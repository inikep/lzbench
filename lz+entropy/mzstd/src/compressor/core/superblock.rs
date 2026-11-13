//! Compressed-size targeted blocks: `ZSTD_c_targetCBlockSize`
//! (`ZSTD_compressBlock_targetCBlockSize` from `zstd_compress.c` and
//! `zstd_compress_superblock.c`).
//!
//! A block of up to 128 KiB (the superblock) is cut into sub-blocks whose
//! compressed size is close to the target. The block's entropy tables are
//! built once for the whole superblock; the first sub-block that can carries
//! their descriptions and the following ones reuse them (`Repeat` modes),
//! so each sub-block is a standard block a receiver can decode on its own.
//! A sub-block that does not compress is merged into its successor, and the
//! tail that never compresses is stored raw.
//!
//! Sizes are estimated, not measured: the split points come from average
//! per-literal and per-sequence costs of the whole superblock, exactly as
//! the reference computes them in integer arithmetic.

use core::num::NonZeroUsize;

use super::block::{self, Build, write_sequence_count};
use super::block_stats::{self, EntropyMetadata, FseMetadata, HufMetadata};
use super::cctx::{BlockState, Engine, no_compress_block, rle_compress_block, write_block_header};
use super::entropy::huf::{self, CTable};
use super::entropy::literals::{self, min_gain};
use super::entropy::sequences::{self, CodeKind, LITERAL_LENGTHS, MATCH_LENGTHS, OFFSETS};
use super::entropy::{FseTables, HufTables, Tables, hist};
use super::error::{CoreError, is_error_code};
use super::mem;
use super::params::TARGET_C_BLOCK_SIZE_BOUNDS;
use super::seq_store::{CODE_COUNT_SLOTS, SeqCodes, SeqDef, SeqStore, update_rep};
use super::window::Segments;
use crate::Backend;
use crate::shared::format::{
    BLOCK_COMPRESSED, BLOCK_HEADER_SIZE, EncodingType, LL_BITS, MIN_MATCH, ML_BITS,
};

/// `ZSTD_TARGETCBLOCKSIZE_MIN`, the smallest target a sub-block is cut to.
const MIN_TARGET: usize = TARGET_C_BLOCK_SIZE_BOUNDS.lower as usize;
/// `BYTESCALE`: fixed-point scale of the per-byte cost estimates.
const BYTE_SCALE: usize = 256;
/// Bytes of the Huffman description the first sub-block's literals header
/// size allows for.
const LITERALS_HEADER_ALLOWANCE: usize = 200;
/// 1 KiB and 16 KiB: where a literals header grows by a byte.
const KB: usize = 1 << 10;

/// Compresses one block into sub-blocks near the target compressed size
/// (`ZSTD_compressBlock_targetCBlockSize`), returning the bytes written,
/// block headers included.
///
/// # Errors
///
/// [`CoreError::DstSizeTooSmall`] when even a stored block does not fit,
/// and the errors of building the block's tables.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
pub(crate) fn compress_block(
    engine: &mut Engine,
    segments: &Segments<'_>,
    view: &mem::Window<'_>,
    start: u32,
    end: u32,
    dst: &mut [u8],
    last_block: bool,
) -> Result<usize, CoreError> {
    let src = view.prefix.slice(start, end);
    let build = block::build_seq_store(engine, segments, view, start, end)?;
    let c_size = compress_block_body(engine, src, dst, build, last_block)?;
    engine.demote_offcode_repeat();
    Ok(c_size)
}

/// `ZSTD_compressBlock_targetCBlockSize_body`.
fn compress_block_body(
    engine: &mut Engine,
    src: &[u8],
    dst: &mut [u8],
    build: Build,
    last_block: bool,
) -> Result<usize, CoreError> {
    if build == Build::Compress {
        // The first block is never an RLE block, as in `ZSTD_compressBlock_internal`.
        if !engine.is_first_block && maybe_rle(&engine.seq_store) && block::is_rle(src) {
            return rle_compress_block(dst, src[0], src.len(), last_block);
        }
        // The superblock may exceed `ZSTD_compressBound` for its input, so
        // a result of zero, one that ran out of room or one that saves
        // nothing falls back to a stored block.
        match compress_super_block(engine, src, dst, last_block) {
            Err(CoreError::DstSizeTooSmall) => {}
            Err(error) => return Err(error),
            Ok(c_size) => {
                let max_c_size = src.len() - min_gain(src.len(), engine.applied.cparams.strategy);
                if c_size != 0 && c_size < max_c_size + BLOCK_HEADER_SIZE {
                    engine.confirm_block_state();
                    return Ok(c_size);
                }
            }
        }
    }
    no_compress_block(dst, src, last_block)
}

/// Whether a block could be one repeated byte (`ZSTD_maybeRLE`).
fn maybe_rle(store: &SeqStore) -> bool {
    store.nb_seq() < 4 && store.literals().len() < 10
}

/// Builds the superblock's tables and writes its sub-blocks
/// (`ZSTD_compressSuperBlock`).
fn compress_super_block(
    engine: &mut Engine,
    src: &[u8],
    dst: &mut [u8],
    last_block: bool,
) -> Result<usize, CoreError> {
    let Engine {
        seq_store,
        prev,
        next,
        applied,
        backend,
        ..
    } = engine;
    let metadata = block_stats::build_block_entropy_stats(
        seq_store,
        &prev.entropy,
        &mut next.entropy,
        applied.cparams.strategy,
        applied.literals_compression_disabled(),
        *backend,
    )?;
    let block = SuperBlock {
        store: seq_store,
        metadata: &metadata,
        target: applied.target_c_block_size,
        backend: *backend,
    };
    compress_sub_blocks(&block, prev, next, src, dst, last_block)
}

/// What the sub-blocks of one superblock share.
struct SuperBlock<'a> {
    /// The block's sequences and literals, with their codes filled.
    store: &'a SeqStore,
    /// The choices and descriptions of the block's entropy tables.
    metadata: &'a EntropyMetadata,
    /// `cctxParams->targetCBlockSize`.
    target: usize,
    backend: Backend,
}

/// What `ZSTD_compressSubBlock` reports.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct SubBlock {
    /// Bytes written, block header included; zero when the sub-block did not
    /// compress.
    size: usize,
    /// Whether the literals section carried the Huffman table description.
    literals_entropy_written: bool,
    /// Whether the sequences section carried the FSE table descriptions.
    sequences_entropy_written: bool,
}

/// `ZSTD_compressSubBlock_multi`.
fn compress_sub_blocks(
    block: &SuperBlock<'_>,
    prev: &BlockState,
    next: &mut BlockState,
    src: &[u8],
    dst: &mut [u8],
    last_block: bool,
) -> Result<usize, CoreError> {
    let store = block.store;
    let metadata = block.metadata;
    let sequences = store.sequences();
    let nb_seqs = sequences.len();
    let codes = store.codes.get(..nb_seqs).unwrap_or_default();
    let literals = store.literals();
    let nb_literals = literals.len();
    let target = MIN_TARGET.max(block.target);
    let mut progress = Progress {
        write_literals_entropy: metadata.huf.kind == EncodingType::Compressed,
        write_sequences_entropy: true,
        ..Progress::default()
    };

    // A general estimation for the full block.
    if let Some(nb_seqs_nonzero) = NonZeroUsize::new(nb_seqs) {
        let estimate = estimate_sub_block_size(
            literals,
            store,
            &next.entropy,
            metadata,
            progress.write_literals_entropy,
            progress.write_sequences_entropy,
        );
        // Quick estimation.
        let avg_lit_cost = (estimate.literals * BYTE_SCALE)
            .checked_div(nb_literals)
            .unwrap_or(BYTE_SCALE);
        let avg_seq_cost = ((estimate.block - estimate.literals) * BYTE_SCALE) / nb_seqs_nonzero;
        let nb_sub_blocks = ((estimate.block + target / 2) / target).max(1);
        let avg_block_budget = (estimate.block * BYTE_SCALE) / nb_sub_blocks;
        // If the full superblock does not compress, give up: the caller
        // stores a single uncompressed block.
        if estimate.block > src.len() {
            return Ok(0);
        }

        // Compress and write sub-blocks. The reference declares a supplement
        // to the budget that is never raised from zero.
        for n in 0..nb_sub_blocks - 1 {
            // The sequences of the current sub-block, plus the literals of
            // the next sequence.
            let seq_count = size_block_sequences(
                &sequences[progress.sequences..],
                avg_block_budget,
                avg_lit_cost,
                avg_seq_cost,
                n == 0,
            );
            // Reaching the last sequence ends the loop: the last sub-block
            // takes the rest.
            if progress.sequences + seq_count == nb_seqs {
                break;
            }
            debug_assert!(seq_count > 0);
            let sp = progress.sequences;
            let lit_size = count_literals(store, sp, seq_count);
            let decompressed = seq_decompressed_size(store, sp, seq_count, lit_size);
            let input = SubBlockInput {
                sequences: &sequences[sp..sp + seq_count],
                codes: &codes[sp..sp + seq_count],
                literals: &literals[progress.literals..progress.literals + lit_size],
                write_literals_entropy: progress.write_literals_entropy,
                write_sequences_entropy: progress.write_sequences_entropy,
                last_block: false,
            };
            let sub =
                compress_sub_block(block, &next.entropy, &input, &mut dst[progress.output..])?;
            // Check compressibility, update state components. Otherwise the
            // sub-block is not compressed yet: it joins the following one.
            if sub.size > 0 && sub.size < decompressed {
                debug_assert!(progress.input + decompressed <= src.len());
                progress.commit(&sub, seq_count, lit_size, decompressed);
            }
        }
    }

    // Write the last block.
    {
        let sp = progress.sequences;
        let lit_size = nb_literals - progress.literals;
        let seq_count = nb_seqs - sp;
        let decompressed = seq_decompressed_size(store, sp, seq_count, lit_size);
        let input = SubBlockInput {
            sequences: &sequences[sp..],
            codes: codes.get(sp..).unwrap_or_default(),
            literals: &literals[progress.literals..],
            write_literals_entropy: progress.write_literals_entropy,
            write_sequences_entropy: progress.write_sequences_entropy,
            last_block,
        };
        let sub = compress_sub_block(block, &next.entropy, &input, &mut dst[progress.output..])?;
        // The literals borrowed from the next sequence must be preserved.
        if sub.size > 0 && sub.size < decompressed {
            debug_assert!(progress.input + decompressed <= src.len());
            progress.commit(&sub, seq_count, lit_size, decompressed);
        }
    }

    if progress.write_literals_entropy {
        // The literals tables were never written: the next block inherits
        // the previous ones.
        next.entropy.huf.clone_from(&prev.entropy.huf);
    }
    if progress.write_sequences_entropy && need_sequence_entropy_tables(&metadata.fse) {
        // The sequence tables were never written, which the decoder needs:
        // the caller stores the block.
        return Ok(0);
    }

    let mut op = progress.output;
    if progress.input < src.len() {
        // Some data is left: the last part of the block is sent uncompressed.
        op += no_compress_block(&mut dst[op..], &src[progress.input..], last_block)?;
        // The repeat offsets are regenerated from the sequences that were
        // coded, since the skipped ones will not be seen by the decoder.
        if progress.sequences < nb_seqs {
            let mut rep = prev.rep;
            for (index, seq) in sequences[..progress.sequences].iter().enumerate() {
                let ll0 = store.sequence_length(index).lit_length == 0;
                update_rep(&mut rep, seq.off_base, ll0);
            }
            next.rep = rep;
        }
    }
    Ok(op)
}

/// Where `compress_sub_blocks` stands in the superblock.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct Progress {
    /// Sequences coded so far (`sp`).
    sequences: usize,
    /// Literals consumed (`lp`).
    literals: usize,
    /// Input bytes regenerated (`ip`).
    input: usize,
    /// Output bytes written (`op`).
    output: usize,
    /// The next sub-block that can carries the Huffman table.
    write_literals_entropy: bool,
    /// The next sub-block that can carries the FSE tables.
    write_sequences_entropy: bool,
}

impl Progress {
    /// Moves past a sub-block that compressed.
    fn commit(&mut self, sub: &SubBlock, seq_count: usize, lit_size: usize, decompressed: usize) {
        self.input += decompressed;
        self.literals += lit_size;
        self.output += sub.size;
        // Entropy only needs to be written once.
        self.write_literals_entropy &= !sub.literals_entropy_written;
        self.write_sequences_entropy &= !sub.sequences_entropy_written;
        self.sequences += seq_count;
    }
}

/// Whether the decoder needs this block's sequence table descriptions: some
/// stream is not predefined or repeated (`ZSTD_needSequenceEntropyTables`).
fn need_sequence_entropy_tables(metadata: &FseMetadata) -> bool {
    [metadata.ll_type, metadata.ml_type, metadata.of_type]
        .into_iter()
        .any(|kind| matches!(kind, EncodingType::Compressed | EncodingType::Rle))
}

/// Literal bytes of `count` sequences from `from` (`countLiterals`).
fn count_literals(store: &SeqStore, from: usize, count: usize) -> usize {
    (from..from + count)
        .map(|index| store.sequence_length(index).lit_length as usize)
        .sum()
}

/// Bytes the sequences from `from` regenerate, `lit_size` of them literals
/// (`ZSTD_seqDecompressedSize`).
fn seq_decompressed_size(store: &SeqStore, from: usize, count: usize, lit_size: usize) -> usize {
    let match_length_sum: usize = (from..from + count)
        .map(|index| store.sequence_length(index).match_length as usize)
        .sum();
    match_length_sum + lit_size
}

/// How many of `sequences` the next sub-block takes to fill its budget
/// (`sizeBlockSequences`). Always at least one.
///
/// Costs are in `1 / BYTE_SCALE` bytes. The first sub-block also pays for
/// the entropy headers. A sub-block keeps growing past its budget until it
/// is deemed compressible.
fn size_block_sequences(
    sequences: &[SeqDef],
    target_budget: usize,
    avg_lit_cost: usize,
    avg_seq_cost: usize,
    first_sub_block: bool,
) -> usize {
    // A generous estimate of the entropy headers.
    let header_size = usize::from(first_sub_block) * 120 * BYTE_SCALE;
    let mut budget = header_size;

    // The first sequence is always taken.
    let first = &sequences[0];
    budget += usize::from(first.lit_length) * avg_lit_cost + avg_seq_cost;
    if budget > target_budget {
        return 1;
    }
    let mut in_size = usize::from(first.lit_length) + (usize::from(first.ml_base) + MIN_MATCH);

    for (n, seq) in sequences.iter().enumerate().skip(1) {
        let current_cost = usize::from(seq.lit_length) * avg_lit_cost + avg_seq_cost;
        budget += current_cost;
        in_size += usize::from(seq.lit_length) + (usize::from(seq.ml_base) + MIN_MATCH);
        // Stop when the sub-block budget is reached, though continue to
        // expand until the sub-block is deemed compressible.
        if budget > target_budget && budget < in_size * BYTE_SCALE {
            return n;
        }
    }
    sequences.len()
}

/// `EstimatedBlockSize`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct EstimatedBlockSize {
    /// The literals section (`estLitSize`).
    literals: usize,
    /// The whole block (`estBlockSize`).
    block: usize,
}

/// Estimates the compressed size of the whole superblock
/// (`ZSTD_estimateSubBlockSize`).
fn estimate_sub_block_size(
    literals: &[u8],
    store: &SeqStore,
    entropy: &Tables,
    metadata: &EntropyMetadata,
    write_literals_entropy: bool,
    write_sequences_entropy: bool,
) -> EstimatedBlockSize {
    let literals = estimate_literals(
        literals,
        &entropy.huf,
        &metadata.huf,
        write_literals_entropy,
    );
    let sequences = estimate_sequences(store, &entropy.fse, &metadata.fse, write_sequences_entropy);
    EstimatedBlockSize {
        literals,
        block: sequences + literals + BLOCK_HEADER_SIZE,
    }
}

/// Size estimate of a literals section (`ZSTD_estimateSubBlockSize_literal`).
fn estimate_literals(
    literals: &[u8],
    huf: &HufTables,
    metadata: &HufMetadata,
    write_entropy: bool,
) -> usize {
    // A hard coded header size.
    const LITERAL_SECTION_HEADER_SIZE: usize = 3;
    match metadata.kind {
        EncodingType::Basic => literals.len(),
        EncodingType::Rle => 1,
        EncodingType::Compressed | EncodingType::Repeat => {
            let mut count = [0u32; 256];
            let mut max_symbol_value = 255;
            if hist::count(&mut count, &mut max_symbol_value, literals, true).is_err() {
                return literals.len();
            }
            let mut estimate = huf.table.estimate_compressed_size(&count, max_symbol_value);
            if write_entropy {
                estimate += metadata.des_size;
            }
            estimate + LITERAL_SECTION_HEADER_SIZE
        }
    }
}

/// Size estimate of the FSE-coded part of a sequences section
/// (`ZSTD_estimateSubBlockSize_symbolType`).
///
/// `count` is the histogram of the stream's `nb_seq` codes.
fn estimate_symbol_type(
    kind: EncodingType,
    count: &[u32; CODE_COUNT_SLOTS],
    nb_seq: usize,
    code_kind: CodeKind,
    table: &super::entropy::fse::CTable,
    additional_bits: Option<&[u8]>,
) -> usize {
    let max = count[..=code_kind.max_code as usize]
        .iter()
        .rposition(|&c| c != 0)
        .unwrap_or(0) as u32;
    let mut bits = match kind {
        // The encoding type was selected, so the stream must be valid.
        EncodingType::Basic if max <= code_kind.default_max => sequences::cross_entropy_cost(
            code_kind.default_norm,
            code_kind.default_norm_log,
            count,
            max,
        ),
        EncodingType::Basic => CoreError::Generic.code(),
        EncodingType::Rle => 0,
        EncodingType::Compressed | EncodingType::Repeat => {
            sequences::fse_bit_cost(table, count, max)
        }
    };
    if is_error_code(bits) {
        return nb_seq * 10;
    }
    // The reference adds each code's extra bits; summing per code value
    // gives the same total.
    for (code, &c) in count[..=max as usize].iter().enumerate() {
        let per_code = match additional_bits {
            Some(extra) => usize::from(extra[code]),
            // For offsets the code is also the number of extra bits.
            None => code,
        };
        bits += c as usize * per_code;
    }
    bits / 8
}

/// Size estimate of a sequences section (`ZSTD_estimateSubBlockSize_sequences`).
fn estimate_sequences(
    store: &SeqStore,
    fse: &FseTables,
    metadata: &FseMetadata,
    write_entropy: bool,
) -> usize {
    // A hard coded header size.
    const SEQUENCES_SECTION_HEADER_SIZE: usize = 3;
    let nb_seq = store.nb_seq();
    if nb_seq == 0 {
        return SEQUENCES_SECTION_HEADER_SIZE;
    }
    let counts = &store.code_counts;
    let mut estimate = estimate_symbol_type(
        metadata.of_type,
        &counts.of,
        nb_seq,
        OFFSETS,
        &fse.offcode,
        None,
    );
    estimate += estimate_symbol_type(
        metadata.ll_type,
        &counts.ll,
        nb_seq,
        LITERAL_LENGTHS,
        &fse.litlength,
        Some(&LL_BITS),
    );
    estimate += estimate_symbol_type(
        metadata.ml_type,
        &counts.ml,
        nb_seq,
        MATCH_LENGTHS,
        &fse.matchlength,
        Some(&ML_BITS),
    );
    if write_entropy {
        estimate += metadata.tables_size;
    }
    estimate + SEQUENCES_SECTION_HEADER_SIZE
}

/// The part of a superblock one sub-block covers.
struct SubBlockInput<'a> {
    sequences: &'a [SeqDef],
    /// The codes of `sequences`.
    codes: &'a [SeqCodes],
    /// The literals of `sequences` (and, for the last sub-block, the
    /// block's trailing literals).
    literals: &'a [u8],
    /// Whether the literals section carries the Huffman table description.
    write_literals_entropy: bool,
    /// Whether the sequences section carries the FSE table descriptions.
    write_sequences_entropy: bool,
    last_block: bool,
}

/// Compresses one sub-block (`ZSTD_compressSubBlock`): a block header, a
/// literals section and a sequences section. A `size` of zero reports that
/// the sub-block did not compress.
fn compress_sub_block(
    block: &SuperBlock<'_>,
    entropy: &Tables,
    input: &SubBlockInput<'_>,
    dst: &mut [u8],
) -> Result<SubBlock, CoreError> {
    if dst.len() < BLOCK_HEADER_SIZE {
        return Err(CoreError::DstSizeTooSmall);
    }
    let mut op = BLOCK_HEADER_SIZE;
    let (literals_size, literals_entropy_written) = compress_sub_block_literals(
        &entropy.huf.table,
        &block.metadata.huf,
        input.literals,
        &mut dst[op..],
        input.write_literals_entropy,
        block.backend,
    )?;
    if literals_size == 0 {
        return Ok(SubBlock::default());
    }
    op += literals_size;
    let (sequences_size, sequences_entropy_written) = compress_sub_block_sequences(
        &entropy.fse,
        &block.metadata.fse,
        input.sequences,
        input.codes,
        &mut dst[op..],
        input.write_sequences_entropy,
        block.backend,
    )?;
    if sequences_size == 0 {
        return Ok(SubBlock::default());
    }
    op += sequences_size;
    write_block_header(
        dst,
        BLOCK_COMPRESSED,
        (op - BLOCK_HEADER_SIZE) as u32,
        input.last_block,
    );
    Ok(SubBlock {
        size: op,
        literals_entropy_written,
        sequences_entropy_written,
    })
}

/// Compresses the literals section of a sub-block
/// (`ZSTD_compressSubBlock_literal`), returning its size (zero when it
/// cannot be compressed) and whether it carries the Huffman table.
///
/// When the table is written, the header size is picked before the table
/// size and compressed size are known, so it allows for the largest table
/// description; if that guess was too small the literals are stored raw.
///
/// The metadata's type says what every sub-block does: `Basic` stores the
/// literals raw, `Rle` as one byte, `Compressed` writes the table in the
/// first sub-block and `Repeat` uses it from the start.
fn compress_sub_block_literals(
    table: &CTable,
    metadata: &HufMetadata,
    literals: &[u8],
    dst: &mut [u8],
    write_entropy: bool,
    backend: Backend,
) -> Result<(usize, bool), CoreError> {
    let lit_size = literals.len();
    let header = if write_entropy {
        LITERALS_HEADER_ALLOWANCE
    } else {
        0
    };
    let lh_size =
        3 + usize::from(lit_size >= KB - header) + usize::from(lit_size >= 16 * KB - header);
    let single_stream = lh_size == 3;
    let kind = if write_entropy {
        metadata.kind
    } else {
        EncodingType::Repeat
    };

    if lit_size == 0 || metadata.kind == EncodingType::Basic {
        return Ok((literals::no_compress(dst, literals)?, false));
    }
    if metadata.kind == EncodingType::Rle {
        return Ok((literals::compress_rle(dst, literals), false));
    }
    debug_assert!(matches!(
        metadata.kind,
        EncodingType::Compressed | EncodingType::Repeat
    ));

    let mut op = lh_size;
    let mut c_lit_size = 0;
    if write_entropy && metadata.kind == EncodingType::Compressed {
        let description = &metadata.des_buffer[..metadata.des_size];
        dst.get_mut(op..op + description.len())
            .ok_or(CoreError::DstSizeTooSmall)?
            .copy_from_slice(description);
        op += description.len();
        c_lit_size += description.len();
    }

    let c_size = if single_stream {
        huf::compress_1x(&mut dst[op..], literals, table, backend)
    } else {
        huf::compress_4x(&mut dst[op..], literals, table, backend)
    };
    op += c_size;
    c_lit_size += c_size;
    if c_size == 0 {
        return Ok((0, false));
    }
    // If we expand and we aren't writing a header, emit uncompressed.
    if !write_entropy && c_lit_size >= lit_size {
        return Ok((literals::no_compress(dst, literals)?, false));
    }
    // If we are writing headers, allow expansion that doesn't change our
    // header size.
    if lh_size < 3 + usize::from(c_lit_size >= KB) + usize::from(c_lit_size >= 16 * KB) {
        debug_assert!(c_lit_size > lit_size);
        return Ok((literals::no_compress(dst, literals)?, false));
    }

    let kind = kind as u32;
    let (size, c_size) = (lit_size as u32, c_lit_size as u32);
    match lh_size {
        3 => {
            // 2 - 2 - 10 - 10
            let lhc = kind + (u32::from(!single_stream) << 2) + (size << 4) + (c_size << 14);
            dst[..3].copy_from_slice(&lhc.to_le_bytes()[..3]);
        }
        4 => {
            // 2 - 2 - 14 - 14
            let lhc = kind + (2 << 2) + (size << 4) + (c_size << 18);
            dst[..4].copy_from_slice(&lhc.to_le_bytes());
        }
        _ => {
            // 2 - 2 - 18 - 18
            let lhc = kind + (3 << 2) + (size << 4) + (c_size << 22);
            dst[..4].copy_from_slice(&lhc.to_le_bytes());
            dst[4] = (c_lit_size >> 10) as u8;
        }
    }
    Ok((op, true))
}

/// Compresses the sequences section of a sub-block
/// (`ZSTD_compressSubBlock_sequences`), returning its size (zero when it
/// cannot be used) and whether it carries the FSE tables.
///
/// The metadata's types are the superblock's: the first sub-block that is
/// kept has them in its header, and the following ones repeat the tables.
fn compress_sub_block_sequences(
    fse: &FseTables,
    metadata: &FseMetadata,
    sequences: &[SeqDef],
    codes: &[SeqCodes],
    dst: &mut [u8],
    write_entropy: bool,
    backend: Backend,
) -> Result<(usize, bool), CoreError> {
    let nb_seq = sequences.len();
    // The sequence count and the symbol compression modes.
    if dst.len() < 3 + 1 {
        return Err(CoreError::DstSizeTooSmall);
    }
    let mut op = write_sequence_count(dst, nb_seq);
    if nb_seq == 0 {
        return Ok((op, false));
    }
    let seq_head = op;
    op += 1;

    if write_entropy {
        dst[seq_head] = ((metadata.ll_type as u8) << 6)
            + ((metadata.of_type as u8) << 4)
            + ((metadata.ml_type as u8) << 2);
        dst.get_mut(op..op + metadata.tables_size)
            .ok_or(CoreError::DstSizeTooSmall)?
            .copy_from_slice(&metadata.tables_buffer[..metadata.tables_size]);
        op += metadata.tables_size;
    } else {
        let repeat = EncodingType::Repeat as u8;
        dst[seq_head] = (repeat << 6) + (repeat << 4) + (repeat << 2);
    }

    let bitstream_size = sequences::encode_slice(&mut dst[op..], fse, sequences, codes, backend)?;
    op += bitstream_size;
    // Decoders up to 1.3.4 report corruption when the last table is two
    // bytes and the bitstream a single one; store such a block instead.
    if write_entropy
        && metadata.last_count_size != 0
        && metadata.last_count_size + bitstream_size < 4
    {
        return Ok((0, false));
    }
    // Decoders up to 1.4.0 report an error when a sequences section is
    // under three bytes, which a repeat after an RLE table can produce.
    if op - seq_head < 4 {
        return Ok((0, false));
    }
    Ok((op, true))
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::compressor::core::entropy::huf::Repeat;
    use crate::compressor::core::params::Strategy;
    use crate::compressor::core::seq_store::offset_to_offbase;

    /// Skewed literals: a few symbols dominate, so Huffman coding pays.
    fn skewed(len: usize) -> Vec<u8> {
        (0..len as u32)
            .map(|i| (i.wrapping_mul(i) % 11) as u8)
            .collect()
    }

    /// A store of `(literals, offset, match length)` sequences and trailing
    /// literals.
    fn store_of(sequences: &[(&[u8], u32, usize)], tail: &[u8]) -> SeqStore {
        let mut store = SeqStore::default();
        store.reserve(1 << 17, 3);
        for &(literals, offset, match_length) in sequences {
            store.store_seq(literals, offset_to_offbase(offset), match_length);
        }
        store.store_last_literals(tail);
        store
    }

    /// The tables and statistics of `store`, as `compress_super_block` builds them.
    fn built(store: &mut SeqStore) -> (Tables, EntropyMetadata) {
        let prev = Tables::default();
        let mut next = Tables::default();
        let metadata = block_stats::build_block_entropy_stats(
            store,
            &prev,
            &mut next,
            Strategy::Lazy,
            false,
            Backend::default(),
        )
        .expect("builds");
        (next, metadata)
    }

    fn many_sequences() -> SeqStore {
        let literals = skewed(60);
        let mut parts: Vec<(&[u8], u32, usize)> = Vec::new();
        for i in 0..64usize {
            parts.push((
                &literals[i % 7..i % 7 + 5 + i % 9],
                [1, 4, 100, 3000, 70_000][i % 5],
                3 + (i * 7) % 40,
            ));
        }
        store_of(&parts, &literals[..20])
    }

    #[test]
    fn a_sub_block_is_sized_by_its_budget_and_grows_until_compressible() {
        // Ten sequences of 10 literals and a 20 byte match, 30 bytes each.
        let sequences = vec![
            SeqDef {
                off_base: 10,
                lit_length: 10,
                ml_base: 17
            };
            10
        ];
        // One byte per literal and two per sequence: 3072 per sequence.
        let size = |budget, first| size_block_sequences(&sequences, budget, 256, 512, first);
        // The first sequence is always taken, even when it exceeds the budget.
        assert_eq!(size(100, false), 1);
        // The entropy headers are charged to the first sub-block only.
        assert_eq!(size(7000, true), 1);
        // The third sequence passes the budget while the sub-block pays.
        assert_eq!(size(7000, false), 2);
        // Literals at three bytes each never pay for themselves, so the
        // sub-block takes everything.
        assert_eq!(
            size_block_sequences(&sequences, 12_000, 768, 512, false),
            10
        );
        // Without a budget the sub-block takes everything too.
        assert_eq!(size(usize::MAX, false), 10);
    }

    #[test]
    fn only_compressed_or_rle_streams_need_their_tables_written() {
        let metadata = |ll_type, of_type, ml_type| FseMetadata {
            ll_type,
            of_type,
            ml_type,
            tables_size: 0,
            tables_buffer: [0; block_stats::MAX_FSE_HEADERS_SIZE],
            last_count_size: 0,
        };
        use EncodingType::{Basic, Compressed, Repeat, Rle};
        assert!(!need_sequence_entropy_tables(&metadata(
            Basic, Repeat, Basic
        )));
        for kind in [Compressed, Rle] {
            assert!(need_sequence_entropy_tables(&metadata(kind, Basic, Basic)));
            assert!(need_sequence_entropy_tables(&metadata(Basic, kind, Basic)));
            assert!(need_sequence_entropy_tables(&metadata(Basic, Basic, kind)));
        }
    }

    #[test]
    fn literal_and_sequence_totals_follow_the_stored_lengths() {
        let mut store = store_of(&[(&[1, 2, 3], 10, 4), (&[4], 20, 9), (&[], 30, 3)], &[5, 6]);
        assert_eq!(count_literals(&store, 0, 3), 4);
        assert_eq!(count_literals(&store, 1, 2), 1);
        assert_eq!(count_literals(&store, 3, 0), 0);
        assert_eq!(seq_decompressed_size(&store, 0, 3, 4), 4 + 9 + 3 + 4);
        assert_eq!(seq_decompressed_size(&store, 1, 1, 1), 9 + 1);
        // A sequence over 64 KiB carries the escape: both sums honour it.
        store.long_length = crate::compressor::core::seq_store::LongLength::Match(1);
        assert_eq!(
            seq_decompressed_size(&store, 0, 3, 4),
            4 + 9 + 3 + 4 + 0x10000
        );
        store.long_length = crate::compressor::core::seq_store::LongLength::Literal(0);
        assert_eq!(count_literals(&store, 0, 1), 3 + 0x10000);
    }

    #[test]
    fn estimates_follow_the_stream_types() {
        let mut store = many_sequences();
        let (tables, metadata) = built(&mut store);
        let literals = store.literals().to_vec();
        let with_tables =
            estimate_sub_block_size(&literals, &store, &tables, &metadata, true, true);
        let without = estimate_sub_block_size(&literals, &store, &tables, &metadata, false, false);
        assert!(with_tables.block >= without.block);
        assert_eq!(
            with_tables.block - without.block,
            (with_tables.literals - without.literals) + metadata.fse.tables_size
        );
        assert_eq!(
            with_tables.block,
            estimate_sequences(&store, &tables.fse, &metadata.fse, true)
                + with_tables.literals
                + BLOCK_HEADER_SIZE
        );

        // Literals: raw, RLE and Huffman coded.
        let huf = &tables.huf;
        let mut huf_metadata = metadata.huf;
        huf_metadata.kind = EncodingType::Basic;
        assert_eq!(
            estimate_literals(&literals, huf, &huf_metadata, true),
            literals.len()
        );
        huf_metadata.kind = EncodingType::Rle;
        assert_eq!(estimate_literals(&literals, huf, &huf_metadata, true), 1);

        // No sequences: the section header alone.
        let mut empty = SeqStore::default();
        let (tables, metadata) = built(&mut empty);
        assert_eq!(
            estimate_sequences(&empty, &tables.fse, &metadata.fse, true),
            3
        );
    }

    #[test]
    fn a_stream_the_chosen_encoding_cannot_code_costs_ten_bits_a_sequence() {
        let table = crate::compressor::core::entropy::fse::CTable::EMPTY;
        let mut count = [0u32; CODE_COUNT_SLOTS];
        count[2] = 3;
        // RLE costs no table bits; the extra bits are all there is.
        assert_eq!(
            estimate_symbol_type(EncodingType::Rle, &count, 3, OFFSETS, &table, None),
            3 * 2 / 8
        );
        // A table without the symbol cannot code it.
        count[3] = 1;
        assert_eq!(
            estimate_symbol_type(EncodingType::Repeat, &count, 4, OFFSETS, &table, None),
            40
        );
        // The predefined distribution stops at offset code 28.
        let mut count = [0u32; CODE_COUNT_SLOTS];
        count[30] = 5;
        assert_eq!(
            estimate_symbol_type(EncodingType::Basic, &count, 5, OFFSETS, &table, None),
            50
        );
        count[30] = 0;
        count[3] = 8;
        assert!(estimate_symbol_type(EncodingType::Basic, &count, 8, OFFSETS, &table, None) > 0);
    }

    /// A Huffman table over all byte values that gives every symbol but the
    /// first eight the longest code, and its metadata.
    fn long_code_table() -> (CTable, HufMetadata) {
        let mut count = [1u32; 256];
        for (symbol, c) in count.iter_mut().take(8).enumerate() {
            *c = 1 << (20 - 2 * symbol);
        }
        let mut table = CTable::EMPTY;
        let log = table.build(&count, 255, 11).expect("builds");
        let mut metadata = HufMetadata {
            kind: EncodingType::Compressed,
            des_size: 0,
            des_buffer: [0; block_stats::MAX_HUF_HEADER_SIZE],
        };
        metadata.des_size = table
            .write(&mut metadata.des_buffer, 255, log)
            .expect("writes");
        (table, metadata)
    }

    fn rare_literals(len: usize) -> Vec<u8> {
        (0..len).map(|i| 8 + (i % 248) as u8).collect()
    }

    #[test]
    fn literals_sections_pick_their_header_from_the_size_and_the_table_mode() {
        let mut store = store_of(&[], &skewed(5000));
        let (tables, metadata) = built(&mut store);
        let literals = store.literals().to_vec();
        let table = &*tables.huf.table;
        let mut dst = vec![0u8; 8000];
        let backend = Backend::default();

        // The first sub-block writes the table: compressed literals.
        let (size, written) =
            compress_sub_block_literals(table, &metadata.huf, &literals, &mut dst, true, backend)
                .expect("fits");
        assert!(written && size < literals.len());
        assert_eq!(dst[0] & 3, EncodingType::Compressed as u8);
        // 5000 literals: a four byte header with four streams.
        assert_eq!((dst[0] >> 2) & 3, 2);
        // The following ones reuse it: a treeless block.
        let (repeated, written) =
            compress_sub_block_literals(table, &metadata.huf, &literals, &mut dst, false, backend)
                .expect("fits");
        assert!(written && repeated < size);
        assert_eq!(dst[0] & 3, EncodingType::Repeat as u8);

        // 700 literals: a three byte header, one stream for a table-less
        // section and, because the table is written, four for 824 or more.
        let (_, written) = compress_sub_block_literals(
            table,
            &metadata.huf,
            &literals[..700],
            &mut dst,
            true,
            backend,
        )
        .expect("fits");
        assert!(written);
        assert_eq!((dst[0] >> 2) & 3, 0);
        let (_, _) = compress_sub_block_literals(
            table,
            &metadata.huf,
            &literals[..900],
            &mut dst,
            true,
            backend,
        )
        .expect("fits");
        assert_eq!((dst[0] >> 2) & 3, 2);
        // Over 16 KiB less the table allowance the header takes five bytes.
        let long = skewed(17_000);
        let (_, written) =
            compress_sub_block_literals(table, &metadata.huf, &long, &mut dst, true, backend)
                .expect("fits");
        assert!(written);
        assert_eq!((dst[0] >> 2) & 3, 3);
    }

    #[test]
    fn literals_that_are_raw_or_one_byte_need_no_table() {
        let table = CTable::EMPTY;
        let mut dst = [0u8; 64];
        let backend = Backend::default();
        let mut metadata = block_stats::HufMetadata {
            kind: EncodingType::Basic,
            des_size: 0,
            des_buffer: [0; block_stats::MAX_HUF_HEADER_SIZE],
        };
        let (size, written) =
            compress_sub_block_literals(&table, &metadata, &[1, 2, 3], &mut dst, true, backend)
                .expect("fits");
        assert_eq!((size, written), (4, false));
        assert_eq!(dst[1..4], [1, 2, 3]);
        // No literals at all are an empty raw section.
        metadata.kind = EncodingType::Compressed;
        let (size, written) =
            compress_sub_block_literals(&table, &metadata, &[], &mut dst, true, backend)
                .expect("fits");
        assert_eq!((size, written), (1, false));
        metadata.kind = EncodingType::Rle;
        let (size, written) =
            compress_sub_block_literals(&table, &metadata, &[7; 40], &mut dst, false, backend)
                .expect("fits");
        assert_eq!((size, written), (3, false));
        assert_eq!(dst[0] & 3, EncodingType::Rle as u8);
        assert_eq!(dst[2], 7);
        // A raw section that does not fit is an error.
        metadata.kind = EncodingType::Basic;
        assert_eq!(
            compress_sub_block_literals(&table, &metadata, &[0; 100], &mut dst, true, backend),
            Err(CoreError::DstSizeTooSmall)
        );
    }

    #[test]
    fn literals_that_expand_beyond_their_header_are_stored_raw() {
        let (table, metadata) = long_code_table();
        let literals = rare_literals(800);
        let mut dst = vec![0u8; 4000];
        let backend = Backend::default();
        // Every literal costs eleven bits: the section outgrows the
        // three byte header that 800 literals were allotted.
        let (size, written) =
            compress_sub_block_literals(&table, &metadata, &literals, &mut dst, true, backend)
                .expect("fits");
        assert_eq!((size, written), (literals.len() + 2, false));
        assert_eq!(dst[0] & 3, EncodingType::Basic as u8);
        assert_eq!(dst[2..802], literals[..]);
        // A repeated table that expands is stored raw too.
        let (size, written) =
            compress_sub_block_literals(&table, &metadata, &literals, &mut dst, false, backend)
                .expect("fits");
        assert_eq!((size, written), (literals.len() + 2, false));
    }

    #[test]
    fn literals_that_do_not_fit_are_reported_not_compressed() {
        let (table, metadata) = long_code_table();
        let literals = rare_literals(800);
        let backend = Backend::default();
        // Room for the header and the description, not for the streams.
        let mut dst = vec![0u8; 3 + metadata.des_size + 2];
        assert_eq!(
            compress_sub_block_literals(&table, &metadata, &literals, &mut dst, true, backend),
            Ok((0, false))
        );
        // Not even room for the description.
        let mut dst = vec![0u8; 4];
        assert_eq!(
            compress_sub_block_literals(&table, &metadata, &literals, &mut dst, true, backend),
            Err(CoreError::DstSizeTooSmall)
        );
    }

    /// One sequence coded with run-length tables: a bitstream of one byte.
    fn single_rle_sequence() -> (SeqStore, FseTables, FseMetadata) {
        let mut store = store_of(&[(&[1, 2, 3, 4, 5], 100, 20)], &[]);
        store.fill_codes(Backend::default());
        let code = store.codes[0];
        let mut fse = FseTables::default();
        fse.litlength.build_rle(code.ll);
        fse.offcode.build_rle(code.of);
        fse.matchlength.build_rle(code.ml);
        let metadata = FseMetadata {
            ll_type: EncodingType::Rle,
            of_type: EncodingType::Rle,
            ml_type: EncodingType::Rle,
            tables_size: 3,
            tables_buffer: {
                let mut buffer = [0; block_stats::MAX_FSE_HEADERS_SIZE];
                buffer[..3].copy_from_slice(&[code.ll, code.of, code.ml]);
                buffer
            },
            last_count_size: 0,
        };
        (store, fse, metadata)
    }

    #[test]
    fn sequences_sections_carry_their_tables_once_and_repeat_them_after() {
        let backend = Backend::default();
        let mut store = many_sequences();
        let (tables, metadata) = built(&mut store);
        let n = store.nb_seq();
        let mut dst = vec![0u8; 4096];
        let sequences = store.sequences();
        let codes = &store.codes[..n];

        let (size, written) = compress_sub_block_sequences(
            &tables.fse,
            &metadata.fse,
            sequences,
            codes,
            &mut dst,
            true,
            backend,
        )
        .expect("fits");
        assert!(written);
        assert_eq!(dst[0] as usize, n);
        let head = (metadata.fse.ll_type as u8) << 6
            | (metadata.fse.of_type as u8) << 4
            | (metadata.fse.ml_type as u8) << 2;
        assert_eq!(dst[1], head);
        assert_eq!(
            dst[2..2 + metadata.fse.tables_size],
            metadata.fse.tables_buffer[..metadata.fse.tables_size]
        );

        let (repeated, written) = compress_sub_block_sequences(
            &tables.fse,
            &metadata.fse,
            sequences,
            codes,
            &mut dst,
            false,
            backend,
        )
        .expect("fits");
        assert!(written);
        assert_eq!(dst[1], 0xFC);
        assert_eq!(size - repeated, metadata.fse.tables_size);

        // A sub-block of 200 sequences counts them in two bytes.
        let literals = skewed(30);
        let parts: Vec<(&[u8], u32, usize)> = (0..200)
            .map(|i| {
                (
                    &literals[i % 5..i % 5 + 3],
                    [1u32, 8, 300][i % 3],
                    3 + i % 11,
                )
            })
            .collect();
        let mut many = store_of(&parts, &literals[..5]);
        let (tables, metadata) = built(&mut many);
        let codes = &many.codes[..200];
        let (_, written) = compress_sub_block_sequences(
            &tables.fse,
            &metadata.fse,
            many.sequences(),
            codes,
            &mut dst,
            true,
            backend,
        )
        .expect("fits");
        assert!(written);
        assert_eq!(dst[0], 0x80);
        assert_eq!(dst[1], 200);
    }

    #[test]
    fn sequences_sections_are_identical_on_every_backend() {
        let mut store = many_sequences();
        let (tables, metadata) = built(&mut store);
        let n = store.nb_seq();
        let mut expected = vec![0u8; 4096];
        let reference = compress_sub_block_sequences(
            &tables.fse,
            &metadata.fse,
            store.sequences(),
            &store.codes[..n],
            &mut expected,
            true,
            Backend::default(),
        )
        .expect("fits");
        for backend in Backend::available() {
            // A run of the sequences from the middle, as a later sub-block holds it.
            for from in [0, 1, n / 2] {
                let mut dst = vec![0u8; 4096];
                let mut baseline = vec![0u8; 4096];
                let ours = compress_sub_block_sequences(
                    &tables.fse,
                    &metadata.fse,
                    &store.sequences()[from..],
                    &store.codes[from..n],
                    &mut dst,
                    false,
                    backend,
                )
                .expect("fits");
                let scalar = compress_sub_block_sequences(
                    &tables.fse,
                    &metadata.fse,
                    &store.sequences()[from..],
                    &store.codes[from..n],
                    &mut baseline,
                    false,
                    Backend::SCALAR,
                )
                .expect("fits");
                assert_eq!(ours, scalar, "{backend} from {from}");
                assert_eq!(dst[..ours.0], baseline[..scalar.0], "{backend} from {from}");
            }
            if backend == Backend::default() {
                assert!(reference.1);
            }
        }
    }

    #[test]
    fn sequences_sections_without_sequences_or_room_say_so() {
        let backend = Backend::default();
        let (store, fse, metadata) = single_rle_sequence();
        let mut dst = [0xAA; 3];
        // The count and the head byte need four bytes.
        assert_eq!(
            compress_sub_block_sequences(&fse, &metadata, &[], &[], &mut dst, true, backend),
            Err(CoreError::DstSizeTooSmall)
        );
        let mut dst = [0xAA; 64];
        // No sequences: just the zero count, and nothing was written.
        assert_eq!(
            compress_sub_block_sequences(&fse, &metadata, &[], &[], &mut dst, true, backend),
            Ok((1, false))
        );
        assert_eq!(dst[0], 0);
        // The tables do not fit behind the head byte.
        let mut dst = [0xAA; 5];
        assert_eq!(
            compress_sub_block_sequences(
                &fse,
                &metadata,
                store.sequences(),
                &store.codes[..1],
                &mut dst,
                true,
                backend
            ),
            Err(CoreError::DstSizeTooSmall)
        );
    }

    #[test]
    fn sections_old_decoders_reject_are_not_produced() {
        let backend = Backend::default();
        let (store, fse, mut metadata) = single_rle_sequence();
        let (sequences, codes) = (store.sequences(), &store.codes[..1]);
        let mut dst = [0u8; 64];
        // Written with its tables the section is 1 + 3 + 1 bytes after the count.
        assert_eq!(
            compress_sub_block_sequences(
                &fse, &metadata, sequences, codes, &mut dst, true, backend
            ),
            Ok((6, true))
        );
        // Repeating run-length tables leaves a body under four bytes, which
        // decoders up to 1.4.0 reject.
        assert_eq!(
            compress_sub_block_sequences(
                &fse, &metadata, sequences, codes, &mut dst, false, backend
            ),
            Ok((0, false))
        );
        // A trailing two byte table and a one byte bitstream trip the 1.3.4 bug.
        metadata.last_count_size = 2;
        assert_eq!(
            compress_sub_block_sequences(
                &fse, &metadata, sequences, codes, &mut dst, true, backend
            ),
            Ok((0, false))
        );
        metadata.last_count_size = 3;
        assert_eq!(
            compress_sub_block_sequences(
                &fse, &metadata, sequences, codes, &mut dst, true, backend
            ),
            Ok((6, true))
        );
    }

    fn input_of<'a>(
        sequences: &'a [SeqDef],
        codes: &'a [SeqCodes],
        literals: &'a [u8],
        last_block: bool,
    ) -> SubBlockInput<'a> {
        SubBlockInput {
            sequences,
            codes,
            literals,
            write_literals_entropy: true,
            write_sequences_entropy: true,
            last_block,
        }
    }

    #[test]
    fn a_sub_block_has_a_header_and_reports_what_it_wrote() {
        let backend = Backend::default();
        let mut store = many_sequences();
        let (tables, metadata) = built(&mut store);
        let n = store.nb_seq();
        let block = SuperBlock {
            store: &store,
            metadata: &metadata,
            target: 1340,
            backend,
        };
        let mut dst = vec![0u8; 4096];
        let input = input_of(store.sequences(), &store.codes[..n], store.literals(), true);
        let sub = compress_sub_block(&block, &tables, &input, &mut dst).expect("fits");
        assert!(sub.size > BLOCK_HEADER_SIZE);
        assert_eq!(
            sub.literals_entropy_written,
            metadata.huf.kind == EncodingType::Compressed
        );
        assert!(sub.sequences_entropy_written);
        let header = u32::from_le_bytes([dst[0], dst[1], dst[2], 0]);
        assert_eq!(header & 1, 1, "last block");
        assert_eq!((header >> 1) & 3, BLOCK_COMPRESSED);
        assert_eq!(header >> 3, (sub.size - BLOCK_HEADER_SIZE) as u32);
        let input = input_of(
            store.sequences(),
            &store.codes[..n],
            store.literals(),
            false,
        );
        let sub = compress_sub_block(&block, &tables, &input, &mut dst).expect("fits");
        assert_eq!(dst[0] & 1, 0);
        assert!(sub.size > 0);
        // No room for a block header.
        assert_eq!(
            compress_sub_block(&block, &tables, &input, &mut dst[..2]),
            Err(CoreError::DstSizeTooSmall)
        );
    }

    #[test]
    fn a_sub_block_whose_sections_do_not_compress_reports_zero() {
        let backend = Backend::default();
        // Literals that do not fit.
        let (table, huf) = long_code_table();
        let mut store = store_of(&[], &rare_literals(800));
        store.fill_codes(backend);
        let mut tables = Tables::default();
        *tables.huf.table = table;
        let (_, mut metadata) = built(&mut store);
        metadata.huf = huf;
        let block = SuperBlock {
            store: &store,
            metadata: &metadata,
            target: 1340,
            backend,
        };
        let input = input_of(&[], &[], store.literals(), false);
        let mut dst = vec![0u8; 3 + 3 + huf.des_size + 2];
        assert_eq!(
            compress_sub_block(&block, &tables, &input, &mut dst),
            Ok(SubBlock::default())
        );

        // Sequences that old decoders would reject.
        let (single, fse, fse_metadata) = single_rle_sequence();
        let tables = Tables {
            fse,
            ..Tables::default()
        };
        let metadata = EntropyMetadata {
            huf: HufMetadata {
                kind: EncodingType::Basic,
                ..huf
            },
            fse: fse_metadata,
        };
        let block = SuperBlock {
            store: &single,
            metadata: &metadata,
            target: 1340,
            backend,
        };
        let mut input = input_of(
            single.sequences(),
            &single.codes[..1],
            &[1, 2, 3, 4, 5],
            false,
        );
        input.write_sequences_entropy = false;
        let mut dst = vec![0u8; 256];
        assert_eq!(
            compress_sub_block(&block, &tables, &input, &mut dst),
            Ok(SubBlock::default())
        );
    }

    #[test]
    fn tables_that_were_never_written_fall_back_to_the_previous_ones() {
        let backend = Backend::default();
        // Literals the chosen table expands: no sub-block can carry it.
        let (table, huf) = long_code_table();
        let literals = rare_literals(800);
        let mut store = store_of(&[], &literals);
        let (_, mut metadata) = built(&mut store);
        metadata.huf = huf;
        let mut prev = BlockState::default();
        prev.entropy.huf.repeat = Repeat::Valid;
        let mut next = BlockState::default();
        *next.entropy.huf.table = table;
        next.entropy.huf.repeat = Repeat::Check;
        let block = SuperBlock {
            store: &store,
            metadata: &metadata,
            target: 1340,
            backend,
        };
        let mut dst = vec![0u8; 2000];
        let size =
            compress_sub_blocks(&block, &prev, &mut next, &literals, &mut dst, true).expect("fits");
        // The whole block is stored, and the next block inherits the
        // previous table state.
        assert_eq!(size, BLOCK_HEADER_SIZE + literals.len());
        assert_eq!(next.entropy.huf, prev.entropy.huf);
        assert_eq!(dst[0] & 1, 1);
        assert_eq!((dst[0] >> 1) & 3, 0);
    }

    #[test]
    fn a_superblock_the_estimate_calls_incompressible_is_left_to_the_caller() {
        let backend = Backend::default();
        let mut store = many_sequences();
        let (tables, metadata) = built(&mut store);
        let prev = BlockState::default();
        let mut next = BlockState {
            entropy: tables,
            rep: [1, 4, 8],
        };
        let block = SuperBlock {
            store: &store,
            metadata: &metadata,
            target: 1340,
            backend,
        };
        // An input shorter than the estimated block size.
        let src = vec![0u8; 10];
        let mut dst = vec![0u8; 4096];
        assert_eq!(
            compress_sub_blocks(&block, &prev, &mut next, &src, &mut dst, false),
            Ok(0)
        );
    }

    #[test]
    fn short_blocks_with_few_literals_may_be_run_length_blocks() {
        let mut store = store_of(&[(&[1; 4], 1, 8)], &[1; 3]);
        assert!(maybe_rle(&store));
        store = store_of(&[(&[1; 12], 1, 8)], &[]);
        assert!(!maybe_rle(&store));
        let none: &[u8] = &[];
        store = store_of(&[(none, 1, 8); 4], &[]);
        assert!(!maybe_rle(&store));
    }
}

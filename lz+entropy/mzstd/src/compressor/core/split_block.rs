//! Post-hoc block splitting (`ZSTD_compressBlock_splitBlock` and the
//! entropy estimators it relies on, from `zstd_compress.c`).
//!
//! After match finding, a block's sequences may be cut into partitions when
//! the estimated cost of coding them separately is lower. The estimate uses
//! the same table construction as real coding, including its error paths,
//! so a partition is chosen exactly when the reference chooses it.

use alloc::vec::Vec;

use super::block::{self, Build};
use super::cctx::{Engine, no_compress_block, rle_compress_block, write_block_header};
use super::entropy::hist;
use super::entropy::huf::{self, Flags};
use super::entropy::sequences::{self, CodeKind, LITERAL_LENGTHS, MATCH_LENGTHS, OFFSETS};
use super::entropy::{FseTables, HufTables, Tables};
use super::error::{CoreError, is_error_code};
use super::mem;
use super::params::Strategy;
use super::seq_store::{
    CODE_COUNT_SLOTS, CodeCounts, LongLength, SeqStore, offbase_is_repcode, offset_to_offbase,
    update_rep,
};
use super::window::Segments;
use crate::Backend;
use crate::shared::format::{
    BLOCK_COMPRESSED, BLOCK_HEADER_SIZE, EncodingType, LL_BITS, LONG_NB_SEQ, MIN_MATCH, ML_BITS,
    REP_NUM,
};

/// `ZSTD_MAX_NB_BLOCK_SPLITS`.
const MAX_NB_BLOCK_SPLITS: usize = 196;
/// `MIN_SEQUENCES_BLOCK_SPLITTING`.
const MIN_SEQUENCES_BLOCK_SPLITTING: usize = 300;
/// `ZSTD_MAX_HUF_HEADER_SIZE`.
const MAX_HUF_HEADER_SIZE: usize = 128;
/// `ZSTD_MAX_FSE_HEADERS_SIZE`.
const MAX_FSE_HEADERS_SIZE: usize = (53 * 9 + 36 * 9 + 32 * 8usize).div_ceil(8);

/// `ZSTD_hufCTablesMetadata_t`.
#[derive(Copy, Clone, Debug)]
struct HufMetadata {
    kind: EncodingType,
    /// Size of the table description; may hold a reference error code.
    des_size: usize,
}

/// `ZSTD_fseCTablesMetadata_t`.
#[derive(Copy, Clone, Debug)]
struct FseMetadata {
    ll_type: EncodingType,
    of_type: EncodingType,
    ml_type: EncodingType,
    tables_size: usize,
}

/// `ZSTD_buildBlockEntropyStats_literals`.
fn build_literals_stats(
    src: &[u8],
    prev: &HufTables,
    next: &mut HufTables,
    disabled: bool,
    optimal_depth: bool,
) -> Result<HufMetadata, CoreError> {
    let src_size = src.len();
    next.clone_from(prev);
    let basic = HufMetadata {
        kind: EncodingType::Basic,
        des_size: 0,
    };
    if disabled {
        return Ok(basic);
    }
    let min_lit_size = if prev.repeat == huf::Repeat::Valid {
        6
    } else {
        63
    };
    if src_size <= min_lit_size {
        return Ok(basic);
    }
    let mut count = [0u32; 256];
    let mut max_symbol_value = 255;
    let largest = hist::count(&mut count, &mut max_symbol_value, src, false)? as usize;
    if largest == src_size {
        return Ok(HufMetadata {
            kind: EncodingType::Rle,
            des_size: 0,
        });
    }
    if largest <= (src_size >> 7) + 4 {
        return Ok(basic);
    }
    let mut repeat = prev.repeat;
    if repeat == huf::Repeat::Check && !prev.table.validate(&count, max_symbol_value) {
        repeat = huf::Repeat::None;
    }
    next.table = huf::CTable::default();
    let flags = Flags {
        optimal_depth,
        ..Flags::default()
    };
    let huff_log = huf::optimal_table_log(11, src_size, max_symbol_value, &count, flags);
    let huff_log = next.table.build(&count, max_symbol_value, huff_log)?;
    let new_c_size = next
        .table
        .estimate_compressed_size(&count, max_symbol_value);
    let mut description = [0u8; MAX_HUF_HEADER_SIZE];
    let h_size = super::error::as_code(next.table.write(
        &mut description,
        max_symbol_value,
        huff_log,
    ));
    if repeat != huf::Repeat::None {
        let old_c_size = prev
            .table
            .estimate_compressed_size(&count, max_symbol_value);
        if old_c_size < src_size
            && (old_c_size <= h_size.wrapping_add(new_c_size)
                || h_size.wrapping_add(12) >= src_size)
        {
            next.clone_from(prev);
            return Ok(HufMetadata {
                kind: EncodingType::Repeat,
                des_size: 0,
            });
        }
    }
    if new_c_size.wrapping_add(h_size) >= src_size {
        next.clone_from(prev);
        return Ok(basic);
    }
    next.repeat = huf::Repeat::Check;
    if is_error_code(h_size) {
        return Err(CoreError::Generic);
    }
    Ok(HufMetadata {
        kind: EncodingType::Compressed,
        des_size: h_size,
    })
}

/// `ZSTD_buildBlockEntropyStats_sequences`.
fn build_sequences_stats(
    store: &mut SeqStore,
    prev: &FseTables,
    next: &mut FseTables,
    strategy: Strategy,
    backend: Backend,
) -> Result<FseMetadata, CoreError> {
    if store.nb_seq() == 0 {
        next.litlength_repeat = super::entropy::FseRepeat::None;
        next.offcode_repeat = super::entropy::FseRepeat::None;
        next.matchlength_repeat = super::entropy::FseRepeat::None;
        return Ok(FseMetadata {
            ll_type: EncodingType::Basic,
            of_type: EncodingType::Basic,
            ml_type: EncodingType::Basic,
            tables_size: 0,
        });
    }
    let mut buffer = [0u8; MAX_FSE_HEADERS_SIZE];
    let stats = sequences::build_statistics(store, prev, next, &mut buffer, strategy, backend)?;
    Ok(FseMetadata {
        ll_type: stats.ll_type,
        of_type: stats.of_type,
        ml_type: stats.ml_type,
        tables_size: stats.size,
    })
}

/// `ZSTD_estimateBlockSize_literal`.
fn estimate_literals(
    literals: &[u8],
    huf: &HufTables,
    metadata: &HufMetadata,
    write_entropy: bool,
) -> usize {
    const KB: usize = 1 << 10;
    let lit_size = literals.len();
    let header = 3 + usize::from(lit_size >= KB) + usize::from(lit_size >= 16 * KB);
    let single_stream = lit_size < 256;
    match metadata.kind {
        EncodingType::Basic => lit_size,
        EncodingType::Rle => 1,
        EncodingType::Compressed | EncodingType::Repeat => {
            let mut count = [0u32; 256];
            let mut max = 255;
            if hist::count(&mut count, &mut max, literals, false).is_err() {
                return lit_size;
            }
            let mut estimate = huf.table.estimate_compressed_size(&count, max);
            if write_entropy {
                estimate += metadata.des_size;
            }
            if !single_stream {
                estimate += 6;
            }
            estimate + header
        }
    }
}

/// `ZSTD_estimateBlockSize_symbolType`.
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
        EncodingType::Basic => sequences::cross_entropy_cost(
            code_kind.default_norm,
            code_kind.default_norm_log,
            count,
            max,
        ),
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
            None => code,
        };
        bits += c as usize * per_code;
    }
    bits >> 3
}

/// `ZSTD_estimateBlockSize`.
fn estimate_block_size(
    store: &SeqStore,
    entropy: &Tables,
    huf_metadata: &HufMetadata,
    fse_metadata: &FseMetadata,
    write_lit_entropy: bool,
) -> usize {
    let literals_size = estimate_literals(
        &store.literals,
        &entropy.huf,
        huf_metadata,
        write_lit_entropy,
    );
    let nb_seq = store.nb_seq();
    let header = 1 + 1 + usize::from(nb_seq >= 128) + usize::from(nb_seq >= LONG_NB_SEQ);
    // Without sequences no codes were made: every histogram is empty.
    let empty = CodeCounts::default();
    let counts = if nb_seq == 0 {
        &empty
    } else {
        &store.code_counts
    };
    let mut seq_size = estimate_symbol_type(
        fse_metadata.of_type,
        &counts.of,
        nb_seq,
        OFFSETS,
        &entropy.fse.offcode,
        None,
    );
    seq_size += estimate_symbol_type(
        fse_metadata.ll_type,
        &counts.ll,
        nb_seq,
        LITERAL_LENGTHS,
        &entropy.fse.litlength,
        Some(&LL_BITS),
    );
    seq_size += estimate_symbol_type(
        fse_metadata.ml_type,
        &counts.ml,
        nb_seq,
        MATCH_LENGTHS,
        &entropy.fse.matchlength,
        Some(&ML_BITS),
    );
    seq_size += fse_metadata.tables_size;
    seq_size + header + literals_size + BLOCK_HEADER_SIZE
}

/// `ZSTD_buildEntropyStatisticsAndEstimateSubBlockSize`.
fn estimate_sub_block_size(engine: &mut Engine, store: &mut SeqStore) -> Result<usize, CoreError> {
    let strategy = engine.applied.cparams.strategy;
    let disabled = engine.applied.literals_compression_disabled();
    let huf_metadata = build_literals_stats(
        &store.literals,
        &engine.prev.entropy.huf,
        &mut engine.next.entropy.huf,
        disabled,
        strategy >= Strategy::BtUltra,
    )?;
    let fse_metadata = build_sequences_stats(
        store,
        &engine.prev.entropy.fse,
        &mut engine.next.entropy.fse,
        strategy,
        engine.backend,
    )?;
    Ok(estimate_block_size(
        store,
        &engine.next.entropy,
        &huf_metadata,
        &fse_metadata,
        huf_metadata.kind == EncodingType::Compressed,
    ))
}

impl SeqStore {
    /// Literal bytes of sequences `[from, to)` (`ZSTD_countSeqStoreLiteralsBytes`).
    fn literal_bytes(&self, from: usize, to: usize, long_length: LongLength) -> usize {
        (from..to)
            .map(|i| {
                let extra = match long_length {
                    LongLength::Literal(pos) if pos as usize == i => 0x10000,
                    _ => 0,
                };
                usize::from(self.sequences[i].lit_length) + extra
            })
            .sum()
    }

    /// Match bytes of all sequences (`ZSTD_countSeqStoreMatchBytes`).
    fn match_bytes(&self) -> usize {
        (0..self.nb_seq())
            .map(|i| {
                let extra = match self.long_length {
                    LongLength::Match(pos) if pos as usize == i => 0x10000,
                    _ => 0,
                };
                usize::from(self.sequences[i].ml_base) + MIN_MATCH + extra
            })
            .sum()
    }

    /// The sequences `[start, end)` as their own store
    /// (`ZSTD_deriveSeqStoreChunk`).
    fn chunk(&self, start: usize, end: usize) -> SeqStore {
        let lit_start = if start > 0 {
            self.literal_bytes(0, start, self.long_length)
        } else {
            0
        };
        let long_length = match self.long_length {
            LongLength::None => LongLength::None,
            LongLength::Literal(pos) | LongLength::Match(pos)
                if (pos as usize) < start || pos as usize > end =>
            {
                LongLength::None
            }
            LongLength::Literal(pos) => LongLength::Literal(pos - start as u32),
            LongLength::Match(pos) => LongLength::Match(pos - start as u32),
        };
        let lit_end = if end == self.nb_seq() {
            self.literals.len()
        } else {
            let shifted = SeqStore {
                sequences: self.sequences[start..end].to_vec(),
                ..SeqStore::default()
            };
            lit_start + shifted.literal_bytes(0, end - start, long_length)
        };
        SeqStore {
            sequences: self.sequences[start..end].to_vec(),
            literals: self.literals[lit_start..lit_end].to_vec(),
            codes: Vec::new(),
            code_counts: CodeCounts::default(),
            long_length,
            max_nb_seq: self.max_nb_seq,
            max_nb_lit: self.max_nb_lit,
        }
    }
}

/// `ZSTD_deriveBlockSplitsHelper`.
fn derive_splits_helper(
    engine: &mut Engine,
    original: &SeqStore,
    splits: &mut Vec<u32>,
    start: usize,
    end: usize,
) {
    let mid = (start + end) / 2;
    if end - start < MIN_SEQUENCES_BLOCK_SPLITTING || splits.len() >= MAX_NB_BLOCK_SPLITS {
        return;
    }
    let mut full = original.chunk(start, end);
    let mut first = original.chunk(start, mid);
    let mut second = original.chunk(mid, end);
    let original_size = estimate_sub_block_size(engine, &mut full);
    let first_size = estimate_sub_block_size(engine, &mut first);
    let second_size = estimate_sub_block_size(engine, &mut second);
    let (Ok(original_size), Ok(first_size), Ok(second_size)) =
        (original_size, first_size, second_size)
    else {
        return;
    };
    if first_size + second_size < original_size {
        derive_splits_helper(engine, original, splits, start, mid);
        splits.push(mid as u32);
        derive_splits_helper(engine, original, splits, mid, end);
    }
}

/// `ZSTD_resolveRepcodeToRawOffset`.
fn resolve_repcode_to_raw_offset(rep: &[u32; REP_NUM], off_base: u32, ll0: bool) -> u32 {
    let adjusted = off_base - 1 + u32::from(ll0);
    if adjusted == REP_NUM as u32 {
        return rep[0].wrapping_sub(1);
    }
    rep[adjusted as usize]
}

/// `ZSTD_seqStore_resolveOffCodes`.
fn resolve_off_codes(d_rep: &mut [u32; REP_NUM], c_rep: &mut [u32; REP_NUM], store: &mut SeqStore) {
    let nb_seq = store.nb_seq();
    let long_lit_len_idx = match store.long_length {
        LongLength::Literal(pos) => pos as usize,
        _ => nb_seq,
    };
    for idx in 0..nb_seq {
        let seq = &mut store.sequences[idx];
        let ll0 = seq.lit_length == 0 && idx != long_lit_len_idx;
        let off_base = seq.off_base;
        if offbase_is_repcode(off_base) {
            let d_raw = resolve_repcode_to_raw_offset(d_rep, off_base, ll0);
            let c_raw = resolve_repcode_to_raw_offset(c_rep, off_base, ll0);
            if d_raw != c_raw {
                seq.off_base = offset_to_offbase(c_raw);
            }
        }
        update_rep(d_rep, seq.off_base, ll0);
        update_rep(c_rep, off_base, ll0);
    }
}

/// The repeat offset histories of a split block: as the decoder will see
/// them (`d`) and as the match finder produced them (`c`).
#[derive(Copy, Clone, Debug)]
struct RepHistories {
    d: [u32; REP_NUM],
    c: [u32; REP_NUM],
}

/// `ZSTD_compressSeqStore_singleBlock`.
fn compress_single_block(
    engine: &mut Engine,
    store: &mut SeqStore,
    reps: &mut RepHistories,
    dst: &mut [u8],
    src: &[u8],
    last_block: bool,
    is_partition: bool,
) -> Result<usize, CoreError> {
    const RLE_MAX_LENGTH: usize = 25;
    let d_rep_original = reps.d;
    if is_partition {
        resolve_off_codes(&mut reps.d, &mut reps.c, store);
    }
    if dst.len() < BLOCK_HEADER_SIZE {
        return Err(CoreError::DstSizeTooSmall);
    }
    let params = engine.applied;
    let mut c_seqs_size = block::entropy_compress(
        store,
        &engine.prev.entropy,
        &mut engine.next.entropy,
        &params,
        &mut dst[BLOCK_HEADER_SIZE..],
        src.len(),
        engine.backend,
    )?;
    if !engine.is_first_block && c_seqs_size < RLE_MAX_LENGTH && block::is_rle(src) {
        c_seqs_size = 1;
    }
    let c_size = match c_seqs_size {
        0 => {
            reps.d = d_rep_original;
            no_compress_block(dst, src, last_block)?
        }
        1 => {
            reps.d = d_rep_original;
            rle_compress_block(dst, src[0], src.len(), last_block)?
        }
        size => {
            engine.confirm_block_state();
            write_block_header(dst, BLOCK_COMPRESSED, size as u32, last_block);
            BLOCK_HEADER_SIZE + size
        }
    };
    engine.demote_offcode_repeat();
    Ok(c_size)
}

/// `ZSTD_compressBlock_splitBlock`.
pub(crate) fn compress_split_block(
    engine: &mut Engine,
    segments: &Segments<'_>,
    view: &mem::Window<'_>,
    start: u32,
    end: u32,
    dst: &mut [u8],
    last_block: bool,
) -> Result<usize, CoreError> {
    let src = view.prefix.slice(start, end);
    if block::build_seq_store(engine, segments, view, start, end)? == Build::NoCompress {
        engine.demote_offcode_repeat();
        return no_compress_block(dst, src, last_block);
    }
    let original = core::mem::take(&mut engine.seq_store);
    let result = split_block_internal(engine, &original, dst, src, last_block);
    engine.seq_store = original;
    result
}

/// `ZSTD_compressBlock_splitBlock_internal`.
fn split_block_internal(
    engine: &mut Engine,
    original: &SeqStore,
    dst: &mut [u8],
    src: &[u8],
    last_block: bool,
) -> Result<usize, CoreError> {
    let nb_seq = original.nb_seq();
    let mut partitions = Vec::new();
    if nb_seq > 4 {
        derive_splits_helper(engine, original, &mut partitions, 0, nb_seq);
    }
    let num_splits = partitions.len();
    partitions.push(nb_seq as u32);

    let mut reps = RepHistories {
        d: engine.prev.rep,
        c: engine.prev.rep,
    };

    if num_splits == 0 {
        let mut whole = original.clone();
        return compress_single_block(engine, &mut whole, &mut reps, dst, src, last_block, false);
    }

    let mut c_size = 0usize;
    let mut ip = 0usize;
    let mut op = 0usize;
    let mut src_bytes_total = 0usize;
    let mut chunk_start = 0usize;
    for (i, &chunk_end) in partitions[..=num_splits].iter().enumerate() {
        let chunk_end = chunk_end as usize;
        let mut chunk = original.chunk(chunk_start, chunk_end);
        let last_partition = i == num_splits;
        let mut src_bytes =
            chunk.literal_bytes(0, chunk.nb_seq(), chunk.long_length) + chunk.match_bytes();
        src_bytes_total += src_bytes;
        let mut last_block_entire_src = false;
        if last_partition {
            src_bytes += src.len() - src_bytes_total;
            last_block_entire_src = last_block;
        }
        let size = compress_single_block(
            engine,
            &mut chunk,
            &mut reps,
            &mut dst[op..],
            &src[ip..ip + src_bytes],
            last_block_entire_src,
            true,
        )?;
        ip += src_bytes;
        op += size;
        c_size += size;
        chunk_start = chunk_end;
    }
    engine.prev.rep = reps.d;
    Ok(c_size)
}

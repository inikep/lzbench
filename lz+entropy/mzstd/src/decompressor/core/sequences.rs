//! The sequences section of a compressed block (RFC 8878
//! section 3.1.1.3.2) and sequence execution (section 3.1.1.4).
//!
//! [`SequenceTables::read_header`] parses the sequence count and the three
//! table descriptions, updating the tables that `Repeat_Mode` reuses.
//! [`SequenceTables::execute`] then decodes the interleaved bitstream and
//! executes every sequence as soon as it is decoded: literals are copied to
//! the output, followed by a match copied from the frame's history.
//!
//! Matches may reach before the frame into a dictionary's content, which is
//! kept apart from the output, as the reference's `extDict` segment: such
//! sequences take the cold exact path, so frames without a dictionary run
//! the same loop as before.
//!
//! The loop is the reference's `ZSTD_decompressSequences_body`: each table
//! row carries its baseline and extra-bit count ([`SeqRow`]), the container
//! is refilled once per sequence and once more in the middle when the
//! sequence's extra bits could exhaust it, and sequences with room to spare
//! in both buffers are executed with wildcopies.

use alloc::boxed::Box;

use fearless_simd::Simd;

use super::BlockError;
use super::bits::BitContainer;
use super::block::Target;
use super::copy::{
    INDEX_LIMIT, MatchCopy, WILDCOPY_OVERLENGTH, exact_match, idx, wildcopy_literals,
    wildcopy_match,
};
use super::dictionary::DecoderEntropy;
use super::fse::{SEQ_TABLE_LOG_MAX, SEQ_TABLE_SIZE, SeqRow, SeqTable, read_distribution, seq_row};
use super::literals::{Literals, encoding_type};
use crate::decompressor::{DecodeLimits, InvalidDataKind};
use crate::shared::format::{
    BLOCK_SIZE_MAX, EncodingType, LL_BITS, LL_DEFAULT_NORM, LL_DEFAULT_NORM_LOG, LL_FSE_LOG,
    LONG_NB_SEQ, MAX_LL, MAX_ML, MAX_OFF, MAX_SEQ, MIN_MATCH, ML_BITS, ML_DEFAULT_NORM,
    ML_DEFAULT_NORM_LOG, ML_FSE_LOG, OF_DEFAULT_NORM, OF_DEFAULT_NORM_LOG, OFF_FSE_LOG, REP_NUM,
};

/// Baselines of a length code alphabet: each code starts where the range of
/// the previous one, `1 << extra_bits` values wide, ends.
const fn baselines<const N: usize>(bits: &[u8; N], first: u32) -> [u32; N] {
    let mut base = [0u32; N];
    let mut value = first;
    let mut code = 0;
    while code < N {
        base[code] = value;
        value += 1 << bits[code];
        code += 1;
    }
    base
}

/// Literal length baselines per code.
const LL_BASE: [u32; MAX_LL as usize + 1] = baselines(&LL_BITS, 0);
/// Match length baselines per code.
const ML_BASE: [u32; MAX_ML as usize + 1] = baselines(&ML_BITS, MIN_MATCH as u32);
/// Offset code `n` reads `n` extra bits.
const OF_BITS: [u8; MAX_OFF as usize + 1] = {
    let mut bits = [0u8; MAX_OFF as usize + 1];
    let mut code = 0;
    while code < bits.len() {
        bits[code] = code as u8;
        code += 1;
    }
    bits
};
/// Offset baselines per code: `(1 << n) - 3` folds the three repeat codes
/// of an offset value into the offset; codes 0 and 1 select repeat offsets
/// and keep the reference's values.
const OF_BASE: [u32; MAX_OFF as usize + 1] = {
    let mut base = [0u32; MAX_OFF as usize + 1];
    base[1] = 1;
    let mut code = 2;
    while code < base.len() {
        base[code] = (1u32 << code) - 3;
        code += 1;
    }
    base
};

// Every sequence symbol fits the `u8` symbols of an FSE table, and every
// table fits a `SeqTable`.
const _: () = assert!(MAX_LL <= MAX_SEQ && MAX_ML <= MAX_SEQ && MAX_OFF <= MAX_SEQ);
const _: () = assert!(
    LL_FSE_LOG <= SEQ_TABLE_LOG_MAX
        && ML_FSE_LOG <= SEQ_TABLE_LOG_MAX
        && OFF_FSE_LOG <= SEQ_TABLE_LOG_MAX
);

// The output is indexed with 32-bit masks (`copy::idx`): a block's offsets
// reach at most a window back, windows are at most `2^MAX_WINDOW_LOG`
// bytes, and a sequence's lengths come from baselines plus at most 16 extra
// bits, so every index stays far below `INDEX_LIMIT`.
const _: () = assert!(
    (1u64 << DecodeLimits::MAX_WINDOW_LOG) + 8 * (BLOCK_SIZE_MAX as u64) < INDEX_LIMIT as u64
);
const _: () = assert!(
    LL_BASE[MAX_LL as usize] < (1 << 17)
        && ML_BASE[MAX_ML as usize] < (1 << 17)
        && LL_BITS[MAX_LL as usize] <= 16
        && ML_BITS[MAX_ML as usize] <= 16
);

/// Most extra bits a sequence can read between two refills without
/// exhausting the container, given the state bits that follow: the
/// reference's `64 - 7 - (LLFSELog + MLFSELog + OffFSELog)`.
const MID_RELOAD_THRESHOLD: u32 = 64 - 7 - (LL_FSE_LOG + ML_FSE_LOG + OFF_FSE_LOG);

/// Static description of one symbol type's tables.
#[derive(Clone, Copy, Debug)]
struct SymbolKind {
    max_symbol: u32,
    max_log: u32,
    default: &'static [i16],
    default_log: u32,
    bases: &'static [u32],
    extra_bits: &'static [u8],
}

const LITERAL_LENGTHS: SymbolKind = SymbolKind {
    max_symbol: MAX_LL,
    max_log: LL_FSE_LOG,
    default: &LL_DEFAULT_NORM,
    default_log: LL_DEFAULT_NORM_LOG,
    bases: &LL_BASE,
    extra_bits: &LL_BITS,
};
const OFFSETS: SymbolKind = SymbolKind {
    max_symbol: MAX_OFF,
    max_log: OFF_FSE_LOG,
    default: &OF_DEFAULT_NORM,
    default_log: OF_DEFAULT_NORM_LOG,
    bases: &OF_BASE,
    extra_bits: &OF_BITS,
};
const MATCH_LENGTHS: SymbolKind = SymbolKind {
    max_symbol: MAX_ML,
    max_log: ML_FSE_LOG,
    default: &ML_DEFAULT_NORM,
    default_log: ML_DEFAULT_NORM_LOG,
    bases: &ML_BASE,
    extra_bits: &ML_BITS,
};

/// Where the history a block may reference starts, its limits, and the
/// dictionary the frame inherits.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockContext<'a> {
    /// Index in the output of the frame's first retained byte.
    pub(crate) history_start: usize,
    /// The frame's window size: the largest permitted offset.
    pub(crate) window: u64,
    /// Largest regenerated block size.
    pub(crate) block_max: usize,
    /// Dictionary content that precedes the frame's first byte, which is
    /// then at `history_start`; empty when the block may not reference
    /// one. Offsets reaching into it are not limited by the window.
    pub(crate) dictionary: &'a [u8],
    /// The dictionary's entropy tables, which repeat modes use until the
    /// frame replaces them.
    pub(crate) entropy: Option<&'a DecoderEntropy>,
}

/// What a [`TableSlot`] currently holds.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum SlotContent {
    /// Nothing `Repeat_Mode` may reuse.
    #[default]
    Empty,
    /// The predefined distribution, kept while no other table replaces it.
    Predefined,
    /// An RLE or a described distribution.
    Custom,
    /// The dictionary's table, inherited at the frame start; the slot's
    /// own table is not used.
    Dictionary,
}

/// One symbol type's table, allocated on first use and rebuilt in place.
#[derive(Debug, Default)]
struct TableSlot {
    table: Option<Box<SeqTable>>,
    content: SlotContent,
}

impl TableSlot {
    /// The table to rebuild, allocating it on first use.
    fn table_mut(&mut self) -> &mut SeqTable {
        self.table.get_or_insert_with(Box::default)
    }

    /// Replaces the table as `mode` asks, returning the description's size.
    fn update(
        &mut self,
        mode: EncodingType,
        kind: SymbolKind,
        src: &[u8],
    ) -> Result<usize, InvalidDataKind> {
        match mode {
            EncodingType::Basic => {
                if self.content != SlotContent::Predefined {
                    self.table_mut().fill(
                        kind.default,
                        kind.default_log,
                        kind.bases,
                        kind.extra_bits,
                    );
                    self.content = SlotContent::Predefined;
                }
                Ok(0)
            }
            EncodingType::Rle => {
                let &symbol = src.first().ok_or(InvalidDataKind::SequencesSection)?;
                if u32::from(symbol) > kind.max_symbol {
                    return Err(InvalidDataKind::SequencesSection);
                }
                let symbol = usize::from(symbol);
                self.table_mut()
                    .fill_rle(kind.bases[symbol], kind.extra_bits[symbol]);
                self.content = SlotContent::Custom;
                Ok(1)
            }
            EncodingType::Compressed => {
                let used = fill_described(self.table_mut(), kind, src)?;
                self.content = SlotContent::Custom;
                Ok(used)
            }
            EncodingType::Repeat => {
                if self.content == SlotContent::Empty {
                    return Err(InvalidDataKind::SequencesSection);
                }
                Ok(0)
            }
        }
    }

    /// The table `Repeat_Mode` and execution use, if any; `dictionary` is
    /// the dictionary's table of the same symbol type.
    fn get<'a>(&'a self, dictionary: Option<&'a SeqTable>) -> Option<&'a SeqTable> {
        match self.content {
            SlotContent::Empty => None,
            SlotContent::Dictionary => dictionary,
            SlotContent::Predefined | SlotContent::Custom => self.table.as_deref(),
        }
    }

    /// Heap bytes the slot holds.
    fn retained_bytes(&self) -> usize {
        self.table
            .as_ref()
            .map_or(0, |table| table.retained_bytes())
    }
}

/// Rebuilds `table` from the FSE table description at the start of `src`,
/// returning the description's size: `FSE_readNCount` and the accuracy
/// limit of `ZSTD_buildSeqTable` and `ZSTD_loadDEntropy`.
fn fill_described(
    table: &mut SeqTable,
    kind: SymbolKind,
    src: &[u8],
) -> Result<usize, InvalidDataKind> {
    let (distribution, used) = read_distribution(src, kind.max_symbol)?;
    if distribution.table_log > kind.max_log {
        return Err(InvalidDataKind::FseTable);
    }
    table.fill(
        distribution.counts(),
        distribution.table_log,
        kind.bases,
        kind.extra_bits,
    );
    Ok(used)
}

/// The three sequence symbol types.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SymbolType {
    /// Literal lengths.
    LiteralLengths,
    /// Offset codes.
    Offsets,
    /// Match lengths.
    MatchLengths,
}

/// Reads the FSE table description of `symbol` at the start of `src`, as
/// a Zstandard-format dictionary stores it, returning the table and the
/// description's size.
pub(crate) fn read_table(
    symbol: SymbolType,
    src: &[u8],
) -> Result<(SeqTable, usize), InvalidDataKind> {
    let kind = match symbol {
        SymbolType::LiteralLengths => LITERAL_LENGTHS,
        SymbolType::Offsets => OFFSETS,
        SymbolType::MatchLengths => MATCH_LENGTHS,
    };
    let mut table = SeqTable::default();
    let used = fill_described(&mut table, kind, src)?;
    Ok((table, used))
}

/// The decoding tables `Repeat_Mode` refers to, per frame.
#[derive(Debug, Default)]
pub(crate) struct SequenceTables {
    literal_lengths: TableSlot,
    offsets: TableSlot,
    match_lengths: TableSlot,
}

/// A block's sequences: how many, their bitstream, and the literals they
/// copy.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SequenceSection<'a> {
    /// Number of sequences.
    pub(crate) count: usize,
    /// The interleaved FSE bitstream.
    pub(crate) stream: &'a [u8],
    /// The block's literals.
    pub(crate) literals: Literals<'a>,
}

/// One decoded sequence.
#[derive(Clone, Copy, Debug, Default)]
struct Sequence {
    literal_length: u32,
    match_length: u32,
    offset: u32,
}

/// The decoding half of the sequence loop: the bitstream, the three FSE
/// states and the repeat offsets.
struct SequenceDecoder<'a> {
    bits: BitContainer,
    /// Literal length, offset and match length states.
    states: [usize; 3],
    repeat: [u32; REP_NUM],
    /// Literal length, offset and match length rows.
    rows: [&'a [SeqRow; SEQ_TABLE_SIZE]; 3],
    stream: &'a [u8],
}

impl SequenceDecoder<'_> {
    /// Decodes the next sequence, the reference's `ZSTD_decodeSequence`;
    /// the states advance unless it is the block's `last`. Fails on a
    /// repeat offset of zero.
    #[inline(always)]
    fn decode(&mut self, last: bool) -> Result<Sequence, InvalidDataKind> {
        let [ll_rows, of_rows, ml_rows] = self.rows;
        let [ll_state, of_state, ml_state] = self.states;
        let ll_row = seq_row(ll_rows, ll_state);
        let of_row = seq_row(of_rows, of_state);
        let ml_row = seq_row(ml_rows, ml_state);
        let bits = &mut self.bits;
        let repeat = &mut self.repeat;

        let of_bits = of_row.extra_bits();
        let offset = if of_bits > 1 {
            let offset = of_row
                .base()
                .wrapping_add(bits.look_nonzero(of_bits) as u32);
            bits.skip(of_bits);
            *repeat = [offset, repeat[0], repeat[1]];
            offset
        } else {
            let ll_zero = ll_row.base() == 0;
            if of_bits == 0 {
                if ll_zero {
                    repeat.swap(0, 1);
                }
                repeat[0]
            } else {
                let index = 1 + usize::from(ll_zero) + bits.read(1) as usize;
                let offset = if index == REP_NUM {
                    repeat[0].wrapping_sub(1)
                } else {
                    repeat[index]
                };
                if offset == 0 {
                    return Err(InvalidDataKind::Offset);
                }
                if index != 1 {
                    repeat[2] = repeat[1];
                }
                repeat[1] = repeat[0];
                repeat[0] = offset;
                offset
            }
        };
        let ml_bits = ml_row.extra_bits();
        let match_length = ml_row.base().wrapping_add(bits.read(ml_bits) as u32);
        let ll_bits = ll_row.extra_bits();
        if ll_bits + ml_bits + of_bits > MID_RELOAD_THRESHOLD {
            bits.reload(self.stream);
        }
        let literal_length = ll_row.base().wrapping_add(bits.read(ll_bits) as u32);
        if !last {
            // The format's order: literal length, match length, offset.
            let ll_state = ll_row.next_state() + bits.read(ll_row.nb_bits()) as usize;
            let ml_state = ml_row.next_state() + bits.read(ml_row.nb_bits()) as usize;
            let of_state = of_row.next_state() + bits.read(of_row.nb_bits()) as usize;
            self.states = [ll_state, of_state, ml_state];
            bits.reload(self.stream);
        }
        Ok(Sequence {
            literal_length,
            match_length,
            offset,
        })
    }
}

/// The executing half of the sequence loop: literals and matches copied
/// into the output, the reference's `ZSTD_execSequence`.
struct Executor<'a, 'b> {
    /// The output, from the lowest byte a match may reach, cut to the
    /// block's maximum and the overlength.
    out: &'a mut [u8],
    op: usize,
    /// The literals and exactly their slack.
    lits: &'b [u8],
    lp: usize,
    block_limit: usize,
    window: usize,
    /// Dictionary content before the output's first byte, read only by the
    /// cold path.
    dictionary: &'b [u8],
}

impl Executor<'_, '_> {
    /// Executes one sequence, checking it as the reference does.
    #[inline(always)]
    fn execute<S: Simd>(
        &mut self,
        simd: S,
        copy: MatchCopy,
        sequence: Sequence,
    ) -> Result<(), BlockError> {
        let literal_length = sequence.literal_length as usize;
        let match_length = sequence.match_length as usize;
        let offset = sequence.offset as usize;
        // Masked operands keep every sum below 2^34, so the compiler knows
        // they do not wrap and can relate the tests below to the copies'
        // own bounds checks.
        let (op, lp) = (idx(self.op), idx(self.lp));
        let lit_stop = lp + literal_length;
        if lit_stop + WILDCOPY_OVERLENGTH > self.lits.len() {
            return Err(InvalidDataKind::LiteralsOverrun.into());
        }
        let lit_end = op + literal_length;
        let match_end = lit_end + match_length;
        if match_end + WILDCOPY_OVERLENGTH <= self.out.len() {
            // Within the block's maximum, with room for the overlength.
            if offset > lit_end || offset > self.window {
                // Into a dictionary, past the window with one, or invalid.
                execute_exactly(
                    self.out,
                    self.lits,
                    [op, lit_end, match_end],
                    [lp, lit_stop],
                    offset,
                    [self.block_limit, self.window],
                    self.dictionary,
                )?;
                self.op = match_end;
                self.lp = lit_stop;
                return Ok(());
            }
            wildcopy_literals(self.out, op, self.lits, lp, literal_length);
            if offset >= 32 {
                // The first chunk here, where the room test above bounds
                // it; most matches need no other.
                // The history ends at the literals' end, so its last
                // `offset` bytes hold the whole chunk.
                let (history, tail) = self.out.split_at_mut(lit_end);
                let source = &history[lit_end - offset..];
                tail[..32].copy_from_slice(&source[..32]);
                if match_length > 32 {
                    wildcopy_match(
                        simd,
                        copy,
                        self.out,
                        lit_end + 32,
                        offset,
                        match_length - 32,
                    );
                }
            } else {
                wildcopy_match(simd, copy, self.out, lit_end, offset, match_length);
            }
        } else {
            execute_exactly(
                self.out,
                self.lits,
                [op, lit_end, match_end],
                [lp, lit_stop],
                offset,
                [self.block_limit, self.window],
                self.dictionary,
            )?;
        }
        self.op = match_end;
        self.lp = lit_stop;
        Ok(())
    }
}

/// Executes a sequence that ends too close to the end of the output for
/// chunked copies, past the block's maximum, or whose match leaves the
/// frame's output or window: the reference's `ZSTD_execSequenceEnd` and
/// the `extDict` branch of `ZSTD_execSequence`. Out of line and cold, so
/// the main loop keeps its registers. `ends` are the sequence's start,
/// literal end and match end in the output; `lit_range` its literals;
/// `limits` the block's end and the window.
///
/// With `dictionary` content before the output's first byte, an offset may
/// reach `dictionary.len()` bytes before it, whatever the window, as
/// `oLitEnd - virtualStart` bounds it in the reference; a match starting
/// there continues into the output.
#[cold]
#[inline(never)]
fn execute_exactly(
    out: &mut [u8],
    lits: &[u8],
    ends: [usize; 3],
    lit_range: [usize; 2],
    offset: usize,
    limits: [usize; 2],
    dictionary: &[u8],
) -> Result<(), BlockError> {
    let [op, lit_end, match_end] = ends;
    let [block_limit, window] = limits;
    if match_end > block_limit {
        return Err(InvalidDataKind::BlockSize.into());
    }
    let reach = if dictionary.is_empty() {
        lit_end.min(window)
    } else {
        lit_end + dictionary.len()
    };
    if offset > reach {
        return Err(InvalidDataKind::Offset.into());
    }
    if match_end > out.len() {
        return Err(BlockError::OutputFull);
    }
    out[op..lit_end].copy_from_slice(&lits[lit_range[0]..lit_range[1]]);
    let mut start = lit_end;
    if offset > lit_end {
        // The match starts `before` bytes before the output, in the
        // dictionary, and continues from the output's first byte.
        let before = offset - lit_end;
        let from = &dictionary[dictionary.len() - before..];
        let count = from.len().min(match_end - lit_end);
        out[lit_end..lit_end + count].copy_from_slice(&from[..count]);
        start += count;
        if start == match_end {
            return Ok(());
        }
    }
    exact_match(out, start, offset, match_end - start);
    Ok(())
}

impl SequenceTables {
    /// Heap bytes the tables hold.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.literal_lengths.retained_bytes()
            + self.offsets.retained_bytes()
            + self.match_lengths.retained_bytes()
    }

    /// Forgets every table at a frame start, keeping their allocations;
    /// with `dictionary_tables`, repeat modes reuse the dictionary's tables
    /// until the frame replaces them (the reference's `fseEntropy`).
    pub(crate) fn reset(&mut self, dictionary_tables: bool) {
        let content = if dictionary_tables {
            SlotContent::Dictionary
        } else {
            SlotContent::Empty
        };
        for slot in [
            &mut self.literal_lengths,
            &mut self.offsets,
            &mut self.match_lengths,
        ] {
            slot.content = content;
        }
    }

    /// Parses the sequences section header, returning the number of
    /// sequences and the header's size.
    pub(crate) fn read_header(&mut self, src: &[u8]) -> Result<(usize, usize), InvalidDataKind> {
        let invalid = InvalidDataKind::SequencesSection;
        let &first = src.first().ok_or(invalid)?;
        let (count, mut used) = match first {
            0..128 => (usize::from(first), 1),
            128..255 => {
                let &second = src.get(1).ok_or(invalid)?;
                (((usize::from(first) - 0x80) << 8) + usize::from(second), 2)
            }
            255 => {
                let bytes = src.get(1..3).ok_or(invalid)?;
                let value = usize::from(u16::from_le_bytes([bytes[0], bytes[1]]));
                (value + LONG_NB_SEQ, 3)
            }
        };
        if count == 0 {
            // Nothing may follow an empty sequences section.
            return if used == src.len() {
                Ok((0, used))
            } else {
                Err(invalid)
            };
        }
        let &modes = src.get(used).ok_or(invalid)?;
        if modes & 3 != 0 {
            return Err(invalid);
        }
        used += 1;
        let rest = |used: usize| src.get(used..).unwrap_or(&[]);
        used +=
            self.literal_lengths
                .update(encoding_type(modes >> 6), LITERAL_LENGTHS, rest(used))?;
        used += self
            .offsets
            .update(encoding_type(modes >> 4), OFFSETS, rest(used))?;
        used += self
            .match_lengths
            .update(encoding_type(modes >> 2), MATCH_LENGTHS, rest(used))?;
        Ok((count, used))
    }

    /// Decodes `count` sequences from `stream` and executes them, writing
    /// the block's output to `target.out[target.op..]`, whose prefix from
    /// `target.context.history_start` is the frame's history. Returns the
    /// end of the block's output.
    ///
    /// `reps` holds the repeat offsets on entry and is updated only when the
    /// whole block succeeds. [`BlockError::OutputFull`] reports that `out`
    /// ends before the block does.
    #[fearless_simd_macros::simd]
    pub(crate) fn execute<S: Simd>(
        &self,
        simd: S,
        copy: MatchCopy,
        section: SequenceSection<'_>,
        reps: &mut [u32; REP_NUM],
        target: Target<'_>,
    ) -> Result<usize, BlockError> {
        let SequenceSection {
            count,
            stream,
            literals,
        } = section;
        let Target { out, op, context } = target;
        // Valid matches never reach before `base`: offsets are bounded by
        // the history and by the window. Cutting the buffer there keeps
        // every index below 2^32 (windows are at most 2^31 bytes), which
        // the copies' index masks rely on.
        let window = usize::try_from(context.window).unwrap_or(usize::MAX);
        let base = context.history_start.max(op.saturating_sub(window));
        // The dictionary precedes the history's first byte, so it is out of
        // reach once the window starts after it.
        let dictionary = if base == context.history_start {
            context.dictionary
        } else {
            &[]
        };
        let cut_end = out.len().min(base.saturating_add(INDEX_LIMIT));
        let out = &mut out[base..cut_end];
        let op = op - base;
        let missing = InvalidDataKind::SequencesSection;
        let entropy = context.entropy;
        let ll_table = self
            .literal_lengths
            .get(entropy.map(|tables| &tables.literal_lengths))
            .ok_or(missing)?;
        let of_table = self
            .offsets
            .get(entropy.map(|tables| &tables.offsets))
            .ok_or(missing)?;
        let ml_table = self
            .match_lengths
            .get(entropy.map(|tables| &tables.match_lengths))
            .ok_or(missing)?;
        let bits = BitContainer::new(stream).ok_or(InvalidDataKind::SequenceStream)?;
        let mut decoder = SequenceDecoder {
            bits,
            states: [0; 3],
            repeat: *reps,
            rows: [ll_table.rows(), of_table.rows(), ml_table.rows()],
            stream,
        };
        decoder.states = [
            decoder.bits.read(ll_table.log()) as usize,
            decoder.bits.read(of_table.log()) as usize,
            decoder.bits.read(ml_table.log()) as usize,
        ];
        decoder.bits.reload(stream);

        // Both buffers are cut so that the fast path's tests are close to
        // the ones the copies' bounds checks make: the literals to exactly
        // their slack, the output to the block's maximum and the
        // overlength, where it is longer.
        let lit_len = literals.len;
        let lits = &literals.bytes[..lit_len + WILDCOPY_OVERLENGTH];
        let block_limit = op + context.block_max;
        let out_len = out.len().min(block_limit + WILDCOPY_OVERLENGTH);
        let mut executor = Executor {
            out: &mut out[..out_len],
            op,
            lits,
            lp: 0,
            block_limit,
            window,
            dictionary,
        };
        // The last sequence leaves the states alone; peeling it keeps that
        // test out of the loop. Callers handle empty sections themselves.
        debug_assert!(count > 0);
        let mut left = count - 1;
        while left != 0 {
            let sequence = decoder.decode(false)?;
            executor.execute(simd, copy, sequence)?;
            left -= 1;
        }
        let sequence = decoder.decode(true)?;
        executor.execute(simd, copy, sequence)?;
        if !decoder.bits.is_exhausted() {
            return Err(InvalidDataKind::SequenceStream.into());
        }
        let Executor { out, op, lp, .. } = executor;
        let rest = lit_len - lp;
        if op + rest > block_limit {
            return Err(InvalidDataKind::BlockSize.into());
        }
        let end = op + rest;
        let target = out.get_mut(op..end).ok_or(BlockError::OutputFull)?;
        target.copy_from_slice(&lits[lp..lit_len]);
        *reps = decoder.repeat;
        Ok(base + end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backend;
    use crate::shared::format::REP_START;
    use alloc::vec::Vec;

    /// Executes `count` sequences after `out` on every backend and copy
    /// mode, requiring them to agree, and appends the output.
    fn run(
        tables: &SequenceTables,
        count: usize,
        stream: &[u8],
        literals: &[u8],
        reps: &mut [u32; REP_NUM],
        out: &mut Vec<u8>,
        context: BlockContext<'_>,
    ) -> Result<(), InvalidDataKind> {
        let start = out.len();
        let mut bytes = literals.to_vec();
        bytes.resize(literals.len() + WILDCOPY_OVERLENGTH, 0xaa);
        let literals = Literals {
            bytes: &bytes,
            len: literals.len(),
        };
        let mut results = Vec::new();
        for backend in Backend::available() {
            for copy in [MatchCopy::Scalar, MatchCopy::Shuffle] {
                let mut buffer = out.clone();
                buffer.resize(start + context.block_max + WILDCOPY_OVERLENGTH, 0);
                let mut repeat = *reps;
                let target = Target {
                    out: &mut buffer,
                    op: start,
                    context,
                };
                let section = SequenceSection {
                    count,
                    stream,
                    literals,
                };
                let result = fearless_simd::dispatch!(backend.0, simd => tables.execute(
                    simd, copy, section, &mut repeat, target
                ));
                results.push(result.map(|end| (buffer[start..end].to_vec(), repeat)));
            }
        }
        assert!(results.windows(2).all(|pair| pair[0] == pair[1]));
        match results.swap_remove(0) {
            Ok((bytes, repeat)) => {
                out.extend_from_slice(&bytes);
                *reps = repeat;
                Ok(())
            }
            Err(BlockError::Invalid(kind)) => Err(kind),
            Err(BlockError::OutputFull) => panic!("the buffer holds a whole block"),
        }
    }

    const CONTEXT: BlockContext<'static> = BlockContext {
        history_start: 0,
        window: 1 << 20,
        block_max: 1 << 17,
        dictionary: &[],
        entropy: None,
    };

    #[test]
    fn baselines_match_the_rfc_tables() {
        // The tables are computed at compile time; recompute them at run
        // time too.
        assert_eq!(baselines(&LL_BITS, 0), LL_BASE);
        assert_eq!(baselines(&ML_BITS, 3), ML_BASE);
        assert_eq!(
            &LL_BASE[..16],
            &::core::array::from_fn::<u32, 16, _>(|i| i as u32)
        );
        assert_eq!(
            &LL_BASE[16..],
            &[
                16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                16384, 32768, 65536
            ]
        );
        assert_eq!(ML_BASE[0], 3);
        assert_eq!(ML_BASE[31], 34);
        assert_eq!(
            &ML_BASE[32..],
            &[
                35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051, 4099, 8195,
                16387, 32771, 65539
            ]
        );
    }

    #[test]
    fn sequence_counts_use_one_two_or_three_bytes() {
        let mut tables = SequenceTables::default();
        assert_eq!(tables.read_header(&[0]), Ok((0, 1)));
        assert_eq!(
            tables.read_header(&[0, 0]),
            Err(InvalidDataKind::SequencesSection)
        );
        // 0x81 0x02 → 0x102 sequences, predefined modes.
        assert_eq!(tables.read_header(&[0x81, 0x02, 0x00]), Ok((0x102, 3)));
        assert_eq!(
            tables.read_header(&[0xff, 0x01, 0x00, 0x00]),
            Ok((LONG_NB_SEQ + 1, 4))
        );
        assert_eq!(
            tables.read_header(&[0xff, 0x01]),
            Err(InvalidDataKind::SequencesSection)
        );
        assert_eq!(
            tables.read_header(&[0x81]),
            Err(InvalidDataKind::SequencesSection)
        );
        assert_eq!(
            tables.read_header(&[1]),
            Err(InvalidDataKind::SequencesSection)
        );
    }

    #[test]
    fn reserved_mode_bits_and_bad_tables_are_rejected() {
        let mut tables = SequenceTables::default();
        assert_eq!(
            tables.read_header(&[1, 0x01]),
            Err(InvalidDataKind::SequencesSection)
        );
        // Repeat mode before any table.
        assert_eq!(
            tables.read_header(&[1, 0xc0]),
            Err(InvalidDataKind::SequencesSection)
        );
        // RLE literal length symbol above 35, and a missing RLE byte.
        assert_eq!(
            tables.read_header(&[1, 0x40, 36]),
            Err(InvalidDataKind::SequencesSection)
        );
        assert_eq!(
            tables.read_header(&[1, 0x40]),
            Err(InvalidDataKind::SequencesSection)
        );
        // An offset table with accuracy log 9 (above 8): low nibble 4.
        assert_eq!(
            tables.read_header(&[1, 0x20, 0x04, 0, 0, 0]),
            Err(InvalidDataKind::FseTable)
        );
        // Once a block set the tables, repeat mode reuses them.
        assert_eq!(tables.read_header(&[1, 0x54, 1, 2, 3]), Ok((1, 5)));
        assert_eq!(tables.read_header(&[1, 0xfc]), Ok((1, 2)));
    }

    /// Writes a sequence bitstream: fields are listed in the order the
    /// decoder reads them, and the marker bit goes on top.
    fn backward_stream(fields: &[(u64, u32)]) -> Vec<u8> {
        let total: u32 = fields.iter().map(|&(_, width)| width).sum();
        let mut value: u128 = 1;
        for &(field, width) in fields {
            value = (value << width) | u128::from(field);
        }
        let bytes = (total as usize + 1).div_ceil(8);
        (0..bytes).map(|i| (value >> (8 * i)) as u8).collect()
    }

    /// RLE tables: literal length code `ll`, offset code `of`, match length
    /// code `ml`.
    fn rle_tables(ll: u8, of: u8, ml: u8) -> SequenceTables {
        let mut tables = SequenceTables::default();
        tables
            .read_header(&[1, 0x54, ll, of, ml])
            .expect("valid RLE tables");
        tables
    }

    #[test]
    fn a_literal_run_and_an_overlapping_match_execute() {
        // Literal length 3, offset code 2 with extra bits 2 (offset value
        // 4 + 2 = 6, offset 3), match length code 2 (5 bytes).
        let tables = rle_tables(3, 2, 2);
        let stream = backward_stream(&[(2, 2)]);
        let mut reps = REP_START;
        let mut out = Vec::new();
        run(&tables, 1, &stream, b"abcXY", &mut reps, &mut out, CONTEXT).expect("valid");
        assert_eq!(out, b"abcabcabXY");
        assert_eq!(reps, [3, 1, 4]);
    }

    #[test]
    fn repeat_offsets_follow_the_update_rules() {
        // Offset code 1 reads one bit: index 1 + bit, or 2 + bit when the
        // literal length is zero.
        let mut out = b"0123456789".to_vec();
        let context = BlockContext {
            history_start: 0,
            ..CONTEXT
        };
        // Literal length 1, offset code 1 with bit 1 → Repeated_Offset3 = 8.
        let tables = rle_tables(1, 1, 0);
        let mut reps = REP_START;
        run(
            &tables,
            1,
            &backward_stream(&[(1, 1)]),
            b"z",
            &mut reps,
            &mut out,
            context,
        )
        .expect("valid");
        assert_eq!(reps, [8, 1, 4]);
        // Literal length 0, offset code 0 → Repeated_Offset2.
        let tables = rle_tables(0, 0, 0);
        run(&tables, 1, &[0x01], b"", &mut reps, &mut out, context).expect("valid");
        assert_eq!(reps, [1, 8, 4]);
        // Literal length 0, offset code 1 with bit 1 → Repeated_Offset1 - 1
        // = 0, which is corrupt.
        let tables = rle_tables(0, 1, 0);
        assert_eq!(
            run(
                &tables,
                1,
                &backward_stream(&[(1, 1)]),
                b"",
                &mut reps,
                &mut out,
                context
            ),
            Err(InvalidDataKind::Offset)
        );
        // Literal length 0, offset code 1 with bit 0 → Repeated_Offset3.
        run(
            &tables,
            1,
            &backward_stream(&[(0, 1)]),
            b"",
            &mut reps,
            &mut out,
            context,
        )
        .expect("valid");
        assert_eq!(reps, [4, 1, 8]);
        // Offset code 0 with literals keeps the order.
        let tables = rle_tables(1, 0, 0);
        run(&tables, 1, &[0x01], b"q", &mut reps, &mut out, context).expect("valid");
        assert_eq!(reps, [4, 1, 8]);
    }

    #[test]
    fn invalid_sequences_are_rejected() {
        let mut reps = REP_START;
        let mut out = Vec::new();
        // Offset 3 with no history.
        let tables = rle_tables(0, 2, 0);
        assert_eq!(
            run(
                &tables,
                1,
                &backward_stream(&[(2, 2)]),
                b"",
                &mut reps,
                &mut out,
                CONTEXT
            ),
            Err(InvalidDataKind::Offset)
        );
        // More literals than the section holds.
        let tables = rle_tables(3, 0, 0);
        assert_eq!(
            run(&tables, 1, &[0x01], b"ab", &mut reps, &mut out, CONTEXT),
            Err(InvalidDataKind::LiteralsOverrun)
        );
        // Leftover bits, and a missing end marker.
        assert_eq!(
            run(&tables, 1, &[0x03], b"abc", &mut reps, &mut out, CONTEXT),
            Err(InvalidDataKind::SequenceStream)
        );
        assert_eq!(
            run(&tables, 1, &[0x00], b"abc", &mut reps, &mut out, CONTEXT),
            Err(InvalidDataKind::SequenceStream)
        );
        // A block larger than its maximum.
        let small = BlockContext {
            block_max: 2,
            ..CONTEXT
        };
        assert_eq!(
            run(&tables, 1, &[0x01], b"abc", &mut reps, &mut out, small),
            Err(InvalidDataKind::BlockSize)
        );
        let tables = rle_tables(0, 0, 0);
        assert_eq!(
            run(
                &tables,
                1,
                &[0x01],
                b"abc",
                &mut reps,
                &mut b"x".to_vec(),
                small
            ),
            Err(InvalidDataKind::BlockSize)
        );
        // Offsets beyond the window.
        let narrow = BlockContext {
            window: 0,
            ..CONTEXT
        };
        assert_eq!(
            run(
                &tables,
                1,
                &[0x01],
                b"",
                &mut reps,
                &mut b"abcdef".to_vec(),
                narrow
            ),
            Err(InvalidDataKind::Offset)
        );
        assert_eq!(reps, REP_START);
        // Tables are required.
        assert_eq!(
            run(
                &SequenceTables::default(),
                1,
                &[0x01],
                b"",
                &mut reps,
                &mut out,
                CONTEXT
            ),
            Err(InvalidDataKind::SequencesSection)
        );
    }

    #[test]
    fn matches_reach_into_the_dictionary_before_the_output() {
        let dictionary = b"0123456789";
        let context = BlockContext {
            dictionary,
            ..CONTEXT
        };
        // Literal length 1, match length 5, offset code 3 with extra bits
        // 0b011: offset 8, seven bytes into the dictionary.
        let tables = rle_tables(1, 3, 2);
        let stream = backward_stream(&[(3, 3)]);
        let mut reps = REP_START;
        let mut out = Vec::new();
        run(&tables, 1, &stream, b"z", &mut reps, &mut out, context).expect("valid");
        assert_eq!(out, b"z34567");
        // Match length 10 at offset 3, from two bytes before the output:
        // "89", then on into the output's own bytes.
        let tables = rle_tables(1, 2, 7);
        let mut out = Vec::new();
        run(
            &tables,
            1,
            &backward_stream(&[(2, 2)]),
            b"z",
            &mut reps,
            &mut out,
            context,
        )
        .expect("valid");
        assert_eq!(out, b"z89z89z89z8");
        // A match entirely inside the dictionary.
        let mut out = Vec::new();
        let tables = rle_tables(0, 3, 0);
        run(&tables, 1, &stream, b"", &mut reps, &mut out, context).expect("valid");
        assert_eq!(out, b"234");
        // Offset 12 reaches before the dictionary; 11 is its first byte.
        let tables = rle_tables(1, 3, 0);
        let mut out = Vec::new();
        assert_eq!(
            run(
                &tables,
                1,
                &backward_stream(&[(7, 3)]),
                b"z",
                &mut reps,
                &mut out,
                context
            ),
            Err(InvalidDataKind::Offset)
        );
        run(
            &tables,
            1,
            &backward_stream(&[(6, 3)]),
            b"z",
            &mut reps,
            &mut out,
            context,
        )
        .expect("valid");
        assert_eq!(out, b"z012");
        // With a dictionary, offsets inside the output may pass the window:
        // literal length 3 and offset 10 in a 9-byte window.
        let wide = BlockContext {
            window: 9,
            ..context
        };
        let tables = rle_tables(3, 3, 0);
        let mut out = b"abcdefgh".to_vec();
        run(
            &tables,
            1,
            &backward_stream(&[(5, 3)]),
            b"xyz",
            &mut reps,
            &mut out,
            wide,
        )
        .expect("valid");
        assert_eq!(out, b"abcdefghxyzbcd");
        // Once the window cuts the history short, the dictionary is out of
        // reach and the window limits offsets again.
        let narrow = BlockContext {
            window: 1,
            ..context
        };
        let tables = rle_tables(0, 2, 0);
        assert_eq!(
            run(
                &tables,
                1,
                &backward_stream(&[(3, 2)]),
                b"",
                &mut reps,
                &mut b"abcdefgh".to_vec(),
                narrow,
            ),
            Err(InvalidDataKind::Offset)
        );
    }

    #[test]
    fn dictionary_tables_serve_repeat_modes_until_replaced() {
        let bytes = crate::decompressor::core::test_dictionary([1, 2, 3], b"0123456789");
        let (entropy, _) = crate::decompressor::core::DecoderEntropy::load(&bytes).expect("valid");
        let mut tables = SequenceTables::default();
        tables.reset(true);
        assert_eq!(tables.read_header(&[1, 0xfc]), Ok((1, 2)));
        let context = BlockContext {
            entropy: Some(&entropy),
            ..CONTEXT
        };
        // Fifteen zero state bits; every symbol is code 0: literal length
        // 0, repeat offset 2 (as the literal length is zero), match length
        // 3, all from the history.
        let mut reps = entropy.reps;
        let mut out = b"abcdef".to_vec();
        run(&tables, 1, &[0x00, 0x80], b"", &mut reps, &mut out, context).expect("valid");
        assert_eq!(out, b"abcdefefe");
        assert_eq!(reps, [2, 1, 3]);
        // Without the dictionary's tables there is nothing to execute.
        assert_eq!(
            run(&tables, 1, &[0x00, 0x80], b"", &mut reps, &mut out, CONTEXT),
            Err(InvalidDataKind::SequencesSection)
        );
        // A described table replaces the dictionary's.
        assert_eq!(tables.read_header(&[1, 0x00]), Ok((1, 2)));
        assert!(tables.literal_lengths.get(None).is_some());
    }

    #[test]
    fn dictionary_tables_read_like_block_tables() {
        // Accuracy log 5 with all states on symbol 0 (see the dictionary
        // tests), then a log above each type's limit.
        let single = [0xf0, 0x03];
        for symbol in [
            SymbolType::LiteralLengths,
            SymbolType::Offsets,
            SymbolType::MatchLengths,
        ] {
            let (table, used) = read_table(symbol, &single).expect("valid");
            assert_eq!((table.log(), used), (5, 2));
        }
        // Log 9 (nibble 4) is beyond the offsets' limit of 8.
        assert_eq!(
            read_table(SymbolType::Offsets, &[0x04, 0, 0, 0]).map(|_| ()),
            Err(InvalidDataKind::FseTable)
        );
    }

    #[test]
    fn reset_forgets_repeat_tables() {
        let mut tables = rle_tables(0, 0, 0);
        tables.reset(false);
        assert_eq!(
            tables.read_header(&[1, 0xfc]),
            Err(InvalidDataKind::SequencesSection)
        );
    }
}

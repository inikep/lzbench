//! FSE (tANS) table descriptions, decoding tables and the two-state weight
//! decoder used by Huffman tree descriptions.
//!
//! [`read_distribution`] parses a normalized distribution (RFC 8878
//! section 4.1.1), [`FseTable::from_distribution`] spreads it into a decoding
//! table (section 4.1.1.1, Appendix A), and [`decode_weights`] runs the
//! interleaved two-state decoder over a Huffman weight stream
//! (section 4.2.1.2).
//!
//! Sequence decoding uses [`SeqTable`] instead of [`FseTable`]: each row
//! also carries its symbol's baseline and extra-bit count, as the
//! reference's `ZSTD_seqSymbol` does, so a sequence field costs one table
//! lookup and one bit read.

use alloc::boxed::Box;

use super::bits::{BitContainer, ForwardBitReader};
use crate::decompressor::InvalidDataKind;
use crate::shared::format::{MAX_LIT, highbit32};

/// Smallest accuracy log a table description can declare.
const MIN_TABLE_LOG: u32 = 5;
/// Largest accuracy log a table description can encode.
const MAX_TABLE_LOG_DESCRIBED: u32 = 15;
/// Largest accuracy log of a Huffman weight table.
const WEIGHT_TABLE_LOG_MAX: u32 = 6;
/// Weights a compressed Huffman description can produce: every literal but
/// the implied last one.
pub(crate) const MAX_WEIGHTS: usize = MAX_LIT as usize;

/// One decoding-table row.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct FseEntry {
    /// Decoded symbol.
    pub(crate) symbol: u8,
    /// Bits to read for the next state.
    pub(crate) nb_bits: u8,
    /// Base of the next state.
    pub(crate) new_state: u16,
}

/// Symbols a distribution can describe.
const MAX_SYMBOLS: usize = MAX_LIT as usize + 1;

/// A normalized distribution: `-1` marks a "less than one" probability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Distribution {
    /// Probabilities of symbols `0..len`; the rest are zero.
    probabilities: [i16; MAX_SYMBOLS],
    /// Symbols described.
    len: usize,
    /// Accuracy log.
    pub(crate) table_log: u32,
}

impl Distribution {
    /// Probabilities of the described symbols.
    pub(crate) fn counts(&self) -> &[i16] {
        &self.probabilities[..self.len.min(MAX_SYMBOLS)]
    }

    /// Appends `count` zero probabilities. The count may overshoot the
    /// symbol limit, which [`read_distribution`] then rejects.
    const fn push_zeros(&mut self, count: usize) {
        self.len += count;
    }

    /// Appends one probability; the caller stays within the symbol limit.
    fn push(&mut self, probability: i16) {
        if let Some(slot) = self.probabilities.get_mut(self.len) {
            *slot = probability;
        }
        self.len += 1;
    }
}

/// Largest accuracy log of a table [`FseTable`] holds.
const FSE_TABLE_LOG_MAX: u32 = WEIGHT_TABLE_LOG_MAX;
/// Rows of the largest [`FseTable`].
const FSE_TABLE_SIZE: usize = 1 << FSE_TABLE_LOG_MAX;

/// An FSE decoding table of `1 << log` rows, for Huffman weights.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FseTable {
    log: u32,
    entries: [FseEntry; FSE_TABLE_SIZE],
}

/// Largest accuracy log of a sequence table (literal and match lengths).
pub(crate) const SEQ_TABLE_LOG_MAX: u32 = 9;
/// Rows of the largest [`SeqTable`]; smaller tables use a prefix.
pub(crate) const SEQ_TABLE_SIZE: usize = 1 << SEQ_TABLE_LOG_MAX;
/// Room for the largest sequence alphabet (match lengths, 53 codes).
pub(crate) const SEQ_SYMBOLS: usize = 64;
/// Masks a state into [`SEQ_TABLE_SIZE`].
const SEQ_STATE_MASK: usize = SEQ_TABLE_SIZE - 1;

/// One sequence decoding row: the next state's base and bit count, and the
/// decoded symbol's baseline and extra-bit count; the reference's
/// `ZSTD_seqSymbol`.
///
/// The sequence loop holds references to rows and reads each field when it
/// needs it, as the reference does, rather than three rows' worth of
/// fields in registers it does not have.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SeqRow {
    base: u32,
    next_state: u16,
    extra_bits: BitCount,
    nb_bits: BitCount,
}

/// A bit count below 32: every sequence field and state transition reads
/// fewer bits. As an enum its range reaches the compiler with each load,
/// so reads by these counts need no masking of the shift amount.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
enum BitCount {
    #[default]
    B0,
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
    B8,
    B9,
    B10,
    B11,
    B12,
    B13,
    B14,
    B15,
    B16,
    B17,
    B18,
    B19,
    B20,
    B21,
    B22,
    B23,
    B24,
    B25,
    B26,
    B27,
    B28,
    B29,
    B30,
    B31,
}

impl BitCount {
    /// The count `count`, which must be below 32. A match makes the
    /// conversion an identity operation after range masking; indexing an
    /// array of the variants would require a table load at runtime.
    const fn new(count: u8) -> Self {
        debug_assert!(count < 32);
        match count & 31 {
            0 => Self::B0,
            1 => Self::B1,
            2 => Self::B2,
            3 => Self::B3,
            4 => Self::B4,
            5 => Self::B5,
            6 => Self::B6,
            7 => Self::B7,
            8 => Self::B8,
            9 => Self::B9,
            10 => Self::B10,
            11 => Self::B11,
            12 => Self::B12,
            13 => Self::B13,
            14 => Self::B14,
            15 => Self::B15,
            16 => Self::B16,
            17 => Self::B17,
            18 => Self::B18,
            19 => Self::B19,
            20 => Self::B20,
            21 => Self::B21,
            22 => Self::B22,
            23 => Self::B23,
            24 => Self::B24,
            25 => Self::B25,
            26 => Self::B26,
            27 => Self::B27,
            28 => Self::B28,
            29 => Self::B29,
            30 => Self::B30,
            _ => Self::B31,
        }
    }
}

impl SeqRow {
    /// A row; both bit counts are below 32.
    pub(crate) const fn new(base: u32, next_state: u16, extra_bits: u8, nb_bits: u8) -> Self {
        Self {
            base,
            next_state,
            extra_bits: BitCount::new(extra_bits),
            nb_bits: BitCount::new(nb_bits),
        }
    }

    /// Baseline of the decoded symbol's value.
    #[inline(always)]
    pub(crate) const fn base(&self) -> u32 {
        self.base
    }

    /// Base of the next state.
    #[inline(always)]
    pub(crate) const fn next_state(&self) -> usize {
        self.next_state as usize
    }

    /// Extra bits that follow the baseline.
    #[inline(always)]
    pub(crate) const fn extra_bits(&self) -> u32 {
        self.extra_bits as u32
    }

    /// Bits to read for the next state.
    #[inline(always)]
    pub(crate) const fn nb_bits(&self) -> u32 {
        self.nb_bits as u32
    }
}

/// A sequence decoding table of `1 << log` rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SeqTable {
    log: u32,
    rows: Box<[SeqRow; SEQ_TABLE_SIZE]>,
}

impl Default for SeqTable {
    fn default() -> Self {
        Self {
            log: 0,
            rows: Box::new([SeqRow::default(); SEQ_TABLE_SIZE]),
        }
    }
}

/// Spreads a distribution over `1 << table_log` cells (RFC 8878 section
/// 4.1.1.1), at most `N`, writing each cell's symbol into `symbols`, and
/// leaves in `next` the first state of each symbol: the reference's
/// `ZSTD_buildFSETable_body`. `M` is `N + 8`, the room the fast path's
/// eight-byte stores need.
///
/// Without "less than one" symbols, which the reference's encoder avoids
/// for small blocks, the symbols are first laid down in order eight at a
/// time and then scattered with a fixed step, which needs no per-cell
/// tests; otherwise each cell skips the cells those symbols hold.
#[inline(always)]
fn spread<const N: usize, const M: usize>(
    counts: &[i16],
    table_log: u32,
    symbols: &mut [u8; N],
    next: &mut [u16; MAX_SYMBOLS],
) {
    const { assert!(N.is_power_of_two() && M == N + 8) };
    let size = 1usize << table_log;
    debug_assert!(size <= N);
    // Every index below is masked into the table, which changes nothing
    // for the distributions `read_distribution` accepts (they fill the
    // table exactly) and spares the bounds checks.
    let mask = (size - 1) & (N - 1);
    let step = (size >> 1) + (size >> 3) + 3;
    // "Less than one" symbols take single cells from the end.
    let mut high = size;
    for ((symbol, &count), next) in counts.iter().enumerate().zip(next.iter_mut()) {
        if count == -1 {
            high -= 1;
            symbols[high & (N - 1)] = symbol as u8;
            *next = 1;
        } else {
            *next = count.max(0) as u16;
        }
    }
    if high == size {
        let mut laid = [0u8; M];
        let mut position = 0usize;
        let mut pattern = 0u64;
        for &count in counts {
            let count = count.max(0) as usize;
            let mut offset = 0;
            loop {
                // A symbol without cells after the last cell writes into
                // the slack, as the reference's does.
                let at = (position + offset).min(N);
                laid[at..at + 8].copy_from_slice(&pattern.to_le_bytes());
                offset += 8;
                if offset >= count {
                    break;
                }
            }
            position += count;
            pattern = pattern.wrapping_add(0x0101_0101_0101_0101);
        }
        let mut position = 0;
        // Tables have at least 32 cells, so they pair up.
        for &[first, second] in laid[..size].as_chunks::<2>().0 {
            symbols[position & mask] = first;
            symbols[(position + step) & mask] = second;
            position = (position + 2 * step) & mask;
        }
    } else {
        let mut position = 0;
        for (symbol, &count) in counts.iter().enumerate() {
            for _ in 0..count.max(0) {
                symbols[position] = symbol as u8;
                position = (position + step) & mask;
                while position >= high {
                    position = (position + step) & mask;
                }
            }
        }
    }
}

/// The bit count and next-state base of a cell whose symbol's state
/// counter reads `next`, in a table of `1 << table_log` cells.
const fn transition(next: u16, table_log: u32) -> (u8, u16) {
    // Every placed symbol has a non-zero count; the low bit, which moves
    // no other count's top bit, only keeps the arithmetic total for a
    // table nothing else would accept.
    let state = next as u32;
    let nb_bits = table_log.wrapping_sub(highbit32(state | 1)) & 31;
    (
        nb_bits as u8,
        (state << nb_bits).wrapping_sub(1u32 << table_log) as u16,
    )
}

/// Reads a normalized distribution describing symbols `0..=max_symbol`.
///
/// Returns the distribution and the number of bytes it occupies. As in the
/// reference decoder, the probabilities must sum exactly to the table size
/// and no symbol above `max_symbol` may appear.
pub(crate) fn read_distribution(
    src: &[u8],
    max_symbol: u32,
) -> Result<(Distribution, usize), InvalidDataKind> {
    let invalid = InvalidDataKind::FseTable;
    let symbol_limit = max_symbol as usize + 1;
    let mut bits = ForwardBitReader::new(src);
    let table_log = bits.read(4) + MIN_TABLE_LOG;
    if table_log > MAX_TABLE_LOG_DESCRIBED {
        return Err(invalid);
    }
    let mut counts = Distribution {
        probabilities: [0; MAX_SYMBOLS],
        len: 0,
        table_log,
    };
    let mut remaining: i32 = (1 << table_log) + 1;
    let mut threshold: i32 = 1 << table_log;
    let mut nb_bits = table_log + 1;
    let mut previous_zero = false;
    loop {
        if previous_zero {
            // Two-bit repeat flags; a value of 3 continues the run.
            loop {
                let repeat = bits.read(2) as usize;
                counts.push_zeros(repeat);
                if repeat != 3 || counts.len >= symbol_limit {
                    break;
                }
            }
            if counts.len >= symbol_limit {
                break;
            }
        }
        let max = (2 * threshold - 1) - remaining;
        let low = bits.peek(nb_bits - 1) as i32;
        let mut value = if low < max {
            bits.consume(nb_bits - 1);
            low
        } else {
            let full = bits.peek(nb_bits) as i32;
            bits.consume(nb_bits);
            if full >= threshold { full - max } else { full }
        };
        value -= 1;
        remaining -= value.abs();
        counts.push(value as i16);
        previous_zero = value == 0;
        if remaining < threshold {
            if remaining <= 1 {
                break;
            }
            nb_bits = highbit32(remaining as u32) + 1;
            threshold = 1 << (nb_bits - 1);
        }
        if counts.len >= symbol_limit {
            break;
        }
    }
    if remaining != 1 || counts.len > symbol_limit {
        return Err(invalid);
    }
    let used = bits.bytes_used();
    if used > src.len() {
        return Err(invalid);
    }
    Ok((counts, used))
}

impl FseTable {
    /// Builds the decoding table of a distribution whose probabilities sum
    /// to `1 << table_log`, at most `1 << 6`.
    pub(crate) fn from_distribution(counts: &[i16], table_log: u32) -> Self {
        debug_assert!(table_log <= FSE_TABLE_LOG_MAX);
        let size = 1usize << table_log;
        let mut symbols = [0u8; FSE_TABLE_SIZE];
        let mut next = [0u16; MAX_SYMBOLS];
        spread::<FSE_TABLE_SIZE, { FSE_TABLE_SIZE + 8 }>(
            counts,
            table_log,
            &mut symbols,
            &mut next,
        );
        let mut entries = [FseEntry::default(); FSE_TABLE_SIZE];
        for (entry, &symbol) in entries.iter_mut().zip(&symbols[..size]) {
            let next_state = &mut next[usize::from(symbol)];
            let (nb_bits, new_state) = transition(*next_state, table_log);
            *entry = FseEntry {
                symbol,
                nb_bits,
                new_state,
            };
            *next_state += 1;
        }
        Self {
            log: table_log,
            entries,
        }
    }

    /// Accuracy log: the bits an initial state takes.
    pub(crate) const fn log(&self) -> u32 {
        self.log
    }

    /// The row for `state`. States come from this table's own transitions,
    /// so they are always in range; the mask only spares the bounds check.
    #[inline(always)]
    pub(crate) fn entry(&self, state: usize) -> FseEntry {
        self.entries[state & (FSE_TABLE_SIZE - 1)]
    }

    /// Decodes the symbol of `state` and advances it past the symbol.
    #[inline(always)]
    fn decode(&self, state: &mut usize, bits: &mut BitContainer) -> u8 {
        let entry = self.entry(*state);
        *state = usize::from(entry.new_state) + bits.read(u32::from(entry.nb_bits)) as usize;
        entry.symbol
    }
}

impl SeqTable {
    /// Symbol baselines and extra-bit counts, built once per alphabet.
    pub(crate) const fn symbol_rows(bases: &[u32], extra_bits: &[u8]) -> [SeqRow; SEQ_SYMBOLS] {
        let mut rows = [SeqRow::new(0, 0, 0, 0); SEQ_SYMBOLS];
        let mut symbol = 0;
        while symbol < bases.len() && symbol < extra_bits.len() && symbol < SEQ_SYMBOLS {
            rows[symbol] = SeqRow::new(bases[symbol], 0, extra_bits[symbol], 0);
            symbol += 1;
        }
        rows
    }

    /// Builds the rows of a valid predefined distribution at compile time.
    /// The scalar spreading walk is the RFC's table construction, independent
    /// of the runtime fill's unrolled fast path.
    pub(crate) const fn predefined_rows(
        counts: &[i16],
        table_log: u32,
        bases: &[u32],
        extra_bits: &[u8],
    ) -> [SeqRow; SEQ_TABLE_SIZE] {
        assert!(table_log <= SEQ_TABLE_LOG_MAX);
        let size = 1usize << table_log;
        let mask = size - 1;
        let step = (size >> 1) + (size >> 3) + 3;
        let mut symbols = [0u8; SEQ_TABLE_SIZE];
        let mut next = [0u16; MAX_SYMBOLS];
        let mut high = size;
        let mut symbol = 0;
        while symbol < counts.len() {
            let count = counts[symbol];
            if count == -1 {
                high -= 1;
                symbols[high] = symbol as u8;
                next[symbol] = 1;
            } else {
                next[symbol] = count as u16;
            }
            symbol += 1;
        }
        let mut position = 0;
        symbol = 0;
        while symbol < counts.len() {
            let mut count = 0;
            while count < counts[symbol] {
                symbols[position] = symbol as u8;
                position = (position + step) & mask;
                while position >= high {
                    position = (position + step) & mask;
                }
                count += 1;
            }
            symbol += 1;
        }
        assert!(position == 0);
        let mut rows = [SeqRow::new(0, 0, 0, 0); SEQ_TABLE_SIZE];
        let mut index = 0;
        while index < size {
            let symbol = symbols[index] as usize;
            let (nb_bits, state) = transition(next[symbol], table_log);
            next[symbol] += 1;
            rows[index] = SeqRow::new(bases[symbol], state, extra_bits[symbol], nb_bits);
            index += 1;
        }
        rows
    }

    /// Independent runtime construction of symbol rows for the tests.
    #[cfg(test)]
    pub(crate) fn fill(
        &mut self,
        counts: &[i16],
        table_log: u32,
        bases: &[u32],
        extra_bits: &[u8],
    ) {
        let mut symbol_rows = [SeqRow::default(); SEQ_SYMBOLS];
        for ((row, &base), &extra) in symbol_rows.iter_mut().zip(bases).zip(extra_bits) {
            *row = SeqRow::new(base, 0, extra, 0);
        }
        self.fill_rows(counts, table_log, &symbol_rows);
    }

    /// Builds rows in the block's selected feature context, without
    /// repeating runtime CPU detection for individual tables or states.
    #[fearless_simd_macros::simd]
    pub(crate) fn fill_rows_on<S: fearless_simd::Simd>(
        &mut self,
        simd: S,
        counts: &[i16],
        table_log: u32,
        symbol_rows: &[SeqRow; SEQ_SYMBOLS],
    ) {
        self.fill_rows(counts, table_log, symbol_rows);
    }

    /// Rebuilds the table for a normalized distribution over precomputed
    /// symbol baselines and extra-bit counts. Inlined so `fill_rows_on`
    /// compiles this body with its token's target features.
    #[inline(always)]
    pub(crate) fn fill_rows(
        &mut self,
        counts: &[i16],
        table_log: u32,
        symbol_rows: &[SeqRow; SEQ_SYMBOLS],
    ) {
        debug_assert!(table_log <= SEQ_TABLE_LOG_MAX);
        let size = 1usize << table_log;
        let mut symbols = [0u8; SEQ_TABLE_SIZE];
        let mut next = [0u16; MAX_SYMBOLS];
        spread::<SEQ_TABLE_SIZE, { SEQ_TABLE_SIZE + 8 }>(
            counts,
            table_log,
            &mut symbols,
            &mut next,
        );
        for (row, &symbol) in self.rows.iter_mut().zip(&symbols[..size]) {
            let symbol = usize::from(symbol);
            let next_state = &mut next[symbol];
            let (nb_bits, state) = transition(*next_state, table_log);
            *next_state += 1;
            *row = SeqRow {
                next_state: state,
                nb_bits: BitCount::new(nb_bits),
                ..symbol_rows[symbol & (SEQ_SYMBOLS - 1)]
            };
        }
        self.log = table_log;
    }

    /// Makes the table a single row that always yields `base` plus
    /// `extra_bits` more bits and reads no state bits.
    pub(crate) fn fill_rle(&mut self, base: u32, extra_bits: u8) {
        self.rows[0] = SeqRow::new(base, 0, extra_bits, 0);
        self.log = 0;
    }

    /// Heap bytes the table holds.
    pub(crate) const fn retained_bytes(&self) -> usize {
        size_of::<[SeqRow; SEQ_TABLE_SIZE]>()
    }

    /// Accuracy log: the bits an initial state takes.
    pub(crate) const fn log(&self) -> u32 {
        self.log
    }

    /// The row for `state`; states come from this table's own
    /// transitions, and the mask only spares the bounds check.
    #[cfg(test)]
    pub(crate) fn row(&self, state: usize) -> SeqRow {
        self.rows()[state & SEQ_STATE_MASK]
    }

    /// The rows, for loops that look them up with [`seq_row`].
    #[inline(always)]
    pub(crate) fn rows(&self) -> &[SeqRow; SEQ_TABLE_SIZE] {
        &self.rows
    }
}

/// The row of `rows` for `state`; states come from the table's own
/// transitions, and the mask only spares the bounds check.
#[inline(always)]
pub(crate) fn seq_row(rows: &[SeqRow; SEQ_TABLE_SIZE], state: usize) -> &SeqRow {
    &rows[state & SEQ_STATE_MASK]
}

impl SeqTable {}

/// Decodes an FSE-compressed Huffman weight series into `weights`, returning
/// how many weights it holds: the reference's `FSE_decompress_wksp`.
///
/// Two states share one table and alternate. While a whole word can be
/// refilled and four more weights fit, four symbols decode per refill with
/// no checks: they read at most 24 bits of the 57 a refill provides. Then
/// each symbol is followed by a refill, and decoding stops once a state
/// update read past the stream start, after emitting the other state's
/// symbol, exactly as the reference does.
pub(crate) fn decode_weights(
    src: &[u8],
    weights: &mut [u8; MAX_WEIGHTS + 1],
) -> Result<usize, InvalidDataKind> {
    let invalid = InvalidDataKind::HuffmanTable;
    let (distribution, used) = read_distribution(src, MAX_LIT).map_err(|_| invalid)?;
    if distribution.table_log > WEIGHT_TABLE_LOG_MAX {
        return Err(invalid);
    }
    let table = FseTable::from_distribution(distribution.counts(), distribution.table_log);
    let stream = src.get(used..).ok_or(invalid)?;
    let mut bits = BitContainer::new(stream).ok_or(invalid)?;
    let log = table.log();
    let mut states = [bits.read(log) as usize, bits.read(log) as usize];
    if bits.overflowed() {
        return Err(invalid);
    }
    bits.reload(stream);
    let capacity = MAX_WEIGHTS;
    let mut count = 0;
    // `&`, not `&&`: the reference refills before it tests the room.
    while bits.reload(stream) & (count + 3 < capacity) {
        for (turn, weight) in [0, 1, 0, 1].into_iter().zip(&mut weights[count..count + 4]) {
            *weight = table.decode(&mut states[turn], &mut bits);
        }
        count += 4;
    }
    let mut turn = 0;
    loop {
        if count + 2 > capacity {
            return Err(invalid);
        }
        weights[count] = table.decode(&mut states[turn], &mut bits);
        count += 1;
        bits.reload(stream);
        if bits.overflowed() {
            weights[count] = table.entry(states[1 - turn]).symbol;
            return Ok(count + 1);
        }
        turn = 1 - turn;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::format::{
        LL_DEFAULT_NORM, LL_DEFAULT_NORM_LOG, ML_DEFAULT_NORM, ML_DEFAULT_NORM_LOG,
        OF_DEFAULT_NORM, OF_DEFAULT_NORM_LOG,
    };
    use alloc::vec::Vec;

    #[test]
    fn bit_counts_represent_every_supported_width() {
        for count in 0..32 {
            let row = SeqRow::new(0, 0, count, count);
            assert_eq!(row.extra_bits(), u32::from(count));
            assert_eq!(row.nb_bits(), u32::from(count));
        }
    }

    /// Writes `(value, width)` fields forward, little-endian.
    fn pack(fields: &[(u32, u32)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut acc = 0u64;
        let mut filled = 0;
        for &(value, width) in fields {
            acc |= u64::from(value) << filled;
            filled += width;
            while filled >= 8 {
                out.push(acc as u8);
                acc >>= 8;
                filled -= 8;
            }
        }
        if filled > 0 {
            out.push(acc as u8);
        }
        out
    }

    #[test]
    fn a_two_symbol_distribution_round_trips() {
        // Accuracy log 5: probabilities 16 and 16, written as values 17 and 17.
        // First field: remaining 33, threshold 32, nbBits 6, max = 63 - 33 = 30;
        // 17 < 30 so 5 bits. Then remaining = 17, threshold 16, nbBits 5,
        // max = 31 - 17 = 14; 17 >= 14 needs the 5-bit form 17 + 14 = 31.
        let src = pack(&[(0, 4), (17, 5), (31, 5)]);
        let (distribution, used) = read_distribution(&src, 1).expect("valid");
        assert_eq!(distribution.table_log, 5);
        assert_eq!(distribution.counts(), [16, 16]);
        assert_eq!(used, 2);
    }

    #[test]
    fn zero_runs_use_repeat_flags() {
        // Log 5: symbol 0 has probability 0 (value 1, five bits), then
        // repeat flags 3 (three more zeros, continue) and 1 (one more), then
        // symbol 5 takes all 32 rows: value 33 is at least the threshold,
        // so it is stored as 33 + max (30) = 63 in six bits.
        let src = pack(&[(0, 4), (1, 5), (3, 2), (1, 2), (63, 6)]);
        let (distribution, _) = read_distribution(&src, 8).expect("valid");
        assert_eq!(distribution.counts(), [0, 0, 0, 0, 0, 32]);
    }

    #[test]
    fn oversized_logs_and_overshooting_symbols_are_rejected() {
        assert_eq!(
            read_distribution(&[0x0b, 0, 0, 0], 35),
            Err(InvalidDataKind::FseTable)
        );
        // A run of zeros past the symbol limit.
        let src = pack(&[(0, 4), (1, 5), (3, 2), (3, 2), (3, 2), (0, 2)]);
        assert_eq!(read_distribution(&src, 3), Err(InvalidDataKind::FseTable));
        // Too few bytes for the declared probabilities.
        assert_eq!(
            read_distribution(&[0x00], 35),
            Err(InvalidDataKind::FseTable)
        );
    }

    /// Rows `(symbol, nb_bits, base)` of an RFC 8878 Appendix A table.
    fn appendix_a(heading: &str) -> Vec<(u8, u8, u16)> {
        let spec = include_str!("../../../zstd-ffi/vendor/zstd/doc/zstd_compression_format.md");
        let section = spec.split(heading).nth(1).expect("heading present");
        section
            .lines()
            .skip_while(|line| !line.starts_with("| ---"))
            .skip(1)
            .take_while(|line| line.starts_with('|'))
            .map(|line| {
                let cells: Vec<u16> = line
                    .split('|')
                    .filter_map(|cell| cell.trim().parse().ok())
                    .collect();
                (cells[1] as u8, cells[2] as u8, cells[3])
            })
            .collect()
    }

    #[test]
    fn predefined_tables_match_appendix_a() {
        let cases: [(&str, &[i16], u32); 3] = [
            (
                "#### Literal Length Code:",
                &LL_DEFAULT_NORM,
                LL_DEFAULT_NORM_LOG,
            ),
            (
                "#### Match Length Code:",
                &ML_DEFAULT_NORM,
                ML_DEFAULT_NORM_LOG,
            ),
            ("#### Offset Code:", &OF_DEFAULT_NORM, OF_DEFAULT_NORM_LOG),
        ];
        for (heading, norm, log) in cases {
            let table = FseTable::from_distribution(norm, log);
            let expected = appendix_a(heading);
            assert_eq!(expected.len(), 1 << log, "{heading}");
            for (state, &(symbol, nb_bits, new_state)) in expected.iter().enumerate() {
                assert_eq!(
                    table.entry(state),
                    FseEntry {
                        symbol,
                        nb_bits,
                        new_state
                    },
                    "{heading} state {state}"
                );
            }
        }
    }

    /// The spread by its definition: one cell at a time, skipping the
    /// cells of "less than one" symbols.
    fn naive_spread(counts: &[i16], table_log: u32) -> (Vec<u8>, Vec<u16>) {
        let size = 1usize << table_log;
        let mut symbols = alloc::vec![0u8; size];
        let mut next = alloc::vec![0u16; MAX_SYMBOLS];
        let mut high = size;
        for (symbol, &count) in counts.iter().enumerate() {
            if count == -1 {
                high -= 1;
                symbols[high] = symbol as u8;
                next[symbol] = 1;
            } else {
                next[symbol] = count as u16;
            }
        }
        let step = (size >> 1) + (size >> 3) + 3;
        let mut position = 0;
        for (symbol, &count) in counts.iter().enumerate() {
            for _ in 0..count.max(0) {
                symbols[position] = symbol as u8;
                position = (position + step) & (size - 1);
                while position >= high {
                    position = (position + step) & (size - 1);
                }
            }
        }
        (symbols, next)
    }

    /// A distribution of `symbols` symbols filling `1 << table_log` cells,
    /// with `low` "less than one" symbols and trailing zeros.
    fn distribution(table_log: u32, symbols: usize, low: usize, seed: u64) -> Vec<i16> {
        let mut counts = alloc::vec![0i16; symbols];
        let mut left = (1i32 << table_log) - low as i32;
        let mut state = seed;
        for (index, count) in counts.iter_mut().enumerate() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if index < low {
                *count = -1;
            } else if index + 3 < symbols && left > 0 {
                let take = ((state >> 33) as i32 % 40).min(left);
                *count = take as i16;
                left -= take;
            }
        }
        // The last described symbol before the trailing zeros takes the rest.
        let last = symbols.saturating_sub(3).max(low);
        counts[last] += left as i16;
        counts
    }

    #[test]
    fn both_spreads_place_symbols_by_their_definition() {
        for table_log in 5..=SEQ_TABLE_LOG_MAX {
            for (low, seed) in [(0, 1), (0, 2), (1, 3), (4, 4), (0, 5)] {
                for symbols in [8, 36, 53] {
                    let counts = distribution(table_log, symbols, low, seed);
                    let mut cells = [0u8; SEQ_TABLE_SIZE];
                    let mut next = [0u16; MAX_SYMBOLS];
                    spread::<SEQ_TABLE_SIZE, { SEQ_TABLE_SIZE + 8 }>(
                        &counts, table_log, &mut cells, &mut next,
                    );
                    let (expected, expected_next) = naive_spread(&counts, table_log);
                    let size = 1 << table_log;
                    assert_eq!(&cells[..size], &expected[..], "log {table_log} low {low}");
                    assert_eq!(&next[..], &expected_next[..]);
                }
            }
        }
        // A full table followed by zero-probability symbols.
        let mut counts = alloc::vec![0i16; 40];
        counts[..32].fill(16);
        let (expected, _) = naive_spread(&counts, 9);
        let mut cells = [0u8; SEQ_TABLE_SIZE];
        let mut next = [0u16; MAX_SYMBOLS];
        spread::<SEQ_TABLE_SIZE, { SEQ_TABLE_SIZE + 8 }>(&counts, 9, &mut cells, &mut next);
        assert_eq!(&cells[..], &expected[..]);
    }

    #[test]
    fn an_rle_sequence_table_reads_no_bits() {
        let mut table = SeqTable::default();
        table.fill_rle(35, 3);
        assert_eq!(table.log(), 0);
        assert_eq!(table.row(0), SeqRow::new(35, 0, 3, 0));
        assert_eq!(table.retained_bytes(), 512 * size_of::<SeqRow>());
    }

    #[test]
    fn sequence_tables_carry_the_weight_tables_transitions_and_baselines() {
        let bases: [u32; 36] = core::array::from_fn(|code| 100 + code as u32);
        let extra: [u8; 36] = core::array::from_fn(|code| code as u8 % 17);
        let mut table = SeqTable::default();
        table.fill(&LL_DEFAULT_NORM, LL_DEFAULT_NORM_LOG, &bases, &extra);
        let reference = FseTable::from_distribution(&LL_DEFAULT_NORM, LL_DEFAULT_NORM_LOG);
        assert_eq!(table.log(), LL_DEFAULT_NORM_LOG);
        for state in 0..1 << LL_DEFAULT_NORM_LOG {
            let entry = reference.entry(state);
            let row = table.row(state);
            assert_eq!(row.base(), bases[usize::from(entry.symbol)]);
            assert_eq!(
                row.extra_bits(),
                u32::from(extra[usize::from(entry.symbol)])
            );
            assert_eq!(
                (row.nb_bits(), row.next_state()),
                (u32::from(entry.nb_bits), usize::from(entry.new_state))
            );
        }
        // States past a small table wrap into its unused rows.
        assert_eq!(
            FseTable::from_distribution(&[32], 5).entry(99),
            FseEntry::default()
        );
    }

    #[test]
    fn weight_streams_without_room_or_states_fail() {
        let mut weights = [0u8; MAX_WEIGHTS + 1];
        // No bitstream after the distribution.
        let distribution = pack(&[(0, 4), (17, 5), (31, 5)]);
        assert_eq!(
            decode_weights(&distribution, &mut weights),
            Err(InvalidDataKind::HuffmanTable)
        );
        // The initial states need ten bits; a one-byte stream has at most 7.
        let mut src = distribution.clone();
        src.push(0x80);
        assert_eq!(
            decode_weights(&src, &mut weights),
            Err(InvalidDataKind::HuffmanTable)
        );
        // Log 7 is too large for weights.
        let mut wide = pack(&[(2, 4), (65, 7), (127, 7)]);
        wide.extend_from_slice(&[0xff, 0xff, 0xff]);
        assert_eq!(
            decode_weights(&wide, &mut weights),
            Err(InvalidDataKind::HuffmanTable)
        );
    }

    #[test]
    fn two_states_alternate_until_the_stream_overflows() {
        let mut weights = [0u8; MAX_WEIGHTS + 1];
        // Symbols 0 and 1 with 16 rows each, so every row reads one bit.
        // The stream holds the two 5-bit initial states (both zero) and two
        // more bits: the first two updates fit, the third overflows and
        // ends the series after the other state's symbol.
        let mut src = pack(&[(0, 4), (17, 5), (31, 5)]);
        src.extend_from_slice(&[0b0000_0001, 0b0001_0000]);
        let count = decode_weights(&src, &mut weights).expect("decodes");
        assert_eq!(count, 4);
        assert!(weights[..count].iter().all(|&w| w <= 1));
    }
}

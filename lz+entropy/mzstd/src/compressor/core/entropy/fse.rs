//! Finite State Entropy encoding (`fse_compress.c` and the encoder half of
//! `fse.h`).
//!
//! Every function mirrors its reference counterpart operation for operation:
//! normalization rounding, table spreading and the header format all feed
//! the compressed bytes, and the size checks decide between encodings.

use super::bits::BitWriter;
use super::hist;
use crate::compressor::core::error::CoreError;
use crate::shared::format::highbit32;

/// Smallest table log the format allows.
pub(crate) const MIN_TABLE_LOG: u32 = 5;
/// Largest table log the encoder builds.
pub(crate) const MAX_TABLE_LOG: u32 = 12;
/// Table log used when none is requested.
const DEFAULT_TABLE_LOG: u32 = 11;
/// Header bound used when the alphabet size is unknown.
const NCOUNT_BOUND: usize = 512;

/// Largest table log of any table this crate encodes with: the sequence
/// tables stop at 9.
const STATE_TABLE_LOG: u32 = 9;
/// Symbols an encoding table can describe: the match length alphabet.
const MAX_SYMBOLS: usize = 64;

/// Per-symbol encoding transform (`FSE_symbolCompressionTransform`).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SymbolTransform {
    pub(crate) delta_find_state: i32,
    pub(crate) delta_nb_bits: u32,
}

/// An FSE encoding table (`FSE_CTable`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CTable {
    /// Table log; `0` for the single-symbol table.
    pub(crate) table_log: u32,
    /// Largest symbol the table was built for.
    pub(crate) max_symbol_value: u32,
    /// Next-state table, sorted by symbol.
    pub(crate) state_table: [u16; 1 << STATE_TABLE_LOG],
    /// Per-symbol transforms.
    pub(crate) symbol_tt: [SymbolTransform; MAX_SYMBOLS],
}

impl Default for CTable {
    fn default() -> Self {
        Self {
            table_log: 0,
            max_symbol_value: 0,
            state_table: [0; 1 << STATE_TABLE_LOG],
            symbol_tt: [SymbolTransform::default(); MAX_SYMBOLS],
        }
    }
}

/// `FSE_TABLESTEP`.
const fn table_step(table_size: u32) -> u32 {
    (table_size >> 1) + (table_size >> 3) + 3
}

/// Smallest table log that can represent `max_symbol_value` symbols drawn
/// from `src_size` samples (`FSE_minTableLog`).
fn min_table_log(src_size: usize, max_symbol_value: u32) -> u32 {
    debug_assert!(src_size > 1);
    let min_bits_src = highbit32(src_size as u32) + 1;
    let min_bits_symbols = highbit32(max_symbol_value) + 2;
    min_bits_src.min(min_bits_symbols)
}

/// `FSE_optimalTableLog_internal`.
pub(crate) fn optimal_table_log_internal(
    max_table_log: u32,
    src_size: usize,
    max_symbol_value: u32,
    minus: u32,
) -> u32 {
    debug_assert!(src_size > 1);
    let max_bits_src = highbit32((src_size - 1) as u32).wrapping_sub(minus);
    let min_bits = min_table_log(src_size, max_symbol_value);
    let mut table_log = if max_table_log == 0 {
        DEFAULT_TABLE_LOG
    } else {
        max_table_log
    };
    if max_bits_src < table_log {
        table_log = max_bits_src;
    }
    if min_bits > table_log {
        table_log = min_bits;
    }
    table_log.clamp(MIN_TABLE_LOG, MAX_TABLE_LOG)
}

/// `FSE_optimalTableLog`.
pub(crate) fn optimal_table_log(max_table_log: u32, src_size: usize, max_symbol_value: u32) -> u32 {
    optimal_table_log_internal(max_table_log, src_size, max_symbol_value, 2)
}

/// Secondary normalization, used when the primary one would starve the most
/// probable symbol (`FSE_normalizeM2`).
fn normalize_m2(
    norm: &mut [i16],
    table_log: u32,
    count: &[u32],
    mut total: usize,
    max_symbol_value: u32,
    low_prob_count: i16,
) -> Result<(), CoreError> {
    const NOT_YET_ASSIGNED: i16 = -2;
    let symbols = max_symbol_value as usize + 1;
    let mut distributed: u32 = 0;
    let low_threshold = (total >> table_log) as u32;
    let mut low_one = ((total * 3) >> (table_log + 1)) as u32;

    for s in 0..symbols {
        if count[s] == 0 {
            norm[s] = 0;
            continue;
        }
        if count[s] <= low_threshold {
            norm[s] = low_prob_count;
            distributed += 1;
            total -= count[s] as usize;
            continue;
        }
        if count[s] <= low_one {
            norm[s] = 1;
            distributed += 1;
            total -= count[s] as usize;
            continue;
        }
        norm[s] = NOT_YET_ASSIGNED;
    }
    let mut to_distribute = (1u32 << table_log) - distributed;

    if to_distribute == 0 {
        return Ok(());
    }

    if total / to_distribute as usize > low_one as usize {
        // Risk of rounding to zero.
        low_one = ((total * 3) / (to_distribute as usize * 2)) as u32;
        for s in 0..symbols {
            if norm[s] == NOT_YET_ASSIGNED && count[s] <= low_one {
                norm[s] = 1;
                distributed += 1;
                total -= count[s] as usize;
            }
        }
        to_distribute = (1u32 << table_log) - distributed;
    }

    if distributed == max_symbol_value + 1 {
        // Every symbol is rare: give the remainder to the most frequent.
        let mut max_v = 0;
        let mut max_c = 0;
        for (s, &c) in count.iter().enumerate().take(symbols) {
            if c > max_c {
                max_v = s;
                max_c = c;
            }
        }
        norm[max_v] += to_distribute as i16;
        return Ok(());
    }

    if total == 0 {
        // Every symbol sat at or below one of the low thresholds.
        let mut s = 0;
        while to_distribute > 0 {
            if norm[s] > 0 {
                to_distribute -= 1;
                norm[s] += 1;
            }
            s = (s + 1) % symbols;
        }
        return Ok(());
    }

    let v_step_log = 62 - u64::from(table_log);
    let mid = (1u64 << (v_step_log - 1)) - 1;
    let r_step = ((1u64 << v_step_log) * u64::from(to_distribute) + mid) / u64::from(total as u32);
    let mut tmp_total = mid;
    for s in 0..symbols {
        if norm[s] == NOT_YET_ASSIGNED {
            let end = tmp_total + u64::from(count[s]) * r_step;
            let s_start = (tmp_total >> v_step_log) as u32;
            let s_end = (end >> v_step_log) as u32;
            let weight = s_end.wrapping_sub(s_start);
            if weight < 1 {
                return Err(CoreError::Generic);
            }
            norm[s] = weight as i16;
            tmp_total = end;
        }
    }
    Ok(())
}

/// Normalizes `count` (summing to `total`) to `2^table_log`
/// (`FSE_normalizeCount`).
///
/// Returns the table log, or `0` when a single symbol holds every count; in
/// that case `norm` is only partially written, as in the reference.
///
/// # Errors
///
/// [`CoreError::Generic`] for a table log too small for the alphabet and
/// [`CoreError::TableLogTooLarge`] above the encoder's limit.
pub(crate) fn normalize_count(
    norm: &mut [i16],
    table_log: u32,
    count: &[u32],
    total: usize,
    max_symbol_value: u32,
    use_low_prob_count: bool,
) -> Result<u32, CoreError> {
    const RTB_TABLE: [u64; 8] = [0, 473195, 504333, 520860, 550000, 700000, 750000, 830000];
    let table_log = if table_log == 0 {
        DEFAULT_TABLE_LOG
    } else {
        table_log
    };
    if table_log < MIN_TABLE_LOG {
        return Err(CoreError::Generic);
    }
    if table_log > MAX_TABLE_LOG {
        return Err(CoreError::TableLogTooLarge);
    }
    if table_log < min_table_log(total, max_symbol_value) {
        return Err(CoreError::Generic);
    }

    let low_prob_count: i16 = if use_low_prob_count { -1 } else { 1 };
    let scale = 62 - u64::from(table_log);
    let step = (1u64 << 62) / u64::from(total as u32);
    let v_step = 1u64 << (scale - 20);
    let mut still_to_distribute: i32 = 1 << table_log;
    let mut largest = 0usize;
    let mut largest_p: i16 = 0;
    let low_threshold = (total >> table_log) as u32;

    for s in 0..=max_symbol_value as usize {
        if count[s] as usize == total {
            return Ok(0);
        }
        if count[s] == 0 {
            norm[s] = 0;
            continue;
        }
        if count[s] <= low_threshold {
            norm[s] = low_prob_count;
            still_to_distribute -= 1;
        } else {
            let scaled = u64::from(count[s]) * step;
            let mut proba = (scaled >> scale) as i16;
            if proba < 8 {
                let rest_to_beat = v_step * RTB_TABLE[proba as usize];
                proba += i16::from(scaled - ((proba as u64) << scale) > rest_to_beat);
            }
            if proba > largest_p {
                largest_p = proba;
                largest = s;
            }
            norm[s] = proba;
            still_to_distribute -= i32::from(proba);
        }
    }
    if -still_to_distribute >= i32::from(norm[largest] >> 1) {
        normalize_m2(
            norm,
            table_log,
            count,
            total,
            max_symbol_value,
            low_prob_count,
        )?;
    } else {
        norm[largest] += still_to_distribute as i16;
    }
    Ok(table_log)
}

/// `FSE_NCountWriteBound`.
const fn ncount_write_bound(max_symbol_value: u32, table_log: u32) -> usize {
    let max_header_size =
        (((max_symbol_value as usize + 1) * table_log as usize + 4 + 2) / 8) + 1 + 2;
    if max_symbol_value != 0 {
        max_header_size
    } else {
        NCOUNT_BOUND
    }
}

/// Writes the normalized distribution header (`FSE_writeNCount`).
///
/// # Errors
///
/// [`CoreError::DstSizeTooSmall`] when `dst` cannot hold the header,
/// [`CoreError::TableLogTooLarge`] / [`CoreError::Generic`] for an
/// unsupported table log, and [`CoreError::Generic`] for a distribution that
/// does not sum to the table size.
pub(crate) fn write_ncount(
    dst: &mut [u8],
    norm: &[i16],
    max_symbol_value: u32,
    table_log: u32,
) -> Result<usize, CoreError> {
    if table_log > MAX_TABLE_LOG {
        return Err(CoreError::TableLogTooLarge);
    }
    if table_log < MIN_TABLE_LOG {
        return Err(CoreError::Generic);
    }
    let write_is_safe = dst.len() >= ncount_write_bound(max_symbol_value, table_log);
    let oend = dst.len();
    let mut out = 0usize;
    let table_size: i32 = 1 << table_log;
    let alphabet_size = max_symbol_value as usize + 1;
    let mut bit_stream: u32 = 0;
    let mut bit_count: i32 = 0;
    let mut symbol = 0usize;
    let mut previous_is_0 = false;

    let emit = |out: &mut usize, bit_stream: u32, dst: &mut [u8]| -> Result<(), CoreError> {
        if !write_is_safe && *out + 2 > oend {
            return Err(CoreError::DstSizeTooSmall);
        }
        dst[*out] = bit_stream as u8;
        dst[*out + 1] = (bit_stream >> 8) as u8;
        *out += 2;
        Ok(())
    };

    // Table size.
    bit_stream += (table_log - MIN_TABLE_LOG) << bit_count;
    bit_count += 4;

    let mut remaining: i32 = table_size + 1;
    let mut threshold: i32 = table_size;
    let mut nb_bits: i32 = table_log as i32 + 1;

    while symbol < alphabet_size && remaining > 1 {
        if previous_is_0 {
            let mut start = symbol;
            while symbol < alphabet_size && norm[symbol] == 0 {
                symbol += 1;
            }
            if symbol == alphabet_size {
                break;
            }
            while symbol >= start + 24 {
                start += 24;
                bit_stream = bit_stream.wrapping_add(0xFFFFu32 << bit_count);
                emit(&mut out, bit_stream, dst)?;
                bit_stream >>= 16;
            }
            while symbol >= start + 3 {
                start += 3;
                bit_stream = bit_stream.wrapping_add(3u32 << bit_count);
                bit_count += 2;
            }
            bit_stream = bit_stream.wrapping_add(((symbol - start) as u32) << bit_count);
            bit_count += 2;
            if bit_count > 16 {
                emit(&mut out, bit_stream, dst)?;
                bit_stream >>= 16;
                bit_count -= 16;
            }
        }
        {
            let mut count = i32::from(norm[symbol]);
            symbol += 1;
            let max = (2 * threshold - 1) - remaining;
            remaining -= count.abs();
            count += 1;
            if count >= threshold {
                count += max;
            }
            bit_stream = bit_stream.wrapping_add((count as u32) << bit_count);
            bit_count += nb_bits;
            bit_count -= i32::from(count < max);
            previous_is_0 = count == 1;
            if remaining < 1 {
                return Err(CoreError::Generic);
            }
            while remaining < threshold {
                nb_bits -= 1;
                threshold >>= 1;
            }
        }
        if bit_count > 16 {
            emit(&mut out, bit_stream, dst)?;
            bit_stream >>= 16;
            bit_count -= 16;
        }
    }

    if remaining != 1 {
        return Err(CoreError::Generic);
    }

    // Flush the remaining bits.
    if !write_is_safe && out + 2 > oend {
        return Err(CoreError::DstSizeTooSmall);
    }
    dst[out] = bit_stream as u8;
    dst[out + 1] = (bit_stream >> 8) as u8;
    out += ((bit_count + 7) / 8) as usize;
    Ok(out)
}

impl CTable {
    /// Builds the table for a normalized distribution (`FSE_buildCTable_wksp`).
    pub(crate) fn build(&mut self, norm: &[i16], max_symbol_value: u32, table_log: u32) {
        let table_size: u32 = 1 << table_log;
        let step = table_step(table_size);
        let symbols = max_symbol_value as usize + 1;
        let mut cumul = [0u32; MAX_SYMBOLS + 1];
        // Sized for the largest table this type holds rather than for
        // MAX_TABLE_LOG: the array is zeroed on every build.
        let mut table_symbol = [0u8; 1 << STATE_TABLE_LOG];
        let mut high_threshold = table_size - 1;

        assert!(table_log <= STATE_TABLE_LOG);
        self.table_log = table_log;
        self.max_symbol_value = max_symbol_value;

        // Symbol start positions.
        for u in 1..=symbols {
            if norm[u - 1] == -1 {
                cumul[u] = cumul[u - 1] + 1;
                table_symbol[high_threshold as usize] = (u - 1) as u8;
                high_threshold = high_threshold.wrapping_sub(1);
            } else {
                cumul[u] = cumul[u - 1] + norm[u - 1] as u32;
            }
        }
        cumul[symbols] = table_size + 1;

        // Spread symbols.
        if high_threshold == table_size - 1 {
            spread_without_low_probability(
                &mut table_symbol,
                &norm[..symbols],
                table_size as usize,
                step as usize,
            );
        } else {
            spread(
                &mut table_symbol,
                &norm[..symbols],
                table_size,
                step,
                high_threshold,
            );
        }

        // Build the state table. Spread symbols are below `symbols`, at
        // most `MAX_SYMBOLS`, and their positions below the table size, so
        // the masks change nothing and leave no bounds checks behind.
        for (u, &s) in (0..table_size).zip(&table_symbol[..table_size as usize]) {
            let s = usize::from(s) & (MAX_SYMBOLS - 1);
            self.state_table[cumul[s] as usize & ((1 << STATE_TABLE_LOG) - 1)] =
                (table_size + u) as u16;
            cumul[s] += 1;
        }

        // Build the symbol transformation table.
        let mut total: u32 = 0;
        for (tt, &count) in self.symbol_tt.iter_mut().zip(&norm[..symbols]) {
            match count {
                0 => {
                    tt.delta_nb_bits = ((table_log + 1) << 16).wrapping_sub(1 << table_log);
                }
                -1 | 1 => {
                    tt.delta_nb_bits = (table_log << 16).wrapping_sub(1 << table_log);
                    tt.delta_find_state = total as i32 - 1;
                    total += 1;
                }
                n => {
                    let n = n as u32;
                    let max_bits_out = table_log - highbit32(n - 1);
                    let min_state_plus = n << max_bits_out;
                    tt.delta_nb_bits = (max_bits_out << 16).wrapping_sub(min_state_plus);
                    tt.delta_find_state = total as i32 - n as i32;
                    total += n;
                }
            }
        }
    }

    /// Builds the single-symbol table (`FSE_buildCTable_rle`).
    pub(crate) fn build_rle(&mut self, symbol: u8) {
        self.table_log = 0;
        self.max_symbol_value = u32::from(symbol);
        self.state_table[0] = 0;
        self.state_table[1] = 0;
        self.symbol_tt[usize::from(symbol)] = SymbolTransform::default();
    }

    /// The reference's approximate maximum cost of `symbol` in bits, its
    /// fractional part rounded up (`FSE_getMaxNbBits`).
    ///
    /// Like the reference it reads whatever transform the symbol holds: one
    /// past the largest symbol of the last build keeps an older table's.
    pub(crate) fn max_nb_bits(&self, symbol: u32) -> u32 {
        self.symbol_tt
            .get(symbol as usize)
            .map_or(0, |tt| tt.delta_nb_bits.wrapping_add((1 << 16) - 1) >> 16)
    }

    /// Approximate cost of `symbol` in `1/2^accuracy_log` bits
    /// (`FSE_bitCost`).
    pub(crate) fn bit_cost(&self, symbol: u32, accuracy_log: u32) -> u32 {
        let tt = self.symbol_tt[symbol as usize];
        let min_nb_bits = tt.delta_nb_bits >> 16;
        let threshold = (min_nb_bits + 1) << 16;
        let table_size = 1u32 << self.table_log;
        let delta_from_threshold =
            threshold.wrapping_sub(tt.delta_nb_bits.wrapping_add(table_size));
        let normalized_delta = (delta_from_threshold << accuracy_log) >> self.table_log;
        let bit_multiplier = 1u32 << accuracy_log;
        (min_nb_bits + 1) * bit_multiplier - normalized_delta
    }
}

/// Symbol spreading around the positions reserved for low-probability
/// symbols above `high_threshold` (the general path of
/// `FSE_buildCTable_wksp`).
fn spread(
    table_symbol: &mut [u8; 1 << STATE_TABLE_LOG],
    norm: &[i16],
    table_size: u32,
    step: u32,
    high_threshold: u32,
) {
    let table_mask = table_size - 1;
    let mut position: u32 = 0;
    for (symbol, &freq) in norm.iter().enumerate() {
        for _ in 0..i32::from(freq).max(0) {
            table_symbol[position as usize & ((1 << STATE_TABLE_LOG) - 1)] = symbol as u8;
            position = (position + step) & table_mask;
            while position > high_threshold {
                position = (position + step) & table_mask;
            }
        }
    }
    debug_assert_eq!(position, 0);
}

/// Symbol spreading for a distribution without low-probability (`-1`)
/// symbols, the branch-light path of `FSE_buildCTable_wksp`.
///
/// The symbols are first laid out in order, `norm[s]` copies of each,
/// eight bytes at a time; the `k`-th of them then lands at `k * step`
/// modulo the table size, which is where the general spreading loop puts
/// the `k`-th occurrence when no position is reserved for `-1` symbols.
fn spread_without_low_probability(
    table_symbol: &mut [u8; 1 << STATE_TABLE_LOG],
    norm: &[i16],
    table_size: usize,
    step: usize,
) {
    const SLOTS: usize = 1 << STATE_TABLE_LOG;
    let mut spread = [0u8; SLOTS + 8];
    let mut pos = 0usize;
    let mut value = 0u64;
    for &n in norm {
        let n = n.max(0) as usize;
        let mut i = 0;
        loop {
            spread[pos + i..pos + i + 8].copy_from_slice(&value.to_le_bytes());
            i += 8;
            if i >= n {
                break;
            }
        }
        pos += n;
        value = value.wrapping_add(0x0101_0101_0101_0101);
    }
    debug_assert_eq!(pos, table_size);
    let mask = table_size - 1;
    let mut position = 0usize;
    for pair in spread[..table_size].as_chunks::<2>().0 {
        table_symbol[position & mask & (SLOTS - 1)] = pair[0];
        table_symbol[(position + step) & mask & (SLOTS - 1)] = pair[1];
        position = (position + 2 * step) & mask;
    }
    debug_assert_eq!(position, 0);
}

/// An encoder state over one table (`FSE_CState_t`).
#[derive(Debug)]
pub(crate) struct CState<'t> {
    value: i64,
    table: &'t CTable,
}

impl<'t> CState<'t> {
    /// `FSE_initCState2`: starts in the cheapest state for `symbol`, which is
    /// the first symbol encoded and the last decoded.
    pub(crate) fn with_first_symbol(table: &'t CTable, symbol: u32) -> Self {
        let tt = table.symbol_tt[symbol as usize];
        let nb_bits_out = (tt.delta_nb_bits.wrapping_add(1 << 15)) >> 16;
        let value = (nb_bits_out << 16).wrapping_sub(tt.delta_nb_bits) as i64;
        let index = (value >> nb_bits_out) + i64::from(tt.delta_find_state);
        Self {
            value: i64::from(table.state_table[index as usize]),
            table,
        }
    }

    /// `FSE_encodeSymbol`.
    ///
    /// Both table indices are masked to their table sizes: a symbol the
    /// table was built for and the state it leads to are always in range,
    /// so the masks change nothing and leave no bounds checks behind.
    #[inline(always)]
    pub(crate) fn encode(&mut self, writer: &mut BitWriter<'_>, symbol: u32) {
        let tt = self.table.symbol_tt[symbol as usize & (MAX_SYMBOLS - 1)];
        let nb_bits_out = ((self.value + i64::from(tt.delta_nb_bits)) >> 16) as u32;
        writer.add_bits(self.value as u64, nb_bits_out);
        let index = (self.value >> nb_bits_out) + i64::from(tt.delta_find_state);
        self.value =
            i64::from(self.table.state_table[index as usize & ((1 << STATE_TABLE_LOG) - 1)]);
    }

    /// `FSE_flushCState`.
    pub(crate) fn flush(&self, writer: &mut BitWriter<'_>) {
        writer.add_bits(self.value as u64, self.table.table_log);
        writer.flush();
    }
}

/// Encodes `src` with `table` (`FSE_compress_usingCTable`).
///
/// Returns `0` when `src` is too short or the stream does not fit.
pub(crate) fn compress_using_ctable(dst: &mut [u8], src: &[u8], table: &CTable) -> usize {
    if src.len() <= 2 {
        return 0;
    }
    let Some(mut writer) = BitWriter::new(dst) else {
        return 0;
    };
    let mut ip = src.len();
    let next = |ip: &mut usize| {
        *ip -= 1;
        u32::from(src[*ip])
    };

    let (mut state1, mut state2);
    if src.len() & 1 != 0 {
        state1 = CState::with_first_symbol(table, next(&mut ip));
        state2 = CState::with_first_symbol(table, next(&mut ip));
        state1.encode(&mut writer, next(&mut ip));
        writer.flush();
    } else {
        state2 = CState::with_first_symbol(table, next(&mut ip));
        state1 = CState::with_first_symbol(table, next(&mut ip));
    }

    // Join to a multiple of four.
    let size = src.len() - 2;
    if size & 2 != 0 {
        state2.encode(&mut writer, next(&mut ip));
        state1.encode(&mut writer, next(&mut ip));
        writer.flush();
    }

    while ip > 0 {
        state2.encode(&mut writer, next(&mut ip));
        state1.encode(&mut writer, next(&mut ip));
        state2.encode(&mut writer, next(&mut ip));
        state1.encode(&mut writer, next(&mut ip));
        writer.flush();
    }

    state2.flush(&mut writer);
    state1.flush(&mut writer);
    writer.close()
}

/// Counts, normalizes and encodes `src` as a header and an FSE stream, the
/// way `HUF_compressWeights` compresses Huffman weights.
///
/// Returns `0` when the input does not compress, `1` when it is a single
/// repeated symbol, and otherwise the size written.
///
/// # Errors
///
/// Whatever header writing or normalization reports.
pub(crate) fn compress_small(
    dst: &mut [u8],
    src: &[u8],
    max_symbol_value: u32,
    max_table_log: u32,
) -> Result<usize, CoreError> {
    if src.len() <= 1 {
        return Ok(0);
    }
    let mut count = [0u32; 256];
    let mut max = max_symbol_value;
    let max_count = hist::count(&mut count, &mut max, src, false)?;
    if max_count as usize == src.len() {
        return Ok(1);
    }
    if max_count == 1 {
        return Ok(0);
    }
    let table_log = optimal_table_log(max_table_log, src.len(), max);
    let mut norm = [0i16; MAX_SYMBOLS];
    normalize_count(&mut norm, table_log, &count, src.len(), max, false)?;
    let header = write_ncount(dst, &norm, max, table_log)?;
    let mut table = CTable::default();
    table.build(&norm, max, table_log);
    let size = compress_using_ctable(&mut dst[header..], src, &table);
    if size == 0 {
        return Ok(0);
    }
    Ok(header + size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fast_spread_matches_the_general_one() {
        for count in histograms() {
            let max = count.len() as u32 - 1;
            let total: usize = count.iter().map(|&c| c as usize).sum();
            for table_log in [5, 6, 9] {
                let mut norm = [0i16; MAX_SYMBOLS];
                if normalize_count(&mut norm, table_log, &count, total, max, false).is_err() {
                    continue;
                }
                let norm = &norm[..count.len()];
                if norm.contains(&-1) {
                    continue;
                }
                let table_size = 1u32 << table_log;
                let step = table_step(table_size);
                let mut general = [0u8; 1 << STATE_TABLE_LOG];
                spread(&mut general, norm, table_size, step, table_size - 1);
                let mut fast = [0u8; 1 << STATE_TABLE_LOG];
                spread_without_low_probability(&mut fast, norm, table_size as usize, step as usize);
                assert_eq!(fast, general, "{count:?} log {table_log}");
            }
        }
    }

    /// Deterministic pseudo-random histograms of every shape the encoder
    /// meets: skewed, flat, sparse and with low-probability symbols.
    fn histograms() -> alloc::vec::Vec<alloc::vec::Vec<u32>> {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut out = alloc::vec::Vec::new();
        for alphabet in [2usize, 3, 5, 13, 36, 53] {
            for shape in 0..40 {
                let mut count = alloc::vec![0u32; alphabet];
                for (s, c) in count.iter_mut().enumerate() {
                    let r = next();
                    *c = match shape % 4 {
                        0 => (r % 1000) as u32,
                        1 => ((r % 7) as u32).pow(4) + u32::from(s == 0) * 5000,
                        2 => u32::from(r % 3 == 0) * (r % 50) as u32,
                        _ => (r % 2) as u32 + if s == alphabet - 1 { 100_000 } else { 0 },
                    };
                }
                count[alphabet - 1] = count[alphabet - 1].max(1);
                count[0] = count[0].max(1);
                out.push(count);
            }
        }
        out
    }

    #[test]
    fn optimal_table_log_matches_the_reference() {
        for max_log in [0, 5, 6, 8, 9, 11, 12] {
            for size in [2usize, 3, 17, 100, 1000, 4096, 1 << 17] {
                for max_symbol in [1u32, 12, 35, 52, 255] {
                    assert_eq!(
                        optimal_table_log(max_log, size, max_symbol),
                        zstd_ffi::oracle::fse_optimal_table_log(max_log, size, max_symbol),
                        "log {max_log} size {size} max {max_symbol}"
                    );
                }
            }
        }
    }

    #[test]
    fn normalization_matches_the_reference() {
        for count in histograms() {
            let total: usize = count.iter().map(|&c| c as usize).sum();
            let max = count.len() as u32 - 1;
            for table_log in [5, 6, 8, 9, 11, 12] {
                for low_prob in [false, true] {
                    let mut norm = alloc::vec![0i16; count.len()];
                    let ours = normalize_count(&mut norm, table_log, &count, total, max, low_prob);
                    let reference =
                        zstd_ffi::oracle::fse_normalize_count(&count, total, table_log, low_prob);
                    match (ours, reference) {
                        (Ok(log), Ok((expected, expected_log))) => {
                            assert_eq!(log as usize, expected_log, "{count:?}");
                            if log != 0 {
                                assert_eq!(norm, expected, "{count:?} log {table_log}");
                            }
                        }
                        (Err(_), Err(_)) => {}
                        (ours, reference) => panic!("{count:?}: {ours:?} vs {reference:?}"),
                    }
                }
            }
        }
    }

    #[test]
    fn normalization_rejects_unsupported_logs() {
        let mut norm = [0i16; 4];
        let count = [3u32, 1, 0, 0];
        assert_eq!(
            normalize_count(&mut norm, 4, &count, 4, 1, true),
            Err(CoreError::Generic)
        );
        assert_eq!(
            normalize_count(&mut norm, 13, &count, 4, 1, true),
            Err(CoreError::TableLogTooLarge)
        );
        assert_eq!(normalize_count(&mut norm, 5, &[4, 0], 4, 1, true), Ok(0));
    }

    #[test]
    fn single_symbol_tables_emit_no_bits() {
        let mut table = CTable::default();
        table.build_rle(7);
        let mut dst = [0u8; 16];
        let mut writer = BitWriter::new(&mut dst).expect("capacity");
        let mut state = CState::with_first_symbol(&table, 7);
        state.encode(&mut writer, 7);
        state.flush(&mut writer);
        assert_eq!(writer.close(), 1);
    }

    #[test]
    fn maximum_symbol_costs_round_fractions_up() {
        // Table log 5: `(deltaNbBits + 0xFFFF) >> 16`, worked by hand. The
        // reference's bound rounds a whole bit up: 16/32 gives 2, 8/32 and
        // 5/32 give 3, 2/32 and a low-probability symbol 5, and an absent
        // one the log plus one.
        let norm = [16i16, 8, 5, 2, -1, 0];
        let mut table = CTable::default();
        table.build(&norm, 5, 5);
        let costs: alloc::vec::Vec<u32> = (0..6).map(|s| table.max_nb_bits(s)).collect();
        assert_eq!(costs, [2, 3, 3, 5, 5, 6]);
        // Past the transforms there is nothing to read.
        assert_eq!(table.max_nb_bits(MAX_SYMBOLS as u32), 0);
    }

    #[test]
    fn short_inputs_are_not_compressed() {
        let mut dst = [0u8; 64];
        assert_eq!(compress_small(&mut dst, &[1], 12, 6), Ok(0));
        assert_eq!(compress_small(&mut dst, &[2, 2, 2], 12, 6), Ok(1));
        assert_eq!(compress_small(&mut dst, &[1, 2, 3], 12, 6), Ok(0));
        let table = CTable::default();
        assert_eq!(compress_using_ctable(&mut dst, &[1, 2], &table), 0);
    }
}

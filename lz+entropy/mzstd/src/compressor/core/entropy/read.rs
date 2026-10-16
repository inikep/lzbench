//! Reading entropy table descriptions back into encoder tables, for
//! Zstandard-format dictionaries (`FSE_readNCount`, `HUF_readStats` and
//! `HUF_readCTable` of the reference's `entropy_common.c` and
//! `huf_compress.c`).
//!
//! A dictionary's entropy section holds the same descriptions a block
//! header does. The encoder only reads them when a dictionary is prepared,
//! so these are straight ports that accept and reject exactly what the
//! reference does; nothing here is on a hot path.

use super::huf::CTable;
use crate::shared::format::highbit32;

/// `FSE_MIN_TABLELOG`.
const MIN_TABLE_LOG: u32 = 5;
/// `FSE_TABLELOG_ABSOLUTE_MAX`.
const TABLE_LOG_ABSOLUTE_MAX: u32 = 15;
/// `HUF_TABLELOG_MAX`.
const HUF_TABLE_LOG_MAX: u32 = 12;
/// `HUF_SYMBOLVALUE_MAX + 1`: weights a Huffman description can hold.
const HUF_WEIGHTS: usize = 256;
/// Table log limit of the FSE table compressing Huffman weights.
const WEIGHT_TABLE_LOG_MAX: u32 = 6;

/// A malformed table description (the reference's `corruption_detected`,
/// `tableLog_tooLarge`, `maxSymbolValue_tooSmall` and `srcSize_wrong`,
/// which a dictionary load reports alike).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct Malformed;

/// A normalized distribution read from a description.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NCount {
    /// Probabilities, `-1` for "less than one"; zero past `max_symbol`.
    pub(crate) counts: [i16; 256],
    /// Largest symbol described.
    pub(crate) max_symbol: u32,
    /// Accuracy log.
    pub(crate) table_log: u32,
    /// Bytes the description occupies.
    pub(crate) size: usize,
}

/// Little-endian 32-bit read at `pos` of a buffer known to hold it.
fn read32(buffer: &[u8], pos: usize) -> u32 {
    let mut word = [0u8; 4];
    word.copy_from_slice(&buffer[pos..pos + 4]);
    u32::from_le_bytes(word)
}

/// `FSE_readNCount` for symbols up to `max_symbol`.
pub(crate) fn read_ncount(src: &[u8], max_symbol: u32) -> Result<NCount, Malformed> {
    if src.len() < 8 {
        // The body needs eight readable bytes; a shorter description is
        // read from a zero-padded copy and must fit the original.
        let mut buffer = [0u8; 8];
        buffer[..src.len()].copy_from_slice(src);
        let ncount = read_ncount_body(&buffer, max_symbol)?;
        if ncount.size > src.len() {
            return Err(Malformed);
        }
        return Ok(ncount);
    }
    read_ncount_body(src, max_symbol)
}

/// Moves the read position by whole bytes, never past `iend - 4`
/// (the reference's end-of-buffer clamp).
fn advance(ip: &mut usize, bit_count: &mut i32, iend: usize) {
    if *ip + 7 <= iend || *ip + (*bit_count >> 3) as usize + 4 <= iend {
        *ip += (*bit_count >> 3) as usize;
        *bit_count &= 7;
    } else {
        *bit_count -= (8 * (iend - 4 - *ip)) as i32;
        *bit_count &= 31;
        *ip = iend - 4;
    }
}

/// `FSE_readNCount_body` over at least eight bytes.
fn read_ncount_body(src: &[u8], max_symbol: u32) -> Result<NCount, Malformed> {
    let iend = src.len();
    let mut ip = 0usize;
    let max_sv1 = max_symbol + 1;
    let mut counts = [0i16; 256];
    let mut charnum: u32 = 0;
    let mut previous0 = false;

    let mut bit_stream = read32(src, ip);
    let mut nb_bits = (bit_stream & 0xF) + MIN_TABLE_LOG;
    if nb_bits > TABLE_LOG_ABSOLUTE_MAX {
        return Err(Malformed);
    }
    bit_stream >>= 4;
    let mut bit_count: i32 = 4;
    let table_log = nb_bits;
    let mut remaining: i32 = (1 << nb_bits) + 1;
    let mut threshold: i32 = 1 << nb_bits;
    nb_bits += 1;

    loop {
        if previous0 {
            // Runs of zero probabilities: two-bit repeat codes, 3 continues.
            let mut repeats = (!bit_stream | 0x8000_0000).trailing_zeros() >> 1;
            while repeats >= 12 {
                charnum += 3 * 12;
                if ip + 7 <= iend {
                    ip += 3;
                } else {
                    // `ip` is past `iend - 7` here: the shift grows.
                    bit_count -= 8 * (iend as i32 - 7 - ip as i32);
                    bit_count &= 31;
                    ip = iend - 4;
                }
                bit_stream = read32(src, ip) >> bit_count;
                repeats = (!bit_stream | 0x8000_0000).trailing_zeros() >> 1;
            }
            charnum += 3 * repeats;
            bit_stream >>= 2 * repeats;
            bit_count += 2 * repeats as i32;

            charnum += bit_stream & 3;
            bit_count += 2;

            if charnum >= max_sv1 {
                break;
            }
            advance(&mut ip, &mut bit_count, iend);
            bit_stream = read32(src, ip) >> bit_count;
        }
        let max = (2 * threshold - 1) - remaining;
        let mut count: i32;
        if ((bit_stream & (threshold - 1) as u32) as i32) < max {
            count = (bit_stream & (threshold - 1) as u32) as i32;
            bit_count += nb_bits as i32 - 1;
        } else {
            count = (bit_stream & (2 * threshold - 1) as u32) as i32;
            if count >= threshold {
                count -= max;
            }
            bit_count += nb_bits as i32;
        }
        count -= 1;
        if count >= 0 {
            remaining -= count;
        } else {
            remaining += count;
        }
        counts[charnum as usize & 0xFF] = count as i16;
        charnum += 1;
        previous0 = count == 0;
        if remaining < threshold {
            if remaining <= 1 {
                break;
            }
            nb_bits = highbit32(remaining as u32) + 1;
            threshold = 1 << (nb_bits - 1);
        }
        if charnum >= max_sv1 {
            break;
        }
        advance(&mut ip, &mut bit_count, iend);
        bit_stream = read32(src, ip) >> bit_count;
    }
    if remaining != 1 || charnum > max_sv1 || bit_count > 32 {
        return Err(Malformed);
    }
    ip += ((bit_count + 7) >> 3) as usize;
    Ok(NCount {
        counts,
        max_symbol: charnum - 1,
        table_log,
        size: ip,
    })
}

/// A backward bit stream (`BIT_DStream_t`): bits are taken from the end,
/// below the final byte's marker bit, and reads past the start yield zeros
/// and mark the stream overflowed.
struct ReverseBits<'a> {
    src: &'a [u8],
    /// Bits not yet read; negative once reads went past the start.
    left: i64,
}

impl<'a> ReverseBits<'a> {
    /// `BIT_initDStream`: the last byte must hold the marker.
    fn new(src: &'a [u8]) -> Result<Self, Malformed> {
        let &last = src.last().ok_or(Malformed)?;
        if last == 0 {
            return Err(Malformed);
        }
        let left = (src.len() as i64 - 1) * 8 + i64::from(highbit32(u32::from(last)));
        Ok(Self { src, left })
    }

    /// Reads `n` bits, most significant first.
    fn read(&mut self, n: u32) -> u32 {
        let mut value = 0u32;
        for _ in 0..n {
            self.left -= 1;
            let bit = if self.left >= 0 {
                (self.src[(self.left >> 3) as usize] >> (self.left & 7)) & 1
            } else {
                0
            };
            value = (value << 1) | u32::from(bit);
        }
        value
    }

    /// Whether reads went past the stream's start (`BIT_DStream_overflow`).
    const fn overflowed(&self) -> bool {
        self.left < 0
    }
}

/// A decoding table row (`FSE_decode_t`).
#[derive(Copy, Clone, Debug, Default)]
struct DecodeRow {
    symbol: u8,
    nb_bits: u8,
    new_state: u16,
}

/// `FSE_buildDTable_internal` for a table of at most 64 rows.
fn build_dtable(ncount: &NCount) -> [DecodeRow; 1 << WEIGHT_TABLE_LOG_MAX] {
    let table_log = ncount.table_log;
    let table_size = 1usize << table_log;
    let mut rows = [DecodeRow::default(); 1 << WEIGHT_TABLE_LOG_MAX];
    let mut symbol_next = [0u16; 256];
    let mut high_threshold = table_size - 1;
    let symbols = ncount.max_symbol as usize + 1;
    for (s, (&count, next)) in ncount.counts[..symbols]
        .iter()
        .zip(&mut symbol_next)
        .enumerate()
    {
        if count == -1 {
            rows[high_threshold].symbol = s as u8;
            high_threshold = high_threshold.wrapping_sub(1);
            *next = 1;
        } else {
            *next = count as u16;
        }
    }
    let mask = table_size - 1;
    let step = (table_size >> 1) + (table_size >> 3) + 3;
    let mut position = 0;
    for s in 0..symbols {
        for _ in 0..ncount.counts[s].max(0) {
            rows[position].symbol = s as u8;
            position = (position + step) & mask;
            while position > high_threshold {
                position = (position + step) & mask;
            }
        }
    }
    for row in &mut rows[..table_size] {
        let next_state = u32::from(symbol_next[usize::from(row.symbol)]);
        symbol_next[usize::from(row.symbol)] += 1;
        row.nb_bits = (table_log - highbit32(next_state)) as u8;
        row.new_state = ((next_state << row.nb_bits) - table_size as u32) as u16;
    }
    rows
}

/// `FSE_decompress_wksp` of a Huffman weight stream into `weights`, at
/// most `weights.len() - 1` of them; returns how many it decoded.
fn decompress_weights(src: &[u8], weights: &mut [u8; HUF_WEIGHTS]) -> Result<usize, Malformed> {
    let ncount = read_ncount(src, 255)?;
    if ncount.table_log > WEIGHT_TABLE_LOG_MAX {
        return Err(Malformed);
    }
    let rows = build_dtable(&ncount);
    let mut bits = ReverseBits::new(&src[ncount.size..])?;
    let table_log = ncount.table_log;
    let mut states = [bits.read(table_log), bits.read(table_log)];
    if bits.overflowed() {
        return Err(Malformed);
    }
    let capacity = HUF_WEIGHTS - 1;
    let mut count = 0;
    let mut turn = 0;
    loop {
        if count + 2 > capacity {
            return Err(Malformed);
        }
        let row = rows[states[turn] as usize];
        weights[count] = row.symbol;
        count += 1;
        states[turn] = u32::from(row.new_state) + bits.read(u32::from(row.nb_bits));
        if bits.overflowed() {
            weights[count] = rows[states[1 - turn] as usize].symbol;
            return Ok(count + 1);
        }
        turn = 1 - turn;
    }
}

/// Weights of a Huffman description (`HUF_readStats`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HufStats {
    /// Weight per symbol, the implied last one included.
    pub(crate) weights: [u8; HUF_WEIGHTS],
    /// Symbols per weight.
    pub(crate) rank: [u32; HUF_TABLE_LOG_MAX as usize + 1],
    /// Symbols described.
    pub(crate) nb_symbols: u32,
    /// Table log.
    pub(crate) table_log: u32,
    /// Bytes the description occupies.
    pub(crate) size: usize,
}

/// `HUF_readStats`.
pub(crate) fn read_stats(src: &[u8]) -> Result<HufStats, Malformed> {
    let (&header, body) = src.split_first().ok_or(Malformed)?;
    let mut weights = [0u8; HUF_WEIGHTS];
    let mut i_size = usize::from(header);
    let o_size;
    if i_size >= 128 {
        o_size = i_size - 127;
        i_size = o_size.div_ceil(2);
        if i_size + 1 > src.len() || o_size >= HUF_WEIGHTS {
            return Err(Malformed);
        }
        for n in (0..o_size).step_by(2) {
            weights[n] = body[n / 2] >> 4;
            weights[n + 1] = body[n / 2] & 15;
        }
    } else {
        if i_size + 1 > src.len() {
            return Err(Malformed);
        }
        o_size = decompress_weights(&body[..i_size], &mut weights)?;
    }

    let mut rank = [0u32; HUF_TABLE_LOG_MAX as usize + 1];
    let mut weight_total: u32 = 0;
    for &weight in &weights[..o_size] {
        if u32::from(weight) > HUF_TABLE_LOG_MAX {
            return Err(Malformed);
        }
        rank[usize::from(weight)] += 1;
        weight_total += (1 << weight) >> 1;
    }
    if weight_total == 0 {
        return Err(Malformed);
    }
    let table_log = highbit32(weight_total) + 1;
    if table_log > HUF_TABLE_LOG_MAX {
        return Err(Malformed);
    }
    let rest = (1u32 << table_log) - weight_total;
    let verif = 1u32 << highbit32(rest);
    let last_weight = highbit32(rest) + 1;
    if verif != rest {
        return Err(Malformed);
    }
    weights[o_size] = last_weight as u8;
    rank[last_weight as usize] += 1;
    if rank[1] < 2 || rank[1] & 1 != 0 {
        return Err(Malformed);
    }
    Ok(HufStats {
        weights,
        rank,
        nb_symbols: o_size as u32 + 1,
        table_log,
        size: i_size + 1,
    })
}

/// Reads a Huffman description into `table` (`HUF_readCTable` with a
/// maximum symbol of 255).
///
/// Returns the bytes read and whether some symbol has weight zero
/// (`hasZeroWeights`).
pub(crate) fn read_ctable(table: &mut CTable, src: &[u8]) -> Result<(usize, bool), Malformed> {
    let stats = read_stats(src)?;
    let has_zero_weights = stats.rank[0] > 0;
    let table_log = stats.table_log;
    let nb_symbols = stats.nb_symbols as usize;

    *table = CTable::default();
    table.table_log = table_log;
    table.max_symbol_value = nb_symbols as u32 - 1;
    for n in 0..nb_symbols {
        let w = u32::from(stats.weights[n]);
        table.nb_bits[n] = if w == 0 { 0 } else { (table_log + 1 - w) as u8 };
    }

    let mut nb_per_rank = [0u16; HUF_TABLE_LOG_MAX as usize + 2];
    let mut val_per_rank = [0u16; HUF_TABLE_LOG_MAX as usize + 2];
    for n in 0..nb_symbols {
        nb_per_rank[usize::from(table.nb_bits[n])] += 1;
    }
    let mut min: u16 = 0;
    for n in (1..=table_log as usize).rev() {
        val_per_rank[n] = min;
        min += nb_per_rank[n];
        min >>= 1;
    }
    for n in 0..nb_symbols {
        let bits = usize::from(table.nb_bits[n]);
        // Symbols without a code take a value the encoder never writes;
        // the reference stores it, this table keeps zero as `build` does.
        if bits != 0 {
            table.value[n] = val_per_rank[bits];
        }
        val_per_rank[bits] = val_per_rank[bits].wrapping_add(1);
    }
    Ok((stats.size, has_zero_weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::core::entropy::fse;

    #[test]
    fn written_distributions_read_back() {
        for (counts, total, max_symbol) in [
            (&[10u32, 0, 3, 1, 1][..], 15usize, 4u32),
            (
                &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 50][..],
                65,
                15,
            ),
            (
                &[
                    100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                ][..],
                101,
                20,
            ),
        ] {
            let table_log = fse::optimal_table_log(9, total, max_symbol);
            let mut norm = [0i16; 64];
            fse::normalize_count(&mut norm, table_log, counts, total, max_symbol, true)
                .expect("normalizes");
            let mut header = [0u8; 64];
            let size =
                fse::write_ncount(&mut header, &norm, max_symbol, table_log).expect("writes");
            let read = read_ncount(&header[..size], 63).expect("reads");
            assert_eq!(read.size, size);
            assert_eq!(read.table_log, table_log);
            assert_eq!(read.max_symbol, max_symbol);
            assert_eq!(
                &read.counts[..=max_symbol as usize],
                &norm[..=max_symbol as usize]
            );
        }
    }

    #[test]
    fn malformed_distributions_are_rejected() {
        // A table log above 15.
        assert_eq!(
            read_ncount(&[0x0F, 0, 0, 0, 0, 0, 0, 0], 255),
            Err(Malformed)
        );
        // Thirty-two "less than one" probabilities fill a table of 32, but
        // not for a maximum symbol of 30.
        assert!(read_ncount(&[0x00; 8], 255).is_ok());
        assert_eq!(read_ncount(&[0x00; 8], 30), Err(Malformed));
        // Too many symbols for the declared maximum.
        assert!(read_ncount(&[0x30, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55], 1).is_err());
        assert!(read_ncount(&[], 255).is_err());
    }

    #[test]
    fn written_huffman_tables_read_back() {
        let mut count = [0u32; 256];
        for (symbol, slot) in count.iter_mut().enumerate() {
            *slot = (symbol as u32 * 7 + 3) % 41 + 1 + u32::from(symbol < 9) * 400;
        }
        for max_symbol in [255u32, 40, 5] {
            let mut table = CTable::default();
            let log = table.build(&count, max_symbol, 11).expect("builds");
            let mut header = [0u8; 300];
            let size = table.write(&mut header, max_symbol, log).expect("writes");
            let mut read = CTable::default();
            let (used, zero) = read_ctable(&mut read, &header[..size]).expect("reads");
            assert_eq!(used, size);
            assert!(!zero);
            assert_eq!(read.table_log, table.table_log);
            assert_eq!(read.nb_bits, table.nb_bits);
            assert_eq!(read.value, table.value);
        }
        // Absent symbols read back as zero weights.
        let mut sparse = [0u32; 256];
        sparse[..4].copy_from_slice(&[9, 0, 4, 4]);
        let mut table = CTable::default();
        let log = table.build(&sparse, 3, 11).expect("builds");
        let mut header = [0u8; 16];
        let size = table.write(&mut header, 3, log).expect("writes");
        let mut read = CTable::default();
        assert!(read_ctable(&mut read, &header[..size]).expect("reads").1);
        assert_eq!(read.nb_bits, table.nb_bits);
    }

    #[test]
    fn malformed_huffman_descriptions_are_rejected() {
        let mut table = CTable::default();
        assert!(read_ctable(&mut table, &[]).is_err());
        // Direct weights longer than the input.
        assert!(read_stats(&[200, 0x11]).is_err());
        // Direct weights above the table log limit.
        assert!(read_stats(&[129, 0xF0]).is_err());
        // All-zero weights.
        assert!(read_stats(&[129, 0x00]).is_err());
        // Weights whose remainder is not a power of two.
        assert!(read_stats(&[131, 0x12, 0x20]).is_err());
        assert!(read_stats(&[131, 0x12, 0x30]).is_ok());
        // A single weight-1 symbol cannot complete a tree.
        assert!(read_stats(&[130, 0x21]).is_err());
        // An FSE header longer than the input, and a stream without marker.
        assert!(read_stats(&[5, 0x00]).is_err());
        assert!(ReverseBits::new(&[0x00]).is_err());
        assert!(ReverseBits::new(&[]).is_err());
    }

    /// Inputs around valid descriptions: the valid ones, every truncation,
    /// single-bit flips and random bytes.
    fn mutations(valid: &[alloc::vec::Vec<u8>], seed: u64) -> alloc::vec::Vec<alloc::vec::Vec<u8>> {
        let mut rng = zstd_ffi::corpus::Rng::new(seed);
        let mut out = alloc::vec::Vec::new();
        for header in valid {
            out.push(header.clone());
            for len in 0..header.len() {
                out.push(header[..len].to_vec());
            }
            for _ in 0..200 {
                let mut flipped = header.clone();
                let bit = (rng.next_u64() % (8 * header.len() as u64)) as usize;
                flipped[bit / 8] ^= 1 << (bit % 8);
                out.push(flipped);
            }
        }
        for _ in 0..3000 {
            let len = (rng.next_u64() % 24) as usize;
            out.push((0..len).map(|_| rng.next_u64() as u8).collect());
        }
        out
    }

    #[test]
    fn distributions_read_like_the_reference() {
        let mut valid = alloc::vec::Vec::new();
        for (seed, max_symbol) in [(1u64, 4u32), (2, 15), (3, 31), (4, 52), (5, 35)] {
            let mut rng = zstd_ffi::corpus::Rng::new(seed);
            let counts: alloc::vec::Vec<u32> = (0..=max_symbol)
                .map(|_| {
                    (rng.next_u64() % 50) as u32 * u32::from(!rng.next_u64().is_multiple_of(3))
                })
                .collect();
            let total: usize = counts.iter().map(|&c| c as usize).sum();
            if total < 2 {
                continue;
            }
            let table_log = fse::optimal_table_log(9, total, max_symbol);
            let mut norm = [0i16; 64];
            if fse::normalize_count(&mut norm, table_log, &counts, total, max_symbol, true).is_err()
            {
                continue;
            }
            let mut header = [0u8; 64];
            let size =
                fse::write_ncount(&mut header, &norm, max_symbol, table_log).expect("writes");
            valid.push(header[..size].to_vec());
        }
        for input in mutations(&valid, 9) {
            for max_symbol in [255u32, 52, 31, 8] {
                let ours = read_ncount(&input, max_symbol);
                let theirs = zstd_ffi::oracle::fse_read_ncount(&input, max_symbol);
                match (ours, theirs) {
                    (Ok(ours), Ok(theirs)) => {
                        assert_eq!(ours.size, theirs.size, "{input:?}");
                        assert_eq!(ours.table_log, theirs.table_log, "{input:?}");
                        assert_eq!(ours.max_symbol, theirs.max_symbol, "{input:?}");
                        assert_eq!(
                            &ours.counts[..=ours.max_symbol as usize],
                            &theirs.counts[..],
                            "{input:?}"
                        );
                    }
                    (Err(_), Err(_)) => {}
                    (ours, theirs) => panic!("{input:?} max {max_symbol}: {ours:?} vs {theirs:?}"),
                }
            }
        }
    }

    #[test]
    fn huffman_descriptions_read_like_the_reference() {
        let mut valid = alloc::vec::Vec::new();
        for seed in 1..12u64 {
            let mut rng = zstd_ffi::corpus::Rng::new(seed);
            let max_symbol = (rng.next_u64() % 256) as u32;
            let count: alloc::vec::Vec<u32> = (0..256)
                .map(|_| {
                    (rng.next_u64() % 300) as u32 * u32::from(!rng.next_u64().is_multiple_of(4))
                })
                .collect();
            let mut table = CTable::default();
            let Ok(log) = table.build(&count, max_symbol, 11) else {
                continue;
            };
            let mut header = [0u8; 300];
            if let Ok(size) = table.write(&mut header, max_symbol, log) {
                valid.push(header[..size].to_vec());
            }
        }
        assert!(valid.len() > 5);
        for input in mutations(&valid, 10) {
            let mut table = CTable::default();
            let ours = read_ctable(&mut table, &input);
            let theirs = zstd_ffi::oracle::huf_read_ctable(&input);
            match (ours, theirs) {
                (Ok((size, zero)), Ok(theirs)) => {
                    assert_eq!(size, theirs.size, "{input:?}");
                    assert_eq!(zero, theirs.has_zero_weights, "{input:?}");
                    assert_eq!(table.table_log, theirs.table_log, "{input:?}");
                    let symbols = theirs.nb_bits.len();
                    assert_eq!(table.max_symbol_value as usize + 1, symbols, "{input:?}");
                    assert_eq!(&table.nb_bits[..symbols], &theirs.nb_bits[..], "{input:?}");
                    assert_eq!(&table.value[..symbols], &theirs.values[..], "{input:?}");
                    assert!(table.nb_bits[symbols..].iter().all(|&b| b == 0));
                }
                (Err(_), Err(_)) => {}
                (ours, theirs) => panic!("{input:?}: {ours:?} vs {theirs:?}"),
            }
        }
    }
}

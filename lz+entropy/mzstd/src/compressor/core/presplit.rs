//! Block boundary selection before match finding (`zstd_preSplit.c`).
//!
//! A full 128 KiB block may be cut short where its byte statistics change.
//! The cut depends only on the input bytes and the strategy-derived level,
//! but it moves every following block boundary, so it is ported literally,
//! including its integer rounding of event counts.

use fearless_simd::Simd;
use fearless_simd_macros::simd;

use crate::Backend;

use super::entropy::hist;

const THRESHOLD_PENALTY_RATE: u64 = 16;
const THRESHOLD_BASE: u64 = THRESHOLD_PENALTY_RATE - 2;
const THRESHOLD_PENALTY: i32 = 3;
const HASH_LENGTH: usize = 2;
const HASH_LOG_MAX: u32 = 10;
const HASH_TABLE_SIZE: usize = 1 << HASH_LOG_MAX;
const KNUTH: u32 = 0x9E37_79B9;
const CHUNK_SIZE: usize = 8 << 10;
const SEGMENT_SIZE: usize = 512;
const KB: usize = 1 << 10;

/// Event histogram of a stretch of input (`Fingerprint`).
#[derive(Clone, Debug)]
struct Fingerprint {
    events: [u32; HASH_TABLE_SIZE],
    nb_events: usize,
}

impl Fingerprint {
    const fn new() -> Self {
        Self {
            events: [0; HASH_TABLE_SIZE],
            nb_events: 0,
        }
    }
}

/// `hash2`: the first byte for 8-bit tables, a hashed pair otherwise. Every
/// result is below `1 << LOG`.
#[inline(always)]
const fn hash2<const LOG: u32>(first: u8, second: u8) -> usize {
    if LOG == 8 {
        return first as usize;
    }
    let pair = u16::from_le_bytes([first, second]) as u32;
    (pair.wrapping_mul(KNUTH) >> (32 - LOG)) as usize
}

/// `addEvents_generic` after clearing (`recordFingerprint_generic`), for one
/// of the reference's compiled-in sampling rates and hash logs
/// (`ZSTD_GEN_RECORD_FINGERPRINT`): with both constant, the stride is a
/// constant and every hashed index provably fits the table.
#[inline(always)]
fn record<const RATE: usize, const LOG: u32>(fp: &mut Fingerprint, src: &[u8]) {
    let events = &mut fp.events;
    events[..1 << LOG].fill(0);
    for pair in src.windows(HASH_LENGTH).step_by(RATE) {
        events[hash2::<LOG>(pair[0], pair[1])] += 1;
    }
    fp.nb_events = (src.len() - HASH_LENGTH + 1) / RATE;
}

/// `fpDistance` over the first `1 << LOG` events.
///
/// At most fifteen 8191-event chunks are compared with one 8191-event
/// chunk. Each product is below 2^30, and their total absolute difference
/// is at most twice that product, below 2^31. Computing in 32 bits keeps
/// the reference's exact result while doubling the vector lane count.
#[inline(always)]
fn distance<const LOG: u32>(fp1: &Fingerprint, fp2: &Fingerprint) -> u64 {
    let len = 1usize << LOG;
    let nb1 = fp1.nb_events as i32;
    let nb2 = fp2.nb_events as i32;
    fp1.events[..len]
        .iter()
        .zip(&fp2.events[..len])
        .fold(0u32, |total, (&a, &b)| {
            total + (a as i32 * nb2 - b as i32 * nb1).unsigned_abs()
        }) as u64
}

/// Whether `new` deviates from `reference` enough to cut
/// (`compareFingerprints`).
#[inline(always)]
fn differs<const LOG: u32>(reference: &Fingerprint, new: &Fingerprint, penalty: i32) -> bool {
    let p50 = reference.nb_events as u64 * new.nb_events as u64;
    let deviation = distance::<LOG>(reference, new);
    let threshold = p50 * (THRESHOLD_BASE + penalty as u64) / THRESHOLD_PENALTY_RATE;
    deviation >= threshold
}

/// `ZSTD_splitBlock_byChunks`: `level` picks the sampling rate and hash log
/// (`records_fs`, `hashParams`).
#[inline(always)]
fn split_by_chunks(block: &[u8], level: usize) -> usize {
    match level {
        0 => split_by_chunks_with::<43, 8>(block),
        1 => split_by_chunks_with::<11, 9>(block),
        2 => split_by_chunks_with::<5, 10>(block),
        _ => split_by_chunks_with::<1, 10>(block),
    }
}

/// [`split_by_chunks`] for one sampling rate and hash log.
#[inline(always)]
fn split_by_chunks_with<const RATE: usize, const LOG: u32>(block: &[u8]) -> usize {
    let block_size = block.len();
    debug_assert_eq!(block_size, 128 * KB);
    let mut past = Fingerprint::new();
    let mut new = Fingerprint::new();
    let mut penalty = THRESHOLD_PENALTY;

    record::<RATE, LOG>(&mut past, &block[..CHUNK_SIZE]);
    let mut pos = CHUNK_SIZE;
    while pos <= block_size - CHUNK_SIZE {
        record::<RATE, LOG>(&mut new, &block[pos..pos + CHUNK_SIZE]);
        if differs::<LOG>(&past, &new, penalty) {
            return pos;
        }
        for (acc, &add) in past.events[..1 << LOG]
            .iter_mut()
            .zip(&new.events[..1 << LOG])
        {
            *acc += add;
        }
        past.nb_events += new.nb_events;
        if penalty > 0 {
            penalty -= 1;
        }
        pos += CHUNK_SIZE;
    }
    block_size
}

/// `ZSTD_splitBlock_fromBorders`: compares the block's ends, then its middle.
#[inline(always)]
fn split_from_borders(block: &[u8]) -> usize {
    let block_size = block.len();
    let mut begin = [0u32; 256];
    let mut end = [0u32; 256];
    hist::add(&mut begin, &block[..SEGMENT_SIZE]);
    hist::add(&mut end, &block[block_size - SEGMENT_SIZE..]);
    let as_fingerprint = |counts: &[u32; 256]| {
        let mut fp = Fingerprint::new();
        fp.events[..256].copy_from_slice(counts);
        fp.nb_events = SEGMENT_SIZE;
        fp
    };
    let past = as_fingerprint(&begin);
    let new = as_fingerprint(&end);
    if !differs::<8>(&past, &new, 0) {
        return block_size;
    }
    let mut middle_counts = [0u32; 256];
    let mid = block_size / 2 - SEGMENT_SIZE / 2;
    hist::add(&mut middle_counts, &block[mid..mid + SEGMENT_SIZE]);
    let middle = as_fingerprint(&middle_counts);
    let from_begin = distance::<8>(&past, &middle);
    let from_end = distance::<8>(&new, &middle);
    let min_distance = (SEGMENT_SIZE * SEGMENT_SIZE / 3) as u64;
    if (from_begin as i64 - from_end as i64).unsigned_abs() < min_distance {
        return 64 * KB;
    }
    if from_begin > from_end {
        32 * KB
    } else {
        96 * KB
    }
}

/// Size of the first block to cut from a full 128 KiB `block`
/// (`ZSTD_splitBlock`); `level` is `0..=4`.
pub(crate) fn split_block(block: &[u8], level: usize, backend: Backend) -> usize {
    fearless_simd::dispatch!(backend.0, simd => split_block_on(simd, block, level))
}

/// Selects the block boundary inside the encoder's feature context, so
/// histogram reductions use its native integer arithmetic.
#[simd]
#[inline(never)]
fn split_block_on<S: Simd>(simd: S, block: &[u8], level: usize) -> usize {
    if level == 0 {
        return split_from_borders(block);
    }
    split_by_chunks(block, level - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn two_halves() -> Vec<u8> {
        let mut block = Vec::with_capacity(128 * KB);
        block.extend((0..64 * KB).map(|i| b"abcd"[i % 4]));
        let mut x = 1u32;
        block.extend((0..64 * KB).map(|_| {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (x >> 16) as u8
        }));
        block
    }

    #[test]
    fn uniform_blocks_are_not_split() {
        let block: Vec<u8> = (0..128 * KB).map(|i| (i % 251) as u8).collect();
        for level in 0..=4 {
            for backend in Backend::available() {
                assert_eq!(
                    split_block(&block, level, backend),
                    128 * KB,
                    "{backend} level {level}"
                );
            }
        }
    }

    #[test]
    fn a_change_of_statistics_is_found() {
        let block = two_halves();
        for backend in Backend::available() {
            assert_eq!(split_block(&block, 4, backend), 64 * KB);
            assert_eq!(split_block(&block, 1, backend), 64 * KB);
            let cut = split_block(&block, 0, backend);
            assert!([32 * KB, 64 * KB, 96 * KB].contains(&cut));
        }
    }

    #[test]
    fn hash2_uses_the_first_byte_for_eight_bits() {
        assert_eq!(hash2::<8>(7, 9), 7);
        assert!(hash2::<10>(7, 9) < 1024);
        assert!(hash2::<9>(255, 255) < 512);
    }

    /// `recordFingerprint_generic`, `fpDistance` and
    /// `ZSTD_splitBlock_byChunks` with run-time parameters, step by step as
    /// the reference writes them.
    fn reference_by_chunks(block: &[u8], level: usize) -> usize {
        const SAMPLING: [usize; 4] = [43, 11, 5, 1];
        const HASH_PARAMS: [u32; 4] = [8, 9, 10, 10];
        let (rate, log) = (SAMPLING[level], HASH_PARAMS[level]);
        let record = |src: &[u8]| {
            let mut events = alloc::vec![0u64; 1 << log];
            let limit = src.len() - HASH_LENGTH + 1;
            let mut n = 0;
            while n < limit {
                let h = if log == 8 {
                    usize::from(src[n])
                } else {
                    let pair = u32::from(src[n]) | (u32::from(src[n + 1]) << 8);
                    (pair.wrapping_mul(KNUTH) >> (32 - log)) as usize
                };
                events[h] += 1;
                n += rate;
            }
            (events, (limit / rate) as u64)
        };
        let distance = |a: &(Vec<u64>, u64), b: &(Vec<u64>, u64)| {
            let mut total = 0u64;
            for n in 0..1usize << log {
                let x = a.0[n] as i64 * b.1 as i64 - b.0[n] as i64 * a.1 as i64;
                total = total.wrapping_add(x.unsigned_abs());
            }
            total
        };
        let mut past = record(&block[..CHUNK_SIZE]);
        let mut penalty = THRESHOLD_PENALTY;
        let mut pos = CHUNK_SIZE;
        while pos <= block.len() - CHUNK_SIZE {
            let new = record(&block[pos..pos + CHUNK_SIZE]);
            let threshold =
                past.1 * new.1 * (THRESHOLD_BASE + penalty as u64) / THRESHOLD_PENALTY_RATE;
            if distance(&past, &new) >= threshold {
                return pos;
            }
            for (acc, add) in past.0.iter_mut().zip(&new.0) {
                *acc += add;
            }
            past.1 += new.1;
            penalty = (penalty - 1).max(0);
            pos += CHUNK_SIZE;
        }
        block.len()
    }

    #[test]
    fn chunk_splits_match_the_reference_loop() {
        let mut x = 0x2545_F491u32;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        let mut blocks = alloc::vec![two_halves()];
        // Text-like, binary-like and drifting statistics, and a change
        // late in the block.
        blocks.push(
            (0..128 * KB)
                .map(|i| b"the quick brown fox "[i % 20])
                .collect(),
        );
        blocks.push((0..128 * KB).map(|_| (next() % 7) as u8 * 31).collect());
        blocks.push(
            (0..128 * KB)
                .map(|i| (next() % (2 + (i / (16 * KB)) as u32 * 30)) as u8)
                .collect(),
        );
        let mut late: Vec<u8> = (0..128 * KB).map(|i| (i % 13) as u8).collect();
        for byte in &mut late[120 * KB..] {
            *byte = next() as u8;
        }
        blocks.push(late);
        for (index, block) in blocks.iter().enumerate() {
            for level in 0..4 {
                let expected = reference_by_chunks(block, level);
                assert_eq!(split_by_chunks(block, level), expected);
                for backend in Backend::available() {
                    assert_eq!(
                        split_block(block, level + 1, backend),
                        expected,
                        "{backend} block {index}, level {level}"
                    );
                }
            }
        }
    }

    #[test]
    fn distance_preserves_the_largest_chunk_products() {
        let mut past = Fingerprint::new();
        let mut new = Fingerprint::new();
        past.nb_events = 15 * (CHUNK_SIZE - 1);
        new.nb_events = CHUNK_SIZE - 1;
        past.events[0] = past.nb_events as u32;
        new.events[1] = new.nb_events as u32;
        let expected = 2 * past.nb_events as u64 * new.nb_events as u64;
        assert_eq!(distance::<10>(&past, &new), expected);
        assert_eq!(distance::<10>(&new, &past), expected);
        assert_eq!(distance::<10>(&new, &new), 0);
    }
}

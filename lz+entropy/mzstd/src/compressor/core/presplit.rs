//! Block boundary selection before match finding (`zstd_preSplit.c`).
//!
//! A full 128 KiB block may be cut short where its byte statistics change.
//! The cut depends only on the input bytes and the strategy-derived level,
//! but it moves every following block boundary, so it is ported literally,
//! including its integer rounding of event counts.

use alloc::boxed::Box;

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
    events: Box<[u32; HASH_TABLE_SIZE]>,
    nb_events: usize,
}

impl Fingerprint {
    fn new() -> Self {
        Self {
            events: Box::new([0; HASH_TABLE_SIZE]),
            nb_events: 0,
        }
    }
}

/// `hash2`: the first byte for 8-bit tables, a hashed pair otherwise.
fn hash2(p: &[u8], hash_log: u32) -> usize {
    if hash_log == 8 {
        return usize::from(p[0]);
    }
    let pair = u32::from(u16::from_le_bytes([p[0], p[1]]));
    (pair.wrapping_mul(KNUTH) >> (32 - hash_log)) as usize
}

/// `addEvents_generic` after clearing (`recordFingerprint_generic`).
fn record(fp: &mut Fingerprint, src: &[u8], sampling_rate: usize, hash_log: u32) {
    fp.events[..1 << hash_log].fill(0);
    fp.nb_events = 0;
    let limit = src.len() - HASH_LENGTH + 1;
    let mut n = 0;
    while n < limit {
        fp.events[hash2(&src[n..], hash_log)] += 1;
        n += sampling_rate;
    }
    fp.nb_events += limit / sampling_rate;
}

/// `fpDistance`.
fn distance(fp1: &Fingerprint, fp2: &Fingerprint, hash_log: u32) -> u64 {
    let mut total = 0u64;
    for n in 0..1usize << hash_log {
        let a = i64::from(fp1.events[n]) * fp2.nb_events as i64;
        let b = i64::from(fp2.events[n]) * fp1.nb_events as i64;
        total = total.wrapping_add((a - b).unsigned_abs());
    }
    total
}

/// Whether `new` deviates from `reference` enough to cut
/// (`compareFingerprints`).
fn differs(reference: &Fingerprint, new: &Fingerprint, penalty: i32, hash_log: u32) -> bool {
    let p50 = reference.nb_events as u64 * new.nb_events as u64;
    let deviation = distance(reference, new, hash_log);
    let threshold = p50 * (THRESHOLD_BASE + penalty as u64) / THRESHOLD_PENALTY_RATE;
    deviation >= threshold
}

/// `ZSTD_splitBlock_byChunks`.
fn split_by_chunks(block: &[u8], level: usize) -> usize {
    const SAMPLING: [usize; 4] = [43, 11, 5, 1];
    const HASH_PARAMS: [u32; 4] = [8, 9, 10, 10];
    let sampling = SAMPLING[level];
    let hash_log = HASH_PARAMS[level];
    let block_size = block.len();
    debug_assert_eq!(block_size, 128 * KB);
    let mut past = Fingerprint::new();
    let mut new = Fingerprint::new();
    let mut penalty = THRESHOLD_PENALTY;

    record(&mut past, &block[..CHUNK_SIZE], sampling, hash_log);
    let mut pos = CHUNK_SIZE;
    while pos <= block_size - CHUNK_SIZE {
        record(&mut new, &block[pos..pos + CHUNK_SIZE], sampling, hash_log);
        if differs(&past, &new, penalty, hash_log) {
            return pos;
        }
        for (acc, &add) in past.events.iter_mut().zip(new.events.iter()) {
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
    if !differs(&past, &new, 0, 8) {
        return block_size;
    }
    let mut middle_counts = [0u32; 256];
    let mid = block_size / 2 - SEGMENT_SIZE / 2;
    hist::add(&mut middle_counts, &block[mid..mid + SEGMENT_SIZE]);
    let middle = as_fingerprint(&middle_counts);
    let from_begin = distance(&past, &middle, 8);
    let from_end = distance(&new, &middle, 8);
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
pub(crate) fn split_block(block: &[u8], level: usize) -> usize {
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
            assert_eq!(split_block(&block, level), 128 * KB, "level {level}");
        }
    }

    #[test]
    fn a_change_of_statistics_is_found() {
        let block = two_halves();
        assert_eq!(split_block(&block, 4), 64 * KB);
        assert_eq!(split_block(&block, 1), 64 * KB);
        let cut = split_block(&block, 0);
        assert!([32 * KB, 64 * KB, 96 * KB].contains(&cut));
    }

    #[test]
    fn hash2_uses_the_first_byte_for_eight_bits() {
        assert_eq!(hash2(&[7, 9], 8), 7);
        assert!(hash2(&[7, 9], 10) < 1024);
    }
}

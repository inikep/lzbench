//! The continuation-byte filter over the sixteen short distance codes.
//!
//! Both the greedy bucket matchers (qualities 7–9) and the high-quality search
//! (qualities 10 and 11) try the distance cache before anything else, and most
//! of those probes fail on one byte: the copy would have to agree with the
//! byte at `cur_ix + best_len`. The probes are not sixteen scattered
//! distances. Codes 0 and 4–9 are the freshest cached distance nudged by
//! `+3..=-3`, which names seven adjacent ring bytes; codes 1 and 10–15 are seven
//! more around the second freshest; codes 2 and 3 are single bytes. Two
//! eight-byte reads therefore fill one `u8x16`, and one comparison tests every
//! code at once.
//!
//! The result is a superset of the codes the scalar filter admits, so callers
//! keep that filter for each code they visit; the reference's order of visits
//! and its decisions are unchanged.

use fearless_simd::{Select, Simd, SimdBase, SimdMask, u8x16, u32x16};

/// The short distance code each byte lane tests.
///
/// Lanes `0..7` hold the freshest distance's run from its farthest byte
/// (distance `+3`) to its nearest (`-3`), lane 7 the third distance; lanes
/// `8..16` repeat that for the second and fourth distances.
const LANE_CODES: [u8; 16] = [9, 7, 5, 0, 4, 6, 8, 2, 15, 13, 11, 1, 10, 12, 14, 3];

/// Maps one byte of a lane mask to the code bits it stands for: the low byte
/// of the mask through the first table, the high byte through the second.
const LANE_CODE_BITS: [[u16; 256]; 2] = {
    let mut tables = [[0u16; 256]; 2];
    let mut half = 0;
    while half < 2 {
        let mut byte = 0;
        while byte < 256 {
            let mut lane = 0;
            while lane < 8 {
                if byte & (1 << lane) != 0 {
                    tables[half][byte] |= 1 << LANE_CODES[half * 8 + lane];
                }
                lane += 1;
            }
            byte += 1;
        }
        half += 1;
    }
    tables
};

/// The lanes of one run, forced through when its bytes cannot be read as one
/// contiguous word.
const RUN_LANES: u16 = 0x7f;

/// Which remembered distance each short code reads (`kDistanceCacheIndex`).
const CODE_SLOTS: [u32; 16] = [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

/// What each short code adds to it (`kDistanceCacheOffset`), as a wrapping
/// `u32`.
const CODE_OFFSETS: [u32; 16] = {
    const SIGNED: [i32; 16] = [0, 0, 0, 0, -1, 1, -2, 2, -3, 3, -1, 1, -2, 2, -3, 3];
    let mut offsets = [0u32; 16];
    let mut code = 0;
    while code < 16 {
        offsets[code] = SIGNED[code] as u32;
        code += 1;
    }
    offsets
};

/// The sixteen short codes' distances in code order, derived from the four
/// remembered ones as `PrepareDistanceCache` does.
///
/// Distances are carried as wrapping `u32`s: an `i32` keeps its bits, so a
/// zero or negative distance stays outside every admissible range.
#[inline(always)]
pub(crate) fn derived_distances<S: Simd>(simd: S, remembered: &[i32; 4]) -> u32x16<S> {
    let slots = u32x16::load_array(simd, CODE_SLOTS);
    let mut distances = u32x16::splat(simd, remembered[0] as u32);
    for (slot, &distance) in remembered.iter().enumerate().skip(1) {
        distances = slots
            .simd_eq(u32x16::splat(simd, slot as u32))
            .select(u32x16::splat(simd, distance as u32), distances);
    }
    distances + u32x16::load_array(simd, CODE_OFFSETS)
}

/// One search position's view of the ring.
#[derive(Copy, Clone)]
pub(crate) struct ProbeSite<'a> {
    /// The addressable ring: never longer than its mask admits, so a probe
    /// past it is the reference's out-of-window test.
    pub(crate) window: &'a [u8],
    pub(crate) cur_ix: usize,
    pub(crate) mask: usize,
    /// The farthest admissible backward distance, below 2^31.
    pub(crate) max_distance: usize,
}

/// One position's filter result, both masks in code order.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Candidates {
    /// Codes whose byte at `best_len` agrees with the continuation byte, or
    /// that could not be read as part of a contiguous run. Not range-checked.
    pub(crate) agree: u16,
    /// Codes whose distance lies in `1..=max_distance`.
    pub(crate) valid: u16,
}

impl ProbeSite<'_> {
    /// Filters the sixteen short codes against `continuation`.
    ///
    /// `distances` are the codes' distances in code order; `remembered` are
    /// the first four of them, from which codes 4–15 must be derived as
    /// [`derived_distances`] derives them — the byte runs are read around
    /// the first two. Without `WIDE`, only codes 0–9 are of interest: the
    /// second run is not read and codes 10–15 are meaningless.
    #[inline(always)]
    pub(crate) fn candidates<S: Simd, const WIDE: bool>(
        self,
        simd: S,
        distances: u32x16<S>,
        remembered: &[i32; 4],
        best_len: usize,
        continuation: u8,
    ) -> Candidates {
        debug_assert!(self.max_distance < 1 << 31);
        // The scalar `backward.wrapping_sub(1) < max_distance`, lane by lane:
        // a zero distance wraps below the bound and a negative one above it.
        let valid = (distances - u32x16::splat(simd, 1))
            .simd_lt(u32x16::splat(simd, self.max_distance as u32))
            .to_bitmask() as u16;

        let (low, low_forced) = self.run(remembered[0], remembered[2], best_len, continuation);
        let (high, high_forced) = if WIDE {
            self.run(remembered[1], remembered[3], best_len, continuation)
        } else {
            // Code 1 alone, in its run lane, beside the fourth distance.
            let second = self.byte(remembered[1], best_len, continuation);
            let fourth = self.byte(remembered[3], best_len, continuation);
            (u64::from(second) << 24 | u64::from(fourth) << 56, 0)
        };
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&low.to_le_bytes());
        bytes[8..].copy_from_slice(&high.to_le_bytes());
        let lanes = u8x16::load_array(simd, bytes)
            .simd_eq(u8x16::splat(simd, continuation))
            .to_bitmask() as u16
            | low_forced
            | high_forced << 8;
        Candidates {
            agree: LANE_CODE_BITS[0][usize::from(lanes as u8)]
                | LANE_CODE_BITS[1][usize::from(lanes >> 8)],
            valid,
        }
    }

    /// Reads the byte at `best_len` for the seven distances around `nudged`
    /// and for `single`, returning them as one word in lane order and the
    /// run lanes that could not be read that way.
    #[inline(always)]
    fn run(self, nudged: i32, single: i32, best_len: usize, continuation: u8) -> (u64, u16) {
        // Lane `i` probes `nudged + 3 - i`; with no wrap inside the run its
        // masked position is `start + i`, the scalar probe's own.
        let start = self
            .cur_ix
            .wrapping_sub((nudged as isize).wrapping_add(3) as usize)
            & self.mask;
        let single = self.byte(single, best_len, continuation);
        // Positions are ring offsets taken modulo 2^64. Below a 32-bit mask
        // nothing here wraps; under an all-ones mask a wrapped start still
        // names each lane's own position, and a run straddling the top of
        // the address space starts out of bounds and is forced through.
        let word = (start.wrapping_add(6) <= self.mask)
            .then(|| self.window.get(start.wrapping_add(best_len)..))
            .flatten()
            .and_then(<[u8]>::first_chunk::<8>);
        match word {
            Some(word) => (
                u64::from_le_bytes(*word) & (u64::MAX >> 8) | u64::from(single) << 56,
                0,
            ),
            None => (u64::from(single) << 56, RUN_LANES),
        }
    }

    /// The byte a copy from `distance` would have to agree on, or a byte that
    /// cannot agree when there is none.
    #[inline(always)]
    fn byte(self, distance: i32, best_len: usize, continuation: u8) -> u8 {
        let at = (self.cur_ix.wrapping_sub(distance as isize as usize) & self.mask)
            .wrapping_add(best_len);
        self.window.get(at).copied().unwrap_or(!continuation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use fearless_simd::{Level, dispatch};

    /// Every SIMD level the host runs, plus the scalar fallback.
    fn levels() -> [Level; 2] {
        [
            Level::try_detect().unwrap_or_else(Level::baseline),
            Level::fallback(),
        ]
    }

    /// The codes the reference's sequential filter would admit, in code order.
    fn admitted_by_scalar_filter(
        site: ProbeSite<'_>,
        distances: &[i64; 16],
        best_len: usize,
        continuation: u8,
    ) -> (u16, u16) {
        let (mut agree, mut valid) = (0u16, 0u16);
        for (code, &distance) in distances.iter().enumerate() {
            let backward = distance as usize;
            if backward.wrapping_sub(1) < site.max_distance {
                valid |= 1 << code;
                let prev_ix = (site.cur_ix - backward) & site.mask;
                if site.window.get(prev_ix + best_len) == Some(&continuation) {
                    agree |= 1 << code;
                }
            }
        }
        (agree, valid)
    }

    /// Widened code-order distances, so extreme test inputs cannot overflow.
    fn scalar_distances(remembered: &[i32; 4]) -> [i64; 16] {
        core::array::from_fn(|code| {
            i64::from(remembered[CODE_SLOTS[code] as usize]) + i64::from(CODE_OFFSETS[code] as i32)
        })
    }

    fn run_filter<S: Simd>(
        simd: S,
        site: ProbeSite<'_>,
        remembered: &[i32; 4],
        best_len: usize,
        continuation: u8,
        wide: bool,
    ) -> Candidates {
        let distances = derived_distances(simd, remembered);
        if wide {
            site.candidates::<S, true>(simd, distances, remembered, best_len, continuation)
        } else {
            site.candidates::<S, false>(simd, distances, remembered, best_len, continuation)
        }
    }

    fn derived_on<S: Simd>(simd: S, remembered: &[i32; 4]) -> [u32; 16] {
        let mut lanes = [0u32; 16];
        derived_distances(simd, remembered).store_array(&mut lanes);
        lanes
    }

    #[test]
    fn lane_tables_name_every_short_code_once() {
        let mut seen = 0u16;
        for code in LANE_CODES {
            seen |= 1 << code;
        }
        assert_eq!(seen, u16::MAX);
        assert_eq!(LANE_CODE_BITS[0][0xff] | LANE_CODE_BITS[1][0xff], u16::MAX);
        assert_eq!(LANE_CODE_BITS[0][0xff] & LANE_CODE_BITS[1][0xff], 0);
        for (lane, &code) in LANE_CODES.iter().enumerate() {
            let code = usize::from(code);
            let (slot, offset) = match lane {
                7 => (2, 0),
                15 => (3, 0),
                _ => (u32::from(lane >= 8), 3 - (lane % 8) as i32),
            };
            assert_eq!(
                (CODE_SLOTS[code], CODE_OFFSETS[code] as i32),
                (slot, offset),
                "lane {lane}"
            );
        }
    }

    #[test]
    fn derived_distances_match_prepare_distance_cache() {
        let remembered = [100, 50, 7, i32::MIN];
        let mut cache = [0i32; 16];
        cache[..4].copy_from_slice(&remembered);
        crate::compressor::core::greedy::hashers::prepare_distance_cache(&mut cache, 16);
        let expected = cache.map(|distance| distance as u32);
        for level in levels() {
            assert_eq!(
                dispatch!(level, simd => derived_on(simd, &remembered)),
                expected
            );
        }
    }

    #[test]
    fn candidates_cover_every_code_the_scalar_filter_admits() {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        // Few byte values, so continuation bytes agree often.
        let ring: Vec<u8> = (0..256).map(|_| (next() % 3) as u8).collect();
        let around =
            |anchor: i64, next: &mut dyn FnMut() -> u64| (anchor + (next() % 13) as i64 - 6) as i32;
        for level in levels() {
            for round in 0..20_000 {
                // Short windows model a ring holding less than its mask admits.
                let len = if round % 5 == 0 {
                    (next() % 256) as usize + 1
                } else {
                    256
                };
                let cur_ix = (next() % 600) as usize;
                let max_distance = cur_ix.min((next() % 200) as usize);
                let site = ProbeSite {
                    window: &ring[..len],
                    cur_ix,
                    mask: 255,
                    max_distance,
                };
                let edge = max_distance as i64;
                let remembered = [
                    around(if round % 2 == 0 { 3 } else { edge }, &mut next),
                    around((next() % 300) as i64, &mut next),
                    around(edge, &mut next),
                    match round % 7 {
                        0 => i32::MIN,
                        1 => i32::MAX,
                        _ => around(1, &mut next),
                    },
                ];
                let best_len = (next() % 12) as usize;
                let continuation = (next() % 3) as u8;
                let (agree, valid) = admitted_by_scalar_filter(
                    site,
                    &scalar_distances(&remembered),
                    best_len,
                    continuation,
                );
                for wide in [true, false] {
                    let codes = if wide { u16::MAX } else { 0x03ff };
                    let found = dispatch!(level, simd => run_filter(
                        simd, site, &remembered, best_len, continuation, wide));
                    assert_eq!(found.valid & codes, valid & codes, "valid: round {round}");
                    assert_eq!(
                        agree & valid & codes & !found.agree,
                        0,
                        "missed codes: level {level:?}, round {round}, cur_ix {cur_ix}, len {len}, \
                         max {max_distance}, cache {remembered:?}, best_len {best_len}, wide {wide}"
                    );
                }
            }
        }
    }

    #[test]
    fn candidates_are_exact_when_both_runs_are_contiguous() {
        let window: Vec<u8> = (0..=255u8).collect();
        let site = ProbeSite {
            window: &window,
            cur_ix: 200,
            mask: 255,
            max_distance: 200,
        };
        let remembered = [10, 40, 7, 90];
        let best_len = 2;
        for level in levels() {
            for continuation in 0..=255u8 {
                let (agree, valid) = admitted_by_scalar_filter(
                    site,
                    &scalar_distances(&remembered),
                    best_len,
                    continuation,
                );
                let found = dispatch!(level, simd => run_filter(
                    simd, site, &remembered, best_len, continuation, true));
                assert_eq!(
                    found.agree & found.valid,
                    agree,
                    "continuation {continuation}"
                );
                assert_eq!(found.valid, valid);
            }
            // Distance 10 at offset 2 reads byte 192: code 0 alone.
            let found =
                dispatch!(level, simd => run_filter(simd, site, &remembered, best_len, 192, true));
            assert_eq!(found.agree, 1);
            // Narrow: the second distance's byte (162) still reaches code 1.
            let found =
                dispatch!(level, simd => run_filter(simd, site, &remembered, best_len, 162, false));
            assert_eq!(found.agree & 0x03ff, 1 << 1);
        }
    }

    #[test]
    fn a_run_that_wraps_or_leaves_the_ring_is_let_through_whole() {
        let window = [5u8; 64];
        let wrapped = ProbeSite {
            window: &window,
            // Distances 9 down to 3 reach positions 63, 0, 1, ...: a wrap.
            cur_ix: 64 + 2 + 4,
            mask: 63,
            max_distance: 64,
        };
        let truncated = ProbeSite {
            window: &window[..40],
            cur_ix: 40,
            mask: 63,
            max_distance: 40,
        };
        let unmasked = ProbeSite {
            window: &window,
            cur_ix: 2,
            mask: usize::MAX,
            max_distance: 2,
        };
        let run_codes = LANE_CODE_BITS[0][usize::from(RUN_LANES)];
        for level in levels() {
            // Distances 7 down to 1 from 40 start at 33: one past best_len,
            // the eight-byte word would end at 42, beyond the 40 held. With
            // no mask, a run reaching before position zero cannot be read.
            for (site, remembered) in [
                (wrapped, [6, 30, 1, 2]),
                (truncated, [4, 30, 1, 2]),
                (unmasked, [1, 30, 1, 2]),
            ] {
                // No byte equals the continuation, so only forced lanes pass.
                let found =
                    dispatch!(level, simd => run_filter(simd, site, &remembered, 1, 9, true));
                assert_eq!(found.agree & run_codes, run_codes);
            }
        }
    }
}

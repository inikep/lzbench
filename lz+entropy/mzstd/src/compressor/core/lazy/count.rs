//! `ZSTD_count` for the lazy match finders: the length of the common run of
//! two byte windows, using the shared word and SIMD scan.
//!
//! The reference compares machine words with an XOR and a trailing-zero
//! count. Most candidates the lazy finders measure differ within the first
//! word, so that word is settled inline; only a match that runs through it
//! continues into the word and vector loops, which are where the long matches
//! of repetitive input spend their time. Every path returns the exact common
//! length, so the backend never changes a frame.

use fearless_simd::Simd;

/// Common length of `src[a..limit]` and `src[b..]` for `b < a`
/// (`ZSTD_count(src + a, src + b, src + limit)`).
///
/// # Panics
///
/// When `a > limit` or `limit > src.len()`: the reference's pointers would
/// leave the window, which its index invariants rule out.
#[inline(always)]
pub(super) fn count<S: Simd>(simd: S, src: &[u8], a: usize, b: usize, limit: usize) -> usize {
    let current = &src[a..limit];
    let Some(earlier) = src.get(b..b + current.len()) else {
        return common_bytes(&src[b..], current);
    };
    common_run(simd, current, earlier)
}

/// Common length of two windows, at most the shorter one's length.
#[inline(always)]
pub(super) fn common_run<S: Simd>(simd: S, x: &[u8], y: &[u8]) -> usize {
    super::super::mem::count_slices_simd(simd, x, y)
}

/// Common length of two windows compared one byte at a time.
#[inline(always)]
fn common_bytes(x: &[u8], y: &[u8]) -> usize {
    x.iter().zip(y).take_while(|(a, b)| a == b).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backend;
    use alloc::vec::Vec;

    fn reference(src: &[u8], a: usize, b: usize, limit: usize) -> usize {
        let mut n = 0;
        while a + n < limit && src[a + n] == src[b + n] {
            n += 1;
        }
        n
    }

    #[test]
    fn counts_equal_the_bytewise_reference_on_every_backend() {
        // A run long enough for every vector width, then a difference at
        // each position and limits at each boundary.
        let mut src: Vec<u8> = (0..400u32).map(|i| (i % 7) as u8).collect();
        src.extend((0..400u32).map(|i| (i % 7) as u8));
        for backend in Backend::available() {
            for diff in [
                0usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 130, 399,
            ] {
                let mut data = src.clone();
                data[400 + diff] ^= 0x80;
                for limit in [400, 401, 407, 408, 409, 424, 433, 464, 528, 799, 800] {
                    let expected = reference(&data, 400, 0, limit);
                    let got = fearless_simd::dispatch!(backend.0, simd => count(simd, &data, 400, 0, limit));
                    assert_eq!(got, expected, "{backend} diff {diff} limit {limit}");
                }
            }
        }
    }

    #[test]
    fn a_short_earlier_window_stops_the_count() {
        let src = [5u8; 20];
        for backend in Backend::available() {
            // `b + len` past the end can only happen when `b > a`.
            let got = fearless_simd::dispatch!(backend.0, simd => count(simd, &src, 2, 10, 20));
            assert_eq!(got, 10, "{backend}");
            let got =
                fearless_simd::dispatch!(backend.0, simd => common_run(simd, &src[..3], &src));
            assert_eq!(got, 3, "{backend}");
        }
    }
}

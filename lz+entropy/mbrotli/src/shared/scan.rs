//! Mask scans over histograms and code-length arrays.
//!
//! Prefix-code construction walks symbol tables of up to 704 entries that are
//! mostly zero for small inputs (absent symbols) or made of long runs of one
//! value (code lengths). Each scan here compares sixteen entries at once at
//! the target's baseline SIMD level ([`super::baseline`]) and walks the
//! resulting bit mask, so a stretch of zeros or of one repeated value costs a
//! compare per sixteen entries instead of a branch per entry. Every scan
//! reports exactly what the plain loop would, in the same order.

use fearless_simd::{Simd, SimdBase, SimdMask, u8x16, u32x16};

use super::baseline::{self, Kernel};

/// Entries compared at once.
const LANES: usize = 16;

/// A mask with every lane set.
const FULL: u32 = (1 << LANES) - 1;

/// Bit `b` is set when `chunk[b] == value`.
#[inline(always)]
fn equal_u32<S: Simd>(simd: S, chunk: &[u32; LANES], value: u32) -> u32 {
    u32x16::load_array_ref(simd, chunk)
        .simd_eq(u32x16::splat(simd, value))
        .to_bitmask() as u32
}

/// Bit `b` is set when `chunk[b] == value`.
#[inline(always)]
fn equal_u8<S: Simd>(simd: S, chunk: &[u8; LANES], value: u8) -> u32 {
    u8x16::load_array_ref(simd, chunk)
        .simd_eq(u8x16::splat(simd, value))
        .to_bitmask() as u32
}

/// Calls `f(i, data[i])` for every non-zero entry, in ascending order.
#[inline(always)]
pub(crate) fn for_each_nonzero_u32(data: &[u32], f: impl FnMut(usize, u32)) {
    baseline::run(NonzeroU32 { data, f });
}

/// Calls `f(i, data[i])` for every non-zero entry, in descending order.
#[inline(always)]
pub(crate) fn for_each_nonzero_u32_rev(data: &[u32], f: impl FnMut(usize, u32)) {
    baseline::run(NonzeroU32Rev { data, f });
}

/// Calls `f(i, data[i])` for every non-zero entry, in ascending order.
#[inline(always)]
pub(crate) fn for_each_nonzero_u8(data: &[u8], f: impl FnMut(usize, u8)) {
    baseline::run(NonzeroU8 { data, f });
}

/// Returns the first index after `start` whose entry differs from
/// `data[start]`, or `data.len()`; `start` itself when it is out of range.
#[inline(always)]
pub(crate) fn run_end_u8(data: &[u8], start: usize) -> usize {
    baseline::run(RunEndU8 { data, start })
}

/// Returns the first index after `start` whose entry differs from
/// `data[start]`, or `data.len()`; `start` itself when it is out of range.
#[inline(always)]
pub(crate) fn run_end_u32(data: &[u32], start: usize) -> usize {
    baseline::run(RunEndU32 { data, start })
}

/// Returns the length of `data` without its trailing zeros.
#[inline(always)]
pub(crate) fn trim_zeros_u8(data: &[u8]) -> usize {
    baseline::run(TrimZerosU8 { data })
}

/// [`for_each_nonzero_u32`] as a kernel.
struct NonzeroU32<'a, F> {
    data: &'a [u32],
    f: F,
}

impl<F: FnMut(usize, u32)> Kernel for NonzeroU32<'_, F> {
    type Output = ();

    #[inline(always)]
    fn run<S: Simd>(mut self, simd: S) {
        let (chunks, tail) = self.data.as_chunks::<LANES>();
        for (c, chunk) in chunks.iter().enumerate() {
            let base = c * LANES;
            let mut mask = !equal_u32(simd, chunk, 0) & FULL;
            if mask == FULL {
                // A dense histogram fills whole chunks; the straight loop
                // beats walking sixteen bits there.
                for (b, &value) in chunk.iter().enumerate() {
                    (self.f)(base + b, value);
                }
                continue;
            }
            while mask != 0 {
                let b = mask.trailing_zeros() as usize;
                (self.f)(base + b, chunk[b]);
                mask &= mask - 1;
            }
        }
        let base = chunks.len() * LANES;
        for (b, &value) in tail.iter().enumerate() {
            if value != 0 {
                (self.f)(base + b, value);
            }
        }
    }
}

/// [`for_each_nonzero_u32_rev`] as a kernel.
struct NonzeroU32Rev<'a, F> {
    data: &'a [u32],
    f: F,
}

impl<F: FnMut(usize, u32)> Kernel for NonzeroU32Rev<'_, F> {
    type Output = ();

    #[inline(always)]
    fn run<S: Simd>(mut self, simd: S) {
        let (chunks, tail) = self.data.as_chunks::<LANES>();
        let base = chunks.len() * LANES;
        for (b, &value) in tail.iter().enumerate().rev() {
            if value != 0 {
                (self.f)(base + b, value);
            }
        }
        for (c, chunk) in chunks.iter().enumerate().rev() {
            let base = c * LANES;
            let mut mask = !equal_u32(simd, chunk, 0) & FULL;
            if mask == FULL {
                for (b, &value) in chunk.iter().enumerate().rev() {
                    (self.f)(base + b, value);
                }
                continue;
            }
            while mask != 0 {
                let b = 31 - mask.leading_zeros() as usize;
                (self.f)(base + b, chunk[b]);
                mask &= !(1 << b);
            }
        }
    }
}

/// [`for_each_nonzero_u8`] as a kernel.
struct NonzeroU8<'a, F> {
    data: &'a [u8],
    f: F,
}

impl<F: FnMut(usize, u8)> Kernel for NonzeroU8<'_, F> {
    type Output = ();

    #[inline(always)]
    fn run<S: Simd>(mut self, simd: S) {
        let (chunks, tail) = self.data.as_chunks::<LANES>();
        for (c, chunk) in chunks.iter().enumerate() {
            let base = c * LANES;
            let mut mask = !equal_u8(simd, chunk, 0) & FULL;
            while mask != 0 {
                let b = mask.trailing_zeros() as usize;
                (self.f)(base + b, chunk[b]);
                mask &= mask - 1;
            }
        }
        let base = chunks.len() * LANES;
        for (b, &value) in tail.iter().enumerate() {
            if value != 0 {
                (self.f)(base + b, value);
            }
        }
    }
}

/// [`run_end_u8`] as a kernel.
struct RunEndU8<'a> {
    data: &'a [u8],
    start: usize,
}

impl Kernel for RunEndU8<'_> {
    type Output = usize;

    #[inline(always)]
    fn run<S: Simd>(self, simd: S) -> usize {
        let data = self.data;
        let Some(&value) = data.get(self.start) else {
            return self.start;
        };
        let mut k = self.start + 1;
        while let Some(chunk) = data.get(k..).and_then(<[u8]>::first_chunk::<LANES>) {
            let differ = !equal_u8(simd, chunk, value) & FULL;
            if differ != 0 {
                return k + differ.trailing_zeros() as usize;
            }
            k += LANES;
        }
        while k < data.len() && data[k] == value {
            k += 1;
        }
        k
    }
}

/// [`run_end_u32`] as a kernel.
struct RunEndU32<'a> {
    data: &'a [u32],
    start: usize,
}

impl Kernel for RunEndU32<'_> {
    type Output = usize;

    #[inline(always)]
    fn run<S: Simd>(self, simd: S) -> usize {
        let data = self.data;
        let Some(&value) = data.get(self.start) else {
            return self.start;
        };
        let mut k = self.start + 1;
        while let Some(chunk) = data.get(k..).and_then(<[u32]>::first_chunk::<LANES>) {
            let differ = !equal_u32(simd, chunk, value) & FULL;
            if differ != 0 {
                return k + differ.trailing_zeros() as usize;
            }
            k += LANES;
        }
        while k < data.len() && data[k] == value {
            k += 1;
        }
        k
    }
}

/// [`trim_zeros_u8`] as a kernel.
struct TrimZerosU8<'a> {
    data: &'a [u8],
}

impl Kernel for TrimZerosU8<'_> {
    type Output = usize;

    #[inline(always)]
    fn run<S: Simd>(self, simd: S) -> usize {
        let data = self.data;
        let mut end = data.len();
        while let Some(chunk) = end
            .checked_sub(LANES)
            .and_then(|start| data.get(start..end))
            .and_then(|chunk| chunk.first_chunk::<LANES>())
        {
            let nonzero = !equal_u8(simd, chunk, 0) & FULL;
            if nonzero != 0 {
                return end - LANES + (32 - nonzero.leading_zeros() as usize);
            }
            end -= LANES;
        }
        while end != 0 && data[end - 1] == 0 {
            end -= 1;
        }
        end
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use fearless_simd::{Level, dispatch};

    use super::*;

    /// Every level the host runs, and the scalar fallback.
    fn levels() -> Vec<Level> {
        crate::backend::Backend::levels()
    }

    /// Arrays with sparse entries, full chunks, long runs and ragged tails.
    fn shapes() -> Vec<Vec<u32>> {
        let mut state = 0x2545_F491u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut shapes = Vec::new();
        // Miri keeps the chunk and tail boundaries but not the long arrays.
        let lens: &[usize] = if cfg!(miri) {
            &[0, 1, 15, 16, 17, 31, 32, 33, 64, 70]
        } else {
            &[0, 1, 15, 16, 17, 31, 32, 33, 64, 70, 256, 704]
        };
        for &len in lens {
            shapes.push((0..len).map(|_| u32::from(next() % 9 == 0) * 3).collect());
            shapes.push((0..len).map(|_| next() % 5 + 1).collect());
            shapes.push((0..len).map(|i| (i / 23 % 3) as u32).collect());
            shapes.push((0..len).map(|i| u32::from(i + 1 == len)).collect());
            shapes.push(alloc::vec![0; len]);
        }
        shapes
    }

    fn nonzero(data: &[u32]) -> Vec<(usize, u32)> {
        (0..data.len())
            .filter(|&i| data[i] != 0)
            .map(|i| (i, data[i]))
            .collect()
    }

    fn run_end<T: PartialEq>(data: &[T], start: usize) -> usize {
        let Some(value) = data.get(start) else {
            return start;
        };
        (start + 1..data.len())
            .find(|&k| data[k] != *value)
            .unwrap_or(data.len())
    }

    #[test]
    fn nonzero_walks_match_the_plain_loop_on_every_level() {
        for data in shapes() {
            let bytes: Vec<u8> = data.iter().map(|&v| v as u8).collect();
            let expected = nonzero(&data);
            let reversed: Vec<_> = expected.iter().rev().copied().collect();
            for level in levels() {
                let mut seen = Vec::new();
                dispatch!(level, simd => NonzeroU32 { data: &data, f: |i, v| seen.push((i, v)) }.run(simd));
                assert_eq!(seen, expected, "{level:?}");
                let mut seen = Vec::new();
                dispatch!(level, simd => NonzeroU32Rev { data: &data, f: |i, v| seen.push((i, v)) }.run(simd));
                assert_eq!(seen, reversed, "{level:?}");
                let mut seen = Vec::new();
                dispatch!(level, simd => NonzeroU8 { data: &bytes, f: |i, v| seen.push((i, u32::from(v))) }.run(simd));
                assert_eq!(seen, expected, "{level:?}");
            }
        }
    }

    #[test]
    fn run_ends_and_trims_match_the_plain_loop_on_every_level() {
        for data in shapes() {
            let bytes: Vec<u8> = data.iter().map(|&v| v as u8).collect();
            let trimmed = bytes.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
            for level in levels() {
                for start in 0..=data.len() {
                    let end = dispatch!(level, simd => RunEndU32 { data: &data, start }.run(simd));
                    assert_eq!(end, run_end(&data, start), "{level:?} at {start}");
                    let end = dispatch!(level, simd => RunEndU8 { data: &bytes, start }.run(simd));
                    assert_eq!(end, run_end(&bytes, start), "{level:?} at {start}");
                }
                let end = dispatch!(level, simd => TrimZerosU8 { data: &bytes }.run(simd));
                assert_eq!(end, trimmed, "{level:?}");
            }
        }
    }

    #[test]
    fn the_baseline_entry_points_agree_with_the_kernels() {
        let data = [0u32, 4, 0, 0, 9, 9, 9, 0];
        let mut seen = Vec::new();
        for_each_nonzero_u32(&data, |i, v| seen.push((i, v)));
        assert_eq!(seen, [(1, 4), (4, 9), (5, 9), (6, 9)]);
        seen.clear();
        for_each_nonzero_u32_rev(&data, |i, v| seen.push((i, v)));
        assert_eq!(seen, [(6, 9), (5, 9), (4, 9), (1, 4)]);
        let bytes = [0u8, 2, 2, 0, 0];
        let mut seen = Vec::new();
        for_each_nonzero_u8(&bytes, |i, v| seen.push((i, v)));
        assert_eq!(seen, [(1, 2), (2, 2)]);
        assert_eq!(run_end_u32(&data, 4), 7);
        assert_eq!(run_end_u8(&bytes, 1), 3);
        assert_eq!(trim_zeros_u8(&bytes), 3);
    }
}

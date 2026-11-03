//! The SIMD level a target guarantees, for code outside dispatched entries.
//!
//! Prefix-code construction, histogram shaping and the match finders' cache
//! hints run in ordinary functions, not inside the encoders' per-level
//! `#[target_feature]` entries. Handing them the encoder's detected level
//! would turn every vector operation into an out-of-line call, so they use
//! the level every build of the target already has: SSE2 on x86 and x86-64,
//! NEON on AArch64, scalar lanes elsewhere. Selecting it needs no detection;
//! `Level::baseline` is a constant, and the selection folds away.

use fearless_simd::{Level, Simd};

/// A computation that runs at any SIMD level.
pub(crate) trait Kernel {
    /// What the computation returns.
    type Output;

    /// Runs the computation with `simd`.
    fn run<S: Simd>(self, simd: S) -> Self::Output;
}

/// Runs `kernel` at the target's baseline level.
#[inline(always)]
pub(crate) fn run<K: Kernel>(kernel: K) -> K::Output {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(simd) = Level::baseline().as_sse2() {
        return kernel.run(simd);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(simd) = Level::baseline().as_neon() {
        return kernel.run(simd);
    }
    // Other targets: whichever level the build guarantees, scalar lanes at
    // worst. Unreachable on the targets above, whose baseline always has
    // SSE2 or NEON.
    fearless_simd::dispatch!(Level::baseline(), simd => kernel.run(simd))
}

/// Asks for the cache line holding `value` (the reference's `PREFETCH_L1`).
///
/// A read hint into L1, as in the reference: the bucket it targets is read by
/// the next search before the store writes it, and a line the encoder's
/// thread reads arrives exclusive, so the later write needs no second
/// request. A hint only: it never faults and never changes a result.
/// `branches` emits it on x86-64, AArch64 and x86 with SSE, and on the other
/// targets it lists (riscv64 with `zicbop`, s390x, PowerPC, any target on
/// nightly); elsewhere it does nothing.
#[inline(always)]
pub(crate) fn prefetch<T>(value: &T) {
    branches::prefetch_read_data::<T, 0>(value);
}

#[cfg(test)]
mod tests {
    use fearless_simd::SimdBase;

    use super::*;

    struct LaneCount;

    impl Kernel for LaneCount {
        type Output = usize;

        fn run<S: Simd>(self, simd: S) -> usize {
            fearless_simd::u8x16::splat(simd, 1)
                .as_slice()
                .iter()
                .map(|&lane| usize::from(lane))
                .sum()
        }
    }

    #[test]
    fn a_kernel_runs_at_the_baseline_level() {
        assert_eq!(run(LaneCount), 16);
    }

    #[test]
    fn prefetching_leaves_the_value_untouched() {
        let values = [7u32; 32];
        prefetch(&values[0]);
        prefetch(&values[31]);
        assert_eq!(values, [7; 32]);
    }
}

//! Logarithms with the exact rounding the reference encoder relies on.
//!
//! `FastLog2` (`c/enc/fast_log.h`) reads a table of single-precision
//! logarithms for small arguments and falls back to `log2` above it. Cost
//! comparisons in the block splitter and the histogram builders are decided on
//! these values, so the rounding is part of the bitstream contract rather than
//! an implementation detail.

use super::tables::LOG2_TABLE;

/// Returns `floor(log2(value))` for a non-zero `value`.
#[inline(always)]
pub(crate) const fn log2_floor_non_zero(value: usize) -> u32 {
    (usize::BITS - 1) - value.leading_zeros()
}

/// Reference logarithm with `log2(0) == 0` (`FastLog2`).
///
/// The table covers every count an entropy sum meets in practice; the
/// library call for larger values is kept out of line and marked cold so
/// the loops summing entropies keep their accumulators in registers rather
/// than spilling them around a call they almost never make.
#[inline]
pub(crate) fn fast_log2(value: usize) -> f64 {
    match LOG2_TABLE.get(value) {
        Some(&entry) => entry,
        None => log2_beyond_table(value),
    }
}

/// The logarithm of a value past the table's end.
#[cold]
#[inline(never)]
fn log2_beyond_table(value: usize) -> f64 {
    #[cfg(not(feature = "no_std"))]
    {
        (value as f64).log2()
    }
    #[cfg(feature = "no_std")]
    {
        libm::log2(value as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "no_std")]
    #[test]
    fn portable_logarithms_match_the_reference_within_one_ulp() {
        for value in [257, 1023, 65535, 1 << 24, usize::MAX] {
            let expected = (value as f64).log2();
            assert!(fast_log2(value).to_bits().abs_diff(expected.to_bits()) <= 1);
        }
    }

    #[test]
    fn floor_log2_matches_the_bit_width() {
        assert_eq!(log2_floor_non_zero(1), 0);
        assert_eq!(log2_floor_non_zero(2), 1);
        assert_eq!(log2_floor_non_zero(3), 1);
        assert_eq!(log2_floor_non_zero(255), 7);
        assert_eq!(log2_floor_non_zero(256), 8);
    }

    #[test]
    fn fast_log2_reads_the_table_below_it_and_computes_above() {
        assert_eq!(fast_log2(0), 0.0);
        assert_eq!(fast_log2(1), 0.0);
        assert_eq!(fast_log2(2), 1.0);
        assert_eq!(fast_log2(255), LOG2_TABLE[255]);
        assert_eq!(fast_log2(256), 8.0);
        assert_eq!(fast_log2(1024), 10.0);
    }
}

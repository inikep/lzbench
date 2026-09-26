//! Symbol histograms and the entropy estimates the splitters decide on.
//!
//! Ports `c/enc/histogram.h`, `BrotliBitsEntropy` from `c/enc/bit_cost.c` and
//! `BrotliOptimizeHuffmanCountsForRle` from `c/enc/entropy_encode.c` of the
//! pinned reference (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! The estimates are floating point in the reference, and the block splitter
//! compares them directly, so the arithmetic has to stay bit-for-bit the same:
//! the same `FastLog2` table, the same accumulation order, the same "at least
//! one bit per symbol" floor.

use super::constants::{NUM_COMMAND_SYMBOLS, NUM_LITERAL_SYMBOLS};
use super::distance::NUM_HISTOGRAM_DISTANCE_SYMBOLS;
use super::fast_log::fast_log2;

/// Occurrence counts over an alphabet of `N` symbols.
#[derive(Clone, Debug)]
pub(crate) struct Histogram<const N: usize> {
    /// Count per symbol.
    pub(crate) data: [u32; N],
    /// Sum of [`Histogram::data`].
    pub(crate) total_count: usize,
    /// What storing this histogram was last estimated to cost, in bits.
    ///
    /// Only the high-quality clusterer maintains this; every other caller
    /// leaves it at infinity. It lives here rather than beside the clusterer
    /// because the reference caches it on the histogram, and a merge that
    /// reused a stale value would pick a different cluster.
    pub(crate) bit_cost: f64,
}

impl<const N: usize> Histogram<N> {
    /// Forgets every symbol counted so far (`HistogramClear`).
    pub(crate) fn clear(&mut self) {
        self.data.fill(0);
        self.total_count = 0;
        self.bit_cost = f64::INFINITY;
    }

    /// Counts one occurrence of `symbol` (`HistogramAdd`).
    #[inline(always)]
    pub(crate) fn add(&mut self, symbol: usize) {
        if let Some(count) = self.data.get_mut(symbol) {
            *count += 1;
            self.total_count += 1;
        }
    }

    /// Adds every count of `other` (`HistogramAddHistogram`).
    ///
    /// The cached bit cost is deliberately left alone: the reference recomputes
    /// it at the point of use rather than after every merge.
    pub(crate) fn add_histogram(&mut self, other: &Self) {
        self.total_count += other.total_count;
        for (mine, &theirs) in self.data.iter_mut().zip(other.data.iter()) {
            *mine += theirs;
        }
    }

    /// Counts every symbol of `values` (`HistogramAddVector`).
    pub(crate) fn add_vector<T: Copy + Into<usize>>(&mut self, values: &[T]) {
        self.total_count += values.len();
        for &value in values {
            if let Some(count) = self.data.get_mut(value.into()) {
                *count += 1;
            }
        }
    }
}

impl<const N: usize> Default for Histogram<N> {
    /// Returns an empty histogram.
    fn default() -> Self {
        Self {
            data: [0u32; N],
            total_count: 0,
            bit_cost: f64::INFINITY,
        }
    }
}

/// Histogram over the literal alphabet.
pub(crate) type HistogramLiteral = Histogram<NUM_LITERAL_SYMBOLS>;

/// Histogram over the insert-and-copy command alphabet.
pub(crate) type HistogramCommand = Histogram<NUM_COMMAND_SYMBOLS>;

/// Histogram over the distance alphabet.
pub(crate) type HistogramDistance = Histogram<NUM_HISTOGRAM_DISTANCE_SYMBOLS>;

/// Returns the Shannon entropy of `population`, floored at one bit per symbol.
///
/// Mirrors `BrotliBitsEntropy`, including its two-at-a-time accumulation: the
/// order the terms are summed in changes the last bits of the result, and the
/// block splitter compares those results against thresholds.
pub(crate) fn bits_entropy(population: &[u32]) -> f64 {
    bits_entropy_of(population.iter().map(|&count| count as usize))
}

/// The entropy of two populations summed, without materialising the sum.
///
/// Same accumulation order as [`bits_entropy`] over the summed counts, so it
/// returns exactly what `bits_entropy` would on the combined histogram.
pub(crate) fn bits_entropy_of_sum(left: &[u32], right: &[u32]) -> f64 {
    bits_entropy_of(
        left.iter()
            .zip(right)
            .map(|(&left, &right)| left as usize + right as usize),
    )
}

/// The shared accumulation of [`bits_entropy`] over `counts`.
///
/// The reference unrolls its loop by two, jumping into the middle for an odd
/// length; the subtractions still happen one count after another in order,
/// so a plain loop over the same counts produces the same double.
#[inline(always)]
fn bits_entropy_of(counts: impl Iterator<Item = usize>) -> f64 {
    let mut sum = 0usize;
    let mut retval = 0f64;
    for count in counts {
        sum += count;
        retval -= count as f64 * fast_log2(count);
    }
    if sum != 0 {
        retval += sum as f64 * fast_log2(sum);
    }
    if retval < sum as f64 {
        // A prefix code cannot spend less than one bit on a symbol.
        retval = sum as f64;
    }
    retval
}

/// Number of equal counts a run needs before it is worth a run-length code.
const STREAK_LIMIT: usize = 1240;

/// Rounds counts so the prefix code they produce is friendlier to run-length
/// coding.
///
/// Mirrors `BrotliOptimizeHuffmanCountsForRle`. It rewrites the histogram in
/// place: neighbouring counts that would produce nearly the same code length
/// are collapsed onto one value, which makes the code-length sequence
/// compressible even though it costs a fraction of a bit per symbol.
///
/// `good_for_rle` is scratch space of at least `length` bytes.
pub(crate) fn optimize_huffman_counts_for_rle(
    length: usize,
    counts: &mut [u32],
    good_for_rle: &mut [u8],
) {
    let mut nonzero_count = 0usize;
    for &count in counts.iter().take(length) {
        if count != 0 {
            nonzero_count += 1;
        }
    }
    if nonzero_count < 16 {
        return;
    }
    let mut length = length;
    while length != 0 && counts[length - 1] == 0 {
        length -= 1;
    }
    if length == 0 {
        return;
    }

    {
        let mut nonzeros = 0usize;
        let mut smallest_nonzero = 1u32 << 30;
        for &count in counts.iter().take(length) {
            if count != 0 {
                nonzeros += 1;
                smallest_nonzero = smallest_nonzero.min(count);
            }
        }
        if nonzeros < 5 {
            // A small histogram is modelled well as it is.
            return;
        }
        if smallest_nonzero < 4 {
            let zeros = length - nonzeros;
            if zeros < 6 {
                for i in 1..length - 1 {
                    if counts[i - 1] != 0 && counts[i] == 0 && counts[i + 1] != 0 {
                        counts[i] = 1;
                    }
                }
            }
        }
        if nonzeros < 28 {
            return;
        }
    }

    // Mark the runs that already code well, so they are not broken up.
    good_for_rle[..length].fill(0);
    {
        let mut symbol = counts[0];
        let mut step = 0usize;
        for i in 0..=length {
            if i == length || counts[i] != symbol {
                if (symbol == 0 && step >= 5) || (symbol != 0 && step >= 7) {
                    for k in 0..step {
                        good_for_rle[i - k - 1] = 1;
                    }
                }
                step = 1;
                if i != length {
                    symbol = counts[i];
                }
            } else {
                step += 1;
            }
        }
    }

    // Collapse the remaining runs. The arithmetic is 24.8 fixed point.
    let mut stride = 0usize;
    let mut limit = 256 * (counts[0] as usize + counts[1] as usize + counts[2] as usize) / 3 + 420;
    let mut sum = 0usize;
    for i in 0..=length {
        let count_i = counts.get(i).copied().unwrap_or(0) as usize;
        if i == length
            || good_for_rle[i] != 0
            || (i != 0 && good_for_rle[i - 1] != 0)
            || (256 * count_i)
                .wrapping_sub(limit)
                .wrapping_add(STREAK_LIMIT)
                >= 2 * STREAK_LIMIT
        {
            if stride >= 4 || (stride >= 3 && sum == 0) {
                let mut count = (sum + stride / 2) / stride;
                if count == 0 {
                    count = 1;
                }
                if sum == 0 {
                    // A run of zeros must not be promoted to ones.
                    count = 0;
                }
                for k in 0..stride {
                    counts[i - k - 1] = count as u32;
                }
            }
            stride = 0;
            sum = 0;
            limit = if i + 2 < length {
                256 * (counts[i] as usize + counts[i + 1] as usize + counts[i + 2] as usize) / 3
                    + 420
            } else if i < length {
                256 * counts[i] as usize
            } else {
                0
            };
        }
        stride += 1;
        if i != length {
            sum += count_i;
            if stride >= 4 {
                limit = (256 * sum + stride / 2) / stride;
            }
            if stride == 4 {
                limit += 120;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_histogram_counts_and_clears() {
        let mut histogram = HistogramLiteral::default();
        assert_eq!(histogram.total_count, 0);
        histogram.add(7);
        histogram.add(7);
        histogram.add(9);
        assert_eq!(histogram.data[7], 2);
        assert_eq!(histogram.data[9], 1);
        assert_eq!(histogram.total_count, 3);
        histogram.clear();
        assert_eq!(histogram.total_count, 0);
        assert!(histogram.data.iter().all(|&count| count == 0));
    }

    #[test]
    fn histograms_add_elementwise() {
        let mut left = HistogramCommand::default();
        let mut right = HistogramCommand::default();
        left.add(1);
        right.add(1);
        right.add(700);
        left.add_histogram(&right);
        assert_eq!(left.data[1], 2);
        assert_eq!(left.data[700], 1);
        assert_eq!(left.total_count, 3);
    }

    #[test]
    fn a_symbol_outside_the_alphabet_is_ignored() {
        let mut histogram = HistogramLiteral::default();
        histogram.add(256);
        assert_eq!(histogram.total_count, 0);
    }

    #[test]
    fn entropy_of_a_uniform_histogram_is_its_log() {
        let population = [4u32; 16];
        // Sixty-four symbols over sixteen equally likely values: four bits
        // each.
        assert!((bits_entropy(&population) - 256.0).abs() < 1e-9);
    }

    #[test]
    fn entropy_never_falls_below_one_bit_per_symbol() {
        let population = [100u32, 0, 0, 0];
        assert_eq!(bits_entropy(&population), 100.0);
        assert_eq!(bits_entropy(&[0u32; 8]), 0.0);
    }

    #[test]
    fn entropy_handles_odd_and_even_lengths() {
        let odd = [1u32, 1, 1];
        let even = [1u32, 1, 1, 0];
        assert_eq!(bits_entropy(&odd), bits_entropy(&even));
    }

    #[test]
    fn rle_optimisation_leaves_small_histograms_alone() {
        let mut counts = [0u32; 256];
        for (index, count) in counts.iter_mut().enumerate().take(10) {
            *count = index as u32 + 1;
        }
        let original = counts;
        let mut scratch = [0u8; 256];
        optimize_huffman_counts_for_rle(256, &mut counts, &mut scratch);
        assert_eq!(counts, original);
    }

    #[test]
    fn rle_optimisation_collapses_a_long_flat_run() {
        let mut counts = [0u32; 256];
        for (index, count) in counts.iter_mut().enumerate() {
            *count = 100 + (index as u32 % 3);
        }
        let mut scratch = [0u8; 256];
        optimize_huffman_counts_for_rle(256, &mut counts, &mut scratch);
        // The neighbouring counts are close enough to be merged onto one
        // value, which is what makes the code lengths run-length codable.
        let distinct: std::collections::BTreeSet<u32> = counts.iter().copied().collect();
        assert!(distinct.len() < 3, "counts were left as {distinct:?}");
        assert!(counts.iter().all(|&count| count > 0));
    }

    #[test]
    fn rle_optimisation_fills_single_zero_gaps() {
        // Two isolated zeros between small counts: cheaper to spend a bit on
        // them than to break the run of equal code lengths.
        let mut counts = [1u32; 40];
        counts[5] = 0;
        counts[15] = 0;
        let mut scratch = [0u8; 40];
        optimize_huffman_counts_for_rle(40, &mut counts, &mut scratch);
        assert!(counts.iter().all(|&count| count > 0));
    }

    #[test]
    fn rle_optimisation_returns_early_for_a_sparse_histogram() {
        // Sixteen to twenty-seven used symbols: enough to look at, not enough
        // to be worth reshaping.
        let mut counts = [0u32; 256];
        for (index, count) in counts.iter_mut().enumerate().take(20) {
            *count = 10 * (index as u32 + 1);
        }
        let original = counts;
        let mut scratch = [0u8; 256];
        optimize_huffman_counts_for_rle(256, &mut counts, &mut scratch);
        assert_eq!(counts, original);
    }

    #[test]
    fn rle_optimisation_preserves_a_long_zero_run() {
        // A zero run of five or more already codes well, so it is marked and
        // left alone rather than being merged into its neighbours.
        let mut counts = [0u32; 256];
        for count in counts.iter_mut().take(30) {
            *count = 100;
        }
        for count in counts.iter_mut().take(70).skip(40) {
            *count = 100;
        }
        let mut scratch = [0u8; 256];
        optimize_huffman_counts_for_rle(256, &mut counts, &mut scratch);
        assert!(
            counts[30..40].iter().all(|&count| count == 0),
            "the zero run was filled in: {:?}",
            &counts[25..45]
        );
    }

    #[test]
    fn rle_optimisation_keeps_an_all_zero_histogram_empty() {
        let mut counts = [0u32; 256];
        let mut scratch = [0u8; 256];
        optimize_huffman_counts_for_rle(256, &mut counts, &mut scratch);
        assert!(counts.iter().all(|&count| count == 0));
    }
}

//! Estimating what a histogram would cost to store as a prefix code.
//!
//! Ports `BrotliPopulationCost` from `c/enc/bit_cost_inc.h` of the pinned
//! reference (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! The high-quality block splitter and the histogram clusterer decide every
//! merge on this number, so it is not an estimate in the usual sense: it is
//! part of the format contract. The rounding, the accumulation order and the
//! four hand-written small-alphabet cases all show up in the emitted bytes.

use super::constants::{CODE_LENGTH_CODES, REPEAT_ZERO_CODE_LENGTH};
use super::fast_log::fast_log2;
use super::histogram::{Histogram, bits_entropy};

/// Cost of a histogram with a single used symbol (`kOneSymbolHistogramCost`).
const ONE_SYMBOL_HISTOGRAM_COST: f64 = 12.0;

/// Cost of a histogram with two used symbols (`kTwoSymbolHistogramCost`).
const TWO_SYMBOL_HISTOGRAM_COST: f64 = 20.0;

/// Cost of a histogram with three used symbols (`kThreeSymbolHistogramCost`).
const THREE_SYMBOL_HISTOGRAM_COST: f64 = 28.0;

/// Cost of a histogram with four used symbols (`kFourSymbolHistogramCost`).
const FOUR_SYMBOL_HISTOGRAM_COST: f64 = 37.0;

/// Symbols per block [`first_used_symbols`] tests for any use at once.
const USED_SCAN_BLOCK: usize = 16;

/// Returns the first five used symbols of `data` and how many were found.
///
/// Five is one more than any closed form of [`population_cost`] needs:
/// finding a fifth is what rules them all out. The histograms clustering
/// prices are often sparse, so whole blocks of unused symbols are skipped
/// with one combined test each instead of a branch per symbol.
#[inline(always)]
fn first_used_symbols(data: &[u32]) -> ([usize; 5], usize) {
    let mut used = [0usize; 5];
    let mut count = 0usize;
    let mut record = |symbol: usize| {
        if let Some(slot) = used.get_mut(count) {
            *slot = symbol;
        }
        count += 1;
        count > 4
    };
    let (blocks, rest) = data.as_chunks::<USED_SCAN_BLOCK>();
    for (block_index, block) in blocks.iter().enumerate() {
        if block.iter().fold(0, |any, &value| any | value) == 0 {
            continue;
        }
        for (offset, &value) in block.iter().enumerate() {
            if value > 0 && record(block_index * USED_SCAN_BLOCK + offset) {
                return (used, count);
            }
        }
    }
    let base = blocks.len() * USED_SCAN_BLOCK;
    for (offset, &value) in rest.iter().enumerate() {
        if value > 0 && record(base + offset) {
            break;
        }
    }
    (used, count)
}

/// Returns the estimated bits needed to store `histogram` and its symbols.
///
/// `data_size` is how much of the alphabet may actually occur, which for
/// distances is narrower than the histogram itself.
///
/// Mirrors `BrotliPopulationCost`. Histograms with at most four used symbols
/// are priced by closed form, because the reference stores them with a simple
/// prefix code whose cost it knows exactly; anything larger is priced as the
/// entropy of the symbols plus the entropy of the code-length code that would
/// describe the tree.
pub(crate) fn population_cost<const N: usize>(histogram: &Histogram<N>, data_size: usize) -> f64 {
    let data = match histogram.data.get(..data_size) {
        Some(data) => data,
        None => &histogram.data,
    };
    if histogram.total_count == 0 {
        return ONE_SYMBOL_HISTOGRAM_COST;
    }

    let (used, count) = first_used_symbols(data);

    match count {
        1 => return ONE_SYMBOL_HISTOGRAM_COST,
        2 => return TWO_SYMBOL_HISTOGRAM_COST + histogram.total_count as f64,
        3 => {
            let counts = [data[used[0]], data[used[1]], data[used[2]]];
            let sum = counts[0] + counts[1] + counts[2];
            let largest = counts[0].max(counts[1]).max(counts[2]);
            return THREE_SYMBOL_HISTOGRAM_COST + f64::from(2 * sum - largest);
        }
        4 => {
            let mut counts = [data[used[0]], data[used[1]], data[used[2]], data[used[3]]];
            // The reference sorts descending with a selection sort; four
            // elements make the order fully determined either way.
            counts.sort_unstable_by(|left, right| right.cmp(left));
            let tail = counts[2] + counts[3];
            let largest = tail.max(counts[0]);
            return FOUR_SYMBOL_HISTOGRAM_COST
                + f64::from(3 * tail + 2 * (counts[0] + counts[1]) - largest);
        }
        _ => {}
    }

    // Compute the entropy of the histogram, and at the same time build a
    // simplified histogram of the code-length codes: the zero repeat code 17 is
    // used, the non-zero repeat code 16 is not.
    let mut max_depth = 1usize;
    let mut depth_histo = [0u32; CODE_LENGTH_CODES];
    let log2total = fast_log2(histogram.total_count);
    let mut bits = 0f64;
    let mut index = 0usize;
    while index < data.len() {
        let value = data[index] as usize;
        if value > 0 {
            // `-log2(P(symbol))`, which the reference rounds to the nearest
            // integer to approximate the depth the tree would give it.
            let log2p = log2total - fast_log2(value);
            // `log2p` lies in `0..=64`, so narrowing through a byte loses
            // nothing and skips the range guards a `usize` cast carries.
            let depth = usize::from(((log2p + 0.5) as u8).min(15));
            bits += value as f64 * log2p;
            if depth > max_depth {
                max_depth = depth;
            }
            if let Some(slot) = depth_histo.get_mut(depth) {
                *slot += 1;
            }
            index += 1;
        } else {
            // Price the run of zeros as the 0 and 17 code-length codes that
            // would describe it.
            let mut reps = 1u32;
            let mut scan = index + 1;
            while scan < data.len() && data[scan] == 0 {
                reps += 1;
                scan += 1;
            }
            index += reps as usize;
            if index == data.len() {
                // A trailing zero run is implicit, so it costs nothing.
                break;
            }
            if reps < 3 {
                depth_histo[0] += reps;
            } else {
                reps -= 2;
                while reps > 0 {
                    depth_histo[REPEAT_ZERO_CODE_LENGTH] += 1;
                    // The three extra bits the 17 code carries.
                    bits += 3.0;
                    reps >>= 3;
                }
            }
        }
    }
    // The estimated cost of storing the code-length code histogram, then its
    // own entropy.
    bits += (18 + 2 * max_depth) as f64;
    bits += bits_entropy(&depth_histo);
    bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::constants::NUM_LITERAL_SYMBOLS;
    use crate::shared::histogram::HistogramLiteral;

    /// Builds a literal histogram from `(symbol, count)` pairs.
    fn histogram(entries: &[(usize, u32)]) -> HistogramLiteral {
        let mut histogram = HistogramLiteral::default();
        for &(symbol, count) in entries {
            for _ in 0..count {
                histogram.add(symbol);
            }
        }
        histogram
    }

    #[test]
    fn an_empty_histogram_costs_one_symbol() {
        let empty = HistogramLiteral::default();
        assert_eq!(
            population_cost(&empty, NUM_LITERAL_SYMBOLS),
            ONE_SYMBOL_HISTOGRAM_COST
        );
    }

    #[test]
    fn the_small_alphabet_cases_use_their_closed_forms() {
        assert_eq!(
            population_cost(&histogram(&[(7, 40)]), NUM_LITERAL_SYMBOLS),
            ONE_SYMBOL_HISTOGRAM_COST
        );
        assert_eq!(
            population_cost(&histogram(&[(7, 40), (9, 10)]), NUM_LITERAL_SYMBOLS),
            TWO_SYMBOL_HISTOGRAM_COST + 50.0
        );
        // Three symbols: 2 * total - max.
        assert_eq!(
            population_cost(&histogram(&[(1, 5), (2, 7), (3, 9)]), NUM_LITERAL_SYMBOLS),
            THREE_SYMBOL_HISTOGRAM_COST + (2.0 * 21.0 - 9.0)
        );
        // Four symbols: 3 * (two smallest) + 2 * (two largest) - max of the
        // two-smallest sum and the largest.
        let cost = population_cost(
            &histogram(&[(1, 1), (2, 2), (3, 3), (4, 10)]),
            NUM_LITERAL_SYMBOLS,
        );
        assert_eq!(cost, FOUR_SYMBOL_HISTOGRAM_COST + (9.0 + 26.0 - 10.0));
    }

    #[test]
    fn a_five_symbol_histogram_leaves_the_closed_forms() {
        let five = histogram(&[(1, 4), (2, 4), (3, 4), (4, 4), (5, 4)]);
        let cost = population_cost(&five, NUM_LITERAL_SYMBOLS);
        assert!(cost > FOUR_SYMBOL_HISTOGRAM_COST);
        // Twenty symbols over five equiprobable values is 20 * log2(5) bits of
        // payload, plus the tree description.
        assert!(cost > 20.0 * 5f64.log2());
    }

    #[test]
    fn a_flat_histogram_costs_about_its_entropy() {
        let mut flat = HistogramLiteral::default();
        for symbol in 0..256 {
            for _ in 0..16 {
                flat.add(symbol);
            }
        }
        let cost = population_cost(&flat, NUM_LITERAL_SYMBOLS);
        // 4096 symbols at eight bits each is the payload; everything above it
        // is the estimated cost of describing a tree whose two hundred and
        // fifty-six depths are all identical.
        let overhead = cost - 8.0 * 4096.0;
        assert!(overhead > 0.0, "overhead was {overhead}");
        assert!(overhead < 0.02 * cost, "overhead was {overhead}");
    }

    #[test]
    fn a_narrower_alphabet_ignores_the_symbols_beyond_it() {
        // Counts past `data_size` are invisible, which is what lets the
        // distance histograms be priced over the alphabet actually in use.
        let mut histogram = HistogramLiteral::default();
        for symbol in [1usize, 2, 3, 4, 5] {
            histogram.add(symbol);
        }
        let wide = population_cost(&histogram, NUM_LITERAL_SYMBOLS);
        let narrow = population_cost(&histogram, 3);
        assert!(narrow < wide);
        // Only symbols one and two remain, so the two-symbol form applies —
        // over the *whole* count, which the reference does not narrow.
        assert_eq!(narrow, TWO_SYMBOL_HISTOGRAM_COST + 5.0);
    }

    #[test]
    fn used_symbols_match_a_symbol_by_symbol_scan() {
        // Lengths around the block width exercise both the skipped blocks and
        // the unblocked tail; positions straddle block boundaries.
        let lengths: &[usize] = if cfg!(miri) {
            &[0, 15, 17, 40]
        } else {
            &[0, 1, 15, 16, 17, 31, 32, 33, 256, 544, 704]
        };
        for &length in lengths {
            for seed in 0..64usize {
                let mut data = vec![0u32; length];
                let used = seed % 7;
                for k in 0..used {
                    if length > 0 {
                        data[(seed * 37 + k * 13 + k * k * 5) % length] = (k + 1) as u32;
                    }
                }
                let mut expected = [0usize; 5];
                let mut count = 0usize;
                for (symbol, &value) in data.iter().enumerate() {
                    if value > 0 {
                        if count < 5 {
                            expected[count] = symbol;
                        }
                        count += 1;
                        if count > 4 {
                            break;
                        }
                    }
                }
                assert_eq!(
                    first_used_symbols(&data),
                    (expected, count),
                    "length {length}, seed {seed}"
                );
            }
        }
    }

    #[test]
    fn a_trailing_zero_run_is_free() {
        // Two histograms with the same used symbols but different alphabet
        // tails cost the same, because the reference stops pricing zeros once
        // nothing follows them.
        let entries = [(0usize, 3u32), (1, 5), (2, 7), (3, 9), (4, 11)];
        let short = population_cost(&histogram(&entries), 5);
        let long = population_cost(&histogram(&entries), NUM_LITERAL_SYMBOLS);
        assert_eq!(short, long);
    }
}

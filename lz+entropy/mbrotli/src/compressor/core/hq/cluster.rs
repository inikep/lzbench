//! Merging histograms that are similar enough to share a prefix code.
//!
//! Ports `c/enc/cluster.c` and `c/enc/cluster_inc.h` from the pinned reference
//! (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! A meta-block may gather hundreds of histograms — one per block type and
//! context — but the format allows at most two hundred and fifty-six, and each
//! one costs a whole prefix code to store. Clustering repeatedly merges the
//! pair whose combination saves the most bits, until no merge pays.
//!
//! The candidate pairs are kept in a bounded array with the best one first,
//! which the reference calls a heap and treats as one only at that first
//! position. Reproducing that — including which pair is displaced when the
//! array is full — is what keeps the merge order, and therefore the context
//! map, identical.

use alloc::vec::Vec;

use crate::shared::bit_cost::population_cost;
use crate::shared::fast_log::fast_log2;
use crate::shared::histogram::Histogram;

/// A candidate merge and what it would cost (`HistogramPair`).
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct HistogramPair {
    /// Lower index of the pair.
    idx1: u32,
    /// Higher index of the pair.
    idx2: u32,
    /// Cost of storing the two histograms combined.
    cost_combo: f64,
    /// Change in total cost from merging: negative means the merge pays.
    cost_diff: f64,
}

/// Returns whether `left` is the worse candidate of the two.
///
/// Mirrors `HistogramPairIsLess`: a larger cost difference is worse, and ties
/// are broken on the span between the indices, which keeps the order total.
fn pair_is_less(left: &HistogramPair, right: &HistogramPair) -> bool {
    if left.cost_diff != right.cost_diff {
        return left.cost_diff > right.cost_diff;
    }
    (left.idx2 - left.idx1) > (right.idx2 - right.idx1)
}

/// Returns how much the context map shrinks when two clusters merge.
///
/// Mirrors `ClusterCostDiff`: the entropy of the cluster-size distribution
/// before the merge, minus after.
fn cluster_cost_diff(size_a: usize, size_b: usize) -> f64 {
    let size_c = size_a + size_b;
    size_a as f64 * fast_log2(size_a) + size_b as f64 * fast_log2(size_b)
        - size_c as f64 * fast_log2(size_c)
}

/// Scratch storage the clusterer reuses across meta-blocks.
pub(crate) struct ClusterArena<const N: usize> {
    pairs: Vec<HistogramPair>,
    cluster_size: Vec<u32>,
    clusters: Vec<u32>,
    new_index: Vec<u32>,
    tmp: Histogram<N>,
    reindexed: Vec<Histogram<N>>,
}

impl<const N: usize> ClusterArena<N> {
    /// Counts all heap-backed scratch storage.
    pub(crate) fn retained_bytes(&self) -> usize {
        (self.pairs.capacity()) * size_of::<HistogramPair>()
            + (self.cluster_size.capacity() + self.clusters.capacity() + self.new_index.capacity())
                * size_of::<u32>()
            + (self.reindexed.capacity()) * size_of::<Histogram<N>>()
    }
}

impl<const N: usize> Default for ClusterArena<N> {
    /// Returns empty scratch storage.
    fn default() -> Self {
        Self {
            pairs: Vec::new(),
            cluster_size: Vec::new(),
            clusters: Vec::new(),
            new_index: Vec::new(),
            tmp: Histogram::default(),
            reindexed: Vec::new(),
        }
    }
}

/// Considers merging `idx1` with `idx2`, queueing it if it looks worthwhile.
///
/// Mirrors `BrotliCompareAndPushToQueue`. The combined cost is only computed
/// when it could beat what is already queued, which is what keeps the quadratic
/// first pass affordable.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors BrotliCompareAndPushToQueue, whose parameters are all needed"
)]
fn compare_and_push_to_queue<const N: usize>(
    out: &[Histogram<N>],
    tmp: &mut Histogram<N>,
    cluster_size: &[u32],
    data_size: usize,
    idx1: u32,
    idx2: u32,
    max_num_pairs: usize,
    pairs: &mut Vec<HistogramPair>,
) {
    if idx1 == idx2 {
        return;
    }
    let (idx1, idx2) = if idx2 < idx1 {
        (idx2, idx1)
    } else {
        (idx1, idx2)
    };

    let mut p = HistogramPair {
        idx1,
        idx2,
        cost_combo: 0.0,
        cost_diff: 0.0,
    };
    p.cost_diff = 0.5
        * cluster_cost_diff(
            cluster_size[idx1 as usize] as usize,
            cluster_size[idx2 as usize] as usize,
        );
    p.cost_diff -= out[idx1 as usize].bit_cost;
    p.cost_diff -= out[idx2 as usize].bit_cost;

    let is_good_pair = if out[idx1 as usize].total_count == 0 {
        p.cost_combo = out[idx2 as usize].bit_cost;
        true
    } else if out[idx2 as usize].total_count == 0 {
        p.cost_combo = out[idx1 as usize].bit_cost;
        true
    } else {
        let threshold = if pairs.is_empty() {
            1e99
        } else {
            pairs[0].cost_diff.max(0.0)
        };
        *tmp = out[idx1 as usize].clone();
        tmp.add_histogram(&out[idx2 as usize]);
        let cost_combo = population_cost(tmp, data_size);
        if cost_combo < threshold - p.cost_diff {
            p.cost_combo = cost_combo;
            true
        } else {
            false
        }
    };

    if !is_good_pair {
        return;
    }
    p.cost_diff += p.cost_combo;
    if !pairs.is_empty() && pair_is_less(&pairs[0], &p) {
        // The new pair is the best so far; the old best keeps its place in the
        // array if there is room.
        if pairs.len() < max_num_pairs {
            let front = pairs[0];
            pairs.push(front);
        }
        pairs[0] = p;
    } else if pairs.len() < max_num_pairs {
        pairs.push(p);
    }
}

/// Merges clusters until no merge pays or `max_clusters` remain.
///
/// Mirrors `BrotliHistogramCombine`. `symbols` maps every input histogram to
/// the cluster it belongs to and is rewritten as clusters merge; `clusters`
/// lists the surviving cluster indices. Returns how many clusters are left.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors BrotliHistogramCombine, whose parameters are all needed"
)]
fn histogram_combine<const N: usize>(
    out: &mut [Histogram<N>],
    tmp: &mut Histogram<N>,
    data_size: usize,
    cluster_size: &mut [u32],
    symbols: &mut [u32],
    clusters: &mut [u32],
    pairs: &mut Vec<HistogramPair>,
    mut num_clusters: usize,
    symbols_size: usize,
    max_clusters: usize,
    max_num_pairs: usize,
) -> usize {
    let mut cost_diff_threshold = 0.0f64;
    let mut min_cluster_size = 1usize;
    pairs.clear();

    for idx1 in 0..num_clusters {
        for idx2 in idx1 + 1..num_clusters {
            compare_and_push_to_queue(
                out,
                tmp,
                cluster_size,
                data_size,
                clusters[idx1],
                clusters[idx2],
                max_num_pairs,
                pairs,
            );
        }
    }

    while num_clusters > min_cluster_size {
        if pairs.is_empty() || pairs[0].cost_diff >= cost_diff_threshold {
            // Nothing left that pays: raise the bar so the loop ends.
            cost_diff_threshold = 1e99;
            min_cluster_size = max_clusters;
            continue;
        }
        let best_idx1 = pairs[0].idx1;
        let best_idx2 = pairs[0].idx2;
        let cost_combo = pairs[0].cost_combo;

        let absorbed = out[best_idx2 as usize].clone();
        out[best_idx1 as usize].add_histogram(&absorbed);
        out[best_idx1 as usize].bit_cost = cost_combo;
        cluster_size[best_idx1 as usize] += cluster_size[best_idx2 as usize];
        for symbol in symbols.iter_mut().take(symbols_size) {
            if *symbol == best_idx2 {
                *symbol = best_idx1;
            }
        }
        if let Some(at) = clusters[..num_clusters]
            .iter()
            .position(|&cluster| cluster == best_idx2)
        {
            clusters.copy_within(at + 1..num_clusters, at);
        }
        num_clusters -= 1;

        {
            // Drop every queued pair that touched either merged cluster,
            // keeping the best of what remains in front.
            let mut copy_to_idx = 0usize;
            for index in 0..pairs.len() {
                let p = pairs[index];
                if p.idx1 == best_idx1
                    || p.idx2 == best_idx1
                    || p.idx1 == best_idx2
                    || p.idx2 == best_idx2
                {
                    continue;
                }
                if pair_is_less(&pairs[0], &p) {
                    let front = pairs[0];
                    pairs[0] = p;
                    pairs[copy_to_idx] = front;
                } else {
                    pairs[copy_to_idx] = p;
                }
                copy_to_idx += 1;
            }
            pairs.truncate(copy_to_idx);
        }

        for &cluster in &clusters[..num_clusters] {
            compare_and_push_to_queue(
                out,
                tmp,
                cluster_size,
                data_size,
                best_idx1,
                cluster,
                max_num_pairs,
                pairs,
            );
        }
    }
    num_clusters
}

/// Returns what moving `histogram` into `candidate` would cost.
///
/// Mirrors `BrotliHistogramBitCostDistance`.
fn bit_cost_distance<const N: usize>(
    histogram: &Histogram<N>,
    candidate: &Histogram<N>,
    tmp: &mut Histogram<N>,
    data_size: usize,
) -> f64 {
    if histogram.total_count == 0 {
        return 0.0;
    }
    *tmp = histogram.clone();
    tmp.add_histogram(candidate);
    population_cost(tmp, data_size) - candidate.bit_cost
}

/// Assigns every input histogram to its cheapest cluster.
///
/// Mirrors `BrotliHistogramRemap`, including its preference for the cluster the
/// previous histogram chose when two are equally good — which is what makes the
/// resulting context map compress.
fn histogram_remap<const N: usize>(
    input: &[Histogram<N>],
    clusters: &[u32],
    num_clusters: usize,
    out: &mut [Histogram<N>],
    tmp: &mut Histogram<N>,
    data_size: usize,
    symbols: &mut [u32],
) {
    for index in 0..input.len() {
        let mut best_out = if index == 0 {
            symbols[0]
        } else {
            symbols[index - 1]
        };
        let mut best_bits =
            bit_cost_distance(&input[index], &out[best_out as usize], tmp, data_size);
        for &cluster in &clusters[..num_clusters] {
            let bits = bit_cost_distance(&input[index], &out[cluster as usize], tmp, data_size);
            if bits < best_bits {
                best_bits = bits;
                best_out = cluster;
            }
        }
        symbols[index] = best_out;
    }

    for &cluster in &clusters[..num_clusters] {
        out[cluster as usize].clear();
    }
    for index in 0..input.len() {
        let source = input[index].clone();
        out[symbols[index] as usize].add_histogram(&source);
    }
}

/// Renumbers the surviving histograms into a dense range.
///
/// Mirrors `BrotliHistogramReindex`: after remapping, `symbols` points at
/// scattered indices; this compacts them so the first use of each comes in
/// increasing order, which is what the context-map encoder expects. Returns how
/// many distinct histograms remain.
fn histogram_reindex<const N: usize>(
    out: &mut [Histogram<N>],
    symbols: &mut [u32],
    length: usize,
    new_index: &mut Vec<u32>,
    reindexed: &mut Vec<Histogram<N>>,
) -> usize {
    new_index.clear();
    new_index.resize(length, u32::MAX);
    let mut next_index = 0u32;
    for &symbol in symbols.iter().take(length) {
        let symbol = symbol as usize;
        if new_index[symbol] == u32::MAX {
            new_index[symbol] = next_index;
            next_index += 1;
        }
    }

    reindexed.clear();
    reindexed.resize(next_index as usize, Histogram::default());
    let mut written = 0u32;
    for slot in symbols.iter_mut().take(length) {
        let symbol = *slot as usize;
        if new_index[symbol] == written {
            reindexed[written as usize] = out[symbol].clone();
            written += 1;
        }
        *slot = new_index[symbol];
    }
    out[..reindexed.len()].clone_from_slice(reindexed);
    reindexed.len()
}

/// Histograms clustered in one batch of the first pass (`max_input_histograms`).
const MAX_INPUT_HISTOGRAMS: usize = 64;

/// Clusters `input` into at most `max_histograms` histograms.
///
/// Mirrors `BrotliClusterHistograms`. Returns the number of histograms written
/// to `out`; `symbols` is filled with the histogram each input maps to.
///
/// The first pass clusters in batches of sixty-four, which bounds the quadratic
/// pair search; the second lets everything that survived compete.
pub(crate) fn cluster_histograms<const N: usize>(
    input: &[Histogram<N>],
    data_size: usize,
    max_histograms: usize,
    arena: &mut ClusterArena<N>,
    out: &mut Vec<Histogram<N>>,
    symbols: &mut Vec<u32>,
) -> usize {
    let in_size = input.len();
    arena.cluster_size.clear();
    arena.cluster_size.resize(in_size, 1);
    arena.clusters.clear();
    arena.clusters.resize(in_size, 0);

    out.clear();
    out.reserve(in_size);
    symbols.clear();
    symbols.resize(in_size, 0);
    for (index, histogram) in input.iter().enumerate() {
        let mut copy = histogram.clone();
        copy.bit_cost = population_cost(histogram, data_size);
        out.push(copy);
        symbols[index] = index as u32;
    }

    let mut num_clusters = 0usize;
    let mut index = 0usize;
    while index < in_size {
        let num_to_combine = (in_size - index).min(MAX_INPUT_HISTOGRAMS);
        for j in 0..num_to_combine {
            arena.clusters[num_clusters + j] = (index + j) as u32;
        }
        let pairs_capacity = MAX_INPUT_HISTOGRAMS * MAX_INPUT_HISTOGRAMS / 2;
        let num_new_clusters = histogram_combine(
            out,
            &mut arena.tmp,
            data_size,
            &mut arena.cluster_size,
            &mut symbols[index..],
            &mut arena.clusters[num_clusters..],
            &mut arena.pairs,
            num_to_combine,
            num_to_combine,
            max_histograms,
            pairs_capacity,
        );
        num_clusters += num_new_clusters;
        index += MAX_INPUT_HISTOGRAMS;
    }

    {
        // The second pass caps the number of pairs it will keep: past that it
        // only tracks the single best.
        let max_num_pairs = (64 * num_clusters).min((num_clusters / 2) * num_clusters);
        num_clusters = histogram_combine(
            out,
            &mut arena.tmp,
            data_size,
            &mut arena.cluster_size,
            symbols,
            &mut arena.clusters,
            &mut arena.pairs,
            num_clusters,
            in_size,
            max_histograms,
            max_num_pairs,
        );
    }

    histogram_remap(
        input,
        &arena.clusters,
        num_clusters,
        out,
        &mut arena.tmp,
        data_size,
        symbols,
    );
    let final_size = histogram_reindex(
        out,
        symbols,
        in_size,
        &mut arena.new_index,
        &mut arena.reindexed,
    );
    out.truncate(final_size);
    final_size
}

/// Merges one batch of block histograms, as the block splitter needs it.
///
/// The splitter clusters its blocks itself rather than through
/// [`cluster_histograms`], because it works from block lengths rather than a
/// prepared array; this exposes the same merge step to it.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors BrotliHistogramCombine, whose parameters are all needed"
)]
pub(crate) fn combine_batch<const N: usize>(
    out: &mut [Histogram<N>],
    tmp: &mut Histogram<N>,
    data_size: usize,
    cluster_size: &mut [u32],
    symbols: &mut [u32],
    clusters: &mut [u32],
    pairs: &mut Vec<HistogramPair>,
    num_clusters: usize,
    symbols_size: usize,
    max_clusters: usize,
    max_num_pairs: usize,
) -> usize {
    histogram_combine(
        out,
        tmp,
        data_size,
        cluster_size,
        symbols,
        clusters,
        pairs,
        num_clusters,
        symbols_size,
        max_clusters,
        max_num_pairs,
    )
}

/// Exposes the move cost to the block splitter, which assigns blocks directly.
pub(crate) fn move_cost<const N: usize>(
    histogram: &Histogram<N>,
    candidate: &Histogram<N>,
    tmp: &mut Histogram<N>,
    data_size: usize,
) -> f64 {
    bit_cost_distance(histogram, candidate, tmp, data_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::constants::NUM_LITERAL_SYMBOLS;
    use crate::shared::histogram::HistogramLiteral;

    /// Builds a histogram over the bytes of `data`.
    fn histogram(data: &[u8]) -> HistogramLiteral {
        let mut histogram = HistogramLiteral::default();
        for &byte in data {
            histogram.add(usize::from(byte));
        }
        histogram
    }

    /// Clusters `input`, returning the histograms and the symbol map.
    fn cluster(
        input: &[HistogramLiteral],
        max_histograms: usize,
    ) -> (Vec<HistogramLiteral>, Vec<u32>) {
        let mut arena = ClusterArena::default();
        let mut out = Vec::new();
        let mut symbols = Vec::new();
        cluster_histograms(
            input,
            NUM_LITERAL_SYMBOLS,
            max_histograms,
            &mut arena,
            &mut out,
            &mut symbols,
        );
        (out, symbols)
    }

    #[test]
    fn a_merge_that_pays_has_a_negative_cost_difference() {
        // Two clusters of one merging into one of two: the map gets shorter.
        assert!(cluster_cost_diff(1, 1) < 0.0);
        assert!(cluster_cost_diff(4, 4) < 0.0);
        // A cluster merging with nothing changes nothing.
        assert_eq!(cluster_cost_diff(0, 5), 0.0);
    }

    #[test]
    fn pair_ordering_breaks_ties_on_the_index_span() {
        let near = HistogramPair {
            idx1: 0,
            idx2: 1,
            cost_combo: 0.0,
            cost_diff: -5.0,
        };
        let far = HistogramPair {
            idx1: 0,
            idx2: 9,
            cost_combo: 0.0,
            cost_diff: -5.0,
        };
        // Equal costs: the wider span is the worse pair.
        assert!(pair_is_less(&far, &near));
        assert!(!pair_is_less(&near, &far));

        let dearer = HistogramPair {
            cost_diff: -1.0,
            ..near
        };
        assert!(pair_is_less(&dearer, &near));
    }

    #[test]
    fn identical_histograms_collapse_into_one() {
        let input = vec![histogram(b"aaaaaaaabbbbbbbb"); 6];
        let (out, symbols) = cluster(&input, 256);
        assert_eq!(out.len(), 1);
        assert_eq!(symbols, vec![0; 6]);
        // The survivor holds every count.
        assert_eq!(out[0].total_count, 16 * 6);
    }

    #[test]
    fn unrelated_histograms_stay_apart() {
        let input = vec![
            histogram(&[0u8; 400]),
            histogram(&[128u8; 400]),
            histogram(&[255u8; 400]),
        ];
        let (out, symbols) = cluster(&input, 256);
        assert_eq!(out.len(), 3, "distinct histograms were merged");
        let mut sorted = symbols.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn the_histogram_limit_is_respected() {
        // Sixteen genuinely different histograms, capped at four.
        let input: Vec<HistogramLiteral> = (0..16u8)
            .map(|index| histogram(&vec![index * 16; 500]))
            .collect();
        let (out, symbols) = cluster(&input, 4);
        assert!(out.len() <= 4, "clustering kept {} histograms", out.len());
        assert!(symbols.iter().all(|&symbol| (symbol as usize) < out.len()));
    }

    #[test]
    fn the_symbol_map_is_dense_and_first_use_ordered() {
        let input: Vec<HistogramLiteral> = (0..40u8)
            .map(|index| histogram(&vec![index * 6; 300]))
            .collect();
        let (out, symbols) = cluster(&input, 8);

        let mut next = 0u32;
        for &symbol in &symbols {
            assert!(symbol <= next, "symbol {symbol} appeared before {next}");
            if symbol == next {
                next += 1;
            }
        }
        assert_eq!(next as usize, out.len());
    }

    #[test]
    fn every_count_survives_clustering() {
        let input: Vec<HistogramLiteral> = (0..30u8)
            .map(|index| histogram(&vec![index; 100 + usize::from(index)]))
            .collect();
        let before: usize = input.iter().map(|histogram| histogram.total_count).sum();
        let (out, _) = cluster(&input, 6);
        let after: usize = out.iter().map(|histogram| histogram.total_count).sum();
        assert_eq!(before, after);
    }

    #[test]
    fn an_empty_histogram_merges_for_free() {
        let input = vec![
            histogram(&[7u8; 500]),
            HistogramLiteral::default(),
            histogram(&[7u8; 500]),
        ];
        let (out, _) = cluster(&input, 256);
        assert_eq!(out.len(), 1, "an empty histogram was kept apart");
    }

    #[test]
    fn clustering_a_single_histogram_is_the_identity() {
        let input = vec![histogram(b"only one")];
        let (out, symbols) = cluster(&input, 256);
        assert_eq!(out.len(), 1);
        assert_eq!(symbols, vec![0]);
        assert_eq!(out[0].total_count, 8);
    }

    #[test]
    fn the_arena_can_be_reused_without_changing_the_result() {
        let input: Vec<HistogramLiteral> = (0..20u8)
            .map(|index| histogram(&vec![index * 11; 400]))
            .collect();
        let other: Vec<HistogramLiteral> =
            (0..7u8).map(|index| histogram(&vec![index; 900])).collect();

        let (expected, expected_symbols) = cluster(&input, 5);

        let mut arena = ClusterArena::default();
        let mut out = Vec::new();
        let mut symbols = Vec::new();
        cluster_histograms(
            &other,
            NUM_LITERAL_SYMBOLS,
            3,
            &mut arena,
            &mut out,
            &mut symbols,
        );
        cluster_histograms(
            &input,
            NUM_LITERAL_SYMBOLS,
            5,
            &mut arena,
            &mut out,
            &mut symbols,
        );
        assert_eq!(symbols, expected_symbols);
        assert_eq!(out.len(), expected.len());
        for (left, right) in out.iter().zip(&expected) {
            assert_eq!(left.data, right.data);
            assert_eq!(left.total_count, right.total_count);
        }
    }

    #[test]
    fn more_than_one_batch_still_clusters() {
        // Past sixty-four histograms the first pass runs in batches, and the
        // second pass has to reconcile them.
        let input: Vec<HistogramLiteral> = (0..200u32)
            .map(|index| histogram(&vec![(index % 5) as u8; 300]))
            .collect();
        let (out, symbols) = cluster(&input, 256);
        assert!(out.len() <= 5, "clustering kept {} histograms", out.len());
        assert!(symbols.iter().all(|&symbol| (symbol as usize) < out.len()));
    }
}

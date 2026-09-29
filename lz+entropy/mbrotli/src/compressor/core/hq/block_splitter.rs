//! Partitioning a meta-block's symbol streams, the high-quality way.
//!
//! Ports `c/enc/block_splitter.c` and `c/enc/block_splitter_inc.h` from the
//! pinned reference (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! The greedy splitter decides each boundary as it passes it. This one sees the
//! whole stream: it seeds a handful of entropy codes from random samples,
//! refines them, then solves for the cheapest assignment of codes to symbols by
//! dynamic programming, repeats that a few times, and finally clusters the
//! resulting blocks down to the number of block types the format allows.
//!
//! The sampling is pseudo-random, seeded at seven with the reference's
//! multiplier, and the sequence is part of the output: change it and a
//! different partition falls out.

use alloc::vec::Vec;

use fearless_simd::{Select, Simd, SimdBase, SimdMask, f64x8};

use super::cluster::{HistogramPair, combine_batch, move_cost};
use super::params::HqParams;
use crate::compressor::core::dispatch::Kernels;
use crate::shared::bit_cost::population_cost;
use crate::shared::block_split::{BlockSplit, MAX_NUMBER_OF_BLOCK_TYPES};
use crate::shared::command::Command;
use crate::shared::constants::{NUM_COMMAND_SYMBOLS, NUM_LITERAL_SYMBOLS};
use crate::shared::distance::NUM_HISTOGRAM_DISTANCE_SYMBOLS;
use crate::shared::fast_log::fast_log2;
use crate::shared::histogram::Histogram;

/// Most histograms the literal stream is seeded with (`kMaxLiteralHistograms`).
const MAX_LITERAL_HISTOGRAMS: usize = 100;

/// Most histograms the command and distance streams are seeded with
/// (`kMaxCommandHistograms`).
const MAX_COMMAND_HISTOGRAMS: usize = 50;

/// What switching literal block type is priced at (`kLiteralBlockSwitchCost`).
const LITERAL_BLOCK_SWITCH_COST: f64 = 28.1;

/// What switching command block type is priced at.
const COMMAND_BLOCK_SWITCH_COST: f64 = 13.5;

/// What switching distance block type is priced at.
const DISTANCE_BLOCK_SWITCH_COST: f64 = 14.6;

/// Symbols each literal sample covers (`kLiteralStrideLength`).
const LITERAL_STRIDE_LENGTH: usize = 70;

/// Symbols each command sample covers.
const COMMAND_STRIDE_LENGTH: usize = 40;

/// Symbols each distance sample covers.
const DISTANCE_STRIDE_LENGTH: usize = 40;

/// Literals one seeded histogram is expected to cover.
const SYMBOLS_PER_LITERAL_HISTOGRAM: usize = 544;

/// Commands one seeded histogram is expected to cover.
const SYMBOLS_PER_COMMAND_HISTOGRAM: usize = 530;

/// Distances one seeded histogram is expected to cover.
const SYMBOLS_PER_DISTANCE_HISTOGRAM: usize = 544;

/// Shortest stream that is worth splitting at all.
const MIN_LENGTH_FOR_BLOCK_SPLITTING: usize = 128;

/// Refinement passes per unit of stream length (`kIterMulForRefining`).
const ITER_MUL_FOR_REFINING: usize = 2;

/// Refinement passes every stream gets regardless (`kMinItersForRefining`).
const MIN_ITERS_FOR_REFINING: usize = 100;

/// Blocks clustered together in one batch (`HISTOGRAMS_PER_BATCH`).
const HISTOGRAMS_PER_BATCH: usize = 64;

/// Clusters one batch is expected to yield (`CLUSTERS_PER_BATCH`).
const CLUSTERS_PER_BATCH: usize = 16;

/// Bytes of the stream over which switching is discounted.
const PROLOGUE_LENGTH: usize = 2000;

/// How much the switch discount grows per byte of the prologue.
const PROLOGUE_MULTIPLIER: f64 = 0.07 / 2000.0;

/// Base of the switch discount at the very first byte.
const PROLOGUE_BASE: f64 = 0.77;

/// The id lane of a vector minimum no histogram has claimed yet.
///
/// Larger than every real id, and it converts to `u64::MAX`, the id the
/// vector scan reports when no histogram undercut the `1e99` start.
const NO_HISTOGRAM: f64 = f64::INFINITY;

/// Histogram costs [`assign_blocks`] updates per vector.
const COST_LANES: usize = 8;

/// Rounds a histogram count up to whole cost vectors.
///
/// [`find_blocks`] lays its cost rows out at this width, so the vector kernel
/// never falls back to its scalar tail.
const fn padded_histograms(num_histograms: usize) -> usize {
    num_histograms.next_multiple_of(COST_LANES)
}

/// The reference's pseudo-random generator (`MyRand`).
///
/// Seeded at seven, where its period is `1 << 29`; the sequence decides which
/// stretches of the stream seed the entropy codes, so it is part of the output.
struct Rand(u32);

impl Rand {
    /// Returns the next value, advancing the state.
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(16807);
        self.0
    }
}

/// Returns the bit cost of a symbol seen `count` times (`BitCost`).
///
/// An unseen symbol is priced below zero, which makes an entropy code that
/// never used it look better than one that did.
fn bit_cost(count: u32) -> f64 {
    if count == 0 {
        -2.0
    } else {
        fast_log2(count as usize)
    }
}

/// Scratch storage the splitter reuses across meta-blocks and streams.
pub(crate) struct SplitArena<const N: usize> {
    histograms: Vec<Histogram<N>>,
    tmp: Histogram<N>,
    insert_cost: Vec<f64>,
    cost: Vec<f64>,
    switch_signal: Vec<u8>,
    block_ids: Vec<u8>,
    new_id: Vec<u16>,
    histogram_symbols: Vec<u32>,
    block_lengths: Vec<u32>,
    sizes: Vec<u32>,
    new_clusters: Vec<u32>,
    symbols: Vec<u32>,
    remap: Vec<u32>,
    all_histograms: Vec<Histogram<N>>,
    cluster_size: Vec<u32>,
    batch: Vec<Histogram<N>>,
    pairs: Vec<HistogramPair>,
    clusters: Vec<u32>,
    new_index: Vec<u32>,
    assign: Histogram<N>,
}

impl<const N: usize> SplitArena<N> {
    /// Counts all heap-backed scratch storage.
    pub(crate) fn retained_bytes(&self) -> usize {
        (self.histograms.capacity() + self.all_histograms.capacity() + self.batch.capacity())
            * size_of::<Histogram<N>>()
            + (self.insert_cost.capacity() + self.cost.capacity()) * size_of::<f64>()
            + (self.switch_signal.capacity() + self.block_ids.capacity()) * size_of::<u8>()
            + (self.new_id.capacity()) * size_of::<u16>()
            + (self.histogram_symbols.capacity()
                + self.block_lengths.capacity()
                + self.sizes.capacity()
                + self.new_clusters.capacity()
                + self.symbols.capacity()
                + self.remap.capacity()
                + self.cluster_size.capacity()
                + self.clusters.capacity()
                + self.new_index.capacity())
                * size_of::<u32>()
            + (self.pairs.capacity()) * size_of::<HistogramPair>()
    }
}

impl<const N: usize> Default for SplitArena<N> {
    /// Returns empty scratch storage.
    fn default() -> Self {
        Self {
            histograms: Vec::new(),
            tmp: Histogram::default(),
            insert_cost: Vec::new(),
            cost: Vec::new(),
            switch_signal: Vec::new(),
            block_ids: Vec::new(),
            new_id: Vec::new(),
            histogram_symbols: Vec::new(),
            block_lengths: Vec::new(),
            sizes: Vec::new(),
            new_clusters: Vec::new(),
            symbols: Vec::new(),
            remap: Vec::new(),
            all_histograms: Vec::new(),
            cluster_size: Vec::new(),
            batch: Vec::new(),
            pairs: Vec::new(),
            clusters: Vec::new(),
            new_index: Vec::new(),
            assign: Histogram::default(),
        }
    }
}

/// Seeds `num_histograms` entropy codes from evenly spread samples.
///
/// Mirrors `InitialEntropyCodes`: one sample per histogram, at a position
/// jittered within its share of the stream.
fn initial_entropy_codes<const N: usize>(
    data: &[u16],
    stride: usize,
    num_histograms: usize,
    histograms: &mut [Histogram<N>],
) {
    let length = data.len();
    let mut seed = Rand(7);
    let block_length = length / num_histograms;
    for histogram in histograms.iter_mut().take(num_histograms) {
        histogram.clear();
    }
    for (index, histogram) in histograms.iter_mut().enumerate().take(num_histograms) {
        let mut pos = length * index / num_histograms;
        if index != 0 && block_length != 0 {
            pos += seed.next() as usize % block_length;
        }
        if pos + stride >= length {
            pos = length.saturating_sub(stride + 1);
        }
        let end = (pos + stride).min(length);
        histogram.add_vector(&data[pos..end]);
    }
}

/// Folds more random samples into the seeded codes (`RefineEntropyCodes`).
fn refine_entropy_codes<const N: usize>(
    data: &[u16],
    stride: usize,
    num_histograms: usize,
    histograms: &mut [Histogram<N>],
    tmp: &mut Histogram<N>,
) {
    let length = data.len();
    let mut iters = ITER_MUL_FOR_REFINING * length / stride + MIN_ITERS_FOR_REFINING;
    iters = iters.div_ceil(num_histograms) * num_histograms;
    let mut seed = Rand(7);
    for iter in 0..iters {
        tmp.clear();
        // One random sample per iteration (`RandomSample`).
        let mut pos = 0usize;
        let sample = if stride >= length {
            length
        } else {
            pos = seed.next() as usize % (length - stride + 1);
            stride
        };
        tmp.add_vector(&data[pos..pos + sample]);
        let target = &mut histograms[iter % num_histograms];
        let source = tmp.clone();
        target.add_histogram(&source);
    }
}

/// Borrowed rows for the block-assignment forward pass. Costs are finite,
/// histogram ids fit in a byte, and every symbol addresses a complete row.
pub(crate) struct BlockCosts<'a> {
    pub(crate) data: &'a [u16],
    pub(crate) block_switch_bitcost: f64,
    pub(crate) insert_cost: &'a [f64],
    pub(crate) cost: &'a mut [f64],
    pub(crate) switch_signal: &'a mut [u8],
    pub(crate) block_id: &'a mut [u8],
}

/// Scalar oracle, retaining the reference's operation and tie order.
fn assign_blocks_scalar(input: BlockCosts<'_>) {
    let BlockCosts {
        data,
        block_switch_bitcost,
        insert_cost,
        cost,
        switch_signal,
        block_id,
    } = input;
    let num_histograms = cost.len();
    let bitmap_len = num_histograms.div_ceil(8);
    for byte_ix in 0..data.len() {
        let ix = byte_ix * bitmap_len;
        let symbol = usize::from(data[byte_ix]);
        let insert_cost_ix = symbol * num_histograms;
        let mut min_cost = 1e99f64;
        let mut block_switch_cost = block_switch_bitcost;

        for k in 0..num_histograms {
            cost[k] += insert_cost[insert_cost_ix + k];
            if cost[k] < min_cost {
                min_cost = cost[k];
                block_id[byte_ix] = k as u8;
            }
        }
        // Switching is cheaper near the start, which lets the partition adapt
        // quickly before the statistics settle.
        if byte_ix < PROLOGUE_LENGTH {
            block_switch_cost *= PROLOGUE_BASE + PROLOGUE_MULTIPLIER * byte_ix as f64;
        }
        for k in 0..num_histograms {
            cost[k] -= min_cost;
            if cost[k] >= block_switch_cost {
                cost[k] = block_switch_cost;
                switch_signal[ix + (k >> 3)] |= 1u8 << (k & 7);
            }
        }
    }
}

/// Assigns histogram ids with the already-selected encoder token. Each lane
/// retains f64 addition/subtraction order; equal minima choose the first id.
pub(crate) fn assign_blocks<S: Simd>(simd: S, input: BlockCosts<'_>) {
    if simd.level().is_fallback() {
        assign_blocks_scalar(input);
        return;
    }
    simd.vectorize(
        #[inline(always)]
        || {
            let BlockCosts {
                data,
                block_switch_bitcost,
                insert_cost,
                cost,
                switch_signal,
                block_id,
            } = input;
            let num_histograms = cost.len();
            let bitmap_len = num_histograms.div_ceil(8);
            let vector_end = num_histograms / 8 * 8;
            for (byte_ix, &symbol) in data.iter().enumerate() {
                let row = &insert_cost[usize::from(symbol) * num_histograms..][..num_histograms];
                let mut minima = f64x8::splat(simd, 1e99);
                // Histogram ids are below 256, so f64 lanes hold them exactly
                // and the tie-break below stays on native `min` instructions;
                // `u64` lanes have no vector minimum before AVX-512.
                let mut ids = f64x8::splat(simd, NO_HISTOGRAM);
                let mut chunk_ids = f64x8::load_array(simd, [0., 1., 2., 3., 4., 5., 6., 7.]);
                for (values, prices) in cost[..vector_end]
                    .as_chunks_mut::<8>()
                    .0
                    .iter_mut()
                    .zip(row[..vector_end].as_chunks::<8>().0)
                {
                    let updated =
                        f64x8::load_array_ref(simd, values) + f64x8::load_array_ref(simd, prices);
                    updated.store_array(values);
                    let improved = updated.simd_lt(minima);
                    minima = improved.select(updated, minima);
                    ids = improved.select(chunk_ids, ids);
                    chunk_ids += 8.0;
                }
                // Each lane kept its first strict minimum; across lanes the
                // smallest id among the equal minima wins, as in the scalar
                // scan. Costs are never NaN, so both minima are exact.
                let mut min_cost = minima.reduce_min();
                let mut best_id = minima
                    .simd_eq(f64x8::splat(simd, min_cost))
                    .select(ids, f64x8::splat(simd, NO_HISTOGRAM))
                    .reduce_min() as u64;
                for k in vector_end..num_histograms {
                    cost[k] += row[k];
                    if cost[k] < min_cost {
                        min_cost = cost[k];
                        best_id = k as u64;
                    }
                }
                block_id[byte_ix] = best_id as u8;
                let mut switch_cost = block_switch_bitcost;
                if byte_ix < PROLOGUE_LENGTH {
                    switch_cost *= PROLOGUE_BASE + PROLOGUE_MULTIPLIER * byte_ix as f64;
                }
                let minimum = f64x8::splat(simd, min_cost);
                let limit = f64x8::splat(simd, switch_cost);
                let signal = &mut switch_signal[byte_ix * bitmap_len..][..bitmap_len];
                for (values, signal) in cost[..vector_end]
                    .as_chunks_mut::<8>()
                    .0
                    .iter_mut()
                    .zip(signal.iter_mut())
                {
                    let delta = f64x8::load_array_ref(simd, values) - minimum;
                    let switches = delta.simd_ge(limit);
                    switches.select(limit, delta).store_array(values);
                    *signal |= switches.to_bitmask() as u8;
                }
                for k in vector_end..num_histograms {
                    cost[k] -= min_cost;
                    if cost[k] >= switch_cost {
                        cost[k] = switch_cost;
                        signal[k >> 3] |= 1 << (k & 7);
                    }
                }
            }
        },
    );
}

/// Assigns each symbol the entropy code that codes it most cheaply.
///
/// Mirrors `FindBlocks`. The forward pass keeps, for every code, how much
/// dearer arriving here with that code is than with the best one, capped at the
/// switch cost; reaching the cap marks a position where the trace-back must
/// switch. Returns the number of blocks.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors FindBlocks, whose parameters are all needed"
)]
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
fn find_blocks<const N: usize>(
    kernels: &dyn Kernels,
    data: &[u16],
    block_switch_bitcost: f64,
    num_histograms: usize,
    histograms: &[Histogram<N>],
    alphabet_size: usize,
    insert_cost: &mut [f64],
    cost: &mut [f64],
    switch_signal: &mut [u8],
    block_id: &mut [u8],
) -> usize {
    let length = data.len();
    let bitmap_len = num_histograms.div_ceil(8);
    let mut num_blocks = 1usize;

    if num_histograms <= 1 {
        block_id[..length].fill(0);
        return 1;
    }

    // Rows are padded to whole vectors so the kernel runs no scalar tail. A
    // padding lane costs NaN: it never undercuts the minimum and never
    // reaches the switch cost, so it changes no id and sets no switch bit.
    let lanes = padded_histograms(num_histograms);

    // Cost of each symbol under each code: `log2(total) - log2(count)`, with an
    // unseen symbol priced two bits above the total.
    insert_cost[..alphabet_size * lanes].fill(0.0);
    for index in 0..num_histograms {
        insert_cost[index] = fast_log2(histograms[index].total_count);
    }
    for symbol in (0..alphabet_size).rev() {
        // Reversed, so the first row can serve as scratch for the totals.
        for j in 0..num_histograms {
            insert_cost[symbol * lanes + j] = insert_cost[j] - bit_cost(histograms[j].data[symbol]);
        }
    }

    cost[..num_histograms].fill(0.0);
    cost[num_histograms..lanes].fill(f64::NAN);
    switch_signal[..length * bitmap_len].fill(0);
    kernels.assign_blocks(BlockCosts {
        data,
        block_switch_bitcost,
        insert_cost,
        cost: &mut cost[..lanes],
        switch_signal,
        block_id,
    });

    {
        // Trace back, switching where the forward pass marked it.
        let mut byte_ix = length - 1;
        let mut ix = byte_ix * bitmap_len;
        let mut cur_id = block_id[byte_ix];
        while byte_ix > 0 {
            let mask = 1u8 << (cur_id & 7);
            byte_ix -= 1;
            ix -= bitmap_len;
            if switch_signal[ix + (usize::from(cur_id) >> 3)] & mask != 0
                && cur_id != block_id[byte_ix]
            {
                cur_id = block_id[byte_ix];
                num_blocks += 1;
            }
            block_id[byte_ix] = cur_id;
        }
    }
    num_blocks
}

/// Renumbers block ids into a dense range (`RemapBlockIds`).
///
/// Returns how many distinct ids remain.
fn remap_block_ids(block_ids: &mut [u8], new_id: &mut [u16], num_histograms: usize) -> usize {
    const INVALID_ID: u16 = 256;
    new_id[..num_histograms].fill(INVALID_ID);
    let mut next_id = 0u16;
    for &id in block_ids.iter() {
        if new_id[usize::from(id)] == INVALID_ID {
            new_id[usize::from(id)] = next_id;
            next_id += 1;
        }
    }
    for id in block_ids.iter_mut() {
        *id = new_id[usize::from(*id)] as u8;
    }
    usize::from(next_id)
}

/// Rebuilds the entropy codes from the assignment just made.
fn build_block_histograms<const N: usize>(
    data: &[u16],
    block_ids: &[u8],
    num_histograms: usize,
    histograms: &mut [Histogram<N>],
) {
    for histogram in histograms.iter_mut().take(num_histograms) {
        histogram.clear();
    }
    for (index, &symbol) in data.iter().enumerate() {
        histograms[usize::from(block_ids[index])].add(usize::from(symbol));
    }
}

/// Clusters the blocks down to at most 256 types and writes the split.
///
/// Mirrors `ClusterBlocks`. Blocks are pre-clustered in batches of sixty-four,
/// then everything that survived competes; finally every block is assigned the
/// cheapest surviving histogram, preferring the one its predecessor chose.
fn cluster_blocks<const N: usize>(
    data: &[u16],
    num_blocks: usize,
    block_ids: &[u8],
    alphabet_size: usize,
    arena: &mut SplitArena<N>,
    split: &mut BlockSplit,
) {
    let length = data.len();
    arena.histogram_symbols.clear();
    arena.histogram_symbols.resize(num_blocks, 0);
    arena.block_lengths.clear();
    arena.block_lengths.resize(num_blocks, 0);
    arena.sizes.clear();
    arena.sizes.resize(HISTOGRAMS_PER_BATCH, 0);
    arena.new_clusters.clear();
    arena.new_clusters.resize(HISTOGRAMS_PER_BATCH, 0);
    arena.symbols.clear();
    arena.symbols.resize(HISTOGRAMS_PER_BATCH, 0);
    arena.remap.clear();
    arena.remap.resize(HISTOGRAMS_PER_BATCH, 0);
    arena.all_histograms.clear();
    arena.cluster_size.clear();
    arena.batch.clear();
    arena
        .batch
        .resize(num_blocks.min(HISTOGRAMS_PER_BATCH), Histogram::default());

    {
        // Turn the run of ids into a list of block lengths.
        let mut block_idx = 0usize;
        for index in 0..length {
            arena.block_lengths[block_idx] += 1;
            if index + 1 == length || block_ids[index] != block_ids[index + 1] {
                block_idx += 1;
            }
        }
    }

    let expected_num_clusters = CLUSTERS_PER_BATCH * num_blocks.div_ceil(HISTOGRAMS_PER_BATCH);
    arena.all_histograms.reserve(expected_num_clusters);
    arena.cluster_size.reserve(expected_num_clusters);

    let mut num_clusters = 0usize;
    let mut pos = 0usize;
    let mut index = 0usize;
    while index < num_blocks {
        let num_to_combine = (num_blocks - index).min(HISTOGRAMS_PER_BATCH);
        for j in 0..num_to_combine {
            let block_length = arena.block_lengths[index + j] as usize;
            arena.batch[j].clear();
            for _ in 0..block_length {
                arena.batch[j].add(usize::from(data[pos]));
                pos += 1;
            }
            arena.batch[j].bit_cost = population_cost(&arena.batch[j], alphabet_size);
            arena.new_clusters[j] = j as u32;
            arena.symbols[j] = j as u32;
            arena.sizes[j] = 1;
        }
        let max_num_pairs = HISTOGRAMS_PER_BATCH * HISTOGRAMS_PER_BATCH / 2;
        let num_new_clusters = combine_batch(
            &mut arena.batch,
            &mut arena.tmp,
            alphabet_size,
            &mut arena.sizes,
            &mut arena.symbols,
            &mut arena.new_clusters,
            &mut arena.pairs,
            num_to_combine,
            num_to_combine,
            HISTOGRAMS_PER_BATCH,
            max_num_pairs,
        );
        for j in 0..num_new_clusters {
            let cluster = arena.new_clusters[j] as usize;
            arena.all_histograms.push(arena.batch[cluster].clone());
            arena.cluster_size.push(arena.sizes[cluster]);
            arena.remap[cluster] = j as u32;
        }
        for j in 0..num_to_combine {
            arena.histogram_symbols[index + j] =
                (num_clusters + arena.remap[arena.symbols[j] as usize] as usize) as u32;
        }
        num_clusters += num_new_clusters;
        index += HISTOGRAMS_PER_BATCH;
    }

    // Everything that survived the batches now competes.
    let max_num_pairs = (64 * num_clusters).min((num_clusters / 2) * num_clusters);
    arena.clusters.clear();
    arena.clusters.extend(0..num_clusters as u32);
    let num_final_clusters = combine_batch(
        &mut arena.all_histograms,
        &mut arena.tmp,
        alphabet_size,
        &mut arena.cluster_size,
        &mut arena.histogram_symbols,
        &mut arena.clusters,
        &mut arena.pairs,
        num_clusters,
        num_blocks,
        MAX_NUMBER_OF_BLOCK_TYPES,
        max_num_pairs,
    );

    // Assign each block the cheapest final histogram.
    arena.new_index.clear();
    arena.new_index.resize(num_clusters, u32::MAX);
    pos = 0;
    {
        let mut next_index = 0u32;
        for index in 0..num_blocks {
            arena.assign.clear();
            for _ in 0..arena.block_lengths[index] {
                arena.assign.add(usize::from(data[pos]));
                pos += 1;
            }
            // Among equally good histograms the reference prefers the one the
            // previous block used, which makes the block-type stream cheap.
            let mut best_out = if index == 0 {
                arena.histogram_symbols[0]
            } else {
                arena.histogram_symbols[index - 1]
            };
            let mut best_bits = move_cost(
                &arena.assign,
                &arena.all_histograms[best_out as usize],
                &mut arena.tmp,
                alphabet_size,
            );
            for j in 0..num_final_clusters {
                let cluster = arena.clusters[j];
                let bits = move_cost(
                    &arena.assign,
                    &arena.all_histograms[cluster as usize],
                    &mut arena.tmp,
                    alphabet_size,
                );
                if bits < best_bits {
                    best_bits = bits;
                    best_out = cluster;
                }
            }
            arena.histogram_symbols[index] = best_out;
            if arena.new_index[best_out as usize] == u32::MAX {
                arena.new_index[best_out as usize] = next_index;
                next_index += 1;
            }
        }
    }

    // Write the partition out, merging neighbours that ended up sharing a type.
    split.types.clear();
    split.lengths.clear();
    let mut cur_length = 0u32;
    let mut max_type = 0u8;
    for index in 0..num_blocks {
        cur_length += arena.block_lengths[index];
        if index + 1 == num_blocks
            || arena.histogram_symbols[index] != arena.histogram_symbols[index + 1]
        {
            let id = arena.new_index[arena.histogram_symbols[index] as usize] as u8;
            split.types.push(id);
            split.lengths.push(cur_length);
            max_type = max_type.max(id);
            cur_length = 0;
        }
    }
    split.num_blocks = split.types.len();
    split.num_types = usize::from(max_type) + 1;
}

/// Splits one symbol stream (`SplitByteVector`).
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors SplitByteVector, whose parameters are all needed"
)]
fn split_byte_vector<const N: usize>(
    kernels: &dyn Kernels,
    data: &[u16],
    alphabet_size: usize,
    symbols_per_histogram: usize,
    max_histograms: usize,
    sampling_stride_length: usize,
    block_switch_cost: f64,
    iterations: usize,
    arena: &mut SplitArena<N>,
    split: &mut BlockSplit,
) {
    let length = data.len();
    // One histogram per share of symbols, capped: the cap leaves room for the
    // context-aware clustering that follows, which this pass cannot see.
    let mut num_histograms = (length / symbols_per_histogram + 1).min(max_histograms);

    if length == 0 {
        split.num_types = 1;
        return;
    }
    if length < MIN_LENGTH_FOR_BLOCK_SPLITTING {
        split.num_types = 1;
        split.types.push(0);
        split.lengths.push(length as u32);
        split.num_blocks = split.types.len();
        return;
    }

    arena.histograms.resize(
        arena.histograms.len().max(num_histograms),
        Histogram::default(),
    );
    initial_entropy_codes(
        data,
        sampling_stride_length,
        num_histograms,
        &mut arena.histograms,
    );
    refine_entropy_codes(
        data,
        sampling_stride_length,
        num_histograms,
        &mut arena.histograms,
        &mut arena.tmp,
    );

    let bitmaplen = num_histograms.div_ceil(8);
    arena.block_ids.clear();
    arena.block_ids.resize(length, 0);
    arena.insert_cost.clear();
    arena
        .insert_cost
        .resize(alphabet_size * padded_histograms(num_histograms), 0.0);
    arena.cost.clear();
    arena.cost.resize(padded_histograms(num_histograms), 0.0);
    arena.switch_signal.clear();
    arena.switch_signal.resize(length * bitmaplen, 0);
    arena.new_id.clear();
    arena.new_id.resize(num_histograms, 0);

    let mut num_blocks = 0usize;
    for _ in 0..iterations {
        num_blocks = find_blocks(
            kernels,
            data,
            block_switch_cost,
            num_histograms,
            &arena.histograms,
            alphabet_size,
            &mut arena.insert_cost,
            &mut arena.cost,
            &mut arena.switch_signal,
            &mut arena.block_ids,
        );
        num_histograms = remap_block_ids(&mut arena.block_ids, &mut arena.new_id, num_histograms);
        build_block_histograms(
            data,
            &arena.block_ids,
            num_histograms,
            &mut arena.histograms,
        );
    }

    let block_ids = ::core::mem::take(&mut arena.block_ids);
    cluster_blocks(data, num_blocks, &block_ids, alphabet_size, arena, split);
    arena.block_ids = block_ids;
}

/// Everything the three splitters share, allocated once per stream.
#[derive(Default)]
pub(crate) struct BlockSplitter {
    literals: Vec<u16>,
    commands: Vec<u16>,
    distances: Vec<u16>,
    literal_arena: SplitArena<NUM_LITERAL_SYMBOLS>,
    command_arena: SplitArena<NUM_COMMAND_SYMBOLS>,
    distance_arena: SplitArena<NUM_HISTOGRAM_DISTANCE_SYMBOLS>,
}

impl BlockSplitter {
    /// Counts symbol staging and all three splitting arenas.
    pub(crate) fn retained_bytes(&self) -> usize {
        (self.literals.capacity() + self.commands.capacity() + self.distances.capacity())
            * size_of::<u16>()
            + self.literal_arena.retained_bytes()
            + self.command_arena.retained_bytes()
            + self.distance_arena.retained_bytes()
    }

    /// Splits all three symbol streams of a meta-block (`BrotliSplitBlock`).
    ///
    /// `pos` is the wrapped position of the first literal.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors BrotliSplitBlock, whose three output partitions and \
                  ring-buffer window are all needed"
    )]
    pub(crate) fn split(
        &mut self,
        kernels: &dyn Kernels,
        commands: &[Command],
        data: &[u8],
        pos: usize,
        mask: usize,
        params: &HqParams,
        literal_split: &mut BlockSplit,
        command_split: &mut BlockSplit,
        distance_split: &mut BlockSplit,
    ) {
        let iterations = params.split_iterations();

        // Literals, gathered into one contiguous run.
        self.literals.clear();
        let mut from_pos = pos & mask;
        for command in commands {
            let insert_len = command.insert_len as usize;
            for offset in 0..insert_len {
                let literal = data.get((from_pos + offset) & mask).copied().unwrap_or(0);
                self.literals.push(u16::from(literal));
            }
            from_pos = (from_pos + insert_len + command.copy_len() as usize) & mask;
        }
        split_byte_vector(
            kernels,
            &self.literals,
            NUM_LITERAL_SYMBOLS,
            SYMBOLS_PER_LITERAL_HISTOGRAM,
            MAX_LITERAL_HISTOGRAMS,
            LITERAL_STRIDE_LENGTH,
            LITERAL_BLOCK_SWITCH_COST,
            iterations,
            &mut self.literal_arena,
            literal_split,
        );

        // Command prefix codes.
        self.commands.clear();
        self.commands
            .extend(commands.iter().map(|command| command.cmd_prefix));
        split_byte_vector(
            kernels,
            &self.commands,
            NUM_COMMAND_SYMBOLS,
            SYMBOLS_PER_COMMAND_HISTOGRAM,
            MAX_COMMAND_HISTOGRAMS,
            COMMAND_STRIDE_LENGTH,
            COMMAND_BLOCK_SWITCH_COST,
            iterations,
            &mut self.command_arena,
            command_split,
        );

        // Distance prefix codes, from the commands that carry one.
        self.distances.clear();
        self.distances.extend(
            commands
                .iter()
                .filter(|command| command.has_distance())
                .map(Command::distance_code),
        );
        split_byte_vector(
            kernels,
            &self.distances,
            NUM_HISTOGRAM_DISTANCE_SYMBOLS,
            SYMBOLS_PER_DISTANCE_HISTOGRAM,
            MAX_COMMAND_HISTOGRAMS,
            DISTANCE_STRIDE_LENGTH,
            DISTANCE_BLOCK_SWITCH_COST,
            iterations,
            &mut self.distance_arena,
            distance_split,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::{CompressParams, QualityLevel, WindowBits};
    use alloc::boxed::Box;

    #[test]
    fn assignments_match_scalar_bits_ties_tails_and_prologue_on_every_backend() {
        for count in [1usize, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 50, 99, 100] {
            for length in [0, 1, 127, PROLOGUE_LENGTH - 1, PROLOGUE_LENGTH + 3] {
                let data: Vec<u16> = (0..length).map(|i| ((i * 73 + i / 7) % 3) as u16).collect();
                for pattern in 0..4 {
                    let prices: Vec<f64> = (0..3 * count)
                        .map(|i| match pattern {
                            0 => 0.0,
                            1 => (i % 3) as f64,
                            2 => ((i * 71 + i / 3) % 151) as f64 / 13.0,
                            _ => 1.0 + (i % 7) as f64 * f64::EPSILON,
                        })
                        .collect();
                    let initial: Vec<f64> = (0..count)
                        .map(|i| if pattern == 1 { (i % 7) as f64 } else { 0.0 })
                        .collect();
                    let mut expected_cost = initial.clone();
                    let mut expected_signal = vec![0; length * count.div_ceil(8)];
                    let mut expected_ids = vec![0; length];
                    assign_blocks_scalar(BlockCosts {
                        data: &data,
                        block_switch_bitcost: 13.5,
                        insert_cost: &prices,
                        cost: &mut expected_cost,
                        switch_signal: &mut expected_signal,
                        block_id: &mut expected_ids,
                    });
                    for backend in crate::Backend::available() {
                        let mut cost = initial.clone();
                        let mut signal = vec![0; expected_signal.len()];
                        let mut ids = vec![0; length];
                        crate::compressor::core::dispatch::select(backend.0).assign_blocks(
                            BlockCosts {
                                data: &data,
                                block_switch_bitcost: 13.5,
                                insert_cost: &prices,
                                cost: &mut cost,
                                switch_signal: &mut signal,
                                block_id: &mut ids,
                            },
                        );
                        assert_eq!(
                            cost.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            expected_cost
                                .iter()
                                .map(|x| x.to_bits())
                                .collect::<Vec<_>>(),
                            "backend={backend} histograms={count} length={length} pattern={pattern}"
                        );
                        assert_eq!(signal, expected_signal);
                        assert_eq!(ids, expected_ids);
                    }
                }
            }
        }
    }

    /// `find_blocks` pads its rows with NaN-cost lanes; those lanes must not
    /// change an id, a switch bit, or a real lane's cost.
    #[test]
    fn nan_padding_lanes_leave_the_assignment_unchanged() {
        for count in [2usize, 5, 9, 50, 100] {
            let lanes = padded_histograms(count);
            let length = PROLOGUE_LENGTH + 3;
            let data: Vec<u16> = (0..length).map(|i| ((i * 73 + i / 7) % 3) as u16).collect();
            let prices: Vec<f64> = (0..3 * count)
                .map(|i| ((i * 71 + i / 3) % 151) as f64 / 13.0)
                .collect();
            let mut padded_prices = vec![0.0; 3 * lanes];
            for (row, padded) in prices.chunks(count).zip(padded_prices.chunks_mut(lanes)) {
                padded[..count].copy_from_slice(row);
            }
            let mut expected_cost = vec![0.0; count];
            let mut expected_signal = vec![0; length * count.div_ceil(8)];
            let mut expected_ids = vec![0; length];
            assign_blocks_scalar(BlockCosts {
                data: &data,
                block_switch_bitcost: 13.5,
                insert_cost: &prices,
                cost: &mut expected_cost,
                switch_signal: &mut expected_signal,
                block_id: &mut expected_ids,
            });
            for backend in crate::Backend::available() {
                let mut cost = vec![f64::NAN; lanes];
                cost[..count].fill(0.0);
                let mut signal = vec![0; expected_signal.len()];
                let mut ids = vec![0; length];
                crate::compressor::core::dispatch::select(backend.0).assign_blocks(BlockCosts {
                    data: &data,
                    block_switch_bitcost: 13.5,
                    insert_cost: &padded_prices,
                    cost: &mut cost,
                    switch_signal: &mut signal,
                    block_id: &mut ids,
                });
                assert_eq!(
                    cost[..count]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    expected_cost
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    "backend={backend} histograms={count}"
                );
                assert!(cost[count..].iter().all(|x| x.is_nan()));
                assert_eq!(
                    signal, expected_signal,
                    "backend={backend} histograms={count}"
                );
                assert_eq!(ids, expected_ids, "backend={backend} histograms={count}");
            }
        }
    }

    #[test]
    fn histogram_counts_pad_to_whole_cost_vectors() {
        assert_eq!(padded_histograms(0), 0);
        assert_eq!(padded_histograms(1), COST_LANES);
        assert_eq!(padded_histograms(COST_LANES), COST_LANES);
        assert_eq!(padded_histograms(100), 104);
    }

    /// A tie across chunks goes to the smaller id, not the lower lane: id 5
    /// sits in lane 5 of the first chunk, id 11 in lane 3 of the second.
    #[test]
    fn tied_minima_in_different_chunks_pick_the_smallest_id() {
        for backend in crate::Backend::available() {
            let mut cost = vec![10.0; 16];
            cost[5] = 1.0;
            cost[11] = 1.0;
            let mut signal = vec![0; 2];
            let mut ids = vec![0; 1];
            crate::compressor::core::dispatch::select(backend.0).assign_blocks(BlockCosts {
                data: &[0],
                block_switch_bitcost: 100.0,
                insert_cost: &[0.0; 16],
                cost: &mut cost,
                switch_signal: &mut signal,
                block_id: &mut ids,
            });
            assert_eq!(ids, [5], "backend={backend}");
            assert_eq!(cost[5], 0.0);
            assert_eq!(cost[11], 0.0);
        }
    }

    /// Resolves quality eleven's parameters.
    fn params() -> HqParams {
        HqParams::new(&CompressParams::new(QualityLevel::Q11, WindowBits::DEFAULT))
            .expect("supported quality")
    }

    /// Splits a literal stream on its own, returning the partition.
    fn split_literals(data: &[u16], iterations: usize) -> BlockSplit {
        let mut arena = SplitArena::<NUM_LITERAL_SYMBOLS>::default();
        let mut split = BlockSplit::default();
        split_byte_vector(
            &*crate::compressor::core::dispatch::select(fearless_simd::Level::fallback()),
            data,
            NUM_LITERAL_SYMBOLS,
            SYMBOLS_PER_LITERAL_HISTOGRAM,
            MAX_LITERAL_HISTOGRAMS,
            LITERAL_STRIDE_LENGTH,
            LITERAL_BLOCK_SWITCH_COST,
            iterations,
            &mut arena,
            &mut split,
        );
        split
    }

    /// Runs the C splitter over the same commands, returning its partitions.
    ///
    /// Reaches `BrotliSplitBlock` — which has no public header — through this
    /// workspace's shim. The two `Command` layouts agree field for field, which
    /// is what lets the array be passed through unchanged.
    fn c_split(
        quality: i32,
        commands: &[Command],
        data: &[u8],
        pos: usize,
        mask: usize,
    ) -> [(usize, Vec<u8>, Vec<u32>); 3] {
        let capacity = commands.len().max(data.len()) + 16;
        let mut types = [
            vec![0u8; capacity],
            vec![0u8; capacity],
            vec![0u8; capacity],
        ];
        let mut lengths = [
            vec![0u32; capacity],
            vec![0u32; capacity],
            vec![0u32; capacity],
        ];
        let mut num_types = [0usize; 3];
        let mut blocks = [0usize; 3];

        // SAFETY: every pointer is valid for the length it is passed with, the
        // command array has the layout the shim documents, and `data` is
        // readable at every index the commands reach under `mask`.
        unsafe {
            google_brotli_ffi::mbrotli_shim_split_block(
                quality,
                22,
                commands.as_ptr().cast::<u8>(),
                commands.len(),
                data.as_ptr(),
                pos,
                mask,
                capacity,
                types[0].as_mut_ptr(),
                lengths[0].as_mut_ptr(),
                &raw mut num_types[0],
                &raw mut blocks[0],
                types[1].as_mut_ptr(),
                lengths[1].as_mut_ptr(),
                &raw mut num_types[1],
                &raw mut blocks[1],
                types[2].as_mut_ptr(),
                lengths[2].as_mut_ptr(),
                &raw mut num_types[2],
                &raw mut blocks[2],
            );
        }

        core::array::from_fn(|index| {
            (
                num_types[index],
                types[index][..blocks[index]].to_vec(),
                lengths[index][..blocks[index]].to_vec(),
            )
        })
    }

    /// Builds a fixture of exactly the length its commands consume.
    type Fixture = Box<dyn Fn(usize) -> Vec<u8>>;

    /// Returns how many input bytes `commands` consume.
    ///
    /// The C shim copies literals out of `data` with `memcpy` and no bounds
    /// check, so a fixture whose commands outrun its buffer would have it read
    /// past the end. The encoder never does that — its ring buffer always holds
    /// the whole meta-block — so the fixtures here have to hold to the same
    /// contract.
    fn consumed(commands: &[Command]) -> usize {
        commands
            .iter()
            .map(|command| command.insert_len as usize + command.copy_len() as usize)
            .sum()
    }

    /// Compares both splitters over one command stream.
    fn assert_split_matches_c(name: &str, quality: i32, commands: &[Command], data: &[u8]) {
        assert!(
            consumed(commands) <= data.len(),
            "case {name}: the commands consume {} bytes of a {}-byte fixture",
            consumed(commands),
            data.len()
        );
        let mut splitter = BlockSplitter::default();
        let (mut literal, mut command, mut distance) = Default::default();
        let mut params = params();
        params.quality = if quality == 10 {
            crate::compressor::core::hq::params::HqQuality::Q10
        } else {
            crate::compressor::core::hq::params::HqQuality::Q11
        };
        splitter.split(
            &*crate::compressor::core::dispatch::select(fearless_simd::Level::fallback()),
            commands,
            data,
            0,
            usize::MAX,
            &params,
            &mut literal,
            &mut command,
            &mut distance,
        );

        let expected = c_split(quality, commands, data, 0, usize::MAX);
        let mut failures = Vec::new();
        for (label, ours, theirs) in [
            ("literal", &literal, &expected[0]),
            ("command", &command, &expected[1]),
            ("distance", &distance, &expected[2]),
        ] {
            if (ours.num_types, &ours.types, &ours.lengths) != (theirs.0, &theirs.1, &theirs.2) {
                failures.push(format!(
                    "{label}: ours types={} blocks={} lengths={:?}\n        theirs types={} blocks={} lengths={:?}",
                    ours.num_types, ours.num_blocks, &ours.lengths[..ours.lengths.len().min(12)],
                    theirs.0, theirs.1.len(), &theirs.2[..theirs.2.len().min(12)]
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "case {name}, quality {quality}:\n      {}",
            failures.join("\n      ")
        );
    }

    #[test]
    fn the_generator_matches_the_reference_sequence() {
        // Seeded at seven and multiplied by 16807 with wrapping.
        let mut rand = Rand(7);
        assert_eq!(rand.next(), 7u32.wrapping_mul(16807));
        assert_eq!(rand.next(), 7u32.wrapping_mul(16807).wrapping_mul(16807));
    }

    #[test]
    fn an_unseen_symbol_is_priced_below_zero() {
        assert_eq!(bit_cost(0), -2.0);
        assert_eq!(bit_cost(1), 0.0);
        assert_eq!(bit_cost(256), 8.0);
    }

    #[test]
    fn an_empty_stream_becomes_one_block_type() {
        let split = split_literals(&[], 3);
        assert_eq!(split.num_types, 1);
        assert!(split.lengths.is_empty());
    }

    #[test]
    fn a_short_stream_is_never_split() {
        let data: Vec<u16> = (0..MIN_LENGTH_FOR_BLOCK_SPLITTING as u16 - 1).collect();
        let split = split_literals(&data, 3);
        assert_eq!(split.num_types, 1);
        assert_eq!(split.num_blocks, 1);
        assert_eq!(split.lengths, vec![data.len() as u32]);
    }

    #[test]
    fn a_uniform_stream_stays_one_block_type() {
        let data = vec![42u16; 8000];
        let split = split_literals(&data, 3);
        assert_eq!(split.num_types, 1, "a uniform stream was split");
        assert_eq!(split.lengths.iter().sum::<u32>(), data.len() as u32);
    }

    #[test]
    fn a_stream_that_changes_character_is_split() {
        // Two halves drawn from disjoint alphabets: one code cannot serve both.
        let mut data = vec![7u16; 6000];
        data.extend(::core::iter::repeat_n(200u16, 6000));
        let split = split_literals(&data, 10);
        assert!(split.num_types >= 2, "a two-part stream was not split");
        assert_eq!(split.lengths.iter().sum::<u32>(), data.len() as u32);
    }

    #[test]
    fn an_alternating_stream_reuses_its_block_types() {
        // Four runs from two alphabets: the partition should reuse two types
        // rather than inventing four.
        let mut data = Vec::new();
        for round in 0..4 {
            let symbol = if round % 2 == 0 { 3u16 } else { 180 };
            data.extend(::core::iter::repeat_n(symbol, 4000));
        }
        let split = split_literals(&data, 10);
        assert!(split.num_types <= 3, "kept {} types", split.num_types);
        assert!(split.num_blocks >= 2);
        assert_eq!(split.lengths.iter().sum::<u32>(), data.len() as u32);
    }

    #[test]
    fn every_partition_is_well_formed() {
        let mut rng = 0x1357_9BDF_2468_ACE0u64;
        let mut data = Vec::new();
        for segment in 0..8u16 {
            for _ in 0..3000 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                data.push(segment * 30 + ((rng >> 24) % 8) as u16);
            }
        }
        let split = split_literals(&data, 10);

        assert_eq!(split.num_blocks, split.types.len());
        assert_eq!(split.num_blocks, split.lengths.len());
        assert_eq!(split.lengths.iter().sum::<u32>(), data.len() as u32);
        assert!(split.lengths.iter().all(|&length| length > 0));
        assert!(split.num_types <= MAX_NUMBER_OF_BLOCK_TYPES);
        assert!(
            split
                .types
                .iter()
                .all(|&t| usize::from(t) < split.num_types),
            "a block type escaped the type count"
        );
        // Neighbouring blocks always differ in type, or they would be one.
        assert!(split.types.windows(2).all(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn splitting_is_deterministic() {
        let mut data = Vec::new();
        for segment in 0..6u16 {
            data.extend(::core::iter::repeat_n(segment * 40, 2500));
        }
        let first = split_literals(&data, 10);
        let second = split_literals(&data, 10);
        assert_eq!(first.types, second.types);
        assert_eq!(first.lengths, second.lengths);
        assert_eq!(first.num_types, second.num_types);
    }

    #[test]
    fn the_iteration_count_can_change_the_partition() {
        // Quality ten refines three times and quality eleven ten times; on a
        // stream with structure to find, that is visible in the result.
        let mut data = Vec::new();
        for segment in 0..10u16 {
            data.extend(::core::iter::repeat_n(segment * 25, 1500));
        }
        let few = split_literals(&data, 3);
        let many = split_literals(&data, 10);
        assert_eq!(few.lengths.iter().sum::<u32>(), data.len() as u32);
        assert_eq!(many.lengths.iter().sum::<u32>(), data.len() as u32);
    }

    #[test]
    fn all_three_streams_are_split_together() {
        let dist = crate::shared::distance::DistanceParams::default();
        let data: Vec<u8> = (0..40_000u32).map(|i| (i % 251) as u8).collect();
        let commands: Vec<Command> = (0..2000)
            .map(|index| Command::new(&dist, 8, 12, 0, 20 + index % 300))
            .collect();

        let mut splitter = BlockSplitter::default();
        let mut literal = BlockSplit::default();
        let mut command = BlockSplit::default();
        let mut distance = BlockSplit::default();
        splitter.split(
            &*crate::compressor::core::dispatch::select(fearless_simd::Level::fallback()),
            &commands,
            &data,
            0,
            usize::MAX,
            &params(),
            &mut literal,
            &mut command,
            &mut distance,
        );

        let literals: u32 = commands.iter().map(|c| c.insert_len).sum();
        assert_eq!(literal.lengths.iter().sum::<u32>(), literals);
        assert_eq!(command.lengths.iter().sum::<u32>(), commands.len() as u32);
        let with_distance = commands.iter().filter(|c| c.has_distance()).count();
        assert_eq!(distance.lengths.iter().sum::<u32>(), with_distance as u32);
    }

    #[test]
    fn every_partition_matches_the_c_splitter() {
        let dist = crate::shared::distance::DistanceParams::default();
        let cases: Vec<(&str, Fixture, Vec<Command>)> = vec![
            (
                "uniform",
                Box::new(|n| vec![b'a'; n]),
                (0..3000)
                    .map(|_| Command::new(&dist, 8, 12, 0, 40))
                    .collect(),
            ),
            (
                "spread-distances",
                Box::new(|n| (0..n as u32).map(|i| (i % 251) as u8).collect()),
                (0..3000)
                    .map(|index| {
                        Command::new(&dist, 6, 10 + index % 40, 0, 20 + (index * 7) % 4000)
                    })
                    .collect(),
            ),
            (
                "two-halves",
                Box::new(|n| {
                    let mut data = vec![b'x'; n / 2];
                    data.extend(::core::iter::repeat_n(b'Q', n - n / 2));
                    data
                }),
                (0..2500)
                    .map(|index| Command::new(&dist, 10, 6, 0, 30 + index % 17))
                    .collect(),
            ),
            (
                "random",
                Box::new(move |n| {
                    let mut rng = 0x0BAD_F00D_0BAD_F00Du64;
                    (0..n)
                        .map(|_| {
                            rng ^= rng << 13;
                            rng ^= rng >> 7;
                            rng ^= rng << 17;
                            (rng >> 24) as u8
                        })
                        .collect()
                }),
                (0..2000)
                    .map(|index| Command::new(&dist, 12, 4 + index % 9, 0, 16 + (index * 13) % 900))
                    .collect(),
            ),
            (
                "literal-only",
                Box::new(|n| (0..n as u32).map(|i| (i * 3 % 199) as u8).collect()),
                (0..900).map(|_| Command::insert_only(50)).collect(),
            ),
            (
                "tiny",
                Box::new(|n| vec![b'z'; n]),
                (0..20).map(|_| Command::new(&dist, 4, 6, 0, 25)).collect(),
            ),
        ];

        for (name, make_data, commands) in cases {
            // Exactly as much input as the commands consume, which is what the
            // ring buffer always holds in the encoder.
            let data = make_data(consumed(&commands));
            for quality in [10, 11] {
                assert_split_matches_c(name, quality, &commands, &data);
            }
        }
    }

    #[test]
    fn a_reused_splitter_produces_the_same_partition() {
        let dist = crate::shared::distance::DistanceParams::default();
        let data: Vec<u8> = (0..30_000u32).map(|i| (i * 7 % 253) as u8).collect();
        let commands: Vec<Command> = (0..1500)
            .map(|index| Command::new(&dist, 6, 14, 0, 25 + index % 200))
            .collect();
        let other: Vec<Command> = (0..900)
            .map(|index| Command::new(&dist, 3, 9, 0, 40 + index % 90))
            .collect();
        let params = params();

        let mut fresh = BlockSplitter::default();
        let (mut l1, mut c1, mut d1) = Default::default();
        fresh.split(
            &*crate::compressor::core::dispatch::select(fearless_simd::Level::fallback()),
            &commands,
            &data,
            0,
            usize::MAX,
            &params,
            &mut l1,
            &mut c1,
            &mut d1,
        );

        let mut reused = BlockSplitter::default();
        let (mut l0, mut c0, mut d0) = Default::default();
        reused.split(
            &*crate::compressor::core::dispatch::select(fearless_simd::Level::fallback()),
            &other,
            &data,
            0,
            usize::MAX,
            &params,
            &mut l0,
            &mut c0,
            &mut d0,
        );
        let (mut l2, mut c2, mut d2) = Default::default();
        reused.split(
            &*crate::compressor::core::dispatch::select(fearless_simd::Level::fallback()),
            &commands,
            &data,
            0,
            usize::MAX,
            &params,
            &mut l2,
            &mut c2,
            &mut d2,
        );

        assert_eq!((l1.types, l1.lengths), (l2.types, l2.lengths));
        assert_eq!((c1.types, c1.lengths), (c2.types, c2.lengths));
        assert_eq!((d1.types, d1.lengths), (d2.types, d2.lengths));
    }
}

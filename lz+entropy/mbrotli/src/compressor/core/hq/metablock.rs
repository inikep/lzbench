//! Building a meta-block the high-quality way.
//!
//! Ports `BrotliBuildMetaBlock`, `ComputeDistanceCost` and
//! `RecomputeDistancePrefixes` from `c/enc/metablock.c`, together with
//! `BrotliBuildHistogramsWithContext` from `c/enc/histogram.c`, of the pinned
//! reference (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! Three things happen here that the greedy builder does not do. The distance
//! alphabet is re-chosen for this block by pricing every legal combination of
//! postfix and direct codes. The literal histograms are gathered per block type
//! *and* per context, sixty-four of them per type, rather than through a fixed
//! static map. And all of those histograms are then clustered down to what the
//! format allows, which is what produces the context map the decoder reads.

use alloc::vec::Vec;

use super::block_splitter::BlockSplitter;
use super::cluster::{ClusterArena, cluster_histograms};
use super::params::HqParams;
use crate::shared::bit_cost::population_cost;
use crate::shared::block_split::BlockSplit;
use crate::shared::command::{Command, prefix_encode_copy_distance};
use crate::shared::constants::NUM_LITERAL_SYMBOLS;
use crate::shared::distance::{DistanceParams, MAX_NPOSTFIX, NUM_HISTOGRAM_DISTANCE_SYMBOLS};
use crate::shared::format::ContextMode;
use crate::shared::histogram::{HistogramCommand, HistogramDistance, HistogramLiteral};
use crate::shared::metablock::{DISTANCE_CONTEXT_BITS, LITERAL_CONTEXT_BITS, MetaBlockSplit};

/// Most histograms a meta-block may end up with, one per block-type byte.
const MAX_NUMBER_OF_HISTOGRAMS: usize = 256;

/// Walks a block split symbol by symbol, reporting the current type.
///
/// Mirrors `BlockSplitIterator`.
struct BlockSplitIterator<'a> {
    split: &'a BlockSplit,
    idx: usize,
    type_: usize,
    length: u32,
}

impl<'a> BlockSplitIterator<'a> {
    /// Starts at the first block of `split`.
    fn new(split: &'a BlockSplit) -> Self {
        Self {
            split,
            idx: 0,
            type_: 0,
            length: split.lengths.first().copied().unwrap_or(0),
        }
    }

    /// Advances one symbol, returning the block type it belongs to.
    fn next(&mut self) -> usize {
        if self.length == 0 && self.idx + 1 < self.split.num_blocks {
            self.idx += 1;
            self.type_ = usize::from(self.split.types[self.idx]);
            self.length = self.split.lengths[self.idx];
        }
        self.length = self.length.saturating_sub(1);
        self.type_
    }
}

/// Rewrites the distance prefixes of `commands` under new parameters.
///
/// Mirrors `RecomputeDistancePrefixes`, which runs once the block has settled
/// on an alphabet different from the one the commands were created with.
fn recompute_distance_prefixes(
    commands: &mut [Command],
    original: &DistanceParams,
    new: &DistanceParams,
) {
    if original.postfix_bits == new.postfix_bits && original.num_direct == new.num_direct {
        return;
    }
    for command in commands {
        if command.has_distance() {
            let distance = command.restore_distance_code(original) as usize;
            let (prefix, extra) =
                prefix_encode_copy_distance(distance, new.num_direct, new.postfix_bits);
            command.dist_prefix = prefix;
            command.dist_extra = extra;
        }
    }
}

/// Prices coding `commands`' distances under `new`.
///
/// Mirrors `ComputeDistanceCost`. Returns `None` when a distance cannot be
/// expressed at all under the candidate alphabet, which rules it out.
fn compute_distance_cost(
    commands: &[Command],
    original: &DistanceParams,
    new: &DistanceParams,
    tmp: &mut HistogramDistance,
) -> Option<f64> {
    tmp.clear();
    let equal_params =
        original.postfix_bits == new.postfix_bits && original.num_direct == new.num_direct;
    let mut extra_bits = 0f64;

    for command in commands {
        if !command.has_distance() {
            continue;
        }
        let dist_prefix = if equal_params {
            command.dist_prefix
        } else {
            let distance = command.restore_distance_code(original);
            if distance > new.max_distance {
                return None;
            }
            prefix_encode_copy_distance(distance as usize, new.num_direct, new.postfix_bits).0
        };
        tmp.add(usize::from(dist_prefix & 0x3FF));
        extra_bits += f64::from(dist_prefix >> 10);
    }
    // Priced over the whole histogram alphabet, not the candidate's own
    // limit: `BrotliPopulationCostDistance` always uses
    // `BROTLI_NUM_HISTOGRAM_DISTANCE_SYMBOLS`, and narrowing it here would make
    // a wide alphabet look cheaper than the reference thinks it is.
    Some(population_cost(tmp, NUM_HISTOGRAM_DISTANCE_SYMBOLS) + extra_bits)
}

/// Chooses the distance alphabet this meta-block will use.
///
/// Mirrors the parameter search at the top of `BrotliBuildMetaBlock`: it walks
/// postfix bits outward and, for each, direct-code counts upward, stopping a
/// row as soon as the cost turns. The awkward `ndirect_msb` carry between rows
/// is the reference's, and it decides which combinations are examined at all.
fn choose_distance_params(
    commands: &[Command],
    large_window: bool,
    original: DistanceParams,
    tmp: &mut HistogramDistance,
) -> DistanceParams {
    let mut best = original;
    let mut best_dist_cost = 1e99f64;
    let mut check_orig = true;
    let mut ndirect_msb = 0u32;

    for npostfix in 0..=MAX_NPOSTFIX {
        while ndirect_msb < 16 {
            let ndirect = ndirect_msb << npostfix;
            let candidate = DistanceParams::for_window(large_window, npostfix, ndirect);
            if npostfix == original.postfix_bits && ndirect == original.num_direct {
                check_orig = false;
            }
            let Some(dist_cost) = compute_distance_cost(commands, &original, &candidate, tmp)
            else {
                break;
            };
            if dist_cost > best_dist_cost {
                break;
            }
            best_dist_cost = dist_cost;
            best = candidate;
            ndirect_msb += 1;
        }
        ndirect_msb = ndirect_msb.saturating_sub(1);
        ndirect_msb /= 2;
    }

    if check_orig
        && let Some(dist_cost) = compute_distance_cost(commands, &original, &original, tmp)
        && dist_cost < best_dist_cost
    {
        best = original;
    }
    best
}

/// Gathers the per-context histograms of a meta-block.
///
/// Mirrors `BrotliBuildHistogramsWithContext`. A literal's histogram is chosen
/// by its block type and, unless context modelling is off, by the two bytes
/// before it; a distance's by its block type and the command's distance
/// context.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors BrotliBuildHistogramsWithContext, whose parameters are all needed"
)]
fn build_histograms_with_context(
    commands: &[Command],
    literal_split: &BlockSplit,
    command_split: &BlockSplit,
    distance_split: &BlockSplit,
    data: &[u8],
    start_pos: usize,
    mask: usize,
    mut prev_byte: u8,
    mut prev_byte2: u8,
    context_mode: Option<ContextMode>,
    literal_histograms: &mut [HistogramLiteral],
    command_histograms: &mut [HistogramCommand],
    distance_histograms: &mut [HistogramDistance],
) {
    let mut pos = start_pos;
    let mut literal_it = BlockSplitIterator::new(literal_split);
    let mut command_it = BlockSplitIterator::new(command_split);
    let mut distance_it = BlockSplitIterator::new(distance_split);

    for command in commands {
        let block_type = command_it.next();
        if let Some(histogram) = command_histograms.get_mut(block_type) {
            histogram.add(usize::from(command.cmd_prefix));
        }
        for _ in 0..command.insert_len {
            let block_type = literal_it.next();
            let context = match context_mode {
                Some(mode) => {
                    (block_type << LITERAL_CONTEXT_BITS) + mode.context(prev_byte, prev_byte2)
                }
                None => block_type,
            };
            let literal = data.get(pos & mask).copied().unwrap_or(0);
            if let Some(histogram) = literal_histograms.get_mut(context) {
                histogram.add(usize::from(literal));
            }
            prev_byte2 = prev_byte;
            prev_byte = literal;
            pos += 1;
        }
        pos += command.copy_len() as usize;
        if command.copy_len() != 0 {
            prev_byte2 = data.get((pos - 2) & mask).copied().unwrap_or(0);
            prev_byte = data.get((pos - 1) & mask).copied().unwrap_or(0);
            if command.cmd_prefix >= 128 {
                let block_type = distance_it.next();
                let context = (block_type << DISTANCE_CONTEXT_BITS) + command.distance_context();
                if let Some(histogram) = distance_histograms.get_mut(context) {
                    histogram.add(usize::from(command.distance_code()));
                }
            }
        }
    }
}

/// Everything the high-quality meta-block builder reuses across blocks.
#[derive(Default)]
pub(crate) struct MetaBlockBuilder {
    splitter: BlockSplitter,
    distance_tmp: HistogramDistance,
    literal_histograms: Vec<HistogramLiteral>,
    distance_histograms: Vec<HistogramDistance>,
    literal_cluster: ClusterArena<NUM_LITERAL_SYMBOLS>,
    distance_cluster: ClusterArena<NUM_HISTOGRAM_DISTANCE_SYMBOLS>,
}

impl MetaBlockBuilder {
    /// Counts splitter, histogram and cluster allocations, excluding inline state.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.splitter.retained_bytes()
            + self.literal_histograms.capacity() * size_of::<HistogramLiteral>()
            + self.distance_histograms.capacity() * size_of::<HistogramDistance>()
            + self.literal_cluster.retained_bytes()
            + self.distance_cluster.retained_bytes()
    }

    /// Builds one meta-block, re-tuning `dist` for it (`BrotliBuildMetaBlock`).
    ///
    /// `commands` is rewritten in place when the distance alphabet changes, and
    /// `dist` is updated to what the block will actually be written with.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors BrotliBuildMetaBlock, whose parameters are all needed"
    )]
    pub(crate) fn build(
        &mut self,
        kernels: &dyn crate::compressor::core::dispatch::Kernels,
        data: &[u8],
        pos: usize,
        mask: usize,
        params: &HqParams,
        prev_byte: u8,
        prev_byte2: u8,
        commands: &mut [Command],
        context_mode: ContextMode,
        dist: &mut DistanceParams,
        mb: &mut MetaBlockSplit,
    ) {
        let original = *dist;
        *dist = choose_distance_params(
            commands,
            params.window.is_large(),
            original,
            &mut self.distance_tmp,
        );
        recompute_distance_prefixes(commands, &original, dist);

        self.splitter.split(
            kernels,
            commands,
            data,
            pos,
            mask,
            params,
            &mut mb.literal_split,
            &mut mb.command_split,
            &mut mb.distance_split,
        );

        // With context modelling on, every block type owns sixty-four literal
        // histograms; with it off, one.
        let context_mode = (!params.disable_literal_context_modeling).then_some(context_mode);
        let literal_context_multiplier = if context_mode.is_some() {
            1usize << LITERAL_CONTEXT_BITS
        } else {
            1
        };

        let literal_histograms_size = mb.literal_split.num_types * literal_context_multiplier;
        self.literal_histograms.clear();
        self.literal_histograms
            .resize(literal_histograms_size, HistogramLiteral::default());

        let distance_histograms_size = mb.distance_split.num_types << DISTANCE_CONTEXT_BITS;
        self.distance_histograms.clear();
        self.distance_histograms
            .resize(distance_histograms_size, HistogramDistance::default());

        mb.command_histograms.clear();
        mb.command_histograms
            .resize(mb.command_split.num_types, HistogramCommand::default());

        build_histograms_with_context(
            commands,
            &mb.literal_split,
            &mb.command_split,
            &mb.distance_split,
            data,
            pos,
            mask,
            prev_byte,
            prev_byte2,
            context_mode,
            &mut self.literal_histograms,
            &mut mb.command_histograms,
            &mut self.distance_histograms,
        );

        // Literal contexts: cluster, then spread the result over every context
        // when modelling is off.
        mb.literal_context_map
            .resize(mb.literal_split.num_types << LITERAL_CONTEXT_BITS, 0);
        cluster_histograms(
            &self.literal_histograms,
            NUM_LITERAL_SYMBOLS,
            MAX_NUMBER_OF_HISTOGRAMS,
            &mut self.literal_cluster,
            &mut mb.literal_histograms,
            &mut mb.literal_context_map,
        );
        if params.disable_literal_context_modeling {
            mb.literal_context_map
                .resize(mb.literal_split.num_types << LITERAL_CONTEXT_BITS, 0);
            for block_type in (0..mb.literal_split.num_types).rev() {
                let assigned = mb.literal_context_map[block_type];
                for context in 0..(1usize << LITERAL_CONTEXT_BITS) {
                    mb.literal_context_map[(block_type << LITERAL_CONTEXT_BITS) + context] =
                        assigned;
                }
            }
        }

        cluster_histograms(
            &self.distance_histograms,
            NUM_HISTOGRAM_DISTANCE_SYMBOLS,
            MAX_NUMBER_OF_HISTOGRAMS,
            &mut self.distance_cluster,
            &mut mb.distance_histograms,
            &mut mb.distance_context_map,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::{CompressMode, CompressParams, QualityLevel, WindowBits};
    use alloc::boxed::Box;

    /// Resolves quality eleven's parameters.
    fn params(mode: CompressMode) -> HqParams {
        HqParams::new(&CompressParams::new(QualityLevel::Q11, WindowBits::DEFAULT).with_mode(mode))
            .expect("supported quality")
    }

    /// Builds commands that copy from a spread of distances.
    fn commands(count: usize, dist: &DistanceParams) -> Vec<Command> {
        (0..count)
            .map(|index| Command::new(dist, 6, 10 + index % 40, 0, 20 + (index * 7) % 4000))
            .collect()
    }

    #[test]
    fn the_split_iterator_walks_every_block_in_turn() {
        let split = BlockSplit {
            num_types: 3,
            num_blocks: 3,
            types: vec![0, 2, 1],
            lengths: vec![2, 3, 1],
        };
        let mut it = BlockSplitIterator::new(&split);
        let seen: Vec<usize> = (0..6).map(|_| it.next()).collect();
        assert_eq!(seen, vec![0, 0, 2, 2, 2, 1]);
    }

    #[test]
    fn an_empty_split_reports_type_zero() {
        let split = BlockSplit::default();
        let mut it = BlockSplitIterator::new(&split);
        assert_eq!(it.next(), 0);
        assert_eq!(it.next(), 0);
    }

    #[test]
    fn recomputing_prefixes_is_a_no_op_under_the_same_parameters() {
        let dist = DistanceParams::default();
        let mut cmds = commands(50, &dist);
        let before = cmds.clone();
        recompute_distance_prefixes(&mut cmds, &dist, &dist);
        assert_eq!(cmds, before);
    }

    #[test]
    fn recomputing_prefixes_preserves_the_distance() {
        let original = DistanceParams::default();
        let new = DistanceParams::new(2, 8);
        let mut cmds = commands(200, &original);
        let distances: Vec<u32> = cmds
            .iter()
            .filter(|c| c.has_distance())
            .map(|c| c.restore_distance_code(&original))
            .collect();

        recompute_distance_prefixes(&mut cmds, &original, &new);

        let after: Vec<u32> = cmds
            .iter()
            .filter(|c| c.has_distance())
            .map(|c| c.restore_distance_code(&new))
            .collect();
        assert_eq!(distances, after);
    }

    #[test]
    fn an_unrepresentable_distance_rules_a_candidate_out() {
        // More postfix bits *widen* the alphabet, so the narrowest candidate is
        // the plain one; a distance past its ceiling is what no candidate can
        // express. RFC 7932 window sizes never produce such a distance, so this
        // guard is unreachable in practice — but it is the reference's, and
        // getting it wrong would silently accept an invalid alphabet.
        let wide = DistanceParams::new(3, 120);
        let narrow = DistanceParams::default();
        assert!(narrow.max_distance < wide.max_distance);

        let mut tmp = HistogramDistance::default();
        let far = vec![Command::new(
            &wide,
            0,
            10,
            0,
            narrow.max_distance as usize + 1,
        )];
        assert!(compute_distance_cost(&far, &wide, &narrow, &mut tmp).is_none());
        // The same command under an alphabet that can hold it prices fine.
        assert!(compute_distance_cost(&far, &wide, &wide, &mut tmp).is_some());
    }

    #[test]
    fn the_chosen_alphabet_can_express_every_distance() {
        let original = DistanceParams::default();
        let mut tmp = HistogramDistance::default();
        let cmds = commands(500, &original);
        let chosen = choose_distance_params(&cmds, false, original, &mut tmp);
        for command in &cmds {
            if command.has_distance() {
                assert!(command.restore_distance_code(&original) <= chosen.max_distance);
            }
        }
        assert!(chosen.postfix_bits <= MAX_NPOSTFIX);
    }

    #[test]
    fn the_chosen_alphabet_never_costs_more_than_the_original() {
        let original = DistanceParams::default();
        let mut tmp = HistogramDistance::default();
        for cmds in [
            // One distance repeated: direct codes make its extra bits vanish.
            (0..300)
                .map(|_| Command::new(&original, 4, 12, 0, 20))
                .collect::<Vec<_>>(),
            // A spread of distances: the plain alphabet is hard to beat.
            commands(500, &original),
        ] {
            let chosen = choose_distance_params(&cmds, false, original, &mut tmp);
            let before = compute_distance_cost(&cmds, &original, &original, &mut tmp)
                .expect("the original alphabet always prices");
            let after = compute_distance_cost(&cmds, &original, &chosen, &mut tmp)
                .expect("the chosen alphabet always prices");
            assert!(
                after <= before,
                "chose a dearer alphabet: {after} vs {before}"
            );
        }
    }

    #[test]
    fn a_repeated_distance_earns_direct_codes() {
        // Every command at the same distance: a direct code spends no extra
        // bits on it at all, which is exactly what the search is looking for.
        let original = DistanceParams::default();
        let mut tmp = HistogramDistance::default();
        let cmds: Vec<Command> = (0..300)
            .map(|_| Command::new(&original, 4, 12, 0, 20))
            .collect();
        let chosen = choose_distance_params(&cmds, false, original, &mut tmp);
        assert!(chosen.num_direct > 0, "no direct codes were chosen");
    }

    /// Builds a meta-block over `data` and returns it with its alphabet.
    fn build(
        data: &[u8],
        cmds: &mut [Command],
        params: &HqParams,
        mode: ContextMode,
    ) -> (MetaBlockSplit, DistanceParams) {
        let mut builder = MetaBlockBuilder::default();
        let mut mb = MetaBlockSplit::default();
        let mut dist = params.dist;
        builder.build(
            &*crate::compressor::core::dispatch::select(fearless_simd::Level::fallback()),
            data,
            0,
            usize::MAX,
            params,
            0,
            0,
            cmds,
            mode,
            &mut dist,
            &mut mb,
        );
        (mb, dist)
    }

    /// Builds a fixture of exactly the length its commands consume.
    type Fixture = Box<dyn Fn(usize) -> Vec<u8>>;

    /// What the meta-block builder decided, in a form both sides can report.
    #[derive(Debug, Eq, PartialEq)]
    struct Shape {
        npostfix: u32,
        ndirect: u32,
        literal_types: usize,
        command_types: usize,
        distance_types: usize,
        literal_histograms: usize,
        command_histograms: usize,
        distance_histograms: usize,
        literal_context_map: Vec<u32>,
        distance_context_map: Vec<u32>,
    }

    /// Runs the C builder over the same commands, returning its decisions.
    fn c_shape(
        quality: i32,
        commands: &mut [Command],
        data: &[u8],
        params: &HqParams,
        context_mode: ContextMode,
    ) -> Shape {
        let capacity = 256 << LITERAL_CONTEXT_BITS;
        let mut literal_context_map = vec![0u32; capacity];
        let mut distance_context_map = vec![0u32; capacity];
        let mut out = [0usize; 6];
        let mut sizes = [0usize; 2];
        let mut npostfix = 0u32;
        let mut ndirect = 0u32;
        let mut data = data.to_vec();

        // SAFETY: every pointer is valid for the length it is passed with, the
        // command array has the layout the shim documents, and the fixture is
        // long enough for every index the commands reach.
        unsafe {
            google_brotli_ffi::mbrotli_shim_build_meta_block(
                quality,
                22,
                match context_mode {
                    ContextMode::Utf8 => 2,
                    ContextMode::Signed => 3,
                },
                i32::from(params.disable_literal_context_modeling),
                data.as_mut_ptr(),
                0,
                usize::MAX,
                0,
                0,
                commands.as_mut_ptr().cast::<u8>(),
                commands.len(),
                capacity,
                &raw mut npostfix,
                &raw mut ndirect,
                &raw mut out[0],
                &raw mut out[1],
                &raw mut out[2],
                &raw mut out[3],
                &raw mut out[4],
                &raw mut out[5],
                literal_context_map.as_mut_ptr(),
                &raw mut sizes[0],
                distance_context_map.as_mut_ptr(),
                &raw mut sizes[1],
            );
        }
        Shape {
            npostfix,
            ndirect,
            literal_types: out[0],
            command_types: out[1],
            distance_types: out[2],
            literal_histograms: out[3],
            command_histograms: out[4],
            distance_histograms: out[5],
            literal_context_map: literal_context_map[..sizes[0]].to_vec(),
            distance_context_map: distance_context_map[..sizes[1]].to_vec(),
        }
    }

    /// Compares both builders over one command stream.
    fn assert_shape_matches_c(
        name: &str,
        quality: QualityLevel,
        commands: &[Command],
        data: &[u8],
        params: &HqParams,
        context_mode: ContextMode,
    ) {
        let consumed: usize = commands
            .iter()
            .map(|c| c.insert_len as usize + c.copy_len() as usize)
            .sum();
        assert!(
            consumed <= data.len(),
            "case {name}: the commands consume {consumed} bytes of a {}-byte fixture",
            data.len()
        );

        let mut ours_commands = commands.to_vec();
        let mut builder = MetaBlockBuilder::default();
        let mut mb = MetaBlockSplit::default();
        let mut dist = params.dist;
        builder.build(
            &*crate::compressor::core::dispatch::select(fearless_simd::Level::fallback()),
            data,
            0,
            usize::MAX,
            params,
            0,
            0,
            &mut ours_commands,
            context_mode,
            &mut dist,
            &mut mb,
        );
        let ours = Shape {
            npostfix: dist.postfix_bits,
            ndirect: dist.num_direct,
            literal_types: mb.literal_split.num_types,
            command_types: mb.command_split.num_types,
            distance_types: mb.distance_split.num_types,
            literal_histograms: mb.literal_histograms.len(),
            command_histograms: mb.command_histograms.len(),
            distance_histograms: mb.distance_histograms.len(),
            literal_context_map: mb.literal_context_map.clone(),
            distance_context_map: mb.distance_context_map.clone(),
        };

        let mut theirs_commands = commands.to_vec();
        let theirs = c_shape(
            usize::from(quality) as i32,
            &mut theirs_commands,
            data,
            params,
            context_mode,
        );

        assert_eq!(
            ours_commands, theirs_commands,
            "case {name}, quality {quality:?}: the rewritten commands differ"
        );
        assert_eq!(ours, theirs, "case {name}, quality {quality:?}");
    }

    #[test]
    fn every_context_map_entry_indexes_a_histogram() {
        let params = params(CompressMode::Generic);
        let data: Vec<u8> = (0..60_000u32).map(|i| (i * 13 % 251) as u8).collect();
        let mut cmds = commands(3000, &params.dist);
        let (mb, _) = build(&data, &mut cmds, &params, ContextMode::Utf8);

        assert_eq!(
            mb.literal_context_map.len(),
            mb.literal_split.num_types << LITERAL_CONTEXT_BITS
        );
        assert!(
            mb.literal_context_map
                .iter()
                .all(|&index| (index as usize) < mb.literal_histograms.len())
        );
        assert_eq!(
            mb.distance_context_map.len(),
            mb.distance_split.num_types << DISTANCE_CONTEXT_BITS
        );
        assert!(
            mb.distance_context_map
                .iter()
                .all(|&index| (index as usize) < mb.distance_histograms.len())
        );
        assert_eq!(mb.command_histograms.len(), mb.command_split.num_types);
    }

    #[test]
    fn the_histograms_account_for_every_symbol() {
        let params = params(CompressMode::Generic);
        let data: Vec<u8> = (0..60_000u32).map(|i| (i % 200) as u8).collect();
        let mut cmds = commands(2000, &params.dist);
        let literals: usize = cmds.iter().map(|c| c.insert_len as usize).sum();
        let distances = cmds.iter().filter(|c| c.has_distance()).count();
        let (mb, _) = build(&data, &mut cmds, &params, ContextMode::Utf8);

        let counted: usize = mb.literal_histograms.iter().map(|h| h.total_count).sum();
        assert_eq!(counted, literals);
        let counted: usize = mb.command_histograms.iter().map(|h| h.total_count).sum();
        assert_eq!(counted, cmds.len());
        let counted: usize = mb.distance_histograms.iter().map(|h| h.total_count).sum();
        assert_eq!(counted, distances);
    }

    #[test]
    fn the_histogram_count_stays_inside_the_format_limit() {
        let params = params(CompressMode::Generic);
        let mut rng = 0x0FED_CBA9_8765_4321u64;
        let data: Vec<u8> = (0..80_000u32)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng >> 24) as u8
            })
            .collect();
        let mut cmds = commands(4000, &params.dist);
        let (mb, _) = build(&data, &mut cmds, &params, ContextMode::Utf8);
        assert!(mb.literal_histograms.len() <= MAX_NUMBER_OF_HISTOGRAMS);
        assert!(mb.distance_histograms.len() <= MAX_NUMBER_OF_HISTOGRAMS);
        assert!(mb.command_histograms.len() <= MAX_NUMBER_OF_HISTOGRAMS);
    }

    #[test]
    fn disabling_context_modelling_gives_every_context_one_histogram() {
        let public = CompressParams::new(QualityLevel::Q11, WindowBits::DEFAULT)
            .with_literal_context_modeling(false);
        let params = HqParams::new(&public).expect("supported quality");
        let data: Vec<u8> = (0..40_000u32).map(|i| (i % 128) as u8).collect();
        let mut cmds = commands(1500, &params.dist);
        let (mb, _) = build(&data, &mut cmds, &params, ContextMode::Utf8);

        // Within one block type every context maps to the same histogram.
        for block_type in 0..mb.literal_split.num_types {
            let base = block_type << LITERAL_CONTEXT_BITS;
            let first = mb.literal_context_map[base];
            assert!(
                mb.literal_context_map[base..base + (1 << LITERAL_CONTEXT_BITS)]
                    .iter()
                    .all(|&index| index == first),
                "block type {block_type} split its contexts"
            );
        }
    }

    #[test]
    fn the_two_context_modes_produce_different_histograms() {
        let params = params(CompressMode::Generic);
        let data: Vec<u8> = (0..40_000u32).map(|i| (i * 3 % 251) as u8).collect();
        let mut utf8_cmds = commands(1200, &params.dist);
        let mut signed_cmds = utf8_cmds.clone();
        let (utf8, _) = build(&data, &mut utf8_cmds, &params, ContextMode::Utf8);
        let (signed, _) = build(&data, &mut signed_cmds, &params, ContextMode::Signed);
        assert_ne!(utf8.literal_context_map, signed.literal_context_map);
    }

    #[test]
    fn a_reused_builder_produces_the_same_meta_block() {
        let params = params(CompressMode::Generic);
        let data: Vec<u8> = (0..50_000u32).map(|i| (i * 5 % 241) as u8).collect();
        let other: Vec<u8> = (0..20_000u32).map(|i| (i % 97) as u8).collect();

        let mut once_cmds = commands(1800, &params.dist);
        let (expected, expected_dist) = build(&data, &mut once_cmds, &params, ContextMode::Utf8);

        let mut builder = MetaBlockBuilder::default();
        let mut warm_cmds = commands(700, &params.dist);
        let mut warm = MetaBlockSplit::default();
        let mut warm_dist = params.dist;
        builder.build(
            &*crate::compressor::core::dispatch::select(fearless_simd::Level::fallback()),
            &other,
            0,
            usize::MAX,
            &params,
            0,
            0,
            &mut warm_cmds,
            ContextMode::Signed,
            &mut warm_dist,
            &mut warm,
        );

        let mut cmds = commands(1800, &params.dist);
        let mut mb = MetaBlockSplit::default();
        let mut dist = params.dist;
        builder.build(
            &*crate::compressor::core::dispatch::select(fearless_simd::Level::fallback()),
            &data,
            0,
            usize::MAX,
            &params,
            0,
            0,
            &mut cmds,
            ContextMode::Utf8,
            &mut dist,
            &mut mb,
        );

        assert_eq!(dist, expected_dist);
        assert_eq!(cmds, once_cmds);
        assert_eq!(mb.literal_context_map, expected.literal_context_map);
        assert_eq!(mb.distance_context_map, expected.distance_context_map);
        assert_eq!(mb.literal_split.types, expected.literal_split.types);
        assert_eq!(mb.command_split.lengths, expected.command_split.lengths);
    }

    #[test]
    fn every_meta_block_matches_the_c_builder() {
        let dist = DistanceParams::default();
        let cases: Vec<(&str, Fixture, Vec<Command>)> = vec![
            (
                "uniform",
                Box::new(|n| vec![b'a'; n]),
                (0..2000)
                    .map(|_| Command::new(&dist, 8, 12, 0, 40))
                    .collect(),
            ),
            (
                "spread-distances",
                Box::new(|n| (0..n as u32).map(|i| (i % 251) as u8).collect()),
                commands(2000, &dist),
            ),
            (
                "text-like",
                Box::new(|n| {
                    let mut data = Vec::with_capacity(n);
                    while data.len() < n {
                        data.extend_from_slice(b"the quick brown fox jumps over the lazy dog. ");
                    }
                    data.truncate(n);
                    data
                }),
                (0..1500)
                    .map(|index| Command::new(&dist, 14, 5 + index % 20, 0, 18 + index % 700))
                    .collect(),
            ),
            (
                "literal-only",
                Box::new(|n| (0..n as u32).map(|i| (i * 3 % 199) as u8).collect()),
                (0..600).map(|_| Command::insert_only(60)).collect(),
            ),
            (
                "tiny",
                Box::new(|n| vec![b'q'; n]),
                (0..15).map(|_| Command::new(&dist, 5, 7, 0, 22)).collect(),
            ),
        ];

        for (name, make_data, cmds) in cases {
            let data = make_data(
                cmds.iter()
                    .map(|c| c.insert_len as usize + c.copy_len() as usize)
                    .sum(),
            );
            for quality in [QualityLevel::Q10, QualityLevel::Q11] {
                for context in [ContextMode::Utf8, ContextMode::Signed] {
                    for modelling in [true, false] {
                        let public = CompressParams::new(quality, WindowBits::DEFAULT)
                            .with_literal_context_modeling(modelling);
                        let resolved = HqParams::new(&public).expect("supported quality");
                        assert_shape_matches_c(name, quality, &cmds, &data, &resolved, context);
                    }
                }
            }
        }
    }
}

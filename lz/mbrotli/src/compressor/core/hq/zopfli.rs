//! The dynamic program that chooses qualities ten and eleven's commands.
//!
//! Ports `c/enc/backward_references_hq.c` from the pinned reference
//! (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! The greedy encoder decides one command at a time and never reconsiders. This
//! one prices every command that could start at every position and then takes
//! the cheapest path through the whole block. Three things make that tractable:
//! only the eight cheapest command starts are ever expanded, a copy shorter
//! than one already known to reach a position is skipped outright, and a very
//! long copy lets the search stride past the positions it covers.
//!
//! Every comparison below is a strict `<` on an `f32`, so the arithmetic in
//! [`super::cost`] is as much a part of the output as the decisions here are.

use alloc::vec::Vec;

use fearless_simd::Simd;

use super::cost::ZopfliCostModel;
use super::h10::{BackwardMatch, BinaryTreeMatcher, HASH_TYPE_LENGTH, STORE_LOOKAHEAD};
use super::nodes::{INFINITY, PosData, StartPosQueue, ZopfliNode};
use super::params::HqParams;
use crate::compressor::core::rfc9841::context::SharedContextInner;
use crate::shared::command::{
    Command, copy_length_code, insert_length_code, prefix_encode_copy_distance,
};
use crate::shared::distance::NUM_DISTANCE_SHORT_CODES;
use crate::shared::format::{COPY_EXTRA, INS_EXTRA};
use crate::shared::match_len::find_match_length;

/// Copy length past which the search strides rather than examining every byte
/// (`BROTLI_LONG_COPY_QUICK_STEP`).
const LONG_COPY_QUICK_STEP: usize = 16_384;

/// One short distance code's probe of the distance cache.
#[derive(Copy, Clone)]
struct CacheProbe {
    /// Which of the four cached distances the code reads.
    slot: u8,
    /// What it adds to that distance.
    offset: i8,
}

impl CacheProbe {
    /// Returns the distance this probe reads from `cache`.
    #[inline(always)]
    fn distance(self, cache: &[i32; 4]) -> usize {
        // The slot is always below four; the mask only spares a range check.
        (cache[usize::from(self.slot) & 3] + i32::from(self.offset)) as usize
    }
}

/// How each of the sixteen short distance codes reads the distance cache
/// (`kDistanceCacheIndex` and `kDistanceCacheOffset`, kept together so the
/// loop over them loads one byte pair per code).
///
/// The first four codes are the cached distances themselves, the next six are
/// the freshest one nudged by up to three either way, and the last six do the
/// same to the second freshest.
const CACHE_PROBES: [CacheProbe; NUM_DISTANCE_SHORT_CODES as usize] = [
    CacheProbe { slot: 0, offset: 0 },
    CacheProbe { slot: 1, offset: 0 },
    CacheProbe { slot: 2, offset: 0 },
    CacheProbe { slot: 3, offset: 0 },
    CacheProbe {
        slot: 0,
        offset: -1,
    },
    CacheProbe { slot: 0, offset: 1 },
    CacheProbe {
        slot: 0,
        offset: -2,
    },
    CacheProbe { slot: 0, offset: 2 },
    CacheProbe {
        slot: 0,
        offset: -3,
    },
    CacheProbe { slot: 0, offset: 3 },
    CacheProbe {
        slot: 1,
        offset: -1,
    },
    CacheProbe { slot: 1, offset: 1 },
    CacheProbe {
        slot: 1,
        offset: -2,
    },
    CacheProbe { slot: 1, offset: 2 },
    CacheProbe {
        slot: 1,
        offset: -3,
    },
    CacheProbe { slot: 1, offset: 3 },
];

/// The mutable encoder state a Zopfli pass consumes and updates.
pub(crate) struct ZopfliState {
    /// The four distances that have short codes.
    pub(crate) dist_cache: [i32; 4],
    /// Literals produced but not yet attached to a command.
    pub(crate) last_insert_len: usize,
    /// Literals in the commands emitted for the current meta-block.
    pub(crate) num_literals: usize,
}

impl Default for ZopfliState {
    /// Returns the state a fresh stream starts from.
    fn default() -> Self {
        Self {
            dist_cache: [4, 11, 15, 16],
            last_insert_len: 0,
            num_literals: 0,
        }
    }
}

/// Reusable storage for one stream's worth of Zopfli passes.
///
/// Allocated once and grown only when a larger block arrives, so no position
/// and no dynamic-programming edge ever allocates.
pub(crate) struct ZopfliWorkspace {
    nodes: Vec<ZopfliNode>,
    queue: StartPosQueue,
    /// Matches at the position being examined, for quality ten.
    scratch: Vec<BackwardMatch>,
    /// Bounded attached-prefix candidates reused across positions and streams.
    prefix_matches: Vec<(usize, usize)>,
    /// Every position's matches end to end, for quality eleven.
    arena: Vec<BackwardMatch>,
    /// How many of [`ZopfliWorkspace::arena`] belong to each position.
    num_matches: Vec<u32>,
    model: ZopfliCostModel,
}

impl ZopfliWorkspace {
    /// Creates a workspace whose buffers are sized per block by `prepare`.
    pub(crate) fn new(alphabet_size: usize) -> Self {
        Self {
            nodes: Vec::new(),
            queue: StartPosQueue::default(),
            scratch: Vec::new(),
            prefix_matches: Vec::new(),
            arena: Vec::new(),
            num_matches: Vec::new(),
            model: ZopfliCostModel::new(alphabet_size),
        }
    }

    /// Returns the bytes this workspace keeps allocated.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.nodes.capacity() * size_of::<ZopfliNode>()
            + (self.scratch.capacity() + self.arena.capacity()) * size_of::<BackwardMatch>()
            + self.num_matches.capacity() * size_of::<u32>()
            + self.model.retained_bytes()
            + self.prefix_matches.capacity() * size_of::<(usize, usize)>()
    }

    /// Sizes every buffer for a block of `num_bytes` and clears the nodes.
    fn prepare(&mut self, num_bytes: usize, alphabet_size: usize) {
        self.nodes.clear();
        self.nodes.resize(num_bytes + 1, ZopfliNode::default());
        self.queue.clear();
        self.scratch.clear();
        self.model.reserve(num_bytes, alphabet_size);
    }

    /// Resets the nodes to the stub without touching anything else.
    fn reset_nodes(&mut self, num_bytes: usize) {
        self.nodes.clear();
        self.nodes.resize(num_bytes + 1, ZopfliNode::default());
    }
}

/// Returns the shortest copy that could still improve a later position.
///
/// Mirrors `ComputeMinimumCopyLength`. Anything shorter reaches a position that
/// is already cheaper to arrive at than the best case here, so it can be skipped
/// without pricing it. The cost floor rises by a bit at each copy-length code
/// bucket, because a longer copy needs an extra length bit.
fn compute_minimum_copy_length(
    start_cost: f32,
    nodes: &[ZopfliNode],
    num_bytes: usize,
    pos: usize,
) -> usize {
    let mut min_cost = start_cost;
    let mut len = 2usize;
    let mut next_len_bucket = 4usize;
    let mut next_len_offset = 10usize;
    while pos + len <= num_bytes && nodes[pos + len].cost() <= min_cost {
        len += 1;
        if len == next_len_offset {
            min_cost += 1.0;
            next_len_offset += next_len_bucket;
            next_len_bucket *= 2;
        }
    }
    len
}

/// Returns the position whose command supplies the next cached distance.
///
/// Mirrors `ComputeDistanceShortcut`. A command that does not update the
/// distance cache — a dictionary reference, or one coded with the last distance
/// — is transparent: the shortcut points straight past it to whatever did.
fn compute_distance_shortcut(
    block_start: usize,
    pos: usize,
    max_backward_limit: usize,
    gap: usize,
    nodes: &[ZopfliNode],
) -> u32 {
    let node = &nodes[pos];
    let c_len = node.copy_length() as usize;
    let i_len = node.insert_length() as usize;
    let dist = node.copy_distance() as usize;
    if pos == 0 {
        0
    } else if dist + c_len <= block_start + pos + gap
        && dist <= max_backward_limit + gap
        && node.distance_code() > 0
    {
        pos as u32
    } else {
        nodes[pos - c_len - i_len].shortcut()
    }
}

/// Fills `dist_cache` with the four distances in effect at `pos`.
///
/// Mirrors `ComputeDistanceCache`, walking the shortcut chain backwards until
/// it has four distances or runs out of block, then falling back on the cache
/// the block started with.
fn compute_distance_cache(
    pos: usize,
    starting_dist_cache: &[i32; 4],
    nodes: &[ZopfliNode],
    dist_cache: &mut [i32; 4],
) {
    let mut idx = 0usize;
    let mut p = nodes[pos].shortcut() as usize;
    while idx < 4 && p > 0 {
        let node = &nodes[p];
        let i_len = node.insert_length() as usize;
        let c_len = node.copy_length() as usize;
        dist_cache[idx] = node.copy_distance() as i32;
        idx += 1;
        // The shortcut invariant guarantees `p >= c_len + i_len >= 2`.
        p = nodes[p - c_len - i_len].shortcut() as usize;
    }
    // The fallback reads the saved cache from *its* beginning, not from the
    // slot being filled: whatever the chain could not supply is taken in order
    // from the distances the block started with. Indexing both sides in
    // parallel would put the wrong distance behind every short code.
    let mut from = 0usize;
    while idx < 4 {
        dist_cache[idx] = starting_dist_cache[from];
        idx += 1;
        from += 1;
    }
}

/// Records `pos`'s distance shortcut and queues it if it is worth starting from.
///
/// Mirrors `EvaluateNode`. Reading the cost has to happen first: computing the
/// shortcut overwrites it, since the two share a word.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors EvaluateNode, whose parameters are all needed"
)]
fn evaluate_node(
    block_start: usize,
    pos: usize,
    max_backward_limit: usize,
    gap: usize,
    starting_dist_cache: &[i32; 4],
    model: &ZopfliCostModel,
    queue: &mut StartPosQueue,
    nodes: &mut [ZopfliNode],
) {
    let node_cost = nodes[pos].cost();
    let shortcut = compute_distance_shortcut(block_start, pos, max_backward_limit, gap, nodes);
    nodes[pos].set_shortcut(shortcut);
    // Only a position cheaper than spelling everything out as literals is worth
    // starting a command from.
    if node_cost <= model.literal_costs(0, pos) {
        let mut posdata = PosData {
            pos,
            distance_cache: [0; 4],
            cost: node_cost,
            costdiff: node_cost - model.literal_costs(0, pos),
        };
        compute_distance_cache(pos, starting_dist_cache, nodes, &mut posdata.distance_cache);
        queue.push(posdata);
    }
}

/// Everything `update_nodes` needs that does not change between positions.
struct UpdateContext<'a> {
    num_bytes: usize,
    block_start: usize,
    ringbuffer: &'a [u8],
    ringbuffer_mask: usize,
    params: &'a HqParams,
    max_backward_limit: usize,
    starting_dist_cache: &'a [i32; 4],
    /// Total attached prefix bytes; the reference's `gap`, zero without one.
    gap: usize,
    /// The attached prefix, when a cached distance may address it.
    attached: Option<&'a SharedContextInner>,
}

/// Measures a cached distance that reaches past the window, or returns `None`
/// when the reference would not price it.
///
/// The cold arm of `UpdateNodes`'s distance-cache loop: a distance beyond the
/// window is either a static-dictionary reference, which the match list
/// already covers, a copy from the attached prefix, or in the "gray" area a
/// decoder could address but this encoder does not hold. Only the prefix copy
/// is measured, and it stops at the end of the attachment it started in,
/// exactly as the reference's `limit` does.
#[cold]
#[inline(never)]
fn prefix_copy_length(
    ctx: &UpdateContext<'_>,
    backward: usize,
    dictionary_start: usize,
    best_len: usize,
    continuation: u8,
    cur_ix_masked: usize,
    max_len: usize,
) -> Option<usize> {
    let gap = ctx.gap;
    if backward > dictionary_start + gap || backward <= dictionary_start {
        return None;
    }
    let sources = ctx.attached?.dictionaries().prefix();
    let logical = (dictionary_start + gap - backward) as u64;
    let (segment, offset) = sources.locate(logical)?;
    let candidate = sources.segment(segment).get(offset..)?;
    let limit = candidate.len().min(max_len);
    if best_len >= limit || candidate.get(best_len) != Some(&continuation) {
        return None;
    }
    let target = ctx.ringbuffer.get(cur_ix_masked..)?;
    Some(prefix_match_length(candidate, target, limit))
}

/// Prices every command that could start at `pos`, returning the longest copy.
///
/// Mirrors `UpdateNodes`. The two halves are the reference's: first the copies
/// reachable through the distance cache, from each of the queued start
/// positions in turn, then the matches the tree found — but only from the two
/// cheapest starts, because a further start with the same distances rarely
/// pays.
fn update_nodes<S: Simd, const INDEPENDENT: bool>(
    simd: S,
    ctx: &UpdateContext<'_>,
    pos: usize,
    matches: &[BackwardMatch],
    model: &ZopfliCostModel,
    queue: &mut StartPosQueue,
    nodes: &mut [ZopfliNode],
) -> usize {
    simd.vectorize(
        #[inline(always)]
        || {
            let cur_ix = ctx.block_start + pos;
            let cur_ix_masked = cur_ix & ctx.ringbuffer_mask;
            let max_distance = cur_ix.min(ctx.max_backward_limit);
            let gap = ctx.gap;
            let dictionary_start = ctx
                .params
                .logical_position(cur_ix)
                .min(ctx.max_backward_limit);
            let max_len = ctx.num_bytes - pos;
            // The addressable ring: a probe past its mask is the reference's
            // out-of-window test, and a shorter buffer is never probed past
            // what it holds.
            let window = ctx
                .ringbuffer
                .get(..=ctx.ringbuffer_mask)
                .unwrap_or(ctx.ringbuffer);
            let max_zopfli_len = ctx.params.max_zopfli_len();
            let max_iters = ctx.params.max_zopfli_candidates();
            let mut result = 0usize;

            evaluate_node(
                ctx.params.logical_position(ctx.block_start),
                pos,
                ctx.max_backward_limit,
                gap,
                ctx.starting_dist_cache,
                model,
                queue,
                nodes,
            );

            let min_len = {
                let posdata = queue.at(0);
                let min_cost =
                    posdata.cost + model.min_cost_cmd() + model.literal_costs(posdata.pos, pos);
                compute_minimum_copy_length(min_cost, nodes, ctx.num_bytes, pos)
            };

            let mut command_scratch = [INFINITY; 32];
            // Command starts in order of increasing cost difference.
            for k in 0..max_iters.min(queue.len()) {
                let (start, start_costdiff, cache) = {
                    let posdata = queue.at(k);
                    (posdata.pos, posdata.costdiff, posdata.distance_cache)
                };
                let inscode = insert_length_code(pos - start);
                let base_cost = start_costdiff
                    + INS_EXTRA[usize::from(inscode)] as f32
                    + model.literal_costs(0, pos);

                // Copies reachable through this start position's distance cache.
                let mut best_len = min_len - 1;
                // Indexed rather than iterated: the probe index is also the
                // short distance code, and a counted loop keeps the body to a
                // register's worth of state.
                let mut j = 0usize;
                while j < CACHE_PROBES.len() {
                    // An independent fragment carries no distance cache.
                    if INDEPENDENT || best_len >= max_len {
                        break;
                    }
                    let code = j;
                    j += 1;
                    // The byte a longer copy would have to agree on; a copy
                    // that would run off the ring ends the search.
                    let Some(&continuation) = window.get(cur_ix_masked + best_len) else {
                        break;
                    };
                    let backward = CACHE_PROBES[code].distance(&cache);
                    // One unsigned compare admits exactly `1..=max_distance`:
                    // a zero distance wraps below it and a negative one wraps
                    // above, and every ordinary window distance is also
                    // within the dictionary bound the reference tests first.
                    let found = if backward.wrapping_sub(1) < max_distance {
                        // An ordinary backward reference into the window,
                        // measured only when it can beat `best_len`.
                        let prev_ix = (cur_ix - backward) & ctx.ringbuffer_mask;
                        (window.get(prev_ix + best_len) == Some(&continuation)).then(|| {
                            find_match_length(simd, ctx.ringbuffer, prev_ix, cur_ix_masked, max_len)
                        })
                    } else {
                        prefix_copy_length(
                            ctx,
                            backward,
                            dictionary_start,
                            best_len,
                            continuation,
                            cur_ix_masked,
                            max_len,
                        )
                    };
                    let Some(len) = found else {
                        continue;
                    };

                    // The continuation byte is only a filter: earlier bytes
                    // can disagree, leaving no newly reachable copy length.
                    if len <= best_len {
                        continue;
                    }

                    let dist_cost = base_cost + model.distance_cost(code);
                    // Price every length the copy newly reaches. The node array
                    // extends one past the block, so the range is always inside
                    // it; an empty one prices nothing.
                    let command_costs =
                        model.command_costs(inscode, code == 0, &mut command_scratch);
                    let reached = nodes
                        .get_mut(pos + best_len + 1..=pos + len)
                        .unwrap_or_default();
                    for (l, node) in (best_len + 1..).zip(reached) {
                        let copycode = copy_length_code(l);
                        let cost = if code == 0 && inscode < 8 && copycode < 16 {
                            base_cost
                        } else {
                            dist_cost
                        } + COPY_EXTRA[usize::from(copycode)] as f32
                            + command_costs[usize::from(copycode) & 31];
                        if cost < node.cost() {
                            node.record(pos, start, l, l, backward, code + 1, cost);
                            result = result.max(l);
                        }
                    }
                    best_len = best_len.max(len);
                }

                // Beyond the second start position only new cached distances help, and
                // those have just been tried.
                if k >= 2 {
                    continue;
                }

                let mut len = min_len;
                for m in matches {
                    let dist = m.distance as usize;
                    let is_dictionary_match = dist > dictionary_start + gap;
                    // Every cached distance has been tried already, so these are always
                    // written with a full distance code.
                    let dist_code = dist + NUM_DISTANCE_SHORT_CODES as usize - 1;
                    let (dist_symbol, _) = prefix_encode_copy_distance(
                        dist_code,
                        ctx.params.dist.num_direct,
                        ctx.params.dist.postfix_bits,
                    );
                    let distnumextra = u32::from(dist_symbol >> 10);
                    let dist_cost = base_cost
                        + distnumextra as f32
                        + model.distance_cost(usize::from(dist_symbol & 0x3FF));

                    // Try every copy length up to this match's. A dictionary word has
                    // only one meaningful length, and a very long copy is not worth
                    // examining shorter prefixes of.
                    let max_match_len = m.length();
                    if len < max_match_len
                        && (is_dictionary_match || max_match_len > max_zopfli_len)
                    {
                        len = max_match_len;
                    }
                    if len <= max_match_len {
                        let command_costs =
                            model.command_costs(inscode, false, &mut command_scratch);
                        let reached = nodes
                            .get_mut(pos + len..=pos + max_match_len)
                            .unwrap_or_default();
                        for (l, node) in (len..).zip(reached) {
                            let len_code = if is_dictionary_match {
                                m.length_code()
                            } else {
                                l
                            };
                            let copycode = copy_length_code(len_code);
                            let cost = dist_cost
                                + COPY_EXTRA[usize::from(copycode)] as f32
                                + command_costs[usize::from(copycode) & 31];
                            if cost < node.cost() {
                                node.record(pos, start, l, len_code, dist, 0, cost);
                                result = result.max(l);
                            }
                        }
                        len = max_match_len + 1;
                    }
                }
            }
            result
        },
    )
}

/// Turns the finished node array into a forward chain of commands.
///
/// Mirrors `ComputeShortestPathFromNodes`: walk back from the end along the
/// command that reached each node, writing into each node the length of the
/// command that follows it.
fn compute_shortest_path_from_nodes(num_bytes: usize, nodes: &mut [ZopfliNode]) -> usize {
    let mut index = num_bytes;
    let mut num_commands = 0usize;
    // Trailing literals are not a command; back up over them.
    while nodes[index].insert_length() == 0 && nodes[index].length == 1 {
        index -= 1;
    }
    nodes[index].set_next(u32::MAX);
    while index != 0 {
        let len = nodes[index].command_length() as usize;
        index -= len;
        nodes[index].set_next(len as u32);
        num_commands += 1;
    }
    num_commands
}

/// Walks the chosen path forward, emitting commands.
///
/// Mirrors `BrotliZopfliCreateCommands`, including its distance-cache update:
/// a dictionary reference and a copy coded with the last distance both leave
/// the cache alone.
fn create_commands(
    num_bytes: usize,
    block_start: usize,
    nodes: &[ZopfliNode],
    params: &HqParams,
    gap: usize,
    state: &mut ZopfliState,
    commands: &mut Vec<Command>,
) {
    let max_backward_limit = params.max_backward_limit();
    let mut pos = 0usize;
    let mut offset = nodes[0].next();
    let mut first = true;

    while offset != u32::MAX {
        let next = &nodes[pos + offset as usize];
        let copy_length = next.copy_length() as usize;
        let mut insert_length = next.insert_length() as usize;
        pos += insert_length;
        offset = next.next();
        if first {
            insert_length += state.last_insert_len;
            state.last_insert_len = 0;
            first = false;
        }

        let distance = next.copy_distance() as usize;
        let len_code = next.length_code() as usize;
        let dictionary_start = params
            .logical_position(block_start + pos)
            .min(max_backward_limit);
        let is_dictionary = distance > dictionary_start + gap;
        let dist_code = next.distance_code() as usize;

        commands.push(Command::new(
            &params.dist,
            insert_length,
            copy_length,
            len_code as i32 - copy_length as i32,
            dist_code,
        ));

        if !is_dictionary && dist_code > 0 {
            state.dist_cache[3] = state.dist_cache[2];
            state.dist_cache[2] = state.dist_cache[1];
            state.dist_cache[1] = state.dist_cache[0];
            state.dist_cache[0] = distance as i32;
        }

        state.num_literals += insert_length;
        pos += copy_length;
    }
    state.last_insert_len += num_bytes - pos;
}

/// Runs the dynamic program over precomputed matches (`ZopfliIterate`).
///
/// Quality eleven's inner loop: every position's matches are already in the
/// arena, so this only prices them.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors ZopfliIterate, whose parameters are all needed"
)]
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
fn zopfli_iterate<S: Simd, const INDEPENDENT: bool>(
    simd: S,
    num_bytes: usize,
    position: usize,
    ringbuffer: &[u8],
    ringbuffer_mask: usize,
    params: &HqParams,
    gap: usize,
    attached: Option<&SharedContextInner>,
    starting_dist_cache: &[i32; 4],
    model: &ZopfliCostModel,
    num_matches: &[u32],
    arena: &[BackwardMatch],
    queue: &mut StartPosQueue,
    nodes: &mut [ZopfliNode],
) -> usize {
    let ctx = UpdateContext {
        num_bytes,
        block_start: position,
        ringbuffer,
        ringbuffer_mask,
        params,
        max_backward_limit: params.max_backward_limit(),
        starting_dist_cache,
        gap,
        attached,
    };
    let max_zopfli_len = params.max_zopfli_len();

    nodes[0].length = 0;
    nodes[0].set_cost(0.0);
    queue.clear();

    let mut cur_match_pos = 0usize;
    let mut i = 0usize;
    while i + 3 < num_bytes {
        let count = num_matches[i] as usize;
        let matches = &arena[cur_match_pos..cur_match_pos + count];
        let mut skip = update_nodes::<_, INDEPENDENT>(simd, &ctx, i, matches, model, queue, nodes);
        if skip < LONG_COPY_QUICK_STEP {
            skip = 0;
        }
        cur_match_pos += count;
        if count == 1 {
            let only = arena[cur_match_pos - 1].length();
            if only > max_zopfli_len {
                skip = skip.max(only);
            }
        }
        if skip > 1 {
            skip -= 1;
            while skip != 0 {
                i += 1;
                if i + 3 >= num_bytes {
                    break;
                }
                evaluate_node(
                    params.logical_position(position),
                    i,
                    ctx.max_backward_limit,
                    gap,
                    starting_dist_cache,
                    model,
                    queue,
                    nodes,
                );
                cur_match_pos += num_matches[i] as usize;
                skip -= 1;
            }
        }
        i += 1;
    }
    compute_shortest_path_from_nodes(num_bytes, nodes)
}

/// Runs the dynamic program while discovering matches
/// (`BrotliZopfliComputeShortestPath`).
///
/// Returns how many leading bytes two windows share, at most `limit`.
///
/// The attached prefix is a plain slice rather than the ring buffer, so it
/// cannot go through [`find_match_length`], which indexes one buffer twice.
fn prefix_match_length(left: &[u8], right: &[u8], limit: usize) -> usize {
    let limit = limit.min(left.len()).min(right.len());
    let (Some(left), Some(right)) = (left.get(..limit), right.get(..limit)) else {
        return 0;
    };
    let (left_words, left_tail) = left.as_chunks::<8>();
    let (right_words, right_tail) = right.as_chunks::<8>();
    let mut matched = 0usize;
    for (left_word, right_word) in left_words.iter().zip(right_words) {
        let difference = u64::from_le_bytes(*left_word) ^ u64::from_le_bytes(*right_word);
        if difference != 0 {
            return matched + (difference.trailing_zeros() >> 3) as usize;
        }
        matched += 8;
    }
    for (left_byte, right_byte) in left_tail.iter().zip(right_tail) {
        if left_byte != right_byte {
            break;
        }
        matched += 1;
    }
    matched
}

/// Shortest attached-dictionary match the high-quality search collects.
///
/// The reference passes a literal `3` to `LookupAllCompoundDictionaryMatches`.
const MIN_PREFIX_MATCH_LENGTH: usize = 3;

/// Most attached-dictionary matches one position may contribute.
///
/// The reference reserves sixty-four slots ahead of the tree's own matches and
/// passes that as the limit.
const MAX_PREFIX_MATCHES: usize = 64;

/// Merges attached-dictionary matches into the tree's, in the reference order.
///
/// Mirrors `MergeMatches`, which the dynamic program relies on: both inputs
/// are ascending in length, and the merged sequence has to stay ascending, with
/// the smaller distance first on a tie. `prefix` is `(distance, length)`
/// because the attached search has no length code to carry.
fn merge_prefix_matches(prefix: &[(usize, usize)], tree: &mut Vec<BackwardMatch>, start: usize) {
    if prefix.is_empty() {
        return;
    }
    // Merge backwards into the retained arena tail. Equal tree matches precede
    // prefix matches, exactly as the forward reference merge orders them.
    let mut right = tree.len();
    let mut left = prefix.len();
    let mut output = right + left;
    tree.resize(output, BackwardMatch::new(0, 0));
    while left > 0 {
        output -= 1;
        let (distance, length) = prefix[left - 1];
        if right > start
            && (length < tree[right - 1].length()
                || (length == tree[right - 1].length()
                    && distance < tree[right - 1].distance as usize))
        {
            right -= 1;
            tree[output] = tree[right];
        } else {
            left -= 1;
            tree[output] = BackwardMatch::new(distance, length);
        }
    }
}

/// Quality ten's inner loop: one tree query per position, priced immediately
/// and then thrown away.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors BrotliZopfliComputeShortestPath, whose parameters are all needed"
)]
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
fn zopfli_compute_shortest_path<S: Simd, const INDEPENDENT: bool>(
    simd: S,
    num_bytes: usize,
    position: usize,
    ringbuffer: &[u8],
    ringbuffer_mask: usize,
    params: &HqParams,
    attached: Option<&SharedContextInner>,
    starting_dist_cache: &[i32; 4],
    matcher: &mut BinaryTreeMatcher,
    workspace: &mut ZopfliWorkspace,
) -> usize {
    let max_backward_limit = params.max_backward_limit();
    let max_zopfli_len = params.max_zopfli_len();
    let short_scan = params.short_scan();
    let gap = attached.map_or(0, SharedContextInner::total_size);
    let store_end = if num_bytes >= STORE_LOOKAHEAD {
        position + num_bytes - STORE_LOOKAHEAD + 1
    } else {
        position
    };

    let ZopfliWorkspace {
        nodes,
        queue,
        scratch,
        prefix_matches,
        model,
        ..
    } = workspace;
    nodes[0].length = 0;
    nodes[0].set_cost(0.0);
    model.set_from_literal_costs(position, ringbuffer, ringbuffer_mask, num_bytes);
    queue.clear();

    let ctx = UpdateContext {
        num_bytes,
        block_start: position,
        ringbuffer,
        ringbuffer_mask,
        params,
        max_backward_limit,
        starting_dist_cache,
        gap,
        attached,
    };
    // Scratch for the attached dictionary's own candidates, merged into the
    // tree's below. Sixty-four is the reference's `LookupAllCompoundDictionary`
    // limit, and it never grows past that.

    let mut i = 0usize;
    while i + HASH_TYPE_LENGTH - 1 < num_bytes {
        let pos = position + i;
        let max_distance = pos.min(max_backward_limit);
        let dictionary_start = params.logical_position(pos).min(max_backward_limit);

        scratch.clear();
        let dictionary_distance = if INDEPENDENT {
            usize::MAX
        } else {
            dictionary_start + gap
        };
        #[cfg(feature = "experimental")]
        let dictionary_distance = if attached.is_some_and(|c| c.static_index.is_some()) {
            params.dist.max_distance as usize + 1
        } else {
            dictionary_distance
        };
        matcher.find_all_matches(
            simd,
            ringbuffer,
            ringbuffer_mask,
            pos,
            num_bytes - i,
            max_distance,
            dictionary_distance,
            params.dist.max_distance as usize,
            short_scan,
            scratch,
        );
        #[cfg(feature = "experimental")]
        if let Some(index) = attached.and_then(|c| c.static_index.as_ref()) {
            let context = params.dictionary_context_mode.context(
                crate::compressor::core::rfc9841::static_index::previous(
                    ringbuffer,
                    pos,
                    ringbuffer_mask,
                    1,
                ),
                crate::compressor::core::rfc9841::static_index::previous(
                    ringbuffer,
                    pos,
                    ringbuffer_mask,
                    2,
                ),
            );
            let start = pos & ringbuffer_mask;
            index.find_all(
                context,
                &ringbuffer[start..start + num_bytes - i],
                dictionary_start + gap,
                params.dist.max_distance as usize,
                scratch,
            );
        }
        if let Some(context) = attached {
            prefix_matches.clear();
            context.find_all_matches(
                ringbuffer,
                ringbuffer_mask,
                pos,
                MIN_PREFIX_MATCH_LENGTH,
                num_bytes - i,
                dictionary_start,
                params.dist.max_distance as usize,
                MAX_PREFIX_MATCHES,
                prefix_matches,
            );
            merge_prefix_matches(prefix_matches, scratch, 0);
        }
        // A copy longer than the cap makes every shorter candidate irrelevant.
        if let Some(&longest) = scratch.last()
            && longest.length() > max_zopfli_len
        {
            scratch.clear();
            scratch.push(longest);
        }

        let mut skip = update_nodes::<_, INDEPENDENT>(simd, &ctx, i, scratch, model, queue, nodes);
        if skip < LONG_COPY_QUICK_STEP {
            skip = 0;
        }
        if scratch.len() == 1 && scratch[0].length() > max_zopfli_len {
            skip = skip.max(scratch[0].length());
        }
        if skip > 1 {
            // The tail of the copy still has to reach the tree, or later
            // positions would not see it.
            matcher.store_range(
                simd,
                ringbuffer,
                ringbuffer_mask,
                pos + 1,
                (pos + skip).min(store_end),
            );
            skip -= 1;
            while skip != 0 {
                i += 1;
                if i + HASH_TYPE_LENGTH > num_bytes {
                    break;
                }
                evaluate_node(
                    params.logical_position(position),
                    i,
                    max_backward_limit,
                    gap,
                    starting_dist_cache,
                    model,
                    queue,
                    nodes,
                );
                skip -= 1;
            }
        }
        i += 1;
    }
    compute_shortest_path_from_nodes(num_bytes, nodes)
}

/// Turns `num_bytes` of input at `position` into quality ten's commands.
///
/// Mirrors `BrotliCreateZopfliBackwardReferences`: one cost model, one pass.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors BrotliCreateZopfliBackwardReferences, whose parameters are all needed"
)]
pub(crate) fn create_zopfli_backward_references<S: Simd, const INDEPENDENT: bool>(
    simd: S,
    num_bytes: usize,
    position: usize,
    ringbuffer: &[u8],
    ringbuffer_mask: usize,
    params: &HqParams,
    attached: Option<&SharedContextInner>,
    matcher: &mut BinaryTreeMatcher,
    workspace: &mut ZopfliWorkspace,
    state: &mut ZopfliState,
    commands: &mut Vec<Command>,
) {
    workspace.prepare(num_bytes, params.dist.alphabet_size_limit as usize);
    let starting_dist_cache = state.dist_cache;
    let gap = attached.map_or(0, SharedContextInner::total_size);
    zopfli_compute_shortest_path::<_, INDEPENDENT>(
        simd,
        num_bytes,
        position,
        ringbuffer,
        ringbuffer_mask,
        params,
        attached,
        &starting_dist_cache,
        matcher,
        workspace,
    );
    create_commands(
        num_bytes,
        position,
        &workspace.nodes,
        params,
        gap,
        state,
        commands,
    );
}

/// Turns `num_bytes` of input at `position` into quality eleven's commands.
///
/// Mirrors `BrotliCreateHqZopfliBackwardReferences`. Two things separate it
/// from quality ten: every position's matches are found once up front, and the
/// whole dynamic program runs twice — the second time priced by the commands
/// the first produced, which is a far better model than the literal-cost
/// estimate it started from.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors BrotliCreateHqZopfliBackwardReferences, whose parameters are all needed"
)]
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
pub(crate) fn create_hq_zopfli_backward_references<S: Simd, const INDEPENDENT: bool>(
    simd: S,
    num_bytes: usize,
    position: usize,
    ringbuffer: &[u8],
    ringbuffer_mask: usize,
    params: &HqParams,
    attached: Option<&SharedContextInner>,
    matcher: &mut BinaryTreeMatcher,
    workspace: &mut ZopfliWorkspace,
    state: &mut ZopfliState,
    commands: &mut Vec<Command>,
) {
    let alphabet_size = params.dist.alphabet_size_limit as usize;
    workspace.prepare(num_bytes, alphabet_size);
    let max_backward_limit = params.max_backward_limit();
    let max_zopfli_len = params.max_zopfli_len();
    let short_scan = params.short_scan();
    let gap = attached.map_or(0, SharedContextInner::total_size);
    let store_end = if num_bytes >= STORE_LOOKAHEAD {
        position + num_bytes - STORE_LOOKAHEAD + 1
    } else {
        position
    };

    workspace.arena.clear();
    // The reference starts its arena at four matches per byte and grows from
    // there; reserving the same up front keeps the peak comparable rather than
    // leaving it to `Vec`'s doubling from empty.
    workspace.arena.reserve(4 * num_bytes);
    workspace.num_matches.clear();
    workspace.num_matches.resize(num_bytes, 0);

    let mut i = 0usize;
    while i + HASH_TYPE_LENGTH - 1 < num_bytes {
        let pos = position + i;
        let max_distance = pos.min(max_backward_limit);
        let dictionary_start = params.logical_position(pos).min(max_backward_limit);
        let max_length = num_bytes - i;

        let cur_match_pos = workspace.arena.len();
        let dictionary_distance = if INDEPENDENT {
            usize::MAX
        } else {
            dictionary_start + gap
        };
        #[cfg(feature = "experimental")]
        let dictionary_distance = if attached.is_some_and(|c| c.static_index.is_some()) {
            params.dist.max_distance as usize + 1
        } else {
            dictionary_distance
        };
        matcher.find_all_matches(
            simd,
            ringbuffer,
            ringbuffer_mask,
            pos,
            max_length,
            max_distance,
            dictionary_distance,
            params.dist.max_distance as usize,
            short_scan,
            &mut workspace.arena,
        );
        #[cfg(feature = "experimental")]
        if let Some(index) = attached.and_then(|c| c.static_index.as_ref()) {
            let context = params.dictionary_context_mode.context(
                crate::compressor::core::rfc9841::static_index::previous(
                    ringbuffer,
                    pos,
                    ringbuffer_mask,
                    1,
                ),
                crate::compressor::core::rfc9841::static_index::previous(
                    ringbuffer,
                    pos,
                    ringbuffer_mask,
                    2,
                ),
            );
            let start = pos & ringbuffer_mask;
            workspace.scratch.clear();
            workspace
                .scratch
                .extend_from_slice(&workspace.arena[cur_match_pos..]);
            index.find_all(
                context,
                &ringbuffer[start..start + max_length],
                dictionary_start + gap,
                params.dist.max_distance as usize,
                &mut workspace.scratch,
            );
            workspace.arena.truncate(cur_match_pos);
            workspace.arena.extend_from_slice(&workspace.scratch);
        }
        if let Some(context) = attached {
            workspace.prefix_matches.clear();
            context.find_all_matches(
                ringbuffer,
                ringbuffer_mask,
                pos,
                MIN_PREFIX_MATCH_LENGTH,
                max_length,
                dictionary_start,
                params.dist.max_distance as usize,
                MAX_PREFIX_MATCHES,
                &mut workspace.prefix_matches,
            );
            merge_prefix_matches(
                &workspace.prefix_matches,
                &mut workspace.arena,
                cur_match_pos,
            );
        }
        let found = workspace.arena.len() - cur_match_pos;
        workspace.num_matches[i] = found as u32;

        if found > 0 {
            let match_len = workspace.arena[workspace.arena.len() - 1].length();
            if match_len > max_zopfli_len {
                // One copy this long settles the next `match_len` positions, so
                // keep only it and skip past them — recording no matches there,
                // which is what makes the arena affordable.
                let skip = match_len - 1;
                let longest = workspace.arena[workspace.arena.len() - 1];
                workspace.arena.truncate(cur_match_pos);
                workspace.arena.push(longest);
                workspace.num_matches[i] = 1;
                matcher.store_range(
                    simd,
                    ringbuffer,
                    ringbuffer_mask,
                    pos + 1,
                    (pos + match_len).min(store_end),
                );
                let stop = (i + 1 + skip).min(num_bytes);
                workspace.num_matches[i + 1..stop].fill(0);
                i += skip;
            }
        }
        i += 1;
    }

    // The state the second pass has to start from again.
    let orig_num_literals = state.num_literals;
    let orig_last_insert_len = state.last_insert_len;
    let orig_dist_cache = state.dist_cache;
    let orig_num_commands = commands.len();

    for iteration in 0..2 {
        workspace.reset_nodes(num_bytes);
        if iteration == 0 {
            workspace.model.set_from_literal_costs(
                position,
                ringbuffer,
                ringbuffer_mask,
                num_bytes,
            );
        } else {
            workspace.model.set_from_commands(
                position,
                ringbuffer,
                ringbuffer_mask,
                &commands[orig_num_commands..],
                orig_last_insert_len,
                num_bytes,
            );
        }
        commands.truncate(orig_num_commands);
        state.num_literals = orig_num_literals;
        state.last_insert_len = orig_last_insert_len;
        state.dist_cache = orig_dist_cache;

        let ZopfliWorkspace {
            nodes,
            queue,
            arena,
            num_matches,
            model,
            ..
        } = workspace;
        zopfli_iterate::<_, INDEPENDENT>(
            simd,
            num_bytes,
            position,
            ringbuffer,
            ringbuffer_mask,
            params,
            gap,
            attached,
            &orig_dist_cache,
            model,
            num_matches,
            arena,
            queue,
            nodes,
        );
        create_commands(
            num_bytes,
            position,
            &workspace.nodes,
            params,
            gap,
            state,
            commands,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::params::HqQuality;
    use super::*;
    use crate::compressor::{CompressParams, QualityLevel, WindowBits};
    use crate::shared::ringbuffer::RingBuffer;
    use fearless_simd::{Level, dispatch};

    #[test]
    fn prefix_merge_preserves_prior_arena_entries_and_tree_ties() {
        let prior = BackwardMatch::new(99, 99);
        let tree_tie = BackwardMatch::dictionary(8, 7, 4);
        let mut arena = vec![
            prior,
            BackwardMatch::new(2, 3),
            tree_tie,
            BackwardMatch::new(1, 10),
        ];
        arena.reserve(4);
        let capacity = arena.capacity();
        merge_prefix_matches(&[(1, 3), (8, 7), (4, 12)], &mut arena, 1);
        let actual: Vec<_> = arena
            .iter()
            .map(|entry| (entry.distance, entry.length(), entry.length_code()))
            .collect();
        assert_eq!(
            actual,
            [
                (99, 99, 99),
                (1, 3, 3),
                (2, 3, 3),
                (8, 7, 4),
                (8, 7, 7),
                (1, 10, 10),
                (4, 12, 12)
            ]
        );
        assert_eq!(arena.capacity(), capacity);
        merge_prefix_matches(&[], &mut arena, 1);
        assert_eq!(arena.len(), 7);
        merge_prefix_matches(&[(3, 5)], &mut arena, 7);
        assert_eq!(arena.last().expect("inserted").length(), 5);
    }

    /// A ring buffer holding `data`, laid out as the encoder would.
    fn ring(params: &HqParams, data: &[u8]) -> RingBuffer {
        let mut buffer = RingBuffer::new(params.rb_bits(), params.lgblock);
        buffer.write(data);
        buffer.clear_margin();
        buffer
    }

    /// Runs our search over one block, returning its commands and state.
    fn our_references(quality: QualityLevel, data: &[u8]) -> (Vec<Command>, ZopfliState, HqParams) {
        let public = CompressParams::new(quality, WindowBits::DEFAULT);
        let params = HqParams::new(&public).expect("supported quality");
        let buffer = ring(&params, data);
        let mut matcher = BinaryTreeMatcher::new(params.lgwin);
        matcher.prepare(false, 0);
        let mut workspace = ZopfliWorkspace::new(params.dist.alphabet_size_limit as usize);
        let mut state = ZopfliState::default();
        let mut commands = Vec::new();
        let level = Level::try_detect().unwrap_or_else(Level::baseline);

        dispatch!(level, simd => matcher.stitch_to_previous_block(
            simd, data.len(), 0, buffer.buffer(), buffer.mask()));
        match params.quality {
            HqQuality::Q10 => {
                dispatch!(level, simd => create_zopfli_backward_references::<_, false>(
                    simd, data.len(), 0, buffer.buffer(), buffer.mask(), &params,
                    None, &mut matcher, &mut workspace, &mut state, &mut commands,
                ))
            }
            HqQuality::Q11 => {
                dispatch!(level, simd => create_hq_zopfli_backward_references::<_, false>(
                    simd, data.len(), 0, buffer.buffer(), buffer.mask(), &params,
                    None, &mut matcher, &mut workspace, &mut state, &mut commands,
                ))
            }
        }
        (commands, state, params)
    }

    /// Runs the C search over the same block, through this workspace's shim.
    fn c_references(
        quality: QualityLevel,
        params: &HqParams,
        data: &[u8],
    ) -> (Vec<Command>, ZopfliState) {
        let buffer = ring(params, data);
        let capacity = data.len() + 16;
        let mut commands = vec![Command::default(); capacity];
        let mut dist_cache = ZopfliState::default().dist_cache;
        let mut last_insert_len = 0usize;
        let mut num_literals = 0usize;

        // SAFETY: the ring buffer is readable at every index the search reaches
        // under its mask, `dist_cache` holds four entries, and `commands` has
        // room for one command per input byte, which the format bounds it by.
        let count = unsafe {
            google_brotli_ffi::mbrotli_shim_zopfli_references(
                usize::from(quality) as i32,
                params.lgwin as i32,
                buffer.buffer().as_ptr(),
                buffer.mask(),
                0,
                data.len(),
                dist_cache.as_mut_ptr(),
                &raw mut last_insert_len,
                &raw mut num_literals,
                commands.as_mut_ptr().cast::<u8>(),
                capacity,
            )
        };
        commands.truncate(count);
        (
            commands,
            ZopfliState {
                dist_cache,
                last_insert_len,
                num_literals,
            },
        )
    }

    /// Compares both searches over one block.
    fn assert_references_match_c(name: &str, quality: QualityLevel, data: &[u8]) {
        let (ours, our_state, params) = our_references(quality, data);
        let (theirs, their_state) = c_references(quality, &params, data);

        let first = ours
            .iter()
            .zip(&theirs)
            .position(|(a, b)| a != b)
            .unwrap_or(ours.len().min(theirs.len()));
        assert_eq!(
            ours.len(),
            theirs.len(),
            "case {name}, quality {quality:?}: {} commands against {}, first difference at {first}",
            ours.len(),
            theirs.len()
        );
        for (index, (ours, theirs)) in ours.iter().zip(&theirs).enumerate() {
            assert_eq!(
                ours, theirs,
                "case {name}, quality {quality:?}, command {index}"
            );
        }
        assert_eq!(
            our_state.dist_cache, their_state.dist_cache,
            "case {name}, quality {quality:?}: distance cache"
        );
        assert_eq!(
            our_state.last_insert_len, their_state.last_insert_len,
            "case {name}, quality {quality:?}: trailing literals"
        );
        assert_eq!(
            our_state.num_literals, their_state.num_literals,
            "case {name}, quality {quality:?}: literal count"
        );
    }

    #[test]
    fn the_command_stream_matches_the_c_search() {
        let mut rng = 0x243F_6A88_85A3_08D3u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 24) as u8
        };
        let mut collisions = Vec::new();
        while collisions.len() < 4000 {
            collisions.extend_from_slice(b"AAAAA");
            collisions.push(next());
        }

        let mut text = Vec::new();
        while text.len() < 4000 {
            text.extend_from_slice(b"the quick brown fox jumps over the lazy dog. ");
        }

        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("empty", Vec::new()),
            ("tiny", b"abc".to_vec()),
            ("short-repeat", b"hello hello hello hello".to_vec()),
            ("zeros", vec![0u8; 3000]),
            ("text", text),
            ("collisions", collisions),
            ("ascending", (0..4000u32).map(|i| (i % 251) as u8).collect()),
            ("random", (0..4000u32).map(|_| next()).collect()),
            (
                "dictionary-words",
                b"time download government information description background ".repeat(40),
            ),
        ];

        for (name, data) in &cases {
            for quality in [QualityLevel::Q10, QualityLevel::Q11] {
                assert_references_match_c(name, quality, data);
            }
        }
    }

    #[test]
    fn every_prefix_of_a_repetitive_block_matches_the_c_search() {
        // The lengths around a repeat period are where a copy that just fits
        // and one that just does not are priced against each other.
        let mut rng = 0x0BAD_C0DE_0BAD_C0DEu64;
        let mut data = Vec::new();
        while data.len() < 2400 {
            data.extend_from_slice(b"AAAAA");
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            data.push((rng >> 24) as u8);
        }
        for len in (0..2400).step_by(7) {
            assert_references_match_c("prefix", QualityLevel::Q10, &data[..len]);
        }
    }
}

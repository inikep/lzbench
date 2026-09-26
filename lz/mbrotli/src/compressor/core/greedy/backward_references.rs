//! Greedy backward-reference search shared by qualities two to nine.
//!
//! Ports `CreateBackwardReferences` from `c/enc/backward_references_inc.h` and
//! `ComputeDistanceCode` from `c/enc/backward_references.c` of the pinned
//! reference (`google/brotli` v1.2.0, commit `028fb5a`), together with
//! `ExtendLastCommand` from `c/enc/encode.c`.
//!
//! The decision order here is the compression format's semantics, not an
//! implementation detail: which candidate wins, when a match is delayed by a
//! byte, which positions are stored and which are skipped all show up in the
//! emitted bytes. The match finder underneath may be accelerated freely, but
//! this sequence may not be reordered.

use alloc::vec::Vec;

use fearless_simd::Simd;

use super::hashers::{
    DistanceCache, MatchQuery, MatchRun, Matcher, RunVisitor, prepare_distance_cache,
};
use super::params::GreedyParams;
use crate::compressor::core::rfc9841::context::SharedContextInner;
use crate::shared::command::Command;
use crate::shared::dictionary::DictionaryStats;
use crate::shared::distance::NUM_DISTANCE_SHORT_CODES;
use crate::shared::ringbuffer::{BlockSpan, Window};
use crate::shared::score::{MIN_SCORE, SearchResult};

/// Score a delayed match has to beat the current one by (`cost_diff_lazy`).
const COST_DIFF_LAZY: usize = 175;

/// How many times in a row a match may be delayed by one byte.
const MAX_DELAYED_IN_A_ROW: usize = 4;

/// Returns the intermediate distance code for `distance`.
///
/// Mirrors `ComputeDistanceCode`. The first sixteen codes are short codes
/// relative to the distance cache; the two magic nibble tables spell out which
/// short code expresses "one less than the last distance", "two more", and so
/// on.
pub(crate) fn compute_distance_code(
    distance: usize,
    max_distance: usize,
    cache: &DistanceCache,
) -> usize {
    if distance <= max_distance {
        let distance_plus_3 = distance + 3;
        let offset0 = distance_plus_3.wrapping_sub(cache[0] as usize);
        let offset1 = distance_plus_3.wrapping_sub(cache[1] as usize);
        if distance == cache[0] as usize {
            return 0;
        }
        if distance == cache[1] as usize {
            return 1;
        }
        if offset0 < 7 {
            return (0x975_0468usize >> (4 * offset0)) & 0xF;
        }
        if offset1 < 7 {
            return (0xFDB_1ACEusize >> (4 * offset1)) & 0xF;
        }
        if distance == cache[2] as usize {
            return 2;
        }
        if distance == cache[3] as usize {
            return 3;
        }
    }
    distance + NUM_DISTANCE_SHORT_CODES as usize - 1
}

/// State the reference search carries between input blocks.
pub(crate) struct ReferenceState {
    /// The four distances that have short codes.
    pub(crate) dist_cache: DistanceCache,
    /// Literals produced but not yet attached to a command.
    pub(crate) last_insert_len: usize,
    /// Literals in the commands emitted for the current meta-block.
    pub(crate) num_literals: usize,
    /// Whether probing the static dictionary is still paying off.
    pub(crate) dictionary: DictionaryStats,
}

impl Default for ReferenceState {
    /// Returns the state a fresh stream starts from.
    fn default() -> Self {
        Self {
            dist_cache: super::hashers::INITIAL_DISTANCE_CACHE,
            last_insert_len: 0,
            num_literals: 0,
            dictionary: DictionaryStats::default(),
        }
    }
}

/// Loop-invariant inputs of one reference search over a block.
///
/// Gathered once so the search loop and the match commit share them. It is
/// `Copy`: the search loop works on a copy of its own, whose fields the
/// compiler can keep in registers because nothing takes its address.
#[derive(Copy, Clone)]
struct Block<'a> {
    params: &'a GreedyParams,
    ringbuffer: &'a [u8],
    /// The ring buffer cut to the window (see [`MatchQuery::window`]).
    window: &'a [u8],
    mask: usize,
    attached: Option<&'a SharedContextInner>,
    /// End of the block's input.
    pos_end: usize,
    /// Last position a store may read a whole hash from.
    store_end: usize,
    max_backward_limit: usize,
    /// Offset the stream starts at (`stream_offset`), zero for ordinary ones.
    position_offset: usize,
    /// Distance shift that addresses the attached dictionary (`gap`).
    gap: usize,
    /// Longest distance the distance alphabet can express.
    max_distance_code: usize,
    /// Window of the random-data heuristic.
    heuristics_window: usize,
    /// Whether the delayed search restarts from nothing (quality five and up).
    extensive: bool,
    /// Cached distances the matcher probes.
    last_distances: usize,
}

impl<'a> Block<'a> {
    /// Builds the query for a search at `position` allowed `max_length` bytes.
    #[inline(always)]
    fn query<const ENABLE_PREFIX: bool>(
        &self,
        cache: &'a DistanceCache,
        position: usize,
        max_length: usize,
    ) -> MatchQuery<'a> {
        let max_backward = position.min(self.max_backward_limit);
        MatchQuery {
            #[cfg(feature = "experimental")]
            custom: if ENABLE_PREFIX {
                self.attached
                    .and_then(|c| c.static_index.as_ref())
                    .map(|index| {
                        index.combination(super::context_model::context(
                            crate::compressor::core::rfc9841::static_index::previous(
                                self.ringbuffer,
                                position,
                                self.mask,
                                1,
                            ),
                            crate::compressor::core::rfc9841::static_index::previous(
                                self.ringbuffer,
                                position,
                                self.mask,
                                2,
                            ),
                        ))
                    })
            } else {
                None
            },
            data: self.ringbuffer,
            window: self.window,
            mask: self.mask,
            cache,
            cur_ix: position,
            max_length,
            max_backward,
            position_offset: self.position_offset,
            dictionary_limit: self.max_backward_limit,
            gap: self.gap,
            max_distance: self.max_distance_code,
        }
    }

    /// Runs the matcher and, with a prefix attached, the prefix search too.
    #[inline(always)]
    fn search<S: Simd, R: MatchRun, const ENABLE_PREFIX: bool>(
        &self,
        simd: S,
        matcher: &mut R,
        state: &mut ReferenceState,
        position: usize,
        max_length: usize,
        out: &mut SearchResult,
    ) {
        let query = self.query::<ENABLE_PREFIX>(&state.dist_cache, position, max_length);
        let dictionary_start = query.dictionary_start();
        matcher.find_longest_match(simd, &mut state.dictionary, query, out);
        if ENABLE_PREFIX && let Some(context) = self.attached {
            context.find_match(
                simd,
                self.ringbuffer,
                self.mask,
                &state.dist_cache,
                position,
                max_length,
                dictionary_start,
                self.max_distance_code,
                out,
            );
        }
    }
}

/// The loop-carried state of a reference search.
///
/// Passed and returned by value so it never has an address the compiler
/// has to keep current in memory.
#[derive(Copy, Clone)]
struct Cursor {
    position: usize,
    insert_length: usize,
    apply_random_heuristics: usize,
}

/// Turns `num_bytes` of input at `position` into commands.
///
/// Appends to `commands` and updates `state`; literals that no command has
/// claimed yet stay in [`ReferenceState::last_insert_len`] for the next call.
/// `ENABLE_PREFIX` mirrors the reference's `ENABLE_COMPOUND_DICTIONARY`, which
/// it uses to compile this function twice per match finder: once with the
/// prefix search in it and once without. It is a const parameter rather than a
/// runtime `is_some()` for the same reason the reference makes it a macro —
/// this is the hottest loop in the crate, and a branch that is always taken
/// the same way still costs a register and an instruction at every position.
/// Measured on an Apple M5 Pro over the eleven `oneshot/q3` corpora, folding
/// the two into one runtime branch cost 2.1% of the geometric-mean throughput
/// and 9.7% on `text-1MiB`.
///
/// The loop itself lives in [`SearchLoop`], which the matcher runs through
/// the view its current layout uses: a finder with more than one layout
/// gets one loop per layout, each carrying only that layout's state.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors CreateBackwardReferences, whose parameters are all needed"
)]
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
pub(crate) fn create_backward_references<
    S: Simd,
    M: Matcher,
    const ENABLE_PREFIX: bool,
    const INDEPENDENT: bool,
>(
    simd: S,
    matcher: &mut M,
    params: &GreedyParams,
    window: Window<'_>,
    span: BlockSpan,
    attached: Option<&SharedContextInner>,
    state: &mut ReferenceState,
    commands: &mut Vec<Command>,
) {
    let num_bytes = span.bytes as usize;
    let position = span.position as usize;
    let pos_end = position + num_bytes;
    #[cfg(feature = "experimental")]
    let position_offset = params.stream_offset;
    #[cfg(not(feature = "experimental"))]
    let position_offset = 0;
    let block = Block {
        params,
        ringbuffer: window.data,
        window: window
            .data
            .get(..window.mask.saturating_add(1))
            .unwrap_or(window.data),
        mask: window.mask,
        attached,
        pos_end,
        // Bound by the run below, which knows its own lookahead.
        store_end: position,
        max_backward_limit: params.max_backward_limit(),
        position_offset,
        // Every distance that addresses the attached dictionary is shifted
        // past the window by this much. Without `ENABLE_PREFIX` it is a
        // compile-time zero, so every `+ gap` folds away.
        gap: if ENABLE_PREFIX {
            attached.map_or(0, SharedContextInner::total_size)
        } else {
            0
        },
        max_distance_code: params.dist.max_distance as usize,
        heuristics_window: params.random_heuristics_window_size(),
        extensive: params.quality.extensive_reference_search(),
        last_distances: matcher.last_distances_to_check(),
    };
    let cursor = Cursor {
        position,
        insert_length: state.last_insert_len,
        apply_random_heuristics: position + block.heuristics_window,
    };
    // The derived cache entries are a function of the four remembered ones,
    // so they are refreshed here and again whenever a command changes them.
    prepare_distance_cache(&mut state.dist_cache, block.last_distances);

    let mut cursor = cursor;
    let mut advance = matcher.checkpoint(position, num_bytes);
    loop {
        cursor = matcher.visit_run(SearchLoop::<S, ENABLE_PREFIX, INDEPENDENT> {
            simd,
            block,
            // At least one position, so every pass makes progress.
            yield_at: cursor.position.saturating_add(advance.max(1)),
            cursor,
            state: &mut *state,
            commands: &mut *commands,
        });
        if cursor.position + M::HASH_TYPE_LENGTH >= pos_end {
            break;
        }
        advance = matcher.checkpoint(cursor.position, pos_end - cursor.position);
    }

    cursor.insert_length += pos_end - cursor.position;
    state.last_insert_len = cursor.insert_length;
}

/// The search loop of [`create_backward_references`], run through one
/// concrete view of the match finder's tables.
struct SearchLoop<'a, S, const ENABLE_PREFIX: bool, const INDEPENDENT: bool> {
    simd: S,
    block: Block<'a>,
    /// Position from which the loop hands control back to the matcher
    /// (see [`Matcher::checkpoint`]).
    yield_at: usize,
    cursor: Cursor,
    state: &'a mut ReferenceState,
    commands: &'a mut Vec<Command>,
}

impl<S: Simd, const ENABLE_PREFIX: bool, const INDEPENDENT: bool> RunVisitor
    for SearchLoop<'_, S, ENABLE_PREFIX, INDEPENDENT>
{
    type Output = Cursor;

    /// Runs the loop and returns where it stopped.
    ///
    /// The loop body is split in two: the search at every position stays
    /// here, and everything a found match entails — the delayed search, the
    /// command, the stores — moves to [`commit_match`]. The split keeps the
    /// loop's state in locals that cross the boundary by value; whether the
    /// commit path is then inlined is left to the compiler, because forcing
    /// it out of line was measured to cost 8–10% on quality two and three
    /// text and 4–5% on the deeper matchers once that state stopped living
    /// in memory.
    #[inline(always)]
    fn visit<R: MatchRun>(self, mut run: R) -> Cursor {
        let Self {
            simd,
            mut block,
            yield_at,
            cursor,
            state,
            commands,
        } = self;
        let pos_end = block.pos_end;
        // `store_end` still holds the block's first position, so a loop
        // resumed after a checkpoint bounds its stores as the first did.
        block.store_end = if pos_end - block.store_end >= R::STORE_LOOKAHEAD {
            pos_end - R::STORE_LOOKAHEAD + 1
        } else {
            block.store_end
        };
        // One bound for both exits: the end of the searchable input and
        // the checkpoint, so resuming costs the loop nothing.
        let loop_end = pos_end.saturating_sub(R::HASH_TYPE_LENGTH).min(yield_at);
        let block = block;
        // Enter the selected feature context after specializing the matcher.
        // This keeps vector operations inline without merging every matcher
        // into one large feature-enabled function at the outer dispatch
        // boundary. The closure moves its captures: the feature context is a
        // separate function, and a capture by reference would be a pointer
        // it dereferences at every use, where a moved value is a local it
        // keeps in a register.
        simd.vectorize(
            #[inline(always)]
            move || {
                // The loop's state lives in locals whose addresses never
                // escape, so the compiler keeps them in registers: the commit
                // path takes the run and the cursor by value and hands them
                // back. The block is copied for the same reason; the commit
                // path borrows the original.
                let hot = block;
                let Cursor {
                    mut position,
                    mut insert_length,
                    mut apply_random_heuristics,
                } = cursor;
                while position < loop_end {
                    let max_length = pos_end - position;
                    let mut sr = SearchResult::empty();
                    hot.search::<S, R, ENABLE_PREFIX>(
                        simd, &mut run, state, position, max_length, &mut sr,
                    );

                    if sr.is_match() {
                        let committed;
                        (run, committed) = commit_match::<S, R, ENABLE_PREFIX, INDEPENDENT>(
                            simd,
                            run,
                            &block,
                            Cursor {
                                position,
                                insert_length,
                                apply_random_heuristics,
                            },
                            state,
                            commands,
                            sr,
                            max_length,
                        );
                        position = committed.position;
                        insert_length = committed.insert_length;
                        apply_random_heuristics = committed.apply_random_heuristics;
                        continue;
                    }

                    insert_length += 1;
                    position += 1;
                    if position <= apply_random_heuristics {
                        continue;
                    }
                    // Nothing has matched for a long time. Storing every
                    // position of incompressible data costs time and floods
                    // the table, so the scan strides forward and only stores
                    // part of what it skips.
                    let (stride, margin_floor) =
                        if position > apply_random_heuristics + 4 * hot.heuristics_window {
                            (4usize, 4usize)
                        } else {
                            (2usize, 2usize)
                        };
                    let margin = (R::STORE_LOOKAHEAD - 1).max(margin_floor);
                    let pos_jump = (position + 4 * stride).min(pos_end.saturating_sub(margin));
                    while position < pos_jump {
                        run.store(hot.ringbuffer, hot.mask, position);
                        insert_length += stride;
                        position += stride;
                    }
                }
                Cursor {
                    position,
                    insert_length,
                    apply_random_heuristics,
                }
            },
        )
    }
}

/// Finishes the match `sr` found at the cursor: delays it while a later
/// start scores better, emits its command, and stores the positions it covers.
///
/// The feature context is re-entered here because the delayed search and the
/// stores use the same vector kernels; when the compiler inlines the call, the
/// nested context is free.
#[expect(
    clippy::too_many_arguments,
    reason = "the second half of CreateBackwardReferences' loop body"
)]
#[inline]
fn commit_match<S: Simd, R: MatchRun, const ENABLE_PREFIX: bool, const INDEPENDENT: bool>(
    simd: S,
    mut matcher: R,
    block: &Block<'_>,
    mut cursor: Cursor,
    state: &mut ReferenceState,
    commands: &mut Vec<Command>,
    mut sr: SearchResult,
    mut max_length: usize,
) -> (R, Cursor) {
    simd.vectorize(
        #[inline(always)]
        move || {
            let pos_end = block.pos_end;
            // A match is available; look one byte ahead for a better one, up
            // to four times in a row.
            let mut delayed = 0usize;
            max_length -= 1;
            loop {
                let mut sr2 = SearchResult {
                    // Below quality five the delayed search starts from the
                    // length it already has, which lets the matcher reject
                    // most candidates without measuring them. Quality five
                    // gives that shortcut up and searches everything again.
                    len: if block.extensive {
                        0
                    } else {
                        (sr.len - 1).min(max_length)
                    },
                    distance: 0,
                    score: MIN_SCORE,
                    len_code_delta: 0,
                };
                block.search::<S, R, ENABLE_PREFIX>(
                    simd,
                    &mut matcher,
                    state,
                    cursor.position + 1,
                    max_length,
                    &mut sr2,
                );
                if sr2.score >= sr.score + COST_DIFF_LAZY {
                    // Emit one more literal and start the match a byte later.
                    cursor.position += 1;
                    cursor.insert_length += 1;
                    sr = sr2;
                    delayed += 1;
                    if delayed < MAX_DELAYED_IN_A_ROW
                        && cursor.position + R::HASH_TYPE_LENGTH < pos_end
                    {
                        max_length -= 1;
                        continue;
                    }
                }
                break;
            }

            let position = cursor.position;
            cursor.apply_random_heuristics = position + 2 * sr.len + block.heuristics_window;
            let dictionary_start = (position + block.position_offset).min(block.max_backward_limit);
            let distance_code = if INDEPENDENT {
                sr.distance + NUM_DISTANCE_SHORT_CODES as usize - 1
            } else {
                compute_distance_code(sr.distance, dictionary_start + block.gap, &state.dist_cache)
            };
            if sr.distance <= dictionary_start + block.gap && distance_code > 0 {
                state.dist_cache[3] = state.dist_cache[2];
                state.dist_cache[2] = state.dist_cache[1];
                state.dist_cache[1] = state.dist_cache[0];
                state.dist_cache[0] = sr.distance as i32;
                prepare_distance_cache(&mut state.dist_cache, block.last_distances);
            }
            commands.push(Command::new(
                &block.params.dist,
                cursor.insert_length,
                sr.len,
                sr.len_code_delta,
                distance_code,
            ));
            state.num_literals += cursor.insert_length;
            cursor.insert_length = 0;

            // Store the positions the match covered, skipping the ones a
            // run-length repeat would only poison the table with.
            let mut range_start = position + 2;
            let range_end = (position + sr.len).min(block.store_end);
            if sr.distance < (sr.len >> 2) {
                range_start =
                    range_end.min(range_start.max(position + sr.len - (sr.distance << 2)));
            }
            matcher.store_range(block.ringbuffer, block.mask, range_start, range_end);
            cursor.position += sr.len;
            (matcher, cursor)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::core::greedy::hashers::{
        self, BucketMatcher, INITIAL_DISTANCE_CACHE, NUM_REMEMBERED_DISTANCES, QuickMatcher,
    };
    use crate::compressor::core::greedy::params::GreedyQuality;
    use crate::compressor::{CompressParams, QualityLevel, WindowBits};
    use fearless_simd::{Level, dispatch};

    fn params(quality: QualityLevel) -> GreedyParams {
        let public = CompressParams::new(quality, WindowBits::DEFAULT);
        GreedyParams::new(&public, 0).expect("supported quality")
    }

    fn run(quality: QualityLevel, data: &[u8]) -> (Vec<Command>, ReferenceState) {
        let params = params(quality);
        let mut matcher = QuickMatcher::<{ 1 << 16 }, 1, 5, false>::new();
        matcher.prepare(true, data.len(), data, true);
        let mut state = ReferenceState::default();
        let mut commands = Vec::new();
        let level = Level::try_detect().unwrap_or_else(Level::baseline);
        let window = Window {
            data,
            mask: usize::MAX,
        };
        let span = BlockSpan {
            position: 0,
            bytes: data.len() as u32,
        };
        dispatch!(level, simd => create_backward_references::<_, _, false, false>(
            simd, &mut matcher, &params, window, span, None, &mut state, &mut commands,
        ));
        (commands, state)
    }

    /// Bytes the commands and the trailing literals account for.
    fn consumed(commands: &[Command], state: &ReferenceState) -> usize {
        commands
            .iter()
            .map(|command| command.insert_len as usize + command.copy_len() as usize)
            .sum::<usize>()
            + state.last_insert_len
    }

    #[test]
    fn every_input_byte_is_accounted_for() {
        for quality in [QualityLevel::Q3, QualityLevel::Q4, QualityLevel::Q5] {
            for payload in [
                b"abcabcabcabcabcabcabcabcabcabc".to_vec(),
                vec![b'z'; 5000],
                (0..5000u32).map(|i| (i % 251) as u8).collect(),
                Vec::new(),
                b"a".to_vec(),
            ] {
                let mut data = payload.clone();
                // The match finder loads whole words past the end.
                data.extend_from_slice(&[0u8; 8]);
                let params = params(quality);
                let mut matcher = QuickMatcher::<{ 1 << 16 }, 1, 5, false>::new();
                matcher.prepare(true, payload.len(), &data, true);
                let mut state = ReferenceState::default();
                let mut commands = Vec::new();
                let level = Level::try_detect().unwrap_or_else(Level::baseline);
                let window = Window {
                    data: &data,
                    mask: usize::MAX,
                };
                let span = BlockSpan {
                    position: 0,
                    bytes: payload.len() as u32,
                };
                dispatch!(level, simd => create_backward_references::<_, _, false, false>(
                    simd, &mut matcher, &params, window, span, None, &mut state, &mut commands,
                ));
                assert_eq!(
                    consumed(&commands, &state),
                    payload.len(),
                    "quality {quality:?}, {} bytes",
                    payload.len()
                );
            }
        }
    }

    #[test]
    fn a_repeated_string_becomes_one_long_copy() {
        let mut data = b"the quick brown fox ".repeat(40);
        data.extend_from_slice(&[0u8; 8]);
        let payload = data.len() - 8;
        let params = params(QualityLevel::Q3);
        let mut matcher = QuickMatcher::<{ 1 << 16 }, 1, 5, false>::new();
        matcher.prepare(true, payload, &data, true);
        let mut state = ReferenceState::default();
        let mut commands = Vec::new();
        let level = Level::try_detect().unwrap_or_else(Level::baseline);
        let window = Window {
            data: &data,
            mask: usize::MAX,
        };
        let span = BlockSpan {
            position: 0,
            bytes: payload as u32,
        };
        dispatch!(level, simd => create_backward_references::<_, _, false, false>(
            simd, &mut matcher, &params, window, span, None, &mut state, &mut commands,
        ));
        assert!(!commands.is_empty());
        let longest = commands
            .iter()
            .map(|command| command.copy_len())
            .max()
            .unwrap_or(0);
        assert!(longest > 500, "longest copy was only {longest}");
    }

    #[test]
    fn incompressible_data_produces_no_commands() {
        let (commands, state) = run(QualityLevel::Q3, &[]);
        assert!(commands.is_empty());
        assert_eq!(state.last_insert_len, 0);
    }

    #[test]
    fn the_distance_cache_only_records_real_distances() {
        let mut data = b"abcdefgh".repeat(200);
        data.extend_from_slice(&[0u8; 8]);
        let payload = data.len() - 8;
        let params = params(QualityLevel::Q3);
        let mut matcher = QuickMatcher::<{ 1 << 16 }, 1, 5, false>::new();
        matcher.prepare(true, payload, &data, true);
        let mut state = ReferenceState::default();
        let mut commands = Vec::new();
        let level = Level::try_detect().unwrap_or_else(Level::baseline);
        let window = Window {
            data: &data,
            mask: usize::MAX,
        };
        let span = BlockSpan {
            position: 0,
            bytes: payload as u32,
        };
        dispatch!(level, simd => create_backward_references::<_, _, false, false>(
            simd, &mut matcher, &params, window, span, None, &mut state, &mut commands,
        ));
        // Only the four remembered entries are history; the rest stay derived.
        assert!(
            state.dist_cache[..NUM_REMEMBERED_DISTANCES]
                .iter()
                .all(|&distance| distance > 0)
        );
    }

    #[test]
    fn distance_codes_prefer_the_cache() {
        let cache: DistanceCache = INITIAL_DISTANCE_CACHE;
        assert_eq!(compute_distance_code(4, 1 << 20, &cache), 0);
        assert_eq!(compute_distance_code(11, 1 << 20, &cache), 1);
        assert_eq!(compute_distance_code(15, 1 << 20, &cache), 2);
        assert_eq!(compute_distance_code(16, 1 << 20, &cache), 3);
        // One less than the last distance has its own short code.
        assert_eq!(compute_distance_code(3, 1 << 20, &cache), 4);
        assert_eq!(compute_distance_code(5, 1 << 20, &cache), 5);
        // Anything else is spelled out.
        assert_eq!(compute_distance_code(1000, 1 << 20, &cache), 1015);
        // Beyond the window a distance is always spelled out.
        assert_eq!(compute_distance_code(4, 3, &cache), 19);
    }

    #[test]
    fn every_short_distance_code_is_in_range() {
        let cache: DistanceCache = INITIAL_DISTANCE_CACHE;
        for distance in 1usize..64 {
            let code = compute_distance_code(distance, 1 << 20, &cache);
            assert!(code < 16 || code == distance + 15, "distance {distance}");
        }
    }

    /// Delegates to `M` but asks the search loop to hand control back every
    /// `interval` positions.
    struct Yielding<M> {
        inner: M,
        interval: usize,
        checkpoints: usize,
    }

    impl<M: Matcher> Matcher for Yielding<M> {
        const HASH_TYPE_LENGTH: usize = M::HASH_TYPE_LENGTH;
        const STORE_LOOKAHEAD: usize = M::STORE_LOOKAHEAD;

        fn visit_run<V: RunVisitor>(&mut self, visitor: V) -> V::Output {
            self.inner.visit_run(visitor)
        }

        fn last_distances_to_check(&self) -> usize {
            self.inner.last_distances_to_check()
        }

        fn checkpoint(&mut self, position: usize, remaining: usize) -> usize {
            self.checkpoints += 1;
            self.inner.checkpoint(position, remaining);
            self.interval
        }

        fn prepare(
            &mut self,
            one_shot: bool,
            input_size: usize,
            data: &[u8],
            clear: bool,
        ) -> hashers::Sweep {
            self.inner.prepare(one_shot, input_size, data, clear)
        }

        fn store(&mut self, data: &[u8], mask: usize, ix: usize) {
            self.inner.store(data, mask, ix);
        }

        fn find_longest_match<S: Simd>(
            &mut self,
            simd: S,
            stats: &mut DictionaryStats,
            query: MatchQuery<'_>,
            out: &mut SearchResult,
        ) {
            self.inner.find_longest_match(simd, stats, query, out);
        }
    }

    /// Runs one block of `payload` through `matcher` at `quality`; `data`
    /// is the payload plus the tail the match finder may read past it.
    fn references<M: Matcher>(
        quality: QualityLevel,
        matcher: &mut M,
        data: &[u8],
        payload: usize,
    ) -> (Vec<Command>, ReferenceState) {
        let params = params(quality);
        matcher.prepare(true, payload, data, true);
        let mut state = ReferenceState::default();
        let mut commands = Vec::new();
        let level = Level::try_detect().unwrap_or_else(Level::baseline);
        let window = Window {
            data,
            mask: usize::MAX,
        };
        let span = BlockSpan {
            position: 0,
            bytes: payload as u32,
        };
        dispatch!(level, simd => create_backward_references::<_, _, false, false>(
            simd, matcher, &params, window, span, None, &mut state, &mut commands,
        ));
        (commands, state)
    }

    #[test]
    fn a_search_resumed_at_every_checkpoint_emits_the_same_commands() {
        let mut data: Vec<u8> = b"the quick brown fox jumps over the lazy dog; "
            .iter()
            .chain(b"a b c d e f g h i j k l m n o p q r s t u v w x y z ")
            .copied()
            .cycle()
            .take(20_000)
            .enumerate()
            .map(|(i, byte)| if i % 97 == 0 { (i % 251) as u8 } else { byte })
            .collect();
        let payload = data.len();
        data.extend_from_slice(&[0u8; 8]);
        // Zero still advances one position per pass.
        for interval in [0, 1, 7, 4096] {
            let expected = references(
                QualityLevel::Q3,
                &mut QuickMatcher::<{ 1 << 16 }, 1, 5, false>::new(),
                &data,
                payload,
            );
            let mut yielding = Yielding {
                inner: QuickMatcher::<{ 1 << 16 }, 1, 5, false>::new(),
                interval,
                checkpoints: 0,
            };
            let actual = references(QualityLevel::Q3, &mut yielding, &data, payload);
            assert!(yielding.checkpoints > 1, "interval {interval}");
            assert_eq!(actual.0, expected.0, "interval {interval}");
            assert_eq!(actual.1.last_insert_len, expected.1.last_insert_len);
            assert_eq!(actual.1.dist_cache, expected.1.dist_cache);

            type Q7 = BucketMatcher<false, { 1 << 15 }, 64, { (1 << 15) * 64 }>;
            let expected = references(QualityLevel::Q7, &mut Q7::new(payload), &data, payload);
            let mut yielding = Yielding {
                inner: Q7::new(payload),
                interval,
                checkpoints: 0,
            };
            let actual = references(QualityLevel::Q7, &mut yielding, &data, payload);
            assert!(yielding.checkpoints > 1, "interval {interval}");
            assert_eq!(actual.0, expected.0, "interval {interval}");
            assert_eq!(actual.1.last_insert_len, expected.1.last_insert_len);
        }
    }

    #[test]
    fn the_yielding_wrapper_stores_and_searches_through_its_matcher() {
        let data = b"abcdefgh abcdefgh ........".to_vec();
        let mut yielding = Yielding {
            inner: QuickMatcher::<{ 1 << 16 }, 1, 5, false>::new(),
            interval: 1,
            checkpoints: 0,
        };
        yielding.prepare(true, data.len(), &data, true);
        yielding.store(&data, usize::MAX, 0);
        let cache = INITIAL_DISTANCE_CACHE;
        let query = MatchQuery {
            #[cfg(feature = "experimental")]
            custom: None,
            data: &data,
            window: &data,
            mask: usize::MAX,
            cache: &cache,
            cur_ix: 9,
            max_length: data.len() - 9,
            max_backward: 9,
            position_offset: 0,
            dictionary_limit: 9,
            gap: 0,
            max_distance: u32::MAX as usize,
        };
        let mut out = SearchResult::empty();
        let mut stats = DictionaryStats::default();
        let level = Level::try_detect().unwrap_or_else(Level::baseline);
        dispatch!(level, simd => yielding.find_longest_match(simd, &mut stats, query, &mut out));
        assert_eq!((out.distance, out.len), (9, 9));
    }

    #[test]
    fn quality_five_searches_more_than_quality_four() {
        // The extensive search resets the delayed candidate length, so it can
        // pick a different, sometimes shorter but nearer, match.
        assert!(GreedyQuality::Q5.extensive_reference_search());
        assert!(!GreedyQuality::Q4.extensive_reference_search());
    }
}

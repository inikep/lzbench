//! Match finders for qualities three to nine.
//!
//! Ports `hash_longest_match_quickly_inc.h` (H3, H4, H54),
//! `hash_longest_match_inc.h` (H5), `hash_longest_match64_inc.h` (H6) and
//! `hash_forgetful_chain_inc.h` (H40, H41, H42) from the pinned reference
//! (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! Which of them runs is decided once, from the caller's parameters, by
//! [`super::params::choose_hasher`]. The hash width and the bucket count are
//! compile-time constants of the matcher type, so the hash itself is a fixed
//! shift; the candidate depth, the chain depth and the number of cached
//! distances are ordinary fields, because they only bound loops and turning
//! five bucket depths into five monomorphisations would cost far more
//! instruction cache than the bound is worth.
//!
//! Qualities seven and up probe more than four cached distances, which is
//! where [`prepare_distance_cache`] earns its keep: the extra entries are
//! near misses derived from the two freshest distances.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use fearless_simd::{Simd, SimdBase, SimdMask, u8x16, u8x32, u32x16};
use fearless_simd_macros::simd;

use super::super::cache_probe::ProbeSite;
use super::params::{BucketShape, ChainShape, HasherPlan};
use super::probe::{StoreOutlook, sample_store_outlook};
use crate::shared::constants::HASH_MUL32;
use crate::shared::dictionary::{self, DictionaryStats};
use crate::shared::fixed::{fixed_table, fixed_table_from_vec};
use crate::shared::match_len::{
    current_window, match_len_at, match_len_at_current, match_len_at_outlined, match_len_windows,
};
use crate::shared::ringbuffer::contiguous;
use crate::shared::score::{
    SearchResult, backward_reference_penalty_using_last_distance, backward_reference_score,
    backward_reference_score_using_last_distance,
};

/// Sixty-four-bit hash multiplier (`kHashMul64`).
const HASH_MUL64: u64 = 0x1FE3_5A7B_D357_9BD3;

/// How many distances a search may probe (`BROTLI_NUM_DISTANCE_SHORT_CODES`).
const NUM_DISTANCE_SHORT_CODES: usize = 16;

/// The sixteen distances a search may probe (`BROTLI_NUM_DISTANCE_SHORT_CODES`).
///
/// Only the first four are real history; the rest are near misses derived from
/// them by [`prepare_distance_cache`].
pub(crate) type DistanceCache = [i32; NUM_DISTANCE_SHORT_CODES];

/// The four cache entries the encoder actually remembers across meta-blocks.
pub(crate) const NUM_REMEMBERED_DISTANCES: usize = 4;

/// Distance cache the reference starts every stream with.
pub(crate) const INITIAL_DISTANCE_CACHE: DistanceCache =
    [4, 11, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// Fills the derived entries of the distance cache (`PrepareDistanceCache`).
///
/// A matcher that probes more than four distances also probes the values one,
/// two and three either side of the two freshest ones. They are recomputed
/// whenever the first four change, and left alone entirely when the matcher
/// only looks at those four.
#[inline]
pub(crate) fn prepare_distance_cache(cache: &mut DistanceCache, num_distances: usize) {
    if num_distances <= NUM_REMEMBERED_DISTANCES {
        return;
    }
    let last = cache[0];
    cache[4] = last - 1;
    cache[5] = last + 1;
    cache[6] = last - 2;
    cache[7] = last + 2;
    cache[8] = last - 3;
    cache[9] = last + 3;
    if num_distances > 10 {
        let next_last = cache[1];
        cache[10] = next_last - 1;
        cache[11] = next_last + 1;
        cache[12] = next_last - 2;
        cache[13] = next_last + 2;
        cache[14] = next_last - 3;
        cache[15] = next_last + 3;
    }
}

/// Reads eight little-endian bytes at `offset`, or zero past the end.
///
/// Every buffer index in this crate is below 2^32 (positions are 32-bit in
/// the format), so `offset` is masked to that width: it changes nothing for
/// a real index and lets the compiler see that `offset + 8` cannot overflow,
/// which leaves a single compare in front of the load.
#[inline(always)]
fn read_u64(data: &[u8], offset: usize) -> u64 {
    debug_assert!(offset <= u32::MAX as usize);
    let offset = offset & u32::MAX as usize;
    match data.get(offset..offset + 8) {
        Some(chunk) => u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])),
        None => 0,
    }
}

/// The longest period [`history_ranges`] looks for. A run that repeats every
/// `p` bytes stores a position under each of its keys every `p` bytes, so
/// short periods are what fill a bucket with one key; longer ones spread out.
#[cfg(not(feature = "no_std"))]
const HISTORY_RUN_MAX_PERIOD: usize = 1024;

/// Calls `store` with the half-open position ranges of `history` worth
/// storing: all of them, except inside runs that repeat a period of at most
/// [`HISTORY_RUN_MAX_PERIOD`] bytes for more than four periods, of which only
/// the last four periods are kept.
///
/// It is the rule `commit_match` applies to a stream's own copies, where a
/// copy whose distance is under a quarter of its length stores only its last
/// `4 * distance` positions: without it, history of zeros or of a short
/// repeated record fills whole buckets with one key, and every search there
/// measures each of them to the full run length. Runs are found through a
/// table of each 4-byte key's last position and confirmed eight bytes at a
/// time; a run is measured once, so the pass is linear in the history.
#[cfg(not(feature = "no_std"))]
fn history_ranges(history: &[u8], mut store: impl FnMut(usize, usize)) {
    const TABLE_BITS: u32 = 12;
    let mut last = [u32::MAX; 1 << TABLE_BITS];
    let mut keep_from = 0;
    let mut measured_until = 0;
    let mut i = 0;
    while i + 8 <= history.len() {
        let key = (read_u32(history, i).wrapping_mul(HASH_MUL32) >> (32 - TABLE_BITS)) as usize;
        let previous = last[key] as usize;
        // History is within the backward limit, so positions fit `u32`.
        last[key] = i as u32;
        let period = i.wrapping_sub(previous);
        if i >= measured_until
            && previous < i
            && period <= HISTORY_RUN_MAX_PERIOD
            && read_u64(history, previous) == read_u64(history, i)
        {
            let mut end = i + 8;
            while end + 8 <= history.len()
                && read_u64(history, end) == read_u64(history, end - period)
            {
                end += 8;
            }
            measured_until = end;
            if end - i > 4 * period {
                store(keep_from, i);
                keep_from = end - 4 * period;
                i = end;
                continue;
            }
        }
        i += 1;
    }
    store(keep_from, history.len());
}

/// Reads four little-endian bytes at `offset`, or zero past the end.
///
/// See [`read_u64`] for the offset mask.
#[inline(always)]
fn read_u32(data: &[u8], offset: usize) -> u32 {
    debug_assert!(offset <= u32::MAX as usize);
    let offset = offset & u32::MAX as usize;
    match data.get(offset..offset + 4) {
        Some(chunk) => u32::from_le_bytes(chunk.try_into().unwrap_or([0; 4])),
        None => 0,
    }
}

/// Reads one byte at `offset`, or zero past the end.
#[inline(always)]
fn read_u8(data: &[u8], offset: usize) -> u8 {
    match data.get(offset) {
        Some(&byte) => byte,
        None => 0,
    }
}

/// Everything a match finder needs from its caller, gathered once.
///
/// Passed by value: it is a handful of words, and keeping it in registers is
/// what stops the search from rebuilding it in memory at every position.
#[derive(Copy, Clone)]
pub(crate) struct MatchQuery<'a> {
    #[cfg(feature = "experimental")]
    pub(crate) custom:
        Option<&'a crate::compressor::core::rfc9841::static_index::StaticCombination>,
    /// The ring buffer being searched, tail copy and margin included.
    pub(crate) data: &'a [u8],
    /// `data` cut to the window: a position the mask admits indexes it, so
    /// a guard against the window's end is also a bounds proof.
    pub(crate) window: &'a [u8],
    /// Mask that turns an absolute position into a buffer index.
    pub(crate) mask: usize,
    /// The four distances that have short codes.
    pub(crate) cache: &'a DistanceCache,
    /// Absolute position the match would start at.
    pub(crate) cur_ix: usize,
    /// Longest match the remaining input allows.
    pub(crate) max_length: usize,
    /// Longest backward distance inside the window.
    pub(crate) max_backward: usize,
    /// Offset the stream starts at (`stream_offset`), zero for ordinary ones.
    pub(crate) position_offset: usize,
    /// Cap on the distance to the start of the stream (`max_backward_limit`).
    pub(crate) dictionary_limit: usize,
    /// Distance shift that addresses the attached dictionary (`gap`).
    pub(crate) gap: usize,
    /// Longest distance the distance alphabet can express.
    pub(crate) max_distance: usize,
}

impl MatchQuery<'_> {
    /// Returns the distance to the start of the stream, capped to the window
    /// (`dictionary_start`).
    ///
    /// Derived here rather than stored: only the dictionary probe and the
    /// attached-prefix search need it, and computing it at every position
    /// for the matcher costs an add and a compare that mostly go unused.
    #[inline(always)]
    pub(crate) fn dictionary_start(&self) -> usize {
        (self.cur_ix + self.position_offset).min(self.dictionary_limit)
    }

    /// Returns the distance at which the static dictionary begins.
    #[inline(always)]
    fn dictionary_distance(&self) -> usize {
        self.dictionary_start() + self.gap
    }

    fn search_dictionary<const SHALLOW: bool>(
        self,
        stats: &mut DictionaryStats,
        out: &mut SearchResult,
    ) {
        let data = self.data.get(self.cur_ix & self.mask..).unwrap_or_default();
        #[cfg(feature = "experimental")]
        if let Some(custom) = self.custom {
            dictionary::search_custom(
                custom,
                stats,
                data,
                self.max_length,
                self.dictionary_distance(),
                self.max_distance,
                out,
                SHALLOW,
            );
            return;
        }
        dictionary::search::<SHALLOW>(
            stats,
            data,
            self.max_length,
            self.dictionary_distance(),
            self.max_distance,
            out,
        );
    }
}

/// What [`Matcher::prepare`] did to the table, which tells a caller how to
/// leave it clean for the next stream.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum Sweep {
    /// Only the slots the first `input_size` positions hash to were cleared.
    /// Replaying the same sweep after the stream clears exactly the slots it
    /// could have dirtied, which is far cheaper than wiping the table.
    Partial,
    /// The whole table was cleared; the stream will dirty it in places no
    /// cheap sweep could find, so the next stream has to wipe it again.
    Full,
    /// The table empties itself on every `prepare` at a cost that does not
    /// depend on the stream, so nothing needs replaying and nothing is dirty.
    SelfCleaning,
}

/// A match finder's tables borrowed for one block of searches and stores.
///
/// The reference hoists its table pointers into `restrict` locals for a
/// whole block, so a store through one never makes the compiler reload the
/// others. A run is the same idea: it holds the tables as slices bound once,
/// and the hot loop stores through those rather than through the finder,
/// which would otherwise reload every field it needs at every position.
pub(crate) trait MatchRun {
    /// Bytes a candidate needs available to be hashed (`HashTypeLength`).
    const HASH_TYPE_LENGTH: usize;

    /// Bytes a store needs available (`StoreLookahead`).
    const STORE_LOOKAHEAD: usize;

    /// Records the position `ix` in the table (`Store`).
    fn store(&mut self, data: &[u8], mask: usize, ix: usize);

    /// Records every position in `start..end` (`StoreRange`).
    fn store_range(&mut self, data: &[u8], mask: usize, start: usize, end: usize);

    /// Searches for the best match at `query.cur_ix` (`FindLongestMatch`).
    ///
    /// `out` is only improved, never worsened: a search that finds nothing
    /// leaves the incoming candidate in place.
    fn find_longest_match<S: Simd>(
        &mut self,
        simd: S,
        stats: &mut DictionaryStats,
        query: MatchQuery<'_>,
        out: &mut SearchResult,
    );
}

/// A finder whose tables need no separate view runs through itself.
///
/// The forwarding methods are inlined by force so the finder's own inline
/// methods land in the search loop rather than behind a call per position.
impl<M: Matcher> MatchRun for &mut M {
    const HASH_TYPE_LENGTH: usize = M::HASH_TYPE_LENGTH;
    const STORE_LOOKAHEAD: usize = M::STORE_LOOKAHEAD;

    #[inline(always)]
    fn store(&mut self, data: &[u8], mask: usize, ix: usize) {
        M::store(self, data, mask, ix);
    }

    #[inline(always)]
    fn store_range(&mut self, data: &[u8], mask: usize, start: usize, end: usize) {
        M::store_range(self, data, mask, start, end);
    }

    #[inline(always)]
    fn find_longest_match<S: Simd>(
        &mut self,
        simd: S,
        stats: &mut DictionaryStats,
        query: MatchQuery<'_>,
        out: &mut SearchResult,
    ) {
        M::find_longest_match(self, simd, stats, query, out);
    }
}

/// A block of searches, generic over the view it runs through.
///
/// A finder whose tables can be laid out more than one way hands the visitor
/// the concrete view its current layout uses, so the visitor's loop is
/// compiled once per view rather than once over a union of them: a loop that
/// carries every layout's state at once spills the one it is actually using.
pub(crate) trait RunVisitor {
    /// What the block of searches produces.
    type Output;

    /// Runs the block through `run`.
    fn visit<R: MatchRun>(self, run: R) -> Self::Output;
}

/// A match finder over the ring buffer.
pub(crate) trait Matcher {
    /// Bytes a candidate needs available to be hashed (`HashTypeLength`).
    const HASH_TYPE_LENGTH: usize;

    /// Bytes a store needs available (`StoreLookahead`).
    const STORE_LOOKAHEAD: usize;

    /// Borrows the tables for a block of searches and stores, as the view
    /// the current layout uses; see [`RunVisitor`].
    fn visit_run<V: RunVisitor>(&mut self, visitor: V) -> V::Output;

    /// Returns how many cached distances a search probes.
    ///
    /// Mirrors the `NUM_LAST_DISTANCES_TO_CHECK` a matcher was instantiated
    /// with; [`prepare_distance_cache`] needs it to decide how much of the
    /// cache to derive.
    fn last_distances_to_check(&self) -> usize {
        NUM_REMEMBERED_DISTANCES
    }

    /// Lets the matcher reorganize its tables before a search resumes at
    /// `position`, with `remaining` bytes of the current block still to
    /// come; returns how many positions the search may advance before it
    /// calls back, or `usize::MAX` if it never has to.
    ///
    /// Only the layout changes: every candidate a search would have seen is
    /// still seen, in the same order, so the output does not depend on when
    /// or whether the matcher reorganizes.
    fn checkpoint(&mut self, _position: usize, _remaining: usize) -> usize {
        usize::MAX
    }

    /// Clears the table before the first block (`Prepare`).
    ///
    /// `clear` may be false only when construction or a previous reset sweep
    /// already left every reachable entry empty. Sweep selection still returns
    /// the information needed to clear the next stream.
    ///
    /// Returns which [`Sweep`] was taken, so a caller that wants to reuse the
    /// matcher for another stream knows how to leave the table clean.
    fn prepare(&mut self, one_shot: bool, input_size: usize, data: &[u8], clear: bool) -> Sweep;

    /// Records the position `ix` in the table (`Store`).
    fn store(&mut self, data: &[u8], mask: usize, ix: usize);

    /// Records every position in `start..end` (`StoreRange`).
    fn store_range(&mut self, data: &[u8], mask: usize, start: usize, end: usize);

    /// Records the three positions that span the previous block boundary.
    ///
    /// Mirrors `StitchToPreviousBlock`: their hashes need bytes from both
    /// blocks, so they could not be computed when the previous block was
    /// processed.
    fn stitch_to_previous_block(
        &mut self,
        num_bytes: usize,
        position: usize,
        data: &[u8],
        mask: usize,
    ) {
        if num_bytes >= Self::HASH_TYPE_LENGTH - 1 && position >= 3 {
            self.store(data, mask, position - 3);
            self.store(data, mask, position - 2);
            self.store(data, mask, position - 1);
        }
    }

    /// Searches for the best match at `query.cur_ix` (`FindLongestMatch`).
    ///
    /// `out` is only improved, never worsened: a search that finds nothing
    /// leaves the incoming candidate in place.
    fn find_longest_match<S: Simd>(
        &mut self,
        simd: S,
        stats: &mut DictionaryStats,
        query: MatchQuery<'_>,
        out: &mut SearchResult,
    );
}

/// Sparse logical slots for short streams. Missing keys have the same zero
/// position as a freshly initialized full quick-matcher table.
///
/// An entry packs the slot above the position plus one, so zero is empty
/// and the map is a zeroed word per entry: half the clearing of a two-word
/// entry, which for a kilobyte of input is most of what the map costs. The
/// packing holds for slots below 2^17 — every quick shape — and positions
/// below [`SMALL_SLOTS_MAX_INPUT`], which [`QuickMatcher::prepare`] only
/// admits for a one-shot stream of at most that many bytes.
#[derive(Default)]
struct SmallSlots {
    entries: Vec<u32>,
    count: usize,
}

/// Bits of a [`SmallSlots`] entry holding the position plus one.
const SMALL_SLOTS_VALUE_BITS: u32 = 15;

/// Mask of the position field of a [`SmallSlots`] entry.
const SMALL_SLOTS_VALUE_MASK: u32 = (1 << SMALL_SLOTS_VALUE_BITS) - 1;

/// Longest one-shot stream whose positions a [`SmallSlots`] entry can hold.
const SMALL_SLOTS_MAX_INPUT: usize = (1 << SMALL_SLOTS_VALUE_BITS) - 1;

impl SmallSlots {
    /// Returns the index `slot` occupies, or the empty index it would take.
    ///
    /// Open addressing over a power-of-two table: the index is always masked
    /// to the table's length, so the loop carries no bounds check. The map
    /// is never more than half full, so an empty index is always found.
    #[inline(always)]
    fn find(&self, slot: usize) -> usize {
        let mask = self.entries.len().wrapping_sub(1);
        let tag = (slot as u32) << SMALL_SLOTS_VALUE_BITS;
        let mut index = slot & mask;
        loop {
            let entry = self.entries[index & mask];
            if entry == 0 || entry & !SMALL_SLOTS_VALUE_MASK == tag {
                return index & mask;
            }
            index = index.wrapping_add(1);
        }
    }

    #[inline(always)]
    fn read(&self, slot: usize) -> u32 {
        if self.entries.is_empty() {
            return 0;
        }
        let entry = self.entries[self.find(slot) & (self.entries.len() - 1)];
        if entry == 0 {
            0
        } else {
            (entry & SMALL_SLOTS_VALUE_MASK) - 1
        }
    }

    #[inline(always)]
    fn write(&mut self, slot: usize, value: u32) {
        self.replace(slot, value);
    }

    /// Writes `value` into `slot`, returning what it held, or zero.
    #[inline(always)]
    fn replace(&mut self, slot: usize, value: u32) -> u32 {
        if 2 * (self.count + 1) > self.entries.len() {
            self.grow();
        }
        let index = self.find(slot) & (self.entries.len() - 1);
        let entry = &mut self.entries[index];
        let previous = if *entry == 0 {
            self.count += 1;
            0
        } else {
            (*entry & SMALL_SLOTS_VALUE_MASK) - 1
        };
        *entry = ((slot as u32) << SMALL_SLOTS_VALUE_BITS)
            | ((value & SMALL_SLOTS_VALUE_MASK).wrapping_add(1) & SMALL_SLOTS_VALUE_MASK);
        previous
    }

    fn grow(&mut self) {
        let size = (self.entries.len() * 2).max(32);
        let previous = ::core::mem::replace(&mut self.entries, vec![0; size]);
        self.count = 0;
        for entry in previous {
            if entry != 0 {
                self.write(
                    (entry >> SMALL_SLOTS_VALUE_BITS) as usize,
                    (entry & SMALL_SLOTS_VALUE_MASK) - 1,
                );
            }
        }
    }

    /// Empties the map, sized so `input_size` distinct keys never grow it.
    fn reset(&mut self, input_size: usize) {
        let size = (2 * input_size).next_power_of_two().max(32);
        self.entries.fill(0);
        if self.entries.len() < size {
            self.entries.resize(size, 0);
        }
        self.count = 0;
    }
}

/// Slot storage a quick run indexes through: the full table or the map.
pub(crate) trait QuickSlots {
    /// Reads slot `slot`.
    fn read(&self, slot: usize) -> u32;

    /// Writes slot `slot`.
    fn write(&mut self, slot: usize, value: u32);

    /// Writes slot `slot`, returning what it held.
    ///
    /// The single-slot shapes read and then overwrite the same slot at
    /// every position; the map finds it once for both.
    fn replace(&mut self, slot: usize, value: u32) -> u32;
}

/// The full table: an array, so a slot masked to its size needs no check.
impl<const N: usize> QuickSlots for &mut [u32; N] {
    #[inline(always)]
    fn read(&self, slot: usize) -> u32 {
        self[slot & (N - 1)]
    }

    #[inline(always)]
    fn write(&mut self, slot: usize, value: u32) {
        self[slot & (N - 1)] = value;
    }

    #[inline(always)]
    fn replace(&mut self, slot: usize, value: u32) -> u32 {
        ::core::mem::replace(&mut self[slot & (N - 1)], value)
    }
}

impl QuickSlots for &mut SmallSlots {
    #[inline(always)]
    fn read(&self, slot: usize) -> u32 {
        SmallSlots::read(self, slot)
    }

    #[inline(always)]
    fn write(&mut self, slot: usize, value: u32) {
        SmallSlots::write(self, slot, value);
    }

    #[inline(always)]
    fn replace(&mut self, slot: usize, value: u32) -> u32 {
        SmallSlots::replace(self, slot, value)
    }
}

/// Quick match finder with one hash bucket sweep (`HashLongestMatchQuickly`).
///
/// `BUCKETS` sizes the table, `SWEEP_BITS` says how many neighbouring slots
/// one hash owns, `HASH_LEN` how many bytes feed the hash, and
/// `USE_DICTIONARY` whether a miss falls back to the static dictionary. The
/// bucket count is compile-time so the hash shift and every slot mask are
/// immediates and the table is an array whose indexing needs no check.
///
/// A `COMPACT` matcher is built for an input of at most two kibibytes. Its
/// first stream indexes a small map instead of the table, so a one-shot call
/// never clears the table; from its second stream on it allocates the table,
/// which a warmed matcher then clears by the partial sweep like any other.
pub(crate) struct QuickMatcher<
    const BUCKETS: usize,
    const SWEEP_BITS: u32,
    const HASH_LEN: u32,
    const USE_DICTIONARY: bool,
    const COMPACT: bool = false,
> {
    /// The full table; `None` until the first stream prepares the matcher,
    /// and while a compact matcher is on its first stream.
    ///
    /// A session of unknown length builds its encoder before it knows its
    /// size, so the table waits for [`Self::materialize`] once the size is
    /// inferred, and the encoder can still be swapped for the compact matcher
    /// without having paid for a table it drops.
    buckets: Option<Box<[u32; BUCKETS]>>,
    /// The map a compact matcher's first stream indexes through.
    compact: SmallSlots,
}

impl<
    const BUCKETS: usize,
    const SWEEP_BITS: u32,
    const HASH_LEN: u32,
    const USE_DICTIONARY: bool,
    const COMPACT: bool,
> QuickMatcher<BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY, COMPACT>
{
    /// Base-2 logarithm of the number of slots.
    const BUCKET_BITS: u32 = BUCKETS.trailing_zeros();

    /// Mask that keeps a slot index inside the table.
    const BUCKET_MASK: usize = BUCKETS - 1;

    /// Number of slots one hash sweeps over.
    const SWEEP: usize = 1usize << SWEEP_BITS;

    /// Mask picking the slot of the sweep a position is stored into.
    const SWEEP_MASK: usize = (Self::SWEEP - 1) << 3;

    /// Returns the bytes this match finder keeps allocated.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.buckets.as_ref().map_or(0, |table| table.len()) * size_of::<u32>()
            + self.compact.entries.capacity() * size_of::<u32>()
    }

    /// Creates an empty matcher; its table is allocated when first prepared.
    pub(crate) fn new() -> Self {
        Self {
            buckets: None,
            compact: SmallSlots::default(),
        }
    }

    /// Allocates a full matcher's table now rather than at its first
    /// preparation; a compact matcher, or one that already holds its table,
    /// is left alone.
    ///
    /// Where the table is allocated matters as much as whether: taken before
    /// the window, a fresh table gets zero pages from the allocator; taken at
    /// the first preparation, after the window, it reuses freed heap that
    /// `calloc` has to clear, which measured 5-18% slower on 8-50 KiB
    /// streams at qualities 2 and 3.
    fn materialize(&mut self) {
        if !COMPACT && self.buckets.is_none() {
            self.buckets = Self::table();
        }
    }

    /// Allocates a zeroed table.
    fn table() -> Option<Box<[u32; BUCKETS]>> {
        Some(fixed_table(0))
    }

    /// Returns the bucket of the bytes at `offset` (`HashBytes`).
    #[inline(always)]
    fn hash(data: &[u8], offset: usize) -> usize {
        let value = read_u64(data, offset) << (64 - 8 * HASH_LEN as u64);
        (value.wrapping_mul(HASH_MUL64) >> (64 - Self::BUCKET_BITS)) as usize
    }
}

/// The matcher a [`QuickRun`] views, for its constants.
type QuickShape<
    const BUCKETS: usize,
    const SWEEP_BITS: u32,
    const HASH_LEN: u32,
    const USE_DICTIONARY: bool,
> = QuickMatcher<BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY>;

/// A [`QuickMatcher`]'s slots borrowed for one block; see [`MatchRun`].
pub(crate) struct QuickRun<
    T: QuickSlots,
    const BUCKETS: usize,
    const SWEEP_BITS: u32,
    const HASH_LEN: u32,
    const USE_DICTIONARY: bool,
> {
    slots: T,
}

impl<
    T: QuickSlots,
    const BUCKETS: usize,
    const SWEEP_BITS: u32,
    const HASH_LEN: u32,
    const USE_DICTIONARY: bool,
> QuickRun<T, BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY>
{
    /// Returns the slot the position `ix` hashing to `key` is stored into.
    #[inline(always)]
    const fn slot_of(key: usize, ix: usize) -> usize {
        if QuickShape::<BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY>::SWEEP == 1 {
            key
        } else {
            (key + (ix & QuickShape::<BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY>::SWEEP_MASK))
                & QuickShape::<BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY>::BUCKET_MASK
        }
    }
}

impl<
    T: QuickSlots,
    const BUCKETS: usize,
    const SWEEP_BITS: u32,
    const HASH_LEN: u32,
    const USE_DICTIONARY: bool,
> MatchRun for QuickRun<T, BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY>
{
    const HASH_TYPE_LENGTH: usize = 8;
    const STORE_LOOKAHEAD: usize = 8;

    #[inline(always)]
    fn store(&mut self, data: &[u8], mask: usize, ix: usize) {
        let key =
            QuickShape::<BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY>::hash(data, ix & mask);
        self.slots.write(Self::slot_of(key, ix), ix as u32);
    }

    fn store_range(&mut self, data: &[u8], mask: usize, start: usize, end: usize) {
        for ix in start..end {
            self.store(data, mask, ix);
        }
    }

    #[inline(always)]
    fn find_longest_match<S: Simd>(
        &mut self,
        simd: S,
        stats: &mut DictionaryStats,
        query: MatchQuery<'_>,
        out: &mut SearchResult,
    ) {
        let data = query.data;
        let slots = &mut self.slots;
        let cur_ix_masked = query.cur_ix & query.mask;
        // The window a candidate is measured against, cut only when one
        // passes the byte compare; most positions never get that far. The
        // four-slot matchers cut it behind the measuring call instead; see
        // `match_len_at_current`.
        let cur = || current_window(data, cur_ix_masked, query.max_length);
        let measure = |prev_ix: usize| {
            if QuickShape::<BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY>::SWEEP >= 4 {
                match_len_at_current(simd, data, prev_ix, cur_ix_masked, query.max_length)
            } else {
                match_len_at_outlined(simd, data, prev_ix, cur())
            }
        };
        let best_len_in = out.len;
        let mut compare_char = read_u8(data, cur_ix_masked + best_len_in);
        let key =
            QuickShape::<BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY>::hash(data, cur_ix_masked);
        let min_score = out.score;
        let mut best_score = out.score;
        let mut best_len = best_len_in;

        out.len_code_delta = 0;

        let cached_backward = query.cache[0] as usize;
        let prev_ix = query.cur_ix.wrapping_sub(cached_backward);
        if prev_ix < query.cur_ix {
            let prev_ix = prev_ix & query.mask;
            if compare_char == read_u8(data, prev_ix + best_len) {
                let len = measure(prev_ix);
                if len >= 4 {
                    let score = backward_reference_score_using_last_distance(len);
                    if best_score < score {
                        out.len = len;
                        out.distance = cached_backward;
                        out.score = score;
                        if QuickShape::<BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY>::SWEEP == 1 {
                            slots.write(key, query.cur_ix as u32);
                            return;
                        }
                        best_len = len;
                        best_score = score;
                        compare_char = read_u8(data, cur_ix_masked + len);
                    }
                }
            }
        }

        if QuickShape::<BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY>::SWEEP == 1 {
            // Only one candidate: the store happens before the comparison, so
            // the slot always ends up holding the current position.
            let prev_ix = slots.replace(key, query.cur_ix as u32) as usize;
            let backward = query.cur_ix.wrapping_sub(prev_ix);
            let prev_ix = prev_ix & query.mask;
            if compare_char != read_u8(data, prev_ix + best_len_in) {
                return;
            }
            if backward == 0 || backward > query.max_backward {
                return;
            }
            let len = measure(prev_ix);
            if len >= 4 {
                let score = backward_reference_score(len, backward);
                if best_score < score {
                    out.len = len;
                    out.distance = backward;
                    out.score = score;
                    // A hit here is final: the reference returns rather than
                    // falling through to the dictionary.
                    return;
                }
            }
            // Anything else falls through to the dictionary search, which is
            // what `H2` — the only single-slot matcher that consults it — is
            // reached by.
        } else {
            for sweep in 0..QuickShape::<BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY>::SWEEP {
                let slot = (key + (sweep << 3))
                    & QuickShape::<BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY>::BUCKET_MASK;
                let prev_ix = slots.read(slot) as usize;
                let backward = query.cur_ix.wrapping_sub(prev_ix);
                let prev_ix = prev_ix & query.mask;
                if compare_char != read_u8(data, prev_ix + best_len) {
                    continue;
                }
                if backward == 0 || backward > query.max_backward {
                    continue;
                }
                let len = measure(prev_ix);
                if len >= 4 {
                    let score = backward_reference_score(len, backward);
                    if best_score < score {
                        best_len = len;
                        out.len = len;
                        compare_char = read_u8(data, cur_ix_masked + len);
                        best_score = score;
                        out.score = score;
                        out.distance = backward;
                    }
                }
            }
        }

        if USE_DICTIONARY && min_score == out.score {
            query.search_dictionary::<true>(stats, out);
        }
        // The sweeping variant writes its own slot last; the single-slot one
        // has already written it, which is why the reference guards this
        // store with `BUCKET_SWEEP != 1`.
        if QuickShape::<BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY>::SWEEP != 1 {
            slots.write(Self::slot_of(key, query.cur_ix), query.cur_ix as u32);
        }
    }
}

impl<
    const BUCKETS: usize,
    const SWEEP_BITS: u32,
    const HASH_LEN: u32,
    const USE_DICTIONARY: bool,
    const COMPACT: bool,
> Matcher for QuickMatcher<BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY, COMPACT>
{
    const HASH_TYPE_LENGTH: usize = 8;
    const STORE_LOOKAHEAD: usize = 8;

    fn visit_run<V: RunVisitor>(&mut self, visitor: V) -> V::Output {
        match &mut self.buckets {
            Some(table) => {
                visitor.visit(
                    QuickRun::<_, BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY> {
                        slots: &mut **table,
                    },
                )
            }
            None => visitor.visit(
                QuickRun::<_, BUCKETS, SWEEP_BITS, HASH_LEN, USE_DICTIONARY> {
                    slots: &mut self.compact,
                },
            ),
        }
    }

    fn prepare(&mut self, one_shot: bool, input_size: usize, data: &[u8], clear: bool) -> Sweep {
        // Clearing only the slots a short input can reach is far cheaper than
        // wiping the whole table, and reaches exactly the same slots the
        // search will later look at.
        let partial_prepare_threshold = BUCKETS >> 5;
        let partial = if one_shot && input_size <= partial_prepare_threshold {
            Sweep::Partial
        } else {
            Sweep::Full
        };
        let Some(table) = &mut self.buckets else {
            if !COMPACT || clear || !one_shot || input_size > SMALL_SLOTS_MAX_INPUT {
                // A full matcher's first stream, a compact matcher past its
                // first stream, or one whose stream the map cannot hold: the
                // map has no part to play, and a fresh table needs no clearing.
                self.buckets = Self::table();
                if self.buckets.is_some() {
                    self.compact = SmallSlots::default();
                    return partial;
                }
            }
            // A fresh map is sized for the input either way, so it never
            // grows and rehashes while the stream stores into it.
            if clear || self.compact.entries.is_empty() {
                self.compact.reset(input_size);
            }
            return partial;
        };
        if !clear {
            return partial;
        }
        if partial == Sweep::Partial {
            for offset in 0..input_size {
                let key = Self::hash(data, offset);
                if Self::SWEEP == 1 {
                    table[key & Self::BUCKET_MASK] = 0;
                } else {
                    for sweep in 0..Self::SWEEP {
                        table[(key + (sweep << 3)) & Self::BUCKET_MASK] = 0;
                    }
                }
            }
        } else {
            table.fill(0);
        }
        partial
    }

    fn store(&mut self, data: &[u8], mask: usize, ix: usize) {
        self.visit_run(Store {
            data,
            mask,
            start: ix,
            end: ix + 1,
        });
    }

    fn store_range(&mut self, data: &[u8], mask: usize, start: usize, end: usize) {
        self.visit_run(Store {
            data,
            mask,
            start,
            end,
        });
    }

    fn find_longest_match<S: Simd>(
        &mut self,
        simd: S,
        stats: &mut DictionaryStats,
        query: MatchQuery<'_>,
        out: &mut SearchResult,
    ) {
        self.visit_run(Search {
            simd,
            stats,
            query,
            out,
        });
    }
}

/// Input length up to which a one-shot stream indexes its buckets through a
/// small map instead of the sparse entry table.
///
/// The table costs 128 KiB (fourteen bucket bits) or 256 KiB (fifteen) to
/// zero, which is most of what a sixteen-byte call pays in total. A kilobyte
/// of input reaches at most a thousand buckets, which a map of two thousand
/// slots indexes after a sixteen-kilobyte clear.
const COMPACT_INPUT_LIMIT: usize = 1024;

/// Positions an on-demand search runs between two looks at its store rate
/// (see [`Matcher::checkpoint`]).
///
/// Long enough that a repetitive input, which stores at almost every
/// position of its first few kibibytes and then copies for the rest, has
/// usually reached its long copy before it is first judged.
const CHECKPOINT_INTERVAL: usize = 1 << 14;

/// Slots a starter block holds; a bucket's fifth store grows it to full depth.
///
/// Quality nine keeps two hundred and fifty-six positions per bucket, a
/// kilobyte each, and a short input activates a bucket for almost every
/// position it hashes. Starting small keeps that input from zeroing a
/// mebibyte it never reads.
const STARTER_SLOTS: usize = 4;

/// Bits of a sparse entry holding the wrapping store counter.
const COUNT_BITS: u32 = 16;

/// Bits of a sparse entry holding the block index and its starter flag.
///
/// Thirty-two thousand buckets of two hundred and sixty slots need
/// twenty-four bits; the flag is the twenty-fifth.
const OFFSET_BITS: u32 = 25;

/// Marks an encoded offset as a starter block.
const STARTER_FLAG: u32 = 1 << (OFFSET_BITS - 1);

/// Mask of the encoded offset inside a sparse entry.
const OFFSET_MASK: u64 = (1 << OFFSET_BITS) - 1;

/// Bit position of the generation stamp inside a sparse entry.
const GENERATION_SHIFT: u32 = COUNT_BITS + OFFSET_BITS;

/// Generations a sparse table lives through before it is wiped.
const MAX_GENERATION: u64 = (1 << (u64::BITS - GENERATION_SHIFT)) - 1;

/// Open-addressing index from bucket key to store counter and block offset.
///
/// Entries pack the key above the counter above the offset; the all-ones
/// word is empty, which no real key reaches.
#[derive(Default)]
struct KeyMap {
    entries: Vec<u64>,
    count: usize,
}

impl KeyMap {
    const EMPTY: u64 = u64::MAX;

    /// Returns the slot `key` occupies, or the empty slot it would take,
    /// with the entry found there.
    ///
    /// Open addressing over a power-of-two table: the index is always masked
    /// to the table's length, so the loop carries no bounds check. The map
    /// is never more than half full, so an empty slot is always found.
    #[inline(always)]
    fn find(&self, key: usize) -> (usize, u64) {
        let mask = self.entries.len().wrapping_sub(1);
        let mut slot = key & mask;
        loop {
            let entry = self.entries[slot & mask];
            if entry == Self::EMPTY || (entry >> 48) as usize == key {
                return (slot & mask, entry);
            }
            slot = slot.wrapping_add(1);
        }
    }

    /// Decodes the counter and offset of an entry; an empty one holds zeros.
    #[inline(always)]
    const fn decode(entry: u64) -> (u16, u32) {
        if entry == Self::EMPTY {
            (0, 0)
        } else {
            ((entry >> 32) as u16, entry as u32)
        }
    }

    /// Returns the slot a store into `key` writes and the counter and
    /// offset it holds, growing the map first when it is more than half
    /// full so the slot stays valid.
    #[inline(always)]
    fn slot_for_write(&mut self, key: usize) -> (usize, u16, u32) {
        if 2 * (self.count + 1) > self.entries.len() {
            self.grow();
        }
        let (slot, entry) = self.find(key);
        let (count, offset) = Self::decode(entry);
        (slot, count, offset)
    }

    /// Records `count` and `offset` for `key` at `slot`, which
    /// [`KeyMap::slot_for_write`] returned for that key.
    #[inline(always)]
    fn write_slot(&mut self, slot: usize, key: usize, count: u16, offset: u32) {
        let mask = self.entries.len().wrapping_sub(1);
        let entry = &mut self.entries[slot & mask];
        self.count += usize::from(*entry == Self::EMPTY);
        *entry = ((key as u64) << 48) | (u64::from(count) << 32) | u64::from(offset);
    }

    /// Returns the counter and offset stored for `key`, or zeros.
    #[inline(always)]
    #[cfg(test)]
    fn get(&self, key: usize) -> (u16, u32) {
        if self.entries.is_empty() {
            return (0, 0);
        }
        Self::decode(self.find(key).1)
    }

    /// Records `count` and `offset` for `key`, growing the map when it is
    /// more than half full.
    #[inline(always)]
    fn set(&mut self, key: usize, count: u16, offset: u32) {
        let (slot, _, _) = self.slot_for_write(key);
        self.write_slot(slot, key, count, offset);
    }

    /// Doubles the table and rehashes entries inside the resized allocation.
    fn grow(&mut self) {
        let old_len = self.entries.len();
        let empty = self
            .entries
            .iter()
            .position(|&entry| entry == Self::EMPTY)
            .unwrap_or(0);
        self.entries.resize((old_len * 2).max(64), Self::EMPTY);
        self.count = 0;
        // Start after an empty slot so wrapping clusters are visited in probe
        // order. With a doubled mask, reinsertion can reach only the new half
        // or already-visited old slots, never an unread old entry.
        for offset in 1..=old_len {
            let slot = (empty + offset) & (old_len - 1);
            let entry = ::core::mem::replace(&mut self.entries[slot], Self::EMPTY);
            if entry != Self::EMPTY {
                self.set((entry >> 48) as usize, (entry >> 32) as u16, entry as u32);
            }
        }
    }

    /// Empties the map, sized so `input_size` distinct keys never grow it.
    fn reset(&mut self, input_size: usize) {
        let size = (2 * input_size).next_power_of_two().max(64);
        self.entries.fill(Self::EMPTY);
        self.entries
            .resize(size.max(self.entries.len()), Self::EMPTY);
        self.count = 0;
    }
}

/// The reference bucket hash plus eight rejection bits below its key.
#[inline(always)]
fn hash_with_tag<const HASH64: bool, const BUCKETS: usize>(data: &[u8], offset: usize) -> usize {
    if HASH64 {
        // H6 tunes the multiplier to a five-byte match and always takes
        // fifteen bits, whatever the bucket count is.
        let hash_mul = HASH_MUL64 << (64 - 5 * 8);
        (read_u64(data, offset).wrapping_mul(hash_mul) >> (64 - 15 - 8)) as usize
    } else {
        (read_u32(data, offset).wrapping_mul(HASH_MUL32) >> (32 - BUCKETS.trailing_zeros() - 8))
            as usize
    }
}

/// Storage owned exclusively by the current bucket layout.
enum Layout<const BUCKETS: usize, const BLOCK: usize, const SLOTS: usize> {
    Compact(CompactLayout),
    Sparse(SparseLayout<BUCKETS, BLOCK>),
    Dense(DenseLayout<BUCKETS, BLOCK, SLOTS>),
}

/// Short-stream bucket heads and linked positions.
#[derive(Default)]
struct CompactLayout {
    compact: KeyMap,
    /// Low word: position; high word: one-based index of the older node.
    chain: Vec<u64>,
}

/// On-demand blocks and their generation-stamped bucket index.
struct SparseLayout<const BUCKETS: usize, const BLOCK: usize> {
    /// Fixed-size index: a masked bucket key proves every lookup in bounds.
    entries: Box<[u64; BUCKETS]>,
    /// Nonzero stamp; old entries retain their block but count as empty.
    generation: u64,
    blocks: Vec<[u32; BLOCK]>,
    block_tags: Vec<[u8; BLOCK]>,
    starters: Vec<[u32; STARTER_SLOTS]>,
    starter_tags: Vec<[u8; STARTER_SLOTS]>,
    /// Positions stored this stream, and the count and stream position at
    /// the last [`Matcher::checkpoint`]: the store rate they give decides
    /// whether the rest of the stream repays the dense table.
    stores: usize,
    checked_stores: usize,
    checked_position: usize,
}

/// Preallocated buckets; only their counters are cleared between streams.
struct DenseLayout<const BUCKETS: usize, const BLOCK: usize, const SLOTS: usize> {
    num: Box<[u16; BUCKETS]>,
    /// Flat integers preserve the allocator's zeroed-page initialization path.
    dense: Box<[u32; SLOTS]>,
    /// Absent for untagged shapes.
    dense_tags: Option<Box<[u8; SLOTS]>>,
}

/// An activated on-demand block, by index into its pool.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum BlockRef {
    /// A block of [`STARTER_SLOTS`] slots.
    Starter(usize),
    /// A block of the shape's full depth.
    Full(usize),
}

/// Returns how many cached distances a bucket of `block` slots probes.
///
/// The reference's `ChooseHasher` uses four below quality seven, ten below
/// quality nine and all sixteen at quality nine; the block depth is the
/// quality less one, so the count follows from the depth.
const fn last_distances_for(block: usize) -> usize {
    if block <= 32 {
        4
    } else if block <= 128 {
        10
    } else {
        16
    }
}

/// Bucketed match finder keeping the most recent positions per hash.
///
/// `HASH64` selects the H6 variant, which hashes eight bytes instead of four
/// and pre-filters candidates on their first four bytes. `BUCKETS` is the
/// number of buckets and `BLOCK` the number of slots per bucket, both
/// compile-time so the hash shift, every block index and every slot mask is
/// an immediate and no table bound has to be held in a register: a search
/// loop that carries a runtime depth spills the values it needs at every
/// candidate. The depth also fixes how many cached distances a search
/// probes ([`last_distances_for`]) and whether slots carry tags. `SLOTS` is
/// `BUCKETS * BLOCK`, passed separately because stable Rust cannot multiply
/// generic constants in an array type; construction checks it at compile time.
///
/// # Storage
///
/// The reference allocates every bucket's block up front and never
/// initialises it, reading a slot only below the counter that guards it.
/// Safe Rust has to initialise what it reads, so the layout follows the
/// input. A matcher built for at least [`BucketMatcher::dense_limit`] bytes
/// gets the reference's dense table: one block per bucket, zeroed once per
/// matcher and never again, with a two-byte counter per bucket cleared per
/// stream. Every other stream, including one of unknown length, activates
/// blocks on demand instead, each starting as a [`STARTER_SLOTS`]-slot block
/// until its fifth store, and indexes them through a table of packed entries
/// — generation stamp, block index, counter — that a new stream empties by
/// bumping the generation. A one-shot stream of at most
/// [`COMPACT_INPUT_LIMIT`] bytes uses neither: it links every store into a
/// per-bucket chain indexed through a small [`KeyMap`], so a sixteen-byte
/// call never zeroes the table and a store is one push.
///
/// Slots fill downwards, as the reference's tagged matchers do: the newest
/// position sits at the lowest occupied slot and older ones follow it
/// upwards, so a scan walks the block by ascending slot from the newest.
///
/// # Equivalence with the tagged reference matchers
///
/// The reference builds `H58`/`H68` in place of `H5`/`H6` when
/// `BROTLI_MAX_SIMD_QUALITY` is defined. Those variants store a one-byte tag
/// beside every position and visit only the slots whose tag matches the
/// current one. They select the same bucket — the tagged `HashBytes` merely
/// keeps eight more low bits, which the key shifts straight back off — and
/// they walk it newest to oldest, exactly as this loop does. A tag is a
/// function of the hashed bytes, so within the same bucket two positions
/// whose first four bytes agree share a tag; a slot the tag mask drops
/// differs in those four bytes, and a candidate that differs there can never
/// reach the reference's `len >= 4` acceptance test. Both matchers also stop
/// at the first candidate beyond `max_backward`, and positions grow
/// monotonically along the ring, so both stop having seen the same prefix of
/// candidates. The accepted-match sets coincide, and so do the streams. Like
/// the pinned C build, only the shallow quality five and six blocks carry
/// tags; deeper blocks measured slower with the mask. The SIMD backends use
/// the tag mask; the scalar backend and starter blocks keep the unfiltered
/// scan as an oracle.
pub(crate) struct BucketMatcher<
    const HASH64: bool,
    const BUCKETS: usize,
    const BLOCK: usize,
    const SLOTS: usize,
> {
    layout: Layout<BUCKETS, BLOCK, SLOTS>,
    /// Dense storage another finder left behind, taken by the first dense
    /// layout instead of a zeroed allocation.
    recycled: Option<(Vec<u32>, Vec<u8>)>,
    /// Total input the matcher was built for; zero when unknown.
    size_hint: usize,
    /// Streams prepared so far, saturating; the layout choice for a deep
    /// shape depends on whether the matcher has been reused.
    streams: u32,
}

impl<const HASH64: bool, const BUCKETS: usize, const BLOCK: usize, const SLOTS: usize>
    BucketMatcher<HASH64, BUCKETS, BLOCK, SLOTS>
{
    /// Whether blocks carry tags (the shallow `H58`/`H68` shapes).
    const TAGGED: bool = BLOCK <= 32;

    /// How many cached distances a search probes.
    const LAST_DISTANCES: usize = last_distances_for(BLOCK);

    /// Slots of the dense table each store the rest of a stream is expected
    /// to make has to repay before the matcher leaves its on-demand layout.
    ///
    /// Leaving costs a clear of the whole table and a copy of every live
    /// bucket, and saves the on-demand index's dependent load and block
    /// bookkeeping at every store that follows. On text a deep shape's store
    /// saves about as much as clearing thirty-two slots costs; a lower bar
    /// switched too early on incompressible and repetitive input, whose store
    /// rate falls once the random-data heuristic starts skipping or long
    /// copies stop storing, and doubled their time. A tagged shape's block
    /// is shallow, so a store saves less and the copy weighs more against
    /// its one- or two-mebibyte table: at thirty-two, forty-eight kibibytes
    /// of incompressible input switched at quality five and took half as
    /// long again, so the bar there is sixteen.
    const PROMOTION_SLOTS_PER_STORE: usize = if Self::TAGGED { 16 } else { 32 };

    /// Returns the bytes this match finder keeps allocated.
    pub(crate) const fn retained_bytes(&self) -> usize {
        let recycled = match &self.recycled {
            Some((words, tags)) => words.capacity() * size_of::<u32>() + tags.capacity(),
            None => 0,
        };
        recycled
            + match &self.layout {
                Layout::Compact(layout) => {
                    (layout.compact.entries.capacity() + layout.chain.capacity()) * size_of::<u64>()
                }
                Layout::Sparse(layout) => {
                    BUCKETS * size_of::<u64>()
                        + layout.blocks.capacity() * size_of::<[u32; BLOCK]>()
                        + layout.block_tags.capacity() * size_of::<[u8; BLOCK]>()
                        + layout.starters.capacity() * size_of::<[u32; STARTER_SLOTS]>()
                        + layout.starter_tags.capacity() * size_of::<[u8; STARTER_SLOTS]>()
                }
                Layout::Dense(layout) => {
                    layout.num.len() * size_of::<u16>()
                        + layout.dense.len() * size_of::<u32>()
                        + match layout.dense_tags {
                            Some(ref dense) => dense.len(),
                            None => 0,
                        }
                }
            }
    }

    /// Creates an empty matcher expecting `size_hint` bytes (zero if unknown).
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn new(size_hint: usize) -> Self {
        const {
            assert!(
                SLOTS == BUCKETS * BLOCK,
                "dense table must match the bucket shape"
            )
        };
        Self {
            layout: Layout::Compact(CompactLayout::default()),
            recycled: None,
            size_hint,
            streams: 0,
        }
    }

    /// Drops an on-demand layout, so the next stream starts from the compact
    /// one as a fresh matcher would.
    ///
    /// A reused sparse layout keeps every bucket earlier streams populated,
    /// which is what makes it fast on similar input; on input from another
    /// size class it holds the union of both streams' buckets. A deep shape's
    /// union of a long response with the next outgrows the default retention
    /// bound and is released every time, so for a new window the matcher
    /// starts over instead.
    fn forget_sparse(&mut self) {
        if matches!(self.layout, Layout::Sparse(_)) {
            self.layout = Layout::Compact(CompactLayout::default());
        }
    }

    /// Takes over dense storage another finder left behind.
    ///
    /// A dense table is read only below its bucket counters, which a dense
    /// layout clears before every stream, so the positions and tags can keep
    /// whatever they held.
    fn recycle(&mut self, parked: &mut ParkedTables) {
        if self.recycled.is_none() {
            self.recycled = parked.dense.take();
        }
    }

    /// Hands the dense storage over for a finder of another plan.
    fn park(self, parked: &mut ParkedTables) {
        let dense = match self.layout {
            Layout::Dense(layout) => Some((
                Vec::from(layout.dense as Box<[u32]>),
                layout
                    .dense_tags
                    .map_or_else(Vec::new, |tags| Vec::from(tags as Box<[u8]>)),
            )),
            _ => self.recycled,
        };
        if let Some(dense) = dense
            && parked
                .dense
                .as_ref()
                .is_none_or(|(words, _)| words.capacity() < dense.0.capacity())
        {
            parked.dense = Some(dense);
        }
    }

    /// The input the layout choice is made for.
    ///
    /// A stream of unknown length is taken to be long when the table is
    /// small: the reference always uses the dense table, and a stream
    /// written in pieces without a size hint would otherwise store every
    /// position through the on-demand index's dependent loads, at several
    /// times the cost of a counter and a block. The deep shapes keep the
    /// on-demand layouts for an unknown length, because their tables cost
    /// mebibytes to clear and only a long stream repays that.
    const fn expected_input(&self) -> usize {
        if self.size_hint == 0 && Self::TAGGED {
            usize::MAX
        } else {
            self.size_hint
        }
    }

    /// Shortest known input that gets the dense table.
    ///
    /// The table is zeroed once per matcher, which costs a clear or the
    /// page faults of its whole size, so it has to be small against what
    /// the matcher will see. Quality five's one mebibyte is worth it from a
    /// thirty-second of that in input: on text the on-demand index's
    /// dependent load and block bookkeeping cost a quarter of the whole
    /// search from sixteen kibibytes up, and from thirty-two the clear is
    /// repaid on text, incompressible and repetitive binary input alike.
    /// Quality six's two mebibytes take a sixteenth: below that a highly
    /// repetitive input stores so little that the clear quadruples its
    /// time, and the stores of text are caught by a mid-stream switch
    /// instead ([`Matcher::checkpoint`]). The deep shapes' eight
    /// to thirty-two mebibytes are cleared for a first stream only when the
    /// input is at least an eighth of the table: a short compressible
    /// stream stores few positions, and clearing sixteen mebibytes for it
    /// costs more than compressing it — a quarter-mebibyte of zeros at
    /// quality seven measured half the reference's speed with the table and
    /// twice it without — while the on-demand layouts touch only what it
    /// stores. A matcher on its second stream has shown it is reused, so
    /// the clear is paid once for every stream that follows, and later
    /// streams only zero the counters; from then on a sixty-fourth of the
    /// table is enough for every shape, because the on-demand index's
    /// dependent load per bucket costs more than the whole search on a
    /// quarter-mebibyte incompressible input at quality seven, and reused
    /// quality six spent 1.7 times as long on 64 KiB of it on demand as with
    /// the table. A matcher that already holds the table reuses it for every
    /// subsequent input.
    const fn dense_limit(&self) -> usize {
        let table_bytes = BUCKETS * BLOCK * size_of::<u32>();
        if self.streams != 0 {
            table_bytes / 64
        } else if !Self::TAGGED {
            table_bytes / 8
        } else if BLOCK <= 16 {
            table_bytes / 32
        } else {
            table_bytes / 16
        }
    }

    /// Shortest known input of ordinary data that gets a tagged shape's
    /// dense table: a hundred-and-twenty-eighth of it, eight kibibytes at
    /// quality five and sixteen at quality six.
    ///
    /// Cold one-shot text, markup, JSON and binary formats measured the dense
    /// table faster from there (q5 8 KiB: 0.84-0.94 of the on-demand time,
    /// q6 16 KiB: 0.86-0.98, both 0.72-0.89 from 32 KiB), well below
    /// [`Self::dense_limit`], which also has to cover the inputs
    /// [`sample_store_outlook`] now tells apart.
    const ORDINARY_DENSE_LIMIT: usize = BUCKETS * BLOCK * size_of::<u32>() / 128;

    /// Whether a stream whose first block is `block` gets the dense table.
    ///
    /// The deep shapes and streams of unknown length follow
    /// [`Self::dense_limit`] alone. A tagged shape that knows its input
    /// samples the first block once the input reaches
    /// [`Self::ORDINARY_DENSE_LIMIT`]: repetitive input never takes the
    /// table, which measured two to twelve times slower on it at every
    /// size up to 128 KiB because long copies leave nothing to store, as
    /// long as the sampled block is at least half the input — `backward65536`
    /// cycled to 256 KiB starts with runs but stores along its 64 KiB-distant
    /// copies, and lost half again its time without the table; incompressible
    /// input, a block too short to sample and repetitive input that outruns
    /// its sample keep [`Self::dense_limit`]; anything else takes the table
    /// from the lower limit.
    fn wants_dense(&self, block: &[u8]) -> bool {
        let expected = self.expected_input();
        if !Self::TAGGED || self.size_hint == 0 {
            return expected >= self.dense_limit();
        }
        if expected < Self::ORDINARY_DENSE_LIMIT {
            return false;
        }
        match sample_store_outlook(block) {
            // A first block covering at least half the input speaks for it.
            Some(StoreOutlook::Repetitive) if expected / 2 <= block.len() => false,
            Some(StoreOutlook::Ordinary) => true,
            _ => expected >= self.dense_limit(),
        }
    }

    /// Re-aims the matcher at a stream of `size_hint` bytes.
    ///
    /// The shape is fixed by the type; only the layout choice the next
    /// preparation makes depends on the hint, so a reused encoder need not
    /// be rebuilt when its input length changes.
    pub(crate) const fn retarget(&mut self, size_hint: usize) {
        self.size_hint = size_hint;
    }

    /// Resets the current layout or promotes it, transferring compatible buffers.
    ///
    /// `block` is the stream's first block, which [`Self::wants_dense`] may
    /// sample.
    fn select_layout(&mut self, one_shot: bool, input_size: usize, block: &[u8]) {
        let compact = matches!(self.layout, Layout::Compact(_))
            && one_shot
            && input_size <= COMPACT_INPUT_LIMIT;
        let dense =
            matches!(self.layout, Layout::Dense(_)) || (!compact && self.wants_dense(block));
        let previous =
            ::core::mem::replace(&mut self.layout, Layout::Compact(CompactLayout::default()));
        self.layout = match previous {
            Layout::Compact(mut layout) if compact => {
                layout.compact.reset(input_size);
                layout.chain.clear();
                layout.chain.reserve(input_size);
                Layout::Compact(layout)
            }
            Layout::Compact(layout) if !dense => Layout::Sparse(layout.into()),
            Layout::Sparse(mut layout) if !dense => {
                if layout.generation == MAX_GENERATION {
                    layout.entries.fill(0);
                    layout.generation = 1;
                    layout.blocks.clear();
                    layout.block_tags.clear();
                    layout.starters.clear();
                    layout.starter_tags.clear();
                } else {
                    layout.generation += 1;
                }
                layout.stores = 0;
                layout.checked_stores = 0;
                layout.checked_position = 0;
                Layout::Sparse(layout)
            }
            previous => {
                let mut layout = match previous {
                    Layout::Dense(layout) => layout,
                    Layout::Sparse(sparse) => {
                        let dense = if sparse.blocks.capacity() * BLOCK
                            >= sparse.starters.capacity() * STARTER_SLOTS
                        {
                            sparse.blocks.into_flattened()
                        } else {
                            sparse.starters.into_flattened()
                        };
                        let dense_tags = if sparse.block_tags.capacity() * BLOCK
                            >= sparse.starter_tags.capacity() * STARTER_SLOTS
                        {
                            sparse.block_tags.into_flattened()
                        } else {
                            sparse.starter_tags.into_flattened()
                        };
                        DenseLayout::from((dense, dense_tags))
                    }
                    Layout::Compact(_) => match self.recycled.take() {
                        Some(storage) => DenseLayout::from(storage),
                        None => DenseLayout::default(),
                    },
                };
                layout.prepare();
                Layout::Dense(layout)
            }
        };
        self.streams = self.streams.saturating_add(1);
    }
}

impl CompactLayout {
    /// Links `ix` in front of the chain of `key`, whose map entry — at
    /// `slot`, already read as `count` stores and chain head `head` — is
    /// then updated to name the new node.
    #[inline(always)]
    fn push_chain(&mut self, key: usize, slot: usize, count: u16, head: u32, ix: u32) {
        self.chain.push(u64::from(ix) | (u64::from(head) << 32));
        let node = self.chain.len() as u32;
        self.compact
            .write_slot(slot, key, count.wrapping_add(1), node);
    }
}

impl<const BUCKETS: usize, const BLOCK: usize, const SLOTS: usize> Default
    for DenseLayout<BUCKETS, BLOCK, SLOTS>
{
    fn default() -> Self {
        Self {
            num: fixed_table(0),
            dense: fixed_table(0),
            dense_tags: (BLOCK <= 32).then(|| fixed_table(0)),
        }
    }
}

impl<const BUCKETS: usize, const BLOCK: usize, const SLOTS: usize> From<(Vec<u32>, Vec<u8>)>
    for DenseLayout<BUCKETS, BLOCK, SLOTS>
{
    fn from((dense, tags): (Vec<u32>, Vec<u8>)) -> Self {
        Self {
            num: fixed_table(0),
            dense: fixed_table_from_vec(dense, 0),
            dense_tags: (BLOCK <= 32).then(|| fixed_table_from_vec(tags, 0)),
        }
    }
}

impl<const BUCKETS: usize, const BLOCK: usize, const SLOTS: usize>
    DenseLayout<BUCKETS, BLOCK, SLOTS>
{
    /// Empties counters while retaining the fixed-size tables for a new stream.
    fn prepare(&mut self) {
        self.num.fill(0);
    }

    /// Borrows the fixed tables for one block without a fallible run wrapper.
    ///
    /// The shape fixes both conversions at compile time. Optional tags stay
    /// optional in the run; an untagged shape does not read the tag pointer.
    #[inline]
    fn dense_run<const HASH64: bool>(&mut self) -> DenseRun<'_, HASH64, BUCKETS, BLOCK> {
        let Ok(dense) = self.dense.as_chunks_mut::<BLOCK>().0.try_into() else {
            unreachable!("dense positions have the bucket shape");
        };
        let tags = if BLOCK <= 32 {
            self.dense_tags.as_deref_mut().map(|tags| {
                let Ok(tags) = tags.as_chunks_mut::<BLOCK>().0.try_into() else {
                    unreachable!("dense tags have the bucket shape");
                };
                tags
            })
        } else {
            None
        };
        DenseRun {
            num: &mut self.num,
            dense,
            tags,
        }
    }
}

impl<const BUCKETS: usize, const BLOCK: usize, const SLOTS: usize>
    From<&SparseLayout<BUCKETS, BLOCK>> for DenseLayout<BUCKETS, BLOCK, SLOTS>
{
    /// Moves a stream's on-demand buckets into a fresh dense table.
    ///
    /// Every live bucket keeps its counter and its slots: a starter's four
    /// slots are the top of a full block, as when it grows. Untagged
    /// shapes only; the tagged ones never switch mid-stream.
    fn from(sparse: &SparseLayout<BUCKETS, BLOCK>) -> Self {
        let mut layout = Self::default();
        let Ok(dense) = layout.dense.as_chunks_mut::<BLOCK>().0.try_into() else {
            unreachable!("dense positions have the bucket shape");
        };
        let dense: &mut [[u32; BLOCK]; BUCKETS] = dense;
        for ((&entry, count), slots) in sparse
            .entries
            .iter()
            .zip(layout.num.iter_mut())
            .zip(dense.iter_mut())
        {
            if entry >> GENERATION_SHIFT != sparse.generation {
                continue;
            }
            let (stored, offset) = sparse.decode_entry(entry);
            *count = stored;
            match SparseLayout::<BUCKETS, BLOCK>::block(offset) {
                Some(BlockRef::Full(index)) => {
                    if let Some(block) = sparse.blocks.get(index) {
                        *slots = *block;
                    }
                }
                Some(BlockRef::Starter(index)) => {
                    if let Some(starter) = sparse.starters.get(index)
                        && let Some(top) = slots.last_chunk_mut::<STARTER_SLOTS>()
                    {
                        *top = *starter;
                    }
                }
                None => {}
            }
        }
        if let Some(tags) = layout.dense_tags.as_deref_mut() {
            let Ok(tags): Result<&mut [[u8; BLOCK]; BUCKETS], _> =
                tags.as_chunks_mut::<BLOCK>().0.try_into()
            else {
                unreachable!("dense tags have the bucket shape");
            };
            for (&entry, bucket) in sparse.entries.iter().zip(tags.iter_mut()) {
                if entry >> GENERATION_SHIFT != sparse.generation {
                    continue;
                }
                let (_, offset) = sparse.decode_entry(entry);
                match SparseLayout::<BUCKETS, BLOCK>::block(offset) {
                    Some(BlockRef::Full(index)) => {
                        if let Some(block) = sparse.block_tags.get(index) {
                            *bucket = *block;
                        }
                    }
                    Some(BlockRef::Starter(index)) => {
                        if let Some(starter) = sparse.starter_tags.get(index)
                            && let Some(top) = bucket.last_chunk_mut::<STARTER_SLOTS>()
                        {
                            *top = *starter;
                        }
                    }
                    None => {}
                }
            }
        }
        layout
    }
}

impl<const BUCKETS: usize, const BLOCK: usize> From<CompactLayout>
    for SparseLayout<BUCKETS, BLOCK>
{
    fn from(compact: CompactLayout) -> Self {
        // Keep the larger u64 allocation; discard its old map/chain encoding.
        let mut entries = if compact.compact.entries.capacity() >= compact.chain.capacity() {
            compact.compact.entries
        } else {
            compact.chain
        };
        entries.clear();
        let entries = fixed_table_from_vec(entries, 0);
        Self {
            entries,
            generation: 1,
            blocks: Vec::new(),
            block_tags: Vec::new(),
            starters: Vec::new(),
            starter_tags: Vec::new(),
            stores: 0,
            checked_stores: 0,
            checked_position: 0,
        }
    }
}

impl<const BUCKETS: usize, const BLOCK: usize> SparseLayout<BUCKETS, BLOCK> {
    const TAGGED: bool = BLOCK <= 32;

    /// Decodes a sparse entry into its counter and encoded block offset.
    /// An entry from an earlier generation still names its block but
    /// counts as empty.
    #[inline(always)]
    const fn decode_entry(&self, entry: u64) -> (u16, u32) {
        let offset = ((entry >> COUNT_BITS) & OFFSET_MASK) as u32;
        let count = if entry >> GENERATION_SHIFT == self.generation {
            entry as u16
        } else {
            0
        };
        (count, offset)
    }

    /// Encodes a counter and block offset as a sparse entry of the current
    /// generation.
    #[inline(always)]
    const fn encode_entry(&self, count: u16, offset: u32) -> u64 {
        (self.generation << GENERATION_SHIFT) | ((offset as u64) << COUNT_BITS) | count as u64
    }

    /// Decodes an on-demand offset into the block it names.
    #[inline(always)]
    const fn block(offset: u32) -> Option<BlockRef> {
        if offset == 0 {
            return None;
        }
        let index = ((offset & !STARTER_FLAG) - 1) as usize;
        if offset & STARTER_FLAG != 0 {
            Some(BlockRef::Starter(index))
        } else {
            Some(BlockRef::Full(index))
        }
    }

    /// Returns the on-demand block a store into a bucket writes, activating
    /// or growing it when `count` stores have already filled what it has.
    ///
    /// The encoded offset comes back with the block so the caller can
    /// record it; it is unchanged whenever the block had room.
    #[inline(always)]
    fn block_for_store(&mut self, count: u16, offset: u32, size_hint: usize) -> (BlockRef, u32) {
        match Self::block(offset) {
            None => {
                let index = self.starters.len();
                self.starters.push([0; STARTER_SLOTS]);
                if Self::TAGGED {
                    self.starter_tags.push([0; STARTER_SLOTS]);
                }
                (BlockRef::Starter(index), (index as u32 + 1) | STARTER_FLAG)
            }
            Some(BlockRef::Starter(index)) if usize::from(count) < STARTER_SLOTS => {
                (BlockRef::Starter(index), offset)
            }
            Some(BlockRef::Starter(index)) => {
                // Both fill downwards from the top, so the starter's slots
                // are the top slots of a full block.
                let mut block = [0u32; BLOCK];
                if let Some(starter) = self.starters.get(index)
                    && let Some(top) = block.last_chunk_mut::<STARTER_SLOTS>()
                {
                    *top = *starter;
                }
                let full = self.blocks.len();
                if full == 1024 {
                    self.reserve_populated_blocks(size_hint);
                }
                self.blocks.push(block);
                if Self::TAGGED {
                    let mut tags = [0u8; BLOCK];
                    if let Some(starter) = self.starter_tags.get(index)
                        && let Some(top) = tags.last_chunk_mut::<STARTER_SLOTS>()
                    {
                        *top = *starter;
                    }
                    self.block_tags.push(tags);
                }
                (BlockRef::Full(full), full as u32 + 1)
            }
            Some(BlockRef::Full(index)) => (BlockRef::Full(index), offset),
        }
    }

    /// Reserves once a sparse stream has demonstrated broad bucket use.
    #[cold]
    fn reserve_populated_blocks(&mut self, size_hint: usize) {
        // A size hint alone over-reserves repetitive streams. Wait until 1024
        // buckets have outgrown their starters, then round the estimate down
        // and cap it so larger pools still prove their demand incrementally.
        let expected = (size_hint / 16).min(BUCKETS).min(1 << 14);
        let expected = expected.checked_ilog2().map_or(0, |bits| 1usize << bits);
        self.blocks
            .reserve(expected.saturating_sub(self.blocks.len()));
        if Self::TAGGED {
            self.block_tags
                .reserve(expected.saturating_sub(self.block_tags.len()));
        }
    }

    /// Stores `ix` with `tag` into `key` of the sparse table.
    #[inline(always)]
    fn push(&mut self, key: usize, ix: u32, tag: u8, size_hint: usize) {
        let key = key & (BUCKETS - 1);
        let (count, offset) = self.decode_entry(self.entries[key]);
        self.push_found(key, count, offset, ix, tag, size_hint);
    }

    /// Stores `ix` with `tag` into `key` of the sparse table, whose entry
    /// was already read as `count` and `offset`.
    ///
    /// A search reads the entry before it scans the bucket and stores the
    /// searched position afterwards; this lets it do both with one lookup.
    #[inline(always)]
    fn push_found(
        &mut self,
        key: usize,
        count: u16,
        offset: u32,
        ix: u32,
        tag: u8,
        size_hint: usize,
    ) {
        self.stores += 1;
        let offset = self.push_block(count, offset, ix, tag, size_hint);
        let entry = self.encode_entry(count.wrapping_add(1), offset);
        self.entries[key & (BUCKETS - 1)] = entry;
    }

    /// Writes `ix` with `tag` into the block a bucket `count` stores deep
    /// keeps at `offset`, activating or growing it first; returns the
    /// encoded offset to record.
    #[inline(always)]
    fn push_block(&mut self, count: u16, offset: u32, ix: u32, tag: u8, size_hint: usize) -> u32 {
        let (block, offset) = self.block_for_store(count, offset, size_hint);
        match block {
            BlockRef::Starter(index) => {
                let slot = !usize::from(count) & (STARTER_SLOTS - 1);
                if let Some(block) = self.starters.get_mut(index) {
                    block[slot] = ix;
                }
                if Self::TAGGED
                    && let Some(tags) = self.starter_tags.get_mut(index)
                {
                    tags[slot] = tag;
                }
            }
            BlockRef::Full(index) => {
                let slot = !usize::from(count) & (BLOCK - 1);
                if let Some(block) = self.blocks.get_mut(index) {
                    block[slot] = ix;
                }
                if Self::TAGGED
                    && let Some(tags) = self.block_tags.get_mut(index)
                {
                    tags[slot] = tag;
                }
            }
        }
        offset
    }

    /// Searches the sparse blocks; see [`MatchRun::find_longest_match`].
    #[inline(always)]
    fn search<S: Simd, const HASH64: bool>(
        &mut self,
        simd: S,
        stats: &mut DictionaryStats,
        query: MatchQuery<'_>,
        out: &mut SearchResult,
        size_hint: usize,
    ) {
        let cur_ix_masked = query.cur_ix & query.mask;
        let min_score = out.score;
        let hash = hash_with_tag::<HASH64, BUCKETS>(query.data, cur_ix_masked);
        let key = hash >> 8;
        let tag = hash as u8;
        let (count, offset) = self.decode_entry(self.entries[key & (BUCKETS - 1)]);
        match Self::block(offset) {
            Some(BlockRef::Full(index)) => {
                let bucket = self.blocks.get(index).unwrap_or(&[0; BLOCK]);
                let tags = if Self::TAGGED {
                    self.block_tags.get(index)
                } else {
                    None
                };
                search_bucket::<S, HASH64, BLOCK, BLOCK>(
                    simd, &query, out, tag, count, bucket, tags,
                );
            }
            Some(BlockRef::Starter(index)) => {
                let bucket = self.starters.get(index).unwrap_or(&[0; STARTER_SLOTS]);
                search_bucket::<S, HASH64, STARTER_SLOTS, BLOCK>(
                    simd, &query, out, tag, count, bucket, None,
                );
            }
            None => {
                search_bucket::<S, HASH64, STARTER_SLOTS, BLOCK>(
                    simd,
                    &query,
                    out,
                    tag,
                    0,
                    &[0; STARTER_SLOTS],
                    None,
                );
            }
        }
        self.push_found(key, count, offset, query.cur_ix as u32, tag, size_hint);
        if min_score == out.score {
            query.search_dictionary::<false>(stats, out);
        }
    }
}

/// Stores `ix` with `tag` into bucket `key` of a dense table.
///
/// The tables are arrays, so with the key masked to the bucket count every
/// index is in range by construction and no check survives.
#[inline(always)]
fn dense_store<const BUCKETS: usize, const BLOCK: usize>(
    num: &mut [u16; BUCKETS],
    dense: &mut [[u32; BLOCK]; BUCKETS],
    tags: Option<&mut [[u8; BLOCK]; BUCKETS]>,
    key: usize,
    ix: u32,
    tag: u8,
) {
    let key = key & (BUCKETS - 1);
    let count = &mut num[key];
    let current = *count;
    *count = current.wrapping_add(1);
    let slot = !usize::from(current) & (BLOCK - 1);
    if BLOCK > 32 {
        dense[key][slot] = ix;
        if let Some(tags) = tags {
            tags[key][slot] = tag;
        }
        return;
    }
    // The shallow shapes store a tag beside every position: one flat index
    // then addresses both tables, as the reference's does, where two
    // bucket rows cost an address computation each. The mask is the table
    // size, a power of two, so it proves the index in range.
    const { assert!((BUCKETS * BLOCK).is_power_of_two()) };
    let index = (key * BLOCK + slot) & (BUCKETS * BLOCK - 1);
    if let Some(entry) = dense.as_flattened_mut().get_mut(index) {
        *entry = ix;
    }
    if let Some(entry) = tags.and_then(|tags| tags.as_flattened_mut().get_mut(index)) {
        *entry = tag;
    }
}

/// Rotates a raw tag-equality mask so that bit `age` names the slot `age`
/// stores older than the newest one, dropping slots no store has filled.
///
/// `bits` is the block capacity, `available` how many of its slots hold
/// positions. Slots fill downwards, so the newest position sits at `newest`
/// and older ones follow it upwards, wrapping to the bottom: a rotation
/// right by `newest` within `bits` lanes brings the newest slot to bit zero
/// and each older one to the bit after it, so ascending bits of the result
/// walk the block newest to oldest.
#[inline(always)]
fn rotate_candidates(equal: u32, bits: u32, newest: u32, available: u32) -> u32 {
    let lanes = u32::MAX.checked_shr(32 - bits).unwrap_or(0);
    let filled = u32::MAX.checked_shr(32 - available).unwrap_or(0);
    let masked = equal & lanes;
    let rotated = (masked >> newest) | (masked.checked_shl(bits - newest).unwrap_or(0) & lanes);
    rotated & filled
}

/// One bit per slot of `tags` whose byte equals `tag`.
///
/// The scalar backend and any block too short for a vector compare report
/// every slot, which keeps the unfiltered scan as an independent oracle for
/// the mask. The block width is a constant, so only one arm survives.
#[inline(always)]
fn tag_equality<S: Simd, const N: usize>(simd: S, tags: &[u8; N], tag: u8) -> u32 {
    if simd.level().is_fallback() {
        return u32::MAX;
    }
    if N == 32
        && let Some(bytes) = tags.first_chunk::<32>()
    {
        return u8x32::load_array_ref(simd, bytes)
            .simd_eq(u8x32::splat(simd, tag))
            .to_bitmask() as u32;
    }
    if N == 16
        && let Some(bytes) = tags.first_chunk::<16>()
    {
        return u32::from(
            u8x16::load_array_ref(simd, bytes)
                .simd_eq(u8x16::splat(simd, tag))
                .to_bitmask() as u16,
        );
    }
    u32::MAX
}

/// The running best of one bucket scan: what every candidate is checked
/// against before it is measured.
///
/// Two words, passed and returned by value, so the candidate loop keeps
/// them in registers.
#[derive(Copy, Clone)]
struct Best {
    /// Kept narrow: an index built from it and a ring position cannot
    /// overflow, which is what lets the candidate reads go unchecked.
    len: u32,
    /// The four bytes ending one past `len`, which a candidate has to
    /// reproduce before it is worth measuring.
    word: u32,
}

/// The match one search has found so far, held in locals until the search
/// ends so the result it is written to never lives in memory mid-search.
#[derive(Copy, Clone)]
struct Found {
    /// Length of the match; zero while nothing has been found.
    len: usize,
    distance: usize,
    /// Score to beat; the incoming result's score while nothing has been
    /// found.
    score: usize,
}

/// Measures the cached-distance candidate at `prev_ix`; returns its length
/// and score when it beats `best_score`, and `None` otherwise.
///
/// Out of line on purpose, like [`accept_candidate`]: the probe loop that
/// calls it keeps only its own filter in registers. `#[simd]` is what keeps
/// it there; see [`accept_candidate`]. It returns by value
/// rather than writing the result, so the result never has to live in
/// memory during the search.
/// A winning score is positive: its nonzero niche keeps the optional pair
/// in two words. Masked ring positions use at most 31 bits and block lengths
/// at most 24; narrow arguments expose those bounds to the optimizer.
#[inline(always)]
#[simd]
fn accept_cached<S: Simd>(
    simd: S,
    data: &[u8],
    cur_ix_masked: u32,
    max_length: u32,
    prev_ix: u32,
    index: usize,
    best_score: usize,
) -> Option<(usize, NonZeroUsize)> {
    let cur_ix_masked = cur_ix_masked as usize;
    let max_length = max_length as usize;
    let prev_ix = prev_ix as usize;
    let len = match_len_at(
        simd,
        data,
        prev_ix,
        current_window(data, cur_ix_masked, max_length),
    );
    // Two-byte matches are only worth scoring for the two freshest cached
    // distances; anything shorter never wins. Written as one comparison:
    // `len >= 3 || (len == 2 && index < 2)`.
    if len + usize::from(index < 2) >= 3 {
        let mut score = backward_reference_score_using_last_distance(len);
        if best_score < score {
            if index != 0 {
                score -= backward_reference_penalty_using_last_distance(index);
            }
            if best_score < score {
                return NonZeroUsize::new(score).map(|score| (len, score));
            }
        }
    }
    None
}

/// Measures the bucket candidate at `prev_ix`, which has already reproduced
/// the four bytes of [`Best::word`]; returns the new running best and its
/// score when it beats `best_score`, and `None` otherwise.
///
/// Out of line on purpose. The candidate loop is a filter — a distance
/// check and a four-byte compare — that rejects most of what it sees, and
/// it runs in registers only while it holds a dozen values at most. The
/// measurement brings the scan loop and the current window with it;
/// inlined, those compete for the loop's registers and the compiler
/// reloads the loop's own invariants from the stack at every candidate.
/// The out-of-line body is the level's feature-enabled `#[simd]` entry: the
/// wrapper inlines into the loop, which calls that entry directly, and the
/// vector compares inline into it. An `#[inline(never)]` body without the
/// level's features would instead call every vector operation above SSE2.
/// It returns by value rather than writing the result, so the result never
/// has to live in memory during the search.
/// Like [`accept_cached`], it uses a positive-score niche and narrow physical
/// ring indices; logical stream positions are not narrowed here.
#[inline(always)]
#[simd]
fn accept_candidate<S: Simd, const HASH64: bool>(
    simd: S,
    data: &[u8],
    cur_ix_masked: u32,
    max_length: u32,
    prev_ix: u32,
    backward: usize,
    best_score: usize,
) -> Option<(Best, NonZeroUsize)> {
    let cur_ix_masked = cur_ix_masked as usize;
    let max_length = max_length as usize;
    let prev_ix = prev_ix as usize;
    let cur = current_window(data, cur_ix_masked, max_length);
    let left = data.get(prev_ix..prev_ix + cur.len())?;
    let len = if HASH64 {
        if left.first_chunk::<4>() != cur.first_chunk::<4>() {
            return None;
        }
        match (left.get(4..), cur.get(4..)) {
            (Some(left), Some(cur)) => match_len_windows(simd, left, cur) + 4,
            _ => return None,
        }
    } else {
        let len = match_len_windows(simd, left, cur);
        if len < 4 {
            return None;
        }
        len
    };
    let score = backward_reference_score(len, backward);
    if best_score < score {
        let best = Best {
            len: len as u32,
            word: read_u32(data, cur_ix_masked + len - 3),
        };
        return NonZeroUsize::new(score).map(|score| (best, score));
    }
    None
}

/// What one search holds fixed while it walks a bucket.
///
/// Passed by value into every candidate check: a handful of words the
/// compiler keeps in registers once the check is inlined, where a reference
/// to a larger query would make it reload each field it needs.
#[derive(Copy, Clone)]
struct BucketScan<'a, S> {
    /// The token the measurement scans with; zero-sized.
    simd: S,
    /// The ring buffer, tail copy and margin included.
    data: &'a [u8],
    /// `data` cut to the window; see [`MatchQuery::window`].
    window: &'a [u8],
    cur_ix: usize,
    mask: usize,
    max_backward: usize,
    /// Longest match the remaining input allows.
    max_length: usize,
}

/// Judges one bucket candidate at `prev_ix`; `false` ends the scan.
///
/// The window is the ring buffer cut to the mask, so the guard against its
/// end is the reference's mask guard and a bounds proof at once.
#[inline(always)]
fn consider<S: Simd, const HASH64: bool>(
    scan: BucketScan<'_, S>,
    prev_ix: u32,
    best: &mut Best,
    found: &mut Found,
) -> bool {
    let backward = scan.cur_ix.wrapping_sub(prev_ix as usize);
    if backward > scan.max_backward {
        return false;
    }
    let prev_ix = prev_ix as usize & scan.mask;
    // The four bytes ending one past `best.len`. Both operands are below
    // 2^32 — the offset is a wrapped `u32` on purpose — so the end cannot
    // overflow and the guard against the window's end bounds the read.
    let start = prev_ix + best.len.wrapping_sub(3) as usize;
    let Some(word) = scan.window.get(start..start + 4) else {
        return true;
    };
    if best.word != u32::from_le_bytes(word.try_into().unwrap_or([0; 4])) {
        return true;
    }
    if let Some((accepted, score)) = accept_candidate::<S, HASH64>(
        scan.simd,
        scan.data,
        (scan.cur_ix & scan.mask) as u32,
        scan.max_length as u32,
        prev_ix as u32,
        backward,
        found.score,
    ) {
        *best = accepted;
        *found = Found {
            len: accepted.len as usize,
            distance: backward,
            score: score.get(),
        };
    }
    true
}

/// Runs the cached-distance probes and the bucket scan for one search.
///
/// `bucket` is the bucket's `N` slots, `count` stores deep; `tags` is its
/// tags, or `None` for an untagged shape; `BLOCK` is the shape's full depth,
/// which fixes how many cached distances are probed first. Shared by every
/// layout so the reference's decision order lives in one place.
#[inline(always)]
fn search_bucket<S: Simd, const HASH64: bool, const N: usize, const BLOCK: usize>(
    simd: S,
    query: &MatchQuery<'_>,
    out: &mut SearchResult,
    tag: u8,
    count: u16,
    bucket: &[u32; N],
    tags: Option<&[u8; N]>,
) {
    let data = query.data;
    let window = query.window;
    let mask = query.mask;
    let cur_ix = query.cur_ix;
    let max_backward = query.max_backward;
    let max_length = query.max_length;
    let available = usize::from(count).min(N);
    // Stores fill downwards from the top of a block, so the `age`-th
    // newest position sits `age` slots above the newest one, wrapping.
    let newest = !usize::from(count.wrapping_sub(1)) & (N - 1);
    // Fetch the bucket's tags before the cache probes so the miss overlaps
    // them, as the reference's prefetch does.
    let equal = match tags {
        Some(tags) if available != 0 => tag_equality::<S, N>(simd, tags, tag),
        _ => u32::MAX,
    };

    let (best_len, mut found) = probe_last_distances::<S, BLOCK>(simd, query, out);

    if available != 0 {
        scan_bucket::<S, HASH64, N>(
            BucketScan {
                simd,
                data,
                window,
                cur_ix,
                mask,
                max_backward,
                max_length,
            },
            bucket,
            equal,
            newest,
            available,
            best_len,
            &mut found,
        );
    }
    write_back(out, found);
}

/// Runs the cached-distance probes and a compact chain walk for one search.
///
/// `head` names the newest node of the bucket's chain and `count` how many
/// stores the bucket has seen; the walk visits the newest `BLOCK` of them in
/// the order a block scan would, judging each with the same filter.
#[inline(always)]
fn search_chain<S: Simd, const HASH64: bool, const BLOCK: usize>(
    simd: S,
    query: &MatchQuery<'_>,
    out: &mut SearchResult,
    count: u16,
    head: u32,
    chain: &[u64],
) {
    let (best_len, mut found) = probe_last_distances::<S, BLOCK>(simd, query, out);
    let mut remaining = usize::from(count).min(BLOCK);
    let mut link = head as usize;
    if remaining != 0 && link != 0 {
        let data = query.data;
        let scan = BucketScan {
            simd,
            data,
            window: query.window,
            cur_ix: query.cur_ix,
            mask: query.mask,
            max_backward: query.max_backward,
            max_length: query.max_length,
        };
        let cur_ix_masked = query.cur_ix & query.mask;
        let mut best = Best {
            len: best_len as u32,
            word: read_u32(data, cur_ix_masked + best_len - 3),
        };
        while remaining != 0
            && let Some(&node) = chain.get(link.wrapping_sub(1))
        {
            if !consider::<S, HASH64>(scan, node as u32, &mut best, &mut found) {
                break;
            }
            link = (node >> 32) as usize;
            remaining -= 1;
        }
    }
    write_back(out, found);
}

/// Records what a search found; an empty result leaves the score alone.
#[inline(always)]
fn write_back(out: &mut SearchResult, found: Found) {
    out.len = found.len;
    out.len_code_delta = 0;
    if found.len != 0 {
        out.distance = found.distance;
        out.score = found.score;
    }
}

/// [`probe_last_distances`] for shapes that probe the derived distances.
///
/// Every code the scalar filter would admit is in the vector filter's
/// candidate set, so visiting those codes in order — and filtering the rest
/// again whenever an accepted copy moves `best_len` — makes the reference's
/// decisions. The cache must hold [`prepare_distance_cache`]'s derived
/// entries: the byte runs are read around the two freshest distances.
#[inline(always)]
fn probe_derived_distances<S: Simd, const BLOCK: usize>(
    simd: S,
    query: &MatchQuery<'_>,
    best_len: &mut usize,
    found: &mut Found,
) {
    let count = last_distances_for(BLOCK).min(NUM_DISTANCE_SHORT_CODES);
    let cache = query.cache;
    let [d0, d1, d2, d3, ..] = *cache;
    debug_assert!({
        let mut derived = *cache;
        prepare_distance_cache(&mut derived, count);
        derived[..count] == cache[..count]
    });
    let remembered = [d0, d1, d2, d3];
    let window = query.window;
    let mask = query.mask;
    let cur_ix = query.cur_ix;
    let cur_ix_masked = cur_ix & mask;
    let max_backward = query.max_backward;
    let site = ProbeSite {
        window,
        cur_ix,
        mask,
        // The scalar test `prev_ix >= cur_ix || backward > max_backward`
        // rejects exactly the distances outside `1..=min(cur_ix, max_backward)`.
        max_distance: cur_ix.min(max_backward),
    };
    let distances = u32x16::load_array(simd, cache.map(|distance| distance as u32));
    let probed = (1u32 << count) - 1;
    let mut next_index = 0u32;
    'probes: while let Some(&continuation) = window.get(cur_ix_masked + *best_len) {
        let candidates = if count > 10 {
            site.candidates::<S, true>(simd, distances, &remembered, *best_len, continuation)
        } else {
            site.candidates::<S, false>(simd, distances, &remembered, *best_len, continuation)
        };
        let mut pending =
            u32::from(candidates.agree & candidates.valid) & probed & (u32::MAX << next_index);
        while pending != 0 {
            let index = pending.trailing_zeros() as usize;
            pending &= pending - 1;
            let backward = cache[index] as usize;
            let prev_ix = cur_ix.wrapping_sub(backward);
            if prev_ix >= cur_ix || backward > max_backward {
                continue;
            }
            let prev_ix = prev_ix & mask;
            let prev_at = prev_ix + *best_len;
            if prev_at >= window.len() || window[prev_at] != continuation {
                continue;
            }
            if let Some((len, score)) = accept_cached(
                simd,
                query.data,
                cur_ix_masked as u32,
                query.max_length as u32,
                prev_ix as u32,
                index,
                found.score,
            ) {
                *best_len = len;
                *found = Found {
                    len,
                    distance: backward,
                    score: score.get(),
                };
                next_index = index as u32 + 1;
                continue 'probes;
            }
        }
        break;
    }
}

/// Tries the cached distances before any bucket is consulted.
///
/// Returns the length the bucket scan's compare offset starts from —
/// raised to three so the scan can compare four bytes unconditionally —
/// and the running result. The incoming length only seeds the probes'
/// compare offset; the result starts empty.
#[inline(always)]
fn probe_last_distances<S: Simd, const BLOCK: usize>(
    simd: S,
    query: &MatchQuery<'_>,
    out: &SearchResult,
) -> (usize, Found) {
    let data = query.data;
    let window = query.window;
    let mask = query.mask;
    let cur_ix = query.cur_ix;
    let cur_ix_masked = cur_ix & mask;
    let max_backward = query.max_backward;
    let max_length = query.max_length;
    let mut best_len = out.len;
    let mut found = Found {
        len: 0,
        distance: 0,
        score: out.score,
    };

    let count = last_distances_for(BLOCK).min(NUM_DISTANCE_SHORT_CODES);
    // Shapes probing the derived distances filter them all with one vector
    // compare. The scalar backend keeps the plain loop as the unfiltered
    // oracle, and four probes are too few to pay for the setup.
    if count > NUM_REMEMBERED_DISTANCES && !simd.level().is_fallback() {
        probe_derived_distances::<S, BLOCK>(simd, query, &mut best_len, &mut found);
    } else {
        // The probe count is a constant, so this loop unrolls.
        for index in 0..count {
            let backward = query.cache[index] as usize;
            let prev_ix = cur_ix.wrapping_sub(backward);
            if prev_ix >= cur_ix || backward > max_backward {
                continue;
            }
            let prev_ix = prev_ix & mask;
            // The reference guards against the mask; the window ends there, so
            // the same guards prove the two byte reads in bounds.
            let cur_at = cur_ix_masked + best_len;
            if cur_at >= window.len() {
                break;
            }
            let prev_at = prev_ix + best_len;
            if prev_at >= window.len() || window[cur_at] != window[prev_at] {
                continue;
            }
            if let Some((len, score)) = accept_cached(
                simd,
                data,
                cur_ix_masked as u32,
                max_length as u32,
                prev_ix as u32,
                index,
                found.score,
            ) {
                best_len = len;
                found = Found {
                    len,
                    distance: backward,
                    score: score.get(),
                };
            }
        }
    }
    // Raising the floor to three lets the bucket loop compare four bytes
    // unconditionally.
    if best_len < 3 {
        best_len = 3;
    }
    (best_len, found)
}

/// Walks a bucket's candidates newest to oldest, updating `found`.
///
/// `equal` is the tag mask, `newest` the slot of the newest position, and
/// `available` how many slots hold positions.
#[inline(always)]
fn scan_bucket<S: Simd, const HASH64: bool, const N: usize>(
    scan: BucketScan<'_, S>,
    bucket: &[u32; N],
    equal: u32,
    newest: usize,
    available: usize,
    best_len: usize,
    found: &mut Found,
) {
    let data = scan.data;
    let cur_ix_masked = scan.cur_ix & scan.mask;
    let mut best = Best {
        len: best_len as u32,
        word: read_u32(data, cur_ix_masked + best_len - 3),
    };
    if N <= 32 {
        // Rotating the mask so bit zero is the newest slot turns the walk
        // into one loop over ascending bits, each `age` bits above the
        // newest, wrapping; the reference rotates its mask the same way.
        let mut ages = rotate_candidates(equal, N as u32, newest as u32, available as u32);
        while ages != 0 {
            let age = ages.trailing_zeros() as usize;
            ages &= ages - 1;
            let slot = (newest + age) & (N - 1);
            let Some(&prev_ix) = bucket.get(slot) else {
                break;
            };
            if !consider::<S, HASH64>(scan, prev_ix, &mut best, found) {
                break;
            }
        }
    } else {
        let mut slot = newest;
        let mut remaining = available;
        while remaining != 0 {
            let prev_ix = bucket[slot & (N - 1)];
            if !consider::<S, HASH64>(scan, prev_ix, &mut best, found) {
                break;
            }
            slot = (slot + 1) & (N - 1);
            remaining -= 1;
        }
    }
}

/// The dense tables of a [`BucketMatcher`], bound as arrays for a block.
///
/// Arrays rather than slices: their sizes are the matcher's constants, so
/// the run carries three pointers and no bounds, and every index the key
/// or a slot mask produces is in range by construction.
pub(crate) struct DenseRun<'a, const HASH64: bool, const BUCKETS: usize, const BLOCK: usize> {
    num: &'a mut [u16; BUCKETS],
    dense: &'a mut [[u32; BLOCK]; BUCKETS],
    tags: Option<&'a mut [[u8; BLOCK]; BUCKETS]>,
}

/// Sparse storage and the current reservation hint, borrowed once per block.
struct SparseRun<'a, const HASH64: bool, const BUCKETS: usize, const BLOCK: usize> {
    layout: &'a mut SparseLayout<BUCKETS, BLOCK>,
    size_hint: usize,
}

/// Compact storage borrowed once per block.
struct CompactRun<'a, const HASH64: bool, const BUCKETS: usize, const BLOCK: usize>(
    &'a mut CompactLayout,
);

impl<const HASH64: bool, const BUCKETS: usize, const BLOCK: usize> MatchRun
    for SparseRun<'_, HASH64, BUCKETS, BLOCK>
{
    const HASH_TYPE_LENGTH: usize = if HASH64 { 8 } else { 4 };
    const STORE_LOOKAHEAD: usize = Self::HASH_TYPE_LENGTH;

    #[inline(always)]
    fn store(&mut self, data: &[u8], mask: usize, ix: usize) {
        let hash = hash_with_tag::<HASH64, BUCKETS>(data, ix & mask);
        self.layout
            .push(hash >> 8, ix as u32, hash as u8, self.size_hint);
    }

    fn store_range(&mut self, data: &[u8], mask: usize, start: usize, end: usize) {
        for ix in start..end {
            self.store(data, mask, ix);
        }
    }

    #[inline(always)]
    fn find_longest_match<S: Simd>(
        &mut self,
        simd: S,
        stats: &mut DictionaryStats,
        query: MatchQuery<'_>,
        out: &mut SearchResult,
    ) {
        self.layout
            .search::<S, HASH64>(simd, stats, query, out, self.size_hint);
    }
}

impl<const HASH64: bool, const BUCKETS: usize, const BLOCK: usize> MatchRun
    for CompactRun<'_, HASH64, BUCKETS, BLOCK>
{
    const HASH_TYPE_LENGTH: usize = if HASH64 { 8 } else { 4 };
    const STORE_LOOKAHEAD: usize = Self::HASH_TYPE_LENGTH;

    #[inline(always)]
    fn store(&mut self, data: &[u8], mask: usize, ix: usize) {
        let hash = hash_with_tag::<HASH64, BUCKETS>(data, ix & mask);
        let key = hash >> 8;
        let (slot, count, head) = self.0.compact.slot_for_write(key);
        self.0.push_chain(key, slot, count, head, ix as u32);
    }

    fn store_range(&mut self, data: &[u8], mask: usize, start: usize, end: usize) {
        for ix in start..end {
            self.store(data, mask, ix);
        }
    }

    #[inline(always)]
    fn find_longest_match<S: Simd>(
        &mut self,
        simd: S,
        stats: &mut DictionaryStats,
        query: MatchQuery<'_>,
        out: &mut SearchResult,
    ) {
        let min_score = out.score;
        let hash = hash_with_tag::<HASH64, BUCKETS>(query.data, query.cur_ix & query.mask);
        let key = hash >> 8;
        let (slot, count, head) = self.0.compact.slot_for_write(key);
        search_chain::<S, HASH64, BLOCK>(simd, &query, out, count, head, &self.0.chain);
        self.0
            .push_chain(key, slot, count, head, query.cur_ix as u32);
        if min_score == out.score {
            query.search_dictionary::<false>(stats, out);
        }
    }
}

impl<const HASH64: bool, const BUCKETS: usize, const BLOCK: usize> MatchRun
    for DenseRun<'_, HASH64, BUCKETS, BLOCK>
{
    const HASH_TYPE_LENGTH: usize = if HASH64 { 8 } else { 4 };
    const STORE_LOOKAHEAD: usize = Self::HASH_TYPE_LENGTH;

    #[inline(always)]
    fn store(&mut self, data: &[u8], mask: usize, ix: usize) {
        let hash = hash_with_tag::<HASH64, BUCKETS>(data, ix & mask);
        dense_store(
            self.num,
            self.dense,
            self.tags.as_deref_mut(),
            hash >> 8,
            ix as u32,
            hash as u8,
        );
    }

    fn store_range(&mut self, data: &[u8], mask: usize, start: usize, end: usize) {
        // A range that does not wrap the ring hashes consecutive bytes, so
        // one slice covers every eight-byte read and none needs a check of
        // its own. The reads may run past the mask into the tail copy, as
        // `store`'s do; a range that wraps would read the tail where `store`
        // reads the ring's start, and takes the loop below. Only the deep,
        // untagged shapes take this path: the shallow tagged ones measured
        // slower with it.
        let masked = start & mask;
        let count = end.saturating_sub(start);
        if BLOCK > 32
            && masked + count <= mask.saturating_add(1)
            && let Some(bytes) = data.get(masked..masked + count + 7)
        {
            let mut tags = self.tags.as_deref_mut();
            for (ix, word) in (start..end).zip(bytes.windows(8)) {
                let hash = hash_with_tag::<HASH64, BUCKETS>(word, 0);
                dense_store(
                    self.num,
                    self.dense,
                    tags.as_deref_mut(),
                    hash >> 8,
                    ix as u32,
                    hash as u8,
                );
            }
            return;
        }
        for ix in start..end {
            self.store(data, mask, ix);
        }
    }

    #[inline(always)]
    fn find_longest_match<S: Simd>(
        &mut self,
        simd: S,
        stats: &mut DictionaryStats,
        query: MatchQuery<'_>,
        out: &mut SearchResult,
    ) {
        let cur_ix_masked = query.cur_ix & query.mask;
        let min_score = out.score;
        let hash = hash_with_tag::<HASH64, BUCKETS>(query.data, cur_ix_masked);
        let key = (hash >> 8) & (BUCKETS - 1);
        let tag = hash as u8;
        let count = self.num[key];
        let bucket = &self.dense[key];
        let tags = self.tags.as_deref().map(|tags| &tags[key]);
        search_bucket::<S, HASH64, BLOCK, BLOCK>(simd, &query, out, tag, count, bucket, tags);
        dense_store(
            self.num,
            self.dense,
            self.tags.as_deref_mut(),
            key,
            query.cur_ix as u32,
            tag,
        );
        if min_score == out.score {
            query.search_dictionary::<false>(stats, out);
        }
    }
}

impl<const HASH64: bool, const BUCKETS: usize, const BLOCK: usize, const SLOTS: usize> Matcher
    for BucketMatcher<HASH64, BUCKETS, BLOCK, SLOTS>
{
    const HASH_TYPE_LENGTH: usize = if HASH64 { 8 } else { 4 };
    const STORE_LOOKAHEAD: usize = Self::HASH_TYPE_LENGTH;

    fn visit_run<V: RunVisitor>(&mut self, visitor: V) -> V::Output {
        match &mut self.layout {
            Layout::Dense(layout) => visitor.visit(layout.dense_run::<HASH64>()),
            Layout::Compact(layout) => visitor.visit(CompactRun::<HASH64, BUCKETS, BLOCK>(layout)),
            Layout::Sparse(layout) => visitor.visit(SparseRun::<HASH64, BUCKETS, BLOCK> {
                layout,
                size_hint: self.size_hint,
            }),
        }
    }

    fn last_distances_to_check(&self) -> usize {
        Self::LAST_DISTANCES
    }

    fn checkpoint(&mut self, position: usize, remaining: usize) -> usize {
        let Layout::Sparse(sparse) = &mut self.layout else {
            return usize::MAX;
        };
        let stores = sparse.stores - sparse.checked_stores;
        let span = position.saturating_sub(sparse.checked_position);
        sparse.checked_stores = sparse.stores;
        sparse.checked_position = position;
        // The size hint covers the rest of the stream; without one only
        // the current block is known to follow.
        let remaining = self.size_hint.saturating_sub(position).max(remaining);
        // Stores the rest of the stream will make at the rate of the last
        // interval, against the table's slots.
        if span != 0
            && stores.saturating_mul(remaining)
                >= span.saturating_mul(SLOTS / Self::PROMOTION_SLOTS_PER_STORE)
        {
            self.layout = Layout::Dense(DenseLayout::from(&*sparse));
            return usize::MAX;
        }
        CHECKPOINT_INTERVAL
    }

    fn prepare(&mut self, one_shot: bool, input_size: usize, data: &[u8], _clear: bool) -> Sweep {
        // Every layout empties itself in time that does not depend on what
        // the stream stored: the dense counters are a fixed memset, the
        // sparse table a generation bump, and the compact map is sized by
        // the input.
        self.select_layout(one_shot, input_size, data.get(..input_size).unwrap_or(data));
        Sweep::SelfCleaning
    }

    fn store(&mut self, data: &[u8], mask: usize, ix: usize) {
        self.visit_run(Store {
            data,
            mask,
            start: ix,
            end: ix + 1,
        });
    }

    fn store_range(&mut self, data: &[u8], mask: usize, start: usize, end: usize) {
        self.visit_run(Store {
            data,
            mask,
            start,
            end,
        });
    }

    fn find_longest_match<S: Simd>(
        &mut self,
        simd: S,
        stats: &mut DictionaryStats,
        query: MatchQuery<'_>,
        out: &mut SearchResult,
    ) {
        self.visit_run(Search {
            simd,
            stats,
            query,
            out,
        });
    }
}

/// A [`RunVisitor`] that stores a range of positions.
struct Store<'a> {
    data: &'a [u8],
    mask: usize,
    start: usize,
    end: usize,
}

impl RunVisitor for Store<'_> {
    type Output = ();

    fn visit<R: MatchRun>(self, mut run: R) {
        run.store_range(self.data, self.mask, self.start, self.end);
    }
}

/// A [`RunVisitor`] that runs one search.
struct Search<'a, S> {
    simd: S,
    stats: &'a mut DictionaryStats,
    query: MatchQuery<'a>,
    out: &'a mut SearchResult,
}

impl<S: Simd> RunVisitor for Search<'_, S> {
    type Output = ();

    fn visit<R: MatchRun>(self, mut run: R) {
        run.find_longest_match(self.simd, self.stats, self.query, self.out);
    }
}

/// Number of buckets the forgetful chains hash into (`BUCKET_BITS` 15).
const CHAIN_BUCKET_BITS: u32 = 15;

/// Number of buckets the forgetful chain hashes into.
const CHAIN_BUCKET_SIZE: usize = 1 << CHAIN_BUCKET_BITS;

/// Address value that terminates a chain after its first node.
///
/// Positions never reach three gibibytes plus sixty-four mebibytes, so a
/// bucket seeded with this always produces a delta larger than any window.
const CHAIN_EMPTY_ADDR: u32 = 0xCCCC_CCCC;

/// Head value the partial preparation seeds a bucket with.
const CHAIN_EMPTY_HEAD: u16 = 0xCCCC;

/// Storage a match finder leaves behind when its encoder switches plans.
///
/// A pooled encoder serving responses on both sides of a plan boundary —
/// short and long inputs at qualities five to nine, or inputs of known and
/// unknown length — switches between the chain finders and the bucket
/// finders at every response. Built from scratch, either zeroes its largest
/// table, which costs more than compressing a short response. Neither needs
/// it zeroed when reused: a dense table is read only below its bucket
/// counters and chain slots only through their addresses, and a new finder
/// clears both. So the buffers wait here for the next finder of their kind.
#[derive(Default)]
pub(crate) struct ParkedTables {
    /// Positions and tags of a dense bucket table.
    dense: Option<(Vec<u32>, Vec<u8>)>,
    /// Slot storage of a chain finder.
    chain_slots: Vec<ChainSlot>,
}

impl ParkedTables {
    /// Returns the bytes kept for a finder of another plan.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.dense.as_ref().map_or(0, |(words, tags)| {
            words.capacity() * size_of::<u32>() + tags.capacity()
        }) + self.chain_slots.capacity() * size_of::<ChainSlot>()
    }
}

/// Returns the forgetful-chain bucket of a four-byte word (`HashBytes`).
#[inline(always)]
const fn hash_chain_word(word: u32) -> usize {
    (word.wrapping_mul(HASH_MUL32) >> (32 - CHAIN_BUCKET_BITS)) as usize
}

/// One node of a forgetful chain.
#[derive(Copy, Clone, Debug, Default)]
struct ChainSlot {
    delta: u16,
    next: u16,
}

/// Forgetful-chain match finder (`HashForgetfulChain`: H40, H41, H42).
///
/// Chains share storage banks, so old nodes are overwritten rather than freed
/// and several chains may end up sharing a tail. A one-byte truncated hash
/// rejects cached-distance candidates before they are compared.
///
/// `NUM_BANKS` and `BANK_BITS` are compile-time because they decide the bank
/// index arithmetic in the inner hop loop: H40 and H41 keep one bank of
/// 65,536 slots, H42 five hundred and twelve banks of 512.
pub(crate) struct ChainMatcher<const NUM_BANKS: usize, const BANK_BITS: u32, const LAST: usize> {
    addr: Box<[u32; CHAIN_BUCKET_SIZE]>,
    head: Box<[u16; CHAIN_BUCKET_SIZE]>,
    tiny_hash: Box<[u8; 1 << 16]>,
    slots: Vec<ChainSlot>,
    /// Compact bank offsets plus one, retained across logical resets.
    bank_offsets: Box<[u32; NUM_BANKS]>,
    // Heap-allocated rather than a `[u16; NUM_BANKS]` field: H42 needs five
    // hundred and twelve of these, and inlining a kibibyte would make every
    // other `MatchFinder` variant carry the same footprint.
    free_slot_idx: Box<[u16; NUM_BANKS]>,
    max_hops: usize,
}

impl<const NUM_BANKS: usize, const BANK_BITS: u32, const LAST: usize>
    ChainMatcher<NUM_BANKS, BANK_BITS, LAST>
{
    /// Slots one bank holds (`BANK_SIZE`).
    const BANK_SIZE: usize = 1usize << BANK_BITS;

    /// Mask that keeps a slot index inside its bank.
    const BANK_MASK: usize = Self::BANK_SIZE - 1;

    /// Mask that maps a bucket key onto a bank.
    const BANK_SELECT: usize = NUM_BANKS - 1;

    /// Creates an empty chain table of the shape `shape` describes.
    /// Returns the bytes this match finder keeps allocated.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.addr.len() * size_of::<u32>()
            + self.head.len() * size_of::<u16>()
            + self.tiny_hash.len()
            + self.slots.capacity() * size_of::<ChainSlot>()
            + self.bank_offsets.len() * size_of::<u32>()
            + self.free_slot_idx.len() * size_of::<u16>()
    }

    pub(crate) fn new(shape: ChainShape) -> Self {
        Self::with_slots(shape, Vec::new())
    }

    /// Creates an empty chain table that takes over `slots`, the slot storage
    /// a finder of another plan left behind.
    ///
    /// The slots keep whatever they held: a chain is only entered through
    /// `addr`, which starts empty, so no slot is read before this matcher
    /// has written it. Reusing them spares the zeroing a fresh bank costs.
    fn with_slots(shape: ChainShape, mut slots: Vec<ChainSlot>) -> Self {
        debug_assert_eq!(shape.num_banks, NUM_BANKS);
        debug_assert_eq!(shape.bank_bits, BANK_BITS);
        debug_assert_eq!(shape.last_distances, LAST);
        let reuse_bank = NUM_BANKS == 1 && slots.len() >= Self::BANK_SIZE;
        if reuse_bank {
            slots.truncate(Self::BANK_SIZE);
        } else {
            // Lazily activated banks are appended from the start; their
            // positions decide nothing but where a bank lives.
            slots.clear();
        }
        let mut matcher = Self {
            addr: fixed_table(CHAIN_EMPTY_ADDR),
            head: fixed_table(0),
            tiny_hash: fixed_table(0),
            slots,
            bank_offsets: fixed_table(0),
            free_slot_idx: fixed_table(0),
            max_hops: shape.max_hops,
        };
        if reuse_bank {
            matcher.bank_offsets[0] = 1;
        } else if NUM_BANKS == 1 {
            // Every store reaches the only bank, so it is materialized up
            // front and `activate_bank` needs no growth path for it.
            matcher.activate_new_bank(0);
        }
        matcher
    }

    /// Records position `ix`, whose four bytes hash to `key` (`Store` after
    /// its `HashBytes`).
    #[inline(always)]
    fn store_key(&mut self, key: usize, ix: usize) {
        let bank = key & Self::BANK_SELECT;
        let bank_base = self.activate_bank(bank);
        let free = self.free_slot_idx.get_mut(bank).map_or(0u16, |slot| {
            let current = *slot;
            *slot = current.wrapping_add(1);
            current
        });
        let idx = usize::from(free) & Self::BANK_MASK;
        let previous = self.addr.get(key).copied().unwrap_or(CHAIN_EMPTY_ADDR);
        let delta = ix.wrapping_sub(previous as usize);
        if let Some(slot) = self.tiny_hash.get_mut(ix as u16 as usize) {
            *slot = key as u8;
        }
        let delta = if delta > 0xFFFF { 0xFFFF } else { delta as u16 };
        let head = self.head.get(key).copied().unwrap_or(0);
        if let Some(slot) = self.slots.get_mut(bank_base + idx) {
            slot.delta = delta;
            slot.next = head;
        }
        if let Some(slot) = self.addr.get_mut(key) {
            *slot = ix as u32;
        }
        if let Some(slot) = self.head.get_mut(key) {
            *slot = idx as u16;
        }
    }

    /// Hands the slot storage over for a finder of another plan.
    fn park(self, parked: &mut ParkedTables) {
        if self.slots.capacity() > parked.chain_slots.capacity() {
            parked.chain_slots = self.slots;
        }
    }

    /// Returns the bucket of the bytes at `offset` (`HashBytes`).
    #[inline(always)]
    fn hash(data: &[u8], offset: usize) -> usize {
        hash_chain_word(read_u32(data, offset))
    }

    /// Returns where `bank` starts in `slots`, materializing it on first use
    /// without changing its circular slot numbering.
    ///
    /// The growth path lives out of line: a call that may reallocate the
    /// slots, reachable from inside the search and store loops, would make
    /// the compiler reload every table pointer after each store. A single
    /// bank is materialized by [`Self::new`], so for H40 and H41 there is no
    /// such call at all.
    #[inline(always)]
    fn activate_bank(&mut self, bank: usize) -> usize {
        if NUM_BANKS == 1 {
            debug_assert_eq!(self.bank_offsets[0], 1);
            return 0;
        }
        let offset = self.bank_offsets[bank];
        if offset != 0 {
            return (offset - 1) as usize;
        }
        self.activate_new_bank(bank)
    }

    /// Appends the storage of a bank used for the first time.
    #[cold]
    #[inline(never)]
    fn activate_new_bank(&mut self, bank: usize) -> usize {
        let start = self.slots.len();
        self.slots
            .resize(start + Self::BANK_SIZE, ChainSlot::default());
        self.bank_offsets[bank] = start as u32 + 1;
        start
    }
}

impl<const NUM_BANKS: usize, const BANK_BITS: u32, const LAST: usize> Matcher
    for ChainMatcher<NUM_BANKS, BANK_BITS, LAST>
{
    const HASH_TYPE_LENGTH: usize = 4;
    const STORE_LOOKAHEAD: usize = 4;

    fn visit_run<V: RunVisitor>(&mut self, visitor: V) -> V::Output {
        visitor.visit(self)
    }

    fn last_distances_to_check(&self) -> usize {
        LAST
    }

    fn prepare(&mut self, one_shot: bool, input_size: usize, data: &[u8], clear: bool) -> Sweep {
        let partial_prepare_threshold = CHAIN_BUCKET_SIZE >> 6;
        let partial = if one_shot && input_size <= partial_prepare_threshold {
            Sweep::Partial
        } else {
            Sweep::Full
        };
        if !clear {
            return partial;
        }
        if partial == Sweep::Partial {
            for offset in 0..input_size {
                let bucket = Self::hash(data, offset);
                if let Some(slot) = self.addr.get_mut(bucket) {
                    *slot = CHAIN_EMPTY_ADDR;
                }
                if let Some(slot) = self.head.get_mut(bucket) {
                    *slot = CHAIN_EMPTY_HEAD;
                }
            }
        } else {
            self.addr.fill(CHAIN_EMPTY_ADDR);
            self.head.fill(0);
        }
        self.tiny_hash.fill(0);
        self.free_slot_idx.fill(0);
        // `slots` is left alone: a chain is only entered through `addr`, and
        // every entry this cleared now reads as empty.
        partial
    }

    #[inline(always)]
    fn store(&mut self, data: &[u8], mask: usize, ix: usize) {
        self.store_key(Self::hash(data, ix & mask), ix);
    }

    /// Records every position in `start..end` (`StoreRange`).
    ///
    /// A match's positions usually lie in one stretch of the window, so the
    /// words they hash are read from one slice: its bounds are checked once,
    /// rather than once per position as [`Self::store`] has to.
    #[inline(always)]
    fn store_range(&mut self, data: &[u8], mask: usize, start: usize, end: usize) {
        let len = end.saturating_sub(start);
        let words = contiguous(data, start, mask, len)
            .and_then(|_| data.get(start & mask..(start & mask) + len + 3));
        let Some(words) = words else {
            for ix in start..end {
                self.store(data, mask, ix);
            }
            return;
        };
        for (ix, word) in (start..end).zip(words.windows(4)) {
            let Ok(word) = <[u8; 4]>::try_from(word) else {
                continue;
            };
            self.store_key(hash_chain_word(u32::from_le_bytes(word)), ix);
        }
    }

    #[inline(always)]
    fn find_longest_match<S: Simd>(
        &mut self,
        simd: S,
        stats: &mut DictionaryStats,
        query: MatchQuery<'_>,
        out: &mut SearchResult,
    ) {
        let data = query.data;
        let mask = query.mask;
        let cur_ix_masked = query.cur_ix & mask;
        let cur = || current_window(data, cur_ix_masked, query.max_length);
        let min_score = out.score;
        let mut best_score = out.score;
        let mut best_len = out.len;
        let key = Self::hash(data, cur_ix_masked);
        let tiny_hash = key as u8;

        out.len = 0;
        out.len_code_delta = 0;

        for index in 0..LAST {
            let backward = query.cache[index] as usize;
            let prev_ix = query.cur_ix.wrapping_sub(backward);
            // Distance code zero is worth trying even for a two-byte match, so
            // it skips the truncated-hash rejection.
            if index > 0
                && self
                    .tiny_hash
                    .get(prev_ix as u16 as usize)
                    .copied()
                    .unwrap_or(0)
                    != tiny_hash
            {
                continue;
            }
            if prev_ix >= query.cur_ix || backward > query.max_backward {
                continue;
            }
            let prev_ix = prev_ix & mask;
            let len = match_len_at(simd, data, prev_ix, cur());
            if len >= 2 {
                let mut score = backward_reference_score_using_last_distance(len);
                if best_score < score {
                    if index != 0 {
                        score -= backward_reference_penalty_using_last_distance(index);
                    }
                    if best_score < score {
                        best_score = score;
                        best_len = len;
                        out.len = best_len;
                        out.distance = backward;
                        out.score = best_score;
                    }
                }
            }
        }
        if best_len < 3 {
            best_len = 3;
        }

        let bank = key & Self::BANK_SELECT;
        let bank_base = self.activate_bank(bank);
        // A candidate beats `best_len` only if it agrees with the current
        // position on the four bytes that end at offset `best_len`. The
        // current side of that test only changes with `best_len`, so it is
        // read here rather than at every hop. `None` means offset `best_len`
        // lies outside the window: no candidate can be longer from then on,
        // so the walk can stop.
        //
        // Both sides are read from `window`, the ring buffer cut to the mask:
        // `at < window.len()` is the reference's `at <= ring_buffer_mask`
        // test, and also proves the four-byte read in bounds. A position the
        // mask admits but the buffer does not hold yet cannot be matched past
        // `best_len` either, so skipping it changes nothing.
        let window = query.window;
        // `tail` is the window from offset `best_len - 3` on, so a candidate
        // at `prev_ix` agrees on those four bytes exactly when
        // `tail[prev_ix..prev_ix + 4]` equals the current position's word,
        // and that read is in bounds exactly when `prev_ix + best_len` is
        // inside the window. Holding the slice instead of `best_len` and the
        // whole window keeps the hop loop to fewer live values.
        let tail_of = |best_len: usize| window.get(best_len - 3..).unwrap_or_default();
        let word_at = |tail: &[u8], at: usize| {
            tail.get(at..at + 4)
                .and_then(<[u8]>::first_chunk::<4>)
                .map(|word| u32::from_le_bytes(*word))
        };
        let mut tail = tail_of(best_len);
        // A bank is materialized whole, so this slice always has `BANK_SIZE`
        // entries; with that length known, the masked hop index needs no
        // bounds check.
        if let (Some(bank_slots), Some(mut word)) = (
            self.slots.get(bank_base..bank_base + Self::BANK_SIZE),
            word_at(tail, cur_ix_masked),
        ) {
            let mut backward = 0usize;
            let mut delta = query
                .cur_ix
                .wrapping_sub(self.addr.get(key).copied().unwrap_or(CHAIN_EMPTY_ADDR) as usize);
            let mut slot = usize::from(self.head.get(key).copied().unwrap_or(0));
            // Counted down, as the reference's `while (hops--)`: one register
            // instead of a counter and its bound.
            let mut hops = self.max_hops;
            while hops != 0 {
                hops -= 1;
                let last = slot;
                backward = backward.wrapping_add(delta);
                if backward > query.max_backward {
                    break;
                }
                let prev_ix = (query.cur_ix.wrapping_sub(backward)) & mask;
                let node = bank_slots
                    .get(last & Self::BANK_MASK)
                    .copied()
                    .unwrap_or_default();
                slot = usize::from(node.next);
                delta = usize::from(node.delta);
                if word_at(tail, prev_ix) != Some(word) {
                    continue;
                }
                let len = match_len_at(simd, data, prev_ix, cur());
                if len >= 4 {
                    let score = backward_reference_score(len, backward);
                    if best_score < score {
                        best_score = score;
                        best_len = len;
                        out.len = best_len;
                        out.distance = backward;
                        out.score = best_score;
                        tail = tail_of(best_len);
                        // Past the window, no later candidate can be longer.
                        let Some(next) = word_at(tail, cur_ix_masked) else {
                            break;
                        };
                        word = next;
                    }
                }
            }
        }
        self.store(data, mask, query.cur_ix);

        if out.score == min_score {
            query.search_dictionary::<false>(stats, out);
        }
    }
}

/// The match finder a stream is using, chosen once from its parameters.
///
/// The tagged reference matchers `H58` and `H68` are not separate variants:
/// they are byte-for-byte equivalent to `H5` and `H6`, as argued on
/// [`BucketMatcher`].
pub(crate) enum MatchFinder {
    /// Short-input storage with the H2 hash and candidate order.
    H2Small(QuickMatcher<{ 1 << 16 }, 0, 5, true, true>),
    /// Short-input storage with the H3 hash and candidate order.
    H3Small(QuickMatcher<{ 1 << 16 }, 1, 5, false, true>),
    /// Short-input storage with the H4 hash and candidate order.
    H4Small(QuickMatcher<{ 1 << 17 }, 2, 5, true, true>),

    /// Quality 2: one candidate slot per bucket, with a dictionary probe.
    H2(QuickMatcher<{ 1 << 16 }, 0, 5, true>),
    /// Quality 3.
    H3(QuickMatcher<{ 1 << 16 }, 1, 5, false>),
    /// Quality 4, small inputs.
    H4(QuickMatcher<{ 1 << 17 }, 2, 5, true>),
    /// Quality 4, large inputs.
    H54(QuickMatcher<{ 1 << 20 }, 2, 7, false>),
    /// Qualities 5 and 6, small windows: `H40`.
    H40(ChainMatcher<1, 16, 4>),
    /// Qualities 7 and 8, small windows: `H41`.
    H41(ChainMatcher<1, 16, 10>),
    /// Quality 9, small windows: `H42`.
    H42(ChainMatcher<512, 9, 16>),
    /// Quality 5, ordinary inputs: fourteen bucket bits, sixteen slots.
    H5Q5(BucketMatcher<false, { 1 << 14 }, 16, { (1 << 14) * 16 }>),
    /// Quality 6, ordinary inputs: fourteen bucket bits, thirty-two slots.
    H5Q6(BucketMatcher<false, { 1 << 14 }, 32, { (1 << 14) * 32 }>),
    /// Quality 7, ordinary inputs: fifteen bucket bits, sixty-four slots.
    H5Q7(BucketMatcher<false, { 1 << 15 }, 64, { (1 << 15) * 64 }>),
    /// Quality 8, ordinary inputs: fifteen bucket bits, 128 slots.
    H5Q8(BucketMatcher<false, { 1 << 15 }, 128, { (1 << 15) * 128 }>),
    /// Quality 9, ordinary inputs: fifteen bucket bits, 256 slots.
    H5Q9(BucketMatcher<false, { 1 << 15 }, 256, { (1 << 15) * 256 }>),
    /// Quality 5, large inputs and wide windows.
    H6Q5(BucketMatcher<true, { 1 << 15 }, 16, { (1 << 15) * 16 }>),
    /// Quality 6, large inputs and wide windows.
    H6Q6(BucketMatcher<true, { 1 << 15 }, 32, { (1 << 15) * 32 }>),
    /// Quality 7, large inputs and wide windows.
    H6Q7(BucketMatcher<true, { 1 << 15 }, 64, { (1 << 15) * 64 }>),
    /// Quality 8, large inputs and wide windows.
    H6Q8(BucketMatcher<true, { 1 << 15 }, 128, { (1 << 15) * 128 }>),
    /// Quality 9, large inputs and wide windows.
    H6Q9(BucketMatcher<true, { 1 << 15 }, 256, { (1 << 15) * 256 }>),
}

impl MatchFinder {
    /// Allocates the bucket matcher `shape` calls for, `HASH64` selecting
    /// `H6` over `H5`.
    ///
    /// The block depth is the quality less one, from four to eight, and it
    /// decides the bucket count with it, so the shape's depth picks the
    /// variant on its own.
    fn bucket(hash64: bool, shape: BucketShape, size_hint: usize) -> Self {
        match (hash64, shape.block_bits) {
            (false, ..=4) => Self::H5Q5(BucketMatcher::new(size_hint)),
            (false, 5) => Self::H5Q6(BucketMatcher::new(size_hint)),
            (false, 6) => Self::H5Q7(BucketMatcher::new(size_hint)),
            (false, 7) => Self::H5Q8(BucketMatcher::new(size_hint)),
            (false, 8..) => Self::H5Q9(BucketMatcher::new(size_hint)),
            (true, ..=4) => Self::H6Q5(BucketMatcher::new(size_hint)),
            (true, 5) => Self::H6Q6(BucketMatcher::new(size_hint)),
            (true, 6) => Self::H6Q7(BucketMatcher::new(size_hint)),
            (true, 7) => Self::H6Q8(BucketMatcher::new(size_hint)),
            (true, 8..) => Self::H6Q9(BucketMatcher::new(size_hint)),
        }
    }
}

impl From<HasherPlan> for MatchFinder {
    /// Allocates the match finder a plan calls for.
    fn from(plan: HasherPlan) -> Self {
        match plan {
            HasherPlan::H2 => Self::H2(QuickMatcher::new()),
            HasherPlan::H3 => Self::H3(QuickMatcher::new()),
            HasherPlan::H4 => Self::H4(QuickMatcher::new()),
            HasherPlan::H54 => Self::H54(QuickMatcher::new()),
            // As in the reference, the number of cached distances is part of
            // the hasher's type, so each probe loop has a constant trip count.
            HasherPlan::Chain(shape) => match (shape.num_banks, shape.last_distances) {
                (1, 4) => Self::H40(ChainMatcher::new(shape)),
                (1, _) => Self::H41(ChainMatcher::new(shape)),
                _ => Self::H42(ChainMatcher::new(shape)),
            },
            HasherPlan::H5(shape) => Self::bucket(false, shape, 0),
            HasherPlan::H6(shape) => Self::bucket(true, shape, 0),
        }
    }
}

/// Runs `body` on whichever concrete matcher `finder` holds.
///
/// The dispatch happens once per block; everything inside `body` is
/// monomorphised on the matcher type it was handed.
macro_rules! with_matcher {
    ($finder:expr, |$matcher:ident| $body:expr) => {
        match $finder {
            MatchFinder::H2Small($matcher) => $body,
            MatchFinder::H3Small($matcher) => $body,
            MatchFinder::H4Small($matcher) => $body,
            MatchFinder::H2($matcher) => $body,
            MatchFinder::H3($matcher) => $body,
            MatchFinder::H4($matcher) => $body,
            MatchFinder::H54($matcher) => $body,
            MatchFinder::H40($matcher) => $body,
            MatchFinder::H41($matcher) => $body,
            MatchFinder::H42($matcher) => $body,
            MatchFinder::H5Q5($matcher) => $body,
            MatchFinder::H5Q6($matcher) => $body,
            MatchFinder::H5Q7($matcher) => $body,
            MatchFinder::H5Q8($matcher) => $body,
            MatchFinder::H5Q9($matcher) => $body,
            MatchFinder::H6Q5($matcher) => $body,
            MatchFinder::H6Q6($matcher) => $body,
            MatchFinder::H6Q7($matcher) => $body,
            MatchFinder::H6Q8($matcher) => $body,
            MatchFinder::H6Q9($matcher) => $body,
        }
    };
}

pub(crate) use with_matcher;

impl MatchFinder {
    /// Selects compact physical storage for an expected short input. Hashes,
    /// logical slots and match order are unchanged; the map grows if needed.
    pub(crate) fn for_input(plan: HasherPlan, size_hint: usize) -> Self {
        if size_hint > 0 && size_hint <= 2048 {
            match plan {
                HasherPlan::H2 => return Self::H2Small(QuickMatcher::new()),
                HasherPlan::H3 => return Self::H3Small(QuickMatcher::new()),
                HasherPlan::H4 => return Self::H4Small(QuickMatcher::new()),
                _ => {}
            }
        }
        match plan {
            HasherPlan::H5(shape) => Self::bucket(false, shape, size_hint),
            HasherPlan::H6(shape) => Self::bucket(true, shape, size_hint),
            _ => Self::from(plan),
        }
    }

    /// Builds the finder `for_input` would, taking over whatever storage of
    /// its kind `parked` holds.
    pub(crate) fn rebuilt(plan: HasherPlan, size_hint: usize, parked: &mut ParkedTables) -> Self {
        let slots = ::core::mem::take(&mut parked.chain_slots);
        let mut finder = match plan {
            HasherPlan::Chain(shape) => match (shape.num_banks, shape.last_distances) {
                (1, 4) => Self::H40(ChainMatcher::with_slots(shape, slots)),
                (1, _) => Self::H41(ChainMatcher::with_slots(shape, slots)),
                _ => Self::H42(ChainMatcher::with_slots(shape, slots)),
            },
            _ => {
                parked.chain_slots = slots;
                Self::for_input(plan, size_hint)
            }
        };
        match &mut finder {
            Self::H5Q5(matcher) => matcher.recycle(parked),
            Self::H5Q6(matcher) => matcher.recycle(parked),
            Self::H5Q7(matcher) => matcher.recycle(parked),
            Self::H5Q8(matcher) => matcher.recycle(parked),
            Self::H5Q9(matcher) => matcher.recycle(parked),
            Self::H6Q5(matcher) => matcher.recycle(parked),
            Self::H6Q6(matcher) => matcher.recycle(parked),
            Self::H6Q7(matcher) => matcher.recycle(parked),
            Self::H6Q8(matcher) => matcher.recycle(parked),
            Self::H6Q9(matcher) => matcher.recycle(parked),
            _ => {}
        }
        finder
    }

    /// Starts a bucket finder's next stream from its compact layout when it
    /// holds an on-demand one; see `BucketMatcher::forget_sparse`.
    pub(crate) fn forget_sparse(&mut self) {
        match self {
            Self::H5Q5(matcher) => matcher.forget_sparse(),
            Self::H5Q6(matcher) => matcher.forget_sparse(),
            Self::H5Q7(matcher) => matcher.forget_sparse(),
            Self::H5Q8(matcher) => matcher.forget_sparse(),
            Self::H5Q9(matcher) => matcher.forget_sparse(),
            Self::H6Q5(matcher) => matcher.forget_sparse(),
            Self::H6Q6(matcher) => matcher.forget_sparse(),
            Self::H6Q7(matcher) => matcher.forget_sparse(),
            Self::H6Q8(matcher) => matcher.forget_sparse(),
            Self::H6Q9(matcher) => matcher.forget_sparse(),
            _ => {}
        }
    }

    /// Drops dense storage this finder took over but has not used yet.
    pub(crate) fn release_recycled(&mut self) {
        match self {
            Self::H5Q5(matcher) => matcher.recycled = None,
            Self::H5Q6(matcher) => matcher.recycled = None,
            Self::H5Q7(matcher) => matcher.recycled = None,
            Self::H5Q8(matcher) => matcher.recycled = None,
            Self::H5Q9(matcher) => matcher.recycled = None,
            Self::H6Q5(matcher) => matcher.recycled = None,
            Self::H6Q6(matcher) => matcher.recycled = None,
            Self::H6Q7(matcher) => matcher.recycled = None,
            Self::H6Q8(matcher) => matcher.recycled = None,
            Self::H6Q9(matcher) => matcher.recycled = None,
            _ => {}
        }
    }

    /// Leaves this finder's reusable storage in `parked` for the next finder
    /// of its kind; the quick finders have none worth keeping, since a
    /// reused quick table has to be wiped anyway.
    pub(crate) fn park(self, parked: &mut ParkedTables) {
        match self {
            Self::H40(matcher) => matcher.park(parked),
            Self::H41(matcher) => matcher.park(parked),
            Self::H42(matcher) => matcher.park(parked),
            Self::H5Q5(matcher) => matcher.park(parked),
            Self::H5Q6(matcher) => matcher.park(parked),
            Self::H5Q7(matcher) => matcher.park(parked),
            Self::H5Q8(matcher) => matcher.park(parked),
            Self::H5Q9(matcher) => matcher.park(parked),
            Self::H6Q5(matcher) => matcher.park(parked),
            Self::H6Q6(matcher) => matcher.park(parked),
            Self::H6Q7(matcher) => matcher.park(parked),
            Self::H6Q8(matcher) => matcher.park(parked),
            Self::H6Q9(matcher) => matcher.park(parked),
            _ => {}
        }
    }

    /// Allocates the full quick tables now; other finders allocate on their
    /// own schedule. See `QuickMatcher::materialize`.
    pub(crate) fn materialize(&mut self) {
        match self {
            Self::H2(matcher) => matcher.materialize(),
            Self::H3(matcher) => matcher.materialize(),
            Self::H4(matcher) => matcher.materialize(),
            Self::H54(matcher) => matcher.materialize(),
            _ => {}
        }
    }

    /// Re-aims the finder at a stream of `size_hint` bytes for `plan`.
    ///
    /// Returns `false` when [`MatchFinder::for_input`] would pick another
    /// variant for that hint — the short-input quick storage against the
    /// full table — in which case the caller has to rebuild; otherwise the
    /// bucket matchers take the hint for their next layout choice and the
    /// rest, whose shape the hint never reached, are unchanged.
    ///
    /// Rebuilding is the cheaper side of that trade for a pooled encoder:
    /// after a long input the full table needs a complete wipe, which costs
    /// more than a fresh short-input map.
    pub(crate) fn retarget(&mut self, plan: HasherPlan, size_hint: usize) -> bool {
        let wants_small = size_hint > 0
            && size_hint <= 2048
            && matches!(plan, HasherPlan::H2 | HasherPlan::H3 | HasherPlan::H4);
        let is_small = matches!(self, Self::H2Small(_) | Self::H3Small(_) | Self::H4Small(_));
        if wants_small != is_small {
            return false;
        }
        match self {
            Self::H5Q5(matcher) => matcher.retarget(size_hint),
            Self::H5Q6(matcher) => matcher.retarget(size_hint),
            Self::H5Q7(matcher) => matcher.retarget(size_hint),
            Self::H5Q8(matcher) => matcher.retarget(size_hint),
            Self::H5Q9(matcher) => matcher.retarget(size_hint),
            Self::H6Q5(matcher) => matcher.retarget(size_hint),
            Self::H6Q6(matcher) => matcher.retarget(size_hint),
            Self::H6Q7(matcher) => matcher.retarget(size_hint),
            Self::H6Q8(matcher) => matcher.retarget(size_hint),
            Self::H6Q9(matcher) => matcher.retarget(size_hint),
            _ => {}
        }
        true
    }

    /// Clears the table before the first block of a stream (`Prepare`).
    ///
    /// Returns which sweep was taken; see [`Matcher::prepare`].
    pub(crate) fn prepare(
        &mut self,
        one_shot: bool,
        input_size: usize,
        data: &[u8],
        clear: bool,
    ) -> Sweep {
        with_matcher!(self, |matcher| matcher
            .prepare(one_shot, input_size, data, clear))
    }

    /// Returns the bytes the chosen match finder keeps allocated.
    pub(crate) fn retained_bytes(&self) -> usize {
        with_matcher!(self, |matcher| matcher.retained_bytes())
    }

    /// Records the positions spanning the previous block boundary.
    pub(crate) fn stitch_to_previous_block(
        &mut self,
        num_bytes: usize,
        position: usize,
        data: &[u8],
        mask: usize,
    ) {
        with_matcher!(self, |matcher| matcher
            .stitch_to_previous_block(num_bytes, position, data, mask));
    }

    /// Records the positions of `history`, the first window bytes, whose
    /// lookahead lies inside it (`HasherPrependCustomDictionary`), except
    /// inside short-period runs (see [`history_ranges`]).
    ///
    /// The last positions, whose lookahead reaches past the history, are left
    /// to [`MatchFinder::stitch_to_previous_block`] once the next block
    /// arrives.
    #[cfg(not(feature = "no_std"))]
    pub(crate) fn prepend(&mut self, data: &[u8], mask: usize, history: &[u8]) {
        fn store_prefix<M: Matcher>(matcher: &mut M, data: &[u8], mask: usize, history: &[u8]) {
            let Some(limit) = (history.len() + 1).checked_sub(M::STORE_LOOKAHEAD) else {
                return;
            };
            history_ranges(history, |start, end| {
                let end = end.min(limit);
                if start < end {
                    matcher.store_range(data, mask, start, end);
                }
            });
        }
        with_matcher!(self, |matcher| store_prefix(matcher, data, mask, history));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Positions [`history_ranges`] keeps, in order.
    #[cfg(not(feature = "no_std"))]
    fn kept(history: &[u8]) -> Vec<(usize, usize)> {
        let mut ranges = Vec::new();
        history_ranges(history, |start, end| ranges.push((start, end)));
        ranges
    }

    #[cfg(not(feature = "no_std"))]
    #[test]
    fn history_runs_keep_only_their_last_four_periods() {
        let total = |ranges: &[(usize, usize)]| ranges.iter().map(|(s, e)| e - s).sum::<usize>();
        // Zeros repeat every byte: the start and the last four positions.
        let zeros = kept(&[0; 1 << 16]);
        assert!(total(&zeros) <= 16, "{zeros:?}");
        assert_eq!(zeros.last().map(|r| r.1), Some(1 << 16));
        // A 256-byte record over and over: its first copy and the last four.
        let records: Vec<u8> = (0..=255u8).cycle().take(1 << 16).collect();
        let ranges = kept(&records);
        assert!(total(&ranges) <= 256 + 8 + 4 * 256, "{ranges:?}");
        let mut state = 1u32;
        let noise: Vec<u8> = (0..1 << 16)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (state >> 24) as u8
            })
            .collect();
        // A run with a period over the limit is stored whole, like text.
        let long: Vec<u8> = noise[..2048]
            .iter()
            .copied()
            .cycle()
            .take(1 << 16)
            .collect();
        assert_eq!(kept(&long), [(0, 1 << 16)]);
        // Neither short history nor input without runs loses a position.
        assert_eq!(kept(&[1, 2, 3]), [(0, 3)]);
        assert_eq!(kept(&noise), [(0, 1 << 16)]);
    }

    #[test]
    fn dense_stores_fill_a_bucket_downwards_through_either_index() {
        // Sixteen slots take the flat index, sixty-four the bucket rows.
        fn check<const BLOCK: usize>() {
            let mut num = [0u16; 4];
            let mut dense = [[0u32; BLOCK]; 4];
            let mut tags = [[0u8; BLOCK]; 4];
            let stores = BLOCK + 4;
            for i in 0..stores {
                dense_store(
                    &mut num,
                    &mut dense,
                    Some(&mut tags),
                    6,
                    100 + i as u32,
                    i as u8,
                );
            }
            assert_eq!(num, [0, 0, stores as u16, 0]);
            for i in stores - BLOCK..stores {
                let slot = BLOCK - 1 - i % BLOCK;
                assert_eq!(dense[2][slot], 100 + i as u32, "block {BLOCK} store {i}");
                assert_eq!(tags[2][slot], i as u8, "block {BLOCK} store {i}");
            }
            for bucket in [0, 1, 3] {
                assert!(dense[bucket].iter().all(|&position| position == 0));
                assert!(tags[bucket].iter().all(|&tag| tag == 0));
            }
        }
        check::<16>();
        check::<64>();
    }

    #[test]
    fn a_dense_store_range_records_what_single_stores_record() {
        fn check<const HASH64: bool, const BLOCK: usize, const SLOTS: usize>() {
            // A 1 KiB ring with a tail, so ranges end inside it, run up to
            // the mask, wrap past it and start on a later lap.
            let data: Vec<u8> = (0..1024 + 64)
                .map(|i: usize| (i * 131 % 251) as u8 ^ (i >> 3) as u8)
                .collect();
            let mask = 1023;
            for (start, end) in [
                (0, 0),
                (5, 200),
                (1000, 1024),
                (1000, 1100),
                (3082, 3372),
                (7, 3),
            ] {
                let mut ranged = DenseLayout::<256, BLOCK, SLOTS>::default();
                let mut single = DenseLayout::<256, BLOCK, SLOTS>::default();
                ranged
                    .dense_run::<HASH64>()
                    .store_range(&data, mask, start, end);
                let mut run = single.dense_run::<HASH64>();
                for ix in start..end {
                    run.store(&data, mask, ix);
                }
                assert_eq!(ranged.num, single.num, "{start}..{end}");
                assert_eq!(ranged.dense, single.dense, "{start}..{end}");
                assert_eq!(ranged.dense_tags, single.dense_tags, "{start}..{end}");
            }
        }
        check::<true, 64, { 256 * 64 }>();
        check::<false, 64, { 256 * 64 }>();
        check::<true, 16, { 256 * 16 }>();
    }

    #[test]
    fn dense_runs_borrow_the_first_and_last_slots_of_the_owned_tables() {
        fn check<const BLOCK: usize, const SLOTS: usize>() {
            let mut layout = DenseLayout::<2, BLOCK, SLOTS>::default();
            let positions = layout.dense.as_ptr();
            let counters = layout.num.as_ptr();
            {
                let run = layout.dense_run::<false>();
                assert_eq!(run.dense.as_ptr().cast::<u32>(), positions);
                assert_eq!(run.num.as_ptr(), counters);
                run.dense[0][0] = 11;
                run.dense[1][BLOCK - 1] = 22;
                run.num[1] = 3;
                assert_eq!(run.tags.is_some(), BLOCK <= 32);
                if let Some(tags) = run.tags {
                    tags[0][0] = 4;
                    tags[1][BLOCK - 1] = 5;
                }
            }
            assert_eq!(layout.dense[0], 11);
            assert_eq!(layout.dense[SLOTS - 1], 22);
            assert_eq!(layout.num[1], 3);
            if let Some(tags) = layout.dense_tags {
                assert_eq!(tags[0], 4);
                assert_eq!(tags[SLOTS - 1], 5);
            }
        }
        check::<16, 32>();
        check::<64, 128>();
    }

    #[test]
    fn compact_slots_reset_to_a_larger_input_extends_an_empty_map() {
        let mut slots = SmallSlots::default();
        slots.reset(16);
        assert_eq!(slots.entries.len(), 32);
        slots.write(3, 7);
        slots.reset(100);
        assert_eq!(slots.entries.len(), 256);
        assert!(slots.entries.iter().all(|&entry| entry == 0));
        assert_eq!((slots.count, slots.read(3)), (0, 0));
    }

    #[test]
    fn compact_slots_preserve_colliding_keys_through_growth_overwrites_and_reset() {
        let mut slots = SmallSlots::default();
        assert_eq!(slots.read(3), 0);
        for key in 0..1024 {
            slots.write(key * 32 + 3, key as u32 + 1);
        }
        for key in 0..1024 {
            assert_eq!(slots.read(key * 32 + 3), key as u32 + 1);
            slots.write(key * 32 + 3, SMALL_SLOTS_MAX_INPUT as u32 - 1);
            assert_eq!(slots.read(key * 32 + 3), SMALL_SLOTS_MAX_INPUT as u32 - 1);
        }
        assert_eq!(slots.count, 1024);
        assert_eq!(slots.read(4), 0);
        let capacity = slots.entries.capacity();
        slots.reset(16);
        assert_eq!(slots.entries.capacity(), capacity);
        assert_eq!(slots.read(3), 0);
        slots.write(3, 7);
        assert_eq!(slots.read(3), 7);
    }

    #[test]
    fn compact_quick_matchers_preserve_the_full_tables_results_on_every_backend() {
        let data = repeated();
        for plan in [HasherPlan::H2, HasherPlan::H3, HasherPlan::H4] {
            for backend in crate::compressor::Backend::available() {
                let expected = with_matcher!(MatchFinder::from(plan), |matcher| {
                    let mut matcher = primed(matcher, &data);
                    search_with(backend.0, &mut matcher, &data, REPEAT_AT)
                });
                let actual = with_matcher!(MatchFinder::for_input(plan, 16), |matcher| {
                    let mut matcher = primed(matcher, &data);
                    search_with(backend.0, &mut matcher, &data, REPEAT_AT)
                });
                assert_eq!(actual, expected, "{backend:?}, {plan:?}");
            }
        }
    }

    #[test]
    fn a_parked_dense_table_and_its_tags_go_to_the_next_bucket_finder() {
        type Q6 = BucketMatcher<false, { 1 << 14 }, 32, { (1 << 14) * 32 }>;
        const SLOTS: usize = (1 << 14) * 32;
        let mut parked = ParkedTables::default();
        assert_eq!(parked.retained_bytes(), 0);

        // A tagged shape's dense layout parks its positions and its tags.
        let mut dense = Q6::new(0);
        dense.select_layout(false, 0, &[]);
        assert!(matches!(dense.layout, Layout::Dense(_)));
        let Layout::Dense(layout) = &dense.layout else {
            unreachable!()
        };
        let positions = layout.dense.as_ptr();
        dense.park(&mut parked);
        assert_eq!(parked.retained_bytes(), SLOTS * size_of::<u32>() + SLOTS);

        // A table no larger than the parked one does not displace it.
        let mut twin = Q6::new(0);
        twin.select_layout(false, 0, &[]);
        twin.park(&mut parked);
        assert_eq!(parked.retained_bytes(), SLOTS * size_of::<u32>() + SLOTS);

        // The next bucket finder's first dense layout takes them over, at
        // the same address, instead of allocating its own.
        let mut next = Q6::new(0);
        next.recycle(&mut parked);
        assert_eq!(parked.retained_bytes(), 0);
        assert!(next.retained_bytes() >= SLOTS * size_of::<u32>() + SLOTS);
        next.select_layout(false, 0, &[]);
        let Layout::Dense(layout) = &next.layout else {
            panic!("a stream of unknown length takes the dense table");
        };
        assert_eq!(layout.dense.as_ptr(), positions);
        assert!(layout.dense_tags.is_some());
        assert!(next.recycled.is_none());
    }

    #[test]
    fn a_single_bank_chain_materializes_its_bank_up_front() {
        let mut matcher = ChainMatcher::<1, 16, 4>::new(Q5_CHAIN);
        assert_eq!(matcher.slots.len(), 1 << 16);
        assert_eq!(matcher.bank_offsets[0], 1);
        let retained = matcher.retained_bytes();
        let data = [b'a'; 64];
        matcher.store(&data, usize::MAX, 0);
        matcher.store_range(&data, usize::MAX, 1, 32);
        assert_eq!(matcher.slots.len(), 1 << 16);
        assert_eq!(matcher.retained_bytes(), retained);
    }

    #[test]
    fn a_cold_q9_chain_allocates_only_the_bank_it_uses() {
        let mut matcher = ChainMatcher::<512, 9, 16>::new(Q9_CHAIN);
        assert!(matcher.slots.is_empty());
        let data = [b'a'; 64];
        matcher.store(&data, usize::MAX, 0);
        assert_eq!(matcher.slots.len(), 512);
        matcher.store(&data, usize::MAX, 1);
        assert_eq!(matcher.slots.len(), 512);
        let retained = matcher.retained_bytes();
        matcher.prepare(false, data.len(), &data, true);
        matcher.store(&data, usize::MAX, 0);
        assert_eq!(matcher.retained_bytes(), retained);
    }

    #[test]
    fn a_short_input_links_each_store_in_front_of_its_bucket_chain() {
        let mut matcher = BucketMatcher::<false, { 1 << 15 }, 256, { (1 << 15) * 256 }>::new(0);
        let data = [b'a'; 64];
        matcher.prepare(true, data.len(), &data, true);
        assert!(matches!(matcher.layout, Layout::Compact(_)));
        assert!(matcher.retained_bytes() < 16 * 1024);
        let Layout::Compact(layout) = &matcher.layout else {
            panic!("compact layout")
        };
        assert!(layout.chain.is_empty());
        for position in 0..5 {
            matcher.store(&data, usize::MAX, position);
        }
        let Layout::Compact(layout) = &matcher.layout else {
            panic!("compact layout")
        };
        // Every store is one node, linked to the previous store into the
        // same bucket: position 4 first, then 3, down to 0 with no link.
        assert_eq!(layout.chain.len(), 5);
        let key = hash_with_tag::<false, { 1 << 15 }>(&data, 0) >> 8;
        let (count, head) = layout.compact.get(key);
        assert_eq!(count, 5);
        let mut link = head as usize;
        let mut walked = Vec::new();
        while link != 0 {
            let node = layout.chain[link - 1];
            walked.push(node as u32);
            link = (node >> 32) as usize;
        }
        assert_eq!(walked, [4, 3, 2, 1, 0]);
    }

    #[test]
    fn a_short_input_scans_only_the_newest_block_of_a_long_chain() {
        // Every position of a run hashes into one bucket. The block keeps
        // sixteen positions, so the chain walk stops after sixteen too and
        // the compact matcher finds what the sparse one does.
        let data = [b'a'; 200];
        let level =
            fearless_simd::Level::try_detect().unwrap_or_else(fearless_simd::Level::baseline);
        let mut compact = BucketMatcher::<false, { 1 << 14 }, 16, { (1 << 14) * 16 }>::new(0);
        compact.prepare(true, data.len(), &data, true);
        assert!(matches!(compact.layout, Layout::Compact(_)));
        let mut sparse = BucketMatcher::<false, { 1 << 14 }, 16, { (1 << 14) * 16 }>::new(
            COMPACT_INPUT_LIMIT + 1,
        );
        sparse.prepare(true, COMPACT_INPUT_LIMIT + 1, &data, true);
        assert!(matches!(sparse.layout, Layout::Sparse(_)));
        compact.store_range(&data, usize::MAX, 0, 100);
        sparse.store_range(&data, usize::MAX, 0, 100);
        let expected = search_with(level, &mut sparse, &data, 100);
        let actual = search_with(level, &mut compact, &data, 100);
        assert_eq!(actual, expected);
        assert!(actual.is_match());
        assert_eq!(SparseLayout::<{ 1 << 15 }, 256>::block(0), None);
    }

    #[test]
    fn every_layout_finds_the_same_match_and_forgets_it_on_prepare() {
        let data = repeated();
        for backend in crate::compressor::Backend::available() {
            let mut expected = None;
            for (one_shot, input_size, size_hint, layout) in [
                (true, data.len(), data.len(), "compact"),
                (
                    true,
                    COMPACT_INPUT_LIMIT + 1,
                    COMPACT_INPUT_LIMIT + 1,
                    "sparse",
                ),
                (false, data.len(), 1 << 20, "dense"),
                // A tagged shape takes an unknown length for a long stream.
                (false, data.len(), 0, "dense"),
            ] {
                let mut matcher =
                    BucketMatcher::<false, { 1 << 14 }, 16, { (1 << 14) * 16 }>::new(size_hint);
                matcher.prepare(one_shot, input_size, &data, true);
                assert_eq!(
                    match matcher.layout {
                        Layout::Compact(_) => "compact",
                        Layout::Sparse(_) => "sparse",
                        Layout::Dense(_) => "dense",
                    },
                    layout,
                    "{backend:?}"
                );
                matcher.store_range(&data, usize::MAX, 0, REPEAT_AT);
                let found = search_with(backend.0, &mut matcher, &data, REPEAT_AT);
                assert_eq!(
                    (found.distance, found.len),
                    (64, 64),
                    "{backend:?} {layout:?}"
                );
                let found = (found.distance, found.len, found.score);
                assert_eq!(
                    *expected.get_or_insert(found),
                    found,
                    "{backend:?} {layout:?}"
                );
                // A new stream of any layout starts from an empty table.
                for (next_one_shot, next_size) in [
                    (one_shot, input_size),
                    (true, 16),
                    (true, COMPACT_INPUT_LIMIT + 1),
                    (false, 0),
                ] {
                    matcher.prepare(next_one_shot, next_size, &data, true);
                    assert!(
                        !search_with(backend.0, &mut matcher, &data, REPEAT_AT).is_match(),
                        "{backend:?} {layout:?} -> {next_one_shot} {next_size}"
                    );
                }
            }
        }
    }

    #[test]
    fn compact_promotion_reuses_the_larger_word_buffer_and_discards_old_entries() {
        for reuse_chain in [false, true] {
            let mut matcher = BucketMatcher::<false, 64, 16, { 64 * 16 }>::new(1);
            matcher.prepare(true, 16, &[], false);
            let Layout::Compact(layout) = &mut matcher.layout else {
                panic!("compact layout");
            };
            let expected = if reuse_chain {
                layout.compact.entries = vec![KeyMap::EMPTY; 32];
                layout.chain = vec![u64::MAX; 64];
                layout.chain.as_ptr()
            } else {
                layout.compact.entries.fill(u64::MAX - 1);
                layout.compact.entries.as_ptr()
            };
            matcher.prepare(false, 0, &[], true);
            let Layout::Sparse(layout) = &matcher.layout else {
                panic!("sparse layout");
            };
            assert_eq!(layout.entries.as_ptr(), expected);
            assert!(layout.entries.iter().all(|&entry| entry == 0));
            assert_eq!(layout.generation, 1);
            assert_eq!(matcher.retained_bytes(), 64 * size_of::<u64>());
        }
    }

    #[test]
    fn sparse_promotion_transfers_full_block_allocations_and_releases_other_pools() {
        fn check<const BLOCK: usize, const SLOTS: usize>(enough_capacity: bool) {
            let mut matcher = BucketMatcher::<false, 64, BLOCK, SLOTS>::new(1);
            matcher.prepare(false, 0, &[], false);
            let Layout::Sparse(layout) = &mut matcher.layout else {
                panic!("sparse layout");
            };
            let capacity = if enough_capacity { 64 } else { 1 };
            layout.blocks = Vec::with_capacity(capacity);
            layout.blocks.push([123; BLOCK]);
            layout.starters.push([456; STARTER_SLOTS]);
            let tagged = BLOCK <= 32;
            if tagged {
                layout.block_tags = Vec::with_capacity(capacity);
                layout.block_tags.push([7; BLOCK]);
                layout.starter_tags.push([8; STARTER_SLOTS]);
            }
            let positions = layout.blocks.as_ptr().cast::<u32>();
            let tags = layout.block_tags.as_ptr().cast::<u8>();
            matcher.retarget(usize::MAX);
            matcher.prepare(false, 0, &[], true);
            let Layout::Dense(layout) = &matcher.layout else {
                panic!("dense layout");
            };
            if enough_capacity {
                assert_eq!(layout.dense.as_ptr(), positions);
                if tagged {
                    assert_eq!(layout.dense_tags.as_ref().unwrap().as_ptr(), tags);
                }
            }
            assert_eq!(&layout.dense[..BLOCK], &[123; BLOCK]);
            assert!(layout.dense[BLOCK..].iter().all(|&position| position == 0));
            assert!(layout.num.iter().all(|&count| count == 0));
            if tagged {
                assert_eq!(&layout.dense_tags.as_ref().unwrap()[..BLOCK], &[7; BLOCK]);
            } else {
                assert!(layout.dense_tags.is_none());
            }
            assert_eq!(
                matcher.retained_bytes(),
                64 * (2 + BLOCK * (4 + usize::from(tagged)))
            );
        }
        for enough in [false, true] {
            check::<16, { 64 * 16 }>(enough);
            check::<64, { 64 * 64 }>(enough);
            check::<256, { 64 * 256 }>(enough);
        }
    }

    #[test]
    fn sparse_promotion_reuses_starters_when_their_allocation_is_larger() {
        let mut matcher = BucketMatcher::<false, 64, 16, { 64 * 16 }>::new(1);
        matcher.prepare(false, 0, &[], false);
        let Layout::Sparse(layout) = &mut matcher.layout else {
            panic!("sparse layout");
        };
        layout.starters = Vec::with_capacity(256);
        layout.starters.push([123; STARTER_SLOTS]);
        layout.starter_tags = Vec::with_capacity(256);
        layout.starter_tags.push([7; STARTER_SLOTS]);
        let positions = layout.starters.as_ptr().cast::<u32>();
        let tags = layout.starter_tags.as_ptr().cast::<u8>();
        matcher.retarget(usize::MAX);
        matcher.prepare(false, 0, &[], true);
        let Layout::Dense(layout) = &matcher.layout else {
            panic!("dense layout");
        };
        assert_eq!(layout.dense.as_ptr(), positions);
        assert_eq!(layout.dense_tags.as_ref().unwrap().as_ptr(), tags);
        assert_eq!(&layout.dense[..STARTER_SLOTS], &[123; STARTER_SLOTS]);
        assert_eq!(
            &layout.dense_tags.as_ref().unwrap()[..STARTER_SLOTS],
            &[7; STARTER_SLOTS]
        );
        assert!(layout.num.iter().all(|&count| count == 0));
        assert_eq!(matcher.retained_bytes(), 64 * (2 + 16 * 5));
    }

    #[test]
    fn prepared_layouts_retain_their_allocations_for_shorter_streams() {
        let data = repeated();
        for size_hint in [1, usize::MAX] {
            let mut matcher =
                BucketMatcher::<false, { 1 << 14 }, 16, { (1 << 14) * 16 }>::new(size_hint);
            matcher.prepare(false, 0, &data, false);
            matcher.store_range(&data, usize::MAX, 0, REPEAT_AT);
            let retained = matcher.retained_bytes();
            let index = match &matcher.layout {
                Layout::Sparse(layout) => layout.entries.as_ptr().cast::<u8>(),
                Layout::Dense(layout) => layout.dense.as_ptr().cast::<u8>(),
                Layout::Compact(_) => panic!("prepared noncompact layout"),
            };
            matcher.retarget(1);
            matcher.prepare(true, 16, &data, true);
            let actual = match &matcher.layout {
                Layout::Sparse(layout) => layout.entries.as_ptr().cast::<u8>(),
                Layout::Dense(layout) => layout.dense.as_ptr().cast::<u8>(),
                Layout::Compact(_) => panic!("keep the allocated table"),
            };
            assert_eq!(actual, index);
            assert_eq!(matcher.retained_bytes(), retained);
            assert!(!search_at(&mut matcher, &data, REPEAT_AT).is_match());
        }
    }

    #[test]
    fn a_deep_shape_takes_the_dense_table_once_it_is_reused() {
        // A quarter-mebibyte hint is below an eighth of the 8 MiB table, so
        // the first stream stays on demand; the second stream, a sixty-fourth
        // being enough for a reused matcher, gets the table.
        let data = repeated();
        let mut matcher = BucketMatcher::<false, { 1 << 15 }, 64, { (1 << 15) * 64 }>::new(1 << 18);
        matcher.prepare(false, data.len(), &data, true);
        assert!(matches!(matcher.layout, Layout::Sparse(_)));
        matcher.prepare(false, data.len(), &data, true);
        assert!(matches!(matcher.layout, Layout::Dense(_)));
        // A hint below a sixty-fourth stays on demand however often it is
        // reused; retargeting to a longer stream changes that.
        let mut short = BucketMatcher::<false, { 1 << 15 }, 64, { (1 << 15) * 64 }>::new(1 << 16);
        for _ in 0..3 {
            short.prepare(false, data.len(), &data, true);
            assert!(matches!(short.layout, Layout::Sparse(_)));
        }
        short.retarget(1 << 18);
        short.prepare(false, data.len(), &data, true);
        assert!(matches!(short.layout, Layout::Dense(_)));
    }

    #[test]
    fn retargeting_keeps_a_finder_only_when_its_variant_would_not_change() {
        let mut small = MatchFinder::for_input(HasherPlan::H2, 16);
        assert!(matches!(small, MatchFinder::H2Small(_)));
        assert!(small.retarget(HasherPlan::H2, 2048));
        assert!(!small.retarget(HasherPlan::H2, 2049));
        assert!(!small.retarget(HasherPlan::H2, 0));
        let mut full = MatchFinder::for_input(HasherPlan::H2, 0);
        assert!(matches!(full, MatchFinder::H2(_)));
        assert!(full.retarget(HasherPlan::H2, 1 << 20));
        assert!(!full.retarget(HasherPlan::H2, 100));
        let mut bucket = MatchFinder::for_input(HasherPlan::H5(Q5_BUCKET), 100);
        assert!(bucket.retarget(HasherPlan::H5(Q5_BUCKET), 1 << 20));
        let MatchFinder::H5Q5(inner) = &bucket else {
            panic!("shape changed");
        };
        assert_eq!(inner.size_hint, 1 << 20);
    }

    #[test]
    fn a_sparse_table_wipes_itself_when_its_generations_run_out() {
        let data = repeated();
        let mut matcher = BucketMatcher::<false, { 1 << 14 }, 16, { (1 << 14) * 16 }>::new(
            COMPACT_INPUT_LIMIT + 1,
        );
        matcher.prepare(true, COMPACT_INPUT_LIMIT + 1, &data, true);
        matcher.store_range(&data, usize::MAX, 0, REPEAT_AT);
        let Layout::Sparse(layout) = &mut matcher.layout else {
            panic!("sparse layout")
        };
        layout.generation = MAX_GENERATION;
        matcher.prepare(true, COMPACT_INPUT_LIMIT + 1, &data, true);
        let Layout::Sparse(layout) = &matcher.layout else {
            panic!("sparse layout")
        };
        assert_eq!(layout.generation, 1);
        assert!(layout.blocks.is_empty());
        assert!(layout.starters.is_empty());
        assert!(!search_at(&mut matcher, &data, REPEAT_AT).is_match());
        // A bump is what an ordinary new stream costs.
        matcher.prepare(true, COMPACT_INPUT_LIMIT + 1, &data, true);
        let Layout::Sparse(layout) = &matcher.layout else {
            panic!("sparse layout")
        };
        assert_eq!(layout.generation, 2);
    }

    #[test]
    fn a_tagged_shape_that_knows_its_input_samples_the_first_block() {
        fn layout<M: Matcher>(matcher: &mut M, input_size: usize, block: &[u8]) {
            matcher.prepare(true, input_size, block, true);
        }
        fn name<const H: bool, const B: usize, const K: usize, const S: usize>(
            matcher: &BucketMatcher<H, B, K, S>,
        ) -> &'static str {
            match matcher.layout {
                Layout::Compact(_) => "compact",
                Layout::Sparse(_) => "sparse",
                Layout::Dense(_) => "dense",
            }
        }
        let mut state = 0x9E37_79B9u32;
        let mut text = Vec::new();
        let words = [
            "alice ", "rabbit ", "queen ", "hatter ", "tea ", "garden ", "door ",
        ];
        while text.len() < 1 << 16 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            text.extend_from_slice(words[(state >> 16) as usize % words.len()].as_bytes());
        }
        let random: Vec<u8> = (0..1 << 16)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect();
        let runs = vec![b'x'; 1 << 16];
        // Quality six: ordinary data from 16 KiB, the length rule from 128 KiB.
        for (size_hint, block, expected) in [
            (20_000, &text[..20_000], "dense"),
            (8_000, &text[..8_000], "sparse"),
            (64 << 10, &random[..], "sparse"),
            (128 << 10, &random[..], "dense"),
            (128 << 10, &runs[..], "sparse"),
            // Runs that cover less than half the input do not speak for it.
            (200 << 10, &runs[..], "dense"),
            // Nothing to sample keeps the length rule.
            (20_000, &[][..], "sparse"),
            // An unknown length is a long stream, whatever it starts with.
            (0, &runs[..], "dense"),
        ] {
            let mut matcher = Q6Bucket::new(size_hint);
            layout(&mut matcher, block.len().max(size_hint.min(1 << 16)), block);
            assert_eq!(
                name(&matcher),
                expected,
                "q6 hint {size_hint}, {} bytes",
                block.len()
            );
        }
        // Quality five: ordinary data from 8 KiB.
        let mut matcher = BucketMatcher::<false, { 1 << 14 }, 16, { (1 << 14) * 16 }>::new(10_000);
        layout(&mut matcher, 10_000, &text[..10_000]);
        assert_eq!(name(&matcher), "dense");
        // A deep shape follows the length rule alone.
        let mut matcher = Q7Bucket::new(1 << 16);
        layout(&mut matcher, 1 << 16, &text);
        assert_eq!(name(&matcher), "sparse");
    }

    #[test]
    fn a_reused_tagged_shape_takes_the_dense_table_from_a_sixty_fourth() {
        fn is_dense<const H: bool, const B: usize, const K: usize, const S: usize>(
            matcher: &BucketMatcher<H, B, K, S>,
        ) -> bool {
            matches!(matcher.layout, Layout::Dense(_))
        }
        let mut state = 0x2545_F491u32;
        let random: Vec<u8> = (0..1 << 16)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect();
        // Quality six: incompressible 64 KiB is below the 128 KiB first-stream
        // limit, but above the 32 KiB a reused matcher needs.
        let mut matcher = Q6Bucket::new(1 << 16);
        matcher.prepare(true, 1 << 16, &random, true);
        assert!(!is_dense(&matcher));
        matcher.prepare(true, 1 << 16, &random, true);
        assert!(is_dense(&matcher));
        // Below a sixty-fourth it stays on demand however often it is reused.
        let mut short = Q6Bucket::new(20_000);
        for _ in 0..3 {
            short.prepare(true, 20_000, &random[..20_000], true);
            assert!(!is_dense(&short));
        }
        // Repetitive input that its first block speaks for never takes it.
        let runs = vec![b'x'; 1 << 16];
        let mut repetitive = Q6Bucket::new(1 << 16);
        for _ in 0..3 {
            repetitive.prepare(true, 1 << 16, &runs, true);
            assert!(!is_dense(&repetitive));
        }
        // Quality five: 24 KiB is below its 32 KiB first-stream limit and
        // above the 16 KiB a reused matcher needs.
        let mut q5 = BucketMatcher::<false, { 1 << 14 }, 16, { (1 << 14) * 16 }>::new(24 << 10);
        q5.prepare(true, 24 << 10, &random[..24 << 10], true);
        assert!(!is_dense(&q5));
        q5.prepare(true, 24 << 10, &random[..24 << 10], true);
        assert!(is_dense(&q5));
    }

    /// The shape quality seven resolves to.
    type Q7Bucket = BucketMatcher<false, { 1 << 15 }, 64, { (1 << 15) * 64 }>;

    #[test]
    fn a_deep_on_demand_table_turns_dense_once_the_rest_of_the_stream_repays_it() {
        let data = repeated();
        let expected = search_at(&mut primed(Q7Bucket::new(1 << 20), &data), &data, REPEAT_AT);
        assert!(expected.is_match());
        // A quarter-mebibyte stream is below the first-stream dense limit.
        let mut matcher = Q7Bucket::new(1 << 18);
        matcher.prepare(true, 1 << 18, &data, true);
        assert!(matches!(matcher.layout, Layout::Sparse(_)));
        // The first look has no interval to judge a rate by.
        assert_eq!(matcher.checkpoint(0, 1 << 16), CHECKPOINT_INTERVAL);
        matcher.store_range(&data, usize::MAX, 0, REPEAT_AT);
        // One store per position, with most of the stream to come.
        assert_eq!(matcher.checkpoint(REPEAT_AT, 1 << 16), usize::MAX);
        assert!(matches!(matcher.layout, Layout::Dense(_)));
        let found = search_at(&mut matcher, &data, REPEAT_AT);
        assert_eq!(
            (found.distance, found.len, found.score),
            (expected.distance, expected.len, expected.score)
        );
        // Dense layouts have nothing left to decide.
        assert_eq!(matcher.checkpoint(1 << 17, 1 << 16), usize::MAX);
    }

    #[test]
    fn a_stream_that_stores_little_keeps_its_on_demand_table() {
        let data = repeated();
        let mut matcher = Q7Bucket::new(1 << 18);
        matcher.prepare(true, 1 << 18, &data, true);
        assert_eq!(matcher.checkpoint(0, 1 << 16), CHECKPOINT_INTERVAL);
        matcher.store_range(&data, usize::MAX, 0, 16);
        // Sixteen stores across sixty-four kibibytes predict too few for
        // the rest of the stream.
        assert_eq!(matcher.checkpoint(1 << 16, 1 << 16), CHECKPOINT_INTERVAL);
        assert!(matches!(matcher.layout, Layout::Sparse(_)));
        // Without a size hint only the block's own remainder counts.
        let mut matcher = Q7Bucket::new(0);
        matcher.prepare(false, 0, &data, true);
        assert!(matches!(matcher.layout, Layout::Sparse(_)));
        matcher.checkpoint(0, 64);
        matcher.store_range(&data, usize::MAX, 0, REPEAT_AT);
        assert_eq!(matcher.checkpoint(REPEAT_AT, 64), CHECKPOINT_INTERVAL);
        // A new stream starts counting again.
        matcher.prepare(false, 0, &data, true);
        let Layout::Sparse(layout) = &matcher.layout else {
            panic!("sparse layout")
        };
        assert_eq!(
            (
                layout.stores,
                layout.checked_stores,
                layout.checked_position
            ),
            (0, 0, 0)
        );
    }

    /// The shape quality six resolves to.
    type Q6Bucket = BucketMatcher<false, { 1 << 14 }, 32, { (1 << 14) * 32 }>;

    #[test]
    fn a_tagged_shape_needs_twice_the_store_rate_of_a_deep_one() {
        let data = repeated();
        let expected = search_at(&mut primed(Q6Bucket::new(1 << 20), &data), &data, REPEAT_AT);
        assert!(expected.is_match());
        // Short enough that the block's remainder, not the hint, decides.
        // No block to sample keeps the length rule, which leaves it sparse.
        let mut matcher = Q6Bucket::new(20_000);
        matcher.prepare(true, 20_000, &[], true);
        assert!(matches!(matcher.layout, Layout::Sparse(_)));
        matcher.checkpoint(0, 20_000);
        matcher.store_range(&data, usize::MAX, 0, REPEAT_AT);
        // One store per position predicts twenty-four thousand more: enough
        // for a deep table of this size (sixteen thousand), not for a
        // tagged one (thirty-two thousand).
        assert_eq!(matcher.checkpoint(REPEAT_AT, 24_576), CHECKPOINT_INTERVAL);
        assert!(matches!(matcher.layout, Layout::Sparse(_)));
        // Stored again, the same positions only deepen their buckets.
        matcher.store_range(&data, usize::MAX, 0, REPEAT_AT);
        assert_eq!(matcher.checkpoint(2 * REPEAT_AT, 40_000), usize::MAX);
        let Layout::Dense(layout) = &matcher.layout else {
            panic!("dense layout")
        };
        assert!(layout.dense_tags.is_some());
        // The tag mask of the copied block still admits the repeat.
        let found = search_at(&mut matcher, &data, REPEAT_AT);
        assert_eq!(
            (found.distance, found.len, found.score),
            (expected.distance, expected.len, expected.score)
        );
    }

    #[test]
    fn compact_layouts_never_ask_for_a_checkpoint() {
        let data = repeated();
        let mut matcher = Q7Bucket::new(data.len());
        matcher.prepare(true, data.len(), &data, true);
        assert!(matches!(matcher.layout, Layout::Compact(_)));
        assert_eq!(matcher.checkpoint(0, data.len()), usize::MAX);
    }

    #[test]
    fn the_dense_copy_keeps_live_buckets_and_forgets_stale_ones() {
        const BLOCK: usize = 64;
        let mut matcher = Q7Bucket::new(1 << 18);
        matcher.prepare(true, 1 << 18, &[], true);
        let Layout::Sparse(layout) = &mut matcher.layout else {
            panic!("sparse layout")
        };
        // A bucket from an earlier stream still names its block.
        layout.push(9, 900, 0, 1 << 18);
        layout.generation += 1;
        // Three stores stay in a starter; seventy wrap a full block.
        for ix in 0..3 {
            layout.push(1, 100 + ix, 0, 1 << 18);
        }
        for ix in 0..70 {
            layout.push(2, 200 + ix, 0, 1 << 18);
        }
        let dense = DenseLayout::<{ 1 << 15 }, BLOCK, { (1 << 15) * BLOCK }>::from(&*layout);
        assert_eq!(dense.num[1], 3);
        assert_eq!(dense.num[2], 70);
        assert_eq!(dense.num[9], 0);
        assert_eq!(dense.num[0], 0);
        let slots = |key: usize| &dense.dense[key * BLOCK..(key + 1) * BLOCK];
        // Slots fill downwards: the newest store sits lowest.
        assert_eq!(slots(1)[BLOCK - 3..], [102, 101, 100]);
        assert!(slots(1)[..BLOCK - 3].iter().all(|&slot| slot == 0));
        for age in 0..BLOCK as u32 {
            let slot = (BLOCK - 70 % BLOCK + age as usize) % BLOCK;
            assert_eq!(slots(2)[slot], 269 - age, "age {age}");
        }
        assert!(slots(9).iter().all(|&slot| slot == 0));
        assert!(dense.dense_tags.is_none());

        // A tagged shape carries each bucket's tags along with its slots.
        const TAGGED: usize = 32;
        let mut matcher = Q6Bucket::new(1 << 16);
        matcher.prepare(true, 1 << 16, &[], true);
        let Layout::Sparse(layout) = &mut matcher.layout else {
            panic!("sparse layout")
        };
        layout.push(9, 900, 90, 1 << 16);
        layout.generation += 1;
        for ix in 0..3u8 {
            layout.push(1, u32::from(ix), 10 + ix, 1 << 16);
        }
        for ix in 0..40u8 {
            layout.push(2, u32::from(ix), 100 + ix, 1 << 16);
        }
        let dense = DenseLayout::<{ 1 << 14 }, TAGGED, { (1 << 14) * TAGGED }>::from(&*layout);
        let Some(tags) = dense.dense_tags.as_deref() else {
            panic!("a tagged shape keeps its tags")
        };
        let tags = |key: usize| &tags[key * TAGGED..(key + 1) * TAGGED];
        assert_eq!(tags(1)[TAGGED - 3..], [12, 11, 10]);
        for age in 0..TAGGED as u8 {
            let slot = (TAGGED - 40 % TAGGED + usize::from(age)) % TAGGED;
            assert_eq!(tags(2)[slot], 139 - age, "age {age}");
        }
        assert!(tags(9).iter().all(|&tag| tag == 0));
    }

    #[test]
    fn a_dense_table_keeps_its_blocks_and_clears_only_its_counters() {
        let data = repeated();
        let mut matcher = BucketMatcher::<false, { 1 << 14 }, 16, { (1 << 14) * 16 }>::new(1 << 20);
        matcher.prepare(false, data.len(), &data, true);
        matcher.store_range(&data, usize::MAX, 0, REPEAT_AT);
        let retained = matcher.retained_bytes();
        matcher.prepare(false, data.len(), &data, true);
        assert_eq!(matcher.retained_bytes(), retained);
        let Layout::Dense(layout) = &matcher.layout else {
            panic!("dense layout")
        };
        assert!(layout.num.iter().all(|&count| count == 0));
        assert!(!search_at(&mut matcher, &data, REPEAT_AT).is_match());
    }

    #[test]
    fn key_map_growth_reuses_reserved_storage() {
        let mut map = KeyMap::default();
        map.reset(8);
        map.entries.reserve_exact(64);
        for key in 0..32 {
            map.set(key * 64 + 63, key as u16 + 1, key as u32 + 7);
        }
        let storage = map.entries.as_ptr();
        map.set(32 * 64 + 63, 33, 39);
        assert_eq!(map.entries.as_ptr(), storage);
        assert_eq!(map.entries.len(), 128);
        assert_eq!(map.count, 33);
        for key in 0..33 {
            assert_eq!(map.get(key * 64 + 63), (key as u16 + 1, key as u32 + 7));
        }
    }

    #[test]
    fn key_map_reset_reuses_reserved_storage_and_clears_old_entries() {
        let mut map = KeyMap::default();
        map.reset(8);
        map.set(63, 7, 11);
        map.entries.reserve_exact(64);
        let storage = map.entries.as_ptr();
        for input_size in [64, 1, 0] {
            map.reset(input_size);
            assert_eq!(map.entries.as_ptr(), storage);
            assert_eq!(map.entries.len(), 128);
            assert_eq!(map.count, 0);
            assert_eq!(map.get(63), (0, 0));
            assert!(map.entries.iter().all(|&entry| entry == KeyMap::EMPTY));
            map.set(63, 7, 11);
        }
    }

    #[test]
    fn key_map_growth_preserves_wrapping_clusters_in_both_hash_halves() {
        for reverse in [false, true] {
            let mut map = KeyMap::default();
            map.reset(8);
            for i in 0..32 {
                let key = if reverse { 31 - i } else { i };
                map.set(key * 64 + 63, key as u16 + 1, key as u32 + 7);
            }
            for _ in 0..4 {
                map.grow();
                assert_eq!(map.count, 32);
                for key in 0..32 {
                    assert_eq!(map.get(key * 64 + 63), (key as u16 + 1, key as u32 + 7));
                }
                assert_eq!(map.get(62), (0, 0));
            }
        }
    }

    #[test]
    fn key_map_growth_from_empty_and_overwrites_match_an_ordered_map() {
        let mut map = KeyMap::default();
        map.grow();
        assert_eq!(map.entries.len(), 64);
        assert_eq!(map.count, 0);
        let mut expected = alloc::collections::BTreeMap::new();
        let mut state = 0x1234_5678_u32;
        for i in 0..2048_u32 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let key = if i & 1 == 0 {
                (state as usize & 511) * 64 + 63
            } else {
                state as usize & 32767
            };
            let value = (i as u16 + 1, state);
            map.set(key, value.0, value.1);
            expected.insert(key, value);
            assert_eq!(map.count, expected.len());
            if i % 64 == 63 {
                for (&key, &value) in &expected {
                    assert_eq!(map.get(key), value);
                }
            }
        }
        for (&key, &value) in &expected {
            assert_eq!(map.get(key), value);
        }
    }

    #[test]
    fn the_key_map_keeps_colliding_keys_through_growth_and_reset() {
        let mut map = KeyMap::default();
        assert_eq!(map.get(3), (0, 0));
        map.reset(4);
        for key in 0..1024 {
            map.set(key * 64 + 3, key as u16 + 1, key as u32 + 7);
        }
        for key in 0..1024 {
            assert_eq!(map.get(key * 64 + 3), (key as u16 + 1, key as u32 + 7));
        }
        assert_eq!(map.get(5), (0, 0));
        map.set(67, 9, 9);
        assert_eq!(map.get(67), (9, 9));
        map.reset(4);
        assert_eq!(map.get(3), (0, 0));
        assert_eq!(map.count, 0);
    }

    #[test]
    fn candidate_masks_walk_a_block_newest_to_oldest() {
        // A block of eight slots whose newest position is at slot 5, holding
        // three positions: slots 5, 6 and 7 become ages 0, 1 and 2.
        assert_eq!(rotate_candidates(u32::MAX, 8, 5, 3), 0b0000_0111);
        // Six positions: slots 5, 6, 7, then 0, 1, 2.
        assert_eq!(rotate_candidates(u32::MAX, 8, 5, 6), 0b0011_1111);
        // A full block visits everything; the equality mask filters it:
        // slots 3, 5, 7 and 1 are ages 1, 3, 5 and 7 from slot 2.
        assert_eq!(rotate_candidates(0b1010_1010, 8, 2, 8), 0b1010_1010);
        // Thirty-two lanes at the top of the word: slot 31 is age 0 and
        // slot 0 age 1.
        assert_eq!(rotate_candidates(u32::MAX, 32, 31, 32), u32::MAX);
        assert_eq!(rotate_candidates(1, 32, 31, 32), 0b10);
        // Nothing stored means nothing to walk, whatever the tags say.
        assert_eq!(rotate_candidates(u32::MAX, 16, 3, 0), 0);
    }

    #[test]
    fn tag_equality_matches_scalar_comparison_on_every_backend() {
        fn check<const N: usize>(backend: crate::compressor::Backend) {
            let tags: [u8; N] = ::core::array::from_fn(|i| (i % 5) as u8);
            for tag in 0..5u8 {
                let expected: u32 = tags
                    .iter()
                    .enumerate()
                    .filter(|&(_, &value)| value == tag)
                    .fold(0, |mask, (slot, _)| mask | 1 << slot);
                let actual = dispatch!(backend.0, simd => tag_equality(simd, &tags, tag));
                if backend == crate::compressor::Backend::SCALAR {
                    assert_eq!(actual, u32::MAX);
                } else {
                    assert_eq!(actual, expected, "{backend:?} {N} {tag}");
                }
            }
        }
        for backend in crate::compressor::Backend::available() {
            check::<16>(backend);
            check::<32>(backend);
            // A starter block is too short for a vector compare.
            assert_eq!(
                dispatch!(backend.0, simd => tag_equality(simd, &[1, 2, 3, 4], 2)),
                u32::MAX
            );
        }
    }

    #[test]
    fn accepted_matches_preserve_scores_and_reject_ties_on_every_backend() {
        let mut data = [0u8; 96];
        data[..8].copy_from_slice(b"abcdefgh");
        data[32..40].copy_from_slice(b"abcdefgh");
        data[8] = 1;
        data[40] = 2;
        for backend in crate::compressor::Backend::available() {
            let score = backward_reference_score(8, 32);
            for hash64 in [false, true] {
                let actual = if hash64 {
                    dispatch!(backend.0, simd => accept_candidate::<_, true>(simd, &data, 32, 16, 0, 32, 0))
                } else {
                    dispatch!(backend.0, simd => accept_candidate::<_, false>(simd, &data, 32, 16, 0, 32, 0))
                };
                let (best, actual_score) = actual.expect("eight-byte match");
                assert_eq!(best.len, 8);
                assert_eq!(best.word, u32::from_le_bytes([b'f', b'g', b'h', 2]));
                assert_eq!(actual_score.get(), score);
            }
            assert!(dispatch!(backend.0, simd => accept_candidate::<_, false>(simd, &data, 32, 16, 0, 32, score)).is_none());
            for (current, limit, previous) in [(95, 16, 0), (32, 16, 96), (32, 0, 0), (32, 3, 0)] {
                assert!(dispatch!(backend.0, simd => accept_candidate::<_, false>(simd, &data, current, limit, previous, 32, 0)).is_none());
                assert!(dispatch!(backend.0, simd => accept_candidate::<_, true>(simd, &data, current, limit, previous, 32, 0)).is_none());
            }
            for index in 0..NUM_DISTANCE_SHORT_CODES {
                let mut expected = backward_reference_score_using_last_distance(8);
                if index != 0 {
                    expected -= backward_reference_penalty_using_last_distance(index);
                }
                let actual =
                    dispatch!(backend.0, simd => accept_cached(simd, &data, 32, 16, 0, index, 0));
                assert_eq!(
                    actual.map(|(length, score)| (length, score.get())),
                    Some((8, expected))
                );
                assert!(dispatch!(backend.0, simd => accept_cached(simd, &data, 32, 16, 0, index, expected)).is_none());
                for length in 0..=3 {
                    let actual = dispatch!(backend.0, simd => accept_cached(simd, &data, 32, length, 0, index, 0));
                    assert_eq!(actual.is_some(), length >= 3 || (length == 2 && index < 2));
                }
            }
            assert!(
                dispatch!(backend.0, simd => accept_cached(simd, &data, 95, 16, 0, 0, 0)).is_none()
            );
            assert!(
                dispatch!(backend.0, simd => accept_cached(simd, &data, 32, 16, 96, 0, 0))
                    .is_none()
            );
        }
    }

    #[test]
    fn sparse_pool_reservation_preserves_positions_and_tags() {
        fn check<const BLOCK: usize, const SLOTS: usize>() {
            let mut matcher = BucketMatcher::<false, { 1 << 14 }, BLOCK, SLOTS>::new(1 << 16);
            matcher.prepare(true, 1 << 16, &[], false);
            assert!(matches!(matcher.layout, Layout::Sparse(_)));
            let Layout::Sparse(layout) = &mut matcher.layout else {
                panic!("sparse layout")
            };
            for key in 0..1030 {
                for store in 0..5 {
                    layout.push(key, (key * 5 + store) as u32, store as u8, 1 << 16);
                }
            }
            for key in 0..1030 {
                let entry = layout.entries[key];
                let (count, offset) = layout.decode_entry(entry);
                assert_eq!(count, 5);
                let Some(BlockRef::Full(index)) = SparseLayout::<{ 1 << 14 }, BLOCK>::block(offset)
                else {
                    panic!("five stores promote a starter");
                };
                for (age, &position) in layout.blocks[index][BLOCK - 5..].iter().enumerate() {
                    assert_eq!(position, (key * 5 + 4 - age) as u32);
                    if BucketMatcher::<false, { 1 << 14 }, BLOCK, SLOTS>::TAGGED {
                        assert_eq!(layout.block_tags[index][BLOCK - 5 + age], (4 - age) as u8);
                    }
                }
            }
        }
        check::<32, { (1 << 14) * 32 }>();
        check::<256, { (1 << 14) * 256 }>();
    }

    #[test]
    fn sparse_pool_reservation_bounds_hints_and_keeps_existing_capacity() {
        let mut matcher = BucketMatcher::<false, { 1 << 15 }, 64, { (1 << 15) * 64 }>::new(0);
        matcher.prepare(false, 0, &[], false);
        let Layout::Sparse(layout) = &mut matcher.layout else {
            panic!("sparse layout")
        };
        layout.reserve_populated_blocks(0);
        assert_eq!(layout.blocks.capacity(), 0);
        layout.reserve_populated_blocks(15);
        assert_eq!(layout.blocks.capacity(), 0);
        layout.reserve_populated_blocks(usize::MAX);
        assert_eq!(layout.blocks.capacity(), 1 << 14);
        assert_eq!(layout.block_tags.capacity(), 0);
        layout.reserve_populated_blocks(0);
        assert_eq!(layout.blocks.capacity(), 1 << 14);
    }

    use fearless_simd::{Level, dispatch};

    /// A payload whose last third repeats the middle third.
    ///
    /// The head deliberately differs, so position zero — which an empty table
    /// hands back for every bucket — is never a match by accident.
    fn repeated() -> Vec<u8> {
        let mut data: Vec<u8> = (0..64u32).map(|i| (i % 97) as u8 + 128).collect();
        let body: Vec<u8> = (0..64u32).map(|i| (i * 7 % 251) as u8 + 1).collect();
        data.extend_from_slice(&body);
        data.extend_from_slice(&body);
        data.extend_from_slice(&[0u8; 8]);
        data
    }

    /// Position the repeat starts at, and the distance back to its original.
    const REPEAT_AT: usize = 128;

    fn query<'a>(data: &'a [u8], cache: &'a DistanceCache, cur_ix: usize) -> MatchQuery<'a> {
        MatchQuery {
            #[cfg(feature = "experimental")]
            custom: None,
            data,
            window: data,
            mask: usize::MAX,
            cache,
            cur_ix,
            max_length: data.len() - cur_ix,
            max_backward: cur_ix,
            position_offset: 0,
            dictionary_limit: cur_ix,
            gap: 0,
            max_distance: u32::MAX as usize,
        }
    }

    fn search_at<M: Matcher>(matcher: &mut M, data: &[u8], cur_ix: usize) -> SearchResult {
        search_with(
            Level::try_detect().unwrap_or_else(Level::baseline),
            matcher,
            data,
            cur_ix,
        )
    }

    fn search_with<M: Matcher>(
        level: Level,
        matcher: &mut M,
        data: &[u8],
        cur_ix: usize,
    ) -> SearchResult {
        // The encoder derives the cache's near-miss entries before any
        // search, and the filtered probes read them as such.
        let mut cache = INITIAL_DISTANCE_CACHE;
        prepare_distance_cache(&mut cache, NUM_DISTANCE_SHORT_CODES);
        let mut out = SearchResult::empty();
        let mut stats = DictionaryStats::default();
        let query = query(data, &cache, cur_ix);
        dispatch!(level, simd => matcher.find_longest_match(simd, &mut stats, query, &mut out));
        out
    }

    /// Fills a matcher with the first `REPEAT_AT` positions of `data`.
    fn primed<M: Matcher>(mut matcher: M, data: &[u8]) -> M {
        matcher.prepare(true, data.len(), data, true);
        matcher.store_range(data, usize::MAX, 0, REPEAT_AT);
        matcher
    }

    /// The bucket shape quality five resolves to.
    const Q5_BUCKET: BucketShape = BucketShape {
        bucket_bits: 14,
        block_bits: 4,
        last_distances: 4,
    };

    /// The bucket shape quality nine resolves to.
    const Q9_BUCKET: BucketShape = BucketShape {
        bucket_bits: 15,
        block_bits: 8,
        last_distances: 16,
    };

    /// The chain shape quality five resolves to.
    const Q5_CHAIN: ChainShape = ChainShape {
        num_banks: 1,
        bank_bits: 16,
        last_distances: 4,
        max_hops: 16,
    };

    /// The chain shape quality nine resolves to.
    const Q7_CHAIN: ChainShape = ChainShape {
        num_banks: 1,
        bank_bits: 16,
        last_distances: 10,
        max_hops: 56,
    };

    const Q9_CHAIN: ChainShape = ChainShape {
        num_banks: 512,
        bank_bits: 9,
        last_distances: 16,
        max_hops: 224,
    };

    #[test]
    fn the_quick_matcher_finds_a_repeat_it_has_stored() {
        let data = repeated();
        let mut matcher = primed(QuickMatcher::<{ 1 << 16 }, 1, 5, false>::new(), &data);
        let found = search_at(&mut matcher, &data, REPEAT_AT);
        assert!(found.is_match());
        assert_eq!((found.distance, found.len), (64, 64));
    }

    #[test]
    fn every_quick_shape_finds_the_same_repeat() {
        let data = repeated();

        let mut h4 = primed(QuickMatcher::<{ 1 << 17 }, 2, 5, true>::new(), &data);
        let found = search_at(&mut h4, &data, REPEAT_AT);
        assert_eq!((found.distance, found.len), (64, 64));

        let mut h54 = primed(QuickMatcher::<{ 1 << 20 }, 2, 7, false>::new(), &data);
        let found = search_at(&mut h54, &data, REPEAT_AT);
        assert_eq!((found.distance, found.len), (64, 64));
    }

    #[test]
    fn the_bucket_matchers_find_a_repeat_they_have_stored() {
        let data = repeated();

        let mut h5 = primed(
            BucketMatcher::<false, { 1 << 14 }, 16, { (1 << 14) * 16 }>::new(0),
            &data,
        );
        let found = search_at(&mut h5, &data, REPEAT_AT);
        assert_eq!((found.distance, found.len), (64, 64));

        let mut h6 = primed(
            BucketMatcher::<true, { 1 << 15 }, 16, { (1 << 15) * 16 }>::new(0),
            &data,
        );
        let found = search_at(&mut h6, &data, REPEAT_AT);
        assert_eq!((found.distance, found.len), (64, 64));

        // The deepest bucket quality nine asks for finds the same repeat.
        let mut deep = primed(
            BucketMatcher::<false, { 1 << 15 }, 256, { (1 << 15) * 256 }>::new(0),
            &data,
        );
        let found = search_at(&mut deep, &data, REPEAT_AT);
        assert_eq!((found.distance, found.len), (64, 64));
    }

    #[test]
    fn the_chain_matchers_find_a_repeat_they_have_stored() {
        let data = repeated();
        let mut h40 = primed(ChainMatcher::<1, 16, 4>::new(Q5_CHAIN), &data);
        let found = search_at(&mut h40, &data, REPEAT_AT);
        assert_eq!((found.distance, found.len), (64, 64));

        // H42 spreads the same chains over five hundred and twelve banks.
        let mut h42 = primed(ChainMatcher::<512, 9, 16>::new(Q9_CHAIN), &data);
        let found = search_at(&mut h42, &data, REPEAT_AT);
        assert_eq!((found.distance, found.len), (64, 64));
    }

    fn probe_on<S: Simd, const BLOCK: usize>(
        simd: S,
        query: &MatchQuery<'_>,
        out: &SearchResult,
    ) -> (usize, usize, usize, usize) {
        let (best_len, found) = probe_last_distances::<S, BLOCK>(simd, query, out);
        (best_len, found.len, found.distance, found.score)
    }

    fn probes_agree<const BLOCK: usize>(
        data: &[u8],
        cache: &DistanceCache,
        cur_ix: usize,
        out: &SearchResult,
    ) {
        let query = query(data, cache, cur_ix);
        let expected =
            dispatch!(Level::fallback(), simd => probe_on::<_, BLOCK>(simd, &query, out));
        let level = Level::try_detect().unwrap_or_else(Level::baseline);
        let actual = dispatch!(level, simd => probe_on::<_, BLOCK>(simd, &query, out));
        assert_eq!(
            actual, expected,
            "block {BLOCK}, cur_ix {cur_ix}, cache {cache:?}, incoming {}",
            out.len
        );
    }

    #[test]
    fn filtered_cache_probes_make_the_scalar_probes_decisions() {
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        // Two symbols with short periodic stretches: most cached distances
        // agree for a while, so a search accepts several copies in turn and
        // the remaining codes are filtered again at each longer length.
        let mut data = Vec::new();
        while data.len() < 4096 {
            let period = (next() % 9 + 1) as usize;
            let stretch: Vec<u8> = (0..period).map(|_| b'a' + (next() % 2) as u8).collect();
            for _ in 0..(next() % 12 + 1) {
                data.extend_from_slice(&stretch);
            }
        }
        for _ in 0..4000 {
            let cur_ix = (next() % (data.len() as u64 - 64)) as usize;
            let mut cache = INITIAL_DISTANCE_CACHE;
            for entry in &mut cache[..NUM_REMEMBERED_DISTANCES] {
                *entry = match next() % 8 {
                    0 => 0,
                    1 => -((next() % 5) as i32),
                    2 => (cur_ix + (next() % 8) as usize) as i32,
                    _ => (next() % 40 + 1) as i32,
                };
            }
            prepare_distance_cache(&mut cache, NUM_DISTANCE_SHORT_CODES);
            let out = SearchResult {
                len: (next() % 5) as usize,
                ..SearchResult::empty()
            };
            probes_agree::<64>(&data, &cache, cur_ix, &out);
            probes_agree::<256>(&data, &cache, cur_ix, &out);
        }
    }

    #[test]
    fn the_derived_distance_cache_brackets_the_two_freshest_entries() {
        let mut cache = INITIAL_DISTANCE_CACHE;
        prepare_distance_cache(&mut cache, 4);
        assert_eq!(cache[4..], [0; 12]);

        prepare_distance_cache(&mut cache, 10);
        assert_eq!(cache[4..10], [3, 5, 2, 6, 1, 7]);
        // Nothing past the tenth entry is touched below the threshold.
        assert_eq!(cache[10..], [0; 6]);

        prepare_distance_cache(&mut cache, 16);
        assert_eq!(cache[10..], [10, 12, 9, 13, 8, 14]);
    }

    #[test]
    fn a_deep_bucket_remembers_more_positions_than_a_shallow_one() {
        // Every position hashes to the same bucket, so the depth is exactly
        // how far back a match can still be found.
        let data = vec![b'a'; 1024];
        fn reach<const BLOCK: usize, const SLOTS: usize>(data: &[u8]) -> usize {
            let mut matcher = BucketMatcher::<false, { 1 << 15 }, BLOCK, SLOTS>::new(0);
            matcher.prepare(true, data.len(), data, true);
            matcher.store_range(data, usize::MAX, 0, 512);
            let found = search_at(&mut matcher, data, 512);
            assert!(found.is_match());
            found.distance
        }
        assert!(reach::<16, { (1 << 15) * 16 }>(&data) <= 16);
        assert!(reach::<256, { (1 << 15) * 256 }>(&data) <= 256);
    }

    #[test]
    fn a_bucket_forgets_all_but_its_newest_sixteen_positions() {
        // Every position hashes to the same bucket, so a store past the
        // sixteenth has to push the oldest one out.
        let data = vec![b'a'; 256];
        let mut matcher = BucketMatcher::<false, { 1 << 14 }, 16, { (1 << 14) * 16 }>::new(0);
        matcher.prepare(true, data.len(), &data, true);
        matcher.store_range(&data, usize::MAX, 0, 100);
        let found = search_at(&mut matcher, &data, 100);
        assert!(found.is_match());
        assert!(found.distance <= 16);
    }

    #[test]
    fn nothing_is_found_when_the_table_holds_no_candidate() {
        let data = repeated();
        let mut matcher = QuickMatcher::<{ 1 << 16 }, 1, 5, false>::new();
        matcher.prepare(true, data.len(), &data, true);
        assert!(!search_at(&mut matcher, &data, REPEAT_AT).is_match());

        let mut chain = ChainMatcher::<1, 16, 4>::new(Q5_CHAIN);
        chain.prepare(true, data.len(), &data, true);
        assert!(!search_at(&mut chain, &data, REPEAT_AT).is_match());

        let mut bucket = BucketMatcher::<false, { 1 << 14 }, 16, { (1 << 14) * 16 }>::new(0);
        bucket.prepare(true, data.len(), &data, true);
        assert!(!search_at(&mut bucket, &data, REPEAT_AT).is_match());
    }

    #[test]
    fn a_full_preparation_clears_what_a_previous_stream_stored() {
        let data = repeated();
        let mut matcher = primed(QuickMatcher::<{ 1 << 16 }, 1, 5, false>::new(), &data);
        assert!(search_at(&mut matcher, &data, REPEAT_AT).is_match());
        matcher.prepare(false, 0, &data, true);
        assert!(!search_at(&mut matcher, &data, REPEAT_AT).is_match());

        let mut chain = primed(ChainMatcher::<1, 16, 4>::new(Q5_CHAIN), &data);
        assert!(search_at(&mut chain, &data, REPEAT_AT).is_match());
        chain.prepare(false, 0, &data, true);
        assert!(!search_at(&mut chain, &data, REPEAT_AT).is_match());

        let mut bucket = primed(
            BucketMatcher::<false, { 1 << 14 }, 16, { (1 << 14) * 16 }>::new(0),
            &data,
        );
        assert!(search_at(&mut bucket, &data, REPEAT_AT).is_match());
        bucket.prepare(false, 0, &data, true);
        assert!(!search_at(&mut bucket, &data, REPEAT_AT).is_match());
    }

    #[test]
    fn every_backend_agrees_on_the_match_it_finds() {
        let data = repeated();
        for block_bits in 4..=8 {
            let shape = BucketShape {
                bucket_bits: if block_bits <= 5 { 14 } else { 15 },
                block_bits,
                last_distances: 4,
            };
            for plan in [
                HasherPlan::H5(shape),
                HasherPlan::H6(BucketShape {
                    bucket_bits: 15,
                    ..shape
                }),
            ] {
                let mut results = Vec::new();
                for backend in crate::compressor::Backend::available() {
                    let finder = MatchFinder::from(plan);
                    let found = with_matcher!(finder, |matcher| {
                        let mut matcher = primed(matcher, &data);
                        search_with(backend.0, &mut matcher, &data, REPEAT_AT)
                    });
                    assert_eq!((found.distance, found.len), (64, 64));
                    results.push(found);
                }
                assert!(results.windows(2).all(|pair| pair[0] == pair[1]));
            }
        }
    }

    /// A payload with three copies of the body, for the stitching tests.
    fn thrice_repeated() -> Vec<u8> {
        let mut data = repeated();
        data.truncate(REPEAT_AT + 64);
        let body: Vec<u8> = data[64..REPEAT_AT].to_vec();
        data.extend_from_slice(&body);
        data.extend_from_slice(&[0u8; 8]);
        data
    }

    #[test]
    fn stitching_stores_the_three_positions_before_the_boundary() {
        let data = thrice_repeated();
        let mut matcher = QuickMatcher::<{ 1 << 16 }, 1, 5, false>::new();
        matcher.prepare(true, data.len(), &data, true);
        matcher.stitch_to_previous_block(64, REPEAT_AT, &data, usize::MAX);
        // Position 125 was stored, so the position 64 further on repeats it.
        let found = search_at(&mut matcher, &data, REPEAT_AT + 61);
        assert!(found.is_match());
        assert_eq!(found.distance, 64);
    }

    #[test]
    fn stitching_does_nothing_at_the_start_of_a_stream() {
        let data = thrice_repeated();
        let mut matcher = QuickMatcher::<{ 1 << 16 }, 1, 5, false>::new();
        matcher.prepare(true, data.len(), &data, true);
        matcher.stitch_to_previous_block(64, 2, &data, usize::MAX);
        matcher.stitch_to_previous_block(1, REPEAT_AT, &data, usize::MAX);
        assert!(!search_at(&mut matcher, &data, REPEAT_AT + 61).is_match());
    }

    #[test]
    fn the_plan_selects_the_matching_finder() {
        assert!(matches!(
            MatchFinder::from(HasherPlan::H3),
            MatchFinder::H3(_)
        ));
        assert!(matches!(
            MatchFinder::from(HasherPlan::H4),
            MatchFinder::H4(_)
        ));
        assert!(matches!(
            MatchFinder::from(HasherPlan::H54),
            MatchFinder::H54(_)
        ));
        assert!(matches!(
            MatchFinder::from(HasherPlan::Chain(Q5_CHAIN)),
            MatchFinder::H40(_)
        ));
        assert!(matches!(
            MatchFinder::from(HasherPlan::Chain(Q7_CHAIN)),
            MatchFinder::H41(_)
        ));
        assert!(matches!(
            MatchFinder::from(HasherPlan::Chain(Q9_CHAIN)),
            MatchFinder::H42(_)
        ));
        assert!(matches!(
            MatchFinder::from(HasherPlan::H5(Q5_BUCKET)),
            MatchFinder::H5Q5(_)
        ));
        assert!(matches!(
            MatchFinder::from(HasherPlan::H5(Q9_BUCKET)),
            MatchFinder::H5Q9(_)
        ));
        assert!(matches!(
            MatchFinder::from(HasherPlan::H6(Q9_BUCKET)),
            MatchFinder::H6Q9(_)
        ));
    }

    #[test]
    fn a_matcher_reports_the_cached_distance_count_its_shape_asked_for() {
        assert_eq!(
            BucketMatcher::<false, { 1 << 15 }, 256, { (1 << 15) * 256 }>::new(0)
                .last_distances_to_check(),
            16
        );
        assert_eq!(
            ChainMatcher::<1, 16, 4>::new(Q5_CHAIN).last_distances_to_check(),
            4
        );
        assert_eq!(
            ChainMatcher::<1, 16, 10>::new(Q7_CHAIN).last_distances_to_check(),
            10
        );
        assert_eq!(
            ChainMatcher::<512, 9, 16>::new(Q9_CHAIN).last_distances_to_check(),
            16
        );
        // The quick matchers always use the plain four.
        assert_eq!(
            QuickMatcher::<{ 1 << 16 }, 1, 5, false>::new().last_distances_to_check(),
            NUM_REMEMBERED_DISTANCES
        );
    }
}

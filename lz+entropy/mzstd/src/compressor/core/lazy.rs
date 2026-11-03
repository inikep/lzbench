//! The `greedy`, `lazy`, `lazy2` and `btlazy2` strategies (`zstd_lazy.c`):
//! one parser driving three searches — hash chains, a binary tree of
//! sorted suffixes (DUBT), and the row-based tag matcher.
//!
//! Like the reference, the parser is specialized on its search and depth,
//! and every search on the minimum match length (and the row finder on its
//! row size); a search picks its specialization with one `match` per call,
//! as `ZSTD_searchMax` does. The SIMD level is dispatched once per block, at
//! the entry points, and its token is passed down to the kernels: the row
//! finder's tag compare ([`row`]) and match-length counting ([`count`]).
//!
//! The reference also specializes every function on the window shape. Here
//! the parser has an extDict variant, and the candidate loops and the binary
//! tree take `EXT` as a constant chosen with one branch per search; the
//! prefix-only case reads the prefix's bytes directly. Positions are window
//! indices; an extDict candidate is read through [`mem::Window`] so a match
//! may start in the external dictionary and continue into the prefix.
//!
//! An attached dictionary (dictMatchState) has its own parser,
//! `lazy_dms_generic`, and each search a `find_dms` variant: the search runs
//! in the context's tables as without a dictionary (the prefix-only
//! candidate loops), then continues in the dictionary's tables
//! ([`DictCtx`]) with the attempts the context's search left, as the
//! reference's `dictMode == ZSTD_dictMatchState` specialization does. The
//! prefix-only and extDict parsers are untouched by it.

mod count;
mod row;

use fearless_simd::Simd;

use super::match_state::{
    BlockInput, DUBT_UNSORTED_MARK, DictMatchState, MatchState, ROW_HASH_CACHE_SIZE, aligned_range,
};
use super::mem::{self, Mem, hash_ptr};
use super::params::ROW_HASH_TAG_BITS;
use super::seq_store::{SeqStore, offbase_to_offset, offset_to_offbase, repcode_to_offbase};
use crate::shared::format::highbit32;

/// `kSearchStrength`.
const SEARCH_STRENGTH: u32 = 8;
/// `kLazySkippingStep`.
const LAZY_SKIPPING_STEP: usize = 8;
/// Offset placeholder before any match is found.
const NO_OFFBASE: u32 = 999_999_999;
const ROW_HASH_CACHE_MASK: u32 = ROW_HASH_CACHE_SIZE as u32 - 1;

/// Per-block constants of a search.
#[derive(Copy, Clone, Debug)]
struct Ctx<'a> {
    window: mem::Window<'a>,
    /// The prefix's bytes: index `i` is `src[i - base]`. They may extend
    /// past `iend`, which bounds every match.
    src: &'a [u8],
    /// Index of the prefix's first byte (`dictLimit`).
    base: u32,
    /// Index one past the block.
    iend: u32,
    mls: u32,
    row_log: u32,
    ext: bool,
}

impl<'a> Ctx<'a> {
    fn new(block: &BlockInput<'a>, mls: u32, row_log: u32, ext: bool) -> Self {
        let prefix = block.window.prefix;
        Self {
            window: block.window,
            src: prefix.slice(prefix.start(), prefix.end()),
            base: prefix.start(),
            iend: block.end,
            mls,
            row_log,
            ext,
        }
    }

    const fn dict_limit(&self) -> u32 {
        self.base
    }

    /// The bytes from prefix index `p` to the block end; empty when `p` is
    /// outside them.
    #[inline(always)]
    fn tail(&self, p: u32) -> &'a [u8] {
        self.src
            .get(p.wrapping_sub(self.base) as usize..self.iend.wrapping_sub(self.base) as usize)
            .unwrap_or_default()
    }

    /// Whether the four bytes at prefix index `p` repeat `offset` bytes
    /// earlier (`MEM_read32(ip - offset) == MEM_read32(ip)`), for an offset
    /// above zero. One slice spanning both words is cut, so one check covers
    /// both loads; a span that leaves the prefix, which the parser's offset
    /// bounds rule out, does not match.
    #[inline(always)]
    fn repeats(&self, p: u32, offset: u32) -> bool {
        let at = p.wrapping_sub(self.base) as usize;
        let Some(from) = at.checked_sub(offset as usize) else {
            return false;
        };
        let Some(span) = self.src.get(from..at + 4) else {
            return false;
        };
        match (span.first_chunk::<4>(), span.last_chunk::<4>()) {
            (Some(earlier), Some(word)) => earlier == word,
            _ => false,
        }
    }

    /// `ZSTD_count` of the prefix positions `a` and `b < a`, up to the block
    /// end.
    #[inline(always)]
    fn count<S: Simd>(&self, simd: S, a: u32, b: u32) -> usize {
        self.count_to(simd, a, b, self.iend)
    }

    /// `ZSTD_count` of the prefix positions `a` and `b`, up to `limit`.
    #[inline(always)]
    fn count_to<S: Simd>(&self, simd: S, a: u32, b: u32, limit: u32) -> usize {
        if a >= limit {
            return 0;
        }
        count::count(
            simd,
            self.src,
            (a - self.base) as usize,
            (b - self.base) as usize,
            (limit - self.base) as usize,
        )
    }

    /// `ZSTD_hashPtr` of prefix index `index` over `bits` bits for a
    /// minimum match of `MLS` (4, 5 or 6).
    ///
    /// The bytes hashed are copied into a fixed-size word first — one bounds
    /// check, and no index kept around for a panic message — and hashed by
    /// [`hash_ptr`] itself. A word past the prefix, which the reference's
    /// index invariants rule out, hashes as zeros.
    #[inline(always)]
    fn hash<const MLS: u32>(&self, index: u32, bits: u32) -> usize {
        let at = index.wrapping_sub(self.base) as usize;
        if MLS == 4 {
            let word: [u8; 4] = self
                .src
                .get(at..at + 4)
                .and_then(|bytes| bytes.try_into().ok())
                .unwrap_or_default();
            hash_ptr(&Mem::new(&word, 0), 0, bits, MLS)
        } else {
            let word: [u8; 8] = self
                .src
                .get(at..at + 8)
                .and_then(|bytes| bytes.try_into().ok())
                .unwrap_or_default();
            hash_ptr(&Mem::new(&word, 0), 0, bits, MLS)
        }
    }

    /// One step of the binary tree with an external dictionary: the common
    /// length of the runs at window indices `a` and `b`, stopping before
    /// `limit` for `a`, and whether `b`'s run sorts below `a`'s — the byte
    /// after the common run is smaller — when that byte is before `limit`
    /// (`false` otherwise). [`TreeRun::step`] takes the prefix-only case.
    #[inline(always)]
    fn ext_tree_step<S: Simd>(&self, simd: S, a: u32, b: u32, limit: u32) -> (usize, bool) {
        let n = self.count_virtual(simd, a, b, limit);
        let at = n as u32;
        let lower = a + at < limit && self.window.byte_at(b + at) < self.window.byte_at(a + at);
        (n, lower)
    }

    /// [`mem::Window::count_virtual`] with the prefix-only case, the common
    /// one, counted by [`count`].
    #[inline(always)]
    fn count_virtual<S: Simd>(&self, simd: S, a: u32, b: u32, limit: u32) -> usize {
        if a >= self.base && b >= self.base {
            self.count_to(simd, a, b, limit)
        } else {
            self.window.count_virtual(a, b, limit)
        }
    }

    /// Length of the match of `ip` with `m` in the external dictionary,
    /// when its first four bytes match; `0` otherwise.
    #[inline(never)]
    fn ext_candidate_length(&self, ip: u32, m: u32) -> usize {
        let dict_limit = self.dict_limit();
        if self.window.ext.read32(m) == self.window.prefix.read32(ip) {
            return self
                .window
                .count_2segments(ip + 4, m + 4, self.iend, dict_limit, dict_limit)
                + 4;
        }
        0
    }
}

/// The searched position of one hash chain or row search, cut once: its
/// bytes up to the block end, and the word a candidate must share with it
/// to beat the best length so far — the reference's
/// `MEM_read32(match + ml - 3) == MEM_read32(ip + ml - 3)` test.
struct Probe<'a> {
    /// The prefix's bytes.
    src: &'a [u8],
    /// Index of `src[0]`.
    base: u32,
    /// The bytes from the searched position to the block end, which may
    /// come before the prefix's end.
    current: &'a [u8],
    /// `current[ml - 3..ml + 1]` for the best length `ml`.
    word: u32,
}

impl<'a> Probe<'a> {
    /// The probe of `ip`, aimed at the initial best length of 3.
    #[inline(always)]
    fn new(ctx: &Ctx<'a>, ip: u32) -> Self {
        // The block may end before the prefix does (long distance matching
        // runs the finder on parts of a block), and no match runs past it.
        let current = ctx
            .src
            .get((ip - ctx.base) as usize..(ctx.iend - ctx.base) as usize)
            .unwrap_or_default();
        let mut probe = Self {
            src: ctx.src,
            base: ctx.base,
            current,
            word: 0,
        };
        probe.aim(3);
        probe
    }

    /// Little-endian `u32` at `at` of `bytes`, `0` past their end.
    #[inline(always)]
    fn word(bytes: &[u8], at: usize) -> u32 {
        let at = at & u32::MAX as usize;
        bytes
            .get(at..at + 4)
            .and_then(|word| word.try_into().ok())
            .map_or(0, u32::from_le_bytes)
    }

    /// Sets the best length to beat to `ml`, which is below the window's
    /// length.
    #[inline(always)]
    fn aim(&mut self, ml: usize) {
        self.word = Self::word(self.current, ml - 3);
    }

    /// Length of the match with prefix index `m < ip` when it shares the
    /// aimed word with the searched position, `0` otherwise.
    #[inline(always)]
    fn length<S: Simd>(&self, simd: S, m: u32, ml: usize) -> usize {
        let at = (m - self.base) as usize;
        // Window positions are 32-bit; the mask states it, so the compiler
        // sees that `probe + 4` cannot overflow and checks only the end.
        let probe = (at + (ml - 3)) & u32::MAX as usize;
        let Some(word) = self.src.get(probe..probe + 4) else {
            return 0;
        };
        if u32::from_le_bytes(word.try_into().unwrap_or_default()) != self.word {
            return 0;
        }
        count::common_run(simd, self.current, self.src.get(at..).unwrap_or_default())
    }

    /// The candidate test of the hash chain and row searches for window
    /// index `m`: its match length when it may beat `ml`, `0` otherwise.
    /// `EXT` states whether `m` may lie in the external dictionary.
    #[inline(always)]
    fn candidate<S: Simd, const EXT: bool>(
        &self,
        simd: S,
        ctx: &Ctx<'_>,
        ip: u32,
        m: u32,
        ml: usize,
    ) -> usize {
        if !EXT || m >= self.base {
            self.length(simd, m, ml)
        } else {
            ctx.ext_candidate_length(ip, m)
        }
    }
}

/// The best of `candidates` for `ip` (`ZSTD_RowFindBestMatch`'s final
/// loop), its offset in `offset`; `EXT` as for [`Probe::candidate`].
#[inline(always)]
fn best_candidate<S: Simd, const EXT: bool>(
    simd: S,
    ctx: &Ctx<'_>,
    ip: u32,
    candidates: &[u32],
    offset: &mut u32,
) -> usize {
    let mut ml: usize = 4 - 1;
    let mut probe = Probe::new(ctx, ip);
    for &match_index in candidates {
        let current_ml = probe.candidate::<S, EXT>(simd, ctx, ip, match_index, ml);
        if current_ml > ml {
            ml = current_ml;
            *offset = offset_to_offbase(ip - match_index);
            if ip as usize + current_ml == ctx.iend as usize {
                break;
            }
            probe.aim(ml);
        }
    }
    ml
}

/// One walk down a hash chain (`ZSTD_HcFindBestMatch`'s loop).
struct ChainWalk<'c> {
    chain: &'c [u32],
    chain_mask: u32,
    /// The first candidate, from the hash table.
    first: u32,
    low_limit: u32,
    /// The walk stops after this index: older ones are overwritten.
    min_chain: u32,
    nb_attempts: u32,
}

/// The longest match for `ip` along `walk`, its offset in `offset`, and
/// the attempts left for an attached dictionary; `EXT` as for
/// [`Probe::candidate`].
#[inline(always)]
fn walk_chain<S: Simd, const EXT: bool>(
    simd: S,
    ctx: &Ctx<'_>,
    walk: &ChainWalk<'_>,
    ip: u32,
    offset: &mut u32,
) -> (usize, u32) {
    let ChainWalk {
        chain,
        chain_mask,
        first: mut match_index,
        low_limit,
        min_chain,
        mut nb_attempts,
    } = *walk;
    let mut ml: usize = 4 - 1;
    let mut probe = Probe::new(ctx, ip);
    while match_index >= low_limit && nb_attempts > 0 {
        let current_ml = probe.candidate::<S, EXT>(simd, ctx, ip, match_index, ml);
        if current_ml > ml {
            ml = current_ml;
            *offset = offset_to_offbase(ip - match_index);
            if ip as usize + current_ml == ctx.iend as usize {
                break;
            }
            probe.aim(ml);
        }
        if match_index <= min_chain {
            break;
        }
        match_index = chain[(match_index & chain_mask) as usize];
        nb_attempts -= 1;
    }
    (ml, nb_attempts)
}

/// A match search (`searchMethod_e`), specialized by the parser.
trait Search {
    /// Whether the search keeps the row finder's hash cache.
    const ROW: bool;

    /// The search's view of the match state for one block: the tables it
    /// reads and writes, and the constants it derives from the parameters.
    type State<'t>;

    /// The state of `ms` for one block of `ctx`.
    fn state<'t>(ms: &'t mut MatchState, ctx: &Ctx<'_>) -> Self::State<'t>;

    /// `ZSTD_searchMax`: the longest match at `ip`, its offset in `offset`.
    /// `lazy_skipping` is the parser's `ms->lazySkipping`.
    fn find<S: Simd>(
        simd: S,
        state: &mut Self::State<'_>,
        ctx: &Ctx<'_>,
        ip: u32,
        lazy_skipping: bool,
        offset: &mut u32,
    ) -> usize;

    /// `ZSTD_searchMax` in dictMatchState mode: [`Self::find`] in the
    /// context's tables, then in the attached dictionary's with the
    /// attempts the context's search left.
    fn find_dms<S: Simd>(
        simd: S,
        state: &mut Self::State<'_>,
        ctx: &Ctx<'_>,
        dict: &DictCtx<'_>,
        ip: u32,
        lazy_skipping: bool,
        offset: &mut u32,
    ) -> usize;

    /// `ZSTD_row_fillHashCache` from the next position to insert; nothing
    /// for searches without the cache, so the parser calls it
    /// unconditionally.
    fn fill_cache(_state: &mut Self::State<'_>, _ctx: &Ctx<'_>, _i_limit: i64) {}
}

/// Hash chains (`search_hashChain`).
struct HashChain;
/// The DUBT binary tree (`search_binaryTree`).
struct BinaryTree;
/// The row-based tag matcher (`search_rowHash`) for a minimum match of
/// `MLS` (4, 5 or 6) and rows of `SLOTS` slots (16, 32 or 64).
///
/// Unlike the other searches it is specialized once per block (`row_block!`)
/// rather than per search: each parser then inlines one row search at its
/// call sites instead of all nine, and no search branches on the parameters
/// (2% faster at the lazy2 levels, with 7% fewer mispredicted branches in
/// Callgrind's model).
struct RowHash<const MLS: u32, const SLOTS: usize>;
/// The row-based tag matcher choosing its [`RowHash`] specialization per
/// search, as `ZSTD_searchMax` does. The dictMatchState parsers use it, so
/// their rarely run code is not multiplied by nine.
struct RowHashAny;

impl Search for HashChain {
    const ROW: bool = false;
    type State<'t> = &'t mut MatchState;

    fn state<'t>(ms: &'t mut MatchState, _ctx: &Ctx<'_>) -> Self::State<'t> {
        ms
    }

    #[inline(always)]
    fn find<S: Simd>(
        simd: S,
        ms: &mut &mut MatchState,
        ctx: &Ctx<'_>,
        ip: u32,
        lazy_skipping: bool,
        offset: &mut u32,
    ) -> usize {
        match ctx.mls {
            5 => hc_find_best_match::<S, 5>(simd, ms, ctx, ip, lazy_skipping, offset).0,
            6 => hc_find_best_match::<S, 6>(simd, ms, ctx, ip, lazy_skipping, offset).0,
            _ => hc_find_best_match::<S, 4>(simd, ms, ctx, ip, lazy_skipping, offset).0,
        }
    }

    #[inline(always)]
    fn find_dms<S: Simd>(
        simd: S,
        ms: &mut &mut MatchState,
        ctx: &Ctx<'_>,
        dict: &DictCtx<'_>,
        ip: u32,
        lazy_skipping: bool,
        offset: &mut u32,
    ) -> usize {
        let (d, l) = (dict, lazy_skipping);
        match ctx.mls {
            5 => hc_find_best_match_dms::<S, 5>(simd, ms, ctx, d, ip, l, offset),
            6 => hc_find_best_match_dms::<S, 6>(simd, ms, ctx, d, ip, l, offset),
            _ => hc_find_best_match_dms::<S, 4>(simd, ms, ctx, d, ip, l, offset),
        }
    }
}

impl Search for BinaryTree {
    const ROW: bool = false;
    type State<'t> = &'t mut MatchState;

    fn state<'t>(ms: &'t mut MatchState, _ctx: &Ctx<'_>) -> Self::State<'t> {
        ms
    }

    #[inline(always)]
    fn find<S: Simd>(
        simd: S,
        ms: &mut &mut MatchState,
        ctx: &Ctx<'_>,
        ip: u32,
        _lazy_skipping: bool,
        offset: &mut u32,
    ) -> usize {
        match (ctx.mls, ctx.ext) {
            (5, false) => bt_find_best_match::<S, 5, false>(simd, ms, ctx, ip, offset),
            (6, false) => bt_find_best_match::<S, 6, false>(simd, ms, ctx, ip, offset),
            (_, false) => bt_find_best_match::<S, 4, false>(simd, ms, ctx, ip, offset),
            (5, true) => bt_find_best_match::<S, 5, true>(simd, ms, ctx, ip, offset),
            (6, true) => bt_find_best_match::<S, 6, true>(simd, ms, ctx, ip, offset),
            (_, true) => bt_find_best_match::<S, 4, true>(simd, ms, ctx, ip, offset),
        }
    }

    #[inline(always)]
    fn find_dms<S: Simd>(
        simd: S,
        ms: &mut &mut MatchState,
        ctx: &Ctx<'_>,
        dict: &DictCtx<'_>,
        ip: u32,
        _lazy_skipping: bool,
        offset: &mut u32,
    ) -> usize {
        match ctx.mls {
            5 => bt_find_best_match_dms::<S, 5>(simd, ms, ctx, dict, ip, offset),
            6 => bt_find_best_match_dms::<S, 6>(simd, ms, ctx, dict, ip, offset),
            _ => bt_find_best_match_dms::<S, 4>(simd, ms, ctx, dict, ip, offset),
        }
    }
}

impl<const MLS: u32, const SLOTS: usize> Search for RowHash<MLS, SLOTS> {
    const ROW: bool = true;
    type State<'t> = RowState<'t>;

    fn state<'t>(ms: &'t mut MatchState, ctx: &Ctx<'_>) -> Self::State<'t> {
        RowState::new(ms, ctx)
    }

    #[inline(always)]
    fn find<S: Simd>(
        simd: S,
        state: &mut RowState<'_>,
        ctx: &Ctx<'_>,
        ip: u32,
        lazy_skipping: bool,
        offset: &mut u32,
    ) -> usize {
        row_find_best_match::<S, MLS, SLOTS>(simd, state, ctx, ip, lazy_skipping, offset).0
    }

    #[inline(always)]
    fn find_dms<S: Simd>(
        simd: S,
        state: &mut RowState<'_>,
        ctx: &Ctx<'_>,
        dict: &DictCtx<'_>,
        ip: u32,
        lazy_skipping: bool,
        offset: &mut u32,
    ) -> usize {
        let (s, d, l) = (state, dict, lazy_skipping);
        row_find_best_match_dms::<S, MLS, SLOTS>(simd, s, ctx, d, ip, l, offset)
    }

    fn fill_cache(state: &mut RowState<'_>, ctx: &Ctx<'_>, i_limit: i64) {
        let idx = state.next_to_update;
        let bits = state.bits;
        let tables = (&*state.tags, &*state.slots);
        row_fill_hash_cache::<MLS>(state.cache, tables, ctx, bits, idx, i_limit);
    }
}

impl Search for RowHashAny {
    const ROW: bool = true;
    type State<'t> = RowState<'t>;

    fn state<'t>(ms: &'t mut MatchState, ctx: &Ctx<'_>) -> Self::State<'t> {
        RowState::new(ms, ctx)
    }

    #[inline(always)]
    fn find<S: Simd>(
        simd: S,
        state: &mut RowState<'_>,
        ctx: &Ctx<'_>,
        ip: u32,
        lazy_skipping: bool,
        offset: &mut u32,
    ) -> usize {
        let (s, l) = (state, lazy_skipping);
        match (ctx.mls, ctx.row_log) {
            (5, 4) => RowHash::<5, 16>::find(simd, s, ctx, ip, l, offset),
            (5, 5) => RowHash::<5, 32>::find(simd, s, ctx, ip, l, offset),
            (5, _) => RowHash::<5, 64>::find(simd, s, ctx, ip, l, offset),
            (6, 4) => RowHash::<6, 16>::find(simd, s, ctx, ip, l, offset),
            (6, 5) => RowHash::<6, 32>::find(simd, s, ctx, ip, l, offset),
            (6, _) => RowHash::<6, 64>::find(simd, s, ctx, ip, l, offset),
            (_, 4) => RowHash::<4, 16>::find(simd, s, ctx, ip, l, offset),
            (_, 5) => RowHash::<4, 32>::find(simd, s, ctx, ip, l, offset),
            (_, _) => RowHash::<4, 64>::find(simd, s, ctx, ip, l, offset),
        }
    }

    #[inline(always)]
    fn find_dms<S: Simd>(
        simd: S,
        state: &mut RowState<'_>,
        ctx: &Ctx<'_>,
        dict: &DictCtx<'_>,
        ip: u32,
        lazy_skipping: bool,
        offset: &mut u32,
    ) -> usize {
        let (s, d, l) = (state, dict, lazy_skipping);
        match (ctx.mls, ctx.row_log) {
            (5, 4) => RowHash::<5, 16>::find_dms(simd, s, ctx, d, ip, l, offset),
            (5, 5) => RowHash::<5, 32>::find_dms(simd, s, ctx, d, ip, l, offset),
            (5, _) => RowHash::<5, 64>::find_dms(simd, s, ctx, d, ip, l, offset),
            (6, 4) => RowHash::<6, 16>::find_dms(simd, s, ctx, d, ip, l, offset),
            (6, 5) => RowHash::<6, 32>::find_dms(simd, s, ctx, d, ip, l, offset),
            (6, _) => RowHash::<6, 64>::find_dms(simd, s, ctx, d, ip, l, offset),
            (_, 4) => RowHash::<4, 16>::find_dms(simd, s, ctx, d, ip, l, offset),
            (_, 5) => RowHash::<4, 32>::find_dms(simd, s, ctx, d, ip, l, offset),
            (_, _) => RowHash::<4, 64>::find_dms(simd, s, ctx, d, ip, l, offset),
        }
    }

    fn fill_cache(state: &mut RowState<'_>, ctx: &Ctx<'_>, i_limit: i64) {
        // Filling the cache depends on the minimum match; the rows it
        // prefetches follow `ctx.row_log`, whatever the row size named here.
        match ctx.mls {
            5 => RowHash::<5, 16>::fill_cache(state, ctx, i_limit),
            6 => RowHash::<6, 16>::fill_cache(state, ctx, i_limit),
            _ => RowHash::<4, 16>::fill_cache(state, ctx, i_limit),
        }
    }
}

/// `lowLimit` of the hash chain and row searches at `curr`.
const fn search_low_limit(ms: &MatchState, curr: u32) -> u32 {
    let max_distance = 1u32 << ms.cparams.window_log;
    let lowest_valid = ms.window.low_limit;
    let within_max_distance = if curr.wrapping_sub(lowest_valid) > max_distance {
        curr - max_distance
    } else {
        lowest_valid
    };
    if ms.loaded_dict_end != 0 {
        lowest_valid
    } else {
        within_max_distance
    }
}

// ---------------------------------------------------------------------------
// Hash chain.

/// `ZSTD_insertAndFindFirstIndex_internal`.
#[inline(always)]
fn insert_and_find_first_index<const MLS: u32>(
    ms: &mut MatchState,
    ctx: &Ctx<'_>,
    ip: u32,
    lazy_skipping: bool,
) -> u32 {
    let hash_log = ms.cparams.hash_log;
    let chain_mask = (1u32 << ms.cparams.chain_log) - 1;
    let mut idx = ms.next_to_update;
    // Local slices keep the tables' pointers and lengths in registers: the
    // stores below could otherwise alias the vectors that own them.
    let hash_table = ms.hash_table.as_mut_slice();
    let chain_table = ms.chain_table.as_mut_slice();
    while idx < ip {
        let h = ctx.hash::<MLS>(idx, hash_log);
        chain_table[(idx & chain_mask) as usize] = hash_table[h];
        hash_table[h] = idx;
        idx += 1;
        if lazy_skipping {
            break;
        }
    }
    let first = hash_table[ctx.hash::<MLS>(ip, hash_log)];
    ms.next_to_update = ip;
    first
}

/// `ZSTD_HcFindBestMatch` in the context's tables: the longest match and
/// the attempts left for an attached dictionary.
#[inline(always)]
fn hc_find_best_match<S: Simd, const MLS: u32>(
    simd: S,
    ms: &mut MatchState,
    ctx: &Ctx<'_>,
    ip: u32,
    lazy_skipping: bool,
    offset: &mut u32,
) -> (usize, u32) {
    let cparams = ms.cparams;
    let chain_size = 1u32 << cparams.chain_log;
    let chain_mask = chain_size - 1;
    let curr = ip;
    let low_limit = search_low_limit(ms, curr);
    let min_chain = curr.saturating_sub(chain_size);
    let nb_attempts = 1u32 << cparams.search_log;

    let first = insert_and_find_first_index::<MLS>(ms, ctx, ip, lazy_skipping);
    let walk = ChainWalk {
        chain: &ms.chain_table,
        chain_mask,
        first,
        low_limit,
        min_chain,
        nb_attempts,
    };
    if ctx.ext {
        walk_chain::<S, true>(simd, ctx, &walk, ip, offset)
    } else {
        walk_chain::<S, false>(simd, ctx, &walk, ip, offset)
    }
}

// ---------------------------------------------------------------------------
// Binary tree (DUBT).

/// `ZSTD_updateDUBT`.
#[inline(always)]
fn update_dubt<const MLS: u32>(ms: &mut MatchState, ctx: &Ctx<'_>, ip: u32) {
    let start = ms.next_to_update;
    ms.next_to_update = ip;
    if start >= ip {
        return;
    }
    let hash_log = ms.cparams.hash_log;
    let bt_mask = (1u32 << (ms.cparams.chain_log - 1)) - 1;
    // The words hashed, cut once (see [`row_update_impl`]); the
    // reference's index invariants keep them in the prefix.
    let from = start.wrapping_sub(ctx.base) as usize;
    let Some(words) = ctx.src.get(from..from + (ip - start) as usize + 7) else {
        return update_dubt_checked::<MLS>(ms, ctx, start, ip);
    };
    let hash_table = ms.hash_table.as_mut_slice();
    // A node's two cells: the next candidate in the unsorted list, and the
    // mark that the node is not sorted yet.
    let nodes = ms.chain_table.as_chunks_mut::<2>().0;
    let mut idx = start;
    for word in words.windows(8) {
        let word = u64::from_le_bytes(word.try_into().unwrap_or_default());
        let h = mem::hash_word::<MLS>(word, hash_log);
        let match_index = core::mem::replace(&mut hash_table[h], idx);
        nodes[(idx & bt_mask) as usize] = [match_index, DUBT_UNSORTED_MARK];
        idx = idx.wrapping_add(1);
    }
}

/// [`update_dubt`] hashing through [`Ctx::hash`], for a range whose
/// words leave the prefix.
#[cold]
#[inline(never)]
fn update_dubt_checked<const MLS: u32>(ms: &mut MatchState, ctx: &Ctx<'_>, start: u32, ip: u32) {
    let hash_log = ms.cparams.hash_log;
    let bt_mask = (1u32 << (ms.cparams.chain_log - 1)) - 1;
    for idx in start..ip {
        let h = ctx.hash::<MLS>(idx, hash_log);
        let match_index = ms.hash_table[h];
        let slot = 2 * (idx & bt_mask) as usize;
        ms.hash_table[h] = idx;
        ms.chain_table[slot] = match_index;
        ms.chain_table[slot + 1] = DUBT_UNSORTED_MARK;
    }
}

/// A tree link not to write (`dummy32`): past every chain table, so the
/// check that guards a write also skips it.
const NO_LINK: usize = usize::MAX;

/// Writes `value` to the tree link `link`: a cell of the chain table, or
/// [`NO_LINK`].
#[inline(always)]
fn set_link(bt: &mut [u32], link: usize, value: u32) {
    if let Some(cell) = bt.get_mut(link) {
        *cell = value;
    }
}

/// The open links of a tree walk (`smallerPtr`, `largerPtr`) and the
/// lengths its run is known to share with each side.
struct TreeLinks {
    smaller: usize,
    larger: usize,
    common_smaller: usize,
    common_larger: usize,
}

impl TreeLinks {
    /// The links of the node at `slot` (its pair of cells).
    #[inline(always)]
    const fn new(slot: usize) -> Self {
        Self {
            smaller: slot,
            larger: slot + 1,
            common_smaller: 0,
            common_larger: 0,
        }
    }

    /// The length every candidate below the open links shares with the run.
    #[inline(always)]
    fn common(&self) -> usize {
        self.common_smaller.min(self.common_larger)
    }

    /// Attaches candidate `m` (whose cells start at `next`, sharing
    /// `match_length` bytes with the run) to the side it sorts on, and
    /// returns that side's child to visit next, or `None` when `m` is at
    /// or below `bt_low` and the walk ends.
    ///
    /// The side is a branch, as in the reference, although it is taken
    /// either way about as often: the walk is a chain of dependent cache
    /// misses, and a predicted branch lets the next node's load start
    /// before the comparison that picks it resolves, where a select would
    /// wait for it on every step.
    #[inline(always)]
    fn descend(
        &mut self,
        bt: &mut [u32],
        m: u32,
        next: usize,
        match_length: usize,
        lower: bool,
        bt_low: u32,
    ) -> Option<u32> {
        let ends = m <= bt_low;
        let child = if lower {
            set_link(bt, self.smaller, m);
            self.common_smaller = match_length;
            if ends {
                self.smaller = NO_LINK;
                return None;
            }
            self.smaller = next + 1;
            next + 1
        } else {
            set_link(bt, self.larger, m);
            self.common_larger = match_length;
            if ends {
                self.larger = NO_LINK;
                return None;
            }
            self.larger = next;
            next
        };
        bt.get(child).copied()
    }

    /// Closes both open links (`*smallerPtr = *largerPtr = 0`).
    #[inline(always)]
    fn close(&self, bt: &mut [u32]) {
        set_link(bt, self.smaller, 0);
        set_link(bt, self.larger, 0);
    }
}

/// The run a tree walk sorts candidates against: the bytes from position
/// `curr` to the walk's limit. Without an external dictionary (`!EXT`)
/// they are cut from the prefix once, so a step costs one check per side.
#[derive(Copy, Clone)]
struct TreeRun<'a> {
    curr: u32,
    limit: u32,
    /// The run's bytes (prefix only).
    current: &'a [u8],
}

impl<'a> TreeRun<'a> {
    #[inline(always)]
    fn new<const EXT: bool>(ctx: &Ctx<'a>, curr: u32, limit: u32) -> Self {
        let current = if EXT {
            &[]
        } else {
            ctx.src
                .get(curr.wrapping_sub(ctx.base) as usize..limit.wrapping_sub(ctx.base) as usize)
                .unwrap_or_default()
        };
        Self {
            curr,
            limit,
            current,
        }
    }

    /// The run's length.
    #[inline(always)]
    const fn len(&self) -> usize {
        (self.limit - self.curr) as usize
    }

    /// One step of a tree walk: the common length of
    /// the run and candidate `m`, which agree on their first `from` bytes,
    /// and whether the candidate sorts below the run.
    #[inline(always)]
    fn step<S: Simd, const EXT: bool>(
        &self,
        simd: S,
        ctx: &Ctx<'_>,
        m: u32,
        from: usize,
    ) -> (usize, bool) {
        if EXT {
            let at = from as u32;
            let (n, lower) = ctx.ext_tree_step(simd, self.curr + at, m + at, self.limit);
            return (from + n, lower);
        }
        let (current, src) = (self.current, ctx.src);
        // Lengths and window offsets fit in 32 bits; saying so lets the
        // compiler drop the overflow checks of the ranges below.
        let from = from & u32::MAX as usize;
        let at = m.wrapping_sub(ctx.base) as usize;
        // `ZSTD_count` inline, a word at a time, for the first
        // `TREE_INLINE_WORDS` words: a call here would make the walk keep its
        // state in memory across every step. Longer runs (run-length input
        // matches whole blocks) continue in the vector scan.
        let mut n = from;
        let mut words = 0;
        while let (Some(x), Some(y)) = (current.get(n..n + 8), src.get(at + n..at + n + 8)) {
            let x = u64::from_le_bytes(x.try_into().unwrap_or_default());
            let y = u64::from_le_bytes(y.try_into().unwrap_or_default());
            let difference = x ^ y;
            if difference != 0 {
                // The words read big-endian order as their bytes do, so the
                // first difference orders them.
                let lower = y.swap_bytes() < x.swap_bytes();
                return (n + (difference.trailing_zeros() >> 3) as usize, lower);
            }
            n += 8;
            words += 1;
            if words == TREE_INLINE_WORDS {
                return long_tree_step(simd, current, src.get(at..).unwrap_or_default(), n);
            }
        }
        while let (Some(x), Some(y)) = (current.get(n), src.get(at + n))
            && x == y
        {
            n += 1;
        }
        let lower = match (current.get(n), src.get(at + n)) {
            (Some(&x), Some(&y)) => y < x,
            _ => false,
        };
        (n, lower)
    }
}

/// Words a tree step compares inline before handing a run to
/// [`long_tree_step`].
const TREE_INLINE_WORDS: usize = 4;

/// The rest of a long tree step, past the `matched` bytes compared inline:
/// the common length of `current` and `candidate` by the vector scan, and
/// whether the candidate sorts below the run. Out of line and cold, so the
/// walk keeps its registers.
#[cold]
#[inline(never)]
fn long_tree_step<S: Simd>(
    simd: S,
    current: &[u8],
    candidate: &[u8],
    matched: usize,
) -> (usize, bool) {
    let (current_rest, candidate_rest) = (
        current.get(matched..).unwrap_or_default(),
        candidate.get(matched..).unwrap_or_default(),
    );
    let n = matched + mem::count_slices_simd(simd, current_rest, candidate_rest);
    let lower = match (current.get(n), candidate.get(n)) {
        (Some(&x), Some(&y)) => y < x,
        _ => false,
    };
    (n, lower)
}

/// `ZSTD_insertDUBT1`: sorts one inserted but unsorted position.
#[inline(always)]
fn insert_dubt1<S: Simd, const EXT: bool>(
    simd: S,
    ms: &mut MatchState,
    ctx: &Ctx<'_>,
    curr: u32,
    mut nb_compares: u32,
    bt_low: u32,
) {
    let cparams = ms.cparams;
    let bt_mask = (1u32 << (cparams.chain_log - 1)) - 1;
    let dict_limit = ctx.dict_limit();
    let iend = if curr >= dict_limit {
        ctx.iend
    } else {
        dict_limit
    };
    let run = TreeRun::new::<EXT>(ctx, curr, iend);
    let window_valid = ms.window.low_limit;
    let max_distance = 1u32 << cparams.window_log;
    let window_low = if curr.wrapping_sub(window_valid) > max_distance {
        curr - max_distance
    } else {
        window_valid
    };
    let bt = ms.chain_table.as_mut_slice();
    let mut links = TreeLinks::new(2 * (curr & bt_mask) as usize);
    let mut match_index = bt[links.smaller];

    while nb_compares > 0 && match_index > window_low {
        let next = 2 * (match_index & bt_mask) as usize;
        let (match_length, lower) = run.step::<S, EXT>(simd, ctx, match_index, links.common());
        if match_length == run.len() {
            break;
        }
        match links.descend(bt, match_index, next, match_length, lower, bt_low) {
            Some(child) => match_index = child,
            None => break,
        }
        nb_compares -= 1;
    }
    links.close(bt);
}

/// `ZSTD_DUBT_findBestMatch` in the context's tree: the longest match and
/// the comparisons left for an attached dictionary, none once a match
/// reaches the block's end.
#[inline(always)]
fn dubt_find_best_match<S: Simd, const MLS: u32, const EXT: bool>(
    simd: S,
    ms: &mut MatchState,
    ctx: &Ctx<'_>,
    ip: u32,
    off_base: &mut u32,
) -> (usize, u32) {
    let cparams = ms.cparams;
    let hash_log = cparams.hash_log;
    let h = ctx.hash::<MLS>(ip, hash_log);
    let mut match_index = ms.hash_table[h];
    let curr = ip;
    let window_low = ms.lowest_match_index(curr, cparams.window_log);
    let bt_mask = (1u32 << (cparams.chain_log - 1)) - 1;
    let bt_low = curr.saturating_sub(bt_mask);
    let unsort_limit = bt_low.max(window_low);
    let mut nb_compares = 1u32 << cparams.search_log;
    let mut nb_candidates = nb_compares;
    let mut previous_candidate = 0u32;

    // Reach the end of the unsorted candidates list.
    {
        let bt = &mut ms.chain_table;
        let mut next_candidate = 2 * (match_index & bt_mask) as usize;
        while match_index > unsort_limit
            && bt[next_candidate + 1] == DUBT_UNSORTED_MARK
            && nb_candidates > 1
        {
            bt[next_candidate + 1] = previous_candidate;
            previous_candidate = match_index;
            match_index = bt[next_candidate];
            next_candidate = 2 * (match_index & bt_mask) as usize;
            nb_candidates -= 1;
        }
        // Nullify the last candidate if it is still unsorted.
        if match_index > unsort_limit && bt[next_candidate + 1] == DUBT_UNSORTED_MARK {
            bt[next_candidate] = 0;
            bt[next_candidate + 1] = 0;
        }
    }

    // Batch sort the stacked candidates.
    match_index = previous_candidate;
    while match_index != 0 {
        let next_candidate_idx = ms.chain_table[2 * (match_index & bt_mask) as usize + 1];
        insert_dubt1::<S, EXT>(simd, ms, ctx, match_index, nb_candidates, unsort_limit);
        match_index = next_candidate_idx;
        nb_candidates += 1;
    }

    // Find the longest match.
    let run = TreeRun::new::<EXT>(ctx, ip, ctx.iend);
    let mut match_end_idx = curr + 8 + 1;
    let mut best_length = 0usize;

    match_index = ms.hash_table[h];
    ms.hash_table[h] = curr;
    let mut links = TreeLinks::new(2 * (curr & bt_mask) as usize);
    let bt = ms.chain_table.as_mut_slice();

    while nb_compares > 0 && match_index > window_low {
        let next = 2 * (match_index & bt_mask) as usize;
        let (match_length, lower) = run.step::<S, EXT>(simd, ctx, match_index, links.common());
        if match_length > best_length {
            if match_length as u32 > match_end_idx - match_index {
                match_end_idx = match_index + match_length as u32;
            }
            let gain = 4 * (match_length - best_length) as i32;
            let cost = (highbit32(curr - match_index + 1) as i32)
                .wrapping_sub(highbit32(*off_base) as i32);
            if gain > cost {
                best_length = match_length;
                *off_base = offset_to_offbase(curr - match_index);
            }
            if match_length == run.len() {
                // dictMatchState also skips the dictionary's tree.
                nb_compares = 0;
                break;
            }
        }
        match links.descend(bt, match_index, next, match_length, lower, bt_low) {
            Some(child) => match_index = child,
            None => break,
        }
        nb_compares -= 1;
    }
    links.close(bt);
    ms.next_to_update = match_end_idx - 8;
    (best_length, nb_compares)
}

/// `ZSTD_BtFindBestMatch`.
#[fearless_simd_macros::simd]
fn bt_find_best_match<S: Simd, const MLS: u32, const EXT: bool>(
    simd: S,
    ms: &mut MatchState,
    ctx: &Ctx<'_>,
    ip: u32,
    off_base: &mut u32,
) -> usize {
    if ip < ms.next_to_update {
        return 0;
    }
    update_dubt::<MLS>(ms, ctx, ip);
    dubt_find_best_match::<S, MLS, EXT>(simd, ms, ctx, ip, off_base).0
}

/// `ZSTD_BtFindBestMatch` in dictMatchState mode: the context's tree, then
/// the dictionary's with the comparisons left
/// (`ZSTD_DUBT_findBetterDictMatch`).
#[fearless_simd_macros::simd]
fn bt_find_best_match_dms<S: Simd, const MLS: u32>(
    simd: S,
    ms: &mut MatchState,
    ctx: &Ctx<'_>,
    dict: &DictCtx<'_>,
    ip: u32,
    off_base: &mut u32,
) -> usize {
    if ip < ms.next_to_update {
        return 0;
    }
    update_dubt::<MLS>(ms, ctx, ip);
    let (best_length, nb_compares) =
        dubt_find_best_match::<S, MLS, false>(simd, ms, ctx, ip, off_base);
    if nb_compares == 0 {
        return best_length;
    }
    dict.better_tree_match::<S, MLS>(simd, ctx, ip, off_base, best_length, nb_compares)
}

// ---------------------------------------------------------------------------
// Row-based match finder.

/// The row finder's tables cut into rows of `ROW` slots: one tag byte and
/// one position per slot, with slot 0 of a tag row holding its head.
///
/// A row number is checked once against the table; every slot index inside
/// a row is masked below `ROW`, so it needs no check of its own.
struct Rows<'t, const ROW: usize> {
    tags: &'t mut [[u8; ROW]],
    slots: &'t mut [[u32; ROW]],
}

impl<'t, const ROW: usize> Rows<'t, ROW> {
    /// Both tables cut to their common number of rows, so one check of a
    /// row number covers both.
    #[inline(always)]
    fn new(tag_table: &'t mut [u8], hash_table: &'t mut [u32]) -> Self {
        let tags = tag_table.as_chunks_mut::<ROW>().0;
        let slots = hash_table.as_chunks_mut::<ROW>().0;
        let rows = tags.len().min(slots.len());
        Self {
            tags: &mut tags[..rows],
            slots: &mut slots[..rows],
        }
    }

    /// [`row_prefetch`] of the rows of `hash`.
    #[inline(always)]
    fn prefetch(&self, hash: u32) {
        row_prefetch(
            self.tags.as_flattened(),
            self.slots.as_flattened(),
            hash,
            ROW.trailing_zeros(),
        );
    }

    /// Inserts position `index`, whose salted hash is `hash`
    /// (`ZSTD_row_update_internalImpl`'s loop body).
    #[inline(always)]
    fn insert(&mut self, hash: u32, index: u32) {
        let row = (hash >> ROW_HASH_TAG_BITS) as usize;
        let tag_row = &mut self.tags[row];
        let pos = row_next_index(tag_row);
        tag_row[pos] = hash as u8;
        self.slots[row][pos] = index;
    }
}

/// `PREFETCH_L1`: hints that the bytes at `at` will be read soon. The
/// pointer is only handed to the prefetch instruction, never dereferenced, so
/// a hint past a table is harmless; targets without one ignore it.
#[inline(always)]
fn prefetch_l1(at: *const u8) {
    branches::prefetch_read_data::<u8, 0>(at);
}

/// `ZSTD_row_prefetch`: the hash and tag rows of `hash`, the first 64 bytes
/// of each, plus the hash row's next 64 bytes from 32-slot rows and the tag
/// row's next 32 bytes for 64-slot rows.
///
/// Each row's address is formed once, and the second line of a row is a
/// constant step from it.
#[inline(always)]
fn row_prefetch(tag_table: &[u8], hash_table: &[u32], hash: u32, row_log: u32) {
    let rel_row = ((hash >> ROW_HASH_TAG_BITS) << row_log) as usize;
    let slots = hash_table.as_ptr().wrapping_add(rel_row).cast::<u8>();
    prefetch_l1(slots);
    if row_log >= 5 {
        prefetch_l1(slots.wrapping_add(64));
    }
    let tags = tag_table.as_ptr().wrapping_add(rel_row);
    prefetch_l1(tags);
    if row_log == 6 {
        prefetch_l1(tags.wrapping_add(32));
    }
}

/// `ZSTD_row_nextIndex` of every head byte for rows of `ROW` slots: the
/// head steps down from the last slot to slot 1, skipping slot 0, which
/// holds it. A head left by a different row size is masked first, as the
/// reference masks it.
const fn next_index_table<const ROW: usize>() -> [u8; 256] {
    let row_mask = ROW - 1;
    let mut table = [0u8; 256];
    let mut head = 0usize;
    while head < 256 {
        let next = head.wrapping_sub(1) & row_mask;
        table[head] = if next == 0 { row_mask } else { next } as u8;
        head += 1;
    }
    table
}

/// The next head of a row, by row size.
struct NextIndex<const ROW: usize>;

impl<const ROW: usize> NextIndex<ROW> {
    const TABLE: [u8; 256] = next_index_table::<ROW>();
}

/// `ZSTD_row_nextIndex`: the next slot of a row, which also becomes its head.
///
/// One load from a 256-byte table replaces the reference's decrement, mask
/// and conditional move, whose wrap the compiler would otherwise branch on.
#[inline(always)]
fn row_next_index<const ROW: usize>(tag_row: &mut [u8; ROW]) -> usize {
    let next = NextIndex::<ROW>::TABLE[usize::from(tag_row[0])];
    tag_row[0] = next;
    usize::from(next) & (ROW - 1)
}

/// Hash of the row match finder over `bits` bits, salted as the reference's
/// is.
#[inline(always)]
fn row_hash<const MLS: u32>(ctx: &Ctx<'_>, index: u32, bits: u32) -> u32 {
    ctx.hash::<MLS>(index, bits) as u32
}

/// `ZSTD_row_fillHashCache`, prefetching the rows of every hash it caches.
fn row_fill_hash_cache<const MLS: u32>(
    cache: &mut [u32; ROW_HASH_CACHE_SIZE],
    tables: (&[u8], &[u32]),
    ctx: &Ctx<'_>,
    bits: u32,
    mut idx: u32,
    i_limit: i64,
) {
    let max_elems = if i64::from(idx) > i_limit {
        0
    } else {
        (i_limit - i64::from(idx) + 1) as u32
    };
    let lim = idx + (ROW_HASH_CACHE_SIZE as u32).min(max_elems);
    while idx < lim {
        let hash = row_hash::<MLS>(ctx, idx, bits);
        row_prefetch(tables.0, tables.1, hash, ctx.row_log);
        cache[(idx & ROW_HASH_CACHE_MASK) as usize] = hash;
        idx += 1;
    }
}

/// `ZSTD_row_nextCachedHash`.
#[inline(always)]
fn row_next_cached_hash<const MLS: u32>(
    cache: &mut [u32; ROW_HASH_CACHE_SIZE],
    ctx: &Ctx<'_>,
    bits: u32,
    idx: u32,
) -> u32 {
    let new_hash = row_hash::<MLS>(ctx, idx + ROW_HASH_CACHE_SIZE as u32, bits);
    core::mem::replace(&mut cache[(idx & ROW_HASH_CACHE_MASK) as usize], new_hash)
}

/// `ZSTD_row_update_internalImpl` with the hash cache.
#[inline(always)]
fn row_update_impl<const MLS: u32, const ROW: usize>(
    state: &mut RowState<'_>,
    ctx: &Ctx<'_>,
    start: u32,
    end: u32,
) {
    if start >= end {
        return;
    }
    // The words hashed, `ROW_HASH_CACHE_SIZE` positions ahead of each
    // inserted one, cut once: the reference's index invariants keep them in
    // the prefix, and each window of the cut is then a word without a check
    // of its own.
    let from = start.wrapping_sub(ctx.base) as usize + ROW_HASH_CACHE_SIZE;
    let words = ctx.src.get(from..from + (end - start) as usize + 7);
    let Some(words) = words else {
        return row_update_checked::<MLS, ROW>(state, ctx, start, end);
    };
    let bits = state.bits;
    let cache = &mut *state.cache;
    let mut rows = Rows::<ROW>::new(state.tags, state.slots);
    let mut idx = start;
    for word in words.windows(8) {
        let word = u64::from_le_bytes(word.try_into().unwrap_or_default());
        let new_hash = mem::hash_word::<MLS>(word, bits) as u32;
        // `ZSTD_row_nextCachedHash` prefetches the rows of the hash it
        // caches, 8 positions ahead.
        rows.prefetch(new_hash);
        let hash = core::mem::replace(&mut cache[(idx & ROW_HASH_CACHE_MASK) as usize], new_hash);
        rows.insert(hash, idx);
        idx = idx.wrapping_add(1);
    }
}

/// [`row_update_impl`] reading each word through [`Ctx::hash`], for a
/// range whose words leave the prefix.
#[cold]
#[inline(never)]
fn row_update_checked<const MLS: u32, const ROW: usize>(
    state: &mut RowState<'_>,
    ctx: &Ctx<'_>,
    start: u32,
    end: u32,
) {
    let bits = state.bits;
    let cache = &mut *state.cache;
    let mut rows = Rows::<ROW>::new(state.tags, state.slots);
    for idx in start..end {
        let hash = row_next_cached_hash::<MLS>(cache, ctx, bits, idx);
        rows.prefetch(cache[(idx & ROW_HASH_CACHE_MASK) as usize]);
        rows.insert(hash, idx);
    }
}

/// `ZSTD_row_update_internal` with the hash cache.
#[inline(always)]
fn row_update<const MLS: u32, const ROW: usize>(state: &mut RowState<'_>, ctx: &Ctx<'_>, ip: u32) {
    const SKIP_THRESHOLD: u32 = 384;
    const MAX_MATCH_START_POSITIONS_TO_UPDATE: u32 = 96;
    const MAX_MATCH_END_POSITIONS_TO_UPDATE: u32 = 32;
    let mut idx = state.next_to_update;
    let target = ip;
    if target - idx > SKIP_THRESHOLD {
        let bound = idx + MAX_MATCH_START_POSITIONS_TO_UPDATE;
        row_update_impl::<MLS, ROW>(state, ctx, idx, bound);
        idx = target - MAX_MATCH_END_POSITIONS_TO_UPDATE;
        let tables = (&*state.tags, &*state.slots);
        let i_limit = i64::from(ip) + 1;
        row_fill_hash_cache::<MLS>(state.cache, tables, ctx, state.bits, idx, i_limit);
    }
    row_update_impl::<MLS, ROW>(state, ctx, idx, target);
    state.next_to_update = target;
}

/// The row finder's view of the match state for one block: its tables, and
/// the constants each search would otherwise derive from the parameters.
struct RowState<'t> {
    tags: &'t mut [u8],
    slots: &'t mut [u32],
    cache: &'t mut [u32; ROW_HASH_CACHE_SIZE],
    /// `nextToUpdate`, kept here and stored back when the block ends.
    next_to_update: u32,
    /// The match state's `nextToUpdate`.
    ms_next_to_update: &'t mut u32,
    /// `rowHashLog + ZSTD_ROW_HASH_TAG_BITS`.
    bits: u32,
    /// The window's lowest valid index.
    lowest_valid: u32,
    /// The longest distance searched: the window size, or no limit once a
    /// dictionary is loaded (`lowLimit` is then the lowest valid index).
    max_distance: u32,
    /// Candidates collected per search (`1 << cappedSearchLog`), at most a
    /// row.
    nb_attempts: usize,
    /// `matchBuffer`, kept across searches so it is never cleared.
    candidates: [u32; 64],
}

impl<'t> RowState<'t> {
    fn new(ms: &'t mut MatchState, ctx: &Ctx<'_>) -> Self {
        let max_distance = if ms.loaded_dict_end != 0 {
            u32::MAX
        } else {
            1u32 << ms.cparams.window_log
        };
        let nb_attempts = 1usize << ms.cparams.search_log.min(ctx.row_log);
        let bits = ms.row_hash_log + ROW_HASH_TAG_BITS;
        let lowest_valid = ms.window.low_limit;
        let rows_len = 1usize << ms.cparams.hash_log;
        let MatchState {
            tag_table,
            hash_table,
            hash_cache,
            next_to_update,
            row_view,
            ..
        } = ms;
        Self {
            tags: aligned_rows(tag_table, rows_len, *row_view),
            slots: aligned_rows(hash_table, rows_len, *row_view),
            cache: hash_cache,
            next_to_update: *next_to_update,
            ms_next_to_update: next_to_update,
            bits,
            lowest_valid,
            max_distance,
            nb_attempts,
            candidates: [0; 64],
        }
    }

    /// `lowLimit` of a search at `curr` ([`search_low_limit`]).
    #[inline(always)]
    const fn low_limit(&self, curr: u32) -> u32 {
        if curr.wrapping_sub(self.lowest_valid) > self.max_distance {
            curr - self.max_distance
        } else {
            self.lowest_valid
        }
    }
}

/// The first `len` entries of `table` from `view` entries past its first
/// 64-byte boundary, as the reference places its tables
/// (`ZSTD_cwksp_reserve_aligned64`): a row of tags or positions then never
/// straddles a cache line, and its vector load never splits. A table
/// without [`ROW_TABLE_SLACK`](super::match_state::ROW_TABLE_SLACK) entries
/// to spare is used from its start.
///
/// The view's position in the allocation does not change the frames: a
/// search reads only the entries the finder wrote since the window's
/// indices were last invalidated, and every older entry, wherever it lies,
/// is below `lowLimit` and ends the search as a zero would.
fn aligned_rows<T>(table: &mut [T], len: usize, view: usize) -> &mut [T] {
    let range = aligned_range(table, len, view);
    &mut table[range]
}

impl Drop for RowState<'_> {
    fn drop(&mut self) {
        *self.ms_next_to_update = self.next_to_update;
    }
}

/// `ZSTD_RowFindBestMatch` in the context's rows: the longest match and
/// the attempts left for an attached dictionary.
#[inline(always)]
fn row_find_best_match<S: Simd, const MLS: u32, const ROW: usize>(
    simd: S,
    state: &mut RowState<'_>,
    ctx: &Ctx<'_>,
    ip: u32,
    lazy_skipping: bool,
    offset: &mut u32,
) -> (usize, usize) {
    let curr = ip;
    let low_limit = state.low_limit(curr);
    let bits = state.bits;
    if !lazy_skipping {
        row_update::<MLS, ROW>(state, ctx, ip);
    }
    let rows = Rows::<ROW>::new(state.tags, state.slots);
    let cache = &mut *state.cache;
    let hash = if !lazy_skipping {
        let hash = row_next_cached_hash::<MLS>(cache, ctx, bits, curr);
        rows.prefetch(cache[(curr & ROW_HASH_CACHE_MASK) as usize]);
        hash
    } else {
        state.next_to_update = curr;
        row_hash::<MLS>(ctx, ip, bits)
    };

    let row = (hash >> ROW_HASH_TAG_BITS) as usize;
    let tag = hash as u8;
    let tag_row = &mut rows.tags[row];
    let slot_row = &mut rows.slots[row];
    let head = u32::from(tag_row[0]) & (ROW as u32 - 1);
    let mut matches = row::match_mask(simd, tag_row, tag, head);
    // Candidates are collected before any is measured, as the reference
    // does: their loads then overlap instead of waiting on each comparison
    // (one fused pass measured 15-20% slower).
    let mut num_matches = 0usize;
    let buffer = &mut state.candidates;
    // Window index `i` of the prefix is at `window + i` (`base`); the
    // pointer only feeds prefetches.
    let window = ctx.src.as_ptr().wrapping_sub(ctx.base as usize);
    if state.nb_attempts >= ROW {
        // At most `ROW - 1` slots match (slot 0 never does), so the
        // attempts cannot run out and the loop counts only the candidates,
        // which also stay below the buffer's 64 slots.
        while let Some(offset) = matches.pop() {
            let match_pos = (head as usize + offset) & (ROW - 1);
            let match_index = slot_row[match_pos];
            if match_index < low_limit {
                break;
            }
            // `PREFETCH_L1(base + matchIndex)`: the candidates' bytes load
            // while the rest are collected.
            prefetch_l1(window.wrapping_add(match_index as usize));
            buffer[num_matches & 63] = match_index;
            num_matches += 1;
        }
    } else {
        // Fewer attempts than a row has slots, all within the buffer.
        let nb_attempts = state.nb_attempts;
        while num_matches < nb_attempts
            && let Some(offset) = matches.pop()
        {
            let match_pos = (head as usize + offset) & (ROW - 1);
            let match_index = slot_row[match_pos];
            if match_index < low_limit {
                break;
            }
            prefetch_l1(window.wrapping_add(match_index as usize));
            buffer[num_matches & 63] = match_index;
            num_matches += 1;
        }
    }

    // Insert the current position too.
    {
        let pos = row_next_index(tag_row);
        tag_row[pos] = tag;
        slot_row[pos] = state.next_to_update;
        state.next_to_update += 1;
    }

    let candidates = &state.candidates[..num_matches.min(64)];
    let ml = if ctx.ext {
        best_candidate::<S, true>(simd, ctx, ip, candidates, offset)
    } else {
        best_prefix_candidate(simd, ctx, ip, candidates, offset)
    };
    (ml, state.nb_attempts.saturating_sub(num_matches))
}

/// [`best_candidate`] without an external dictionary: every candidate is a
/// prefix index below `ip`.
///
/// The best length's probe word is compared against `hay`, the prefix
/// shifted by that length, so a candidate costs one load and one check; a
/// candidate that shares it is measured with its first word inline.
#[inline(always)]
fn best_prefix_candidate<S: Simd>(
    simd: S,
    ctx: &Ctx<'_>,
    ip: u32,
    candidates: &[u32],
    offset: &mut u32,
) -> usize {
    let src = ctx.src;
    let base = ctx.base;
    // The block may end before the prefix does (long distance matching
    // runs the finder on parts of a block), and no match runs past it.
    let current = src
        .get((ip - base) as usize..(ctx.iend - base) as usize)
        .unwrap_or_default();
    // The searched position is at least 16 bytes before the block's end
    // (`ilimit`); a shorter tail takes the general loop.
    let Some(first) = current.first_chunk::<8>() else {
        return best_candidate::<S, false>(simd, ctx, ip, candidates, offset);
    };
    let first = u64::from_le_bytes(*first);
    let mut ml: usize = 4 - 1;
    // `current[ml - 3..ml + 1]`, and `hay[i]` is `src[i + ml - 3]`.
    let mut word = first as u32;
    let mut hay = src;
    for &match_index in candidates {
        let at = match_index.wrapping_sub(base) as usize;
        match hay.get(at..at + 4) {
            Some(bytes) if u32::from_le_bytes(bytes.try_into().unwrap_or_default()) == word => {}
            _ => continue,
        }
        let current_ml = match src.get(at..at + 8) {
            Some(y) => match first ^ u64::from_le_bytes(y.try_into().unwrap_or_default()) {
                0 => count::common_run(simd, current, &src[at..]),
                difference => (difference.trailing_zeros() >> 3) as usize,
            },
            None => count::common_run(simd, current, &src[at..]),
        };
        if current_ml > ml {
            ml = current_ml;
            *offset = offset_to_offbase(ip - match_index);
            // A match below the tail's length leaves the bytes of the next
            // probe word in it.
            let (Some(next), Some(next_hay)) = (current.get(ml - 3..ml + 1), src.get(ml - 3..))
            else {
                break;
            };
            word = u32::from_le_bytes(next.try_into().unwrap_or_default());
            hay = next_hay;
        }
    }
    ml
}

// ---------------------------------------------------------------------------
// Attached dictionary (dictMatchState).

/// An attached dictionary as the dictMatchState searches read it
/// (`ms->dictMatchState`): its tables and content, in its own index space.
/// A dictionary index `d` stands for the context's index `d + index_delta`.
#[derive(Copy, Clone)]
struct DictCtx<'a> {
    /// The dictionary's tables and parameters.
    ms: &'a MatchState,
    /// The dictionary's content: index `i` is `src[i - start]`.
    src: &'a [u8],
    /// Index of the content's first byte (`dictLowestIndex`, the
    /// dictionary's `dictLimit`).
    start: u32,
    /// Index one past the content (`dictHighLimit`, `dmsSize`).
    end: u32,
    /// The dictionary's `lowLimit`, the binary tree's lower bound.
    low_limit: u32,
    /// `dictIndexDelta`: the context's prefix start minus [`Self::end`].
    index_delta: u32,
    /// The dictionary's tag rows from their first 64-byte boundary, as its
    /// load wrote them; empty unless its tables are rows.
    tags: &'a [u8],
    /// The dictionary's position rows, viewed as [`Self::tags`] is.
    slots: &'a [u32],
}

impl<'a> DictCtx<'a> {
    /// The view of `dms` for a context whose prefix starts at
    /// `prefix_lowest`.
    fn new(dms: &DictMatchState<'a>, prefix_lowest: u32) -> Self {
        let ms = dms.ms;
        let content = dms.window.prefix;
        let (start, end) = (content.start(), content.end());
        let rows_len = 1usize << ms.cparams.hash_log;
        Self {
            ms,
            src: content.slice(start, end),
            start,
            end,
            low_limit: ms.window.low_limit,
            index_delta: prefix_lowest.wrapping_sub(end),
            tags: ms
                .tag_table
                .get(aligned_range(&ms.tag_table, rows_len, ms.row_view))
                .unwrap_or_default(),
            slots: ms
                .hash_table
                .get(aligned_range(&ms.hash_table, rows_len, ms.row_view))
                .unwrap_or_default(),
        }
    }

    /// The content from dictionary index `d`; empty outside it.
    #[inline(always)]
    fn from(&self, d: u32) -> &'a [u8] {
        self.src
            .get(d.wrapping_sub(self.start) as usize..)
            .unwrap_or_default()
    }

    /// `ZSTD_count_2segments(ip, dictBase + d, iend, dictEnd, prefixStart)`:
    /// the common length of `current` (the bytes from a searched position to
    /// the block end) and the content from index `d`, continuing at the
    /// prefix's start, `prefix`, when the content ends. An index past the
    /// content counts nothing, as the reference's pointer arithmetic does
    /// (a tree walk's known length can carry a candidate there).
    #[inline(always)]
    fn count_2segments<S: Simd>(&self, simd: S, current: &[u8], prefix: &[u8], d: u32) -> usize {
        let Some(content) = self.src.get(d.wrapping_sub(self.start) as usize..) else {
            return 0;
        };
        let n = count::common_run(simd, current, content);
        if n < content.len() {
            return n;
        }
        n + count::common_run(simd, current.get(n..).unwrap_or_default(), prefix)
    }

    /// Length of the match of `current`, whose first four bytes are `word`,
    /// with dictionary index `d` when their first four bytes agree
    /// (`MEM_read32(match) == MEM_read32(ip)`), `0` otherwise.
    #[inline(always)]
    fn candidate_length<S: Simd>(
        &self,
        simd: S,
        ctx: &Ctx<'_>,
        current: &[u8],
        word: [u8; 4],
        d: u32,
    ) -> usize {
        if self.from(d).first_chunk::<4>() != Some(&word) {
            return 0;
        }
        let rest = current.get(4..).unwrap_or_default();
        self.count_2segments(simd, rest, ctx.src, d + 4) + 4
    }

    /// The repeat offset `offset` at prefix index `curr`, in the prefix or
    /// in the dictionary (the parser's `isDxS` repcode checks): its length
    /// when its first four bytes match, `None` otherwise.
    #[inline(always)]
    fn repeat<S: Simd>(&self, simd: S, ctx: &Ctx<'_>, curr: u32, offset: u32) -> Option<usize> {
        let prefix_lowest = ctx.dict_limit();
        let rep_index = curr.wrapping_sub(offset);
        // `ZSTD_index_overlap_check`: no four bytes straddle the boundary.
        if prefix_lowest.wrapping_sub(1).wrapping_sub(rep_index) < 3 {
            return None;
        }
        if rep_index < prefix_lowest {
            let current = ctx.tail(curr);
            let word = *current.first_chunk::<4>()?;
            let d = rep_index.wrapping_sub(self.index_delta);
            match self.candidate_length(simd, ctx, current, word, d) {
                0 => None,
                length => Some(length),
            }
        } else if ctx.repeats(curr, offset) {
            Some(ctx.count(simd, curr + 4, rep_index + 4) + 4)
        } else {
            None
        }
    }

    /// The dictionary part of `ZSTD_HcFindBestMatch`: walks the
    /// dictionary's chain for `ip` with the `nb_attempts` left, improving
    /// on the context's best length `ml`.
    #[inline(always)]
    fn chain_match<S: Simd, const MLS: u32>(
        &self,
        simd: S,
        ctx: &Ctx<'_>,
        ip: u32,
        mut ml: usize,
        mut nb_attempts: u32,
        offset: &mut u32,
    ) -> usize {
        let dms = self.ms;
        let chain_size = 1u32 << dms.cparams.chain_log;
        let chain_mask = chain_size - 1;
        let min_chain = self.end.saturating_sub(chain_size);
        let chain = dms.chain_table.as_slice();
        let h = ctx.hash::<MLS>(ip, dms.cparams.hash_log);
        let mut match_index = dms.hash_table.get(h).copied().unwrap_or_default();
        let current = ctx.tail(ip);
        let Some(&word) = current.first_chunk::<4>() else {
            return ml;
        };
        while match_index >= self.start && nb_attempts > 0 {
            let current_ml = self.candidate_length(simd, ctx, current, word, match_index);
            if current_ml > ml {
                ml = current_ml;
                *offset =
                    offset_to_offbase(ip.wrapping_sub(match_index.wrapping_add(self.index_delta)));
                if current_ml == current.len() {
                    break;
                }
            }
            if match_index <= min_chain {
                break;
            }
            match_index = chain
                .get((match_index & chain_mask) as usize)
                .copied()
                .unwrap_or_default();
            nb_attempts -= 1;
        }
        ml
    }

    /// `ZSTD_DUBT_findBetterDictMatch`: walks the dictionary's sorted tree
    /// for `ip` with the `nb_compares` left, improving on the context's
    /// `best_length`. The tree is only read.
    #[inline(always)]
    fn better_tree_match<S: Simd, const MLS: u32>(
        &self,
        simd: S,
        ctx: &Ctx<'_>,
        ip: u32,
        off_base: &mut u32,
        mut best_length: usize,
        mut nb_compares: u32,
    ) -> usize {
        let dms = self.ms;
        let h = ctx.hash::<MLS>(ip, dms.cparams.hash_log);
        let mut dict_match_index = dms.hash_table.get(h).copied().unwrap_or_default();
        let (high, low) = (self.end, self.low_limit);
        let bt = dms.chain_table.as_slice();
        let bt_mask = (1u32 << (dms.cparams.chain_log - 1)) - 1;
        let bt_low = if bt_mask >= high.wrapping_sub(low) {
            low
        } else {
            high - bt_mask
        };
        let current = ctx.tail(ip);
        let (mut common_smaller, mut common_larger) = (0usize, 0usize);
        while nb_compares > 0 && dict_match_index > low {
            let next = 2 * (dict_match_index & bt_mask) as usize;
            let known = common_smaller.min(common_larger);
            let rest = current.get(known..).unwrap_or_default();
            let match_length =
                known + self.count_2segments(simd, rest, ctx.src, dict_match_index + known as u32);
            if match_length > best_length {
                let match_index = dict_match_index.wrapping_add(self.index_delta);
                let gain = 4 * (match_length - best_length) as i32;
                let cost = (highbit32(ip.wrapping_sub(match_index).wrapping_add(1)) as i32)
                    .wrapping_sub(highbit32(off_base.wrapping_add(1)) as i32);
                if gain > cost {
                    best_length = match_length;
                    *off_base = offset_to_offbase(ip.wrapping_sub(match_index));
                }
                if match_length == current.len() {
                    break;
                }
            }
            if dict_match_index <= bt_low {
                break;
            }
            // `match[matchLength]`: past the content, the match continues
            // at the prefix's start.
            let match_end = dict_match_index + match_length as u32;
            let match_byte = if match_end >= high {
                ctx.src.get((match_end - high) as usize)
            } else {
                self.from(match_end).first()
            };
            let lower = matches!(
                (match_byte, current.get(match_length)),
                (Some(m), Some(c)) if m < c
            );
            if lower {
                common_smaller = match_length;
                dict_match_index = bt.get(next + 1).copied().unwrap_or_default();
            } else {
                common_larger = match_length;
                dict_match_index = bt.get(next).copied().unwrap_or_default();
            }
            nb_compares -= 1;
        }
        best_length
    }
}

/// `ZSTD_HcFindBestMatch` in dictMatchState mode.
#[inline(always)]
fn hc_find_best_match_dms<S: Simd, const MLS: u32>(
    simd: S,
    ms: &mut MatchState,
    ctx: &Ctx<'_>,
    dict: &DictCtx<'_>,
    ip: u32,
    lazy_skipping: bool,
    offset: &mut u32,
) -> usize {
    let (ml, nb_attempts) = hc_find_best_match::<S, MLS>(simd, ms, ctx, ip, lazy_skipping, offset);
    dict.chain_match::<S, MLS>(simd, ctx, ip, ml, nb_attempts, offset)
}

/// `ZSTD_RowFindBestMatch` in dictMatchState mode: the dictionary's rows
/// are prefetched before the context's are updated, and searched after
/// with the attempts left, as the reference does.
#[inline(always)]
fn row_find_best_match_dms<S: Simd, const MLS: u32, const ROW: usize>(
    simd: S,
    state: &mut RowState<'_>,
    ctx: &Ctx<'_>,
    dict: &DictCtx<'_>,
    ip: u32,
    lazy_skipping: bool,
    offset: &mut u32,
) -> usize {
    // The dictionary's rows are hashed unsalted, over its own row bits and
    // with the context's row size.
    let hash = row_hash::<MLS>(ctx, ip, dict.ms.row_hash_log + ROW_HASH_TAG_BITS);
    row_prefetch(dict.tags, dict.slots, hash, ROW.trailing_zeros());
    let (mut ml, mut nb_attempts) =
        row_find_best_match::<S, MLS, ROW>(simd, state, ctx, ip, lazy_skipping, offset);

    let row = (hash >> ROW_HASH_TAG_BITS) as usize;
    let tags = dict.tags.as_chunks::<ROW>().0;
    let slots = dict.slots.as_chunks::<ROW>().0;
    let (Some(tag_row), Some(slot_row)) = (tags.get(row), slots.get(row)) else {
        return ml;
    };
    let head = u32::from(tag_row[0]) & (ROW as u32 - 1);
    let mut matches = row::match_mask(simd, tag_row, hash as u8, head);
    let buffer = &mut state.candidates;
    // Dictionary index `i` is at `window + i`; the pointer only feeds
    // prefetches.
    let window = dict.src.as_ptr().wrapping_sub(dict.start as usize);
    let mut num_matches = 0usize;
    while nb_attempts > 0
        && let Some(offset) = matches.pop()
    {
        let match_pos = (head as usize + offset) & (ROW - 1);
        let match_index = slot_row[match_pos];
        if match_index < dict.start {
            break;
        }
        prefetch_l1(window.wrapping_add(match_index as usize));
        buffer[num_matches & 63] = match_index;
        num_matches += 1;
        nb_attempts -= 1;
    }

    let current = ctx.tail(ip);
    let Some(&word) = current.first_chunk::<4>() else {
        return ml;
    };
    for &match_index in &buffer[..num_matches.min(64)] {
        let current_ml = dict.candidate_length(simd, ctx, current, word, match_index);
        if current_ml > ml {
            ml = current_ml;
            *offset =
                offset_to_offbase(ip.wrapping_sub(match_index.wrapping_add(dict.index_delta)));
            if current_ml == current.len() {
                break;
            }
        }
    }
    ml
}

// ---------------------------------------------------------------------------
// Parsers.

/// `(int)(a*m - highbit32(offBase) + k)` as the reference computes it.
#[inline(always)]
fn gain(length: usize, multiplier: i64, off_base: u32, bonus: i64) -> i32 {
    (length as i64 * multiplier - i64::from(highbit32(off_base)) + bonus) as i32
}

/// `ZSTD_compressBlock_lazy_generic` for the no-dictionary window.
#[inline(always)]
fn lazy_generic<S: Simd, M: Search, const DEPTH: u32>(
    simd: S,
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
) -> usize {
    let prefix = block.window.prefix;
    let base = prefix.start();
    let src = prefix.slice(base, block.end);
    let istart = (block.start - base) as usize;
    let iend = src.len();
    let margin = if M::ROW { 8 + ROW_HASH_CACHE_SIZE } else { 8 };
    let ilimit = iend.saturating_sub(margin);
    let cache_limit = i64::from(block.end) - margin as i64;
    let prefix_lowest = (ms.window.dict_limit - base) as usize;
    let ctx = Ctx::new(
        block,
        ms.cparams.min_match.clamp(4, 6),
        ms.cparams.search_log.clamp(4, 6),
        false,
    );
    let count = |a: usize, b: usize| mem::count_at_simd(simd, src, a, b);
    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];
    let mut offset_saved1 = 0u32;
    let mut offset_saved2 = 0u32;

    let mut ip = istart;
    let mut anchor = istart;
    ip += usize::from(ip == prefix_lowest);
    {
        let curr = base + ip as u32;
        let window_low = ms.lowest_prefix_index(curr, ms.cparams.window_log);
        let max_rep = curr - window_low;
        if offset_2 > max_rep {
            offset_saved2 = offset_2;
            offset_2 = 0;
        }
        if offset_1 > max_rep {
            offset_saved1 = offset_1;
            offset_1 = 0;
        }
    }
    let mut lazy_skipping = false;
    let mut state = M::state(ms, &ctx);
    M::fill_cache(&mut state, &ctx, cache_limit);

    while ip < ilimit {
        let mut match_length = 0usize;
        let mut off_base = repcode_to_offbase(1);
        let mut start = ip + 1;
        let mut store_now = false;

        if offset_1 > 0 && ctx.repeats(base + (ip + 1) as u32, offset_1) {
            match_length = count(ip + 1 + 4, ip + 1 + 4 - offset_1 as usize) + 4;
            if DEPTH == 0 {
                store_now = true;
            }
        }

        if !store_now {
            let mut found = NO_OFFBASE;
            let ml2 = M::find(
                simd,
                &mut state,
                &ctx,
                base + ip as u32,
                lazy_skipping,
                &mut found,
            );
            if ml2 > match_length {
                match_length = ml2;
                start = ip;
                off_base = found;
            }

            if match_length < 4 {
                let step = ((ip - anchor) >> SEARCH_STRENGTH) + 1;
                ip += step;
                lazy_skipping = step > LAZY_SKIPPING_STEP;
                continue;
            }

            if DEPTH >= 1 {
                while ip < ilimit {
                    ip += 1;
                    if off_base != 0 && offset_1 > 0 && ctx.repeats(base + ip as u32, offset_1) {
                        let ml_rep = count(ip + 4, ip + 4 - offset_1 as usize) + 4;
                        let gain2 = (ml_rep * 3) as i32;
                        let gain1 = gain(match_length, 3, off_base, 1);
                        if ml_rep >= 4 && gain2 > gain1 {
                            match_length = ml_rep;
                            off_base = repcode_to_offbase(1);
                            start = ip;
                        }
                    }
                    {
                        let mut candidate = NO_OFFBASE;
                        let ml2 = M::find(
                            simd,
                            &mut state,
                            &ctx,
                            base + ip as u32,
                            lazy_skipping,
                            &mut candidate,
                        );
                        let gain2 = gain(ml2, 4, candidate, 0);
                        let gain1 = gain(match_length, 4, off_base, 4);
                        if ml2 >= 4 && gain2 > gain1 {
                            match_length = ml2;
                            off_base = candidate;
                            start = ip;
                            continue;
                        }
                    }
                    if DEPTH == 2 && ip < ilimit {
                        ip += 1;
                        if off_base != 0 && offset_1 > 0 && ctx.repeats(base + ip as u32, offset_1)
                        {
                            let ml_rep = count(ip + 4, ip + 4 - offset_1 as usize) + 4;
                            let gain2 = (ml_rep * 4) as i32;
                            let gain1 = gain(match_length, 4, off_base, 1);
                            if ml_rep >= 4 && gain2 > gain1 {
                                match_length = ml_rep;
                                off_base = repcode_to_offbase(1);
                                start = ip;
                            }
                        }
                        let mut candidate = NO_OFFBASE;
                        let ml2 = M::find(
                            simd,
                            &mut state,
                            &ctx,
                            base + ip as u32,
                            lazy_skipping,
                            &mut candidate,
                        );
                        let gain2 = gain(ml2, 4, candidate, 0);
                        let gain1 = gain(match_length, 4, off_base, 7);
                        if ml2 >= 4 && gain2 > gain1 {
                            match_length = ml2;
                            off_base = candidate;
                            start = ip;
                            continue;
                        }
                    }
                    break;
                }
            }

            // Catch up.
            if off_base > 3 {
                let offset = offbase_to_offset(off_base) as usize;
                while start > anchor
                    && start - offset > prefix_lowest
                    && src[start - 1] == src[start - offset - 1]
                {
                    start -= 1;
                    match_length += 1;
                }
                offset_2 = offset_1;
                offset_1 = offbase_to_offset(off_base);
            }
        }

        // _storeSequence:
        seq_store.store_seq(&src[anchor..start], off_base, match_length);
        ip = start + match_length;
        anchor = ip;
        if lazy_skipping {
            M::fill_cache(&mut state, &ctx, cache_limit);
            lazy_skipping = false;
        }

        while ip <= ilimit && offset_2 > 0 && ctx.repeats(base + ip as u32, offset_2) {
            let ml = count(ip + 4, ip + 4 - offset_2 as usize) + 4;
            core::mem::swap(&mut offset_1, &mut offset_2);
            seq_store.store_seq_only(0, repcode_to_offbase(1), ml);
            ip += ml;
            anchor = ip;
        }
    }

    drop(state);
    ms.lazy_skipping = lazy_skipping;
    offset_saved2 = if offset_saved1 != 0 && offset_1 != 0 {
        offset_saved1
    } else {
        offset_saved2
    };
    rep[0] = if offset_1 != 0 {
        offset_1
    } else {
        offset_saved1
    };
    rep[1] = if offset_2 != 0 {
        offset_2
    } else {
        offset_saved2
    };
    iend - anchor
}

/// `ZSTD_compressBlock_lazy_extDict_generic`.
#[inline(always)]
fn lazy_ext_generic<S: Simd, M: Search, const DEPTH: u32>(
    simd: S,
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
) -> usize {
    let window = block.window;
    let prefix = window.prefix;
    let istart = i64::from(block.start);
    let iend = i64::from(block.end);
    let ilimit = if M::ROW {
        iend - 8 - ROW_HASH_CACHE_SIZE as i64
    } else {
        iend - 8
    };
    let dict_limit = ms.window.dict_limit;
    let prefix_start = i64::from(dict_limit);
    let dict_start = ms.window.low_limit;
    let window_log = ms.cparams.window_log;
    let ctx = Ctx::new(
        block,
        ms.cparams.min_match.clamp(4, 6),
        ms.cparams.search_log.clamp(4, 6),
        true,
    );
    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];
    // A repeat-offset candidate at `rep_index` for position `curr`.
    let (lowest_valid, dict_loaded) = (ms.window.low_limit, ms.loaded_dict_end != 0);
    // `ZSTD_getLowestMatchIndex`.
    let lowest_match_index = move |curr: u32| {
        let max_distance = 1u32 << window_log;
        if dict_loaded || curr.wrapping_sub(lowest_valid) <= max_distance {
            lowest_valid
        } else {
            curr - max_distance
        }
    };
    let rep_ok = |curr: u32, offset: u32| {
        let window_low = lowest_match_index(curr);
        let rep_index = curr.wrapping_sub(offset);
        let overlap_ok = dict_limit.wrapping_sub(1).wrapping_sub(rep_index) >= 3;
        if overlap_ok
            && offset <= curr - window_low
            && window.read32_at(rep_index) == prefix.read32(curr)
        {
            Some(ctx.count_virtual(simd, curr + 4, rep_index + 4, block.end) + 4)
        } else {
            None
        }
    };

    let mut lazy_skipping = false;
    let mut ip = istart;
    let mut anchor = istart;
    ip += i64::from(ip == prefix_start);
    let mut state = M::state(ms, &ctx);
    M::fill_cache(&mut state, &ctx, ilimit);

    while ip < ilimit {
        let mut match_length = 0usize;
        let mut off_base = repcode_to_offbase(1);
        let mut start = ip + 1;
        let mut curr = ip as u32;
        let mut store_now = false;

        if let Some(length) = rep_ok(curr + 1, offset_1) {
            match_length = length;
            if DEPTH == 0 {
                store_now = true;
            }
        }

        if !store_now {
            let mut candidate = NO_OFFBASE;
            let ml2 = M::find(
                simd,
                &mut state,
                &ctx,
                ip as u32,
                lazy_skipping,
                &mut candidate,
            );
            if ml2 > match_length {
                match_length = ml2;
                start = ip;
                off_base = candidate;
            }
            if match_length < 4 {
                let step = ((ip - anchor) >> SEARCH_STRENGTH) as usize;
                ip += step as i64 + 1;
                lazy_skipping = step > LAZY_SKIPPING_STEP;
                continue;
            }
            if DEPTH >= 1 {
                while ip < ilimit {
                    ip += 1;
                    curr += 1;
                    if off_base != 0
                        && let Some(rep_length) = rep_ok(curr, offset_1)
                    {
                        let gain2 = (rep_length * 3) as i32;
                        let gain1 = gain(match_length, 3, off_base, 1);
                        if rep_length >= 4 && gain2 > gain1 {
                            match_length = rep_length;
                            off_base = repcode_to_offbase(1);
                            start = ip;
                        }
                    }
                    {
                        let mut candidate = NO_OFFBASE;
                        let ml2 = M::find(
                            simd,
                            &mut state,
                            &ctx,
                            ip as u32,
                            lazy_skipping,
                            &mut candidate,
                        );
                        let gain2 = gain(ml2, 4, candidate, 0);
                        let gain1 = gain(match_length, 4, off_base, 4);
                        if ml2 >= 4 && gain2 > gain1 {
                            match_length = ml2;
                            off_base = candidate;
                            start = ip;
                            continue;
                        }
                    }
                    if DEPTH == 2 && ip < ilimit {
                        ip += 1;
                        curr += 1;
                        if off_base != 0
                            && let Some(rep_length) = rep_ok(curr, offset_1)
                        {
                            let gain2 = (rep_length * 4) as i32;
                            let gain1 = gain(match_length, 4, off_base, 1);
                            if rep_length >= 4 && gain2 > gain1 {
                                match_length = rep_length;
                                off_base = repcode_to_offbase(1);
                                start = ip;
                            }
                        }
                        let mut candidate = NO_OFFBASE;
                        let ml2 = M::find(
                            simd,
                            &mut state,
                            &ctx,
                            ip as u32,
                            lazy_skipping,
                            &mut candidate,
                        );
                        let gain2 = gain(ml2, 4, candidate, 0);
                        let gain1 = gain(match_length, 4, off_base, 7);
                        if ml2 >= 4 && gain2 > gain1 {
                            match_length = ml2;
                            off_base = candidate;
                            start = ip;
                            continue;
                        }
                    }
                    break;
                }
            }

            if off_base > 3 {
                let match_index = (start as u32).wrapping_sub(offbase_to_offset(off_base));
                let mut m = match_index;
                let m_start = if match_index < dict_limit {
                    dict_start
                } else {
                    dict_limit
                };
                while start > anchor
                    && m > m_start
                    && prefix.byte((start - 1) as u32) == window.byte_at(m - 1)
                {
                    start -= 1;
                    m -= 1;
                    match_length += 1;
                }
                offset_2 = offset_1;
                offset_1 = offbase_to_offset(off_base);
            }
        }

        seq_store.store_seq(
            prefix.slice(anchor as u32, start as u32),
            off_base,
            match_length,
        );
        ip = start + match_length as i64;
        anchor = ip;
        if lazy_skipping {
            M::fill_cache(&mut state, &ctx, ilimit);
            lazy_skipping = false;
        }

        while ip <= ilimit {
            let Some(ml) = rep_ok(ip as u32, offset_2) else {
                break;
            };
            core::mem::swap(&mut offset_1, &mut offset_2);
            seq_store.store_seq_only(0, repcode_to_offbase(1), ml);
            ip += ml as i64;
            anchor = ip;
        }
    }
    drop(state);
    ms.lazy_skipping = lazy_skipping;
    rep[0] = offset_1;
    rep[1] = offset_2;
    (iend - anchor) as usize
}

/// `ZSTD_compressBlock_lazy_generic` for a prefix with an attached
/// dictionary (`dictMode == ZSTD_dictMatchState`): repeat offsets and
/// matches may reach into the dictionary, whose indices sit just below the
/// prefix's (`dictIndexDelta`), and every search continues in the
/// dictionary's tables with what the context's left of its budget.
#[inline(always)]
fn lazy_dms_generic<S: Simd, M: Search, const DEPTH: u32>(
    simd: S,
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
    dms: &DictMatchState<'_>,
) -> usize {
    let prefix = block.window.prefix;
    let istart = i64::from(block.start);
    let iend = i64::from(block.end);
    let ilimit = if M::ROW {
        iend - 8 - ROW_HASH_CACHE_SIZE as i64
    } else {
        iend - 8
    };
    let prefix_lowest_index = ms.window.dict_limit;
    let prefix_lowest = i64::from(prefix_lowest_index);
    let ctx = Ctx::new(
        block,
        ms.cparams.min_match.clamp(4, 6),
        ms.cparams.search_log.clamp(4, 6),
        false,
    );
    let dict = DictCtx::new(dms, prefix_lowest_index);
    let repeat = |curr: i64, offset: u32| dict.repeat(simd, &ctx, curr as u32, offset);
    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];

    let mut ip = istart;
    let mut anchor = istart;
    let dict_and_prefix_length = (ip - prefix_lowest) + i64::from(dict.end - dict.start);
    ip += i64::from(dict_and_prefix_length == 0);
    let mut lazy_skipping = false;
    let mut state = M::state(ms, &ctx);
    M::fill_cache(&mut state, &ctx, ilimit);

    while ip < ilimit {
        let mut match_length = 0usize;
        let mut off_base = repcode_to_offbase(1);
        let mut start = ip + 1;
        let mut store_now = false;

        if let Some(length) = repeat(ip + 1, offset_1) {
            match_length = length;
            if DEPTH == 0 {
                store_now = true;
            }
        }

        if !store_now {
            let mut found = NO_OFFBASE;
            let ml2 = M::find_dms(
                simd,
                &mut state,
                &ctx,
                &dict,
                ip as u32,
                lazy_skipping,
                &mut found,
            );
            if ml2 > match_length {
                match_length = ml2;
                start = ip;
                off_base = found;
            }

            if match_length < 4 {
                let step = ((ip - anchor) >> SEARCH_STRENGTH) as usize + 1;
                ip += step as i64;
                lazy_skipping = step > LAZY_SKIPPING_STEP;
                continue;
            }

            if DEPTH >= 1 {
                while ip < ilimit {
                    ip += 1;
                    if let Some(ml_rep) = repeat(ip, offset_1) {
                        let gain2 = (ml_rep * 3) as i32;
                        let gain1 = gain(match_length, 3, off_base, 1);
                        if ml_rep >= 4 && gain2 > gain1 {
                            match_length = ml_rep;
                            off_base = repcode_to_offbase(1);
                            start = ip;
                        }
                    }
                    {
                        let mut candidate = NO_OFFBASE;
                        let ml2 = M::find_dms(
                            simd,
                            &mut state,
                            &ctx,
                            &dict,
                            ip as u32,
                            lazy_skipping,
                            &mut candidate,
                        );
                        let gain2 = gain(ml2, 4, candidate, 0);
                        let gain1 = gain(match_length, 4, off_base, 4);
                        if ml2 >= 4 && gain2 > gain1 {
                            match_length = ml2;
                            off_base = candidate;
                            start = ip;
                            continue;
                        }
                    }
                    if DEPTH == 2 && ip < ilimit {
                        ip += 1;
                        if let Some(ml_rep) = repeat(ip, offset_1) {
                            let gain2 = (ml_rep * 4) as i32;
                            let gain1 = gain(match_length, 4, off_base, 1);
                            if ml_rep >= 4 && gain2 > gain1 {
                                match_length = ml_rep;
                                off_base = repcode_to_offbase(1);
                                start = ip;
                            }
                        }
                        let mut candidate = NO_OFFBASE;
                        let ml2 = M::find_dms(
                            simd,
                            &mut state,
                            &ctx,
                            &dict,
                            ip as u32,
                            lazy_skipping,
                            &mut candidate,
                        );
                        let gain2 = gain(ml2, 4, candidate, 0);
                        let gain1 = gain(match_length, 4, off_base, 7);
                        if ml2 >= 4 && gain2 > gain1 {
                            match_length = ml2;
                            off_base = candidate;
                            start = ip;
                            continue;
                        }
                    }
                    break;
                }
            }

            // Catch up, in the dictionary for a match that starts there.
            if off_base > 3 {
                let offset = offbase_to_offset(off_base);
                let match_index = (start as u32).wrapping_sub(offset);
                if match_index < prefix_lowest_index {
                    let mut m = match_index.wrapping_sub(dict.index_delta);
                    while start > anchor
                        && m > dict.start
                        && dict.from(m - 1).first() == Some(&prefix.byte((start - 1) as u32))
                    {
                        start -= 1;
                        m -= 1;
                        match_length += 1;
                    }
                } else {
                    let mut m = match_index;
                    while start > anchor
                        && m > prefix_lowest_index
                        && prefix.byte((start - 1) as u32) == prefix.byte(m - 1)
                    {
                        start -= 1;
                        m -= 1;
                        match_length += 1;
                    }
                }
                offset_2 = offset_1;
                offset_1 = offset;
            }
        }

        // _storeSequence:
        seq_store.store_seq(
            prefix.slice(anchor as u32, start as u32),
            off_base,
            match_length,
        );
        ip = start + match_length as i64;
        anchor = ip;
        if lazy_skipping {
            M::fill_cache(&mut state, &ctx, ilimit);
            lazy_skipping = false;
        }

        while ip <= ilimit {
            let Some(ml) = repeat(ip, offset_2) else {
                break;
            };
            core::mem::swap(&mut offset_1, &mut offset_2);
            seq_store.store_seq_only(0, repcode_to_offbase(1), ml);
            ip += ml as i64;
            anchor = ip;
        }
    }

    drop(state);
    ms.lazy_skipping = lazy_skipping;
    rep[0] = offset_1;
    rep[1] = offset_2;
    (iend - anchor) as usize
}

/// Defines the block compressors of one search and depth: each dispatches
/// the match state's backend once and runs the parser specialized for it.
macro_rules! block_compressors {
    ($($name:ident, $ext_name:ident => $search:ty, $depth:expr;)*) => {$(
        pub(crate) fn $name(
            ms: &mut MatchState,
            seq_store: &mut SeqStore,
            rep: &mut [u32; 3],
            block: &BlockInput<'_>,
        ) -> usize {
            let backend = ms.backend;
            fearless_simd::dispatch!(backend.0, simd => {
                lazy_generic::<_, $search, $depth>(simd, ms, seq_store, rep, block)
            })
        }

        pub(crate) fn $ext_name(
            ms: &mut MatchState,
            seq_store: &mut SeqStore,
            rep: &mut [u32; 3],
            block: &BlockInput<'_>,
        ) -> usize {
            let backend = ms.backend;
            fearless_simd::dispatch!(backend.0, simd => {
                lazy_ext_generic::<_, $search, $depth>(simd, ms, seq_store, rep, block)
            })
        }
    )*};
}

block_compressors! {
    greedy, greedy_ext => HashChain, 0;
    lazy, lazy_ext => HashChain, 1;
    lazy2, lazy2_ext => HashChain, 2;
    btlazy2, btlazy2_ext => BinaryTree, 2;
}

/// Runs `$run!` with the [`RowHash`] specialization for the minimum match
/// and row size of `$ms`, clamped as the parsers pass them to [`Ctx::new`]
/// (the reference's `ZSTD_searchMax` switch, taken once per block).
macro_rules! row_block {
    ($ms:expr, $run:ident) => {
        match ($ms.cparams.min_match.clamp(4, 6), $ms.cparams.search_log.clamp(4, 6)) {
            (5, 4) => $run!(RowHash<5, 16>),
            (5, 5) => $run!(RowHash<5, 32>),
            (5, _) => $run!(RowHash<5, 64>),
            (6, 4) => $run!(RowHash<6, 16>),
            (6, 5) => $run!(RowHash<6, 32>),
            (6, _) => $run!(RowHash<6, 64>),
            (_, 4) => $run!(RowHash<4, 16>),
            (_, 5) => $run!(RowHash<4, 32>),
            (_, _) => $run!(RowHash<4, 64>),
        }
    };
}

/// Defines the row finder's block compressors of one depth, as
/// `block_compressors!` does, with the search specialized by `row_block!`
/// inside the backend's dispatch.
macro_rules! row_block_compressors {
    ($($name:ident, $ext_name:ident => $depth:expr;)*) => {$(
        pub(crate) fn $name(
            ms: &mut MatchState,
            seq_store: &mut SeqStore,
            rep: &mut [u32; 3],
            block: &BlockInput<'_>,
        ) -> usize {
            let backend = ms.backend;
            fearless_simd::dispatch!(backend.0, simd => {
                macro_rules! run {
                    ($search:ty) => {
                        lazy_generic::<_, $search, $depth>(simd, ms, seq_store, rep, block)
                    };
                }
                row_block!(ms, run)
            })
        }

        pub(crate) fn $ext_name(
            ms: &mut MatchState,
            seq_store: &mut SeqStore,
            rep: &mut [u32; 3],
            block: &BlockInput<'_>,
        ) -> usize {
            let backend = ms.backend;
            fearless_simd::dispatch!(backend.0, simd => {
                macro_rules! run {
                    ($search:ty) => {
                        lazy_ext_generic::<_, $search, $depth>(simd, ms, seq_store, rep, block)
                    };
                }
                row_block!(ms, run)
            })
        }
    )*};
}

row_block_compressors! {
    greedy_row, greedy_row_ext => 0;
    lazy_row, lazy_row_ext => 1;
    lazy2_row, lazy2_row_ext => 2;
}

/// Indexes dictionary content up to `target` (`ZSTD_loadDictionaryContent`
/// for the lazy strategies): the row finder clears its tags and runs
/// `ZSTD_row_update`, the hash chain `ZSTD_insertAndFindFirstIndex`.
pub(crate) fn load_dictionary(
    ms: &mut MatchState,
    window: &mem::Window<'_>,
    target: u32,
    row: bool,
) {
    let block = BlockInput {
        window: *window,
        start: target,
        end: window.prefix.end(),
        backend: ms.backend,
        dms: None,
        symbol_costs: None,
    };
    if row {
        ms.tag_table.fill(0);
        let row_log = ms.cparams.search_log.clamp(4, 6);
        let ctx = Ctx::new(&block, ms.cparams.min_match.min(6), row_log, false);
        match (ctx.mls, row_log) {
            (5, 4) => row_load::<5, 16>(ms, &ctx, target),
            (5, 5) => row_load::<5, 32>(ms, &ctx, target),
            (5, _) => row_load::<5, 64>(ms, &ctx, target),
            (6, 4) => row_load::<6, 16>(ms, &ctx, target),
            (6, 5) => row_load::<6, 32>(ms, &ctx, target),
            (6, _) => row_load::<6, 64>(ms, &ctx, target),
            (_, 4) => row_load::<4, 16>(ms, &ctx, target),
            (_, 5) => row_load::<4, 32>(ms, &ctx, target),
            (_, _) => row_load::<4, 64>(ms, &ctx, target),
        }
    } else {
        // `ZSTD_insertAndFindFirstIndex` hashes with the unclamped minimum
        // match, unlike the searches.
        let ctx = Ctx::new(&block, ms.cparams.min_match, 0, false);
        match ctx.mls {
            5 => insert_and_find_first_index::<5>(ms, &ctx, target, false),
            6 => insert_and_find_first_index::<6>(ms, &ctx, target, false),
            7 => insert_and_find_first_index::<7>(ms, &ctx, target, false),
            _ => insert_and_find_first_index::<4>(ms, &ctx, target, false),
        };
    }
}

/// `ZSTD_row_update` for a dictionary: every position up to `target`,
/// without the hash cache or the skipping a search's update does.
fn row_load<const MLS: u32, const ROW: usize>(ms: &mut MatchState, ctx: &Ctx<'_>, target: u32) {
    let mut state = RowState::new(ms, ctx);
    let bits = state.bits;
    let mut rows = Rows::<ROW>::new(state.tags, state.slots);
    for idx in state.next_to_update..target {
        rows.insert(ctx.hash::<MLS>(idx, bits) as u32, idx);
    }
    state.next_to_update = target;
}

/// Defines the dictMatchState block compressors of one search and depth
/// (`ZSTD_compressBlock_*_dictMatchState{,_row}`): each dispatches the
/// match state's backend once and runs the parser with the block's attached
/// dictionary. A block without one, which `select_block_compressor` never
/// routes here, takes the prefix-only compressor `$no_dict`.
macro_rules! dms_block_compressors {
    ($($name:ident, $no_dict:ident => $search:ty, $depth:expr;)*) => {$(
        pub(crate) fn $name(
            ms: &mut MatchState,
            seq_store: &mut SeqStore,
            rep: &mut [u32; 3],
            block: &BlockInput<'_>,
        ) -> usize {
            let Some(dms) = block.dms else {
                return $no_dict(ms, seq_store, rep, block);
            };
            let backend = ms.backend;
            fearless_simd::dispatch!(backend.0, simd => {
                lazy_dms_generic::<_, $search, $depth>(simd, ms, seq_store, rep, block, &dms)
            })
        }
    )*};
}

dms_block_compressors! {
    greedy_dms, greedy => HashChain, 0;
    lazy_dms, lazy => HashChain, 1;
    lazy2_dms, lazy2 => HashChain, 2;
    btlazy2_dms, btlazy2 => BinaryTree, 2;
    greedy_row_dms, greedy_row => RowHashAny, 0;
    lazy_row_dms, lazy_row => RowHashAny, 1;
    lazy2_row_dms, lazy2_row => RowHashAny, 2;
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use alloc::boxed::Box;

    use super::super::advanced_tests;
    use super::super::cctx::{Engine, RequestedParams, compress_bound};
    use super::super::cdict::{AttachPref, CDict};
    use super::super::dict_tests;
    use super::super::match_state::{BlockCompressor, ROW_TABLE_SLACK};
    use super::super::params::{ParamSwitch, Strategy, level_cparams};
    use super::super::stream::{EndOp, Stream};
    use super::super::{compress_once, compress_once_into};
    use super::*;
    use crate::Backend;
    use crate::dictionary::DictionaryContent;
    use zstd_ffi::corpus;
    use zstd_ffi::oracle::{self, Directive, Params};

    /// A prefix-only block over `bytes`, starting at window index 2.
    fn prefix_block(bytes: &[u8]) -> BlockInput<'_> {
        BlockInput {
            window: mem::Window {
                prefix: Mem::new(bytes, 2),
                ext: Mem::new(&[], 2),
            },
            start: 2,
            end: 2 + bytes.len() as u32,
            backend: Backend::default(),
            dms: None,
            symbol_costs: None,
        }
    }

    #[test]
    fn next_index_tables_follow_the_reference() {
        fn check<const ROW: usize>() {
            let mask = ROW as u32 - 1;
            for head in 0..=255u8 {
                let mut row = [0u8; ROW];
                row[0] = head;
                let next = u32::from(head).wrapping_sub(1) & mask;
                let expected = if next == 0 { mask } else { next };
                assert_eq!(row_next_index(&mut row), expected as usize, "{ROW} {head}");
                assert_eq!(u32::from(row[0]), expected);
            }
            // Built at run time too, not only into the constant.
            assert_eq!(next_index_table::<ROW>(), NextIndex::<ROW>::TABLE);
        }
        check::<16>();
        check::<32>();
        check::<64>();
    }

    #[test]
    fn row_tables_start_at_a_cache_line() {
        let mut slots = vec![0u32; 1024 + ROW_TABLE_SLACK];
        let view = aligned_rows(&mut slots, 1024, 0);
        assert_eq!(view.len(), 1024);
        assert_eq!(view.as_ptr().addr() % 64, 0);
        let mut tags = vec![0u8; 512 + ROW_TABLE_SLACK];
        let view = aligned_rows(&mut tags, 512, 0);
        assert_eq!(view.len(), 512);
        assert_eq!(view.as_ptr().addr() % 64, 0);
        // A moved view starts whole rows later, still on a cache line.
        let last = ROW_TABLE_SLACK - 64;
        let start = aligned_rows(&mut tags, 512, 0).as_ptr().addr();
        let moved = aligned_rows(&mut tags, 512, last);
        assert_eq!((moved.len(), moved.as_ptr().addr() - start), (512, last));
        // Without room to move, a table is used from its start.
        let mut short = vec![0u8; 100];
        let start = short.as_ptr();
        assert_eq!(aligned_rows(&mut short, 200, 0).as_ptr(), start);
        assert_eq!(aligned_rows(&mut short, 200, 0).len(), 100);
    }

    /// The per-index fallback of a row update inserts exactly what the
    /// word loop inserts.
    #[test]
    fn checked_row_updates_insert_like_the_word_loop() {
        fn check<const MLS: u32, const ROW: usize>(bytes: &[u8]) {
            let block = prefix_block(bytes);
            let row_log = ROW.trailing_zeros();
            let mut cparams = level_cparams(6, u64::MAX);
            cparams.search_log = row_log;
            cparams.min_match = MLS;
            let ctx = Ctx::new(&block, MLS, row_log, false);
            let mut fast = MatchState::new(cparams, ParamSwitch::Enable);
            let mut checked = MatchState::new(cparams, ParamSwitch::Enable);
            for (ms, by_index) in [(&mut fast, false), (&mut checked, true)] {
                let mut state = RowState::new(ms, &ctx);
                for (start, end) in [(12, 300), (300, 301), (301, 2000)] {
                    if by_index {
                        row_update_checked::<MLS, ROW>(&mut state, &ctx, start, end);
                    } else {
                        row_update_impl::<MLS, ROW>(&mut state, &ctx, start, end);
                    }
                }
            }
            // Each state's rows start where its own allocation aligns.
            let rows = |ms: &mut MatchState| {
                let len = 1usize << ms.cparams.hash_log;
                let view = ms.row_view;
                let tags = aligned_rows(&mut ms.tag_table, len, view).to_vec();
                (tags, aligned_rows(&mut ms.hash_table, len, view).to_vec())
            };
            assert!(rows(&mut fast) == rows(&mut checked), "{MLS} {ROW}");
            assert_eq!(fast.hash_cache, checked.hash_cache, "{MLS} {ROW}");
        }
        let bytes = corpus::text(4096, 61);
        check::<4, 16>(&bytes);
        check::<5, 32>(&bytes);
        check::<6, 64>(&bytes);
    }

    /// The per-index fallback of a tree update inserts exactly what the
    /// word loop inserts.
    #[test]
    fn checked_tree_updates_insert_like_the_word_loop() {
        let bytes = corpus::text(4096, 62);
        let block = prefix_block(&bytes);
        let cparams = level_cparams(13, u64::MAX);
        let ctx = Ctx::new(&block, 5, 4, false);
        let mut fast = MatchState::new(cparams, ParamSwitch::Disable);
        let mut checked = MatchState::new(cparams, ParamSwitch::Disable);
        fast.next_to_update = 12;
        update_dubt::<5>(&mut fast, &ctx, 3000);
        update_dubt::<5>(&mut fast, &ctx, 3000);
        update_dubt_checked::<5>(&mut checked, &ctx, 12, 3000);
        assert_eq!(fast.next_to_update, 3000);
        assert!(fast.hash_table == checked.hash_table);
        assert!(fast.chain_table == checked.chain_table);
    }

    /// A tree step in the prefix measures and orders runs as the
    /// external-dictionary step does, from any known common length.
    #[test]
    fn prefix_tree_steps_match_the_general_step() {
        let mut bytes = corpus::text(600, 63);
        bytes.extend_from_within(..300);
        let block = prefix_block(&bytes);
        let end = block.end;
        for backend in Backend::available() {
            fearless_simd::dispatch!(backend.0, simd => {
                let ctx = Ctx::new(&block, 4, 4, false);
                for (curr, limit) in [(602, end), (650, end), (700, 720)] {
                    let prefix = TreeRun::new::<false>(&ctx, curr, limit);
                    let general = TreeRun::new::<true>(&ctx, curr, limit);
                    for m in 2..curr {
                        let (full, _) = general.step::<_, true>(simd, &ctx, m, 0);
                        for known in [0usize, 1, 7, 8, 9, 30].map(|k| k.min(full)) {
                            let expected = general.step::<_, true>(simd, &ctx, m, known);
                            let got = prefix.step::<_, false>(simd, &ctx, m, known);
                            assert_eq!(got, expected, "{backend} {curr} {m} {known}");
                        }
                    }
                }
            });
        }
    }

    /// The reference's parameters for `requested`, row finder switch
    /// included.
    fn oracle_params(requested: &RequestedParams) -> Params {
        with_row_switch(advanced_tests::oracle_params(requested), requested)
    }

    /// `params` with the row finder switch of `requested`.
    fn with_row_switch(params: Params, requested: &RequestedParams) -> Params {
        let row = match requested.use_row_match_finder {
            ParamSwitch::Auto => 0,
            ParamSwitch::Enable => 1,
            ParamSwitch::Disable => 2,
        };
        params.with(zstd_ffi::ZSTD_c_useRowMatchFinder, row)
    }

    /// Every search specialization: strategies with and without the row
    /// finder, minimum matches 4 to 6 and row sizes 16 to 64.
    fn requests() -> Vec<RequestedParams> {
        let mut all = Vec::new();
        for strategy in [
            Strategy::Greedy,
            Strategy::Lazy,
            Strategy::Lazy2,
            Strategy::BtLazy2,
        ] {
            for row in [ParamSwitch::Enable, ParamSwitch::Disable] {
                for (min_match, search_log) in [(4, 4), (5, 5), (6, 6), (7, 3)] {
                    let mut requested = RequestedParams::default();
                    requested.cparams.strategy = Some(strategy);
                    requested.cparams.min_match = min_match;
                    requested.cparams.search_log = search_log;
                    requested.use_row_match_finder = row;
                    all.push(requested);
                }
            }
        }
        all
    }

    #[test]
    fn every_backend_matches_the_reference_one_shot() {
        let inputs = [
            corpus::text(200_000, 41),
            corpus::binary(120_000, 42),
            corpus::mixed(90_000, 43),
            vec![b'a'; 70_000],
        ];
        for backend in Backend::available() {
            let mut engine = Engine::with_backend(backend);
            for level in 5..=15 {
                let requested = RequestedParams {
                    compression_level: level,
                    ..RequestedParams::default()
                };
                for (i, input) in inputs.iter().enumerate() {
                    let ours = compress_once(&mut engine, &requested, input, 1).expect("compress");
                    let theirs =
                        oracle::compress2(input, &oracle_params(&requested)).expect("oracle");
                    assert!(ours == theirs, "{backend} level {level} input {i}");
                }
            }
            for (r, requested) in requests().iter().enumerate() {
                for (i, input) in inputs.iter().enumerate() {
                    let ours = compress_once(&mut engine, requested, input, 1).expect("compress");
                    let theirs =
                        oracle::compress2(input, &oracle_params(requested)).expect("oracle");
                    assert!(ours == theirs, "{backend} request {r} input {i}");
                }
            }
        }
    }

    /// Long distance matching runs the finders on parts of a block whose end
    /// comes before the end of the prefix they see; no match may run past
    /// that end (a regression: the row finder's candidates once did).
    #[test]
    fn every_backend_matches_the_reference_with_long_distance_matching() {
        let input = corpus::text(300_000, 1);
        let mut requested = RequestedParams {
            compression_level: 10,
            ..RequestedParams::default()
        };
        requested.cparams.chain_log = 10;
        requested.cparams.hash_log = 20;
        requested.cparams.min_match = 5;
        requested.ldm.enable = ParamSwitch::Enable;
        requested.ldm.min_match_length = 23;
        for backend in Backend::available() {
            let mut engine = Engine::with_backend(backend);
            for strategy in [Strategy::Greedy, Strategy::Lazy2, Strategy::BtLazy2] {
                for row in [ParamSwitch::Enable, ParamSwitch::Disable] {
                    requested.cparams.strategy = Some(strategy);
                    requested.use_row_match_finder = row;
                    let ours = compress_once(&mut engine, &requested, &input, 1).expect("compress");
                    let theirs =
                        oracle::compress2(&input, &oracle_params(&requested)).expect("oracle");
                    assert!(ours == theirs, "{backend} {strategy:?} {row:?}");
                }
            }
        }
    }

    /// A reused engine continues its indices and keeps the previous
    /// frames' table entries, as the reference does; they lie below the new
    /// window and must never be followed, whatever finder wrote them.
    #[test]
    fn reused_engines_match_fresh_reference_frames() {
        let inputs = [
            corpus::text(90_000, 71),
            corpus::binary(60_000, 72),
            corpus::text(3_000, 73),
            corpus::mixed(120_000, 74),
        ];
        let mut sequence = requests();
        for level in [5, 9, 12, 13, 15, 3, 16, 19, 22, 13] {
            sequence.push(RequestedParams {
                compression_level: level,
                ..RequestedParams::default()
            });
        }
        for backend in Backend::available() {
            let mut engine = Engine::with_backend(backend);
            for (r, requested) in sequence.iter().enumerate() {
                let input = &inputs[r % inputs.len()];
                let ours = compress_once(&mut engine, requested, input, r as u64).expect("ours");
                let theirs = oracle::compress2(input, &oracle_params(requested)).expect("oracle");
                assert!(ours == theirs, "{backend} frame {r}");
            }
        }
    }

    fn stream(
        engine: &mut Engine,
        requested: &RequestedParams,
        dict: Option<&CDict>,
        chunks: &[&[u8]],
    ) -> Vec<u8> {
        let mut state = Stream::new();
        let mut out = Vec::new();
        let mut buffer = vec![0u8; 1 << 17];
        for (n, &data) in chunks.iter().enumerate() {
            let op = if n + 1 == chunks.len() {
                EndOp::End
            } else {
                EndOp::Continue
            };
            let mut in_pos = 0;
            loop {
                let step = state
                    .compress(
                        engine,
                        requested,
                        dict,
                        &mut buffer[..],
                        &data[in_pos..],
                        op,
                    )
                    .expect("stream");
                in_pos += step.consumed;
                out.extend_from_slice(&buffer[..step.produced]);
                if in_pos == data.len() && (op != EndOp::End || step.remaining == 0) {
                    break;
                }
            }
        }
        out
    }

    /// Small windows wrap the streaming buffer, so every finder runs its
    /// extDict variant.
    #[test]
    fn every_backend_matches_the_reference_on_ext_dict_windows() {
        let input = [corpus::text(60_000, 51), corpus::mixed(60_000, 52)].concat();
        let chunks: Vec<&[u8]> = input.chunks(7_001).collect();
        let steps: Vec<(&[u8], Directive)> = chunks
            .iter()
            .enumerate()
            .map(|(n, &data)| {
                let op = if n + 1 == chunks.len() {
                    Directive::End
                } else {
                    Directive::Continue
                };
                (data, op)
            })
            .collect();
        for backend in Backend::available() {
            let mut engine = Engine::with_backend(backend);
            for (r, mut requested) in requests().into_iter().enumerate() {
                requested.cparams.window_log = 12;
                let theirs =
                    oracle::compress_stream(&oracle_params(&requested), &steps).expect("oracle");
                let ours = stream(&mut engine, &requested, None, &chunks);
                assert!(ours == theirs, "{backend} request {r}");
            }
        }
    }

    /// The row finder's parsers are specialized per block on the minimum
    /// match and row size (`row_block!`): every one of the nine pairs, at
    /// every depth, on prefix-only and extDict windows, writes the
    /// reference's frames.
    #[test]
    fn every_row_specialization_matches_the_reference() {
        let input = [corpus::text(40_000, 71), corpus::mixed(30_000, 72)].concat();
        let chunks: Vec<&[u8]> = input.chunks(5_003).collect();
        let steps: Vec<(&[u8], Directive)> = chunks
            .iter()
            .enumerate()
            .map(|(n, &data)| {
                let op = if n + 1 == chunks.len() {
                    Directive::End
                } else {
                    Directive::Continue
                };
                (data, op)
            })
            .collect();
        for backend in Backend::available() {
            let mut engine = Engine::with_backend(backend);
            for strategy in [Strategy::Greedy, Strategy::Lazy, Strategy::Lazy2] {
                for min_match in 4..=6 {
                    for search_log in 4..=6 {
                        let mut requested = RequestedParams::default();
                        requested.cparams.strategy = Some(strategy);
                        requested.cparams.min_match = min_match;
                        requested.cparams.search_log = search_log;
                        requested.use_row_match_finder = ParamSwitch::Enable;
                        let case =
                            alloc::format!("{backend} {strategy:?} {min_match} {search_log}");
                        let ours =
                            compress_once(&mut engine, &requested, &input, 1).expect("compress");
                        let theirs =
                            oracle::compress2(&input, &oracle_params(&requested)).expect("oracle");
                        assert!(ours == theirs, "{case} prefix");

                        requested.cparams.window_log = 12;
                        let theirs = oracle::compress_stream(&oracle_params(&requested), &steps)
                            .expect("oracle");
                        let ours = stream(&mut engine, &requested, None, &chunks);
                        assert!(ours == theirs, "{case} extDict");
                    }
                }
            }
        }
    }

    /// Whether frames attaching a dictionary made for `strategy` run this
    /// module's dictMatchState compressors.
    fn lazy_family(strategy: Strategy) -> bool {
        matches!(
            strategy,
            Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 | Strategy::BtLazy2
        )
    }

    /// Compresses every input with `dict` under `requested` on every
    /// backend, one reused engine per backend, and compares with the
    /// reference loading the same dictionary. Checks nothing and returns
    /// `false` when the dictionary's strategy is not a lazy one: attaching
    /// it would run another module's finder.
    fn check_attached(
        requested: &RequestedParams,
        dict: &[u8],
        content: DictionaryContent,
        inputs: &[Vec<u8>],
    ) -> bool {
        let cdict = CDict::new(Box::from(dict), content, requested).expect("prepares");
        if !lazy_family(cdict.cparams().strategy) {
            return false;
        }
        let params = with_row_switch(dict_tests::dict_params(requested, dict, content), requested);
        let expected: Vec<Vec<u8>> = inputs
            .iter()
            .map(|input| oracle::compress2(input, &params).expect("oracle"))
            .collect();
        for backend in Backend::available() {
            let mut engine = Engine::with_backend(backend);
            for (index, (input, expected)) in inputs.iter().zip(&expected).enumerate() {
                let mut dst = vec![0u8; compress_bound(input.len()).expect("bound")];
                let size = compress_once_into(
                    &mut engine,
                    requested,
                    Some(&cdict),
                    input,
                    index as u64,
                    &mut dst,
                )
                .expect("compresses");
                assert!(
                    dst[..size] == expected[..],
                    "{backend} input {index} ({} bytes), {requested:?}: {size} vs {} bytes",
                    input.len(),
                    expected.len()
                );
            }
        }
        true
    }

    /// A raw dictionary: text, then records like the inputs.
    fn raw_dictionary(records: &[Vec<u8>]) -> Vec<u8> {
        [corpus::text(20_000, 9), records[..200].concat()].concat()
    }

    /// Every lazy compression level attaches trained and raw dictionaries
    /// as the reference does, forced or by the size heuristic (small
    /// inputs attach, large ones copy).
    #[test]
    fn attached_dictionaries_match_the_reference_at_every_lazy_level() {
        let (trained, records) = dict_tests::trained();
        let raw = raw_dictionary(&records);
        let inputs = dict_tests::inputs(&records);
        let mut checked = 0;
        for level in 4..=12 {
            for pref in [AttachPref::ForceAttach, AttachPref::Default] {
                let requested = dict_tests::request(level, None, pref);
                for (dict, content) in [
                    (&trained, DictionaryContent::Auto),
                    (&raw, DictionaryContent::Auto),
                    (&trained, DictionaryContent::Raw),
                ] {
                    checked += usize::from(check_attached(&requested, dict, content, &inputs));
                }
            }
        }
        assert!(checked >= 30, "{checked}");
    }

    /// Every lazy strategy, forced, attaches as the reference does.
    #[test]
    fn attached_dictionaries_match_the_reference_for_every_lazy_strategy() {
        let (trained, records) = dict_tests::trained();
        let raw = raw_dictionary(&records);
        let inputs = dict_tests::inputs(&records);
        for strategy in [
            Strategy::Greedy,
            Strategy::Lazy,
            Strategy::Lazy2,
            Strategy::BtLazy2,
        ] {
            for level in [3, 9, 15] {
                let requested = dict_tests::request(level, Some(strategy), AttachPref::ForceAttach);
                assert!(check_attached(
                    &requested,
                    &trained,
                    DictionaryContent::Zstd,
                    &inputs
                ));
                assert!(check_attached(
                    &requested,
                    &raw,
                    DictionaryContent::Raw,
                    &inputs
                ));
            }
        }
    }

    /// The hash chain and the row finder, every row size and minimum match
    /// (3 and 7 clamp to 4 and 6 in the searches, not in the dictionary's
    /// load), with fewer attempts than a row has slots and with more.
    #[test]
    fn attached_dictionaries_match_the_reference_for_every_search_specialization() {
        let (trained, records) = dict_tests::trained();
        let inputs = dict_tests::inputs(&records);
        for strategy in [
            Strategy::Greedy,
            Strategy::Lazy,
            Strategy::Lazy2,
            Strategy::BtLazy2,
        ] {
            for row in [ParamSwitch::Enable, ParamSwitch::Disable] {
                for (min_match, search_log) in [
                    (3, 5),
                    (4, 4),
                    (5, 5),
                    (6, 6),
                    (7, 3),
                    (4, 7),
                    (5, 6),
                    (6, 4),
                    (6, 5),
                ] {
                    let mut requested =
                        dict_tests::request(6, Some(strategy), AttachPref::ForceAttach);
                    requested.cparams.min_match = min_match;
                    requested.cparams.search_log = search_log;
                    requested.use_row_match_finder = row;
                    assert!(check_attached(
                        &requested,
                        &trained,
                        DictionaryContent::Auto,
                        &inputs
                    ));
                }
            }
        }
    }

    /// Tables smaller than the dictionary: its chain and tree stop at their
    /// size, and the context's tables differ in size from the dictionary's
    /// once the input's size shrinks them.
    #[test]
    fn attached_dictionaries_match_the_reference_with_small_tables() {
        let (trained, records) = dict_tests::trained();
        let raw = raw_dictionary(&records);
        let inputs = dict_tests::inputs(&records);
        for strategy in [Strategy::Greedy, Strategy::Lazy2, Strategy::BtLazy2] {
            for row in [ParamSwitch::Enable, ParamSwitch::Disable] {
                for (chain_log, hash_log) in [(8, 10), (10, 8), (12, 14)] {
                    let mut requested =
                        dict_tests::request(9, Some(strategy), AttachPref::ForceAttach);
                    requested.cparams.chain_log = chain_log;
                    requested.cparams.hash_log = hash_log;
                    requested.use_row_match_finder = row;
                    assert!(check_attached(
                        &requested,
                        &trained,
                        DictionaryContent::Auto,
                        &inputs
                    ));
                    assert!(check_attached(
                        &requested,
                        &raw,
                        DictionaryContent::Raw,
                        &inputs
                    ));
                }
            }
        }
    }

    /// The dictMatchState compressors run the prefix-only parsers for a
    /// block without an attached dictionary.
    #[test]
    fn dms_compressors_without_a_dictionary_run_the_prefix_parsers() {
        let bytes = corpus::text(20_000, 64);
        let block = prefix_block(&bytes);
        let pairs: [(BlockCompressor, BlockCompressor, Strategy, ParamSwitch); 7] = [
            (greedy_dms, greedy, Strategy::Greedy, ParamSwitch::Disable),
            (lazy_dms, lazy, Strategy::Lazy, ParamSwitch::Disable),
            (lazy2_dms, lazy2, Strategy::Lazy2, ParamSwitch::Disable),
            (
                btlazy2_dms,
                btlazy2,
                Strategy::BtLazy2,
                ParamSwitch::Disable,
            ),
            (
                greedy_row_dms,
                greedy_row,
                Strategy::Greedy,
                ParamSwitch::Enable,
            ),
            (lazy_row_dms, lazy_row, Strategy::Lazy, ParamSwitch::Enable),
            (
                lazy2_row_dms,
                lazy2_row,
                Strategy::Lazy2,
                ParamSwitch::Enable,
            ),
        ];
        for (dms, plain, strategy, row) in pairs {
            let mut cparams = level_cparams(7, u64::MAX);
            cparams.strategy = strategy;
            let run = |compressor: BlockCompressor| {
                let mut ms = MatchState::new(cparams, row);
                let mut seq_store = SeqStore::default();
                seq_store.reserve(bytes.len(), cparams.min_match);
                let mut rep = [1, 4, 8];
                let last = compressor(&mut ms, &mut seq_store, &mut rep, &block);
                (
                    last,
                    rep,
                    seq_store.sequences().to_vec(),
                    seq_store.literals().to_vec(),
                )
            };
            let expected = run(plain);
            assert!(!expected.2.is_empty(), "{strategy:?}");
            assert!(run(dms) == expected, "{strategy:?} {row:?}");
        }
    }

    /// A forced attachment spanning many blocks, in a window that passes
    /// the dictionary: the blocks after it run without it.
    #[test]
    fn attached_dictionaries_match_the_reference_once_the_window_passes_them() {
        let (trained, records) = dict_tests::trained();
        let input = [records.concat(), corpus::text(150_000, 21)].concat();
        let inputs = [input];
        for strategy in [
            Strategy::Greedy,
            Strategy::Lazy,
            Strategy::Lazy2,
            Strategy::BtLazy2,
        ] {
            for row in [ParamSwitch::Enable, ParamSwitch::Disable] {
                for window_log in [17, 20] {
                    let mut requested =
                        dict_tests::request(7, Some(strategy), AttachPref::ForceAttach);
                    requested.cparams.window_log = window_log;
                    requested.use_row_match_finder = row;
                    assert!(check_attached(
                        &requested,
                        &trained,
                        DictionaryContent::Auto,
                        &inputs
                    ));
                }
            }
        }
    }

    /// Inputs shorter than a block's search limits, down to nothing.
    #[test]
    fn tiny_inputs_attach_dictionaries_like_the_reference() {
        let (trained, records) = dict_tests::trained();
        let raw = raw_dictionary(&records);
        let joined = records[records.len() - 10..].concat();
        let inputs: Vec<Vec<u8>> = [
            0usize, 1, 2, 7, 8, 9, 15, 16, 17, 24, 25, 31, 32, 33, 40, 64, 100, 200, 500,
        ]
        .iter()
        .map(|&len| joined[..len].to_vec())
        .collect();
        for strategy in [
            Strategy::Greedy,
            Strategy::Lazy,
            Strategy::Lazy2,
            Strategy::BtLazy2,
        ] {
            for row in [ParamSwitch::Enable, ParamSwitch::Disable] {
                for pref in [AttachPref::ForceAttach, AttachPref::Default] {
                    let mut requested = dict_tests::request(5, Some(strategy), pref);
                    requested.use_row_match_finder = row;
                    assert!(check_attached(
                        &requested,
                        &trained,
                        DictionaryContent::Auto,
                        &inputs
                    ));
                    assert!(check_attached(
                        &requested,
                        &raw,
                        DictionaryContent::Raw,
                        &inputs
                    ));
                }
            }
        }
    }

    /// Streams of unknown size attach the dictionary; every chunking and
    /// every backend produces the reference's frame, also when a small
    /// window wraps the stream's buffer.
    #[test]
    fn streamed_frames_attach_dictionaries_like_the_reference() {
        let (trained, records) = dict_tests::trained();
        let input = [records[..1500].concat(), corpus::mixed(40_000, 31)].concat();
        let chunkings = [1_000usize, 7_001, 65_536, input.len()];
        let mut requests = Vec::new();
        for strategy in [
            Strategy::Greedy,
            Strategy::Lazy,
            Strategy::Lazy2,
            Strategy::BtLazy2,
        ] {
            for row in [ParamSwitch::Enable, ParamSwitch::Disable] {
                for window_log in [0, 15] {
                    let mut requested = dict_tests::request(8, Some(strategy), AttachPref::Default);
                    requested.use_row_match_finder = row;
                    requested.cparams.window_log = window_log;
                    requests.push(requested);
                }
            }
        }
        for requested in &requests {
            let content = DictionaryContent::Auto;
            let cdict = CDict::new(Box::from(&trained[..]), content, requested).expect("prepares");
            let params = with_row_switch(
                dict_tests::dict_params(requested, &trained, content),
                requested,
            );
            for size in chunkings {
                let chunks: Vec<&[u8]> = input.chunks(size).collect();
                let steps: Vec<(&[u8], Directive)> = chunks
                    .iter()
                    .enumerate()
                    .map(|(n, &data)| {
                        let op = if n + 1 == chunks.len() {
                            Directive::End
                        } else {
                            Directive::Continue
                        };
                        (data, op)
                    })
                    .collect();
                let theirs = oracle::compress_stream(&params, &steps).expect("oracle");
                for backend in Backend::available() {
                    let mut engine = Engine::with_backend(backend);
                    let ours = stream(&mut engine, requested, Some(&cdict), &chunks);
                    assert!(ours == theirs, "{backend} chunks of {size}, {requested:?}");
                }
            }
        }
    }
}

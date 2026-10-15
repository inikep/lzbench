//! Match finder state shared by every strategy (`ZSTD_MatchState_t`).

use alloc::vec::Vec;

use super::entropy::Tables;
use super::ldm::RawSeqStore;
use super::mem;
use super::opt::OptState;
use super::params::{CParams, ParamSwitch, Strategy, row_match_finder_used};
use super::seq_store::SeqStore;
use super::window::WindowState;
use crate::Backend;

/// `ZSTD_HASHLOG3_MAX`.
const HASHLOG3_MAX: u32 = 17;
/// `ZSTD_ROW_HASH_CACHE_SIZE`.
pub(crate) const ROW_HASH_CACHE_SIZE: usize = 8;
/// Entries the row match finder's tables carry past their size, so that the
/// finder can start both at a 64-byte boundary, as the reference's
/// workspace aligns them.
pub(crate) const ROW_TABLE_SLACK: usize = 64;

/// The block a match finder works on.
#[derive(Copy, Clone, Debug)]
pub(crate) struct BlockInput<'a> {
    /// History and the block itself.
    pub(crate) window: mem::Window<'a>,
    /// Index of the block's first byte.
    pub(crate) start: u32,
    /// Index one past the block's last byte.
    pub(crate) end: u32,
    /// The SIMD level the block's kernels run with.
    pub(crate) backend: Backend,
    /// The attached dictionary, when the match finder runs in
    /// dictMatchState mode.
    pub(crate) dms: Option<DictMatchState<'a>>,
    /// The entropy tables the block inherits; the optimal parser seeds its
    /// first block's statistics from them when a dictionary validated them
    /// (`ms->opt.symbolCosts`). `None` where nothing is inherited.
    pub(crate) symbol_costs: Option<&'a Tables>,
}

/// An attached dictionary (`ms->dictMatchState`): its match finder tables
/// and its bytes, which the context reads but never changes.
#[derive(Copy, Clone, Debug)]
pub(crate) struct DictMatchState<'a> {
    /// The dictionary's tables, window state and parameters.
    pub(crate) ms: &'a MatchState,
    /// The dictionary's content, addressed by the dictionary's own indices
    /// (`ext` is empty).
    pub(crate) window: mem::Window<'a>,
}

/// A block compressor: fills `seq_store` with the block's sequences, updates
/// the repeat offsets, and returns the length of the trailing literals.
pub(crate) type BlockCompressor =
    fn(&mut MatchState, &mut SeqStore, &mut [u32; 3], &BlockInput<'_>) -> usize;

/// Match finder tables and window state.
#[derive(Debug)]
pub(crate) struct MatchState {
    pub(crate) window: WindowState,
    /// Index of the end of a loaded or attached dictionary, `0` when none
    /// is in effect.
    pub(crate) loaded_dict_end: u32,
    /// Whether a dictionary is attached (`ms->dictMatchState != NULL`);
    /// the dictionary itself comes with each call's segments.
    pub(crate) dict_match_state: bool,
    /// Index from which table updates resume.
    pub(crate) next_to_update: u32,
    /// Hash log of the 3-byte table, `0` when unused.
    pub(crate) hash_log3: u32,
    /// Row match finder: hash log of rows.
    pub(crate) row_hash_log: u32,
    /// Row match finder: per-row tags and heads.
    pub(crate) tag_table: Vec<u8>,
    /// Row match finder: hashes of upcoming positions.
    pub(crate) hash_cache: [u32; ROW_HASH_CACHE_SIZE],
    pub(crate) hash_table: Vec<u32>,
    pub(crate) hash_table3: Vec<u32>,
    pub(crate) chain_table: Vec<u32>,
    /// Treat the next input as non-contiguous.
    pub(crate) force_non_contiguous: bool,
    /// Optimal parser state.
    pub(crate) opt: OptState,
    pub(crate) cparams: CParams,
    /// Lazy match finders only insert searched positions.
    pub(crate) lazy_skipping: bool,
    /// Whether the row match finder is in use.
    pub(crate) use_row_match_finder: ParamSwitch,
    /// Long matches offered to the optimal parser for the current block.
    pub(crate) ldm_seq_store: Option<RawSeqStore>,
    /// The SIMD level the match finders' kernels run with, selected once
    /// per block outside their loops.
    pub(crate) backend: Backend,
}

impl MatchState {
    /// A state for `cparams`, with fresh indices and clean tables.
    #[cfg(test)]
    pub(crate) fn new(cparams: CParams, use_row_match_finder: ParamSwitch) -> Self {
        let mut state = Self::empty(cparams, use_row_match_finder);
        state.reset(cparams, use_row_match_finder, true);
        state
    }

    /// A state with fresh indices and no tables; [`Self::reset`] sizes them.
    pub(crate) fn empty(cparams: CParams, use_row_match_finder: ParamSwitch) -> Self {
        Self {
            window: WindowState::new(),
            loaded_dict_end: 0,
            dict_match_state: false,
            next_to_update: 0,
            hash_log3: 0,
            row_hash_log: 0,
            tag_table: Vec::new(),
            hash_cache: [0; ROW_HASH_CACHE_SIZE],
            hash_table: Vec::new(),
            hash_table3: Vec::new(),
            chain_table: Vec::new(),
            force_non_contiguous: false,
            opt: OptState::default(),
            cparams,
            lazy_skipping: false,
            use_row_match_finder,
            ldm_seq_store: None,
            backend: Backend::default(),
        }
    }

    /// `ZSTD_reset_matchState` for a compression context, with clean
    /// tables.
    ///
    /// With `reset_index` the window restarts from index 2; otherwise indices
    /// continue after the previous input, as the reference does when it
    /// reuses a context.
    pub(crate) fn reset(
        &mut self,
        cparams: CParams,
        use_row_match_finder: ParamSwitch,
        reset_index: bool,
    ) {
        self.reset_for(cparams, use_row_match_finder, reset_index, false);
    }

    /// `ZSTD_reset_matchState` for a dictionary's own tables
    /// (`ZSTD_resetTarget_CDict`): fresh indices, clean tables, and neither
    /// the 3-byte table nor the optimal parser's state.
    pub(crate) fn reset_for_dictionary(
        &mut self,
        cparams: CParams,
        use_row_match_finder: ParamSwitch,
    ) {
        self.reset_for(cparams, use_row_match_finder, true, true);
    }

    #[inline]
    fn reset_for(
        &mut self,
        cparams: CParams,
        use_row_match_finder: ParamSwitch,
        reset_index: bool,
        for_dictionary: bool,
    ) {
        let strategy = cparams.strategy;
        let row = row_match_finder_used(strategy, use_row_match_finder);
        let chain_size = if strategy != Strategy::Fast && !row {
            1usize << cparams.chain_log
        } else {
            0
        };
        let h_size = 1usize << cparams.hash_log;
        let hash_log3 = if cparams.min_match == 3 && !for_dictionary {
            HASHLOG3_MAX.min(cparams.window_log)
        } else {
            0
        };
        let h3_size = if hash_log3 != 0 {
            1usize << hash_log3
        } else {
            0
        };

        if reset_index {
            self.window = WindowState::new();
        }
        self.hash_log3 = hash_log3;
        self.lazy_skipping = false;

        // ZSTD_invalidateMatchState.
        self.window.clear();
        self.next_to_update = self.window.dict_limit;
        self.loaded_dict_end = 0;
        self.dict_match_state = false;
        self.opt.lit_length_sum = 0;

        // The reference zeroes its tables only when indices restart
        // (`ZSTD_cwksp_mark_tables_dirty`, then `ZSTD_cwksp_clean_tables`),
        // whatever the strategy; when they continue, every entry left from
        // the previous input lies below the new window, so it is kept. Every
        // finder tests an entry against the window before following it —
        // fast and dfast directly, the hash chain and row searches at
        // `lowLimit`, the binary trees at `windowLow`/`btLow`, the 3-byte
        // table at its own low bound — and rewrites a tree node's cells when
        // it inserts the node, so a stale entry acts as the zero it
        // replaces. The row finder's tag table is likewise never re-zeroed
        // (`ZSTD_cwksp_reserve_aligned_init_once`); its salt only permutes
        // rows and tags, so it is not modelled.
        let keep_stale = !reset_index;
        let slack = if row { ROW_TABLE_SLACK } else { 0 };
        for (table, size) in [
            (&mut self.hash_table, h_size + slack),
            (&mut self.chain_table, chain_size),
            (&mut self.hash_table3, h3_size),
        ] {
            if !keep_stale {
                table.clear();
            }
            grow_zeroed(table, size);
        }

        if !(row && keep_stale) {
            self.tag_table.clear();
        }
        if row {
            self.tag_table.resize(h_size + slack, 0);
            let row_log = cparams.search_log.clamp(4, 6);
            self.row_hash_log = cparams.hash_log - row_log;
        }
        if strategy >= Strategy::BtOpt && !for_dictionary {
            self.opt.allocate();
        }
        self.cparams = cparams;
        self.use_row_match_finder = use_row_match_finder;
    }

    /// `ZSTD_getLowestPrefixIndex`.
    pub(crate) const fn lowest_prefix_index(&self, curr: u32, window_log: u32) -> u32 {
        let max_distance = 1u32 << window_log;
        let lowest_valid = self.window.dict_limit;
        let within_window = if curr.wrapping_sub(lowest_valid) > max_distance {
            curr - max_distance
        } else {
            lowest_valid
        };
        if self.loaded_dict_end != 0 {
            lowest_valid
        } else {
            within_window
        }
    }

    /// `ZSTD_getLowestMatchIndex`.
    pub(crate) const fn lowest_match_index(&self, curr: u32, window_log: u32) -> u32 {
        let max_distance = 1u32 << window_log;
        let lowest_valid = self.window.low_limit;
        let within_window = if curr.wrapping_sub(lowest_valid) > max_distance {
            curr - max_distance
        } else {
            lowest_valid
        };
        if self.loaded_dict_end != 0 {
            lowest_valid
        } else {
            within_window
        }
    }

    /// Copies a dictionary's tables, untagging fast and dfast entries, and
    /// takes over its window (`ZSTD_resetCCtx_byCopyingCDict`). Both states
    /// must have the same table parameters.
    pub(crate) fn copy_dictionary_tables(&mut self, dict: &Self) {
        let tagged = super::params::cdict_indices_are_tagged(&dict.cparams);
        let copy = |dst: &mut [u32], src: &[u32]| {
            if tagged {
                for (dst, &src) in dst.iter_mut().zip(src) {
                    *dst = src >> super::params::SHORT_CACHE_TAG_BITS;
                }
            } else {
                dst.copy_from_slice(src);
            }
        };
        let h_size = 1usize << dict.cparams.hash_log;
        let row = row_match_finder_used(dict.cparams.strategy, dict.use_row_match_finder);
        if row {
            // Row tables are read from their first 64-byte boundary.
            let dst = aligned_range(&self.hash_table, h_size);
            let src = aligned_range(&dict.hash_table, h_size);
            copy(&mut self.hash_table[dst], &dict.hash_table[src]);
            let dst = aligned_range(&self.tag_table, h_size);
            let src = aligned_range(&dict.tag_table, h_size);
            self.tag_table[dst].copy_from_slice(&dict.tag_table[src]);
        } else {
            copy(&mut self.hash_table[..h_size], &dict.hash_table[..h_size]);
        }
        if !self.chain_table.is_empty() {
            let chain_size = dict.chain_table.len();
            copy(&mut self.chain_table[..chain_size], &dict.chain_table);
        }
        // The dictionary never fills a 3-byte table.
        self.hash_table3.fill(0);
        self.window = dict.window;
        self.next_to_update = dict.next_to_update;
        self.loaded_dict_end = dict.loaded_dict_end;
    }

    /// `ZSTD_overflowCorrectIfNeeded` before indexing `[ip, iend)`: rebases
    /// every index when they come too close to overflowing, which also
    /// forgets a loaded or attached dictionary.
    pub(crate) fn overflow_correct_if_needed(&mut self, ip: u32, iend: u32) {
        let cycle_log = super::params::cycle_log(self.cparams.chain_log, self.cparams.strategy);
        let max_dist = 1u32 << self.cparams.window_log;
        if self
            .window
            .need_overflow_correction(cycle_log, max_dist, self.loaded_dict_end, ip, iend)
        {
            let correction = self.window.correct_overflow(cycle_log, max_dist, ip);
            self.reduce_index(correction);
            self.next_to_update = self.next_to_update.saturating_sub(correction);
            self.loaded_dict_end = 0;
            self.dict_match_state = false;
        }
    }

    /// Subtracts `reducer` from every stored index, squashing older ones to
    /// zero (`ZSTD_reduceIndex`).
    pub(crate) fn reduce_index(&mut self, reducer: u32) {
        reduce_table(&mut self.hash_table, reducer, false);
        let preserve_mark = self.cparams.strategy == Strategy::BtLazy2;
        reduce_table(&mut self.chain_table, reducer, preserve_mark);
        reduce_table(&mut self.hash_table3, reducer, false);
    }
}

/// Where the first `len` entries of a row table start at its first 64-byte
/// boundary, as the reference's workspace aligns row tables; a table
/// without [`ROW_TABLE_SLACK`] entries to spare is used from its start.
pub(crate) fn aligned_range<T>(table: &[T], len: usize) -> core::ops::Range<usize> {
    const ALIGN: usize = 64;
    let skip = table.as_ptr().addr().wrapping_neg() % ALIGN / size_of::<T>().max(1);
    if skip < ROW_TABLE_SLACK && skip + len <= table.len() {
        skip..skip + len
    } else {
        0..len.min(table.len())
    }
}

/// Sizes `table` to `len` entries, new entries zero.
///
/// An empty table is allocated zeroed, which the allocator can serve
/// without writing (most of a large table is never touched by a small
/// input); a table holding entries is resized in place, zeroing only the
/// newly exposed tail.
fn grow_zeroed(table: &mut Vec<u32>, len: usize) {
    if table.is_empty() && table.capacity() < len {
        *table = alloc::vec![0; len];
    } else {
        table.resize(len, 0);
    }
}

/// `ZSTD_DUBT_UNSORTED_MARK`.
pub(crate) const DUBT_UNSORTED_MARK: u32 = 1;

/// `ZSTD_reduceTable_internal`.
fn reduce_table(table: &mut [u32], reducer: u32, preserve_mark: bool) {
    let threshold = reducer + super::window::WINDOW_START_INDEX;
    for cell in table {
        *cell = if preserve_mark && *cell == DUBT_UNSORTED_MARK {
            DUBT_UNSORTED_MARK
        } else if *cell < threshold {
            0
        } else {
            *cell - reducer
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::core::params::level_cparams;

    #[test]
    fn reducing_indices_shifts_recent_entries_and_clears_older_ones() {
        let mut ms = MatchState::new(level_cparams(13, u64::MAX), ParamSwitch::Disable);
        assert_eq!(ms.cparams.strategy, Strategy::BtLazy2);
        let reducer = 1000;
        let threshold = reducer + super::super::window::WINDOW_START_INDEX;
        for table in [&mut ms.hash_table, &mut ms.chain_table] {
            table[..4].copy_from_slice(&[0, threshold - 1, threshold, threshold + 77]);
        }
        ms.chain_table[4] = DUBT_UNSORTED_MARK;
        ms.hash_table[4] = DUBT_UNSORTED_MARK;
        ms.reduce_index(reducer);
        let shifted = [0, 0, threshold - reducer, threshold + 77 - reducer];
        assert_eq!(ms.hash_table[..4], shifted);
        assert_eq!(ms.chain_table[..4], shifted);
        // The binary tree's unsorted mark survives in the chain table only.
        assert_eq!(ms.chain_table[4], DUBT_UNSORTED_MARK);
        assert_eq!(ms.hash_table[4], 0);
    }

    #[test]
    fn tables_follow_the_strategy() {
        let fast = MatchState::new(level_cparams(1, u64::MAX), ParamSwitch::Disable);
        assert!(fast.chain_table.is_empty());
        assert_eq!(fast.hash_table.len(), 1 << fast.cparams.hash_log);
        let row = MatchState::new(level_cparams(6, u64::MAX), ParamSwitch::Enable);
        assert!(row.chain_table.is_empty());
        assert_eq!(row.tag_table.len(), row.hash_table.len());
        let lazy = MatchState::new(level_cparams(6, u64::MAX), ParamSwitch::Disable);
        assert_eq!(lazy.chain_table.len(), 1 << lazy.cparams.chain_log);
        let opt = MatchState::new(level_cparams(19, u64::MAX), ParamSwitch::Disable);
        assert!(opt.hash_log3 > 0);
    }

    #[test]
    fn continuing_indices_keep_every_table() {
        for level in [1, 3, 6, 13, 19] {
            for row in [ParamSwitch::Disable, ParamSwitch::Enable] {
                let cparams = level_cparams(level, u64::MAX);
                let mut ms = MatchState::new(cparams, row);
                for table in [&mut ms.hash_table, &mut ms.chain_table, &mut ms.hash_table3] {
                    table.fill(77);
                }
                ms.tag_table.fill(77);
                ms.reset(cparams, row, false);
                for table in [&ms.hash_table, &ms.chain_table, &ms.hash_table3] {
                    assert!(table.iter().all(|&v| v == 77), "{level} {row:?}");
                }
                assert!(ms.tag_table.iter().all(|&v| v == 77), "{level} {row:?}");
                // Restarting indices always cleans.
                ms.reset(cparams, row, true);
                for table in [&ms.hash_table, &ms.chain_table, &ms.hash_table3] {
                    assert!(table.iter().all(|&v| v == 0), "{level} {row:?}");
                }
                assert!(ms.tag_table.iter().all(|&v| v == 0), "{level} {row:?}");
            }
        }
        assert!(
            !MatchState::new(level_cparams(19, u64::MAX), ParamSwitch::Disable)
                .hash_table3
                .is_empty()
        );
    }

    #[test]
    fn growing_a_table_zeroes_only_new_entries() {
        let mut table = alloc::vec::Vec::new();
        grow_zeroed(&mut table, 4);
        assert_eq!(table, [0; 4]);
        table.fill(5);
        grow_zeroed(&mut table, 6);
        assert_eq!(table, [5, 5, 5, 5, 0, 0]);
        grow_zeroed(&mut table, 2);
        assert_eq!(table, [5, 5]);
        table.clear();
        grow_zeroed(&mut table, 3);
        assert_eq!(table, [0; 3]);
    }

    #[test]
    fn reducing_squashes_old_indices_and_keeps_marks() {
        let mut table = [0, 1, 5, 100, 1000];
        reduce_table(&mut table, 50, true);
        assert_eq!(table, [0, 1, 0, 50, 950]);
        let mut table = [1, 100];
        reduce_table(&mut table, 50, false);
        assert_eq!(table, [0, 50]);
    }

    #[test]
    fn lowest_indices_respect_the_window() {
        let mut state = MatchState::new(level_cparams(1, u64::MAX), ParamSwitch::Disable);
        state.window.dict_limit = 10;
        state.window.low_limit = 5;
        assert_eq!(state.lowest_prefix_index(100, 10), 10);
        assert_eq!(state.lowest_prefix_index(5000, 10), 5000 - 1024);
        assert_eq!(state.lowest_match_index(100, 10), 5);
        state.loaded_dict_end = 1;
        assert_eq!(state.lowest_prefix_index(5000, 10), 10);
        assert_eq!(state.lowest_match_index(5000, 10), 5);
    }
}

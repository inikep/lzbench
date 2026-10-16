//! The sliding window's index bookkeeping (`ZSTD_window_t` and its
//! functions in `zstd_compress_internal.h`).
//!
//! The reference compares raw pointers: a new input is contiguous with the
//! window when it starts exactly where the last one ended, and an external
//! dictionary shrinks when new input is written over its memory. Those
//! decisions change which matches exist, so this port models memory as
//! [`Segment`]s and keeps the reference's arithmetic within a segment.
//! Addresses in different segments never compare as adjacent or overlapping,
//! exactly as two distinct allocations never do in the reference.

use super::cdict::CDict;
use super::mem;

/// `ZSTD_WINDOW_START_INDEX`.
pub(crate) const WINDOW_START_INDEX: u32 = 2;
/// `HASH_READ_SIZE`.
pub(crate) const HASH_READ_SIZE: u32 = 8;
/// `ZSTD_CURRENT_MAX` on 64-bit targets.
pub(crate) const CURRENT_MAX: u32 = 3500 << 20;

/// Runs index overflow correction as early as the reference allows, as the
/// reference does when built with `ZSTD_WINDOW_OVERFLOW_CORRECT_FREQUENTLY`.
/// Test-only: the C oracle reads the same variable when it is built.
pub(crate) const OVERFLOW_CORRECT_FREQUENTLY: bool =
    option_env!("MZSTD_OVERFLOW_CORRECT_FREQUENTLY").is_some();

/// A memory region within which addresses compare meaningfully.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) enum Segment {
    /// The static placeholder the reference initializes its window with.
    Initial,
    /// The compressor's own input buffer.
    Buffer,
    /// Caller memory handed to one compression call.
    Input(u64),
    /// The bytes of the dictionary the frame is compressed with.
    Dict,
}

/// An address: a segment and a byte offset that may lie outside it, as a
/// base pointer does.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct Addr {
    pub(crate) segment: Segment,
    pub(crate) offset: i64,
}

impl Addr {
    /// `offset` bytes into `segment`.
    pub(crate) const fn new(segment: Segment, offset: i64) -> Self {
        Self { segment, offset }
    }

    /// This address moved by `delta` bytes.
    pub(crate) const fn add(self, delta: i64) -> Self {
        Self {
            segment: self.segment,
            offset: self.offset + delta,
        }
    }
}

/// Resolves segments to their bytes for one block.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Segments<'a> {
    /// The compressor's input buffer.
    pub(crate) buffer: &'a [u8],
    /// The current call's input, and its identity.
    pub(crate) input: (&'a [u8], u64),
    /// The dictionary's bytes ([`Segment::Dict`]).
    pub(crate) dict: &'a [u8],
    /// The dictionary the frame was begun with, if any; an attached one's
    /// tables are read from it.
    pub(crate) cdict: Option<&'a CDict>,
}

impl<'a> Segments<'a> {
    /// Segments for a call reading `input`, with the frame's dictionary.
    pub(crate) fn input(input: &'a [u8], id: u64, cdict: Option<&'a CDict>) -> Self {
        Self {
            buffer: &[],
            input: (input, id),
            dict: cdict.map_or(&[], CDict::bytes),
            cdict,
        }
    }

    fn bytes(&self, segment: Segment) -> &'a [u8] {
        match segment {
            Segment::Initial => b" \0",
            Segment::Buffer => self.buffer,
            Segment::Input(id) => {
                assert_eq!(id, self.input.1, "window refers to a released input");
                self.input.0
            }
            Segment::Dict => self.dict,
        }
    }
}

/// `ZSTD_window_t`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct WindowState {
    /// Where the next contiguous input would start.
    pub(crate) next_src: Addr,
    /// Address of index 0 for prefix indices.
    pub(crate) base: Addr,
    /// Address of index 0 for external-dictionary indices.
    pub(crate) dict_base: Addr,
    /// Below this index, history lives in the external dictionary.
    pub(crate) dict_limit: u32,
    /// Below this index, nothing is valid.
    pub(crate) low_limit: u32,
    /// Overflow corrections since [`WindowState::new`].
    pub(crate) nb_overflow_corrections: u32,
}

impl Default for WindowState {
    fn default() -> Self {
        Self::new()
    }
}

impl WindowState {
    /// `ZSTD_window_init`.
    pub(crate) const fn new() -> Self {
        let base = Addr::new(Segment::Initial, 0);
        Self {
            next_src: base.add(WINDOW_START_INDEX as i64),
            base,
            dict_base: base,
            dict_limit: WINDOW_START_INDEX,
            low_limit: WINDOW_START_INDEX,
            nb_overflow_corrections: 0,
        }
    }

    /// Index of `addr`, which must lie in the prefix segment.
    pub(crate) fn index_of(&self, addr: Addr) -> u32 {
        debug_assert_eq!(addr.segment, self.base.segment);
        (addr.offset - self.base.offset) as u32
    }

    /// Index one past the most recent input (`nextSrc - base`).
    pub(crate) fn end_index(&self) -> u32 {
        self.index_of(self.next_src)
    }

    /// `ZSTD_window_clear`.
    pub(crate) fn clear(&mut self) {
        let end = self.end_index();
        self.low_limit = end;
        self.dict_limit = end;
    }

    /// `ZSTD_window_hasExtDict`.
    pub(crate) const fn has_ext_dict(&self) -> bool {
        self.low_limit < self.dict_limit
    }

    /// Appends `len` bytes at `src` (`ZSTD_window_update`).
    ///
    /// Returns whether the input continued the current segment.
    pub(crate) fn update(&mut self, src: Addr, len: usize, force_non_contiguous: bool) -> bool {
        let mut contiguous = true;
        if len == 0 {
            return contiguous;
        }
        if src != self.next_src || force_non_contiguous {
            let distance_from_base = self.next_src.offset - self.base.offset;
            self.low_limit = self.dict_limit;
            self.dict_limit = distance_from_base as u32;
            self.dict_base = self.base;
            self.base = src.add(-distance_from_base);
            if self.dict_limit - self.low_limit < HASH_READ_SIZE {
                self.low_limit = self.dict_limit;
            }
            contiguous = false;
        }
        self.next_src = src.add(len as i64);
        // Input written over the dictionary invalidates that part of it.
        if src.segment == self.dict_base.segment {
            let input_end = src.offset + len as i64;
            let dict_low = self.dict_base.offset + i64::from(self.low_limit);
            let dict_high = self.dict_base.offset + i64::from(self.dict_limit);
            if input_end > dict_low && src.offset < dict_high {
                let high_input_idx = input_end - self.dict_base.offset;
                let low_limit_max = if high_input_idx > i64::from(self.dict_limit) {
                    self.dict_limit
                } else {
                    high_input_idx as u32
                };
                self.low_limit = low_limit_max;
            }
        }
        contiguous
    }

    /// `ZSTD_window_enforceMaxDist`: raises the limits so nothing older than
    /// `max_dist` before `block_end` stays reachable, and then forgets a
    /// loaded or attached dictionary.
    pub(crate) fn enforce_max_dist(
        &mut self,
        block_end: u32,
        max_dist: u32,
        loaded_dict_end: &mut u32,
        dict_match_state: &mut bool,
    ) {
        let loaded = *loaded_dict_end;
        if block_end > max_dist.wrapping_add(loaded) {
            let new_low_limit = block_end - max_dist;
            if self.low_limit < new_low_limit {
                self.low_limit = new_low_limit;
            }
            if self.dict_limit < self.low_limit {
                self.dict_limit = self.low_limit;
            }
            *loaded_dict_end = 0;
            *dict_match_state = false;
        }
    }

    /// `ZSTD_checkDictValidity`: a dictionary stays usable for the block
    /// ending at `block_end` only while the whole block is within
    /// `max_dist` of its end and the window still starts where it ended.
    pub(crate) const fn check_dict_validity(
        &self,
        block_end: u32,
        max_dist: u32,
        loaded_dict_end: &mut u32,
        dict_match_state: &mut bool,
    ) {
        let loaded = *loaded_dict_end;
        if block_end > loaded.wrapping_add(max_dist) || loaded != self.dict_limit {
            *loaded_dict_end = 0;
            *dict_match_state = false;
        }
    }

    /// `ZSTD_window_canOverflowCorrect`.
    fn can_overflow_correct(
        &self,
        cycle_log: u32,
        max_dist: u32,
        loaded_dict_end: u32,
        src: u32,
    ) -> bool {
        let cycle_size = 1u32 << cycle_log;
        let min_index_to_overflow_correct = cycle_size
            .wrapping_add(max_dist.max(cycle_size))
            .wrapping_add(WINDOW_START_INDEX);
        let adjustment = self.nb_overflow_corrections + 1;
        let adjusted_index = (min_index_to_overflow_correct.wrapping_mul(adjustment))
            .max(min_index_to_overflow_correct);
        let index_large_enough = src > adjusted_index;
        let dictionary_invalidated = src > max_dist.wrapping_add(loaded_dict_end);
        index_large_enough && dictionary_invalidated
    }

    /// `ZSTD_window_needOverflowCorrection` for the block `[src, src_end)`.
    pub(crate) fn need_overflow_correction(
        &self,
        cycle_log: u32,
        max_dist: u32,
        loaded_dict_end: u32,
        src: u32,
        src_end: u32,
    ) -> bool {
        if OVERFLOW_CORRECT_FREQUENTLY
            && self.can_overflow_correct(cycle_log, max_dist, loaded_dict_end, src)
        {
            return true;
        }
        src_end > CURRENT_MAX
    }

    /// `ZSTD_window_correctOverflow`: rebases indices so `src` lands just
    /// above `max_dist`, and returns the correction to subtract from every
    /// stored index.
    pub(crate) fn correct_overflow(&mut self, cycle_log: u32, max_dist: u32, src: u32) -> u32 {
        let cycle_size = 1u32 << cycle_log;
        let cycle_mask = cycle_size - 1;
        let curr = src;
        let current_cycle = curr & cycle_mask;
        let current_cycle_correction = if current_cycle < WINDOW_START_INDEX {
            cycle_size.max(WINDOW_START_INDEX)
        } else {
            0
        };
        let new_current = current_cycle + current_cycle_correction + max_dist.max(cycle_size);
        let correction = curr - new_current;
        debug_assert!(OVERFLOW_CORRECT_FREQUENTLY || correction > 1 << 28);

        self.base = self.base.add(i64::from(correction));
        self.dict_base = self.dict_base.add(i64::from(correction));
        if self.low_limit < correction + WINDOW_START_INDEX {
            self.low_limit = WINDOW_START_INDEX;
        } else {
            self.low_limit -= correction;
        }
        if self.dict_limit < correction + WINDOW_START_INDEX {
            self.dict_limit = WINDOW_START_INDEX;
        } else {
            self.dict_limit -= correction;
        }
        self.nb_overflow_corrections += 1;
        correction
    }

    /// The segments visible to a block ending at index `block_end`.
    pub(crate) fn view<'a>(&self, segments: &Segments<'a>, block_end: u32) -> mem::Window<'a> {
        let prefix_bytes = segments.bytes(self.base.segment);
        let from = (self.base.offset + i64::from(self.dict_limit)) as usize;
        let to = (self.base.offset + i64::from(block_end)) as usize;
        let prefix = mem::Mem::new(&prefix_bytes[from..to], self.dict_limit);
        let ext = if self.low_limit < self.dict_limit {
            let dict_bytes = segments.bytes(self.dict_base.segment);
            let from = (self.dict_base.offset + i64::from(self.low_limit)) as usize;
            let to = (self.dict_base.offset + i64::from(self.dict_limit)) as usize;
            mem::Mem::new(&dict_bytes[from..to], self.low_limit)
        } else {
            mem::Mem::new(&[], self.dict_limit)
        };
        mem::Window { prefix, ext }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_input_starts_at_the_window_start_index() {
        let mut window = WindowState::new();
        let src = Addr::new(Segment::Input(1), 0);
        assert!(!window.update(src, 100, false));
        assert_eq!(window.index_of(src), WINDOW_START_INDEX);
        assert_eq!(window.low_limit, window.dict_limit);
        assert!(!window.has_ext_dict());
        assert!(window.update(src.add(100), 50, false));
        assert_eq!(window.end_index(), WINDOW_START_INDEX + 150);
        assert!(window.update(src, 0, true));
    }

    #[test]
    fn a_new_segment_turns_the_prefix_into_an_external_dictionary() {
        let mut window = WindowState::new();
        window.update(Addr::new(Segment::Buffer, 0), 1000, false);
        let input = Addr::new(Segment::Input(2), 0);
        assert!(!window.update(input, 10, false));
        assert!(window.has_ext_dict());
        assert_eq!(window.low_limit, WINDOW_START_INDEX);
        assert_eq!(window.dict_limit, WINDOW_START_INDEX + 1000);
        assert_eq!(window.index_of(input), WINDOW_START_INDEX + 1000);
    }

    #[test]
    fn input_written_over_the_dictionary_shrinks_it() {
        let mut window = WindowState::new();
        window.update(Addr::new(Segment::Buffer, 0), 1000, false);
        // Wrapping to the start of the same buffer overwrites old history.
        window.update(Addr::new(Segment::Buffer, 0), 300, true);
        assert_eq!(window.low_limit, WINDOW_START_INDEX + 300);
        assert_eq!(window.dict_limit, WINDOW_START_INDEX + 1000);
    }

    #[test]
    fn a_tiny_dictionary_is_dropped() {
        let mut window = WindowState::new();
        window.update(Addr::new(Segment::Buffer, 0), 5, false);
        window.update(Addr::new(Segment::Input(3), 0), 5, false);
        assert!(!window.has_ext_dict());
    }

    #[test]
    fn max_distance_raises_both_limits() {
        let mut window = WindowState::new();
        window.update(Addr::new(Segment::Input(1), 0), 10_000, false);
        let mut loaded = 0;
        let mut attached = true;
        window.enforce_max_dist(window.end_index(), 1024, &mut loaded, &mut attached);
        assert!(!attached);
        assert_eq!(window.low_limit, window.end_index() - 1024);
        assert_eq!(window.dict_limit, window.low_limit);
        window.clear();
        assert_eq!(window.low_limit, window.end_index());
    }

    #[test]
    fn overflow_correction_preserves_the_cycle_and_window() {
        let mut window = WindowState::new();
        let src = Addr::new(Segment::Input(1), 0);
        window.update(src, 16, false);
        let far = CURRENT_MAX + 4096;
        assert!(window.need_overflow_correction(16, 1 << 20, 0, far, far + 1));
        assert!(!window.need_overflow_correction(16, 1 << 20, 0, 100, 200));
        let correction = window.correct_overflow(16, 1 << 20, far);
        assert_eq!((far - correction) & 0xFFFF, far & 0xFFFF);
        assert!(far - correction >= 1 << 20);
        assert_eq!(window.low_limit, WINDOW_START_INDEX);
        assert_eq!(window.nb_overflow_corrections, 1);
        assert!(!window.can_overflow_correct(16, 1 << 20, 0, 10));
    }

    #[test]
    fn views_expose_prefix_and_dictionary_bytes() {
        let buffer: alloc::vec::Vec<u8> = (0..100).collect();
        let input = [200u8, 201, 202];
        let mut window = WindowState::new();
        window.update(Addr::new(Segment::Buffer, 0), 100, false);
        window.update(Addr::new(Segment::Input(9), 0), 3, false);
        let segments = Segments {
            buffer: &buffer,
            input: (&input, 9),
            dict: &[],
            cdict: None,
        };
        let view = window.view(&segments, window.end_index());
        assert_eq!(view.ext.byte(WINDOW_START_INDEX), 0);
        assert_eq!(view.prefix.byte(window.dict_limit), 200);
        let empty = WindowState::new();
        let view = empty.view(&segments, WINDOW_START_INDEX);
        assert_eq!(view.ext.end(), view.ext.start());
    }
}

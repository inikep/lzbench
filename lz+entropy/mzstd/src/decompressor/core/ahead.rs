//! Write prefetches for the output the next block will write.
//!
//! An output that starts uninitialized is zeroed a block's room at a time
//! before the block ([`super::block::BlockOutput`]): that is the first
//! touch of memory that is usually not cached, and a `memset` alone cannot
//! overlap its cache misses with any decoding. The reference avoids the
//! burst by decoding a block's literals into the end of its own output
//! (`ZSTD_in_dst`), so its literal loop fetches the output's lines while it
//! computes. Here the literal loops, which wait on their table lookups and
//! leave memory bandwidth unused, instead prefetch for writing the lines
//! the next block's room will cover, one per round: that room's zeroing,
//! or a caller's slice written in place, then finds them cached. The
//! sequence loop of a fixed output does the same once per sequence
//! ([`prefetch_write`]), for blocks whose literals no loop decodes.
//!
//! A prefetch is a hint: it never faults, reads or writes no data and
//! changes no result, so the addresses need not even be inside the
//! output.

use core::ptr;

/// Bytes of a cache line.
const LINE: usize = 64;
/// Cache lines one [`WriteAhead::step`] prefetches. Literal loops run a
/// round per 20 to 40 literals, and blocks at high levels carry few: one
/// line a round left most of the next room unfetched there, while four
/// measured no better than two (silesia.tar through `mzstd_decompress`,
/// levels 1, 8 and 19, 2026-11-08).
const LINES: usize = 2;

/// The addresses of the bytes a later block will write that are not yet
/// prefetched, as a cursor that advances a cache line at a time.
///
/// It holds raw addresses only to hand them to the prefetch instruction;
/// nothing is ever read or written through them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WriteAhead {
    next: *const u8,
    end: *const u8,
}

/// Prefetches for writing the cache line of the byte `at` bytes after the
/// start of `out`, wherever that is: inside `out` or in the same buffer
/// past its end.
///
/// The sequence loop calls it once per sequence with an offset a block
/// beyond its write position while the next block's whole room lies in
/// the output ([`WriteAhead::covers`]), so the lines that room covers
/// arrive while sequences execute; blocks whose literals are stored raw
/// run no literal loop that could prefetch for them. Otherwise it passes
/// its write position itself, a line it is about to write: an address
/// past the output may be unmapped, where a hint costs a page walk for
/// nothing.
#[inline(always)]
pub(crate) fn prefetch_write(out: &[u8], at: usize) {
    // Miri cannot run the hint's inline assembly; a hint changes nothing
    // it could check.
    #[cfg(not(miri))]
    branches::prefetch_write_data::<u8, 0>(out.as_ptr().wrapping_add(at));
    #[cfg(miri)]
    let _ = (out, at);
}

impl WriteAhead {
    /// Prefetches nothing.
    pub(crate) const NONE: Self = Self {
        next: ptr::null(),
        end: ptr::null(),
    };

    /// The bytes `start..end` of the allocation that starts at `base`;
    /// nothing when the range is empty.
    pub(crate) fn within(base: *const u8, start: usize, end: usize) -> Self {
        if start >= end {
            return Self::NONE;
        }
        Self {
            next: base.wrapping_add(start),
            end: base.wrapping_add(end),
        }
    }

    /// Whether at least `len` bytes are left to prefetch: asked before
    /// any step, whether the range holds a whole room of `len` bytes.
    #[inline(always)]
    pub(crate) fn covers(&self, len: usize) -> bool {
        (self.end as usize).saturating_sub(self.next as usize) >= len && len != 0
    }

    /// Prefetches the next [`LINES`] cache lines for writing, if any is
    /// left.
    #[inline(always)]
    pub(crate) fn step(&mut self) {
        if self.next < self.end {
            // Miri cannot run the hint's inline assembly; a hint changes
            // nothing it could check.
            #[cfg(not(miri))]
            for line in 0..LINES {
                branches::prefetch_write_data::<u8, 0>(self.next.wrapping_add(line * LINE));
            }
            self.next = self.next.wrapping_add(LINES * LINE);
        }
    }

    /// Bytes not yet prefetched.
    #[cfg(test)]
    fn remaining(&self) -> usize {
        (self.end as usize).saturating_sub(self.next as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_range_prefetches_nothing() {
        let bytes = [0u8; 4];
        assert_eq!(WriteAhead::within(bytes.as_ptr(), 3, 3), WriteAhead::NONE);
        assert_eq!(WriteAhead::within(bytes.as_ptr(), 4, 2), WriteAhead::NONE);
        let mut none = WriteAhead::NONE;
        none.step();
        assert_eq!(none, WriteAhead::NONE);
    }

    #[test]
    fn steps_advance_by_their_lines_and_stop_at_the_end() {
        let step = LINES * LINE;
        let bytes = [0u8; 3 * LINES * LINE];
        let mut ahead = WriteAhead::within(bytes.as_ptr(), 8, 8 + 2 * step + 1);
        assert_eq!(ahead.remaining(), 2 * step + 1);
        ahead.step();
        assert_eq!(ahead.remaining(), step + 1);
        ahead.step();
        ahead.step();
        assert_eq!(ahead.remaining(), 0);
        let done = ahead;
        ahead.step();
        assert_eq!(ahead, done);
    }

    #[test]
    fn a_write_prefetch_changes_nothing_wherever_it_points() {
        let bytes = [7u8; 16];
        for at in [0, 15, 16, 1 << 17, usize::MAX] {
            prefetch_write(&bytes, at);
            prefetch_write(&bytes[..0], at);
        }
        assert_eq!(bytes, [7; 16]);
    }

    #[test]
    fn a_range_covers_lengths_up_to_what_is_left_of_it() {
        let bytes = [0u8; 4 * LINES * LINE];
        let step = LINES * LINE;
        let mut ahead = WriteAhead::within(bytes.as_ptr(), 8, 8 + 2 * step);
        assert!(ahead.covers(1));
        assert!(ahead.covers(2 * step));
        assert!(!ahead.covers(2 * step + 1));
        assert!(!ahead.covers(0));
        ahead.step();
        assert!(ahead.covers(step));
        assert!(!ahead.covers(step + 1));
        assert!(!WriteAhead::NONE.covers(1));
        assert!(!WriteAhead::NONE.covers(0));
    }

    #[test]
    fn addresses_past_the_allocation_are_only_hints() {
        let bytes = [7u8; 16];
        let mut ahead = WriteAhead::within(bytes.as_ptr(), 0, 1 << 20);
        for _ in 0..64 {
            ahead.step();
        }
        assert_eq!(bytes, [7; 16]);
    }
}

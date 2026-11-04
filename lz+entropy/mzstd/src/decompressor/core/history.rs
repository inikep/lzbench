//! The streaming decoder's output buffer, which doubles as the window.
//!
//! Blocks are regenerated after the filled part of one buffer; the caller's
//! output is filled from its undelivered tail. The buffer is kept
//! initialized past the filled part, with room for a whole block and the
//! copies' overlength, so blocks decode in place. Before each block, once
//! everything is delivered, the buffer drops all but the last window of
//! history, but only after it has grown by at least another window (or
//! block) beyond it, so each byte is moved at most once on average and
//! history never has to wrap. Delivery only writes to the caller, so an
//! uninitialized destination serves as well as an initialized one.

use ::core::fmt;
use ::core::mem::MaybeUninit;
use alloc::vec::Vec;

use super::copy::WILDCOPY_OVERLENGTH;
use crate::decompressor::DecodeError;
use crate::shared::format::BLOCK_SIZE_MAX;

/// Regenerated output and the history blocks reference.
#[derive(Default)]
pub(crate) struct History {
    /// Initialized storage; `buffer[..filled]` is output.
    buffer: Vec<u8>,
    filled: usize,
    delivered: usize,
}

impl fmt::Debug for History {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("History")
            .field("len", &self.filled)
            .field("delivered", &self.delivered)
            .finish()
    }
}

/// Grows `buffer` to at least `len` initialized bytes: a first allocation
/// is zeroed by the allocator, a later one grows in place where it can.
pub(crate) fn grow_zeroed(buffer: &mut Vec<u8>, len: usize) -> Result<(), DecodeError> {
    if buffer.len() >= len {
        return Ok(());
    }
    if buffer.capacity() == 0 {
        let mut fresh = Vec::new();
        fresh
            .try_reserve_exact(len)
            .map_err(|_| DecodeError::AllocationFailed)?;
        fresh.resize(len, 0);
        *buffer = fresh;
        return Ok(());
    }
    buffer
        .try_reserve(len - buffer.len())
        .map_err(|_| DecodeError::AllocationFailed)?;
    buffer.resize(len, 0);
    Ok(())
}

/// A caller destination that delivered bytes are copied into.
///
/// Delivery only ever writes, so an uninitialized slice serves as well as
/// an initialized one; each gets its own instantiation of the streaming
/// state machine, so no call chooses a variant at run time.
pub(crate) trait Sink {
    /// Capacity of the destination.
    fn capacity(&self) -> usize;
    /// Writes `bytes` at `at`; `at + bytes.len()` never exceeds the capacity.
    fn put(&mut self, at: usize, bytes: &[u8]);
}

impl Sink for [u8] {
    #[inline]
    fn capacity(&self) -> usize {
        self.len()
    }

    #[inline]
    fn put(&mut self, at: usize, bytes: &[u8]) {
        self[at..at + bytes.len()].copy_from_slice(bytes);
    }
}

impl Sink for [MaybeUninit<u8>] {
    #[inline]
    fn capacity(&self) -> usize {
        self.len()
    }

    #[inline]
    fn put(&mut self, at: usize, bytes: &[u8]) {
        // `write_copy_of_slice` needs Rust 1.93; this compiles to a `memcpy`.
        for (slot, &byte) in self[at..at + bytes.len()].iter_mut().zip(bytes) {
            slot.write(byte);
        }
    }
}

impl History {
    /// Forgets everything at a frame start, keeping the allocation.
    pub(crate) fn reset(&mut self) {
        self.filled = 0;
        self.delivered = 0;
    }

    /// Bytes regenerated but not yet delivered.
    pub(crate) const fn pending(&self) -> usize {
        self.filled - self.delivered
    }

    /// Copies pending bytes into `output` from `at` on, returning how
    /// many; no byte past them is written.
    pub(crate) fn deliver<O: Sink + ?Sized>(&mut self, output: &mut O, at: usize) -> usize {
        let pending = &self.buffer[self.delivered..self.filled];
        let count = pending.len().min(output.capacity() - at);
        output.put(at, &pending[..count]);
        self.delivered += count;
        count
    }

    /// Prepares for the next block of a frame with `window` bytes of
    /// history and blocks of at most `block_max` bytes. Returns the buffer
    /// and where the block starts; [`Self::commit`] then records its end.
    /// All output must have been delivered.
    pub(crate) fn prepare(
        &mut self,
        window: u64,
        block_max: usize,
    ) -> Result<(&mut [u8], usize), DecodeError> {
        debug_assert_eq!(self.pending(), 0);
        let keep = usize::try_from(window).unwrap_or(usize::MAX);
        let slack = keep.max(BLOCK_SIZE_MAX);
        if self.filled > keep.saturating_add(slack) {
            let drop = self.filled - keep;
            self.buffer.copy_within(drop..self.filled, 0);
            self.filled = keep;
            self.delivered = keep;
        }
        grow_zeroed(
            &mut self.buffer,
            self.filled + block_max + WILDCOPY_OVERLENGTH,
        )?;
        Ok((&mut self.buffer, self.filled))
    }

    /// Records that the block [`Self::prepare`] started ends at `end`.
    pub(crate) fn commit(&mut self, end: usize) {
        debug_assert!(end >= self.filled && end <= self.buffer.len());
        self.filled = end;
    }

    /// Drops the buffer's allocation; nothing may be pending.
    pub(crate) fn release(&mut self) {
        *self = Self::default();
    }

    /// Heap bytes the buffer holds.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.buffer.capacity()
    }

    /// Bytes the buffer currently holds, for tests and accounting.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.filled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Appends `bytes` as one block.
    fn append(history: &mut History, window: u64, bytes: &[u8]) {
        let (buffer, start) = history.prepare(window, BLOCK_SIZE_MAX).expect("room");
        buffer[start..start + bytes.len()].copy_from_slice(bytes);
        history.commit(start + bytes.len());
    }

    #[test]
    fn delivery_drains_the_pending_tail() {
        let mut history = History::default();
        append(&mut history, 1024, b"hello");
        assert_eq!(history.pending(), 5);
        let mut out = [0u8; 3];
        assert_eq!(history.deliver(&mut out[..], 0), 3);
        assert_eq!(&out, b"hel");
        assert_eq!(history.deliver(&mut out[..], 1), 2);
        assert_eq!(&out, b"hlo");
        assert_eq!(history.deliver(&mut out[..], 0), 0);
        assert!(alloc::format!("{history:?}").contains("len: 5"));
    }

    #[test]
    fn history_is_compacted_only_past_a_window_of_slack() {
        let mut history = History::default();
        let window = BLOCK_SIZE_MAX as u64;
        let block = alloc::vec![7u8; BLOCK_SIZE_MAX];
        append(&mut history, window, &block);
        history.deliver(&mut alloc::vec![0; BLOCK_SIZE_MAX][..], 0);
        append(&mut history, window, &block);
        history.deliver(&mut alloc::vec![0; BLOCK_SIZE_MAX][..], 0);
        // Exactly window + slack: kept.
        history.prepare(window, BLOCK_SIZE_MAX).expect("room");
        assert_eq!(history.len(), 2 * BLOCK_SIZE_MAX);
        append(&mut history, window, &[1]);
        history.deliver(&mut [0][..], 0);
        history.prepare(window, BLOCK_SIZE_MAX).expect("room");
        assert_eq!(history.len(), BLOCK_SIZE_MAX);
        assert_eq!(history.pending(), 0);
        history.reset();
        assert_eq!(history.len(), 0);
    }

    #[test]
    fn uninitialized_delivery_writes_only_the_delivered_range() {
        let mut history = History::default();
        append(&mut history, 1024, b"hello");
        let mut out = [MaybeUninit::new(0xAAu8); 4];
        assert_eq!(history.deliver(&mut out[..], 1), 3);
        assert_eq!(history.pending(), 2);
        // SAFETY: every element was initialized with a sentinel.
        let bytes = out.map(|slot| unsafe { slot.assume_init() });
        assert_eq!(bytes, [0xAA, b'h', b'e', b'l']);
        let sink: &mut [MaybeUninit<u8>] = &mut out;
        assert_eq!(sink.capacity(), 4);
        assert_eq!(history.deliver(sink, 4), 0);
    }
}

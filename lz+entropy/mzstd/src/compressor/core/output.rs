//! Initialized destinations for frame compression. A caller-owned slice
//! retains its exact capacity; a Vec initializes only the space needed by
//! the next block and reuses its unwritten tail between blocks; uninitialized
//! memory is zeroed the same way, just ahead of each block.

use alloc::vec::Vec;
use core::mem::MaybeUninit;

use borrowed_buf::BorrowedBuf;

use super::error::CoreError;

pub(crate) trait Output {
    /// Writable tail at `position`, relative to the current call's
    /// destination. `needed` is a growth hint, not a capacity requirement:
    /// the slice adapter keeps the caller's exact limit, and the encoder
    /// performs its usual capacity checks on the returned slice.
    fn prepare(&mut self, position: usize, needed: usize) -> Result<&mut [u8], CoreError>;
}

pub(crate) struct SliceOutput<'a>(pub(crate) &'a mut [u8]);

impl Output for SliceOutput<'_> {
    #[inline]
    fn prepare(&mut self, position: usize, _needed: usize) -> Result<&mut [u8], CoreError> {
        self.0.get_mut(position..).ok_or(CoreError::DstSizeTooSmall)
    }
}

pub(crate) struct VecOutput<'a> {
    pub(crate) buffer: &'a mut Vec<u8>,
    pub(crate) start: usize,
    /// End of the frame's compressBound reservation, including the prefix.
    pub(crate) limit: usize,
}

impl Output for VecOutput<'_> {
    fn prepare(&mut self, position: usize, needed: usize) -> Result<&mut [u8], CoreError> {
        let position = self
            .start
            .checked_add(position)
            .ok_or(CoreError::DstSizeTooSmall)?;
        if position > self.limit {
            return Err(CoreError::DstSizeTooSmall);
        }
        let end = position.saturating_add(needed).min(self.limit);
        if end > self.buffer.len() {
            self.buffer.resize(end, 0);
        }
        Ok(&mut self.buffer[position..])
    }
}

/// Caller memory that may be uninitialized, as [`VecOutput`] with the
/// caller's length as its limit: the filled prefix grows by zeroing just
/// ahead of each block, so each byte is zeroed at most once and only where
/// the encoder may write.
pub(crate) struct UninitOutput<'a>(BorrowedBuf<'a, u8>);

impl<'a> From<&'a mut [MaybeUninit<u8>]> for UninitOutput<'a> {
    fn from(output: &'a mut [MaybeUninit<u8>]) -> Self {
        Self(BorrowedBuf::from(output))
    }
}

impl Output for UninitOutput<'_> {
    fn prepare(&mut self, position: usize, needed: usize) -> Result<&mut [u8], CoreError> {
        let capacity = self.0.capacity();
        if position > capacity {
            return Err(CoreError::DstSizeTooSmall);
        }
        let end = position.saturating_add(needed).min(capacity);
        if let Some(grow) = end.checked_sub(self.0.len()).filter(|&grow| grow > 0) {
            self.0.unfilled().append_fill(grow, 0);
        }
        Ok(&mut self.0.filled_mut()[position..])
    }
}

/// The caller's output of one streaming call.
pub(crate) trait Destination {
    /// The engine's view of the destination from some position on.
    type Tail<'a>: Output
    where
        Self: 'a;

    fn capacity(&self) -> usize;

    /// The destination from `at` on, for the engine to write blocks into.
    fn tail(&mut self, at: usize) -> Self::Tail<'_>;

    /// Copies `bytes` to `at`; the caller has checked that they fit.
    fn put(&mut self, at: usize, bytes: &[u8]);
}

impl Destination for [u8] {
    type Tail<'a> = SliceOutput<'a>;

    #[inline]
    fn capacity(&self) -> usize {
        self.len()
    }

    #[inline]
    fn tail(&mut self, at: usize) -> SliceOutput<'_> {
        SliceOutput(&mut self[at..])
    }

    #[inline]
    fn put(&mut self, at: usize, bytes: &[u8]) {
        self[at..at + bytes.len()].copy_from_slice(bytes);
    }
}

impl Destination for [MaybeUninit<u8>] {
    type Tail<'a> = UninitOutput<'a>;

    #[inline]
    fn capacity(&self) -> usize {
        self.len()
    }

    #[inline]
    fn tail(&mut self, at: usize) -> UninitOutput<'_> {
        UninitOutput::from(&mut self[at..])
    }

    #[inline]
    fn put(&mut self, at: usize, bytes: &[u8]) {
        // `write_copy_of_slice` needs Rust 1.93; this compiles to a `memcpy`.
        for (slot, &byte) in self[at..at + bytes.len()].iter_mut().zip(bytes) {
            slot.write(byte);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    /// Reads back memory a test initialized in full.
    fn bytes(output: &[MaybeUninit<u8>]) -> Vec<u8> {
        // SAFETY: each test initializes every element before reading.
        output
            .iter()
            .map(|slot| unsafe { slot.assume_init() })
            .collect()
    }

    #[test]
    fn uninit_output_zeroes_only_up_to_the_hint() {
        let mut storage = [MaybeUninit::new(0xAA); 16];
        let mut output = UninitOutput::from(&mut storage[..]);
        let tail = output.prepare(2, 4).expect("in range");
        assert_eq!(tail, &[0; 4]);
        tail.copy_from_slice(b"abcd");
        assert_eq!(
            bytes(&storage),
            b"\0\0abcd\xAA\xAA\xAA\xAA\xAA\xAA\xAA\xAA\xAA\xAA"
        );
    }

    #[test]
    fn uninit_output_keeps_written_bytes_when_it_grows() {
        let mut storage = [MaybeUninit::new(0xAA); 8];
        let mut output = UninitOutput::from(&mut storage[..]);
        output
            .prepare(0, 3)
            .expect("in range")
            .copy_from_slice(b"abc");
        let tail = output.prepare(1, 4).expect("in range");
        assert_eq!(tail, b"bc\0\0");
        // A smaller hint still sees everything zeroed so far.
        assert_eq!(output.prepare(4, 0).expect("in range"), &[0]);
        assert_eq!(bytes(&storage[..5]), b"abc\0\0");
    }

    #[test]
    fn uninit_output_stops_at_its_capacity() {
        let mut storage = [MaybeUninit::uninit(); 6];
        let mut output = UninitOutput::from(&mut storage[..]);
        assert_eq!(output.prepare(4, usize::MAX).expect("in range"), &[0, 0]);
        assert_eq!(output.prepare(6, 1).expect("at the end"), &[] as &[u8]);
        assert_eq!(output.prepare(7, 1), Err(CoreError::DstSizeTooSmall));
    }

    #[test]
    fn slice_output_returns_the_exact_tail() {
        let mut storage = [7u8; 4];
        let mut output = SliceOutput(&mut storage);
        assert_eq!(output.prepare(1, 0).expect("in range"), &[7, 7, 7]);
        assert_eq!(output.prepare(5, 0), Err(CoreError::DstSizeTooSmall));
    }

    #[test]
    fn destinations_copy_and_expose_their_tails() {
        let mut initialized = [0u8; 5];
        let destination: &mut [u8] = &mut initialized;
        assert_eq!(destination.capacity(), 5);
        destination.put(1, b"xy");
        assert_eq!(
            destination.tail(3).prepare(0, 1).expect("in range"),
            &[0, 0]
        );
        assert_eq!(initialized, *b"\0xy\0\0");

        let mut uninit = [MaybeUninit::new(0xAA); 5];
        let destination: &mut [MaybeUninit<u8>] = &mut uninit;
        assert_eq!(destination.capacity(), 5);
        destination.put(1, b"xy");
        assert_eq!(destination.tail(4).prepare(0, 9).expect("in range"), &[0]);
        assert_eq!(bytes(&uninit), b"\xAAxy\xAA\0");
    }

    #[test]
    fn vec_output_grows_to_the_hint_within_its_limit() {
        let mut buffer = vec![1u8];
        let mut output = VecOutput {
            buffer: &mut buffer,
            start: 1,
            limit: 4,
        };
        assert_eq!(output.prepare(1, 10).expect("in range"), &[0, 0]);
        assert_eq!(output.prepare(4, 0), Err(CoreError::DstSizeTooSmall));
        assert_eq!(buffer, [1, 0, 0, 0]);
    }
}

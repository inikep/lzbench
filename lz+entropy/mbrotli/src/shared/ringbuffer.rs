//! Sliding window over the input, with the reference's exact layout.
//!
//! Ports `RingBuffer` from `c/enc/ringbuffer.h` and the seven-byte clearing
//! that `CopyInputToRingBuffer` performs in `c/enc/encode.c` of the pinned
//! reference (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! The layout matters for the emitted bytes, not just for correctness. Match
//! finding reads whole words past the current position, and the reference
//! defines exactly what those bytes are: a copy of the start of the window in
//! the tail, zeros in the seven-byte margin, and the sentinel `241` at the
//! first tail byte until a lap writes over it. Reproducing the same filler is
//! what makes the encoder deterministic on data it never actually copied.

use alloc::vec::Vec;

/// Bytes of margin past the window, so eight-byte loads always have data.
const SLACK_FOR_EIGHT_BYTE_HASHING: usize = 7;

/// Bytes reserved before the window for the two wrap-around copies.
const HEAD_ROOM: usize = 2;

/// Sentinel the reference leaves at the first tail byte.
const TAIL_SENTINEL: u8 = 241;

/// The window a search runs over.
#[derive(Copy, Clone)]
pub(crate) struct Window<'a> {
    /// The ring buffer holding the input seen so far.
    pub(crate) data: &'a [u8],
    /// Mask that turns an absolute position into a buffer index.
    pub(crate) mask: usize,
}

/// The stretch of input one call processes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct BlockSpan {
    /// Wrapped position the stretch starts at.
    pub(crate) position: u32,
    /// Number of bytes in the stretch.
    pub(crate) bytes: u32,
}

/// Circular window holding the input the encoder can still refer back to.
pub(crate) struct RingBuffer {
    size: usize,
    mask: usize,
    tail_size: usize,
    total_size: usize,
    cur_size: usize,
    // The reference omits tail mirroring for a short first write.
    tail_start: usize,
    pos: u32,
    /// Bytes the caller expects the stream to hold, zero when unknown.
    expected_input: usize,
    data: Vec<u8>,
}

impl RingBuffer {
    /// Creates an empty window of `1 << rb_bits` bytes (`RingBufferSetup`).
    ///
    /// `lgblock` sizes the tail: the copy of the window head that lets a match
    /// finder read a whole word past the wrap point without a branch.
    pub(crate) fn new(rb_bits: usize, lgblock: usize) -> Self {
        let size = 1usize << rb_bits;
        let tail_size = 1usize << lgblock;
        Self {
            size,
            mask: size - 1,
            tail_size,
            total_size: size + tail_size,
            cur_size: 0,
            tail_start: 0,
            pos: 0,
            expected_input: 0,
            data: Vec::new(),
        }
    }

    /// Tells the window how many bytes the stream is expected to hold.
    ///
    /// The prefix a stream writes before its first wrap grows with every
    /// block; knowing the total up front lets the first growth reserve all
    /// of it, instead of reallocating and copying the prefix at every
    /// doubling. Only capacity changes: the layout is the same with or
    /// without a hint, and the reservation never exceeds the full window, so
    /// a wrong hint costs at most what a long stream allocates anyway.
    pub(crate) const fn expect_input(&mut self, bytes: usize) {
        self.expected_input = bytes;
    }

    /// Returns the mask that turns an absolute position into a buffer index.
    pub(crate) const fn mask(&self) -> usize {
        self.mask
    }

    /// Returns the window contents, indexed by masked position.
    ///
    /// The slice is longer than the window: it also holds the tail copy and the
    /// margin, so a caller may read a whole word starting at any valid index.
    pub(crate) fn buffer(&self) -> &[u8] {
        match self.data.get(HEAD_ROOM..) {
            Some(buffer) => buffer,
            None => &[],
        }
    }

    /// Returns the bytes this window keeps allocated.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.data.capacity()
    }

    /// Returns whether any input has been written yet.
    pub(crate) const fn is_allocated(&self) -> bool {
        self.cur_size != 0
    }

    /// Restores the window to the state its constructor left it in.
    ///
    /// The bytes are left where they are rather than wiped. Nothing can read
    /// them: a backward reference is bounded by the distance to the start of
    /// the stream, so the next stream never looks further back than it has
    /// written, and `write` re-establishes the head bytes, the tail mirror and
    /// the sentinel while `clear_margin` re-zeroes the margin. Wiping a window
    /// that can be mebibytes wide would cost more than the allocation reuse
    /// saves.
    pub(crate) fn reset(&mut self) {
        self.cur_size = 0;
        self.tail_start = 0;
        self.pos = 0;
    }

    /// Grows the backing storage to `buflen` bytes (`RingBufferInitBuffer`).
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    fn init_buffer(&mut self, buflen: usize) {
        // Previously initialized capacity remains usable after reset. History
        // outside the written range is never a valid match; write establishes
        // the sentinel/tail and clear_margin establishes the lookahead bytes.
        self.data
            .resize(HEAD_ROOM + buflen + SLACK_FOR_EIGHT_BYTE_HASHING, 0);
        self.cur_size = buflen;
        // The two head bytes and the margin are zero, exactly as the reference
        // leaves them; `vec!` already provided that for a fresh allocation and
        // the copy above never reaches past `keep`.
        for index in 0..SLACK_FOR_EIGHT_BYTE_HASHING {
            if let Some(byte) = self.data.get_mut(HEAD_ROOM + self.cur_size + index) {
                *byte = 0;
            }
        }
        if let Some(byte) = self.data.get_mut(0) {
            *byte = 0;
        }
        if let Some(byte) = self.data.get_mut(1) {
            *byte = 0;
        }
    }

    /// Appends `bytes` to a window still holding only the written prefix,
    /// which then ends at `end`.
    ///
    /// Leaves exactly the layout `init_buffer(end)` followed by a copy would:
    /// the two zero head bytes, the prefix, and the zeroed margin. The bytes
    /// are appended rather than copied over a zero-filled extension, so each
    /// byte of the window is written once instead of twice.
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    fn append_prefix(&mut self, bytes: &[u8], end: usize) {
        let start = HEAD_ROOM + self.pos as usize;
        // Capacity kept across a reset may still hold a longer stream; the
        // margin of the previous write is overwritten by the new bytes.
        self.data.truncate(start);
        // One growth for all three appends below, and for the rest of the
        // expected stream: a stream that will fill the window materializes
        // the full layout next, so that is what it reserves.
        let expected = if self.expected_input >= self.size.saturating_sub(8) {
            self.total_size
        } else {
            self.expected_input.max(end)
        };
        self.data
            .reserve(HEAD_ROOM + expected + SLACK_FOR_EIGHT_BYTE_HASHING - self.data.len());
        self.data.resize(start, 0);
        self.data.extend_from_slice(bytes);
        self.data
            .extend_from_slice(&[0; SLACK_FOR_EIGHT_BYTE_HASHING]);
        self.cur_size = end;
        if let Some(head) = self.data.get_mut(..HEAD_ROOM) {
            head.fill(0);
        }
    }

    /// Writes `index`-th byte of the window, ignoring an out-of-range index.
    fn set(&mut self, index: usize, value: u8) {
        if let Some(byte) = self.data.get_mut(HEAD_ROOM + index) {
            *byte = value;
        }
    }

    /// Copies the head of `bytes` into the tail mirror (`RingBufferWriteTail`).
    fn write_tail(&mut self, bytes: &[u8]) {
        let masked_pos = (self.pos as usize) & self.mask;
        if masked_pos >= self.tail_size {
            return;
        }
        let count = bytes.len().min(self.tail_size - masked_pos);
        let start = HEAD_ROOM + self.size + masked_pos;
        if let Some(target) = self.data.get_mut(start..start + count)
            && let Some(source) = bytes.get(..count)
        {
            target.copy_from_slice(source);
        }
    }

    /// Appends `bytes` to the window (`RingBufferWrite`).
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn write(&mut self, bytes: &[u8]) {
        let end = self.pos as usize + bytes.len();
        if self.cur_size < self.total_size && end < self.size.saturating_sub(8) {
            // Before the first wrap, only the written prefix and its seven
            // lookahead bytes can be read. Keep the logical mask unchanged.
            // Materialize the full layout before reaching the last two bytes
            // or reading into the tail, whose sentinel must remain exact.
            if self.pos == 0 {
                self.tail_start = if bytes.len() < self.tail_size {
                    bytes.len()
                } else {
                    0
                };
            }
            self.append_prefix(bytes, end);
            self.pos = end as u32;
            return;
        }
        if self.cur_size < self.total_size {
            let mirrored_end = (self.pos as usize).min(self.tail_size);
            self.init_buffer(self.total_size);
            self.set(self.size - 2, 0);
            self.set(self.size - 1, 0);
            self.set(self.size, TAIL_SENTINEL);
            // Replay the tail copies deferred while storing only a prefix.
            // A short first write left this range untouched in the reference.
            if self.tail_start < mirrored_end {
                self.data.copy_within(
                    HEAD_ROOM + self.tail_start..HEAD_ROOM + mirrored_end,
                    HEAD_ROOM + self.size + self.tail_start,
                );
            }
        }

        let masked_pos = (self.pos as usize) & self.mask;
        self.write_tail(bytes);
        if masked_pos + bytes.len() <= self.size {
            let start = HEAD_ROOM + masked_pos;
            if let Some(target) = self.data.get_mut(start..start + bytes.len()) {
                target.copy_from_slice(bytes);
            }
        } else {
            let head = (self.total_size - masked_pos).min(bytes.len());
            let start = HEAD_ROOM + masked_pos;
            if let Some(target) = self.data.get_mut(start..start + head)
                && let Some(source) = bytes.get(..head)
            {
                target.copy_from_slice(source);
            }
            let wrapped = self.size - masked_pos;
            if let Some(source) = bytes.get(wrapped..) {
                let count = source.len();
                if let Some(target) = self.data.get_mut(HEAD_ROOM..HEAD_ROOM + count) {
                    target.copy_from_slice(source);
                }
            }
        }

        let not_first_lap = (self.pos & (1u32 << 31)) != 0;
        let pos_mask = (1u32 << 31) - 1;
        let last_but_one = self.buffer().get(self.size - 2).copied().unwrap_or(0);
        let last = self.buffer().get(self.size - 1).copied().unwrap_or(0);
        if let Some(byte) = self.data.get_mut(0) {
            *byte = last_but_one;
        }
        if let Some(byte) = self.data.get_mut(1) {
            *byte = last;
        }
        self.pos = (self.pos & pos_mask) + ((bytes.len() as u32) & pos_mask);
        if not_first_lap {
            self.pos |= 1u32 << 31;
        }
    }

    /// Clears the seven bytes that follow the written data on the first lap.
    ///
    /// Hashing loads whole words, so without this the hash of the last few
    /// positions would depend on memory the encoder never wrote.
    pub(crate) fn clear_margin(&mut self) {
        if self.pos as usize > self.mask {
            return;
        }
        let start = HEAD_ROOM + self.pos as usize;
        let end = (start + SLACK_FOR_EIGHT_BYTE_HASHING).min(self.data.len());
        if let Some(target) = self.data.get_mut(start..end) {
            target.fill(0);
        }
    }
}

/// Wraps a 64-bit input position into the 32-bit space positions are stored in.
///
/// Mirrors `WrapPosition`: the first three gibibytes are contiguous, and after
/// that positions alternate between two gibibyte-wide halves so that the
/// "already lapped" property survives the truncation.
pub(crate) const fn wrap_position(position: u64) -> u32 {
    let result = position as u32;
    let gb = position >> 30;
    if gb > 2 {
        (result & ((1u32 << 30) - 1)) | ((((gb - 1) & 1) as u32 + 1) << 30)
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a window the way `ComputeRbBits` would for `lgwin` and `lgblock`.
    fn ring(lgwin: usize, lgblock: usize) -> RingBuffer {
        RingBuffer::new(1 + lgwin.max(lgblock), lgblock)
    }

    #[test]
    fn a_short_first_write_only_allocates_what_it_holds() {
        let mut rb = ring(16, 16);
        assert!(!rb.is_allocated());
        rb.write(b"hello");
        assert!(rb.is_allocated());
        assert_eq!(&rb.buffer()[..5], b"hello");
        // The margin is zeroed so eight-byte loads are defined.
        assert_eq!(&rb.buffer()[5..12], &[0u8; 7]);
    }

    #[test]
    fn writes_before_the_first_wrap_only_allocate_the_written_prefix() {
        let mut rb = ring(16, 16);
        let payload = vec![7u8; 1 << 16];
        rb.write(&payload);
        assert_eq!(rb.cur_size, payload.len());
        assert_eq!(rb.buffer()[0], 7);
        rb.write(&[8; 16]);
        assert_eq!(rb.cur_size, payload.len() + 16);
        assert_eq!(&rb.buffer()[payload.len()..payload.len() + 16], &[8; 16]);
        assert_eq!(rb.mask(), rb.size - 1);
    }

    #[test]
    fn an_expected_input_is_reserved_by_the_first_write_without_changing_the_layout() {
        let blocks: Vec<Vec<u8>> = (0..4u8).map(|block| vec![block + 1; 1000]).collect();
        let mut plain = ring(16, 10);
        let mut hinted = ring(16, 10);
        hinted.expect_input(4000);
        hinted.write(&blocks[0]);
        let reserved = hinted.retained_bytes();
        assert!(reserved >= HEAD_ROOM + 4000 + SLACK_FOR_EIGHT_BYTE_HASHING);
        plain.write(&blocks[0]);
        for block in &blocks[1..] {
            plain.write(block);
            hinted.write(block);
        }
        assert_eq!(hinted.retained_bytes(), reserved);
        assert_eq!(hinted.data, plain.data);
        assert_eq!(hinted.cur_size, plain.cur_size);
    }

    #[test]
    fn an_expected_input_that_fills_the_window_reserves_the_full_layout() {
        let mut rb = ring(10, 8);
        rb.expect_input(1 << 20);
        rb.write(&[1; 16]);
        assert!(rb.retained_bytes() >= HEAD_ROOM + rb.total_size + SLACK_FOR_EIGHT_BYTE_HASHING);
        assert_eq!(&rb.buffer()[..16], &[1; 16]);
        assert_eq!(&rb.buffer()[16..23], &[0; 7]);
    }

    #[test]
    fn the_tail_sentinel_survives_until_a_lap_writes_over_it() {
        let mut rb = ring(16, 16);
        // A write that starts past the tail leaves the sentinel in place.
        let mut payload = vec![1u8; 1 << 16];
        payload.truncate(1 << 16);
        rb.write(&vec![2u8; rb.size]);
        assert_eq!(rb.buffer()[rb.size], 2);
    }

    #[test]
    fn materializing_the_tail_preserves_previous_writes_and_the_short_write_sentinel() {
        for first in [3, 16] {
            let mut rb = RingBuffer::new(6, 4);
            rb.write(&vec![2; first]);
            rb.write(&vec![3; 52 - first]);
            rb.write(&[4; 12]);
            assert_eq!(&rb.buffer()[..first], vec![2; first]);
            assert_eq!(&rb.buffer()[52..64], &[4; 12]);
            assert_eq!(&rb.data[..2], &[4; 2]);
            assert_eq!(rb.buffer()[64], if first < 16 { TAIL_SENTINEL } else { 2 });
            assert_eq!(
                &rb.buffer()[64 + first.min(16)..80],
                &rb.buffer()[first.min(16)..16]
            );
            rb.write(&[9; 8]);
            assert_eq!(&rb.buffer()[..8], &[9; 8]);
            assert_eq!(&rb.buffer()[64..72], &[9; 8]);
            rb.reset();
            rb.write(&[5; 16]);
            rb.clear_margin();
            assert_eq!(&rb.buffer()[..16], &[5; 16]);
            assert_eq!(&rb.buffer()[16..23], &[0; 7]);
        }
    }

    #[test]
    fn writes_wrap_around_and_mirror_into_the_tail() {
        let mut rb = ring(10, 16);
        let window = rb.size;
        rb.write(&vec![1u8; window - 4]);
        rb.write(&[9, 9, 9, 9, 8, 8, 8, 8]);
        assert_eq!(&rb.buffer()[window - 4..window], &[9, 9, 9, 9]);
        assert_eq!(&rb.buffer()[..4], &[8, 8, 8, 8]);
        assert_eq!(&rb.buffer()[window..window + 4], &[8, 8, 8, 8]);
    }

    #[test]
    fn the_head_bytes_mirror_the_end_of_the_window() {
        let mut rb = ring(10, 16);
        let window = rb.size;
        rb.write(&vec![5u8; window]);
        assert_eq!(rb.data[0], 5);
        assert_eq!(rb.data[1], 5);
        assert_eq!(rb.buffer()[window - 1], 5);
    }

    #[test]
    fn clearing_the_margin_only_touches_the_first_lap() {
        let mut rb = ring(10, 16);
        rb.write(&[3u8; 8]);
        rb.clear_margin();
        assert_eq!(&rb.buffer()[8..15], &[0u8; 7]);
    }

    #[test]
    fn position_wrapping_keeps_the_lap_parity() {
        assert_eq!(wrap_position(0), 0);
        assert_eq!(wrap_position(1234), 1234);
        assert_eq!(wrap_position((1u64 << 30) - 1), (1 << 30) - 1);
        assert_eq!(wrap_position(3u64 << 30), 1 << 30);
        assert_eq!(wrap_position(4u64 << 30), 2 << 30);
        assert_eq!(wrap_position(5u64 << 30), 1 << 30);
        assert_eq!(wrap_position((3u64 << 30) + 17), (1 << 30) + 17);
    }
}

//! Backward bit readers for Huffman and FSE bitstreams, and the forward
//! reader for table descriptions.
//!
//! Zstandard writes these streams forward and ends them with a single `1` bit
//! followed by zero padding up to the byte boundary, so the decoder starts at
//! the last byte, skips the padding and the marker bit, and reads towards the
//! first byte. A stream is valid only when every bit is consumed exactly.
//!
//! [`BitContainer`] is the hot-path reader, a port of the reference's
//! `BIT_DStream_t`: a 64-bit container that callers refill with
//! [`BitContainer::reload`] at the points the reference does, so the bits
//! they read are the reference's. Reading past the first byte is not checked
//! per read; it leaves the consumed count above 64, which
//! [`BitContainer::is_exhausted`] then rejects, exactly as the reference's
//! `BIT_endOfDStream` does, and which [`BitContainer::overflowed`] reports
//! as the reference's FSE decoders test it.

use crate::shared::format::highbit32;

/// Loads up to eight bytes from `src[start..]` as a little-endian word,
/// padding with zeros past the end.
fn load_word(src: &[u8], start: usize) -> u64 {
    let tail = src.get(start..).unwrap_or(&[]);
    if let Some(word) = tail.first_chunk::<8>() {
        return u64::from_le_bytes(*word);
    }
    let mut word = [0u8; 8];
    word[..tail.len()].copy_from_slice(tail);
    u64::from_le_bytes(word)
}

/// Width of the bit container in bits.
const CONTAINER_BITS: u32 = 64;

/// Loads the eight bytes at `src[pos..]` as a little-endian word; zero when
/// they are not all there, which callers never ask for.
///
/// Streams are block parts, far below 2^32 bytes, so masking `pos` to 32
/// bits changes nothing but lets the compiler see that `pos + 8` cannot
/// overflow, leaving one compare in front of the load.
#[inline(always)]
fn read64(src: &[u8], pos: usize) -> u64 {
    let pos = pos & u32::MAX as usize;
    match src.get(pos..pos + 8) {
        Some(word) => u64::from_le_bytes(word.try_into().unwrap_or([0; 8])),
        None => 0,
    }
}

/// A backward reader with a 64-bit container, refilled on request.
///
/// The container holds the stream bytes `[pos, pos + 8)`; its low `avail`
/// bits are unread. Counting the unread bits rather than the read ones
/// makes a read a subtraction, a shift and a mask, with no negation. Reads
/// past the start of the stream make `avail` wrap below zero, where it
/// stays: [`BitContainer::is_exhausted`] then fails. Streams shorter than eight bytes are
/// loaded once, with their missing high bytes counted as consumed. The
/// stream slice is passed to each reload rather than held, so the reader is
/// a small `Copy` value the hot loops keep in registers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BitContainer {
    container: u64,
    avail: u32,
    pos: usize,
}

impl BitContainer {
    /// Opens `src`, or returns `None` when it is empty or its last byte holds
    /// no end marker.
    #[inline]
    pub(crate) fn new(src: &[u8]) -> Option<Self> {
        let &last = src.last()?;
        if last == 0 {
            return None;
        }
        let marker = 8 - highbit32(u32::from(last));
        if src.len() >= 8 {
            let pos = src.len() - 8;
            return Some(Self {
                container: read64(src, pos),
                avail: CONTAINER_BITS - marker,
                pos,
            });
        }
        let container = src
            .iter()
            .rev()
            .fold(0u64, |word, &byte| (word << 8) | u64::from(byte));
        Some(Self {
            container,
            avail: CONTAINER_BITS - marker - (8 - src.len() as u32) * 8,
            pos: 0,
        })
    }

    /// The next `count` bits without consuming them; `count <= 56`, and may
    /// be zero.
    #[inline(always)]
    pub(crate) const fn look(&self, count: u32) -> u64 {
        // The reference's `BIT_getMiddleBits`: a shift and, with BMI2, a
        // `bzhi`. A zero count masks everything away whatever the shift.
        let start = self.avail.wrapping_sub(count);
        (self.container >> (start & 63)) & ((1u64 << count) - 1)
    }

    /// [`Self::look`] for `1 <= count <= 56`, one shift shorter.
    #[inline(always)]
    pub(crate) const fn look_nonzero(&self, count: u32) -> u64 {
        let consumed = CONTAINER_BITS.wrapping_sub(self.avail);
        (self.container << (consumed & 63)) >> ((CONTAINER_BITS - count) & 63)
    }

    /// Consumes `count` bits.
    #[inline(always)]
    pub(crate) const fn skip(&mut self, count: u32) {
        self.avail = self.avail.wrapping_sub(count);
    }

    /// Reads and consumes `count` bits; `count <= 56`, and may be zero.
    #[inline(always)]
    pub(crate) const fn read(&mut self, count: u32) -> u64 {
        let value = self.look(count);
        self.skip(count);
        value
    }

    /// Whether the fast refill applies: the container can move a whole
    /// word towards the start.
    #[inline(always)]
    pub(crate) const fn can_reload_fast(&self) -> bool {
        self.pos >= 8
    }

    /// Refills the container when [`Self::can_reload_fast`] holds; the
    /// reference's `BIT_reloadDStreamFast`.
    #[inline(always)]
    pub(crate) fn reload_fast(&mut self, src: &[u8]) {
        debug_assert!(self.can_reload_fast() && self.avail <= CONTAINER_BITS);
        let bytes = (CONTAINER_BITS - self.avail) >> 3;
        self.pos -= bytes as usize;
        self.avail += bytes * 8;
        self.container = read64(src, self.pos);
    }

    /// Refills the container as far as the stream allows; the reference's
    /// `BIT_reloadDStream`. Returns whether at least 57 bits are now
    /// available (`BIT_DStream_unfinished`).
    #[inline(always)]
    pub(crate) fn reload(&mut self, src: &[u8]) -> bool {
        // While a whole word can be refilled, every refill so far was a
        // full one, and callers read at most 57 bits between refills, so
        // nothing can have been read past the start yet.
        if self.can_reload_fast() {
            self.reload_fast(src);
            return true;
        }
        if self.pos == 0 || self.avail > CONTAINER_BITS {
            // At the start there is nothing to refill; overflow is permanent.
            return false;
        }
        let (next, full) = self.reload_tail(src);
        *self = next;
        full
    }

    /// The cautious end of [`Self::reload`], near the start of the stream.
    ///
    /// Out of line and by value: a reference would make the hot loops keep
    /// the container in memory rather than in registers.
    #[cold]
    #[inline(never)]
    fn reload_tail(mut self, src: &[u8]) -> (Self, bool) {
        if self.pos == 0 {
            return (self, false);
        }
        let wanted = ((CONTAINER_BITS - self.avail) >> 3) as usize;
        let step = wanted.min(self.pos);
        self.pos -= step;
        self.avail += (step as u32) * 8;
        self.container = read64(src, self.pos);
        (self, step == wanted)
    }

    /// Whether every bit was consumed, and no more.
    #[inline]
    pub(crate) const fn is_exhausted(&self) -> bool {
        self.pos == 0 && self.avail == 0
    }

    /// Whether more bits were consumed than the stream holds: the
    /// reference's `BIT_DStream_overflow`.
    #[inline(always)]
    pub(crate) const fn overflowed(&self) -> bool {
        self.avail > CONTAINER_BITS
    }
}

/// A backward reader for the Huffman fast loops, the reference's
/// `HUF_DecompressFastArgs` streams: the container keeps unread bits at its
/// top and a sentinel `1` below them, shifted up with every read, so the
/// consumed count is implicit — the sentinel's position — and a lookup is a
/// single shift.
///
/// `pos` indexes the buffer that holds all four streams. Each refill moves
/// back by whole consumed bytes; callers bound the rounds so that `pos`
/// stays at least seven bytes into its stream, and read at most 57 bits
/// between refills, which keeps the sentinel in the container.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sentinel {
    bits: u64,
    pos: usize,
}

impl Sentinel {
    /// Converts a container refilled with [`BitContainer::reload_fast`],
    /// at most seven bits consumed, over the stream that starts at `start`
    /// in `src`.
    #[inline(always)]
    pub(crate) fn from_container(bits: BitContainer, src: &[u8], start: usize) -> Self {
        let consumed = CONTAINER_BITS - bits.avail;
        debug_assert!(consumed <= 7);
        let pos = start + bits.pos;
        Self {
            bits: (read64(src, pos) | 1) << (consumed & 7),
            pos,
        }
    }

    /// The container again, for the stream that starts at `start` in
    /// `src`.
    #[inline(always)]
    pub(crate) fn into_container(self, src: &[u8], start: usize) -> BitContainer {
        // A fast loop may load a word that starts in the preceding stream.
        // Move its base to this stream's first byte and remove those prefix
        // bits from the available count. A consumed prefix wraps avail above
        // 64, preserving the cautious reader's permanent overflow state.
        let pos = self.pos.max(start);
        let prefix_bits = ((pos - self.pos) as u32).wrapping_mul(8);
        BitContainer {
            container: read64(src, pos),
            avail: (CONTAINER_BITS - self.bits.trailing_zeros()).wrapping_sub(prefix_bits),
            pos: pos - start,
        }
    }

    /// Bytes of the stream left below the container's start.
    #[inline(always)]
    pub(crate) const fn remaining(&self, start: usize) -> usize {
        self.pos - start
    }

    /// Moves back by the consumed whole bytes and refills, for callers that
    /// keep the position at most `limit`, itself at most `src.len() - 8`.
    /// The position is clamped to `limit`, which never changes it but shows
    /// the word in bounds, so the fast loops refill without a branch.
    /// Returns `false`, leaving the sentinel as it was, only when `limit`
    /// is out of range.
    #[inline(always)]
    pub(crate) fn reload_within(&mut self, src: &[u8], limit: usize) -> bool {
        let consumed = self.bits.trailing_zeros();
        let pos = (self.pos - (consumed >> 3) as usize).min(limit);
        let Some(word) = src.get(pos..).and_then(<[u8]>::first_chunk::<8>) else {
            return false;
        };
        self.pos = pos;
        self.bits = (u64::from_le_bytes(*word) | 1) << (consumed & 7);
        true
    }

    /// The next `64 - shift` bits.
    #[inline(always)]
    pub(crate) const fn peek(&self, shift: u32) -> usize {
        (self.bits >> shift) as usize
    }

    /// Consumes `count` bits.
    #[inline(always)]
    pub(crate) const fn skip(&mut self, count: u32) {
        self.bits <<= count & 63;
    }
}

/// A forward little-endian bit reader for FSE table descriptions.
///
/// Bits past the end read as zeros; [`ForwardBitReader::bytes_used`] tells
/// the caller how many bytes the read prefix spans so it can reject
/// descriptions that ran past their input.
#[derive(Clone, Debug)]
pub(crate) struct ForwardBitReader<'a> {
    src: &'a [u8],
    position: usize,
}

impl<'a> ForwardBitReader<'a> {
    /// Starts reading at the first bit of `src`.
    pub(crate) const fn new(src: &'a [u8]) -> Self {
        Self { src, position: 0 }
    }

    /// The next `count` bits, lowest first; `count <= 32`.
    pub(crate) fn peek(&self, count: u32) -> u32 {
        debug_assert!(count <= 32);
        let word = load_word(self.src, self.position / 8) >> (self.position % 8);
        (word & ((1u64 << count) - 1)) as u32
    }

    /// Consumes `count` bits.
    pub(crate) const fn consume(&mut self, count: u32) {
        self.position += count as usize;
    }

    /// Reads and consumes `count` bits; `count <= 32`.
    pub(crate) fn read(&mut self, count: u32) -> u32 {
        let value = self.peek(count);
        self.consume(count);
        value
    }

    /// Bytes spanned by the bits read so far, rounded up.
    pub(crate) const fn bytes_used(&self) -> usize {
        self.position.div_ceil(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_unterminated_streams_are_rejected() {
        assert!(BitContainer::new(&[]).is_none());
        assert!(BitContainer::new(&[0x12, 0x00]).is_none());
    }

    #[test]
    fn the_marker_bit_and_padding_are_skipped() {
        // 0b0000_0101: marker at bit 2, two payload bits `01`.
        let mut bits = BitContainer::new(&[0b0000_0101]).expect("terminated");
        assert_eq!(bits.read(1), 0);
        assert_eq!(bits.read(1), 1);
        assert!(bits.is_exhausted());
        assert!(!bits.overflowed());
    }

    #[test]
    fn reads_cross_byte_boundaries_from_the_end() {
        // Bits from the top: byte 1 = 0b1_0110011 (marker then 0110011),
        // byte 0 = 0b1010_0101.
        let src = [0b1010_0101, 0b1011_0011];
        let mut bits = BitContainer::new(&src).expect("terminated");
        assert_eq!(bits.read(7), 0b011_0011);
        assert_eq!(bits.read(4), 0b1010);
        assert_eq!(bits.read(4), 0b0101);
        assert!(bits.is_exhausted());
    }

    #[test]
    fn reading_past_the_start_overflows_until_reset() {
        let src = [0b0000_0111];
        let mut bits = BitContainer::new(&src).expect("terminated");
        bits.read(2);
        assert!(!bits.overflowed());
        bits.read(2);
        assert!(bits.overflowed());
        assert!(!bits.is_exhausted());
        // A refill cannot undo it.
        assert!(!bits.reload(&src));
        assert!(bits.overflowed());
    }

    #[test]
    fn refills_continue_wide_streams_where_reads_left_off() {
        // Bytes 1..=19, then the marker byte.
        let mut src: [u8; 20] = core::array::from_fn(|index| index as u8 + 1);
        src[19] = 0x80;
        let mut bits = BitContainer::new(&src).expect("terminated");
        assert_eq!(bits.read(7), 0);
        assert_eq!(bits.read(32), 0x1312_1110);
        // A whole word refill, then a cautious one that still moves back
        // as far as asked, then one that reaches the start early.
        assert!(bits.reload(&src));
        assert_eq!(bits.read(32), 0x0f0e_0d0c);
        assert!(bits.reload(&src));
        assert_eq!(bits.read(32), 0x0b0a_0908);
        assert!(!bits.reload(&src));
        assert_eq!(bits.read(32), 0x0706_0504);
        assert_eq!(bits.read(24), 0x03_0201);
        assert!(bits.is_exhausted());
        assert!(!bits.overflowed());
    }

    #[test]
    fn sentinels_refill_only_within_the_buffer() {
        // Bytes 1..=15, then a byte of padding and the marker.
        let mut src: [u8; 16] = core::array::from_fn(|index| index as u8 + 1);
        src[15] = 0x01;
        let mut bits = BitContainer::new(&src).expect("terminated");
        bits.reload_fast(&src);
        let mut sentinel = Sentinel::from_container(bits, &src, 0);
        // The marker byte is consumed: the top bits are byte 14.
        assert_eq!(sentinel.peek(56), 0x0f);
        sentinel.skip(8);
        let limit = src.len() - 8;
        assert!(sentinel.reload_within(&src, limit));
        assert_eq!(sentinel.peek(56), 0x0e);
        // A limit past the end refuses the refill and leaves the sentinel
        // as it was.
        sentinel.skip(16);
        let before = sentinel;
        assert!(!sentinel.reload_within(&src[..11], limit));
        assert_eq!(sentinel.peek(56), before.peek(56));
        assert_eq!(sentinel.remaining(0), before.remaining(0));
        assert!(sentinel.reload_within(&src, limit));
        assert_eq!(sentinel.peek(56), 0x0c);
        // A position past the limit is clamped to it.
        sentinel.skip(8);
        assert!(sentinel.reload_within(&src, 2));
        assert_eq!(sentinel.remaining(0), 2);
        assert_eq!(sentinel.peek(56), 0x0a);
        assert!(sentinel.reload_within(&src, limit));
        assert_eq!(sentinel.remaining(0), 2);
        assert_eq!(sentinel.into_container(&src, 0).look(8), 0x0a);
    }

    #[test]
    fn sentinels_crossing_stream_starts_keep_only_the_stream_bits() {
        let src: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
        let start = 8;
        let word = u64::from_le_bytes(src[start..start + 8].try_into().expect("whole word"));
        // The common word starts in a preceding stream. Every remaining
        // bit count, including an exact end, refers to this stream only.
        for prefix in 1..=8 {
            let pos = start - prefix;
            for consumed in 0..=64 - prefix as u32 * 8 {
                let sentinel = Sentinel {
                    bits: (read64(&src, pos) | 1) << consumed,
                    pos,
                };
                let mut container = sentinel.into_container(&src, start);
                let available = 64 - consumed - prefix as u32 * 8;
                assert_eq!(container.pos, 0);
                assert!(!container.overflowed());
                assert_eq!(container.read(available), word & ((1 << available) - 1));
                assert!(container.is_exhausted());
            }
        }
    }

    #[test]
    fn sentinels_read_past_stream_starts_remain_overflowed() {
        let src: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(29).wrapping_add(17));
        let start = 16;
        for prefix in 1..=16 {
            let pos = start - prefix;
            let consumed = 64u32.saturating_sub(prefix as u32 * 8) + 1;
            let sentinel = Sentinel {
                bits: (read64(&src, pos) | 1) << consumed,
                pos,
            };
            let mut container = sentinel.into_container(&src, start);
            assert_eq!(container.pos, 0);
            assert!(container.overflowed());
            assert!(!container.is_exhausted());
            assert!(!container.reload(&src[start..]));
            container.read(7);
            assert!(container.overflowed());
            assert!(!container.is_exhausted());
        }
    }

    #[test]
    fn forward_reads_are_little_endian_and_count_bytes() {
        let mut bits = ForwardBitReader::new(&[0b1010_0101, 0b0000_0011]);
        assert_eq!(bits.read(4), 0b0101);
        assert_eq!(bits.bytes_used(), 1);
        assert_eq!(bits.peek(6), 0b11_1010);
        assert_eq!(bits.read(6), 0b11_1010);
        assert_eq!(bits.bytes_used(), 2);
        assert_eq!(bits.read(16), 0);
        assert_eq!(bits.bytes_used(), 4);
    }
}

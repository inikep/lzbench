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
//! `BIT_endOfDStream` does.
//!
//! [`ReverseBitReader`] reloads on every read and reads zeros past the start;
//! the Huffman weight decoder relies on its [`ReverseBitReader::overflowed`]
//! test, which mirrors the reference's FSE overflow handling.

use crate::shared::format::highbit32;

/// A backward reader over one bitstream.
#[derive(Clone, Debug)]
pub(crate) struct ReverseBitReader<'a> {
    src: &'a [u8],
    /// Number of unread bits; negative once reads ran past the first byte.
    position: i64,
}

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

impl<'a> ReverseBitReader<'a> {
    /// Opens `src`, or returns `None` when it is empty or its last byte holds
    /// no end marker.
    pub(crate) fn new(src: &'a [u8]) -> Option<Self> {
        let &last = src.last()?;
        if last == 0 {
            return None;
        }
        let whole = i64::try_from(src.len() - 1).ok()?.checked_mul(8)?;
        Some(Self {
            src,
            position: whole + i64::from(highbit32(u32::from(last))),
        })
    }

    /// Bits `[start, start + count)` of the stream as an integer whose most
    /// significant bit is the highest-numbered one; bits below zero read as
    /// zeros. `count` must not exceed 56.
    fn bits_at(&self, start: i64, count: u32) -> u64 {
        debug_assert!(count <= 56);
        if count == 0 {
            return 0;
        }
        let end = start + i64::from(count);
        if end <= 0 {
            return 0;
        }
        let (low, shift_in) = if start < 0 {
            (0, (-start) as u32)
        } else {
            (start, 0)
        };
        let width = (end - low) as u32;
        let byte = (low / 8) as usize;
        let word = load_word(self.src, byte) >> (low % 8);
        (word & ((1u64 << width) - 1)) << shift_in
    }

    /// The next `count` bits without consuming them; `count <= 56`.
    pub(crate) fn peek(&self, count: u32) -> u64 {
        self.bits_at(self.position - i64::from(count), count)
    }

    /// Consumes `count` bits.
    pub(crate) fn consume(&mut self, count: u32) {
        self.position -= i64::from(count);
    }

    /// Reads and consumes `count` bits; `count <= 56`.
    pub(crate) fn read(&mut self, count: u32) -> u64 {
        let value = self.peek(count);
        self.consume(count);
        value
    }

    /// Whether more bits were consumed than the stream holds.
    pub(crate) const fn overflowed(&self) -> bool {
        self.position < 0
    }

    /// Whether every bit was consumed, and no more.
    #[cfg(test)]
    pub(crate) const fn is_exhausted(&self) -> bool {
        self.position == 0
    }
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
        if self.avail > CONTAINER_BITS {
            // Read past the start: corrupt, and caught by `is_exhausted`.
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
        BitContainer {
            container: read64(src, self.pos),
            avail: CONTAINER_BITS - self.bits.trailing_zeros(),
            pos: self.pos - start,
        }
    }

    /// Bytes of the stream left below the container's start.
    #[inline(always)]
    pub(crate) const fn remaining(&self, start: usize) -> usize {
        self.pos - start
    }

    /// Moves back by the consumed whole bytes and refills. Positions
    /// never pass `last`, the start of `src`'s last whole word: clamping to
    /// it changes nothing but lets the compiler drop the bounds check.
    #[inline(always)]
    pub(crate) fn reload(&mut self, src: &[u8], last: usize) {
        let consumed = self.bits.trailing_zeros();
        self.pos -= (consumed >> 3) as usize;
        let at = self.pos.min(last);
        let word = src[at..at + 8].try_into().unwrap_or([0; 8]);
        self.bits = (u64::from_le_bytes(word) | 1) << (consumed & 7);
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
        assert!(ReverseBitReader::new(&[]).is_none());
        assert!(ReverseBitReader::new(&[0x12, 0x00]).is_none());
    }

    #[test]
    fn the_marker_bit_and_padding_are_skipped() {
        // 0b0000_0101: marker at bit 2, two payload bits `01`.
        let mut bits = ReverseBitReader::new(&[0b0000_0101]).expect("terminated");
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
        let mut bits = ReverseBitReader::new(&src).expect("terminated");
        assert_eq!(bits.read(7), 0b011_0011);
        assert_eq!(bits.read(4), 0b1010);
        assert_eq!(bits.read(4), 0b0101);
        assert!(bits.is_exhausted());
    }

    #[test]
    fn reading_past_the_start_yields_zeros_and_overflows() {
        let mut bits = ReverseBitReader::new(&[0b0000_0111]).expect("terminated");
        assert_eq!(bits.peek(4), 0b1100);
        assert_eq!(bits.read(4), 0b1100);
        assert!(bits.overflowed());
        assert!(!bits.is_exhausted());
        assert_eq!(bits.read(8), 0);
    }

    #[test]
    fn wide_reads_use_whole_words() {
        let src: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0x80];
        let mut bits = ReverseBitReader::new(&src).expect("terminated");
        assert_eq!(bits.read(7), 0);
        assert_eq!(bits.read(32), 0x0b0a_0908);
        assert_eq!(bits.read(32), 0x0706_0504);
        assert_eq!(bits.read(24), 0x03_0201);
        assert!(bits.is_exhausted());
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

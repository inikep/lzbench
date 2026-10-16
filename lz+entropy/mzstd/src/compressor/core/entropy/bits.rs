//! The forward bit writer behind FSE streams, sequence streams and Huffman
//! streams.
//!
//! Bits accumulate least significant first in a 64-bit container and are
//! written out eight bytes at a time. Capacity follows the reference
//! `BIT_CStream_t` exactly: the last eight bytes of the destination are a
//! landing zone for those wide stores, so a stream that reaches them reports
//! "does not fit" (`0`) rather than the reference's size. Callers branch on
//! that result to choose a different encoding, so the boundary is part of
//! the output contract.

/// Container width.
const CONTAINER_BYTES: usize = 8;

/// A forward bit stream over a caller-owned destination.
#[derive(Debug)]
pub(crate) struct BitWriter<'a> {
    dst: &'a mut [u8],
    /// Next byte to write.
    pos: usize,
    /// The furthest `pos` may advance: `dst.len() - 8`.
    end: usize,
    /// Pending bits, least significant first.
    container: u64,
    /// Number of pending bits.
    bit_pos: u32,
}

impl<'a> BitWriter<'a> {
    /// Starts a stream at the beginning of `dst`.
    ///
    /// Returns `None` when `dst` cannot hold a single container, where the
    /// reference reports `dstSize_tooSmall`.
    pub(crate) fn new(dst: &'a mut [u8]) -> Option<Self> {
        if dst.len() <= CONTAINER_BYTES {
            return None;
        }
        let end = dst.len() - CONTAINER_BYTES;
        Some(Self {
            dst,
            pos: 0,
            end,
            container: 0,
            bit_pos: 0,
        })
    }

    /// Appends the low `nb_bits` bits of `value`; `nb_bits` is below 32.
    #[inline(always)]
    pub(crate) fn add_bits(&mut self, value: u64, nb_bits: u32) {
        debug_assert!(nb_bits < 32);
        debug_assert!(nb_bits + self.bit_pos < 64);
        let mask = (1u64 << nb_bits) - 1;
        self.container |= (value & mask) << self.bit_pos;
        self.bit_pos += nb_bits;
    }

    /// Appends `nb_bits` bits of a value with no bits above them.
    #[inline(always)]
    pub(crate) fn add_bits_fast(&mut self, value: u64, nb_bits: u32) {
        debug_assert!(value >> nb_bits == 0);
        debug_assert!(nb_bits + self.bit_pos < 64);
        self.container |= value << self.bit_pos;
        self.bit_pos += nb_bits;
    }

    /// Writes every whole pending byte.
    ///
    /// Past the end of the usable area the bytes land at its limit, as the
    /// reference's clamped write position puts them, and the overflow
    /// surfaces in [`Self::close`]: the position only grows, so once past
    /// the limit it stays there.
    #[inline(always)]
    pub(crate) fn flush(&mut self) {
        let nb_bytes = (self.bit_pos >> 3) as usize;
        let at = self.pos.min(self.end);
        if let Some(window) = self.dst.get_mut(at..at + CONTAINER_BYTES) {
            window.copy_from_slice(&self.container.to_le_bytes());
        }
        self.pos += nb_bytes;
        self.bit_pos &= 7;
        self.container >>= nb_bytes * 8;
    }

    /// Adds the end mark and returns the stream size, or `0` when the stream
    /// did not fit.
    pub(crate) fn close(mut self) -> usize {
        self.add_bits_fast(1, 1);
        self.flush();
        if self.pos >= self.end {
            return 0;
        }
        self.pos + usize::from(self.bit_pos > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::BitWriter;

    #[test]
    fn bits_are_packed_least_significant_first_with_an_end_mark() {
        let mut dst = [0u8; 16];
        let mut writer = BitWriter::new(&mut dst).expect("capacity");
        writer.add_bits(0b101, 3);
        writer.add_bits(0xFF, 4);
        writer.flush();
        let size = writer.close();
        assert_eq!(size, 1);
        assert_eq!(dst[0], 0b1111_1101);
    }

    #[test]
    fn a_destination_of_one_container_is_refused() {
        let mut dst = [0u8; 8];
        assert!(BitWriter::new(&mut dst).is_none());
    }

    #[test]
    fn reaching_the_landing_zone_reports_no_fit() {
        let mut dst = [0u8; 10];
        let mut writer = BitWriter::new(&mut dst).expect("capacity");
        for _ in 0..3 {
            writer.add_bits(0xFF, 8);
            writer.flush();
        }
        assert_eq!(writer.close(), 0);

        let mut dst = [0u8; 10];
        let mut writer = BitWriter::new(&mut dst).expect("capacity");
        writer.add_bits(0xFF, 8);
        writer.flush();
        assert_eq!(writer.close(), 2);
    }
}

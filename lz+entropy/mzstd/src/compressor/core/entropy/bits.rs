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

    /// The pending bits and their number, for a caller that packs a run
    /// of values itself and hands the result to [`Self::append`].
    #[inline(always)]
    pub(crate) const fn pending(&self) -> (u64, u32) {
        (self.container, self.bit_pos)
    }

    /// The next `N` bytes of the destination, for a caller that packs a
    /// run of values there itself and then calls [`Self::advance`]; `None`
    /// when fewer are left.
    ///
    /// Packing in place spares the copy [`Self::append`] makes, whose wide
    /// loads would wait on the narrow overlapping stores that filled its
    /// source.
    #[inline(always)]
    pub(crate) fn room<const N: usize>(&mut self) -> Option<&mut [u8; N]> {
        self.dst.get_mut(self.pos..)?.first_chunk_mut::<N>()
    }

    /// Takes over a run packed into [`Self::room`] from [`Self::pending`]:
    /// its first `bytes` bytes are whole bytes of the stream, at least
    /// eight fewer than the room, and `container` holds the `bit_pos` bits
    /// left over, fewer than eight.
    #[inline(always)]
    pub(crate) const fn advance(&mut self, bytes: usize, container: u64, bit_pos: u32) {
        debug_assert!(bit_pos < 8);
        debug_assert!(self.pos + bytes + CONTAINER_BYTES <= self.dst.len());
        self.pos += bytes;
        self.container = container;
        self.bit_pos = bit_pos;
    }

    /// Takes over a run packed from [`Self::pending`]: `bytes` are the
    /// whole bytes the run completed, in stream order, and `container`
    /// holds the `bit_pos` bits left over, fewer than eight.
    ///
    /// The stream is the one [`Self::add_bits`] and [`Self::flush`] would
    /// have written for the same values. Bytes that do not fit are
    /// dropped and the position still advances, so the overflow surfaces
    /// in [`Self::close`] as it does for a flush.
    pub(crate) fn append(&mut self, bytes: &[u8], container: u64, bit_pos: u32) {
        debug_assert!(bit_pos < 8);
        if let Some(room) = self
            .pos
            .checked_add(bytes.len())
            .and_then(|end| self.dst.get_mut(self.pos..end))
        {
            room.copy_from_slice(bytes);
        }
        self.pos = self.pos.saturating_add(bytes.len());
        self.container = container;
        self.bit_pos = bit_pos;
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

    /// The values a packed run is compared with: widths from none to 56
    /// bits, which with fewer than eight pending bits still fit a
    /// container.
    fn values() -> impl Iterator<Item = (u64, u32)> {
        (0..200u64).map(|i| {
            let width = (i * 7 % 57) as u32;
            let value = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) & ((1 << width) - 1);
            (value, width)
        })
    }

    /// Packs `values` the way a caller of [`BitWriter::append`] does:
    /// whole bytes into a buffer, the rest left in the container.
    fn pack(mut container: u64, mut bit_pos: u32) -> (alloc::vec::Vec<u8>, u64, u32) {
        let mut bytes = alloc::vec::Vec::new();
        for (value, width) in values() {
            container |= value << bit_pos;
            bit_pos += width;
            let whole = (bit_pos >> 3) as usize;
            bytes.extend_from_slice(&container.to_le_bytes()[..whole]);
            container >>= whole * 8;
            bit_pos &= 7;
        }
        (bytes, container, bit_pos)
    }

    #[test]
    fn an_appended_run_is_the_stream_of_its_values() {
        let mut expected = [0u8; 1024];
        let mut direct = BitWriter::new(&mut expected).expect("capacity");
        direct.add_bits(0b101, 3);
        direct.flush();
        for (value, width) in values() {
            direct.add_bits_fast(value, width);
            direct.flush();
        }
        let size = direct.close();
        assert_ne!(size, 0);

        let mut dst = [0u8; 1024];
        let mut writer = BitWriter::new(&mut dst).expect("capacity");
        writer.add_bits(0b101, 3);
        writer.flush();
        let (container, bit_pos) = writer.pending();
        assert_eq!((container, bit_pos), (0b101, 3));
        let (bytes, container, bit_pos) = pack(container, bit_pos);
        writer.append(&bytes, container, bit_pos);
        assert_eq!(writer.close(), size);
        assert_eq!(dst[..size], expected[..size]);
    }

    #[test]
    fn a_run_packed_in_place_is_the_stream_of_its_values() {
        const ROOM: usize = 2048;
        let mut expected = [0u8; ROOM + 64];
        let mut direct = BitWriter::new(&mut expected).expect("capacity");
        direct.add_bits(0b11, 2);
        direct.flush();
        for (value, width) in values() {
            direct.add_bits_fast(value, width);
            direct.flush();
        }
        let size = direct.close();
        assert_ne!(size, 0);

        let mut dst = [0u8; ROOM + 64];
        let mut writer = BitWriter::new(&mut dst).expect("capacity");
        writer.add_bits(0b11, 2);
        writer.flush();
        let (container, bit_pos) = writer.pending();
        let (bytes, container, bit_pos) = pack(container, bit_pos);
        assert!(bytes.len() + 8 <= ROOM);
        let room = writer.room::<ROOM>().expect("room");
        room[..bytes.len()].copy_from_slice(&bytes);
        writer.advance(bytes.len(), container, bit_pos);
        assert_eq!(writer.close(), size);
        assert_eq!(dst[..size], expected[..size]);
    }

    #[test]
    fn room_is_refused_where_the_destination_ends() {
        let mut dst = [0u8; 40];
        let mut writer = BitWriter::new(&mut dst).expect("capacity");
        assert!(writer.room::<40>().is_some());
        assert!(writer.room::<41>().is_none());
        writer.add_bits(0xFFFF, 16);
        writer.flush();
        assert!(writer.room::<38>().is_some());
        assert!(writer.room::<39>().is_none());
    }

    #[test]
    fn an_appended_run_that_does_not_fit_reports_no_fit() {
        let (bytes, container, bit_pos) = pack(0, 0);
        // The run needs its bytes, the end mark and the landing zone.
        for (capacity, fits) in [
            (bytes.len() + 9, true),
            (bytes.len() + 8, false),
            (bytes.len(), false),
            (9, false),
        ] {
            let mut dst = alloc::vec![0xAAu8; capacity];
            let mut writer = BitWriter::new(&mut dst).expect("capacity");
            writer.append(&bytes, container, bit_pos);
            // A later run after the overflow changes nothing.
            if !fits {
                writer.append(&[1, 2, 3], 0, 0);
            }
            let size = writer.close();
            assert_eq!(size != 0, fits, "{capacity}");
            if fits {
                assert_eq!(dst[..bytes.len()], bytes[..]);
            }
        }
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

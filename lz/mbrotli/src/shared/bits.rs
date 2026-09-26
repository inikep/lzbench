//! Least-significant-bit-first bit writer used by the fast encoders.
//!
//! The layout matches `BrotliWriteBits` in `c/enc/write_bits.h` of the pinned
//! reference: bits are written into increasing byte addresses and, within a
//! byte, from the least significant bit upwards. Every write also clears the
//! bits above the new position in the byte it lands in, so the writer never
//! needs a separate "prepare storage" step after the first byte.

use alloc::vec::Vec;

/// Largest number of bits a single [`BitWriter::write`] call accepts.
pub(crate) const MAX_BITS_PER_WRITE: u32 = 56;

/// Bytes a write touches, and therefore the headroom the buffer must keep.
pub(crate) const WRITE_SLACK: usize = 8;

/// Statically selected initialized storage for a bit writer.
///
/// Slice storage has a fixed bound. Vector storage initializes only requested
/// ranges, so its unwritten reserved capacity never needs clearing or copying.
pub(crate) trait ByteBuffer {
    fn bytes(&self) -> &[u8];
    fn window(&mut self, range: ::core::ops::Range<usize>) -> Option<&mut [u8]>;

    #[inline(always)]
    fn copy_bytes(&mut self, start: usize, data: &[u8]) -> bool {
        let Some(window) = self.window(start..start + data.len()) else {
            return false;
        };
        window.copy_from_slice(data);
        true
    }
}

impl ByteBuffer for [u8] {
    #[inline(always)]
    fn bytes(&self) -> &[u8] {
        self
    }

    #[inline(always)]
    fn window(&mut self, range: ::core::ops::Range<usize>) -> Option<&mut [u8]> {
        self.get_mut(range)
    }
}

/// Keeps infrequent allocation and initialization out of each symbol write.
#[cold]
#[inline(never)]
fn grow_bit_output(storage: &mut Vec<u8>, required: usize) {
    let end = required.checked_add(255).map_or(required, |end| end & !255);
    storage.resize(required.max(end.min(storage.capacity())), 0);
}

impl ByteBuffer for Vec<u8> {
    #[inline(always)]
    fn bytes(&self) -> &[u8] {
        self
    }

    #[inline(always)]
    fn window(&mut self, range: ::core::ops::Range<usize>) -> Option<&mut [u8]> {
        if self.len() < range.end {
            // Amortize initialization over small batches of bit writes.
            grow_bit_output(self, range.end);
        }
        self.get_mut(range)
    }

    #[inline(always)]
    fn copy_bytes(&mut self, start: usize, data: &[u8]) -> bool {
        if self.len() < start {
            self.resize(start, 0);
        }
        let initialized = (self.len() - start).min(data.len());
        self[start..start + initialized].copy_from_slice(&data[..initialized]);
        // Uncompressed blocks are copied straight into spare capacity by Vec.
        // No zero-fill is needed before this safe initialized append.
        self.extend_from_slice(&data[initialized..]);
        true
    }
}

/// Cursor into a caller-owned byte buffer that appends individual bits.
pub(crate) struct BitWriter<'a, B: ByteBuffer + ?Sized = [u8]> {
    storage: &'a mut B,
    position: usize,
    overflowed: bool,
}

impl<'a> BitWriter<'a> {
    /// Creates a writer that resumes at bit `position` of `storage`.
    ///
    /// The caller owns the invariant that `storage[position >> 3]` already
    /// holds the partially filled byte, exactly as the reference encoder seeds
    /// the first two bytes with the stream header.
    pub(crate) const fn new(storage: &'a mut [u8], position: usize) -> Self {
        Self {
            storage,
            position,
            overflowed: false,
        }
    }
}

impl<'a> BitWriter<'a, Vec<u8>> {
    /// Appends bits into a growable vector, retaining its existing prefix.
    pub(crate) const fn append(storage: &'a mut Vec<u8>, position: usize) -> Self {
        Self {
            storage,
            position,
            overflowed: false,
        }
    }
}

impl<B: ByteBuffer + ?Sized> BitWriter<'_, B> {
    /// Returns the current bit position.
    pub(crate) const fn position(&self) -> usize {
        self.position
    }

    /// Returns `true` when a write was refused because the buffer was full.
    ///
    /// A writer that has overflowed keeps its position advancing so callers can
    /// still finish their control flow, but the produced bytes are meaningless
    /// and the encoder turns this into an error.
    pub(crate) const fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// Returns the byte at `index`, or zero when it lies past the buffer.
    pub(crate) fn byte(&self, index: usize) -> u8 {
        match self.storage.bytes().get(index) {
            Some(&byte) => byte,
            None => 0,
        }
    }

    /// Overwrites the byte at `index`, ignoring an index past the buffer.
    ///
    /// The encoder uses this to restore the partial byte it carried into a
    /// meta-block after deciding to store that meta-block uncompressed.
    pub(crate) fn set_byte(&mut self, index: usize, value: u8) {
        let Some(end) = index.checked_add(1) else {
            self.overflowed = true;
            return;
        };
        match self
            .storage
            .window(index..end)
            .and_then(|bytes| bytes.first_mut())
        {
            Some(byte) => *byte = value,
            None => self.overflowed = true,
        }
    }

    /// Appends the `n_bits` low bits of `bits`.
    ///
    /// Like the reference writer this materialises a whole machine word, which
    /// clears the seven bytes that follow the new position. Callers therefore
    /// have to provide [`WRITE_SLACK`] bytes of headroom past the largest
    /// position they will reach.
    ///
    /// # Panics
    ///
    /// Never panics; a write past the end of the buffer sets the overflow flag
    /// instead.
    #[inline(always)]
    pub(crate) fn write(&mut self, n_bits: u32, bits: u64) {
        debug_assert!(n_bits <= MAX_BITS_PER_WRITE);
        debug_assert!(n_bits == 64 || (bits >> n_bits) == 0);
        let start = self.position >> 3;
        let Some(window) = self.storage.window(start..start + WRITE_SLACK) else {
            self.overflowed = true;
            self.position += n_bits as usize;
            return;
        };
        let value = u64::from(window[0]) | (bits << (self.position & 7));
        window.copy_from_slice(&value.to_le_bytes());
        self.position += n_bits as usize;
    }

    /// Overwrites `n_bits` bits that were already emitted at `position`.
    ///
    /// Mirrors `UpdateBits`; quality 0 uses it to widen the `MLEN` field of a
    /// meta-block it decided to extend.
    pub(crate) fn update(&mut self, mut n_bits: u32, mut bits: u32, mut position: usize) {
        while n_bits > 0 {
            let byte_position = position >> 3;
            let unchanged = (position & 7) as u32;
            let changed = n_bits.min(8 - unchanged);
            let total = unchanged + changed;
            let Some(byte) = self
                .storage
                .window((byte_position)..(byte_position) + 1)
                .and_then(|bytes| bytes.first_mut())
            else {
                self.overflowed = true;
                return;
            };
            let mask = !((1u32 << total) - 1) | ((1u32 << unchanged) - 1);
            let kept = u32::from(*byte) & mask;
            let fresh = bits & ((1u32 << changed) - 1);
            *byte = ((fresh << unchanged) | kept) as u8;
            n_bits -= changed;
            bits >>= changed;
            position += changed as usize;
        }
    }

    /// Drops everything written after bit `position` and resumes there.
    pub(crate) fn rewind(&mut self, position: usize) {
        let mask = (1u16 << (position & 7)) - 1;
        if let Some(byte) = self
            .storage
            .window((position >> 3)..(position >> 3) + 1)
            .and_then(|bytes| bytes.first_mut())
        {
            *byte &= mask as u8;
        } else {
            self.overflowed = true;
        }
        self.position = position;
    }

    /// Advances to the next byte boundary, leaving the skipped bits zero.
    pub(crate) const fn align(&mut self) {
        self.position = (self.position + 7) & !7;
    }

    /// Advances to the next byte boundary and clears the byte landed on.
    ///
    /// Mirrors `JumpToByteBoundary`: the caller reads that byte back as the
    /// partial byte carried into the next meta-block, so in a reused buffer it
    /// has to be cleared rather than left holding an older stream's bits.
    pub(crate) fn jump_to_byte_boundary(&mut self) {
        self.align();
        self.prepare_storage();
    }

    /// Clears the byte at the current, byte-aligned position.
    ///
    /// Mirrors `BrotliWriteBitsPrepareStorage`.
    pub(crate) fn prepare_storage(&mut self) {
        debug_assert_eq!(self.position & 7, 0);
        match self
            .storage
            .window((self.position >> 3)..(self.position >> 3) + 1)
            .and_then(|bytes| bytes.first_mut())
        {
            Some(byte) => *byte = 0,
            None => self.overflowed = true,
        }
    }

    /// Copies `data` verbatim at the current, byte-aligned position.
    pub(crate) fn write_bytes(&mut self, data: &[u8]) {
        debug_assert_eq!(self.position & 7, 0);
        let start = self.position >> 3;
        if !self.storage.copy_bytes(start, data) {
            self.overflowed = true;
            self.position += data.len() << 3;
            return;
        }
        self.position += data.len() << 3;
        match self
            .storage
            .window((self.position >> 3)..(self.position >> 3) + 1)
            .and_then(|bytes| bytes.first_mut())
        {
            Some(byte) => *byte = 0,
            None => self.overflowed = true,
        }
    }
}

/// Bytes [`inject_byte_padding`] can write, and therefore the headroom a
/// flushing encoder has to keep past the bytes it already completed.
///
/// The seal is at most seven carried bits plus the six of the header, so it
/// spans at most two bytes; the third is the fresh partial byte the writer
/// would resume in, which this function never leaves behind.
pub(crate) const BYTE_PADDING_SLACK: usize = 3;

/// Realigns the stream to a byte boundary with an empty metadata meta-block.
///
/// Ports `InjectBytePaddingBlock` from `c/enc/encode.c`. The six bits it emits
/// are `ISLAST = 0`, `MNIBBLES = 3` (the metadata escape), one reserved zero
/// and `MSKIPBYTES = 0`, which a decoder reads as a metadata block carrying no
/// bytes. Everything above them in the final byte is left zero, so the stream
/// resumes on a byte boundary and every byte handed out so far decodes to the
/// input the encoder has already consumed.
///
/// `last_bytes` and `last_bytes_bits` are the partial byte the encoder carries
/// between meta-blocks. On return they are zero: the padded stream is aligned,
/// so there is nothing left to carry.
///
/// Returns the number of bytes written at the start of `dst`, which is zero
/// when the stream was already aligned — the reference injects nothing in that
/// case, and emitting a bare metadata block would change the output. `dst`
/// must hold [`BYTE_PADDING_SLACK`] bytes; a shorter buffer writes nothing and
/// reports zero, leaving the carried byte untouched so the caller can retry.
pub(crate) fn inject_byte_padding(
    last_bytes: &mut u16,
    last_bytes_bits: &mut u32,
    dst: &mut [u8],
) -> usize {
    if *last_bytes_bits == 0 {
        return 0;
    }
    let Some(window) = dst.first_chunk_mut::<BYTE_PADDING_SLACK>() else {
        return 0;
    };

    let seal_bits = *last_bytes_bits;
    // `0x6` is the six-bit header itself: bit 0 is ISLAST, bits 1 and 2 are
    // MNIBBLES = 3, bit 3 is reserved and bits 4 and 5 are MSKIPBYTES = 0.
    let seal = u32::from(*last_bytes) | (0x6u32 << seal_bits);
    *last_bytes = 0;
    *last_bytes_bits = 0;

    window[0] = seal as u8;
    window[1] = (seal >> 8) as u8;
    window[2] = (seal >> 16) as u8;
    ((seal_bits as usize + 6) + 7) >> 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growing_storage_matches_fixed_storage_through_rewinds_and_byte_writes() {
        fn exercise<B: ByteBuffer + ?Sized>(writer: &mut BitWriter<'_, B>, count: usize) -> usize {
            let start = writer.position();
            writer.write(3, 5);
            for value in 0..count {
                writer.write(7, (value & 127) as u64);
            }
            writer.update(2, 2, start);
            writer.rewind(writer.position() - 1);
            writer.write(1, 0);
            writer.jump_to_byte_boundary();
            writer.write_bytes(&vec![37; count]);
            writer.set_byte(start >> 3, 6);
            assert_eq!(writer.byte(start >> 3), 6);
            assert_eq!(writer.byte(usize::MAX), 0);
            assert!(!writer.overflowed());
            writer.set_byte(usize::MAX, 0);
            assert!(writer.overflowed());
            writer.position()
        }

        for prefix in [0, 1, 7, 64] {
            for count in (0..=65).chain([127, 128, 255, 256, 1023, 4097]) {
                let mut fixed = vec![0; 1 << 14];
                fixed[..prefix].fill(91);
                let mut growing = vec![91; prefix];
                let expected = exercise(&mut BitWriter::new(&mut fixed, prefix * 8), count);
                let actual = exercise(&mut BitWriter::append(&mut growing, prefix * 8), count);
                assert_eq!(actual, expected);
                assert_eq!(&growing[..actual / 8], &fixed[..actual / 8]);
                assert!(growing.len() < fixed.len());
            }
        }
    }

    fn writer_output(bits: &[(u32, u64)]) -> (Vec<u8>, usize) {
        let mut storage = vec![0u8; 64];
        let mut writer = BitWriter::new(&mut storage, 0);
        for &(n, value) in bits {
            writer.write(n, value);
        }
        let position = writer.position();
        assert!(!writer.overflowed());
        (storage, position)
    }

    #[test]
    fn writes_bits_least_significant_first() {
        let (storage, position) = writer_output(&[(1, 1), (2, 2), (5, 0b10101)]);
        assert_eq!(position, 8);
        assert_eq!(storage[0], 0b1010_1101);
    }

    #[test]
    fn writes_across_a_byte_boundary() {
        let (storage, position) = writer_output(&[(4, 0xF), (8, 0xA5)]);
        assert_eq!(position, 12);
        assert_eq!(storage[0], 0x5F);
        assert_eq!(storage[1], 0x0A);
    }

    #[test]
    fn writes_the_widest_supported_field() {
        let value = (1u64 << 56) - 1;
        let (storage, position) = writer_output(&[(3, 5), (56, value)]);
        assert_eq!(position, 59);
        assert_eq!(storage[0], 0b1111_1101);
        assert_eq!(&storage[1..7], &[0xFF; 6]);
        assert_eq!(storage[7], 0b0000_0111);
    }

    #[test]
    fn zero_width_writes_leave_the_stream_untouched() {
        let (storage, position) = writer_output(&[(3, 5), (0, 0), (3, 5)]);
        assert_eq!(position, 6);
        assert_eq!(storage[0], 0b0010_1101);
    }

    #[test]
    fn clears_the_bytes_that_follow_the_write() {
        let mut storage = vec![0xFFu8; 16];
        storage[0] = 0;
        let mut writer = BitWriter::new(&mut storage, 0);
        writer.write(4, 0);
        assert_eq!(&storage[..8], &[0u8; 8]);
        assert_eq!(&storage[8..], &[0xFFu8; 8]);
    }

    #[test]
    fn update_rewrites_an_already_emitted_field() {
        let mut storage = vec![0u8; 16];
        let mut writer = BitWriter::new(&mut storage, 0);
        writer.write(3, 0b101);
        let patched = writer.position();
        writer.write(20, 0);
        writer.write(4, 0xF);
        writer.update(20, 0x0F_F0F, patched);
        assert_eq!(writer.position(), 27);
        let value = u32::from_le_bytes([storage[0], storage[1], storage[2], storage[3]]);
        assert_eq!(value & 0b111, 0b101);
        assert_eq!((value >> 3) & 0xF_FFFF, 0x0F_F0F);
        assert_eq!((value >> 23) & 0xF, 0xF);
    }

    #[test]
    fn rewind_discards_bits_written_after_the_mark() {
        let mut storage = vec![0u8; 16];
        let mut writer = BitWriter::new(&mut storage, 0);
        writer.write(5, 0b10101);
        let mark = writer.position();
        writer.write(20, 0xF_FFFF);
        writer.rewind(mark);
        assert_eq!(writer.position(), 5);
        assert_eq!(storage[0], 0b0001_0101);
    }

    #[test]
    fn align_moves_to_the_next_byte_boundary() {
        let mut storage = vec![0u8; 8];
        let mut writer = BitWriter::new(&mut storage, 0);
        writer.write(3, 0b111);
        writer.align();
        assert_eq!(writer.position(), 8);
        writer.align();
        assert_eq!(writer.position(), 8);
    }

    #[test]
    fn jumping_to_a_byte_boundary_clears_the_byte_landed_on() {
        let mut storage = vec![0xFFu8; 16];
        storage[0] = 0;
        let mut writer = BitWriter::new(&mut storage, 0);
        writer.write(3, 0b101);
        writer.jump_to_byte_boundary();
        assert_eq!(writer.position(), 8);
        assert_eq!(storage[0], 0b101);
        assert_eq!(storage[1], 0);
    }

    #[test]
    fn preparing_storage_clears_only_the_current_byte() {
        let mut storage = vec![0xFFu8; 4];
        let mut writer = BitWriter::new(&mut storage, 16);
        writer.prepare_storage();
        assert_eq!(storage, vec![0xFF, 0xFF, 0x00, 0xFF]);
    }

    #[test]
    fn preparing_storage_past_the_buffer_reports_an_overflow() {
        let mut storage = vec![0u8; 1];
        let mut writer = BitWriter::new(&mut storage, 64);
        writer.prepare_storage();
        assert!(writer.overflowed());
    }

    #[test]
    fn write_bytes_copies_verbatim_and_clears_the_next_byte() {
        let mut storage = vec![0xFFu8; 16];
        storage[0] = 0;
        let mut writer = BitWriter::new(&mut storage, 0);
        writer.write(8, 0x41);
        writer.write_bytes(&[1, 2, 3]);
        let position = writer.position();
        assert_eq!(&storage[..5], &[0x41, 1, 2, 3, 0]);
        assert_eq!(position, 32);
    }

    #[test]
    fn reports_overflow_instead_of_panicking() {
        let mut storage = vec![0u8; WRITE_SLACK + 1];
        let mut writer = BitWriter::new(&mut storage, 0);
        writer.write(8, 0xFF);
        assert!(!writer.overflowed());
        writer.write(8, 0xFF);
        assert!(!writer.overflowed());
        writer.write(8, 0xFF);
        assert!(writer.overflowed());
    }

    #[test]
    fn overflow_is_reported_for_rewind_update_and_byte_copies() {
        let mut storage = vec![0u8; 1];
        let mut writer = BitWriter::new(&mut storage, 0);
        writer.write_bytes(&[1, 2, 3]);
        assert!(writer.overflowed());

        let mut storage = vec![0u8; 1];
        let mut writer = BitWriter::new(&mut storage, 0);
        writer.update(8, 0xFF, 64);
        assert!(writer.overflowed());

        let mut storage = vec![0u8; 1];
        let mut writer = BitWriter::new(&mut storage, 0);
        writer.rewind(64);
        assert!(writer.overflowed());
    }

    #[test]
    fn set_byte_overwrites_and_reports_an_out_of_range_index() {
        let mut storage = vec![0u8; 2];
        let mut writer = BitWriter::new(&mut storage, 0);
        writer.set_byte(1, 0xAB);
        assert!(!writer.overflowed());
        writer.set_byte(9, 0xCD);
        assert!(writer.overflowed());
        assert_eq!(storage[1], 0xAB);
    }

    #[test]
    fn byte_reads_past_the_buffer_as_zero() {
        let mut storage = vec![7u8; 1];
        let writer = BitWriter::new(&mut storage, 0);
        assert_eq!(writer.byte(0), 7);
        assert_eq!(writer.byte(9), 0);
    }

    #[test]
    fn byte_padding_seals_every_unaligned_position() {
        // Reproduces `InjectBytePaddingBlock` by hand at every bit offset a
        // meta-block can leave behind, carrying a byte whose low bits are set
        // so the seal has to preserve them.
        for bits in 1..8u32 {
            let carried = u16::from(u8::MAX >> (8 - bits));
            let mut last_bytes = carried;
            let mut last_bytes_bits = bits;
            let mut dst = [0u8; BYTE_PADDING_SLACK];

            let written = inject_byte_padding(&mut last_bytes, &mut last_bytes_bits, &mut dst);

            assert_eq!(written, ((bits as usize + 6) + 7) >> 3, "bits {bits}");
            assert_eq!(last_bytes, 0, "bits {bits}: a carried byte survived");
            assert_eq!(last_bytes_bits, 0, "bits {bits}: the stream stayed open");

            let expected = u32::from(carried) | (0x6u32 << bits);
            let mut seen = 0u32;
            for (index, byte) in dst[..written].iter().enumerate() {
                seen |= u32::from(*byte) << (index * 8);
            }
            assert_eq!(seen, expected, "bits {bits}: wrong seal");
            // Nothing above the six header bits is set, so the stream resumes
            // on a byte boundary with a zero partial byte.
            assert_eq!(seen >> (bits + 6), 0, "bits {bits}: bits above the seal");
        }
    }

    #[test]
    fn byte_padding_writes_nothing_when_already_aligned() {
        let mut last_bytes = 0u16;
        let mut last_bytes_bits = 0u32;
        let mut dst = [0xAAu8; BYTE_PADDING_SLACK];

        assert_eq!(
            inject_byte_padding(&mut last_bytes, &mut last_bytes_bits, &mut dst),
            0
        );
        assert_eq!(
            dst, [0xAA; BYTE_PADDING_SLACK],
            "an aligned stream was padded"
        );
    }

    #[test]
    fn byte_padding_leaves_the_carry_alone_when_the_buffer_is_short() {
        let mut last_bytes = 0x1u16;
        let mut last_bytes_bits = 3u32;
        let mut dst = [0u8; BYTE_PADDING_SLACK - 1];

        assert_eq!(
            inject_byte_padding(&mut last_bytes, &mut last_bytes_bits, &mut dst),
            0
        );
        assert_eq!(last_bytes, 0x1, "the carried byte was consumed");
        assert_eq!(last_bytes_bits, 3, "the carried bit count was consumed");
    }
}

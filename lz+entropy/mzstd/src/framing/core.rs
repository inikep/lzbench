//! Wire constants of the framed container, shared by its encoder, decoder
//! and seek reader.
//!
//! Every container is a sequence of Zstandard frames: skippable frames
//! carry the container header and every chunk header, regular Zstandard
//! frames carry visible payload, and a final skippable frame holds a seek
//! table in the layout of zstd's `contrib/seekable_format`.

/// Magic number of the skippable frames carrying the container header and
/// every chunk.
pub(crate) const CHUNK_MAGIC: u32 = 0x184D_2A5B;
/// Magic number of the seek table's skippable frame (`contrib/seekable_format`).
pub(crate) const SEEK_TABLE_MAGIC: u32 = 0x184D_2A5E;
/// Last four bytes of every container: the seek table footer's magic number.
pub(crate) const SEEKABLE_MAGIC: u32 = 0x8F92_EAB1;
/// Content of the container header frame before its flags byte.
pub(crate) const SIGNATURE: [u8; 4] = *b"MZFR";
/// Magic number and size field of a skippable frame.
pub(crate) const FRAME_HEADER: usize = 8;
/// Length of the container header frame: frame header, signature, flags.
pub(crate) const CONTAINER_HEADER: usize = FRAME_HEADER + SIGNATURE.len() + 1;
/// Container flag of the full profile: a footer, any number of resources.
pub(crate) const FLAG_CONTAINER: u8 = 4;
/// Version bits of the container flags; only zero is defined.
#[cfg(feature = "decompression")]
pub(crate) const VERSION_MASK: u8 = 3;
/// Number of frames, descriptor and magic number closing the seek table.
pub(crate) const SEEK_FOOTER: usize = 9;
/// Compressed and decompressed size of one seek table entry.
pub(crate) const SEEK_ENTRY: usize = 8;
/// Seek table descriptor bits that must be zero: the checksum flag (bit 7),
/// which this format does not use, and the reserved bits 2-6.
#[cfg(feature = "decompression")]
pub(crate) const SEEK_DESCRIPTOR_MASK: u8 = 0xFC;
/// Most frames a seek table may list (`ZSTD_SEEKABLE_MAXFRAMES`).
#[cfg(feature = "compression")]
pub(crate) const MAX_FRAMES: u64 = 0x800_0000;
/// Footer content: type, directory offset and file size.
#[cfg(any(feature = "compression", not(feature = "no_std")))]
pub(crate) const FOOTER_CONTENT: usize = 17;
/// Largest value a varint may hold.
#[cfg(feature = "compression")]
pub(crate) const MAX_VARINT: u64 = (1 << 63) - 1;
/// Most bytes of a varint.
#[cfg(feature = "decompression")]
pub(crate) const MAX_VARINT_LEN: usize = 9;
/// Most dictionary references one chunk may carry.
pub(crate) const MAX_REFERENCES: usize = 16;
/// Size of an identity: a hash-type byte and 32 hash bytes.
pub(crate) const ID_LEN: usize = 33;
/// Hash-type byte of every identity.
pub(crate) const ID_TYPE: u8 = 3;
/// Chunk flag: resource is hidden (first chunk only).
pub(crate) const FLAG_HIDDEN: u8 = 1;
/// Chunk flag: whole-resource identity follows (last chunk only).
pub(crate) const FLAG_CHECKSUM: u8 = 2;
/// Chunk flag: the content is the Zstandard frame following the header frame.
pub(crate) const FLAG_DETACHED: u8 = 4;
/// Largest content of a chunk: no frame may exceed 32 bits.
#[cfg(feature = "compression")]
pub(crate) const MAX_FRAME_CONTENT: u64 = u32::MAX as u64 - FRAME_HEADER as u64;

/// Bytes of the varint encoding `value`.
#[cfg(feature = "compression")]
pub(crate) const fn varint_len(value: u64) -> usize {
    let bits = 64 - (value | 1).leading_zeros() as usize;
    bits.div_ceil(7)
}

/// The flags byte permitted for a data chunk type.
#[cfg(feature = "decompression")]
pub(crate) const fn data_flags(kind: u8) -> u8 {
    match kind {
        2 => FLAG_HIDDEN | FLAG_CHECKSUM | FLAG_DETACHED,
        3 => FLAG_HIDDEN | FLAG_DETACHED,
        4 => FLAG_DETACHED,
        5 => FLAG_CHECKSUM | FLAG_DETACHED,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "compression")]
    #[test]
    fn varint_lengths_follow_seven_bit_groups() {
        assert_eq!(varint_len(0), 1);
        assert_eq!(varint_len(127), 1);
        assert_eq!(varint_len(128), 2);
        assert_eq!(varint_len(16_383), 2);
        assert_eq!(varint_len(16_384), 3);
        assert_eq!(varint_len(MAX_VARINT), 9);
    }

    #[cfg(feature = "decompression")]
    #[test]
    fn data_flags_permit_hidden_first_checksum_last_and_detached_bodies() {
        assert_eq!(data_flags(2), 7);
        assert_eq!(data_flags(3), 5);
        assert_eq!(data_flags(4), 4);
        assert_eq!(data_flags(5), 6);
        assert_eq!(data_flags(1), 0);
        assert_eq!(CONTAINER_HEADER, 13);
    }
}

//! Wire constants of the Zstandard frame format (RFC 8878) shared by both
//! codecs.

/// Magic number opening every Zstandard frame.
pub(crate) const MAGIC_NUMBER: u32 = 0xFD2F_B528;
/// First of the sixteen skippable-frame magic numbers.
#[cfg(feature = "decompression")]
pub(crate) const MAGIC_SKIPPABLE_START: u32 = 0x184D_2A50;
/// Mask selecting the skippable-frame magic family.
#[cfg(feature = "decompression")]
pub(crate) const MAGIC_SKIPPABLE_MASK: u32 = 0xFFFF_FFF0;
/// Size of a skippable frame's magic number and length field.
#[cfg(feature = "decompression")]
pub(crate) const SKIPPABLE_HEADER_SIZE: usize = 8;

/// Size of a block header.
pub(crate) const BLOCK_HEADER_SIZE: usize = 3;
/// Largest block content: 128 KiB.
pub(crate) const BLOCK_SIZE_MAX: usize = 1 << 17;
/// Size of the optional content checksum.
#[cfg(feature = "decompression")]
pub(crate) const CHECKSUM_SIZE: usize = 4;
/// Largest frame header: magic, descriptor, window, dictionary ID and size.
pub(crate) const FRAME_HEADER_SIZE_MAX: usize = 18;
/// Smallest window a frame header can describe, as a power of two.
pub(crate) const WINDOW_LOG_ABSOLUTE_MIN: u32 = 10;

/// Block types (RFC 8878 section 3.1.1.2.2).
pub(crate) const BLOCK_RAW: u32 = 0;
pub(crate) const BLOCK_RLE: u32 = 1;
pub(crate) const BLOCK_COMPRESSED: u32 = 2;

/// Literals and sequence table encodings.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum EncodingType {
    /// Predefined distribution, or raw literals.
    Basic = 0,
    /// A single repeated symbol.
    Rle = 1,
    /// A table described in the stream.
    Compressed = 2,
    /// The previous block's table.
    Repeat = 3,
}

/// Number of repeat offsets.
pub(crate) const REP_NUM: usize = 3;
/// Repeat offsets at the start of a frame.
pub(crate) const REP_START: [u32; REP_NUM] = [1, 4, 8];

/// Shortest match the format can express.
pub(crate) const MIN_MATCH: usize = 3;

/// Above this many sequences the count takes three bytes.
pub(crate) const LONG_NB_SEQ: usize = 0x7F00;

/// Literals alphabet size minus one.
pub(crate) const MAX_LIT: u32 = 255;
/// Largest match length code.
pub(crate) const MAX_ML: u32 = 52;
/// Largest literal length code.
pub(crate) const MAX_LL: u32 = 35;
/// Largest offset code the predefined distribution covers.
pub(crate) const DEFAULT_MAX_OFF: u32 = 28;
/// Largest offset code.
pub(crate) const MAX_OFF: u32 = 31;
/// Largest of the length code alphabets.
pub(crate) const MAX_SEQ: u32 = 52;
/// Accuracy logs of the sequence tables.
pub(crate) const ML_FSE_LOG: u32 = 9;
pub(crate) const LL_FSE_LOG: u32 = 9;
pub(crate) const OFF_FSE_LOG: u32 = 8;
/// Largest Huffman table log for literals.
pub(crate) const LIT_HUF_LOG: u32 = 11;

/// Extra bits per literal length code.
pub(crate) const LL_BITS: [u8; MAX_LL as usize + 1] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16,
];
/// Predefined literal length distribution.
pub(crate) const LL_DEFAULT_NORM: [i16; MAX_LL as usize + 1] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1,
];
pub(crate) const LL_DEFAULT_NORM_LOG: u32 = 6;

/// Extra bits per match length code.
pub(crate) const ML_BITS: [u8; MAX_ML as usize + 1] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];
/// Predefined match length distribution.
pub(crate) const ML_DEFAULT_NORM: [i16; MAX_ML as usize + 1] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];
pub(crate) const ML_DEFAULT_NORM_LOG: u32 = 6;

/// Predefined offset code distribution.
pub(crate) const OF_DEFAULT_NORM: [i16; DEFAULT_MAX_OFF as usize + 1] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];
pub(crate) const OF_DEFAULT_NORM_LOG: u32 = 5;

/// Position of the highest set bit; `value` must be non-zero.
#[inline]
pub(crate) const fn highbit32(value: u32) -> u32 {
    31 - value.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predefined_distributions_sum_to_their_table_size() {
        let sum = |norm: &[i16]| norm.iter().map(|&n| i32::from(n).abs()).sum::<i32>();
        assert_eq!(sum(&LL_DEFAULT_NORM), 1 << LL_DEFAULT_NORM_LOG);
        assert_eq!(sum(&ML_DEFAULT_NORM), 1 << ML_DEFAULT_NORM_LOG);
        assert_eq!(sum(&OF_DEFAULT_NORM), 1 << OF_DEFAULT_NORM_LOG);
    }

    #[test]
    fn highbit_is_the_floor_log2() {
        assert_eq!(highbit32(1), 0);
        assert_eq!(highbit32(2), 1);
        assert_eq!(highbit32(255), 7);
        assert_eq!(highbit32(u32::MAX), 31);
    }

    #[test]
    fn encoding_types_carry_their_wire_values() {
        assert_eq!(EncodingType::Basic as u8, 0);
        assert_eq!(EncodingType::Rle as u8, 1);
        assert_eq!(EncodingType::Compressed as u8, 2);
        assert_eq!(EncodingType::Repeat as u8, 3);
    }
}

//! Command tables specific to the quality 0 and quality 1 fast encoders.
//!
//! Source: <https://github.com/google/brotli/tree/028fb5a> (v1.2.0), files
//! `c/enc/compress_fragment.c`, `c/enc/compress_fragment_two_pass.c` and
//! `c/enc/encode.c`. Distributed by Google under the MIT licence; see
//! `brotli-ffi/vendor/brotli/LICENSE`.
//!
//! Tables every quality shares live in
//! [`crate::shared::tables`].

/// Seed counts the quality 0 command histogram starts each block with.
pub(crate) const CMD_HISTO_SEED: [u32; 128] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
];

/// Bit depths of the quality 0 first-block command code
/// (`kDefaultCommandDepths`).
pub(crate) const DEFAULT_COMMAND_DEPTHS: [u8; 128] = [
    0, 4, 4, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 0, 0, 0, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7,
    7, 7, 10, 10, 10, 10, 10, 10, 0, 4, 4, 5, 5, 5, 6, 6, 7, 8, 8, 9, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 5, 5, 5,
    5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 8, 10, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 0, 0, 0, 0,
];

/// Bit patterns of the quality 0 first-block command code
/// (`kDefaultCommandBits`).
pub(crate) const DEFAULT_COMMAND_BITS: [u16; 128] = [
    0, 0, 8, 9, 3, 35, 7, 71, 39, 103, 23, 47, 175, 111, 239, 31, 0, 0, 0, 4, 12, 2, 10, 6, 13, 29,
    11, 43, 27, 59, 87, 55, 15, 79, 319, 831, 191, 703, 447, 959, 0, 14, 1, 25, 5, 21, 19, 51, 119,
    159, 95, 223, 479, 991, 63, 575, 127, 639, 383, 895, 255, 767, 511, 1023, 14, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 59, 7, 39, 23, 55, 30, 1, 17, 9, 25, 5, 0, 8, 4, 12, 2, 10, 6,
    21, 13, 29, 3, 19, 11, 15, 47, 31, 95, 63, 127, 255, 767, 2815, 1791, 3839, 511, 2559, 1535,
    3583, 1023, 3071, 2047, 4095, 0, 0, 0, 0,
];

/// Pre-compressed quality 0 first-block command code (`kDefaultCommandCode`).
pub(crate) const DEFAULT_COMMAND_CODE: [u8; 57] = [
    0xff, 0x77, 0xd5, 0xbf, 0xe7, 0xde, 0xea, 0x9e, 0x51, 0x5d, 0xde, 0xc6, 0x70, 0x57, 0xbc, 0x58,
    0x58, 0x58, 0xd8, 0xd8, 0x58, 0xd5, 0xcb, 0x8c, 0xea, 0xe0, 0xc3, 0x87, 0x1f, 0x83, 0xc1, 0x60,
    0x1c, 0x67, 0xb2, 0xaa, 0x06, 0x83, 0xc1, 0x60, 0x30, 0x18, 0xcc, 0xa1, 0xce, 0x88, 0x54, 0x94,
    0x46, 0xe1, 0xb0, 0xd0, 0x4e, 0xb2, 0xf7, 0x04, 0x00,
];

/// Number of valid bits in [`DEFAULT_COMMAND_CODE`].
pub(crate) const DEFAULT_COMMAND_CODE_NUM_BITS: usize = 448;

/// Extra-bit counts of the quality 1 compact command alphabet
/// (`kNumExtraBits`).
pub(crate) const NUM_EXTRA_BITS: [u32; 128] = [
    0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12, 14, 24, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9,
    10, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7,
    7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19,
    20, 20, 21, 21, 22, 22, 23, 23, 24, 24,
];

/// Insert-length bases of the quality 1 compact command alphabet
/// (`kInsertOffset`).
pub(crate) const INSERT_OFFSET: [u32; 24] = [
    0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 26, 34, 50, 66, 98, 130, 194, 322, 578, 1090, 2114, 6210,
    22594,
];

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn command_tables_have_the_reference_lengths_and_checksums() {
        assert_eq!(NUM_EXTRA_BITS.len(), 128);
        assert_eq!(INSERT_OFFSET.len(), 24);
        assert_eq!(DEFAULT_COMMAND_DEPTHS.len(), 128);
        assert_eq!(DEFAULT_COMMAND_BITS.len(), 128);
        assert_eq!(CMD_HISTO_SEED.len(), 128);
        assert_eq!(DEFAULT_COMMAND_CODE.len(), 57);
        assert_eq!(DEFAULT_COMMAND_CODE_NUM_BITS, 448);

        assert_eq!(NUM_EXTRA_BITS.iter().sum::<u32>(), 834);
        assert_eq!(INSERT_OFFSET.iter().sum::<u32>(), 33577);
        assert_eq!(
            DEFAULT_COMMAND_DEPTHS
                .iter()
                .map(|&d| u32::from(d))
                .sum::<u32>(),
            753
        );
        assert_eq!(
            DEFAULT_COMMAND_BITS
                .iter()
                .map(|&b| u32::from(b))
                .sum::<u32>(),
            40961
        );
        assert_eq!(CMD_HISTO_SEED.iter().sum::<u32>(), 104);
        assert_eq!(
            DEFAULT_COMMAND_CODE
                .iter()
                .map(|&b| u32::from(b))
                .sum::<u32>(),
            8285
        );
    }
}

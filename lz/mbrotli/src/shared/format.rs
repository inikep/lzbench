//! Constant tables translated from Google's Brotli reference encoder.
//!
//! Source: <https://github.com/google/brotli/tree/028fb5a> (v1.2.0), files
//! `c/common/context.c`, `c/common/constants.c`, `c/enc/command.c` and
//! `c/enc/encode.c`. Distributed by Google under the MIT licence; see
//! `brotli-ffi/vendor/brotli/LICENSE`.

/// Context lookup table for `CONTEXT_UTF8`.
///
/// The context of a literal is `LUT[p1] | LUT[256 + p2]`, where `p1` and `p2`
/// are the two preceding bytes. `CONTEXT_LSB6` and `CONTEXT_MSB6` are the
/// remaining two tables of `_kBrotliContextLookupTable`; no quality this
/// encoder implements ever selects them.
pub(crate) const CONTEXT_LUT_UTF8: [u8; 512] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    8, 12, 16, 12, 12, 20, 12, 16, 24, 28, 12, 12, 32, 12, 36, 12, 44, 44, 44, 44, 44, 44, 44, 44,
    44, 44, 32, 32, 24, 40, 28, 12, 12, 48, 52, 52, 52, 48, 52, 52, 52, 48, 52, 52, 52, 52, 52, 48,
    52, 52, 52, 52, 52, 48, 52, 52, 52, 52, 52, 24, 12, 28, 12, 12, 12, 56, 60, 60, 60, 56, 60, 60,
    60, 56, 60, 60, 60, 60, 60, 56, 60, 60, 60, 60, 60, 56, 60, 60, 60, 60, 60, 24, 12, 28, 12, 0,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
    2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
    1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
];

/// Context lookup table for `CONTEXT_SIGNED`.
///
/// `ChooseContextMode` picks this over [`CONTEXT_LUT_UTF8`] at quality ten and
/// above when the block is not mostly UTF-8; it buckets the two preceding
/// bytes by magnitude rather than by character class.
pub(crate) const CONTEXT_LUT_SIGNED: [u8; 512] = [
    0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 48, 48, 48, 48,
    48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 56, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7,
];

/// The literal context model a meta-block is coded with (`ContextType`).
///
/// The numeric values are what `BrotliStoreMetaBlock` writes into the header,
/// so they are format, not implementation.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
#[cfg(feature = "compression")]
pub(crate) enum ContextMode {
    /// Second-order model tuned for UTF-8 text (`CONTEXT_UTF8`).
    #[default]
    Utf8,
    /// Second-order model tuned for signed integers (`CONTEXT_SIGNED`).
    Signed,
}

#[cfg(feature = "compression")]
impl ContextMode {
    /// Returns the lookup table this mode computes contexts from.
    pub(crate) const fn lut(self) -> &'static [u8; 512] {
        match self {
            Self::Utf8 => &CONTEXT_LUT_UTF8,
            Self::Signed => &CONTEXT_LUT_SIGNED,
        }
    }

    /// Returns the two-bit code the meta-block header carries.
    pub(crate) const fn code(self) -> u64 {
        match self {
            Self::Utf8 => 2,
            Self::Signed => 3,
        }
    }

    /// Returns the context of a literal preceded by `prev1` and `prev2`.
    ///
    /// Mirrors `BROTLI_CONTEXT`.
    #[inline(always)]
    pub(crate) fn context(self, prev1: u8, prev2: u8) -> usize {
        let lut = self.lut();
        usize::from(lut[usize::from(prev1)] | lut[256 + usize::from(prev2)])
    }
}

/// Number of insert-and-copy length codes (`BROTLI_NUM_INS_COPY_CODES`).
pub(crate) const NUM_INS_COPY_CODES: usize = 24;

/// Insert-length code bases (`kBrotliInsBase`).
pub(crate) const INS_BASE: [u32; NUM_INS_COPY_CODES] = [
    0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 26, 34, 50, 66, 98, 130, 194, 322, 578, 1090, 2114, 6210,
    22594,
];

/// Insert-length code extra-bit counts (`kBrotliInsExtra`).
pub(crate) const INS_EXTRA: [u32; NUM_INS_COPY_CODES] = [
    0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12, 14, 24,
];

/// Copy-length code bases (`kBrotliCopyBase`).
pub(crate) const COPY_BASE: [u32; NUM_INS_COPY_CODES] = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18, 22, 30, 38, 54, 70, 102, 134, 198, 326, 582, 1094, 2118,
];

/// Copy-length code extra-bit counts (`kBrotliCopyExtra`).
pub(crate) const COPY_EXTRA: [u32; NUM_INS_COPY_CODES] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 24,
];

/// Number of block-length symbols (`BROTLI_NUM_BLOCK_LEN_SYMBOLS`).
pub(crate) const NUM_BLOCK_LEN_SYMBOLS: usize = 26;

/// Block-length prefix code ranges (`_kBrotliPrefixCodeRanges`).
///
/// Each entry is `(offset, nbits)`: the code covers
/// `offset..offset + (1 << nbits)`.
pub(crate) const PREFIX_CODE_RANGES: [(u32, u32); NUM_BLOCK_LEN_SYMBOLS] = [
    (1, 2),
    (5, 2),
    (9, 2),
    (13, 2),
    (17, 3),
    (25, 3),
    (33, 3),
    (41, 3),
    (49, 4),
    (65, 4),
    (81, 4),
    (97, 4),
    (113, 5),
    (145, 5),
    (177, 5),
    (209, 5),
    (241, 6),
    (305, 6),
    (369, 7),
    (497, 8),
    (753, 9),
    (1265, 10),
    (2289, 11),
    (4337, 12),
    (8433, 13),
    (16625, 24),
];

/// Two-context map over UTF-8 prefixes (`kStaticContextMapSimpleUTF8`).
#[cfg(feature = "compression")]
pub(crate) const STATIC_CONTEXT_MAP_SIMPLE_UTF8: [u32; 64] = [
    0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Three-context map over UTF-8 prefixes (`kStaticContextMapContinuation`).
#[cfg(feature = "compression")]
pub(crate) const STATIC_CONTEXT_MAP_CONTINUATION: [u32; 64] = [
    1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Number of contexts the complex static map distinguishes.
#[cfg(feature = "compression")]
pub(crate) const MAX_STATIC_CONTEXTS: usize = 13;

/// Thirteen-context map over UTF-8 prefixes (`kStaticContextMapComplexUTF8`).
///
/// The rows group the source classes: special, line feed, space, punctuation,
/// quotes, percent, opening and closing brackets, colons, full stop, greater
/// than, digits, upper case and lower case.
#[cfg(feature = "compression")]
pub(crate) const STATIC_CONTEXT_MAP_COMPLEX_UTF8: [u32; 64] = [
    11, 11, 12, 12, //
    0, 0, 0, 0, //
    1, 1, 9, 9, //
    2, 2, 2, 2, //
    1, 1, 1, 1, //
    8, 3, 3, 3, //
    1, 1, 1, 1, //
    2, 2, 2, 2, //
    8, 4, 4, 4, //
    8, 7, 4, 4, //
    8, 0, 0, 0, //
    3, 3, 3, 3, //
    5, 5, 10, 5, //
    5, 5, 10, 5, //
    6, 6, 6, 6, //
    6, 6, 6, 6, //
];

#[cfg(all(test, feature = "compression"))]
mod tests {
    use super::*;

    #[test]
    fn context_lut_matches_the_reference_checksum() {
        assert_eq!(CONTEXT_LUT_UTF8.len(), 512);
        assert_eq!(
            CONTEXT_LUT_UTF8.iter().map(|&v| u32::from(v)).sum::<u32>(),
            4394
        );
        assert_eq!(CONTEXT_LUT_UTF8[usize::from(b'a')], 56);
        assert_eq!(CONTEXT_LUT_UTF8[usize::from(b'b')], 60);
        assert_eq!(CONTEXT_LUT_UTF8[256 + usize::from(b'a')], 3);
        assert_eq!(CONTEXT_LUT_UTF8[usize::from(b' ')], 8);
    }

    #[test]
    fn command_tables_have_the_reference_lengths_and_checksums() {
        assert_eq!(INS_BASE.iter().sum::<u32>(), 33_577);
        assert_eq!(COPY_BASE.iter().sum::<u32>(), 4_866);
        assert_eq!(INS_EXTRA.iter().sum::<u32>(), 120);
        assert_eq!(COPY_EXTRA.iter().sum::<u32>(), 94);
    }

    #[test]
    fn prefix_code_ranges_are_contiguous() {
        let mut next = 1u32;
        for &(offset, nbits) in &PREFIX_CODE_RANGES {
            assert_eq!(offset, next);
            next = offset + (1u32 << nbits);
        }
    }

    #[test]
    fn static_context_maps_stay_within_their_context_counts() {
        assert!(STATIC_CONTEXT_MAP_SIMPLE_UTF8.iter().all(|&v| v < 2));
        assert!(STATIC_CONTEXT_MAP_CONTINUATION.iter().all(|&v| v < 3));
        assert!(
            STATIC_CONTEXT_MAP_COMPLEX_UTF8
                .iter()
                .all(|&v| (v as usize) < MAX_STATIC_CONTEXTS)
        );
    }
}

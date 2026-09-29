//! The distance alphabet: its parameters and the constants that bound it.
//!
//! Ports `BrotliInitDistanceParams` from `c/enc/metablock.c` and the distance
//! constants of `c/common/constants.h` from the pinned reference
//! (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! Every quality from three upwards writes distances through this alphabet;
//! qualities four and above may also tune it per meta-block, which is why the
//! parameters are a value rather than a global.

/// Largest number of postfix bits the format allows (`BROTLI_MAX_NPOSTFIX`).
pub(crate) const MAX_NPOSTFIX: u32 = 3;

/// Largest number of direct distance codes (`BROTLI_MAX_NDIRECT`).
pub(crate) const MAX_NDIRECT: u32 = 120;

/// Number of short distance codes (`BROTLI_NUM_DISTANCE_SHORT_CODES`).
pub(crate) const NUM_DISTANCE_SHORT_CODES: u32 = 16;

/// Largest number of distance bits in RFC 7932 (`BROTLI_MAX_DISTANCE_BITS`).
pub(crate) const MAX_DISTANCE_BITS: u32 = 24;

/// Largest number of distance bits in RFC 9841
/// (`BROTLI_LARGE_MAX_DISTANCE_BITS`).
///
/// This is a property of the format rather than of any encoder: it fixes how
/// wide a distance symbol may be, and therefore how many bits a simple prefix
/// code spends per distance symbol in a large-window stream.
pub(crate) const LARGE_MAX_DISTANCE_BITS: u32 = 62;

/// Largest distance a large-window stream may express
/// (`BROTLI_MAX_ALLOWED_DISTANCE`).
///
/// RFC 9841 permits wider distances still, but stopping at `(1 << 31) - 4`
/// keeps every distance calculation inside a signed 32-bit range, which is
/// what makes a 32-bit decoder safe. The reference encoder draws the line in
/// the same place.
pub(crate) const MAX_ALLOWED_DISTANCE: u32 = 0x7FFF_FFFC;

/// Distance symbols a histogram reserves
/// (`BROTLI_NUM_HISTOGRAM_DISTANCE_SYMBOLS`).
pub(crate) const NUM_HISTOGRAM_DISTANCE_SYMBOLS: usize = 544;

/// Distance alphabet size assuming no postfix or direct codes.
///
/// `MAX_SIMPLE_DISTANCE_ALPHABET_SIZE` in `c/enc/brotli_bit_stream.c`, computed
/// for the large-window bit count even though this encoder never uses it.
pub(crate) const MAX_SIMPLE_DISTANCE_ALPHABET_SIZE: usize = 140;

/// Resolved distance alphabet (`BrotliDistanceParams`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct DistanceParams {
    /// Number of postfix bits, `NPOSTFIX`.
    pub(crate) postfix_bits: u32,
    /// Number of direct distance codes, `NDIRECT`.
    pub(crate) num_direct: u32,
    /// Size of the distance alphabet that is written to the stream.
    pub(crate) alphabet_size_max: u32,
    /// Size of the distance alphabet that can actually occur.
    pub(crate) alphabet_size_limit: u32,
    /// Largest distance this alphabet can express.
    pub(crate) max_distance: u32,
}

impl DistanceParams {
    /// Builds the RFC 7932 alphabet for `postfix_bits` and `num_direct`.
    ///
    /// Mirrors `BrotliInitDistanceParams` with `large_window` false: every
    /// symbol the alphabet can express is also one the stream may use, so the
    /// limit and the maximum alphabet coincide.
    pub(crate) const fn new(postfix_bits: u32, num_direct: u32) -> Self {
        let alphabet_size_max =
            NUM_DISTANCE_SHORT_CODES + num_direct + (MAX_DISTANCE_BITS << (postfix_bits + 1));
        let max_distance = num_direct + (1u32 << (MAX_DISTANCE_BITS + postfix_bits + 2))
            - (1u32 << (postfix_bits + 2));
        Self {
            postfix_bits,
            num_direct,
            alphabet_size_max,
            alphabet_size_limit: alphabet_size_max,
            max_distance,
        }
    }

    /// Builds the RFC 9841 large-window alphabet.
    ///
    /// Mirrors `BrotliInitDistanceParams` with `large_window` true. Two things
    /// change against [`DistanceParams::new`]: the alphabet written to the
    /// stream is sized for 62 distance bits rather than 24, and the symbols
    /// that may actually occur stop at [`MAX_ALLOWED_DISTANCE`]. The two are no
    /// longer equal, which is why the meta-block writer keeps them apart.
    pub(crate) const fn new_large(postfix_bits: u32, num_direct: u32) -> Self {
        let alphabet_size_max =
            NUM_DISTANCE_SHORT_CODES + num_direct + (LARGE_MAX_DISTANCE_BITS << (postfix_bits + 1));
        let limit = distance_code_limit(MAX_ALLOWED_DISTANCE, postfix_bits, num_direct);
        Self {
            postfix_bits,
            num_direct,
            alphabet_size_max,
            alphabet_size_limit: limit.max_alphabet_size,
            max_distance: limit.max_distance,
        }
    }

    /// Builds the alphabet a stream with this window uses.
    pub(crate) const fn for_window(large_window: bool, postfix_bits: u32, num_direct: u32) -> Self {
        if large_window {
            Self::new_large(postfix_bits, num_direct)
        } else {
            Self::new(postfix_bits, num_direct)
        }
    }
}

/// The largest alphabet that cannot express a distance past a given limit.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct DistanceCodeLimit {
    /// Number of distance symbols that alphabet holds.
    pub(crate) max_alphabet_size: u32,
    /// Largest distance those symbols can express.
    pub(crate) max_distance: u32,
}

/// Returns the largest alphabet whose distances all stay within `max_distance`.
///
/// Ports `BrotliCalculateDistanceCodeLimit` from `c/common/constants.h`.
///
/// Distance symbols above the short codes and the direct codes do not cover
/// consecutive distances: `1 << NPOSTFIX` consecutive symbols form one
/// interleaved group, and two consecutive groups share an extra-bit width. The
/// alphabet is cut at a group boundary so that neither side ever has to reason
/// about a half-represented group.
///
/// `max_distance` is expected to leave room for `max_distance + 1`, which
/// [`MAX_ALLOWED_DISTANCE`] — the only value production code passes — does with
/// a whole bit to spare. `num_direct` is at most [`MAX_NDIRECT`] and
/// `postfix_bits` at most [`MAX_NPOSTFIX`], so every other sum and shift here
/// is bounded by construction.
const fn distance_code_limit(
    max_distance: u32,
    postfix_bits: u32,
    num_direct: u32,
) -> DistanceCodeLimit {
    if max_distance <= num_direct {
        // Only reachable for a limit smaller than the direct codes, which no
        // caller can currently ask for; kept so the function is total.
        return DistanceCodeLimit {
            max_alphabet_size: max_distance + NUM_DISTANCE_SHORT_CODES,
            max_distance,
        };
    }
    // Work from the first distance the alphabet must not be able to express,
    // with the directly encoded region and the postfix interleaving removed.
    let forbidden = max_distance + 1;
    let postfix = (1u32 << postfix_bits) - 1;
    let offset = ((forbidden - num_direct - 1) >> postfix_bits) + 4;

    // Floor of the base-2 logarithm of `offset / 2`.
    let mut distance_bits = 0u32;
    let mut rest = offset / 2;
    while rest != 0 {
        distance_bits += 1;
        rest >>= 1;
    }
    distance_bits -= 1;

    // One bit of the range is addressed by the subrange ("half") instead.
    // `offset >= 4` forces `distance_bits >= 1`, so the shift below never
    // underflows.
    let half = (offset >> distance_bits) & 1;
    let group = ((distance_bits - 1) << 1) | half;
    if group == 0 {
        // Cannot happen for a limit above 128; kept for completeness.
        return DistanceCodeLimit {
            max_alphabet_size: num_direct + NUM_DISTANCE_SHORT_CODES,
            max_distance: num_direct,
        };
    }
    // That group covers the forbidden distance, so the last permitted group is
    // the one below it, and the extra-bit width has to be recomputed for it.
    let group = group - 1;
    let distance_bits = (group >> 1) + 1;
    let extra = (1u32 << distance_bits) - 1;
    let start = (1u32 << (distance_bits + 1)) - 4 + ((group & 1) << distance_bits);

    DistanceCodeLimit {
        max_alphabet_size: ((group << postfix_bits) | postfix)
            + num_direct
            + NUM_DISTANCE_SHORT_CODES
            + 1,
        max_distance: ((start + extra) << postfix_bits) + postfix + num_direct + 1,
    }
}

impl Default for DistanceParams {
    /// Returns the alphabet with neither postfix nor direct codes.
    fn default() -> Self {
        Self::new(0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Every `(NPOSTFIX, NDIRECT)` pair the format can express.
    fn legal_pairs() -> Vec<(u32, u32)> {
        let mut pairs = Vec::new();
        for postfix_bits in 0..=MAX_NPOSTFIX {
            for direct_codes in 0..=MAX_NDIRECT {
                let groups = (direct_codes >> postfix_bits) & 0x0F;
                if (groups << postfix_bits) == direct_codes {
                    pairs.push((postfix_bits, direct_codes));
                }
            }
        }
        pairs
    }

    #[test]
    fn the_ordinary_alphabet_is_unchanged_by_the_extension() {
        let default = DistanceParams::new(0, 0);
        assert_eq!(default.alphabet_size_max, 64);
        assert_eq!(default.alphabet_size_limit, 64);
        assert_eq!(default.max_distance, (1 << 26) - 4);
        assert_eq!(DistanceParams::for_window(false, 0, 0), default);
    }

    #[test]
    fn the_large_alphabet_is_sized_for_sixty_two_distance_bits() {
        let large = DistanceParams::new_large(0, 0);
        assert_eq!(large.alphabet_size_max, 16 + (62 << 1));
        assert_eq!(large.alphabet_size_limit, 74);
        assert_eq!(large.max_distance, MAX_ALLOWED_DISTANCE);
        assert_eq!(DistanceParams::for_window(true, 0, 0), large);
    }

    #[test]
    fn the_large_alphabet_never_outgrows_a_distance_histogram() {
        for (postfix_bits, direct_codes) in legal_pairs() {
            let large = DistanceParams::new_large(postfix_bits, direct_codes);
            assert!(
                large.alphabet_size_limit as usize <= NUM_HISTOGRAM_DISTANCE_SYMBOLS,
                "npostfix {postfix_bits}, ndirect {direct_codes}"
            );
            assert!(
                large.alphabet_size_limit <= large.alphabet_size_max,
                "npostfix {postfix_bits}, ndirect {direct_codes}"
            );
        }
        // The 544-symbol histogram is exactly the widest legal alphabet.
        assert_eq!(
            DistanceParams::new_large(MAX_NPOSTFIX, 120).alphabet_size_limit as usize,
            NUM_HISTOGRAM_DISTANCE_SYMBOLS
        );
    }

    #[test]
    fn a_large_alphabet_reaches_further_than_the_ordinary_one() {
        for (postfix_bits, direct_codes) in legal_pairs() {
            let ordinary = DistanceParams::new(postfix_bits, direct_codes);
            let large = DistanceParams::new_large(postfix_bits, direct_codes);
            assert!(
                large.max_distance >= ordinary.max_distance,
                "npostfix {postfix_bits}, ndirect {direct_codes}"
            );
            assert!(
                large.alphabet_size_max > ordinary.alphabet_size_max,
                "npostfix {postfix_bits}, ndirect {direct_codes}"
            );
            assert!(
                large.max_distance <= MAX_ALLOWED_DISTANCE,
                "npostfix {postfix_bits}, ndirect {direct_codes}"
            );
        }
    }

    #[test]
    fn a_limit_below_the_direct_codes_degenerates_cleanly() {
        assert_eq!(
            distance_code_limit(8, 0, 16),
            DistanceCodeLimit {
                max_alphabet_size: 8 + NUM_DISTANCE_SHORT_CODES,
                max_distance: 8,
            }
        );
        // A limit small enough that no complete group survives.
        assert_eq!(
            distance_code_limit(9, 0, 8),
            DistanceCodeLimit {
                max_alphabet_size: 8 + NUM_DISTANCE_SHORT_CODES,
                max_distance: 8,
            }
        );
    }

    #[test]
    fn the_default_alphabet_is_the_one_without_postfix_or_direct_codes() {
        assert_eq!(DistanceParams::default(), DistanceParams::new(0, 0));
    }
}

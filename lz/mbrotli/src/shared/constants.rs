//! Format constants shared by every encoder in this crate.
//!
//! Every value here is fixed by RFC 7932 and by Google's Brotli reference
//! encoder (`google/brotli` v1.2.0, commit `028fb5a`, MIT licence). Changing
//! any of them changes the emitted bitstream, so they are asserted in
//! [`tests`] rather than being treated as tunables.

/// Multiplier used by both fast hash functions (`c/enc/hash_base.h`).
pub(crate) const HASH_MUL32: u32 = 0x1E35_A7BD;

/// Bytes of the sliding window reserved as a margin (`BROTLI_WINDOW_GAP`).
pub(crate) const WINDOW_GAP: usize = 16;

/// Number of symbols in the full Brotli command alphabet.
pub(crate) const NUM_COMMAND_SYMBOLS: usize = 704;

/// Number of symbols in the Brotli literal alphabet.
pub(crate) const NUM_LITERAL_SYMBOLS: usize = 256;

/// Number of symbols in the Brotli code-length alphabet.
pub(crate) const CODE_LENGTH_CODES: usize = 18;

/// Code-length symbol that repeats the previous non-zero length.
pub(crate) const REPEAT_PREVIOUS_CODE_LENGTH: usize = 16;

/// Code-length symbol that repeats a run of zero lengths.
pub(crate) const REPEAT_ZERO_CODE_LENGTH: usize = 17;

/// Code length assumed to precede the first emitted length.
pub(crate) const INITIAL_REPEATED_CODE_LENGTH: u8 = 8;

/// Bytes of slack a fast-path output buffer needs beyond `2 * len + 503`.
///
/// The bit writer materialises up to eight bytes around the current bit
/// position, so the scratch buffer keeps one machine word of headroom.
pub(crate) const OUTPUT_SLACK: usize = 8;

/// Constant term of the per-block output reservation, `2 * len + 503`.
pub(crate) const OUTPUT_RESERVE_CONST: usize = 503;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_the_reference_encoder() {
        assert_eq!(HASH_MUL32, 0x1E35_A7BD);
        assert_eq!(WINDOW_GAP, 16);
        assert_eq!(NUM_LITERAL_SYMBOLS, 256);
        assert_eq!(NUM_COMMAND_SYMBOLS, 704);
        assert_eq!(CODE_LENGTH_CODES, 18);
        assert_eq!(REPEAT_PREVIOUS_CODE_LENGTH, 16);
        assert_eq!(REPEAT_ZERO_CODE_LENGTH, 17);
        assert_eq!(INITIAL_REPEATED_CODE_LENGTH, 8);
        assert_eq!(OUTPUT_RESERVE_CONST, 503);
        assert_eq!(OUTPUT_SLACK, 8);
    }
}

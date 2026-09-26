//! The base-128 integer encoding every RFC 9841 container field is built on.
//!
//! [RFC 9841 section 4] defines one integer encoding and uses it in two places:
//! the serialized dictionary stream reads it forwards, and the framing format's
//! final footer reads it backwards. Both live here so the two can never
//! disagree about what a byte sequence means.
//!
//! A varint is a little-endian base-128 sequence: every byte carries seven
//! value bits, and the high bit says whether another byte follows. The RFC caps
//! a varint at nine bytes and sixty-three value bits, which is what makes
//! `u64` the natural carrier and also what makes the ninth byte a special case
//! — nine bytes are exactly sixty-three bits, so a ninth byte that claims a
//! tenth is invalid rather than merely large.
//!
//! [RFC 9841 section 4]: https://www.rfc-editor.org/rfc/rfc9841.html#section-4

use alloc::vec::Vec;

use thiserror::Error;

/// Most bytes one varint may occupy (`63 / 7`).
pub(crate) const MAX_VARINT_BYTES: usize = 9;

/// Largest value a varint may carry, from the RFC's sixty-three-bit cap.
pub(crate) const MAX_VARINT: u64 = (1u64 << 63) - 1;

/// Why a varint could not be read or written.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum VarintError {
    /// The bytes ran out before a terminating byte was seen.
    #[error("a varint was cut off after {available} byte(s)")]
    Truncated {
        /// How many bytes were left to read.
        available: usize,
    },
    /// A ninth byte asked for a tenth, so the value would exceed 63 bits.
    #[error("a varint may not exceed {MAX_VARINT_BYTES} bytes or 63 value bits")]
    Overlong,
    /// A value larger than the encoding can carry was offered for writing.
    #[error("{value} exceeds the largest varint value of {MAX_VARINT}")]
    OutOfRange {
        /// The value that was offered.
        value: u64,
    },
}

/// Returns how many bytes `value` occupies once encoded.
///
/// Defined for every `u64`, so a caller can size a buffer before checking the
/// range; [`write`] is what refuses an out-of-range value.
pub(crate) const fn encoded_len(value: u64) -> usize {
    let mut len = 1;
    let mut rest = value >> 7;
    while rest != 0 {
        len += 1;
        rest >>= 7;
    }
    len
}

/// Reads one varint from the front of `bytes`.
///
/// Returns the value and how many bytes it occupied.
///
/// # Errors
///
/// Returns [`VarintError::Truncated`] when no byte in `bytes` terminates the
/// sequence, and [`VarintError::Overlong`] when a ninth byte sets its
/// continuation bit.
pub(crate) fn read(bytes: &[u8]) -> Result<(u64, usize), VarintError> {
    let mut value = 0u64;
    for (index, &byte) in bytes.iter().take(MAX_VARINT_BYTES).enumerate() {
        value |= u64::from(byte & 0x7F) << (7 * index);
        if byte & 0x80 == 0 {
            return Ok((value, index + 1));
        }
        if index + 1 == MAX_VARINT_BYTES {
            return Err(VarintError::Overlong);
        }
    }
    Err(VarintError::Truncated {
        available: bytes.len(),
    })
}

/// Appends `value` to `out` as a varint.
///
/// # Errors
///
/// Returns [`VarintError::OutOfRange`] past [`MAX_VARINT`].
pub(crate) fn write(value: u64, out: &mut Vec<u8>) -> Result<(), VarintError> {
    if value > MAX_VARINT {
        return Err(VarintError::OutOfRange { value });
    }
    let mut rest = value;
    loop {
        let byte = (rest & 0x7F) as u8;
        rest >>= 7;
        if rest == 0 {
            out.push(byte);
            return Ok(());
        }
        out.push(byte | 0x80);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encodes a value and reads it back, returning the encoding.
    fn round_trip(value: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        write(value, &mut bytes).expect("the value is in range");
        assert_eq!(bytes.len(), encoded_len(value));
        assert_eq!(read(&bytes), Ok((value, bytes.len())));
        bytes
    }

    #[test]
    fn small_values_take_one_byte() {
        assert_eq!(round_trip(0), vec![0x00]);
        assert_eq!(round_trip(1), vec![0x01]);
        assert_eq!(round_trip(127), vec![0x7F]);
    }

    #[test]
    fn the_low_group_comes_first() {
        assert_eq!(round_trip(128), vec![0x80, 0x01]);
        assert_eq!(round_trip(300), vec![0xAC, 0x02]);
        assert_eq!(round_trip(16_383), vec![0xFF, 0x7F]);
    }

    #[test]
    fn the_largest_value_takes_nine_bytes() {
        let bytes = round_trip(MAX_VARINT);
        assert_eq!(bytes.len(), MAX_VARINT_BYTES);
        assert_eq!(bytes[MAX_VARINT_BYTES - 1], 0x7F);
    }

    #[test]
    fn every_bit_position_round_trips() {
        for bit in 0..63 {
            round_trip(1u64 << bit);
            round_trip((1u64 << bit) - 1);
        }
    }

    #[test]
    fn a_value_past_the_cap_is_refused() {
        let mut bytes = Vec::new();
        assert_eq!(
            write(1u64 << 63, &mut bytes),
            Err(VarintError::OutOfRange { value: 1u64 << 63 })
        );
        assert!(bytes.is_empty());
        assert_eq!(
            write(u64::MAX, &mut bytes),
            Err(VarintError::OutOfRange { value: u64::MAX })
        );
    }

    #[test]
    fn a_truncated_sequence_is_refused() {
        assert_eq!(read(&[]), Err(VarintError::Truncated { available: 0 }));
        assert_eq!(read(&[0x80]), Err(VarintError::Truncated { available: 1 }));
        assert_eq!(
            read(&[0x80, 0x80, 0x80]),
            Err(VarintError::Truncated { available: 3 })
        );
    }

    #[test]
    fn a_ninth_continuation_byte_is_refused() {
        assert_eq!(read(&[0x80; MAX_VARINT_BYTES]), Err(VarintError::Overlong));
        assert_eq!(read(&[0xFF; 12]), Err(VarintError::Overlong));
    }

    #[test]
    fn trailing_bytes_are_left_for_the_caller() {
        assert_eq!(read(&[0x01, 0xAA, 0xBB]), Ok((1, 1)));
        assert_eq!(read(&[0x80, 0x01, 0xAA]), Ok((128, 2)));
    }

    #[test]
    fn a_noncanonical_encoding_is_accepted_as_the_rfc_allows() {
        // The RFC caps the length but does not require the shortest one, so a
        // padded encoding of zero is a valid encoding of zero.
        assert_eq!(read(&[0x80, 0x00]), Ok((0, 2)));
        assert_eq!(read(&[0x81, 0x80, 0x00]), Ok((1, 3)));
        // The writer only ever produces the canonical form.
        assert_eq!(encoded_len(0), 1);
    }
}

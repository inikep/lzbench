//! Safe implementation behind the exported C functions.
//!
//! The exported functions in the crate root only document the contract. Every
//! check a C caller's arguments go through lives here: pointer validation
//! first, then the one `unsafe` conversion into slices, then the safe codec
//! call inside a panic guard, and finally the mapping of the private
//! [`FfiError`] onto the public status code.

use crate::MbrotliResult;
use mbrotli::{
    Compressor, ConfigError, DecodeConfigError, DecodeError, DecoderConfig, Decompressor,
    EncodeError, EncoderConfig, Quality, Window,
};
use std::ffi::c_int;
use std::panic::{self, AssertUnwindSafe};
use thiserror::Error;

/// Why a C call failed, before it is reduced to an [`MbrotliResult`].
#[derive(Debug, Error)]
pub(crate) enum FfiError {
    /// `output_len` was null.
    #[error("the output length pointer is null")]
    NullLength,
    /// `output_len` was not aligned for a `size_t`.
    #[error("the output length pointer is misaligned")]
    MisalignedLength,
    /// A buffer pointer was null although its length was not zero.
    #[error("a buffer pointer is null but its length is {len}")]
    NullBuffer {
        /// The length the null pointer was declared with.
        len: usize,
    },
    /// A buffer claimed more than `isize::MAX` bytes, or ended past the
    /// address space.
    #[error("a buffer of {len} bytes cannot exist in the address space")]
    OversizedBuffer {
        /// The declared length.
        len: usize,
    },
    /// The input, the output and the length slot were not pairwise disjoint.
    #[error("the input, output and length arguments overlap")]
    Overlap,
    /// A quality outside `0..=11`.
    #[error("quality {0} is outside the 0..=11 Brotli defines")]
    Quality(c_int),
    /// A window outside `10..=24`.
    #[error("a window of {0} bits is outside the 10..=24 RFC 7932 expresses")]
    Window(c_int),
    /// The encoder refused a configuration built from valid parameters.
    #[error("the encoder configuration was rejected")]
    Config(#[from] ConfigError),
    /// The encoder failed, including an output buffer that was too small.
    #[error("compression failed")]
    Encode(#[from] EncodeError),
    /// The decoder could not be constructed.
    #[error("the decoder configuration was rejected")]
    DecodeConfig(#[from] DecodeConfigError),
    /// The decoder failed, including an output buffer that was too small.
    #[error("decompression failed")]
    Decode(#[from] DecodeError),
    /// The codec panicked; the panic was stopped at the C boundary.
    #[error("the codec panicked")]
    Panic,
}

impl From<&FfiError> for MbrotliResult {
    fn from(error: &FfiError) -> Self {
        match error {
            FfiError::NullLength
            | FfiError::MisalignedLength
            | FfiError::NullBuffer { .. }
            | FfiError::OversizedBuffer { .. }
            | FfiError::Overlap
            | FfiError::Quality(_)
            | FfiError::Window(_)
            | FfiError::Config(_) => Self::InvalidParameter,
            FfiError::Encode(EncodeError::OutputTooSmall { .. })
            | FfiError::Decode(DecodeError::OutputTooSmall { .. }) => Self::OutputTooSmall,
            FfiError::Encode(_)
            | FfiError::DecodeConfig(_)
            | FfiError::Decode(_)
            | FfiError::Panic => Self::Error,
        }
    }
}

/// A byte range described by a C pointer and length, not yet trusted.
#[derive(Copy, Clone, Debug)]
struct Region {
    start: usize,
    len: usize,
}

impl Region {
    /// Validates one buffer argument.
    ///
    /// A null pointer is accepted only with a zero length. A length above
    /// `isize::MAX`, or one that runs past the end of the address space, is
    /// refused because no allocation can have it.
    fn new(address: usize, len: usize) -> Result<Self, FfiError> {
        if address == 0 && len != 0 {
            return Err(FfiError::NullBuffer { len });
        }
        if len > isize::MAX.unsigned_abs() || address.checked_add(len).is_none() {
            return Err(FfiError::OversizedBuffer { len });
        }
        Ok(Self {
            start: address,
            len,
        })
    }

    /// Whether the two ranges share a byte. Empty ranges share nothing.
    const fn overlaps(self, other: Self) -> bool {
        // Both ends were checked not to overflow in `Region::new`.
        self.len != 0
            && other.len != 0
            && self.start < other.start + other.len
            && other.start < self.start + self.len
    }
}

/// The validated `output_len` argument: non-null, aligned, and ending inside
/// the address space.
#[derive(Copy, Clone, Debug)]
struct LengthSlot {
    pointer: *mut usize,
    region: Region,
}

impl LengthSlot {
    /// Validates the length pointer.
    ///
    /// A slot that fails here is never written, since it may not be memory.
    fn new(pointer: *mut usize) -> Result<Self, FfiError> {
        if pointer.is_null() {
            return Err(FfiError::NullLength);
        }
        if !pointer.is_aligned() {
            return Err(FfiError::MisalignedLength);
        }
        let region = Region::new(pointer.addr(), size_of::<usize>())?;
        Ok(Self { pointer, region })
    }
}

/// The validated buffer arguments of one call, ready to become slices.
struct Arguments {
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
}

impl Arguments {
    /// Checks every property of the buffers that can be checked without
    /// dereferencing them: null pointers, impossible lengths, and that the
    /// input, the output and the length slot are pairwise disjoint.
    fn new(
        input: *const u8,
        input_len: usize,
        output: *mut u8,
        output_len: usize,
        slot: LengthSlot,
    ) -> Result<Self, FfiError> {
        let input_region = Region::new(input.addr(), input_len)?;
        let output_region = Region::new(output.addr(), output_len)?;
        if input_region.overlaps(slot.region)
            || output_region.overlaps(slot.region)
            || output_region.overlaps(input_region)
        {
            return Err(FfiError::Overlap);
        }
        Ok(Self {
            input,
            input_len,
            output,
            output_len,
        })
    }

    /// Converts the validated pointers into slices.
    ///
    /// Empty buffers become empty slices whatever their pointer is, so a null
    /// or dangling pointer with a zero length is never turned into a slice.
    ///
    /// # Safety
    ///
    /// Beyond what [`Arguments::new`] checked, a non-empty `input` must be
    /// valid for reads of `input_len` bytes and a non-empty `output` valid for
    /// writes of `output_len` bytes, both for the lifetime `'a`, and nothing
    /// else may access either buffer during that lifetime.
    unsafe fn slices<'a>(&self) -> (&'a [u8], &'a mut [u8]) {
        let input: &'a [u8] = if self.input_len == 0 {
            &[]
        } else {
            // SAFETY: non-null (checked), at most `isize::MAX` bytes (checked),
            // readable for `input_len` bytes and unaliased by any mutable
            // access (caller), and disjoint from the output (checked).
            unsafe { std::slice::from_raw_parts(self.input, self.input_len) }
        };
        let output: &'a mut [u8] = if self.output_len == 0 {
            &mut []
        } else {
            // SAFETY: non-null (checked), at most `isize::MAX` bytes (checked),
            // writable for `output_len` bytes and not accessed elsewhere
            // (caller), and disjoint from the input and length slot (checked).
            unsafe { std::slice::from_raw_parts_mut(self.output, self.output_len) }
        };
        (input, output)
    }
}

/// Runs one C call end to end and returns its status.
///
/// On success the number of bytes written is stored through `length`; on any
/// failure after `length` has been validated, zero is stored instead.
///
/// # Safety
///
/// As documented on `mbrotli_compress` and `mbrotli_decompress`: a non-null,
/// aligned `length` must be valid for reading and writing a `size_t`, and the
/// buffers must satisfy [`Arguments::slices`].
pub(crate) unsafe fn call(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    length: *mut usize,
    codec: impl FnOnce(&[u8], &mut [u8]) -> Result<usize, FfiError>,
) -> MbrotliResult {
    let slot = match LengthSlot::new(length) {
        Ok(slot) => slot,
        Err(error) => return MbrotliResult::from(&error),
    };
    // SAFETY: `slot` is non-null and aligned, and the caller guarantees that
    // such a pointer is valid for reading a `size_t`.
    let output_len = unsafe { slot.pointer.read() };
    let outcome =
        Arguments::new(input, input_len, output, output_len, slot).and_then(|arguments| {
            // SAFETY: the buffers passed `Arguments::new`, and the caller
            // guarantees their validity and exclusivity for the duration of this
            // call, which outlives both slices.
            let (src, dst) = unsafe { arguments.slices() };
            guarded(|| codec(src, dst))
        });
    let (status, written) = match outcome {
        Ok(written) => (MbrotliResult::Ok, written),
        Err(error) => (MbrotliResult::from(&error), 0),
    };
    // SAFETY: `slot` is non-null and aligned and the caller guarantees it is
    // writable. Both slices are dead by now, and neither overlapped the slot
    // (checked in `Arguments::new`), so nothing aliases this write.
    unsafe { slot.pointer.write(written) };
    status
}

/// Runs `body`, turning a panic into [`FfiError::Panic`] so it never unwinds
/// into C.
fn guarded(body: impl FnOnce() -> Result<usize, FfiError>) -> Result<usize, FfiError> {
    // `AssertUnwindSafe` is sound here: after a panic the only state the
    // closure could have left half-updated is the output buffer, whose
    // contents are unspecified after any failure, and the codec it built,
    // which is dropped with the closure.
    panic::catch_unwind(AssertUnwindSafe(body)).unwrap_or(Err(FfiError::Panic))
}

/// Converts the C `quality` argument.
fn quality(value: c_int) -> Result<Quality, FfiError> {
    u8::try_from(value)
        .ok()
        .and_then(|value| Quality::try_from(value).ok())
        .ok_or(FfiError::Quality(value))
}

/// Converts the C `lgwin` argument into an RFC 7932 window.
fn window(value: c_int) -> Result<Window, FfiError> {
    u8::try_from(value)
        .ok()
        .and_then(|value| Window::standard(value).ok())
        .ok_or(FfiError::Window(value))
}

/// Compresses `src` into `dst` as one complete stream, exactly as Google's
/// `BrotliEncoderCompress` does.
///
/// Like that function, a stream longer than [`compress_bound`] — which only
/// incompressible input produces — or one that does not fit in a `dst` of at
/// least that bound is replaced by a [`stored_stream`], so the bound is always
/// sufficient and never exceeded.
///
/// # Errors
///
/// [`FfiError::Quality`] or [`FfiError::Window`] for parameters outside the
/// accepted ranges, [`FfiError::Encode`] when the stream does not fit in a
/// `dst` smaller than the bound or the encoder fails.
pub(crate) fn compress(
    src: &[u8],
    dst: &mut [u8],
    quality_value: c_int,
    lgwin: c_int,
) -> Result<usize, FfiError> {
    let config = EncoderConfig::default()
        .with_quality(quality(quality_value)?)
        .with_window(window(lgwin)?);
    if src.is_empty() {
        return empty_stream(dst);
    }
    let mut encoder = Compressor::new(config)?;
    let outcome = encoder.compress_to_slice(src, dst);
    let Some(bound) = bound(src.len()) else {
        return Ok(outcome?);
    };
    match outcome {
        Ok(written) if written <= bound => Ok(written),
        _ if dst.len() >= bound => stored_stream(src, dst),
        outcome => Ok(outcome?),
    }
}

/// The stream `BrotliEncoderCompress` writes for an empty input whatever its
/// window: a 16-bit window header followed by an empty last meta-block.
const EMPTY_STREAM: u8 = 0x06;

/// Writes [`EMPTY_STREAM`], so an empty input compresses to the same byte as
/// in Google's one-shot API rather than to a header naming the chosen window.
///
/// # Errors
///
/// [`FfiError::Encode`] with `OutputTooSmall` when `dst` is empty.
fn empty_stream(dst: &mut [u8]) -> Result<usize, FfiError> {
    let Some(first) = dst.first_mut() else {
        return Err(EncodeError::OutputTooSmall { provided: 0 }.into());
    };
    *first = EMPTY_STREAM;
    Ok(1)
}

/// The largest meta-block a stored stream uses: 2^24 bytes, the most an
/// uncompressed meta-block with six length nibbles can declare.
const STORED_CHUNK: usize = 1 << 24;

/// Wraps non-empty `src` in uncompressed meta-blocks, byte for byte as
/// Google's `MakeUncompressedStream`: a 10-bit window header and an empty
/// metadata block, one uncompressed meta-block per [`STORED_CHUNK`], and an
/// empty last meta-block.
///
/// # Errors
///
/// [`FfiError::Encode`] with `OutputTooSmall` when `dst` is shorter than the
/// stream, which a `dst` of at least [`compress_bound`] bytes never is.
fn stored_stream(src: &[u8], dst: &mut [u8]) -> Result<usize, FfiError> {
    let provided = dst.len();
    let mut rest = &mut dst[..];
    let mut put = |bytes: &[u8]| {
        let (head, tail) = std::mem::take(&mut rest)
            .split_at_mut_checked(bytes.len())
            .ok_or(EncodeError::OutputTooSmall { provided })?;
        head.copy_from_slice(bytes);
        rest = tail;
        Ok::<(), FfiError>(())
    };
    // Window bits 10 with ISLAST clear, then an empty metadata block and its
    // padding.
    put(&[0x21, 0x03])?;
    for chunk in src.chunks(STORED_CHUNK) {
        let header = stored_header(chunk.len());
        let header_len = if chunk.len() > 1 << 20 { 4 } else { 3 };
        put(&header.to_le_bytes()[..header_len])?;
        put(chunk)?;
    }
    // An empty last meta-block: ISLAST and ISLASTEMPTY.
    put(&[0x03])?;
    Ok(provided - rest.len())
}

/// The header of an uncompressed, non-last meta-block of `len` bytes, where
/// `len` is `1..=STORED_CHUNK`.
///
/// ISLAST is bit 0, MNIBBLES bits 1–2 (4, 5 or 6 nibbles encoded as 0, 1, 2),
/// MLEN-1 starts at bit 3 and ISUNCOMPRESSED follows the last nibble.
const fn stored_header(len: usize) -> u32 {
    let extra_nibbles: u32 = if len > 1 << 20 {
        2
    } else if len > 1 << 16 {
        1
    } else {
        0
    };
    // `len - 1 < 2^24`, so the shifted value fits in 27 bits.
    let mlen_minus_one = (len - 1) as u32;
    (extra_nibbles << 1) | (mlen_minus_one << 3) | (1 << (19 + 4 * extra_nibbles))
}

/// Decompresses exactly one complete stream from `src` into `dst`.
///
/// # Errors
///
/// [`FfiError::Decode`] for corrupt, truncated or trailing input and for a
/// `dst` too small to hold the result.
pub(crate) fn decompress(src: &[u8], dst: &mut [u8]) -> Result<usize, FfiError> {
    let mut decoder = Decompressor::new(DecoderConfig::default())?;
    Ok(decoder.decompress_to_slice(src, dst)?)
}

/// Google's `BrotliEncoderMaxCompressedSize`: a stored stream's size for
/// `input_len` bytes, or `None` when it does not fit in a `usize`.
///
/// Two header bytes, at most four bytes of meta-block header per 16 KiB (the
/// stored chunks are far larger, so this over-counts), and the last byte.
const fn bound(input_len: usize) -> Option<usize> {
    if input_len == 0 {
        return Some(2);
    }
    input_len.checked_add(2 + 4 * (input_len >> 14) + 3 + 1)
}

/// [`bound`], or zero when it overflows.
pub(crate) const fn compress_bound(input_len: usize) -> usize {
    match bound(input_len) {
        Some(bound) => bound,
        None => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn region_accepts_null_only_when_empty() {
        assert!(Region::new(0, 0).is_ok());
        assert!(matches!(
            Region::new(0, 3),
            Err(FfiError::NullBuffer { len: 3 })
        ));
    }

    #[test]
    fn region_refuses_lengths_no_allocation_can_have() {
        let too_long = isize::MAX.unsigned_abs() + 1;
        assert!(matches!(
            Region::new(16, too_long),
            Err(FfiError::OversizedBuffer { .. })
        ));
        assert!(matches!(
            Region::new(usize::MAX, 2),
            Err(FfiError::OversizedBuffer { len: 2 })
        ));
        assert!(Region::new(usize::MAX - 2, 2).is_ok());
    }

    #[test]
    fn regions_overlap_only_when_they_share_a_byte() {
        let region = |start, len| Region { start, len };
        assert!(region(10, 5).overlaps(region(14, 1)));
        assert!(region(14, 1).overlaps(region(10, 5)));
        assert!(region(10, 5).overlaps(region(0, 100)));
        assert!(!region(10, 5).overlaps(region(15, 5)));
        assert!(!region(15, 5).overlaps(region(10, 5)));
        assert!(!region(10, 0).overlaps(region(10, 5)));
        assert!(!region(10, 5).overlaps(region(12, 0)));
    }

    #[test]
    fn quality_accepts_exactly_zero_through_eleven() {
        for value in 0..=11 {
            assert_eq!(quality(value).map(u8::from).ok(), u8::try_from(value).ok());
        }
        for value in [-1, 12, 255, 256, c_int::MIN, c_int::MAX] {
            assert!(matches!(quality(value), Err(FfiError::Quality(v)) if v == value));
        }
    }

    #[test]
    fn window_accepts_exactly_ten_through_twenty_four() {
        for value in 10..=24 {
            assert_eq!(
                window(value).map(Window::bits).ok(),
                u8::try_from(value).ok()
            );
        }
        for value in [-1, 0, 9, 25, 30, 266, c_int::MIN, c_int::MAX] {
            assert!(matches!(window(value), Err(FfiError::Window(v)) if v == value));
        }
    }

    #[test]
    fn guarded_turns_a_panic_into_an_error() {
        let outcome = guarded(|| panic!("codec bug"));
        assert!(matches!(outcome, Err(FfiError::Panic)));
        assert_eq!(guarded(|| Ok(7)).ok(), Some(7));
    }

    #[test]
    fn errors_map_onto_the_documented_status_codes() {
        let invalid = [
            FfiError::NullLength,
            FfiError::MisalignedLength,
            FfiError::NullBuffer { len: 1 },
            FfiError::OversizedBuffer { len: 1 },
            FfiError::Overlap,
            FfiError::Quality(12),
            FfiError::Window(9),
            FfiError::Config(ConfigError::StandardWindow { requested: 9 }),
        ];
        for error in &invalid {
            assert_eq!(MbrotliResult::from(error), MbrotliResult::InvalidParameter);
        }
        let too_small = [
            FfiError::Encode(EncodeError::OutputTooSmall { provided: 0 }),
            FfiError::Decode(DecodeError::OutputTooSmall { written: 0 }),
        ];
        for error in &too_small {
            assert_eq!(MbrotliResult::from(error), MbrotliResult::OutputTooSmall);
        }
        let failed = [
            FfiError::Encode(EncodeError::AbandonedSession),
            FfiError::Decode(DecodeError::UnexpectedEndOfInput),
            FfiError::DecodeConfig(DecodeConfigError::StandardWindow { max_bits: 9 }),
            FfiError::Panic,
        ];
        for error in &failed {
            assert_eq!(MbrotliResult::from(error), MbrotliResult::Error);
        }
    }

    #[test]
    fn codec_errors_keep_their_source() {
        let error = FfiError::from(DecodeError::UnexpectedEndOfInput);
        assert_eq!(error.to_string(), "decompression failed");
        let source = error.source().expect("the decoder error is kept");
        assert_eq!(
            source.to_string(),
            DecodeError::UnexpectedEndOfInput.to_string()
        );
        assert!(FfiError::Quality(12).to_string().contains("12"));
    }

    #[test]
    fn compress_writes_a_stream_decompress_restores() {
        let payload = b"slice to slice, slice to slice, slice to slice".repeat(8);
        let mut compressed = vec![0; compress_bound(payload.len())];
        let written = compress(&payload, &mut compressed, 9, 22).expect("compresses");
        let mut restored = vec![0; payload.len()];
        let read = decompress(&compressed[..written], &mut restored).expect("decompresses");
        assert_eq!(&restored[..read], payload.as_slice());
    }

    #[test]
    fn compress_refuses_bad_parameters_before_encoding() {
        let mut output = [0; 16];
        assert!(matches!(
            compress(b"x", &mut output, 12, 22),
            Err(FfiError::Quality(12))
        ));
        assert!(matches!(
            compress(b"x", &mut output, 5, 25),
            Err(FfiError::Window(25))
        ));
    }

    #[test]
    fn empty_input_compresses_to_the_single_byte_google_writes() {
        let mut output = [0xaa; 2];
        assert_eq!(compress(b"", &mut output, 11, 10).ok(), Some(1));
        assert_eq!(output, [EMPTY_STREAM, 0xaa]);
        assert_eq!(decompress(&output[..1], &mut []).ok(), Some(0));
        assert!(matches!(
            compress(b"", &mut [], 11, 22),
            Err(FfiError::Encode(EncodeError::OutputTooSmall {
                provided: 0
            }))
        ));
    }

    #[test]
    fn compress_bound_matches_google_and_is_zero_on_overflow() {
        assert_eq!(compress_bound(0), 2);
        assert_eq!(compress_bound(1), 7);
        assert_eq!(compress_bound(1 << 14), (1 << 14) + 10);
        assert_eq!(compress_bound(usize::MAX), 0);
        assert_eq!(compress_bound(usize::MAX - 10), 0);
    }

    #[test]
    fn stored_headers_use_the_fewest_nibbles_that_fit() {
        // MLEN-1 = 0 with four nibbles, uncompressed bit at 19.
        assert_eq!(stored_header(1), 1 << 19);
        assert_eq!(stored_header(1 << 16), (0xffff << 3) | (1 << 19));
        assert_eq!(
            stored_header((1 << 16) + 1),
            2 | (0x1_0000 << 3) | (1 << 23)
        );
        assert_eq!(stored_header(1 << 20), 2 | (0xf_ffff << 3) | (1 << 23));
        assert_eq!(
            stored_header((1 << 20) + 1),
            4 | (0x10_0000 << 3) | (1 << 27)
        );
        assert_eq!(
            stored_header(STORED_CHUNK),
            4 | (0xff_ffff << 3) | (1 << 27)
        );
    }

    #[test]
    fn stored_streams_fit_the_bound_and_decompress() {
        for len in [1, 1 << 16, (1 << 16) + 1, (1 << 20) + 1, STORED_CHUNK + 1] {
            let src: Vec<u8> = (0..len).map(|index| (index * 31 % 251) as u8).collect();
            let mut dst = vec![0; compress_bound(len)];
            let written = stored_stream(&src, &mut dst).expect("the bound suffices");
            assert!(written <= dst.len());
            assert_eq!(&dst[..2], &[0x21, 0x03]);
            assert_eq!(dst[written - 1], 0x03);
            let mut restored = vec![0; len];
            assert_eq!(decompress(&dst[..written], &mut restored).ok(), Some(len));
            assert_eq!(restored, src);
        }
    }

    #[test]
    fn stored_stream_reports_a_short_destination() {
        let mut dst = [0; 7];
        assert!(matches!(
            stored_stream(b"four", &mut dst),
            Err(FfiError::Encode(EncodeError::OutputTooSmall {
                provided: 7
            }))
        ));
        assert_eq!(stored_stream(b"four", &mut [0; 10]).ok(), Some(10));
    }

    #[test]
    fn incompressible_input_falls_back_to_a_stored_stream_within_the_bound() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let src: Vec<u8> = (0..20_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state.to_le_bytes()[0]
            })
            .collect();
        let mut dst = vec![0; compress_bound(src.len())];
        let written = compress(&src, &mut dst, 0, 10).expect("the bound suffices");
        let mut stored = vec![0; dst.len()];
        let stored_len = stored_stream(&src, &mut stored).expect("fits");
        assert_eq!(&dst[..written], &stored[..stored_len]);

        // Below the bound there is no fallback, as in Google's API.
        let mut short = vec![0; compress_bound(src.len()) - 1];
        assert!(matches!(
            compress(&src, &mut short, 0, 10),
            Err(FfiError::Encode(EncodeError::OutputTooSmall { .. }))
        ));
    }
}

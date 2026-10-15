//! Safe implementation behind the exported C functions.
//!
//! The exported functions in the crate root only document the contract. Every
//! check a C caller's arguments go through lives here: pointer validation
//! first, then the one `unsafe` conversion into slices, then the safe codec
//! call inside a panic guard, and finally the mapping of the private
//! [`FfiError`] onto the public status code.

use crate::MzstdResult;
use borrowed_buf::BorrowedBuf;
use mzstd::{
    Compressor, ConfigError, DecodeError, DecodeLimits, DecoderConfig, Decompressor, EncodeError,
    EncoderConfig, Level,
};
use std::ffi::c_int;
use std::mem::MaybeUninit;
use std::panic::{self, AssertUnwindSafe};
use thiserror::Error;

/// Why a C call failed, before it is reduced to an [`MzstdResult`].
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
    /// A compression level outside `-131072..=22`.
    #[error("compression level {level} is out of range")]
    Level {
        /// The rejected level.
        level: c_int,
        /// Why it was rejected.
        #[source]
        source: ConfigError,
    },
    /// The encoder failed, including an output buffer that was too small.
    #[error("compression failed")]
    Encode(#[from] EncodeError),
    /// The decoder failed, including an output buffer that was too small.
    #[error("decompression failed")]
    Decode(#[from] DecodeError),
    /// The codec panicked; the panic was stopped at the C boundary.
    #[error("the codec panicked")]
    Panic,
}

impl From<&FfiError> for MzstdResult {
    fn from(error: &FfiError) -> Self {
        match error {
            FfiError::NullLength
            | FfiError::MisalignedLength
            | FfiError::NullBuffer { .. }
            | FfiError::OversizedBuffer { .. }
            | FfiError::Overlap
            | FfiError::Level { .. } => Self::InvalidParameter,
            FfiError::Encode(EncodeError::OutputTooSmall { .. })
            | FfiError::Decode(DecodeError::OutputTooSmall { .. }) => Self::OutputTooSmall,
            FfiError::Encode(_) | FfiError::Decode(_) | FfiError::Panic => Self::Error,
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
    /// else may access either buffer during that lifetime. `output` need not
    /// be initialized: it is only ever viewed as `MaybeUninit<u8>`.
    unsafe fn slices<'a>(&self) -> (&'a [u8], &'a mut [MaybeUninit<u8>]) {
        let input: &'a [u8] = if self.input_len == 0 {
            &[]
        } else {
            // SAFETY: non-null (checked), at most `isize::MAX` bytes (checked),
            // readable for `input_len` bytes and unaliased by any mutable
            // access (caller), and disjoint from the output (checked).
            unsafe { std::slice::from_raw_parts(self.input, self.input_len) }
        };
        let output: &'a mut [MaybeUninit<u8>] = if self.output_len == 0 {
            &mut []
        } else {
            // SAFETY: non-null (checked), at most `isize::MAX` bytes (checked),
            // writable for `output_len` bytes and not accessed elsewhere
            // (caller), and disjoint from the input and length slot (checked).
            // `MaybeUninit<u8>` has the layout of `u8` and admits any contents,
            // so a C buffer that was never initialized is a valid slice of it.
            unsafe {
                std::slice::from_raw_parts_mut(
                    self.output.cast::<MaybeUninit<u8>>(),
                    self.output_len,
                )
            }
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
/// As documented on `mzstd_compress` and `mzstd_decompress`: a non-null,
/// aligned `length` must be valid for reading and writing a `size_t`, and the
/// buffers must satisfy [`Arguments::slices`].
pub(crate) unsafe fn call(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    length: *mut usize,
    codec: impl FnOnce(&[u8], &mut [MaybeUninit<u8>]) -> Result<usize, FfiError>,
) -> MzstdResult {
    let slot = match LengthSlot::new(length) {
        Ok(slot) => slot,
        Err(error) => return MzstdResult::from(&error),
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
        Ok(written) => (MzstdResult::Ok, written),
        Err(error) => (MzstdResult::from(&error), 0),
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

/// Converts the C `level` argument.
///
/// # Errors
///
/// [`FfiError::Level`] outside `-131072..=22`.
fn level(value: c_int) -> Result<Level, FfiError> {
    Level::try_from(value).map_err(|source| FfiError::Level {
        level: value,
        source,
    })
}

/// Compresses `src` into `dst` as one frame, exactly as zstd's
/// `ZSTD_compress` does at `level_value`.
///
/// # Errors
///
/// [`FfiError::Level`] for a level out of range, [`FfiError::Encode`] when
/// the frame does not fit in `dst` or the encoder fails.
pub(crate) fn compress(
    src: &[u8],
    dst: &mut [MaybeUninit<u8>],
    level_value: c_int,
) -> Result<usize, FfiError> {
    let config = EncoderConfig::default().with_level(level(level_value)?);
    // A configuration holding only a valid level has nothing else to reject,
    // so the error arm reports a validation bug rather than panicking.
    let mut encoder = match Compressor::new(config) {
        Ok(encoder) => encoder,
        Err(source) => {
            return Err(FfiError::Level {
                level: level_value,
                source,
            });
        }
    };
    Ok(encoder.compress_to_uninit(src, dst)?)
}

/// Decodes every frame of `src` into `dst`, as zstd's `ZSTD_decompress`
/// does.
///
/// An empty `src` decodes to nothing, and every window the format allows is
/// accepted, since `dst` already bounds the output. `dst` is zeroed first,
/// because the decoder writes into initialized memory.
///
/// # Errors
///
/// [`FfiError::Decode`] for corrupt or truncated input and for a `dst` too
/// small to hold the result.
pub(crate) fn decompress(src: &[u8], dst: &mut [MaybeUninit<u8>]) -> Result<usize, FfiError> {
    if src.is_empty() {
        return Ok(0);
    }
    let mut decoder = Decompressor::from(DecoderConfig::DEFAULT.with_limits(LIMITS));
    let mut zeroed = BorrowedBuf::<u8>::from(dst);
    Ok(decoder.decompress_to_slice(src, zeroed.unfilled().ensure_init())?)
}

/// Decoder limits of the C ABI: the widest window the format allows, and no
/// input or output budget beyond the buffers themselves.
const LIMITS: DecodeLimits = {
    match DecodeLimits::DEFAULT.with_max_window_log(DecodeLimits::MAX_WINDOW_LOG) {
        Ok(limits) => limits,
        Err(_) => DecodeLimits::DEFAULT,
    }
};

/// zstd's `ZSTD_compressBound`, or zero when it overflows.
pub(crate) const fn compress_bound(input_len: usize) -> usize {
    match Compressor::max_compressed_size(input_len) {
        Ok(bound) => bound,
        Err(_) => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    /// A destination of `len` bytes that were never initialized.
    fn uninit(len: usize) -> Vec<MaybeUninit<u8>> {
        vec![MaybeUninit::uninit(); len]
    }

    /// Reads bytes the code under test reported as written.
    ///
    /// Callers pass only the prefix a successful call initialized, or bytes
    /// the test itself initialized.
    fn read(bytes: &[MaybeUninit<u8>]) -> Vec<u8> {
        // SAFETY: see above; every byte passed in is initialized.
        bytes
            .iter()
            .map(|byte| unsafe { byte.assume_init() })
            .collect()
    }

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
    fn level_accepts_exactly_the_reference_range() {
        for value in [-131_072, -5, -1, 0, 1, 3, 19, 22] {
            assert!(level(value).is_ok(), "{value}");
        }
        for value in [-131_073, 23, c_int::MIN, c_int::MAX] {
            assert!(
                matches!(level(value), Err(FfiError::Level { level, .. }) if level == value),
                "{value}"
            );
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
        let Err(level_error) = level(99) else {
            panic!("level 99 is out of range");
        };
        let invalid = [
            FfiError::NullLength,
            FfiError::MisalignedLength,
            FfiError::NullBuffer { len: 1 },
            FfiError::OversizedBuffer { len: 1 },
            FfiError::Overlap,
            level_error,
        ];
        for error in &invalid {
            assert_eq!(MzstdResult::from(error), MzstdResult::InvalidParameter);
        }
        let too_small = [
            FfiError::Encode(EncodeError::OutputTooSmall { provided: 0 }),
            FfiError::Decode(DecodeError::OutputTooSmall { written: 0 }),
        ];
        for error in &too_small {
            assert_eq!(MzstdResult::from(error), MzstdResult::OutputTooSmall);
        }
        let failed = [
            FfiError::Encode(EncodeError::InvalidState),
            FfiError::Decode(DecodeError::UnexpectedEndOfInput),
            FfiError::Panic,
        ];
        for error in &failed {
            assert_eq!(MzstdResult::from(error), MzstdResult::Error);
        }
    }

    #[test]
    fn errors_keep_their_source() {
        let error = FfiError::from(DecodeError::UnexpectedEndOfInput);
        assert_eq!(error.to_string(), "decompression failed");
        let source = error.source().expect("the decoder error is kept");
        assert_eq!(
            source.to_string(),
            DecodeError::UnexpectedEndOfInput.to_string()
        );
        let Err(error) = level(23) else {
            panic!("level 23 is out of range");
        };
        assert!(error.to_string().contains("23"));
        assert!(error.source().is_some());
    }

    #[test]
    fn compress_writes_a_frame_decompress_restores() {
        let payload = b"slice to slice, slice to slice, slice to slice".repeat(8);
        let mut compressed = uninit(compress_bound(payload.len()));
        let written = compress(&payload, &mut compressed, 9).expect("compresses");
        let compressed = read(&compressed[..written]);
        let mut restored = uninit(payload.len());
        let len = decompress(&compressed, &mut restored).expect("decompresses");
        assert_eq!(read(&restored[..len]), payload);
    }

    #[test]
    fn decompress_reports_a_short_destination() {
        let payload = b"short destination".repeat(10);
        let mut compressed = uninit(compress_bound(payload.len()));
        let written = compress(&payload, &mut compressed, 1).expect("compresses");
        let compressed = read(&compressed[..written]);
        let mut short = uninit(payload.len() - 1);
        assert!(matches!(
            decompress(&compressed, &mut short),
            Err(FfiError::Decode(DecodeError::OutputTooSmall { .. }))
        ));
    }

    #[test]
    fn empty_input_decodes_to_nothing() {
        assert_eq!(decompress(&[], &mut []).ok(), Some(0));
        assert_eq!(decompress(&[], &mut uninit(4)).ok(), Some(0));
    }

    #[test]
    fn the_widest_window_is_accepted() {
        assert_eq!(LIMITS.max_window_size(), 1 << DecodeLimits::MAX_WINDOW_LOG);
    }

    #[test]
    fn compress_bound_is_zero_on_overflow() {
        assert_eq!(compress_bound(0), 64);
        assert_eq!(compress_bound(usize::MAX), 0);
    }
}

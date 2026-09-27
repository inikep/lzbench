//! C ABI for [mbrotli](https://docs.rs/mbrotli): one-shot Brotli compression
//! and decompression into caller-owned buffers.
//!
//! The crate builds a `cdylib` and a `staticlib` named `mbrotli_ffi`. The C
//! declarations live in `include/mbrotli.h`:
//!
//! ```c
//! typedef enum {
//!     MBROTLI_OK = 0,
//!     MBROTLI_INVALID_PARAMETER = 1,
//!     MBROTLI_OUTPUT_TOO_SMALL = 2,
//!     MBROTLI_ERROR = 3
//! } mbrotli_result;
//!
//! mbrotli_result mbrotli_compress(const uint8_t *input, size_t input_len,
//!                                 uint8_t *output, size_t *output_len,
//!                                 int quality, int lgwin);
//! mbrotli_result mbrotli_decompress(const uint8_t *input, size_t input_len,
//!                                   uint8_t *output, size_t *output_len);
//! size_t mbrotli_compress_bound(size_t input_len);
//! ```
//!
//! Every call is self-contained: it builds its codec, runs it and releases it,
//! so the functions are safe to call from any number of threads at once. No
//! panic unwinds into C; one would be reported as [`MbrotliResult::Error`].
//!
//! # Examples
//!
//! ```
//! use mbrotli_ffi::{MbrotliResult, mbrotli_compress, mbrotli_compress_bound, mbrotli_decompress};
//!
//! let payload = b"hello, hello, hello from C";
//! let mut compressed = vec![0u8; mbrotli_compress_bound(payload.len())];
//! let mut compressed_len = compressed.len();
//! // SAFETY: every pointer describes a live buffer of the stated length.
//! let status = unsafe {
//!     mbrotli_compress(
//!         payload.as_ptr(),
//!         payload.len(),
//!         compressed.as_mut_ptr(),
//!         &mut compressed_len,
//!         11,
//!         22,
//!     )
//! };
//! assert_eq!(status, MbrotliResult::Ok);
//!
//! let mut restored = [0u8; 64];
//! let mut restored_len = restored.len();
//! // SAFETY: as above.
//! let status = unsafe {
//!     mbrotli_decompress(
//!         compressed.as_ptr(),
//!         compressed_len,
//!         restored.as_mut_ptr(),
//!         &mut restored_len,
//!     )
//! };
//! assert_eq!(status, MbrotliResult::Ok);
//! assert_eq!(&restored[..restored_len], payload);
//! ```

mod core;

use std::ffi::c_int;

/// Status returned by every fallible function; `mbrotli_result` in C.
///
/// The discriminants are part of the ABI and never change.
///
/// # Examples
///
/// ```
/// use mbrotli_ffi::MbrotliResult;
///
/// assert_eq!(MbrotliResult::Ok as i32, 0);
/// assert_eq!(MbrotliResult::Error as i32, 3);
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MbrotliResult {
    /// `MBROTLI_OK`: the call succeeded and `*output_len` holds the number of
    /// bytes written.
    Ok = 0,
    /// `MBROTLI_INVALID_PARAMETER`: an argument was rejected before any work
    /// was done — a null `output_len`, a null buffer with a non-zero length,
    /// overlapping buffers, or a quality or window out of range.
    InvalidParameter = 1,
    /// `MBROTLI_OUTPUT_TOO_SMALL`: the result does not fit in the output
    /// buffer.
    OutputTooSmall = 2,
    /// `MBROTLI_ERROR`: any other failure, such as corrupt, truncated or
    /// trailing compressed input, or an allocation failure.
    Error = 3,
}

/// Compresses `input` into `output` as one complete Brotli stream.
///
/// `quality` is `0..=11` (11 is the densest and slowest) and `lgwin` is the
/// window size in bits, `10..=24` (22 is the reference encoder's default).
/// On entry `*output_len` is the capacity of `output`; on return it is the
/// number of bytes written on success and `0` on any other status except a
/// null or misaligned `output_len`, which is left untouched. An output of
/// [`mbrotli_compress_bound`]`(input_len)` bytes is always large enough.
///
/// The produced stream is byte-identical to Google's `BrotliEncoderCompress`
/// with the same quality and window and `BROTLI_MODE_GENERIC`, including its
/// two special cases: an empty input becomes the single byte `0x06`, and when
/// the compressed stream would exceed [`mbrotli_compress_bound`] — or does not
/// fit in an output of at least that bound — the input is stored in
/// uncompressed meta-blocks instead.
///
/// # Errors
///
/// - [`MbrotliResult::InvalidParameter`] for a null or misaligned
///   `output_len`, a null `input` or `output` with a non-zero length, a length
///   above `PTRDIFF_MAX`, overlapping arguments, or `quality`/`lgwin` out of
///   range.
/// - [`MbrotliResult::OutputTooSmall`] when the stream does not fit. The
///   contents of `output` are then unspecified.
/// - [`MbrotliResult::Error`] for any other failure.
///
/// # Safety
///
/// - `output_len` must be null or point to a `size_t` valid for reads and
///   writes.
/// - When `input_len` is not zero, `input` must be null or valid for reads of
///   `input_len` bytes.
/// - When `*output_len` is not zero, `output` must be null or valid for
///   writes of `*output_len` bytes.
/// - No other thread may write to `input`, or access `output` or
///   `output_len`, during the call.
///
/// Null pointers, lengths above `PTRDIFF_MAX` and overlapping arguments are
/// detected and rejected; pointers into freed or too-short memory are not.
///
/// # Examples
///
/// ```
/// use mbrotli_ffi::{MbrotliResult, mbrotli_compress};
///
/// let mut output = [0u8; 1];
/// let mut output_len = output.len();
/// // SAFETY: an empty input may be null; `output` has `output_len` bytes.
/// let status = unsafe {
///     mbrotli_compress(std::ptr::null(), 0, output.as_mut_ptr(), &mut output_len, 5, 22)
/// };
/// assert_eq!(status, MbrotliResult::Ok);
/// // The one-byte empty stream Google's `BrotliEncoderCompress` also writes.
/// assert_eq!(&output[..output_len], &[0x06]);
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mbrotli_compress(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: *mut usize,
    quality: c_int,
    lgwin: c_int,
) -> MbrotliResult {
    // SAFETY: forwarded verbatim from this function's own contract.
    unsafe {
        core::call(input, input_len, output, output_len, |src, dst| {
            core::compress(src, dst, quality, lgwin)
        })
    }
}

/// Decompresses one complete Brotli stream from `input` into `output`.
///
/// `input` must hold exactly one stream: bytes after its end are an error.
/// On entry `*output_len` is the capacity of `output`; on return it is the
/// number of bytes written on success and `0` on any other status except a
/// null or misaligned `output_len`, which is left untouched. Brotli does not
/// record the decompressed size, so the caller must know or bound it.
///
/// Streams with an RFC 9841 large window are accepted as well as ordinary
/// RFC 7932 ones.
///
/// # Errors
///
/// - [`MbrotliResult::InvalidParameter`] for a null or misaligned
///   `output_len`, a null `input` or `output` with a non-zero length, a length
///   above `PTRDIFF_MAX`, or overlapping arguments.
/// - [`MbrotliResult::OutputTooSmall`] when the output fills up before the
///   stream ends. `output` then holds as much of the data as fits. Decoding
///   stops there, so a stream that is corrupt or truncated only past that
///   point is reported this way too; a larger buffer then yields
///   [`MbrotliResult::Error`].
/// - [`MbrotliResult::Error`] for corrupt, truncated or trailing input and any
///   other failure. The contents of `output` are then unspecified.
///
/// # Safety
///
/// As [`mbrotli_compress`]: `output_len` must be null or point to a readable
/// and writable `size_t`, a non-null `input` must be readable for `input_len`
/// bytes, a non-null `output` writable for `*output_len` bytes, and no other
/// thread may write to `input` or access `output` or `output_len` during the
/// call.
///
/// # Examples
///
/// ```
/// use mbrotli_ffi::{MbrotliResult, mbrotli_decompress};
///
/// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
/// let mut output = [0u8; 5];
/// let mut output_len = output.len();
/// // SAFETY: both buffers are live and have the stated lengths.
/// let status = unsafe {
///     mbrotli_decompress(compressed.as_ptr(), compressed.len(), output.as_mut_ptr(), &mut output_len)
/// };
/// assert_eq!(status, MbrotliResult::Ok);
/// assert_eq!(&output[..output_len], b"hello");
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mbrotli_decompress(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: *mut usize,
) -> MbrotliResult {
    // SAFETY: forwarded verbatim from this function's own contract.
    unsafe { core::call(input, input_len, output, output_len, core::decompress) }
}

/// Returns an output size that [`mbrotli_compress`] never exceeds for
/// `input_len` bytes, at any quality and window.
///
/// The value equals Google's `BrotliEncoderMaxCompressedSize`: the size of the
/// input stored in uncompressed meta-blocks, a few bytes more than
/// `input_len`. Returns `0` when it does not fit in a `size_t`; every bound is
/// at least two, so zero is never a valid one.
///
/// # Examples
///
/// ```
/// use mbrotli_ffi::mbrotli_compress_bound;
///
/// assert_eq!(mbrotli_compress_bound(1000), 1006);
/// assert_eq!(mbrotli_compress_bound(usize::MAX), 0);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn mbrotli_compress_bound(input_len: usize) -> usize {
    core::compress_bound(input_len)
}

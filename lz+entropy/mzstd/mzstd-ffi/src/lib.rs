//! C ABI for [mzstd](https://docs.rs/mzstd): one-shot Zstandard compression
//! and decompression into caller-owned buffers.
//!
//! The crate builds a `cdylib` and a `staticlib` named `mzstd_ffi`. The C
//! declarations live in `include/mzstd.h`:
//!
//! ```c
//! typedef enum {
//!     MZSTD_OK = 0,
//!     MZSTD_INVALID_PARAMETER = 1,
//!     MZSTD_OUTPUT_TOO_SMALL = 2,
//!     MZSTD_ERROR = 3
//! } mzstd_result;
//!
//! mzstd_result mzstd_compress(const uint8_t *input, size_t input_len,
//!                             uint8_t *output, size_t *output_len,
//!                             int level);
//! mzstd_result mzstd_decompress(const uint8_t *input, size_t input_len,
//!                               uint8_t *output, size_t *output_len);
//! size_t mzstd_compress_bound(size_t input_len);
//! ```
//!
//! Every call is self-contained: it builds its codec, runs it and releases it,
//! so the functions are safe to call from any number of threads at once. No
//! panic unwinds into C; one would be reported as [`MzstdResult::Error`].
//!
//! # Examples
//!
//! ```
//! use mzstd_ffi::{MzstdResult, mzstd_compress, mzstd_compress_bound, mzstd_decompress};
//!
//! let payload = b"hello, hello, hello from C";
//! let mut compressed = vec![0u8; mzstd_compress_bound(payload.len())];
//! let mut compressed_len = compressed.len();
//! // SAFETY: every pointer describes a live buffer of the stated length.
//! let status = unsafe {
//!     mzstd_compress(
//!         payload.as_ptr(),
//!         payload.len(),
//!         compressed.as_mut_ptr(),
//!         &mut compressed_len,
//!         19,
//!     )
//! };
//! assert_eq!(status, MzstdResult::Ok);
//!
//! let mut restored = [0u8; 64];
//! let mut restored_len = restored.len();
//! // SAFETY: as above.
//! let status = unsafe {
//!     mzstd_decompress(
//!         compressed.as_ptr(),
//!         compressed_len,
//!         restored.as_mut_ptr(),
//!         &mut restored_len,
//!     )
//! };
//! assert_eq!(status, MzstdResult::Ok);
//! assert_eq!(&restored[..restored_len], payload);
//! ```

mod core;

use std::ffi::c_int;

/// Status returned by every fallible function; `mzstd_result` in C.
///
/// The discriminants are part of the ABI and never change.
///
/// # Examples
///
/// ```
/// use mzstd_ffi::MzstdResult;
///
/// assert_eq!(MzstdResult::Ok as i32, 0);
/// assert_eq!(MzstdResult::Error as i32, 3);
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MzstdResult {
    /// `MZSTD_OK`: the call succeeded and `*output_len` holds the number of
    /// bytes written.
    Ok = 0,
    /// `MZSTD_INVALID_PARAMETER`: an argument was rejected before any work
    /// was done — a null `output_len`, a null buffer with a non-zero length,
    /// overlapping buffers, or a level out of range.
    InvalidParameter = 1,
    /// `MZSTD_OUTPUT_TOO_SMALL`: the result does not fit in the output
    /// buffer.
    OutputTooSmall = 2,
    /// `MZSTD_ERROR`: any other failure, such as corrupt or truncated
    /// compressed input, a frame that needs a dictionary, or an allocation
    /// failure.
    Error = 3,
}

/// Compresses `input` into `output` as one Zstandard frame.
///
/// `level` is `-131072..=22`; `0` selects the default level, 3. On entry
/// `*output_len` is the capacity of `output`; on return it is the number of
/// bytes written on success and `0` on any other status except a null or
/// misaligned `output_len`, which is left untouched. An output of
/// [`mzstd_compress_bound`]`(input_len)` bytes is always large enough.
///
/// The frame is byte-identical to the one zstd 1.5.7's `ZSTD_compress`
/// writes at the same level: content size recorded, no checksum. A smaller
/// output fails exactly when `ZSTD_compress` fails: the encoder needs room
/// for the largest frame header (18 bytes) and some slack past the frame, so
/// an output exactly as long as the frame may be refused.
///
/// # Errors
///
/// - [`MzstdResult::InvalidParameter`] for a null or misaligned
///   `output_len`, a null `input` or `output` with a non-zero length, a length
///   above `PTRDIFF_MAX`, overlapping arguments, or a level out of range.
///   `ZSTD_compress` clamps such a level instead.
/// - [`MzstdResult::OutputTooSmall`] when the frame does not fit. The
///   contents of `output` are then unspecified.
/// - [`MzstdResult::Error`] for any other failure.
///
/// # Safety
///
/// - `output_len` must be null or point to a `size_t` valid for reads and
///   writes.
/// - When `input_len` is not zero, `input` must be null or valid for reads of
///   `input_len` bytes.
/// - When `*output_len` is not zero, `output` must be null or valid for
///   writes of `*output_len` bytes. It need not be initialized: the call only
///   writes to it, and on success exactly the reported bytes are initialized.
/// - No other thread may write to `input`, or access `output` or
///   `output_len`, during the call.
///
/// Null pointers, lengths above `PTRDIFF_MAX` and overlapping arguments are
/// detected and rejected; pointers into freed or too-short memory are not.
///
/// # Examples
///
/// ```
/// use mzstd_ffi::{MzstdResult, mzstd_compress};
///
/// let mut output = [0u8; 18];
/// let mut output_len = output.len();
/// // SAFETY: an empty input may be null; `output` has `output_len` bytes.
/// let status = unsafe {
///     mzstd_compress(std::ptr::null(), 0, output.as_mut_ptr(), &mut output_len, 3)
/// };
/// assert_eq!(status, MzstdResult::Ok);
/// // Magic number, frame header with a zero content size, one empty block.
/// assert_eq!(&output[..output_len], &[0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x00, 0x01, 0x00, 0x00]);
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mzstd_compress(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: *mut usize,
    level: c_int,
) -> MzstdResult {
    // SAFETY: forwarded verbatim from this function's own contract.
    unsafe {
        core::call(input, input_len, output, output_len, |src, dst| {
            core::compress(src, dst, level)
        })
    }
}

/// Decompresses every frame in `input` into `output`, one after another.
///
/// Skippable frames are skipped, so `input` may hold any sequence of
/// Zstandard and skippable frames, including none: an empty input decodes to
/// nothing, as with `ZSTD_decompress`. On entry `*output_len` is the
/// capacity of `output`; on return it is the number of bytes written on
/// success and `0` on any other status except a null or misaligned
/// `output_len`, which is left untouched. Zstandard frames usually record
/// their content size, but nothing requires it; the caller must know or
/// bound the result.
///
/// Every window size the format allows, up to 2 GiB, is accepted: the output
/// buffer bounds the memory the call uses. `output` is zeroed before decoding.
///
/// # Errors
///
/// - [`MzstdResult::InvalidParameter`] for a null or misaligned
///   `output_len`, a null `input` or `output` with a non-zero length, a length
///   above `PTRDIFF_MAX`, or overlapping arguments.
/// - [`MzstdResult::OutputTooSmall`] when the output fills up before the
///   input ends.
/// - [`MzstdResult::Error`] for corrupt or truncated input, a checksum or
///   content size mismatch, a frame that needs a dictionary, and any other
///   failure. The contents of `output` are then unspecified.
///
/// # Safety
///
/// As [`mzstd_compress`]: `output_len` must be null or point to a readable
/// and writable `size_t`, a non-null `input` must be readable for `input_len`
/// bytes, a non-null `output` writable for `*output_len` bytes (it need not be
/// initialized), and no other thread may write to `input` or access `output`
/// or `output_len` during the call.
///
/// # Examples
///
/// ```
/// use mzstd_ffi::{MzstdResult, mzstd_decompress};
///
/// // A frame holding one raw block with "hello".
/// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x05, 0x29, 0x00, 0x00, b'h', b'e', b'l', b'l', b'o'];
/// let mut output = [0u8; 5];
/// let mut output_len = output.len();
/// // SAFETY: both buffers are live and have the stated lengths.
/// let status = unsafe {
///     mzstd_decompress(frame.as_ptr(), frame.len(), output.as_mut_ptr(), &mut output_len)
/// };
/// assert_eq!(status, MzstdResult::Ok);
/// assert_eq!(&output[..output_len], b"hello");
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mzstd_decompress(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: *mut usize,
) -> MzstdResult {
    // SAFETY: forwarded verbatim from this function's own contract.
    unsafe { core::call(input, input_len, output, output_len, core::decompress) }
}

/// Returns an output size that [`mzstd_compress`] never exceeds for
/// `input_len` bytes, at any level.
///
/// The value equals zstd's `ZSTD_compressBound`. Returns `0` when it does not
/// fit in a `size_t`; every bound is at least 64, so zero is never a valid
/// one.
///
/// # Examples
///
/// ```
/// use mzstd_ffi::mzstd_compress_bound;
///
/// assert_eq!(mzstd_compress_bound(1000), 1066);
/// assert_eq!(mzstd_compress_bound(usize::MAX), 0);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn mzstd_compress_bound(input_len: usize) -> usize {
    core::compress_bound(input_len)
}

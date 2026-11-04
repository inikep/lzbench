//! C ABI for [mbrotli](https://docs.rs/mbrotli): one-shot and streaming Brotli compression
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
//! One-shot calls are self-contained. Session handles retain their codec and
//! require exclusive access; distinct handles can run on different threads. No
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

/// Status returned by buffer operations and session reinitialization;
/// `mbrotli_result` in C. Session creation returns a pointer or null instead.
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
    /// A session needs more input; an encoder flush has been fully delivered.
    NeedsInput = 4,
    /// A session needs output space; retry with the unconsumed input suffix.
    NeedsOutput = 5,
    /// The stream has finished and all output has been delivered.
    Finished = 6,
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
/// Bytes of `output` past the ones reported are never written.
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
/// `output` serves as the decoder's history, so no separate window is copied
/// out of. Bytes of `output` past the ones reported may be overwritten only
/// when its capacity is at most 1 KiB, which is zeroed first; larger
/// capacities are written only up to the result.
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
/// bytes, a non-null `output` writable for `*output_len` bytes (it need not be
/// initialized), and no other
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

/// Operation passed as an integer to session `process` functions.
/// Unknown values are rejected without advancing the session.
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MbrotliOperation {
    /// Accept more input without declaring EOF.
    Process = 0,
    /// Flush an encoder; decoders accept only Process and Finish.
    Flush = 1,
    /// Declare the final input suffix and finish the stream.
    Finish = 2,
}

/// Opaque owning encoder session. Only use pointers returned by
/// [`mbrotli_encoder_session_create`]; release with [`mbrotli_encoder_session_destroy`].
/// This is a pointer-only ABI token, not the layout of the allocation behind it.
/// Never allocate, copy, dereference or pass a handle object by value; only its
/// pointer may cross the ABI. Internal session layout changes do not affect C.
#[repr(C)]
pub struct MbrotliEncoderSession {
    _private: [u8; 0],
}

/// Creates an independent encoder session with default stream settings.
/// `quality` is 0..=11; `lgwin` is 10..=24. Input size is initially unknown.
/// Returns null on invalid configuration, codec failure or a caught panic.
/// The handle must eventually be destroyed, including after an error.
///
/// # Examples
///
/// ```
/// use mbrotli_ffi::{mbrotli_encoder_session_create, mbrotli_encoder_session_destroy};
/// let session = mbrotli_encoder_session_create(5, 22);
/// assert!(!session.is_null());
/// // SAFETY: this is the live handle just created, with no concurrent access.
/// unsafe { mbrotli_encoder_session_destroy(session) };
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn mbrotli_encoder_session_create(
    quality: c_int,
    lgwin: c_int,
) -> *mut MbrotliEncoderSession {
    core::sessions::encoder_create(quality, lgwin)
}

/// Releases a session, abandoning any unfinished stream. Null is a no-op.
///
/// # Safety
/// A non-null handle must be a live result of [`mbrotli_encoder_session_create`],
/// used exclusively and destroyed exactly once. No pointer into it may be used
/// after this call. Misaligned pointers are rejected without dereferencing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mbrotli_encoder_session_destroy(session: *mut MbrotliEncoderSession) {
    // SAFETY: forwarded from the public handle contract.
    unsafe { core::sessions::encoder_destroy(session) }
}

/// Abandons the current stream and starts a new one with the same configuration.
/// Retained workspace is reused according to the codec's retention policy.
///
/// # Errors
/// Returns InvalidParameter for a null or misaligned handle, Error on codec
/// failure or panic, and Ok on success. A poisoned session can be reinitialized.
///
/// # Safety
/// The handle must be null or a live, exclusively accessed result of
/// [`mbrotli_encoder_session_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mbrotli_encoder_session_reinit(
    session: *mut MbrotliEncoderSession,
) -> MbrotliResult {
    // SAFETY: forwarded from the public handle contract.
    unsafe { core::sessions::encoder_reinit(session) }
}

/// Advances the encoder into a possibly uninitialized output buffer.
///
/// On entry `*input_len` is the available input and `*output_len` the output
/// capacity. On return they are the bytes consumed and produced in this call.
/// Advance the input by the consumed count before retrying. NeedsOutput is
/// backpressure, not a failure. NeedsInput requests more input; Finished ends
/// the stream. Empty buffers may be null. No buffers are retained across calls.
/// Once Finish is submitted, repeat Finish with exactly its unconsumed suffix.
/// Process, Flush and Finish are accepted. Repeat Flush until NeedsInput;
/// then input may resume. On Error both counts are zero and output is unspecified.
/// Streaming bytes follow the Rust session API, without the one-shot bound or
/// Google's one-shot empty-stream/fallback special cases.
/// Exactly the reported output prefix is initialized; bytes after it are untouched
/// on successful calls.
///
/// # Errors
/// InvalidParameter rejects null/misaligned handles or count pointers, unknown
/// operations, impossible buffer lengths, and overlapping buffers/counts/handle.
/// Rejected arguments leave both counts unchanged and do not advance the session.
/// Error reports a codec failure or panic; call reinit before further processing.
/// A caught panic zeros both counts and leaves output unspecified.
///
/// # Safety
/// The handle must be null or a live result of [`mbrotli_encoder_session_create`].
/// Non-null aligned count pointers must be readable and writable `size_t` slots.
/// Nonempty, non-null buffers must be valid for their declared lengths: input
/// readable and output writable (initialization is unnecessary). No concurrent
/// access to the handle, counts or output, or mutation of input, is permitted.
/// Dangling pointers, handles of the wrong kind and undersized allocations
/// cannot be detected and must never be supplied.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mbrotli_encoder_session_process(
    session: *mut MbrotliEncoderSession,
    input: *const u8,
    input_len: *mut usize,
    output: *mut u8,
    output_len: *mut usize,
    operation: c_int,
) -> MbrotliResult {
    // SAFETY: forwarded verbatim from this function's contract.
    unsafe {
        core::sessions::encoder_process(session, input, input_len, output, output_len, operation)
    }
}

/// Drains accepted input without declaring EOF, into uninitialized output.
/// Equivalent to [`mbrotli_encoder_session_process`] with empty input and
/// Flush. Repeat while NeedsOutput is returned.
/// For an encoder, NeedsInput means flushing is complete.
///
/// # Errors
/// Same statuses and count/output guarantees as [`mbrotli_encoder_session_process`].
///
/// # Safety
/// Same handle and output contracts as [`mbrotli_encoder_session_process`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mbrotli_encoder_session_flush(
    session: *mut MbrotliEncoderSession,
    output: *mut u8,
    output_len: *mut usize,
) -> MbrotliResult {
    let mut input_len = 0;
    // SAFETY: empty input is valid and the remaining contract is forwarded.
    unsafe {
        core::sessions::encoder_process(
            session,
            std::ptr::null(),
            &mut input_len,
            output,
            output_len,
            1,
        )
    }
}

/// Finishes the stream after all input has been accepted, into uninitialized output.
/// Equivalent to [`mbrotli_encoder_session_process`] with empty input and
/// Finish. Repeat while NeedsOutput is returned.
/// The process operation's final-suffix rules still apply.
///
/// # Errors
/// Same statuses and count/output guarantees as [`mbrotli_encoder_session_process`].
///
/// # Safety
/// Same handle and output contracts as [`mbrotli_encoder_session_process`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mbrotli_encoder_session_finish(
    session: *mut MbrotliEncoderSession,
    output: *mut u8,
    output_len: *mut usize,
) -> MbrotliResult {
    let mut input_len = 0;
    // SAFETY: empty input is valid and the remaining contract is forwarded.
    unsafe {
        core::sessions::encoder_process(
            session,
            std::ptr::null(),
            &mut input_len,
            output,
            output_len,
            2,
        )
    }
}

/// Opaque owning decoder session. Only use pointers returned by
/// [`mbrotli_decoder_session_create`]; release with [`mbrotli_decoder_session_destroy`].
/// This is a pointer-only ABI token, not the layout of the allocation behind it.
/// Never allocate, copy, dereference or pass a handle object by value; only its
/// pointer may cross the ABI. Internal session layout changes do not affect C.
#[repr(C)]
pub struct MbrotliDecoderSession {
    _private: [u8; 0],
}

/// Creates an independent decoder session with default stream settings.
/// Returns null on invalid configuration, codec failure or a caught panic.
/// The handle must eventually be destroyed, including after an error.
///
/// # Examples
///
/// ```
/// use mbrotli_ffi::{mbrotli_decoder_session_create, mbrotli_decoder_session_destroy};
/// let session = mbrotli_decoder_session_create();
/// assert!(!session.is_null());
/// // SAFETY: this is the live handle just created, with no concurrent access.
/// unsafe { mbrotli_decoder_session_destroy(session) };
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn mbrotli_decoder_session_create() -> *mut MbrotliDecoderSession {
    core::sessions::decoder_create()
}

/// Releases a session, abandoning any unfinished stream. Null is a no-op.
///
/// # Safety
/// A non-null handle must be a live result of [`mbrotli_decoder_session_create`],
/// used exclusively and destroyed exactly once. No pointer into it may be used
/// after this call. Misaligned pointers are rejected without dereferencing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mbrotli_decoder_session_destroy(session: *mut MbrotliDecoderSession) {
    // SAFETY: forwarded from the public handle contract.
    unsafe { core::sessions::decoder_destroy(session) }
}

/// Abandons the current stream and starts a new one with the same configuration.
/// Retained workspace is reused according to the codec's retention policy.
///
/// # Errors
/// Returns InvalidParameter for a null or misaligned handle, Error on codec
/// failure or panic, and Ok on success. A poisoned session can be reinitialized.
///
/// # Safety
/// The handle must be null or a live, exclusively accessed result of
/// [`mbrotli_decoder_session_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mbrotli_decoder_session_reinit(
    session: *mut MbrotliDecoderSession,
) -> MbrotliResult {
    // SAFETY: forwarded from the public handle contract.
    unsafe { core::sessions::decoder_reinit(session) }
}

/// Advances the decoder into a possibly uninitialized output buffer.
///
/// On entry `*input_len` is the available input and `*output_len` the output
/// capacity. On return they are the bytes consumed and produced in this call.
/// Advance the input by the consumed count before retrying. NeedsOutput is
/// backpressure, not a failure. NeedsInput requests more input; Finished ends
/// the stream. Empty buffers may be null. No buffers are retained across calls.
/// Once Finish is submitted, repeat Finish with exactly its unconsumed suffix.
/// Only Process and Finish are accepted. Finish declares EOF and detects
/// truncation. Decoding uses the default single-member policy; trailing input
/// is left unconsumed when Finished is returned. On codec Error, counts retain the exact partial progress.
/// Exactly the reported output prefix is initialized; bytes after it are untouched
/// on successful calls, and on decoder errors.
///
/// # Errors
/// InvalidParameter rejects null/misaligned handles or count pointers, unknown
/// operations, impossible buffer lengths, and overlapping buffers/counts/handle.
/// Rejected arguments leave both counts unchanged and do not advance the session.
/// Error reports a codec failure or panic; call reinit before further processing.
/// A caught panic zeros both counts and leaves output unspecified.
///
/// # Safety
/// The handle must be null or a live result of [`mbrotli_decoder_session_create`].
/// Non-null aligned count pointers must be readable and writable `size_t` slots.
/// Nonempty, non-null buffers must be valid for their declared lengths: input
/// readable and output writable (initialization is unnecessary). No concurrent
/// access to the handle, counts or output, or mutation of input, is permitted.
/// Dangling pointers, handles of the wrong kind and undersized allocations
/// cannot be detected and must never be supplied.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mbrotli_decoder_session_process(
    session: *mut MbrotliDecoderSession,
    input: *const u8,
    input_len: *mut usize,
    output: *mut u8,
    output_len: *mut usize,
    operation: c_int,
) -> MbrotliResult {
    // SAFETY: forwarded verbatim from this function's contract.
    unsafe {
        core::sessions::decoder_process(session, input, input_len, output, output_len, operation)
    }
}

/// Drains accepted input without declaring EOF, into uninitialized output.
/// Equivalent to [`mbrotli_decoder_session_process`] with empty input and
/// Process. Repeat while NeedsOutput is returned.
/// The process operation's final-suffix rules still apply.
///
/// # Errors
/// Same statuses and count/output guarantees as [`mbrotli_decoder_session_process`].
///
/// # Safety
/// Same handle and output contracts as [`mbrotli_decoder_session_process`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mbrotli_decoder_session_flush(
    session: *mut MbrotliDecoderSession,
    output: *mut u8,
    output_len: *mut usize,
) -> MbrotliResult {
    let mut input_len = 0;
    // SAFETY: empty input is valid and the remaining contract is forwarded.
    unsafe {
        core::sessions::decoder_process(
            session,
            std::ptr::null(),
            &mut input_len,
            output,
            output_len,
            0,
        )
    }
}

/// Finishes the stream after all input has been accepted, into uninitialized output.
/// Equivalent to [`mbrotli_decoder_session_process`] with empty input and
/// Finish. Repeat while NeedsOutput is returned.
/// The process operation's final-suffix rules still apply.
///
/// # Errors
/// Same statuses and count/output guarantees as [`mbrotli_decoder_session_process`].
///
/// # Safety
/// Same handle and output contracts as [`mbrotli_decoder_session_process`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mbrotli_decoder_session_finish(
    session: *mut MbrotliDecoderSession,
    output: *mut u8,
    output_len: *mut usize,
) -> MbrotliResult {
    let mut input_len = 0;
    // SAFETY: empty input is valid and the remaining contract is forwarded.
    unsafe {
        core::sessions::decoder_process(
            session,
            std::ptr::null(),
            &mut input_len,
            output,
            output_len,
            2,
        )
    }
}

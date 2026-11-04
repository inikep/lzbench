//! Incremental compression and decompression through session handles.
//!
//! An encoder handle writes one frame at a time, byte-identical to zstd
//! 1.5.7's `ZSTD_compressStream2` given the same inputs, output capacities and
//! operations; a decoder handle reads any sequence of Zstandard and skippable
//! frames, as `ZSTD_decompressStream` does. Handles keep their buffers between
//! frames, so a reset is much cheaper than a new handle.

use crate::MzstdResult;
use crate::core::session;
use std::ffi::c_int;

/// The `pledged_size` that leaves a frame's size undeclared;
/// `MZSTD_CONTENT_SIZE_UNKNOWN` in C, equal to zstd's
/// `ZSTD_CONTENTSIZE_UNKNOWN`.
///
/// # Examples
///
/// ```
/// assert_eq!(mzstd_ffi::MZSTD_CONTENT_SIZE_UNKNOWN, u64::MAX);
/// ```
pub const MZSTD_CONTENT_SIZE_UNKNOWN: u64 = u64::MAX;

/// What [`mzstd_encoder_process`] should do; `mzstd_operation` in C, zstd's
/// `ZSTD_EndDirective`.
///
/// The function takes the discriminant as an `int`, so a value C passes that
/// names no operation is rejected rather than undefined. The discriminants
/// are part of the ABI and never change.
///
/// # Examples
///
/// ```
/// use mzstd_ffi::MzstdOperation;
///
/// assert_eq!(MzstdOperation::Continue as i32, 0);
/// assert_eq!(MzstdOperation::End as i32, 2);
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MzstdOperation {
    /// `MZSTD_CONTINUE`: take input and compress whole blocks as they fill.
    Continue = 0,
    /// `MZSTD_FLUSH`: make everything taken so far decodable without ending
    /// the frame. The current block ends early, which costs ratio.
    Flush = 1,
    /// `MZSTD_END`: compress everything left and end the frame.
    End = 2,
}

/// Whether more input may follow a [`mzstd_decoder_process`] call;
/// `mzstd_decode_operation` in C.
///
/// The function takes the discriminant as an `int`; the discriminants are
/// part of the ABI and never change.
///
/// # Examples
///
/// ```
/// use mzstd_ffi::MzstdDecodeOperation;
///
/// assert_eq!(MzstdDecodeOperation::Finish as i32, 1);
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MzstdDecodeOperation {
    /// `MZSTD_DECODE_PROCESS`: more input may follow; an empty input does not
    /// end it.
    Process = 0,
    /// `MZSTD_DECODE_FINISH`: this is all the remaining input. Every later
    /// call of the operation must use it too, with exactly the input not yet
    /// consumed.
    Finish = 1,
}

/// What a session needs after a successful process call; `mzstd_status` in
/// C.
///
/// Handle the reported byte counts before acting on it. The discriminants are
/// part of the ABI and never change.
///
/// # Examples
///
/// ```
/// use mzstd_ffi::MzstdStatus;
///
/// assert_eq!(MzstdStatus::Finished as i32, 2);
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MzstdStatus {
    /// `MZSTD_NEEDS_INPUT`: the call did all it could with its input; pass
    /// more, or the next operation.
    NeedsInput = 0,
    /// `MZSTD_NEEDS_OUTPUT`: output is waiting; call again with room for it.
    NeedsOutput = 1,
    /// `MZSTD_FINISHED`: the frame (encoder) or the operation (decoder) is
    /// complete and all its output delivered. Later calls report this again
    /// and move nothing until the handle is reset.
    Finished = 2,
}

/// An incremental encoder; the opaque `mzstd_encoder` in C.
///
/// Created by [`mzstd_encoder_create`] and released by
/// [`mzstd_encoder_free`]. A handle may move between threads but must not be
/// used by two calls at once.
#[derive(Debug)]
pub struct MzstdEncoder(pub(crate) session::Encoder);

/// An incremental decoder; the opaque `mzstd_decoder` in C.
///
/// Created by [`mzstd_decoder_create`] and released by
/// [`mzstd_decoder_free`]. A handle may move between threads but must not be
/// used by two calls at once.
#[derive(Debug)]
pub struct MzstdDecoder(pub(crate) session::Decoder);

/// Creates an encoder at `level` and starts its first frame, of undeclared
/// size.
///
/// `level` is `-131072..=22`; `0` selects the default level, 3. Frames record
/// their content size when it is known and carry no checksum, as with
/// `ZSTD_compressStream2` on a fresh context. On return `*encoder` holds the
/// new handle on success and null on any other status, except a null or
/// misaligned `encoder`, which is left untouched.
///
/// # Errors
///
/// - [`MzstdResult::InvalidParameter`] for a null or misaligned `encoder` or
///   a level out of range.
/// - [`MzstdResult::Error`] for any other failure.
///
/// # Safety
///
/// `encoder` must be null or point to an `mzstd_encoder *` valid for writes.
///
/// # Examples
///
/// ```
/// use mzstd_ffi::{MzstdResult, mzstd_encoder_create, mzstd_encoder_free};
///
/// let mut encoder = std::ptr::null_mut();
/// // SAFETY: `encoder` is a writable pointer slot.
/// assert_eq!(unsafe { mzstd_encoder_create(19, &mut encoder) }, MzstdResult::Ok);
/// // SAFETY: the handle came from `mzstd_encoder_create`.
/// unsafe { mzstd_encoder_free(encoder) };
///
/// // SAFETY: as above.
/// assert_eq!(unsafe { mzstd_encoder_create(23, &mut encoder) }, MzstdResult::InvalidParameter);
/// assert!(encoder.is_null());
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mzstd_encoder_create(
    level: c_int,
    encoder: *mut *mut MzstdEncoder,
) -> MzstdResult {
    // SAFETY: forwarded verbatim from this function's own contract.
    unsafe { session::create(encoder, || session::encoder(level)) }
}

/// Abandons the encoder's current frame, if any, and starts a new one.
///
/// `pledged_size` declares the new frame's exact input size, which selects
/// parameters for it and is recorded in its header, as with
/// `ZSTD_CCtx_setPledgedSrcSize`; [`MZSTD_CONTENT_SIZE_UNKNOWN`] leaves it
/// undeclared. A reset also clears an earlier failure.
///
/// # Errors
///
/// - [`MzstdResult::InvalidParameter`] for a null or misaligned `encoder`.
/// - [`MzstdResult::Error`] when a panic interrupted an earlier call: such a
///   handle can only be released.
///
/// # Safety
///
/// `encoder` must be null or a live handle from [`mzstd_encoder_create`],
/// and no other call may use it at the same time.
///
/// # Examples
///
/// See [`mzstd_encoder_process`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mzstd_encoder_reset(
    encoder: *mut MzstdEncoder,
    pledged_size: u64,
) -> MzstdResult {
    // SAFETY: forwarded verbatim from this function's own contract.
    unsafe {
        session::apply(encoder, |encoder| {
            session::reset_encoder(encoder, pledged_size);
        })
    }
}

/// Moves the encoder's frame forward by one step, as `ZSTD_compressStream2`.
///
/// On entry `*input_len` is the number of bytes offered at `input` and
/// `*output_len` the capacity of `output`; on success they are the bytes
/// consumed and produced, and `*status` says what the session needs next.
/// After any other status both lengths are `0` and `*status` is untouched,
/// except when an out-pointer is null or misaligned, in which case none of
/// them is written.
///
/// `operation` is an [`MzstdOperation`]. An `MZSTD_END` or `MZSTD_FLUSH`
/// that reports `MZSTD_NEEDS_OUTPUT` must be repeated, with the unconsumed
/// input, until it reports `MZSTD_FINISHED` or `MZSTD_NEEDS_INPUT`
/// respectively. The same sequence of inputs, capacities and operations
/// produces the same bytes as the reference. `output` may be uninitialized;
/// treat only the produced bytes as written, since scratch bytes past them may
/// be overwritten too.
///
/// # Errors
///
/// - [`MzstdResult::InvalidParameter`] for a null or misaligned `encoder`,
///   `input_len`, `output_len` or `status`, a null buffer with a non-zero
///   length, a length above `PTRDIFF_MAX`, overlapping arguments, or an
///   undefined `operation`. The session is unaffected.
/// - [`MzstdResult::Error`] for input that contradicts a pledged size, for a
///   session that failed earlier and was not reset, and for any other
///   failure. The frame is then abandoned.
///
/// # Safety
///
/// - `encoder` must be null or a live handle from [`mzstd_encoder_create`].
/// - `input_len`, `output_len` and `status` must each be null or valid for
///   reads and writes of their type.
/// - When `*input_len` is not zero, `input` must be null or valid for reads of
///   `*input_len` bytes; when `*output_len` is not zero, `output` must be null
///   or valid for writes of `*output_len` bytes.
/// - No other thread may use `encoder`, write to `input`, or access `output`
///   or the out-pointers during the call, and no buffer may overlap the
///   handle's internal memory.
///
/// # Examples
///
/// ```
/// use mzstd_ffi::{
///     MzstdOperation, MzstdResult, MzstdStatus, mzstd_encoder_create, mzstd_encoder_free,
///     mzstd_encoder_process, mzstd_encoder_reset,
/// };
///
/// let payload = b"one chunk, then another chunk, then the end".repeat(4);
/// let mut encoder = std::ptr::null_mut();
/// // SAFETY: `encoder` is a writable pointer slot.
/// assert_eq!(unsafe { mzstd_encoder_create(3, &mut encoder) }, MzstdResult::Ok);
/// // SAFETY: the handle is live.
/// assert_eq!(unsafe { mzstd_encoder_reset(encoder, payload.len() as u64) }, MzstdResult::Ok);
///
/// let mut frame = Vec::new();
/// let mut input = payload.as_slice();
/// let mut buffer = [0u8; 32];
/// loop {
///     let (mut input_len, mut output_len) = (input.len(), buffer.len());
///     let mut status = MzstdStatus::NeedsInput;
///     // SAFETY: the handle is live and every buffer has its stated length.
///     let result = unsafe {
///         mzstd_encoder_process(
///             encoder,
///             input.as_ptr(),
///             &mut input_len,
///             buffer.as_mut_ptr(),
///             &mut output_len,
///             MzstdOperation::End as i32,
///             &mut status,
///         )
///     };
///     assert_eq!(result, MzstdResult::Ok);
///     input = &input[input_len..];
///     frame.extend_from_slice(&buffer[..output_len]);
///     if status == MzstdStatus::Finished {
///         break;
///     }
/// }
/// // SAFETY: the handle is live and released once.
/// unsafe { mzstd_encoder_free(encoder) };
/// assert!(input.is_empty());
/// assert_eq!(&frame[..4], &[0x28, 0xb5, 0x2f, 0xfd]);
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mzstd_encoder_process(
    encoder: *mut MzstdEncoder,
    input: *const u8,
    input_len: *mut usize,
    output: *mut u8,
    output_len: *mut usize,
    operation: c_int,
    status: *mut MzstdStatus,
) -> MzstdResult {
    // SAFETY: forwarded verbatim from this function's own contract.
    unsafe {
        session::process(
            encoder, input, input_len, output, output_len, operation, status,
        )
    }
}

/// Releases an encoder, abandoning an unfinished frame. A null `encoder` is
/// ignored.
///
/// # Safety
///
/// `encoder` must be null or a live handle from [`mzstd_encoder_create`] that
/// no other call is using; it is dangling afterwards.
///
/// # Examples
///
/// ```
/// // SAFETY: null is always accepted.
/// unsafe { mzstd_ffi::mzstd_encoder_free(std::ptr::null_mut()) };
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mzstd_encoder_free(encoder: *mut MzstdEncoder) {
    // SAFETY: forwarded verbatim from this function's own contract.
    unsafe { session::free(encoder) }
}

/// Creates a decoder and starts its first operation.
///
/// It accepts frames whose window is at most `2^window_log_max` bytes;
/// `window_log_max` is `10..=31`, or `0` for the default, 27 (128 MiB), as
/// zstd's `ZSTD_d_windowLogMax`. Unlike the one-shot `mzstd_decompress`, a
/// session allocates its window itself, so the limit bounds its memory. On
/// return `*decoder` holds the new handle on success and null on any other
/// status, except a null or misaligned `decoder`, which is left untouched.
///
/// # Errors
///
/// - [`MzstdResult::InvalidParameter`] for a null or misaligned `decoder` or
///   a window log out of range.
/// - [`MzstdResult::Error`] for any other failure.
///
/// # Safety
///
/// `decoder` must be null or point to an `mzstd_decoder *` valid for writes.
///
/// # Examples
///
/// ```
/// use mzstd_ffi::{MzstdResult, mzstd_decoder_create, mzstd_decoder_free};
///
/// let mut decoder = std::ptr::null_mut();
/// // SAFETY: `decoder` is a writable pointer slot.
/// assert_eq!(unsafe { mzstd_decoder_create(0, &mut decoder) }, MzstdResult::Ok);
/// // SAFETY: the handle came from `mzstd_decoder_create`.
/// unsafe { mzstd_decoder_free(decoder) };
///
/// // SAFETY: as above.
/// assert_eq!(unsafe { mzstd_decoder_create(32, &mut decoder) }, MzstdResult::InvalidParameter);
/// assert!(decoder.is_null());
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mzstd_decoder_create(
    window_log_max: c_int,
    decoder: *mut *mut MzstdDecoder,
) -> MzstdResult {
    // SAFETY: forwarded verbatim from this function's own contract.
    unsafe { session::create(decoder, || session::decoder(window_log_max)) }
}

/// Abandons the decoder's current operation, finished or not, and starts a
/// new one. A reset also clears an earlier failure.
///
/// # Errors
///
/// - [`MzstdResult::InvalidParameter`] for a null or misaligned `decoder`.
/// - [`MzstdResult::Error`] when a panic interrupted an earlier call: such a
///   handle can only be released.
///
/// # Safety
///
/// `decoder` must be null or a live handle from [`mzstd_decoder_create`],
/// and no other call may use it at the same time.
///
/// # Examples
///
/// See [`mzstd_decoder_process`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mzstd_decoder_reset(decoder: *mut MzstdDecoder) -> MzstdResult {
    // SAFETY: forwarded verbatim from this function's own contract.
    unsafe { session::apply(decoder, session::reset_decoder) }
}

/// Decodes as much of the offered input as it can, as
/// `ZSTD_decompressStream`.
///
/// On entry `*input_len` is the number of bytes offered at `input` and
/// `*output_len` the capacity of `output`; on success they are the bytes
/// consumed and produced, and `*status` says what the session needs next.
/// After any other status both lengths are `0` and `*status` is untouched,
/// except when an out-pointer is null or misaligned, in which case none of
/// them is written.
///
/// `operation` is an [`MzstdDecodeOperation`]. Use `MZSTD_DECODE_PROCESS`
/// while more input may arrive; once the input is known to end, use
/// `MZSTD_DECODE_FINISH` on every remaining call, each time offering exactly
/// the input not yet consumed, until the status is `MZSTD_FINISHED`. Frames
/// may be concatenated, and skippable frames are skipped; a frame boundary
/// reports `MZSTD_NEEDS_INPUT` until `MZSTD_DECODE_FINISH` confirms that the
/// input ends there. `output` may be uninitialized; no byte after the
/// produced ones is written.
///
/// # Errors
///
/// - [`MzstdResult::InvalidParameter`] for a null or misaligned `decoder`,
///   `input_len`, `output_len` or `status`, a null buffer with a non-zero
///   length, a length above `PTRDIFF_MAX`, overlapping arguments, or an
///   undefined `operation`. The session is unaffected.
/// - [`MzstdResult::Error`] for corrupt input, a window above the limit, a
///   checksum or content size mismatch, a frame that needs a dictionary,
///   input that ends inside a frame, a changed operation or final input after
///   `MZSTD_DECODE_FINISH`, a session that failed earlier and was not reset,
///   and any other failure. The output produced by the operation so far is
///   then unvalidated.
///
/// # Safety
///
/// As [`mzstd_encoder_process`], with a live handle from
/// [`mzstd_decoder_create`].
///
/// # Examples
///
/// ```
/// use mzstd_ffi::{
///     MzstdDecodeOperation, MzstdResult, MzstdStatus, mzstd_decoder_create, mzstd_decoder_free,
///     mzstd_decoder_process, mzstd_decoder_reset,
/// };
///
/// // "hello" as one raw block, delivered two bytes at a time.
/// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x05, 0x29, 0x00, 0x00, b'h', b'e', b'l', b'l', b'o'];
/// let mut decoder = std::ptr::null_mut();
/// // SAFETY: `decoder` is a writable pointer slot.
/// assert_eq!(unsafe { mzstd_decoder_create(0, &mut decoder) }, MzstdResult::Ok);
/// for _ in 0..2 {
///     let mut remaining = &frame[..];
///     let mut decoded = Vec::new();
///     loop {
///         let mut buffer = [0u8; 2];
///         let (mut input_len, mut output_len) = (remaining.len(), buffer.len());
///         let mut status = MzstdStatus::NeedsInput;
///         // SAFETY: the handle is live and every buffer has its stated length.
///         let result = unsafe {
///             mzstd_decoder_process(
///                 decoder,
///                 remaining.as_ptr(),
///                 &mut input_len,
///                 buffer.as_mut_ptr(),
///                 &mut output_len,
///                 MzstdDecodeOperation::Finish as i32,
///                 &mut status,
///             )
///         };
///         assert_eq!(result, MzstdResult::Ok);
///         remaining = &remaining[input_len..];
///         decoded.extend_from_slice(&buffer[..output_len]);
///         if status == MzstdStatus::Finished {
///             break;
///         }
///     }
///     assert_eq!(decoded, b"hello");
///     // SAFETY: the handle is live.
///     assert_eq!(unsafe { mzstd_decoder_reset(decoder) }, MzstdResult::Ok);
/// }
/// // SAFETY: the handle is live and released once.
/// unsafe { mzstd_decoder_free(decoder) };
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mzstd_decoder_process(
    decoder: *mut MzstdDecoder,
    input: *const u8,
    input_len: *mut usize,
    output: *mut u8,
    output_len: *mut usize,
    operation: c_int,
    status: *mut MzstdStatus,
) -> MzstdResult {
    // SAFETY: forwarded verbatim from this function's own contract.
    unsafe {
        session::process(
            decoder, input, input_len, output, output_len, operation, status,
        )
    }
}

/// Releases a decoder, abandoning an unfinished operation. A null `decoder`
/// is ignored.
///
/// # Safety
///
/// `decoder` must be null or a live handle from [`mzstd_decoder_create`] that
/// no other call is using; it is dangling afterwards.
///
/// # Examples
///
/// ```
/// // SAFETY: null is always accepted.
/// unsafe { mzstd_ffi::mzstd_decoder_free(std::ptr::null_mut()) };
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mzstd_decoder_free(decoder: *mut MzstdDecoder) {
    // SAFETY: forwarded verbatim from this function's own contract.
    unsafe { session::free(decoder) }
}

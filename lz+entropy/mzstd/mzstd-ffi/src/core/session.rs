//! Safe implementation behind the exported session functions.
//!
//! A session handle is a boxed [`MzstdEncoder`] or [`MzstdDecoder`] the C
//! caller holds between calls. Each exported function forwards here: the
//! out-pointers are validated first and the handle is dereferenced only after
//! every buffer has passed [`Arguments::new`]. The codec runs inside
//! [`run`], which stops a panic at the boundary and poisons the handle, since
//! a session interrupted mid-step may hold half-updated state.

use super::{Arguments, FfiError, Slot, compressor, guarded};
use crate::{
    MZSTD_CONTENT_SIZE_UNKNOWN, MzstdDecodeOperation, MzstdDecoder, MzstdEncoder, MzstdOperation,
    MzstdResult, MzstdStatus,
};
use mzstd::{
    DecodeError, DecodeLimits, DecodeOperation, DecoderConfig, DecoderSessionOwned, DecoderStatus,
    Decompressor, EncoderSessionOwned, EncoderStatus, InputSize, Operation, StreamConfig,
};
use std::ffi::c_int;
use std::mem::MaybeUninit;

/// The state behind an [`MzstdEncoder`]: one frame at a time on a reused
/// compressor.
#[derive(Debug)]
pub(crate) struct Encoder {
    session: EncoderSessionOwned,
    poisoned: bool,
}

/// The state behind an [`MzstdDecoder`]: one operation over any number of
/// frames on a reused decompressor.
#[derive(Debug)]
pub(crate) struct Decoder {
    session: DecoderSessionOwned,
    poisoned: bool,
}

/// What one successful process call moved, in the C ABI's terms.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct Step {
    consumed: usize,
    produced: usize,
    status: MzstdStatus,
}

impl From<EncoderStatus> for MzstdStatus {
    fn from(status: EncoderStatus) -> Self {
        match status {
            EncoderStatus::NeedsInput => Self::NeedsInput,
            EncoderStatus::NeedsOutput => Self::NeedsOutput,
            EncoderStatus::Finished => Self::Finished,
        }
    }
}

impl From<DecoderStatus> for MzstdStatus {
    fn from(status: DecoderStatus) -> Self {
        match status {
            DecoderStatus::NeedsInput => Self::NeedsInput,
            DecoderStatus::NeedsOutput => Self::NeedsOutput,
            DecoderStatus::Finished => Self::Finished,
        }
    }
}

/// The operations a handle's process call takes and how it runs one step.
pub(crate) trait Handle {
    /// The parsed C `operation` argument.
    type Operation: Copy;

    /// Parses the C `operation` argument.
    ///
    /// # Errors
    ///
    /// [`FfiError::Operation`] for a value that names no operation.
    fn operation(value: c_int) -> Result<Self::Operation, FfiError>;

    /// Runs one step of the session.
    ///
    /// # Errors
    ///
    /// The session's own failure, as [`FfiError::Encode`] or
    /// [`FfiError::Decode`].
    fn step(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        operation: Self::Operation,
    ) -> Result<Step, FfiError>;

    /// Whether a panic interrupted an earlier call.
    fn poisoned(&mut self) -> &mut bool;
}

impl Handle for MzstdEncoder {
    type Operation = Operation;

    fn operation(value: c_int) -> Result<Operation, FfiError> {
        match value {
            v if v == MzstdOperation::Continue as c_int => Ok(Operation::Continue),
            v if v == MzstdOperation::Flush as c_int => Ok(Operation::Flush),
            v if v == MzstdOperation::End as c_int => Ok(Operation::End),
            value => Err(FfiError::Operation { value }),
        }
    }

    fn step(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        operation: Operation,
    ) -> Result<Step, FfiError> {
        let progress = self.0.session.process_uninit(input, output, operation)?;
        Ok(Step {
            consumed: progress.consumed,
            produced: progress.produced,
            status: progress.status.into(),
        })
    }

    fn poisoned(&mut self) -> &mut bool {
        &mut self.0.poisoned
    }
}

impl Handle for MzstdDecoder {
    type Operation = DecodeOperation;

    fn operation(value: c_int) -> Result<DecodeOperation, FfiError> {
        match value {
            v if v == MzstdDecodeOperation::Process as c_int => Ok(DecodeOperation::Process),
            v if v == MzstdDecodeOperation::Finish as c_int => Ok(DecodeOperation::Finish),
            value => Err(FfiError::Operation { value }),
        }
    }

    fn step(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        operation: DecodeOperation,
    ) -> Result<Step, FfiError> {
        let progress = self
            .0
            .session
            .process_uninit(input, output, operation)
            .map_err(DecodeError::from)?;
        Ok(Step {
            consumed: progress.consumed,
            produced: progress.produced,
            status: progress.status.into(),
        })
    }

    fn poisoned(&mut self) -> &mut bool {
        &mut self.0.poisoned
    }
}

/// A new encoder at `level_value` with its first frame started, of unknown
/// size.
///
/// # Errors
///
/// [`FfiError::Level`] for a level out of range.
pub(crate) fn encoder(level_value: c_int) -> Result<MzstdEncoder, FfiError> {
    let session = compressor(level_value)?.into_session(StreamConfig::default());
    Ok(MzstdEncoder(Encoder {
        session,
        poisoned: false,
    }))
}

/// A new decoder of concatenated frames whose windows are at most
/// `2^window_log_max` bytes, `0` selecting the default 2^27.
///
/// # Errors
///
/// [`FfiError::WindowLog`] for a window log out of range.
pub(crate) fn decoder(window_log_max: c_int) -> Result<MzstdDecoder, FfiError> {
    let config = DecoderConfig::DEFAULT.with_limits(window_limits(window_log_max)?);
    Ok(MzstdDecoder(Decoder {
        session: Decompressor::from(config).into_session(),
        poisoned: false,
    }))
}

/// Decoder limits for the C `window_log_max` argument: the default for `0`,
/// otherwise `10..=31`, and no input or output budget.
///
/// # Errors
///
/// [`FfiError::WindowLog`] for any other value.
pub(crate) fn window_limits(value: c_int) -> Result<DecodeLimits, FfiError> {
    if value == 0 {
        return Ok(DecodeLimits::DEFAULT);
    }
    // A value no `u8` holds is out of range as well; `0` stands in for it,
    // since `with_max_window_log` refuses it too.
    let log = u8::try_from(value).unwrap_or(0);
    DecodeLimits::DEFAULT
        .with_max_window_log(log)
        .map_err(|source| FfiError::WindowLog { value, source })
}

/// The frame a C `pledged_size` describes.
pub(crate) fn stream(pledged_size: u64) -> StreamConfig {
    StreamConfig::from(if pledged_size == MZSTD_CONTENT_SIZE_UNKNOWN {
        InputSize::Unknown
    } else {
        InputSize::Exact(pledged_size)
    })
}

/// Starts the encoder's next frame, abandoning any unfinished one.
pub(crate) fn reset_encoder(encoder: &mut MzstdEncoder, pledged_size: u64) {
    encoder.0.session.reinit(stream(pledged_size));
}

/// Starts the decoder's next operation, abandoning any unfinished one.
pub(crate) fn reset_decoder(decoder: &mut MzstdDecoder) {
    decoder.0.session.reinit();
}

/// Runs `body` on `handle` unless it is poisoned, and poisons it when `body`
/// panics.
///
/// # Errors
///
/// [`FfiError::Poisoned`] for a poisoned handle, [`FfiError::Panic`] when
/// `body` panics, and any error `body` returns.
fn run<H: Handle, T>(
    handle: &mut H,
    body: impl FnOnce(&mut H) -> Result<T, FfiError>,
) -> Result<T, FfiError> {
    if *handle.poisoned() {
        return Err(FfiError::Poisoned);
    }
    let outcome = guarded(|| body(handle));
    if matches!(outcome, Err(FfiError::Panic)) {
        *handle.poisoned() = true;
    }
    outcome
}

/// Builds a handle and stores it through `out`, or stores null on failure.
///
/// # Safety
///
/// A non-null, aligned `out` must be valid for writing a pointer.
pub(crate) unsafe fn create<H>(
    out: *mut *mut H,
    build: impl FnOnce() -> Result<H, FfiError>,
) -> MzstdResult {
    let slot = match Slot::new(out) {
        Ok(slot) => slot,
        Err(error) => return MzstdResult::from(&error),
    };
    let (status, handle) = match guarded(|| build().map(Box::new)) {
        Ok(handle) => (MzstdResult::Ok, Box::into_raw(handle)),
        Err(error) => (MzstdResult::from(&error), std::ptr::null_mut()),
    };
    // SAFETY: `slot` is non-null and aligned and the caller guarantees it is
    // writable.
    unsafe { slot.pointer.write(handle) };
    status
}

/// Releases a handle `create` returned. Null and misaligned pointers, which
/// `create` never returns, are ignored.
///
/// # Safety
///
/// A non-null, aligned `handle` must have come from [`create`] for the same
/// `H` and not have been released, and no other call may be using it.
pub(crate) unsafe fn free<H>(handle: *mut H) {
    if handle.is_null() || !handle.is_aligned() {
        return;
    }
    // SAFETY: the caller guarantees `handle` is a live `Box<H>` from
    // `create`, now owned by this call alone.
    let handle = unsafe { Box::from_raw(handle) };
    // A panic while dropping only leaks what was not yet released.
    let _ = guarded(|| {
        drop(handle);
        Ok(())
    });
}

/// Runs `body` on a live handle and returns its status.
///
/// # Safety
///
/// A non-null, aligned `handle` must have come from [`create`] for the same
/// `H` and not have been released, and no other call may be using it.
pub(crate) unsafe fn apply<H: Handle>(handle: *mut H, body: impl FnOnce(&mut H)) -> MzstdResult {
    let outcome = Slot::new(handle).and_then(|slot| {
        // SAFETY: non-null and aligned (checked); live and unshared (caller).
        let handle = unsafe { &mut *slot.pointer };
        run(handle, |handle| {
            body(handle);
            Ok(())
        })
    });
    match outcome {
        Ok(()) => MzstdResult::Ok,
        Err(error) => MzstdResult::from(&error),
    }
}

/// The out-pointers of one process call, validated.
struct Outputs {
    input_len: Slot<usize>,
    output_len: Slot<usize>,
    status: Slot<MzstdStatus>,
}

impl Outputs {
    /// Validates the three out-pointers. None of them is written when any
    /// fails, since it may not be memory.
    fn new(
        input_len: *mut usize,
        output_len: *mut usize,
        status: *mut MzstdStatus,
    ) -> Result<Self, FfiError> {
        Ok(Self {
            input_len: Slot::new(input_len)?,
            output_len: Slot::new(output_len)?,
            status: Slot::new(status)?,
        })
    }
}

/// Runs one process call end to end and returns its status.
///
/// On success the bytes consumed and produced are stored through `input_len`
/// and `output_len` and the session's status through `status`. On any failure
/// after the three have been validated, zero is stored through both lengths
/// and `status` is left untouched.
///
/// # Safety
///
/// As documented on `mzstd_encoder_process` and `mzstd_decoder_process`:
/// non-null, aligned out-pointers must be valid for reads and writes of their
/// type, a non-null, aligned `handle` must be a live handle of type `H` that
/// no other call is using, and the buffers must satisfy
/// [`Arguments::slices`].
pub(crate) unsafe fn process<H: Handle>(
    handle: *mut H,
    input: *const u8,
    input_len: *mut usize,
    output: *mut u8,
    output_len: *mut usize,
    operation: c_int,
    status: *mut MzstdStatus,
) -> MzstdResult {
    let outputs = match Outputs::new(input_len, output_len, status) {
        Ok(outputs) => outputs,
        Err(error) => return MzstdResult::from(&error),
    };
    // SAFETY: both slots are non-null and aligned, and the caller guarantees
    // such pointers are valid for reading a `size_t`.
    let (offered, capacity) = unsafe {
        (
            outputs.input_len.pointer.read(),
            outputs.output_len.pointer.read(),
        )
    };
    let outcome = Slot::new(handle).and_then(|handle| {
        let operation = H::operation(operation)?;
        let slots = [
            outputs.input_len.region,
            outputs.output_len.region,
            outputs.status.region,
            handle.region,
        ];
        let arguments = Arguments::new(input, offered, output, capacity, &slots)?;
        // SAFETY: the buffers passed `Arguments::new`, and the caller
        // guarantees their validity and exclusivity for this call, which
        // outlives both slices.
        let (src, dst) = unsafe { arguments.slices() };
        // SAFETY: non-null and aligned (checked), disjoint from both buffers
        // and every out-pointer (checked), live and unshared (caller).
        let handle = unsafe { &mut *handle.pointer };
        run(handle, |handle| handle.step(src, dst, operation))
    });
    let (result, consumed, produced) = match outcome {
        Ok(step) => {
            // SAFETY: non-null, aligned and writable (checked and caller),
            // and nothing borrowed from the call is alive any more.
            unsafe { outputs.status.pointer.write(step.status) };
            (MzstdResult::Ok, step.consumed, step.produced)
        }
        Err(error) => (MzstdResult::from(&error), 0, 0),
    };
    // SAFETY: as for the reads above; the slices and the handle borrow are
    // dead, and the slots were checked disjoint from both buffers.
    unsafe {
        outputs.input_len.pointer.write(consumed);
        outputs.output_len.pointer.write(produced);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use mzstd::DecodeConfigError;

    /// A destination of `len` bytes that were never initialized.
    fn uninit(len: usize) -> Vec<MaybeUninit<u8>> {
        vec![MaybeUninit::uninit(); len]
    }

    #[test]
    fn statuses_map_one_to_one() {
        assert_eq!(
            MzstdStatus::from(EncoderStatus::NeedsInput),
            MzstdStatus::NeedsInput
        );
        assert_eq!(
            MzstdStatus::from(EncoderStatus::NeedsOutput),
            MzstdStatus::NeedsOutput
        );
        assert_eq!(
            MzstdStatus::from(EncoderStatus::Finished),
            MzstdStatus::Finished
        );
        assert_eq!(
            MzstdStatus::from(DecoderStatus::NeedsInput),
            MzstdStatus::NeedsInput
        );
        assert_eq!(
            MzstdStatus::from(DecoderStatus::NeedsOutput),
            MzstdStatus::NeedsOutput
        );
        assert_eq!(
            MzstdStatus::from(DecoderStatus::Finished),
            MzstdStatus::Finished
        );
    }

    #[test]
    fn operations_parse_exactly_the_header_values() {
        assert_eq!(MzstdEncoder::operation(0).ok(), Some(Operation::Continue));
        assert_eq!(MzstdEncoder::operation(1).ok(), Some(Operation::Flush));
        assert_eq!(MzstdEncoder::operation(2).ok(), Some(Operation::End));
        assert_eq!(
            MzstdDecoder::operation(0).ok(),
            Some(DecodeOperation::Process)
        );
        assert_eq!(
            MzstdDecoder::operation(1).ok(),
            Some(DecodeOperation::Finish)
        );
        for value in [-1, 3, c_int::MIN, c_int::MAX] {
            assert!(matches!(
                MzstdEncoder::operation(value),
                Err(FfiError::Operation { value: v }) if v == value
            ));
        }
        for value in [-1, 2, c_int::MAX] {
            assert!(matches!(
                MzstdDecoder::operation(value),
                Err(FfiError::Operation { value: v }) if v == value
            ));
        }
    }

    #[test]
    fn window_limits_accept_zero_and_the_format_range() {
        assert_eq!(window_limits(0).ok(), Some(DecodeLimits::DEFAULT));
        for log in [10, 27, 31] {
            let limits = window_limits(log).expect("in range");
            assert_eq!(c_int::from(limits.max_window_log()), log);
        }
        for value in [-1, 9, 32, 256, c_int::MIN, c_int::MAX] {
            assert!(matches!(
                window_limits(value),
                Err(FfiError::WindowLog {
                    value: v,
                    source: DecodeConfigError::WindowLog { .. }
                }) if v == value
            ));
        }
    }

    #[test]
    fn the_unknown_sentinel_leaves_the_size_undeclared() {
        assert_eq!(
            stream(MZSTD_CONTENT_SIZE_UNKNOWN).input_size(),
            InputSize::Unknown
        );
        assert_eq!(stream(0).input_size(), InputSize::Exact(0));
        assert_eq!(
            stream(u64::MAX - 1).input_size(),
            InputSize::Exact(u64::MAX - 1)
        );
    }

    #[test]
    fn a_panic_poisons_the_handle_for_good() {
        let mut handle = decoder(0).expect("default window");
        let outcome: Result<(), _> = run(&mut handle, |_| panic!("codec bug"));
        assert!(matches!(outcome, Err(FfiError::Panic)));
        let pointer = &raw mut handle;
        // SAFETY: `pointer` is a live, unshared handle.
        assert_eq!(unsafe { apply(pointer, reset_decoder) }, MzstdResult::Error);
        let (mut input_len, mut output_len) = (0, 0);
        let mut status = MzstdStatus::NeedsOutput;
        // SAFETY: as above; both buffers are empty.
        let result = unsafe {
            process(
                pointer,
                std::ptr::null(),
                &mut input_len,
                std::ptr::null_mut(),
                &mut output_len,
                MzstdDecodeOperation::Finish as c_int,
                &mut status,
            )
        };
        assert_eq!(result, MzstdResult::Error);
        assert_eq!(status, MzstdStatus::NeedsOutput);
    }

    #[test]
    fn an_error_does_not_poison_the_handle() {
        let mut handle = decoder(0).expect("default window");
        let truncated = [0x28, 0xb5, 0x2f, 0xfd];
        assert!(matches!(
            run(&mut handle, |handle| handle.step(
                &truncated,
                &mut uninit(8),
                DecodeOperation::Finish
            )),
            Err(FfiError::Decode(DecodeError::UnexpectedEndOfInput))
        ));
        reset_decoder(&mut handle);
        let frame = [
            0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k',
        ];
        let step = run(&mut handle, |handle| {
            handle.step(&frame, &mut uninit(8), DecodeOperation::Finish)
        })
        .expect("decodes after a reset");
        assert_eq!(
            step,
            Step {
                consumed: frame.len(),
                produced: 2,
                status: MzstdStatus::Finished
            }
        );
    }

    #[test]
    fn an_encoder_writes_one_frame_per_reset() {
        let mut handle = encoder(1).expect("level 1");
        let mut frames = Vec::new();
        for pledged in [MZSTD_CONTENT_SIZE_UNKNOWN, 5] {
            reset_encoder(&mut handle, pledged);
            let mut output = uninit(64);
            let step = run(&mut handle, |handle| {
                handle.step(b"hello", &mut output, Operation::End)
            })
            .expect("compresses");
            assert_eq!((step.consumed, step.status), (5, MzstdStatus::Finished));
            frames.push(step.produced);
        }
        // Both frames know their size: a first `End` records the input's.
        assert_eq!(frames[0], frames[1]);
    }
}

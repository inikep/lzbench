//! Owned session storage, validation and panic containment for C handles.

use super::{Arguments, FfiError, LengthSlot, Region, quality, window};
use crate::{MbrotliDecoderSession, MbrotliEncoderSession, MbrotliResult};
use mbrotli::{
    Compressor, DecodeOperation, DecoderConfig, DecoderSessionOwned, DecoderStatus, Decompressor,
    EncoderConfig, EncoderSessionOwned, EncoderStatus, Operation,
};
use std::ffi::c_int;
use std::mem::MaybeUninit;
use std::panic::{AssertUnwindSafe, catch_unwind};

// This allocation is private to the loaded library, not a C-layout object.
// C only stores an opaque pointer returned by create and returns it to this
// library. Changing T, field order or allocation size therefore cannot change
// the C ABI. Keeping T inline avoids a second allocation and indirection.
struct Handle<T> {
    session: T,
    poisoned: bool,
}

type Step = (usize, usize, MbrotliResult);

fn create<T>(build: impl FnOnce() -> Result<T, FfiError>) -> *mut Handle<T> {
    catch_unwind(AssertUnwindSafe(|| {
        build().map(|session| {
            Box::into_raw(Box::new(Handle {
                session,
                poisoned: false,
            }))
        })
    }))
    .ok()
    .and_then(Result::ok)
    .unwrap_or(std::ptr::null_mut())
}

fn handle_region<T>(handle: *mut Handle<T>) -> Result<Region, FfiError> {
    if handle.is_null() || !handle.is_aligned() {
        return Err(FfiError::NullBuffer {
            len: size_of::<Handle<T>>(),
        });
    }
    Region::new(handle.addr(), size_of::<Handle<T>>())
}

/// Caller guarantees a live, exclusive allocation of the matching handle type.
unsafe fn destroy<T>(handle: *mut Handle<T>) {
    if handle_region(handle).is_ok() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: caller transfers the live Box allocation back exactly once;
            // null and alignment were checked above, and no aliases are in use.
            drop(unsafe { Box::from_raw(handle) });
        }));
    }
}

/// Caller guarantees a live and exclusively accessed handle, or null.
unsafe fn reinit<T>(
    handle: *mut Handle<T>,
    reset: impl FnOnce(&mut T) -> Result<(), FfiError>,
) -> MbrotliResult {
    if handle_region(handle).is_err() {
        return MbrotliResult::InvalidParameter;
    }
    // SAFETY: validated alignment/null; caller guarantees lifetime and exclusivity.
    let handle = unsafe { &mut *handle };
    handle.poisoned = true;
    match catch_unwind(AssertUnwindSafe(|| reset(&mut handle.session))) {
        Ok(Ok(())) => {
            handle.poisoned = false;
            MbrotliResult::Ok
        }
        _ => MbrotliResult::Error,
    }
}

/// Validates every region before constructing references or changing counts.
/// Caller supplies valid storage and a live exclusive handle, as in the C API.
unsafe fn process<T>(
    handle: *mut Handle<T>,
    input: *const u8,
    input_len: *mut usize,
    output: *mut u8,
    output_len: *mut usize,
    step: impl FnOnce(&mut T, &[u8], &mut [MaybeUninit<u8>]) -> Result<Step, FfiError>,
) -> MbrotliResult {
    let validated = (|| {
        let state = handle_region(handle)?;
        let src_count = LengthSlot::new(input_len)?;
        let dst_count = LengthSlot::new(output_len)?;
        if state.overlaps(src_count.region)
            || state.overlaps(dst_count.region)
            || src_count.region.overlaps(dst_count.region)
        {
            return Err(FfiError::Overlap);
        }
        // SAFETY: null/alignment/overlap were checked; caller provides readable slots.
        let (available, capacity) = unsafe { (input_len.read(), output_len.read()) };
        let args = Arguments::new(input, available, output, capacity, dst_count)?;
        let src = Region::new(input.addr(), available)?;
        let dst = Region::new(output.addr(), capacity)?;
        if src.overlaps(src_count.region)
            || dst.overlaps(src_count.region)
            || state.overlaps(src)
            || state.overlaps(dst)
        {
            return Err(FfiError::Overlap);
        }
        Ok(args)
    })();
    let args = match validated {
        Ok(args) => args,
        Err(_) => return MbrotliResult::InvalidParameter,
    };
    // SAFETY: caller guarantees a live exclusive handle; all argument regions
    // have been checked to be disjoint from it and each other.
    let handle = unsafe { &mut *handle };
    let result = if handle.poisoned {
        (0, 0, MbrotliResult::Error)
    } else {
        // Poison before entering user-independent codec code: panic cannot leave
        // a partially modified state usable by a subsequent process call.
        handle.poisoned = true;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: validation above plus caller storage/exclusivity guarantees.
            let (src, dst) = unsafe { args.slices() };
            step(&mut handle.session, src, dst)
        }));
        match outcome {
            Ok(Ok(progress)) => {
                handle.poisoned = progress.2 == MbrotliResult::Error;
                progress
            }
            _ => (0, 0, MbrotliResult::Error),
        }
    };
    // SAFETY: both writable slots were validated and are disjoint; buffer borrows
    // have ended. Neither slot overlaps the live handle reference.
    unsafe {
        input_len.write(result.0);
        output_len.write(result.1);
    }
    result.2
}

pub(crate) fn encoder_create(quality_value: c_int, lgwin: c_int) -> *mut MbrotliEncoderSession {
    create(|| {
        let config = EncoderConfig::default()
            .with_quality(quality(quality_value)?)
            .with_window(window(lgwin)?);
        Ok(Compressor::new(config)?.into_session(Default::default())?)
    })
    .cast()
}

pub(crate) fn decoder_create() -> *mut MbrotliDecoderSession {
    create(|| Ok(Decompressor::new(DecoderConfig::default())?.into_session(Default::default())?))
        .cast()
}

pub(crate) unsafe fn encoder_destroy(handle: *mut MbrotliEncoderSession) {
    // SAFETY: matching live allocation, transferred by the caller.
    unsafe { destroy(handle.cast::<Handle<EncoderSessionOwned>>()) }
}

pub(crate) unsafe fn decoder_destroy(handle: *mut MbrotliDecoderSession) {
    // SAFETY: matching live allocation, transferred by the caller.
    unsafe { destroy(handle.cast::<Handle<DecoderSessionOwned>>()) }
}

pub(crate) unsafe fn encoder_reinit(handle: *mut MbrotliEncoderSession) -> MbrotliResult {
    // SAFETY: the caller guarantees the matching live, exclusive handle.
    unsafe {
        reinit(handle.cast::<Handle<EncoderSessionOwned>>(), |session| {
            Ok(session.reinit(Default::default())?)
        })
    }
}

pub(crate) unsafe fn decoder_reinit(handle: *mut MbrotliDecoderSession) -> MbrotliResult {
    // SAFETY: the caller guarantees the matching live, exclusive handle.
    unsafe {
        reinit(handle.cast::<Handle<DecoderSessionOwned>>(), |session| {
            Ok(session.reinit(Default::default())?)
        })
    }
}

pub(crate) unsafe fn encoder_process(
    handle: *mut MbrotliEncoderSession,
    input: *const u8,
    input_len: *mut usize,
    output: *mut u8,
    output_len: *mut usize,
    operation: c_int,
) -> MbrotliResult {
    let op = match operation {
        0 => Operation::Process,
        1 => Operation::Flush,
        2 => Operation::Finish,
        _ => return MbrotliResult::InvalidParameter,
    };
    // SAFETY: all raw arguments are forwarded with their original contracts.
    unsafe {
        process(
            handle.cast::<Handle<EncoderSessionOwned>>(),
            input,
            input_len,
            output,
            output_len,
            |session, src, dst| {
                let p = session.process_uninit(src, dst, op)?;
                let status = match p.status {
                    EncoderStatus::NeedsInput => MbrotliResult::NeedsInput,
                    EncoderStatus::NeedsOutput => MbrotliResult::NeedsOutput,
                    EncoderStatus::Finished => MbrotliResult::Finished,
                };
                Ok((p.consumed, p.produced, status))
            },
        )
    }
}

pub(crate) unsafe fn decoder_process(
    handle: *mut MbrotliDecoderSession,
    input: *const u8,
    input_len: *mut usize,
    output: *mut u8,
    output_len: *mut usize,
    operation: c_int,
) -> MbrotliResult {
    let op = match operation {
        0 => DecodeOperation::Process,
        2 => DecodeOperation::Finish,
        _ => return MbrotliResult::InvalidParameter,
    };
    // SAFETY: all raw arguments are forwarded with their original contracts.
    unsafe {
        process(
            handle.cast::<Handle<DecoderSessionOwned>>(),
            input,
            input_len,
            output,
            output_len,
            |session, src, dst| match session.process_uninit(src, dst, op) {
                Ok(p) => {
                    let status = match p.status {
                        DecoderStatus::NeedsInput => MbrotliResult::NeedsInput,
                        DecoderStatus::NeedsOutput => MbrotliResult::NeedsOutput,
                        DecoderStatus::Finished => MbrotliResult::Finished,
                    };
                    Ok((p.consumed, p.produced, status))
                }
                Err(failure) => Ok((failure.consumed, failure.produced, MbrotliResult::Error)),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panics_and_errors_poison_until_successful_reinit() {
        assert!(create::<u64>(|| panic!("constructor failure")).is_null());
        assert!(create::<u64>(|| Err(FfiError::Panic)).is_null());
        let handle = create(|| Ok(0u64));
        assert!(!handle.is_null());
        // SAFETY: handle belongs to this test, all counts are live and disjoint,
        // and empty buffers may be null. Destroy transfers ownership exactly once.
        unsafe {
            let (mut read, mut written) = (0, 0);
            let status = process(
                handle,
                std::ptr::null(),
                &mut read,
                std::ptr::null_mut(),
                &mut written,
                |_, _, _| panic!("process failure"),
            );
            assert_eq!(status, MbrotliResult::Error);
            assert_eq!((read, written), (0, 0));
            assert!((*handle).poisoned);
            assert_eq!(
                reinit(handle, |_| panic!("reset failure")),
                MbrotliResult::Error
            );
            assert_eq!(
                reinit(handle, |_| Err(FfiError::Panic)),
                MbrotliResult::Error
            );
            assert!((*handle).poisoned);
            assert_eq!(reinit(handle, |_| Ok(())), MbrotliResult::Ok);
            assert!(!(*handle).poisoned);
            assert_eq!(
                process(
                    handle,
                    std::ptr::null(),
                    &mut read,
                    std::ptr::null_mut(),
                    &mut written,
                    |_, _, _| Err(FfiError::Panic)
                ),
                MbrotliResult::Error
            );
            destroy(handle);
        }
    }
}

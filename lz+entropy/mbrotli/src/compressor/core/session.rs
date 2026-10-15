//! Public-session ownership and error boundary over the shared block state machine.

use core::mem::MaybeUninit;

use super::stream::{Buffers, Destination, Output, Phase, StreamState};
use crate::compressor::dictionary::PreparedDictionary;
use crate::compressor::encoder::Compressor;
use crate::compressor::error::EncodeError;
use crate::compressor::session::{InputSize, Operation, Progress, StreamConfig};

/// Ceiling C's `UpdateSizeHint` puts on an inferred size hint.
const MAX_INFERRED_SIZE_HINT: usize = 1 << 30;

/// Largest stream of unknown length whose storage is sized for its inferred
/// total, when that stream is finished in the call that first encodes it.
///
/// Up to here the small-input storage pays: the compact and quick slot maps
/// and the on-demand bucket layouts against a cleared dense table (1 KiB JSON
/// at quality 6: 62 -> 22 us). Above it, a fresh encoder sized for a known
/// total measured slower on JSON and HTML than one sized for an unknown total,
/// as before inference existed: up to +24% at qualities 2, 3 and 5 from 6 KiB
/// and at quality 6 from 10 KiB; every quality 2-6 point at or below 4 KiB was
/// within 2% of it or faster.
const SIZED_STORAGE_LIMIT: usize = 4096;

/// Exclusive stream state over the compressor's retained buffers and encoder.
#[derive(Debug)]
pub(crate) struct SessionCore<'c, 'd> {
    compressor: &'c mut Compressor,
    dictionary: Option<&'d PreparedDictionary>,
    operation: OperationState,
}

/// Owned stream state: the compressor and dictionary move in.
///
/// Holds the same [`OperationState`] a borrowed [`SessionCore`] does, beside
/// the compressor rather than over a reference to it, so nothing here points
/// into its own fields. It has no `Drop`: dropping it drops the compressor
/// too, and [`Self::into_compressor`] runs the one shared release path.
#[derive(Debug)]
pub(crate) struct OwnedSessionCore<D> {
    compressor: Compressor,
    dictionary: Option<D>,
    operation: OperationState,
}

/// Non-borrowing operation shared by raw guards and framed drivers.
#[derive(Debug)]
pub(crate) struct OperationState {
    state: StreamState,
    /// The stream as started, which an inferred size hint re-lowers.
    stream: StreamConfig,
    /// Whether the size hint is still to be inferred.
    hint: SizeHint,
    #[cfg(feature = "experimental")]
    logical_position: u64,
}

/// Where a stream's size hint stands.
///
/// A declared length is the hint from the start. A stream of unknown length
/// infers it the way C's `BrotliEncoderCompressStream` does: `UpdateSizeHint`
/// runs right before every `EncodeData` while the hint is still zero, and
/// sets it to the bytes gathered plus the caller's remaining input. The first
/// `EncodeData` comes when a block fills or a flush or finish is requested,
/// so the hint is everything known at the first call that does either.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SizeHint {
    /// Declared, or already inferred.
    Settled,
    /// Unknown, and no input has been encoded yet.
    Open,
}

impl<'c, 'd> SessionCore<'c, 'd> {
    /// Starts after stream validation and workspace acquisition have succeeded.
    pub(crate) fn new(
        compressor: &'c mut Compressor,
        dictionary: Option<&'d PreparedDictionary>,
        limit: usize,
        stream: StreamConfig,
    ) -> Self {
        Self {
            compressor,
            dictionary,
            operation: OperationState::new(limit, stream),
        }
    }
    pub(crate) fn process(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: Operation,
    ) -> Result<Progress, EncodeError> {
        self.operation
            .process(self.compressor, self.dictionary, input, output, operation)
    }
    pub(crate) fn process_uninit(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        operation: Operation,
    ) -> Result<Progress, EncodeError> {
        self.operation.run(
            self.compressor,
            self.dictionary,
            input,
            Destination::Uninit(output),
            operation,
        )
    }
    pub(crate) const fn is_finished(&self) -> bool {
        self.operation.is_finished(self.compressor)
    }
}

impl<D: AsRef<PreparedDictionary>> OwnedSessionCore<D> {
    /// Starts after stream validation and workspace acquisition have succeeded.
    pub(crate) fn new(
        compressor: Compressor,
        dictionary: Option<D>,
        limit: usize,
        stream: StreamConfig,
    ) -> Self {
        Self {
            compressor,
            dictionary,
            operation: OperationState::new(limit, stream),
        }
    }
    pub(crate) fn process(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: Operation,
    ) -> Result<Progress, EncodeError> {
        self.operation.process(
            &mut self.compressor,
            self.dictionary.as_ref().map(AsRef::as_ref),
            input,
            output,
            operation,
        )
    }
    pub(crate) fn process_uninit(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        operation: Operation,
    ) -> Result<Progress, EncodeError> {
        self.operation.run(
            &mut self.compressor,
            self.dictionary.as_ref().map(AsRef::as_ref),
            input,
            Destination::Uninit(output),
            operation,
        )
    }
    pub(crate) const fn is_finished(&self) -> bool {
        self.operation.is_finished(&self.compressor)
    }
    /// Releases the current operation and starts a fresh one with the same
    /// dictionary, through the same path `Compressor::start` uses.
    ///
    /// A rejected start leaves the operation failed, so it encodes nothing.
    pub(crate) fn reinit(&mut self, stream: StreamConfig) -> Result<(), EncodeError> {
        self.operation.release(&mut self.compressor);
        let dictionary = self.dictionary.as_ref().map(AsRef::as_ref);
        match self.compressor.begin(dictionary, stream) {
            Ok(limit) => {
                self.operation = OperationState::new(limit, stream);
                Ok(())
            }
            Err(error) => {
                self.operation.poison();
                Err(error)
            }
        }
    }
    /// Ends the operation exactly as dropping a borrowed session does.
    pub(crate) fn into_compressor(self) -> Compressor {
        let Self {
            mut compressor,
            operation,
            ..
        } = self;
        operation.release(&mut compressor);
        compressor
    }
}

impl OperationState {
    pub(crate) fn new(limit: usize, stream: StreamConfig) -> Self {
        Self {
            state: StreamState::new(
                limit,
                cfg!(feature = "experimental") && stream.stream_offset() != 0,
            ),
            stream,
            hint: match stream.input_size() {
                InputSize::Unknown => SizeHint::Open,
                InputSize::Exact(_) => SizeHint::Settled,
            },
            #[cfg(feature = "experimental")]
            logical_position: stream.stream_offset(),
        }
    }

    /// Runs one call into an initialized slice.
    pub(crate) fn process(
        &mut self,
        compressor: &mut Compressor,
        dictionary: Option<&PreparedDictionary>,
        input: &[u8],
        output: &mut [u8],
        operation: Operation,
    ) -> Result<Progress, EncodeError> {
        self.run(
            compressor,
            dictionary,
            input,
            Destination::Slice(output),
            operation,
        )
    }

    /// Validates session state and logical positions, then runs the shared scheduler.
    fn run(
        &mut self,
        compressor: &mut Compressor,
        dictionary: Option<&PreparedDictionary>,
        input: &[u8],
        output: Destination<'_>,
        operation: Operation,
    ) -> Result<Progress, EncodeError> {
        if self.state.phase == Phase::Failed {
            return Err(EncodeError::InvalidState {
                attempted: "process a stream that has already failed",
            });
        }
        #[cfg(feature = "experimental")]
        if self.state.phase != Phase::Finished
            && self
                .logical_position
                .checked_add(input.len() as u64)
                .is_none_or(|end| end > (1u64 << 63) - 1)
        {
            return Err(EncodeError::StreamPositionOverflow {
                position: self.logical_position,
                input_bytes: input.len() as u64,
            });
        }

        if let Err(error) = self.infer_size_hint(compressor, input.len(), operation) {
            self.state.phase = Phase::Failed;
            return Err(error);
        }

        let Compressor {
            workspace,
            staging,
            pending,
            served,
            ..
        } = &mut *compressor;
        let Some(encoder) = workspace.encoder() else {
            self.state.phase = Phase::Failed;
            return Err(EncodeError::InternalInvariant {
                detail: "a session outlived the encoder it was started with",
            });
        };
        let outcome = self.state.process(
            encoder,
            dictionary.map(PreparedDictionary::inner),
            Buffers {
                staging,
                pending,
                served,
                allow_pending: true,
            },
            input,
            Output::new(output),
            operation,
        );
        let progress = match outcome {
            Ok(progress) => progress,
            Err(error) => return Err(EncodeError::from_core(error, 0)),
        };
        #[cfg(feature = "experimental")]
        {
            self.logical_position += progress.consumed as u64;
        }
        Ok(progress)
    }

    /// Settles an open size hint when this call is the one C would first
    /// encode in: a block fills, or a flush or finish is requested.
    ///
    /// The hint is the staged bytes plus this call's input, capped as C caps
    /// it. A call with nothing to encode leaves it open, as C's zero total does.
    /// Only a finish makes that the whole stream; after a flush or a full
    /// block the stream goes on, so storage stays sized for an unknown total,
    /// as it does for a finished stream above [`SIZED_STORAGE_LIMIT`].
    fn infer_size_hint(
        &mut self,
        compressor: &mut Compressor,
        input: usize,
        operation: Operation,
    ) -> Result<(), EncodeError> {
        if self.hint == SizeHint::Settled {
            return Ok(());
        }
        let known = compressor.staging.len().saturating_add(input);
        if known == 0 || (known < self.state.block_limit() && operation == Operation::Process) {
            return Ok(());
        }
        self.hint = SizeHint::Settled;
        let expected_input = if operation == Operation::Finish && known <= SIZED_STORAGE_LIMIT {
            known
        } else {
            0
        };
        compressor.resolve_size_hint(
            self.stream,
            known.min(MAX_INFERRED_SIZE_HINT),
            expected_input,
        )
    }

    /// Termination is observable only after all pending output was delivered.
    #[must_use]
    pub(crate) const fn is_finished(&self, compressor: &Compressor) -> bool {
        matches!(self.state.phase, Phase::Finished) && !compressor.has_pending()
    }
}

impl OperationState {
    /// Makes every later call report `InvalidState`.
    pub(crate) fn poison(&mut self) {
        self.state.phase = Phase::Failed;
    }
    pub(crate) fn release(&self, compressor: &mut Compressor) {
        if self.state.phase != Phase::Finished {
            compressor.workspace.invalidate();
        }
        compressor.staging.clear();
        compressor.pending.clear();
        compressor.served = 0;
        compressor.active = false;
        compressor.finish_operation();
    }
}

impl Drop for SessionCore<'_, '_> {
    fn drop(&mut self) {
        self.operation.release(self.compressor);
    }
}

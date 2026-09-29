//! Public-session ownership and error boundary over the shared block state machine.

use super::stream::{Buffers, Destination, Output, Phase, StreamState};
use crate::compressor::dictionary::PreparedDictionary;
use crate::compressor::encoder::Compressor;
use crate::compressor::error::EncodeError;
use crate::compressor::session::{Operation, Progress, StreamConfig};

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
    #[cfg(feature = "experimental")]
    logical_position: u64,
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
        #[cfg(not(feature = "experimental"))]
        let _ = stream;
        Self {
            state: StreamState::new(
                limit,
                cfg!(feature = "experimental") && stream.stream_offset() != 0,
            ),
            #[cfg(feature = "experimental")]
            logical_position: stream.stream_offset(),
        }
    }

    /// Validates session state and logical positions, then runs the shared scheduler.
    pub(crate) fn process(
        &mut self,
        compressor: &mut Compressor,
        dictionary: Option<&PreparedDictionary>,
        input: &[u8],
        output: &mut [u8],
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
            Output::new(Destination::Slice(output)),
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

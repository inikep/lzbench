//! Per-operation stream state shared by the borrowed and owned sessions.
//!
//! [`OperationState`] holds neither the decoder nor the dictionary: every call
//! receives both, so the same state machine runs over `&mut Decompressor` and
//! a borrowed `DictionaryRef` in `DecoderSession`, and over an owned
//! `Decompressor` and dictionary in `DecoderSessionOwned`.

use super::{Input, Output, Stop};
use crate::Window;
use crate::decompressor::session::{DecodeFailure, DecodeOperation, DecodeProgress, DecoderStatus};
use crate::decompressor::{DecodeError, DecodeStreamConfig, Decompressor, MemberMode, OutputSize};
use crate::dictionary::DictionaryRef;

/// Progress and lifecycle flags of one decoding operation.
#[derive(Debug)]
pub(crate) struct OperationState {
    stream: DecodeStreamConfig,
    total_in: u64,
    total_out: u64,
    members: u64,
    window: Option<Window>,
    final_end: u64,
    finishing: bool,
    boundary: bool,
    finished: bool,
    failed: bool,
}

/// How one call delivers decoded bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Delivery {
    /// Copy history into the caller's slice as it is decoded.
    Slice,
    /// Decode into the ring up to this capacity; the owned Vec takes it.
    Collect(usize),
    /// Decode the first member straight into the caller's slice.
    Linear,
}

impl OperationState {
    /// Validates the stream against `decoder` and claims it for one operation.
    pub(crate) fn start(
        decoder: &mut Decompressor,
        stream: DecodeStreamConfig,
    ) -> Result<Self, DecodeError> {
        if decoder.active {
            return Err(DecodeError::AbandonedSession);
        }
        if let (OutputSize::Exact(expected), Some(limit)) = (
            stream.output_size(),
            decoder.config.limits().max_output_bytes(),
        ) && expected > limit
        {
            return Err(DecodeError::OutputLimitExceeded { limit });
        }
        if decoder
            .config
            .limits()
            .max_workspace_bytes()
            .is_some_and(|limit| decoder.retained_bytes() > limit)
        {
            decoder.recover();
        }
        decoder.workspace.reset(decoder.config);
        decoder.active = true;
        Ok(Self {
            stream,
            total_in: 0,
            total_out: 0,
            members: 0,
            window: None,
            final_end: 0,
            finishing: false,
            boundary: false,
            finished: false,
            failed: false,
        })
    }

    /// Runs one call of the incremental contract against `decoder`.
    ///
    /// `dictionary` must be the one the operation started with on every call.
    /// `delivery` lets the one-shot APIs decode into history: the owned Vec
    /// collects until the first wrap, and a slice holds the first member of
    /// a fresh operation. Public streaming always passes `Delivery::Slice`.
    pub(crate) fn process(
        &mut self,
        decoder: &mut Decompressor,
        dictionary: Option<DictionaryRef<'_>>,
        input: &[u8],
        output: &mut [u8],
        operation: DecodeOperation,
        delivery: Delivery,
    ) -> Result<DecodeProgress, DecodeFailure> {
        let invalid = |error| DecodeFailure {
            error,
            consumed: 0,
            produced: 0,
        };
        if self.failed {
            return Err(invalid(DecodeError::InvalidState));
        }
        if self.finished {
            return Ok(DecodeProgress {
                consumed: 0,
                produced: 0,
                status: DecoderStatus::Finished,
            });
        }
        let Some(end) = self.total_in.checked_add(input.len() as u64) else {
            self.failed = true;
            return Err(invalid(DecodeError::SizeOverflow));
        };
        if self.finishing {
            if operation != DecodeOperation::Finish || end != self.final_end {
                self.failed = true;
                return Err(invalid(DecodeError::InvalidState));
            }
        } else if operation == DecodeOperation::Finish {
            self.final_end = end;
            self.finishing = true;
        }
        let config = decoder.config;
        let limits = config.limits();
        let mut input = Input::new(input, self.total_in, limits.max_input_bytes());
        let mut output = Output {
            collect: match delivery {
                Delivery::Collect(capacity) => Some(capacity),
                Delivery::Slice | Delivery::Linear => None,
            },
            // Only a call that starts the operation begins its first member
            // at `output[0]`; the member boundary below turns it off.
            linear: delivery == Delivery::Linear && self.total_in == 0 && self.total_out == 0,
            bytes: output,
            produced: 0,
            total_before: self.total_out,
            limit: limits.max_output_bytes(),
            exact: self.stream.output_size(),
        };
        let result = (|| loop {
            if self.boundary {
                if config.member_mode() == MemberMode::Single
                    || (self.finishing && input.consumed == input.bytes.len())
                {
                    if let OutputSize::Exact(expected) = self.stream.output_size() {
                        let actual = self.total_out + output.produced as u64;
                        if actual != expected {
                            return Err(DecodeError::OutputSizeMismatch { expected, actual });
                        }
                    }
                    self.finished = true;
                    return Ok(DecoderStatus::Finished);
                }
                if input.consumed == input.bytes.len() {
                    return Ok(DecoderStatus::NeedsInput);
                }
                decoder.workspace.reset(config);
                self.boundary = false;
            }
            let outcome =
                decoder
                    .workspace
                    .run(decoder.backend, &mut input, &mut output, config, dictionary);
            if let Some(window) = decoder.workspace.window {
                self.window = Some(window);
            }
            match outcome? {
                Stop::Input if self.finishing => {
                    return Err(DecodeError::UnexpectedEndOfInput);
                }
                Stop::Input => return Ok(DecoderStatus::NeedsInput),
                Stop::Output => return Ok(DecoderStatus::NeedsOutput),
                Stop::Member => {
                    output.linear = false;
                    self.members = self
                        .members
                        .checked_add(1)
                        .ok_or(DecodeError::SizeOverflow)?;
                    self.boundary = true;
                }
            }
        })();
        self.total_in += input.consumed as u64;
        self.total_out += output.produced as u64;
        match result {
            Ok(status) => Ok(DecodeProgress {
                consumed: input.consumed,
                produced: output.produced,
                status,
            }),
            Err(error) => {
                self.failed = true;
                Err(DecodeFailure {
                    error,
                    consumed: input.consumed,
                    produced: output.produced,
                })
            }
        }
    }

    /// Makes every later call report `InvalidState`.
    pub(crate) const fn poison(&mut self) {
        self.failed = true;
    }

    /// Ends the operation, leaving `decoder` ready for the next one.
    ///
    /// The single release path: borrowed sessions run it on drop, owned
    /// sessions when they hand the decoder back.
    pub(crate) fn release(&self, decoder: &mut Decompressor) {
        decoder.active = false;
        decoder.workspace.reset(decoder.config);
        decoder.trim(decoder.retention());
    }

    /// Whether all members and the exact-size contract were validated.
    pub(crate) const fn is_finished(&self) -> bool {
        self.finished
    }
    /// Total accepted compressed bytes, including failing calls.
    pub(crate) const fn total_in(&self) -> u64 {
        self.total_in
    }
    /// Total payload bytes delivered, including failing calls.
    pub(crate) const fn total_out(&self) -> u64 {
        self.total_out
    }
    /// Number of validated members whose output has been delivered.
    pub(crate) const fn members_decoded(&self) -> u64 {
        self.members
    }
    /// Most recently accepted window header.
    pub(crate) const fn window(&self) -> Option<Window> {
        self.window
    }
}

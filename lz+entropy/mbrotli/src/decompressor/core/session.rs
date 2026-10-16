//! Per-operation stream state shared by the borrowed and owned sessions.
//!
//! [`OperationState`] holds neither the decoder nor the dictionary: every call
//! receives both, so the same state machine runs over `&mut Decompressor` and
//! a borrowed `DictionaryRef` in `DecoderSession`, and over an owned
//! `Decompressor` and dictionary in `DecoderSessionOwned`.

use core::mem::MaybeUninit;

use super::{Input, Output, Sink, Stop};
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
    /// Decode the first member straight into the caller's slice, which also
    /// serves as its history. The slice starts with everything the operation
    /// produced so far, so a call continues at `total_out`: every call of an
    /// operation must pass `Linear` with the same bytes in front, which lets
    /// a caller grow the slice between calls without losing history.
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
        self.run(decoder, dictionary, input, output, operation, delivery)
    }

    /// Runs one call of the streaming contract into uninitialized memory.
    ///
    /// Always delivers as [`Delivery::Slice`]: `output` cannot serve as
    /// history, so every byte is copied out of the ring.
    pub(crate) fn process_uninit(
        &mut self,
        decoder: &mut Decompressor,
        dictionary: Option<DictionaryRef<'_>>,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        operation: DecodeOperation,
    ) -> Result<DecodeProgress, DecodeFailure> {
        self.run(
            decoder,
            dictionary,
            input,
            output,
            operation,
            Delivery::Slice,
        )
    }

    fn run<'a, O: Sink<'a>>(
        &mut self,
        decoder: &mut Decompressor,
        dictionary: Option<DictionaryRef<'_>>,
        input: &[u8],
        output: O,
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
        // A linear slice holds the operation's earlier output in front of
        // the space this call may write.
        let start = match delivery {
            Delivery::Linear => usize::try_from(self.total_out)
                .ok()
                .filter(|&start| start <= output.len()),
            Delivery::Slice | Delivery::Collect(_) => Some(0),
        };
        let Some(start) = start else {
            self.failed = true;
            return Err(invalid(DecodeError::InternalInvariant));
        };
        let config = decoder.config;
        let limits = config.limits();
        let mut input = Input::new(input, self.total_in, limits.max_input_bytes());
        let mut output = Output {
            collect: match delivery {
                Delivery::Collect(capacity) => Some(capacity),
                Delivery::Slice | Delivery::Linear => None,
            },
            // The first member begins at `output[0]` and stays linear across
            // calls until the member boundary below turns it off.
            linear: delivery == Delivery::Linear && self.members == 0 && !self.boundary,
            bytes: output,
            produced: start,
            total_before: self.total_out - start as u64,
            limit: limits.max_output_bytes(),
            exact: self.stream.output_size(),
        };
        let result = (|| loop {
            if self.boundary {
                if config.member_mode() == MemberMode::Single
                    || (self.finishing && input.consumed == input.bytes.len())
                {
                    if let OutputSize::Exact(expected) = self.stream.output_size() {
                        let actual = output.total_before + output.produced as u64;
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
        let produced = output.produced - start;
        self.total_in += input.consumed as u64;
        self.total_out += produced as u64;
        match result {
            Ok(status) => Ok(DecodeProgress {
                consumed: input.consumed,
                produced,
                status,
            }),
            Err(error) => {
                self.failed = true;
                Err(DecodeFailure {
                    error,
                    consumed: input.consumed,
                    produced,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompressor::DecoderConfig;

    /// `hello` as one stored meta-block.
    const HELLO: [u8; 9] = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];

    fn started(decoder: &mut Decompressor) -> OperationState {
        OperationState::start(decoder, DecodeStreamConfig::default()).expect("starts")
    }

    #[test]
    fn a_linear_call_resumes_after_the_bytes_already_produced() {
        let mut decoder = Decompressor::new(DecoderConfig::default()).expect("a decoder");
        let mut operation = started(&mut decoder);
        let mut output = [0; 5];
        let first = operation
            .process(
                &mut decoder,
                None,
                &HELLO,
                &mut output[..2],
                DecodeOperation::Finish,
                Delivery::Linear,
            )
            .expect("decodes");
        assert_eq!(first.produced, 2);
        assert_eq!(first.status, DecoderStatus::NeedsOutput);

        let second = operation
            .process(
                &mut decoder,
                None,
                &HELLO[first.consumed..],
                &mut output,
                DecodeOperation::Finish,
                Delivery::Linear,
            )
            .expect("decodes");
        assert_eq!(second.produced, 3);
        assert_eq!(second.status, DecoderStatus::Finished);
        assert_eq!(&output, b"hello");
        assert_eq!(operation.total_out(), 5);
    }

    #[test]
    fn a_linear_slice_shorter_than_the_output_so_far_fails_the_operation() {
        let mut decoder = Decompressor::new(DecoderConfig::default()).expect("a decoder");
        let mut operation = started(&mut decoder);
        let mut output = [0; 5];
        let first = operation
            .process(
                &mut decoder,
                None,
                &HELLO,
                &mut output[..2],
                DecodeOperation::Finish,
                Delivery::Linear,
            )
            .expect("decodes");

        let failure = operation
            .process(
                &mut decoder,
                None,
                &HELLO[first.consumed..],
                &mut output[..1],
                DecodeOperation::Finish,
                Delivery::Linear,
            )
            .expect_err("the slice lost history");
        assert!(matches!(failure.error, DecodeError::InternalInvariant));
        assert_eq!((failure.consumed, failure.produced), (0, 0));
        let again = operation.process(
            &mut decoder,
            None,
            &HELLO[first.consumed..],
            &mut output,
            DecodeOperation::Finish,
            Delivery::Linear,
        );
        assert!(matches!(
            again,
            Err(DecodeFailure {
                error: DecodeError::InvalidState,
                ..
            })
        ));
        operation.release(&mut decoder);
    }
}

//! Shared block scheduling for one-shot and incremental compression.

use alloc::vec::Vec;
use core::mem::MaybeUninit;

use super::driver::Encoder;
use super::fast::FastEncoder;
use super::rfc9841::context::SharedContextInner;
use crate::compressor::session::{EncoderStatus, Operation, Progress};
use crate::compressor::{BrotliCompressError, BrotliResult};
use crate::shared::uninit::copy_to_uninit;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) enum Phase {
    Open,
    Flushed,
    Finished,
    Failed,
}

/// State without allocation ownership; buffers belong to the compressor.
#[derive(Debug)]
pub(super) struct StreamState {
    pub(super) phase: Phase,
    limit: usize,
    flint: bool,
}

/// The caller's destination. One-shot vectors append without initializing spare capacity.
pub(super) enum Destination<'a> {
    Slice(&'a mut [u8]),
    /// Write-only: safe code cannot read these bytes back, so the fast
    /// encoders' in-place bit writer never runs here and every block is
    /// copied from owned encoded storage instead.
    Uninit(&'a mut [MaybeUninit<u8>]),
    Append(&'a mut Vec<u8>),
}

/// Output cursor for one call, with no ownership of the caller's bytes.
pub(super) struct Output<'a> {
    destination: Destination<'a>,
    produced: usize,
}

impl<'a> Output<'a> {
    pub(super) const fn new(destination: Destination<'a>) -> Self {
        Self {
            destination,
            produced: 0,
        }
    }

    /// Copies only what fits; an append destination always accepts the whole part.
    pub(super) fn append(&mut self, bytes: &[u8]) -> usize {
        let count = match &mut self.destination {
            Destination::Slice(output) => {
                let count = bytes.len().min(output.len() - self.produced);
                output[self.produced..self.produced + count].copy_from_slice(&bytes[..count]);
                count
            }
            Destination::Uninit(output) => {
                let count = bytes.len().min(output.len() - self.produced);
                copy_to_uninit(
                    &mut output[self.produced..self.produced + count],
                    &bytes[..count],
                );
                count
            }
            Destination::Append(output) => {
                output.extend_from_slice(bytes);
                bytes.len()
            }
        };
        self.produced += count;
        count
    }
}

/// Borrowed retained storage; one-shot calls leave both vectors unallocated.
pub(super) struct Buffers<'a> {
    pub(super) staging: &'a mut Vec<u8>,
    pub(super) pending: &'a mut Vec<u8>,
    /// Completed prefix: `served <= pending_end <= pending.len()`.
    pub(super) pending_end: &'a mut usize,
    pub(super) served: &'a mut usize,
    /// A session keeps overflow for the next call; a one-shot slice reports it.
    pub(super) allow_pending: bool,
}

impl Buffers<'_> {
    fn drain(&mut self, output: &mut Output<'_>) -> bool {
        *self.served += output.append(&self.pending[*self.served..*self.pending_end]);
        if *self.served < *self.pending_end {
            return false;
        }
        *self.pending_end = 0;
        *self.served = 0;
        true
    }
}

/// Encodes and delivers a block without borrowing staging mutably at the same time.
struct Delivery<'a, 'o> {
    output: &'a mut Output<'o>,
    pending: &'a mut Vec<u8>,
    pending_end: &'a mut usize,
    served: &'a mut usize,
    allow_pending: bool,
}

impl Delivery<'_, '_> {
    fn encode(
        &mut self,
        encoder: &mut Encoder,
        attached: Option<&SharedContextInner>,
        input: &[u8],
        operation: Operation,
    ) -> BrotliResult<()> {
        if operation != Operation::Flush
            && let Encoder::Fast(fast) = encoder
            && let Destination::Slice(output) = &mut self.output.destination
        {
            let tail = &mut output[self.output.produced..];
            if tail.len() >= FastEncoder::fragment_reserve(input.len())? {
                self.output.produced +=
                    fast.encode_block_into(input, operation == Operation::Finish, tail)?;
                return Ok(());
            }
        }
        if operation != Operation::Flush
            && let Encoder::Fast(fast) = encoder
            && let Destination::Append(output) = &mut self.output.destination
        {
            self.output.produced +=
                fast.encode_block_append(input, operation == Operation::Finish, output)?;
            return Ok(());
        }
        if operation != Operation::Flush
            && let Encoder::Greedy(greedy) = encoder
            && let Destination::Append(output) = &mut self.output.destination
        {
            self.output.produced += greedy.encode_block_append(
                input,
                operation == Operation::Finish,
                attached,
                output,
            )?;
            return Ok(());
        }
        // A short caller buffer cannot use the fixed-slice encoder. Encode
        // straight into retained pending storage instead of first writing a
        // scratch buffer and then copying its unserved tail.
        if operation != Operation::Flush
            && self.allow_pending
            && let Encoder::Fast(fast) = encoder
        {
            // Keep the initialized backing length across drains. Growth only
            // initializes a new tail; completed bytes have a separate boundary.
            let reserve = FastEncoder::fragment_reserve(input.len())?;
            crate::shared::fixed::grow_zeroed(self.pending, reserve);
            let complete = fast.encode_block_into(
                input,
                operation == Operation::Finish,
                self.pending.as_mut_slice(),
            )?;
            *self.pending_end = complete;
            *self.served = self.output.append(&self.pending[..complete]);
            if *self.served == complete {
                *self.pending_end = 0;
                *self.served = 0;
            }
            return Ok(());
        }
        let bytes = match operation {
            Operation::Flush => encoder.flush_block(input, attached)?,
            _ => encoder.encode_block_with(input, operation == Operation::Finish, attached)?,
        };
        let direct = self.output.append(bytes);
        if direct < bytes.len() {
            if !self.allow_pending {
                return Err(BrotliCompressError::OutputTooSmall);
            }
            let suffix = &bytes[direct..];
            let initialized = self.pending.len().min(suffix.len());
            self.pending[..initialized].copy_from_slice(&suffix[..initialized]);
            self.pending.extend_from_slice(&suffix[initialized..]);
            *self.pending_end = suffix.len();
            *self.served = 0;
        }
        Ok(())
    }
}

impl StreamState {
    pub(super) const fn new(limit: usize, flint: bool) -> Self {
        Self {
            phase: Phase::Open,
            limit,
            flint,
        }
    }

    /// Returns the input one block holds: the most a stream gathers before
    /// it has to encode, which is when C's `UpdateSizeHint` runs.
    pub(super) const fn block_limit(&self) -> usize {
        if self.flint { 2 } else { self.limit }
    }

    /// Drains output first, then borrows complete blocks or stages an undecided tail.
    /// A full final block is held on Process until a following byte or Finish is known.
    pub(super) fn process(
        &mut self,
        encoder: &mut Encoder,
        attached: Option<&SharedContextInner>,
        mut buffers: Buffers<'_>,
        input: &[u8],
        mut output: Output<'_>,
        operation: Operation,
    ) -> BrotliResult<Progress> {
        let mut consumed = 0;
        let status = loop {
            if !buffers.drain(&mut output) {
                break EncoderStatus::NeedsOutput;
            }
            if self.phase == Phase::Finished {
                break EncoderStatus::Finished;
            }
            if self.phase == Phase::Flushed
                && consumed == input.len()
                && operation == Operation::Flush
            {
                break EncoderStatus::NeedsInput;
            }
            let limit = self.block_limit();
            let take = (limit - buffers.staging.len()).min(input.len() - consumed);
            let at_end = consumed + take == input.len();
            let restart = self.flint
                && buffers.staging.len() + take == 2
                && (!at_end || operation != Operation::Finish);
            if at_end && operation == Operation::Process && !restart {
                buffers
                    .staging
                    .extend_from_slice(&input[consumed..consumed + take]);
                consumed += take;
                if take != 0 {
                    self.phase = Phase::Open;
                }
                break EncoderStatus::NeedsInput;
            }
            let block_operation = if restart {
                Operation::Flush
            } else if at_end {
                operation
            } else {
                Operation::Process
            };
            let block = if buffers.staging.is_empty() {
                // The whole input is borrowed on one-shot calls. Streaming also
                // borrows complete blocks rather than copying them through staging.
                &input[consumed..consumed + take]
            } else {
                buffers
                    .staging
                    .extend_from_slice(&input[consumed..consumed + take]);
                buffers.staging.as_slice()
            };
            let outcome = Delivery {
                output: &mut output,
                pending: buffers.pending,
                pending_end: buffers.pending_end,
                served: buffers.served,
                allow_pending: buffers.allow_pending,
            }
            .encode(encoder, attached, block, block_operation);
            if let Err(error) = outcome {
                self.phase = Phase::Failed;
                return Err(error);
            }
            consumed += take;
            buffers.staging.clear();
            self.phase = match block_operation {
                Operation::Finish => Phase::Finished,
                Operation::Flush if !restart => Phase::Flushed,
                _ => Phase::Open,
            };
            if restart {
                self.flint = false;
            }
        };
        Ok(Progress {
            consumed,
            produced: output.produced,
            status,
        })
    }
}

/// Finish through the same scheduler without allocating session staging or pending storage.
pub(super) fn finish(
    encoder: &mut Encoder,
    attached: Option<&SharedContextInner>,
    input: &[u8],
    destination: Destination<'_>,
) -> BrotliResult<usize> {
    let mut state = StreamState::new(encoder.block_size_limit(), false);
    let mut staging = Vec::new();
    let mut pending = Vec::new();
    let mut pending_end = 0;
    let mut served = 0;
    state
        .process(
            encoder,
            attached,
            Buffers {
                staging: &mut staging,
                pending: &mut pending,
                pending_end: &mut pending_end,
                served: &mut served,
                allow_pending: false,
            },
            input,
            Output::new(destination),
            Operation::Finish,
        )
        .map(|progress| progress.produced)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_fragments_drain_pending_bytes_once_across_output_boundaries() {
        use crate::compressor::{Backend, CompressParams, QualityLevel, WindowBits};

        fn initialized(bytes: &[MaybeUninit<u8>]) -> Vec<u8> {
            // SAFETY: callers pass exactly the prefix Output reported writing.
            // Buffers start uninitialized, so Miri detects an incorrect report.
            bytes
                .iter()
                .map(|byte| unsafe { byte.assume_init() })
                .collect()
        }

        let len = if cfg!(miri) { 64 } else { 1024 };
        let input: Vec<u8> = (0..len).map(|i| ((i * 37 + i / 11) % 256) as u8).collect();
        for backend in Backend::available() {
            for quality in [QualityLevel::Q0, QualityLevel::Q1] {
                for uninit in [false, true] {
                    for first_output in [0, 1, 7, 2048] {
                        for (tail_case, tail) in [
                            None,
                            Some(b"".as_slice()),
                            Some(b"a trailing fragment".as_slice()),
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            // Miri samples all transitions while keeping every
                            // quality/backend; native tests cross all cases.
                            if cfg!(miri)
                                && !matches!(
                                    (uninit, first_output, tail_case),
                                    (false, 0, 0) | (true, 1, 1) | (true, 2048, 2)
                                )
                            {
                                continue;
                            }
                            let first_output = if cfg!(miri) && first_output == 2048 {
                                128
                            } else {
                                first_output
                            };
                            let params = CompressParams::new(
                                quality,
                                WindowBits::standard(10).expect("window"),
                            );
                            let mut reference = FastEncoder::<false>::new(backend, &params)
                                .expect("reference encoder");
                            let mut encoder =
                                Encoder::Fast(FastEncoder::new(backend, &params).expect("encoder"));
                            let mut pending = Vec::new();
                            let mut pending_end = 0;
                            let mut initialized_len = 0;
                            let mut served = 0;
                            let mut actual = Vec::new();
                            let mut expected = Vec::new();
                            for (data, operation) in [
                                (input.as_slice(), Operation::Process),
                                (tail.unwrap_or_default(), Operation::Flush),
                                (b"final fragment".as_slice(), Operation::Finish),
                            ] {
                                if operation == Operation::Flush && tail.is_none() {
                                    continue;
                                }
                                let encoded = match operation {
                                    Operation::Flush => reference.flush_block(data),
                                    _ => {
                                        reference.encode_block(data, operation == Operation::Finish)
                                    }
                                }
                                .expect("reference block");
                                expected.extend_from_slice(encoded);
                                let mut bytes = vec![0; first_output];
                                let mut uninitialized = vec![MaybeUninit::uninit(); first_output];
                                let produced = {
                                    let destination = if uninit {
                                        Destination::Uninit(&mut uninitialized)
                                    } else {
                                        Destination::Slice(&mut bytes)
                                    };
                                    let mut output = Output::new(destination);
                                    Delivery {
                                        output: &mut output,
                                        pending: &mut pending,
                                        pending_end: &mut pending_end,
                                        served: &mut served,
                                        allow_pending: true,
                                    }
                                    .encode(&mut encoder, None, data, operation)
                                    .expect("block");
                                    output.produced
                                };
                                assert!(pending.len() >= initialized_len);
                                initialized_len = pending.len();
                                if cfg!(miri) && first_output == 128 {
                                    assert_eq!(pending_end, 0, "complete output must fit");
                                }
                                if uninit {
                                    actual.extend_from_slice(&initialized(
                                        &uninitialized[..produced],
                                    ));
                                } else {
                                    actual.extend_from_slice(&bytes[..produced]);
                                }
                                let mut staging = Vec::new();
                                let mut buffers = Buffers {
                                    staging: &mut staging,
                                    pending: &mut pending,
                                    pending_end: &mut pending_end,
                                    served: &mut served,
                                    allow_pending: true,
                                };
                                if *buffers.pending_end != 0 {
                                    let old_served = *buffers.served;
                                    let destination = if uninit {
                                        Destination::Uninit(&mut [])
                                    } else {
                                        Destination::Slice(&mut [])
                                    };
                                    assert!(!buffers.drain(&mut Output::new(destination)));
                                    assert_eq!(*buffers.served, old_served);
                                }
                                while *buffers.pending_end != 0 {
                                    let mut chunk = [0; 17];
                                    let mut uninitialized = [MaybeUninit::uninit(); 17];
                                    let destination = if uninit {
                                        Destination::Uninit(&mut uninitialized)
                                    } else {
                                        Destination::Slice(&mut chunk)
                                    };
                                    let mut output = Output::new(destination);
                                    buffers.drain(&mut output);
                                    let produced = output.produced;
                                    assert!(produced > 0);
                                    if uninit {
                                        actual.extend_from_slice(&initialized(
                                            &uninitialized[..produced],
                                        ));
                                    } else {
                                        actual.extend_from_slice(&chunk[..produced]);
                                    }
                                }
                                assert_eq!(*buffers.served, 0);
                                assert_eq!(*buffers.pending_end, 0);
                                assert_eq!(buffers.pending.len(), initialized_len);
                                assert_eq!(
                                    actual, expected,
                                    "{backend:?}, {quality:?}, {first_output}, uninit {uninit}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn finished_phase_drains_only_the_completed_prefix_of_retained_storage() {
        use crate::compressor::{Backend, CompressParams, QualityLevel, WindowBits};

        let params =
            CompressParams::new(QualityLevel::Q1, WindowBits::standard(10).expect("window"));
        let mut reference = FastEncoder::<false>::new(Backend::SCALAR, &params).expect("reference");
        let expected = reference
            .encode_block(b"tail", true)
            .expect("reference block");
        let mut encoder =
            Encoder::Fast(FastEncoder::new(Backend::SCALAR, &params).expect("encoder"));
        let mut state = StreamState::new(encoder.block_size_limit(), false);
        let mut staging = Vec::new();
        let mut pending = vec![0xa5; 2048];
        let mut pending_end = 0;
        let mut served = 0;
        let progress = state
            .process(
                &mut encoder,
                None,
                Buffers {
                    staging: &mut staging,
                    pending: &mut pending,
                    pending_end: &mut pending_end,
                    served: &mut served,
                    allow_pending: true,
                },
                b"tail",
                Output::new(Destination::Slice(&mut [])),
                Operation::Finish,
            )
            .expect("encode final block");
        assert_eq!(progress.status, EncoderStatus::NeedsOutput);
        assert_eq!(progress.consumed, 4);
        assert_eq!(progress.produced, 0);
        assert_eq!(state.phase, Phase::Finished);
        assert_eq!(pending_end, expected.len());
        let mut bytes = [0; 128];
        let progress = state
            .process(
                &mut encoder,
                None,
                Buffers {
                    staging: &mut staging,
                    pending: &mut pending,
                    pending_end: &mut pending_end,
                    served: &mut served,
                    allow_pending: true,
                },
                b"ignored after final block",
                Output::new(Destination::Slice(&mut bytes)),
                Operation::Finish,
            )
            .expect("drain final block");
        assert_eq!(progress.status, EncoderStatus::Finished);
        assert_eq!(progress.consumed, 0);
        assert_eq!(&bytes[..progress.produced], expected);
        assert_eq!(pending.len(), 2048);
        assert_eq!(pending_end, 0);
        assert_eq!(served, 0);
    }

    #[test]
    fn pending_output_is_delivered_once_and_empty_destinations_preserve_it() {
        let mut staging = Vec::new();
        let mut pending = b"0123456789stale padding".to_vec();
        let initialized_len = pending.len();
        let mut pending_end = 10;
        let mut served = 0;
        let mut buffers = Buffers {
            staging: &mut staging,
            pending: &mut pending,
            pending_end: &mut pending_end,
            served: &mut served,
            allow_pending: true,
        };
        assert!(!buffers.drain(&mut Output::new(Destination::Slice(&mut []))));
        assert_eq!(*buffers.served, 0);
        for (expected, drained) in [(b"0123".as_slice(), false), (b"4567", false), (b"89", true)] {
            let mut bytes = [0; 4];
            let mut output = Output::new(Destination::Slice(&mut bytes));
            assert_eq!(buffers.drain(&mut output), drained);
            assert_eq!(output.produced, expected.len());
            assert_eq!(&bytes[..expected.len()], expected);
        }
        assert_eq!(buffers.pending.len(), initialized_len);
        assert_eq!(*buffers.pending_end, 0);
        assert_eq!(*buffers.served, 0);
        assert!(buffers.drain(&mut Output::new(Destination::Slice(&mut []))));
    }
}

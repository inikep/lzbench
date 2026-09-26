//! Shared block scheduling for one-shot and incremental compression.

use alloc::vec::Vec;

use super::driver::Encoder;
use super::fast::FastEncoder;
use super::rfc9841::context::SharedContextInner;
use crate::compressor::session::{EncoderStatus, Operation, Progress};
use crate::compressor::{BrotliCompressError, BrotliResult};

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
    fn append(&mut self, bytes: &[u8]) -> usize {
        let count = match &mut self.destination {
            Destination::Slice(output) => {
                let count = bytes.len().min(output.len() - self.produced);
                output[self.produced..self.produced + count].copy_from_slice(&bytes[..count]);
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
    pub(super) served: &'a mut usize,
    /// A session keeps overflow for the next call; a one-shot slice reports it.
    pub(super) allow_pending: bool,
}

impl Buffers<'_> {
    fn drain(&mut self, output: &mut Output<'_>) -> bool {
        *self.served += output.append(&self.pending[*self.served..]);
        if *self.served < self.pending.len() {
            return false;
        }
        self.pending.clear();
        *self.served = 0;
        true
    }
}

/// Encodes and delivers a block without borrowing staging mutably at the same time.
struct Delivery<'a, 'o> {
    output: &'a mut Output<'o>,
    pending: &'a mut Vec<u8>,
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
        let bytes = match operation {
            Operation::Flush => encoder.flush_block(input, attached)?,
            _ => encoder.encode_block_with(input, operation == Operation::Finish, attached)?,
        };
        let direct = self.output.append(bytes);
        if direct < bytes.len() {
            if !self.allow_pending {
                return Err(BrotliCompressError::OutputTooSmall);
            }
            self.pending.clear();
            self.pending.extend_from_slice(&bytes[direct..]);
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
            let limit = if self.flint { 2 } else { self.limit };
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
    let mut served = 0;
    state
        .process(
            encoder,
            attached,
            Buffers {
                staging: &mut staging,
                pending: &mut pending,
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
    fn pending_output_is_delivered_once_and_empty_destinations_preserve_it() {
        let mut staging = Vec::new();
        let mut pending = b"0123456789".to_vec();
        let mut served = 0;
        let mut buffers = Buffers {
            staging: &mut staging,
            pending: &mut pending,
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
        assert!(buffers.pending.is_empty());
        assert_eq!(*buffers.served, 0);
        assert!(buffers.drain(&mut Output::new(Destination::Slice(&mut []))));
    }
}

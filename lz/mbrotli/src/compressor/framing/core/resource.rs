//! Non-borrowing resource state; dictionaries are supplied only while driving it.
use super::{Container, bytes};
use crate::compressor::core::session::OperationState;
use crate::compressor::framing::{
    FramedEncodeError as E, FramedEncoderStatus as Status, ResourceEncoding, ResourceOptions,
};
use crate::dictionary::PreparedDictionary;
use crate::{Compressor, EncoderStatus, InputSize, Operation, StreamConfig};
use alloc::vec::Vec;

#[derive(Debug, Default)]
pub(in crate::compressor::framing) struct Resource {
    raw: Option<OperationState>,
    options: ResourceOptions,
    references: Vec<u8>,
    input: Vec<u8>,
    content: Vec<u8>,
    first: bool,
    queued_final: bool,
    pub(super) finished: bool,
    pub(super) total_in: u64,
    pub(super) total_out: u64,
    expected: InputSize,
    boundary: Option<(Operation, u64)>,
}
impl Resource {
    pub(super) fn retained_bytes(&self) -> usize {
        self.references.capacity() + self.input.capacity() + self.content.capacity()
    }
    pub(super) fn clear(&mut self, compressor: &mut Compressor) {
        if let Some(raw) = self.raw.take() {
            raw.release(compressor);
        }
        self.input.clear();
        self.content.clear();
        self.references.clear();
        self.boundary = None;
    }
    pub(super) fn begin(
        &mut self,
        core: &mut Container,
        compressor: &mut Compressor,
        options: ResourceOptions,
        stream: StreamConfig,
        encoding: ResourceEncoding<'_>,
        references: Vec<u8>,
    ) -> Result<(), E> {
        if stream.stream_offset() != 0 {
            return Err(E::Invalid(
                "a new resource requires a stream header; its offset must be zero",
            ));
        }
        core.begin()?;
        if self.input.capacity() < core.config.chunk_bytes {
            self.input
                .try_reserve_exact(core.config.chunk_bytes - self.input.len())
                .map_err(|_| E::AllocationFailed)?;
        }
        let dictionary = match encoding {
            ResourceEncoding::Shared { dictionary, .. } => Some(dictionary),
            _ => None,
        };
        let raw = if !matches!(encoding, ResourceEncoding::Uncompressed) {
            let limit = compressor.begin(dictionary, stream)?;
            Some(OperationState::new(limit, stream))
        } else {
            None
        };
        self.raw = raw;
        self.options = options;
        self.references = references;
        self.input.clear();
        self.content.clear();
        self.first = true;
        self.queued_final = false;
        self.finished = false;
        self.total_in = 0;
        self.total_out = 0;
        self.expected = stream.input_size();
        self.boundary = None;
        core.active = true;
        core.after_resource = false;
        core.metadata_pending = false;
        Ok(())
    }
    fn emit(
        &mut self,
        core: &mut Container,
        compressor: &mut Compressor,
        dictionary: Option<&PreparedDictionary>,
        last: bool,
    ) -> Result<(), E> {
        core.drain()?;
        core.check_buffer(
            core.config
                .chunk_bytes
                .saturating_mul(4)
                .saturating_add(8192),
        )?;
        if core.chunks >= core.config.max_chunks {
            return Err(E::Limit {
                kind: "chunk count",
                limit: core.config.max_chunks,
            });
        }
        self.content.clear();
        let capacity = self
            .input
            .len()
            .checked_mul(2)
            .and_then(|v| v.checked_add(1024))
            .ok_or(E::Overflow)?;
        if self.content.capacity() < capacity {
            self.content
                .try_reserve_exact(capacity)
                .map_err(|_| E::AllocationFailed)?;
        }
        if let Some(raw) = &mut self.raw {
            let mut remaining = self.input.as_slice();
            let mut output = [0; 8192];
            loop {
                let p = raw.process(
                    compressor,
                    dictionary,
                    remaining,
                    &mut output,
                    if last {
                        Operation::Finish
                    } else {
                        Operation::Flush
                    },
                )?;
                remaining = &remaining[p.consumed..];
                core.check_buffer(
                    self.content
                        .capacity()
                        .saturating_add(p.produced)
                        .saturating_add(self.input.capacity()),
                )?;
                self.content
                    .try_reserve_exact(p.produced)
                    .map_err(|_| E::AllocationFailed)?;
                self.content.extend_from_slice(&output[..p.produced]);
                if matches!(
                    p.status,
                    EncoderStatus::Finished | EncoderStatus::NeedsInput
                ) {
                    break;
                }
                if p.consumed == 0 && p.produced == 0 {
                    return Err(E::Invalid("encoder made no progress"));
                }
            }
        } else {
            self.content.extend_from_slice(&self.input);
        }
        let kind = match (self.first, last) {
            (true, true) => 2,
            (true, false) => 3,
            (false, false) => 4,
            (false, true) => 5,
        };
        let codec = if self.raw.is_none() {
            0
        } else if !self.first {
            1
        } else if self.references.is_empty() {
            2
        } else {
            3
        };
        let mut header = bytes(48 + self.references.len())?;
        header.extend_from_slice(&[kind, codec]);
        if codec != 0 {
            super::number(self.input.len() as u64, &mut header)?;
        }
        if codec == 3 {
            header.extend_from_slice(&self.references);
        }
        header.push(
            u8::from(self.first && self.options.hidden)
                | (u8::from(last && self.options.id.is_some()) << 1),
        );
        if last && let Some(id) = self.options.id {
            header.push(3);
            header.extend_from_slice(&id.0);
        }
        core.queue(header, &self.content, true)?;
        self.input.clear();
        self.first = false;
        if last {
            self.queued_final = true;
        }
        Ok(())
    }
    pub(super) fn process(
        &mut self,
        call: Call<'_, '_>,
        input: &[u8],
        output: &mut [u8],
        operation: Operation,
    ) -> Result<Status, E> {
        let Call {
            core,
            compressor,
            dictionary,
            aggregate,
            total_in,
            consumed,
            produced,
        } = call;
        if self.finished {
            return Ok(Status::Finished);
        }
        let end = self
            .total_in
            .checked_add(input.len() as u64)
            .ok_or(E::Overflow)?;
        if let Some((op, boundary)) = self.boundary {
            if op != operation || boundary != end {
                return Err(E::InvalidState);
            }
        } else if operation != Operation::Process {
            self.boundary = Some((operation, end));
        }
        if let InputSize::Exact(expected) = self.expected
            && (end > expected || (operation == Operation::Finish && end != expected))
        {
            return Err(E::InputSizeMismatch {
                scope: "resource",
                expected,
                actual: end,
            });
        }
        let aggregate_end = total_in
            .checked_add(input.len() as u64)
            .ok_or(E::Overflow)?;
        if let InputSize::Exact(expected) = aggregate
            && aggregate_end > expected
        {
            return Err(E::InputSizeMismatch {
                scope: "container",
                expected,
                actual: aggregate_end,
            });
        }
        loop {
            let n = core.output(&mut output[*produced..]);
            *produced += n;
            self.total_out += n as u64;
            if core.has_pending() {
                return Ok(Status::NeedsOutput);
            }
            if self.queued_final {
                self.finished = true;
                core.active = false;
                core.after_resource = true;
                core.resources += 1;
                if let Some(raw) = self.raw.take() {
                    raw.release(compressor);
                }
                return Ok(Status::Finished);
            }
            if *consumed < input.len() {
                if self.input.len() == core.config.chunk_bytes {
                    self.emit(core, compressor, dictionary, false)?;
                    continue;
                }
                let n = (input.len() - *consumed).min(core.config.chunk_bytes - self.input.len());
                self.input
                    .extend_from_slice(&input[*consumed..*consumed + n]);
                *consumed += n;
                self.total_in += n as u64;
                *total_in += n as u64;
                continue;
            }
            match operation {
                Operation::Process => return Ok(Status::NeedsInput),
                Operation::Flush => {
                    if !self.input.is_empty() {
                        self.emit(core, compressor, dictionary, false)?;
                        continue;
                    }
                    self.boundary = None;
                    return Ok(Status::NeedsInput);
                }
                Operation::Finish => self.emit(core, compressor, dictionary, true)?,
            }
        }
    }
}

pub(super) struct Call<'a, 'd> {
    pub(super) core: &'a mut Container,
    pub(super) compressor: &'a mut Compressor,
    pub(super) dictionary: Option<&'d PreparedDictionary>,
    pub(super) aggregate: InputSize,
    pub(super) total_in: &'a mut u64,
    pub(super) consumed: &'a mut usize,
    pub(super) produced: &'a mut usize,
}

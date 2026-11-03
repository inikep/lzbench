//! Non-borrowing resource state; dictionaries are supplied only while driving it.
//!
//! Input is staged up to the chunk size. Each emitted chunk is one
//! independent Zstandard frame, compressed in one call exactly as
//! [`Compressor::compress_into`] does, so it decodes without any earlier
//! chunk.

use alloc::vec::Vec;

use super::{Body, Container, bytes};
use crate::compressor::framing::{
    FramedEncodeError as E, FramedEncoderStatus as Status, ResourceEncoding, ResourceOptions,
};
use crate::dictionary::PreparedDictionary;
use crate::framing::core::{FLAG_CHECKSUM, FLAG_DETACHED, FLAG_HIDDEN, ID_LEN, ID_TYPE};
use crate::shared::format::{BLOCK_HEADER_SIZE, BLOCK_SIZE_MAX, CHECKSUM_SIZE, MAGIC_NUMBER};
use crate::shared::xxhash::Xxh64;
use crate::{Compressor, EncodeError, InputSize, Operation, StreamConfig};

#[derive(Debug, Default)]
pub(in crate::compressor::framing) struct Resource {
    compressed: bool,
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

    pub(super) fn clear(&mut self) {
        self.input.clear();
        self.content.clear();
        self.references.clear();
        self.boundary = None;
    }

    pub(super) fn begin(
        &mut self,
        core: &mut Container,
        compressor: &Compressor,
        options: ResourceOptions,
        stream: StreamConfig,
        encoding: ResourceEncoding<'_>,
        references: Vec<u8>,
    ) -> Result<(), E> {
        core.begin()?;
        if let ResourceEncoding::Shared { dictionary, .. } = encoding
            && !dictionary.serves(&compressor.requested)
        {
            return Err(E::Encode(EncodeError::DictionaryMismatch));
        }
        if self.input.capacity() < core.config.chunk_bytes {
            self.input
                .try_reserve_exact(core.config.chunk_bytes - self.input.len())
                .map_err(|_| E::AllocationFailed)?;
        }
        self.compressed = !matches!(encoding, ResourceEncoding::Uncompressed);
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

    /// Reserves `size` bytes of content staging within the budget.
    fn reserve_content(&mut self, core: &Container, size: usize) -> Result<(), E> {
        if self.content.capacity() < size {
            core.check_buffer(
                size.saturating_add(self.input.capacity())
                    .saturating_add(self.references.capacity()),
            )?;
            self.content
                .try_reserve_exact(size)
                .map_err(|_| E::AllocationFailed)?;
        }
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
        // Visible payload must stay a top-level Zstandard frame; hidden
        // payload is wrapped in the chunk's skippable frame.
        let detached = !self.options.hidden;
        let codec = if !self.compressed && !detached {
            0
        } else if self.references.is_empty() {
            2
        } else {
            3
        };
        self.content.clear();
        if self.compressed {
            let bound =
                Compressor::max_compressed_size(self.input.len()).map_err(|_| E::Overflow)?;
            self.reserve_content(core, bound)?;
            match (dictionary, codec) {
                (Some(dictionary), 3) => {
                    compressor.compress_with_dictionary_into(
                        dictionary,
                        &self.input,
                        &mut self.content,
                    )?;
                }
                (None, 2) => {
                    compressor.compress_into(&self.input, &mut self.content)?;
                }
                _ => return Err(E::InvalidState),
            }
        } else if detached {
            self.reserve_content(core, stored_size(self.input.len()))?;
            stored_frame(
                &self.input,
                compressor.config().checksum(),
                &mut self.content,
            );
        } else {
            self.reserve_content(core, self.input.len())?;
            self.content.extend_from_slice(&self.input);
        }
        let kind = match (self.first, last) {
            (true, true) => 2,
            (true, false) => 3,
            (false, false) => 4,
            (false, true) => 5,
        };
        let mut header = bytes(32 + ID_LEN + self.references.len())?;
        header.extend_from_slice(&[kind, codec]);
        if codec != 0 {
            super::number(self.input.len() as u64, &mut header)?;
        }
        if codec == 3 {
            header.extend_from_slice(&self.references);
        }
        let checksum = last && self.options.id.is_some();
        header.push(
            if self.first && self.options.hidden {
                FLAG_HIDDEN
            } else {
                0
            } | if checksum { FLAG_CHECKSUM } else { 0 }
                | if detached { FLAG_DETACHED } else { 0 },
        );
        if checksum && let Some(id) = self.options.id {
            header.push(ID_TYPE);
            header.extend_from_slice(&id.0);
        }
        if detached {
            super::number(self.content.len() as u64, &mut header)?;
            // A chunk holds at most 16 MiB of input.
            let decoded = u32::try_from(self.input.len()).map_err(|_| E::Overflow)?;
            core.queue(
                header,
                &[],
                true,
                Some(Body {
                    bytes: &self.content,
                    decoded,
                }),
            )?;
        } else {
            core.queue(header, &self.content, true, None)?;
        }
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
        } else if operation != Operation::Continue {
            self.boundary = Some((operation, end));
        }
        if let InputSize::Exact(expected) = self.expected
            && (end > expected || (operation == Operation::End && end != expected))
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
                Operation::Continue => return Ok(Status::NeedsInput),
                Operation::Flush => {
                    if !self.input.is_empty() {
                        self.emit(core, compressor, dictionary, false)?;
                        continue;
                    }
                    self.boundary = None;
                    return Ok(Status::NeedsInput);
                }
                Operation::End => self.emit(core, compressor, dictionary, true)?,
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

/// Size of the frame [`stored_frame`] writes for `len` bytes.
fn stored_size(len: usize) -> usize {
    let blocks = len.div_ceil(BLOCK_SIZE_MAX).max(1);
    4 + 1 + 8 + blocks * BLOCK_HEADER_SIZE + len + CHECKSUM_SIZE
}

/// Appends `input` as one single-segment Zstandard frame of raw blocks,
/// with a content checksum when `checksum` is set.
fn stored_frame(input: &[u8], checksum: bool, output: &mut Vec<u8>) {
    output.extend_from_slice(&MAGIC_NUMBER.to_le_bytes());
    let len = input.len() as u64;
    let checksum_flag = if checksum { 4 } else { 0 };
    // Single segment: the content size field doubles as the window.
    if len <= 0xFF {
        output.push(0x20 | checksum_flag);
        output.push(len as u8);
    } else if len <= 0xFFFF + 0x100 {
        output.push(0x60 | checksum_flag);
        output.extend_from_slice(&((len - 0x100) as u16).to_le_bytes());
    } else if len <= u64::from(u32::MAX) {
        output.push(0xA0 | checksum_flag);
        output.extend_from_slice(&(len as u32).to_le_bytes());
    } else {
        output.push(0xE0 | checksum_flag);
        output.extend_from_slice(&len.to_le_bytes());
    }
    let mut blocks = input.chunks(BLOCK_SIZE_MAX).peekable();
    if blocks.peek().is_none() {
        output.extend_from_slice(&[1, 0, 0]);
    }
    while let Some(block) = blocks.next() {
        let last = u32::from(blocks.peek().is_none());
        let header = last | ((block.len() as u32) << 3);
        output.extend_from_slice(&header.to_le_bytes()[..BLOCK_HEADER_SIZE]);
        output.extend_from_slice(block);
    }
    if checksum {
        let mut hasher = Xxh64::new(0);
        hasher.update(input);
        output.extend_from_slice(&(hasher.digest() as u32).to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "decompression")]
    #[test]
    fn stored_frames_decode_at_every_content_size_field_width() {
        let mut decoder = crate::Decompressor::default();
        for len in [
            0,
            1,
            255,
            256,
            0x10FF,
            0x1_00FF,
            0x1_0100,
            BLOCK_SIZE_MAX + 1,
        ] {
            let input: Vec<u8> = (0..len).map(|i| (i * 7) as u8).collect();
            for checksum in [false, true] {
                let mut frame = Vec::new();
                stored_frame(&input, checksum, &mut frame);
                assert!(frame.len() <= stored_size(len));
                assert_eq!(decoder.decompress(&frame).unwrap(), input);
            }
        }
    }

    #[test]
    fn an_empty_stored_frame_is_one_empty_last_raw_block() {
        let mut frame = Vec::new();
        stored_frame(&[], false, &mut frame);
        assert_eq!(frame, [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0, 1, 0, 0]);
    }
}

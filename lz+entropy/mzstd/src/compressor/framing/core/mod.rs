//! Container ordering, checked wire sizes, durable output cursors and the
//! seek table.
//!
//! Every queued chunk becomes one skippable header frame, followed by its
//! Zstandard body frame when the content is detached. The container keeps
//! one seek table entry per frame and streams the table out after the
//! footer, so a contrib seekable decoder can index the container.

use alloc::{boxed::Box, vec::Vec};
pub(super) mod driver;
pub(super) mod engine;
mod metadata;
pub(super) mod resource;
use super::{
    DictionaryReference, FramingConfig, FramingError, MetadataEncoding, MetadataField,
    MetadataKind, MetadataOptions,
};
use crate::compressor::Compressor;
use crate::framing::core::{
    CHUNK_MAGIC, CONTAINER_HEADER, FLAG_CONTAINER, FOOTER_CONTENT, FRAME_HEADER, ID_LEN, ID_TYPE,
    MAX_FRAME_CONTENT, MAX_FRAMES, MAX_REFERENCES, MAX_VARINT, SEEK_ENTRY, SEEK_FOOTER,
    SEEK_TABLE_MAGIC, SEEKABLE_MAGIC, SIGNATURE, varint_len,
};

/// Seek table entries emitted per refill of the pending buffer.
const SEEK_BATCH: usize = 1024;

/// Appends `value` as a varint, reserving fallibly.
pub(super) fn number(value: u64, output: &mut Vec<u8>) -> Result<(), FramingError> {
    if value > MAX_VARINT {
        return Err(FramingError::Overflow);
    }
    output
        .try_reserve(varint_len(value))
        .map_err(|_| FramingError::AllocationFailed)?;
    let mut rest = value;
    while rest >= 0x80 {
        output.push((rest as u8) | 0x80);
        rest >>= 7;
    }
    output.push(rest as u8);
    Ok(())
}

/// A detached Zstandard frame following its chunk's header frame.
#[derive(Clone, Copy, Debug)]
pub(super) struct Body<'a> {
    /// The complete frame.
    pub(super) bytes: &'a [u8],
    /// Its decompressed size, for the seek table.
    pub(super) decoded: u32,
}

#[derive(Debug)]
struct Record {
    offset: u64,
    kind: u8,
    header: Vec<u8>,
    metadata: Vec<u8>,
    repeat_header: Vec<u8>,
}

/// Where the seek table's emission stands.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Seek {
    /// Not started.
    Idle,
    /// The frame header is out; entries from this one follow.
    Entries(usize),
    /// Every entry is out; the table footer follows.
    Footer,
}

#[derive(Debug)]
pub(super) struct Container {
    pub(super) config: FramingConfig,
    pending: Vec<u8>,
    cursor: usize,
    offset: u64,
    pub(super) chunks: u64,
    pub(super) resources: u64,
    pub(super) active: bool,
    pub(super) after_resource: bool,
    metadata_pending: bool,
    metadata_bytes: usize,
    records: Vec<Record>,
    /// Compressed and decompressed size of every frame so far.
    frames: Vec<[u32; 2]>,
    /// Most frames the seek table may list.
    frame_limit: u64,
    retained: usize,
    finishing: bool,
    repeat_cursor: usize,
    repeat_offset: u64,
    directory_offset: u64,
    directory_done: bool,
    footer_done: bool,
    seek: Seek,
    finished: bool,
    repeat_fields: Option<Box<[[u8; 2]]>>,
}

impl Container {
    pub(super) const fn offset(&self) -> u64 {
        self.offset
    }

    pub(super) fn validate(config: FramingConfig) -> Result<(), FramingError> {
        if !config.container && (config.central_directory || config.repeat_metadata) {
            return Err(FramingError::Invalid(
                "a single-resource profile cannot contain a directory or repeated metadata",
            ));
        }
        if config.repeat_metadata && !config.central_directory {
            return Err(FramingError::Invalid(
                "repeat metadata requires a central directory",
            ));
        }
        if config.chunk_bytes == 0
            || config.chunk_bytes > (1 << 24)
            || config.max_buffer_bytes < config.chunk_bytes.saturating_mul(4).saturating_add(8192)
        {
            return Err(FramingError::Invalid(
                "chunk size must be 1..=16 MiB and fit four times plus 8 KiB in the buffer budget",
            ));
        }
        Ok(())
    }

    pub(super) fn new(config: FramingConfig) -> Self {
        Self {
            config,
            pending: Vec::new(),
            cursor: 0,
            offset: CONTAINER_HEADER as u64,
            chunks: 0,
            resources: 0,
            active: false,
            after_resource: false,
            metadata_pending: false,
            metadata_bytes: 0,
            records: Vec::new(),
            frames: Vec::new(),
            frame_limit: MAX_FRAMES,
            retained: 0,
            finishing: false,
            repeat_cursor: 0,
            repeat_offset: 0,
            directory_offset: 0,
            directory_done: false,
            footer_done: false,
            seek: Seek::Idle,
            finished: false,
            repeat_fields: None,
        }
    }

    /// Bytes of the seek table entries' storage.
    fn frame_bytes(&self) -> usize {
        self.frames.capacity() * size_of::<[u32; 2]>()
    }

    pub(super) fn check_buffer(&self, extra: usize) -> Result<(), FramingError> {
        if self
            .retained
            .saturating_add(self.frame_bytes())
            .saturating_add(extra)
            > self.config.max_buffer_bytes
        {
            Err(FramingError::Limit {
                kind: "retained framing bytes",
                limit: self.config.max_buffer_bytes as u64,
            })
        } else {
            Ok(())
        }
    }

    pub(super) fn reset(&mut self) {
        let pending = ::core::mem::take(&mut self.pending);
        let records = ::core::mem::take(&mut self.records);
        let frames = ::core::mem::take(&mut self.frames);
        *self = Self::new(self.config);
        self.pending = pending;
        self.pending.clear();
        self.records = records;
        self.records.clear();
        self.frames = frames;
        self.frames.clear();
        self.retained = self.records.capacity() * size_of::<Record>();
    }

    /// Records one more frame for the seek table.
    fn push_frame(&mut self, compressed: usize, decoded: u32) -> Result<(), FramingError> {
        let compressed = u32::try_from(compressed).map_err(|_| FramingError::Overflow)?;
        if self.frames.len() == self.frames.capacity() {
            let target = self.frames.capacity().saturating_mul(2).max(16);
            self.check_buffer(
                (target - self.frames.capacity()).saturating_mul(size_of::<[u32; 2]>()),
            )?;
            self.frames
                .try_reserve_exact(target - self.frames.len())
                .map_err(|_| FramingError::AllocationFailed)?;
        }
        self.frames.push([compressed, decoded]);
        Ok(())
    }

    pub(super) fn start(&mut self) -> Result<(), FramingError> {
        self.reset();
        self.pending
            .try_reserve_exact(CONTAINER_HEADER)
            .map_err(|_| FramingError::AllocationFailed)?;
        self.pending.extend_from_slice(&CHUNK_MAGIC.to_le_bytes());
        self.pending
            .extend_from_slice(&((SIGNATURE.len() + 1) as u32).to_le_bytes());
        self.pending.extend_from_slice(&SIGNATURE);
        self.pending.push(if self.config.container {
            FLAG_CONTAINER
        } else {
            0
        });
        self.push_frame(CONTAINER_HEADER, 0)
    }

    pub(super) fn retained_bytes(&self) -> usize {
        self.retained + self.frame_bytes() + self.pending.capacity()
    }

    pub(super) const fn has_pending(&self) -> bool {
        self.cursor < self.pending.len()
    }

    pub(super) fn drain(&self) -> Result<(), FramingError> {
        if self.has_pending() {
            Err(FramingError::OutputPending)
        } else {
            Ok(())
        }
    }

    pub(super) fn output(&mut self, output: &mut [u8]) -> usize {
        let n = output.len().min(self.pending.len() - self.cursor);
        output[..n].copy_from_slice(&self.pending[self.cursor..self.cursor + n]);
        self.cursor += n;
        if self.cursor == self.pending.len() {
            self.pending.clear();
            self.cursor = 0;
        }
        n
    }

    pub(super) const fn is_finished(&self) -> bool {
        self.finished && !self.has_pending()
    }

    fn idle(&self) -> Result<(), FramingError> {
        self.drain()?;
        if self.active {
            return Err(FramingError::AbandonedResource);
        }
        if self.finishing {
            return Err(FramingError::Invalid(
                "resource is unfinished, abandoned, or container is finishing",
            ));
        }
        Ok(())
    }

    pub(super) fn begin(&mut self) -> Result<(), FramingError> {
        self.idle()?;
        if self.chunks >= self.config.max_chunks {
            return Err(FramingError::Limit {
                kind: "chunk count",
                limit: self.config.max_chunks,
            });
        }
        if self.resources >= self.config.max_resources
            || (!self.config.container && self.resources != 0)
        {
            return Err(FramingError::Limit {
                kind: "resource count",
                limit: self.config.max_resources,
            });
        }
        self.check_buffer(
            self.config
                .chunk_bytes
                .saturating_mul(4)
                .saturating_add(8192),
        )?;
        Ok(())
    }

    /// Queues one chunk: a skippable frame of `header` and inline
    /// `content`, then `body` when the content is detached.
    pub(super) fn queue(
        &mut self,
        header: Vec<u8>,
        content: &[u8],
        record: bool,
        body: Option<Body<'_>>,
    ) -> Result<(), FramingError> {
        if !self.pending.is_empty() {
            return Err(FramingError::Invalid("pending chunk must be drained"));
        }
        if self.chunks >= self.config.max_chunks {
            return Err(FramingError::Limit {
                kind: "chunk count",
                limit: self.config.max_chunks,
            });
        }
        let frames = 1 + usize::from(body.is_some());
        if (self.frames.len() + frames) as u64 > self.frame_limit {
            return Err(FramingError::Limit {
                kind: "seek table frames",
                limit: self.frame_limit,
            });
        }
        let frame_size = header
            .len()
            .checked_add(content.len())
            .filter(|n| *n as u64 <= MAX_FRAME_CONTENT)
            .ok_or(FramingError::Overflow)?;
        let body_len = body.map_or(0, |b| b.bytes.len());
        let total = (FRAME_HEADER + frame_size)
            .checked_add(body_len)
            .ok_or(FramingError::Overflow)?;
        let next_offset = self
            .offset
            .checked_add(total as u64)
            .filter(|v| *v <= MAX_VARINT)
            .ok_or(FramingError::Overflow)?;
        let kind = header[0];
        let record_capacity = if record && self.records.len() == self.records.capacity() {
            self.records.capacity().saturating_mul(2).max(4)
        } else {
            self.records.capacity()
        };
        let record_bytes = if record {
            (record_capacity - self.records.capacity()).saturating_mul(size_of::<Record>())
                + header.len()
                + FRAME_HEADER
        } else {
            0
        };
        self.check_buffer(
            total
                .saturating_mul(2)
                .saturating_add(self.pending.capacity())
                .saturating_add(record_bytes)
                .saturating_add(self.config.chunk_bytes * 2),
        )?;
        let mut complete_header = bytes(FRAME_HEADER + header.len())?;
        complete_header.extend_from_slice(&CHUNK_MAGIC.to_le_bytes());
        complete_header.extend_from_slice(&(frame_size as u32).to_le_bytes());
        complete_header.extend_from_slice(&header);
        let mut pending = bytes(total)?;
        pending.extend_from_slice(&complete_header);
        pending.extend_from_slice(content);
        if let Some(body) = body {
            pending.extend_from_slice(body.bytes);
        }
        if record && record_capacity > self.records.capacity() {
            self.records
                .try_reserve_exact(record_capacity - self.records.len())
                .map_err(|_| FramingError::AllocationFailed)?;
        }
        let listed = self.frames.len();
        self.push_frame(FRAME_HEADER + frame_size, 0)?;
        if let Some(body) = body
            && let Err(error) = self.push_frame(body.bytes.len(), body.decoded)
        {
            self.frames.truncate(listed);
            return Err(error);
        }
        if record {
            self.records.push(Record {
                offset: self.offset,
                kind,
                header: complete_header,
                metadata: Vec::new(),
                repeat_header: Vec::new(),
            });
            self.retained += record_bytes;
        }
        self.pending = pending;
        self.offset = next_offset;
        self.chunks += 1;
        Ok(())
    }

    /// Validates and serializes dictionary references.
    pub(super) fn references(
        &self,
        references: &[DictionaryReference],
    ) -> Result<Vec<u8>, FramingError> {
        let serialized = references
            .iter()
            .filter(|r| {
                matches!(
                    r,
                    DictionaryReference::SerializedId(_)
                        | DictionaryReference::SerializedResource(_)
                )
            })
            .count();
        if references.is_empty()
            || references.len() > MAX_REFERENCES
            || (serialized != 0 && references.len() != 1)
        {
            return Err(FramingError::Invalid(
                "a chunk takes one serialized dictionary or 1..=16 prefixes",
            ));
        }
        let size = 1 + references.len() * (1 + ID_LEN);
        self.check_buffer(size)?;
        let mut bytes = bytes(size)?;
        bytes.push(references.len() as u8);
        for reference in references {
            let (flag, id, pointer) = match *reference {
                DictionaryReference::PrefixId(id) => (2, Some(id), None),
                DictionaryReference::SerializedId(id) => (6, Some(id), None),
                DictionaryReference::PrefixResource(offset) => (0, None, Some(offset)),
                DictionaryReference::SerializedResource(offset) => (4, None, Some(offset)),
                DictionaryReference::PrefixChunk(offset) => (1, None, Some(offset)),
            };
            bytes.push(flag);
            if let Some(id) = id {
                bytes.push(ID_TYPE);
                bytes.extend_from_slice(&id.0);
            }
            if let Some(offset) = pointer {
                let valid = self
                    .records
                    .iter()
                    .any(|r| r.offset == offset && (flag & 3 == 1 || matches!(r.kind, 2 | 3)));
                if !valid {
                    return Err(FramingError::Invalid(
                        "dictionary pointer must address an earlier content chunk or resource",
                    ));
                }
                number(offset, &mut bytes)?;
            }
        }
        Ok(bytes)
    }

    pub(super) fn repeat_metadata_fields(&mut self, codes: &[[u8; 2]]) -> Result<(), FramingError> {
        self.idle()?;
        if !self.config.repeat_metadata || self.records.iter().any(|r| matches!(r.kind, 1 | 6 | 7))
        {
            return Err(FramingError::Invalid(
                "select repeated fields before any metadata with repetition enabled",
            ));
        }
        if codes.len() > 678 {
            return Err(FramingError::Limit {
                kind: "repeated field codes",
                limit: 678,
            });
        }
        for (i, code) in codes.iter().enumerate() {
            if !(code.iter().all(u8::is_ascii_uppercase) || matches!(code, b"id" | b"mt"))
                || codes[..i].contains(code)
            {
                return Err(FramingError::Invalid(
                    "invalid or duplicate repeated field code",
                ));
            }
        }
        let bytes = size_of_val(codes);
        self.check_buffer(bytes)?;
        let mut selection = Vec::new();
        selection
            .try_reserve_exact(codes.len())
            .map_err(|_| FramingError::AllocationFailed)?;
        selection.extend_from_slice(codes);
        self.retained -= self.repeat_fields.as_ref().map_or(0, |v| size_of_val(&**v));
        self.repeat_fields = Some(selection.into_boxed_slice());
        self.retained += bytes;
        Ok(())
    }

    fn metadata_references(
        &self,
        encoding: MetadataEncoding<'_>,
        repeated: bool,
    ) -> Result<Vec<u8>, FramingError> {
        if let MetadataEncoding::Shared { references, .. } = encoding {
            if repeated
                && references.iter().any(|r| {
                    !matches!(
                        r,
                        DictionaryReference::PrefixId(_) | DictionaryReference::SerializedId(_)
                    )
                })
            {
                return Err(FramingError::Invalid(
                    "pre-encoded repeated metadata requires external dictionary references",
                ));
            }
            self.references(references)
        } else {
            Ok(Vec::new())
        }
    }

    pub(super) fn metadata(
        &mut self,
        compressor: &mut Compressor,
        kind: MetadataKind,
        fields: &[MetadataField<'_>],
        options: MetadataOptions<'_>,
    ) -> Result<(), FramingError> {
        self.idle()?;
        if !self.config.container {
            return Err(FramingError::Invalid(
                "metadata requires a container footer",
            ));
        }
        if self.metadata_pending || (kind == MetadataKind::Footer && !self.after_resource) {
            return Err(FramingError::Invalid(
                "metadata is not adjacent to its resource",
            ));
        }
        let mut length = 0usize;
        let mut name = false;
        let mut modified = false;
        for field in fields {
            let valid = if field.code.iter().all(u8::is_ascii_uppercase) {
                true
            } else if kind == MetadataKind::Resource && field.code == *b"id" && !name {
                name = true;
                ::core::str::from_utf8(field.value).is_ok()
            } else if kind == MetadataKind::Resource && field.code == *b"mt" && !modified {
                modified = true;
                field.value.len() == 8
            } else {
                false
            };
            if !valid {
                return Err(FramingError::Invalid(
                    "invalid or duplicate reserved metadata field",
                ));
            }
            length = length
                .checked_add(2 + varint_len(field.value.len() as u64))
                .and_then(|v| v.checked_add(field.value.len()))
                .ok_or(FramingError::Overflow)?;
        }
        let total = self
            .metadata_bytes
            .checked_add(length)
            .ok_or(FramingError::Overflow)?;
        if total > self.config.max_metadata_bytes {
            return Err(FramingError::Limit {
                kind: "metadata bytes",
                limit: self.config.max_metadata_bytes as u64,
            });
        }
        let bound = Compressor::max_compressed_size(length).map_err(|_| FramingError::Overflow)?;
        // Original and repeated inputs/outputs, pending storage and queue copies
        // coexist. Include output capacities, not merely compressed lengths.
        self.check_buffer(
            bound
                .saturating_mul(8)
                .saturating_add(length.saturating_mul(4))
                .saturating_add(self.pending.capacity())
                .saturating_add(8192),
        )?;
        if self.chunks >= self.config.max_chunks {
            return Err(FramingError::Limit {
                kind: "chunk count",
                limit: self.config.max_chunks,
            });
        }
        let repeat = self.config.repeat_metadata && kind != MetadataKind::Global;
        let references = self.metadata_references(options.encoding, false)?;
        let repeat_references = if repeat {
            self.metadata_references(options.repeated_encoding, true)?
        } else {
            Vec::new()
        };
        let content = metadata::serialize(fields, None, length)?;
        let code = match kind {
            MetadataKind::Resource => 1,
            MetadataKind::Footer => 6,
            MetadataKind::Global => 7,
        };
        let (header, content) =
            metadata::encode(compressor, code, content, options.encoding, references)?;
        let (repeat_header, repeated) = if repeat {
            let selected = metadata::serialize(fields, self.repeat_fields.as_deref(), length)?;
            let (mut header, content) = metadata::encode(
                compressor,
                8,
                selected,
                options.repeated_encoding,
                repeat_references,
            )?;
            header.push(code);
            (header, content)
        } else {
            (Vec::new(), Vec::new())
        };
        let repeat_bytes = repeat_header.capacity().saturating_add(repeated.capacity());
        self.check_buffer(
            repeat_bytes
                .saturating_add(content.capacity())
                .saturating_add(8192),
        )?;
        self.retained += repeat_bytes;
        if let Err(error) = self.queue(header, &content, true, None) {
            self.retained -= repeat_bytes;
            return Err(error);
        }
        if let Some(record) = self.records.last_mut() {
            record.metadata = repeated;
            record.repeat_header = repeat_header;
        }
        self.metadata_bytes = total;
        self.metadata_pending = kind == MetadataKind::Resource;
        self.after_resource = false;
        Ok(())
    }

    pub(super) fn padding(&mut self, bytes: usize) -> Result<(), FramingError> {
        self.idle()?;
        self.check_buffer(bytes.saturating_mul(3).saturating_add(8192))?;
        let mut content = self::bytes(bytes)?;
        content.resize(bytes, 0);
        self.queue(copy(&[0])?, &content, false, None)
    }

    /// Queues the next piece of the suffix; repeated until `finished`.
    pub(super) fn finish(&mut self) -> Result<(), FramingError> {
        if self.active || self.metadata_pending {
            return Err(FramingError::Invalid(
                "resource is missing, unfinished, or abandoned",
            ));
        }
        if !self.config.container && self.resources != 1 {
            return Err(FramingError::Invalid(
                "single-resource profile requires exactly one resource",
            ));
        }
        self.drain()?;
        self.finishing = true;
        if self.finished {
            return Ok(());
        }
        if self.config.container {
            if self.config.repeat_metadata {
                while self.repeat_cursor < self.records.len() {
                    let record = &self.records[self.repeat_cursor];
                    if matches!(record.kind, 1 | 6) {
                        self.check_buffer(
                            record
                                .metadata
                                .len()
                                .saturating_mul(3)
                                .saturating_add(record.repeat_header.len())
                                .saturating_add(8192),
                        )?;
                        let header = copy(&record.repeat_header)?;
                        let content = copy(&record.metadata)?;
                        let offset = self.offset;
                        self.queue(header, &content, true, None)?;
                        if self.repeat_offset == 0 {
                            self.repeat_offset = offset;
                        }
                    }
                    self.repeat_cursor += 1;
                    if self.has_pending() {
                        return Ok(());
                    }
                }
            }
            if self.config.central_directory && !self.directory_done {
                let bound = self
                    .records
                    .iter()
                    .try_fold(9usize, |sum, r| sum.checked_add(18 + r.header.len()))
                    .ok_or(FramingError::Overflow)?;
                self.check_buffer(bound.saturating_mul(3).saturating_add(8192))?;
                let mut content = bytes(bound)?;
                number(self.repeat_offset, &mut content)?;
                for record in &self.records {
                    number(record.offset, &mut content)?;
                    number(record.header.len() as u64, &mut content)?;
                    content.extend_from_slice(&record.header);
                }
                let offset = self.offset;
                self.queue(copy(&[9])?, &content, false, None)?;
                self.directory_offset = offset;
                self.directory_done = true;
                return Ok(());
            }
            if !self.footer_done {
                // The seek table lists every frame up to the footer's.
                let frames = self.frames.len() as u64 + 1;
                let footer = (FRAME_HEADER + FOOTER_CONTENT) as u64;
                let size = frames
                    .checked_mul(SEEK_ENTRY as u64)
                    .and_then(|n| n.checked_add((FRAME_HEADER + SEEK_FOOTER) as u64))
                    .and_then(|n| n.checked_add(footer))
                    .and_then(|n| n.checked_add(self.offset))
                    .ok_or(FramingError::Overflow)?;
                let mut content = bytes(FOOTER_CONTENT - 1)?;
                content.extend_from_slice(&self.directory_offset.to_le_bytes());
                content.extend_from_slice(&size.to_le_bytes());
                self.queue(copy(&[10])?, &content, false, None)?;
                self.footer_done = true;
                return Ok(());
            }
        }
        self.seek_table()
    }

    /// Queues the next piece of the seek table.
    fn seek_table(&mut self) -> Result<(), FramingError> {
        match self.seek {
            Seek::Idle => {
                let size = self
                    .frames
                    .len()
                    .checked_mul(SEEK_ENTRY)
                    .and_then(|n| n.checked_add(SEEK_FOOTER))
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or(FramingError::Overflow)?;
                self.reserve_pending(FRAME_HEADER)?;
                self.pending
                    .extend_from_slice(&SEEK_TABLE_MAGIC.to_le_bytes());
                self.pending.extend_from_slice(&size.to_le_bytes());
                self.seek = Seek::Entries(0);
            }
            Seek::Entries(start) => {
                let end = (start + SEEK_BATCH).min(self.frames.len());
                self.reserve_pending((end - start) * SEEK_ENTRY)?;
                for [compressed, decoded] in &self.frames[start..end] {
                    self.pending.extend_from_slice(&compressed.to_le_bytes());
                    self.pending.extend_from_slice(&decoded.to_le_bytes());
                }
                self.seek = if end == self.frames.len() {
                    Seek::Footer
                } else {
                    Seek::Entries(end)
                };
            }
            Seek::Footer => {
                self.reserve_pending(SEEK_FOOTER)?;
                self.pending
                    .extend_from_slice(&(self.frames.len() as u32).to_le_bytes());
                self.pending.push(0);
                self.pending
                    .extend_from_slice(&SEEKABLE_MAGIC.to_le_bytes());
                self.finished = true;
            }
        }
        Ok(())
    }

    /// Grows the drained pending buffer to hold `size` bytes.
    fn reserve_pending(&mut self, size: usize) -> Result<(), FramingError> {
        if self.pending.capacity() < size {
            self.check_buffer(size)?;
            self.pending
                .try_reserve_exact(size)
                .map_err(|_| FramingError::AllocationFailed)?;
        }
        Ok(())
    }
}

/// Fallible bounded staging allocation.
pub(super) fn bytes(capacity: usize) -> Result<Vec<u8>, FramingError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| FramingError::AllocationFailed)?;
    Ok(bytes)
}

pub(super) fn copy(source: &[u8]) -> Result<Vec<u8>, FramingError> {
    let mut bytes = bytes(source.len())?;
    bytes.extend_from_slice(source);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn scalar_and_each_host_backend_preserve_framed_bytes() {
        use crate::compressor::framing::{
            FramedCompressor, FramedEncodeConfig, FramedInput, FramedItem, FramedResource,
        };
        use crate::{Backend, EncoderConfig, Level};
        let data = b"same framing and raw resource bytes across backends".repeat(20);
        let items = [FramedItem::Resource(FramedResource::from(data.as_slice()))];
        let input = FramedInput::from(items.as_slice());
        for level in [1, 3, 19] {
            let config = FramedEncodeConfig::default().with_encoder_config(
                EncoderConfig::default().with_level(Level::try_from(level).unwrap()),
            );
            let expected = FramedCompressor::builder(config)
                .with_backend(Backend::SCALAR)
                .build()
                .unwrap()
                .compress(input)
                .unwrap();
            for backend in Backend::available() {
                assert_eq!(
                    FramedCompressor::builder(config)
                        .with_backend(backend)
                        .build()
                        .unwrap()
                        .compress(input)
                        .unwrap(),
                    expected
                );
            }
        }
    }

    #[test]
    fn varints_write_seven_bit_groups_and_reject_values_above_63_bits() {
        let mut bytes = Vec::new();
        number(300, &mut bytes).unwrap();
        assert_eq!(bytes, [0xac, 0x02]);
        assert!(matches!(
            number(u64::MAX, &mut Vec::new()),
            Err(FramingError::Overflow)
        ));
    }

    #[test]
    fn chunk_limits_and_offset_overflow_fail_without_queuing() {
        let mut container = Container::new(FramingConfig {
            max_chunks: 0,
            ..Default::default()
        });
        assert!(matches!(
            container.begin(),
            Err(FramingError::Limit {
                kind: "chunk count",
                limit: 0
            })
        ));
        container.config.max_chunks = 1;
        container.offset = MAX_VARINT;
        assert!(matches!(
            container.queue(vec![0], b"x", false, None),
            Err(FramingError::Overflow)
        ));
        assert!(container.pending.is_empty());
        assert_eq!(container.chunks, 0);
    }

    #[test]
    fn the_seek_table_frame_limit_is_checked_before_queuing() {
        let mut container = Container::new(FramingConfig::default());
        container.frame_limit = 2;
        container.frames = vec![[0, 0]; 2];
        assert!(matches!(
            container.queue(vec![0], b"", false, None),
            Err(FramingError::Limit {
                kind: "seek table frames",
                limit: 2
            })
        ));
        assert_eq!(container.chunks, 0);
    }

    #[test]
    fn oversized_frames_overflow_the_seek_table_entry() {
        let mut container = Container::new(FramingConfig::default());
        assert!(matches!(
            container.push_frame(u32::MAX as usize + 1, 0),
            Err(FramingError::Overflow)
        ));
    }
}

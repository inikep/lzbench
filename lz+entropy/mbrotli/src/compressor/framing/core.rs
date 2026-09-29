//! Container ordering, checked wire sizes, and durable output cursors.

#[cfg(test)]
use alloc::vec;
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
use crate::compressor::core::rfc9841::varint;

pub(super) fn number(value: u64, output: &mut Vec<u8>) -> Result<(), FramingError> {
    if value > varint::MAX_VARINT {
        return Err(FramingError::Overflow);
    }
    output
        .try_reserve(varint::encoded_len(value))
        .map_err(|_| FramingError::AllocationFailed)?;
    varint::write(value, output).map_err(|_| FramingError::Overflow)
}

#[derive(Debug)]
struct Record {
    offset: u64,
    kind: u8,
    header: Vec<u8>,
    metadata: Vec<u8>,
    repeat_header: Vec<u8>,
}

#[derive(Debug)]
pub(super) struct Container {
    pub(super) config: FramingConfig,
    pending: Vec<u8>,
    cursor: usize,
    offset: u64,
    chunks: u64,
    pub(super) resources: u64,
    pub(super) active: bool,
    pub(super) after_resource: bool,
    metadata_pending: bool,
    metadata_bytes: usize,
    records: Vec<Record>,
    retained: usize,
    finishing: bool,
    repeat_cursor: usize,
    repeat_offset: u64,
    directory_offset: u64,
    directory_done: bool,
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
            offset: 5,
            chunks: 0,
            resources: 0,
            active: false,
            after_resource: false,
            metadata_pending: false,
            metadata_bytes: 0,
            records: Vec::new(),
            retained: 0,
            finishing: false,
            repeat_cursor: 0,
            repeat_offset: 0,
            directory_offset: 0,
            directory_done: false,
            finished: false,
            repeat_fields: None,
        }
    }

    pub(super) fn check_buffer(&self, extra: usize) -> Result<(), FramingError> {
        if self.retained.saturating_add(extra) > self.config.max_buffer_bytes {
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
        *self = Self::new(self.config);
        self.pending = pending;
        self.pending.clear();
        self.records = records;
        self.records.clear();
        self.retained = self.records.capacity() * size_of::<Record>();
    }
    pub(super) fn start(&mut self) -> Result<(), FramingError> {
        self.reset();
        self.pending
            .try_reserve_exact(5)
            .map_err(|_| FramingError::AllocationFailed)?;
        self.pending.extend_from_slice(&[
            0x91,
            10,
            66,
            82,
            if self.config.container { 4 } else { 0 },
        ]);
        Ok(())
    }
    pub(super) fn retained_bytes(&self) -> usize {
        self.retained + self.pending.capacity()
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
        self.drain()?;
        Ok(())
    }

    pub(super) fn queue(
        &mut self,
        header: Vec<u8>,
        content: &[u8],
        record: bool,
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
        let length = header
            .len()
            .checked_add(content.len())
            .ok_or(FramingError::Overflow)?;
        let total = length
            .checked_add(varint::encoded_len(length as u64))
            .ok_or(FramingError::Overflow)?;
        let next_offset = self
            .offset
            .checked_add(total as u64)
            .filter(|v| *v <= varint::MAX_VARINT)
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
                + 9
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
        let mut complete_header = bytes(header.len() + 9)?;
        number(length as u64, &mut complete_header)?;
        complete_header.extend_from_slice(&header);
        let mut pending = bytes(total)?;
        pending.extend_from_slice(&complete_header);
        pending.extend_from_slice(content);
        if record {
            if record_capacity > self.records.capacity() {
                self.records
                    .try_reserve_exact(record_capacity - self.records.len())
                    .map_err(|_| FramingError::AllocationFailed)?;
            }
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

    pub(super) fn references(
        &self,
        references: &[DictionaryReference],
    ) -> Result<Vec<u8>, FramingError> {
        if references.is_empty() || references.len() > 16 {
            return Err(FramingError::Invalid(
                "shared chunks require 1..=16 dictionary references",
            ));
        }
        let mut serialized = 0;
        let mut prefixes = 0;
        self.check_buffer(1 + references.len() * 35)?;
        let mut bytes = bytes(1 + references.len() * 35)?;
        bytes.push(references.len() as u8);
        for reference in references {
            let (flag, id, pointer) = match *reference {
                DictionaryReference::PrefixId(id) => (2, Some(id), None),
                DictionaryReference::SerializedId(id) => (6, Some(id), None),
                DictionaryReference::PrefixResource(offset) => (0, None, Some(offset)),
                DictionaryReference::SerializedResource(offset) => (4, None, Some(offset)),
                DictionaryReference::PrefixChunk(offset) => (1, None, Some(offset)),
            };
            if flag & 4 != 0 {
                serialized += 1;
            } else {
                prefixes += 1;
            }
            if serialized > 1 || prefixes > 15 {
                return Err(FramingError::Invalid(
                    "at most one serialized and fifteen prefix references",
                ));
            }
            bytes.push(flag);
            if let Some(id) = id {
                bytes.push(3);
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
                .checked_add(2 + varint::encoded_len(field.value.len() as u64))
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
        self.drain()?;
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
        if let Err(error) = self.queue(header, &content, true) {
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
        self.drain()?;
        let mut content = self::bytes(bytes)?;
        content.resize(bytes, 0);
        self.queue(copy(&[0])?, &content, false)
    }

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
        if self.config.container && !self.finished {
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
                        self.queue(header, &content, true)?;
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
                self.queue(copy(&[9])?, &content, false)?;
                self.directory_offset = offset;
                self.directory_done = true;
                return Ok(());
            }
            let content = footer(self.offset, self.directory_offset)?;
            self.queue(copy(&[10])?, &content, false)?;
            self.finished = true;
        }
        self.finished = true;
        Ok(())
    }
}

/// The file-size field counts its own bytes and the enclosing length varint.
fn footer(offset: u64, directory: u64) -> Result<Vec<u8>, FramingError> {
    let mut size = offset;
    loop {
        let length = 1 + varint::encoded_len(size) + varint::encoded_len(directory);
        let next = offset
            .checked_add((length + varint::encoded_len(length as u64)) as u64)
            .filter(|v| *v <= varint::MAX_VARINT)
            .ok_or(FramingError::Overflow)?;
        if size == next {
            break;
        }
        size = next;
    }
    let mut bytes = bytes(18)?;
    number(size, &mut bytes)?;
    bytes.reverse();
    let start = bytes.len();
    number(directory, &mut bytes)?;
    bytes[start..].reverse();
    Ok(bytes)
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

    #[test]
    fn scalar_and_each_host_backend_preserve_framed_bytes() {
        use crate::compressor::framing::{
            FramedCompressor, FramedEncodeConfig, FramedInput, FramedItem, FramedResource,
        };
        use crate::{Backend, EncoderConfig, Quality};
        let data = b"same framing and raw resource bytes across backends".repeat(20);
        let items = [FramedItem::Resource(FramedResource::from(data.as_slice()))];
        let input = FramedInput::from(items.as_slice());
        for quality in [Quality::Q0, Quality::Q5, Quality::Q11] {
            let config = FramedEncodeConfig::default()
                .with_encoder_config(EncoderConfig::default().with_quality(quality));
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
    fn footer_size_converges_across_varint_width_boundaries() {
        for offset in [0, 120, 127, 128, 16375, 16383, 16384, (1 << 56) - 10] {
            let content = footer(offset, 5).expect("footer");
            let mut reversed = content.clone();
            reversed.reverse();
            let (directory, used) = varint::read(&reversed).expect("directory");
            let (size, _) = varint::read(&reversed[used..]).expect("size");
            assert_eq!(directory, 5);
            let length = content.len() + 1;
            assert_eq!(
                size,
                offset + (length + varint::encoded_len(length as u64)) as u64
            );
        }
        assert!(matches!(
            footer(varint::MAX_VARINT, 0),
            Err(FramingError::Overflow)
        ));
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
        container.output(&mut [0; 5]);
        container.config.max_chunks = 1;
        container.offset = varint::MAX_VARINT;
        assert!(matches!(
            container.queue(vec![0], b"x", false),
            Err(FramingError::Overflow)
        ));
        assert!(container.pending.is_empty());
        assert_eq!(container.chunks, 0);
    }
}

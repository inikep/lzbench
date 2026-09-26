//! Bounded container state and event transitions; no caller borrows are stored.
#[cfg(not(feature = "no_std"))]
pub(in crate::decompressor::framing) mod seek;
mod wire;
use super::*;
use crate::decompressor::core::{Input, Output, Stop, Stream};
use crate::dictionary::{DecodeDictionary, DecodeDictionaryLimits, DictionaryAttachment};
use crate::framing::{DictionaryReference, MetadataKind};
use crate::{Backend, DecodeOperation, DecoderConfig, OutputSize, WindowLimit};
use ::core::ops::Range;
use FramedDecodeError as E;
use FramedLimitKind as L;
use alloc::vec::Vec;

#[derive(Debug, Clone, Copy)]
pub(super) enum Tag {
    Input,
    Output,
    Finished,
    Stream,
    Header,
    ChunkEnd,
    Start,
    Data(ResourceDataPosition),
    End,
    Metadata(usize),
    DirectoryStart,
    DirectoryEntry(usize),
    DirectoryEnd,
    Padding,
    Footer,
}
#[derive(Debug, Clone, Copy)]
enum Phase {
    Detect,
    Header,
    Begin,
    Content,
    Semantic,
    ChunkEnd,
    ResourceEnd,
    Directory,
    DirectoryEnd,
    Finish,
    Failed,
}
#[derive(Debug)]
pub(super) struct Resource {
    pub header: ResourceHeader,
    pub size: u64,
    pub metadata: Option<usize>,
    pub footer: Option<usize>,
    pub checksum: Option<crate::framing::DictionaryId>,
    cache: Range<usize>,
    complete: bool,
}
#[derive(Debug)]
pub(super) struct Meta {
    pub value: Option<Metadata>,
    pub scope: MetadataScope,
    pub original: Option<ChunkOffset>,
    pub kind: MetadataKind,
}
#[derive(Debug)]
struct Content {
    chunk: usize,
    cache: Range<usize>,
    metadata: Option<usize>,
    resource: Option<usize>,
}

#[derive(Debug)]
pub(super) struct Engine {
    pub config: FramedDecodeConfig,
    pub stream: FramedDecodeStreamConfig,
    pub codec: Stream,
    pub format: Option<StreamInfo>,
    pub total_in: u64,
    pub total_out: u64,
    pub total_decoded: u64,
    pub completed: u64,
    pub finished: bool,
    pub layout: FramedLayout,
    pub resources: Vec<Resource>,
    pub metadata: Vec<Meta>,
    phase: Phase,
    final_end: Option<u64>,
    pending: Vec<u8>,
    scratch: Vec<u8>,
    cache: Vec<u8>,
    contents: Vec<Content>,
    dictionary: Option<DecodeDictionary>,
    codec_config: DecoderConfig,
    codec_active: bool,
    codec_ended: bool,
    current: usize,
    chunk_left: u64,
    chunk_out: u64,
    cache_start: usize,
    previous: Option<ChunkType>,
    partial: bool,
    pending_meta: Option<usize>,
    repeat_count: usize,
    original_cursor: usize,
    repeat_start: Option<ChunkOffset>,
    metadata_bytes: u64,
    metadata_fields: u64,
    directory_cursor: usize,
    directory_pointer: bool,
    collector_bytes: usize,
}

pub(super) fn check(value: u64, limit: Option<u64>, kind: L) -> Result<(), E> {
    if let Some(limit) = limit
        && value > limit
    {
        return Err(E::LimitExceeded { kind, limit });
    }
    Ok(())
}
fn plus(a: u64, b: usize) -> Result<u64, E> {
    a.checked_add(b as u64).ok_or(E::SizeOverflow)
}

impl Engine {
    pub fn new(
        config: FramedDecodeConfig,
        stream: FramedDecodeStreamConfig,
        codec: Stream,
    ) -> Self {
        Self {
            config,
            stream,
            codec,
            format: None,
            total_in: 0,
            total_out: 0,
            total_decoded: 0,
            completed: 0,
            finished: false,
            layout: FramedLayout::default(),
            resources: Vec::new(),
            metadata: Vec::new(),
            phase: Phase::Detect,
            final_end: None,
            pending: Vec::new(),
            scratch: Vec::new(),
            cache: Vec::new(),
            contents: Vec::new(),
            dictionary: None,
            codec_config: DecoderConfig::default(),
            codec_active: false,
            codec_ended: false,
            current: 0,
            chunk_left: 0,
            chunk_out: 0,
            cache_start: 0,
            previous: None,
            partial: false,
            pending_meta: None,
            repeat_count: 0,
            original_cursor: 0,
            repeat_start: None,
            metadata_bytes: 0,
            metadata_fields: 0,
            directory_cursor: 0,
            directory_pointer: false,
            collector_bytes: 0,
        }
    }
    pub fn clear(&mut self) {
        let mut codec = ::core::mem::take(&mut self.codec);
        codec.reset(DecoderConfig::default());
        let mut next = Self::new(self.config, FramedDecodeStreamConfig::default(), codec);
        // Retain capacities, never semantic identities or dictionary borrows.
        self.pending.clear();
        next.pending = ::core::mem::take(&mut self.pending);
        self.scratch.clear();
        next.scratch = ::core::mem::take(&mut self.scratch);
        self.cache.clear();
        next.cache = ::core::mem::take(&mut self.cache);
        self.contents.clear();
        next.contents = ::core::mem::take(&mut self.contents);
        self.resources.clear();
        next.resources = ::core::mem::take(&mut self.resources);
        self.metadata.clear();
        next.metadata = ::core::mem::take(&mut self.metadata);
        self.layout.chunks.clear();
        next.layout.chunks = ::core::mem::take(&mut self.layout.chunks);
        *self = next;
    }
    /// Makes every later call report `InvalidState`.
    pub const fn poison(&mut self) {
        self.phase = Phase::Failed;
    }
    pub fn fits_policy(&self) -> bool {
        self.budget(0, 0).is_ok()
    }
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    fn framing_bytes(&self) -> usize {
        self.collector_bytes
            + self.pending.capacity()
            + self.scratch.capacity()
            + self.layout.chunks.capacity() * size_of::<ChunkInfo>()
            + self.resources.capacity() * size_of::<Resource>()
            + self.metadata.capacity() * size_of::<Meta>()
            + self.contents.capacity() * size_of::<Content>()
            + self
                .layout
                .chunks
                .iter()
                .map(|c| {
                    c.header_bytes.capacity()
                        + c.dictionaries.capacity() * size_of::<DictionaryReference>()
                })
                .sum::<usize>()
            + self
                .metadata
                .iter()
                .filter_map(|m| m.value.as_ref())
                .map(|m| {
                    m.bytes.capacity() + m.fields.capacity() * size_of::<([u8; 2], Range<usize>)>()
                })
                .sum::<usize>()
            + self
                .layout
                .directory
                .as_ref()
                .map_or(0, |d| d.entries.capacity() * size_of::<DirectoryEntry>())
    }
    fn dictionary_bytes(&self) -> usize {
        self.cache.capacity()
            + self
                .dictionary
                .as_ref()
                .map_or(0, DecodeDictionary::retained_bytes)
    }
    pub fn retained_bytes(&self) -> usize {
        self.framing_bytes() + self.dictionary_bytes() + self.codec.retained_bytes()
    }
    fn budget(&self, framing: usize, dictionary: usize) -> Result<(), E> {
        let limits = self.config.limits();
        check(
            plus(self.framing_bytes() as u64, framing)?,
            limits.max_framing_bytes.map(|x| x as u64),
            L::FramingBytes,
        )?;
        check(
            plus(self.dictionary_bytes() as u64, dictionary)?,
            limits.max_dictionary_bytes.map(|x| x as u64),
            L::DictionaryBytes,
        )?;
        check(
            plus(plus(self.retained_bytes() as u64, framing)?, dictionary)?,
            limits.max_workspace_bytes.map(|x| x as u64),
            L::WorkspaceBytes,
        )
    }
    fn reserve_pending(&mut self) -> Result<(), E> {
        if self.pending.len() == self.pending.capacity() {
            let target = self.pending.capacity().saturating_mul(2).max(16);
            self.budget(target, 0)?;
            self.pending
                .try_reserve_exact(target - self.pending.len())
                .map_err(|_| E::AllocationFailed)?;
        }
        Ok(())
    }
    fn location(&self, consumed: usize) -> FramedDecodeLocation {
        FramedDecodeLocation {
            offset: self.total_in.saturating_add(consumed as u64),
            chunk: self
                .layout
                .chunks
                .get(self.current)
                .and_then(|c| ::core::num::NonZeroU64::new(c.offset.0)),
            resource: self
                .resources
                .last()
                .and_then(|r| r.header.index.0.checked_add(1))
                .and_then(::core::num::NonZeroU64::new),
        }
    }
    fn byte(&self, input: &[u8], consumed: &mut usize) -> Result<Option<u8>, E> {
        let Some(&b) = input.get(*consumed) else {
            return Ok(None);
        };
        check(
            plus(
                self.total_in,
                consumed.checked_add(1).ok_or(E::SizeOverflow)?,
            )?,
            self.config.limits().max_input_bytes,
            L::InputBytes,
        )?;
        *consumed += 1;
        Ok(Some(b))
    }
    fn wait(&self) -> Result<Tag, E> {
        if self.final_end.is_some() {
            Err(E::UnexpectedEndOfInput)
        } else {
            Ok(Tag::Input)
        }
    }
    fn add_resource(&mut self, source: ResourceSource, hidden: bool) -> Result<(), E> {
        check(
            plus(self.resources.len() as u64, 1)?,
            self.config.limits().max_resources,
            L::Resources,
        )?;
        self.budget(
            if self.resources.len() == self.resources.capacity() {
                (self.resources.len() + 1)
                    .checked_mul(size_of::<Resource>())
                    .ok_or(E::SizeOverflow)?
            } else {
                0
            },
            0,
        )?;
        self.resources
            .try_reserve_exact(1)
            .map_err(|_| E::AllocationFailed)?;
        self.resources.push(Resource {
            header: ResourceHeader {
                index: ResourceIndex(self.resources.len() as u64),
                source,
                hidden,
            },
            size: 0,
            metadata: self.pending_meta.take(),
            footer: None,
            checksum: None,
            cache: self.cache.len()..self.cache.len(),
            complete: false,
        });
        Ok(())
    }
    fn validate_end(&self) -> Result<(), E> {
        if self.partial || self.pending_meta.is_some() || (self.codec_active && !self.codec_ended) {
            return Err(E::UnexpectedEndOfInput);
        }
        if let OutputSize::Exact(expected) = self.stream.output_size()
            && self.total_out != expected
        {
            return Err(E::DeclaredSizeMismatch {
                expected,
                actual: self.total_out,
            });
        }
        if self.repeat_count != 0 {
            let originals = self
                .metadata
                .iter()
                .filter(|m| m.original.is_none() && m.kind != MetadataKind::Global)
                .count();
            if originals != self.repeat_count || self.layout.directory.is_none() {
                return Err(E::InvalidDirectory);
            }
            // Codes have a fixed finite domain. Check each selected code once
            // per original/copy pair, regardless of custom-code multiplicity.
            let mut selected = [false; 678];
            for copy in self.metadata.iter().filter(|m| m.original.is_some()) {
                for field in copy.value.as_ref().ok_or(E::InvalidState)?.fields() {
                    selected[field_code(field.code)] = true;
                }
            }
            let originals = self
                .metadata
                .iter()
                .filter(|m| m.original.is_none() && m.kind != MetadataKind::Global);
            let copies = self.metadata.iter().filter(|m| m.original.is_some());
            for (original, copy) in originals.zip(copies) {
                let original = original.value.as_ref().ok_or(E::InvalidState)?;
                let copy = copy.value.as_ref().ok_or(E::InvalidState)?;
                for (code, &present) in selected.iter().enumerate() {
                    if present
                        && !original
                            .fields()
                            .filter(|f| field_code(f.code) == code)
                            .eq(copy.fields().filter(|f| field_code(f.code) == code))
                    {
                        return Err(E::InvalidMetadata);
                    }
                }
            }
        }
        Ok(())
    }
    fn order(&mut self) -> Result<(), E> {
        let c = &self.layout.chunks[self.current];
        let k = c.kind;
        if k == ChunkType::Padding {
            return Ok(());
        }
        let full = matches!(self.format, Some(StreamInfo::Framed(h)) if h.has_footer());
        if !full
            && !matches!(
                k,
                ChunkType::Data
                    | ChunkType::FirstPartial
                    | ChunkType::MiddlePartial
                    | ChunkType::LastPartial
            )
        {
            return Err(E::InvalidOrder);
        }
        if self.partial && !matches!(k, ChunkType::MiddlePartial | ChunkType::LastPartial) {
            return Err(E::InvalidOrder);
        }
        if self.pending_meta.is_some() && !matches!(k, ChunkType::Data | ChunkType::FirstPartial) {
            return Err(E::InvalidOrder);
        }
        if self.layout.directory.is_some() && k != ChunkType::Footer {
            return Err(E::InvalidOrder);
        }
        if self.repeat_start.is_some()
            && self.layout.directory.is_none()
            && !matches!(k, ChunkType::RepeatMetadata | ChunkType::CentralDirectory)
        {
            return Err(E::InvalidOrder);
        }
        match k {
            ChunkType::Data | ChunkType::FirstPartial => {
                if !full && !self.resources.is_empty() {
                    return Err(E::InvalidOrder);
                }
                let source = ResourceSource::Framed {
                    first_chunk: c.offset,
                };
                let hidden = c.flags & 1 != 0;
                self.add_resource(source, hidden)?;
                self.partial = k == ChunkType::FirstPartial;
            }
            ChunkType::MiddlePartial | ChunkType::LastPartial if !self.partial => {
                return Err(E::InvalidOrder);
            }
            ChunkType::LastPartial => self.partial = false,
            ChunkType::FooterMetadata
                if !matches!(
                    self.previous,
                    Some(ChunkType::Data | ChunkType::LastPartial)
                ) =>
            {
                return Err(E::InvalidOrder);
            }
            ChunkType::RepeatMetadata if self.repeat_start.is_none() => {
                self.repeat_start = Some(c.offset);
            }
            _ => {}
        }
        Ok(())
    }
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    fn prepare(&mut self, resolver: Option<DictionaryResolverRef<'_>>) -> Result<(), E> {
        let c = &self.layout.chunks[self.current];
        let Some(codec) = c.codec else { return Ok(()) };
        if codec == Codec::KeepDecoder {
            if !self.codec_active
                || self.codec_ended
                || (c.kind == ChunkType::RepeatMetadata
                    && self.previous != Some(ChunkType::RepeatMetadata))
            {
                return Err(E::InvalidChunk);
            }
            return Ok(());
        }
        if self.codec_active && !self.codec_ended {
            return Err(E::InvalidChunk);
        }
        self.codec_active = codec != Codec::Uncompressed;
        self.codec_ended = false;
        self.dictionary = None;
        if !self.codec_active {
            return Ok(());
        }
        let mut attachments = [DictionaryAttachment::Raw(&[]); 255];
        let mut serialized = 0;
        for (i, reference) in c.dictionaries.iter().enumerate() {
            let (bytes, is_serialized) = match *reference {
                DictionaryReference::PrefixId(id) | DictionaryReference::SerializedId(id) => {
                    let is_serialized = matches!(reference, DictionaryReference::SerializedId(_));
                    let request = ExternalDictionaryRequest {
                        id,
                        kind: if is_serialized {
                            ExternalDictionaryKind::Serialized
                        } else {
                            ExternalDictionaryKind::Prefix
                        },
                    };
                    let bytes =
                        resolver
                            .and_then(|r| r.resolve(request))
                            .ok_or(E::MissingDictionary {
                                request,
                                location: self.location(0),
                            })?;
                    (bytes, is_serialized)
                }
                DictionaryReference::PrefixChunk(offset)
                | DictionaryReference::PrefixResource(offset)
                | DictionaryReference::SerializedResource(offset) => {
                    if self.config.internal_dictionaries() == InternalDictionaryPolicy::Reject {
                        return Err(E::InternalDictionaryReferencesDisabled);
                    }
                    let target = self
                        .contents
                        .iter()
                        .find(|r| self.layout.chunks[r.chunk].offset.0 == offset)
                        .ok_or(E::InvalidDictionaryReference)?;
                    if c.kind == ChunkType::RepeatMetadata
                        && self.layout.chunks[target.chunk].kind != ChunkType::RepeatMetadata
                    {
                        return Err(E::InvalidDictionaryReference);
                    }
                    if matches!(reference, DictionaryReference::PrefixChunk(_)) {
                        if let Some(m) = target.metadata {
                            (
                                self.metadata[m]
                                    .value
                                    .as_ref()
                                    .ok_or(E::InvalidState)?
                                    .as_bytes(),
                                false,
                            )
                        } else {
                            (&self.cache[target.cache.clone()], false)
                        }
                    } else {
                        let r = target.resource.and_then(|r| self.resources.get(r)).filter(|r| r.complete && matches!(r.header.source, ResourceSource::Framed { first_chunk } if first_chunk.0 == offset)).ok_or(E::InvalidDictionaryReference)?;
                        (
                            &self.cache[r.cache.clone()],
                            matches!(reference, DictionaryReference::SerializedResource(_)),
                        )
                    }
                }
            };
            serialized += usize::from(is_serialized);
            if serialized > 1 {
                return Err(E::InvalidDictionaryReference);
            }
            attachments[i] = if is_serialized {
                DictionaryAttachment::Serialized(bytes)
            } else {
                DictionaryAttachment::Raw(bytes)
            };
        }
        let remaining_dict = self
            .config
            .limits()
            .max_dictionary_bytes
            .map(|n| n.saturating_sub(self.dictionary_bytes()));
        let remaining_work = self
            .config
            .limits()
            .max_workspace_bytes
            .map(|n| n.saturating_sub(self.retained_bytes()));
        let owned = match (remaining_dict, remaining_work) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        if !c.dictionaries.is_empty() {
            self.dictionary = Some(
                DecodeDictionary::new(
                    &attachments[..c.dictionaries.len()],
                    DecodeDictionaryLimits {
                        max_source_bytes: None,
                        max_owned_bytes: owned,
                    },
                )
                .map_err(|source| E::Dictionary {
                    source,
                    location: self.location(0),
                })?,
            );
        }
        let window = if codec == Codec::Brotli {
            WindowLimit::standard(self.config.window_limit().max_bits().min(24))
                .map_err(|_| E::InvalidState)?
        } else {
            self.config.window_limit()
        };
        self.codec_config = DecoderConfig::default().with_window_limit(window);
        self.codec.reset(self.codec_config);
        Ok(())
    }
    fn register(&mut self, metadata: Option<usize>) -> Result<(), E> {
        let c = &self.layout.chunks[self.current];
        if c.codec.is_none() {
            return Ok(());
        }
        self.budget(
            if self.contents.len() == self.contents.capacity() {
                (self.contents.len() + 1)
                    .checked_mul(size_of::<Content>())
                    .ok_or(E::SizeOverflow)?
            } else {
                0
            },
            0,
        )?;
        self.contents
            .try_reserve_exact(1)
            .map_err(|_| E::AllocationFailed)?;
        let resource = if is_data(c.kind) {
            Some(self.resources.len() - 1)
        } else {
            None
        };
        self.contents.push(Content {
            chunk: self.current,
            cache: self.cache_start..self.cache.len(),
            metadata,
            resource,
        });
        Ok(())
    }
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    fn finish_metadata(&mut self) -> Result<usize, E> {
        let c = &self.layout.chunks[self.current];
        let kind = match c.kind {
            ChunkType::Metadata => MetadataKind::Resource,
            ChunkType::FooterMetadata => MetadataKind::Footer,
            ChunkType::GlobalMetadata => MetadataKind::Global,
            _ => c.repeated_kind.ok_or(E::InvalidMetadata)?,
        };
        // At most one field per three bytes; bound the temporary index before allocation.
        let index_bytes = (self.scratch.len() / 3)
            .checked_mul(size_of::<([u8; 2], Range<usize>)>())
            .ok_or(E::SizeOverflow)?;
        self.budget(
            index_bytes
                .checked_add(if self.metadata.len() == self.metadata.capacity() {
                    (self.metadata.len() + 1)
                        .checked_mul(size_of::<Meta>())
                        .ok_or(E::SizeOverflow)?
                } else {
                    0
                })
                .ok_or(E::SizeOverflow)?,
            0,
        )?;
        let fields = wire::metadata(&self.scratch, kind)?;
        check(
            plus(self.metadata_fields, fields.len())?,
            self.config.limits().max_metadata_fields,
            L::MetadataFields,
        )?;
        let mut scope = match kind {
            MetadataKind::Resource => {
                MetadataScope::Resource(ResourceIndex(self.resources.len() as u64))
            }
            MetadataKind::Footer => MetadataScope::Footer(ResourceIndex(
                self.resources.len().checked_sub(1).ok_or(E::InvalidOrder)? as u64,
            )),
            MetadataKind::Global => MetadataScope::Global,
        };
        let mut original = None;
        if c.kind == ChunkType::RepeatMetadata {
            let original_index = self.metadata[self.original_cursor..]
                .iter()
                .position(|m| m.original.is_none() && m.kind != MetadataKind::Global)
                .map(|index| index + self.original_cursor)
                .ok_or(E::InvalidMetadata)?;
            let m = &self.metadata[original_index];
            if m.kind != kind {
                return Err(E::InvalidMetadata);
            }
            let value = m.value.as_ref().ok_or(E::InvalidState)?;
            let mut checked = [false; 678];
            for (code, _) in &fields {
                let index = field_code(*code);
                if checked[index] {
                    continue;
                }
                checked[index] = true;
                let a = fields
                    .iter()
                    .filter(|(k, _)| k == code)
                    .map(|(_, r)| &self.scratch[r.clone()]);
                let b = value.fields().filter(|f| f.code == *code).map(|f| f.value);
                if !a.eq(b) {
                    return Err(E::InvalidMetadata);
                }
            }
            original = Some(value.source);
            scope = m.scope;
            self.repeat_count += 1;
            self.original_cursor = original_index + 1;
        }
        self.metadata
            .try_reserve_exact(1)
            .map_err(|_| E::AllocationFailed)?;
        self.metadata_fields += fields.len() as u64;
        let i = self.metadata.len();
        self.metadata.push(Meta {
            value: Some(Metadata {
                source: c.offset,
                bytes: ::core::mem::take(&mut self.scratch),
                fields,
            }),
            scope,
            original,
            kind,
        });
        if original.is_none() {
            match kind {
                MetadataKind::Resource => self.pending_meta = Some(i),
                MetadataKind::Footer => {
                    self.resources.last_mut().ok_or(E::InvalidOrder)?.footer = Some(i)
                }
                MetadataKind::Global => {}
            }
        }
        self.register(Some(i))?;
        Ok(i)
    }
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    fn payload(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        consumed: &mut usize,
        produced: &mut usize,
        backend: Backend,
    ) -> Result<Option<Tag>, E> {
        let raw = self.format == Some(StreamInfo::Raw);
        let data = raw || is_data(self.layout.chunks[self.current].kind);
        let position = ResourceDataPosition {
            resource: self
                .resources
                .last()
                .map_or(ResourceIndex(0), |r| r.header.index),
            chunk: if raw {
                None
            } else {
                Some(self.layout.chunks[self.current].offset)
            },
            offset: self.resources.last().map_or(0, |r| r.size),
        };
        let mut metadata_buffer = [0; 4096];
        let destination = if data { output } else { &mut metadata_buffer };
        let limits = self.config.limits();
        let mut allowance = u64::MAX - self.chunk_out;
        let mut limiting = L::DecodedBytes;
        let mut ceiling = u64::MAX;
        let budgets = [
            (
                limits.max_decoded_bytes,
                self.total_decoded,
                L::DecodedBytes,
            ),
            (
                if data {
                    limits.max_output_bytes
                } else {
                    limits.max_metadata_bytes
                },
                if data {
                    self.total_out
                } else {
                    self.metadata_bytes
                },
                if data {
                    L::OutputBytes
                } else {
                    L::MetadataBytes
                },
            ),
            (
                if data {
                    limits.max_resource_bytes
                } else {
                    None
                },
                position.offset,
                L::ResourceBytes,
            ),
        ];
        for (limit, used, kind) in budgets {
            if let Some(limit) = limit
                && limit.saturating_sub(used) < allowance
            {
                allowance = limit.saturating_sub(used);
                limiting = kind;
                ceiling = limit;
            }
        }
        if data && let OutputSize::Exact(expected) = self.stream.output_size() {
            allowance = allowance.min(expected.saturating_sub(self.total_out));
        }
        let declared = if raw {
            None
        } else {
            self.layout.chunks[self.current].declared_size
        };
        let codec = if raw {
            Codec::Brotli
        } else {
            self.layout.chunks[self.current]
                .codec
                .ok_or(E::InvalidState)?
        };
        let available = if raw {
            input.len() - *consumed
        } else {
            usize::try_from(self.chunk_left.min((input.len() - *consumed) as u64))
                .map_err(|_| E::SizeOverflow)?
        };
        let mut n = 0;
        let outcome;
        if codec == Codec::Uncompressed {
            if self.chunk_left == 0 {
                self.phase = Phase::Semantic;
                return Ok(None);
            }
            if allowance == 0 {
                if data
                    && let OutputSize::Exact(expected) = self.stream.output_size()
                    && self.total_out >= expected
                {
                    return Err(E::DeclaredSizeMismatch {
                        expected,
                        actual: self.total_out.checked_add(1).ok_or(E::SizeOverflow)?,
                    });
                }
                return Err(E::LimitExceeded {
                    kind: limiting,
                    limit: ceiling,
                });
            }
            if destination.is_empty() {
                return Ok(Some(Tag::Output));
            }
            if available == 0 {
                return self.wait().map(Some);
            }
            n = available
                .min(destination.len())
                .min(usize::try_from(allowance).unwrap_or(usize::MAX));
            if let Some(limit) = limits.max_input_bytes {
                n = n.min(
                    usize::try_from(limit.saturating_sub(self.total_in + *consumed as u64))
                        .unwrap_or(usize::MAX),
                );
            }
            if n == 0 {
                return Err(E::LimitExceeded {
                    kind: L::InputBytes,
                    limit: limits.max_input_bytes.unwrap_or(0),
                });
            }
            destination[..n].copy_from_slice(&input[*consumed..*consumed + n]);
            *consumed += n;
            self.chunk_left -= n as u64;
            outcome = Ok(if self.chunk_left == 0 {
                Stop::Input
            } else {
                Stop::Output
            });
        } else {
            let owned_other = self
                .framing_bytes()
                .checked_add(self.dictionary_bytes())
                .ok_or(E::SizeOverflow)?;
            let workspace = limits
                .max_workspace_bytes
                .map(|n| n.saturating_sub(owned_other));
            self.codec.set_framed_workspace_limit(workspace);
            let mut source = Input::new(
                &input[*consumed..*consumed + available],
                self.total_in + *consumed as u64,
                limits.max_input_bytes,
            );
            let mut target = Output {
                bytes: destination,
                collect: None,
                linear: false,
                produced: 0,
                total_before: self.chunk_out,
                limit: Some(
                    self.chunk_out
                        .checked_add(allowance)
                        .ok_or(E::SizeOverflow)?,
                ),
                // A content boundary may split a meta-block. The raw exact-size
                // guard runs before discovering that another input byte is needed;
                // validate actual regenerated content here instead.
                exact: OutputSize::Unknown,
            };
            outcome = self.codec.run(
                backend,
                &mut source,
                &mut target,
                self.codec_config,
                self.dictionary.as_ref().map(Into::into),
            );
            *consumed += source.consumed;
            if !raw {
                self.chunk_left -= source.consumed as u64;
            }
            n += target.produced;
        }
        self.chunk_out = plus(self.chunk_out, n)?;
        self.total_decoded = plus(self.total_decoded, n)?;
        if data {
            *produced = n;
            self.total_out = plus(self.total_out, n)?;
            let r = self.resources.last_mut().ok_or(E::InvalidState)?;
            r.size = plus(r.size, n)?;
            if !raw
                && self.config.internal_dictionaries() == InternalDictionaryPolicy::Retain
                && n != 0
            {
                if self.cache.len().checked_add(n).ok_or(E::SizeOverflow)? > self.cache.capacity() {
                    let target = self
                        .cache
                        .len()
                        .checked_add(n)
                        .ok_or(E::SizeOverflow)?
                        .max(self.cache.capacity().saturating_mul(2));
                    self.budget(0, target)?;
                    self.cache
                        .try_reserve_exact(target - self.cache.len())
                        .map_err(|_| E::AllocationFailed)?;
                }
                self.cache.extend_from_slice(&destination[..n]);
                self.resources.last_mut().ok_or(E::InvalidState)?.cache.end = self.cache.len();
            }
        } else {
            self.metadata_bytes = plus(self.metadata_bytes, n)?;
            if self.scratch.len().checked_add(n).ok_or(E::SizeOverflow)? > self.scratch.capacity() {
                let target = self
                    .scratch
                    .len()
                    .checked_add(n)
                    .ok_or(E::SizeOverflow)?
                    .max(self.scratch.capacity().saturating_mul(2));
                self.budget(target, 0)?;
                self.scratch
                    .try_reserve_exact(target - self.scratch.len())
                    .map_err(|_| E::AllocationFailed)?;
            }
            self.scratch.extend_from_slice(&destination[..n]);
        }
        let stop = outcome.map_err(|source| match source {
            crate::DecodeError::InputLimitExceeded { limit } => E::LimitExceeded {
                kind: L::InputBytes,
                limit,
            },
            crate::DecodeError::OutputLimitExceeded { .. } => {
                if data
                    && let OutputSize::Exact(expected) = self.stream.output_size()
                    && self.total_out >= expected
                {
                    return E::DeclaredSizeMismatch {
                        expected,
                        actual: self.total_out.saturating_add(1),
                    };
                }
                E::LimitExceeded {
                    kind: limiting,
                    limit: ceiling,
                }
            }
            crate::DecodeError::OutputSizeMismatch { expected, actual } => {
                E::DeclaredSizeMismatch { expected, actual }
            }
            crate::DecodeError::MemoryLimitExceeded { .. } => E::LimitExceeded {
                kind: L::WorkspaceBytes,
                limit: limits.max_workspace_bytes.unwrap_or(usize::MAX) as u64,
            },
            source => E::Decode {
                source,
                location: self.location(*consumed),
            },
        })?;
        if let Some(expected) = declared
            && self.chunk_out > expected
        {
            return Err(E::DeclaredSizeMismatch {
                expected,
                actual: self.chunk_out,
            });
        }
        if stop == Stop::Member {
            self.codec_ended = true;
        }
        if raw {
            if stop == Stop::Member {
                self.phase = Phase::ResourceEnd;
            } else if stop == Stop::Input && self.final_end.is_some() {
                return Err(E::UnexpectedEndOfInput);
            }
        } else if self.chunk_left == 0 && stop != Stop::Output {
            if let Some(expected) = declared
                && expected != self.chunk_out
            {
                return Err(E::DeclaredSizeMismatch {
                    expected,
                    actual: self.chunk_out,
                });
            }
            self.phase = Phase::Semantic;
        } else if stop == Stop::Member {
            return Err(E::InvalidChunk);
        }
        if data && n != 0 {
            return Ok(Some(Tag::Data(position)));
        }
        if matches!(self.phase, Phase::Semantic | Phase::ResourceEnd) {
            return Ok(None);
        }
        if stop == Stop::Output {
            if data {
                Ok(Some(Tag::Output))
            } else {
                Ok(None)
            }
        } else {
            self.wait().map(Some)
        }
    }

    fn directory(&mut self, input: &[u8], consumed: &mut usize) -> Result<Option<Tag>, E> {
        if !self.directory_pointer {
            let mut p = 0;
            if let Some(pointer) = wire::number(&self.pending, &mut p)? {
                let repeated = if pointer == 0 {
                    None
                } else {
                    Some(ChunkOffset(pointer))
                };
                if repeated != self.repeat_start {
                    return Err(E::InvalidDirectory);
                }
                self.layout.directory = Some(CentralDirectory {
                    offset: self.layout.chunks[self.current].offset,
                    repeated_metadata: repeated,
                    entries: Vec::new(),
                });
                self.pending.clear();
                self.directory_pointer = true;
                return Ok(Some(Tag::DirectoryStart));
            }
        } else if self.pending.is_empty() && self.chunk_left == 0 {
            if self.layout.chunks[self.directory_cursor..self.current]
                .iter()
                .any(|c| c.codec.is_some())
            {
                return Err(E::InvalidDirectory);
            }
            self.phase = Phase::DirectoryEnd;
            return Ok(None);
        } else {
            let mut p = 0;
            if let Some(offset) = wire::number(&self.pending, &mut p)?
                && let Some(length) = wire::number(&self.pending, &mut p)?
            {
                let expected = self.layout.chunks[self.directory_cursor..self.current]
                    .iter()
                    .position(|c| c.codec.is_some())
                    .map(|i| i + self.directory_cursor)
                    .ok_or(E::InvalidDirectory)?;
                let c = &self.layout.chunks[expected];
                if offset != c.offset.0 || length != c.header_bytes.len() as u64 {
                    return Err(E::InvalidDirectory);
                }
                let bytes = &self.pending[p..];
                if bytes.len() > c.header_bytes.len() || bytes != &c.header_bytes[..bytes.len()] {
                    return Err(E::InvalidDirectory);
                }
                if bytes.len() == c.header_bytes.len() {
                    let entries = &self
                        .layout
                        .directory
                        .as_ref()
                        .ok_or(E::InvalidState)?
                        .entries;
                    self.budget(
                        if entries.len() == entries.capacity() {
                            (entries.len() + 1)
                                .checked_mul(size_of::<DirectoryEntry>())
                                .ok_or(E::SizeOverflow)?
                        } else {
                            0
                        },
                        0,
                    )?;
                    let directory = self.layout.directory.as_mut().ok_or(E::InvalidState)?;
                    directory
                        .entries
                        .try_reserve_exact(1)
                        .map_err(|_| E::AllocationFailed)?;
                    let i = directory.entries.len();
                    directory.entries.push(DirectoryEntry {
                        chunk_index: expected,
                    });
                    self.directory_cursor = expected + 1;
                    self.pending.clear();
                    return Ok(Some(Tag::DirectoryEntry(i)));
                }
            }
        }
        if self.chunk_left == 0 {
            return Err(E::InvalidDirectory);
        }
        self.reserve_pending()?;
        let Some(b) = self.byte(input, consumed)? else {
            return self.wait().map(Some);
        };
        self.pending.push(b);
        self.chunk_left -= 1;
        Ok(None)
    }

    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    fn drive(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        consumed: &mut usize,
        produced: &mut usize,
        backend: Backend,
        resolver: Option<DictionaryResolverRef<'_>>,
    ) -> Result<Tag, E> {
        loop {
            match self.phase {
                Phase::Detect => {
                    if self.pending.is_empty() {
                        let Some(&first) = input.get(*consumed) else {
                            return self.wait();
                        };
                        if first != 0x91 {
                            if self.config.input_mode() != InputMode::Auto {
                                return Err(E::InvalidSignature);
                            }
                            self.format = Some(StreamInfo::Raw);
                            self.add_resource(ResourceSource::Raw, false)?;
                            self.codec_config = DecoderConfig::default()
                                .with_window_limit(self.config.window_limit());
                            self.codec.reset(self.codec_config);
                            self.codec_active = true;
                            self.phase = Phase::Begin;
                            return Ok(Tag::Stream);
                        }
                    }
                    self.reserve_pending()?;
                    let Some(b) = self.byte(input, consumed)? else {
                        return self.wait();
                    };
                    let i = self.pending.len();
                    if i == 1 && b == 0 {
                        return Err(E::UnexpectedInputKind(
                            UnexpectedInputKind::SerializedDictionary,
                        ));
                    }
                    if i < 4 && b != [0x91, 0x0a, 0x42, 0x52][i] {
                        return Err(E::InvalidSignature);
                    }
                    self.pending.push(b);
                    if i == 4 {
                        if b & 3 != 0 {
                            return Err(E::UnsupportedVersion(b & 3));
                        }
                        self.format = Some(StreamInfo::Framed(ContainerHeader { flags: b }));
                        self.pending.clear();
                        self.phase = Phase::Header;
                        return Ok(Tag::Stream);
                    }
                }
                Phase::Header => {
                    if self.pending.is_empty() && *consumed == input.len() {
                        if self.final_end.is_some()
                            && matches!(self.format, Some(StreamInfo::Framed(h)) if !h.has_footer())
                        {
                            if self.resources.len() != 1 {
                                return Err(E::InvalidOrder);
                            }
                            self.validate_end()?;
                            self.phase = Phase::Finish;
                            continue;
                        }
                        return self.wait();
                    }
                    self.reserve_pending()?;
                    let Some(b) = self.byte(input, consumed)? else {
                        return self.wait();
                    };
                    self.pending.push(b);
                    let offset = self.total_in + *consumed as u64 - self.pending.len() as u64;
                    if let Some(mut c) = wire::header(&self.pending, offset, |bytes| {
                        self.budget(
                            bytes
                                .checked_add(
                                    if self.layout.chunks.len() == self.layout.chunks.capacity() {
                                        (self.layout.chunks.len() + 1)
                                            .checked_mul(size_of::<ChunkInfo>())
                                            .ok_or(E::SizeOverflow)?
                                    } else {
                                        0
                                    },
                                )
                                .ok_or(E::SizeOverflow)?,
                            0,
                        )
                    })? {
                        check(
                            plus(self.layout.chunks.len() as u64, 1)?,
                            self.config.limits().max_chunks,
                            L::Chunks,
                        )?;
                        c.offset.0.checked_add(c.length).ok_or(E::SizeOverflow)?;
                        self.chunk_left = c.length - self.pending.len() as u64;
                        self.chunk_out = 0;
                        self.cache_start = self.cache.len();
                        c.header_bytes = ::core::mem::take(&mut self.pending);
                        self.layout
                            .chunks
                            .try_reserve_exact(1)
                            .map_err(|_| E::AllocationFailed)?;
                        self.current = self.layout.chunks.len();
                        self.layout.chunks.push(c);
                        self.order()?;
                        self.phase = Phase::Begin;
                        return Ok(Tag::Header);
                    }
                }
                Phase::Begin => {
                    self.phase = Phase::Content;
                    if self.format == Some(StreamInfo::Raw) {
                        return Ok(Tag::Start);
                    }
                    self.prepare(resolver)?;
                    let k = self.layout.chunks[self.current].kind;
                    if k == ChunkType::CentralDirectory {
                        self.phase = Phase::Directory;
                        self.directory_pointer = false;
                    }
                    if matches!(k, ChunkType::Data | ChunkType::FirstPartial) {
                        return Ok(Tag::Start);
                    }
                }
                Phase::Content => {
                    let raw = self.format == Some(StreamInfo::Raw);
                    if raw || self.layout.chunks[self.current].codec.is_some() {
                        if let Some(tag) =
                            self.payload(input, output, consumed, produced, backend)?
                        {
                            return Ok(tag);
                        }
                    } else {
                        let k = self.layout.chunks[self.current].kind;
                        if self.chunk_left == 0 {
                            self.phase = Phase::Semantic;
                            continue;
                        }
                        if k == ChunkType::Footer
                            && self.chunk_left + self.pending.len() as u64 > 18
                        {
                            return Err(E::InvalidFooter);
                        }
                        if k == ChunkType::Footer {
                            self.reserve_pending()?;
                        }
                        let Some(b) = self.byte(input, consumed)? else {
                            return self.wait();
                        };
                        self.chunk_left -= 1;
                        if k == ChunkType::Padding {
                            if b != 0 {
                                return Err(E::InvalidChunk);
                            }
                        } else {
                            self.pending.push(b);
                        }
                    }
                }
                Phase::Semantic => {
                    let c = &self.layout.chunks[self.current];
                    self.phase = Phase::ChunkEnd;
                    match c.kind {
                        ChunkType::Padding => return Ok(Tag::Padding),
                        ChunkType::Metadata
                        | ChunkType::FooterMetadata
                        | ChunkType::GlobalMetadata
                        | ChunkType::RepeatMetadata => {
                            return self.finish_metadata().map(Tag::Metadata);
                        }
                        ChunkType::Footer => {
                            self.pending.reverse();
                            let mut p = 0;
                            let directory =
                                wire::number(&self.pending, &mut p)?.ok_or(E::InvalidFooter)?;
                            let size =
                                wire::number(&self.pending, &mut p)?.ok_or(E::InvalidFooter)?;
                            let end = c.offset.0.checked_add(c.length).ok_or(E::SizeOverflow)?;
                            if p != self.pending.len()
                                || (size != 0 && size != end)
                                || directory
                                    != self.layout.directory.as_ref().map_or(0, |d| d.offset.0)
                            {
                                return Err(E::InvalidFooter);
                            }
                            self.validate_end()?;
                            self.layout.footer = Some(ContainerFooter {
                                offset: c.offset,
                                file_size: (size != 0).then_some(size),
                                directory: (directory != 0).then_some(ChunkOffset(directory)),
                            });
                            self.pending.clear();
                            return Ok(Tag::Footer);
                        }
                        _ => self.register(None)?,
                    }
                }
                Phase::ChunkEnd => {
                    let c = &self.layout.chunks[self.current];
                    self.phase = if matches!(c.kind, ChunkType::Data | ChunkType::LastPartial) {
                        Phase::ResourceEnd
                    } else if c.kind == ChunkType::Footer {
                        Phase::Finish
                    } else {
                        Phase::Header
                    };
                    if c.kind != ChunkType::Padding {
                        self.previous = Some(c.kind);
                    }
                    return Ok(Tag::ChunkEnd);
                }
                Phase::ResourceEnd => {
                    let r = self.resources.last_mut().ok_or(E::InvalidState)?;
                    r.complete = true;
                    if self.format != Some(StreamInfo::Raw) {
                        r.checksum = self.layout.chunks[self.current].checksum;
                    }
                    self.completed += 1;
                    self.phase = if self.format == Some(StreamInfo::Raw) {
                        Phase::Finish
                    } else {
                        Phase::Header
                    };
                    return Ok(Tag::End);
                }
                Phase::Directory => {
                    if let Some(tag) = self.directory(input, consumed)? {
                        return Ok(tag);
                    }
                }
                Phase::DirectoryEnd => {
                    self.phase = Phase::ChunkEnd;
                    return Ok(Tag::DirectoryEnd);
                }
                Phase::Finish => {
                    self.validate_end()?;
                    self.finished = true;
                    return Ok(Tag::Finished);
                }
                Phase::Failed => return Err(E::InvalidState),
            }
        }
    }
    // The by-value failure contract preserves progress even when allocation fails;
    // boxing this 128-byte record would require an allocation on the error path.
    #[expect(
        clippy::result_large_err,
        reason = "allocation-independent progress is part of the public contract"
    )]
    pub fn process(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: DecodeOperation,
        backend: Backend,
        resolver: Option<DictionaryResolverRef<'_>>,
    ) -> Result<(usize, usize, Tag), FramedDecodeFailure> {
        let mut consumed = 0;
        let mut produced = 0;
        let position = ResourceDataPosition {
            resource: self
                .resources
                .last()
                .map_or(ResourceIndex(0), |r| r.header.index),
            chunk: if self.format == Some(StreamInfo::Raw) {
                None
            } else {
                self.layout.chunks.get(self.current).map(|c| c.offset)
            },
            offset: self.resources.last().map_or(0, |r| r.size),
        };
        let result = (|| {
            if matches!(self.phase, Phase::Failed) {
                return Err(E::InvalidState);
            }
            if self.finished {
                return Ok(Tag::Finished);
            }
            let end = plus(self.total_in, input.len())?;
            if let Some(final_end) = self.final_end {
                if operation != DecodeOperation::Finish || end != final_end {
                    return Err(E::InvalidState);
                }
            } else if operation == DecodeOperation::Finish {
                self.final_end = Some(end);
            }
            self.drive(
                input,
                output,
                &mut consumed,
                &mut produced,
                backend,
                resolver,
            )
        })();
        self.total_in += consumed as u64;
        match result {
            Ok(tag) => Ok((consumed, produced, tag)),
            Err(error) => {
                self.phase = Phase::Failed;
                Err(FramedDecodeFailure {
                    error,
                    consumed,
                    produced,
                    last_output: (produced != 0).then_some(ProducedFragment {
                        range: 0..produced,
                        position,
                    }),
                })
            }
        }
    }
    pub fn event<'a>(&'a self, tag: Tag, output: &'a [u8]) -> Option<FramedEvent<'a>> {
        Some(match tag {
            Tag::Input | Tag::Output | Tag::Finished => return None,
            Tag::Stream => FramedEvent::StreamStart(self.format?),
            Tag::Header => FramedEvent::ChunkHeader(&self.layout.chunks[self.current]),
            Tag::ChunkEnd => FramedEvent::ChunkEnd(ChunkSummary {
                offset: self.layout.chunks[self.current].offset,
                decoded_size: self.chunk_out,
            }),
            Tag::Start => FramedEvent::ResourceStart(self.resources.last()?.header),
            Tag::Data(position) => FramedEvent::ResourceData(ResourceData {
                position,
                bytes: output,
            }),
            Tag::End => {
                let r = self.resources.last()?;
                FramedEvent::ResourceDataEnd(ResourceSummary {
                    index: r.header.index,
                    size: r.size,
                    checksum: r.checksum,
                })
            }
            Tag::Metadata(i) => {
                let m = &self.metadata[i];
                FramedEvent::Metadata(MetadataEvent {
                    scope: m.scope,
                    original: m.original,
                    metadata: m.value.as_ref()?,
                })
            }
            Tag::DirectoryStart => {
                let d = self.layout.directory.as_ref()?;
                FramedEvent::DirectoryStart(DirectoryHeader {
                    offset: d.offset,
                    repeated_metadata: d.repeated_metadata,
                })
            }
            Tag::DirectoryEntry(i) => {
                FramedEvent::DirectoryEntry(&self.layout.directory.as_ref()?.entries[i])
            }
            Tag::DirectoryEnd => FramedEvent::DirectoryEnd,
            Tag::Padding => {
                let c = &self.layout.chunks[self.current];
                FramedEvent::Padding(PaddingInfo {
                    offset: c.offset,
                    length: c.length,
                })
            }
            Tag::Footer => FramedEvent::Footer(self.layout.footer?),
        })
    }
    pub fn reserve_collection<T>(&mut self, buffer: &mut Vec<T>) -> Result<(), E> {
        if buffer.len() == buffer.capacity() {
            let capacity = buffer.len().checked_add(1).ok_or(E::SizeOverflow)?;
            let replacement = capacity
                .checked_mul(size_of::<T>())
                .ok_or(E::SizeOverflow)?;
            self.budget(replacement, 0)?;
            let previous = buffer.capacity();
            buffer
                .try_reserve_exact(1)
                .map_err(|_| E::AllocationFailed)?;
            self.collector_bytes = self
                .collector_bytes
                .checked_add((buffer.capacity() - previous) * size_of::<T>())
                .ok_or(E::SizeOverflow)?;
        }
        Ok(())
    }
    pub fn result<B>(&mut self, data: Vec<B>) -> Result<FramedOutput<B>, E> {
        if !self.finished || data.len() != self.resources.len() {
            return Err(E::InvalidState);
        }
        let extra = data
            .len()
            .checked_mul(size_of::<DecodedResource<B>>())
            .and_then(|n| n.checked_add(self.metadata.len() * size_of::<Metadata>()))
            .and_then(|n| n.checked_add(self.repeat_count * size_of::<RepeatedMetadata>()))
            .ok_or(E::SizeOverflow)?;
        self.budget(extra, 0)?;
        let mut resources = Vec::new();
        let mut global_metadata = Vec::new();
        resources
            .try_reserve_exact(data.len())
            .map_err(|_| E::AllocationFailed)?;
        global_metadata
            .try_reserve_exact(self.metadata.len())
            .map_err(|_| E::AllocationFailed)?;
        self.layout
            .repeated_metadata
            .try_reserve_exact(self.repeat_count)
            .map_err(|_| E::AllocationFailed)?;
        for (r, data) in self.resources.iter().zip(data) {
            resources.push(DecodedResource {
                index: r.header.index,
                source: r.header.source,
                hidden: r.header.hidden,
                metadata: r.metadata.and_then(|i| self.metadata[i].value.take()),
                data,
                footer_metadata: r.footer.and_then(|i| self.metadata[i].value.take()),
                checksum: r.checksum,
            });
        }
        for m in &mut self.metadata {
            if let Some(value) = m.value.take() {
                if let Some(original) = m.original {
                    self.layout.repeated_metadata.push(RepeatedMetadata {
                        original,
                        kind: m.kind,
                        metadata: value,
                    });
                } else {
                    global_metadata.push(value);
                }
            }
        }
        Ok(FramedOutput {
            structure: match self.format.ok_or(E::InvalidState)? {
                StreamInfo::Raw => OutputStructure::Raw,
                StreamInfo::Framed(header) => OutputStructure::Framed {
                    header,
                    layout: ::core::mem::take(&mut self.layout),
                },
            },
            resources,
            global_metadata,
        })
    }
}
// Metadata has already validated these codes before indexing this fixed domain.
fn field_code(code: [u8; 2]) -> usize {
    match &code {
        b"id" => 676,
        b"mt" => 677,
        _ => usize::from(code[0] - b'A') * 26 + usize::from(code[1] - b'A'),
    }
}
fn is_data(kind: ChunkType) -> bool {
    matches!(
        kind,
        ChunkType::Data
            | ChunkType::FirstPartial
            | ChunkType::MiddlePartial
            | ChunkType::LastPartial
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_and_host_backends_decode_the_same_framed_resource() {
        let bytes = [
            0x91, 10, 66, 82, 0, 13, 2, 2, 5, 0, 0x0b, 2, 0x80, b'h', b'e', b'l', b'l', b'o', 3,
        ];
        let mut baseline = FramedDecompressor::builder(FramedDecodeConfig::default())
            .with_backend(Backend::SCALAR)
            .build()
            .unwrap();
        let expected = baseline.decompress(&bytes).unwrap();
        for backend in Backend::available() {
            let mut d = FramedDecompressor::builder(FramedDecodeConfig::default())
                .with_backend(backend)
                .build()
                .unwrap();
            assert_eq!(d.decompress(&bytes).unwrap(), expected);
        }
    }
}

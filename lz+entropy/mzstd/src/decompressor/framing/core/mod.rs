//! Bounded container state and event transitions; no caller borrows are stored.
//!
//! The engine walks the container frame by frame. Chunk headers and
//! structural content are parsed byte by byte into bounded staging; every
//! compressed chunk content is one Zstandard frame, decoded by the
//! decoder's own streaming [`Stream`] in single-frame mode with the chunk's
//! dictionary and an output ceiling fixed when the chunk begins.

#[cfg(not(feature = "no_std"))]
pub(in crate::decompressor::framing) mod seek;
mod wire;

use super::*;
use crate::decompressor::core::Stream;
use crate::dictionary::DictionaryContent;
use crate::dictionary::decode::{DecodeParts, parse_decode_parts};
use crate::framing::core::{
    CHUNK_MAGIC, CONTAINER_HEADER, FRAME_HEADER, SEEK_DESCRIPTOR_MASK, SEEK_ENTRY, SEEK_FOOTER,
    SEEK_TABLE_MAGIC, SEEKABLE_MAGIC, SIGNATURE, VERSION_MASK,
};
use crate::framing::{DictionaryReference, MetadataKind};
use crate::{DecodeError, DecodeLimits, DecodeOperation, DecoderConfig, DecoderStatus, FrameMode};
use ::core::ops::Range;
use FramedDecodeError as E;
use FramedLimitKind as L;
use alloc::vec::Vec;

/// First four bytes of a Zstandard-format dictionary.
const DICTIONARY_MAGIC: u32 = 0xEC30_A437;
/// Footer content after its type byte: directory offset and file size.
const FOOTER_FIELDS: usize = 16;

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
    SeekTable,
}

#[derive(Debug, Clone, Copy)]
enum Phase {
    Detect,
    Header,
    ChunkHeader,
    Begin,
    Content,
    Semantic,
    ChunkEnd,
    ResourceEnd,
    Directory,
    DirectoryEnd,
    SeekTable,
    Finish,
    Failed,
}

/// Where the seek table's verification stands.
#[derive(Debug, Clone, Copy, Default)]
struct SeekCursor {
    /// The size field has been read.
    sized: bool,
    /// Entries still to read.
    entries: u64,
    /// The container header frame's entry has been checked.
    header: bool,
    /// Chunk whose frames come next.
    chunk: usize,
    /// Its body frame comes next.
    body: bool,
}

/// What bounds the decoded size of the current content.
#[derive(Debug, Clone, Copy)]
enum Ceiling {
    /// No ceiling.
    None,
    /// A configured budget.
    Budget(L, u64),
    /// The caller's exact payload contract.
    Exact(u64),
    /// The chunk's declared size.
    Declared(u64),
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
    /// Its whole payload is in the cache for later references.
    retained: bool,
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
    /// Its whole content is in the cache.
    retained: bool,
    metadata: Option<usize>,
    resource: Option<usize>,
}

/// One resolved dictionary source.
enum Source<'a> {
    External(&'a [u8]),
    Cache(Range<usize>),
    Metadata(usize),
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
    /// Owned bytes of the current chunk's dictionary.
    dictionary: Vec<u8>,
    /// Its parsed ID and entropy tables.
    dictionary_parts: Option<DecodeParts>,
    /// The references it was built from, to reuse it for the next chunk.
    dictionary_refs: Vec<DictionaryReference>,
    /// The current content is decoded with the dictionary.
    dictionary_active: bool,
    /// The current content is a Zstandard frame (or raw input).
    codec_active: bool,
    /// Raw input completed its last frame.
    codec_ended: bool,
    /// Ceiling on the current content's decoded size.
    limit: u64,
    ceiling: Ceiling,
    /// Detection bytes still to feed to the raw decoder.
    replay: usize,
    current: usize,
    chunk_left: u64,
    chunk_out: u64,
    cache_start: usize,
    previous: Option<ChunkType>,
    partial: bool,
    hidden: bool,
    pending_meta: Option<usize>,
    repeat_count: usize,
    original_cursor: usize,
    repeat_start: Option<ChunkOffset>,
    metadata_bytes: u64,
    metadata_fields: u64,
    directory_cursor: usize,
    directory_pointer: bool,
    seek: SeekCursor,
    /// Offset of the seek table's frame.
    seek_offset: u64,
    /// The cache still grows; it stops for good once its budget is spent.
    retaining: bool,
    pub collector_bytes: usize,
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
            dictionary: Vec::new(),
            dictionary_parts: None,
            dictionary_refs: Vec::new(),
            dictionary_active: false,
            codec_active: false,
            codec_ended: false,
            limit: u64::MAX,
            ceiling: Ceiling::None,
            replay: 0,
            current: 0,
            chunk_left: 0,
            chunk_out: 0,
            cache_start: 0,
            previous: None,
            partial: false,
            hidden: false,
            pending_meta: None,
            repeat_count: 0,
            original_cursor: 0,
            repeat_start: None,
            metadata_bytes: 0,
            metadata_fields: 0,
            directory_cursor: 0,
            directory_pointer: false,
            seek: SeekCursor::default(),
            seek_offset: 0,
            retaining: true,
            collector_bytes: 0,
        }
    }

    pub fn clear(&mut self) {
        let backend = self.codec.backend();
        let mut codec = ::core::mem::replace(
            &mut self.codec,
            Stream::new(DecoderConfig::DEFAULT, backend),
        );
        codec.reset(DecoderConfig::DEFAULT);
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
        self.dictionary.clear();
        next.dictionary = ::core::mem::take(&mut self.dictionary);
        self.dictionary_refs.clear();
        next.dictionary_refs = ::core::mem::take(&mut self.dictionary_refs);
        *self = next;
    }

    /// Makes every later call report `InvalidState`.
    pub const fn poison(&mut self) {
        self.phase = Phase::Failed;
    }

    pub fn fits_policy(&self) -> bool {
        self.budget(0, 0).is_ok()
    }

    pub(super) fn framing_bytes(&self) -> usize {
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
            + self.dictionary.capacity()
            + self.dictionary_refs.capacity() * size_of::<DictionaryReference>()
            + self
                .dictionary_parts
                .as_ref()
                .map_or(0, DecodeParts::retained_bytes)
    }

    pub fn retained_bytes(&self) -> usize {
        self.framing_bytes() + self.dictionary_bytes() + self.codec.retained_bytes()
    }

    pub(super) fn budget(&self, framing: usize, dictionary: usize) -> Result<(), E> {
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
            retained: self.retaining,
        });
        Ok(())
    }

    fn validate_end(&self) -> Result<(), E> {
        let raw = self.format == Some(StreamInfo::Raw);
        if self.partial || self.pending_meta.is_some() || (raw && !self.codec_ended) {
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
        if self.layout.footer.is_some() {
            return Err(E::InvalidOrder);
        }
        let c = &self.layout.chunks[self.current];
        let k = c.kind;
        if k == ChunkType::Padding {
            return Ok(());
        }
        let full = matches!(self.format, Some(StreamInfo::Framed(h)) if h.has_footer());
        if !full && !is_data(k) {
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
        let detached = c.body_length.is_some();
        match k {
            ChunkType::Data | ChunkType::FirstPartial => {
                if !full && !self.resources.is_empty() {
                    return Err(E::InvalidOrder);
                }
                let source = ResourceSource::Framed {
                    first_chunk: c.offset,
                };
                let hidden = c.flags & crate::framing::core::FLAG_HIDDEN != 0;
                self.add_resource(source, hidden)?;
                self.hidden = hidden;
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
        // Visible payload is always a top-level Zstandard frame; hidden
        // payload always stays inside its skippable frame.
        if is_data(k) && detached == self.hidden {
            return Err(E::InvalidChunk);
        }
        Ok(())
    }

    /// Whether the seek table may follow what has been read.
    fn seek_order(&self) -> Result<(), E> {
        if self.partial || self.pending_meta.is_some() {
            return Err(E::InvalidOrder);
        }
        let full = matches!(self.format, Some(StreamInfo::Framed(h)) if h.has_footer());
        if (full && self.layout.footer.is_none()) || (!full && self.resources.len() != 1) {
            return Err(E::InvalidOrder);
        }
        Ok(())
    }

    /// The tightest ceiling on content about to be decoded.
    fn ceiling(&self, data: bool, declared: Option<u64>) -> (u64, Ceiling) {
        let limits = self.config.limits();
        let mut allowance = u64::MAX;
        let mut ceiling = Ceiling::None;
        let offset = self.resources.last().map_or(0, |r| r.size);
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
                offset,
                L::ResourceBytes,
            ),
        ];
        for (limit, used, kind) in budgets {
            if let Some(limit) = limit
                && limit.saturating_sub(used) < allowance
            {
                allowance = limit.saturating_sub(used);
                ceiling = Ceiling::Budget(kind, limit);
            }
        }
        if data
            && let OutputSize::Exact(expected) = self.stream.output_size()
            && expected.saturating_sub(self.total_out) < allowance
        {
            allowance = expected.saturating_sub(self.total_out);
            ceiling = Ceiling::Exact(expected);
        }
        if let Some(declared) = declared
            && declared < allowance
        {
            allowance = declared;
            ceiling = Ceiling::Declared(declared);
        }
        (allowance, ceiling)
    }

    /// The error for content that decoded past its ceiling.
    const fn exceeded(&self) -> E {
        match self.ceiling {
            Ceiling::None => E::SizeOverflow,
            Ceiling::Budget(kind, limit) => E::LimitExceeded { kind, limit },
            Ceiling::Exact(expected) | Ceiling::Declared(expected) => E::DeclaredSizeMismatch {
                expected,
                actual: expected.saturating_add(1),
            },
        }
    }

    /// The error for a reference to content the cache stopped retaining.
    fn evicted(&self) -> E {
        E::LimitExceeded {
            kind: L::DictionaryBytes,
            limit: self
                .config
                .limits()
                .max_dictionary_bytes
                .map_or(u64::MAX, |n| n as u64),
        }
    }

    /// Codec settings for the current content.
    fn codec_config(&self, mode: FrameMode) -> Result<DecoderConfig, E> {
        let limits = DecodeLimits::DEFAULT
            .with_max_window_log(self.config.max_window_log())
            .map_err(|_| E::InvalidState)?
            .with_max_output_bytes((self.limit != u64::MAX).then_some(self.limit))
            .with_max_input_bytes(if mode == FrameMode::Concatenated {
                self.config.limits().max_input_bytes
            } else {
                None
            });
        Ok(DecoderConfig::DEFAULT
            .with_frame_mode(mode)
            .with_limits(limits))
    }

    /// Switches to raw Zstandard input, replaying the detection bytes.
    fn raw(&mut self) -> Result<Tag, E> {
        self.format = Some(StreamInfo::Raw);
        self.add_resource(ResourceSource::Raw, false)?;
        (self.limit, self.ceiling) = self.ceiling(true, None);
        let config = self.codec_config(FrameMode::Concatenated)?;
        self.codec.reset(config);
        self.codec_active = true;
        self.replay = 0;
        self.phase = Phase::Begin;
        Ok(Tag::Stream)
    }

    /// Resolves the dictionary of the current chunk into owned storage.
    fn attach(&mut self, resolver: Option<DictionaryResolverRef<'_>>) -> Result<(), E> {
        let c = &self.layout.chunks[self.current];
        self.dictionary_active = !c.dictionaries.is_empty();
        if !self.dictionary_active || c.dictionaries == self.dictionary_refs {
            return Ok(());
        }
        let mut sources = [const { None }; crate::framing::core::MAX_REFERENCES];
        let mut serialized = false;
        let mut size = 0usize;
        for (slot, reference) in sources.iter_mut().zip(&c.dictionaries) {
            let source = match *reference {
                DictionaryReference::PrefixId(id) | DictionaryReference::SerializedId(id) => {
                    let is_serialized = matches!(reference, DictionaryReference::SerializedId(_));
                    serialized |= is_serialized;
                    let request = ExternalDictionaryRequest {
                        id,
                        kind: if is_serialized {
                            ExternalDictionaryKind::Serialized
                        } else {
                            ExternalDictionaryKind::Prefix
                        },
                    };
                    Source::External(resolver.and_then(|r| r.resolve(request)).ok_or(
                        E::MissingDictionary {
                            request,
                            location: self.location(0),
                        },
                    )?)
                }
                DictionaryReference::PrefixChunk(offset)
                | DictionaryReference::PrefixResource(offset)
                | DictionaryReference::SerializedResource(offset) => {
                    if self.config.internal_dictionaries() == InternalDictionaryPolicy::Reject {
                        return Err(E::InternalDictionaryReferencesDisabled);
                    }
                    serialized |= matches!(reference, DictionaryReference::SerializedResource(_));
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
                        match target.metadata {
                            Some(m) => Source::Metadata(m),
                            None if target.retained => Source::Cache(target.cache.clone()),
                            None => return Err(self.evicted()),
                        }
                    } else {
                        let r = target
                            .resource
                            .and_then(|r| self.resources.get(r))
                            .filter(|r| {
                                r.complete
                                    && matches!(r.header.source, ResourceSource::Framed { first_chunk } if first_chunk.0 == offset)
                            })
                            .ok_or(E::InvalidDictionaryReference)?;
                        if !r.retained {
                            return Err(self.evicted());
                        }
                        Source::Cache(r.cache.clone())
                    }
                }
            };
            size = size
                .checked_add(match &source {
                    Source::External(bytes) => bytes.len(),
                    Source::Cache(range) => range.len(),
                    Source::Metadata(m) => self.metadata[*m]
                        .value
                        .as_ref()
                        .ok_or(E::InvalidState)?
                        .bytes
                        .len(),
                })
                .ok_or(E::SizeOverflow)?;
            *slot = Some(source);
        }
        let count = c.dictionaries.len();
        // Release the previous dictionary before budgeting the next one.
        self.dictionary.clear();
        self.dictionary_parts = None;
        self.dictionary_refs.clear();
        let refs = count * size_of::<DictionaryReference>();
        let grow = size.saturating_sub(self.dictionary.capacity())
            + refs
                .saturating_sub(self.dictionary_refs.capacity() * size_of::<DictionaryReference>());
        self.budget(0, grow)?;
        self.dictionary
            .try_reserve_exact(size)
            .map_err(|_| E::AllocationFailed)?;
        self.dictionary_refs
            .try_reserve_exact(count)
            .map_err(|_| E::AllocationFailed)?;
        for source in sources.iter().flatten() {
            match source {
                Source::External(bytes) => self.dictionary.extend_from_slice(bytes),
                Source::Cache(range) => self
                    .dictionary
                    .extend_from_slice(&self.cache[range.clone()]),
                Source::Metadata(m) => {
                    let bytes = &self.metadata[*m]
                        .value
                        .as_ref()
                        .ok_or(E::InvalidState)?
                        .bytes;
                    self.dictionary.extend_from_slice(bytes);
                }
            }
        }
        let content = if serialized {
            DictionaryContent::Zstd
        } else {
            DictionaryContent::Raw
        };
        let parts =
            parse_decode_parts(&self.dictionary, content).map_err(|source| E::Dictionary {
                source,
                location: self.location(0),
            })?;
        self.dictionary_parts = Some(parts);
        self.budget(0, 0)?;
        let c = &self.layout.chunks[self.current];
        self.dictionary_refs.extend_from_slice(&c.dictionaries);
        Ok(())
    }

    fn prepare(&mut self, resolver: Option<DictionaryResolverRef<'_>>) -> Result<(), E> {
        let c = &self.layout.chunks[self.current];
        let Some(codec) = c.codec else {
            self.codec_active = false;
            return Ok(());
        };
        let data = is_data(c.kind);
        (self.limit, self.ceiling) = self.ceiling(data, c.declared_size);
        self.codec_active = codec != Codec::Uncompressed;
        self.dictionary_active = false;
        if !self.codec_active {
            return Ok(());
        }
        self.attach(resolver)?;
        let config = self.codec_config(FrameMode::Single)?;
        self.codec.reset(config);
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
        let retained = metadata.is_some() || resource.is_some_and(|r| self.resources[r].retained);
        self.contents.push(Content {
            chunk: self.current,
            cache: self.cache_start..self.cache.len(),
            retained,
            metadata,
            resource,
        });
        Ok(())
    }

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
                    self.resources.last_mut().ok_or(E::InvalidOrder)?.footer = Some(i);
                }
                MetadataKind::Global => {}
            }
        }
        self.register(Some(i))?;
        Ok(i)
    }

    /// Maps a codec failure to the framing error it reports.
    fn codec_error(&self, error: DecodeError, consumed: usize) -> E {
        match error {
            DecodeError::InputLimitExceeded { limit } => E::LimitExceeded {
                kind: L::InputBytes,
                limit,
            },
            DecodeError::OutputLimitExceeded { .. } => self.exceeded(),
            DecodeError::AllocationFailed => E::AllocationFailed,
            source => E::Decode {
                source,
                location: self.location(consumed),
            },
        }
    }

    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    fn payload(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        consumed: &mut usize,
        produced: &mut usize,
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
        let room = limits.max_input_bytes.map(|limit| {
            usize::try_from(limit.saturating_sub(self.total_in + *consumed as u64))
                .unwrap_or(usize::MAX)
        });
        let mut available = if raw {
            input.len() - *consumed
        } else {
            usize::try_from(self.chunk_left.min((input.len() - *consumed) as u64))
                .map_err(|_| E::SizeOverflow)?
        };
        let n;
        if !self.codec_active {
            if self.chunk_left == 0 {
                self.phase = Phase::Semantic;
                return Ok(None);
            }
            let allowance = self.limit.saturating_sub(self.chunk_out);
            if allowance == 0 {
                return Err(self.exceeded());
            }
            if destination.is_empty() {
                return Ok(Some(Tag::Output));
            }
            if available == 0 {
                return self.wait().map(Some);
            }
            let mut take = available
                .min(destination.len())
                .min(usize::try_from(allowance).unwrap_or(usize::MAX));
            if let Some(room) = room {
                take = take.min(room);
            }
            if take == 0 {
                return Err(E::LimitExceeded {
                    kind: L::InputBytes,
                    limit: limits.max_input_bytes.unwrap_or(0),
                });
            }
            destination[..take].copy_from_slice(&input[*consumed..*consumed + take]);
            *consumed += take;
            self.chunk_left -= take as u64;
            n = take;
            if self.chunk_left == 0 {
                self.phase = Phase::Semantic;
            }
        } else {
            let replaying = raw && self.replay < self.pending.len();
            if !raw && let Some(room) = room {
                if available != 0 && room == 0 {
                    return Err(E::LimitExceeded {
                        kind: L::InputBytes,
                        limit: limits.max_input_bytes.unwrap_or(0),
                    });
                }
                available = available.min(room);
            }
            let operation = if raw && !replaying && self.final_end.is_some() {
                DecodeOperation::Finish
            } else {
                DecodeOperation::Process
            };
            let dictionary = if self.dictionary_active {
                self.dictionary_parts
                    .as_ref()
                    .map(|parts| parts.view(&self.dictionary))
            } else {
                None
            };
            let feed = if replaying {
                &self.pending[self.replay..]
            } else {
                &input[*consumed..*consumed + available]
            };
            let result = self.codec.process(feed, destination, operation, dictionary);
            let (used, made, status) = match result {
                Ok(p) => (p.consumed, p.produced, Ok(p.status)),
                Err(f) => (f.consumed, f.produced, Err(f.error)),
            };
            if replaying {
                self.replay += used;
            } else {
                *consumed += used;
                if !raw {
                    self.chunk_left -= used as u64;
                }
            }
            n = made;
            self.account(data, &destination[..n], n)?;
            if data {
                *produced = n;
            }
            let status = status.map_err(|error| self.codec_error(error, *consumed))?;
            check(
                self.retained_bytes() as u64,
                limits.max_workspace_bytes.map(|x| x as u64),
                L::WorkspaceBytes,
            )?;
            match status {
                DecoderStatus::Finished if raw => {
                    self.codec_ended = true;
                    self.phase = Phase::ResourceEnd;
                }
                DecoderStatus::Finished => {
                    let c = &self.layout.chunks[self.current];
                    if self.codec.frames_decoded() != 1 || self.chunk_left != 0 {
                        return Err(E::InvalidChunk);
                    }
                    if let Some(expected) = c.declared_size
                        && expected != self.chunk_out
                    {
                        return Err(E::DeclaredSizeMismatch {
                            expected,
                            actual: self.chunk_out,
                        });
                    }
                    self.phase = Phase::Semantic;
                }
                DecoderStatus::NeedsOutput => {
                    if data && n == 0 {
                        return Ok(Some(Tag::Output));
                    }
                }
                DecoderStatus::NeedsInput if replaying => {}
                DecoderStatus::NeedsInput if !raw && self.chunk_left == 0 => {
                    return Err(E::InvalidChunk);
                }
                DecoderStatus::NeedsInput if *consumed < input.len() && !raw => {}
                DecoderStatus::NeedsInput => {
                    if data && n != 0 {
                        return Ok(Some(Tag::Data(position)));
                    }
                    return self.wait().map(Some);
                }
            }
            if data && n != 0 {
                return Ok(Some(Tag::Data(position)));
            }
            return Ok(None);
        }
        self.account(data, &destination[..n], n)?;
        if data {
            *produced = n;
            return Ok(Some(Tag::Data(position)));
        }
        Ok(None)
    }

    /// Counts `n` regenerated bytes, retaining resource content for later
    /// references and collecting metadata.
    fn account(&mut self, data: bool, bytes: &[u8], n: usize) -> Result<(), E> {
        let raw = self.format == Some(StreamInfo::Raw);
        self.chunk_out = plus(self.chunk_out, n)?;
        self.total_decoded = plus(self.total_decoded, n)?;
        if data {
            self.total_out = plus(self.total_out, n)?;
            let r = self.resources.last_mut().ok_or(E::InvalidState)?;
            r.size = plus(r.size, n)?;
            if !raw
                && self.config.internal_dictionaries() == InternalDictionaryPolicy::Retain
                && n != 0
                && self.resources.last().is_some_and(|r| r.retained)
            {
                let needed = self.cache.len().checked_add(n).ok_or(E::SizeOverflow)?;
                if needed > self.cache.capacity() {
                    // Grow geometrically while the budget allows, then to the
                    // exact need; past the budget, stop retaining: only a
                    // later reference to this content fails.
                    let doubled = needed.max(self.cache.capacity().saturating_mul(2));
                    let target = if self.budget(0, doubled - self.cache.capacity()).is_ok() {
                        doubled
                    } else {
                        needed
                    };
                    if let Err(error) = self.budget(0, target - self.cache.capacity()) {
                        if !matches!(error, E::LimitExceeded { .. }) {
                            return Err(error);
                        }
                        self.retaining = false;
                        self.resources.last_mut().ok_or(E::InvalidState)?.retained = false;
                        return Ok(());
                    }
                    self.cache
                        .try_reserve_exact(target - self.cache.len())
                        .map_err(|_| E::AllocationFailed)?;
                }
                self.cache.extend_from_slice(bytes);
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
                self.budget(target - self.scratch.capacity(), 0)?;
                self.scratch
                    .try_reserve_exact(target - self.scratch.len())
                    .map_err(|_| E::AllocationFailed)?;
            }
            self.scratch.extend_from_slice(bytes);
        }
        Ok(())
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

    /// The seek table entry the next frame of the container must have.
    fn next_frame(&mut self) -> Option<[u64; 2]> {
        if !self.seek.header {
            self.seek.header = true;
            return Some([CONTAINER_HEADER as u64, 0]);
        }
        let c = self.layout.chunks.get(self.seek.chunk)?;
        if self.seek.body {
            self.seek.body = false;
            self.seek.chunk += 1;
            return Some([c.body_length?, c.declared_size?]);
        }
        let frame = c.length - c.body_length.unwrap_or(0);
        if c.body_length.is_some() {
            self.seek.body = true;
        } else {
            self.seek.chunk += 1;
        }
        Some([frame, 0])
    }

    /// Frames the seek table must list: the header frame and every chunk's.
    fn frame_count(&self) -> u64 {
        1 + self
            .layout
            .chunks
            .iter()
            .map(|c| 1 + u64::from(c.body_length.is_some()))
            .sum::<u64>()
    }

    /// Reads and checks the seek table frame after its magic number.
    fn seek_table(&mut self, input: &[u8], consumed: &mut usize) -> Result<Option<Tag>, E> {
        let need = if !self.seek.sized {
            4
        } else if self.seek.entries != 0 {
            SEEK_ENTRY
        } else {
            SEEK_FOOTER
        };
        while self.pending.len() < need {
            self.reserve_pending()?;
            let Some(b) = self.byte(input, consumed)? else {
                return self.wait().map(Some);
            };
            self.pending.push(b);
        }
        if !self.seek.sized {
            let size = u64::from(wire::le32(&self.pending, 0).ok_or(E::InvalidState)?);
            let entries = size
                .checked_sub(SEEK_FOOTER as u64)
                .filter(|n| n % SEEK_ENTRY as u64 == 0)
                .ok_or(E::InvalidSeekTable)?
                / SEEK_ENTRY as u64;
            if entries != self.frame_count() {
                return Err(E::InvalidSeekTable);
            }
            self.seek.sized = true;
            self.seek.entries = entries;
        } else if self.seek.entries != 0 {
            let compressed = wire::le32(&self.pending, 0).ok_or(E::InvalidState)?;
            let decompressed = wire::le32(&self.pending, 4).ok_or(E::InvalidState)?;
            if self.next_frame() != Some([u64::from(compressed), u64::from(decompressed)]) {
                return Err(E::InvalidSeekTable);
            }
            self.seek.entries -= 1;
        } else {
            let frames = wire::le32(&self.pending, 0).ok_or(E::InvalidState)?;
            let descriptor = self.pending[4];
            let magic = wire::le32(&self.pending, 5).ok_or(E::InvalidState)?;
            if u64::from(frames) != self.frame_count()
                || descriptor & SEEK_DESCRIPTOR_MASK != 0
                || magic != SEEKABLE_MAGIC
            {
                return Err(E::InvalidSeekTable);
            }
            let end = plus(self.total_in, *consumed)?;
            if let Some(footer) = self.layout.footer
                && footer.file_size.is_some_and(|size| size != end)
            {
                return Err(E::InvalidFooter);
            }
            self.layout.seek_table = Some(SeekTableInfo {
                offset: ChunkOffset(self.seek_offset),
                frames: u64::from(frames),
            });
            self.pending.clear();
            self.phase = Phase::Finish;
            return Ok(Some(Tag::SeekTable));
        }
        self.pending.clear();
        Ok(None)
    }

    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    fn drive(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        consumed: &mut usize,
        produced: &mut usize,
        resolver: Option<DictionaryResolverRef<'_>>,
    ) -> Result<Tag, E> {
        loop {
            match self.phase {
                Phase::Detect => {
                    self.reserve_pending()?;
                    let Some(b) = self.byte(input, consumed)? else {
                        return self.wait();
                    };
                    self.pending.push(b);
                    let n = self.pending.len();
                    if n < 4 {
                        continue;
                    }
                    let magic = wire::le32(&self.pending, 0).ok_or(E::InvalidState)?;
                    if magic == DICTIONARY_MAGIC {
                        return Err(E::UnexpectedInputKind(
                            UnexpectedInputKind::SerializedDictionary,
                        ));
                    }
                    let matches = magic == CHUNK_MAGIC
                        && (n < 8 || wire::le32(&self.pending, 4) == Some(5))
                        && (n < 12 || self.pending[8..12] == SIGNATURE);
                    if !matches {
                        if self.config.input_mode() != InputMode::Auto {
                            return Err(E::InvalidSignature);
                        }
                        return self.raw();
                    }
                    if n == CONTAINER_HEADER {
                        if b & VERSION_MASK != 0 {
                            return Err(E::UnsupportedVersion(b & VERSION_MASK));
                        }
                        self.format = Some(StreamInfo::Framed(ContainerHeader { flags: b }));
                        self.pending.clear();
                        self.phase = Phase::Header;
                        return Ok(Tag::Stream);
                    }
                }
                Phase::Header => {
                    self.reserve_pending()?;
                    let Some(b) = self.byte(input, consumed)? else {
                        return self.wait();
                    };
                    self.pending.push(b);
                    if self.pending.len() < 4 {
                        continue;
                    }
                    match wire::le32(&self.pending, 0).ok_or(E::InvalidState)? {
                        CHUNK_MAGIC => self.phase = Phase::ChunkHeader,
                        SEEK_TABLE_MAGIC => {
                            self.seek_order()?;
                            self.seek_offset =
                                self.total_in + *consumed as u64 - self.pending.len() as u64;
                            self.pending.clear();
                            self.seek = SeekCursor::default();
                            self.phase = Phase::SeekTable;
                        }
                        _ if self.layout.footer.is_some() => return Err(E::InvalidOrder),
                        _ => return Err(E::InvalidChunk),
                    }
                }
                Phase::ChunkHeader => {
                    self.reserve_pending()?;
                    let Some(b) = self.byte(input, consumed)? else {
                        return self.wait();
                    };
                    self.pending.push(b);
                    if self.pending.len() < FRAME_HEADER {
                        continue;
                    }
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
                        self.chunk_left = c
                            .body_length
                            .unwrap_or(c.length - self.pending.len() as u64);
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
                        if let Some(tag) = self.payload(input, output, consumed, produced)? {
                            return Ok(tag);
                        }
                    } else {
                        let k = self.layout.chunks[self.current].kind;
                        if k == ChunkType::Footer
                            && self.chunk_left + self.pending.len() as u64 != FOOTER_FIELDS as u64
                        {
                            return Err(E::InvalidFooter);
                        }
                        if self.chunk_left == 0 {
                            self.phase = Phase::Semantic;
                            continue;
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
                            let field = |at: usize| {
                                self.pending
                                    .get(at..at + 8)
                                    .and_then(|b| b.try_into().ok())
                                    .map(u64::from_le_bytes)
                                    .ok_or(E::InvalidFooter)
                            };
                            let directory = field(0)?;
                            let size = field(8)?;
                            if directory != self.layout.directory.as_ref().map_or(0, |d| d.offset.0)
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
                Phase::SeekTable => {
                    if let Some(tag) = self.seek_table(input, consumed)? {
                        return Ok(tag);
                    }
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
    // boxing this record would require an allocation on the error path.
    #[expect(
        clippy::result_large_err,
        reason = "allocation-independent progress is part of the public contract"
    )]
    pub fn process(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: DecodeOperation,
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
            self.drive(input, output, &mut consumed, &mut produced, resolver)
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
            Tag::SeekTable => FramedEvent::SeekTable(self.layout.seek_table?),
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

const fn is_data(kind: ChunkType) -> bool {
    matches!(
        kind,
        ChunkType::Data
            | ChunkType::FirstPartial
            | ChunkType::MiddlePartial
            | ChunkType::LastPartial
    )
}

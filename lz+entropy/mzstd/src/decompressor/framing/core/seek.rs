//! Seek table and directory indexing, and sparse, dependency-ordered
//! execution over positional reads.
//!
//! Opening reads the container header, the seek table at the end, the
//! footer it names and the central directory the footer names. The seek
//! table lists every frame, so each directory record, each detached body
//! and every gap between them is matched against it without scanning
//! payload; gaps may hold padding only.

use super::*;
use crate::framing::core::{FLAG_CONTAINER, FLAG_HIDDEN, FOOTER_CONTENT};
use FramedSeekError as S;
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug)]
struct IndexedResource {
    chunks: Range<usize>,
    metadata: Option<usize>,
    footer: Option<usize>,
}

#[derive(Debug)]
pub(in crate::decompressor::framing) struct Index {
    pub header: ContainerHeader,
    pub footer: ContainerFooter,
    pub infos: Vec<ResourceInfo>,
    pub metadata: Vec<Option<Metadata>>,
    chunks: Vec<ChunkInfo>,
    resources: Vec<IndexedResource>,
    owners: Vec<Option<usize>>,
    bytes: usize,
    config: FramedDecodeConfig,
}

fn available_limits(engine: &Engine) -> FramedDecodeLimits {
    let mut limits = engine.config.limits();
    limits.max_framing_bytes = limits
        .max_framing_bytes
        .map(|n| n.saturating_sub(engine.framing_bytes()));
    limits.max_workspace_bytes = limits
        .max_workspace_bytes
        .map(|n| n.saturating_sub(engine.retained_bytes()));
    limits
}

fn storage(bytes: usize, extra: usize, limits: FramedDecodeLimits) -> Result<(), E> {
    let total = bytes.checked_add(extra).ok_or(E::SizeOverflow)? as u64;
    check(
        total,
        limits.max_framing_bytes.map(|n| n as u64),
        L::FramingBytes,
    )?;
    check(
        total,
        limits.max_workspace_bytes.map(|n| n as u64),
        L::WorkspaceBytes,
    )
}

fn reserve<T>(
    vec: &mut Vec<T>,
    extra: usize,
    bytes: &mut usize,
    limits: FramedDecodeLimits,
) -> Result<(), E> {
    let needed = vec.len().checked_add(extra).ok_or(E::SizeOverflow)?;
    if needed > vec.capacity() {
        let allocation = needed.checked_mul(size_of::<T>()).ok_or(E::SizeOverflow)?;
        storage(*bytes, allocation, limits)?;
        let old = vec.capacity();
        vec.try_reserve_exact(extra)
            .map_err(|_| E::AllocationFailed)?;
        *bytes = bytes
            .checked_add((vec.capacity() - old) * size_of::<T>())
            .ok_or(E::SizeOverflow)?;
    }
    Ok(())
}

fn read_at(source: &mut (impl Read + Seek), offset: u64, bytes: &mut [u8]) -> Result<(), S> {
    source.seek(SeekFrom::Start(offset))?;
    source.read_exact(bytes)?;
    Ok(())
}

fn read_number(source: &mut impl Read, left: &mut u64) -> Result<u64, S> {
    let mut bytes = [0; 9];
    for i in 0..bytes.len() {
        if *left == 0 {
            return Err(E::InvalidDirectory.into());
        }
        source.read_exact(&mut bytes[i..=i])?;
        *left -= 1;
        if bytes[i] < 128 {
            return Ok(wire::number(&bytes[..=i], &mut 0)?.ok_or(E::InvalidDirectory)?);
        }
    }
    Err(E::InvalidDirectory.into())
}

/// Reads the header of the chunk whose frame starts at `offset` and spans
/// `frame` bytes; only padding, directory and footer chunks are read so.
fn read_structural(
    source: &mut (impl Read + Seek),
    offset: u64,
    frame: u64,
) -> Result<ChunkType, S> {
    let mut bytes = [0; FRAME_HEADER + 1];
    let n = if frame > FRAME_HEADER as u64 {
        bytes.len()
    } else {
        FRAME_HEADER
    };
    if frame < FRAME_HEADER as u64 {
        return Err(E::InvalidDirectory.into());
    }
    read_at(source, offset, &mut bytes[..n])?;
    if wire::le32(&bytes, 0) != Some(CHUNK_MAGIC)
        || wire::le32(&bytes, 4).map(u64::from) != Some(frame - FRAME_HEADER as u64)
    {
        return Err(E::InvalidDirectory.into());
    }
    Ok(match (n, bytes[FRAME_HEADER]) {
        (FRAME_HEADER, _) | (_, 0) => ChunkType::Padding,
        (_, 9) => ChunkType::CentralDirectory,
        (_, 10) => ChunkType::Footer,
        _ => return Err(E::InvalidDirectory.into()),
    })
}

impl Index {
    pub fn open(source: &mut (impl Read + Seek), engine: &Engine) -> Result<Self, S> {
        // Retained owner allocations coexist with the index during opening.
        let limits = available_limits(engine);
        let config = engine.config.with_limits(limits);
        let length = source.seek(SeekFrom::End(0))?;
        let mut signature = [0; CONTAINER_HEADER];
        read_at(source, 0, &mut signature)?;
        if wire::le32(&signature, 0) != Some(CHUNK_MAGIC)
            || wire::le32(&signature, 4) != Some(5)
            || signature[8..12] != SIGNATURE
        {
            return Err(E::InvalidSignature.into());
        }
        let flags = signature[12];
        if flags & VERSION_MASK != 0 {
            return Err(E::UnsupportedVersion(flags & VERSION_MASK).into());
        }
        let header = ContainerHeader { flags };
        if flags & FLAG_CONTAINER == 0 {
            return Err(S::CentralDirectoryRequired);
        }
        // The seek table footer is the last nine bytes.
        let mut tail = [0; SEEK_FOOTER];
        let tail_offset = length
            .checked_sub(SEEK_FOOTER as u64)
            .ok_or(E::InvalidSeekTable)?;
        read_at(source, tail_offset, &mut tail)?;
        let frames = u64::from(wire::le32(&tail, 0).ok_or(E::InvalidState)?);
        if tail[4] & SEEK_DESCRIPTOR_MASK != 0 || wire::le32(&tail, 5) != Some(SEEKABLE_MAGIC) {
            return Err(E::InvalidSeekTable.into());
        }
        let table = frames
            .checked_mul(SEEK_ENTRY as u64)
            .and_then(|n| n.checked_add((FRAME_HEADER + SEEK_FOOTER) as u64))
            .ok_or(E::SizeOverflow)?;
        let seek_start = length
            .checked_sub(table)
            .filter(|start| *start >= CONTAINER_HEADER as u64)
            .ok_or(E::InvalidSeekTable)?;
        let mut frame_header = [0; FRAME_HEADER];
        read_at(source, seek_start, &mut frame_header)?;
        if wire::le32(&frame_header, 0) != Some(SEEK_TABLE_MAGIC)
            || wire::le32(&frame_header, 4).map(u64::from) != Some(table - FRAME_HEADER as u64)
        {
            return Err(E::InvalidSeekTable.into());
        }
        // Each listed frame holds at least a frame header.
        if frames < 3 || frames > seek_start / FRAME_HEADER as u64 {
            return Err(E::InvalidSeekTable.into());
        }
        let mut index = Self {
            header,
            footer: ContainerFooter {
                offset: ChunkOffset(0),
                file_size: None,
                directory: None,
            },
            infos: Vec::new(),
            metadata: Vec::new(),
            chunks: Vec::new(),
            resources: Vec::new(),
            owners: Vec::new(),
            bytes: 0,
            config,
        };
        let limits = config.limits();
        let count = usize::try_from(frames).map_err(|_| E::SizeOverflow)?;
        let mut sizes: Vec<[u32; 2]> = Vec::new();
        reserve(&mut sizes, count, &mut index.bytes, limits)?;
        let mut entries = [0; 8192];
        let mut left = count;
        while left != 0 {
            let take = left.min(entries.len() / SEEK_ENTRY);
            source.read_exact(&mut entries[..take * SEEK_ENTRY])?;
            for entry in entries[..take * SEEK_ENTRY].as_chunks::<SEEK_ENTRY>().0 {
                sizes.push([
                    wire::le32(entry, 0).ok_or(E::InvalidState)?,
                    wire::le32(entry, 4).ok_or(E::InvalidState)?,
                ]);
            }
            left -= take;
        }
        let total = sizes
            .iter()
            .try_fold(0u64, |sum, [c, _]| sum.checked_add(u64::from(*c)))
            .ok_or(E::SizeOverflow)?;
        if total != seek_start || sizes[0] != [CONTAINER_HEADER as u32, 0] {
            return Err(E::InvalidSeekTable.into());
        }
        // The footer is the last listed frame.
        let [footer_frame, footer_decoded] = sizes[count - 1];
        let footer_offset = seek_start - u64::from(footer_frame);
        let mut footer = [0; FRAME_HEADER + FOOTER_CONTENT];
        if u64::from(footer_frame) != footer.len() as u64 || footer_decoded != 0 {
            return Err(E::InvalidFooter.into());
        }
        read_at(source, footer_offset, &mut footer)?;
        let field = |at: usize| {
            footer[at..at + 8]
                .try_into()
                .map(u64::from_le_bytes)
                .map_err(|_| E::InvalidState)
        };
        if wire::le32(&footer, 0) != Some(CHUNK_MAGIC)
            || wire::le32(&footer, 4) != Some(FOOTER_CONTENT as u32)
            || footer[FRAME_HEADER] != 10
        {
            return Err(E::InvalidFooter.into());
        }
        let directory_offset = field(FRAME_HEADER + 1)?;
        let file_size = field(FRAME_HEADER + 9)?;
        if file_size != 0 && file_size != length {
            return Err(E::InvalidFooter.into());
        }
        if directory_offset == 0 {
            return Err(S::CentralDirectoryRequired);
        }
        if !(CONTAINER_HEADER as u64..footer_offset).contains(&directory_offset) {
            return Err(E::InvalidDirectory.into());
        }
        index.footer = ContainerFooter {
            offset: ChunkOffset(footer_offset),
            file_size: (file_size != 0).then_some(file_size),
            directory: Some(ChunkOffset(directory_offset)),
        };
        let mut directory = [0; FRAME_HEADER + 1];
        read_at(source, directory_offset, &mut directory)?;
        let directory_size = u64::from(wire::le32(&directory, 4).ok_or(E::InvalidState)?);
        if wire::le32(&directory, 0) != Some(CHUNK_MAGIC)
            || directory_size == 0
            || directory[FRAME_HEADER] != 9
        {
            return Err(E::InvalidDirectory.into());
        }
        let directory_end = directory_offset + FRAME_HEADER as u64 + directory_size;
        if directory_end > footer_offset {
            return Err(E::InvalidDirectory.into());
        }
        let mut accepted = (CONTAINER_HEADER as u64)
            .checked_add(directory_size + FRAME_HEADER as u64)
            .and_then(|n| n.checked_add(length - footer_offset))
            .ok_or(E::SizeOverflow)?;
        check(accepted, limits.max_input_bytes, L::InputBytes)?;
        source.seek(SeekFrom::Start(directory_offset + directory.len() as u64))?;
        let mut left = directory_size - 1;
        let repeated = read_number(source, &mut left)?;
        let mut previous_end = CONTAINER_HEADER as u64;
        while left != 0 {
            check(index.chunks.len() as u64 + 3, limits.max_chunks, L::Chunks)?;
            let offset = read_number(source, &mut left)?;
            let size = read_number(source, &mut left)?;
            if offset < previous_end || offset >= directory_offset || size > left {
                return Err(E::InvalidDirectory.into());
            }
            let size = usize::try_from(size).map_err(|_| E::SizeOverflow)?;
            let mut bytes = Vec::new();
            reserve(&mut bytes, size, &mut index.bytes, limits)?;
            bytes.resize(size, 0);
            source.read_exact(&mut bytes)?;
            left -= size as u64;
            let mut chunk =
                wire::header(&bytes, offset, |extra| storage(index.bytes, extra, limits))?
                    .ok_or(E::InvalidDirectory)?;
            if chunk.codec.is_none() {
                return Err(E::InvalidDirectory.into());
            }
            previous_end = offset.checked_add(chunk.length).ok_or(E::SizeOverflow)?;
            if previous_end > directory_offset {
                return Err(E::InvalidDirectory.into());
            }
            accepted = plus(accepted, bytes.len())?;
            check(accepted, limits.max_input_bytes, L::InputBytes)?;
            index.bytes = index
                .bytes
                .checked_add(chunk.dictionaries.capacity() * size_of::<DictionaryReference>())
                .ok_or(E::SizeOverflow)?;
            chunk.header_bytes = bytes;
            reserve(&mut index.chunks, 1, &mut index.bytes, limits)?;
            reserve(&mut index.metadata, 1, &mut index.bytes, limits)?;
            reserve(&mut index.owners, 1, &mut index.bytes, limits)?;
            index.chunks.push(chunk);
            index.metadata.push(None);
            index.owners.push(None);
        }
        index.frames(source, &sizes, directory_offset, footer_offset)?;
        index.structure(repeated)?;
        index.bytes -= sizes.capacity() * size_of::<[u32; 2]>();
        index.config = engine.config;
        Ok(index)
    }

    /// Matches every listed frame with the container header, a directory
    /// record, its body, the directory, the footer or padding.
    fn frames(
        &self,
        source: &mut (impl Read + Seek),
        sizes: &[[u32; 2]],
        directory: u64,
        footer: u64,
    ) -> Result<(), S> {
        let limits = self.config.limits();
        let mut offset = CONTAINER_HEADER as u64;
        let mut record = 0;
        let mut chunks = self.chunks.len() as u64 + 2;
        let mut frame = 1;
        let mut directory_seen = false;
        while frame < sizes.len() - 1 {
            let [compressed, decompressed] = sizes[frame];
            let compressed = u64::from(compressed);
            if decompressed != 0 {
                return Err(E::InvalidSeekTable.into());
            }
            if let Some(chunk) = self.chunks.get(record).filter(|c| c.offset.0 == offset) {
                if compressed != chunk.length - chunk.body_length.unwrap_or(0) {
                    return Err(E::InvalidSeekTable.into());
                }
                offset += compressed;
                if let Some(body) = chunk.body_length {
                    frame += 1;
                    let [length, decoded] = *sizes.get(frame).ok_or(E::InvalidSeekTable)?;
                    if u64::from(length) != body || Some(u64::from(decoded)) != chunk.declared_size
                    {
                        return Err(E::InvalidSeekTable.into());
                    }
                    offset += body;
                }
                record += 1;
            } else {
                let kind = read_structural(source, offset, compressed)?;
                match kind {
                    ChunkType::CentralDirectory if offset == directory && !directory_seen => {
                        directory_seen = true;
                    }
                    ChunkType::Padding => {
                        chunks = chunks.checked_add(1).ok_or(E::SizeOverflow)?;
                        check(chunks, limits.max_chunks, L::Chunks)?;
                    }
                    _ => return Err(E::InvalidDirectory.into()),
                }
                offset += compressed;
            }
            frame += 1;
        }
        if offset != footer || record != self.chunks.len() || !directory_seen {
            return Err(E::InvalidDirectory.into());
        }
        Ok(())
    }

    fn structure(&mut self, repeated: u64) -> Result<(), E> {
        let mut partial = false;
        let mut hidden = false;
        let mut pending = None;
        let mut previous = None;
        let mut repeat_start = None;
        let mut originals = 0;
        let mut repeats = 0;
        let mut original_cursor = 0;
        for (i, c) in self.chunks.iter().enumerate() {
            let kind = c.kind;
            if partial && !matches!(kind, ChunkType::MiddlePartial | ChunkType::LastPartial)
                || pending.is_some() && !matches!(kind, ChunkType::Data | ChunkType::FirstPartial)
                || repeat_start.is_some() && kind != ChunkType::RepeatMetadata
            {
                return Err(E::InvalidOrder);
            }
            match kind {
                ChunkType::Metadata => {
                    pending = Some(i);
                    originals += 1;
                }
                ChunkType::Data | ChunkType::FirstPartial => {
                    check(
                        self.infos.len() as u64 + 1,
                        self.config.limits().max_resources,
                        L::Resources,
                    )?;
                    reserve(
                        &mut self.resources,
                        1,
                        &mut self.bytes,
                        self.config.limits(),
                    )?;
                    reserve(&mut self.infos, 1, &mut self.bytes, self.config.limits())?;
                    self.resources.push(IndexedResource {
                        chunks: i..i + 1,
                        metadata: pending.take(),
                        footer: None,
                    });
                    hidden = c.flags & FLAG_HIDDEN != 0;
                    self.infos.push(ResourceInfo {
                        index: ResourceIndex(self.infos.len() as u64),
                        hidden,
                        checksum: None,
                        decoded_size: Some(0),
                    });
                    partial = kind == ChunkType::FirstPartial;
                }
                ChunkType::MiddlePartial | ChunkType::LastPartial => {
                    if !partial {
                        return Err(E::InvalidOrder);
                    }
                    partial = kind != ChunkType::LastPartial;
                }
                ChunkType::FooterMetadata => {
                    if !matches!(previous, Some(ChunkType::Data | ChunkType::LastPartial)) {
                        return Err(E::InvalidOrder);
                    }
                    self.resources.last_mut().ok_or(E::InvalidOrder)?.footer = Some(i);
                    originals += 1;
                }
                ChunkType::RepeatMetadata => {
                    repeat_start.get_or_insert(c.offset.0);
                    let relative = self.chunks[original_cursor..i]
                        .iter()
                        .position(|c| {
                            matches!(c.kind, ChunkType::Metadata | ChunkType::FooterMetadata)
                        })
                        .ok_or(E::InvalidMetadata)?;
                    original_cursor += relative + 1;
                    let original = &self.chunks[original_cursor - 1];
                    let expected = if original.kind == ChunkType::Metadata {
                        MetadataKind::Resource
                    } else {
                        MetadataKind::Footer
                    };
                    if c.repeated_kind != Some(expected) {
                        return Err(E::InvalidMetadata);
                    }
                    repeats += 1;
                }
                ChunkType::GlobalMetadata => {}
                _ => return Err(E::InvalidDirectory),
            }
            if is_data(kind) {
                if c.body_length.is_some() == hidden {
                    return Err(E::InvalidChunk);
                }
                let owner = self.resources.len() - 1;
                self.owners[i] = Some(owner);
                self.resources[owner].chunks.end = i + 1;
                let info = &mut self.infos[owner];
                let size = c
                    .declared_size
                    .unwrap_or(c.length - c.header_bytes.len() as u64);
                info.decoded_size = Some(
                    info.decoded_size
                        .ok_or(E::SizeOverflow)?
                        .checked_add(size)
                        .ok_or(E::SizeOverflow)?,
                );
                info.checksum = c.checksum;
            }
            previous = Some(kind);
        }
        if partial || pending.is_some() {
            return Err(E::InvalidOrder);
        }
        if repeat_start.unwrap_or(0) != repeated || repeats != 0 && repeats != originals {
            return Err(E::InvalidDirectory);
        }
        for (i, c) in self.chunks.iter().enumerate() {
            for reference in &c.dictionaries {
                self.dependency(i, *reference)?;
            }
        }
        Ok(())
    }

    fn dependency(
        &self,
        current: usize,
        reference: DictionaryReference,
    ) -> Result<Option<Range<usize>>, E> {
        let (offset, resource) = match reference {
            DictionaryReference::PrefixChunk(n) => (n, false),
            DictionaryReference::PrefixResource(n) | DictionaryReference::SerializedResource(n) => {
                (n, true)
            }
            _ => return Ok(None),
        };
        let target = self
            .chunks
            .binary_search_by_key(&offset, |c| c.offset.0)
            .map_err(|_| E::InvalidDictionaryReference)?;
        if target >= current
            || self.chunks[current].kind == ChunkType::RepeatMetadata
                && self.chunks[target].kind != ChunkType::RepeatMetadata
        {
            return Err(E::InvalidDictionaryReference);
        }
        let range = if resource {
            let r = self.owners[target]
                .and_then(|i| self.resources.get(i))
                .ok_or(E::InvalidDictionaryReference)?;
            if r.chunks.start != target || r.chunks.end > current {
                return Err(E::InvalidDictionaryReference);
            }
            r.chunks.clone()
        } else {
            target..target + 1
        };
        Ok(Some(range))
    }

    pub fn resource_range(&self, index: ResourceIndex) -> Result<Range<usize>, S> {
        let i = usize::try_from(index.0).map_err(|_| S::ResourceNotFound)?;
        Ok(self
            .resources
            .get(i)
            .ok_or(S::ResourceNotFound)?
            .chunks
            .clone())
    }

    pub fn metadata_chunk(&self, index: ResourceIndex, footer: bool) -> Result<Option<usize>, S> {
        let i = usize::try_from(index.0).map_err(|_| S::ResourceNotFound)?;
        let resource = self.resources.get(i).ok_or(S::ResourceNotFound)?;
        Ok(if footer {
            resource.footer
        } else {
            resource.metadata
        })
    }

    pub fn cache_metadata(&mut self, slot: usize, value: Metadata) -> Result<(), E> {
        let extra = value
            .bytes
            .capacity()
            .checked_add(value.fields.capacity() * size_of::<([u8; 2], Range<usize>)>())
            .ok_or(E::SizeOverflow)?;
        storage(self.bytes, extra, self.config.limits())?;
        self.bytes += extra;
        self.metadata[slot] = Some(value);
        Ok(())
    }
}

/// Cancellation guard also handles a forgotten child when its parent is dropped.
#[derive(Debug)]
pub(in crate::decompressor::framing) struct Lease<'a> {
    pub owner: &'a mut FramedDecompressor,
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.owner.cancel();
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Needed {
    decode: bool,
    retain: bool,
    retain_resource: bool,
}

#[derive(Debug)]
pub(in crate::decompressor::framing) struct Operation<'a> {
    owner: &'a mut FramedDecompressor,
    plan: Vec<Needed>,
    target: Range<usize>,
    next: usize,
    active: Option<usize>,
    input: [u8; 8192],
    input_start: usize,
    input_end: usize,
    position: u64,
}

impl<'a> Operation<'a> {
    pub fn new(
        owner: &'a mut FramedDecompressor,
        index: &Index,
        target: Range<usize>,
    ) -> Result<Self, S> {
        // A forgotten resource may leave workspace populated; it contains no
        // borrowed dictionary pointers and can always be cancelled here.
        owner.cancel();
        owner.active = true;
        let mut plan = Vec::new();
        let mut bytes = index.bytes;
        reserve(
            &mut plan,
            index.chunks.len(),
            &mut bytes,
            available_limits(&owner.engine),
        )?;
        plan.resize(index.chunks.len(), Needed::default());
        for slot in &mut plan[target.clone()] {
            slot.decode = true;
        }
        // Every edge points backwards, so one reverse pass computes the closure.
        // Earlier fragments of a resource precede a later one, so whole-resource
        // accounting and limits see the resource from its start.
        for i in (0..plan.len()).rev() {
            if !plan[i].decode {
                continue;
            }
            plan[i].retain |= plan[i].retain_resource;
            if let Some(resource) = index.owners[i]
                && i > index.resources[resource].chunks.start
            {
                plan[i - 1].decode = true;
                plan[i - 1].retain_resource |= plan[i].retain_resource;
            }
            for reference in &index.chunks[i].dictionaries {
                if let Some(range) = index.dependency(i, *reference)? {
                    if index.config.internal_dictionaries() == InternalDictionaryPolicy::Reject {
                        return Err(E::InternalDictionaryReferencesDisabled.into());
                    }
                    let last = range.end - 1;
                    plan[last].decode = true;
                    plan[last].retain = true;
                    plan[last].retain_resource |= range.len() > 1;
                }
            }
        }
        owner.engine.collector_bytes = bytes;
        owner.engine.format = Some(StreamInfo::Framed(index.header));
        owner.engine.budget(0, 0)?;
        Ok(Self {
            owner,
            plan,
            target,
            next: 0,
            active: None,
            input: [0; 8192],
            input_start: 0,
            input_end: 0,
            position: 0,
        })
    }

    fn begin(
        &mut self,
        source: &mut (impl Read + Seek),
        index: &Index,
        resolver: Option<DictionaryResolverRef<'_>>,
        slot: usize,
    ) -> Result<(), S> {
        let original = &index.chunks[slot];
        let engine = &mut self.owner.engine;
        // Validate exact bytes before any payload can enter a codec or dictionary.
        let mut matched = 0;
        while matched < original.header_bytes.len() {
            let n = (original.header_bytes.len() - matched).min(self.input.len());
            read_at(
                source,
                original.offset.0 + matched as u64,
                &mut self.input[..n],
            )?;
            if self.input[..n] != original.header_bytes[matched..matched + n] {
                return Err(S::DirectoryMismatch);
            }
            matched += n;
        }
        engine.budget(
            original.header_bytes.len()
                + original.dictionaries.len() * size_of::<DictionaryReference>()
                + size_of::<ChunkInfo>() * (engine.layout.chunks.len() + 1),
            0,
        )?;
        let mut chunk = wire::header(&original.header_bytes, original.offset.0, |n| {
            engine.budget(n, 0)
        })?
        .ok_or(E::InvalidChunk)?;
        chunk
            .header_bytes
            .try_reserve_exact(matched)
            .map_err(|_| E::AllocationFailed)?;
        chunk.header_bytes.extend_from_slice(&original.header_bytes);
        engine
            .layout
            .chunks
            .try_reserve_exact(1)
            .map_err(|_| E::AllocationFailed)?;
        engine.current = engine.layout.chunks.len();
        engine.layout.chunks.push(chunk);
        if is_data(original.kind) {
            let owner = index.owners[slot].ok_or(E::InvalidState)?;
            if index.resources[owner].chunks.start == slot {
                engine.pending_meta = None;
                engine.add_resource(
                    ResourceSource::Framed {
                        first_chunk: original.offset,
                    },
                    original.flags & FLAG_HIDDEN != 0,
                )?;
                engine
                    .resources
                    .last_mut()
                    .ok_or(E::InvalidState)?
                    .header
                    .index = ResourceIndex(owner as u64);
            }
        }
        engine.previous = slot.checked_sub(1).map(|i| index.chunks[i].kind);
        let body = original.body_length.unwrap_or(0);
        let frame = original.length - body;
        engine.chunk_left = original
            .body_length
            .unwrap_or(original.length - matched as u64);
        engine.chunk_out = 0;
        engine.cache_start = engine.cache.len();
        engine.phase = Phase::Content;
        engine.total_in = plus(engine.total_in, matched)?;
        check(
            engine.total_in,
            engine.config.limits().max_input_bytes,
            L::InputBytes,
        )?;
        engine.prepare(resolver)?;
        self.position = original.offset.0
            + if original.body_length.is_some() {
                frame
            } else {
                matched as u64
            };
        self.input_start = 0;
        self.input_end = 0;
        self.active = Some(slot);
        Ok(())
    }

    fn finish(&mut self, index: &Index, slot: usize) -> Result<(), S> {
        let engine = &mut self.owner.engine;
        let chunk = &index.chunks[slot];
        if is_data(chunk.kind) {
            engine.register(None)?;
            if matches!(chunk.kind, ChunkType::Data | ChunkType::LastPartial) {
                engine.resources.last_mut().ok_or(E::InvalidState)?.complete = true;
            }
        } else {
            let kind = match chunk.kind {
                ChunkType::Metadata => MetadataKind::Resource,
                ChunkType::FooterMetadata => MetadataKind::Footer,
                ChunkType::GlobalMetadata => MetadataKind::Global,
                _ => chunk.repeated_kind.ok_or(E::InvalidMetadata)?,
            };
            engine.budget(
                (engine.scratch.len() / 3) * size_of::<([u8; 2], Range<usize>)>()
                    + (engine.metadata.len() + 1) * size_of::<Meta>(),
                0,
            )?;
            let fields = wire::metadata(&engine.scratch, kind)?;
            engine.metadata_fields = plus(engine.metadata_fields, fields.len())?;
            check(
                engine.metadata_fields,
                engine.config.limits().max_metadata_fields,
                L::MetadataFields,
            )?;
            engine
                .metadata
                .try_reserve_exact(1)
                .map_err(|_| E::AllocationFailed)?;
            let metadata = engine.metadata.len();
            engine.metadata.push(Meta {
                value: Some(Metadata {
                    source: chunk.offset,
                    bytes: ::core::mem::take(&mut engine.scratch),
                    fields,
                }),
                scope: MetadataScope::Global,
                original: None,
                kind,
            });
            engine.register(Some(metadata))?;
        }
        self.active = None;
        Ok(())
    }

    pub fn read(
        &mut self,
        source: &mut (impl Read + Seek),
        index: &Index,
        resolver: Option<DictionaryResolverRef<'_>>,
        output: &mut [u8],
    ) -> Result<usize, S> {
        loop {
            if let Some(slot) = self.active {
                if matches!(self.owner.engine.phase, Phase::Semantic) {
                    self.finish(index, slot)?;
                    continue;
                }
                let engine = &mut self.owner.engine;
                if self.input_start == self.input_end && engine.chunk_left != 0 {
                    let n = engine.chunk_left.min(self.input.len() as u64) as usize;
                    read_at(source, self.position, &mut self.input[..n])?;
                    self.position += n as u64;
                    self.input_start = 0;
                    self.input_end = n;
                }
                let mut discard = [0; 8192];
                let deliver = self.target.contains(&slot) && is_data(index.chunks[slot].kind);
                let destination = if deliver { &mut *output } else { &mut discard };
                let mut consumed = 0;
                let mut produced = 0;
                // Retention is selective: only actual dictionary dependencies
                // enter the cache, never the requested streaming payload alone.
                let config = engine.config;
                if !self.plan[slot].retain {
                    engine.config =
                        config.with_internal_dictionaries(InternalDictionaryPolicy::Reject);
                }
                let result = engine.payload(
                    &self.input[self.input_start..self.input_end],
                    destination,
                    &mut consumed,
                    &mut produced,
                );
                engine.config = config;
                self.input_start += consumed;
                engine.total_in = plus(engine.total_in, consumed)?;
                let tag = result?;
                if deliver && produced != 0 {
                    return Ok(produced);
                }
                if consumed == 0
                    && produced == 0
                    && matches!(tag, Some(Tag::Input))
                    && !matches!(engine.phase, Phase::Semantic)
                {
                    return Err(E::UnexpectedEndOfInput.into());
                }
            } else {
                while self.next < self.plan.len() && !self.plan[self.next].decode {
                    self.next += 1;
                }
                if self.next == self.plan.len() {
                    return Ok(0);
                }
                let slot = self.next;
                self.next += 1;
                self.begin(source, index, resolver, slot)?;
            }
        }
    }

    pub fn metadata(
        &mut self,
        source: &mut (impl Read + Seek),
        index: &Index,
        resolver: Option<DictionaryResolverRef<'_>>,
        slot: usize,
    ) -> Result<Metadata, S> {
        let mut discard = [0; 8192];
        while self.read(source, index, resolver, &mut discard)? != 0 {}
        self.owner
            .engine
            .metadata
            .iter_mut()
            .find_map(|m| {
                if m.value
                    .as_ref()
                    .is_some_and(|m| m.source == index.chunks[slot].offset)
                {
                    m.value.take()
                } else {
                    None
                }
            })
            .ok_or(S::Decode(E::InvalidState))
    }
}

impl Drop for Operation<'_> {
    fn drop(&mut self) {
        self.owner.cancel();
        // The parent lease still owns the operation boundary. Forgetting that
        // lease must preserve the reusable owner's abandoned-session contract.
        self.owner.active = true;
    }
}

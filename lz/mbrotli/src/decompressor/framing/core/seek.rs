//! Directory indexing and sparse, dependency-ordered execution over positional reads.
use super::*;
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
    for i in 0..9 {
        if *left == 0 {
            return Err(E::InvalidDirectory.into());
        }
        source.read_exact(&mut bytes[i..i + 1])?;
        *left -= 1;
        if bytes[i] < 128 {
            return Ok(wire::number(&bytes[..=i], &mut 0)?.ok_or(E::InvalidDirectory)?);
        }
    }
    Err(E::InvalidDirectory.into())
}
fn read_header(
    source: &mut (impl Read + Seek),
    offset: u64,
    end: u64,
    limits: FramedDecodeLimits,
) -> Result<ChunkInfo, S> {
    // Non-content headers contain only length and type, bounded by ten bytes.
    let mut bytes = [0; 10];
    for n in 1..=bytes.len() {
        if offset.checked_add(n as u64).ok_or(E::SizeOverflow)? > end {
            return Err(E::InvalidFooter.into());
        }
        read_at(source, offset + n as u64 - 1, &mut bytes[n - 1..n])?;
        if let Some(mut chunk) = wire::header(&bytes[..n], offset, |extra| {
            if extra == 0 {
                Ok(())
            } else {
                Err(E::InvalidDirectory)
            }
        })? {
            if chunk.codec.is_some() {
                return Err(E::InvalidDirectory.into());
            }
            storage(0, n, limits)?;
            chunk
                .header_bytes
                .try_reserve_exact(n)
                .map_err(|_| E::AllocationFailed)?;
            chunk.header_bytes.extend_from_slice(&bytes[..n]);
            return Ok(chunk);
        }
    }
    Err(E::InvalidDirectory.into())
}
impl Index {
    pub fn open(source: &mut (impl Read + Seek), engine: &Engine) -> Result<Self, S> {
        // Retained owner allocations coexist with the index during opening.
        let limits = available_limits(engine);
        let config = engine.config.with_limits(limits);
        let length = source.seek(SeekFrom::End(0))?;
        let mut signature = [0; 5];
        read_at(source, 0, &mut signature)?;
        if signature[..4] != [0x91, 10, 66, 82] {
            return Err(E::InvalidSignature.into());
        }
        if signature[4] & 3 != 0 {
            return Err(E::UnsupportedVersion(signature[4] & 3).into());
        }
        let header = ContainerHeader {
            flags: signature[4],
        };
        if !header.has_footer() {
            return Err(S::CentralDirectoryRequired);
        }
        // Footer numbers are reversed on the wire. Read exactly their bounded suffix.
        let mut reversed = [0; 18];
        let mut used = 0;
        let mut values = [0; 2];
        for value in &mut values {
            let start = used;
            loop {
                if used - start == 9 || length <= used as u64 + 5 {
                    return Err(E::InvalidFooter.into());
                }
                read_at(
                    source,
                    length - used as u64 - 1,
                    &mut reversed[used..used + 1],
                )?;
                used += 1;
                if reversed[used - 1] < 128 {
                    break;
                }
            }
            *value = wire::number(&reversed[start..used], &mut 0)?.ok_or(E::InvalidFooter)?;
        }
        let tag_offset = length
            .checked_sub(used as u64 + 1)
            .ok_or(E::InvalidFooter)?;
        let mut tag = [0];
        read_at(source, tag_offset, &mut tag)?;
        if tag[0] != 10 || (values[1] != 0 && values[1] != length) {
            return Err(E::InvalidFooter.into());
        }
        let mut footer_offset = None;
        for n in 1..=9 {
            let Some(offset) = tag_offset.checked_sub(n) else {
                break;
            };
            if offset < 5 {
                break;
            }
            let mut bytes = [0; 9];
            read_at(source, offset, &mut bytes[..n as usize])?;
            let mut cursor = 0;
            if let Ok(Some(size)) = wire::number(&bytes[..n as usize], &mut cursor)
                && cursor == n as usize
                && size == used as u64 + 1
            {
                footer_offset = Some(offset);
                break;
            }
        }
        let footer_offset = footer_offset.ok_or(E::InvalidFooter)?;
        let directory_offset = values[0];
        if directory_offset == 0 {
            return Err(S::CentralDirectoryRequired);
        }
        if !(5..footer_offset).contains(&directory_offset) {
            return Err(E::InvalidDirectory.into());
        }
        let directory = read_header(source, directory_offset, footer_offset, limits)?;
        let directory_end = directory_offset
            .checked_add(directory.length)
            .ok_or(E::SizeOverflow)?;
        if directory.kind != ChunkType::CentralDirectory || directory_end > footer_offset {
            return Err(E::InvalidDirectory.into());
        }
        let footer = ContainerFooter {
            offset: ChunkOffset(footer_offset),
            file_size: (values[1] != 0).then_some(values[1]),
            directory: Some(ChunkOffset(directory_offset)),
        };
        let mut index = Self {
            header,
            footer,
            infos: Vec::new(),
            metadata: Vec::new(),
            chunks: Vec::new(),
            resources: Vec::new(),
            owners: Vec::new(),
            bytes: directory.header_bytes.capacity(),
            config,
        };
        let limits = config.limits();
        let mut accepted = 5_u64
            .checked_add(directory.length)
            .and_then(|n| n.checked_add(length - footer_offset))
            .ok_or(E::SizeOverflow)?;
        check(accepted, limits.max_input_bytes, L::InputBytes)?;
        storage(0, directory.header_bytes.capacity(), limits)?;
        source.seek(SeekFrom::Start(
            directory_offset + directory.header_bytes.len() as u64,
        ))?;
        let mut left = directory.length - directory.header_bytes.len() as u64;
        let repeated = read_number(source, &mut left)?;
        let mut previous_end = 5;
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
        check(index.chunks.len() as u64 + 2, limits.max_chunks, L::Chunks)?;
        let mut chunks = index.chunks.len() as u64 + 2;
        let mut start = 5;
        for chunk in &index.chunks {
            index.padding(source, start, chunk.offset.0, &mut chunks, &mut accepted)?;
            start = chunk.offset.0 + chunk.length;
        }
        index.padding(source, start, directory_offset, &mut chunks, &mut accepted)?;
        index.padding(
            source,
            directory_end,
            footer_offset,
            &mut chunks,
            &mut accepted,
        )?;
        index.structure(repeated)?;
        index.bytes -= directory.header_bytes.capacity();
        index.config = engine.config;
        Ok(index)
    }
    fn padding(
        &self,
        source: &mut (impl Read + Seek),
        mut start: u64,
        end: u64,
        chunks: &mut u64,
        accepted: &mut u64,
    ) -> Result<(), S> {
        let mut limits = self.config.limits();
        limits.max_framing_bytes = limits
            .max_framing_bytes
            .map(|n| n.saturating_sub(self.bytes));
        limits.max_workspace_bytes = limits
            .max_workspace_bytes
            .map(|n| n.saturating_sub(self.bytes));
        // Gaps may contain padding only. Read their bounded headers and skip
        // their bodies, so omitted content and the all-chunk limit are checked
        // without scanning resource payload or allocating padding storage.
        while start < end {
            *chunks = chunks.checked_add(1).ok_or(E::SizeOverflow)?;
            check(*chunks, limits.max_chunks, L::Chunks)?;
            let chunk = read_header(source, start, end, limits)?;
            let next = start.checked_add(chunk.length).ok_or(E::SizeOverflow)?;
            if chunk.kind != ChunkType::Padding || next > end {
                return Err(E::InvalidDirectory.into());
            }
            *accepted = plus(*accepted, chunk.header_bytes.len())?;
            check(*accepted, limits.max_input_bytes, L::InputBytes)?;
            start = next;
        }
        Ok(())
    }
    fn structure(&mut self, repeated: u64) -> Result<(), E> {
        let mut partial = false;
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
            if c.codec == Some(Codec::KeepDecoder)
                && (i == 0
                    || self.chunks[i - 1].codec == Some(Codec::Uncompressed)
                    || kind == ChunkType::RepeatMetadata
                        && previous != Some(ChunkType::RepeatMetadata))
            {
                return Err(E::InvalidChunk);
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
                    self.infos.push(ResourceInfo {
                        index: ResourceIndex(self.infos.len() as u64),
                        hidden: c.flags & 1 != 0,
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
        // Include earlier partial fragments to enforce whole-resource limits.
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
            if index.chunks[i].codec == Some(Codec::KeepDecoder) {
                plan[i - 1].decode = true;
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
                    original.flags & 1 != 0,
                )?;
                engine
                    .resources
                    .last_mut()
                    .ok_or(E::InvalidState)?
                    .header
                    .index = ResourceIndex(owner as u64);
            }
        }
        // A skipped interval cannot contribute codec state. The backward closure
        // guarantees that KeepDecoder always has its actual preceding content.
        if original.codec != Some(Codec::KeepDecoder) {
            engine.codec_active = false;
            engine.codec_ended = false;
        }
        engine.previous = slot.checked_sub(1).map(|i| index.chunks[i].kind);
        engine.chunk_left = original.length - matched as u64;
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
        self.position = original.offset.0 + matched as u64;
        self.input_start = 0;
        self.input_end = 0;
        self.active = Some(slot);
        Ok(())
    }
    fn finish(&mut self, index: &Index, slot: usize) -> Result<(), S> {
        let engine = &mut self.owner.engine;
        let chunk = &index.chunks[slot];
        let continues = index
            .chunks
            .get(slot + 1)
            .is_some_and(|c| c.codec == Some(Codec::KeepDecoder));
        if engine.codec_active && engine.codec_ended == continues {
            return Err(E::InvalidChunk.into());
        }
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
                    self.owner.backend,
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_and_host_backends_agree_on_seek_payload() {
        let bytes = b"\x91\x0aBR\x04\x0d\x02\x02\x05\x00\x0b\x02\x80hello\x03\x09\x09\x00\x05\x05\x0d\x02\x02\x05\x00\x03\x0a\x00\x13";
        for backend in ::core::iter::once(Backend::SCALAR).chain(Backend::available()) {
            let mut d = FramedDecompressor::builder(Default::default())
                .with_backend(backend)
                .build()
                .unwrap();
            let mut reader = d.framed_seek_reader(std::io::Cursor::new(bytes)).unwrap();
            let mut output = Vec::new();
            reader
                .resource(ResourceIndex(0))
                .unwrap()
                .read_to_end(&mut output)
                .unwrap();
            assert_eq!(output, b"hello");
        }
    }
}

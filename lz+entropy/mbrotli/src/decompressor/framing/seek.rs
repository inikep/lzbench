use super::{core::seek as core, *};
use std::io::{Read, Seek};

/// Failure opening an indexed container or reading an indexed resource.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FramedSeekError {
    /// Original source I/O failure.
    #[error("framed source failed: {0}")]
    Io(#[from] std::io::Error),
    /// Typed framing, codec, dictionary or policy failure.
    #[error("{0}")]
    Decode(#[from] FramedDecodeError),
    /// Random access requires a full container with a central directory.
    #[error("central directory is required for random access")]
    CentralDirectoryRequired,
    /// No logical resource has this index.
    #[error("resource index is out of range")]
    ResourceNotFound,
    /// An original header differs from its directory copy.
    #[error("central directory entry does not match the original chunk")]
    DirectoryMismatch,
}

/// Structural properties of a logical resource, without decoding its payload.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResourceInfo {
    pub(super) index: ResourceIndex,
    pub(super) hidden: bool,
    pub(super) checksum: Option<crate::framing::DictionaryId>,
    pub(super) decoded_size: Option<u64>,
}
impl ResourceInfo {
    /// Canonical zero-based identity, including hidden resources.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let info = &reader.resources()[0];
    /// assert_eq!(info.index(), ResourceIndex(0));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn index(&self) -> ResourceIndex {
        self.index
    }
    /// Whether the resource carries the hidden flag.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let info = &reader.resources()[0];
    /// assert!(!info.hidden());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn hidden(&self) -> bool {
        self.hidden
    }
    /// Declared checksum; this decoder does not authenticate checksums.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let info = &reader.resources()[0];
    /// // This resource declares no checksum.
    /// assert_eq!(info.checksum(), None);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn checksum(&self) -> Option<crate::framing::DictionaryId> {
        self.checksum
    }
    /// Size derived from copied headers, validated when payload is read.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let info = &reader.resources()[0];
    /// assert_eq!(info.decoded_size(), Some(3));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn decoded_size(&self) -> Option<u64> {
        self.decoded_size
    }
}

/// Indexed RFC 9841 container over an exclusively owned seekable source.
///
/// Opening validates the footer and directory structure without decompressing
/// payload or metadata. Each access validates the original headers and codec
/// invariants on its dependency path. Untouched resources remain unvalidated;
/// use the sequential framed decoder for full object validation.
/// Source offsets are relative to byte zero; the source must contain one object.
///
/// # Examples
/// ```
/// use mbrotli::framing::*;
/// use std::io::{Cursor, Read};
/// // One uncompressed resource and its central directory.
/// let bytes = b"\x91\x0aBR\x04\x04\x02\x00\x00x\x08\x09\x00\x05\x04\x04\x02\x00\x00\x03\x0a\x00\x0a";
/// let mut decoder = FramedDecompressor::new(Default::default())?;
/// let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
/// assert_eq!(reader.resources()[0].decoded_size(), Some(1));
/// let mut output = Vec::new();
/// reader.resource(ResourceIndex(0))?.read_to_end(&mut output)?;
/// assert_eq!(output, b"x");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct FramedSeekReader<'d, 'dict, R> {
    source: R,
    owner: core::Lease<'d>,
    resolver: Option<DictionaryResolverRef<'dict>>,
    index: core::Index,
}
impl FramedDecompressor {
    /// Reads the footer and central directory without decoding resource payload.
    /// # Errors
    /// Returns source, structural, budget or abandoned-session errors. Containers
    /// without a usable directory return `CentralDirectoryRequired`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let source = Cursor::new(bytes);
    /// let mut reader = decoder.framed_seek_reader(source)?;
    /// assert_eq!(reader.resources().len(), 1);
    /// let mut payload = Vec::new();
    /// reader.resource(ResourceIndex(0))?.read_to_end(&mut payload)?;
    /// assert_eq!(payload, b"abc");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn framed_seek_reader<R: Read + Seek>(
        &mut self,
        source: R,
    ) -> Result<FramedSeekReader<'_, 'static, R>, FramedSeekError> {
        FramedSeekReader::open(self, None, source)
    }
    /// Opens indexed framing with the existing external dictionary resolver.
    /// Dictionaries are resolved lazily at codec attachment boundaries.
    /// # Errors
    /// As [`Self::framed_seek_reader`]. Dictionary errors occur during access.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04(\x02\x03\x00\x01\x02\x03\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x00;,\x09\x00\x05((\x02\x03\x00\x01\x02\x03\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x07\x00\x03\x0a\x00.";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// struct Dictionaries;
    /// impl DictionaryResolver for Dictionaries {
    ///     fn resolve(&self, request: ExternalDictionaryRequest) -> Option<&[u8]> {
    ///         (request.id == DictionaryId([7; 32])
    ///             && request.kind == ExternalDictionaryKind::Prefix)
    ///             .then_some(b"shared prefix".as_slice())
    ///     }
    /// }
    /// let dictionaries = Dictionaries;
    /// let mut reader = decoder.framed_seek_reader_with_dictionaries(
    ///     &dictionaries,
    ///     Cursor::new(bytes),
    /// )?;
    /// // The fixture's empty SharedBrotli resource requests the prefix above.
    /// let mut payload = Vec::new();
    /// reader.resource(ResourceIndex(0))?.read_to_end(&mut payload)?;
    /// assert!(payload.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn framed_seek_reader_with_dictionaries<'d, 'dict, R: Read + Seek>(
        &'d mut self,
        dictionaries: impl Into<DictionaryResolverRef<'dict>>,
        source: R,
    ) -> Result<FramedSeekReader<'d, 'dict, R>, FramedSeekError> {
        FramedSeekReader::open(self, Some(dictionaries.into()), source)
    }
}
impl<'d, 'dict, R: Read + Seek> FramedSeekReader<'d, 'dict, R> {
    fn open(
        owner: &'d mut FramedDecompressor,
        resolver: Option<DictionaryResolverRef<'dict>>,
        mut source: R,
    ) -> Result<Self, FramedSeekError> {
        // Reuse the owner's lifecycle check and normal cancellation/retention.
        drop(owner.start(Default::default())?);
        let index = core::Index::open(&mut source, &owner.engine)?;
        owner.active = true;
        Ok(Self {
            source,
            owner: core::Lease { owner },
            resolver,
            index,
        })
    }
    /// Container header; payload has not necessarily been validated.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// assert!(reader.header().has_footer());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn header(&self) -> ContainerHeader {
        self.index.header
    }
    /// Structurally validated terminal footer.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let footer = reader.footer();
    /// assert_eq!(footer.directory, Some(ChunkOffset(12)));
    /// assert_eq!(footer.file_size, None);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn footer(&self) -> ContainerFooter {
        self.index.footer
    }
    /// Logical resources in wire order, including hidden and empty resources.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let resources = reader.resources();
    /// assert_eq!(resources.len(), 1);
    /// for info in resources {
    ///     println!("resource {:?}: {:?} bytes", info.index(), info.decoded_size());
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn resources(&self) -> &[ResourceInfo] {
        &self.index.infos
    }
    /// Structural information, or `None` for an invalid index.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let info = reader.resource_info(ResourceIndex(0)).unwrap();
    /// assert_eq!(info.decoded_size(), Some(3));
    /// assert!(reader.resource_info(ResourceIndex(1)).is_none());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn resource_info(&self, index: ResourceIndex) -> Option<&ResourceInfo> {
        usize::try_from(index.0)
            .ok()
            .and_then(|i| self.index.infos.get(i))
    }
    /// Lazily decodes and caches original resource metadata.
    /// # Errors
    /// Returns an invalid-index, I/O, header, dictionary, codec or limit error.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x01\x00id\x01x\x06\x02\x00\x00abc\x06\x06\x00AA\x01z\x12\x09\x00\x05\x03\x06\x01\x00\x0c\x04\x06\x02\x00\x00\x13\x03\x06\x06\x00\x03\x0a\x00\x1a";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let index = ResourceIndex(0);
    /// let metadata = reader.resource_metadata(index)?.unwrap();
    /// assert_eq!(metadata.name(), Some("x"));
    /// // A later request reuses the cached metadata.
    /// assert_eq!(reader.resource_metadata(index)?.unwrap().name(), Some("x"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn resource_metadata(
        &mut self,
        index: ResourceIndex,
    ) -> Result<Option<&Metadata>, FramedSeekError> {
        self.metadata(index, false)
    }
    /// Lazily decodes and caches original footer metadata.
    /// # Errors
    /// As [`Self::resource_metadata`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x01\x00id\x01x\x06\x02\x00\x00abc\x06\x06\x00AA\x01z\x12\x09\x00\x05\x03\x06\x01\x00\x0c\x04\x06\x02\x00\x00\x13\x03\x06\x06\x00\x03\x0a\x00\x1a";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let metadata = reader.resource_footer_metadata(ResourceIndex(0))?.unwrap();
    /// assert_eq!(metadata.as_bytes(), b"AA\x01z");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn resource_footer_metadata(
        &mut self,
        index: ResourceIndex,
    ) -> Result<Option<&Metadata>, FramedSeekError> {
        self.metadata(index, true)
    }
    fn metadata(
        &mut self,
        index: ResourceIndex,
        footer: bool,
    ) -> Result<Option<&Metadata>, FramedSeekError> {
        let slot = self.index.metadata_chunk(index, footer)?;
        let Some(slot) = slot else { return Ok(None) };
        if self.index.metadata[slot].is_none() {
            let mut operation =
                core::Operation::new(self.owner.owner, &self.index, slot..slot + 1)?;
            let result = operation.metadata(&mut self.source, &self.index, self.resolver, slot);
            drop(operation);
            self.index.cache_metadata(slot, result?)?;
        }
        Ok(self.index.metadata[slot].as_ref())
    }
    /// Starts a fresh streaming decode operation for one logical resource.
    ///
    /// Partial chunks and dictionary/continuation dependencies are handled
    /// internally. Dropping the returned reader early performs no I/O and leaves
    /// this reader reusable. Bytes remain provisional until EOF validates the
    /// entire selected payload. Independent calls reset aggregate decode counters.
    /// # Errors
    /// Returns `ResourceNotFound` or a dependency/planning/budget error. Source,
    /// codec and dictionary failures during `Read` retain a `FramedSeekError`
    /// inside `std::io::Error`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let index = ResourceIndex(0);
    /// {
    ///     let mut resource = reader.resource(index)?;
    ///     let mut prefix = [0; 1];
    ///     resource.read_exact(&mut prefix)?;
    ///     assert_eq!(&prefix, b"a");
    /// } // Dropping before EOF cancels this operation without I/O.
    ///
    /// // Opening it again starts at the beginning of the logical resource.
    /// let mut resource = reader.resource(index)?;
    /// let mut destination = Vec::new();
    /// std::io::copy(&mut resource, &mut destination)?;
    /// assert_eq!(destination, b"abc");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn resource(
        &mut self,
        index: ResourceIndex,
    ) -> Result<ResourceReader<'_, 'dict, R>, FramedSeekError> {
        let range = self.index.resource_range(index)?;
        let info = &self.index.infos
            [usize::try_from(index.0).map_err(|_| FramedSeekError::ResourceNotFound)?];
        let operation = core::Operation::new(self.owner.owner, &self.index, range)?;
        Ok(ResourceReader {
            source: &mut self.source,
            index: &self.index,
            info,
            resolver: self.resolver,
            operation,
            total_out: 0,
            failed: false,
        })
    }
    /// Inspects the source; its physical cursor position is unspecified.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let source = reader.get_ref();
    /// assert_eq!(*source.get_ref(), bytes);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn get_ref(&self) -> &R {
        &self.source
    }
    /// Accesses the source for inspection or source-specific controls.
    ///
    /// Moving the underlying source position or modifying source contents may
    /// invalidate the seek reader's internal assumptions. Avoid seeking or writing
    /// through this reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let source: &mut Cursor<_> = reader.get_mut();
    /// // Inspect the backing bytes without moving the cursor or modifying content.
    /// assert_eq!(*source.get_ref(), bytes);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn get_mut(&mut self) -> &mut R {
        &mut self.source
    }
    /// Returns the source at its unspecified physical position, with no I/O.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let source = reader.into_inner();
    /// assert_eq!(source.into_inner(), bytes);
    /// // The decoder's borrow has ended; it can open another container.
    /// let another_reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// assert_eq!(another_reader.resources().len(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_inner(self) -> R {
        self.source
    }
}

/// Streaming decoded payload of one indexed logical resource.
///
/// Implements `Read`, but not decoded `Seek`. Drop cancels local codec state
/// without I/O. Metadata and dependency payload never appear in this stream.
#[derive(Debug)]
pub struct ResourceReader<'a, 'dict, R> {
    source: &'a mut R,
    index: &'a core::Index,
    info: &'a ResourceInfo,
    resolver: Option<DictionaryResolverRef<'dict>>,
    operation: core::Operation<'a>,
    total_out: u64,
    failed: bool,
}
impl<R: Read + Seek> ResourceReader<'_, '_, R> {
    /// Canonical identity of the selected resource.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let resource = reader.resource(ResourceIndex(0))?;
    /// assert_eq!(resource.index(), ResourceIndex(0));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn index(&self) -> ResourceIndex {
        self.info.index
    }
    /// Directory-derived structural information.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let resource = reader.resource(ResourceIndex(0))?;
    /// assert_eq!(resource.info().decoded_size(), Some(3));
    /// assert!(!resource.info().hidden());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn info(&self) -> &ResourceInfo {
        self.info
    }
    /// Bytes successfully delivered to this reader's caller.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let mut resource = reader.resource(ResourceIndex(0))?;
    /// assert_eq!(resource.total_out(), 0);
    /// resource.read_exact(&mut [0; 1])?;
    /// assert_eq!(resource.total_out(), 1);
    /// std::io::copy(&mut resource, &mut std::io::sink())?;
    /// assert_eq!(resource.total_out(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn total_out(&self) -> u64 {
        self.total_out
    }
}
impl<R: Read + Seek> Read for ResourceReader<'_, '_, R> {
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::{Cursor, Read};
    /// # let bytes = b"\x91\x0aBR\x04\x06\x02\x00\x00abc\x08\x09\x00\x05\x04\x06\x02\x00\x00\x03\x0a\x00\x0c";
    /// # let mut decoder = FramedDecompressor::new(Default::default())?;
    /// # let mut reader = decoder.framed_seek_reader(Cursor::new(bytes))?;
    /// let mut resource = reader.resource(ResourceIndex(0))?;
    /// let mut buffer = [0; 2];
    /// let mut payload = Vec::new();
    /// loop {
    ///     let n = resource.read(&mut buffer)?;
    ///     if n == 0 {
    ///         break;
    ///     }
    ///     payload.extend_from_slice(&buffer[..n]);
    /// }
    /// assert_eq!(payload, b"abc");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        if self.failed {
            return Err(std::io::Error::other(FramedSeekError::Decode(
                FramedDecodeError::InvalidState,
            )));
        }
        match self
            .operation
            .read(self.source, self.index, self.resolver, output)
        {
            Ok(n) => {
                self.total_out += n as u64;
                Ok(n)
            }
            Err(error) => {
                self.failed = true;
                Err(std::io::Error::other(error))
            }
        }
    }
}

use crate::framing::{DictionaryId, DictionaryReference, MetadataField, MetadataKind};
use ::core::ops::Range;
use alloc::vec::Vec;

/// Zero-based logical resource index, including hidden resources.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourceIndex(pub u64);
/// Wire offset relative to this container's first signature byte.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ChunkOffset(pub u64);
/// Validated main header. Reserved flag bits are preserved.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContainerHeader {
    /// Original container flags.
    pub flags: u8,
}
impl ContainerHeader {
    /// Whether the full profile requires a final footer.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// assert!(ContainerHeader { flags: 4 }.has_footer());
    /// assert!(!ContainerHeader { flags: 0 }.has_footer());
    /// ```
    pub const fn has_footer(self) -> bool {
        self.flags & 4 != 0
    }
}
/// Content encoding from the wire.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Codec {
    /// Literal content.
    Uncompressed,
    /// Continue the preceding compressed stream.
    KeepDecoder,
    /// RFC 7932 stream, without extended window headers.
    Brotli,
    /// RFC 9841 stream and explicit dictionary references.
    SharedBrotli,
}
/// Defined chunk types.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ChunkType {
    /// Zero padding.
    Padding = 0,
    /// Metadata for the next resource.
    Metadata = 1,
    /// Complete resource.
    Data = 2,
    /// First resource fragment.
    FirstPartial = 3,
    /// Intermediate resource fragment.
    MiddlePartial = 4,
    /// Last resource fragment.
    LastPartial = 5,
    /// Metadata for the preceding resource.
    FooterMetadata = 6,
    /// Container metadata.
    GlobalMetadata = 7,
    /// Copy of earlier resource metadata.
    RepeatMetadata = 8,
    /// Exact header index.
    CentralDirectory = 9,
    /// Terminal footer.
    Footer = 10,
}
/// Original, validated chunk header; payload bytes are not duplicated here.
#[derive(Debug, Eq, PartialEq)]
pub struct ChunkInfo {
    /// Original wire offset.
    pub offset: ChunkOffset,
    /// Total wire size including the initial varint.
    pub length: u64,
    /// Chunk kind.
    pub kind: ChunkType,
    /// Content codec, where present.
    pub codec: Option<Codec>,
    /// Declared decoded size, where present.
    pub declared_size: Option<u64>,
    /// Dictionary attachments in wire order.
    pub dictionaries: Vec<DictionaryReference>,
    /// Original header bytes, including nonminimal varints.
    pub header_bytes: Vec<u8>,
    /// Resource flags, when applicable.
    pub flags: u8,
    /// Supplied checksum, not authenticated by this API.
    pub checksum: Option<DictionaryId>,
    /// Original metadata kind for a repeated chunk.
    pub repeated_kind: Option<MetadataKind>,
}
/// Lossless decoded metadata with a validated field index.
#[derive(Debug, Eq, PartialEq)]
pub struct Metadata {
    /// Chunk containing this serialization.
    pub source: ChunkOffset,
    pub(super) bytes: Vec<u8>,
    pub(super) fields: Vec<([u8; 2], Range<usize>)>,
}
impl Metadata {
    /// Original decoded field serialization, suitable for dictionary content.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = b"\x91\x0aBR\x04\x06\x01\x00id\x01x\x03\x02\x00\x00\x03\x0a\x00\x00";
    /// let output = decoder.decompress(input)?;
    /// let metadata = output.resources[0].metadata.as_ref().unwrap();
    /// assert_eq!(metadata.as_bytes(), b"id\x01x");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Iterates fields in original order, preserving custom-code repetitions.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = b"\x91\x0aBR\x04\x06\x01\x00id\x01x\x03\x02\x00\x00\x03\x0a\x00\x00";
    /// let output = decoder.decompress(input)?;
    /// let metadata = output.resources[0].metadata.as_ref().unwrap();
    /// let fields: Vec<_> = metadata.fields().collect();
    /// assert_eq!(fields[0].code, *b"id");
    /// assert_eq!(fields[0].value, b"x");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn fields(&self) -> impl Iterator<Item = MetadataField<'_>> {
        self.fields.iter().map(|(code, range)| MetadataField {
            code: *code,
            value: &self.bytes[range.clone()],
        })
    }
    /// Optional UTF-8 resource name, without path normalization.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = b"\x91\x0aBR\x04\x06\x01\x00id\x01x\x03\x02\x00\x00\x03\x0a\x00\x00";
    /// let output = decoder.decompress(input)?;
    /// let metadata = output.resources[0].metadata.as_ref().unwrap();
    /// assert_eq!(metadata.name(), Some("x"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn name(&self) -> Option<&str> {
        self.fields()
            .find(|field| field.code == *b"id")
            .and_then(|field| ::core::str::from_utf8(field.value).ok())
    }
    /// Optional signed microseconds since the Unix epoch.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = b"\x91\x0aBR\x04\x06\x01\x00id\x01x\x03\x02\x00\x00\x03\x0a\x00\x00";
    /// let output = decoder.decompress(input)?;
    /// let metadata = output.resources[0].metadata.as_ref().unwrap();
    /// assert_eq!(metadata.timestamp(), None); // This record has no mt field.
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn timestamp(&self) -> Option<i64> {
        self.fields()
            .find(|field| field.code == *b"mt")
            .and_then(|field| field.value.try_into().ok())
            .map(i64::from_le_bytes)
    }
}
/// Wire identity of a resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceSource {
    /// Normalized raw member, with no invented framing offset.
    Raw,
    /// Framed resource's first data chunk.
    Framed {
        /// First data chunk offset.
        first_chunk: ChunkOffset,
    },
}
/// One logical resource, with owned payload or a range in caller storage.
#[derive(Debug, Eq, PartialEq)]
pub struct DecodedResource<B> {
    /// Logical resource index.
    pub index: ResourceIndex,
    /// Wire identity.
    pub source: ResourceSource,
    /// Whether implicit extraction is discouraged.
    pub hidden: bool,
    /// Metadata preceding the data.
    pub metadata: Option<Metadata>,
    /// Payload storage.
    pub data: B,
    /// Metadata following the data.
    pub footer_metadata: Option<Metadata>,
    /// Supplied whole-resource checksum, without authentication.
    pub checksum: Option<DictionaryId>,
}
/// A repeated metadata record and its original wire identity.
#[derive(Debug, Eq, PartialEq)]
pub struct RepeatedMetadata {
    /// Original metadata chunk.
    pub original: ChunkOffset,
    /// Original metadata kind.
    pub kind: MetadataKind,
    /// Repeated fields, which can be a consistent subset.
    pub metadata: Metadata,
}
/// Entry referencing an existing exact chunk-header record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectoryEntry {
    /// Index into `FramedLayout::chunks`.
    pub chunk_index: usize,
}
/// Central directory information.
#[derive(Debug, Eq, PartialEq)]
pub struct CentralDirectory {
    /// Directory wire offset.
    pub offset: ChunkOffset,
    /// Beginning of the repeated metadata series.
    pub repeated_metadata: Option<ChunkOffset>,
    /// Verified entries in original order.
    pub entries: Vec<DirectoryEntry>,
}
/// Validated footer fields; no checksum verification is implied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContainerFooter {
    /// Footer wire offset.
    pub offset: ChunkOffset,
    /// Optional declared file size; wire zero becomes `None`.
    pub file_size: Option<u64>,
    /// Optional central directory offset.
    pub directory: Option<ChunkOffset>,
}
/// Physical structure without a second copy of resource payload.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct FramedLayout {
    /// Every wire chunk including padding and footer.
    pub chunks: Vec<ChunkInfo>,
    /// Repeated metadata in wire order.
    pub repeated_metadata: Vec<RepeatedMetadata>,
    /// Optional verified central directory.
    pub directory: Option<CentralDirectory>,
    /// Required footer in the full profile.
    pub footer: Option<ContainerFooter>,
}
/// Distinguishes actual framing from a normalized raw member.
#[derive(Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum OutputStructure {
    /// No framing structures were present.
    Raw,
    /// Actual container structures.
    Framed {
        /// Main header.
        header: ContainerHeader,
        /// Validated wire layout.
        layout: FramedLayout,
    },
}
/// Fully validated structured result, independent of decoder and source lifetimes.
#[derive(Debug, Eq, PartialEq)]
pub struct FramedOutput<B = Vec<u8>> {
    /// Actual top-level format.
    pub structure: OutputStructure,
    /// Resources in wire order.
    pub resources: Vec<DecodedResource<B>>,
    /// Global metadata in wire order.
    pub global_metadata: Vec<Metadata>,
}
/// Selected input format, available after detection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamInfo {
    /// Raw Brotli member.
    Raw,
    /// Framing main header.
    Framed(ContainerHeader),
}
/// Identity and initial properties of a resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceHeader {
    /// Logical index.
    pub index: ResourceIndex,
    /// Wire identity.
    pub source: ResourceSource,
    /// Visibility flag.
    pub hidden: bool,
}
/// Resource-local position of one delivered fragment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceDataPosition {
    /// Resource receiving the fragment.
    pub resource: ResourceIndex,
    /// Wire chunk, absent only for raw input.
    pub chunk: Option<ChunkOffset>,
    /// Beginning offset in decoded resource bytes.
    pub offset: u64,
}
/// Borrowed view into the caller's output slice.
#[derive(Debug)]
pub struct ResourceData<'a> {
    /// Resource-local position.
    pub position: ResourceDataPosition,
    /// Exactly the produced output prefix.
    pub bytes: &'a [u8],
}
/// Locally completed resource payload; later container checks can still fail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceSummary {
    /// Resource index.
    pub index: ResourceIndex,
    /// Decoded resource length.
    pub size: u64,
    /// Supplied checksum.
    pub checksum: Option<DictionaryId>,
}
/// Locally completed chunk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkSummary {
    /// Chunk identity.
    pub offset: ChunkOffset,
    /// Regenerated content length.
    pub decoded_size: u64,
}
/// Metadata's semantic target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetadataScope {
    /// Before-resource metadata.
    Resource(ResourceIndex),
    /// After-resource metadata.
    Footer(ResourceIndex),
    /// Container metadata.
    Global,
}
/// Metadata event, including explicit repeated origin when applicable.
#[derive(Debug)]
pub struct MetadataEvent<'a> {
    /// Logical target.
    pub scope: MetadataScope,
    /// Original chunk for a repeat, otherwise absent.
    pub original: Option<ChunkOffset>,
    /// Validated, lossless metadata.
    pub metadata: &'a Metadata,
}
/// Beginning of a directory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectoryHeader {
    /// Wire location.
    pub offset: ChunkOffset,
    /// Optional repeated-series location.
    pub repeated_metadata: Option<ChunkOffset>,
}
/// Padding description, without allocating its bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PaddingInfo {
    /// Wire location.
    pub offset: ChunkOffset,
    /// Total wire length.
    pub length: u64,
}
/// One incremental semantic event. Borrowed events prevent session mutation.
#[derive(Debug)]
#[non_exhaustive]
pub enum FramedEvent<'a> {
    /// Input format selected.
    StreamStart(StreamInfo),
    /// One original header accepted.
    ChunkHeader(&'a ChunkInfo),
    /// Local chunk validation completed.
    ChunkEnd(ChunkSummary),
    /// New logical resource.
    ResourceStart(ResourceHeader),
    /// Resource payload in caller output.
    ResourceData(ResourceData<'a>),
    /// Payload complete, before potential footer metadata.
    ResourceDataEnd(ResourceSummary),
    /// Complete metadata content validated.
    Metadata(MetadataEvent<'a>),
    /// Directory begins.
    DirectoryStart(DirectoryHeader),
    /// One verified directory entry.
    DirectoryEntry(&'a DirectoryEntry),
    /// Directory completed.
    DirectoryEnd,
    /// Padding validated.
    Padding(PaddingInfo),
    /// Footer validated.
    Footer(ContainerFooter),
}
/// Explicit external dictionary interpretation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalDictionaryKind {
    /// Literal prefix bytes, never sniffed as serialized content.
    Prefix,
    /// Serialized RFC 9841 dictionary.
    Serialized,
}
/// One dictionary lookup at an attachment boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExternalDictionaryRequest {
    /// Caller-managed identity.
    pub id: DictionaryId,
    /// Required interpretation.
    pub kind: ExternalDictionaryKind,
}
/// Immutable dictionary lookup. No `Send` or `Sync` requirement is imposed.
pub trait DictionaryResolver {
    /// Returns stable bytes for the operation, or `None` for a missing mapping.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// struct Prefix(Vec<u8>);
    /// impl DictionaryResolver for Prefix {
    ///     fn resolve(&self, request: ExternalDictionaryRequest) -> Option<&[u8]> {
    ///         (request.id == DictionaryId([7; 32])
    ///             && request.kind == ExternalDictionaryKind::Prefix)
    ///             .then_some(self.0.as_slice())
    ///     }
    /// }
    /// let dictionary = Prefix(b"shared words".to_vec());
    /// let request = ExternalDictionaryRequest {
    ///     id: DictionaryId([7; 32]), kind: ExternalDictionaryKind::Prefix,
    /// };
    /// assert_eq!(dictionary.resolve(request), Some(b"shared words".as_slice()));
    /// ```
    fn resolve(&self, request: ExternalDictionaryRequest) -> Option<&[u8]>;
}

impl<T: DictionaryResolver + ?Sized> DictionaryResolver for &T {
    /// Resolves through the borrowed resolver, so a `&'static` one can be owned.
    fn resolve(&self, request: ExternalDictionaryRequest) -> Option<&[u8]> {
        (**self).resolve(request)
    }
}
impl<T: DictionaryResolver + ?Sized> DictionaryResolver for alloc::boxed::Box<T> {
    /// Resolves through the boxed resolver.
    fn resolve(&self, request: ExternalDictionaryRequest) -> Option<&[u8]> {
        (**self).resolve(request)
    }
}
#[cfg(target_has_atomic = "ptr")]
impl<T: DictionaryResolver + ?Sized> DictionaryResolver for alloc::sync::Arc<T> {
    /// Resolves through the shared resolver, so sessions can share one without copying.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use mbrotli::framing::*;
    /// struct Prefix;
    /// impl DictionaryResolver for Prefix {
    ///     fn resolve(&self, _: ExternalDictionaryRequest) -> Option<&[u8]> { Some(b"prefix") }
    /// }
    /// let shared = Arc::new(Prefix);
    /// let request = ExternalDictionaryRequest {
    ///     id: DictionaryId([7; 32]), kind: ExternalDictionaryKind::Prefix,
    /// };
    /// assert_eq!(shared.resolve(request), Some(&b"prefix"[..]));
    /// # Ok::<(), FramedDecodeError>(())
    /// ```
    fn resolve(&self, request: ExternalDictionaryRequest) -> Option<&[u8]> {
        (**self).resolve(request)
    }
}

/// The resolver type of an owned framed session started without dictionaries.
///
/// It is the default type parameter of
/// [`FramedDecoderSessionOwned`](super::FramedDecoderSessionOwned). A session
/// started by [`into_session`](super::FramedDecompressor::into_session) has no
/// resolver at all, exactly like
/// [`start`](super::FramedDecompressor::start); this type is never consulted
/// there. Used as a resolver in its own right, it resolves nothing.
///
/// # Examples
///
/// ```
/// use mbrotli::framing::*;
/// let request = ExternalDictionaryRequest {
///     id: DictionaryId([7; 32]), kind: ExternalDictionaryKind::Prefix,
/// };
/// assert_eq!(NoDictionaries.resolve(request), None);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct NoDictionaries;

impl DictionaryResolver for NoDictionaries {
    /// Resolves no dictionary.
    fn resolve(&self, _: ExternalDictionaryRequest) -> Option<&[u8]> {
        None
    }
}

/// Borrowed external dictionary lookup accepted by framed decoder entry points.
///
/// A shared reference to any [`DictionaryResolver`] converts automatically.
/// Conversion allocates no storage and the resolver remains borrowed for the
/// session's lifetime.
///
/// # Examples
///
/// ```
/// use mbrotli::framing::*;
/// struct Dictionaries;
/// impl DictionaryResolver for Dictionaries {
///     fn resolve(&self, _: ExternalDictionaryRequest) -> Option<&[u8]> { None }
/// }
/// let dictionaries = Dictionaries;
/// let mut decoder = FramedDecompressor::new(Default::default())?;
/// let session = decoder.start_with_dictionaries(&dictionaries, Default::default())?;
/// assert_eq!(session.total_in(), 0);
/// # Ok::<(), FramedDecodeError>(())
/// ```
#[derive(Clone, Copy)]
pub struct DictionaryResolverRef<'dict>(&'dict dyn DictionaryResolver);

impl<'dict, T: DictionaryResolver> From<&'dict T> for DictionaryResolverRef<'dict> {
    fn from(resolver: &'dict T) -> Self {
        Self(resolver)
    }
}
impl<'dict> DictionaryResolverRef<'dict> {
    pub(super) fn resolve(self, request: ExternalDictionaryRequest) -> Option<&'dict [u8]> {
        self.0.resolve(request)
    }
}
impl ::core::fmt::Debug for DictionaryResolverRef<'_> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        f.debug_struct("DictionaryResolverRef")
            .finish_non_exhaustive()
    }
}

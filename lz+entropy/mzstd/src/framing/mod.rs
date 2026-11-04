//! Structured, seekable containers of Zstandard frames.
//!
//! A container holds ordered resources (payloads), metadata about them,
//! padding, an optional central directory and a footer. On the wire it is
//! nothing but Zstandard frames:
//!
//! - visible resource payload is a sequence of regular Zstandard frames, so
//!   any Zstandard decoder (`zstd -d`, `ZSTD_decompress`) extracts the
//!   concatenation of the visible resources and skips everything else;
//! - the container header, every chunk header, metadata, hidden resources,
//!   padding, the directory and the footer live in skippable frames;
//! - the container ends with a seek table in the layout of zstd's
//!   `contrib/seekable_format`, so seekable decoders map decompressed
//!   offsets to frames without reading the container.
//!
//! Resources are cut into chunks of a configured size; each chunk is an
//! independent Zstandard frame, so any chunk decodes on its own. With
//! `compression`, `FramedCompressor` writes containers; with
//! `decompression`, `FramedDecompressor` validates and decodes them, sequentially or, with std, by random access through a
//! seekable source. The wire format is specified in the repository's
//! `architecture/framing.md`.
//!
//! The types of this module are shared by both sides and need no codec.

pub(crate) mod core;

/// A caller-supplied 256-bit identity, for instance a SHA-256 hash.
///
/// It names an external dictionary for a
/// [`DictionaryReference`], or records a whole-resource checksum. No key
/// or hashing policy is implied, and nothing here computes or verifies it.
/// It is unrelated to the 32-bit
/// [`dictionary::DictionaryId`](crate::dictionary::DictionaryId) a
/// Zstandard-format dictionary declares.
///
/// # Examples
///
/// ```
/// use mzstd::framing::DictionaryId;
/// let id = DictionaryId([7; 32]);
/// assert_eq!(id.0[0], 7);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DictionaryId(pub [u8; 32]);

/// One dictionary source of a chunk, in attachment order.
///
/// A Zstandard frame uses one dictionary. A chunk therefore names either a
/// single serialized (Zstandard-format) dictionary, or one to sixteen
/// prefixes whose contents, concatenated in order, form one raw-content
/// dictionary. Offsets are relative to the first byte of the container.
///
/// # Examples
///
/// ```
/// use mzstd::framing::{DictionaryId, DictionaryReference};
/// let references = [
///     DictionaryReference::PrefixId(DictionaryId([1; 32])),
///     DictionaryReference::PrefixResource(13),
/// ];
/// assert_eq!(references.len(), 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DictionaryReference {
    /// An application-resolved external prefix (raw content).
    PrefixId(DictionaryId),
    /// An application-resolved external Zstandard-format dictionary.
    SerializedId(DictionaryId),
    /// A complete, earlier resource, by the offset of its first chunk,
    /// whose payload is used as a prefix.
    PrefixResource(u64),
    /// A complete, earlier resource, by the offset of its first chunk,
    /// whose payload is a Zstandard-format dictionary.
    SerializedResource(u64),
    /// The decoded content of an earlier individual chunk, used as a prefix.
    PrefixChunk(u64),
}

/// Where metadata applies.
///
/// # Examples
///
/// ```
/// use mzstd::framing::MetadataKind;
/// assert_ne!(MetadataKind::Resource, MetadataKind::Footer);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetadataKind {
    /// Applies to the next resource; permits `id`, `mt`, and uppercase codes.
    Resource,
    /// Applies to the preceding resource; permits uppercase codes only.
    Footer,
    /// Applies to the container; permits uppercase codes only.
    Global,
}

/// One borrowed metadata field. Codes and reserved value shapes are validated.
///
/// A code is two uppercase ASCII letters, which may repeat, or one of the
/// reserved lowercase codes of resource metadata: `id`, a UTF-8 name, and
/// `mt`, an eight-byte little-endian signed timestamp in microseconds since
/// the Unix epoch, each at most once.
///
/// # Examples
///
/// ```
/// use mzstd::framing::MetadataField;
/// let field = MetadataField { code: *b"id", value: b"index.html" };
/// assert_eq!(field.value, b"index.html");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MetadataField<'a> {
    /// Two uppercase ASCII letters, or a recognized lowercase code.
    pub code: [u8; 2],
    /// Raw field content. `id` is UTF-8; `mt` is an eight-byte signed timestamp.
    pub value: &'a [u8],
}

#[cfg(feature = "compression")]
pub use crate::compressor::framing::*;

#[cfg(feature = "decompression")]
pub use crate::decompressor::framing::*;

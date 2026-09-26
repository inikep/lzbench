//! Codec-neutral experimental RFC 9841 framing types.

/// A caller-supplied 256-bit HighwayHash value. No key or hashing policy is implied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DictionaryId(pub [u8; 32]);

/// An explicit dictionary source, in decoder attachment order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictionaryReference {
    /// An application-resolved external prefix dictionary.
    PrefixId(DictionaryId),
    /// An application-resolved serialized dictionary.
    SerializedId(DictionaryId),
    /// A complete, earlier resource containing prefix bytes.
    PrefixResource(u64),
    /// A complete, earlier resource containing a serialized dictionary.
    SerializedResource(u64),
    /// The contents of an earlier individual chunk, used as a prefix.
    PrefixChunk(u64),
}

/// Where metadata applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataKind {
    /// Applies to the next resource; permits `id`, `mt`, and uppercase codes.
    Resource,
    /// Applies to the preceding resource; permits uppercase codes only.
    Footer,
    /// Applies to the container; permits uppercase codes only.
    Global,
}

/// One borrowed metadata field. Codes and reserved value shapes are validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

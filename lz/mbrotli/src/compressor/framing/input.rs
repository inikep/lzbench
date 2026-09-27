use super::{DictionaryReference, MetadataField, MetadataKind, MetadataOptions, ResourceOptions};
use crate::dictionary::PreparedDictionary;
use crate::{InputSize, StreamConfig};

/// Borrowed commands for one complete container, in wire order.
#[derive(Clone, Copy, Debug)]
pub struct FramedInput<'a> {
    /// Resources, metadata and padding; suffix records are generated automatically.
    pub items: &'a [FramedItem<'a>],
    /// None repeats all fields; an empty slice requests empty repeated chunks.
    pub repeat_metadata_fields: Option<&'a [[u8; 2]]>,
}
impl<'a> From<&'a [FramedItem<'a>]> for FramedInput<'a> {
    fn from(items: &'a [FramedItem<'a>]) -> Self {
        Self {
            items,
            repeat_metadata_fields: None,
        }
    }
}
/// A complete structured-input command.
#[derive(Clone, Copy, Debug)]
pub enum FramedItem<'a> {
    /// One complete resource, including hidden or empty resources.
    Resource(FramedResource<'a>),
    /// Metadata with independently configured original and repeated encoding.
    Metadata {
        /// Scope and adjacency of this chunk.
        kind: MetadataKind,
        /// Borrowed ordered fields, preserving multiplicity.
        fields: &'a [MetadataField<'a>],
        /// Original and repeated encoding.
        options: MetadataOptions<'a>,
    },
    /// A zero-filled padding chunk.
    Padding {
        /// Number of content bytes, excluding the chunk header.
        bytes: usize,
    },
}
/// Borrowed payload and explicit per-resource encoding policy.
#[derive(Clone, Copy, Debug)]
pub struct FramedResource<'a> {
    /// Complete payload; never copied into whole-resource staging.
    pub data: &'a [u8],
    /// Visibility and caller-supplied checksum.
    pub options: ResourceOptions,
    /// Raw input size hint and zero stream offset.
    pub stream: StreamConfig,
    /// Resource codec and optional borrowed dictionary.
    pub encoding: ResourceEncoding<'a>,
}
impl<'a> From<&'a [u8]> for FramedResource<'a> {
    fn from(data: &'a [u8]) -> Self {
        Self {
            data,
            options: ResourceOptions::default(),
            stream: InputSize::Exact(data.len() as u64).into(),
            encoding: ResourceEncoding::Brotli,
        }
    }
}
/// Encoding of resource payload, with explicit dictionary attachment order.
#[derive(Clone, Copy, Debug)]
pub enum ResourceEncoding<'a> {
    /// Store bytes verbatim.
    Uncompressed,
    /// Ordinary Brotli, or Shared Brotli without references for Large Window.
    Brotli,
    /// Shared Brotli with a prepared dictionary borrowed for this resource.
    Shared {
        /// Dictionary remains externally owned and is never cloned.
        dictionary: &'a PreparedDictionary,
        /// Container-relative references in attachment order.
        references: &'a [DictionaryReference],
    },
}

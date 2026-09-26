//! Immutable dictionaries for the enabled codecs.
//!
//! Prepared encoder dictionaries require `compression`; decode-only dictionaries
//! and borrowed decoding views require `decompression`.

#[cfg(feature = "compression")]
pub use crate::compressor::dictionary::*;
#[cfg(feature = "decompression")]
mod decode;
#[cfg(feature = "decompression")]
pub use decode::{
    DecodeDictionary, DecodeDictionaryError, DecodeDictionaryLimits, DictionaryAttachment,
    DictionaryRef,
};

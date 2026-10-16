//! Zstandard dictionaries.
//!
//! A dictionary is history both sides agree on before a frame starts: bytes
//! the encoder may copy from without ever writing them, and, in the Zstandard
//! dictionary format, entropy tables and repeat offsets the first block may
//! reuse. The decoder must be given the same dictionary, out of band; a frame
//! names it by its ID when the dictionary has one.
//!
//! A dictionary's bytes are either *raw content* (any bytes) or the
//! [Zstandard format]: the magic number `0xEC30A437`, a 32-bit ID, entropy
//! tables, three repeat offsets and the content. [`DictionaryContent`] says
//! how to read them; [`DictionaryContent::Auto`] decides by the magic
//! number, as the reference does.
//!
//! With the `compression` feature, a `DictionaryBuilder` prepares a
//! `PreparedDictionary` for one encoder configuration, which
//! `Compressor::compress_with_dictionary` and its slice, vector, session and
//! `std::io` twins compress with. With the `decompression` feature, a
//! `DecodeDictionary` holds a parsed dictionary for decoding, and decoder
//! entry points borrow it through a `DictionaryRef`:
//! `Decompressor::decompress_with_dictionary` and its twins.
//!
//! [Zstandard format]: https://www.rfc-editor.org/rfc/rfc8878#name-dictionary-format

use core::fmt;
use core::num::NonZeroU32;

#[cfg(feature = "decompression")]
pub(crate) mod decode;

#[cfg(feature = "decompression")]
pub use decode::{DecodeDictionary, DictionaryRef};

#[cfg(feature = "compression")]
pub use crate::compressor::dictionary::{DictionaryBuilder, PreparedDictionary};

/// `ZSTD_MAGIC_DICTIONARY`.
pub(crate) const MAGIC_DICTIONARY: u32 = 0xEC30_A437;

/// How dictionary bytes are interpreted (`ZSTD_dictContentType_e`).
///
/// # Examples
///
/// ```
/// use mzstd::dictionary::DictionaryContent;
///
/// assert_eq!(DictionaryContent::default(), DictionaryContent::Auto);
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum DictionaryContent {
    /// The Zstandard format when the bytes start with its magic number, raw
    /// content otherwise (`ZSTD_dct_auto`).
    #[default]
    Auto,
    /// Raw content, even when the bytes start with the magic number
    /// (`ZSTD_dct_rawContent`).
    Raw,
    /// The Zstandard format; anything else is rejected
    /// (`ZSTD_dct_fullDict`).
    Zstd,
}

/// A dictionary's ID, as a Zstandard-format dictionary declares it and a
/// frame header names it.
///
/// Raw-content dictionaries have no ID, and neither do frames compressed
/// with one.
///
/// # Examples
///
/// ```
/// use core::num::NonZeroU32;
/// use mzstd::dictionary::DictionaryId;
///
/// let id = DictionaryId::from(NonZeroU32::new(7).unwrap());
/// assert_eq!(u32::from(id), 7);
/// assert_eq!(id.to_string(), "7");
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DictionaryId(NonZeroU32);

impl DictionaryId {
    /// The ID for a raw `u32`, `None` for zero (no ID).
    pub(crate) const fn new(id: u32) -> Option<Self> {
        match NonZeroU32::new(id) {
            Some(id) => Some(Self(id)),
            None => None,
        }
    }
}

impl From<NonZeroU32> for DictionaryId {
    fn from(id: NonZeroU32) -> Self {
        Self(id)
    }
}

impl From<DictionaryId> for NonZeroU32 {
    fn from(id: DictionaryId) -> Self {
        id.0
    }
}

impl From<DictionaryId> for u32 {
    fn from(id: DictionaryId) -> Self {
        id.0.get()
    }
}

impl fmt::Display for DictionaryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// Why dictionary bytes were rejected.
///
/// # Examples
///
/// ```
/// use mzstd::dictionary::DictionaryError;
///
/// assert_eq!(
///     DictionaryError::NotZstdFormat.to_string(),
///     "dictionary does not start with the Zstandard dictionary magic number",
/// );
/// ```
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum DictionaryError {
    /// [`DictionaryContent::Zstd`] was requested for bytes that are not in
    /// the Zstandard dictionary format.
    #[error("dictionary does not start with the Zstandard dictionary magic number")]
    NotZstdFormat,
    /// A Zstandard-format dictionary's entropy tables or repeat offsets are
    /// malformed.
    #[error("Zstandard dictionary is corrupted")]
    Corrupted,
    /// The encoder configuration a dictionary was to be prepared for is
    /// invalid.
    #[cfg(feature = "compression")]
    #[error("invalid configuration for the dictionary")]
    Config(#[from] crate::ConfigError),
}

/// How a codec reads a dictionary, decided from its first bytes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum Layout {
    /// Every byte is content.
    Raw,
    /// The Zstandard format with this ID; the entropy section starts at
    /// byte 8.
    Zstd {
        /// The declared ID, possibly zero.
        id: u32,
    },
}

/// Whether `bytes` start with the dictionary magic number and an ID.
pub(crate) fn zstd_header(bytes: &[u8]) -> Option<u32> {
    let magic = bytes.get(..4)?;
    let id = bytes.get(4..8)?;
    (u32::from_le_bytes(magic.try_into().ok()?) == MAGIC_DICTIONARY)
        .then(|| u32::from_le_bytes(id.try_into().unwrap_or_default()))
}

/// Classifies `bytes` for `content`, or reports that they are not in the
/// Zstandard format although `content` requires it.
pub(crate) fn layout(bytes: &[u8], content: DictionaryContent) -> Result<Layout, DictionaryError> {
    match (content, zstd_header(bytes)) {
        (DictionaryContent::Raw, _) | (DictionaryContent::Auto, None) => Ok(Layout::Raw),
        (_, Some(id)) => Ok(Layout::Zstd { id }),
        (DictionaryContent::Zstd, None) => Err(DictionaryError::NotZstdFormat),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn layout_follows_the_magic_number_and_the_requested_content() {
        let mut zstd = MAGIC_DICTIONARY.to_le_bytes().to_vec();
        zstd.extend_from_slice(&9u32.to_le_bytes());
        assert_eq!(
            layout(&zstd, DictionaryContent::Auto),
            Ok(Layout::Zstd { id: 9 })
        );
        assert_eq!(
            layout(&zstd, DictionaryContent::Zstd),
            Ok(Layout::Zstd { id: 9 })
        );
        assert_eq!(layout(&zstd, DictionaryContent::Raw), Ok(Layout::Raw));
        assert_eq!(
            layout(b"raw bytes", DictionaryContent::Auto),
            Ok(Layout::Raw)
        );
        assert_eq!(
            layout(b"raw bytes", DictionaryContent::Zstd),
            Err(DictionaryError::NotZstdFormat)
        );
        assert_eq!(layout(&zstd[..7], DictionaryContent::Auto), Ok(Layout::Raw));
        assert_eq!(zstd_header(&zstd[..7]), None);
    }

    #[test]
    fn ids_convert_and_display() {
        assert_eq!(DictionaryId::new(0), None);
        let id = DictionaryId::new(42).expect("non-zero");
        assert_eq!(u32::from(id), 42);
        assert_eq!(NonZeroU32::from(id).get(), 42);
        assert_eq!(DictionaryId::from(NonZeroU32::from(id)), id);
        assert_eq!(alloc::format!("{id}"), "42");
        assert_eq!(
            DictionaryError::Corrupted.to_string(),
            "Zstandard dictionary is corrupted"
        );
    }
}

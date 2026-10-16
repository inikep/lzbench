//! Dictionaries for decoding.
//!
//! A [`DecodeDictionary`] owns a dictionary's bytes and what the decoder
//! needs from them, parsed once: its ID and, in the Zstandard format, its
//! entropy tables and repeat offsets. Decoder entry points borrow it through
//! a [`DictionaryRef`].

use alloc::boxed::Box;
use core::fmt;

use super::{DictionaryContent, DictionaryError, DictionaryId, Layout, layout};
use crate::decompressor::{DecoderEntropy, DictView};

/// What the decoder reads from a dictionary's bytes, parsed once
/// (`ZSTD_loadEntropy_intoDDict`).
///
/// It does not hold the bytes: whoever owns them pairs them with these
/// parts through [`Self::view`].
#[derive(Clone, Debug)]
pub(crate) struct DecodeParts {
    /// The declared ID; zero for raw content.
    pub(crate) id: u32,
    /// Entropy tables and repeat offsets of a Zstandard-format dictionary.
    pub(crate) entropy: Option<DecoderEntropy>,
}

impl DecodeParts {
    /// The decoder's view of the dictionary whose bytes are `bytes`, the
    /// ones these parts were parsed from.
    ///
    /// All of `bytes` precede every frame as history, the format's header
    /// and tables included: the reference's `ZSTD_DDict` does the same, and
    /// encoders only reference the content after the tables.
    pub(crate) fn view<'a>(&'a self, bytes: &'a [u8]) -> DictView<'a> {
        DictView {
            content: bytes,
            id: self.id,
            entropy: self.entropy.as_ref(),
        }
    }

    /// Heap bytes the parts hold.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.entropy
            .as_ref()
            .map_or(0, DecoderEntropy::retained_bytes)
    }
}

/// Parses `bytes` for decoding as `content` asks, as
/// `ZSTD_createDDict_advanced` does: raw content has ID zero and no
/// tables; the Zstandard format has its declared ID and its entropy
/// section, which must be valid. Empty bytes are an empty raw dictionary
/// whatever `content` says, as `ZSTD_DCtx_loadDictionary` treats them.
///
/// # Errors
///
/// [`DictionaryError::NotZstdFormat`] when `content` is
/// [`DictionaryContent::Zstd`] and the bytes lack the magic number, and
/// [`DictionaryError::Corrupted`] when the entropy section is malformed.
pub(crate) fn parse_decode_parts(
    bytes: &[u8],
    content: DictionaryContent,
) -> Result<DecodeParts, DictionaryError> {
    if bytes.is_empty() {
        return Ok(DecodeParts {
            id: 0,
            entropy: None,
        });
    }
    match layout(bytes, content)? {
        Layout::Raw => Ok(DecodeParts {
            id: 0,
            entropy: None,
        }),
        Layout::Zstd { id } => {
            let (entropy, _) =
                DecoderEntropy::load(bytes).map_err(|_| DictionaryError::Corrupted)?;
            Ok(DecodeParts {
                id,
                entropy: Some(entropy),
            })
        }
    }
}

/// An immutable dictionary for decoding, built once and shared by any
/// number of operations.
///
/// Built from raw content or from a dictionary in the [Zstandard format]
/// (for instance one trained by `zstd --train`); [`DictionaryContent`]
/// says how to read the bytes. The decoder must use the dictionary the
/// frames were compressed with: a frame that names a dictionary ID decodes
/// only with a dictionary declaring that ID, while one that names none
/// decodes with any dictionary, so a wrong raw dictionary goes undetected
/// unless the frame carries a checksum.
///
/// A [`crate::Decompressor`] borrows it per operation and keeps no
/// reference to it afterwards. To share one between threads or owned
/// sessions, wrap it in an `Arc`.
///
/// [Zstandard format]: https://www.rfc-editor.org/rfc/rfc8878#name-dictionary-format
///
/// # Examples
///
/// ```
/// use mzstd::dictionary::{DecodeDictionary, DictionaryContent};
/// use mzstd::Decompressor;
///
/// let dictionary = DecodeDictionary::new(b"hello, hello", DictionaryContent::Raw)?;
/// assert_eq!(dictionary.id(), None);
/// assert_eq!(dictionary.content_len(), 12);
///
/// // "hello" as one raw block, which needs no history.
/// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x05, 0x29, 0x00, 0x00, b'h', b'e', b'l', b'l', b'o'];
/// let mut decoder = Decompressor::default();
/// assert_eq!(decoder.decompress_with_dictionary(&dictionary, &frame)?, b"hello");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct DecodeDictionary {
    bytes: Box<[u8]>,
    parts: DecodeParts,
}

impl fmt::Debug for DecodeDictionary {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecodeDictionary")
            .field("id", &self.id())
            .field("content_len", &self.content_len())
            .field("entropy", &self.parts.entropy.is_some())
            .finish()
    }
}

impl DecodeDictionary {
    /// Copies and parses `bytes` as `content` asks.
    ///
    /// [`DictionaryContent::Auto`] reads bytes that start with the
    /// Zstandard dictionary magic number as that format and anything else,
    /// including fewer than eight bytes, as raw content;
    /// [`DictionaryContent::Raw`] always reads raw content. Empty bytes make
    /// a dictionary that changes nothing.
    ///
    /// # Errors
    ///
    /// [`DictionaryError::NotZstdFormat`] when `content` is
    /// [`DictionaryContent::Zstd`] and the bytes are not in that format,
    /// and [`DictionaryError::Corrupted`] when a Zstandard-format
    /// dictionary's entropy tables or repeat offsets are malformed, exactly
    /// when the reference rejects them.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::{DecodeDictionary, DictionaryContent, DictionaryError};
    ///
    /// // The magic number and an ID, but no entropy tables.
    /// let mut truncated = 0xEC30_A437u32.to_le_bytes().to_vec();
    /// truncated.extend_from_slice(&5u32.to_le_bytes());
    /// assert_eq!(
    ///     DecodeDictionary::new(&truncated, DictionaryContent::Auto).unwrap_err(),
    ///     DictionaryError::Corrupted,
    /// );
    /// // The same bytes as raw content.
    /// let raw = DecodeDictionary::new(&truncated, DictionaryContent::Raw)?;
    /// assert_eq!((raw.id(), raw.content_len()), (None, 8));
    /// assert_eq!(
    ///     DecodeDictionary::new(b"text", DictionaryContent::Zstd).unwrap_err(),
    ///     DictionaryError::NotZstdFormat,
    /// );
    /// # Ok::<(), DictionaryError>(())
    /// ```
    pub fn new(bytes: &[u8], content: DictionaryContent) -> Result<Self, DictionaryError> {
        let parts = parse_decode_parts(bytes, content)?;
        Ok(Self {
            bytes: bytes.into(),
            parts,
        })
    }

    /// The ID the dictionary declares; `None` for raw content and for
    /// Zstandard-format dictionaries that declare zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DecodeDictionary;
    /// let raw = DecodeDictionary::try_from(&b"any bytes"[..])?;
    /// assert_eq!(raw.id(), None);
    /// # Ok::<(), mzstd::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub const fn id(&self) -> Option<DictionaryId> {
        DictionaryId::new(self.parts.id)
    }

    /// Bytes that precede every frame as history: all of them, the
    /// Zstandard format's header and tables included, as in the reference
    /// decoder.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DecodeDictionary;
    /// let raw = DecodeDictionary::try_from(&b"history"[..])?;
    /// assert_eq!(raw.content_len(), 7);
    /// # Ok::<(), mzstd::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub fn content_len(&self) -> usize {
        self.bytes.len()
    }

    /// Heap bytes the dictionary holds: its copy of the bytes and its
    /// decoding tables.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DecodeDictionary;
    /// let raw = DecodeDictionary::try_from(&b"history"[..])?;
    /// assert_eq!(raw.retained_bytes(), 7);
    /// # Ok::<(), mzstd::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.bytes.len() + self.parts.retained_bytes()
    }

    /// The decoder's view.
    pub(crate) fn view(&self) -> DictView<'_> {
        self.parts.view(&self.bytes)
    }
}

impl TryFrom<&[u8]> for DecodeDictionary {
    type Error = DictionaryError;

    /// Parses `bytes` with [`DictionaryContent::Auto`].
    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        Self::new(bytes, DictionaryContent::Auto)
    }
}

impl AsRef<Self> for DecodeDictionary {
    /// The dictionary itself, so that owned sessions accept it as well as
    /// an `Arc` or a `Box` of it.
    fn as_ref(&self) -> &Self {
        self
    }
}

/// A borrowed dictionary for one decoding operation.
///
/// Decoder entry points take `impl Into<DictionaryRef>`, so a reference to
/// a [`DecodeDictionary`] converts implicitly. A session keeps the borrow
/// until it is dropped; the decoder keeps no dictionary reference between
/// operations.
///
/// # Examples
///
/// ```
/// use mzstd::dictionary::{DecodeDictionary, DictionaryRef};
/// use mzstd::{DecodeOperation, DecoderStatus, Decompressor};
///
/// let dictionary = DecodeDictionary::try_from(&b"history"[..])?;
/// let view = DictionaryRef::from(&dictionary);
/// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
/// let mut decoder = Decompressor::default();
/// let mut session = decoder.start_with_dictionary(view);
/// let mut output = [0u8; 8];
/// let progress = session.process(&frame, &mut output, DecodeOperation::Finish)?;
/// assert_eq!(progress.status, DecoderStatus::Finished);
/// assert_eq!(&output[..progress.produced], b"ok");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum DictionaryRef<'a> {
    /// A dictionary built for decoding.
    Decode(&'a DecodeDictionary),
    /// A dictionary prepared for compression, which also decodes the
    /// frames it compresses, without a second copy of its bytes.
    #[cfg(feature = "compression")]
    Prepared(&'a crate::dictionary::PreparedDictionary),
}

impl<'a> From<&'a DecodeDictionary> for DictionaryRef<'a> {
    fn from(dictionary: &'a DecodeDictionary) -> Self {
        Self::Decode(dictionary)
    }
}

#[cfg(feature = "compression")]
impl<'a> From<&'a crate::dictionary::PreparedDictionary> for DictionaryRef<'a> {
    /// Borrows a prepared dictionary for decoding.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DictionaryBuilder;
    /// use mzstd::{Compressor, Decompressor};
    ///
    /// let dictionary = DictionaryBuilder::new(b"shared words for small payloads").build()?;
    /// let frame = Compressor::default().compress_with_dictionary(&dictionary, b"small payloads")?;
    /// let restored = Decompressor::default().decompress_with_dictionary(&dictionary, &frame)?;
    /// assert_eq!(restored, b"small payloads");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn from(dictionary: &'a crate::dictionary::PreparedDictionary) -> Self {
        Self::Prepared(dictionary)
    }
}

impl<'a> DictionaryRef<'a> {
    /// The decoder's view of the dictionary.
    pub(crate) fn view(self) -> DictView<'a> {
        match self {
            Self::Decode(dictionary) => dictionary.view(),
            #[cfg(feature = "compression")]
            Self::Prepared(dictionary) => dictionary.decode_view(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::MAGIC_DICTIONARY;
    use alloc::vec::Vec;

    /// The magic number, `id`, and no entropy tables.
    fn header_only(id: u32) -> Vec<u8> {
        let mut bytes = MAGIC_DICTIONARY.to_le_bytes().to_vec();
        bytes.extend_from_slice(&id.to_le_bytes());
        bytes
    }

    #[test]
    fn empty_bytes_are_an_empty_raw_dictionary_for_every_content_type() {
        for content in [
            DictionaryContent::Auto,
            DictionaryContent::Raw,
            DictionaryContent::Zstd,
        ] {
            let dictionary = DecodeDictionary::new(&[], content).expect("empty");
            assert_eq!(dictionary.id(), None);
            assert_eq!(dictionary.content_len(), 0);
            assert_eq!(dictionary.retained_bytes(), 0);
        }
    }

    #[test]
    fn raw_content_keeps_every_byte_and_has_no_id() {
        let bytes = header_only(5);
        let raw = DecodeDictionary::new(&bytes, DictionaryContent::Raw).expect("raw");
        let view = raw.view();
        assert_eq!(view.content, &bytes[..]);
        assert_eq!(view.id, 0);
        assert!(view.entropy.is_none());
        // Short bytes are raw content when the format is not required.
        let short = DecodeDictionary::try_from(&bytes[..7]).expect("short");
        assert_eq!(short.content_len(), 7);
        assert_eq!(
            DecodeDictionary::new(&bytes[..7], DictionaryContent::Zstd).map(|_| ()),
            Err(DictionaryError::NotZstdFormat)
        );
    }

    #[test]
    fn zstd_dictionaries_need_valid_entropy_tables() {
        let bytes = header_only(5);
        assert_eq!(
            DecodeDictionary::try_from(&bytes[..]).map(|_| ()),
            Err(DictionaryError::Corrupted)
        );
        let bytes = crate::decompressor::test_dictionary([1, 2, 3], b"content");
        let dictionary = DecodeDictionary::new(&bytes, DictionaryContent::Zstd).expect("valid");
        assert_eq!(dictionary.id().map(u32::from), Some(7));
        assert_eq!(dictionary.content_len(), bytes.len());
        assert!(dictionary.retained_bytes() > bytes.len());
        let view = DictionaryRef::from(&dictionary).view();
        assert_eq!(view.id, 7);
        assert_eq!(view.entropy.map(|entropy| entropy.reps), Some([1, 2, 3]));
        let debug = alloc::format!("{dictionary:?}");
        assert!(debug.contains("entropy: true"), "{debug}");
        assert!(core::ptr::eq(dictionary.as_ref(), &dictionary));
    }
}

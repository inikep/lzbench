//! Frame header queries (`ZSTD_getFrameHeader_advanced`).

use super::core::{FrameHeader, MAGIC_SIZE, header_size};
use super::{DecodeError, InvalidDataKind};
use crate::FrameFormat;
use crate::dictionary::DictionaryId;
use crate::shared::format::{MAGIC_NUMBER, MAGIC_SKIPPABLE_MASK, MAGIC_SKIPPABLE_START};

/// The largest window log the format allows on 64-bit targets
/// (`ZSTD_WINDOWLOG_MAX`).
const WINDOW_LOG_MAX: u32 = 31;

/// Size of a skippable frame's magic number and size field.
const SKIPPABLE_HEADER_SIZE: usize = 8;

/// Which kind of frame a header opens.
///
/// # Examples
///
/// ```
/// use mzstd::decompressor::FrameKind;
///
/// assert_ne!(FrameKind::Zstandard, FrameKind::Skippable);
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum FrameKind {
    /// A Zstandard frame.
    Zstandard,
    /// A skippable frame; only [`FrameHeaderInfo::content_size`], its payload
    /// length, is meaningful.
    Skippable,
}

/// What the header of a frame declares, read without decoding the frame
/// (`ZSTD_FrameHeader`).
///
/// # Examples
///
/// ```
/// use mzstd::FrameFormat;
/// use mzstd::decompressor::{FrameHeaderInfo, FrameKind};
///
/// // "hello" as one raw block in a single-segment frame.
/// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x05, 0x29, 0x00, 0x00, b'h', b'e', b'l', b'l', b'o'];
/// let header = FrameHeaderInfo::parse(&frame, FrameFormat::Standard)?;
/// assert_eq!(header.kind, FrameKind::Zstandard);
/// assert_eq!(header.content_size, Some(5));
/// assert_eq!(header.header_size, 6);
/// // The same frame without its magic number.
/// let magicless = FrameHeaderInfo::parse(&frame[4..], FrameFormat::Magicless)?;
/// assert_eq!(magicless.content_size, Some(5));
/// # Ok::<(), mzstd::DecodeError>(())
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub struct FrameHeaderInfo {
    /// The frame's kind.
    pub kind: FrameKind,
    /// Bytes the header occupies, magic number included when the format has
    /// one.
    pub header_size: usize,
    /// The declared decompressed size, or `None` when the header omits it.
    /// For a skippable frame, its payload size.
    pub content_size: Option<u64>,
    /// The largest back-reference distance the frame uses.
    pub window_size: u64,
    /// The dictionary the frame names, if any.
    pub dictionary_id: Option<DictionaryId>,
    /// Whether a content checksum follows the last block.
    pub checksum: bool,
}

impl FrameHeaderInfo {
    /// Reads the header at the start of `src`.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::UnexpectedEndOfInput`] when `src` ends before the
    ///   header does.
    /// - [`DecodeError::InvalidData`] for an unknown magic number (never for
    ///   [`FrameFormat::Magicless`]) or a reserved header bit.
    /// - [`DecodeError::WindowLimitExceeded`] for a window above the
    ///   format's maximum of 2^31 bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::DecodeError;
    /// use mzstd::FrameFormat;
    /// use mzstd::decompressor::FrameHeaderInfo;
    ///
    /// assert!(matches!(
    ///     FrameHeaderInfo::parse(&[0x28, 0xb5], FrameFormat::Standard),
    ///     Err(DecodeError::UnexpectedEndOfInput)
    /// ));
    /// assert!(matches!(
    ///     FrameHeaderInfo::parse(b"not a frame", FrameFormat::Standard),
    ///     Err(DecodeError::InvalidData { .. })
    /// ));
    /// ```
    pub fn parse(src: &[u8], format: FrameFormat) -> Result<Self, DecodeError> {
        let magic_size = match format {
            FrameFormat::Standard => MAGIC_SIZE,
            FrameFormat::Magicless => 0,
        };
        if format == FrameFormat::Standard {
            Self::check_magic_prefix(src)?;
            if let Some(&bytes) = src.first_chunk::<MAGIC_SIZE>()
                && u32::from_le_bytes(bytes) & MAGIC_SKIPPABLE_MASK == MAGIC_SKIPPABLE_START
            {
                return Self::skippable(src);
            }
        }
        let descriptor = *src
            .get(magic_size)
            .ok_or(DecodeError::UnexpectedEndOfInput)?;
        let size = magic_size + header_size(descriptor);
        let header = FrameHeader::parse(
            src.get(magic_size..size)
                .ok_or(DecodeError::UnexpectedEndOfInput)?,
        )?;
        // The window byte follows the descriptor in a segmented frame.
        if descriptor & 0x20 == 0 {
            let window_log = u32::from(src[magic_size + 1] >> 3) + 10;
            if window_log > WINDOW_LOG_MAX {
                return Err(DecodeError::WindowLimitExceeded {
                    window_size: header.window_size,
                    limit: 1 << WINDOW_LOG_MAX,
                });
            }
        }
        Ok(Self {
            kind: FrameKind::Zstandard,
            header_size: size,
            content_size: header.content_size,
            window_size: header.window_size,
            dictionary_id: DictionaryId::new(header.dictionary_id),
            checksum: header.checksum,
        })
    }

    /// Fails early when the leading bytes cannot start any frame: missing
    /// bytes are filled in from the magic number, then from a skippable one,
    /// as the reference does.
    fn check_magic_prefix(src: &[u8]) -> Result<(), DecodeError> {
        let count = src.len().min(MAGIC_SIZE);
        let mut standard = MAGIC_NUMBER.to_le_bytes();
        standard[..count].copy_from_slice(&src[..count]);
        let mut skippable = MAGIC_SKIPPABLE_START.to_le_bytes();
        skippable[..count].copy_from_slice(&src[..count]);
        if u32::from_le_bytes(standard) == MAGIC_NUMBER
            || u32::from_le_bytes(skippable) & MAGIC_SKIPPABLE_MASK == MAGIC_SKIPPABLE_START
        {
            Ok(())
        } else {
            Err(InvalidDataKind::Magic.into())
        }
    }

    /// A skippable frame's magic number and size field.
    fn skippable(src: &[u8]) -> Result<Self, DecodeError> {
        let field = src
            .get(MAGIC_SIZE..SKIPPABLE_HEADER_SIZE)
            .and_then(|field| field.first_chunk::<4>())
            .ok_or(DecodeError::UnexpectedEndOfInput)?;
        Ok(Self {
            kind: FrameKind::Skippable,
            header_size: SKIPPABLE_HEADER_SIZE,
            content_size: Some(u64::from(u32::from_le_bytes(*field))),
            window_size: 0,
            dictionary_id: None,
            checksum: false,
        })
    }
}

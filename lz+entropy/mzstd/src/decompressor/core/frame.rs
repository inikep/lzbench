//! Frame headers and per-frame decoding state (RFC 8878 section 3.1.1).
//!
//! [`FrameHeader::parse`] reads a frame header after its magic number.
//! [`FrameDecoder`] validates it against the caller's limits, then
//! regenerates the frame's blocks one at a time, tracking the produced size
//! against `Frame_Content_Size` and hashing the output for the optional
//! content checksum. A frame starts from the operation's dictionary, if
//! any: its ID must match the frame's, its tables and repeat offsets seed
//! the first block, and its content precedes the frame's output.

use ::core::ops::Range;

use super::BlockError;
use super::block::{BlockDecoder, BlockHeader};
use super::dictionary::DictView;
use super::sequences::BlockContext;
use crate::Backend;
use crate::decompressor::{DecodeError, DecodeLimits, InvalidDataKind};
use crate::dictionary::DictionaryId;
use crate::shared::format::{
    BLOCK_SIZE_MAX, CHECKSUM_SIZE, FRAME_HEADER_SIZE_MAX, WINDOW_LOG_ABSOLUTE_MIN,
};
use crate::shared::xxhash::Xxh64;

/// Size of a frame magic number.
pub(crate) const MAGIC_SIZE: usize = 4;

// The largest header (window byte, four-byte dictionary ID, eight-byte
// content size) and its magic number make the format's maximum.
const _: () = assert!(MAGIC_SIZE + header_size(0xc3) == FRAME_HEADER_SIZE_MAX);

/// A parsed frame header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FrameHeader {
    /// Largest back-reference distance, and the history a decoder keeps.
    pub(crate) window_size: u64,
    /// Declared decompressed size, when present.
    pub(crate) content_size: Option<u64>,
    /// Declared dictionary; zero means none.
    pub(crate) dictionary_id: u32,
    /// Whether a content checksum follows the last block.
    pub(crate) checksum: bool,
}

/// Size of a frame header, from its descriptor byte to its last field
/// (the magic number excluded).
pub(crate) const fn header_size(descriptor: u8) -> usize {
    let single_segment = descriptor & 0x20 != 0;
    let window = if single_segment { 0 } else { 1 };
    let dictionary = match descriptor & 3 {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => 4,
    };
    let content_size = match descriptor >> 6 {
        0 => {
            if single_segment {
                1
            } else {
                0
            }
        }
        1 => 2,
        2 => 4,
        _ => 8,
    };
    1 + window + dictionary + content_size
}

/// Reads a little-endian field of up to eight bytes.
fn read_le(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .rev()
        .fold(0u64, |value, &byte| (value << 8) | u64::from(byte))
}

impl FrameHeader {
    /// Parses a header of exactly [`header_size`] bytes.
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, InvalidDataKind> {
        let invalid = InvalidDataKind::FrameHeader;
        let (&descriptor, mut fields) = bytes.split_first().ok_or(invalid)?;
        if bytes.len() != header_size(descriptor) || descriptor & 0x08 != 0 {
            return Err(invalid);
        }
        let single_segment = descriptor & 0x20 != 0;
        let mut window_size = 0;
        if !single_segment {
            let (&window, rest) = fields.split_first().ok_or(invalid)?;
            let log = WINDOW_LOG_ABSOLUTE_MIN + u32::from(window >> 3);
            let base = 1u64 << log;
            window_size = base + (base / 8) * u64::from(window & 7);
            fields = rest;
        }
        let dictionary_bytes = match descriptor & 3 {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 4,
        };
        let (dictionary, fields) = fields.split_at_checked(dictionary_bytes).ok_or(invalid)?;
        let content_size = match fields.len() {
            0 => None,
            2 => Some(read_le(fields) + 256),
            _ => Some(read_le(fields)),
        };
        if single_segment {
            window_size = content_size.unwrap_or(0);
        }
        Ok(Self {
            window_size,
            content_size,
            dictionary_id: read_le(dictionary) as u32,
            checksum: descriptor & 0x04 != 0,
        })
    }
}

/// Decoding state of the frame in progress.
#[derive(Debug)]
pub(crate) struct FrameDecoder {
    header: Option<FrameHeader>,
    block_max: usize,
    produced: u64,
    hasher: Xxh64,
    blocks: BlockDecoder,
}

impl Default for FrameDecoder {
    fn default() -> Self {
        Self {
            header: None,
            block_max: 0,
            produced: 0,
            hasher: Xxh64::new(0),
            blocks: BlockDecoder::default(),
        }
    }
}

impl FrameDecoder {
    /// Validates `header` against `limits` and the operation's
    /// `dictionary`, and starts its frame from that dictionary.
    ///
    /// As in `ZSTD_decodeFrameHeader`, a frame that names a dictionary ID
    /// needs a dictionary with that ID: without a dictionary it fails with
    /// [`DecodeError::DictionaryUnsupported`], with another one with
    /// [`DecodeError::DictionaryMismatch`]. A frame that names none decodes
    /// with any dictionary, or without one.
    pub(crate) fn begin(
        &mut self,
        header: FrameHeader,
        limits: &DecodeLimits,
        dictionary: Option<DictView<'_>>,
    ) -> Result<(), DecodeError> {
        if let Some(frame) = DictionaryId::new(header.dictionary_id) {
            let Some(view) = dictionary else {
                return Err(DecodeError::DictionaryUnsupported {
                    id: header.dictionary_id,
                });
            };
            let expected = DictionaryId::new(view.id);
            if expected != Some(frame) {
                return Err(DecodeError::DictionaryMismatch { expected, frame });
            }
        }
        let limit = limits.max_window_size();
        if header.window_size > limit {
            return Err(DecodeError::WindowLimitExceeded {
                window_size: header.window_size,
                limit,
            });
        }
        // At most 128 KiB, so the conversion is lossless.
        self.block_max = header.window_size.min(BLOCK_SIZE_MAX as u64) as usize;
        self.header = Some(header);
        self.produced = 0;
        self.hasher.reset(0);
        self.blocks.reset(dictionary.and_then(|view| view.entropy));
        Ok(())
    }

    /// Heap bytes held between blocks.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.blocks.retained_bytes()
    }

    /// Drops the entropy tables and buffers, keeping the header.
    pub(crate) fn release(&mut self) {
        self.blocks = BlockDecoder::default();
    }

    /// The header of the frame in progress.
    pub(crate) const fn header(&self) -> Option<&FrameHeader> {
        self.header.as_ref()
    }

    /// Largest block of the frame in progress.
    pub(crate) const fn block_max(&self) -> usize {
        self.block_max
    }

    /// Whether the frame in progress ends with a content checksum.
    pub(crate) fn has_checksum(&self) -> bool {
        self.header.is_some_and(|header| header.checksum)
    }

    /// Regenerates one block into `out` after `out[history]`, this frame's
    /// retained history, on `backend`, returning the end of its output.
    /// `dictionary` must be the one the frame began with.
    ///
    /// The dictionary's content precedes the frame's first byte while the
    /// frame's output before the block is below its window: the reference
    /// encoder only references a dictionary from blocks that end within
    /// one window of it, and the frame's first window is then still in
    /// `out` from `history.start`.
    ///
    /// Callers that grow their buffer give it room for
    /// [`Self::block_max`] bytes plus
    /// [`WILDCOPY_OVERLENGTH`](super::copy::WILDCOPY_OVERLENGTH); a block that
    /// does not fit a caller-sized buffer fails with
    /// [`DecodeError::OutputTooSmall`].
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn decode_block(
        &mut self,
        backend: Backend,
        block: &BlockHeader,
        content: &[u8],
        out: &mut [u8],
        history: Range<usize>,
        dictionary: Option<DictView<'_>>,
    ) -> Result<usize, DecodeError> {
        let Range {
            start: history_start,
            end: op,
        } = history;
        let header = self.header.ok_or(DecodeError::InvalidState)?;
        let before = match dictionary {
            Some(view) if self.produced < header.window_size => view.content,
            _ => &[],
        };
        let context = BlockContext {
            history_start,
            window: header.window_size,
            block_max: self.block_max,
            dictionary: before,
            entropy: dictionary.and_then(|view| view.entropy),
        };
        let end = self
            .blocks
            .decode(backend, block, content, out, op, context)
            .map_err(|error| match error {
                BlockError::Invalid(kind) => DecodeError::from(kind),
                BlockError::OutputFull => DecodeError::OutputTooSmall { written: op },
            })?;
        self.account_block(block, &out[op..end])?;
        Ok(end)
    }

    /// Records a block's regenerated bytes, `produced`, written by the
    /// caller or by [`Self::decode_block`]: feeds the checksum and checks
    /// the frame's declared size.
    pub(crate) fn account_block(
        &mut self,
        block: &BlockHeader,
        produced: &[u8],
    ) -> Result<(), DecodeError> {
        let header = self.header.ok_or(DecodeError::InvalidState)?;
        if header.checksum {
            self.hasher.update(produced);
        }
        self.produced += produced.len() as u64;
        if let Some(declared) = header.content_size
            && (self.produced > declared || (block.last && self.produced != declared))
        {
            return Err(DecodeError::ContentSizeMismatch {
                declared,
                decoded: self.produced,
            });
        }
        Ok(())
    }

    /// Checks the stored content checksum against the frame's output.
    pub(crate) fn verify_checksum(&self, stored: [u8; CHECKSUM_SIZE]) -> Result<(), DecodeError> {
        let expected = u32::from_le_bytes(stored);
        let computed = self.hasher.digest() as u32;
        if expected == computed {
            Ok(())
        } else {
            Err(DecodeError::ChecksumMismatch { expected, computed })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompressor::core::block::BlockKind;
    use alloc::vec::Vec;

    #[test]
    fn header_sizes_follow_the_descriptor() {
        assert_eq!(header_size(0x00), 2);
        assert_eq!(header_size(0x20), 2);
        assert_eq!(header_size(0x03), 6);
        assert_eq!(header_size(0x40), 4);
        assert_eq!(header_size(0x80), 6);
        assert_eq!(header_size(0xe2), 11);
    }

    #[test]
    fn windows_content_sizes_and_dictionaries_parse() {
        // Window exponent 1, mantissa 3: 2 KiB + 3 * 256.
        let header = FrameHeader::parse(&[0x04, 0x0b]).expect("valid");
        assert_eq!(header.window_size, 2048 + 3 * 256);
        assert_eq!(header.content_size, None);
        assert!(header.checksum);
        // Single segment with a one-byte size: the window is the size.
        let header = FrameHeader::parse(&[0x20, 200]).expect("valid");
        assert_eq!(header.window_size, 200);
        assert_eq!(header.content_size, Some(200));
        // Two-byte sizes are offset by 256; a two-byte dictionary ID.
        let header = FrameHeader::parse(&[0x62, 0x34, 0x12, 0x01, 0x00]).expect("valid");
        assert_eq!(header.dictionary_id, 0x1234);
        assert_eq!(header.content_size, Some(257));
        let header = FrameHeader::parse(&[0xa0, 1, 0, 0, 0]).expect("valid");
        assert_eq!(header.content_size, Some(1));
        let header = FrameHeader::parse(&[0xe1, 7, 0, 0, 0, 0, 0, 0, 0, 1]).expect("valid");
        assert_eq!(header.dictionary_id, 7);
        assert_eq!(header.content_size, Some(1 << 56));
    }

    #[test]
    fn reserved_bits_and_wrong_lengths_are_rejected() {
        assert_eq!(
            FrameHeader::parse(&[0x08, 0]),
            Err(InvalidDataKind::FrameHeader)
        );
        assert_eq!(
            FrameHeader::parse(&[0x00]),
            Err(InvalidDataKind::FrameHeader)
        );
        assert_eq!(FrameHeader::parse(&[]), Err(InvalidDataKind::FrameHeader));
        // The unused bit is ignored.
        assert!(FrameHeader::parse(&[0x10, 0]).is_ok());
    }

    #[test]
    fn begin_enforces_dictionary_and_window_limits() {
        let mut frame = FrameDecoder::default();
        let limits = DecodeLimits::default();
        let with_dictionary = FrameHeader::parse(&[0x01, 0x00, 9]).expect("valid");
        assert!(matches!(
            frame.begin(with_dictionary, &limits, None),
            Err(DecodeError::DictionaryUnsupported { id: 9 })
        ));
        // Exponent 18: a 2^28-byte window exceeds the default 2^27.
        let wide = FrameHeader::parse(&[0x00, 18 << 3]).expect("valid");
        assert!(matches!(
            frame.begin(wide, &limits, None),
            Err(DecodeError::WindowLimitExceeded {
                window_size: 0x1000_0000,
                limit: 0x0800_0000
            })
        ));
        let empty = FrameHeader::parse(&[0x20, 0]).expect("valid");
        frame.begin(empty, &limits, None).expect("valid");
        assert_eq!(frame.block_max(), 0);
        assert!(!frame.has_checksum());
        assert_eq!(frame.header(), Some(&empty));
    }

    #[test]
    fn frames_need_the_dictionary_they_name() {
        let mut frame = FrameDecoder::default();
        let limits = DecodeLimits::default();
        let view = |id| DictView {
            content: b"history",
            id,
            entropy: None,
        };
        let named = FrameHeader::parse(&[0x01, 0x00, 9]).expect("valid");
        let anonymous = FrameHeader::parse(&[0x00, 0x00]).expect("valid");
        frame.begin(named, &limits, Some(view(9))).expect("same ID");
        frame
            .begin(anonymous, &limits, Some(view(9)))
            .expect("frames without an ID take any dictionary");
        frame
            .begin(anonymous, &limits, Some(view(0)))
            .expect("or one without an ID");
        let expected = DictionaryId::new(4);
        assert!(matches!(
            frame.begin(named, &limits, Some(view(4))),
            Err(DecodeError::DictionaryMismatch { expected: found, .. }) if found == expected
        ));
        // A raw-content dictionary has no ID, so it never matches one.
        assert!(matches!(
            frame.begin(named, &limits, Some(view(0))),
            Err(DecodeError::DictionaryMismatch { expected: None, .. })
        ));
    }

    #[test]
    fn dictionary_content_is_reachable_within_the_first_window() {
        let mut frame = FrameDecoder::default();
        // A 1 KiB window, no content size.
        let header = FrameHeader::parse(&[0x00, 0x00]).expect("valid");
        let view = DictView {
            content: b"0123456789",
            id: 0,
            entropy: None,
        };
        frame
            .begin(header, &DecodeLimits::default(), Some(view))
            .expect("valid");
        // Raw literals "x", one sequence with RLE tables: literal length 0,
        // match length 3, offset code 3 with extra bits 0b111: offset 12,
        // two bytes before the dictionary's ten.
        let content = [1 << 3, b'x', 1, 0x54, 0, 3, 0, 0x0f];
        let block = BlockHeader {
            last: false,
            kind: BlockKind::Compressed,
            size: content.len(),
        };
        let mut buffer = alloc::vec![0u8; 1024 + (1 << 17) + 64];
        let error = frame
            .decode_block(
                Backend::default(),
                &block,
                &content,
                &mut buffer,
                0..0,
                Some(view),
            )
            .expect_err("offset 12 reaches past the dictionary");
        assert!(matches!(
            error,
            DecodeError::InvalidData {
                kind: InvalidDataKind::Offset
            }
        ));
        // Offset code 2 with extra bits 0b11: offset 4, into "6789".
        let content = [1 << 3, b'x', 1, 0x54, 0, 2, 0, 0x07];
        let end = frame
            .decode_block(
                Backend::default(),
                &block,
                &content,
                &mut buffer,
                0..0,
                Some(view),
            )
            .expect("valid");
        assert_eq!(&buffer[..end], b"678x");
        // Past the first window the dictionary is out of reach.
        let raw = BlockHeader {
            last: false,
            kind: BlockKind::Raw,
            size: 1024,
        };
        let filler = alloc::vec![b'.'; 1024];
        let end = frame
            .decode_block(
                Backend::default(),
                &raw,
                &filler,
                &mut buffer,
                0..end,
                Some(view),
            )
            .expect("raw");
        // With no retained history, the same match has nothing to copy.
        let error = frame
            .decode_block(
                Backend::default(),
                &block,
                &content,
                &mut buffer,
                end..end,
                Some(view),
            )
            .expect_err("the dictionary is out of reach");
        assert!(matches!(
            error,
            DecodeError::InvalidData {
                kind: InvalidDataKind::Offset
            }
        ));
    }

    /// Decodes a block after `out` and appends its output.
    fn decode(
        frame: &mut FrameDecoder,
        block: &BlockHeader,
        content: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<(), DecodeError> {
        let start = out.len();
        let mut buffer = out.clone();
        buffer.resize(start + (1 << 17) + 32, 0);
        let end = frame.decode_block(
            Backend::default(),
            block,
            content,
            &mut buffer,
            0..start,
            None,
        )?;
        out.extend_from_slice(&buffer[start..end]);
        Ok(())
    }

    #[test]
    fn blocks_are_checked_against_the_declared_size_and_checksum() {
        let limits = DecodeLimits::default();
        let mut frame = FrameDecoder::default();
        let mut out = Vec::new();
        let raw = |last, size| BlockHeader {
            last,
            kind: BlockKind::Raw,
            size,
        };
        assert!(matches!(
            decode(&mut frame, &raw(true, 1), b"a", &mut out),
            Err(DecodeError::InvalidState)
        ));
        frame
            .begin(
                FrameHeader::parse(&[0x24, 3]).expect("valid"),
                &limits,
                None,
            )
            .expect("valid");
        decode(&mut frame, &raw(false, 2), b"ab", &mut out).expect("fits");
        assert!(matches!(
            decode(&mut frame, &raw(true, 0), b"", &mut out),
            Err(DecodeError::ContentSizeMismatch {
                declared: 3,
                decoded: 2
            })
        ));
        frame
            .begin(
                FrameHeader::parse(&[0x24, 3]).expect("valid"),
                &limits,
                None,
            )
            .expect("valid");
        out.clear();
        decode(&mut frame, &raw(true, 3), b"abc", &mut out).expect("fits");
        // XXH64("abc") = 0x44BC2CF5AD770999.
        frame
            .verify_checksum(0xAD77_0999u32.to_le_bytes())
            .expect("matches");
        assert!(matches!(
            frame.verify_checksum([0; 4]),
            Err(DecodeError::ChecksumMismatch {
                expected: 0,
                computed: 0xAD77_0999
            })
        ));
        // Producing more than declared fails before the last block.
        frame
            .begin(
                FrameHeader::parse(&[0x20, 1]).expect("valid"),
                &limits,
                None,
            )
            .expect("valid");
        assert!(matches!(
            decode(&mut frame, &raw(false, 1), b"a", &mut out),
            Ok(())
        ));
        assert!(matches!(
            decode(&mut frame, &raw(false, 1), b"b", &mut out),
            Err(DecodeError::ContentSizeMismatch {
                declared: 1,
                decoded: 2
            })
        ));
    }
}

//! The literals section of a compressed block (RFC 8878 section 3.1.1.3.1).
//!
//! Raw literals are borrowed straight from the block; RLE and
//! Huffman-coded literals are regenerated into a reusable buffer. The
//! Huffman table of the last `Compressed_Literals_Block` is kept for
//! `Treeless_Literals_Block`s later in the same frame; before the frame's
//! first one, a Zstandard-format dictionary's table serves them.
//!
//! Sequence execution copies literals in whole chunks, so every
//! [`Literals`] view carries [`WILDCOPY_OVERLENGTH`] readable bytes past
//! its end: raw literals are borrowed in place when the block holds that
//! many more bytes after them, and copied into the buffer otherwise, as
//! the reference does.

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::copy::WILDCOPY_OVERLENGTH;
use super::huffman::HuffmanTable;
use crate::decompressor::InvalidDataKind;
use crate::shared::format::EncodingType;

/// Smallest compressed block: a one-byte literals header and one more byte.
const MIN_COMPRESSED_BLOCK: usize = 2;
/// Fewest literals four Huffman streams can carry.
const MIN_LITERALS_FOR_4_STREAMS: usize = 6;
/// Bytes the reference decoder requires before reading a Huffman header.
const MIN_HUFFMAN_SECTION: usize = 5;

/// Maps the two-bit literals block type onto the shared encoding names.
pub(crate) const fn encoding_type(bits: u8) -> EncodingType {
    match bits & 3 {
        0 => EncodingType::Basic,
        1 => EncodingType::Rle,
        2 => EncodingType::Compressed,
        _ => EncodingType::Repeat,
    }
}

/// A block's literals, followed by at least [`WILDCOPY_OVERLENGTH`]
/// readable bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Literals<'a> {
    /// The literals, then the slack.
    pub(crate) bytes: &'a [u8],
    /// How many leading bytes of `bytes` are literals.
    pub(crate) len: usize,
}

impl<'a> Literals<'a> {
    /// The literals themselves.
    #[cfg(test)]
    pub(crate) fn as_slice(&self) -> &'a [u8] {
        &self.bytes[..self.len]
    }
}

/// The tree a `Treeless_Literals_Block` reuses.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum TreeSource {
    /// None: the frame has neither described a tree nor inherited one.
    #[default]
    Missing,
    /// The decoder's own table, from the frame's last
    /// `Compressed_Literals_Block`.
    Own,
    /// The dictionary's table, until the frame describes a tree.
    Dictionary,
}

/// Literals state carried across the blocks of one frame.
#[derive(Debug, Default)]
pub(crate) struct LiteralsDecoder {
    /// Table of the last Huffman-compressed literals block, allocated on
    /// first use.
    huffman: Option<Box<HuffmanTable>>,
    /// Which table `Treeless_Literals_Block`s reuse.
    tree: TreeSource,
    /// Regenerated literals and their slack.
    buffer: Vec<u8>,
}

impl LiteralsDecoder {
    /// The buffer cut to `size` literals and their slack; the caller
    /// overwrites the literals. It only ever grows, so blocks of varying
    /// sizes zero nothing but its new tail.
    fn buffer(&mut self, size: usize) -> &mut [u8] {
        let len = size + WILDCOPY_OVERLENGTH;
        if self.buffer.len() < len {
            self.buffer.resize(len, 0);
        }
        &mut self.buffer[..len]
    }
}

/// Reads `N` little-endian header bytes into a `u32`.
fn header_value(content: &[u8], bytes: usize) -> Option<u32> {
    let header = content.get(..bytes)?;
    Some(
        header
            .iter()
            .rev()
            .fold(0u32, |value, &byte| (value << 8) | u32::from(byte)),
    )
}

impl LiteralsDecoder {
    /// Heap bytes held between blocks.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.buffer.capacity()
            + self.huffman.as_ref().map_or(0, |table| {
                size_of::<HuffmanTable>() + table.retained_bytes()
            })
    }

    /// Forgets the previous Huffman table at a frame start, keeping its
    /// allocation; with `dictionary_tree`, treeless blocks reuse the
    /// dictionary's table until the frame describes its own (the
    /// reference's `litEntropy`).
    pub(crate) fn reset(&mut self, dictionary_tree: bool) {
        self.tree = if dictionary_tree {
            TreeSource::Dictionary
        } else {
            TreeSource::Missing
        };
    }

    /// Decodes the literals section at the start of a compressed block's
    /// `content`, returning the literals and the section's size.
    /// `dictionary` is the dictionary's table, which treeless blocks use
    /// until the frame describes a tree.
    #[inline(always)]
    pub(crate) fn decode<'a, S: fearless_simd::Simd>(
        &'a mut self,
        simd: S,
        content: &'a [u8],
        block_max: usize,
        dictionary: Option<&HuffmanTable>,
    ) -> Result<(Literals<'a>, usize), InvalidDataKind> {
        let invalid = InvalidDataKind::LiteralsSection;
        if content.len() < MIN_COMPRESSED_BLOCK {
            return Err(invalid);
        }
        let first = content[0];
        let size_format = (first >> 2) & 3;
        match encoding_type(first) {
            kind @ (EncodingType::Basic | EncodingType::Rle) => {
                let header_size = match size_format {
                    0 | 2 => 1,
                    1 => 2,
                    _ => 3,
                };
                let header = header_value(content, header_size).ok_or(invalid)?;
                let size = (header >> if header_size == 1 { 3 } else { 4 }) as usize;
                if size > block_max {
                    return Err(invalid);
                }
                if kind == EncodingType::Basic {
                    let literals = content
                        .get(header_size..header_size + size)
                        .ok_or(invalid)?;
                    let used = header_size + size;
                    if content.len() - used >= WILDCOPY_OVERLENGTH {
                        let bytes = &content[header_size..];
                        return Ok((Literals { bytes, len: size }, used));
                    }
                    let buffer = self.buffer(size);
                    buffer[..size].copy_from_slice(literals);
                    return Ok((
                        Literals {
                            bytes: buffer,
                            len: size,
                        },
                        used,
                    ));
                }
                let &byte = content.get(header_size).ok_or(invalid)?;
                let buffer = self.buffer(size);
                buffer[..size].fill(byte);
                Ok((
                    Literals {
                        bytes: buffer,
                        len: size,
                    },
                    header_size + 1,
                ))
            }
            kind @ (EncodingType::Compressed | EncodingType::Repeat) => {
                if kind == EncodingType::Repeat && self.tree == TreeSource::Missing {
                    return Err(invalid);
                }
                if content.len() < MIN_HUFFMAN_SECTION {
                    return Err(invalid);
                }
                let (header_size, size_bits) = match size_format {
                    0 | 1 => (3, 10),
                    2 => (4, 14),
                    _ => (5, 18),
                };
                let four_streams = size_format != 0;
                let header = header_value(content, header_size.min(4)).ok_or(invalid)?;
                let mask = (1u32 << size_bits) - 1;
                let size = ((header >> 4) & mask) as usize;
                let compressed = if header_size == 5 {
                    ((header >> 22) | (u32::from(content[4]) << 10)) as usize
                } else {
                    ((header >> (4 + size_bits)) & mask) as usize
                };
                if size > block_max || (four_streams && size < MIN_LITERALS_FOR_4_STREAMS) {
                    return Err(invalid);
                }
                let section = content
                    .get(header_size..header_size + compressed)
                    .ok_or(invalid)?;
                let streams = if kind == EncodingType::Compressed {
                    self.tree = TreeSource::Missing;
                    let table = self.huffman.get_or_insert_with(Box::default);
                    let used = table.read(section)?;
                    self.tree = TreeSource::Own;
                    section.get(used..).ok_or(invalid)?
                } else {
                    section
                };
                let len = size + WILDCOPY_OVERLENGTH;
                if self.buffer.len() < len {
                    self.buffer.resize(len, 0);
                }
                let buffer = &mut self.buffer[..len];
                let out = &mut buffer[..size];
                if self.tree == TreeSource::Dictionary {
                    let table = dictionary.ok_or(invalid)?;
                    table.decode_streams_shared(simd, streams, out, four_streams)?;
                } else {
                    let table = self.huffman.as_deref_mut().ok_or(invalid)?;
                    table.decode_streams(simd, streams, out, four_streams)?;
                }
                Ok((
                    Literals {
                        bytes: buffer,
                        len: size,
                    },
                    header_size + compressed,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(content: &[u8]) -> Result<(Vec<u8>, usize), InvalidDataKind> {
        let mut decoder = LiteralsDecoder::default();
        decoder
            .decode(fearless_simd::Fallback::new(), content, 1 << 17, None)
            .map(|(literals, used)| {
                assert!(literals.bytes.len() >= literals.len + WILDCOPY_OVERLENGTH);
                (literals.as_slice().to_vec(), used)
            })
    }

    #[test]
    fn raw_literals_use_all_three_size_formats() {
        // One-byte header: size 3 in bits 3..8.
        assert_eq!(
            decode(&[3 << 3, b'a', b'b', b'c', 0]),
            Ok((b"abc".to_vec(), 4))
        );
        // Two-byte header, size format 01: size 2.
        assert_eq!(
            decode(&[0b0000_0100 | (2 << 4), 0, b'x', b'y']),
            Ok((b"xy".to_vec(), 4))
        );
        // Three-byte header, size format 11: size 1.
        assert_eq!(
            decode(&[0b0000_1100 | (1 << 4), 0, 0, b'z']),
            Ok((b"z".to_vec(), 4))
        );
    }

    #[test]
    fn rle_literals_repeat_one_byte() {
        let (literals, used) = decode(&[1 | (5 << 3), b'q']).expect("valid");
        assert_eq!(literals, b"qqqqq");
        assert_eq!(used, 2);
        // A two-byte RLE header needs its byte too.
        assert_eq!(
            decode(&[0b0000_0101, 0]),
            Err(InvalidDataKind::LiteralsSection)
        );
    }

    #[test]
    fn sizes_beyond_the_block_or_the_content_are_rejected() {
        let mut decoder = LiteralsDecoder::default();
        assert_eq!(
            decoder
                .decode(
                    fearless_simd::Fallback::new(),
                    &[4 << 3, 0, 0, 0, 0],
                    3,
                    None
                )
                .map(|_| ()),
            Err(InvalidDataKind::LiteralsSection)
        );
        assert_eq!(decode(&[9 << 3, 1]), Err(InvalidDataKind::LiteralsSection));
        assert_eq!(decode(&[0]), Err(InvalidDataKind::LiteralsSection));
    }

    #[test]
    fn treeless_literals_need_a_previous_table() {
        assert_eq!(
            decode(&[3, 0, 0, 0, 0, 0]),
            Err(InvalidDataKind::LiteralsSection)
        );
    }

    #[test]
    fn compressed_literals_check_their_header_sizes() {
        // Fewer than five bytes.
        assert_eq!(decode(&[2, 0, 0, 0]), Err(InvalidDataKind::LiteralsSection));
        // Four streams with fewer than six literals: size format 01, size 5.
        let header = 2 | (1 << 2) | (5 << 4) | (10 << 14);
        let bytes = u32::to_le_bytes(header);
        assert_eq!(
            decode(&[bytes[0], bytes[1], bytes[2], 0, 0]),
            Err(InvalidDataKind::LiteralsSection)
        );
        // A compressed size past the content.
        let header = 2 | (4 << 4) | (100 << 14);
        let bytes = u32::to_le_bytes(header);
        assert_eq!(
            decode(&[bytes[0], bytes[1], bytes[2], 0, 0]),
            Err(InvalidDataKind::LiteralsSection)
        );
    }

    #[test]
    fn huffman_literals_decode_and_seed_treeless_blocks() {
        // Single stream, two literals, compressed size 5: the RFC example
        // tree (four bytes) and one stream byte decoding A then B.
        let tree = [132u8, 0x43, 0x20, 0x10];
        let stream = 0b0000_1101u8; // A then B.
        let header = 2 | (2 << 4) | (5 << 14);
        let bytes = u32::to_le_bytes(header);
        let mut content = alloc::vec![bytes[0], bytes[1], bytes[2]];
        content.extend_from_slice(&tree);
        content.push(stream);
        let mut decoder = LiteralsDecoder::default();
        let (literals, used) = decoder
            .decode(fearless_simd::Fallback::new(), &content, 1 << 17, None)
            .expect("valid");
        assert_eq!(literals.as_slice(), [0, 1]);
        assert_eq!(used, 8);
        // A treeless block reuses the table.
        let header = 3 | (2 << 4) | (1 << 14);
        let bytes = u32::to_le_bytes(header);
        let treeless = [bytes[0], bytes[1], bytes[2], stream, 0];
        let (literals, used) = decoder
            .decode(fearless_simd::Fallback::new(), &treeless, 1 << 17, None)
            .expect("valid");
        assert_eq!(literals.as_slice(), [0, 1]);
        assert_eq!(used, 4);
        // Size formats 10 and 11 read 14- and 18-bit fields. A six-byte
        // section holds only the jump table, so its streams are missing.
        let header = 3 | (2 << 2) | (6 << 4) | (6 << 18);
        let mut wide = u32::to_le_bytes(header).to_vec();
        wide.extend_from_slice(&[1, 0, 1, 0, 1, 0]);
        assert_eq!(
            decoder
                .decode(fearless_simd::Fallback::new(), &wide, 1 << 17, None)
                .map(|_| ()),
            Err(InvalidDataKind::HuffmanStream)
        );
        let header = 3 | (3 << 2) | (6 << 4) | (10 << 22);
        let mut widest = u32::to_le_bytes(header).to_vec();
        widest.push(0);
        widest.extend_from_slice(&[1, 0, 1, 0, 1, 0]);
        widest.extend_from_slice(&[0b0000_1101; 4]);
        // Six literals over four streams: 2, 2, 2, 0 symbols; the last
        // stream decodes nothing, so its marker byte alone is exhausted
        // only when it is 0x01.
        *widest.last_mut().expect("stream") = 0x01;
        let (literals, used) = decoder
            .decode(fearless_simd::Fallback::new(), &widest, 1 << 17, None)
            .expect("valid");
        assert_eq!(literals.as_slice(), [0, 1, 0, 1, 0, 1]);
        assert_eq!(used, 15);
    }

    #[test]
    fn treeless_blocks_use_the_dictionary_tree_until_the_frame_describes_one() {
        let mut dictionary = HuffmanTable::default();
        // The RFC example tree: A = 0 and B = 1 at one and two bits.
        dictionary.read(&[132u8, 0x43, 0x20, 0x10]).expect("valid");
        dictionary.build_shared();
        let stream = 0b0000_1101u8; // A then B.
        let header = 3 | (2 << 4) | (1 << 14);
        let bytes = u32::to_le_bytes(header);
        let treeless = [bytes[0], bytes[1], bytes[2], stream, 0];
        let mut decoder = LiteralsDecoder::default();
        // Without an inherited tree, treeless blocks are invalid.
        assert_eq!(
            decoder
                .decode(
                    fearless_simd::Fallback::new(),
                    &treeless,
                    1 << 17,
                    Some(&dictionary)
                )
                .map(|_| ()),
            Err(InvalidDataKind::LiteralsSection)
        );
        decoder.reset(true);
        // The dictionary's tree is required once the frame inherits it.
        assert_eq!(
            decoder
                .decode(fearless_simd::Fallback::new(), &treeless, 1 << 17, None)
                .map(|_| ()),
            Err(InvalidDataKind::LiteralsSection)
        );
        let (literals, used) = decoder
            .decode(
                fearless_simd::Fallback::new(),
                &treeless,
                1 << 17,
                Some(&dictionary),
            )
            .expect("valid");
        assert_eq!((literals.as_slice(), used), (&[0u8, 1][..], 4));
        // A described tree replaces it: symbols 1 and 2 at one bit each.
        let tree = [129u8, 0x01];
        let header = 2 | (2 << 4) | (3 << 14);
        let bytes = u32::to_le_bytes(header);
        let mut content = alloc::vec![bytes[0], bytes[1], bytes[2]];
        content.extend_from_slice(&tree);
        content.push(0b0000_0110);
        let (own, _) = decoder
            .decode(
                fearless_simd::Fallback::new(),
                &content,
                1 << 17,
                Some(&dictionary),
            )
            .expect("valid");
        let own = own.as_slice().to_vec();
        assert!(own.iter().all(|&symbol| symbol == 1 || symbol == 2));
        let mut treeless = treeless;
        treeless[3] = 0b0000_0110;
        let (again, _) = decoder
            .decode(
                fearless_simd::Fallback::new(),
                &treeless,
                1 << 17,
                Some(&dictionary),
            )
            .expect("the frame's own tree");
        assert_eq!(again.as_slice(), own);
    }

    #[test]
    fn encoding_types_follow_the_two_low_bits() {
        assert_eq!(encoding_type(0b100), EncodingType::Basic);
        assert_eq!(encoding_type(0b101), EncodingType::Rle);
        assert_eq!(encoding_type(0b110), EncodingType::Compressed);
        assert_eq!(encoding_type(0b111), EncodingType::Repeat);
    }
}

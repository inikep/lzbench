//! Block headers and block regeneration (RFC 8878 section 3.1.1.2).
//!
//! A [`BlockDecoder`] holds the entropy state a frame carries from one
//! compressed block to the next: the literals Huffman table, the three
//! sequence tables and the repeat offsets. A Zstandard-format dictionary
//! seeds all of them at the frame start.

use ::core::ops::Range;

use fearless_simd::Simd;

use super::BlockError;
use super::copy::MatchCopy;
use super::dictionary::DecoderEntropy;
use super::literals::LiteralsDecoder;
use super::sequences::{BlockContext, SequenceSection, SequenceTables};
use crate::Backend;
use crate::decompressor::InvalidDataKind;
use crate::shared::format::{
    BLOCK_COMPRESSED, BLOCK_HEADER_SIZE, BLOCK_RAW, BLOCK_RLE, REP_NUM, REP_START,
};

/// How a block's content regenerates its output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BlockKind {
    /// Content copied verbatim.
    Raw,
    /// One byte repeated `size` times.
    Rle,
    /// Literals and sequences.
    Compressed,
}

/// A parsed three-byte block header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BlockHeader {
    /// Whether this block ends the frame.
    pub(crate) last: bool,
    /// The block type.
    pub(crate) kind: BlockKind,
    /// Content size, or the regenerated size of an RLE block.
    pub(crate) size: usize,
}

impl BlockHeader {
    /// Parses a header, rejecting the reserved type and sizes above the
    /// frame's `block_max`.
    pub(crate) fn parse(
        bytes: [u8; BLOCK_HEADER_SIZE],
        block_max: usize,
    ) -> Result<Self, InvalidDataKind> {
        let value = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]);
        let kind = match (value >> 1) & 3 {
            BLOCK_RAW => BlockKind::Raw,
            BLOCK_RLE => BlockKind::Rle,
            BLOCK_COMPRESSED => BlockKind::Compressed,
            _ => return Err(InvalidDataKind::BlockType),
        };
        let size = (value >> 3) as usize;
        if size > block_max {
            return Err(InvalidDataKind::BlockSize);
        }
        Ok(Self {
            last: value & 1 != 0,
            kind,
            size,
        })
    }

    /// Bytes of content that follow the header.
    pub(crate) const fn content_size(&self) -> usize {
        match self.kind {
            BlockKind::Rle => 1,
            BlockKind::Raw | BlockKind::Compressed => self.size,
        }
    }
}

/// Entropy state of one frame's compressed blocks.
#[derive(Debug)]
pub(crate) struct BlockDecoder {
    literals: LiteralsDecoder,
    sequences: SequenceTables,
    reps: [u32; REP_NUM],
}

impl Default for BlockDecoder {
    fn default() -> Self {
        Self {
            literals: LiteralsDecoder::default(),
            sequences: SequenceTables::default(),
            reps: REP_START,
        }
    }
}

impl BlockDecoder {
    /// Heap bytes held between blocks.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.literals.retained_bytes() + self.sequences.retained_bytes()
    }

    /// Restores the frame-start state, keeping allocations: the frame
    /// inherits `dictionary`'s tables and repeat offsets
    /// (`ZSTD_copyDDictParameters`), or starts with none and the format's
    /// initial offsets.
    pub(crate) fn reset(&mut self, dictionary: Option<&DecoderEntropy>) {
        let inherited = dictionary.is_some();
        self.literals.reset(inherited);
        self.sequences.reset(inherited);
        self.reps = dictionary.map_or(REP_START, |entropy| entropy.reps);
    }

    /// Regenerates one block into `out[op..]`, returning the end of its
    /// output; `out[context.history_start..op]` is the frame's history.
    ///
    /// `content` holds exactly [`BlockHeader::content_size`] bytes.
    /// [`BlockError::OutputFull`] reports that `out` ends before the block
    /// does, and cannot grow there. Compressed blocks run on `backend`,
    /// selected here once per block.
    pub(crate) fn decode(
        &mut self,
        backend: Backend,
        header: &BlockHeader,
        content: &[u8],
        out: &mut dyn BlockOutput,
        op: usize,
        context: BlockContext<'_>,
    ) -> Result<usize, BlockError> {
        match header.kind {
            BlockKind::Raw => {
                out.put(op, content)?;
                Ok(op + content.len())
            }
            BlockKind::Rle => {
                let &byte = content.first().ok_or(InvalidDataKind::BlockSize)?;
                let end = op + header.size;
                out.target(op..end)?.fill(byte);
                Ok(end)
            }
            BlockKind::Compressed => {
                let copy = if backend.has_fast_shuffle() {
                    MatchCopy::Shuffle
                } else {
                    MatchCopy::Scalar
                };
                let target = Target { out, op, context };
                fearless_simd::dispatch!(backend.block_decode_level(), simd => self.decode_compressed(simd, copy, content, target))
            }
        }
    }

    /// Decodes a compressed block's literals and sequences; the two hot
    /// loops run with `simd`'s target features.
    fn decode_compressed<S: Simd>(
        &mut self,
        simd: S,
        copy: MatchCopy,
        content: &[u8],
        target: Target<'_>,
    ) -> Result<usize, BlockError> {
        let Target { out, op, context } = target;
        let tree = context.entropy.map(|entropy| &entropy.huffman);
        let (literals, used) = self
            .literals
            .decode(simd, content, context.block_max, tree)?;
        let sequences = content.get(used..).unwrap_or(&[]);
        let (count, header_size) = self.sequences.read_header_on(simd, sequences)?;
        if count == 0 {
            // The literals section already bounds its size by the block's.
            out.put(op, &literals.bytes[..literals.len])?;
            return Ok(op + literals.len);
        }
        let stream = sequences.get(header_size..).unwrap_or(&[]);
        let section = SequenceSection {
            count,
            stream,
            literals,
        };
        self.sequences.execute(
            simd,
            copy,
            section,
            &mut self.reps,
            Target { out, op, context },
        )
    }
}

/// A block's output buffer: an initialized prefix, history included, that
/// some outputs extend on demand.
///
/// Safe code can only write initialized memory, so an output that starts
/// uninitialized must be zeroed before the decoder writes it. Extending it
/// a chunk at a time just ahead of the writes keeps the zeroed lines in the
/// cache for the copies that overwrite them, where zeroing a whole block
/// ahead would write each byte twice to the next level.
pub(crate) trait BlockOutput {
    /// The initialized buffer.
    fn buffer(&mut self) -> &mut [u8];

    /// Extends the buffer to at least `end` bytes, or as far as it may
    /// grow, returning whether it grew at all.
    fn grow(&mut self, end: usize) -> bool;

    /// Whether [`Self::grow`] or [`Self::append`] may ever extend the
    /// buffer; a buffer that never grows runs the sequence loop without
    /// stops.
    fn grows(&self) -> bool {
        true
    }

    /// Cuts the buffer at `at` and appends `literals`, then a match of
    /// `match_length` bytes from `offset` bytes before the literals' end
    /// (ignored without a match), writing each byte once. Returns `false`,
    /// having changed nothing, where the output cannot append, the match
    /// reaches before the buffer, or the sequence passes the block's room.
    fn append(
        &mut self,
        _at: usize,
        _literals: &[u8],
        _offset: usize,
        _match_length: usize,
    ) -> bool {
        false
    }
}

impl dyn BlockOutput + '_ {
    /// `range` of the buffer, extended to its end first if needed.
    pub(crate) fn target(&mut self, range: Range<usize>) -> Result<&mut [u8], BlockError> {
        if self.buffer().len() < range.end {
            self.grow(range.end);
        }
        self.buffer().get_mut(range).ok_or(BlockError::OutputFull)
    }

    /// Writes `bytes` at `at`: over the buffer where it reaches that far,
    /// else appended where the output can, else into room grown for them.
    pub(crate) fn put(&mut self, at: usize, bytes: &[u8]) -> Result<(), BlockError> {
        let end = at + bytes.len();
        if self.buffer().len() < end && self.append(at, bytes, 0, 0) {
            return Ok(());
        }
        self.target(at..end)?.copy_from_slice(bytes);
        Ok(())
    }
}

/// A buffer that is initialized throughout and never grows.
impl BlockOutput for &mut [u8] {
    fn buffer(&mut self) -> &mut [u8] {
        self
    }

    fn grow(&mut self, _end: usize) -> bool {
        false
    }

    fn grows(&self) -> bool {
        false
    }
}

/// Where a block's output goes: `out[op..]`, after the history that
/// `context` bounds.
pub(crate) struct Target<'a> {
    /// The output buffer, history included.
    pub(crate) out: &'a mut dyn BlockOutput,
    /// Where the block's output starts.
    pub(crate) op: usize,
    /// The history's start, the frame's limits and its dictionary.
    pub(crate) context: BlockContext<'a>,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::decompressor::core::copy::WILDCOPY_OVERLENGTH;
    use alloc::vec::Vec;

    /// An output that zeroes `step` bytes at a time up to `limit`, as a
    /// caller's vector does, and appends byte by byte when `appends`.
    pub(crate) struct Growing {
        pub(crate) buffer: Vec<u8>,
        pub(crate) limit: usize,
        pub(crate) step: usize,
        pub(crate) appends: bool,
    }

    impl BlockOutput for Growing {
        fn buffer(&mut self) -> &mut [u8] {
            &mut self.buffer
        }

        fn grow(&mut self, end: usize) -> bool {
            let target = end.max(self.buffer.len() + self.step).min(self.limit);
            if target <= self.buffer.len() {
                return false;
            }
            self.buffer.resize(target, 0);
            true
        }

        fn append(
            &mut self,
            at: usize,
            literals: &[u8],
            offset: usize,
            match_length: usize,
        ) -> bool {
            let lit_end = at + literals.len();
            if !self.appends
                || at > self.buffer.len()
                || lit_end + match_length > self.limit
                || (match_length > 0 && !(1..=lit_end).contains(&offset))
            {
                return false;
            }
            self.buffer.truncate(at);
            self.buffer.extend_from_slice(literals);
            for _ in 0..match_length {
                let byte = self.buffer[self.buffer.len() - offset];
                self.buffer.push(byte);
            }
            true
        }
    }

    #[test]
    fn fixed_buffers_never_grow_and_report_running_out() {
        let mut storage = *b"abcdef";
        let mut fixed = &mut storage[..];
        let out: &mut dyn BlockOutput = &mut fixed;
        assert!(!out.grow(10));
        assert!(!out.append(0, b"xy", 0, 0));
        out.put(1, b"XY").expect("fits");
        assert_eq!(out.target(4..6).expect("fits"), b"ef");
        assert_eq!(out.target(4..7), Err(BlockError::OutputFull));
        assert_eq!(out.put(5, b"zz"), Err(BlockError::OutputFull));
        assert_eq!(&storage, b"aXYdef");
    }

    #[test]
    fn growing_buffers_append_or_zero_room_for_writes() {
        let mut growing = Growing {
            buffer: b"ab".to_vec(),
            limit: 8,
            step: 3,
            appends: true,
        };
        let out: &mut dyn BlockOutput = &mut growing;
        // Past the buffer, bytes are appended where the output can.
        out.put(2, b"cd").expect("appended");
        assert_eq!(out.buffer(), b"abcd");
        // Within it, they are written in place.
        out.put(1, b"B").expect("in place");
        assert_eq!(out.buffer(), b"aBcd");
        // Without appends, room is zeroed a step at a time, to the limit.
        growing.appends = false;
        let out: &mut dyn BlockOutput = &mut growing;
        out.put(4, b"e").expect("grown");
        assert_eq!(out.buffer(), b"aBcde\0\0");
        assert_eq!(out.put(7, b"fg"), Err(BlockError::OutputFull));
        assert_eq!(out.buffer().len(), 8);
    }

    const CONTEXT: BlockContext<'static> = BlockContext {
        history_start: 0,
        window: 1 << 17,
        block_max: 1 << 17,
        dictionary: &[],
        entropy: None,
    };

    /// Decodes one block after `out` and appends its output.
    fn decode(
        decoder: &mut BlockDecoder,
        header: &BlockHeader,
        content: &[u8],
        out: &mut Vec<u8>,
        context: BlockContext<'_>,
    ) -> Result<(), BlockError> {
        let start = out.len();
        let mut buffer = out.clone();
        buffer.resize(start + context.block_max + WILDCOPY_OVERLENGTH, 0);
        let end = decoder.decode(
            Backend::SCALAR,
            header,
            content,
            &mut buffer.as_mut_slice(),
            start,
            context,
        )?;
        out.extend_from_slice(&buffer[start..end]);
        Ok(())
    }

    #[test]
    fn headers_carry_the_last_flag_type_and_size() {
        let raw = BlockHeader::parse([0x21, 0, 0], 100).expect("raw");
        assert_eq!(
            raw,
            BlockHeader {
                last: true,
                kind: BlockKind::Raw,
                size: 4
            }
        );
        assert_eq!(raw.content_size(), 4);
        let rle = BlockHeader::parse([0x2a, 0x01, 0], 100).expect("rle");
        assert_eq!(rle.kind, BlockKind::Rle);
        assert_eq!(rle.size, 37);
        assert!(!rle.last);
        assert_eq!(rle.content_size(), 1);
        let compressed = BlockHeader::parse([0x15, 0, 0], 100).expect("compressed");
        assert_eq!(compressed.kind, BlockKind::Compressed);
        assert_eq!(compressed.content_size(), 2);
    }

    #[test]
    fn reserved_types_and_oversized_blocks_are_rejected() {
        assert_eq!(
            BlockHeader::parse([0x06, 0, 0], 100),
            Err(InvalidDataKind::BlockType)
        );
        assert_eq!(
            BlockHeader::parse([0x00, 0x04, 0], 100),
            Err(InvalidDataKind::BlockSize)
        );
    }

    #[test]
    fn raw_rle_and_literal_only_blocks_regenerate() {
        let mut decoder = BlockDecoder::default();
        let mut out = Vec::new();
        let raw = BlockHeader::parse([0x18, 0, 0], 100).expect("raw");
        decode(&mut decoder, &raw, b"abc", &mut out, CONTEXT).expect("raw");
        let rle = BlockHeader::parse([0x1a, 0, 0], 100).expect("rle");
        decode(&mut decoder, &rle, b"z", &mut out, CONTEXT).expect("rle");
        assert_eq!(out, b"abczzz");
        // Raw literals "hi" and no sequences.
        let compressed = BlockHeader::parse([0x1c, 0, 0], 100).expect("compressed");
        decode(
            &mut decoder,
            &compressed,
            &[2 << 3, b'h', b'i', 0],
            &mut out,
            CONTEXT,
        )
        .expect("compressed");
        assert_eq!(out, b"abczzzhi");
        assert_eq!(
            decode(&mut decoder, &rle, b"", &mut out, CONTEXT),
            Err(BlockError::Invalid(InvalidDataKind::BlockSize))
        );
    }

    #[test]
    fn compressed_blocks_execute_sequences_and_keep_offsets() {
        let mut decoder = BlockDecoder::default();
        let mut out = b"abc".to_vec();
        // Raw literals "x", one sequence with RLE tables: literal length 1,
        // offset code 0 (repeat 1 = 1), match length code 0 (3 bytes).
        let content = [1 << 3, b'x', 1, 0x54, 1, 0, 0, 0x01];
        let header = BlockHeader {
            last: true,
            kind: BlockKind::Compressed,
            size: content.len(),
        };
        decode(&mut decoder, &header, &content, &mut out, CONTEXT).expect("valid");
        assert_eq!(out, b"abcxxxx");
        decoder.reset(None);
        assert_eq!(decoder.reps, REP_START);
    }

    #[test]
    fn a_dictionary_seeds_offsets_and_repeat_tables() {
        let bytes =
            crate::decompressor::core::dictionary::tests::dictionary([2, 4, 8], b"ABCDEFGH");
        let (entropy, _) = DecoderEntropy::load(&bytes).expect("valid");
        let mut decoder = BlockDecoder::default();
        decoder.reset(Some(&entropy));
        assert_eq!(decoder.reps, [2, 4, 8]);
        // Raw literals "x", one sequence with every table repeated: the
        // dictionary's tables give literal length 0 (code 0), offset code 0
        // and match length 3 (code 0) with no extra bits; the stream holds
        // only the fifteen initial state bits. A zero literal length makes
        // offset code 0 select repeat offset 2, four bytes back into the
        // dictionary.
        let content = [1 << 3, b'x', 1, 0xfc, 0x00, 0x80];
        let header = BlockHeader {
            last: true,
            kind: BlockKind::Compressed,
            size: content.len(),
        };
        let mut context = CONTEXT;
        context.dictionary = b"ABCDEFGH";
        context.entropy = Some(&entropy);
        let mut out = Vec::new();
        decode(&mut decoder, &header, &content, &mut out, context).expect("valid");
        assert_eq!(out, b"EFGx");
        assert_eq!(decoder.reps, [4, 2, 8]);
        // Without the tables, repeat modes have nothing to repeat.
        decoder.reset(None);
        assert_eq!(
            decode(&mut decoder, &header, &content, &mut Vec::new(), CONTEXT),
            Err(BlockError::Invalid(InvalidDataKind::SequencesSection))
        );
    }
}

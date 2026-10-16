//! What a dictionary gives the decoder: history before each frame and, in
//! the Zstandard format, entropy tables and repeat offsets.
//!
//! [`DecoderEntropy::load`] is the reference's `ZSTD_loadDEntropy`: it reads
//! the Huffman table, the offset, match length and literal length FSE tables
//! and the three repeat offsets that follow a dictionary's magic number and
//! ID. [`DictView`] is the borrowed form the frame and block decoders
//! consume, whatever owns the bytes.

use ::core::fmt;

use super::fse::SeqTable;
use super::huffman::HuffmanTable;
use super::sequences::{SymbolType, read_table};
use crate::decompressor::InvalidDataKind;
use crate::shared::format::REP_NUM;

/// Bytes of a Zstandard-format dictionary before its entropy section: the
/// magic number and the ID.
const HEADER_SIZE: usize = 8;

/// The entropy section of a Zstandard-format dictionary
/// (`ZSTD_entropyDTables_t`).
#[derive(Clone)]
pub(crate) struct DecoderEntropy {
    /// The literals tree, with its pair rows built, so that decoding never
    /// modifies it.
    pub(crate) huffman: HuffmanTable,
    /// Literal length table.
    pub(crate) literal_lengths: SeqTable,
    /// Offset code table.
    pub(crate) offsets: SeqTable,
    /// Match length table.
    pub(crate) match_lengths: SeqTable,
    /// Repeat offsets a frame starts with.
    pub(crate) reps: [u32; REP_NUM],
}

impl fmt::Debug for DecoderEntropy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecoderEntropy")
            .field("reps", &self.reps)
            .finish_non_exhaustive()
    }
}

impl DecoderEntropy {
    /// Parses the entropy section of the Zstandard-format dictionary
    /// `dictionary` (magic number and ID included), returning the tables
    /// and the section's end. Fails exactly where `ZSTD_loadDEntropy` does:
    /// on malformed or oversized tables, and on repeat offsets of zero or
    /// beyond the content that follows them.
    pub(crate) fn load(dictionary: &[u8]) -> Result<(Self, usize), InvalidDataKind> {
        if dictionary.len() <= HEADER_SIZE {
            return Err(InvalidDataKind::HuffmanTable);
        }
        let mut position = HEADER_SIZE;
        let rest = |position: usize| dictionary.get(position..).unwrap_or_default();
        let mut huffman = HuffmanTable::default();
        position += huffman.read(rest(position))?;
        huffman.build_shared();
        let (offsets, used) = read_table(SymbolType::Offsets, rest(position))?;
        position += used;
        let (match_lengths, used) = read_table(SymbolType::MatchLengths, rest(position))?;
        position += used;
        let (literal_lengths, used) = read_table(SymbolType::LiteralLengths, rest(position))?;
        position += used;
        let (reps, content) = rest(position)
            .split_first_chunk::<{ 4 * REP_NUM }>()
            .ok_or(InvalidDataKind::Offset)?;
        let reps: [u32; REP_NUM] = ::core::array::from_fn(|index| {
            let bytes = [
                reps[4 * index],
                reps[4 * index + 1],
                reps[4 * index + 2],
                reps[4 * index + 3],
            ];
            u32::from_le_bytes(bytes)
        });
        let valid = |rep: &u32| *rep != 0 && *rep as usize <= content.len();
        if !reps.iter().all(valid) {
            return Err(InvalidDataKind::Offset);
        }
        let entropy = Self {
            huffman,
            literal_lengths,
            offsets,
            match_lengths,
            reps,
        };
        Ok((entropy, position + 4 * REP_NUM))
    }

    /// Heap bytes the tables hold.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.huffman.retained_bytes()
            + self.literal_lengths.retained_bytes()
            + self.offsets.retained_bytes()
            + self.match_lengths.retained_bytes()
    }
}

/// A dictionary as the decoder consumes it, borrowed from whatever owns
/// its bytes and tables.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DictView<'a> {
    /// History before every frame's first byte.
    pub(crate) content: &'a [u8],
    /// The dictionary's ID; zero for raw content and for Zstandard-format
    /// dictionaries that declare none.
    pub(crate) id: u32,
    /// Entropy tables and repeat offsets, for Zstandard-format
    /// dictionaries.
    pub(crate) entropy: Option<&'a DecoderEntropy>,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// A minimal Zstandard-format dictionary with ID 7: a two-symbol
    /// Huffman tree, FSE tables of one symbol each, `reps` and `content`.
    pub(crate) fn dictionary(reps: [u32; 3], content: &[u8]) -> Vec<u8> {
        let mut bytes = 0xEC30_A437u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&7u32.to_le_bytes());
        // Huffman: one direct weight (symbol 0 at weight 1), symbol 1
        // implied.
        bytes.extend_from_slice(&[128, 0x10]);
        // Each FSE table gives all of its states to symbol 0.
        for _ in 0..3 {
            bytes.extend_from_slice(&fse_single_symbol());
        }
        for rep in reps {
            bytes.extend_from_slice(&rep.to_le_bytes());
        }
        bytes.extend_from_slice(content);
        bytes
    }

    /// A distribution of accuracy log 5 whose 32 states all go to symbol 0.
    fn fse_single_symbol() -> Vec<u8> {
        // Four bits of log - 5 = 0, then symbol 0's count + 1 = 33 in six
        // bits: values whose low five bits reach 30 take six bits and
        // decode to `value - 30` (`FSE_readNCount`), so 33 is written 63.
        let bits: u32 = 63 << 4;
        alloc::vec![bits as u8, (bits >> 8) as u8]
    }

    #[test]
    fn a_minimal_dictionary_loads_its_tables_and_offsets() {
        let bytes = dictionary([1, 4, 8], b"content!");
        let (entropy, end) = DecoderEntropy::load(&bytes).expect("valid");
        assert_eq!(end, bytes.len() - 8);
        assert_eq!(entropy.reps, [1, 4, 8]);
        assert_eq!(entropy.offsets.log(), 5);
        assert!(entropy.retained_bytes() > 0);
        assert!(alloc::format!("{entropy:?}").contains("reps"));
    }

    #[test]
    fn repeat_offsets_must_be_non_zero_and_within_the_content() {
        for reps in [[0, 1, 1], [1, 9, 1], [1, 1, u32::MAX]] {
            let bytes = dictionary(reps, b"content!");
            assert_eq!(
                DecoderEntropy::load(&bytes).map(|_| ()),
                Err(InvalidDataKind::Offset),
                "{reps:?}"
            );
        }
        // Eight content bytes allow an offset of eight.
        let bytes = dictionary([8, 8, 8], b"content!");
        assert!(DecoderEntropy::load(&bytes).is_ok());
    }

    #[test]
    fn truncated_dictionaries_are_rejected() {
        let bytes = dictionary([1, 1, 1], b"x");
        for len in 0..bytes.len() - 1 {
            assert!(DecoderEntropy::load(&bytes[..len]).is_err(), "length {len}");
        }
        assert!(DecoderEntropy::load(&bytes).is_ok());
    }
}

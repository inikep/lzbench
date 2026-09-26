//! Static dictionary word lists, built in and custom.
//!
//! A static dictionary word is addressed by its length and its index among the
//! words of that length. RFC 7932 fixes one such list; [RFC 9841 section 5]
//! lets a shared dictionary supply its own, described by twenty-eight
//! `SIZE_BITS_BY_LENGTH` entries covering lengths four to thirty-one followed
//! by the word bytes laid end to end, shortest length first.
//!
//! Ports `BrotliSizeBitsToOffsets` and `ParseWordList` from
//! `c/common/shared_dictionary.c` of the pinned reference (`google/brotli`
//! v1.2.0, commit `028fb5a`). The built-in tables are the ones
//! [`shared::dictionary`](crate::shared::dictionary)
//! already holds, borrowed rather than copied.
//!
//! [RFC 9841 section 5]: https://www.rfc-editor.org/rfc/rfc9841.html#section-5

use alloc::vec::Vec;

use alloc::borrow::Cow;

use thiserror::Error;

use crate::shared::dictionary::{
    BUILTIN_OFFSETS_BY_LENGTH, BUILTIN_SIZE_BITS_BY_LENGTH, BUILTIN_WORDS,
};

use super::transform::MAX_WORD_LENGTH;

/// Shortest word a static dictionary may hold
/// (`SHARED_BROTLI_MIN_DICTIONARY_WORD_LENGTH`).
pub(crate) const MIN_WORD_LENGTH: usize = 4;

/// Word lengths the wire describes, one `SIZE_BITS_BY_LENGTH` byte each.
pub(crate) const NUM_ENCODED_LENGTHS: usize = MAX_WORD_LENGTH - MIN_WORD_LENGTH + 1;

/// Largest `SIZE_BITS_BY_LENGTH` value the RFC allows (`BROTLI_MAX_SIZE_BITS`).
pub(crate) const MAX_SIZE_BITS: u8 = 15;

/// Why a word list could not be built from its parts.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum WordListError {
    /// A length claimed more words than the RFC allows.
    #[error("length {length} claims 2^{bits} words, past the limit of 2^{MAX_SIZE_BITS}")]
    TooManySizeBits {
        /// The word length that claimed it.
        length: usize,
        /// The exponent it claimed.
        bits: u8,
    },
    /// The word data is not the length the size bits describe.
    #[error("the size bits describe {expected} bytes of words, but {found} were given")]
    DataLength {
        /// How many bytes the size bits describe.
        expected: usize,
        /// How many bytes were given.
        found: usize,
    },
}

/// One list of static dictionary words, built in or custom.
///
/// Immutable once built, and cheap to construct for the built-in list: the word
/// bytes are borrowed from static storage rather than copied.
#[derive(Debug, Clone)]
pub(crate) struct WordList {
    /// Base-2 logarithm of the word count at each length, zero for none.
    ///
    /// Indexed by length, so entries below [`MIN_WORD_LENGTH`] are always zero.
    size_bits_by_length: [u8; MAX_WORD_LENGTH + 1],
    /// Byte offset at which the words of each length begin.
    offsets_by_length: [u32; MAX_WORD_LENGTH + 1],
    /// Every word, shortest length first.
    data: Cow<'static, [u8]>,
}

impl WordList {
    /// Returns the RFC 7932 word list every ordinary stream uses.
    pub(crate) fn builtin() -> Self {
        let mut size_bits_by_length = [0u8; MAX_WORD_LENGTH + 1];
        let mut offsets_by_length = [0u32; MAX_WORD_LENGTH + 1];
        // The shared tables run to thirty-two entries because the reference
        // indexes them by a length that may reach thirty-two mid-computation;
        // only the first thirty-two matter and only the first thirty-two exist.
        size_bits_by_length.copy_from_slice(&BUILTIN_SIZE_BITS_BY_LENGTH[..=MAX_WORD_LENGTH]);
        offsets_by_length.copy_from_slice(&BUILTIN_OFFSETS_BY_LENGTH[..=MAX_WORD_LENGTH]);
        Self {
            size_bits_by_length,
            offsets_by_length,
            data: Cow::Borrowed(&BUILTIN_WORDS[..]),
        }
    }

    /// Builds a word list from the wire's size bits and word bytes.
    ///
    /// `size_bits` is indexed by length minus [`MIN_WORD_LENGTH`], exactly as
    /// the wire stores it.
    ///
    /// # Errors
    ///
    /// Returns [`WordListError::TooManySizeBits`] past the RFC's fifteen-bit
    /// ceiling, and [`WordListError::DataLength`] when `data` is not the length
    /// the size bits describe.
    pub(crate) fn from_parts(
        size_bits: &[u8; NUM_ENCODED_LENGTHS],
        data: Cow<'static, [u8]>,
    ) -> Result<Self, WordListError> {
        let mut size_bits_by_length = [0u8; MAX_WORD_LENGTH + 1];
        for (index, &bits) in size_bits.iter().enumerate() {
            let length = MIN_WORD_LENGTH + index;
            if bits > MAX_SIZE_BITS {
                return Err(WordListError::TooManySizeBits { length, bits });
            }
            size_bits_by_length[length] = bits;
        }
        let (offsets_by_length, expected) = offsets(&size_bits_by_length);
        if data.len() != expected {
            return Err(WordListError::DataLength {
                expected,
                found: data.len(),
            });
        }
        Ok(Self {
            size_bits_by_length,
            offsets_by_length,
            data,
        })
    }

    /// Returns the base-2 logarithm of how many words of `length` exist.
    ///
    /// Zero means the list holds no word of that length, not one word.
    pub(crate) fn size_bits(&self, length: usize) -> u8 {
        self.size_bits_by_length
            .get(length)
            .copied()
            .unwrap_or_default()
    }

    /// Returns how many words of `length` the list holds.
    pub(crate) fn word_count(&self, length: usize) -> usize {
        match self.size_bits(length) {
            0 => 0,
            bits => 1usize << bits,
        }
    }

    /// Returns the byte offset at which the words of `length` begin.
    pub(crate) fn offset(&self, length: usize) -> usize {
        self.offsets_by_length
            .get(length)
            .copied()
            .unwrap_or_default() as usize
    }

    /// Returns one word, or an empty slice when it does not exist.
    ///
    /// `index` is the word's position among the words of its own length, which
    /// is how a static dictionary reference addresses it.
    pub(crate) fn word(&self, length: usize, index: usize) -> &[u8] {
        if index >= self.word_count(length) {
            return &[];
        }
        let start = self.offset(length) + index * length;
        self.data.get(start..start + length).unwrap_or_default()
    }

    /// Returns every word byte, shortest length first.
    pub(crate) fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns whether the list holds no word at all.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns how many bytes [`WordList::serialize`] will append.
    pub(crate) fn wire_len(&self) -> usize {
        NUM_ENCODED_LENGTHS + self.data.len()
    }

    /// Appends the list to `out` in the RFC 9841 word list layout.
    pub(crate) fn serialize(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.size_bits_by_length[MIN_WORD_LENGTH..=MAX_WORD_LENGTH]);
        out.extend_from_slice(&self.data);
    }

    /// Returns the size bits in the order the wire stores them.
    #[cfg(test)]
    pub(crate) fn encoded_size_bits(&self) -> [u8; NUM_ENCODED_LENGTHS] {
        let mut encoded = [0u8; NUM_ENCODED_LENGTHS];
        encoded.copy_from_slice(&self.size_bits_by_length[MIN_WORD_LENGTH..=MAX_WORD_LENGTH]);
        encoded
    }
}

/// Turns size bits into start offsets, and returns the total word byte count.
///
/// Mirrors `BrotliSizeBitsToOffsets`. Every product fits a `u32` because a
/// length is at most thirty-one and an exponent at most fifteen, so one length
/// contributes at most `31 << 15` bytes and thirty-two lengths at most
/// `32 * 31 << 15`, comfortably inside the type.
fn offsets(size_bits_by_length: &[u8; MAX_WORD_LENGTH + 1]) -> ([u32; MAX_WORD_LENGTH + 1], usize) {
    let mut offsets = [0u32; MAX_WORD_LENGTH + 1];
    let mut position = 0u32;
    for (length, &bits) in size_bits_by_length.iter().enumerate() {
        offsets[length] = position;
        if bits != 0 {
            position += (length as u32) << bits;
        }
    }
    (offsets, position as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a word list holding `count` words of `length`, filled with `fill`.
    fn list(length: usize, bits: u8, fill: u8) -> WordList {
        let mut size_bits = [0u8; NUM_ENCODED_LENGTHS];
        size_bits[length - MIN_WORD_LENGTH] = bits;
        let data = vec![fill; length << bits];
        WordList::from_parts(&size_bits, Cow::Owned(data)).expect("well formed")
    }

    #[test]
    fn the_builtin_list_is_the_reference_one() {
        let words = WordList::builtin();

        assert_eq!(words.data().len(), 122_784);
        assert_eq!(words.word(4, 0), b"time");
        assert_eq!(words.word_count(4), 1024);
        assert_eq!(words.size_bits(4), 10);
    }

    #[test]
    fn the_builtin_offsets_tile_every_length() {
        let words = WordList::builtin();
        let mut expected = 0usize;

        for length in MIN_WORD_LENGTH..=MAX_WORD_LENGTH {
            assert_eq!(words.offset(length), expected, "length {length}");
            expected += length * words.word_count(length);
        }
        assert_eq!(expected, words.data().len());
    }

    #[test]
    fn a_zero_exponent_means_no_words_rather_than_one() {
        let words = WordList::from_parts(&[0; NUM_ENCODED_LENGTHS], Cow::Owned(Vec::new()))
            .expect("well formed");

        assert_eq!(words.word_count(4), 0);
        assert_eq!(words.word(4, 0), b"");
        assert!(words.is_empty());
    }

    #[test]
    fn words_are_addressed_by_length_and_index() {
        let mut size_bits = [0u8; NUM_ENCODED_LENGTHS];
        size_bits[0] = 1;
        let data = b"abcdefgh".to_vec();
        let words = WordList::from_parts(&size_bits, Cow::Owned(data)).expect("well formed");

        assert_eq!(words.word(4, 0), b"abcd");
        assert_eq!(words.word(4, 1), b"efgh");
        assert_eq!(words.word(4, 2), b"");
        assert_eq!(words.word(5, 0), b"");
    }

    #[test]
    fn an_exponent_past_the_limit_is_refused() {
        let mut size_bits = [0u8; NUM_ENCODED_LENGTHS];
        size_bits[0] = MAX_SIZE_BITS + 1;

        assert_eq!(
            WordList::from_parts(&size_bits, Cow::Owned(Vec::new())).err(),
            Some(WordListError::TooManySizeBits {
                length: MIN_WORD_LENGTH,
                bits: MAX_SIZE_BITS + 1,
            })
        );
    }

    #[test]
    fn data_of_the_wrong_length_is_refused() {
        let mut size_bits = [0u8; NUM_ENCODED_LENGTHS];
        size_bits[0] = 1;

        assert_eq!(
            WordList::from_parts(&size_bits, Cow::Owned(vec![0; 7])).err(),
            Some(WordListError::DataLength {
                expected: 8,
                found: 7,
            })
        );
    }

    #[test]
    fn the_largest_list_a_length_may_hold_is_accepted() {
        let words = list(MAX_WORD_LENGTH, MAX_SIZE_BITS, b'z');

        assert_eq!(words.word_count(MAX_WORD_LENGTH), 1 << MAX_SIZE_BITS);
        assert_eq!(words.data().len(), MAX_WORD_LENGTH << MAX_SIZE_BITS);
    }

    #[test]
    fn serializing_round_trips_through_the_wire_layout() {
        let words = list(6, 2, b'q');
        let mut bytes = Vec::new();
        words.serialize(&mut bytes);

        assert_eq!(bytes.len(), words.wire_len());
        let (encoded, data) = bytes.split_at(NUM_ENCODED_LENGTHS);
        let size_bits: [u8; NUM_ENCODED_LENGTHS] = encoded.try_into().expect("28 bytes");
        let parsed = WordList::from_parts(&size_bits, Cow::Owned(data.to_vec()))
            .expect("what was written parses");

        assert_eq!(parsed.encoded_size_bits(), words.encoded_size_bits());
        assert_eq!(parsed.data(), words.data());
    }

    #[test]
    fn the_encoded_size_bits_cover_lengths_four_to_thirty_one() {
        let words = list(MAX_WORD_LENGTH, 1, b'a');
        let encoded = words.encoded_size_bits();

        assert_eq!(encoded.len(), NUM_ENCODED_LENGTHS);
        assert_eq!(encoded[NUM_ENCODED_LENGTHS - 1], 1);
        assert_eq!(encoded[0], 0);
    }

    #[test]
    fn a_length_outside_the_covered_range_holds_nothing() {
        let words = WordList::builtin();

        assert_eq!(words.word_count(3), 0);
        assert_eq!(words.word_count(32), 0);
        assert_eq!(words.size_bits(99), 0);
        assert_eq!(words.offset(99), 0);
    }
}

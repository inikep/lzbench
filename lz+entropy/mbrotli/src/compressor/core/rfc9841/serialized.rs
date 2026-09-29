//! The serialized shared dictionary stream: parsing, and writing it back.
//!
//! [RFC 9841 section 5] defines one self-describing byte stream that carries an
//! LZ77 prefix, custom static dictionary word lists, custom transform lists,
//! the combinations of the two a decoder may fall through, and a sixty-four
//! entry context map that picks the first combination. This module is the codec
//! for it, and nothing above it needs to know the layout.
//!
//! Ports `DecodeSharedDictionary`, `DryParseDictionary`, `ParseDictionary`,
//! `ParseWordList`, `ParseTransformsList` and `ParsePrefixSuffixTable` from
//! `c/common/shared_dictionary.c` of the pinned reference (`google/brotli`
//! v1.2.0, commit `028fb5a`), which the reference compiles only when
//! `BROTLI_EXPERIMENTAL` is defined.
//!
//! Two deliberate differences from that reference, both recorded in
//! `architecture/shared-dictionary.md`:
//!
//! - the reference caps `LZ77_DICTIONARY_LENGTH` at `0x3FFFFFFF` with the
//!   comment that the limit "is not specified"; this parser applies the RFC's
//!   own ceiling, the largest sliding window large-window Brotli can address,
//!   which is fifteen bytes smaller;
//! - the reference stops parsing once the structure is complete and ignores
//!   whatever follows; [`parse`] reports how many bytes it consumed, and the
//!   caller decides whether a tail is allowed. The public API refuses one.
//!
//! [RFC 9841 section 5]: https://www.rfc-editor.org/rfc/rfc9841.html#section-5

use alloc::boxed::Box;
use alloc::vec::Vec;

use alloc::borrow::Cow;

use thiserror::Error;

use super::transform::{TransformList, TransformListError};
use super::varint::{self, VarintError};
use super::words::{MAX_SIZE_BITS, MIN_WORD_LENGTH, NUM_ENCODED_LENGTHS, WordList, WordListError};

/// The two bytes every serialized dictionary starts with.
///
/// The first is an invalid `WBITS` combination for Brotli and large-window
/// Brotli, so a decoder can tell a dictionary from a stream by its first byte.
pub(crate) const MAGIC: [u8; 2] = [0x91, 0x00];

/// Word lists, transform lists or combinations one dictionary may hold
/// (`SHARED_BROTLI_NUM_DICTIONARY_CONTEXTS`).
pub(crate) const MAX_LISTS: usize = 64;

/// Literal contexts the context map covers, one entry each.
pub(crate) const NUM_CONTEXTS: usize = 64;

/// Largest LZ77 prefix the RFC allows: the widest sliding window it can address.
///
/// [RFC 9841 section 5] caps `LZ77_DICTIONARY_LENGTH` at "the maximum possible
/// sliding window size of brotli or large window brotli", which is
/// `(1 << 30) - 16`.
///
/// [RFC 9841 section 5]: https://www.rfc-editor.org/rfc/rfc9841.html#section-5
pub(crate) const MAX_LZ77_DICTIONARY_LENGTH: u64 = (1 << 30) - 16;

/// Which word or transform list a combination names.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum ListRef {
    /// The RFC 7932 list, which the wire spells as the count of custom lists.
    Builtin,
    /// One of the dictionary's own lists, by index.
    Custom(u8),
}

impl ListRef {
    /// Returns the byte the wire stores, given how many custom lists exist.
    const fn encode(self, custom: u8) -> u8 {
        match self {
            Self::Builtin => custom,
            Self::Custom(index) => index,
        }
    }

    /// Reads one wire byte, given how many custom lists exist.
    ///
    /// The byte names a custom list when it is below the count and the built-in
    /// list when it equals it; anything larger is invalid.
    const fn decode(byte: u8, custom: u8) -> Option<Self> {
        if byte == custom {
            Some(Self::Builtin)
        } else if byte < custom {
            Some(Self::Custom(byte))
        } else {
            None
        }
    }
}

/// One pairing of a word list with a transform list.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct Combination {
    /// The word list this combination draws words from.
    pub(crate) words: ListRef,
    /// The transform list this combination draws transforms from.
    pub(crate) transforms: ListRef,
}

/// Ceilings the parser and the builder check before allocating.
///
/// Every field bounds one dimension of the format independently, so a caller
/// can loosen the one their dictionaries need without loosening the rest.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct SerializedLimits {
    /// Largest whole stream, checked before anything inside it is read.
    pub(crate) max_total_bytes: u64,
    /// Largest LZ77 prefix.
    pub(crate) max_prefix_bytes: u64,
    /// Most word lists.
    pub(crate) max_word_lists: usize,
    /// Largest total of every word list's word bytes.
    pub(crate) max_word_bytes: u64,
    /// Most transform lists.
    pub(crate) max_transform_lists: usize,
    /// Largest total of every transform list's wire bytes.
    pub(crate) max_transform_bytes: u64,
    /// Most combinations.
    pub(crate) max_combinations: usize,
}

impl SerializedLimits {
    /// Returns limits wide enough for any dictionary the format can express.
    ///
    /// Used where the bytes are this crate's own — re-parsing what it just
    /// wrote — rather than a caller's. Every caller-facing path takes the
    /// limits from
    /// [`DictionaryLimits`](crate::dictionary::DictionaryLimits) instead.
    #[cfg(test)]
    pub(crate) const fn permissive() -> Self {
        Self {
            max_total_bytes: u64::MAX,
            max_prefix_bytes: MAX_LZ77_DICTIONARY_LENGTH,
            max_word_lists: MAX_LISTS,
            max_word_bytes: u64::MAX,
            max_transform_lists: MAX_LISTS,
            max_transform_bytes: u64::MAX,
            max_combinations: MAX_LISTS,
        }
    }
}

/// Why a serialized dictionary could not be parsed or built.
#[derive(Error, Debug, Clone, Eq, PartialEq)]
pub(crate) enum SerializedError {
    /// The stream did not start with the two magic bytes.
    #[error("a serialized dictionary starts with {MAGIC:02X?}, not {found:02X?}")]
    BadMagic {
        /// The first two bytes that were found, or fewer at the end of input.
        found: Vec<u8>,
    },
    /// The stream ended in the middle of a field.
    #[error("the dictionary ends after {position} bytes, mid-{field}")]
    Truncated {
        /// Which field was being read.
        field: &'static str,
        /// How many bytes the stream held.
        position: usize,
    },
    /// A varint field was malformed.
    #[error("the {field} varint is malformed: {source}")]
    Varint {
        /// Which field carried it.
        field: &'static str,
        /// What was wrong with it.
        #[source]
        source: VarintError,
    },
    /// A boolean field held something other than zero or one.
    #[error("the {field} flag must be 0 or 1, not {value}")]
    NotABoolean {
        /// Which field carried it.
        field: &'static str,
        /// The byte that was found.
        value: u8,
    },
    /// The LZ77 prefix is longer than the format allows.
    #[error("an LZ77 prefix of {length} bytes exceeds the format's {MAX_LZ77_DICTIONARY_LENGTH}")]
    PrefixTooLongForFormat {
        /// The length the stream declared.
        length: u64,
    },
    /// One of the counted dimensions exceeds sixty-four.
    #[error("a dictionary holds at most {MAX_LISTS} {what}, not {count}")]
    TooManyLists {
        /// Which dimension overflowed.
        what: &'static str,
        /// How many were declared.
        count: usize,
    },
    /// A dictionary declared no combinations, which the RFC forbids.
    #[error("a dictionary with custom lists must declare at least one combination")]
    NoCombinations,
    /// A combination named a list that does not exist.
    #[error("a combination names {what} {index}, past the {available} available")]
    UndefinedList {
        /// Which side of the combination named it.
        what: &'static str,
        /// The index it named.
        index: u8,
        /// How many exist, the built-in one excluded.
        available: u8,
    },
    /// A context map entry named a combination that does not exist.
    #[error("context {context} maps to combination {index}, past the {available} declared")]
    UndefinedCombination {
        /// Which of the sixty-four contexts named it.
        context: usize,
        /// The combination it named.
        index: u8,
        /// How many combinations exist.
        available: usize,
    },
    /// A word list was malformed.
    #[error("word list {index} is malformed: {source}")]
    WordList {
        /// Which list.
        index: usize,
        /// What was wrong with it.
        #[source]
        source: WordListError,
    },
    /// A transform list was malformed.
    #[error("transform list {index} is malformed: {source}")]
    TransformList {
        /// Which list.
        index: usize,
        /// What was wrong with it.
        #[source]
        source: TransformListError,
    },
    /// A resource limit was exceeded.
    #[error("the dictionary's {what} of {found} exceeds the limit of {limit}")]
    LimitExceeded {
        /// Which limit was hit.
        what: &'static str,
        /// What the stream asked for.
        found: u64,
        /// What the limit allows.
        limit: u64,
    },
    /// Bytes followed the end of the dictionary structure.
    #[error("{extra} byte(s) follow the end of the dictionary")]
    TrailingBytes {
        /// How many bytes were left over.
        extra: usize,
    },
}

/// A parsed or programmatically built serialized shared dictionary.
///
/// Owns everything it describes, so it outlives whatever it was parsed from.
#[derive(Debug, Clone, Default)]
pub(crate) struct SerializedDictionaryData {
    /// The LZ77 prefix, absent when `LZ77_DICTIONARY_LENGTH` is zero.
    prefix: Option<Box<[u8]>>,
    /// Custom word lists, in wire order.
    word_lists: Vec<WordList>,
    /// Custom transform lists, in wire order.
    transform_lists: Vec<TransformList>,
    /// Combinations, in wire order; empty when there are no custom lists,
    /// because the wire leaves the whole block out in that case.
    combinations: Vec<Combination>,
    /// The context map, absent when `CONTEXT_ENABLED` is zero.
    context_map: Option<[u8; NUM_CONTEXTS]>,
}

impl SerializedDictionaryData {
    /// Conservative heap bound while this description and prepared indexes coexist.
    pub(crate) fn allocation_bound(&self) -> usize {
        self.prefix()
            .len()
            .saturating_add(self.word_lists.capacity() * size_of::<WordList>())
            .saturating_add(self.transform_lists.capacity() * size_of::<TransformList>())
            .saturating_add(self.combinations.capacity() * size_of::<Combination>())
            .saturating_add(
                self.word_lists
                    .iter()
                    .map(|w| w.data().len())
                    .sum::<usize>(),
            )
            .saturating_add(
                self.transform_lists
                    .iter()
                    .map(|t| t.wire_len() + t.stringlet_count() * size_of::<u16>())
                    .sum::<usize>(),
            )
    }

    /// Returns the LZ77 prefix, or an empty slice when there is none.
    pub(crate) fn prefix(&self) -> &[u8] {
        self.prefix.as_deref().unwrap_or_default()
    }

    /// Returns whether the stream carries an LZ77 prefix at all.
    ///
    /// A present but empty prefix is impossible: the wire spells "no prefix"
    /// and "an empty prefix" the same way, as a zero length.
    pub(crate) fn has_prefix(&self) -> bool {
        self.prefix.is_some()
    }

    /// Returns the custom word lists in wire order.
    pub(crate) fn word_lists(&self) -> &[WordList] {
        &self.word_lists
    }

    /// Returns the custom transform lists in wire order.
    pub(crate) fn transform_lists(&self) -> &[TransformList] {
        &self.transform_lists
    }

    /// Returns the combinations in wire order.
    ///
    /// Empty exactly when the dictionary defines no custom list, which the wire
    /// spells by leaving the whole combination block out.
    pub(crate) fn combinations(&self) -> &[Combination] {
        &self.combinations
    }

    /// Returns the context map, or `None` when the dictionary is not context based.
    pub(crate) fn context_map(&self) -> Option<&[u8; NUM_CONTEXTS]> {
        self.context_map.as_ref()
    }

    /// Returns whether the dictionary replaces the static dictionary at all.
    pub(crate) fn is_custom_static(&self) -> bool {
        !self.word_lists.is_empty() || !self.transform_lists.is_empty()
    }

    /// Assembles a dictionary from parts and checks every cross reference.
    ///
    /// This is the one constructor: [`parse`] decodes the wire into these parts
    /// and then calls it, so a parsed dictionary and a programmatically built
    /// one go through exactly the same validation.
    ///
    /// # Errors
    ///
    /// Returns the [`SerializedError`] naming the first rule broken:
    /// [`SerializedError::TooManyLists`] past sixty-four of anything,
    /// [`SerializedError::NoCombinations`] when custom lists are present with
    /// no combination to use them, [`SerializedError::UndefinedList`] and
    /// [`SerializedError::UndefinedCombination`] for a dangling reference, and
    /// [`SerializedError::LimitExceeded`] for a resource ceiling.
    pub(crate) fn assemble(
        prefix: Option<Box<[u8]>>,
        word_lists: Vec<WordList>,
        transform_lists: Vec<TransformList>,
        combinations: Vec<Combination>,
        context_map: Option<[u8; NUM_CONTEXTS]>,
        limits: &SerializedLimits,
    ) -> Result<Self, SerializedError> {
        check_count("word lists", word_lists.len(), limits.max_word_lists)?;
        check_count(
            "transform lists",
            transform_lists.len(),
            limits.max_transform_lists,
        )?;
        check_count("combinations", combinations.len(), limits.max_combinations)?;

        let prefix_len = prefix.as_ref().map_or(0, |bytes| bytes.len() as u64);
        if prefix_len > MAX_LZ77_DICTIONARY_LENGTH {
            return Err(SerializedError::PrefixTooLongForFormat { length: prefix_len });
        }
        check_limit("LZ77 prefix", prefix_len, limits.max_prefix_bytes)?;

        let word_bytes = word_lists
            .iter()
            .map(|list| list.data().len() as u64)
            .sum::<u64>();
        check_limit("word data", word_bytes, limits.max_word_bytes)?;
        let transform_bytes = transform_lists
            .iter()
            .map(|list| list.wire_len() as u64)
            .sum::<u64>();
        check_limit(
            "transform data",
            transform_bytes,
            limits.max_transform_bytes,
        )?;

        let custom = !word_lists.is_empty() || !transform_lists.is_empty();
        if custom && combinations.is_empty() {
            return Err(SerializedError::NoCombinations);
        }
        // Both counts are at most sixty-four, so neither cast can lose a bit.
        let words_available = word_lists.len() as u8;
        let transforms_available = transform_lists.len() as u8;
        for combination in &combinations {
            check_reference("word list", combination.words, words_available)?;
            check_reference(
                "transform list",
                combination.transforms,
                transforms_available,
            )?;
        }
        if let Some(map) = &context_map {
            for (context, &index) in map.iter().enumerate() {
                if usize::from(index) >= combinations.len() {
                    return Err(SerializedError::UndefinedCombination {
                        context,
                        index,
                        available: combinations.len(),
                    });
                }
            }
        }
        Ok(Self {
            prefix,
            word_lists,
            transform_lists,
            combinations: if custom { combinations } else { Vec::new() },
            context_map: if custom { context_map } else { None },
        })
    }

    /// Returns how many bytes [`SerializedDictionaryData::serialize`] produces.
    ///
    /// Exact, so a caller can size a buffer once and check a limit before the
    /// bytes exist.
    pub(crate) fn wire_len(&self) -> usize {
        let prefix = self.prefix();
        let mut len = MAGIC.len() + varint::encoded_len(prefix.len() as u64) + prefix.len();
        len += 1 + self
            .word_lists
            .iter()
            .map(WordList::wire_len)
            .sum::<usize>();
        len += 1 + self
            .transform_lists
            .iter()
            .map(TransformList::wire_len)
            .sum::<usize>();
        if self.is_custom_static() {
            len += 1 + self.combinations.len() * 2 + 1;
            if self.context_map.is_some() {
                len += NUM_CONTEXTS;
            }
        }
        len
    }

    /// Writes the dictionary in the canonical RFC 9841 encoding.
    ///
    /// Canonical means: the shortest varint for the prefix length, the
    /// combination block present exactly when a custom list is, the parameter
    /// block present exactly when a transform shifts, and the context map
    /// present exactly when the dictionary is context based. A stream this
    /// writes always parses back to an equal dictionary.
    ///
    /// # Errors
    ///
    /// Returns [`SerializedError::Varint`] only if the prefix is longer than a
    /// varint can express, which [`SerializedDictionaryData::assemble`] has
    /// already made impossible.
    pub(crate) fn serialize(&self, out: &mut Vec<u8>) -> Result<(), SerializedError> {
        out.reserve(self.wire_len());
        out.extend_from_slice(&MAGIC);
        let prefix = self.prefix();
        // Written through a `match` rather than `map_err` so the arm the
        // format's own ceiling makes unreachable is not a closure of its own:
        // `assemble` caps the prefix at `MAX_LZ77_DICTIONARY_LENGTH`, which is
        // four orders of magnitude below what a varint can carry.
        match varint::write(prefix.len() as u64, out) {
            Ok(()) => {}
            Err(source) => {
                return Err(SerializedError::Varint {
                    field: "LZ77_DICTIONARY_LENGTH",
                    source,
                });
            }
        }
        out.extend_from_slice(prefix);

        out.push(u8::try_from(self.word_lists.len()).unwrap_or(u8::MAX));
        for list in &self.word_lists {
            list.serialize(out);
        }
        out.push(u8::try_from(self.transform_lists.len()).unwrap_or(u8::MAX));
        for list in &self.transform_lists {
            list.serialize(out);
        }

        if self.is_custom_static() {
            out.push(u8::try_from(self.combinations.len()).unwrap_or(u8::MAX));
            let words = u8::try_from(self.word_lists.len()).unwrap_or(u8::MAX);
            let transforms = u8::try_from(self.transform_lists.len()).unwrap_or(u8::MAX);
            for combination in &self.combinations {
                out.push(combination.words.encode(words));
                out.push(combination.transforms.encode(transforms));
            }
            match &self.context_map {
                Some(map) => {
                    out.push(1);
                    out.extend_from_slice(map);
                }
                None => out.push(0),
            }
        }
        Ok(())
    }
}

/// Returns an error when `count` exceeds either the format's cap or `limit`.
fn check_count(what: &'static str, count: usize, limit: usize) -> Result<(), SerializedError> {
    if count > MAX_LISTS {
        return Err(SerializedError::TooManyLists { what, count });
    }
    check_limit(what, count as u64, limit as u64)
}

/// Returns an error when `found` exceeds `limit`.
fn check_limit(what: &'static str, found: u64, limit: u64) -> Result<(), SerializedError> {
    if found > limit {
        return Err(SerializedError::LimitExceeded { what, found, limit });
    }
    Ok(())
}

/// Returns an error when a combination names a custom list that is absent.
fn check_reference(
    what: &'static str,
    reference: ListRef,
    available: u8,
) -> Result<(), SerializedError> {
    match reference {
        ListRef::Custom(index) if index >= available => Err(SerializedError::UndefinedList {
            what,
            index,
            available,
        }),
        _ => Ok(()),
    }
}

/// A cursor over the wire bytes that never reads past the end.
struct Reader<'a> {
    /// The whole stream.
    bytes: &'a [u8],
    /// How many bytes have been consumed.
    position: usize,
}

impl<'a> Reader<'a> {
    /// Starts a cursor at the front of `bytes`.
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    /// Reads one byte.
    fn u8(&mut self, field: &'static str) -> Result<u8, SerializedError> {
        let Some(&byte) = self.bytes.get(self.position) else {
            return Err(self.truncated(field));
        };
        self.position += 1;
        Ok(byte)
    }

    /// Reads one byte that must be zero or one.
    fn boolean(&mut self, field: &'static str) -> Result<bool, SerializedError> {
        match self.u8(field)? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(SerializedError::NotABoolean { field, value }),
        }
    }

    /// Reads a little-endian sixteen-bit field.
    fn u16_le(&mut self, field: &'static str) -> Result<u16, SerializedError> {
        let Some(chunk) = self
            .bytes
            .get(self.position..)
            .and_then(<[u8]>::first_chunk::<2>)
        else {
            return Err(self.truncated(field));
        };
        self.position += 2;
        Ok(u16::from_le_bytes(*chunk))
    }

    /// Reads a varint.
    fn varint(&mut self, field: &'static str) -> Result<u64, SerializedError> {
        let rest = self.bytes.get(self.position..).unwrap_or_default();
        let (value, len) =
            varint::read(rest).map_err(|source| SerializedError::Varint { field, source })?;
        self.position += len;
        Ok(value)
    }

    /// Takes `count` bytes.
    fn take(&mut self, field: &'static str, count: usize) -> Result<&'a [u8], SerializedError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or(self.truncated(field))?;
        let Some(slice) = self.bytes.get(self.position..end) else {
            return Err(self.truncated(field));
        };
        self.position = end;
        Ok(slice)
    }

    /// Takes a fixed-size array of bytes.
    fn array<const N: usize>(&mut self, field: &'static str) -> Result<[u8; N], SerializedError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(field, N)?);
        Ok(out)
    }

    /// Builds the truncation error for the field being read.
    fn truncated(&self, field: &'static str) -> SerializedError {
        SerializedError::Truncated {
            field,
            position: self.bytes.len(),
        }
    }
}

/// Parses one serialized dictionary from the front of `bytes`.
///
/// Returns the dictionary and how many bytes it occupied, so a container that
/// embeds one can carry on afterwards. Trailing bytes are not an error here;
/// the public entry point is what refuses them.
///
/// Every count and length is checked against `limits` before the bytes it
/// describes are copied, so a hostile stream cannot make this allocate more
/// than the caller allowed.
///
/// # Errors
///
/// Returns the [`SerializedError`] naming the first rule broken.
pub(crate) fn parse(
    bytes: &[u8],
    limits: &SerializedLimits,
) -> Result<(SerializedDictionaryData, usize), SerializedError> {
    check_limit("total size", bytes.len() as u64, limits.max_total_bytes)?;
    let mut reader = Reader::new(bytes);
    let magic = reader
        .array::<2>("magic")
        .map_err(|_| SerializedError::BadMagic {
            found: bytes.get(..bytes.len().min(2)).unwrap_or_default().to_vec(),
        })?;
    if magic != MAGIC {
        return Err(SerializedError::BadMagic {
            found: magic.to_vec(),
        });
    }

    let prefix_len = reader.varint("LZ77_DICTIONARY_LENGTH")?;
    if prefix_len > MAX_LZ77_DICTIONARY_LENGTH {
        return Err(SerializedError::PrefixTooLongForFormat { length: prefix_len });
    }
    check_limit("LZ77 prefix", prefix_len, limits.max_prefix_bytes)?;
    let prefix = if prefix_len == 0 {
        None
    } else {
        // `prefix_len` is at most a gigabyte, so the conversion is lossless on
        // every target this crate supports; `take` is what proves the bytes
        // exist.
        let Ok(count) = usize::try_from(prefix_len) else {
            return Err(SerializedError::LimitExceeded {
                what: "LZ77 prefix",
                found: prefix_len,
                limit: usize::MAX as u64,
            });
        };
        Some(Box::from(reader.take("the LZ77 prefix", count)?))
    };

    let num_word_lists = usize::from(reader.u8("NUM_CUSTOM_WORD_LISTS")?);
    check_count("word lists", num_word_lists, limits.max_word_lists)?;
    let mut word_bytes = 0u64;
    let mut word_lists = Vec::with_capacity(num_word_lists);
    for index in 0..num_word_lists {
        let size_bits = reader.array::<NUM_ENCODED_LENGTHS>("SIZE_BITS_BY_LENGTH")?;
        let expected = word_data_len(&size_bits)
            .map_err(|source| SerializedError::WordList { index, source })?;
        word_bytes += expected as u64;
        check_limit("word data", word_bytes, limits.max_word_bytes)?;
        let data = reader.take("a word list's words", expected)?;
        // Length was checked above; keep the typed error should word-list
        // validation gain another invariant, without a never-called closure.
        let list = match WordList::from_parts(&size_bits, Cow::Owned(data.to_vec())) {
            Ok(list) => list,
            Err(source) => return Err(SerializedError::WordList { index, source }),
        };
        word_lists.push(list);
    }

    let num_transform_lists = usize::from(reader.u8("NUM_CUSTOM_TRANSFORM_LISTS")?);
    check_count(
        "transform lists",
        num_transform_lists,
        limits.max_transform_lists,
    )?;
    let mut transform_bytes = 0u64;
    let mut transform_lists = Vec::with_capacity(num_transform_lists);
    for index in 0..num_transform_lists {
        let list = parse_transform_list(&mut reader, index)?;
        transform_bytes += list.wire_len() as u64;
        check_limit(
            "transform data",
            transform_bytes,
            limits.max_transform_bytes,
        )?;
        transform_lists.push(list);
    }

    let custom = num_word_lists > 0 || num_transform_lists > 0;
    let mut combinations = Vec::new();
    let mut context_map = None;
    if custom {
        let count = usize::from(reader.u8("NUM_DICTIONARIES")?);
        if count == 0 {
            return Err(SerializedError::NoCombinations);
        }
        check_count("combinations", count, limits.max_combinations)?;
        combinations.reserve(count);
        // Both counts came from a byte and were capped at sixty-four above.
        let words_available = num_word_lists as u8;
        let transforms_available = num_transform_lists as u8;
        for _ in 0..count {
            let words = reader.u8("a combination's word list index")?;
            let words =
                ListRef::decode(words, words_available).ok_or(SerializedError::UndefinedList {
                    what: "word list",
                    index: words,
                    available: words_available,
                })?;
            let transforms = reader.u8("a combination's transform list index")?;
            let transforms = ListRef::decode(transforms, transforms_available).ok_or(
                SerializedError::UndefinedList {
                    what: "transform list",
                    index: transforms,
                    available: transforms_available,
                },
            )?;
            combinations.push(Combination { words, transforms });
        }
        if reader.boolean("CONTEXT_ENABLED")? {
            context_map = Some(reader.array::<NUM_CONTEXTS>("CONTEXT_MAP")?);
        }
    }

    let consumed = reader.position;
    let dictionary = SerializedDictionaryData::assemble(
        prefix,
        word_lists,
        transform_lists,
        combinations,
        context_map,
        limits,
    )?;
    Ok((dictionary, consumed))
}

/// Parses one serialized dictionary and refuses anything after it.
///
/// # Errors
///
/// As [`parse`], plus [`SerializedError::TrailingBytes`] when the structure
/// ends before the input does. The reference ignores such a tail; refusing it
/// is what makes a dictionary's bytes and its meaning one to one.
pub(crate) fn parse_exact(
    bytes: &[u8],
    limits: &SerializedLimits,
) -> Result<SerializedDictionaryData, SerializedError> {
    let (dictionary, consumed) = parse(bytes, limits)?;
    match bytes.len().checked_sub(consumed) {
        Some(0) | None => Ok(dictionary),
        Some(extra) => Err(SerializedError::TrailingBytes { extra }),
    }
}

/// Returns how many word bytes the size bits describe.
///
/// Cannot overflow: a length is at most thirty-one and an exponent at most
/// fifteen, so the total is under a megabyte.
fn word_data_len(size_bits: &[u8; NUM_ENCODED_LENGTHS]) -> Result<usize, WordListError> {
    let mut total = 0usize;
    for (index, &bits) in size_bits.iter().enumerate() {
        let length = MIN_WORD_LENGTH + index;
        if bits > MAX_SIZE_BITS {
            return Err(WordListError::TooManySizeBits { length, bits });
        }
        if bits != 0 {
            total += length << bits;
        }
    }
    Ok(total)
}

/// Parses one transform list, prefix and suffix table included.
fn parse_transform_list(
    reader: &mut Reader<'_>,
    index: usize,
) -> Result<TransformList, SerializedError> {
    let block_len = usize::from(reader.u16_le("PREFIX_SUFFIX_LENGTH")?);
    let block = reader.take("a transform list's prefix and suffix data", block_len)?;
    let count = usize::from(reader.u8("NTRANSFORMS")?);
    let triples = reader
        .take("a transform list's transforms", count * 3)?
        .to_vec();
    // The parameter block is on the wire if and only if some transform shifts,
    // which is decided by the triples that were just read.
    let shifts = triples
        .as_chunks::<3>()
        .0
        .iter()
        .any(|triple| triple[1] == SHIFT_FIRST || triple[1] == SHIFT_ALL);
    let params = if shifts {
        reader
            .take("a transform list's parameters", count * 2)?
            .to_vec()
    } else {
        Vec::new()
    };
    TransformList::from_parts(
        Cow::Owned(block.to_vec()),
        Cow::Owned(triples),
        Cow::Owned(params),
    )
    .map_err(|source| SerializedError::TransformList { index, source })
}

/// Transform id of `ShiftFirst`, which is what puts a parameter block on the wire.
const SHIFT_FIRST: u8 = 21;

/// Transform id of `ShiftAll`.
const SHIFT_ALL: u8 = 22;

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds the shortest valid dictionary: magic and a zero prefix length.
    fn minimal() -> Vec<u8> {
        vec![MAGIC[0], MAGIC[1], 0, 0, 0]
    }

    /// Parses under limits that refuse nothing the format allows.
    fn parse_permissive(bytes: &[u8]) -> Result<SerializedDictionaryData, SerializedError> {
        parse_exact(bytes, &SerializedLimits::permissive())
    }

    #[test]
    fn truncated_transform_list_from_timeout_corpus_is_rejected() {
        // AFL saved this input as a timeout; isolated replays terminate normally.
        let bytes = [
            0x91, 0x00, 0x0b, 0x61, 0x62, 0x96, 0x00, 0x03, 0x61, 0x62, 0x63, 0x10, 0x00, 0x63,
            0x00, 0x1e,
        ];
        assert!(matches!(
            parse_permissive(&bytes),
            Err(SerializedError::Truncated { position: 16, .. })
        ));
    }

    #[test]
    fn overflowing_read_length_preserves_cursor() {
        let mut reader = Reader::new(&[1, 2]);
        assert_eq!(reader.u8("first").expect("first byte"), 1);
        assert!(matches!(
            reader.take("overflow", usize::MAX),
            Err(SerializedError::Truncated { .. })
        ));
        assert_eq!(reader.u8("second").expect("cursor unchanged"), 2);
    }

    /// Round-trips a dictionary through its own encoding.
    fn round_trip(dictionary: &SerializedDictionaryData) -> SerializedDictionaryData {
        let mut bytes = Vec::new();
        dictionary.serialize(&mut bytes).expect("in range");
        assert_eq!(bytes.len(), dictionary.wire_len());
        let parsed = parse_permissive(&bytes).expect("what was written parses");
        let mut again = Vec::new();
        parsed.serialize(&mut again).expect("in range");
        assert_eq!(again, bytes, "serializing is stable across a round trip");
        parsed
    }

    /// Builds a one-word-list dictionary with the given word bytes.
    fn with_word_list(words: &[u8], length: usize) -> SerializedDictionaryData {
        let mut size_bits = [0u8; NUM_ENCODED_LENGTHS];
        size_bits[length - MIN_WORD_LENGTH] = 1;
        let list =
            WordList::from_parts(&size_bits, Cow::Owned(words.to_vec())).expect("the parts agree");
        SerializedDictionaryData::assemble(
            None,
            vec![list],
            Vec::new(),
            vec![Combination {
                words: ListRef::Custom(0),
                transforms: ListRef::Builtin,
            }],
            None,
            &SerializedLimits::permissive(),
        )
        .expect("the parts are consistent")
    }

    #[test]
    fn the_shortest_dictionary_parses() {
        let parsed = parse_permissive(&minimal()).expect("valid");

        assert!(!parsed.has_prefix());
        assert!(parsed.word_lists().is_empty());
        assert!(parsed.transform_lists().is_empty());
        assert!(parsed.combinations().is_empty());
        assert!(!parsed.is_custom_static());
    }

    #[test]
    fn a_prefix_only_dictionary_round_trips() {
        let dictionary = SerializedDictionaryData::assemble(
            Some(Box::from(&b"a prefix"[..])),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            &SerializedLimits::permissive(),
        )
        .expect("valid");

        let parsed = round_trip(&dictionary);

        assert_eq!(parsed.prefix(), b"a prefix");
        assert!(parsed.has_prefix());
    }

    #[test]
    fn a_word_list_dictionary_round_trips() {
        let dictionary = with_word_list(b"abcdefgh", 4);
        let parsed = round_trip(&dictionary);

        assert_eq!(parsed.word_lists().len(), 1);
        assert_eq!(parsed.word_lists()[0].word(4, 1), b"efgh");
        assert_eq!(parsed.combinations().len(), 1);
        assert_eq!(parsed.combinations()[0].words, ListRef::Custom(0));
        assert_eq!(parsed.combinations()[0].transforms, ListRef::Builtin);
    }

    #[test]
    fn a_transform_list_dictionary_round_trips() {
        let transforms = TransformList::from_parts(
            Cow::Owned(vec![1, b'!', 0]),
            Cow::Owned(vec![0, 0, 1]),
            Cow::Owned(Vec::new()),
        )
        .expect("well formed");
        let dictionary = SerializedDictionaryData::assemble(
            None,
            Vec::new(),
            vec![transforms],
            vec![Combination {
                words: ListRef::Builtin,
                transforms: ListRef::Custom(0),
            }],
            None,
            &SerializedLimits::permissive(),
        )
        .expect("valid");

        let parsed = round_trip(&dictionary);

        assert_eq!(parsed.transform_lists().len(), 1);
        assert_eq!(parsed.transform_lists()[0].stringlet(0), b"!");
    }

    #[test]
    fn a_context_map_round_trips() {
        let mut map = [0u8; NUM_CONTEXTS];
        map[7] = 1;
        let mut size_bits = [0u8; NUM_ENCODED_LENGTHS];
        size_bits[0] = 1;
        let list = WordList::from_parts(&size_bits, Cow::Owned(b"abcdefgh".to_vec()))
            .expect("the parts agree");
        let dictionary = SerializedDictionaryData::assemble(
            None,
            vec![list],
            Vec::new(),
            vec![
                Combination {
                    words: ListRef::Custom(0),
                    transforms: ListRef::Builtin,
                },
                Combination {
                    words: ListRef::Builtin,
                    transforms: ListRef::Builtin,
                },
            ],
            Some(map),
            &SerializedLimits::permissive(),
        )
        .expect("valid");

        let parsed = round_trip(&dictionary);

        assert_eq!(parsed.context_map().map(|map| map[7]), Some(1));
        assert_eq!(parsed.context_map().map(|map| map[0]), Some(0));
    }

    #[test]
    fn the_wrong_magic_is_refused() {
        assert!(matches!(
            parse_permissive(&[0x91, 0x01, 0, 0, 0]),
            Err(SerializedError::BadMagic { .. })
        ));
        assert!(matches!(
            parse_permissive(&[0x00]),
            Err(SerializedError::BadMagic { .. })
        ));
        assert!(matches!(
            parse_permissive(&[]),
            Err(SerializedError::BadMagic { .. })
        ));
    }

    #[test]
    fn every_truncation_of_a_valid_dictionary_is_refused() {
        let dictionary = with_word_list(b"abcdefgh", 4);
        let mut bytes = Vec::new();
        dictionary.serialize(&mut bytes).expect("in range");

        for cut in 0..bytes.len() {
            assert!(
                parse_permissive(&bytes[..cut]).is_err(),
                "a dictionary cut to {cut} bytes was accepted"
            );
        }
        assert!(parse_permissive(&bytes).is_ok());
    }

    #[test]
    fn trailing_bytes_are_refused_but_reported_by_the_prefix_parser() {
        let mut bytes = minimal();
        bytes.extend_from_slice(b"tail");

        assert_eq!(
            parse_permissive(&bytes).err(),
            Some(SerializedError::TrailingBytes { extra: 4 })
        );
        let (_, consumed) =
            parse(&bytes, &SerializedLimits::permissive()).expect("the head parses");
        assert_eq!(consumed, bytes.len() - 4);
    }

    #[test]
    fn a_prefix_past_the_format_ceiling_is_refused() {
        let mut bytes = vec![MAGIC[0], MAGIC[1]];
        varint::write(MAX_LZ77_DICTIONARY_LENGTH + 1, &mut bytes).expect("in range");

        assert_eq!(
            parse_permissive(&bytes).err(),
            Some(SerializedError::PrefixTooLongForFormat {
                length: MAX_LZ77_DICTIONARY_LENGTH + 1,
            })
        );
    }

    #[test]
    fn a_prefix_past_the_caller_limit_is_refused_before_it_is_copied() {
        let mut bytes = vec![MAGIC[0], MAGIC[1]];
        varint::write(1 << 20, &mut bytes).expect("in range");
        let limits = SerializedLimits {
            max_prefix_bytes: 16,
            ..SerializedLimits::permissive()
        };

        // The declared prefix is never present in `bytes`, so accepting the
        // length would mean reading a megabyte that does not exist.
        assert_eq!(
            parse_exact(&bytes, &limits).err(),
            Some(SerializedError::LimitExceeded {
                what: "LZ77 prefix",
                found: 1 << 20,
                limit: 16,
            })
        );
    }

    #[test]
    fn a_stream_larger_than_the_limit_is_refused_before_it_is_read() {
        let limits = SerializedLimits {
            max_total_bytes: 2,
            ..SerializedLimits::permissive()
        };

        assert_eq!(
            parse_exact(&minimal(), &limits).err(),
            Some(SerializedError::LimitExceeded {
                what: "total size",
                found: 5,
                limit: 2,
            })
        );
    }

    #[test]
    fn more_than_sixty_four_word_lists_are_refused() {
        let bytes = vec![MAGIC[0], MAGIC[1], 0, 65];

        assert_eq!(
            parse_permissive(&bytes).err(),
            Some(SerializedError::TooManyLists {
                what: "word lists",
                count: 65,
            })
        );
    }

    #[test]
    fn more_than_sixty_four_transform_lists_are_refused() {
        let bytes = vec![MAGIC[0], MAGIC[1], 0, 0, 65];

        assert_eq!(
            parse_permissive(&bytes).err(),
            Some(SerializedError::TooManyLists {
                what: "transform lists",
                count: 65,
            })
        );
    }

    #[test]
    fn a_word_list_count_past_the_caller_limit_is_refused() {
        let mut size_bits = [0u8; NUM_ENCODED_LENGTHS];
        size_bits[0] = 1;
        let mut bytes = vec![MAGIC[0], MAGIC[1], 0, 1];
        bytes.extend_from_slice(&size_bits);
        bytes.extend_from_slice(b"abcdefgh");
        bytes.extend_from_slice(&[0, 1, 0, 64, 0]);
        let limits = SerializedLimits {
            max_word_lists: 0,
            ..SerializedLimits::permissive()
        };

        assert_eq!(
            parse_exact(&bytes, &limits).err(),
            Some(SerializedError::LimitExceeded {
                what: "word lists",
                found: 1,
                limit: 0,
            })
        );
    }

    #[test]
    fn a_dictionary_with_custom_lists_and_no_combinations_is_refused() {
        let mut size_bits = [0u8; NUM_ENCODED_LENGTHS];
        size_bits[0] = 1;
        let mut bytes = vec![MAGIC[0], MAGIC[1], 0, 1];
        bytes.extend_from_slice(&size_bits);
        bytes.extend_from_slice(b"abcdefgh");
        bytes.extend_from_slice(&[0, 0]);

        assert_eq!(
            parse_permissive(&bytes).err(),
            Some(SerializedError::NoCombinations)
        );
    }

    #[test]
    fn a_combination_naming_a_missing_word_list_is_refused() {
        let mut size_bits = [0u8; NUM_ENCODED_LENGTHS];
        size_bits[0] = 1;
        let mut bytes = vec![MAGIC[0], MAGIC[1], 0, 1];
        bytes.extend_from_slice(&size_bits);
        bytes.extend_from_slice(b"abcdefgh");
        // One transform list count, one combination naming word list 2, which
        // is past both the one custom list and the built-in it would be at 1.
        bytes.extend_from_slice(&[0, 1, 2, 0, 0]);

        assert_eq!(
            parse_permissive(&bytes).err(),
            Some(SerializedError::UndefinedList {
                what: "word list",
                index: 2,
                available: 1,
            })
        );
    }

    #[test]
    fn a_context_entry_naming_a_missing_combination_is_refused() {
        let mut size_bits = [0u8; NUM_ENCODED_LENGTHS];
        size_bits[0] = 1;
        let mut bytes = vec![MAGIC[0], MAGIC[1], 0, 1];
        bytes.extend_from_slice(&size_bits);
        bytes.extend_from_slice(b"abcdefgh");
        bytes.extend_from_slice(&[0, 1, 0, 0, 1]);
        bytes.extend_from_slice(&[3u8; NUM_CONTEXTS]);

        assert_eq!(
            parse_permissive(&bytes).err(),
            Some(SerializedError::UndefinedCombination {
                context: 0,
                index: 3,
                available: 1,
            })
        );
    }

    #[test]
    fn a_context_flag_that_is_not_a_boolean_is_refused() {
        let mut size_bits = [0u8; NUM_ENCODED_LENGTHS];
        size_bits[0] = 1;
        let mut bytes = vec![MAGIC[0], MAGIC[1], 0, 1];
        bytes.extend_from_slice(&size_bits);
        bytes.extend_from_slice(b"abcdefgh");
        bytes.extend_from_slice(&[0, 1, 0, 0, 2]);

        assert_eq!(
            parse_permissive(&bytes).err(),
            Some(SerializedError::NotABoolean {
                field: "CONTEXT_ENABLED",
                value: 2,
            })
        );
    }

    #[test]
    fn a_zero_combination_count_is_refused() {
        let mut size_bits = [0u8; NUM_ENCODED_LENGTHS];
        size_bits[0] = 1;
        let mut bytes = vec![MAGIC[0], MAGIC[1], 0, 1];
        bytes.extend_from_slice(&size_bits);
        bytes.extend_from_slice(b"abcdefgh");
        bytes.extend_from_slice(&[0, 0, 0]);

        assert_eq!(
            parse_permissive(&bytes).err(),
            Some(SerializedError::NoCombinations)
        );
    }

    #[test]
    fn a_malformed_varint_names_its_field() {
        let bytes = vec![MAGIC[0], MAGIC[1], 0xFF];

        assert!(matches!(
            parse_permissive(&bytes),
            Err(SerializedError::Varint {
                field: "LZ77_DICTIONARY_LENGTH",
                ..
            })
        ));
    }

    #[test]
    fn a_noncanonical_prefix_length_is_accepted_as_the_rfc_allows() {
        // The RFC caps a varint's length but does not require the shortest
        // encoding, so a padded zero is a valid zero.
        let bytes = vec![MAGIC[0], MAGIC[1], 0x80, 0x00, 0, 0];
        let parsed = parse_permissive(&bytes).expect("valid");

        assert!(!parsed.has_prefix());
        // Writing it back produces the canonical form, which is shorter.
        let mut written = Vec::new();
        parsed.serialize(&mut written).expect("in range");
        assert_eq!(written, minimal());
    }

    #[test]
    fn a_combination_may_name_the_builtin_lists() {
        let mut bytes = vec![MAGIC[0], MAGIC[1], 0, 0, 1];
        // One transform list: a bare terminator and one identity transform.
        bytes.extend_from_slice(&[1, 0, 0, 1, 0, 0, 0]);
        // One combination, naming the built-in words and the custom transforms.
        bytes.extend_from_slice(&[1, 0, 0, 0]);
        let parsed = parse_permissive(&bytes).expect("valid");

        assert_eq!(parsed.combinations()[0].words, ListRef::Builtin);
        assert_eq!(parsed.combinations()[0].transforms, ListRef::Custom(0));
    }

    #[test]
    fn assembling_without_custom_lists_drops_the_combination_block() {
        let dictionary = SerializedDictionaryData::assemble(
            Some(Box::from(&b"prefix"[..])),
            Vec::new(),
            Vec::new(),
            vec![Combination {
                words: ListRef::Builtin,
                transforms: ListRef::Builtin,
            }],
            Some([0; NUM_CONTEXTS]),
            &SerializedLimits::permissive(),
        )
        .expect("valid");

        // The wire has nowhere to put either, so neither is kept.
        assert!(dictionary.combinations().is_empty());
        assert!(dictionary.context_map().is_none());
    }

    #[test]
    fn a_prefix_longer_than_the_format_allows_is_refused_when_assembled() {
        // Constructing the bytes would need a gigabyte, so the check is made
        // against a shorter buffer through the limit that stands in for it.
        let limits = SerializedLimits {
            max_prefix_bytes: 4,
            ..SerializedLimits::permissive()
        };
        let outcome = SerializedDictionaryData::assemble(
            Some(Box::from(&b"too long"[..])),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            &limits,
        );

        assert_eq!(
            outcome.err(),
            Some(SerializedError::LimitExceeded {
                what: "LZ77 prefix",
                found: 8,
                limit: 4,
            })
        );
    }

    #[test]
    fn the_list_reference_encoding_names_the_builtin_by_the_custom_count() {
        assert_eq!(ListRef::Builtin.encode(3), 3);
        assert_eq!(ListRef::Custom(1).encode(3), 1);
        assert_eq!(ListRef::decode(3, 3), Some(ListRef::Builtin));
        assert_eq!(ListRef::decode(1, 3), Some(ListRef::Custom(1)));
        assert_eq!(ListRef::decode(4, 3), None);
        assert_eq!(ListRef::decode(0, 0), Some(ListRef::Builtin));
    }
}

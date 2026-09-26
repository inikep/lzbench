//! Serialized shared dictionaries: the RFC 9841 dictionary stream.
//!
//! **Experimental.** Everything in this module is behind the `experimental`
//! feature. [RFC 9841] specifies the format, but the reference encoder compiles
//! its parser out unless `BROTLI_EXPERIMENTAL` is defined and has never shipped
//! it as a stable API, so this surface is validated against the RFC and against
//! fixtures rather than against byte identity with a pinned C encoder. It may
//! change in a patch release.
//!
//! A serialized dictionary is a self-describing byte stream that can carry
//! three separable things:
//!
//! - an **LZ77 prefix**, bytes placed in front of the stream that a backward
//!   distance may reach, exactly what
//!   [`DictionaryBuilder::add_prefix`](super::DictionaryBuilder::add_prefix)
//!   attaches by hand;
//! - **custom static dictionaries**, word lists and transform lists that
//!   replace the ones RFC 7932 fixes;
//! - a **context map**, which picks the first word and transform list to try
//!   from the literal context of the position being coded.
//!
//! The type that results is a description, not a workspace: it is immutable,
//! carries no encoder state, and is turned into the same
//! [`PreparedDictionary`](super::PreparedDictionary) that a hand-attached
//! prefix produces.
//!
//! # Examples
//!
//! Round-tripping a dictionary through its wire form:
//!
//! ```
//! use mbrotli::dictionary::SerializedDictionary;
//!
//! let dictionary = SerializedDictionary::builder()
//!     .with_prefix(&b"HTTP/1.1 200 OK\r\n"[..])
//!     .build()?;
//!
//! let bytes = dictionary.to_bytes();
//! assert_eq!(&bytes[..2], &[0x91, 0x00]);
//!
//! let parsed = SerializedDictionary::try_from(&bytes[..])?;
//! assert_eq!(parsed.prefix(), b"HTTP/1.1 200 OK\r\n");
//! # Ok::<(), mbrotli::dictionary::SerializedDictionaryError>(())
//! ```
//!
//! [RFC 9841]: https://www.rfc-editor.org/rfc/rfc9841.html

use alloc::boxed::Box;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

use alloc::borrow::Cow;

use thiserror::Error;

use crate::compressor::core::rfc9841::serialized::{
    self, Combination, ListRef, MAX_LISTS, NUM_CONTEXTS, SerializedDictionaryData, SerializedError,
    SerializedLimits,
};
use crate::compressor::core::rfc9841::transform::{
    MAX_STRINGLET_BYTES, MAX_WORD_LENGTH, TransformList as CoreTransformList,
    TransformListError as CoreTransformListError, TransformScratch,
};
use crate::compressor::core::rfc9841::words::{
    MAX_SIZE_BITS, MIN_WORD_LENGTH, NUM_ENCODED_LENGTHS, WordList as CoreWordList,
    WordListError as CoreWordListError,
};

use super::DictionaryLimits;

/// Lowers the public limits into the ones the codec checks against.
impl From<DictionaryLimits> for SerializedLimits {
    /// Flattens the caller's ceilings into the codec's flat form.
    ///
    /// The LZ77 prefix inside a serialized dictionary is bounded by the same
    /// `max_prefix_bytes` a hand-attached prefix is, because it becomes one.
    fn from(value: DictionaryLimits) -> Self {
        Self {
            max_total_bytes: value.max_serialized_bytes(),
            max_prefix_bytes: value.max_prefix_bytes(),
            max_word_lists: value.max_word_lists(),
            max_word_bytes: value.max_word_bytes(),
            max_transform_lists: value.max_transform_lists(),
            max_transform_bytes: value.max_transform_bytes(),
            max_combinations: value.max_combinations(),
        }
    }
}

/// Literal contexts a context map covers.
///
/// Fixed by RFC 7932's literal context model, which
/// [RFC 9841 section 3.1] reuses unchanged.
///
/// [RFC 9841 section 3.1]: https://www.rfc-editor.org/rfc/rfc9841.html#section-3.1
pub const CONTEXTS: usize = NUM_CONTEXTS;

/// Word lists, transform lists or combinations one dictionary may hold.
pub const MAX_LIST_COUNT: usize = MAX_LISTS;

/// Transforms one transform list may hold, from its one-byte count.
pub const MAX_TRANSFORMS: usize = 255;

/// Distinct prefixes and suffixes one transform list may hold.
///
/// The zero-length terminator is one of them, so a list that uses the empty
/// string as a prefix or a suffix spends nothing extra on it.
pub const MAX_STRINGLETS: usize = 256;

/// How many trailing or leading bytes an omit transform drops.
///
/// Between one and nine; dropping nothing is [`TransformOperation::Identity`],
/// and the format has no way to spell a larger cut.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::OmitLength;
///
/// assert_eq!(u8::from(OmitLength::try_from(3)?), 3);
/// assert!(OmitLength::try_from(0).is_err());
/// assert!(OmitLength::try_from(10).is_err());
/// # Ok::<(), mbrotli::dictionary::OmitLengthOutOfRange>(())
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct OmitLength(u8);

impl OmitLength {
    /// The smallest cut, one byte.
    pub const MIN: Self = Self(1);
    /// The largest cut the format can express, nine bytes.
    pub const MAX: Self = Self(9);
}

impl TryFrom<u8> for OmitLength {
    type Error = OmitLengthOutOfRange;

    /// Accepts one to nine.
    ///
    /// # Errors
    ///
    /// Returns [`OmitLengthOutOfRange`] for zero or ten and above.
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if (Self::MIN.0..=Self::MAX.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(OmitLengthOutOfRange { value })
        }
    }
}

impl From<OmitLength> for u8 {
    /// Returns how many bytes the cut drops.
    fn from(value: OmitLength) -> Self {
        value.0
    }
}

impl ::core::fmt::Display for OmitLength {
    /// Prints the number of bytes dropped.
    fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Error returned when an omit length is outside one to nine.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
#[error("an omit transform drops between 1 and 9 bytes, not {value}")]
pub struct OmitLengthOutOfRange {
    /// The value that was offered.
    pub value: u8,
}

/// What a transform does to the word between its prefix and its suffix.
///
/// The twenty-three operations of
/// [RFC 9841 section 3.1.1], which are RFC 7932's twenty-one plus the two
/// scalar shifts.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::{OmitLength, TransformOperation};
///
/// let cut = TransformOperation::OmitLast(OmitLength::try_from(2)?);
/// assert_eq!(u8::from(cut), 2);
/// assert_eq!(u8::from(TransformOperation::FermentFirst), 10);
/// assert_eq!(TransformOperation::try_from(22), Ok(TransformOperation::ShiftAll(0)));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// [RFC 9841 section 3.1.1]:
///     https://www.rfc-editor.org/rfc/rfc9841.html#section-3.1.1
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TransformOperation {
    /// Copies the word unchanged.
    Identity,
    /// Drops bytes from the end of the word.
    OmitLast(OmitLength),
    /// Uppercases the first character of the word.
    ///
    /// RFC 7932's `UppercaseFirst`, under the name RFC 9841 gives it. The
    /// casing model is the format's own — an exclusive-or against a bit for
    /// ASCII and a fixed byte for longer UTF-8 sequences — and is deliberately
    /// not locale aware.
    FermentFirst,
    /// Uppercases every character of the word, by the same model.
    FermentAll,
    /// Drops bytes from the start of the word.
    OmitFirst(OmitLength),
    /// Adds a signed offset to the first encoded scalar of the word.
    ///
    /// The parameter is the raw sixteen-bit value the wire carries; RFC 9841
    /// sign-extends it into the addend that is actually applied.
    ShiftFirst(u16),
    /// Adds the same offset to every encoded scalar of the word.
    ShiftAll(u16),
}

impl TransformOperation {
    /// Returns the parameter the wire carries for this operation.
    const fn parameter(self) -> u16 {
        match self {
            Self::ShiftFirst(parameter) | Self::ShiftAll(parameter) => parameter,
            _ => 0,
        }
    }

    /// Returns whether this operation puts a parameter block on the wire.
    const fn shifts(self) -> bool {
        matches!(self, Self::ShiftFirst(_) | Self::ShiftAll(_))
    }
}

impl From<TransformOperation> for u8 {
    /// Returns the numeric operation id RFC 9841 assigns.
    fn from(value: TransformOperation) -> Self {
        match value {
            TransformOperation::Identity => 0,
            TransformOperation::OmitLast(length) => length.0,
            TransformOperation::FermentFirst => 10,
            TransformOperation::FermentAll => 11,
            TransformOperation::OmitFirst(length) => 11 + length.0,
            TransformOperation::ShiftFirst(_) => 21,
            TransformOperation::ShiftAll(_) => 22,
        }
    }
}

impl TryFrom<u8> for TransformOperation {
    type Error = UndefinedTransformOperation;

    /// Reads a numeric operation id, with a zero shift parameter.
    ///
    /// The parameter lives in a separate wire field, so an id alone decodes to
    /// a shift of zero; [`TransformList`] pairs the two back up.
    ///
    /// # Errors
    ///
    /// Returns [`UndefinedTransformOperation`] for twenty-three and above.
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Identity),
            1..=9 => Ok(Self::OmitLast(OmitLength(value))),
            10 => Ok(Self::FermentFirst),
            11 => Ok(Self::FermentAll),
            12..=20 => Ok(Self::OmitFirst(OmitLength(value - 11))),
            21 => Ok(Self::ShiftFirst(0)),
            22 => Ok(Self::ShiftAll(0)),
            _ => Err(UndefinedTransformOperation { value }),
        }
    }
}

/// Error returned when an operation id is not one RFC 9841 defines.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
#[error("RFC 9841 defines transform operations 0 to 22, not {value}")]
pub struct UndefinedTransformOperation {
    /// The id that was offered.
    pub value: u8,
}

/// Which word list or transform list a combination draws from.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::ListSelector;
///
/// assert_eq!(ListSelector::from(2), ListSelector::Custom(2));
/// assert_eq!(ListSelector::default(), ListSelector::Builtin);
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum ListSelector {
    /// The RFC 7932 list, which every ordinary Brotli stream uses.
    #[default]
    Builtin,
    /// One of the dictionary's own lists, by position.
    Custom(u8),
}

impl From<u8> for ListSelector {
    /// Selects the dictionary's own list at `index`.
    fn from(index: u8) -> Self {
        Self::Custom(index)
    }
}

impl From<ListSelector> for ListRef {
    /// Lowers the public selector into the one the codec uses.
    fn from(value: ListSelector) -> Self {
        match value {
            ListSelector::Builtin => Self::Builtin,
            ListSelector::Custom(index) => Self::Custom(index),
        }
    }
}

impl From<ListRef> for ListSelector {
    /// Lifts the codec's selector into the public one.
    fn from(value: ListRef) -> Self {
        match value {
            ListRef::Builtin => Self::Builtin,
            ListRef::Custom(index) => Self::Custom(index),
        }
    }
}

/// One pairing of a word list with a transform list.
///
/// A dictionary declares up to sixty-four of these. A decoder resolving a
/// static dictionary reference starts at the combination the context map names
/// and falls through the rest in declaration order when the reference reaches
/// past the words the current one holds.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::{DictionaryCombination, ListSelector};
///
/// // The dictionary's own words, with the transforms RFC 7932 fixes.
/// let combination = DictionaryCombination::new(ListSelector::Custom(0), ListSelector::Builtin);
///
/// assert_eq!(combination.words(), ListSelector::Custom(0));
/// assert_eq!(combination.transforms(), ListSelector::Builtin);
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct DictionaryCombination {
    /// Where the words come from.
    words: ListSelector,
    /// Where the transforms come from.
    transforms: ListSelector,
}

impl DictionaryCombination {
    /// Pairs one word list with one transform list.
    #[must_use]
    pub const fn new(words: ListSelector, transforms: ListSelector) -> Self {
        Self { words, transforms }
    }

    /// Returns where the words come from.
    #[must_use]
    pub const fn words(self) -> ListSelector {
        self.words
    }

    /// Returns where the transforms come from.
    #[must_use]
    pub const fn transforms(self) -> ListSelector {
        self.transforms
    }
}

impl From<DictionaryCombination> for Combination {
    /// Lowers the public combination into the one the codec uses.
    fn from(value: DictionaryCombination) -> Self {
        Self {
            words: value.words.into(),
            transforms: value.transforms.into(),
        }
    }
}

impl From<Combination> for DictionaryCombination {
    /// Lifts the codec's combination into the public one.
    fn from(value: Combination) -> Self {
        Self {
            words: value.words.into(),
            transforms: value.transforms.into(),
        }
    }
}

/// Which combination each of the sixty-four literal contexts starts from.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::ContextMap;
///
/// let mut map = ContextMap::uniform(0);
/// map.set(7, 1);
///
/// assert_eq!(map[7], 1);
/// assert_eq!(map[0], 0);
/// assert_eq!(map.as_ref().len(), 64);
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ContextMap([u8; CONTEXTS]);

impl ContextMap {
    /// Points every context at the same combination.
    #[must_use]
    pub const fn uniform(combination: u8) -> Self {
        Self([combination; CONTEXTS])
    }

    /// Points one context at a combination.
    ///
    /// A `context` of sixty-four or more is ignored: the map has exactly
    /// [`CONTEXTS`] entries and cannot grow. Whether the combination exists is
    /// checked when the dictionary is built, not here.
    pub const fn set(&mut self, context: usize, combination: u8) {
        if context < CONTEXTS {
            self.0[context] = combination;
        }
    }
}

impl Default for ContextMap {
    /// Points every context at the first combination.
    fn default() -> Self {
        Self::uniform(0)
    }
}

impl From<[u8; CONTEXTS]> for ContextMap {
    /// Takes the entries as they are.
    fn from(value: [u8; CONTEXTS]) -> Self {
        Self(value)
    }
}

impl From<ContextMap> for [u8; CONTEXTS] {
    /// Returns the entries as they are.
    fn from(value: ContextMap) -> Self {
        value.0
    }
}

impl AsRef<[u8]> for ContextMap {
    /// Borrows the sixty-four entries.
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl ::core::ops::Index<usize> for ContextMap {
    type Output = u8;

    /// Returns the combination one context starts from.
    ///
    /// # Panics
    ///
    /// Panics when `context` is sixty-four or more, as slice indexing does.
    /// [`ContextMap::as_ref`] is the non-panicking way to read one.
    fn index(&self, context: usize) -> &Self::Output {
        &self.0[context]
    }
}

/// A custom list of static dictionary words.
///
/// Words are grouped by length, four to thirty-one bytes, and each non-empty
/// group holds a power-of-two number of words: the format addresses a word by a
/// fixed-width index, so there is no way to spell any other count.
/// [`WordListBuilder`] pads the groups for you.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::WordList;
///
/// let list = WordList::builder()
///     .add_word(b"alpha")
///     .add_word(b"bravo")
///     .add_word(b"charlie")
///     .build()?;
///
/// // "alpha" and "bravo" are both five bytes, so that group is already a
/// // usable size; "charlie" is alone at seven and is padded to two, because
/// // the format spells "one word" and "no words" the same way.
/// assert_eq!(list.word_count(5), 2);
/// assert_eq!(list.word_count(7), 2);
/// assert_eq!(list.word(5, 0), b"alpha");
/// # Ok::<(), mbrotli::dictionary::WordListError>(())
/// ```
#[derive(Debug, Clone)]
pub struct WordList {
    /// The list the codec and the encoder read.
    inner: CoreWordList,
}

impl WordList {
    /// Returns the RFC 7932 word list, the one an ordinary stream uses.
    ///
    /// Costs nothing but the offset tables: the hundred and twenty kilobytes of
    /// words are borrowed from static storage, not copied.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::WordList;
    ///
    /// assert_eq!(WordList::builtin().word(4, 0), b"time");
    /// ```
    #[must_use]
    pub fn builtin() -> Self {
        Self {
            inner: CoreWordList::builtin(),
        }
    }

    /// Starts building a word list from individual words.
    #[must_use]
    pub fn builder() -> WordListBuilder {
        WordListBuilder::default()
    }

    /// Borrows the list as the view a parsed dictionary hands out.
    ///
    /// The read accessors live on [`WordListView`], so an owned list and one
    /// inside a [`SerializedDictionary`] are read exactly the same way.
    #[must_use]
    pub const fn as_view(&self) -> WordListView<'_> {
        WordListView::new(&self.inner)
    }

    /// Returns how many words of `length` the list holds.
    ///
    /// Zero for a length the list does not cover, including every length below
    /// four and above thirty-one.
    #[must_use]
    pub fn word_count(&self, length: usize) -> usize {
        self.as_view().word_count(length)
    }

    /// Returns one word, or an empty slice when it does not exist.
    ///
    /// `index` counts within the words of `length`, which is how a static
    /// dictionary reference addresses one.
    #[must_use]
    pub fn word(&self, length: usize, index: usize) -> &[u8] {
        self.as_view().word(length, index)
    }

    /// Returns every word byte, shortest length first.
    ///
    /// This is the block the wire carries verbatim.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.as_view().as_bytes()
    }

    /// Returns the list the codec reads.
    pub(crate) fn into_inner(self) -> CoreWordList {
        self.inner
    }

    /// Wraps a list the codec produced.
    pub(crate) const fn from_inner(inner: CoreWordList) -> Self {
        Self { inner }
    }
}

impl Default for WordList {
    /// Returns the RFC 7932 word list.
    fn default() -> Self {
        Self::builtin()
    }
}

/// A borrowed view of one word list.
///
/// What [`SerializedDictionary::word_list`] hands out, so reading a parsed
/// dictionary's words copies nothing.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::{SerializedDictionary, WordList};
///
/// let dictionary = SerializedDictionary::builder()
///     .add_word_list(WordList::builder().add_word(b"alpha").add_word(b"bravo").build()?)
///     .build()?;
///
/// let words = dictionary.word_list(0).expect("the list was added");
///
/// assert_eq!(words.word_count(5), 2);
/// assert_eq!(words.word(5, 1), b"bravo");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Copy, Clone, Debug)]
pub struct WordListView<'a> {
    /// The list being viewed.
    inner: &'a CoreWordList,
}

impl<'a> WordListView<'a> {
    /// Borrows the codec's list.
    pub(crate) const fn new(inner: &'a CoreWordList) -> Self {
        Self { inner }
    }

    /// Returns how many words of `length` the list holds.
    #[must_use]
    pub fn word_count(&self, length: usize) -> usize {
        self.inner.word_count(length)
    }

    /// Returns one word, or an empty slice when it does not exist.
    #[must_use]
    pub fn word(&self, length: usize, index: usize) -> &'a [u8] {
        self.inner.word(length, index)
    }

    /// Returns every word byte, shortest length first.
    #[must_use]
    pub fn as_bytes(&self) -> &'a [u8] {
        self.inner.data()
    }

    /// Copies the view into an owned list.
    #[must_use]
    pub fn to_owned_list(&self) -> WordList {
        WordList::from_inner(self.inner.clone())
    }
}

/// Collects words and lays them out in the format's fixed-width groups.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::{WordList, WordListError};
///
/// // A word shorter than four bytes has no representation in the format.
/// assert!(matches!(
///     WordList::builder().add_word(b"ab").build(),
///     Err(WordListError::WordLength { length: 2, .. })
/// ));
/// ```
#[derive(Debug, Default, Clone)]
pub struct WordListBuilder {
    /// Words added so far, in the order they were added.
    words: Vec<Box<[u8]>>,
}

impl WordListBuilder {
    /// Adds one word.
    ///
    /// Order within a length group is preserved; order between groups is not,
    /// because the format stores the groups shortest first.
    #[must_use]
    pub fn add_word<B>(mut self, word: B) -> Self
    where
        B: AsRef<[u8]>,
    {
        self.words.push(Box::from(word.as_ref()));
        self
    }

    /// Lays the words out, padding each group to a power of two.
    ///
    /// A group that is not already a power of two is filled by repeating its
    /// last word. The repeats are ordinary words a reference may reach, and
    /// they change nothing a decoder does: the same bytes are simply reachable
    /// through more than one index.
    ///
    /// # Errors
    ///
    /// Returns [`WordListError::WordLength`] for a word outside four to
    /// thirty-one bytes, [`WordListError::TooManyWords`] when a group would
    /// need more than `2^15` entries, and [`WordListError::Empty`] when no word
    /// was added.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::WordList;
    ///
    /// let list = WordList::builder().add_word(b"lorem").add_word(b"ipsum").add_word(b"dolor").build()?;
    ///
    /// // Three five-byte words round up to four; the last one repeats.
    /// assert_eq!(list.word_count(5), 4);
    /// assert_eq!(list.word(5, 3), b"dolor");
    /// # Ok::<(), mbrotli::dictionary::WordListError>(())
    /// ```
    pub fn build(self) -> Result<WordList, WordListError> {
        if self.words.is_empty() {
            return Err(WordListError::Empty);
        }
        let mut groups: Vec<Vec<Box<[u8]>>> = vec![Vec::new(); MAX_WORD_LENGTH + 1];
        for word in self.words {
            let length = word.len();
            let Some(group) = groups.get_mut(length) else {
                return Err(WordListError::WordLength {
                    length,
                    word: truncate_for_report(&word),
                });
            };
            if length < MIN_WORD_LENGTH {
                return Err(WordListError::WordLength {
                    length,
                    word: truncate_for_report(&word),
                });
            }
            group.push(word);
        }

        let mut size_bits = [0u8; NUM_ENCODED_LENGTHS];
        let mut data = Vec::new();
        for length in MIN_WORD_LENGTH..=MAX_WORD_LENGTH {
            let group = &groups[length];
            if group.is_empty() {
                continue;
            }
            let bits = ceil_log2(group.len());
            if bits > MAX_SIZE_BITS {
                return Err(WordListError::TooManyWords {
                    length,
                    count: group.len(),
                    limit: 1usize << MAX_SIZE_BITS,
                });
            }
            size_bits[length - MIN_WORD_LENGTH] = bits;
            let target = 1usize << bits;
            for index in 0..target {
                // Past the last real word the group repeats it, which is what
                // pads the count up to a power of two. `group` is not empty, so
                // `last` is only `None` on a path the check above rules out and
                // the fallback is an empty write rather than a panic.
                let word = match group.get(index).or_else(|| group.last()) {
                    Some(word) => word.as_ref(),
                    None => &[],
                };
                data.extend_from_slice(word);
            }
        }
        CoreWordList::from_parts(&size_bits, Cow::Owned(data))
            .map(WordList::from_inner)
            .map_err(WordListError::from)
    }
}

/// Returns the smallest exponent that can address `count` words.
///
/// Never zero for a non-empty group: the format spells "no words of this
/// length" as a zero exponent, so a group of one word is stored as two.
fn ceil_log2(count: usize) -> u8 {
    if count <= 2 {
        return 1;
    }
    let bits = usize::BITS - (count - 1).leading_zeros();
    // A group larger than `2^15` is rejected by the caller, so the cast is
    // lossless on every path that keeps its result.
    u8::try_from(bits).unwrap_or(u8::MAX)
}

/// Returns a short, printable stand-in for a word in an error message.
fn truncate_for_report(word: &[u8]) -> Box<[u8]> {
    Box::from(word.get(..word.len().min(16)).unwrap_or_default())
}

/// Error returned when a word list cannot be built.
#[derive(Error, Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum WordListError {
    /// No word was added.
    #[error("a word list must hold at least one word")]
    Empty,
    /// A word is outside the four-to-thirty-one-byte range the format covers.
    #[error("a static dictionary word is 4 to 31 bytes, not {length}")]
    WordLength {
        /// How long the word was.
        length: usize,
        /// The first sixteen bytes of it.
        word: Box<[u8]>,
    },
    /// A length group holds more words than the format can index.
    #[error("length {length} holds {count} words, past the limit of {limit}")]
    TooManyWords {
        /// The length whose group overflowed.
        length: usize,
        /// How many words it held.
        count: usize,
        /// How many it may hold.
        limit: usize,
    },
    /// The size bits and the word data disagree about how long the data is.
    ///
    /// Only reachable from parsing, where the two arrive independently.
    #[error("the size bits describe {expected} bytes of words, but {found} were given")]
    DataLength {
        /// How many bytes the size bits describe.
        expected: usize,
        /// How many bytes were given.
        found: usize,
    },
    /// A length claims more words than the format allows.
    #[error("length {length} claims 2^{bits} words, past the limit of 2^15")]
    TooManySizeBits {
        /// The length that claimed it.
        length: usize,
        /// The exponent it claimed.
        bits: u8,
    },
}

impl From<CoreWordListError> for WordListError {
    /// Lifts the codec's word list error into the public one.
    fn from(value: CoreWordListError) -> Self {
        match value {
            CoreWordListError::TooManySizeBits { length, bits } => {
                Self::TooManySizeBits { length, bits }
            }
            CoreWordListError::DataLength { expected, found } => {
                Self::DataLength { expected, found }
            }
        }
    }
}

/// A custom list of word transformations.
///
/// Each transform is a prefix, an operation and a suffix. The prefixes and
/// suffixes live in one shared table that transforms index into, so a string
/// used by several transforms is stored once; [`TransformListBuilder`] does
/// that deduplication for you, which is also what makes the encoding canonical.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::{TransformList, TransformOperation};
///
/// let list = TransformList::builder()
///     .add_transform(b"", TransformOperation::Identity, b"")
///     .add_transform(b"<", TransformOperation::FermentFirst, b">")
///     .build()?;
///
/// assert_eq!(list.len(), 2);
/// assert_eq!(list.apply(1, b"tag"), b"<Tag>".to_vec());
/// # Ok::<(), mbrotli::dictionary::TransformListError>(())
/// ```
#[derive(Debug, Clone)]
pub struct TransformList {
    /// The list the codec and the encoder read.
    inner: CoreTransformList,
}

impl TransformList {
    /// Returns the RFC 7932 transform list, the one an ordinary stream uses.
    ///
    /// Costs nothing but the stringlet offsets: the tables themselves are
    /// borrowed from static storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::TransformList;
    ///
    /// let builtin = TransformList::builtin();
    ///
    /// assert_eq!(builtin.len(), 121);
    /// assert_eq!(builtin.apply(0, b"word"), b"word".to_vec());
    /// ```
    #[must_use]
    pub fn builtin() -> Self {
        Self {
            inner: CoreTransformList::builtin(),
        }
    }

    /// Starts building a transform list.
    #[must_use]
    pub fn builder() -> TransformListBuilder {
        TransformListBuilder::default()
    }

    /// Borrows the list as the view a parsed dictionary hands out.
    #[must_use]
    pub const fn as_view(&self) -> TransformListView<'_> {
        TransformListView::new(&self.inner)
    }

    /// Returns how many transforms the list defines.
    #[must_use]
    pub fn len(&self) -> usize {
        self.as_view().len()
    }

    /// Returns whether the list defines no transform at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.as_view().is_empty()
    }

    /// Returns one transform's prefix, or an empty slice past the end.
    #[must_use]
    pub fn prefix(&self, index: usize) -> &[u8] {
        self.as_view().prefix(index)
    }

    /// Returns one transform's suffix, or an empty slice past the end.
    #[must_use]
    pub fn suffix(&self, index: usize) -> &[u8] {
        self.as_view().suffix(index)
    }

    /// Returns what one transform does to the word between prefix and suffix.
    ///
    /// `None` past the end of the list.
    #[must_use]
    pub fn operation(&self, index: usize) -> Option<TransformOperation> {
        self.as_view().operation(index)
    }

    /// Applies one transform to `word` and returns the result.
    ///
    /// Allocates, so it is a convenience for inspecting and testing a list
    /// rather than something compression calls: the encoder applies transforms
    /// through a reusable buffer and allocates nothing per candidate.
    ///
    /// A `word` longer than thirty-one bytes is truncated to it, which is the
    /// longest word the format can hold. An index past the end returns the
    /// truncated word unchanged; use [`Self::operation`] to check whether the
    /// transform exists before applying it.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::{OmitLength, TransformList, TransformOperation};
    ///
    /// let list = TransformList::builder()
    ///     .add_transform(b"", TransformOperation::OmitLast(OmitLength::try_from(3)?), b"!")
    ///     .build()?;
    ///
    /// assert_eq!(list.apply(0, b"shorten"), b"shor!".to_vec());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn apply(&self, index: usize, word: &[u8]) -> Vec<u8> {
        self.as_view().apply(index, word)
    }

    /// Returns the list the codec reads.
    pub(crate) fn into_inner(self) -> CoreTransformList {
        self.inner
    }

    /// Wraps a list the codec produced.
    pub(crate) const fn from_inner(inner: CoreTransformList) -> Self {
        Self { inner }
    }
}

impl Default for TransformList {
    /// Returns the RFC 7932 transform list.
    fn default() -> Self {
        Self::builtin()
    }
}

/// A borrowed view of one transform list.
///
/// What [`SerializedDictionary::transform_list`] hands out.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::{SerializedDictionary, TransformList, TransformOperation};
///
/// let dictionary = SerializedDictionary::builder()
///     .add_transform_list(
///         TransformList::builder()
///             .add_transform(b"[", TransformOperation::FermentAll, b"]")
///             .build()?,
///     )
///     .build()?;
///
/// let transforms = dictionary.transform_list(0).expect("the list was added");
///
/// assert_eq!(transforms.apply(0, b"loud"), b"[LOUD]".to_vec());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Copy, Clone, Debug)]
pub struct TransformListView<'a> {
    /// The list being viewed.
    inner: &'a CoreTransformList,
}

impl<'a> TransformListView<'a> {
    /// Borrows the codec's list.
    pub(crate) const fn new(inner: &'a CoreTransformList) -> Self {
        Self { inner }
    }

    /// Returns how many transforms the list defines.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether the list defines no transform at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Returns one transform's prefix, or an empty slice past the end.
    #[must_use]
    pub fn prefix(&self, index: usize) -> &'a [u8] {
        match self.inner.transform(index) {
            Some((prefix, _, _)) => self.inner.stringlet(usize::from(prefix)),
            None => &[],
        }
    }

    /// Returns one transform's suffix, or an empty slice past the end.
    #[must_use]
    pub fn suffix(&self, index: usize) -> &'a [u8] {
        match self.inner.transform(index) {
            Some((_, _, suffix)) => self.inner.stringlet(usize::from(suffix)),
            None => &[],
        }
    }

    /// Returns what one transform does, or `None` past the end of the list.
    ///
    /// A shift operation carries the parameter the wire pairs it with, which is
    /// stored in a separate block from the operation id.
    #[must_use]
    pub fn operation(&self, index: usize) -> Option<TransformOperation> {
        let (_, operation, _) = self.inner.transform(index)?;
        match TransformOperation::try_from(operation).ok()? {
            TransformOperation::ShiftFirst(_) => {
                Some(TransformOperation::ShiftFirst(self.inner.parameter(index)))
            }
            TransformOperation::ShiftAll(_) => {
                Some(TransformOperation::ShiftAll(self.inner.parameter(index)))
            }
            other => Some(other),
        }
    }

    /// Applies one transform to `word` and returns the result.
    ///
    /// Allocates; compression applies transforms through a reusable buffer
    /// instead. A `word` longer than thirty-one bytes is truncated to it.
    /// An index past the end returns the truncated word unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::{TransformList, TransformOperation};
    /// let list = TransformList::builder()
    ///     .add_transform(b"[", TransformOperation::Identity, b"]")
    ///     .build()?;
    /// let view = list.as_view();
    /// assert_eq!(view.apply(0, b"word"), b"[word]");
    /// assert_eq!(view.apply(view.len(), b"word"), b"word");
    /// assert_eq!(view.operation(view.len()), None);
    /// assert_eq!(view.apply(view.len(), &[b'x'; 40]), [b'x'; 31]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn apply(&self, index: usize, word: &[u8]) -> Vec<u8> {
        let mut scratch = TransformScratch::default();
        self.inner.apply(index, word, &mut scratch).to_vec()
    }

    /// Copies the view into an owned list.
    #[must_use]
    pub fn to_owned_list(&self) -> TransformList {
        TransformList::from_inner(self.inner.clone())
    }
}

/// Collects transforms and lays out the prefix and suffix table they share.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::{TransformList, TransformOperation};
///
/// // The two transforms share one stored copy of the suffix.
/// let list = TransformList::builder()
///     .add_transform(b"", TransformOperation::Identity, b" ")
///     .add_transform(b"", TransformOperation::FermentAll, b" ")
///     .build()?;
///
/// assert_eq!(list.suffix(0), b" ");
/// assert_eq!(list.suffix(1), b" ");
/// # Ok::<(), mbrotli::dictionary::TransformListError>(())
/// ```
#[derive(Debug, Default, Clone)]
pub struct TransformListBuilder {
    /// Each transform as it was added, before the stringlets are interned.
    transforms: Vec<PendingTransform>,
}

/// One transform as the builder holds it, before its strings are interned.
#[derive(Debug, Clone)]
struct PendingTransform {
    /// What goes before the word.
    prefix: Box<[u8]>,
    /// What happens to the word.
    operation: TransformOperation,
    /// What goes after the word.
    suffix: Box<[u8]>,
}

impl TransformListBuilder {
    /// Adds one transform.
    #[must_use]
    pub fn add_transform<P, S>(
        mut self,
        prefix: P,
        operation: TransformOperation,
        suffix: S,
    ) -> Self
    where
        P: AsRef<[u8]>,
        S: AsRef<[u8]>,
    {
        self.transforms.push(PendingTransform {
            prefix: Box::from(prefix.as_ref()),
            operation,
            suffix: Box::from(suffix.as_ref()),
        });
        self
    }

    /// Interns the prefixes and suffixes and validates the list.
    ///
    /// The stringlet table is laid out in first-use order with duplicates
    /// merged, and always ends with the zero-length terminator the format
    /// requires. That layout is deterministic, so building the same transforms
    /// twice produces the same bytes.
    ///
    /// # Errors
    ///
    /// Returns [`TransformListError::StringletTooLong`] for a prefix or suffix
    /// past two hundred and fifty-five bytes,
    /// [`TransformListError::TooManyStringlets`] past two hundred and
    /// fifty-five distinct ones, and [`TransformListError::TooManyTransforms`]
    /// past two hundred and fifty-five transforms.
    pub fn build(self) -> Result<TransformList, TransformListError> {
        if self.transforms.len() > MAX_TRANSFORMS {
            return Err(TransformListError::TooManyTransforms {
                count: self.transforms.len(),
                limit: MAX_TRANSFORMS,
            });
        }
        let mut empty_slots: Vec<usize> = Vec::new();
        // First-use order, duplicates merged, with the empty string resolved
        // last: every list ends with a zero-length terminator anyway, so the
        // empty prefix or suffix is that terminator rather than a second copy.
        let mut table: Vec<Box<[u8]>> = Vec::new();
        let mut triples: Vec<u8> = Vec::with_capacity(self.transforms.len() * 3);
        let mut params = Vec::with_capacity(self.transforms.len() * 2);
        let mut shifts = false;
        for transform in &self.transforms {
            let operation = &transform.operation;
            let prefix_id = intern(&mut table, &transform.prefix)?;
            let suffix_id = intern(&mut table, &transform.suffix)?;
            // The empty string's id is the terminator's, which is only known
            // once the table is complete; `None` marks the two slots to fill.
            triples.push(prefix_id.unwrap_or_default());
            triples.push(u8::from(*operation));
            triples.push(suffix_id.unwrap_or_default());
            if prefix_id.is_none() {
                empty_slots.push(triples.len() - 3);
            }
            if suffix_id.is_none() {
                empty_slots.push(triples.len() - 1);
            }
            params.extend_from_slice(&operation.parameter().to_le_bytes());
            shifts |= operation.shifts();
        }
        // The terminator takes the id after the last interned string, and the
        // whole table including it may hold at most 256 entries.
        let Ok(terminator) = u8::try_from(table.len()) else {
            return Err(TransformListError::TooManyStringlets {
                count: table.len() + 1,
                limit: MAX_STRINGLETS,
            });
        };
        for slot in empty_slots {
            if let Some(id) = triples.get_mut(slot) {
                *id = terminator;
            }
        }

        let mut block = Vec::new();
        for string in &table {
            block.push(u8::try_from(string.len()).unwrap_or(u8::MAX));
            block.extend_from_slice(string);
        }
        block.push(0);

        CoreTransformList::from_parts(
            Cow::Owned(block),
            Cow::Owned(triples),
            Cow::Owned(if shifts { params } else { Vec::new() }),
        )
        .map(TransformList::from_inner)
        .map_err(TransformListError::from)
    }
}

/// Interns one prefix or suffix and returns its id.
///
/// `None` stands for the empty string, whose id is the terminator's and is only
/// known once the whole table is laid out.
fn intern(table: &mut Vec<Box<[u8]>>, string: &[u8]) -> Result<Option<u8>, TransformListError> {
    if string.len() > MAX_STRINGLET_BYTES {
        return Err(TransformListError::StringletTooLong {
            length: string.len(),
            limit: MAX_STRINGLET_BYTES,
        });
    }
    if string.is_empty() {
        return Ok(None);
    }
    if let Some(index) = table.iter().position(|held| held.as_ref() == string) {
        // An index into a table that was itself capped at 255 entries.
        return Ok(u8::try_from(index).ok());
    }
    if table.len() >= MAX_STRINGLETS - 1 {
        return Err(TransformListError::TooManyStringlets {
            count: table.len() + 2,
            limit: MAX_STRINGLETS,
        });
    }
    table.push(Box::from(string));
    Ok(u8::try_from(table.len() - 1).ok())
}

/// Error returned when a transform list cannot be built or parsed.
#[derive(Error, Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum TransformListError {
    /// A prefix or suffix is longer than one length byte can describe.
    #[error("a prefix or suffix is at most {limit} bytes, not {length}")]
    StringletTooLong {
        /// How long it was.
        length: usize,
        /// How long it may be.
        limit: usize,
    },
    /// More distinct prefixes and suffixes than the format can index.
    #[error("a transform list holds at most {limit} distinct strings, not {count}")]
    TooManyStringlets {
        /// How many were needed.
        count: usize,
        /// How many may be held.
        limit: usize,
    },
    /// More transforms than a one-byte count can express.
    #[error("a transform list holds at most {limit} transforms, not {count}")]
    TooManyTransforms {
        /// How many were offered.
        count: usize,
        /// How many may be held.
        limit: usize,
    },
    /// The parsed prefix and suffix block is malformed.
    ///
    /// Only reachable from parsing: the builder lays the block out itself.
    #[error("the prefix and suffix data is malformed: {detail}")]
    MalformedStringlets {
        /// What was wrong with it.
        detail: String,
    },
    /// A parsed transform refers to something that does not exist.
    ///
    /// Only reachable from parsing.
    #[error("a transform refers to something undefined: {detail}")]
    UndefinedReference {
        /// What was wrong with it.
        detail: String,
    },
}

impl From<CoreTransformListError> for TransformListError {
    /// Lifts the codec's transform list error into the public one.
    ///
    /// The codec distinguishes more shapes of malformed input than a caller of
    /// the builder can produce, so the parse-only ones collapse into two
    /// variants that carry the detail as text.
    fn from(value: CoreTransformListError) -> Self {
        match value {
            CoreTransformListError::TooManyTransforms { count } => Self::TooManyTransforms {
                count,
                limit: MAX_TRANSFORMS,
            },
            CoreTransformListError::TooManyStringlets => Self::TooManyStringlets {
                count: MAX_STRINGLETS + 1,
                limit: MAX_STRINGLETS,
            },
            CoreTransformListError::EmptyStringlets
            | CoreTransformListError::StringletOverrun { .. }
            | CoreTransformListError::MisplacedTerminator => Self::MalformedStringlets {
                detail: value.to_string(),
            },
            CoreTransformListError::UndefinedStringlet { .. }
            | CoreTransformListError::UndefinedOperation { .. }
            | CoreTransformListError::ParameterLength { .. }
            | CoreTransformListError::UnusedParameter { .. } => Self::UndefinedReference {
                detail: value.to_string(),
            },
        }
    }
}

/// An immutable RFC 9841 shared dictionary description.
///
/// Built by [`SerializedDictionaryBuilder`] or parsed from bytes, and turned
/// into a [`PreparedDictionary`](super::PreparedDictionary) by
/// [`DictionaryBuilder::add_serialized`](super::DictionaryBuilder::add_serialized).
/// It holds no encoder state and is `Send` and `Sync`, so one description can
/// be parsed once and reused everywhere.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::{
///     DictionaryCombination, ListSelector, SerializedDictionary, TransformOperation, WordList,
/// };
///
/// let dictionary = SerializedDictionary::builder()
///     .with_prefix(&b"a prefix the stream may copy from"[..])
///     .add_word_list(WordList::builder().add_word(b"payload").build()?)
///     .add_combination(DictionaryCombination::new(
///         ListSelector::Custom(0),
///         ListSelector::Builtin,
///     ))
///     .build()?;
///
/// assert_eq!(dictionary.word_list_count(), 1);
/// assert_eq!(dictionary.combination_count(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct SerializedDictionary {
    /// The parsed description the codec and the encoder read.
    inner: SerializedDictionaryData,
}

impl SerializedDictionary {
    /// Starts building a dictionary.
    #[must_use]
    pub fn builder() -> SerializedDictionaryBuilder {
        SerializedDictionaryBuilder::default()
    }

    /// Parses a dictionary stream under the given limits.
    ///
    /// Every count and length is checked before the bytes it describes are
    /// copied, so a hostile stream cannot allocate more than `limits` allows.
    /// Bytes after the end of the structure are refused: the reference ignores
    /// such a tail, and refusing it is what makes a dictionary's bytes and its
    /// meaning one to one.
    ///
    /// # Errors
    ///
    /// Returns [`SerializedDictionaryError`] naming the first rule broken.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::{DictionaryLimits, SerializedDictionary, SerializedDictionaryError};
    ///
    /// let bytes = SerializedDictionary::builder()
    ///     .with_prefix(&b"0123456789"[..])
    ///     .build()?
    ///     .to_bytes();
    ///
    /// let limits = DictionaryLimits::default().with_max_prefix_bytes(4);
    ///
    /// assert!(matches!(
    ///     SerializedDictionary::parse(&bytes, limits),
    ///     Err(SerializedDictionaryError::LimitExceeded { .. })
    /// ));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn parse(
        bytes: &[u8],
        limits: DictionaryLimits,
    ) -> Result<Self, SerializedDictionaryError> {
        let inner = serialized::parse_exact(bytes, &SerializedLimits::from(limits))?;
        Ok(Self { inner })
    }

    /// Returns the LZ77 prefix, or an empty slice when there is none.
    #[must_use]
    pub fn prefix(&self) -> &[u8] {
        self.inner.prefix()
    }

    /// Returns how many custom word lists the dictionary carries.
    #[must_use]
    pub fn word_list_count(&self) -> usize {
        self.inner.word_lists().len()
    }

    /// Returns a view of one custom word list, or `None` past the end.
    ///
    /// A view rather than the owned [`WordList`], so reading a parsed
    /// dictionary's words copies nothing.
    #[must_use]
    pub fn word_list(&self, index: usize) -> Option<WordListView<'_>> {
        self.inner.word_lists().get(index).map(WordListView::new)
    }

    /// Returns how many custom transform lists the dictionary carries.
    #[must_use]
    pub fn transform_list_count(&self) -> usize {
        self.inner.transform_lists().len()
    }

    /// Returns a view of one custom transform list, or `None` past the end.
    #[must_use]
    pub fn transform_list(&self, index: usize) -> Option<TransformListView<'_>> {
        self.inner
            .transform_lists()
            .get(index)
            .map(TransformListView::new)
    }

    /// Returns how many combinations the dictionary declares.
    ///
    /// Zero when the dictionary carries no custom list, in which case the
    /// implicit combination is the RFC 7932 word and transform lists.
    #[must_use]
    pub fn combination_count(&self) -> usize {
        self.inner.combinations().len()
    }

    /// Returns the combinations in declaration order.
    ///
    /// Declaration order is fall-through order: a reference that reaches past
    /// the words of one combination continues into the next.
    pub fn combinations(&self) -> impl ExactSizeIterator<Item = DictionaryCombination> + '_ {
        self.inner
            .combinations()
            .iter()
            .copied()
            .map(DictionaryCombination::from)
    }

    /// Returns the context map, or `None` when the dictionary is not context based.
    #[must_use]
    pub fn context_map(&self) -> Option<ContextMap> {
        self.inner.context_map().copied().map(ContextMap::from)
    }

    /// Returns whether the dictionary replaces the RFC 7932 static dictionary.
    #[must_use]
    pub fn is_custom_static(&self) -> bool {
        self.inner.is_custom_static()
    }

    /// Returns how many bytes [`SerializedDictionary::to_bytes`] will produce.
    #[must_use]
    pub fn serialized_len(&self) -> usize {
        self.inner.wire_len()
    }

    /// Returns the dictionary in its canonical wire encoding.
    ///
    /// Canonical means the shortest varint for the prefix length, the
    /// combination block present exactly when a custom list is, the parameter
    /// block present exactly when a transform shifts, and the context map
    /// present exactly when the dictionary is context based. Parsing the result
    /// yields an equal dictionary, and serializing that yields equal bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::SerializedDictionary;
    ///
    /// let dictionary = SerializedDictionary::builder().with_prefix(&b"bytes"[..]).build()?;
    /// let bytes = dictionary.to_bytes();
    ///
    /// assert_eq!(bytes.len(), dictionary.serialized_len());
    /// assert_eq!(SerializedDictionary::try_from(&bytes[..])?.to_bytes(), bytes);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.serialized_len());
        self.write_to(&mut out);
        out
    }

    /// Appends the canonical wire encoding to `out`.
    ///
    /// The same bytes [`SerializedDictionary::to_bytes`] returns, without the
    /// intermediate allocation, for a caller assembling a larger container.
    /// Preserves existing bytes and grows `out` as needed; it does not add a
    /// length prefix for the enclosing container.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::SerializedDictionary;
    /// let dictionary = SerializedDictionary::builder().with_prefix(&b"prefix"[..]).build()?;
    /// let mut container = b"header".to_vec();
    /// let start = container.len();
    /// dictionary.write_to(&mut container);
    /// assert_eq!(&container[..start], b"header");
    /// assert_eq!(container.len() - start, dictionary.serialized_len());
    /// let parsed = SerializedDictionary::try_from(&container[start..])?;
    /// assert_eq!(parsed.prefix(), b"prefix");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write_to(&self, out: &mut Vec<u8>) {
        // `serialize` can only fail on a prefix longer than a varint can
        // express, which no constructed dictionary holds: both the builder and
        // the parser cap it at the format's own ceiling first.
        let _ = self.inner.serialize(out);
    }

    /// Returns the description the codec and the encoder read.
    pub(crate) const fn data(&self) -> &SerializedDictionaryData {
        &self.inner
    }
}

impl TryFrom<&[u8]> for SerializedDictionary {
    type Error = SerializedDictionaryError;

    /// Parses a dictionary stream under the default limits.
    ///
    /// # Errors
    ///
    /// As [`SerializedDictionary::parse`].
    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        Self::parse(bytes, DictionaryLimits::default())
    }
}

/// Collects the parts of a serialized dictionary and validates them together.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::{
///     ContextMap, DictionaryCombination, ListSelector, SerializedDictionary, WordList,
/// };
///
/// let mut map = ContextMap::uniform(0);
/// map.set(1, 1);
///
/// let dictionary = SerializedDictionary::builder()
///     .add_word_list(WordList::builder().add_word(b"first").build()?)
///     .add_word_list(WordList::builder().add_word(b"second").build()?)
///     .add_combination(DictionaryCombination::new(ListSelector::Custom(0), ListSelector::Builtin))
///     .add_combination(DictionaryCombination::new(ListSelector::Custom(1), ListSelector::Builtin))
///     .with_context_map(map)
///     .build()?;
///
/// assert_eq!(dictionary.combination_count(), 2);
/// assert!(dictionary.context_map().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Default, Clone)]
pub struct SerializedDictionaryBuilder {
    /// The LZ77 prefix, if one was set.
    prefix: Option<Box<[u8]>>,
    /// Custom word lists, in declaration order.
    word_lists: Vec<WordList>,
    /// Custom transform lists, in declaration order.
    transform_lists: Vec<TransformList>,
    /// Combinations, in declaration order.
    combinations: Vec<DictionaryCombination>,
    /// The context map, if one was set.
    context_map: Option<ContextMap>,
    /// The ceilings [`SerializedDictionaryBuilder::build`] checks against.
    limits: DictionaryLimits,
}

impl SerializedDictionaryBuilder {
    /// Sets the LZ77 prefix, replacing any previously set.
    #[must_use]
    pub fn with_prefix<B>(mut self, bytes: B) -> Self
    where
        B: Into<Box<[u8]>>,
    {
        self.prefix = Some(bytes.into());
        self
    }

    /// Adds one custom word list after the ones already added.
    #[must_use]
    pub fn add_word_list(mut self, list: WordList) -> Self {
        self.word_lists.push(list);
        self
    }

    /// Adds one custom transform list after the ones already added.
    #[must_use]
    pub fn add_transform_list(mut self, list: TransformList) -> Self {
        self.transform_lists.push(list);
        self
    }

    /// Adds one combination after the ones already added.
    ///
    /// Declaration order is fall-through order.
    #[must_use]
    pub fn add_combination(mut self, combination: DictionaryCombination) -> Self {
        self.combinations.push(combination);
        self
    }

    /// Sets the context map, making the dictionary context based.
    #[must_use]
    pub const fn with_context_map(mut self, map: ContextMap) -> Self {
        self.context_map = Some(map);
        self
    }

    /// Replaces the resource limits the dictionary is built under.
    #[must_use]
    pub const fn with_limits(mut self, limits: DictionaryLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Validates the parts against each other and against the limits.
    ///
    /// When a custom list was added but no combination was, one combination is
    /// implied: the first custom word list if there is one and the built-in
    /// otherwise, paired the same way for transforms. That is the only
    /// combination such a dictionary could sensibly mean, and leaving it out
    /// would make the common case need a line of ceremony.
    ///
    /// # Errors
    ///
    /// Returns [`SerializedDictionaryError`] naming the first rule broken:
    /// a dangling combination or context map entry, a count past sixty-four, or
    /// a resource limit.
    pub fn build(self) -> Result<SerializedDictionary, SerializedDictionaryError> {
        let custom = !self.word_lists.is_empty() || !self.transform_lists.is_empty();
        let mut combinations = self.combinations;
        if custom && combinations.is_empty() {
            combinations.push(DictionaryCombination::new(
                if self.word_lists.is_empty() {
                    ListSelector::Builtin
                } else {
                    ListSelector::Custom(0)
                },
                if self.transform_lists.is_empty() {
                    ListSelector::Builtin
                } else {
                    ListSelector::Custom(0)
                },
            ));
        }
        let inner = SerializedDictionaryData::assemble(
            self.prefix,
            self.word_lists
                .into_iter()
                .map(WordList::into_inner)
                .collect(),
            self.transform_lists
                .into_iter()
                .map(TransformList::into_inner)
                .collect(),
            combinations.into_iter().map(Combination::from).collect(),
            self.context_map.map(<[u8; CONTEXTS]>::from),
            &SerializedLimits::from(self.limits),
        )?;
        Ok(SerializedDictionary { inner })
    }
}

/// Error returned when a serialized dictionary cannot be parsed or built.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::{SerializedDictionary, SerializedDictionaryError};
///
/// assert!(matches!(
///     SerializedDictionary::try_from(&b"not a dictionary"[..]),
///     Err(SerializedDictionaryError::BadMagic { .. })
/// ));
/// ```
#[derive(Error, Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum SerializedDictionaryError {
    /// The stream did not start with the two bytes every dictionary starts with.
    #[error("a serialized dictionary starts with 0x91 0x00, not {found:02X?}")]
    BadMagic {
        /// The first two bytes that were found, or fewer at the end of input.
        found: Box<[u8]>,
    },
    /// The stream ended in the middle of a field.
    #[error("the dictionary ends after {length} bytes, mid-{field}")]
    Truncated {
        /// Which field was being read.
        field: &'static str,
        /// How many bytes the stream held.
        length: usize,
    },
    /// A field held a value the format does not define.
    #[error("{detail}")]
    Malformed {
        /// What was wrong with it.
        detail: String,
    },
    /// A reference names something the dictionary does not contain.
    #[error("{detail}")]
    UndefinedReference {
        /// What was wrong with it.
        detail: String,
    },
    /// A resource limit was exceeded.
    #[error("the dictionary's {what} of {found} exceeds the limit of {limit}")]
    LimitExceeded {
        /// Which limit was hit.
        what: &'static str,
        /// What was asked for.
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

impl From<SerializedError> for SerializedDictionaryError {
    /// Lifts the codec's error into the public one, keeping the detail as text.
    ///
    /// The codec names every field and rule separately because it is what
    /// enforces them; a caller needs to know which of five kinds of thing went
    /// wrong and to be able to print the rest.
    fn from(value: SerializedError) -> Self {
        match value {
            SerializedError::BadMagic { found } => Self::BadMagic {
                found: found.into_boxed_slice(),
            },
            SerializedError::Truncated { field, position } => Self::Truncated {
                field,
                length: position,
            },
            SerializedError::LimitExceeded { what, found, limit } => {
                Self::LimitExceeded { what, found, limit }
            }
            SerializedError::TrailingBytes { extra } => Self::TrailingBytes { extra },
            SerializedError::UndefinedList { .. }
            | SerializedError::UndefinedCombination { .. }
            | SerializedError::NoCombinations => Self::UndefinedReference {
                detail: value.to_string(),
            },
            other => Self::Malformed {
                detail: other.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a dictionary carrying one word list, one transform list, a
    /// prefix, two combinations and a context map.
    fn rich() -> SerializedDictionary {
        let mut map = ContextMap::uniform(0);
        map.set(5, 1);
        map.set(CONTEXTS, 1);
        SerializedDictionary::builder()
            .with_prefix(&b"prefix bytes"[..])
            .add_word_list(
                WordList::builder()
                    .add_word(b"alpha")
                    .add_word(b"bravo")
                    .build()
                    .expect("well formed"),
            )
            .add_transform_list(
                TransformList::builder()
                    .add_transform(b"", TransformOperation::Identity, b"")
                    .add_transform(b"<", TransformOperation::FermentAll, b">")
                    .build()
                    .expect("well formed"),
            )
            .add_combination(DictionaryCombination::new(
                ListSelector::Custom(0),
                ListSelector::Custom(0),
            ))
            .add_combination(DictionaryCombination::default())
            .with_context_map(map)
            .with_limits(DictionaryLimits::default())
            .build()
            .expect("the parts are consistent")
    }

    #[test]
    fn an_omit_length_accepts_one_to_nine() {
        assert_eq!(u8::from(OmitLength::try_from(1).expect("in range")), 1);
        assert_eq!(u8::from(OmitLength::MIN), 1);
        assert_eq!(u8::from(OmitLength::MAX), 9);
        assert_eq!(
            OmitLength::try_from(0),
            Err(OmitLengthOutOfRange { value: 0 })
        );
        assert_eq!(
            OmitLength::try_from(10),
            Err(OmitLengthOutOfRange { value: 10 })
        );
        assert_eq!(OmitLength::MAX.to_string(), "9");
        assert_eq!(
            OmitLengthOutOfRange { value: 0 }.to_string(),
            "an omit transform drops between 1 and 9 bytes, not 0"
        );
    }

    #[test]
    fn every_operation_id_round_trips() {
        for id in 0..=22u8 {
            let operation = TransformOperation::try_from(id).expect("defined");
            assert_eq!(u8::from(operation), id, "id {id}");
        }
        assert_eq!(
            TransformOperation::try_from(23),
            Err(UndefinedTransformOperation { value: 23 })
        );
        assert_eq!(
            UndefinedTransformOperation { value: 23 }.to_string(),
            "RFC 9841 defines transform operations 0 to 22, not 23"
        );
    }

    #[test]
    fn a_shift_carries_its_parameter_and_the_rest_carry_none() {
        assert_eq!(TransformOperation::ShiftFirst(7).parameter(), 7);
        assert_eq!(TransformOperation::ShiftAll(7).parameter(), 7);
        assert_eq!(TransformOperation::Identity.parameter(), 0);
        assert!(TransformOperation::ShiftAll(0).shifts());
        assert!(!TransformOperation::FermentAll.shifts());
    }

    #[test]
    fn a_list_selector_converts_both_ways() {
        assert_eq!(ListSelector::from(2), ListSelector::Custom(2));
        assert_eq!(ListSelector::default(), ListSelector::Builtin);
        assert_eq!(ListRef::from(ListSelector::Builtin), ListRef::Builtin);
        assert_eq!(ListRef::from(ListSelector::Custom(1)), ListRef::Custom(1));
        assert_eq!(ListSelector::from(ListRef::Builtin), ListSelector::Builtin);
        assert_eq!(
            ListSelector::from(ListRef::Custom(1)),
            ListSelector::Custom(1)
        );
    }

    #[test]
    fn a_combination_converts_both_ways() {
        let public = DictionaryCombination::new(ListSelector::Custom(1), ListSelector::Builtin);

        assert_eq!(public.words(), ListSelector::Custom(1));
        assert_eq!(public.transforms(), ListSelector::Builtin);
        assert_eq!(
            DictionaryCombination::from(Combination::from(public)),
            public
        );
        assert_eq!(
            DictionaryCombination::default().words(),
            ListSelector::Builtin
        );
    }

    #[test]
    fn a_context_map_reads_back_what_was_set() {
        let mut map = ContextMap::uniform(2);
        map.set(0, 1);
        // Past the end is ignored rather than panicking; the map cannot grow.
        map.set(CONTEXTS, 9);

        assert_eq!(map[0], 1);
        assert_eq!(map[1], 2);
        assert_eq!(map.as_ref().len(), CONTEXTS);
        assert_eq!(ContextMap::default()[0], 0);
        assert_eq!(<[u8; CONTEXTS]>::from(map)[0], 1);
        assert_eq!(ContextMap::from([3u8; CONTEXTS])[63], 3);
    }

    #[test]
    #[should_panic(expected = "the len is 64")]
    fn indexing_a_context_map_past_its_end_panics() {
        let _ = ContextMap::default()[CONTEXTS];
    }

    #[test]
    fn the_builtin_lists_are_the_defaults() {
        assert_eq!(
            WordList::default().word(4, 0),
            WordList::builtin().word(4, 0)
        );
        assert_eq!(
            TransformList::default().len(),
            TransformList::builtin().len()
        );
        assert_eq!(WordList::builtin().as_bytes().len(), 122_784);
        assert!(!TransformList::builtin().is_empty());
    }

    #[test]
    fn a_word_list_is_read_the_same_owned_and_borrowed() {
        let list = WordList::builder()
            .add_word(b"alpha")
            .add_word(b"bravo")
            .build()
            .expect("well formed");
        let view = list.as_view();

        assert_eq!(view.word_count(5), list.word_count(5));
        assert_eq!(view.word(5, 0), list.word(5, 0));
        assert_eq!(view.as_bytes(), list.as_bytes());
        assert_eq!(view.to_owned_list().as_bytes(), list.as_bytes());
    }

    #[test]
    fn a_word_list_builder_refuses_what_the_format_cannot_hold() {
        assert_eq!(
            WordList::builder().build().err(),
            Some(WordListError::Empty)
        );
        assert!(matches!(
            WordList::builder().add_word(b"ab").build(),
            Err(WordListError::WordLength { length: 2, .. })
        ));
        let long = [b'z'; 32];
        assert!(matches!(
            WordList::builder().add_word(long).build(),
            Err(WordListError::WordLength { length: 32, .. })
        ));
    }

    #[test]
    fn a_word_list_group_past_the_format_ceiling_is_refused() {
        let mut builder = WordList::builder();
        for index in 0..=(1usize << MAX_SIZE_BITS) {
            builder = builder.add_word(format!("{index:08}"));
        }

        assert!(matches!(
            builder.build(),
            Err(WordListError::TooManyWords { length: 8, .. })
        ));
    }

    #[test]
    fn a_word_reported_in_an_error_is_truncated() {
        let word = [b'q'; 40];

        assert_eq!(truncate_for_report(&word).len(), 16);
        assert_eq!(truncate_for_report(b"short").len(), 5);
    }

    #[test]
    fn a_group_of_one_word_is_stored_as_two() {
        assert_eq!(ceil_log2(1), 1);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(1 << 15), 15);
    }

    #[test]
    fn a_transform_list_is_read_the_same_owned_and_borrowed() {
        let list = TransformList::builder()
            .add_transform(b"<", TransformOperation::FermentAll, b">")
            .build()
            .expect("well formed");
        let view = list.as_view();

        assert_eq!(view.len(), list.len());
        assert_eq!(view.is_empty(), list.is_empty());
        assert_eq!(view.prefix(0), list.prefix(0));
        assert_eq!(view.suffix(0), list.suffix(0));
        assert_eq!(view.operation(0), list.operation(0));
        assert_eq!(view.apply(0, b"loud"), list.apply(0, b"loud"));
        assert_eq!(view.to_owned_list().len(), list.len());
        assert_eq!(list.apply(0, b"loud"), b"<LOUD>".to_vec());
    }

    #[test]
    fn an_index_past_a_transform_list_reads_as_nothing() {
        let list = TransformList::builder()
            .add_transform(b"<", TransformOperation::Identity, b">")
            .build()
            .expect("well formed");

        assert_eq!(list.prefix(9), b"");
        assert_eq!(list.suffix(9), b"");
        assert_eq!(list.operation(9), None);
        assert_eq!(list.apply(9, b"word"), b"word".to_vec());
    }

    #[test]
    fn an_empty_transform_list_is_empty() {
        let list = TransformList::builder().build().expect("well formed");

        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn a_shift_transform_keeps_its_parameter_through_a_round_trip() {
        let list = TransformList::builder()
            .add_transform(b"", TransformOperation::ShiftFirst(0x1234), b"")
            .add_transform(b"", TransformOperation::ShiftAll(0x8000), b"")
            .add_transform(b"", TransformOperation::Identity, b"")
            .build()
            .expect("well formed");

        assert_eq!(
            list.operation(0),
            Some(TransformOperation::ShiftFirst(0x1234))
        );
        assert_eq!(
            list.operation(1),
            Some(TransformOperation::ShiftAll(0x8000))
        );
        assert_eq!(list.operation(2), Some(TransformOperation::Identity));
    }

    #[test]
    fn a_transform_list_builder_refuses_what_the_format_cannot_hold() {
        let long = vec![b'p'; MAX_STRINGLET_BYTES + 1];
        assert!(matches!(
            TransformList::builder()
                .add_transform(&long[..], TransformOperation::Identity, b"")
                .build(),
            Err(TransformListError::StringletTooLong { .. })
        ));

        let mut builder = TransformList::builder();
        for index in 0..=MAX_TRANSFORMS {
            builder = builder.add_transform(b"", TransformOperation::Identity, b"");
            let _ = index;
        }
        assert!(matches!(
            builder.build(),
            Err(TransformListError::TooManyTransforms { .. })
        ));
    }

    #[test]
    fn too_many_distinct_strings_are_refused() {
        // Two distinct strings per transform, so the stringlet table runs out
        // before the two-hundred-and-fifty-five transform ceiling does.
        let mut builder = TransformList::builder();
        for index in 0..MAX_STRINGLETS / 2 {
            builder = builder.add_transform(
                format!("p{index:04}"),
                TransformOperation::Identity,
                format!("s{index:04}"),
            );
        }

        assert!(matches!(
            builder.build(),
            Err(TransformListError::TooManyStringlets { .. })
        ));
    }

    #[test]
    fn identical_strings_are_stored_once() {
        let list = TransformList::builder()
            .add_transform(b"same", TransformOperation::Identity, b"same")
            .add_transform(b"same", TransformOperation::FermentAll, b"same")
            .build()
            .expect("well formed");

        // Both transforms name the one stored copy, and the empty terminator
        // is the only other stringlet.
        assert_eq!(list.prefix(0), b"same");
        assert_eq!(list.suffix(1), b"same");
        assert_eq!(list.apply(1, b"x"), b"sameXsame".to_vec());
    }

    #[test]
    fn a_dictionary_reports_everything_it_was_built_from() {
        let dictionary = rich();

        assert_eq!(dictionary.prefix(), b"prefix bytes");
        assert_eq!(dictionary.word_list_count(), 1);
        assert_eq!(dictionary.transform_list_count(), 1);
        assert_eq!(dictionary.combination_count(), 2);
        assert!(dictionary.is_custom_static());
        assert_eq!(dictionary.context_map().map(|map| map[5]), Some(1));
        assert_eq!(
            dictionary.word_list(0).map(|list| list.word_count(5)),
            Some(2)
        );
        assert_eq!(dictionary.transform_list(0).map(|list| list.len()), Some(2));
        assert!(dictionary.word_list(1).is_none());
        assert!(dictionary.transform_list(1).is_none());
        assert_eq!(dictionary.combinations().len(), 2);
        assert_eq!(
            dictionary.combinations().next().map(|c| c.words()),
            Some(ListSelector::Custom(0))
        );
    }

    #[test]
    fn writing_to_a_buffer_matches_the_owned_bytes() {
        let dictionary = rich();
        let mut out = vec![0xAA];
        dictionary.write_to(&mut out);

        assert_eq!(out[0], 0xAA);
        assert_eq!(&out[1..], dictionary.to_bytes().as_slice());
        assert_eq!(out.len() - 1, dictionary.serialized_len());
        assert!(!dictionary.data().prefix().is_empty());
    }

    #[test]
    fn parsing_under_explicit_limits_agrees_with_the_default_ones() {
        let bytes = rich().to_bytes();
        let parsed = SerializedDictionary::parse(&bytes, DictionaryLimits::default())
            .expect("within the defaults");

        assert_eq!(
            parsed.to_bytes(),
            SerializedDictionary::try_from(&bytes[..])
                .expect("the same bytes")
                .to_bytes()
        );
    }

    #[test]
    fn a_builder_with_one_custom_list_implies_its_combination() {
        let with_words = SerializedDictionary::builder()
            .add_word_list(
                WordList::builder()
                    .add_word(b"word")
                    .build()
                    .expect("valid"),
            )
            .build()
            .expect("valid");
        let with_transforms = SerializedDictionary::builder()
            .add_transform_list(
                TransformList::builder()
                    .add_transform(b"", TransformOperation::Identity, b"")
                    .build()
                    .expect("valid"),
            )
            .build()
            .expect("valid");

        assert_eq!(
            with_words.combinations().next().map(|c| c.words()),
            Some(ListSelector::Custom(0))
        );
        assert_eq!(
            with_words.combinations().next().map(|c| c.transforms()),
            Some(ListSelector::Builtin)
        );
        assert_eq!(
            with_transforms.combinations().next().map(|c| c.words()),
            Some(ListSelector::Builtin)
        );
        assert_eq!(
            with_transforms
                .combinations()
                .next()
                .map(|c| c.transforms()),
            Some(ListSelector::Custom(0))
        );
    }

    #[test]
    fn an_empty_dictionary_carries_nothing() {
        let dictionary = SerializedDictionary::default();

        assert!(dictionary.prefix().is_empty());
        assert_eq!(dictionary.word_list_count(), 0);
        assert_eq!(dictionary.transform_list_count(), 0);
        assert_eq!(dictionary.combination_count(), 0);
        assert!(dictionary.context_map().is_none());
        assert!(!dictionary.is_custom_static());
        assert_eq!(dictionary.to_bytes(), vec![0x91, 0x00, 0, 0, 0]);
    }

    #[test]
    fn each_codec_error_lifts_into_its_public_shape() {
        let cases = [
            (
                SerializedError::BadMagic { found: vec![1, 2] },
                "a serialized dictionary starts with 0x91 0x00",
            ),
            (
                SerializedError::Truncated {
                    field: "magic",
                    position: 1,
                },
                "mid-magic",
            ),
            (
                SerializedError::TrailingBytes { extra: 3 },
                "follow the end of the dictionary",
            ),
            (SerializedError::NoCombinations, "at least one combination"),
            (
                SerializedError::LimitExceeded {
                    what: "word data",
                    found: 9,
                    limit: 4,
                },
                "exceeds the limit of 4",
            ),
            (
                SerializedError::NotABoolean {
                    field: "CONTEXT_ENABLED",
                    value: 2,
                },
                "must be 0 or 1",
            ),
        ];

        for (error, fragment) in cases {
            let lifted = SerializedDictionaryError::from(error);
            assert!(
                lifted.to_string().contains(fragment),
                "{lifted} does not mention {fragment}"
            );
        }
    }

    #[test]
    fn each_word_list_error_lifts_into_its_public_shape() {
        assert_eq!(
            WordListError::from(CoreWordListError::TooManySizeBits {
                length: 4,
                bits: 16,
            }),
            WordListError::TooManySizeBits {
                length: 4,
                bits: 16
            }
        );
        assert_eq!(
            WordListError::from(CoreWordListError::DataLength {
                expected: 8,
                found: 7,
            }),
            WordListError::DataLength {
                expected: 8,
                found: 7,
            }
        );
    }

    #[test]
    fn each_transform_list_error_lifts_into_its_public_shape() {
        assert!(matches!(
            TransformListError::from(CoreTransformListError::TooManyTransforms { count: 300 }),
            TransformListError::TooManyTransforms { count: 300, .. }
        ));
        assert!(matches!(
            TransformListError::from(CoreTransformListError::TooManyStringlets),
            TransformListError::TooManyStringlets { .. }
        ));
        assert!(matches!(
            TransformListError::from(CoreTransformListError::EmptyStringlets),
            TransformListError::MalformedStringlets { .. }
        ));
        assert!(matches!(
            TransformListError::from(CoreTransformListError::MisplacedTerminator),
            TransformListError::MalformedStringlets { .. }
        ));
        assert!(matches!(
            TransformListError::from(CoreTransformListError::StringletOverrun { length: 4 }),
            TransformListError::MalformedStringlets { .. }
        ));
        assert!(matches!(
            TransformListError::from(CoreTransformListError::UndefinedOperation {
                index: 0,
                operation: 30,
            }),
            TransformListError::UndefinedReference { .. }
        ));
        assert!(matches!(
            TransformListError::from(CoreTransformListError::UndefinedStringlet {
                index: 0,
                stringlet: 3,
                count: 1,
            }),
            TransformListError::UndefinedReference { .. }
        ));
        assert!(matches!(
            TransformListError::from(CoreTransformListError::ParameterLength {
                expected: 2,
                found: 0,
            }),
            TransformListError::UndefinedReference { .. }
        ));
        assert!(matches!(
            TransformListError::from(CoreTransformListError::UnusedParameter {
                index: 0,
                parameter: 7,
            }),
            TransformListError::UndefinedReference { .. }
        ));
    }

    #[test]
    fn the_public_limits_lower_into_the_codec_ones() {
        let limits = DictionaryLimits::default()
            .with_max_serialized_bytes(11)
            .with_max_prefix_bytes(12)
            .with_max_word_lists(3)
            .with_max_word_bytes(13)
            .with_max_transform_lists(4)
            .with_max_transform_bytes(14)
            .with_max_combinations(5);
        let lowered = SerializedLimits::from(limits);

        assert_eq!(lowered.max_total_bytes, 11);
        assert_eq!(lowered.max_prefix_bytes, 12);
        assert_eq!(lowered.max_word_lists, 3);
        assert_eq!(lowered.max_word_bytes, 13);
        assert_eq!(lowered.max_transform_lists, 4);
        assert_eq!(lowered.max_transform_bytes, 14);
        assert_eq!(lowered.max_combinations, 5);
    }

    #[test]
    fn interning_reports_the_first_use_of_each_string() {
        let mut table = Vec::new();

        assert_eq!(intern(&mut table, b""), Ok(None));
        assert_eq!(intern(&mut table, b"a"), Ok(Some(0)));
        assert_eq!(intern(&mut table, b"b"), Ok(Some(1)));
        assert_eq!(intern(&mut table, b"a"), Ok(Some(0)));
        assert_eq!(table.len(), 2);
    }
}

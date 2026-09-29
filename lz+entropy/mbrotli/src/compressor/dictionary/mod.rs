//! Immutable dictionaries a stream can be compressed against.
//!
//! [RFC 9841] lets a caller place up to fifteen LZ77 prefix dictionaries in
//! front of a stream: bytes the encoder may copy from but never emits, which a
//! decoder has to be given out of band in the same order.
//!
//! A [`PreparedDictionary`] is the result of indexing those bytes. It is
//! immutable, it owns everything it needs, and it holds no per-stream state —
//! no history, no distance cache, no input position — so any number of
//! compressors may borrow one at the same time. There is no `Arc`, no lock and
//! no atomic inside it; a caller who wants shared ownership wraps it in an
//! `Arc` themselves, and that is their policy rather than this crate's.
//!
//! Preparing is the expensive half and compressing is the cheap half, which is
//! the whole reason the two are separate types.
//!
//! [RFC 9841]: https://www.rfc-editor.org/rfc/rfc9841.html

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::core::rfc9841::context::{Budget, SharedContextInner};
use super::shared::SharedBrotliError;

#[cfg(feature = "decompression")]
pub use crate::dictionary::{
    DecodeDictionary, DecodeDictionaryError, DecodeDictionaryLimits, DictionaryAttachment,
    DictionaryRef,
};
use thiserror::Error;

#[cfg(feature = "experimental")]
mod serialized;

#[cfg(feature = "experimental")]
pub use serialized::{
    CONTEXTS, ContextMap, DictionaryCombination, ListSelector, MAX_LIST_COUNT, MAX_STRINGLETS,
    MAX_TRANSFORMS, OmitLength, OmitLengthOutOfRange, SerializedDictionary,
    SerializedDictionaryBuilder, SerializedDictionaryError, TransformList, TransformListBuilder,
    TransformListError, TransformListView, TransformOperation, UndefinedTransformOperation,
    WordList, WordListBuilder, WordListError, WordListView,
};

/// Dictionaries one prepared dictionary may hold, as RFC 9841 fixes it.
const MAX_ATTACHMENTS: usize = 15;

/// An indexed set of prefix dictionaries, ready to compress against.
///
/// Built by [`DictionaryBuilder`]. Immutable and shareable: every method takes
/// `&self`, so one dictionary can back any number of compressors at once,
/// on any number of threads, with no synchronisation of this crate's making.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::DictionaryBuilder;
/// use mbrotli::{Compressor, EncoderConfig, Quality};
///
/// let dictionary = DictionaryBuilder::new()
///     .add_prefix(&b"HTTP/1.1 200 OK\r\nContent-Type: "[..])
///     .build()?;
///
/// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
/// let payload = b"Content-Type: text/html; charset=utf-8";
///
/// let with = encoder.compress_with_dictionary(&dictionary, payload)?;
/// let without = encoder.compress(payload)?;
/// assert!(with.len() < without.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Sharing one dictionary between workers needs no lock, because nothing in it
/// is mutable:
///
/// ```
/// use mbrotli::dictionary::DictionaryBuilder;
/// use mbrotli::{Compressor, EncoderConfig, Quality};
/// use std::sync::Arc;
///
/// let dictionary = Arc::new(
///     DictionaryBuilder::new()
///         .add_prefix(&b"a shared prefix, indexed once"[..])
///         .build()?,
/// );
///
/// let workers: Vec<_> = (0..4)
///     .map(|worker| {
///         let dictionary = Arc::clone(&dictionary);
///         std::thread::spawn(move || {
///             let config = EncoderConfig::default().with_quality(Quality::Q5);
///             let mut encoder = Compressor::new(config).expect("a legal configuration");
///             encoder
///                 .compress_with_dictionary(&dictionary, format!("worker {worker}").as_bytes())
///                 .expect("compression")
///         })
///     })
///     .collect();
///
/// for worker in workers {
///     assert!(!worker.join().expect("the worker finished").is_empty());
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct PreparedDictionary {
    /// The dictionary bytes and the indexes built over them.
    inner: SharedContextInner,
}

impl PreparedDictionary {
    /// Returns how many prefix dictionaries were attached.
    ///
    /// Zero for a dictionary containing only custom static words/transforms.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    ///
    /// let dictionary = DictionaryBuilder::new()
    ///     .add_prefix(&b"oldest"[..])
    ///     .add_prefix(&b"newest"[..])
    ///     .build()?;
    ///
    /// assert_eq!(dictionary.attachment_count(), 2);
    /// # Ok::<(), mbrotli::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub fn attachment_count(&self) -> usize {
        self.inner.dictionaries().prefix().segment_count()
    }

    /// Returns how many dictionary bytes the caller handed over.
    ///
    /// Includes prefix bytes and custom word/transform source bytes, but not
    /// serialized framing or built-in dictionary bytes. The decoder must attach
    /// the same dictionaries in the same order.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    ///
    /// let dictionary = DictionaryBuilder::new().add_prefix(&b"twelve bytes"[..]).build()?;
    ///
    /// assert_eq!(dictionary.source_bytes(), 12);
    /// # Ok::<(), mbrotli::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub fn source_bytes(&self) -> usize {
        let bytes = self.inner.dictionaries().source_size();
        #[cfg(feature = "experimental")]
        let bytes = bytes
            + self
                .inner
                .static_index
                .as_ref()
                .map_or(0, |index| index.source_bytes);
        bytes
    }

    /// Returns how much memory this dictionary occupies.
    ///
    /// Counts the dictionary bytes and the prepared indexes together. Reading
    /// it needs no synchronisation, because there is none to take.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    ///
    /// let dictionary = DictionaryBuilder::new()
    ///     .add_prefix(&b"long enough to be worth indexing"[..])
    ///     .build()?;
    ///
    /// assert!(dictionary.retained_bytes() > dictionary.source_bytes());
    /// # Ok::<(), mbrotli::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.inner.allocated_size()
    }

    /// Returns the backward distance that addresses `offset` in the prefix.
    ///
    /// RFC 9841 places the attached prefix immediately beyond the ordinary
    /// sliding window: distances `1..=max_backward` are the stream's own
    /// history, `max_backward + 1` is the *last* prefix byte, and
    /// `max_backward + prefix_length` is the very first. Custom static source
    /// bytes do not contribute to `prefix_length`. `max_backward` is the
    /// largest distance the window can express at the position the copy starts
    /// from.
    ///
    /// Returns `None` for an offset past the end of the prefix, and when the
    /// distance would not fit a `u64`. Nothing wraps.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    ///
    /// let dictionary = DictionaryBuilder::new()
    ///     .add_prefix(&b"oldest"[..])
    ///     .add_prefix(&b"newest"[..])
    ///     .build()?;
    ///
    /// // Twelve prefix bytes: the last is one past the window, the first twelve past.
    /// assert_eq!(dictionary.backward_distance(11, 1000), Some(1001));
    /// assert_eq!(dictionary.backward_distance(0, 1000), Some(1012));
    /// assert_eq!(dictionary.backward_distance(12, 1000), None);
    /// # Ok::<(), mbrotli::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub fn backward_distance(&self, offset: u64, max_backward: u64) -> Option<u64> {
        self.inner
            .dictionaries()
            .prefix()
            .distance_of(offset, max_backward)
    }

    /// Returns the prefix offset a backward `distance` addresses.
    ///
    /// The inverse of [`PreparedDictionary::backward_distance`], and the
    /// mapping a decoder performs. Returns `None` when the distance falls
    /// inside the ordinary sliding window or past the end of the prefix.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    ///
    /// let dictionary = DictionaryBuilder::new()
    ///     .add_prefix(&b"oldest"[..])
    ///     .add_prefix(&b"newest"[..])
    ///     .build()?;
    ///
    /// assert_eq!(dictionary.prefix_offset(1001, 1000), Some(11));
    /// assert_eq!(dictionary.prefix_offset(1012, 1000), Some(0));
    /// // Inside the window, and past the whole prefix.
    /// assert_eq!(dictionary.prefix_offset(1000, 1000), None);
    /// assert_eq!(dictionary.prefix_offset(1013, 1000), None);
    /// # Ok::<(), mbrotli::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub fn prefix_offset(&self, distance: u64, max_backward: u64) -> Option<u64> {
        self.inner
            .dictionaries()
            .prefix()
            .address_of(distance, max_backward)
    }

    /// Returns the longest match this dictionary offers at the start of `input`.
    ///
    /// A diagnostic, not part of the compression contract: it answers how well
    /// a candidate dictionary covers a corpus, which is the question worth
    /// asking before shipping one. The candidate order it resolves ties by is
    /// an implementation detail that may change; do not build application
    /// behaviour on it.
    ///
    /// Returns `None` when nothing matched, and when `input` is shorter than
    /// the eight bytes the prepared index is keyed on.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    ///
    /// let dictionary = DictionaryBuilder::new()
    ///     .add_prefix(&b"HTTP/1.1 200 OK\r\nContent-Type: "[..])
    ///     .build()?;
    ///
    /// let found = dictionary
    ///     .longest_match(b"Content-Type: text/html")
    ///     .expect("the header is in the dictionary");
    ///
    /// assert_eq!(found.length(), 14);
    /// assert!(dictionary.longest_match(b"nothing alike").is_none());
    /// # Ok::<(), mbrotli::dictionary::DictionaryError>(())
    /// ```
    #[cfg(feature = "diagnostics")]
    #[must_use]
    pub fn longest_match(&self, input: &[u8]) -> Option<PrefixMatch> {
        self.inner
            .longest_prefix_match(input)
            .map(PrefixMatch::from)
    }

    /// Returns the indexes a match finder consults.
    pub(crate) const fn inner(&self) -> &SharedContextInner {
        &self.inner
    }
}

impl AsRef<Self> for PreparedDictionary {
    /// Borrows the dictionary itself, so an owned dictionary, a `&'static`
    /// one and an `Arc` of one all satisfy `AsRef<PreparedDictionary>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use mbrotli::dictionary::{DictionaryBuilder, PreparedDictionary};
    ///
    /// let shared = Arc::new(DictionaryBuilder::new().add_prefix(&b"prefix"[..]).build()?);
    /// let view: &PreparedDictionary = shared.as_ref().as_ref();
    /// assert!(std::ptr::eq(view, &*shared));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn as_ref(&self) -> &Self {
        self
    }
}

/// Where a dictionary matched an input, and for how long.
///
/// Returned by [`PreparedDictionary::longest_match`]. The offset is into the
/// *logical* prefix — every attachment laid end to end in attachment order —
/// not into any one attachment, because a match may run from one attachment
/// into the next.
#[cfg(feature = "diagnostics")]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PrefixMatch {
    /// Where the match starts in the logical prefix.
    offset: u64,
    /// How many bytes matched.
    length: usize,
}

#[cfg(feature = "diagnostics")]
impl PrefixMatch {
    /// Returns where the match starts in the logical prefix.
    ///
    /// Zero is the first byte of the first attachment, which is the oldest byte
    /// a backward distance can reach.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    ///
    /// let dictionary = DictionaryBuilder::new()
    ///     .add_prefix(&b"the quick brown fox"[..])
    ///     .build()?;
    ///
    /// let found = dictionary.longest_match(b"quick brown foxes").expect("a match");
    /// assert_eq!(found.prefix_offset(), 4);
    /// # Ok::<(), mbrotli::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub const fn prefix_offset(self) -> u64 {
        self.offset
    }

    /// Returns how many bytes matched.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    ///
    /// let dictionary = DictionaryBuilder::new()
    ///     .add_prefix(&b"the quick brown fox"[..])
    ///     .build()?;
    ///
    /// let found = dictionary.longest_match(b"quick brown foxes").expect("a match");
    /// assert_eq!(found.length(), 15);
    /// # Ok::<(), mbrotli::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub const fn length(self) -> usize {
        self.length
    }
}

#[cfg(feature = "diagnostics")]
impl From<super::core::rfc9841::context::PrefixMatch> for PrefixMatch {
    /// Lifts the private search result into the public one.
    fn from(value: super::core::rfc9841::context::PrefixMatch) -> Self {
        Self {
            offset: value.offset,
            length: value.length,
        }
    }
}

/// How much memory preparing a dictionary may spend.
///
/// These are implementation resource limits, not wire-format limits: they
/// change which dictionaries this crate agrees to build, never what a stream
/// built with one looks like. They exist because dictionary bytes usually
/// arrive from somewhere less trusted than the code that compresses with them,
/// and a prepared index is several times the size of what it indexes.
///
/// The defaults are sized for ordinary production dictionaries — a few
/// megabytes of prefix — with a wide margin. Raise them deliberately; a caller
/// who has already validated the bytes is the only one who knows it is safe to.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::DictionaryLimits;
///
/// let limits = DictionaryLimits::default().with_max_prefix_bytes(1 << 20);
///
/// assert_eq!(limits.max_prefix_bytes(), 1 << 20);
/// assert_eq!(
///     limits.max_retained_bytes(),
///     DictionaryLimits::default().max_retained_bytes()
/// );
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DictionaryLimits {
    /// Largest total of every attached source byte.
    max_source_bytes: u64,
    /// Largest logical LZ77 prefix.
    max_prefix_bytes: u64,
    /// Largest peak allocation preparing the dictionary may reach.
    max_retained_bytes: u64,
    /// Most prefix dictionaries one prepared dictionary may hold.
    max_attachments: usize,
    /// Largest serialized dictionary stream that may be parsed.
    #[cfg(feature = "experimental")]
    max_serialized_bytes: u64,
    /// Largest total of every custom word list's word bytes.
    #[cfg(feature = "experimental")]
    max_word_bytes: u64,
    /// Most custom word lists.
    #[cfg(feature = "experimental")]
    max_word_lists: usize,
    /// Largest total of every custom transform list's wire bytes.
    #[cfg(feature = "experimental")]
    max_transform_bytes: u64,
    /// Most custom transform lists.
    #[cfg(feature = "experimental")]
    max_transform_lists: usize,
    /// Most word-and-transform-list combinations.
    #[cfg(feature = "experimental")]
    max_combinations: usize,
    #[cfg(feature = "experimental")]
    max_transformed_word_bytes: u64,
    #[cfg(feature = "experimental")]
    max_static_entries: u64,
}

impl DictionaryLimits {
    /// Sets the total transformed-word storage limit (default 128 MiB).
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn with_max_transformed_word_bytes(mut self, bytes: u64) -> Self {
        self.max_transformed_word_bytes = bytes;
        self
    }

    /// Returns the total transformed-word storage limit.
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn max_transformed_word_bytes(self) -> u64 {
        self.max_transformed_word_bytes
    }

    /// Sets the number of entries in the static search index (default 8 million).
    /// This bounds the flat index used instead of a trie.
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn with_max_static_entries(mut self, count: u64) -> Self {
        self.max_static_entries = count;
        self
    }

    /// Returns the maximum number of static search entries.
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn max_static_entries(self) -> u64 {
        self.max_static_entries
    }
    /// Default ceiling on the attached source bytes: 64 MiB.
    ///
    /// Far above any ordinary production dictionary, and at the size where the
    /// prepared index stops scaling its bucket count, so a larger dictionary
    /// costs proportionally more to index than it repays.
    pub const DEFAULT_MAX_SOURCE_BYTES: u64 = 64 << 20;

    /// Default ceiling on the logical LZ77 prefix: 64 MiB.
    pub const DEFAULT_MAX_PREFIX_BYTES: u64 = 64 << 20;

    /// Default ceiling on the peak allocation of preparing one: 1 GiB.
    ///
    /// Preparation costs roughly eight bytes per source byte at its peak — a
    /// four-byte chain link and a four-byte index entry for every position —
    /// plus the bucket tables. A dictionary at
    /// [`DictionaryLimits::DEFAULT_MAX_PREFIX_BYTES`] therefore fits this with
    /// room to spare, and the two defaults do not contradict each other.
    pub const DEFAULT_MAX_RETAINED_BYTES: u64 = 1 << 30;

    /// Default ceiling on the prefix attachment count: fifteen.
    ///
    /// Also the format's own ceiling, so this default refuses nothing RFC 9841
    /// allows; lowering it is how a caller refuses more than they expect.
    pub const DEFAULT_MAX_ATTACHMENTS: usize = MAX_ATTACHMENTS;

    /// Default ceiling on a serialized dictionary stream: 128 MiB.
    ///
    /// Checked before a single byte inside the stream is read, so it bounds the
    /// whole parse rather than any one field. Comfortably above a maximal
    /// prefix plus a maximal set of custom lists.
    #[cfg(feature = "experimental")]
    pub const DEFAULT_MAX_SERIALIZED_BYTES: u64 = 128 << 20;

    /// Default ceiling on the total custom word bytes: 16 MiB.
    ///
    /// A single word list holds at most `31 << 15` bytes, just under a
    /// megabyte, so this leaves room for sixteen maximal lists. A dictionary
    /// that wants all sixty-four has to say so.
    #[cfg(feature = "experimental")]
    pub const DEFAULT_MAX_WORD_BYTES: u64 = 16 << 20;

    /// Default ceiling on the custom word list count: sixty-four, the format's.
    #[cfg(feature = "experimental")]
    pub const DEFAULT_MAX_WORD_LISTS: usize = MAX_LIST_COUNT;

    /// Default ceiling on the total custom transform bytes: 8 MiB.
    ///
    /// A single transform list occupies at most about sixty-six kilobytes, so
    /// this leaves room for every list the format allows.
    #[cfg(feature = "experimental")]
    pub const DEFAULT_MAX_TRANSFORM_BYTES: u64 = 8 << 20;

    /// Default ceiling on the custom transform list count: sixty-four.
    #[cfg(feature = "experimental")]
    pub const DEFAULT_MAX_TRANSFORM_LISTS: usize = MAX_LIST_COUNT;

    /// Default ceiling on the combination count: sixty-four, the format's.
    #[cfg(feature = "experimental")]
    pub const DEFAULT_MAX_COMBINATIONS: usize = MAX_LIST_COUNT;

    /// Sets the largest total of attached source bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(
    ///     DictionaryLimits::default().with_max_source_bytes(4096).max_source_bytes(),
    ///     4096
    /// );
    /// ```
    #[must_use]
    pub const fn with_max_source_bytes(mut self, bytes: u64) -> Self {
        self.max_source_bytes = bytes;
        self
    }

    /// Returns the largest total of attached source bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(
    ///     DictionaryLimits::default().max_source_bytes(),
    ///     DictionaryLimits::DEFAULT_MAX_SOURCE_BYTES
    /// );
    /// ```
    #[must_use]
    pub const fn max_source_bytes(self) -> u64 {
        self.max_source_bytes
    }

    /// Sets the largest logical LZ77 prefix.
    ///
    /// The prefix is every attachment laid end to end, which is what a backward
    /// distance past the sliding window addresses.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(
    ///     DictionaryLimits::default().with_max_prefix_bytes(4096).max_prefix_bytes(),
    ///     4096
    /// );
    /// ```
    #[must_use]
    pub const fn with_max_prefix_bytes(mut self, bytes: u64) -> Self {
        self.max_prefix_bytes = bytes;
        self
    }

    /// Returns the largest logical LZ77 prefix.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(
    ///     DictionaryLimits::default().max_prefix_bytes(),
    ///     DictionaryLimits::DEFAULT_MAX_PREFIX_BYTES
    /// );
    /// ```
    #[must_use]
    pub const fn max_prefix_bytes(self) -> u64 {
        self.max_prefix_bytes
    }

    /// Sets the largest allocation preparing a dictionary may reach.
    ///
    /// Bounds the *peak*, not just the finished dictionary: the build holds its
    /// scratch tables and the finished ones at once. It is checked against an
    /// upper bound computed before anything is allocated, so a dictionary that
    /// would exceed it is refused rather than allocated and discarded.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(
    ///     DictionaryLimits::default().with_max_retained_bytes(1 << 20).max_retained_bytes(),
    ///     1 << 20
    /// );
    /// ```
    #[must_use]
    pub const fn with_max_retained_bytes(mut self, bytes: u64) -> Self {
        self.max_retained_bytes = bytes;
        self
    }

    /// Returns the largest allocation preparing a dictionary may reach.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(
    ///     DictionaryLimits::default().max_retained_bytes(),
    ///     DictionaryLimits::DEFAULT_MAX_RETAINED_BYTES
    /// );
    /// ```
    #[must_use]
    pub const fn max_retained_bytes(self) -> u64 {
        self.max_retained_bytes
    }

    /// Sets the largest number of prefix dictionaries one dictionary may hold.
    ///
    /// Never raises the ceiling past the format's own fifteen; a larger value
    /// is silently the same as fifteen.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::{DictionaryBuilder, DictionaryError, DictionaryLimits};
    ///
    /// let outcome = DictionaryBuilder::new()
    ///     .add_prefix(&b"one"[..])
    ///     .add_prefix(&b"two"[..])
    ///     .with_limits(DictionaryLimits::default().with_max_attachments(1))
    ///     .build();
    ///
    /// assert!(matches!(
    ///     outcome,
    ///     Err(DictionaryError::TooManyAttachments { attached: 2, limit: 1 })
    /// ));
    /// ```
    #[must_use]
    pub const fn with_max_attachments(mut self, attachments: usize) -> Self {
        self.max_attachments = attachments;
        self
    }

    /// Returns the largest number of prefix dictionaries one may hold.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(DictionaryLimits::default().max_attachments(), 15);
    /// ```
    #[must_use]
    pub const fn max_attachments(self) -> usize {
        self.max_attachments
    }

    /// Sets the largest serialized dictionary stream that may be parsed.
    ///
    /// Checked before any field inside the stream is read.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// let limits = DictionaryLimits::default().with_max_serialized_bytes(4096);
    ///
    /// assert_eq!(limits.max_serialized_bytes(), 4096);
    /// ```
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn with_max_serialized_bytes(mut self, bytes: u64) -> Self {
        self.max_serialized_bytes = bytes;
        self
    }

    /// Returns the largest serialized dictionary stream that may be parsed.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(
    ///     DictionaryLimits::default().max_serialized_bytes(),
    ///     DictionaryLimits::DEFAULT_MAX_SERIALIZED_BYTES
    /// );
    /// ```
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn max_serialized_bytes(self) -> u64 {
        self.max_serialized_bytes
    }

    /// Sets the largest total of every custom word list's word bytes.
    ///
    /// Checked cumulatively as the lists are parsed, so a stream is refused at
    /// the list that crosses the ceiling rather than after all of them.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(DictionaryLimits::default().with_max_word_bytes(64).max_word_bytes(), 64);
    /// ```
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn with_max_word_bytes(mut self, bytes: u64) -> Self {
        self.max_word_bytes = bytes;
        self
    }

    /// Returns the largest total of every custom word list's word bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(
    ///     DictionaryLimits::default().max_word_bytes(),
    ///     DictionaryLimits::DEFAULT_MAX_WORD_BYTES
    /// );
    /// ```
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn max_word_bytes(self) -> u64 {
        self.max_word_bytes
    }

    /// Sets the largest number of custom word lists.
    ///
    /// Never raises the ceiling past the format's own sixty-four.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(DictionaryLimits::default().with_max_word_lists(2).max_word_lists(), 2);
    /// ```
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn with_max_word_lists(mut self, lists: usize) -> Self {
        self.max_word_lists = lists;
        self
    }

    /// Returns the largest number of custom word lists.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(DictionaryLimits::default().max_word_lists(), 64);
    /// ```
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn max_word_lists(self) -> usize {
        self.max_word_lists
    }

    /// Sets the largest total of every custom transform list's wire bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// let limits = DictionaryLimits::default().with_max_transform_bytes(1024);
    ///
    /// assert_eq!(limits.max_transform_bytes(), 1024);
    /// ```
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn with_max_transform_bytes(mut self, bytes: u64) -> Self {
        self.max_transform_bytes = bytes;
        self
    }

    /// Returns the largest total of every custom transform list's wire bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(
    ///     DictionaryLimits::default().max_transform_bytes(),
    ///     DictionaryLimits::DEFAULT_MAX_TRANSFORM_BYTES
    /// );
    /// ```
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn max_transform_bytes(self) -> u64 {
        self.max_transform_bytes
    }

    /// Sets the largest number of custom transform lists.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// let limits = DictionaryLimits::default().with_max_transform_lists(1);
    ///
    /// assert_eq!(limits.max_transform_lists(), 1);
    /// ```
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn with_max_transform_lists(mut self, lists: usize) -> Self {
        self.max_transform_lists = lists;
        self
    }

    /// Returns the largest number of custom transform lists.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(DictionaryLimits::default().max_transform_lists(), 64);
    /// ```
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn max_transform_lists(self) -> usize {
        self.max_transform_lists
    }

    /// Sets the largest number of word-and-transform-list combinations.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(DictionaryLimits::default().with_max_combinations(4).max_combinations(), 4);
    /// ```
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn with_max_combinations(mut self, combinations: usize) -> Self {
        self.max_combinations = combinations;
        self
    }

    /// Returns the largest number of word-and-transform-list combinations.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// assert_eq!(DictionaryLimits::default().max_combinations(), 64);
    /// ```
    #[cfg(feature = "experimental")]
    #[must_use]
    pub const fn max_combinations(self) -> usize {
        self.max_combinations
    }
}

impl Default for DictionaryLimits {
    /// Returns the documented production defaults.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryLimits;
    ///
    /// let limits = DictionaryLimits::default();
    ///
    /// assert_eq!(limits.max_prefix_bytes(), 64 << 20);
    /// assert_eq!(limits.max_retained_bytes(), 1 << 30);
    /// ```
    fn default() -> Self {
        Self {
            max_source_bytes: Self::DEFAULT_MAX_SOURCE_BYTES,
            max_prefix_bytes: Self::DEFAULT_MAX_PREFIX_BYTES,
            max_retained_bytes: Self::DEFAULT_MAX_RETAINED_BYTES,
            max_attachments: Self::DEFAULT_MAX_ATTACHMENTS,
            #[cfg(feature = "experimental")]
            max_serialized_bytes: Self::DEFAULT_MAX_SERIALIZED_BYTES,
            #[cfg(feature = "experimental")]
            max_word_bytes: Self::DEFAULT_MAX_WORD_BYTES,
            #[cfg(feature = "experimental")]
            max_word_lists: Self::DEFAULT_MAX_WORD_LISTS,
            #[cfg(feature = "experimental")]
            max_transform_bytes: Self::DEFAULT_MAX_TRANSFORM_BYTES,
            #[cfg(feature = "experimental")]
            max_transform_lists: Self::DEFAULT_MAX_TRANSFORM_LISTS,
            #[cfg(feature = "experimental")]
            max_combinations: Self::DEFAULT_MAX_COMBINATIONS,
            #[cfg(feature = "experimental")]
            max_transformed_word_bytes: 128 << 20,
            #[cfg(feature = "experimental")]
            max_static_entries: 8_000_000,
        }
    }
}

impl From<DictionaryLimits> for Budget {
    /// Flattens the public limits into the form the checks compare against.
    fn from(value: DictionaryLimits) -> Self {
        Self {
            max_total_source_bytes: value.max_source_bytes,
            max_prefix_bytes: value.max_prefix_bytes,
            max_allocated_bytes: value.max_retained_bytes,
            max_attachments: value.max_attachments,
        }
    }
}

/// Collects prefix dictionaries in attachment order and indexes them all at once.
///
/// Call order is prefix order: the first dictionary attached holds the oldest
/// bytes, and the last one the bytes immediately before the stream's own
/// output. A decoder has to attach exactly the same bytes in exactly the same
/// order.
///
/// Nothing is validated or indexed until [`DictionaryBuilder::build`], which is
/// all-or-nothing: it returns a whole dictionary or an error, never a partly
/// usable one.
///
/// Building does not ask for a quality. The indexes a dictionary carries are the
/// same whichever quality later reads them, so one prepared dictionary serves
/// every quality that can consult one.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::{DictionaryBuilder, DictionaryLimits};
///
/// let dictionary = DictionaryBuilder::new()
///     .add_prefix(&b"oldest bytes"[..])
///     .add_prefix(&b"newest bytes"[..])
///     .with_limits(DictionaryLimits::default().with_max_prefix_bytes(1 << 20))
///     .build()?;
///
/// assert_eq!(dictionary.attachment_count(), 2);
/// # Ok::<(), mbrotli::dictionary::DictionaryError>(())
/// ```
#[derive(Debug, Default)]
pub struct DictionaryBuilder {
    /// The limits [`DictionaryBuilder::build`] checks against.
    limits: DictionaryLimits,
    /// Owned dictionary bytes, oldest first.
    attachments: Vec<Box<[u8]>>,
    /// The custom static dictionary, if a serialized one supplied it.
    #[cfg(feature = "experimental")]
    custom_static: Option<crate::compressor::core::rfc9841::serialized::SerializedDictionaryData>,
}

impl DictionaryBuilder {
    /// Creates a builder with nothing attached and the default limits.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::{DictionaryBuilder, DictionaryError};
    ///
    /// // A dictionary with no bytes in it is refused rather than built.
    /// assert!(matches!(DictionaryBuilder::new().build(), Err(DictionaryError::Empty)));
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Attaches one LZ77 prefix dictionary after the ones already attached.
    ///
    /// The bytes are moved into the builder and then into the dictionary: no
    /// reference counting, no borrow of the caller's buffer to keep alive.
    /// Passing a `Vec<u8>` or a `Box<[u8]>` moves it without copying; passing a
    /// `&[u8]` copies it once.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    ///
    /// let dictionary = DictionaryBuilder::new()
    ///     .add_prefix(b"moved without copying".to_vec())
    ///     .add_prefix(&b"copied once"[..])
    ///     .build()?;
    ///
    /// assert_eq!(dictionary.attachment_count(), 2);
    /// # Ok::<(), mbrotli::dictionary::DictionaryError>(())
    /// ```
    #[must_use]
    pub fn add_prefix<B>(mut self, bytes: B) -> Self
    where
        B: Into<Box<[u8]>>,
    {
        self.attachments.push(bytes.into());
        self
    }

    /// Attaches everything an RFC 9841 serialized dictionary describes.
    ///
    /// **Experimental**, behind the `experimental` feature.
    ///
    /// The dictionary's LZ77 prefix becomes one attachment, in the position it
    /// is added in, exactly as
    /// [`DictionaryBuilder::add_prefix`] would place it. Its custom word and
    /// transform lists replace the RFC 7932 static dictionary for every stream
    /// compressed against the result.
    ///
    /// The description is borrowed and copied out of, so one parsed
    /// [`SerializedDictionary`] can back any number of prepared dictionaries.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::{DictionaryBuilder, SerializedDictionary};
    ///
    /// let described = SerializedDictionary::builder()
    ///     .with_prefix(&b"Content-Type: text/html"[..])
    ///     .build()?;
    ///
    /// let dictionary = DictionaryBuilder::new().add_serialized(&described).build()?;
    ///
    /// assert_eq!(dictionary.attachment_count(), 1);
    /// assert_eq!(dictionary.source_bytes(), 23);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg(feature = "experimental")]
    #[must_use]
    pub fn add_serialized(mut self, dictionary: &SerializedDictionary) -> Self {
        if dictionary.data().has_prefix() {
            self.attachments.push(Box::from(dictionary.prefix()));
        }
        if dictionary.is_custom_static() {
            self.custom_static = Some(dictionary.data().clone());
        }
        self
    }

    /// Replaces the resource limits the dictionary is built under.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::{DictionaryBuilder, DictionaryError, DictionaryLimits};
    ///
    /// let outcome = DictionaryBuilder::new()
    ///     .add_prefix(&b"far too long for this limit"[..])
    ///     .with_limits(DictionaryLimits::default().with_max_prefix_bytes(8))
    ///     .build();
    ///
    /// assert!(matches!(outcome, Err(DictionaryError::TooLarge { limit: 8, .. })));
    /// ```
    #[must_use]
    pub const fn with_limits(mut self, limits: DictionaryLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Validates and indexes every attachment, producing the dictionary.
    ///
    /// This is where the expensive work happens: the counts and sizes are
    /// checked, the logical prefix is laid out, and one hash index is built per
    /// attachment. Compressing afterwards reuses all of it.
    ///
    /// # Errors
    ///
    /// Returns [`DictionaryError::Empty`] when nothing was attached or every
    /// attachment was empty, [`DictionaryError::TooManyAttachments`] past
    /// fifteen attachments, [`DictionaryError::TooLarge`] when an attachment or
    /// the whole prefix exceeds its limit, and
    /// [`DictionaryError::PreparationTooLarge`] when the indexes would exceed
    /// the allocation limit. Nothing is retained on failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    ///
    /// let dictionary = DictionaryBuilder::new().add_prefix(&b"a prefix"[..]).build()?;
    ///
    /// assert_eq!(dictionary.attachment_count(), 1);
    /// assert_eq!(dictionary.source_bytes(), 8);
    /// # Ok::<(), mbrotli::dictionary::DictionaryError>(())
    /// ```
    pub fn build(self) -> Result<PreparedDictionary, DictionaryError> {
        #[cfg(feature = "experimental")]
        let has_static = self.custom_static.is_some();
        #[cfg(not(feature = "experimental"))]
        let has_static = false;
        if !has_static && self.attachments.iter().all(|bytes| bytes.is_empty()) {
            return Err(DictionaryError::Empty);
        }
        #[cfg(feature = "experimental")]
        if let Some(data) = &self.custom_static {
            let word_bytes = data
                .word_lists()
                .iter()
                .map(|w| w.data().len() as u64)
                .sum::<u64>();
            let transform_bytes = data
                .transform_lists()
                .iter()
                .map(|t| t.wire_len() as u64)
                .sum::<u64>();
            for (what, found, limit) in [
                (
                    "serialized bytes",
                    data.wire_len() as u64,
                    self.limits.max_serialized_bytes,
                ),
                (
                    "word lists",
                    data.word_lists().len() as u64,
                    self.limits.max_word_lists as u64,
                ),
                ("word bytes", word_bytes, self.limits.max_word_bytes),
                (
                    "transform lists",
                    data.transform_lists().len() as u64,
                    self.limits.max_transform_lists as u64,
                ),
                (
                    "transform bytes",
                    transform_bytes,
                    self.limits.max_transform_bytes,
                ),
                (
                    "combinations",
                    data.combinations().len() as u64,
                    self.limits.max_combinations as u64,
                ),
            ] {
                if found > limit {
                    return Err(DictionaryError::LimitExceeded { what, found, limit });
                }
            }
            let source = self.attachments.iter().map(|p| p.len() as u64).sum::<u64>()
                + word_bytes
                + transform_bytes;
            if source > self.limits.max_source_bytes {
                return Err(DictionaryError::TooLarge {
                    bytes: source,
                    limit: self.limits.max_source_bytes,
                });
            }
        }
        let budget = Budget::from(self.limits);
        #[cfg(feature = "experimental")]
        let description_bytes = self
            .custom_static
            .as_ref()
            .map_or(0, |d| d.allocation_bound()) as u64;
        #[cfg(feature = "experimental")]
        let construction_bytes = description_bytes
            .saturating_add((self.attachments.capacity() * size_of::<Box<[u8]>>()) as u64);
        #[cfg(feature = "experimental")]
        let budget = Budget {
            max_allocated_bytes: budget
                .max_allocated_bytes
                .saturating_sub(construction_bytes),
            ..budget
        };
        let inner = SharedContextInner::new(self.attachments, &budget).map_err(|error| {
            #[cfg(feature = "experimental")]
            let error = match error {
                SharedBrotliError::SharedContextTooLarge { bytes, .. } => {
                    SharedBrotliError::SharedContextTooLarge {
                        bytes: bytes.saturating_add(construction_bytes),
                        limit: self.limits.max_retained_bytes,
                    }
                }
                other => other,
            };
            DictionaryError::from_core(error)
        })?;
        #[cfg(feature = "experimental")]
        let inner = {
            let mut inner = inner;
            if let Some(data) = self.custom_static {
                inner.static_index = Some(
                    super::core::rfc9841::static_index::StaticIndex::prepare(
                        &data,
                        self.limits
                            .max_retained_bytes
                            .saturating_sub(inner.allocated_size() as u64)
                            .saturating_sub(description_bytes),
                        self.limits.max_transformed_word_bytes,
                        self.limits.max_static_entries,
                    )
                    .map_err(DictionaryError::from_core)?,
                );
            }
            inner
        };
        Ok(PreparedDictionary { inner })
    }
}

/// Error returned when a dictionary cannot be prepared.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::{DictionaryBuilder, DictionaryError};
///
/// let mut builder = DictionaryBuilder::new();
/// for _ in 0..16 {
///     builder = builder.add_prefix(&b"payload"[..]);
/// }
///
/// assert!(matches!(
///     builder.build(),
///     Err(DictionaryError::TooManyAttachments { attached: 16, limit: 15 })
/// ));
/// ```
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum DictionaryError {
    /// An experimental serialized-description preparation limit was exceeded.
    #[cfg(feature = "experimental")]
    #[error("dictionary {what} of {found} exceeds the limit of {limit}")]
    LimitExceeded {
        /// The bounded resource.
        what: &'static str,
        /// The description's requirement.
        found: u64,
        /// The caller's ceiling.
        limit: u64,
    },
    /// Nothing was attached, or every attachment was empty.
    ///
    /// A dictionary with no bytes in it can shorten no stream, so it is refused
    /// rather than built: compressing against one would be indistinguishable
    /// from compressing without it, which is exactly the confusion this crate
    /// avoids elsewhere by refusing a dictionary a quality cannot use.
    #[error("a dictionary with no bytes in it cannot shorten a stream")]
    Empty,
    /// More than fifteen prefix dictionaries were attached.
    ///
    /// RFC 9841 gives a distance no way to say which of a sixteenth
    /// dictionary's bytes it meant, so the limit is the format's, not this
    /// implementation's.
    #[error("a dictionary holds at most {limit} attachments, not {attached}")]
    TooManyAttachments {
        /// How many attachments the builder was given.
        attached: usize,
        /// How many it may hold.
        limit: usize,
    },
    /// An attachment, or the whole logical prefix, is larger than allowed.
    #[error("{bytes} dictionary bytes exceed the limit of {limit}")]
    TooLarge {
        /// How many bytes were offered.
        bytes: u64,
        /// How many were allowed.
        limit: u64,
    },
    /// Indexing the dictionary would allocate more than the limit allows.
    ///
    /// Reported from an upper bound computed before anything is built, so the
    /// allocation the limit refused is never actually made.
    #[error("preparing a dictionary would allocate {bytes} bytes, past the limit of {limit}")]
    PreparationTooLarge {
        /// The predicted allocation.
        bytes: u64,
        /// How many bytes were allowed.
        limit: u64,
    },
}

impl DictionaryError {
    /// Lifts the low-level preparation error into the public one.
    const fn from_core(error: SharedBrotliError) -> Self {
        match error {
            // The limit the core reports is the effective one: the caller's
            // ceiling, or the format's fifteen when the caller asked for more.
            SharedBrotliError::TooManyPrefixDictionaries { attached, limit } => {
                Self::TooManyAttachments { attached, limit }
            }
            SharedBrotliError::DictionaryTooLarge { bytes, limit } => {
                Self::TooLarge { bytes, limit }
            }
            SharedBrotliError::SharedContextTooLarge { bytes, limit } => {
                Self::PreparationTooLarge { bytes, limit }
            }
            // Preparation raises no other variant: the rest belong to an
            // encoding operation, which has not started here.
            SharedBrotliError::UnsupportedLargeWindow { .. } => Self::Empty,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn the_attachment_limit_is_configurable_and_never_above_the_format_one() {
        let limits = DictionaryLimits::default();

        assert_eq!(limits.max_attachments(), MAX_ATTACHMENTS);
        assert_eq!(limits.with_max_attachments(2).max_attachments(), 2);
        assert_eq!(
            limits.max_source_bytes(),
            DictionaryLimits::DEFAULT_MAX_SOURCE_BYTES
        );

        let outcome = DictionaryBuilder::new()
            .add_prefix(&b"one"[..])
            .add_prefix(&b"two"[..])
            .with_limits(limits.with_max_attachments(1))
            .build();

        assert!(matches!(
            outcome,
            Err(DictionaryError::TooManyAttachments {
                attached: 2,
                limit: 1
            })
        ));

        // Raising it past the format's own ceiling changes nothing.
        let mut builder = DictionaryBuilder::new().with_limits(limits.with_max_attachments(999));
        for _ in 0..=MAX_ATTACHMENTS {
            builder = builder.add_prefix(&b"payload"[..]);
        }
        assert!(matches!(
            builder.build(),
            Err(DictionaryError::TooManyAttachments {
                limit: MAX_ATTACHMENTS,
                ..
            })
        ));
    }

    #[test]
    fn a_prepared_dictionary_is_send_and_sync() {
        const fn assert_send<T: Send>() {}
        const fn assert_sync<T: Sync>() {}
        assert_send::<PreparedDictionary>();
        assert_sync::<PreparedDictionary>();
        assert_send::<DictionaryBuilder>();
    }

    #[test]
    fn an_empty_builder_is_refused_rather_than_built() {
        assert_eq!(
            DictionaryBuilder::new().build().unwrap_err(),
            DictionaryError::Empty
        );
        assert_eq!(
            DictionaryBuilder::new()
                .add_prefix(&b""[..])
                .add_prefix(&b""[..])
                .build()
                .unwrap_err(),
            DictionaryError::Empty
        );
    }

    #[test]
    fn attachment_order_is_call_order() {
        let dictionary = DictionaryBuilder::new()
            .add_prefix(&b"oldest"[..])
            .add_prefix(&b"middle"[..])
            .add_prefix(&b"newest"[..])
            .build()
            .expect("prepared");
        let prefix = dictionary.inner().dictionaries().prefix();
        assert_eq!(prefix.segment(0), b"oldest");
        assert_eq!(prefix.segment(1), b"middle");
        assert_eq!(prefix.segment(2), b"newest");
        assert_eq!(dictionary.attachment_count(), 3);
        assert_eq!(dictionary.source_bytes(), 18);
    }

    #[test]
    fn every_owned_byte_form_is_accepted() {
        let boxed: Box<[u8]> = b"boxed".to_vec().into_boxed_slice();
        let dictionary = DictionaryBuilder::new()
            .add_prefix(b"vector".to_vec())
            .add_prefix(boxed)
            .add_prefix(&b"borrowed"[..])
            .build()
            .expect("prepared");
        assert_eq!(dictionary.attachment_count(), 3);
        assert_eq!(dictionary.source_bytes(), 6 + 5 + 8);
    }

    #[test]
    fn the_format_limit_on_attachments_is_enforced() {
        let mut builder = DictionaryBuilder::new();
        for _ in 0..MAX_ATTACHMENTS {
            builder = builder.add_prefix(&b"payload"[..]);
        }
        assert_eq!(
            builder
                .build()
                .expect("fifteen is legal")
                .attachment_count(),
            MAX_ATTACHMENTS
        );

        let mut builder = DictionaryBuilder::new();
        for _ in 0..=MAX_ATTACHMENTS {
            builder = builder.add_prefix(&b"payload"[..]);
        }
        assert_eq!(
            builder.build().unwrap_err(),
            DictionaryError::TooManyAttachments {
                attached: 16,
                limit: 15
            }
        );
    }

    #[test]
    fn every_limit_is_checked_before_anything_is_built() {
        let too_long = DictionaryBuilder::new()
            .add_prefix(&b"nine byte"[..])
            .with_limits(DictionaryLimits::default().with_max_prefix_bytes(8))
            .build();
        assert_eq!(
            too_long.unwrap_err(),
            DictionaryError::TooLarge { bytes: 9, limit: 8 }
        );

        let too_much_source = DictionaryBuilder::new()
            .add_prefix(&b"four"[..])
            .add_prefix(&b"five!"[..])
            .with_limits(DictionaryLimits::default().with_max_source_bytes(8))
            .build();
        assert_eq!(
            too_much_source.unwrap_err(),
            DictionaryError::TooLarge { bytes: 9, limit: 8 }
        );

        let too_much_index = DictionaryBuilder::new()
            .add_prefix(&b"eight!!!"[..])
            .with_limits(DictionaryLimits::default().with_max_retained_bytes(1024))
            .build();
        assert!(matches!(
            too_much_index.unwrap_err(),
            DictionaryError::PreparationTooLarge { limit: 1024, .. }
        ));
    }

    #[test]
    fn a_refusal_leaves_nothing_behind() {
        // The same shape without the limit still prepares, so no state outside
        // the failed builder was disturbed.
        assert!(
            DictionaryBuilder::new()
                .add_prefix(&b"nine byte"[..])
                .with_limits(DictionaryLimits::default().with_max_prefix_bytes(8))
                .build()
                .is_err()
        );
        assert!(
            DictionaryBuilder::new()
                .add_prefix(&b"nine byte"[..])
                .build()
                .is_ok()
        );
    }

    #[test]
    fn the_limits_expose_their_documented_defaults() {
        let limits = DictionaryLimits::default();
        assert_eq!(
            limits.max_source_bytes(),
            DictionaryLimits::DEFAULT_MAX_SOURCE_BYTES
        );
        assert_eq!(
            limits.max_prefix_bytes(),
            DictionaryLimits::DEFAULT_MAX_PREFIX_BYTES
        );
        assert_eq!(
            limits.max_retained_bytes(),
            DictionaryLimits::DEFAULT_MAX_RETAINED_BYTES
        );

        let tightened = limits
            .with_max_source_bytes(1)
            .with_max_prefix_bytes(2)
            .with_max_retained_bytes(3);
        assert_eq!(tightened.max_source_bytes(), 1);
        assert_eq!(tightened.max_prefix_bytes(), 2);
        assert_eq!(tightened.max_retained_bytes(), 3);
        assert_ne!(tightened, limits);

        let budget = Budget::from(tightened);
        assert_eq!(budget.max_total_source_bytes, 1);
        assert_eq!(budget.max_prefix_bytes, 2);
        assert_eq!(budget.max_allocated_bytes, 3);
    }

    #[test]
    fn addressing_and_its_inverse_agree() {
        let dictionary = DictionaryBuilder::new()
            .add_prefix(&b"oldest"[..])
            .add_prefix(&b"newest"[..])
            .build()
            .expect("prepared");
        let max_backward = 1u64 << 20;

        for offset in 0..12u64 {
            let distance = dictionary
                .backward_distance(offset, max_backward)
                .expect("inside the prefix");
            assert!(distance > max_backward);
            assert_eq!(
                dictionary.prefix_offset(distance, max_backward),
                Some(offset)
            );
        }
        assert_eq!(
            dictionary.backward_distance(11, max_backward),
            Some(max_backward + 1)
        );
        assert_eq!(
            dictionary.backward_distance(0, max_backward),
            Some(max_backward + 12)
        );
        assert_eq!(dictionary.backward_distance(12, max_backward), None);
        assert_eq!(dictionary.prefix_offset(max_backward, max_backward), None);
        assert_eq!(
            dictionary.prefix_offset(max_backward + 13, max_backward),
            None
        );
        assert_eq!(dictionary.prefix_offset(u64::MAX, u64::MAX), None);
    }

    #[test]
    fn retained_bytes_cover_the_sources_and_the_indexes() {
        let short = DictionaryBuilder::new()
            .add_prefix(&b"tiny"[..])
            .build()
            .expect("prepared");
        let indexed = DictionaryBuilder::new()
            .add_prefix(&b"long enough to be indexed"[..])
            .build()
            .expect("prepared");
        assert!(short.retained_bytes() > short.source_bytes());
        assert!(indexed.retained_bytes() > short.retained_bytes());
    }

    #[test]
    fn a_preparation_error_says_what_was_refused() {
        assert!(DictionaryError::Empty.to_string().contains("no bytes"));
        assert!(
            DictionaryError::TooManyAttachments {
                attached: 16,
                limit: 15
            }
            .to_string()
            .contains("15")
        );
        assert!(
            DictionaryError::TooLarge {
                bytes: 99,
                limit: 8
            }
            .to_string()
            .contains("99")
        );
        assert!(
            DictionaryError::PreparationTooLarge {
                bytes: 99,
                limit: 8
            }
            .to_string()
            .contains("99")
        );
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn the_longest_match_diagnostic_reports_what_it_found() {
        let dictionary = DictionaryBuilder::new()
            .add_prefix(&b"the quick brown fox"[..])
            .build()
            .expect("prepared");

        let found = dictionary
            .longest_match(b"quick brown foxes")
            .expect("a match");
        assert_eq!(found.prefix_offset(), 4);
        assert_eq!(found.length(), 15);
        assert!(dictionary.longest_match(b"nothing here at all").is_none());
        // Fewer than the eight bytes the index is keyed on.
        assert!(dictionary.longest_match(b"the").is_none());
    }
}

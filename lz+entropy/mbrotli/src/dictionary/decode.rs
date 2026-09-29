#[cfg(feature = "compression")]
use super::PreparedDictionary;
use crate::shared::decode_dictionary::Prefixes;

/// One external dictionary attachment, applied in caller order.
///
/// Use [`DecodeDictionary::new`] to copy and validate a sequence. Attachment
/// order and bytes must match the encoder's effective dictionary. RAW input is
/// never detected as serialized data based on its contents.
#[derive(Clone, Copy, Debug)]
pub enum DictionaryAttachment<'a> {
    /// Literal LZ77 prefix bytes; never interpreted as serialized data.
    Raw(&'a [u8]),
    /// RFC 9841 serialized description with prefixes and custom static lists.
    #[cfg(feature = "experimental")]
    Serialized(&'a [u8]),
}

/// Explicit budgets for constructing a dictionary without encoder indexes.
///
/// `None` means unlimited; `Some(0)` allows no bytes of that resource. These
/// construction budgets are separate from [`crate::DecodeLimits`]. See
/// [`DecodeDictionary::new`] for an example using both limits.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DecodeDictionaryLimits {
    /// Sum of supplied attachment lengths, including empty attachments.
    pub max_source_bytes: Option<u64>,
    /// Peak live requested dictionary heap bytes during construction.
    pub max_owned_bytes: Option<usize>,
}

/// Owned immutable external dictionary for decoding.
///
/// Construction copies source bytes without building encoder match indexes.
/// Pass `&dictionary` directly to the decoder's dictionary methods. Its bytes
/// must match those used for encoding; raw Brotli cannot reliably detect a
/// wrong dictionary. Source buffers can be dropped after construction.
///
/// # Examples
///
/// With both codec features enabled, prepare the same prefix for each direction:
///
/// ```
/// # #[cfg(feature = "compression")]
/// # {
/// use mbrotli::{Compressor, DecoderConfig, Decompressor, EncoderConfig, Quality};
/// use mbrotli::dictionary::{DecodeDictionary, DictionaryAttachment, DictionaryBuilder};
/// let prefix = b"shared words for repeated payloads";
/// let prepared = DictionaryBuilder::new().add_prefix(&prefix[..]).build()?;
/// let dictionary = DecodeDictionary::new(
///     &[DictionaryAttachment::Raw(prefix)], Default::default())?;
/// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
/// let compressed = encoder.compress_with_dictionary(&prepared, prefix)?;
/// let mut decoder = Decompressor::new(DecoderConfig::default())?;
/// assert_eq!(decoder.decompress_with_dictionary(&dictionary, &compressed)?, prefix);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct DecodeDictionary {
    prefixes: Prefixes,
}

impl DecodeDictionary {
    /// Applies attachments sequentially and owns their bytes after success.
    ///
    /// # Errors
    /// Rejects more than fifteen prefix slots, arithmetic overflow, exceeded
    /// budgets and failed allocations. Empty RAW attachments occupy a slot.
    /// With `experimental`, also rejects malformed serialized attachments and
    /// conflicting active custom static dictionaries.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::{DecodeDictionary, DecodeDictionaryLimits, DictionaryAttachment};
    /// let limits = DecodeDictionaryLimits {
    ///     max_source_bytes: Some(1024),
    ///     max_owned_bytes: Some(4096),
    /// };
    /// let source = b"shared prefix".to_vec();
    /// let dictionary = DecodeDictionary::new(&[DictionaryAttachment::Raw(&source)], limits)?;
    /// drop(source); // The dictionary owns its copy.
    /// assert_eq!(dictionary.attachment_count(), 1);
    /// assert!(dictionary.retained_bytes() >= b"shared prefix".len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        attachments: &[DictionaryAttachment<'_>],
        limits: DecodeDictionaryLimits,
    ) -> Result<Self, DecodeDictionaryError> {
        Ok(Self {
            prefixes: Prefixes::build(attachments, limits)?,
        })
    }
    /// Returns the number of effective LZ77 prefix slots.
    pub const fn attachment_count(&self) -> usize {
        self.prefixes.count()
    }
    /// Returns owned heap storage, excluding this object's inline fields.
    pub const fn retained_bytes(&self) -> usize {
        self.prefixes.retained_bytes()
    }
}

impl AsRef<Self> for DecodeDictionary {
    /// Borrows the dictionary itself, so an owned dictionary, a `&'static`
    /// one and an `Arc` of one all satisfy `AsRef<DecodeDictionary>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use mbrotli::dictionary::{DecodeDictionary, DictionaryAttachment};
    ///
    /// let shared = Arc::new(DecodeDictionary::new(
    ///     &[DictionaryAttachment::Raw(b"prefix")], Default::default())?);
    /// let view: &DecodeDictionary = shared.as_ref().as_ref();
    /// assert!(std::ptr::eq(view, &*shared));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn as_ref(&self) -> &Self {
        self
    }
}

/// Borrowed immutable dictionary source for a single decoding operation.
///
/// Decoder entry points accept `impl Into<DictionaryRef>`, so a shared reference
/// to either supported dictionary type suffices without an explicit conversion.
/// A session keeps the borrow until dropped; the decoder retains no dictionary
/// reference between operations.
///
/// # Examples
///
/// ```
/// use mbrotli::{DecodeOperation, DecoderConfig, Decompressor};
/// use mbrotli::dictionary::{DecodeDictionary, DictionaryAttachment, DictionaryRef};
/// let dictionary = DecodeDictionary::new(
///     &[DictionaryAttachment::Raw(b"prefix")], Default::default())?;
/// let view = DictionaryRef::from(&dictionary);
/// let mut decoder = Decompressor::new(DecoderConfig::default())?;
/// let mut session = decoder.start_with_dictionary(view, Default::default())?;
/// // This empty member needs no references, but uses the same dictionary API.
/// session.process(&[0x3b], &mut [], DecodeOperation::Finish)?;
/// assert!(session.is_finished());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub enum DictionaryRef<'a> {
    /// Reuses a prepared encoder dictionary without copying its payload.
    #[cfg(feature = "compression")]
    Prepared(&'a PreparedDictionary),
    /// Uses a dictionary built without encoder search indexes.
    DecodeOnly(&'a DecodeDictionary),
}

#[cfg(feature = "compression")]
impl<'a> From<&'a PreparedDictionary> for DictionaryRef<'a> {
    fn from(value: &'a PreparedDictionary) -> Self {
        Self::Prepared(value)
    }
}
impl<'a> From<&'a DecodeDictionary> for DictionaryRef<'a> {
    fn from(value: &'a DecodeDictionary) -> Self {
        Self::DecodeOnly(value)
    }
}

impl DictionaryRef<'_> {
    #[cfg(feature = "experimental")]
    pub(crate) fn custom_word(
        self,
        address: u64,
        length: usize,
        context: usize,
        scratch: &mut [u8; crate::shared::dictionary::transform::SCRATCH_BYTES],
    ) -> Option<Result<usize, crate::DecodeError>> {
        match self {
            #[cfg(feature = "compression")]
            Self::Prepared(value) => value
                .inner()
                .static_index
                .as_ref()
                .map(|index| index.decode_word(address, length, context, scratch)),
            Self::DecodeOnly(value) => value
                .prefixes
                .description
                .as_ref()
                .map(|description| description.resolve(address, length, context, scratch)),
        }
    }

    pub(crate) fn prefix_len(self) -> u64 {
        match self {
            #[cfg(feature = "compression")]
            Self::Prepared(value) => value.inner().dictionaries().prefix().total_len(),
            Self::DecodeOnly(value) => value.prefixes.total(),
        }
    }
}

impl<'a> DictionaryRef<'a> {
    /// Contiguous prefix bytes from `offset` to the end of the segment holding
    /// it, empty past the prefix end. Lets a copy emit whole runs instead of
    /// byte by byte.
    pub(crate) fn prefix_run(self, offset: u64) -> &'a [u8] {
        match self {
            #[cfg(feature = "compression")]
            Self::Prepared(value) => value.inner().dictionaries().prefix().run_from(offset),
            Self::DecodeOnly(value) => value.prefixes.run_from(offset),
        }
    }
}

/// Failure while constructing an immutable decode-only dictionary.
///
/// # Examples
///
/// ```
/// use mbrotli::dictionary::{
///     DecodeDictionary, DecodeDictionaryError, DecodeDictionaryLimits, DictionaryAttachment,
/// };
/// let limits = DecodeDictionaryLimits { max_source_bytes: Some(0), ..Default::default() };
/// let result = DecodeDictionary::new(&[DictionaryAttachment::Raw(b"prefix")], limits);
/// assert!(matches!(result, Err(DecodeDictionaryError::SourceLimitExceeded { limit: 0 })));
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DecodeDictionaryError {
    /// The serialized description violates its format.
    #[cfg(feature = "experimental")]
    #[error("invalid serialized decode dictionary")]
    InvalidSerializedDictionary,
    /// Two active custom static descriptions cannot be combined.
    #[cfg(feature = "experimental")]
    #[error("conflicting custom static dictionaries")]
    ConflictingStaticDictionaries,
    /// The format permits at most fifteen prefix attachments.
    #[error("more than fifteen dictionary prefix attachments")]
    TooManyAttachments,
    /// Supplied source buffers exceed the explicit budget.
    #[error("dictionary source exceeds {limit} bytes")]
    SourceLimitExceeded {
        /// Configured source budget.
        limit: u64,
    },
    /// Peak owned storage exceeds the explicit budget.
    #[error("dictionary storage exceeds {limit} bytes")]
    MemoryLimitExceeded {
        /// Configured memory budget.
        limit: usize,
    },
    /// A fallible dictionary allocation failed.
    #[error("dictionary allocation failed")]
    AllocationFailed,
    /// A size exceeds the counter or address space.
    #[error("dictionary size overflow")]
    SizeOverflow,
}

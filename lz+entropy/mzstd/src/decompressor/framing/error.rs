use super::{ChunkOffset, ExternalDictionaryRequest, ResourceDataPosition, ResourceIndex};
use crate::DecodeError;
use crate::dictionary::DictionaryError;
use ::core::ops::Range;

/// The resource budget that was exhausted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FramedLimitKind {
    /// InputBytes ceiling.
    InputBytes,
    /// OutputBytes ceiling.
    OutputBytes,
    /// DecodedBytes ceiling.
    DecodedBytes,
    /// ResourceBytes ceiling.
    ResourceBytes,
    /// WorkspaceBytes ceiling.
    WorkspaceBytes,
    /// FramingBytes ceiling.
    FramingBytes,
    /// DictionaryBytes ceiling.
    DictionaryBytes,
    /// MetadataBytes ceiling.
    MetadataBytes,
    /// MetadataFields ceiling.
    MetadataFields,
    /// Resources ceiling.
    Resources,
    /// Chunks ceiling.
    Chunks,
}

/// Recognized non-resource input kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnexpectedInputKind {
    /// A standalone Zstandard-format dictionary.
    SerializedDictionary,
}

/// Point where a failure was detected, not necessarily the corrupt bit.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FramedDecodeLocation {
    /// Accepted wire position.
    pub offset: u64,
    /// Current chunk, if any.
    pub(super) chunk: Option<::core::num::NonZeroU64>,
    /// Current resource, if any.
    pub(super) resource: Option<::core::num::NonZeroU64>,
}

impl FramedDecodeLocation {
    /// Current chunk, when a chunk header has been accepted.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::framing::*;
    /// assert_eq!(FramedDecodeLocation::default().chunk(), None);
    /// ```
    #[must_use]
    pub const fn chunk(self) -> Option<ChunkOffset> {
        match self.chunk {
            Some(n) => Some(ChunkOffset(n.get())),
            None => None,
        }
    }

    /// Current logical resource, when known.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::framing::*;
    /// assert_eq!(FramedDecodeLocation::default().resource(), None);
    /// ```
    #[must_use]
    pub const fn resource(self) -> Option<ResourceIndex> {
        match self.resource {
            Some(n) => Some(ResourceIndex(n.get() - 1)),
            None => None,
        }
    }
}

/// Typed framing failure. Codec and dictionary errors preserve their sources.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FramedDecodeError {
    /// Invalid framing signature.
    #[error("invalid framing signature")]
    InvalidSignature,
    /// Invalid framing header.
    #[error("invalid framing header")]
    InvalidHeader,
    /// Invalid chunk header or content.
    #[error("invalid chunk header or content")]
    InvalidChunk,
    /// Invalid chunk order.
    #[error("invalid chunk order")]
    InvalidOrder,
    /// Invalid metadata fields.
    #[error("invalid metadata fields")]
    InvalidMetadata,
    /// Invalid internal dictionary reference.
    #[error("invalid internal dictionary reference")]
    InvalidDictionaryReference,
    /// Internal dictionary references are disabled by policy.
    #[error("internal dictionary references are disabled by policy")]
    InternalDictionaryReferencesDisabled,
    /// Central directory does not match wire records.
    #[error("central directory does not match wire records")]
    InvalidDirectory,
    /// Invalid container footer.
    #[error("invalid container footer")]
    InvalidFooter,
    /// The seek table does not list the container's frames.
    #[error("seek table does not match the container's frames")]
    InvalidSeekTable,
    /// Unexpected end of input.
    #[error("unexpected end of input")]
    UnexpectedEndOfInput,
    /// Data follows the top-level object.
    #[error("data follows the top-level object")]
    TrailingData,
    /// Output slice cannot hold resource payload.
    #[error("output slice cannot hold resource payload")]
    OutputTooSmall,
    /// Framing size overflow.
    #[error("framing size overflow")]
    SizeOverflow,
    /// Framing allocation failed.
    #[error("framing allocation failed")]
    AllocationFailed,
    /// Decoder has an abandoned session.
    #[error("decoder has an abandoned session")]
    AbandonedSession,
    /// Invalid framed decoder state.
    #[error("invalid framed decoder state")]
    InvalidState,
    /// A recognized input kind is not a resource stream.
    #[error("unexpected input kind: {0:?}")]
    UnexpectedInputKind(UnexpectedInputKind),
    /// Unsupported version bits.
    #[error("unsupported framing version {0}")]
    UnsupportedVersion(u8),
    /// Decoded size violates a declared contract.
    #[error("declared size {expected} differs from decoded size {actual}")]
    DeclaredSizeMismatch {
        /// Wire or caller contract.
        expected: u64,
        /// Decoded size or established lower bound.
        actual: u64,
    },
    /// A numeric ceiling was exceeded.
    #[error("framing {kind:?} exceeds {limit}")]
    LimitExceeded {
        /// Exhausted budget.
        kind: FramedLimitKind,
        /// Configured ceiling.
        limit: u64,
    },
    /// An explicit external lookup failed.
    #[error("missing dictionary {request:?} at {location:?}")]
    MissingDictionary {
        /// Failed lookup.
        request: ExternalDictionaryRequest,
        /// Detection context.
        location: FramedDecodeLocation,
    },
    /// Zstandard core failure with framing context.
    #[error("Zstandard decoding failed at {location:?}: {source}")]
    Decode {
        /// Underlying codec error.
        #[source]
        source: DecodeError,
        /// Detection context.
        location: FramedDecodeLocation,
    },
    /// Dictionary preparation failure with framing context.
    #[error("dictionary preparation failed at {location:?}: {source}")]
    Dictionary {
        /// Underlying dictionary error.
        #[source]
        source: DictionaryError,
        /// Detection context.
        location: FramedDecodeLocation,
    },
}

/// Last payload fragment written during a failing operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProducedFragment {
    /// Only this fragment's range in the destination.
    pub range: Range<usize>,
    /// Resource-local identity.
    pub position: ResourceDataPosition,
}

/// Terminal failure with exact accepted and produced prefixes.
#[derive(Debug, thiserror::Error)]
#[error("{error}")]
pub struct FramedDecodeFailure {
    /// Underlying error.
    #[source]
    pub error: FramedDecodeError,
    /// Accepted input bytes.
    pub consumed: usize,
    /// Delivered payload bytes.
    pub produced: usize,
    /// Last written fragment, if any.
    pub last_output: Option<ProducedFragment>,
}

impl FramedDecodeFailure {
    /// Discards progress after it has been handled.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let failure = decoder.decompress_to_slice(&[], &mut []).unwrap_err();
    /// assert_eq!((failure.consumed, failure.produced), (0, 0));
    /// assert!(matches!(failure.into_error(), FramedDecodeError::UnexpectedEndOfInput));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn into_error(self) -> FramedDecodeError {
        self.error
    }
}

impl From<FramedDecodeFailure> for FramedDecodeError {
    fn from(failure: FramedDecodeFailure) -> Self {
        failure.error
    }
}

#[cfg(not(feature = "no_std"))]
impl From<FramedDecodeError> for std::io::Error {
    fn from(error: FramedDecodeError) -> Self {
        use std::io::ErrorKind;
        let kind = match &error {
            FramedDecodeError::UnexpectedEndOfInput => ErrorKind::UnexpectedEof,
            FramedDecodeError::AllocationFailed => ErrorKind::OutOfMemory,
            FramedDecodeError::OutputTooSmall
            | FramedDecodeError::AbandonedSession
            | FramedDecodeError::InvalidState
            | FramedDecodeError::LimitExceeded { .. } => ErrorKind::Other,
            _ => ErrorKind::InvalidData,
        };
        Self::new(kind, error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::core::error::Error as _;
    use alloc::string::ToString;

    #[test]
    fn locations_report_one_based_storage_as_zero_based_indices() {
        let location = FramedDecodeLocation {
            offset: 9,
            chunk: ::core::num::NonZeroU64::new(13),
            resource: ::core::num::NonZeroU64::new(1),
        };
        assert_eq!(location.chunk(), Some(ChunkOffset(13)));
        assert_eq!(location.resource(), Some(ResourceIndex(0)));
    }

    #[test]
    fn codec_failures_keep_their_source() {
        let error = FramedDecodeError::Decode {
            source: DecodeError::UnexpectedEndOfInput,
            location: FramedDecodeLocation::default(),
        };
        assert!(error.to_string().starts_with("Zstandard decoding failed"));
        assert!(error.source().is_some());
        let failure = FramedDecodeFailure {
            error: FramedDecodeError::InvalidSeekTable,
            consumed: 0,
            produced: 0,
            last_output: None,
        };
        assert_eq!(
            FramedDecodeError::from(failure).to_string(),
            "seek table does not match the container's frames"
        );
    }

    #[cfg(not(feature = "no_std"))]
    #[test]
    fn io_conversion_classifies_errors() {
        use std::io::ErrorKind;
        let kind = |error| std::io::Error::from(error).kind();
        assert_eq!(
            kind(FramedDecodeError::UnexpectedEndOfInput),
            ErrorKind::UnexpectedEof
        );
        assert_eq!(
            kind(FramedDecodeError::AllocationFailed),
            ErrorKind::OutOfMemory
        );
        assert_eq!(kind(FramedDecodeError::InvalidState), ErrorKind::Other);
        assert_eq!(
            kind(FramedDecodeError::InvalidChunk),
            ErrorKind::InvalidData
        );
    }
}

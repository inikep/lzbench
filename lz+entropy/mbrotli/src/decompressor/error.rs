use thiserror::Error;

/// Invalid decoder policy values.
///
/// # Examples
///
/// ```
/// use mbrotli::{DecodeConfigError, WindowLimit};
/// assert!(matches!(WindowLimit::standard(25),
///     Err(DecodeConfigError::StandardWindow { max_bits: 25 })));
/// assert!(matches!(WindowLimit::large(63),
///     Err(DecodeConfigError::LargeWindow { max_bits: 63 })));
/// ```
#[derive(Debug, Clone, Copy, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum DecodeConfigError {
    /// Standard window limit outside `10..=24`.
    #[error("standard window limit must be 10..=24 bits, got {max_bits}")]
    StandardWindow {
        /// Requested limit.
        max_bits: u8,
    },
    /// Extended window limit outside `10..=62`.
    #[error("large window limit must be 10..=62 bits, got {max_bits}")]
    LargeWindow {
        /// Requested limit.
        max_bits: u8,
    },
}

/// Format structure in which invalid data was detected.
///
/// Carried by [`DecodeError::InvalidData`]. Describes the malformed structure,
/// not a byte offset; incremental callers obtain accepted-byte counts from
/// [`super::DecodeFailure`].
#[derive(Debug, Clone, Copy, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum InvalidDataKind {
    /// Invalid stream header.
    #[error("stream header")]
    Header,
    /// Invalid meta-block header or length.
    #[error("meta-block")]
    MetaBlock,
    /// Invalid prefix code.
    #[error("Huffman code")]
    Huffman,
    /// Invalid entropy context mapping.
    #[error("context map")]
    ContextMap,
    /// Invalid backward distance.
    #[error("distance")]
    Distance,
    /// Invalid external or static dictionary address.
    #[error("dictionary reference")]
    DictionaryReference,
    /// Nonzero alignment padding.
    #[error("padding")]
    Padding,
}

/// Decompression failure. Raw Brotli does not authenticate dictionary identity.
///
/// One-shot methods return this directly; sessions wrap it in
/// [`super::DecodeFailure`] to preserve partial progress. I/O adapters wrap it
/// in `std::io::Error` when standard I/O is available. Include a fallback arm
/// when matching this non-exhaustive enum.
///
/// # Examples
///
/// ```
/// use mbrotli::{DecodeError, DecoderConfig, Decompressor};
/// let mut decoder = Decompressor::new(DecoderConfig::default())?;
/// match decoder.decompress(&[0x3b, 0xaa]) {
///     Err(DecodeError::TrailingData { offset }) => assert_eq!(offset, 1),
///     other => panic!("expected a trailing-data error, got {other:?}"),
/// }
/// // A failed one-shot operation does not prevent decoder reuse.
/// assert!(decoder.decompress(&[0x3b])?.is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DecodeError {
    /// The input violates the wire grammar.
    #[error("invalid Brotli {kind}")]
    InvalidData {
        /// Invalid structure.
        kind: InvalidDataKind,
    },
    /// Final input ended before a complete stream.
    #[error("unexpected end of compressed input")]
    UnexpectedEndOfInput,
    /// A strict one-shot call found bytes after its single member.
    #[error("trailing compressed data at byte {offset}")]
    TrailingData {
        /// Absolute operation offset.
        offset: u64,
    },
    /// The caller's destination cannot hold the payload.
    #[error("output slice is full after {written} bytes")]
    OutputTooSmall {
        /// Payload bytes already written.
        written: usize,
    },
    /// Regenerated output differs from the declared exact size.
    #[error("expected {expected} output bytes, decoded {actual}")]
    OutputSizeMismatch {
        /// Declared exact size.
        expected: u64,
        /// Final size or proven lower bound.
        actual: u64,
    },
    /// Compressed input budget exhausted.
    #[error("compressed input exceeds {limit} bytes")]
    InputLimitExceeded {
        /// Configured budget.
        limit: u64,
    },
    /// Regenerated output budget exhausted.
    #[error("decoded output exceeds {limit} bytes")]
    OutputLimitExceeded {
        /// Configured budget.
        limit: u64,
    },
    /// The declared window exceeds caller policy.
    #[error("declared {declared}-bit window exceeds allowed {allowed} bits")]
    WindowLimitExceeded {
        /// Header window.
        declared: u8,
        /// Configured limit.
        allowed: u8,
    },
    /// Caller policy disallows extended headers.
    #[error("large window headers are disabled")]
    LargeWindowDisabled,
    /// Live requested decoder allocation would exceed the budget.
    #[error("decoder workspace exceeds {limit} bytes")]
    MemoryLimitExceeded {
        /// Configured budget.
        limit: usize,
    },
    /// Fallible heap allocation failed.
    #[error("decoder allocation failed")]
    AllocationFailed,
    /// Arithmetic or addressable storage overflow.
    #[error("decoder size overflow")]
    SizeOverflow,
    /// A forgotten session must be explicitly recovered.
    #[error("decoder has an abandoned session")]
    AbandonedSession,
    /// The operation violates the session lifecycle or final-input contract.
    #[error("invalid decoder state")]
    InvalidState,
    /// An internal invariant failed; this indicates a library defect.
    #[error("decoder internal invariant failed")]
    InternalInvariant,
}

impl From<InvalidDataKind> for DecodeError {
    fn from(kind: InvalidDataKind) -> Self {
        Self::InvalidData { kind }
    }
}

#[cfg(not(feature = "no_std"))]
impl From<DecodeError> for std::io::Error {
    fn from(error: DecodeError) -> Self {
        use std::io::ErrorKind;
        let kind = match &error {
            DecodeError::InvalidData { .. }
            | DecodeError::TrailingData { .. }
            | DecodeError::OutputSizeMismatch { .. } => ErrorKind::InvalidData,
            DecodeError::UnexpectedEndOfInput => ErrorKind::UnexpectedEof,
            DecodeError::InvalidState
            | DecodeError::AbandonedSession
            | DecodeError::OutputTooSmall { .. } => ErrorKind::InvalidInput,
            DecodeError::AllocationFailed => ErrorKind::OutOfMemory,
            _ => ErrorKind::Other,
        };
        Self::new(kind, error)
    }
}

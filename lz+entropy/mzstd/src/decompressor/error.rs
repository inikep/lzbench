use ::core::fmt;

use thiserror::Error;

use crate::dictionary::DictionaryId;

/// An invalid decoder policy value.
///
/// # Examples
///
/// ```
/// use mzstd::{DecodeConfigError, DecodeLimits};
/// assert!(matches!(
///     DecodeLimits::default().with_max_window_log(9),
///     Err(DecodeConfigError::WindowLog { log: 9 })
/// ));
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum DecodeConfigError {
    /// A window limit outside `10..=31`.
    #[error("window log limit must be 10..=31, got {log}")]
    WindowLog {
        /// The requested log.
        log: u8,
    },
}

/// The structure in which invalid compressed data was detected.
///
/// Carried by [`DecodeError::InvalidData`]. It names the malformed part of
/// the format, not a byte offset; sessions report accepted-byte counts
/// separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum InvalidDataKind {
    /// Bytes where a frame should start match no frame magic number.
    #[error("frame magic number")]
    Magic,
    /// A frame header with its reserved bit set.
    #[error("frame header")]
    FrameHeader,
    /// A skippable frame whose size field overflows.
    #[error("skippable frame")]
    SkippableFrame,
    /// A block with the reserved block type.
    #[error("block type")]
    BlockType,
    /// A block larger than the frame allows, compressed or regenerated.
    #[error("block size")]
    BlockSize,
    /// A malformed literals section header or size.
    #[error("literals section")]
    LiteralsSection,
    /// An invalid Huffman tree description.
    #[error("Huffman tree description")]
    HuffmanTable,
    /// A Huffman bitstream that is not consumed exactly.
    #[error("Huffman bitstream")]
    HuffmanStream,
    /// An invalid FSE table description.
    #[error("FSE table description")]
    FseTable,
    /// A malformed sequences section header or table mode.
    #[error("sequences section")]
    SequencesSection,
    /// A sequence bitstream that is not consumed exactly.
    #[error("sequence bitstream")]
    SequenceStream,
    /// A sequence copying more literals than the block holds.
    #[error("literal length")]
    LiteralsOverrun,
    /// A zero offset, or one reaching before the frame or past its window.
    #[error("match offset")]
    Offset,
}

/// A decompression failure.
///
/// One-shot methods return it directly; sessions wrap it in
/// [`super::DecodeFailure`] together with the progress of the failing call.
/// The `std::io` adapters wrap it in `std::io::Error` (with `std`), from
/// which `get_ref` and `downcast_ref::<DecodeError>()` recover it.
///
/// # Examples
///
/// ```
/// use mzstd::{DecodeError, Decompressor, InvalidDataKind};
/// let mut decoder = Decompressor::default();
/// assert!(matches!(
///     decoder.decompress(b"not zstd"),
///     Err(DecodeError::InvalidData { kind: InvalidDataKind::Magic })
/// ));
/// assert!(matches!(
///     decoder.decompress(&[0x28, 0xb5, 0x2f, 0xfd]),
///     Err(DecodeError::UnexpectedEndOfInput)
/// ));
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DecodeError {
    /// The input violates the Zstandard format.
    #[error("invalid Zstandard {kind}")]
    InvalidData {
        /// The malformed structure.
        #[source]
        kind: InvalidDataKind,
    },
    /// The input ended inside a frame, or before any frame.
    #[error("unexpected end of compressed input")]
    UnexpectedEndOfInput,
    /// Bytes followed the frame of a [`super::FrameMode::Single`] operation.
    #[error("trailing data at byte {offset}")]
    TrailingData {
        /// Offset of the first byte after the frame.
        offset: u64,
    },
    /// The caller's output slice is full.
    #[error("output slice is full after {written} bytes")]
    OutputTooSmall {
        /// Bytes written before the slice filled.
        written: usize,
    },
    /// The frame's content checksum does not match its output.
    #[error("content checksum mismatch: stored {expected:#010x}, computed {computed:#010x}")]
    ChecksumMismatch {
        /// The checksum stored in the frame.
        expected: u32,
        /// The low 32 bits of the XXH64 of the decoded content.
        computed: u32,
    },
    /// The frame regenerated a different size than its header declares.
    #[error("frame declares {declared} content bytes, decoded {decoded}")]
    ContentSizeMismatch {
        /// `Frame_Content_Size` from the header.
        declared: u64,
        /// Bytes regenerated, or a lower bound once it exceeds `declared`.
        decoded: u64,
    },
    /// The frame names a dictionary, and the operation has none.
    ///
    /// Decode such frames with the dictionary they were compressed with,
    /// through the `*_with_dictionary` methods of
    /// [`Decompressor`](super::Decompressor).
    #[error("frame requires dictionary {id}, and none was supplied")]
    DictionaryUnsupported {
        /// The frame's `Dictionary_ID`.
        id: u32,
    },
    /// The frame names a dictionary ID other than the one the operation's
    /// dictionary declares.
    ///
    /// A frame that names no dictionary decodes with any dictionary. A
    /// raw-content dictionary declares no ID, so it never matches a frame
    /// that names one, as in the reference decoder.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DecodeDictionary;
    /// use mzstd::{DecodeError, Decompressor};
    /// // A frame naming dictionary 7, with an empty last raw block.
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x21, 0x07, 0x00, 0x01, 0x00, 0x00];
    /// let raw = DecodeDictionary::try_from(&b"raw content"[..])?;
    /// let error = Decompressor::default()
    ///     .decompress_with_dictionary(&raw, &frame)
    ///     .unwrap_err();
    /// assert!(matches!(
    ///     error,
    ///     DecodeError::DictionaryMismatch { expected: None, frame } if u32::from(frame) == 7
    /// ));
    /// assert_eq!(
    ///     error.to_string(),
    ///     "frame requires dictionary 7, but the dictionary declares no ID"
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[error("frame requires dictionary {frame}, but {}", Supplied(*.expected))]
    DictionaryMismatch {
        /// The ID the operation's dictionary declares, if any.
        expected: Option<DictionaryId>,
        /// The frame's `Dictionary_ID`.
        frame: DictionaryId,
    },
    /// The frame's window exceeds the configured limit.
    #[error("frame window of {window_size} bytes exceeds the {limit}-byte limit")]
    WindowLimitExceeded {
        /// The window the frame declares.
        window_size: u64,
        /// The configured limit.
        limit: u64,
    },
    /// The compressed input budget is exhausted.
    #[error("compressed input exceeds {limit} bytes")]
    InputLimitExceeded {
        /// The configured budget.
        limit: u64,
    },
    /// The decompressed output budget is exhausted.
    #[error("decompressed output exceeds {limit} bytes")]
    OutputLimitExceeded {
        /// The configured budget.
        limit: u64,
    },
    /// A heap allocation failed.
    #[error("decoder allocation failed")]
    AllocationFailed,
    /// The call violates the session contract, or the session already failed.
    #[error("invalid decoder state")]
    InvalidState,
}

/// Describes the dictionary an operation supplied, for
/// [`DecodeError::DictionaryMismatch`].
struct Supplied(Option<DictionaryId>);

impl fmt::Display for Supplied {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(id) => write!(formatter, "dictionary {id} was supplied"),
            None => formatter.write_str("the dictionary declares no ID"),
        }
    }
}

impl From<InvalidDataKind> for DecodeError {
    fn from(kind: InvalidDataKind) -> Self {
        Self::InvalidData { kind }
    }
}

#[cfg(not(feature = "no_std"))]
impl From<DecodeError> for std::io::Error {
    /// Maps format errors to `InvalidData`, truncation to `UnexpectedEof`,
    /// contract violations to `InvalidInput`, allocation failure to
    /// `OutOfMemory` and exhausted limits to `Other`.
    fn from(error: DecodeError) -> Self {
        use std::io::ErrorKind;
        let kind = match &error {
            DecodeError::InvalidData { .. }
            | DecodeError::TrailingData { .. }
            | DecodeError::ChecksumMismatch { .. }
            | DecodeError::ContentSizeMismatch { .. }
            | DecodeError::DictionaryUnsupported { .. }
            | DecodeError::DictionaryMismatch { .. } => ErrorKind::InvalidData,
            DecodeError::UnexpectedEndOfInput => ErrorKind::UnexpectedEof,
            DecodeError::OutputTooSmall { .. } | DecodeError::InvalidState => {
                ErrorKind::InvalidInput
            }
            DecodeError::AllocationFailed => ErrorKind::OutOfMemory,
            DecodeError::WindowLimitExceeded { .. }
            | DecodeError::InputLimitExceeded { .. }
            | DecodeError::OutputLimitExceeded { .. } => ErrorKind::Other,
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
    fn invalid_data_keeps_its_kind_as_the_source() {
        let error = DecodeError::from(InvalidDataKind::Offset);
        assert_eq!(error.to_string(), "invalid Zstandard match offset");
        let source = error.source().expect("a source");
        assert_eq!(source.to_string(), "match offset");
    }

    #[test]
    fn messages_name_the_values_involved() {
        let checksum = DecodeError::ChecksumMismatch {
            expected: 1,
            computed: 0xabcd,
        };
        assert_eq!(
            checksum.to_string(),
            "content checksum mismatch: stored 0x00000001, computed 0x0000abcd"
        );
        let window = DecodeError::WindowLimitExceeded {
            window_size: 1 << 28,
            limit: 1 << 27,
        };
        assert_eq!(
            window.to_string(),
            "frame window of 268435456 bytes exceeds the 134217728-byte limit"
        );
        assert_eq!(
            DecodeConfigError::WindowLog { log: 40 }.to_string(),
            "window log limit must be 10..=31, got 40"
        );
        let frame = DictionaryId::new(7).expect("non-zero");
        let mismatch = DecodeError::DictionaryMismatch {
            expected: DictionaryId::new(9),
            frame,
        };
        assert_eq!(
            mismatch.to_string(),
            "frame requires dictionary 7, but dictionary 9 was supplied"
        );
        let missing = DecodeError::DictionaryMismatch {
            expected: None,
            frame,
        };
        assert_eq!(
            missing.to_string(),
            "frame requires dictionary 7, but the dictionary declares no ID"
        );
        assert_eq!(
            DecodeError::DictionaryUnsupported { id: 7 }.to_string(),
            "frame requires dictionary 7, and none was supplied"
        );
    }

    #[cfg(not(feature = "no_std"))]
    #[test]
    fn io_errors_classify_and_keep_the_decode_error() {
        use std::io::ErrorKind;
        let cases = [
            (
                DecodeError::from(InvalidDataKind::Magic),
                ErrorKind::InvalidData,
            ),
            (
                DecodeError::TrailingData { offset: 1 },
                ErrorKind::InvalidData,
            ),
            (
                DecodeError::ChecksumMismatch {
                    expected: 0,
                    computed: 1,
                },
                ErrorKind::InvalidData,
            ),
            (
                DecodeError::ContentSizeMismatch {
                    declared: 0,
                    decoded: 1,
                },
                ErrorKind::InvalidData,
            ),
            (
                DecodeError::DictionaryUnsupported { id: 1 },
                ErrorKind::InvalidData,
            ),
            (
                DecodeError::DictionaryMismatch {
                    expected: None,
                    frame: DictionaryId::new(1).expect("non-zero"),
                },
                ErrorKind::InvalidData,
            ),
            (DecodeError::UnexpectedEndOfInput, ErrorKind::UnexpectedEof),
            (
                DecodeError::OutputTooSmall { written: 0 },
                ErrorKind::InvalidInput,
            ),
            (DecodeError::InvalidState, ErrorKind::InvalidInput),
            (DecodeError::AllocationFailed, ErrorKind::OutOfMemory),
            (
                DecodeError::WindowLimitExceeded {
                    window_size: 2,
                    limit: 1,
                },
                ErrorKind::Other,
            ),
            (
                DecodeError::InputLimitExceeded { limit: 1 },
                ErrorKind::Other,
            ),
            (
                DecodeError::OutputLimitExceeded { limit: 1 },
                ErrorKind::Other,
            ),
        ];
        for (error, kind) in cases {
            let io = std::io::Error::from(error);
            assert_eq!(io.kind(), kind);
            assert!(io.get_ref().is_some_and(|inner| inner.is::<DecodeError>()));
        }
    }
}

//! What can go wrong while a stream is being encoded.
//!
//! Configuration mistakes are reported by
//! [`ConfigError`](super::ConfigError) when the compressor is built, and
//! dictionary mistakes by
//! [`DictionaryError`](super::dictionary::DictionaryError) when the dictionary
//! is built. What is left — the failures that need an operation in flight to
//! happen at all — is [`EncodeError`].

use super::config::{Quality, SizeOverflow};
use super::internal::BrotliCompressError;
use thiserror::Error;

/// Error returned by an encoding operation.
///
/// Every variant describes something about the *operation*: the destination was
/// too small, the dictionary cannot be used at this quality, an allocation was
/// refused. A configuration that could never work is rejected earlier, by
/// [`Compressor::new`](crate::Compressor::new), and never reaches here.
///
/// # Examples
///
/// ```
/// use mbrotli::{Compressor, EncodeError, EncoderConfig, Quality};
///
/// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
/// let mut cramped = [0u8; 1];
///
/// assert!(matches!(
///     encoder.compress_to_slice(b"a payload that will not fit in one byte", &mut cramped),
///     Err(EncodeError::OutputTooSmall { provided: 1 })
/// ));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum EncodeError {
    /// Logical placement plus input exceeds the RFC's 63-bit position range.
    #[error("stream position {position} plus {input_bytes} input bytes exceeds 63 bits")]
    StreamPositionOverflow {
        /// Position before accepting this input.
        position: u64,
        /// Number of input bytes offered.
        input_bytes: u64,
    },
    /// The caller's destination cannot hold the whole stream.
    ///
    /// Size one with [`Compressor::max_compressed_size`] to make this
    /// impossible.
    ///
    /// [`Compressor::max_compressed_size`]: crate::Compressor::max_compressed_size
    #[error("a destination of {provided} bytes cannot hold the compressed stream")]
    OutputTooSmall {
        /// How many bytes the caller offered.
        provided: usize,
    },
    /// An allocation the operation needed was refused.
    #[error("an allocation of {requested} bytes failed")]
    AllocationFailed {
        /// How many bytes were asked for.
        requested: usize,
    },
    /// The compressed-size bound does not fit in a `usize`.
    #[error(transparent)]
    Bound(#[from] SizeOverflow),
    /// This quality cannot compress against an attached dictionary.
    ///
    /// The reference compiles its compound-dictionary search only for the match
    /// finders qualities five and above select, and silently ignores the
    /// dictionary elsewhere. This crate refuses instead: a stream compressed
    /// without the dictionary it was given decodes perfectly well, so the
    /// mistake would stay invisible until a decoder that does attach it
    /// produced the wrong bytes.
    #[error("quality {} cannot compress against a prepared dictionary", quality.get())]
    DictionaryUnsupportedForQuality {
        /// The quality that was asked for.
        quality: Quality,
    },
    /// A non-zero stream offset was asked for on an unsupported path.
    ///
    /// Continuations require `experimental` and quality two or above.
    #[error(
        "stream offset {offset} requires experimental continuation support at quality 2 or above"
    )]
    UnsupportedStreamOffset {
        /// The offset that was asked for.
        offset: u64,
    },
    /// A session was abandoned without being dropped, and its state is unknown.
    ///
    /// `::core::mem::forget` can skip a session's destructor, which is the one way
    /// a compressor can be left holding state no operation has cleaned up.
    /// Rather than trusting it, the compressor refuses until
    /// [`Compressor::recover`](crate::Compressor::recover) has put it back into
    /// a known state.
    #[error("a previous session was abandoned; call Compressor::recover before encoding again")]
    AbandonedSession,
    /// An operation was asked for that the session's state does not allow.
    #[error("the session cannot {attempted} in its current state")]
    InvalidState {
        /// What was attempted.
        attempted: &'static str,
    },
    /// An invariant inside the encoder was violated.
    ///
    /// No valid caller input can cause this; it is a defect in this crate.
    #[error("an internal encoder invariant was violated: {detail}")]
    InternalInvariant {
        /// What the encoder reported.
        detail: &'static str,
    },
}

impl EncodeError {
    /// Lifts a low-level encoder error into the public one.
    ///
    /// `provided` is the size of the destination the operation was given, which
    /// only the caller of the operation knows; the encoders report a short
    /// buffer without it.
    pub(crate) fn from_core(error: BrotliCompressError, provided: usize) -> Self {
        match error {
            BrotliCompressError::OutputTooSmall => Self::OutputTooSmall { provided },
            BrotliCompressError::BoundOverflow => Self::Bound(SizeOverflow),
            // A large window at a quality that cannot carry one is refused when
            // the compressor is built, a dictionary at a quality that cannot
            // read one is refused before the encoder is asked, and every
            // quality has an encoder. None of the three can be reached from a
            // validated configuration.
            BrotliCompressError::Shared(_) | BrotliCompressError::UnsupportedQuality(_) => {
                Self::InternalInvariant {
                    detail: "a validated configuration reached an encoder that refused it",
                }
            }
            BrotliCompressError::BufferOverflow => Self::InternalInvariant {
                detail: "the encoder's scratch buffer was too small",
            },
            // The encoders perform no I/O; the variant exists for the low-level
            // error type alone.
            #[cfg(not(feature = "no_std"))]
            BrotliCompressError::IOError(_) => Self::InternalInvariant {
                detail: "an encoder reported an I/O failure it cannot perform",
            },
        }
    }
}

#[cfg(not(feature = "no_std"))]
impl From<EncodeError> for std::io::Error {
    /// Carries an encoding failure through a [`std::io`] adapter.
    ///
    /// A short destination becomes [`std::io::ErrorKind::WriteZero`] and an
    /// allocation failure [`std::io::ErrorKind::OutOfMemory`], so a caller can
    /// tell the two apart without downcasting; everything else keeps the
    /// original error as its source.
    fn from(value: EncodeError) -> Self {
        let kind = match value {
            EncodeError::OutputTooSmall { .. } => std::io::ErrorKind::WriteZero,
            EncodeError::AllocationFailed { .. } => std::io::ErrorKind::OutOfMemory,
            EncodeError::UnsupportedStreamOffset { .. }
            | EncodeError::StreamPositionOverflow { .. }
            | EncodeError::DictionaryUnsupportedForQuality { .. } => {
                std::io::ErrorKind::InvalidInput
            }
            EncodeError::AbandonedSession | EncodeError::InvalidState { .. } => {
                std::io::ErrorKind::InvalidData
            }
            EncodeError::Bound(_) | EncodeError::InternalInvariant { .. } => {
                std::io::ErrorKind::Other
            }
        };
        Self::new(kind, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::shared::SharedBrotliError;
    use ::core::error::Error as _;
    use alloc::string::ToString;

    #[test]
    fn a_short_destination_reports_what_it_was_given() {
        let error = EncodeError::from_core(BrotliCompressError::OutputTooSmall, 7);
        assert!(matches!(error, EncodeError::OutputTooSmall { provided: 7 }));
        assert!(error.to_string().contains('7'));
    }

    #[test]
    fn a_refused_dictionary_names_the_quality() {
        // The refusal is raised by the compressor, which knows the quality as a
        // `Quality`, rather than lifted from a low-level error.
        let error = EncodeError::DictionaryUnsupportedForQuality {
            quality: Quality::Q3,
        };
        assert!(error.to_string().contains('3'));
    }

    #[test]
    fn unreachable_low_level_failures_become_internal_invariants() {
        for error in [
            BrotliCompressError::BufferOverflow,
            BrotliCompressError::UnsupportedQuality(5),
            BrotliCompressError::Shared(SharedBrotliError::UnsupportedLargeWindow { quality: 0 }),
            #[cfg(not(feature = "no_std"))]
            BrotliCompressError::IOError(std::io::Error::other("nowhere")),
        ] {
            assert!(matches!(
                EncodeError::from_core(error, 0),
                EncodeError::InternalInvariant { .. }
            ));
        }
    }

    #[test]
    fn a_bound_overflow_travels_as_its_own_error() {
        let error = EncodeError::from_core(BrotliCompressError::BoundOverflow, 0);
        assert!(matches!(error, EncodeError::Bound(SizeOverflow)));
        assert_eq!(error.to_string(), SizeOverflow.to_string());
        assert!(EncodeError::from(SizeOverflow).source().is_none());
    }

    #[test]
    #[cfg(not(feature = "no_std"))]
    fn every_variant_maps_to_a_distinguishable_io_kind() {
        let cases = [
            (
                EncodeError::OutputTooSmall { provided: 1 },
                std::io::ErrorKind::WriteZero,
            ),
            (
                EncodeError::AllocationFailed { requested: 8 },
                std::io::ErrorKind::OutOfMemory,
            ),
            (
                EncodeError::UnsupportedStreamOffset { offset: 1 },
                std::io::ErrorKind::InvalidInput,
            ),
            (
                EncodeError::DictionaryUnsupportedForQuality {
                    quality: Quality::Q0,
                },
                std::io::ErrorKind::InvalidInput,
            ),
            (
                EncodeError::AbandonedSession,
                std::io::ErrorKind::InvalidData,
            ),
            (
                EncodeError::InvalidState {
                    attempted: "process",
                },
                std::io::ErrorKind::InvalidData,
            ),
            (EncodeError::Bound(SizeOverflow), std::io::ErrorKind::Other),
            (
                EncodeError::InternalInvariant { detail: "defect" },
                std::io::ErrorKind::Other,
            ),
        ];
        for (error, expected) in cases {
            let message = error.to_string();
            let io = std::io::Error::from(error);
            assert_eq!(io.kind(), expected);
            // The original error survives as the source, so nothing is lost.
            assert_eq!(
                io.get_ref().map(std::string::ToString::to_string),
                Some(message)
            );
        }
    }
}

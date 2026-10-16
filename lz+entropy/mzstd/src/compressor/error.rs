//! What can go wrong while a frame is being encoded.
//!
//! Configuration mistakes are reported by [`ConfigError`](super::ConfigError)
//! when the compressor is built; what is left needs an operation in flight.

use thiserror::Error;

use super::config::SizeOverflow;
use super::core::error::CoreError;

/// Error returned by an encoding operation.
///
/// # Examples
///
/// ```
/// use mzstd::{Compressor, EncodeError};
///
/// let mut encoder = Compressor::new(Default::default())?;
/// let mut cramped = [0u8; 4];
/// assert!(matches!(
///     encoder.compress_to_slice(b"a payload that will not fit in four bytes", &mut cramped),
///     Err(EncodeError::OutputTooSmall { provided: 4 })
/// ));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Error, Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum EncodeError {
    /// The caller's destination cannot hold the whole frame.
    ///
    /// Size it with [`Compressor::max_compressed_size`] to make this
    /// impossible.
    ///
    /// [`Compressor::max_compressed_size`]: crate::Compressor::max_compressed_size
    #[error("a destination of {provided} bytes cannot hold the compressed frame")]
    OutputTooSmall {
        /// How many bytes the caller offered.
        provided: usize,
    },
    /// The input does not match the size declared for the stream.
    #[error("the stream was declared as {declared} bytes but received a different amount")]
    InputSizeMismatch {
        /// The declared size.
        declared: u64,
    },
    /// The session has already failed or finished.
    #[error("the session cannot accept this operation in its current state")]
    InvalidState,
    /// The compressed-size bound does not fit in a `usize`.
    #[error(transparent)]
    Bound(#[from] SizeOverflow),
    /// An internal invariant of the encoder failed.
    #[error("internal encoder failure")]
    Internal,
    /// The dictionary was prepared for a configuration other than the
    /// compressor's (frame options aside).
    ///
    /// Prepare it with [`DictionaryBuilder::with_config`] for the
    /// compressor's [`config`](crate::Compressor::config).
    ///
    /// [`DictionaryBuilder::with_config`]: crate::dictionary::DictionaryBuilder::with_config
    #[error("the dictionary was prepared for a different encoder configuration")]
    DictionaryMismatch,
}

impl EncodeError {
    /// Translates a core failure at the public boundary.
    pub(crate) fn from_core(error: CoreError, provided: usize, declared: Option<u64>) -> Self {
        match error {
            CoreError::DstSizeTooSmall => Self::OutputTooSmall { provided },
            CoreError::SrcSizeWrong => Self::InputSizeMismatch {
                declared: declared.unwrap_or(0),
            },
            CoreError::StageWrong => Self::InvalidState,
            CoreError::Generic
            | CoreError::TableLogTooLarge
            | CoreError::MaxSymbolValueTooLarge
            | CoreError::MaxSymbolValueTooSmall => Self::Internal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_errors_map_to_public_variants() {
        assert_eq!(
            EncodeError::from_core(CoreError::DstSizeTooSmall, 3, None),
            EncodeError::OutputTooSmall { provided: 3 }
        );
        assert_eq!(
            EncodeError::from_core(CoreError::SrcSizeWrong, 0, Some(9)),
            EncodeError::InputSizeMismatch { declared: 9 }
        );
        assert_eq!(
            EncodeError::from_core(CoreError::StageWrong, 0, None),
            EncodeError::InvalidState
        );
        assert_eq!(
            EncodeError::from_core(CoreError::Generic, 0, None),
            EncodeError::Internal
        );
        assert_eq!(
            EncodeError::from(SizeOverflow),
            EncodeError::Bound(SizeOverflow)
        );
    }
}

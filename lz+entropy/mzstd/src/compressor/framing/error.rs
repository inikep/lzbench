use crate::{ConfigError, EncodeError};
use thiserror::Error;

/// Failure to validate, encode, or deliver a container.
///
/// # Examples
///
/// ```
/// use mzstd::framing::FramedEncodeError;
/// let error = FramedEncodeError::Limit { kind: "chunk count", limit: 3 };
/// assert_eq!(error.to_string(), "framing chunk count limit exceeded (limit 3)");
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FramedEncodeError {
    /// Invalid owner configuration.
    #[error("invalid framed encoder configuration: {0}")]
    Config(#[from] ConfigError),
    /// Drain output before retrying this uncommitted command.
    #[error("framing output is pending")]
    OutputPending,
    /// A forgotten container requires recovery.
    #[error("a framed session was abandoned; recover the owner")]
    AbandonedSession,
    /// A resource guard was dropped or forgotten before completion.
    #[error("a framed resource was abandoned")]
    AbandonedResource,
    /// A terminal operation cannot be continued with this command.
    #[error("invalid framed encoder state")]
    InvalidState,
    /// Fixed output cannot hold the canonical stream.
    #[error("framed output destination is too small")]
    OutputTooSmall,
    /// A framing allocation was refused.
    #[error("framing allocation failed")]
    AllocationFailed,
    /// Payload size differs from the declared resource or container contract.
    #[error("{scope} input size mismatch: expected {expected}, actual {actual}")]
    InputSizeMismatch {
        /// Whether the contract belongs to a resource or the container.
        scope: &'static str,
        /// Declared payload length.
        expected: u64,
        /// Offered or accepted payload length.
        actual: u64,
    },
    /// Invalid options, references, metadata, or chunk order.
    #[error("invalid framing operation: {0}")]
    Invalid(&'static str),
    /// A configured resource ceiling would be exceeded before allocation.
    #[error("framing {kind} limit exceeded (limit {limit})")]
    Limit {
        /// Storage or count budget that rejected the operation.
        kind: &'static str,
        /// Configured maximum in bytes or entries.
        limit: u64,
    },
    /// A size or offset cannot be represented: a varint above 63 bits, a
    /// frame above 32 bits, or a seek table of too many frames.
    #[error("framing size or offset overflow")]
    Overflow,
    /// Compression failed. The resource must be abandoned.
    #[error("resource encoding failed: {0}")]
    Encode(#[from] EncodeError),
    /// Sink failure; the unwritten suffix is retained for retry.
    #[error("container output failed: {0}")]
    #[cfg(not(feature = "no_std"))]
    Io(#[from] std::io::Error),
}

#[cfg(not(feature = "no_std"))]
impl From<FramingError> for std::io::Error {
    fn from(error: FramingError) -> Self {
        match error {
            FramingError::Io(error) => error,
            other => Self::other(other),
        }
    }
}

/// Recoverable finalization failure retaining the writer and its pending bytes.
#[derive(Debug)]
#[cfg(not(feature = "no_std"))]
pub struct FramingFinishError<T> {
    /// Writer to retry or recover the sink from.
    pub writer: T,
    /// Failure reported by the last finalization attempt.
    pub error: FramingError,
}

/// Compatibility name for the canonical framed encoding error.
pub type FramingError = FramedEncodeError;

/// Position at which a framed operation detected a failure.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FramedEncodeLocation {
    /// Zero-based resource index, including hidden and empty resources.
    pub resource_index: Option<u64>,
    /// Structured-input item index; absent for native commands.
    pub item_index: Option<usize>,
    /// Accepted bytes in the current resource.
    pub resource_input_offset: Option<u64>,
    /// Delivered bytes relative to the container's start.
    pub wire_offset: u64,
}

/// Exact progress retained even when an encoding operation fails.
#[derive(Debug, Error)]
#[error("{error}")]
pub struct FramedEncodeFailure {
    /// Typed cause, retaining the source chain.
    #[source]
    pub error: FramedEncodeError,
    /// Accepted input; per call for sessions, cumulative for one-shot.
    pub consumed: usize,
    /// Delivered output; per call for sessions, cumulative for one-shot.
    pub produced: usize,
    /// Context at detection.
    pub location: FramedEncodeLocation,
}

impl FramedEncodeFailure {
    /// Discards progress after the caller has accounted for it.
    #[must_use]
    pub fn into_error(self) -> FramedEncodeError {
        self.error
    }
}

impl From<FramedEncodeFailure> for FramedEncodeError {
    fn from(failure: FramedEncodeFailure) -> Self {
        failure.error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::core::error::Error as _;
    use alloc::string::ToString;

    #[test]
    fn failures_keep_their_cause_as_the_source() {
        let failure = FramedEncodeFailure {
            error: FramedEncodeError::Encode(EncodeError::InvalidState),
            consumed: 1,
            produced: 2,
            location: FramedEncodeLocation::default(),
        };
        assert_eq!(
            failure.to_string(),
            "resource encoding failed: the session cannot accept this operation in its current state"
        );
        assert!(failure.source().is_some());
        assert!(matches!(
            FramedEncodeError::from(failure),
            FramedEncodeError::Encode(EncodeError::InvalidState)
        ));
    }

    #[cfg(not(feature = "no_std"))]
    #[test]
    fn io_conversion_preserves_transport_errors_and_wraps_the_rest() {
        let io = std::io::Error::from(FramedEncodeError::Io(std::io::Error::from(
            std::io::ErrorKind::WouldBlock,
        )));
        assert_eq!(io.kind(), std::io::ErrorKind::WouldBlock);
        let other = std::io::Error::from(FramedEncodeError::OutputPending);
        assert_eq!(other.kind(), std::io::ErrorKind::Other);
        assert_eq!(other.to_string(), "framing output is pending");
    }
}

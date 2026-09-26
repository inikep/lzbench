use crate::EncodeError;
use thiserror::Error;
/// Failure to validate, encode, or deliver a container.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FramedEncodeError {
    /// Invalid owner configuration.
    #[error("invalid framed encoder configuration: {0}")]
    Config(#[from] crate::ConfigError),
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
    /// A wire size or offset cannot fit the RFC's 63-bit varint.
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
    pub fn into_error(self) -> FramedEncodeError {
        self.error
    }
}

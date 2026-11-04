//! What can go wrong while a frame is compressed in parallel.

use std::io;
use std::sync::Arc;

use thiserror::Error;

use crate::compressor::error::EncodeError;

/// Error of a parallel compression.
///
/// A failed batch reports [`Self::BatchFailed`] and a failed stream
/// [`Self::StreamFailed`], each with the failure that caused it as its
/// [`source`](std::error::Error::source). The cause is shared, so the source
/// downcasts to `Arc<ParallelEncodeError>`; matching on the variant's field
/// reaches it directly.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::{ParallelCompressor, ParallelConfig, ParallelEncodeError, TaskCount};
/// use mzstd::{InputSize, Operation};
///
/// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
/// let (mut session, _queue) = compressor.start(InputSize::Exact(4).into(), TaskCount::ONE);
/// let error = session.process(b"too long", &mut [], Operation::Continue).unwrap_err();
/// let ParallelEncodeError::StreamFailed(cause) = error else { unreachable!() };
/// assert_eq!(*cause, ParallelEncodeError::InputSizeMismatch { declared: 4, actual: 8 });
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Error, Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum ParallelEncodeError {
    /// A segment failed to compress.
    #[error("a segment failed to compress")]
    Encode(#[from] EncodeError),
    /// The input does not match the declared size.
    #[error("the stream was declared as {declared} bytes but received {actual}")]
    InputSizeMismatch {
        /// The declared size.
        declared: u64,
        /// The bytes received.
        actual: u64,
    },
    /// A batch task never reported, because it was leaked with
    /// [`core::mem::forget`] instead of being run or dropped.
    #[error("the batch cannot be assembled: a task never reported")]
    InvalidState,
    /// A task of a batch failed; the cause is the source.
    #[error("the parallel batch failed")]
    BatchFailed(#[source] Arc<ParallelEncodeError>),
    /// A streaming session failed, on this call or an earlier one; the cause
    /// is the source.
    #[error("the parallel stream failed")]
    StreamFailed(#[source] Arc<ParallelEncodeError>),
    /// Every handle of a stream's task queue was dropped while the stream
    /// still had a task to queue.
    #[error("every handle of the parallel task queue was dropped")]
    TaskQueueClosed,
    /// A task was dropped without running.
    #[error("the task for segment {segment} was dropped without running")]
    TaskAbandoned {
        /// The first segment of the task.
        segment: u64,
    },
    /// A task panicked; its engine was discarded.
    #[error("the task for segment {segment} panicked")]
    TaskPanicked {
        /// The first segment of the task.
        segment: u64,
    },
    /// The length of a [`RandomAccessSource`](super::RandomAccessSource)
    /// could not be read.
    #[error("the parallel source's length could not be read")]
    SourceMetadata(#[source] SourceIoError),
    /// Reading a [`RandomAccessSource`](super::RandomAccessSource) failed.
    #[error("reading segment {segment} of the parallel source failed")]
    SourceRead {
        /// The segment being read, or whose range of the source was being
        /// hashed for the checksum.
        segment: u64,
        /// The source's failure.
        #[source]
        source: SourceIoError,
    },
    /// A source's length or [identity](super::SourceIdentity) changed
    /// between preparing and finishing its batch.
    #[error("the parallel source changed during compression")]
    SourceChanged,
}

/// An I/O failure of a [`RandomAccessSource`](super::RandomAccessSource).
///
/// It shares the [`io::Error`], so [`ParallelEncodeError`] stays cloneable;
/// [`AsRef`] reaches it. Two values are equal when their kinds and messages
/// are.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::SourceIoError;
/// use std::io;
///
/// let error = SourceIoError::from(io::Error::from(io::ErrorKind::UnexpectedEof));
/// assert_eq!(error.as_ref().kind(), io::ErrorKind::UnexpectedEof);
/// assert_eq!(error.clone(), error);
/// ```
#[derive(Error, Debug, Clone)]
#[error(transparent)]
pub struct SourceIoError(Arc<io::Error>);

impl From<io::Error> for SourceIoError {
    fn from(error: io::Error) -> Self {
        Self(Arc::new(error))
    }
}

impl AsRef<io::Error> for SourceIoError {
    fn as_ref(&self) -> &io::Error {
        &self.0
    }
}

impl PartialEq for SourceIoError {
    fn eq(&self, other: &Self) -> bool {
        self.0.kind() == other.0.kind() && self.0.to_string() == other.0.to_string()
    }
}

impl Eq for SourceIoError {}

impl ParallelEncodeError {
    /// The [`io::ErrorKind`] this failure carries through an adapter.
    ///
    /// Never `Interrupted` or `WouldBlock`: `std::io` retries those, and a
    /// failed stream stays failed.
    fn io_kind(&self) -> io::ErrorKind {
        match self {
            Self::BatchFailed(cause) | Self::StreamFailed(cause) => cause.io_kind(),
            Self::Encode(EncodeError::OutputTooSmall { .. }) => io::ErrorKind::WriteZero,
            Self::Encode(
                EncodeError::InputSizeMismatch { .. }
                | EncodeError::Bound(_)
                | EncodeError::DictionaryMismatch,
            )
            | Self::InputSizeMismatch { .. } => io::ErrorKind::InvalidInput,
            Self::Encode(EncodeError::InvalidState | EncodeError::Internal)
            | Self::InvalidState
            | Self::TaskQueueClosed
            | Self::TaskAbandoned { .. }
            | Self::TaskPanicked { .. } => io::ErrorKind::Other,
            Self::SourceMetadata(error) | Self::SourceRead { source: error, .. } => {
                match error.as_ref().kind() {
                    io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock => io::ErrorKind::Other,
                    kind => kind,
                }
            }
            Self::SourceChanged => io::ErrorKind::InvalidData,
        }
    }
}

impl From<ParallelEncodeError> for io::Error {
    /// Carries a parallel failure through a [`std::io`] adapter.
    ///
    /// The kind is that of the root cause, looked up through
    /// [`ParallelEncodeError::StreamFailed`] and
    /// [`ParallelEncodeError::BatchFailed`]: a declared size the input
    /// contradicts is [`io::ErrorKind::InvalidInput`], a source failure keeps
    /// its own kind (except the retryable `Interrupted` and `WouldBlock`), a
    /// changed source is [`io::ErrorKind::InvalidData`], and failures of
    /// tasks and queues are [`io::ErrorKind::Other`]. The error itself stays the
    /// source, so the whole chain survives.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::compressor::parallel::ParallelEncodeError;
    /// use std::io::ErrorKind;
    /// use std::sync::Arc;
    ///
    /// let cause = ParallelEncodeError::InputSizeMismatch { declared: 4, actual: 8 };
    /// let error = std::io::Error::from(ParallelEncodeError::StreamFailed(Arc::new(cause)));
    /// assert_eq!(error.kind(), ErrorKind::InvalidInput);
    /// assert!(error.get_ref().unwrap().is::<ParallelEncodeError>());
    /// ```
    fn from(value: ParallelEncodeError) -> Self {
        Self::new(value.io_kind(), value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::config::SizeOverflow;
    use std::error::Error as _;

    fn kind(error: ParallelEncodeError) -> io::ErrorKind {
        io::Error::from(error).kind()
    }

    #[test]
    fn io_conversion_maps_every_variant_to_a_non_retryable_kind() {
        let cases = [
            (
                ParallelEncodeError::Encode(EncodeError::OutputTooSmall { provided: 1 }),
                io::ErrorKind::WriteZero,
            ),
            (
                ParallelEncodeError::Encode(EncodeError::InputSizeMismatch { declared: 1 }),
                io::ErrorKind::InvalidInput,
            ),
            (
                ParallelEncodeError::Encode(EncodeError::Bound(SizeOverflow)),
                io::ErrorKind::InvalidInput,
            ),
            (
                ParallelEncodeError::Encode(EncodeError::InvalidState),
                io::ErrorKind::Other,
            ),
            (
                ParallelEncodeError::Encode(EncodeError::Internal),
                io::ErrorKind::Other,
            ),
            (
                ParallelEncodeError::InputSizeMismatch {
                    declared: 1,
                    actual: 2,
                },
                io::ErrorKind::InvalidInput,
            ),
            (ParallelEncodeError::InvalidState, io::ErrorKind::Other),
            (ParallelEncodeError::TaskQueueClosed, io::ErrorKind::Other),
            (
                ParallelEncodeError::TaskAbandoned { segment: 0 },
                io::ErrorKind::Other,
            ),
            (
                ParallelEncodeError::TaskPanicked { segment: 0 },
                io::ErrorKind::Other,
            ),
            (
                ParallelEncodeError::SourceMetadata(
                    io::Error::from(io::ErrorKind::NotFound).into(),
                ),
                io::ErrorKind::NotFound,
            ),
            (
                ParallelEncodeError::SourceRead {
                    segment: 1,
                    source: io::Error::from(io::ErrorKind::UnexpectedEof).into(),
                },
                io::ErrorKind::UnexpectedEof,
            ),
            (
                ParallelEncodeError::SourceRead {
                    segment: 1,
                    source: io::Error::from(io::ErrorKind::Interrupted).into(),
                },
                io::ErrorKind::Other,
            ),
            (
                ParallelEncodeError::SourceMetadata(
                    io::Error::from(io::ErrorKind::WouldBlock).into(),
                ),
                io::ErrorKind::Other,
            ),
            (
                ParallelEncodeError::SourceChanged,
                io::ErrorKind::InvalidData,
            ),
        ];
        for (error, expected) in cases {
            let label = format!("{error:?}");
            let kind = kind(error);
            assert_eq!(kind, expected, "{label}");
            assert_ne!(kind, io::ErrorKind::Interrupted);
            assert_ne!(kind, io::ErrorKind::WouldBlock);
        }
    }

    #[test]
    fn io_conversion_takes_the_kind_of_a_wrapped_cause_and_keeps_the_chain() {
        let root = Arc::new(ParallelEncodeError::Encode(EncodeError::OutputTooSmall {
            provided: 3,
        }));
        let stream = ParallelEncodeError::StreamFailed(Arc::clone(&root));
        let batch = ParallelEncodeError::BatchFailed(Arc::new(stream));
        let error = io::Error::from(batch);
        assert_eq!(error.kind(), io::ErrorKind::WriteZero);
        let outer = error
            .get_ref()
            .and_then(|error| error.downcast_ref::<ParallelEncodeError>())
            .unwrap();
        assert!(matches!(outer, ParallelEncodeError::BatchFailed(_)));
        let middle = outer.source().unwrap();
        assert_eq!(middle.to_string(), "the parallel stream failed");
        // `Arc<T>` is the error the chain exposes.
        let cause = middle
            .source()
            .and_then(|error| error.downcast_ref::<Arc<ParallelEncodeError>>())
            .unwrap();
        assert_eq!(cause, &root);
    }

    #[test]
    fn messages_name_what_failed() {
        assert_eq!(
            ParallelEncodeError::TaskAbandoned { segment: 3 }.to_string(),
            "the task for segment 3 was dropped without running"
        );
        assert_eq!(
            ParallelEncodeError::TaskPanicked { segment: 4 }.to_string(),
            "the task for segment 4 panicked"
        );
        assert_eq!(
            ParallelEncodeError::TaskQueueClosed.to_string(),
            "every handle of the parallel task queue was dropped"
        );
        assert_eq!(
            ParallelEncodeError::BatchFailed(Arc::new(ParallelEncodeError::InvalidState))
                .to_string(),
            "the parallel batch failed"
        );
    }

    #[test]
    fn source_failures_keep_their_io_error_in_the_chain() {
        let error = ParallelEncodeError::SourceRead {
            segment: 7,
            source: io::Error::new(io::ErrorKind::PermissionDenied, "denied").into(),
        };
        assert_eq!(
            error.to_string(),
            "reading segment 7 of the parallel source failed"
        );
        let source = error.source().unwrap();
        assert_eq!(source.to_string(), "denied");
        let source = source.downcast_ref::<SourceIoError>().unwrap();
        assert_eq!(source.as_ref().kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(
            ParallelEncodeError::SourceMetadata(io::Error::other("gone").into()).to_string(),
            "the parallel source's length could not be read"
        );
        assert_eq!(
            ParallelEncodeError::SourceChanged.to_string(),
            "the parallel source changed during compression"
        );
    }

    #[test]
    fn source_io_errors_compare_by_kind_and_message() {
        let error = |kind, message| SourceIoError::from(io::Error::new(kind, message));
        let denied = error(io::ErrorKind::PermissionDenied, "denied");
        assert_eq!(denied, error(io::ErrorKind::PermissionDenied, "denied"));
        assert_ne!(denied, error(io::ErrorKind::NotFound, "denied"));
        assert_ne!(denied, error(io::ErrorKind::PermissionDenied, "other"));
    }
}

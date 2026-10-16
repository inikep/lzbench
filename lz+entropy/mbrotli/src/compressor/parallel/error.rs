//! Errors retain public codec and I/O sources without exposing private mechanics.
use super::{SegmentId, TaskId};
use crate::compressor::{ConfigError, EncodeError};
use std::io;

/// Invalid parallel or underlying serial configuration.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ParallelConfigError {
    /// Segment size is outside 64 KiB–16 MiB.
    #[error("invalid parallel segment size: {bytes}")]
    InvalidSegmentSize {
        /// Rejected byte count.
        bytes: usize,
    },
    /// Task count is zero or cannot fit a task ID.
    #[error("invalid parallel task count: {count}")]
    InvalidTaskCount {
        /// Rejected task count.
        count: usize,
    },
    /// The serial configuration is invalid.
    #[error("invalid encoder configuration: {0}")]
    Encoder(#[from] ConfigError),
    /// Large Window fragments are not yet supported.
    #[error("parallel compression currently requires a standard Brotli window")]
    UnsupportedParallelWindow,
}

/// Failure of planning, task execution, or output assembly.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ParallelEncodeError {
    /// A previously observed task failure; the typed cause remains in the chain.
    #[error("parallel batch failed: {0}")]
    BatchFailed(#[source] std::sync::Arc<ParallelEncodeError>),
    /// A streaming session failed earlier; the typed cause remains in the chain.
    #[error("parallel stream failed: {0}")]
    StreamFailed(#[source] std::sync::Arc<ParallelEncodeError>),
    /// Every task queue handle was dropped while the stream still had work.
    #[error("parallel task queue was closed by its executors")]
    TaskQueueClosed,
    /// Configuration validation failed.
    #[error("invalid parallel configuration: {0}")]
    Config(#[from] ParallelConfigError),
    /// A checked length, count, or memory estimate overflowed.
    #[error("parallel size calculation overflowed")]
    SizeOverflow,
    /// Fallible allocation was refused.
    #[error("parallel allocation failed: {0}")]
    Allocation(#[from] std::collections::TryReserveError),
    /// The worst-case staging bound exceeds the caller's limit.
    #[error("parallel memory staging bound exceeds configured limit")]
    MemoryStagingLimit,
    /// Aggregate temporary memory exceeds the caller's limit.
    #[error("parallel worker memory estimate exceeds configured limit")]
    WorkerMemoryLimit,
    /// Source metadata could not be read.
    #[error("source metadata failed: {0}")]
    SourceMetadata(#[source] io::Error),
    /// Source metadata differs from the planning snapshot.
    #[error("source changed during compression")]
    SourceChanged,
    /// A segment source read failed.
    #[error("source read failed for {task:?}, {segment:?}: {source}")]
    SourceRead {
        /// Owning task.
        task: TaskId,
        /// Input segment.
        segment: SegmentId,
        /// Original I/O failure.
        #[source]
        source: io::Error,
    },
    /// The worker could not encode its segment.
    #[error("encoding failed for {task:?}, {segment:?}: {source}")]
    Encode {
        /// Owning task.
        task: TaskId,
        /// Input segment.
        segment: SegmentId,
        /// Public encoder failure.
        #[source]
        source: EncodeError,
    },
    /// Private spool creation, writing, or reading failed.
    #[error("staging failed for {task:?}: {source}")]
    Staging {
        /// Owning task.
        task: TaskId,
        /// Original I/O failure.
        #[source]
        source: io::Error,
    },
    /// A caller dropped a task without running it.
    #[error("parallel task {task:?} was abandoned")]
    TaskAbandoned {
        /// Abandoned task.
        task: TaskId,
    },
    /// A worker or source panicked under unwinding; its workspace is discarded.
    #[error("parallel task {task:?} panicked")]
    TaskPanicked {
        /// Panicking task.
        task: TaskId,
    },
    /// Cancellation was requested.
    #[error("parallel batch was cancelled")]
    Cancelled,
    /// Task extraction has already occurred.
    #[error("parallel tasks were already taken")]
    TasksAlreadyTaken,
    /// Tasks have not been extracted or are still pending.
    #[error("parallel tasks are not ready")]
    NotReady,
    /// Private artifact metadata did not match the plan.
    #[error("parallel fragment invariant failed: {0}")]
    FragmentInvariant(&'static str),
    /// A generic writer accepted a prefix before failing.
    #[error("assembly failed after {bytes_written} bytes: {source}")]
    AssemblyIo {
        /// Exact bytes accepted by the destination.
        bytes_written: u64,
        /// Original I/O failure.
        #[source]
        source: io::Error,
    },
}

impl From<ParallelEncodeError> for io::Error {
    /// Carries a parallel failure through a [`std::io`] adapter.
    ///
    /// The kind is that of the root cause, looked up through
    /// [`ParallelEncodeError::StreamFailed`] and
    /// [`ParallelEncodeError::BatchFailed`]; the error itself stays the
    /// source, so the whole chain survives. Allocation failures and memory
    /// limits become [`io::ErrorKind::OutOfMemory`], and invalid configuration
    /// [`io::ErrorKind::InvalidInput`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::ParallelEncodeError;
    /// use std::io::ErrorKind;
    ///
    /// let error = std::io::Error::from(ParallelEncodeError::WorkerMemoryLimit);
    /// assert_eq!(error.kind(), ErrorKind::OutOfMemory);
    /// ```
    fn from(value: ParallelEncodeError) -> Self {
        Self::new(value.io_kind(), value)
    }
}
impl ParallelEncodeError {
    /// The [`io::ErrorKind`] this failure carries through an adapter.
    ///
    /// Never `Interrupted` or `WouldBlock`, even from an I/O cause: `std::io`
    /// retries those, and a failed stream stays failed.
    fn io_kind(&self) -> io::ErrorKind {
        match self {
            Self::BatchFailed(cause) | Self::StreamFailed(cause) => cause.io_kind(),
            Self::Encode { source, .. } => source.io_kind(),
            Self::Allocation(_) | Self::MemoryStagingLimit | Self::WorkerMemoryLimit => {
                io::ErrorKind::OutOfMemory
            }
            Self::Config(_) | Self::SizeOverflow => io::ErrorKind::InvalidInput,
            Self::SourceChanged => io::ErrorKind::InvalidData,
            Self::SourceMetadata(source)
            | Self::SourceRead { source, .. }
            | Self::Staging { source, .. }
            | Self::AssemblyIo { source, .. } => match source.kind() {
                io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock => io::ErrorKind::Other,
                kind => kind,
            },
            Self::TaskQueueClosed
            | Self::TaskAbandoned { .. }
            | Self::TaskPanicked { .. }
            | Self::Cancelled
            | Self::TasksAlreadyTaken
            | Self::NotReady
            | Self::FragmentInvariant(_) => io::ErrorKind::Other,
        }
    }
}

/// A consuming writer finish failure, preserving ownership and exact progress.
#[derive(Debug)]
pub struct ParallelFinishError<W> {
    /// The caller's writer, including any accepted output prefix.
    pub writer: W,
    /// Cause of failure.
    pub error: ParallelEncodeError,
    /// Exact number of bytes accepted by the writer.
    pub bytes_written: u64,
}
impl<W> std::fmt::Display for ParallelFinishError<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}
impl<W: std::fmt::Debug> std::error::Error for ParallelFinishError<W> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{error::Error as _, sync::Arc};

    fn kind(error: ParallelEncodeError) -> io::ErrorKind {
        io::Error::from(error).kind()
    }
    fn io_cause(kind: io::ErrorKind) -> io::Error {
        io::Error::new(kind, "cause")
    }

    #[test]
    fn io_conversion_maps_every_variant_to_a_non_retryable_kind() {
        let task = TaskId(1);
        let segment = SegmentId(2);
        let allocation = Vec::<u8>::new().try_reserve(usize::MAX).unwrap_err();
        let cases = [
            (
                ParallelEncodeError::Allocation(allocation),
                io::ErrorKind::OutOfMemory,
            ),
            (
                ParallelEncodeError::MemoryStagingLimit,
                io::ErrorKind::OutOfMemory,
            ),
            (
                ParallelEncodeError::WorkerMemoryLimit,
                io::ErrorKind::OutOfMemory,
            ),
            (
                ParallelEncodeError::Config(ParallelConfigError::InvalidTaskCount { count: 0 }),
                io::ErrorKind::InvalidInput,
            ),
            (
                ParallelEncodeError::SizeOverflow,
                io::ErrorKind::InvalidInput,
            ),
            (
                ParallelEncodeError::SourceChanged,
                io::ErrorKind::InvalidData,
            ),
            (
                ParallelEncodeError::Encode {
                    task,
                    segment,
                    source: EncodeError::AllocationFailed { requested: 8 },
                },
                io::ErrorKind::OutOfMemory,
            ),
            (
                ParallelEncodeError::SourceMetadata(io_cause(io::ErrorKind::NotFound)),
                io::ErrorKind::NotFound,
            ),
            (
                ParallelEncodeError::SourceRead {
                    task,
                    segment,
                    source: io_cause(io::ErrorKind::Interrupted),
                },
                io::ErrorKind::Other,
            ),
            (
                ParallelEncodeError::Staging {
                    task,
                    source: io_cause(io::ErrorKind::WouldBlock),
                },
                io::ErrorKind::Other,
            ),
            (
                ParallelEncodeError::AssemblyIo {
                    bytes_written: 3,
                    source: io_cause(io::ErrorKind::BrokenPipe),
                },
                io::ErrorKind::BrokenPipe,
            ),
            (ParallelEncodeError::TaskQueueClosed, io::ErrorKind::Other),
            (
                ParallelEncodeError::TaskAbandoned { task },
                io::ErrorKind::Other,
            ),
            (
                ParallelEncodeError::TaskPanicked { task },
                io::ErrorKind::Other,
            ),
            (ParallelEncodeError::Cancelled, io::ErrorKind::Other),
            (ParallelEncodeError::TasksAlreadyTaken, io::ErrorKind::Other),
            (ParallelEncodeError::NotReady, io::ErrorKind::Other),
            (
                ParallelEncodeError::FragmentInvariant("detail"),
                io::ErrorKind::Other,
            ),
        ];
        for (error, expected) in cases {
            let label = format!("{error:?}");
            assert_eq!(kind(error), expected, "{label}");
        }
    }

    #[test]
    fn io_conversion_takes_the_kind_of_a_wrapped_cause_and_keeps_the_chain() {
        let root = Arc::new(ParallelEncodeError::WorkerMemoryLimit);
        let stream = ParallelEncodeError::StreamFailed(Arc::clone(&root));
        let batch = ParallelEncodeError::BatchFailed(Arc::new(stream));
        let error = io::Error::from(batch);
        assert_eq!(error.kind(), io::ErrorKind::OutOfMemory);
        let inner = error
            .get_ref()
            .and_then(|e| e.downcast_ref::<ParallelEncodeError>())
            .unwrap();
        assert!(matches!(inner, ParallelEncodeError::BatchFailed(_)));
        assert!(inner.source().is_some());
    }
}

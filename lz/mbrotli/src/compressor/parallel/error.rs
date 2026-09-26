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

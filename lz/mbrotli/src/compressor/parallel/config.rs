//! Validated format and per-operation resource choices.
use super::ParallelConfigError;
use std::{num::NonZeroUsize, path::PathBuf};

/// Fixed segment size, independent of task count (64 KiB through 16 MiB).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SegmentSize(NonZeroUsize);
impl SegmentSize {
    /// Default segment size of four mebibytes.
    pub const DEFAULT: Self = Self(NonZeroUsize::new(4 << 20).unwrap());
    /// Number of input bytes in a full segment.
    pub const fn get(self) -> usize {
        self.0.get()
    }
}
impl TryFrom<usize> for SegmentSize {
    type Error = ParallelConfigError;
    /// Validates the inclusive 64 KiB–16 MiB range.
    /// # Errors
    /// Returns `InvalidSegmentSize` outside that range.
    fn try_from(bytes: usize) -> Result<Self, Self::Error> {
        match NonZeroUsize::new(bytes) {
            Some(n) if (64 << 10..=16 << 20).contains(&bytes) => Ok(Self(n)),
            _ => Err(ParallelConfigError::InvalidSegmentSize { bytes }),
        }
    }
}
impl Default for SegmentSize {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Requested number of tasks, from one through `u32::MAX`.
/// Large values are capped by the segment count and checked against memory limits.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TaskCount(NonZeroUsize);
impl TaskCount {
    /// One caller-run task.
    pub const ONE: Self = Self(NonZeroUsize::MIN);
    /// Number of requested tasks.
    pub const fn get(self) -> usize {
        self.0.get()
    }
    /// Queries the operating system's available parallelism.
    /// # Errors
    /// Propagates the operating system's query failure.
    pub fn available() -> std::io::Result<Self> {
        std::thread::available_parallelism().map(Self)
    }
}
impl TryFrom<usize> for TaskCount {
    type Error = ParallelConfigError;
    /// Validates a positive task count that fits a task ID.
    /// # Errors
    /// Returns `InvalidTaskCount` for zero or counts above `u32::MAX`.
    fn try_from(count: usize) -> Result<Self, Self::Error> {
        match NonZeroUsize::new(count) {
            Some(n) if u32::try_from(count).is_ok() => Ok(Self(n)),
            _ => Err(ParallelConfigError::InvalidTaskCount { count }),
        }
    }
}

/// How source immutability is checked before destination mutation.
/// Metadata checks cannot prove absence of every in-place write; applications
/// requiring stronger guarantees should supply an immutable snapshot source.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SourceConsistency {
    /// The caller guarantees unchanged bytes and length.
    AssumeImmutable,
    /// Compare source lengths before and after task execution.
    VerifyLength,
    /// Also compare identity metadata when the source provides it.
    VerifyLengthAndIdentity,
}

/// Deterministic segmentation and aggregate resource policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParallelConfig {
    pub(super) segment_size: SegmentSize,
    pub(super) minimum_parallel_size: u64,
    pub(super) aggregate_memory_limit: Option<usize>,
    pub(super) max_retained_workers: usize,
    pub(super) source_consistency: SourceConsistency,
}
impl From<SegmentSize> for ParallelConfig {
    fn from(segment_size: SegmentSize) -> Self {
        Self {
            segment_size,
            ..Self::default()
        }
    }
}
impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            segment_size: SegmentSize::DEFAULT,
            minimum_parallel_size: 8 << 20,
            aggregate_memory_limit: None,
            max_retained_workers: 0,
            source_consistency: SourceConsistency::VerifyLengthAndIdentity,
        }
    }
}
impl ParallelConfig {
    /// Uses serial encoding below this threshold when input fits one segment.
    /// Zero forces fragment encoding for every nonempty input.
    pub const fn with_minimum_parallel_size(mut self, bytes: u64) -> Self {
        self.minimum_parallel_size = bytes;
        self
    }
    /// Sets a conservative aggregate temporary-memory ceiling.
    pub const fn with_aggregate_memory_limit(mut self, bytes: Option<usize>) -> Self {
        self.aggregate_memory_limit = bytes;
        self
    }
    /// Sets the maximum number of idle workers retained after a batch.
    pub const fn with_max_retained_workers(mut self, count: usize) -> Self {
        self.max_retained_workers = count;
        self
    }
    /// Sets source metadata verification policy.
    pub const fn with_source_consistency(mut self, policy: SourceConsistency) -> Self {
        self.source_consistency = policy;
        self
    }
}

/// Maximum combined capacity approved for in-memory artifacts and descriptors.
#[derive(Clone, Debug)]
pub struct MemoryStaging {
    pub(super) max_total_bytes: usize,
}
impl From<usize> for MemoryStaging {
    fn from(max_total_bytes: usize) -> Self {
        Self { max_total_bytes }
    }
}
/// Directory under which private, automatically deleted task spools are created.
#[derive(Clone, Debug)]
pub struct DirectoryStaging {
    pub(super) directory: PathBuf,
}
impl From<PathBuf> for DirectoryStaging {
    fn from(directory: PathBuf) -> Self {
        Self { directory }
    }
}
/// Private intermediate storage. Directory mode keeps payload memory bounded.
#[derive(Clone, Debug)]
pub enum Staging {
    /// Per-task memory buffers with a preflight worst-case limit.
    Memory(MemoryStaging),
    /// One private spool per task, removed on success, failure, or drop.
    Directory(DirectoryStaging),
}
/// Scheduling and staging choices for one operation.
#[derive(Clone, Debug)]
pub struct BatchConfig {
    pub(super) task_count: TaskCount,
    pub(super) staging: Staging,
}
impl BatchConfig {
    /// Combines an explicit task count and storage policy.
    pub const fn new(task_count: TaskCount, staging: Staging) -> Self {
        Self {
            task_count,
            staging,
        }
    }
    /// Stages in memory using checked bounds derived from the source length and
    /// configured segment size during preparation. Effective task count is capped
    /// by the segment count. No caller-supplied staging ceiling is required.
    ///
    /// The aggregate memory limit in [`ParallelConfig`] still applies, including
    /// worker estimates for the effective task count. This does not query free
    /// system memory, choose a directory, or create threads.
    ///
    /// # Examples
    /// ```
    /// use mbrotli::EncoderConfig;
    /// use mbrotli::compressor::parallel::{
    ///     BatchConfig, ParallelCompressor, ParallelConfig, TaskCount,
    /// };
    /// let mut compressor = ParallelCompressor::new(
    ///     EncoderConfig::default(), ParallelConfig::default())?;
    /// let mut batch = compressor.prepare_slice(b"automatic staging",
    ///     BatchConfig::auto(TaskCount::ONE))?;
    /// batch.run_inline()?;
    /// let (output, _) = batch.finish_to_writer(Vec::new())?;
    /// assert!(!output.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn auto(task_count: TaskCount) -> Self {
        Self::memory(task_count, usize::MAX)
    }
    /// Stages in memory, rejecting worst-case bounds above `max_total_bytes`.
    /// The ceiling includes payload and descriptors; use
    /// [`super::ParallelSizeEstimate::maximum_staging_memory_bytes`] for the
    /// complete required bound.
    pub const fn memory(task_count: TaskCount, max_total_bytes: usize) -> Self {
        Self::new(
            task_count,
            Staging::Memory(MemoryStaging { max_total_bytes }),
        )
    }
    /// Stages under an existing directory using one temporary file per task.
    pub fn directory(task_count: TaskCount, directory: impl Into<PathBuf>) -> Self {
        Self::new(
            task_count,
            Staging::Directory(DirectoryStaging {
                directory: directory.into(),
            }),
        )
    }
}
/// Aggregate policy for idle codec workspaces.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ParallelRetentionPolicy {
    /// Keep at most the configured worker count.
    CurrentPlan,
    /// Also constrain combined retained bytes.
    Bounded {
        /// Maximum combined idle allocation size.
        max_bytes: usize,
    },
    /// Keep every current idle worker until another trim or batch completion.
    Aggressive,
    /// Release every idle worker.
    ReleaseAll,
}

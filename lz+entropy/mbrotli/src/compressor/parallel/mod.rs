//! Caller-scheduled compression of fixed independent segments into one stream.
//!
//! Tasks own their codec workspace and never wait for other tasks or for payload
//! draining. The caller chooses the executor. Finish only after submitted tasks
//! can run: blocking a one-thread pool's sole worker would prevent progress.
//! Parallel bytes differ from serial bytes, but are invariant under task count,
//! execution order, input source, staging backend, and supported SIMD backend.
//!
//! # Examples
//! ```
//! use mbrotli::{EncoderConfig, Quality};
//! use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, BatchConfig, TaskCount};
//! let mut compressor = ParallelCompressor::new(
//!     EncoderConfig::default().with_quality(Quality::Q5), ParallelConfig::default())?;
//! let input = b"caller-selected threads ".repeat(4000);
//! let mut batch = compressor.prepare_slice(&input, BatchConfig::memory(TaskCount::try_from(2)?, 1 << 20))?;
//! std::thread::scope(|scope| {
//!     for task in batch.take_tasks().unwrap() { scope.spawn(move || task.run()); }
//! });
//! let mut output = Vec::new();
//! let result = batch.finish_into(&mut output)?;
//! assert_eq!(result.stats.input_bytes, input.len() as u64);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod config;
mod core;
mod error;
mod source;

use crate::compressor::{Backend, EncoderConfig};
pub use config::{
    BatchConfig, DirectoryStaging, MemoryStaging, ParallelConfig, ParallelRetentionPolicy,
    SegmentSize, SourceConsistency, Staging, TaskCount,
};
pub use error::{ParallelConfigError, ParallelEncodeError, ParallelFinishError};
pub use source::{ArcBytesSource, FileSource, RandomAccessSource, SeekSource, SourceIdentity};
use std::{io::Write, ops::Range, sync::Arc, time::Duration};

/// Input-order identifier of a fixed segment.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SegmentId(u64);
impl From<SegmentId> for u64 {
    fn from(id: SegmentId) -> Self {
        id.0
    }
}
/// Input-order identifier of a contiguous task group.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TaskId(u32);
impl From<TaskId> for u32 {
    fn from(id: TaskId) -> Self {
        id.0
    }
}
/// Storage used for intermediate compressed bytes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StagingKind {
    /// Bounded in-memory artifacts.
    Memory,
    /// Private temporary files.
    Directory,
}
/// Structural and workspace diagnostics for a successful operation.
#[derive(Clone, Debug)]
pub struct ParallelStats {
    /// Original source length.
    pub input_bytes: u64,
    /// Exact destination bytes, including stream termination.
    pub output_bytes: u64,
    /// Configured fixed segment size.
    pub segment_size: usize,
    /// Number of nonempty segments.
    pub segment_count: u64,
    /// Caller-requested scheduling granularity.
    pub requested_tasks: usize,
    /// Number of task values generated.
    pub effective_tasks: usize,
    /// Whether canonical serial encoding was selected deterministically.
    pub serial_fallback: bool,
    /// Intermediate storage backend.
    pub staging_kind: StagingKind,
    /// Worst-case approved staged payload size.
    pub maximum_staged_bytes: u64,
    /// Total raw boundary-prefix input bytes.
    pub context_prefix_bytes: u64,
    /// Workers reused from earlier batches.
    pub workers_reused: usize,
    /// Workers created for this batch.
    pub workers_created: usize,
    /// Idle workspace bytes retained after completion.
    pub retained_worker_bytes: usize,
}
/// Appended destination range and successful operation statistics.
#[derive(Clone, Debug)]
pub struct ParallelOutput {
    /// Newly appended compressed stream; preexisting destination bytes survive.
    pub range: Range<usize>,
    /// Operation statistics.
    pub stats: ParallelStats,
}
/// Conservative preflight bounds; file-scale lengths remain `u64`.
#[derive(Clone, Debug)]
pub struct ParallelSizeEstimate {
    /// Source byte length.
    pub input_bytes: u64,
    /// Number of deterministic segments.
    pub segment_count: u64,
    /// Maximum staged payload size.
    pub maximum_staged_bytes: u64,
    /// Maximum complete output size.
    pub maximum_final_bytes: u64,
    /// Estimated active codec, source-buffer, and metadata allocation size.
    pub estimated_active_workspace_bytes: usize,
}
impl ParallelSizeEstimate {
    /// Returns the complete preflight memory-staging bound, including payload
    /// and segment descriptors, even when estimated with directory staging.
    /// Convert it with `usize::try_from` before passing it to [`BatchConfig::memory`].
    /// Worker and assembly storage remain subject to the separate aggregate limit.
    ///
    /// # Errors
    /// Returns [`ParallelEncodeError::SizeOverflow`] if the combined bound cannot
    /// fit in `u64`, including for manually constructed or modified estimates.
    pub const fn maximum_staging_memory_bytes(&self) -> Result<u64, ParallelEncodeError> {
        core::staging_memory_bound(self.maximum_staged_bytes, self.segment_count)
    }
}
/// Nonblocking task-completion state.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BatchPoll {
    /// Some tasks have not published their completion.
    Pending {
        /// Published task count.
        completed: usize,
        /// Planned task count.
        total: usize,
    },
    /// All tasks have completed successfully.
    Ready,
}
/// Result of a non-consuming timed wait.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WaitStatus {
    /// All tasks completed successfully.
    Ready,
    /// Deadline elapsed; tasks remain live and are not cancelled.
    TimedOut,
}

/// Reusable planner and private worker reservoir. No threads are created here.
/// # Examples
///
/// ## Thread scope
/// ```rust
/// use mbrotli::compressor::parallel::{
///     BatchConfig, ParallelCompressor, ParallelConfig, TaskCount,
/// };
/// use mbrotli::{EncoderConfig, Quality};
///
/// let input = "brotli ".repeat(10);
///
/// let tasks = TaskCount::available()?;
/// let config = EncoderConfig::default().with_quality(Quality::Q0);
///
/// let mut compressor = ParallelCompressor::new(config, ParallelConfig::default())?;
/// let mut prepared = compressor.prepare_slice(input.as_bytes(), BatchConfig::auto(tasks))?;
///
/// let tasks = prepared.take_tasks()?;
///
/// std::thread::scope(|scope| {
///     for task in tasks {
///         scope.spawn(move || task.run());
///     }
/// });
///
/// let mut output = Vec::new();
/// prepared.finish_into(&mut output)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Rayon
///
/// Add `rayon` to your dependencies to use its parallel iterator traits.
///
/// ```rust
/// use rayon::prelude::*;
/// use mbrotli::compressor::parallel::{
///     BatchConfig, ParallelCompressor, ParallelConfig, TaskCount,
/// };
/// use mbrotli::{EncoderConfig, Quality};
///
/// let input = "brotli ".repeat(10);
///
/// let tasks = TaskCount::try_from(rayon::current_num_threads().max(1))?;
/// let config = EncoderConfig::default().with_quality(Quality::Q0);
///
/// let mut compressor = ParallelCompressor::new(config, ParallelConfig::default())?;
/// let mut prepared = compressor.prepare_slice(input.as_bytes(), BatchConfig::auto(tasks))?;
///
/// prepared
///     .take_tasks()?
///     .into_par_iter()
///     .for_each(|task| task.run());
///
/// let mut output = Vec::new();
/// prepared.finish_into(&mut output)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ParallelCompressor {
    inner: core::Compressor,
}
impl ParallelCompressor {
    /// Validates configuration and detects the backend once; workspaces are lazy.
    /// # Errors
    /// Returns invalid serial configuration or unsupported Large Window errors.
    pub fn new(
        encoder: EncoderConfig,
        parallel: ParallelConfig,
    ) -> Result<Self, ParallelConfigError> {
        Self::with_backend(encoder, parallel, Backend::default())
    }
    /// Constructs a compressor pinned to a host-validated opaque backend.
    /// # Errors
    /// Returns invalid serial configuration or unsupported Large Window errors.
    pub fn with_backend(
        encoder: EncoderConfig,
        parallel: ParallelConfig,
        backend: Backend,
    ) -> Result<Self, ParallelConfigError> {
        core::Compressor::new(encoder, parallel, backend).map(|inner| Self { inner })
    }
    /// Serial algorithm settings shared by workers.
    pub const fn encoder_config(&self) -> &EncoderConfig {
        &self.inner.encoder
    }
    /// Replaces serial algorithm settings for subsequent batches.
    ///
    /// Changed settings release all idle workers; identical settings preserve
    /// them. The parallel configuration and selected backend are preserved.
    /// A live batch exclusively borrows this compressor, so reconfiguration is
    /// only available after the batch is finished or dropped.
    ///
    /// # Errors
    /// Returns the same configuration errors as [`Self::new`], including
    /// unsupported Large Window settings. On error, the compressor and its
    /// retained workers are unchanged.
    ///
    /// # Examples
    /// ```
    /// use mbrotli::{EncoderConfig, Quality};
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig};
    ///
    /// let mut compressor = ParallelCompressor::new(
    ///     EncoderConfig::default(), ParallelConfig::default())?;
    /// compressor.reconfigure(
    ///     EncoderConfig::default().with_quality(Quality::Q1))?;
    /// assert_eq!(compressor.encoder_config().quality(), Quality::Q1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reconfigure(&mut self, config: EncoderConfig) -> Result<(), ParallelConfigError> {
        self.inner.reconfigure(config)
    }
    /// Fixed segmentation and resource policy.
    pub const fn parallel_config(&self) -> &ParallelConfig {
        &self.inner.parallel
    }
    /// Replaces segmentation/resource policy and applies its retention ceiling.
    pub fn reconfigure_parallel(&mut self, config: ParallelConfig) {
        self.inner.parallel = config;
        self.inner.trim(ParallelRetentionPolicy::CurrentPlan);
    }
    /// Idle workers available for reuse.
    pub fn retained_worker_count(&self) -> usize {
        self.inner.workers.len()
    }
    /// Combined retained allocation capacities, excluding caller-owned data.
    pub fn retained_bytes(&self) -> usize {
        self.inner.retained_bytes()
    }
    /// Applies an aggregate policy to idle worker allocations.
    pub fn trim(&mut self, policy: ParallelRetentionPolicy) {
        self.inner.trim(policy);
    }
    /// Computes checked bounds without reading the source or creating workers.
    /// # Errors
    /// Returns overflow or configured resource-limit errors.
    pub fn estimate_source(
        &self,
        source_len: u64,
        config: &BatchConfig,
    ) -> Result<ParallelSizeEstimate, ParallelEncodeError> {
        self.inner.plan(source_len, config).map(|p| p.estimate)
    }
    /// Prepares tasks borrowing immutable input without copying the whole slice.
    /// # Errors
    /// Returns planning, allocation, or staging-creation errors.
    pub fn prepare_slice<'encoder, 'input>(
        &'encoder mut self,
        input: &'input [u8],
        config: BatchConfig,
    ) -> Result<ScopedParallelBatch<'encoder, 'input>, ParallelEncodeError> {
        core::Batch::prepare(&mut self.inner, core::Input::Slice(input), config)
            .map(|inner| ScopedParallelBatch { inner })
    }
    /// Prepares detached tasks from an owned source or shared source handle.
    /// Conversion into `Arc<S>` happens once, before task planning. Concrete
    /// sources and `Arc<dyn RandomAccessSource>` both use this entry point.
    /// Existing `Arc` handles and custom conversions may need explicit `S`, for
    /// example `prepare_source::<FileSource, _>(shared_file, config)`, because
    /// `Into<Arc<S>>` admits more than one conversion target.
    ///
    /// # Examples
    /// ```
    /// use std::io::Cursor;
    /// use mbrotli::EncoderConfig;
    /// use mbrotli::compressor::parallel::{
    ///     BatchConfig, ParallelCompressor, ParallelConfig, SeekSource, TaskCount,
    /// };
    /// let mut compressor = ParallelCompressor::new(
    ///     EncoderConfig::default(), ParallelConfig::default())?;
    /// let source = SeekSource::from(Cursor::new(b"generic input".to_vec()));
    /// let mut batch = compressor.prepare_source(source,
    ///     BatchConfig::memory(TaskCount::ONE, 4096))?;
    /// batch.run_inline()?;
    /// let (output, _) = batch.finish_to_writer(Vec::new())?;
    /// assert!(!output.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    /// # Errors
    /// Returns source metadata, planning, allocation, or staging errors.
    pub fn prepare_source<S, T>(
        &mut self,
        source: T,
        config: BatchConfig,
    ) -> Result<OwnedParallelBatch<'_>, ParallelEncodeError>
    where
        S: RandomAccessSource + ?Sized,
        T: Into<Arc<S>>,
    {
        let source = Arc::new(core::source::SharedSource(source.into()));
        core::Batch::prepare(&mut self.inner, core::Input::Source(source), config)
            .map(|inner| ScopedParallelBatch { inner })
    }
}

/// Single-use caller-schedulable task. Under `panic = "unwind"`, `run` catches
/// worker/source panics and reports them to the batch. Abort builds cannot recover.
#[must_use = "run the task or its batch will report abandonment"]
pub struct ScopedParallelTask<'input> {
    inner: core::Task<'input>,
}
/// A task owning only `'static` input handles, usable with detached spawners.
pub type OwnedParallelTask = ScopedParallelTask<'static>;
impl ScopedParallelTask<'_> {
    /// This task's stable input-order identifier.
    pub const fn id(&self) -> TaskId {
        self.inner.id
    }
    /// Fixed segments assigned to this task, in input order.
    pub fn segment_range(&self) -> Range<SegmentId> {
        SegmentId(self.inner.range.start)..SegmentId(self.inner.range.end)
    }
    /// Runs synchronously on the calling thread and publishes exactly one result.
    pub fn run(self) {
        self.inner.run();
    }
}

/// Coordinator exclusively borrowing its parent compressor. Drop cancels without
/// waiting for detached tasks; their remaining resources are released on completion.
pub struct ScopedParallelBatch<'encoder, 'input> {
    inner: core::Batch<'encoder, 'input>,
}
/// Coordinator for tasks with owned, `'static` source handles.
pub type OwnedParallelBatch<'encoder> = ScopedParallelBatch<'encoder, 'static>;
impl<'input> ScopedParallelBatch<'_, 'input> {
    /// Number of task values generated (zero for empty input).
    pub fn task_count(&self) -> usize {
        self.inner.plan.tasks
    }
    /// Number of nonempty source segments.
    pub fn segment_count(&self) -> u64 {
        self.inner.plan.estimate.segment_count
    }
    /// Transfers single-use tasks to the caller's executor.
    /// # Errors
    /// Returns `TasksAlreadyTaken` on repeated extraction.
    pub fn take_tasks(&mut self) -> Result<Vec<ScopedParallelTask<'input>>, ParallelEncodeError> {
        self.inner.take_tasks().map(|tasks| {
            tasks
                .into_iter()
                .map(|inner| ScopedParallelTask { inner })
                .collect()
        })
    }
    /// Executes all tasks on this thread, using exactly the external task path.
    /// # Errors
    /// Returns extraction, source, codec, staging, or cancellation failures.
    pub fn run_inline(&mut self) -> Result<(), ParallelEncodeError> {
        for task in self.take_tasks()? {
            task.run();
        }
        self.wait()
    }
    /// Observes completions without blocking.
    /// # Errors
    /// Returns deterministic task failure after all completions are collected.
    pub fn poll(&mut self) -> Result<BatchPoll, ParallelEncodeError> {
        self.inner.poll()
    }
    /// Waits for submitted tasks. The caller must keep their executor runnable.
    /// # Errors
    /// Returns `NotReady` for untaken tasks, or the task/source failure.
    pub fn wait(&mut self) -> Result<(), ParallelEncodeError> {
        self.inner.wait(None).map(|_| ())
    }
    /// Waits up to `timeout`; timeout does not cancel tasks.
    /// # Errors
    /// Returns `NotReady` for untaken tasks, or the task/source failure.
    pub fn wait_timeout(&mut self, timeout: Duration) -> Result<WaitStatus, ParallelEncodeError> {
        self.inner.wait(Some(timeout))
    }
    /// Requests cooperative cancellation. Repeated calls are harmless.
    pub fn cancel(&self) {
        self.inner.cancel();
    }
    /// Appends exactly one stream, restoring the original vector length on error.
    /// # Errors
    /// Returns task/source/validation, allocation, or assembly errors.
    pub fn finish_into(mut self, dst: &mut Vec<u8>) -> Result<ParallelOutput, ParallelEncodeError> {
        self.inner.finish_into(dst)
    }
    /// Assembles after validation; does not flush the writer or promise durability.
    /// A failure returns the writer and exact accepted-byte count; retry is consuming.
    /// # Errors
    /// Returns task/source/validation errors before writes, or partial I/O progress.
    pub fn finish_to_writer<W: Write>(
        mut self,
        writer: W,
    ) -> Result<(W, ParallelStats), ParallelFinishError<W>> {
        self.inner.finish_to_writer(writer)
    }
}

impl std::fmt::Debug for ParallelCompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelCompressor")
            .field("encoder", self.encoder_config())
            .field("parallel", self.parallel_config())
            .field("retained_workers", &self.retained_worker_count())
            .finish_non_exhaustive()
    }
}
impl std::fmt::Debug for ScopedParallelTask<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopedParallelTask")
            .field("id", &self.id())
            .field("segments", &self.segment_range())
            .finish_non_exhaustive()
    }
}
impl std::fmt::Debug for ScopedParallelBatch<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopedParallelBatch")
            .field("tasks", &self.task_count())
            .field("segments", &self.segment_count())
            .finish_non_exhaustive()
    }
}

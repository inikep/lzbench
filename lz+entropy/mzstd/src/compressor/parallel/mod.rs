//! Parallel compression of one frame with caller-run tasks.
//!
//! The input is cut into fixed segments, each compressed by its own task
//! into blocks of one frame; tasks are handed to the caller, who runs them on
//! whatever threads it likes. The frame is a standard Zstandard frame any
//! decoder accepts. Its bytes depend on the configuration and the segment
//! size, never on how many tasks there were or in which order they ran — but
//! they differ from the serial encoder's, and from the reference's
//! multithreaded mode.
//!
//! An input that fits in one segment is compressed serially by
//! [`ParallelCompressor::prepare_slice`] and matches
//! [`Compressor::compress`](crate::Compressor::compress).
//!
//! # Examples
//!
//! ```
//! use mzstd::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
//! use mzstd::EncoderConfig;
//!
//! let mut compressor = ParallelCompressor::new(EncoderConfig::default(), ParallelConfig::default())?;
//! let input = vec![7u8; 10 << 20];
//! let mut batch = compressor.prepare_slice(&input, TaskCount::try_from(4)?);
//! std::thread::scope(|scope| {
//!     for task in batch.take_tasks() {
//!         scope.spawn(move || task.run());
//!     }
//! });
//! let frame = batch.finish()?;
//! assert!(frame.len() < input.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod core;
mod error;
mod io;
mod session;

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use std::sync::Mutex;

pub use error::ParallelEncodeError;
pub use io::{ParallelEncoderReader, ParallelEncoderWriter};
pub use session::{
    OwnedParallelTask, ParallelEncoderSession, ParallelEncoderSessionOwned, ParallelEncoderStatus,
    ParallelProgress, ParallelTaskQueue, ParallelTasks,
};

use self::core::{EnginePool, FramePlan, compress_segment, compress_serial, lock};
use super::config::{ConfigError, EncoderConfig, Parameter};
use super::core::cctx::RequestedParams;
use super::error::EncodeError;
use super::internal::requested_params;
use crate::shared::xxhash::Xxh64;

/// Size of the segments a frame is cut into.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::SegmentSize;
///
/// assert_eq!(SegmentSize::DEFAULT.get(), 4 << 20);
/// assert!(SegmentSize::try_from(1000).is_err());
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SegmentSize(usize);

impl SegmentSize {
    /// The smallest segment, 64 KiB.
    pub const MIN: Self = Self(64 << 10);
    /// The largest segment, 256 MiB.
    pub const MAX: Self = Self(256 << 20);
    /// The default segment, 4 MiB.
    pub const DEFAULT: Self = Self(4 << 20);

    /// The size in bytes.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl Default for SegmentSize {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl TryFrom<usize> for SegmentSize {
    type Error = ConfigError;

    /// Validates a segment size.
    ///
    /// # Errors
    ///
    /// [`ConfigError::OutOfRange`] outside 64 KiB to 256 MiB.
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if !(Self::MIN.0..=Self::MAX.0).contains(&value) {
            return Err(ConfigError::OutOfRange {
                parameter: Parameter::SegmentSize,
                value: value as i64,
                min: Self::MIN.0 as i64,
                max: Self::MAX.0 as i64,
            });
        }
        Ok(Self(value))
    }
}

/// How many tasks a batch is split into, at most.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::TaskCount;
///
/// assert_eq!(TaskCount::try_from(8)?.get(), 8);
/// assert!(TaskCount::try_from(0).is_err());
/// # Ok::<(), mzstd::ConfigError>(())
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TaskCount(usize);

impl TaskCount {
    /// One task.
    pub const ONE: Self = Self(1);
    /// The largest task count.
    pub const MAX: Self = Self(1 << 16);

    /// The count.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl Default for TaskCount {
    fn default() -> Self {
        Self::ONE
    }
}

impl TryFrom<usize> for TaskCount {
    type Error = ConfigError;

    /// Validates a task count.
    ///
    /// # Errors
    ///
    /// [`ConfigError::OutOfRange`] outside `1..=65536`.
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if value == 0 || value > Self::MAX.0 {
            return Err(ConfigError::OutOfRange {
                parameter: Parameter::TaskCount,
                value: value as i64,
                min: 1,
                max: Self::MAX.0 as i64,
            });
        }
        Ok(Self(value))
    }
}

/// Configuration of parallel compression.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::{ParallelConfig, SegmentSize};
///
/// let config = ParallelConfig::default().with_segment_size(SegmentSize::try_from(1 << 20)?);
/// assert_eq!(config.segment_size().get(), 1 << 20);
/// # Ok::<(), mzstd::ConfigError>(())
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct ParallelConfig {
    segment_size: SegmentSize,
}

impl ParallelConfig {
    /// Sets the segment size.
    #[must_use]
    pub const fn with_segment_size(mut self, segment_size: SegmentSize) -> Self {
        self.segment_size = segment_size;
        self
    }

    /// The segment size.
    #[must_use]
    pub const fn segment_size(&self) -> SegmentSize {
        self.segment_size
    }
}

/// A compressor that splits a frame across caller-run tasks.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::{ParallelCompressor, ParallelConfig};
///
/// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
/// let frame = compressor.compress(b"small inputs are compressed serially")?;
/// assert_eq!(frame, mzstd::Compressor::new(Default::default())?.compress(b"small inputs are compressed serially")?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct ParallelCompressor {
    config: EncoderConfig,
    requested: RequestedParams,
    parallel: ParallelConfig,
    pool: Arc<EnginePool>,
}

impl ParallelCompressor {
    /// Creates a parallel compressor.
    ///
    /// # Errors
    ///
    /// [`ConfigError`] when a parameter is out of range.
    pub fn new(config: EncoderConfig, parallel: ParallelConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        Ok(Self {
            config,
            requested: requested_params(&config),
            parallel,
            pool: Arc::new(EnginePool::default()),
        })
    }

    /// The encoder configuration.
    #[must_use]
    pub const fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// The parallel configuration.
    #[must_use]
    pub const fn parallel_config(&self) -> &ParallelConfig {
        &self.parallel
    }

    /// Bytes retained by idle engines.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.pool.retained_bytes()
    }

    /// Releases every idle engine.
    pub fn release(&mut self) {
        self.pool.release();
    }

    /// Plans the compression of `input` into at most `tasks` tasks.
    ///
    /// Segments are grouped contiguously; the grouping does not change the
    /// frame.
    pub fn prepare_slice<'a>(
        &mut self,
        input: &'a [u8],
        tasks: TaskCount,
    ) -> ScopedParallelBatch<'a> {
        let segment = self.parallel.segment_size.get();
        let shared = Arc::new(BatchShared::default());
        let plan = FramePlan::new(&self.requested, Some(input.len() as u64));
        if input.len() <= segment {
            let task = ScopedParallelTask {
                work: Work::Serial {
                    requested: self.requested,
                    input,
                },
                index: 0,
                segment: 0,
                shared: Arc::clone(&shared),
                pool: Arc::clone(&self.pool),
                running: false,
                published: false,
            };
            return ScopedParallelBatch {
                tasks: vec![task],
                shared,
                expected: 1,
                serial: true,
                plan,
                input,
            };
        }
        let segments = input.len().div_ceil(segment);
        let task_count = tasks.get().min(segments);
        let per_task = segments / task_count;
        let extra = segments % task_count;
        let mut list = Vec::with_capacity(task_count);
        let mut first = 0;
        for t in 0..task_count {
            let count = per_task + usize::from(t < extra);
            list.push(ScopedParallelTask {
                work: Work::Segments {
                    plan,
                    input,
                    first,
                    count,
                    segment,
                },
                index: t,
                segment: first as u64,
                shared: Arc::clone(&shared),
                pool: Arc::clone(&self.pool),
                running: false,
                published: false,
            });
            first += count;
        }
        ScopedParallelBatch {
            tasks: list,
            shared,
            expected: task_count,
            serial: false,
            plan,
            input,
        }
    }

    /// Compresses `input`, running every task on the calling thread.
    ///
    /// # Errors
    ///
    /// [`ParallelEncodeError::BatchFailed`] when a segment fails.
    pub fn compress(&mut self, input: &[u8]) -> Result<Vec<u8>, ParallelEncodeError> {
        self.prepare_slice(input, TaskCount::ONE).finish()
    }
}

/// One task's result: its index and its blocks.
type Slot = (usize, Result<Vec<u8>, ParallelEncodeError>);

/// Results of a batch, one slot per task.
#[derive(Debug, Default)]
struct BatchShared {
    slots: Mutex<Vec<Slot>>,
}

/// What a scoped task compresses.
#[derive(Debug)]
enum Work<'a> {
    /// The whole input, serially.
    Serial {
        requested: RequestedParams,
        input: &'a [u8],
    },
    /// `count` segments starting at segment `first`.
    Segments {
        plan: FramePlan,
        input: &'a [u8],
        first: usize,
        count: usize,
        segment: usize,
    },
}

/// One unit of work of a batch; run it on any thread.
///
/// Dropping it unrun makes [`ScopedParallelBatch::finish`] report
/// [`ParallelEncodeError::TaskAbandoned`], and a panic while it runs
/// [`ParallelEncodeError::TaskPanicked`]; the panic itself still unwinds.
#[derive(Debug)]
pub struct ScopedParallelTask<'a> {
    work: Work<'a>,
    index: usize,
    /// The first segment the task covers.
    segment: u64,
    shared: Arc<BatchShared>,
    pool: Arc<EnginePool>,
    /// Whether compression has started, so an unwinding drop reports a panic.
    running: bool,
    published: bool,
}

impl ScopedParallelTask<'_> {
    /// Compresses this task's segments.
    pub fn run(mut self) {
        self.running = true;
        let mut engine = self.pool.take();
        let result = match &self.work {
            Work::Serial { requested, input } => compress_serial(&mut engine, requested, input),
            Work::Segments {
                plan,
                input,
                first,
                count,
                segment,
            } => {
                let overlap = plan.overlap();
                let mut out = Vec::new();
                let mut result = Ok(());
                for s in *first..first + count {
                    let start = s * segment;
                    let end = (start + segment).min(input.len());
                    let from = start.saturating_sub(overlap);
                    match compress_segment(
                        &mut engine,
                        plan,
                        &input[from..end],
                        start - from,
                        &mut out,
                    ) {
                        Ok(()) => {}
                        Err(error) => {
                            result = Err(EncodeError::from_core(error, 0, None));
                            break;
                        }
                    }
                }
                result.map(|()| out)
            }
        };
        // An engine that unwound is dropped instead of being returned here.
        self.pool.give(engine);
        self.publish(result.map_err(ParallelEncodeError::from));
    }

    fn publish(&mut self, result: Result<Vec<u8>, ParallelEncodeError>) {
        self.published = true;
        lock(&self.shared.slots).push((self.index, result));
    }
}

impl Drop for ScopedParallelTask<'_> {
    fn drop(&mut self) {
        if !self.published {
            let segment = self.segment;
            self.publish(Err(if self.running {
                ParallelEncodeError::TaskPanicked { segment }
            } else {
                ParallelEncodeError::TaskAbandoned { segment }
            }));
        }
    }
}

/// A planned compression of one input.
///
/// Tasks taken with [`Self::take_tasks`] should be run before
/// [`Self::finish`]; any that were not taken are run by `finish` on the
/// calling thread.
#[derive(Debug)]
pub struct ScopedParallelBatch<'a> {
    tasks: Vec<ScopedParallelTask<'a>>,
    shared: Arc<BatchShared>,
    expected: usize,
    serial: bool,
    plan: FramePlan,
    input: &'a [u8],
}

impl<'a> ScopedParallelBatch<'a> {
    /// Takes the tasks.
    pub fn take_tasks(&mut self) -> Vec<ScopedParallelTask<'a>> {
        ::core::mem::take(&mut self.tasks)
    }

    /// Assembles the frame.
    ///
    /// # Errors
    ///
    /// [`ParallelEncodeError::BatchFailed`] with the failure of the first
    /// failing task as its source: a segment that failed to compress
    /// ([`ParallelEncodeError::Encode`]), a task dropped without running
    /// ([`ParallelEncodeError::TaskAbandoned`]) or one that panicked
    /// ([`ParallelEncodeError::TaskPanicked`]).
    /// [`ParallelEncodeError::InvalidState`] when a task was leaked and never
    /// reported.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::compressor::parallel::{ParallelCompressor, ParallelConfig, ParallelEncodeError, TaskCount};
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let input = vec![5u8; 9 << 20];
    /// let mut batch = compressor.prepare_slice(&input, TaskCount::try_from(2)?);
    /// drop(batch.take_tasks());
    /// let Err(ParallelEncodeError::BatchFailed(cause)) = batch.finish() else { unreachable!() };
    /// assert_eq!(*cause, ParallelEncodeError::TaskAbandoned { segment: 0 });
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn finish(self) -> Result<Vec<u8>, ParallelEncodeError> {
        let mut out = Vec::new();
        self.finish_into(&mut out)?;
        Ok(out)
    }

    /// Appends the frame to `out`; on failure `out` is left as it was.
    ///
    /// # Errors
    ///
    /// As [`Self::finish`].
    pub fn finish_into(mut self, out: &mut Vec<u8>) -> Result<(), ParallelEncodeError> {
        for task in ::core::mem::take(&mut self.tasks) {
            task.run();
        }
        let mut slots = ::core::mem::take(&mut *lock(&self.shared.slots));
        if slots.len() != self.expected {
            return Err(ParallelEncodeError::InvalidState);
        }
        slots.sort_by_key(|(index, _)| *index);
        let mut parts = Vec::with_capacity(slots.len());
        for (_, result) in slots {
            match result {
                Ok(bytes) => parts.push(bytes),
                Err(error) => return Err(ParallelEncodeError::BatchFailed(Arc::new(error))),
            }
        }
        if self.serial {
            parts.iter().for_each(|part| out.extend_from_slice(part));
            return Ok(());
        }
        let header = self.plan.header();
        out.reserve(header.len() + parts.iter().map(Vec::len).sum::<usize>() + 7);
        out.extend_from_slice(&header);
        parts.iter().for_each(|part| out.extend_from_slice(part));
        let mut hasher = Xxh64::new(0);
        if self.plan.applied.fparams.checksum {
            hasher.update(self.input);
        }
        out.extend_from_slice(&self.plan.epilogue(&hasher));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_task_that_unwinds_reports_a_panic_to_its_batch() {
        let mut compressor =
            ParallelCompressor::new(EncoderConfig::default(), ParallelConfig::default()).unwrap();
        let mut batch = compressor.prepare_slice(b"input", TaskCount::ONE);
        let mut tasks = batch.take_tasks();
        // What an unwinding `run` leaves behind: started, never published.
        tasks[0].running = true;
        drop(tasks);
        assert_eq!(
            batch.finish(),
            Err(ParallelEncodeError::BatchFailed(Arc::new(
                ParallelEncodeError::TaskPanicked { segment: 0 }
            )))
        );
    }
}

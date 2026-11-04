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
//! [`ParallelCompressor::prepare_source`] plans the same frame over a
//! [`RandomAccessSource`] the tasks read their segments from, so they own
//! everything they touch and run on detached threads.
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
mod source;

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use std::sync::Mutex;

pub use error::{ParallelEncodeError, SourceIoError};
pub use io::{ParallelEncoderReader, ParallelEncoderWriter};
pub use session::{
    OwnedParallelTask, ParallelEncoderSession, ParallelEncoderSessionOwned, ParallelEncoderStatus,
    ParallelProgress, ParallelTaskQueue, ParallelTasks,
};
pub use source::{ArcBytesSource, FileSource, RandomAccessSource, SeekSource, SourceIdentity};

use self::core::{
    EnginePool, FramePlan, Input, SharedSource, compress_segment, compress_serial, lock,
};
use super::config::{ConfigError, EncoderConfig, Parameter};
use super::core::cctx::Engine;
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

    /// Plans the compression of a borrowed `input` into at most `tasks`
    /// tasks, without compressing or copying anything yet.
    ///
    /// The input is cut into segments of the configured
    /// [`SegmentSize`], and the segments are split into contiguous runs, one
    /// per task, their counts differing by at most one. There are never more
    /// tasks than segments. The frame depends on the segment size, never on
    /// `tasks` or on the order the tasks run in. An input that fits in one
    /// segment, empty input included, becomes a single task that compresses
    /// it serially, and the frame is byte-identical to
    /// [`Compressor::compress`](crate::Compressor::compress).
    ///
    /// The tasks borrow `input`, so they run within its lifetime, for
    /// example on [`std::thread::scope`] or a scoped pool. Take them with
    /// [`ScopedParallelBatch::take_tasks`], run each on any thread, then
    /// call [`ScopedParallelBatch::finish`]. `finish` runs the tasks that
    /// were never taken on the calling thread. Each task reads its segments
    /// in place, together with the overlap before them, so the input is
    /// never copied. For tasks that must outlive the input, use
    /// [`Self::prepare_source`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::compressor::parallel::{
    ///     ParallelCompressor, ParallelConfig, SegmentSize, TaskCount,
    /// };
    ///
    /// let config = ParallelConfig::default().with_segment_size(SegmentSize::MIN);
    /// let mut compressor = ParallelCompressor::new(Default::default(), config)?;
    /// let input = b"borrowed by scoped tasks ".repeat(10_000);
    ///
    /// // 250 000 bytes are four 64 KiB segments, so three tasks.
    /// let mut batch = compressor.prepare_slice(&input, TaskCount::try_from(3)?);
    /// let tasks = batch.take_tasks();
    /// assert_eq!(tasks.len(), 3);
    /// std::thread::scope(|scope| {
    ///     for task in tasks {
    ///         scope.spawn(move || task.run());
    ///     }
    /// });
    /// let frame = batch.finish()?;
    /// // The task count does not change the frame.
    /// assert_eq!(frame, compressor.compress(&input)?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn prepare_slice<'a>(
        &mut self,
        input: &'a [u8],
        tasks: TaskCount,
    ) -> ScopedParallelBatch<'a> {
        self.prepare(Input::Slice(input), input.len() as u64, tasks)
    }

    /// Plans the compression of an owned or shared source into at most
    /// `tasks` tasks that own everything they read, so they run on detached
    /// threads.
    ///
    /// The source is converted into an `Arc<S>` once, and its length and
    /// [identity](RandomAccessSource::identity) are recorded. Each task reads
    /// its segments and their overlap; a frame with a checksum reads the
    /// whole source once more when the batch is finished. The frame is
    /// byte-identical to [`Self::prepare_slice`] over the same bytes.
    ///
    /// Concrete sources and `Arc<dyn RandomAccessSource>` both enter here.
    /// An existing `Arc` or a custom conversion may need `S` named, for
    /// example `prepare_source::<FileSource, _>(shared_file, tasks)`,
    /// because `Into<Arc<S>>` admits more than one target.
    ///
    /// # Errors
    ///
    /// [`ParallelEncodeError::SourceMetadata`] when the source's length
    /// cannot be read.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::compressor::parallel::{
    ///     ParallelCompressor, ParallelConfig, SeekSource, SegmentSize, TaskCount,
    /// };
    /// use std::io::Cursor;
    ///
    /// let config = ParallelConfig::default().with_segment_size(SegmentSize::MIN);
    /// let mut compressor = ParallelCompressor::new(Default::default(), config)?;
    /// let input = b"read by detached tasks ".repeat(10_000);
    /// let source = SeekSource::from(Cursor::new(input.clone()));
    ///
    /// let mut batch = compressor.prepare_source(source, TaskCount::try_from(4)?)?;
    /// let workers: Vec<_> = batch
    ///     .take_tasks()
    ///     .into_iter()
    ///     .map(|task| std::thread::spawn(move || task.run()))
    ///     .collect();
    /// workers.into_iter().for_each(|worker| worker.join().unwrap());
    /// assert_eq!(batch.finish()?, compressor.compress(&input)?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn prepare_source<S, T>(
        &mut self,
        source: T,
        tasks: TaskCount,
    ) -> Result<OwnedParallelBatch, ParallelEncodeError>
    where
        S: RandomAccessSource + ?Sized,
        T: Into<Arc<S>>,
    {
        let input = Input::Source(Arc::new(SharedSource(source.into())));
        let len = input.len()?;
        Ok(self.prepare(input, len, tasks))
    }

    /// Plans `len` bytes of `input` into at most `tasks` tasks.
    fn prepare<'a>(&self, input: Input<'a>, len: u64, tasks: TaskCount) -> ScopedParallelBatch<'a> {
        let segment = self.parallel.segment_size.get();
        let shared = Arc::new(BatchShared::default());
        let plan = FramePlan::new(&self.requested, Some(len));
        let identity = input.identity();
        let task = |work, index, segment| ScopedParallelTask {
            work,
            index,
            segment,
            shared: Arc::clone(&shared),
            pool: Arc::clone(&self.pool),
            running: false,
            published: false,
        };
        let serial = len <= segment as u64;
        let list = if serial {
            let work = Work::Serial {
                requested: self.requested,
                input: input.clone(),
                len,
            };
            vec![task(work, 0, 0)]
        } else {
            let segments = len.div_ceil(segment as u64);
            // At most `TaskCount::MAX` tasks, so the count fits a `usize`.
            let task_count = (tasks.get() as u64).min(segments);
            let per_task = segments / task_count;
            let extra = segments % task_count;
            let mut first = 0;
            (0..task_count)
                .map(|t| {
                    let count = per_task + u64::from(t < extra);
                    let work = Work::Segments {
                        plan,
                        input: input.clone(),
                        len,
                        first,
                        count,
                        segment,
                    };
                    let task = task(work, t as usize, first);
                    first += count;
                    task
                })
                .collect()
        };
        ScopedParallelBatch {
            expected: list.len(),
            tasks: list,
            shared,
            serial,
            plan,
            input,
            len,
            identity,
            segment,
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
    /// The whole input of `len` bytes, serially.
    Serial {
        requested: RequestedParams,
        input: Input<'a>,
        len: u64,
    },
    /// `count` segments of `segment` bytes starting at segment `first`, of
    /// an input of `len` bytes.
    Segments {
        plan: FramePlan,
        input: Input<'a>,
        len: u64,
        first: u64,
        count: u64,
        segment: usize,
    },
}

impl Work<'_> {
    /// Compresses the work with `engine`: a whole frame, or the segments'
    /// blocks.
    ///
    /// # Errors
    ///
    /// [`ParallelEncodeError::SourceRead`] when a source fails, and
    /// [`ParallelEncodeError::Encode`] when a segment fails to compress.
    fn compress(&self, engine: &mut Engine) -> Result<Vec<u8>, ParallelEncodeError> {
        let mut buffer = Vec::new();
        match self {
            Self::Serial {
                requested,
                input,
                len,
            } => {
                let data = input.read(0..*len, 0, &mut buffer)?;
                Ok(compress_serial(engine, requested, data)?)
            }
            Self::Segments {
                plan,
                input,
                len,
                first,
                count,
                segment,
            } => {
                let overlap = plan.overlap() as u64;
                let segment = *segment as u64;
                let mut out = Vec::new();
                for s in *first..first + count {
                    let start = s * segment;
                    let end = (start + segment).min(*len);
                    let from = start.saturating_sub(overlap);
                    let data = input.read(from..end, s, &mut buffer)?;
                    compress_segment(engine, plan, data, (start - from) as usize, &mut out)
                        .map_err(|error| EncodeError::from_core(error, 0, None))?;
                }
                Ok(out)
            }
        }
    }
}

/// One unit of work of a batch; run it on any thread.
///
/// Dropping it unrun makes [`ScopedParallelBatch::finish`] report
/// [`ParallelEncodeError::TaskAbandoned`], and a panic while it runs,
/// including one inside a [`RandomAccessSource`],
/// [`ParallelEncodeError::TaskPanicked`]; the panic itself still unwinds.
/// The tasks of [`ParallelCompressor::prepare_source`] borrow nothing and
/// are `'static`.
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
    /// Compresses this task's segments, reading them first from a source.
    pub fn run(mut self) {
        self.running = true;
        let mut engine = self.pool.take();
        let result = self.work.compress(&mut engine);
        // An engine that unwound is dropped instead of being returned here.
        self.pool.give(engine);
        self.publish(result);
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
    input: Input<'a>,
    /// The input's length when the batch was planned.
    len: u64,
    /// A source's identity when the batch was planned.
    identity: Option<SourceIdentity>,
    segment: usize,
}

/// A batch over an owned source, from
/// [`ParallelCompressor::prepare_source`]; its tasks are `'static`.
pub type OwnedParallelBatch = ScopedParallelBatch<'static>;

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
    /// ([`ParallelEncodeError::Encode`]), a source that failed to read it
    /// ([`ParallelEncodeError::SourceRead`]), a task dropped without running
    /// ([`ParallelEncodeError::TaskAbandoned`]) or one that panicked
    /// ([`ParallelEncodeError::TaskPanicked`]).
    /// [`ParallelEncodeError::InvalidState`] when a task was leaked and never
    /// reported. For a source, [`ParallelEncodeError::SourceChanged`] when
    /// its length or identity differs from when the batch was planned,
    /// [`ParallelEncodeError::SourceMetadata`] when its length cannot be
    /// read, and [`ParallelEncodeError::SourceRead`] when reading it for the
    /// checksum fails.
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
        self.input.verify(self.len, self.identity.as_ref())?;
        if self.serial {
            parts.iter().for_each(|part| out.extend_from_slice(part));
            return Ok(());
        }
        // Hashed before `out` is touched: reading a source may fail.
        let mut hasher = Xxh64::new(0);
        if self.plan.applied.fparams.checksum {
            self.input.hash(self.len, self.segment, &mut hasher)?;
        }
        let header = self.plan.header();
        out.reserve(header.len() + parts.iter().map(Vec::len).sum::<usize>() + 7);
        out.extend_from_slice(&header);
        parts.iter().for_each(|part| out.extend_from_slice(part));
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

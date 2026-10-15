//! Incremental parallel streams: input arrives in chunks, segments become
//! tasks.
//!
//! [`ParallelCompressor::start`] returns a session and a
//! [`ParallelTaskQueue`]. Every full segment the session accepts becomes a
//! task on that queue, and the caller's executors take and run them. The
//! session delivers finished parts in input order, so output can be written
//! while later segments still compress.

use core::marker::PhantomData;
use core::mem::MaybeUninit;

use super::core::{QueueHandle, Stream, StreamTask};
use super::{ParallelCompressor, ParallelEncodeError, TaskCount};
use crate::compressor::session::{InputSize, Operation, StreamConfig};

/// What a parallel session needs next.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::ParallelEncoderStatus;
///
/// assert_ne!(ParallelEncoderStatus::Pending, ParallelEncoderStatus::NeedsInput);
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ParallelEncoderStatus {
    /// The operation has done all it can; more input or a new operation is
    /// needed.
    NeedsInput,
    /// Compressed bytes are waiting; call again with room for them.
    NeedsOutput,
    /// Queued tasks must complete first: keep executors running, call
    /// [`ParallelEncoderSession::wait`], then call again.
    Pending,
    /// The frame is complete and delivered.
    Finished,
}

/// What one parallel `process` call did.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::{ParallelEncoderStatus, ParallelProgress};
///
/// let progress = ParallelProgress {
///     consumed: 4,
///     produced: 0,
///     status: ParallelEncoderStatus::NeedsInput,
/// };
/// assert_eq!(progress.consumed, 4);
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ParallelProgress {
    /// Bytes taken from the input.
    pub consumed: usize,
    /// Bytes written to the output.
    pub produced: usize,
    /// What the session needs next.
    pub status: ParallelEncoderStatus,
}

/// A segment compression queued by a streaming session; run it on any
/// thread.
///
/// Dropping it unrun fails the stream with
/// [`ParallelEncodeError::TaskAbandoned`], and a panic while it runs with
/// [`ParallelEncodeError::TaskPanicked`], so the session never waits for it
/// forever.
#[derive(Debug)]
pub struct OwnedParallelTask {
    inner: StreamTask,
}

impl OwnedParallelTask {
    /// Compresses the segment.
    pub fn run(self) {
        self.inner.run();
    }
}

/// Where a streaming session's tasks wait to be run.
///
/// Clone one handle per executor. [`Self::recv`] blocks until a task is
/// queued and returns `None` once the session has finished, failed or been
/// dropped, so `for task in queue { task.run() }` is a complete executor
/// loop.
///
/// Dropping every handle abandons the tasks still queued: the session then
/// fails with [`ParallelEncodeError::TaskAbandoned`] or
/// [`ParallelEncodeError::TaskQueueClosed`] instead of waiting forever. On a
/// single thread, run queued tasks with [`Self::next_task`] before waiting.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
/// use mzstd::Operation;
///
/// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
/// let (mut session, queue) = compressor.start(Default::default(), TaskCount::ONE);
/// let mut output = [0u8; 64];
/// session.process(b"queued on one thread", &mut output, Operation::End)?;
/// while let Some(task) = queue.next_task() {
///     task.run();
/// }
/// session.wait()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ParallelTaskQueue {
    inner: QueueHandle,
}

impl ParallelTaskQueue {
    /// Blocks until a task is queued; returns `None` once the session has
    /// ended.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (session, queue) = compressor.start(Default::default(), TaskCount::ONE);
    /// drop(session);
    /// assert!(queue.recv().is_none());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn recv(&self) -> Option<OwnedParallelTask> {
        self.inner.recv().map(|inner| OwnedParallelTask { inner })
    }

    /// Takes the next queued task without blocking, if any.
    #[must_use]
    pub fn next_task(&self) -> Option<OwnedParallelTask> {
        self.inner
            .try_recv()
            .map(|inner| OwnedParallelTask { inner })
    }

    /// Number of tasks queued and not yet taken.
    #[must_use]
    pub fn queued(&self) -> usize {
        self.inner.queued()
    }
}

impl IntoIterator for ParallelTaskQueue {
    type Item = OwnedParallelTask;
    type IntoIter = ParallelTasks;

    /// Yields queued tasks until the session ends, blocking between them.
    fn into_iter(self) -> ParallelTasks {
        ParallelTasks { queue: self }
    }
}

/// Blocking iterator over a [`ParallelTaskQueue`], ending with the session.
#[derive(Debug)]
pub struct ParallelTasks {
    queue: ParallelTaskQueue,
}

impl Iterator for ParallelTasks {
    type Item = OwnedParallelTask;

    fn next(&mut self) -> Option<Self::Item> {
        self.queue.recv()
    }
}

macro_rules! session_methods {
    () => {
        /// Moves the frame forward by one step without blocking.
        ///
        /// Takes what fits the window from `input`, delivers finished parts
        /// into `output` in input order, and reports what it needs next:
        ///
        /// - [`ParallelEncoderStatus::NeedsOutput`]: delivered bytes are
        ///   waiting.
        /// - [`ParallelEncoderStatus::Pending`]: input is left over because
        ///   the window is full, or a `Flush` or `End` waits for queued
        ///   tasks.
        /// - [`ParallelEncoderStatus::NeedsInput`]: everything this
        ///   operation can do is done. A `Continue` call may still have
        ///   tasks running.
        /// - [`ParallelEncoderStatus::Finished`]: the frame was delivered;
        ///   later calls consume and produce nothing.
        ///
        /// A `Flush` reports `NeedsInput` only once everything accepted has
        /// been delivered. Once an `End` has queued the last part, later
        /// calls consume no input whatever their operation, and only
        /// deliver.
        ///
        /// # Errors
        ///
        /// [`ParallelEncodeError::StreamFailed`] with the cause as its source
        /// when the input contradicts a declared size, a task fails, or the
        /// queue is closed, and on every later call.
        ///
        /// # Examples
        ///
        /// ```
        /// use mzstd::compressor::parallel::{
        ///     ParallelCompressor, ParallelConfig, ParallelEncoderStatus, TaskCount,
        /// };
        /// use mzstd::Operation;
        ///
        /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
        /// let (mut session, queue) = compressor.start(Default::default(), TaskCount::ONE);
        /// let mut output = [0u8; 64];
        ///
        /// let progress = session.process(b"accepted", &mut output, Operation::Continue)?;
        /// assert_eq!(progress.consumed, 8);
        /// assert_eq!(progress.status, ParallelEncoderStatus::NeedsInput);
        ///
        /// let progress = session.process(b"", &mut output, Operation::End)?;
        /// assert_eq!(progress.status, ParallelEncoderStatus::Pending);
        /// queue.next_task().unwrap().run();
        /// session.wait()?;
        /// let progress = session.process(b"", &mut output, Operation::End)?;
        /// assert_eq!(progress.status, ParallelEncoderStatus::Finished);
        /// # Ok::<(), Box<dyn std::error::Error>>(())
        /// ```
        pub fn process(
            &mut self,
            input: &[u8],
            output: &mut [u8],
            operation: Operation,
        ) -> Result<ParallelProgress, ParallelEncodeError> {
            self.stream.process(input, output, operation)
        }

        /// [`Self::process`] into uninitialized memory, such as
        /// [`Vec::spare_capacity_mut`], without zeroing it first.
        ///
        /// It produces the same bytes, `consumed` and `produced` counts,
        /// statuses and errors as [`Self::process`], and calls of the two
        /// may be mixed on one frame. On success exactly
        /// `output[..produced]` is initialized and no byte after it is
        /// written; after an error, treat none of `output` as initialized.
        ///
        /// # Errors
        ///
        /// As [`Self::process`].
        ///
        /// # Examples
        ///
        /// ```
        /// use mzstd::compressor::parallel::{
        ///     ParallelCompressor, ParallelConfig, ParallelEncoderStatus, TaskCount,
        /// };
        /// use mzstd::Operation;
        ///
        /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
        /// let (mut session, queue) = compressor.start(Default::default(), TaskCount::ONE);
        /// let mut frame: Vec<u8> = Vec::with_capacity(64);
        /// let mut input = &b"delivered into spare capacity"[..];
        /// loop {
        ///     if frame.len() == frame.capacity() {
        ///         frame.reserve(64);
        ///     }
        ///     let progress =
        ///         session.process_uninit(input, frame.spare_capacity_mut(), Operation::End)?;
        ///     input = &input[progress.consumed..];
        ///     // SAFETY: `process_uninit` initialized the first `produced` spare bytes.
        ///     unsafe { frame.set_len(frame.len() + progress.produced) };
        ///     match progress.status {
        ///         ParallelEncoderStatus::Finished => break,
        ///         ParallelEncoderStatus::Pending => {
        ///             while let Some(task) = queue.next_task() {
        ///                 task.run();
        ///             }
        ///             session.wait()?;
        ///         }
        ///         _ => {}
        ///     }
        /// }
        /// assert_eq!(frame, mzstd::Compressor::default().compress(b"delivered into spare capacity")?);
        /// # Ok::<(), Box<dyn std::error::Error>>(())
        /// ```
        pub fn process_uninit(
            &mut self,
            input: &[u8],
            output: &mut [MaybeUninit<u8>],
            operation: Operation,
        ) -> Result<ParallelProgress, ParallelEncodeError> {
            self.stream.process(input, output, operation)
        }

        /// Makes everything accepted so far decodable, without taking input.
        ///
        /// Shorthand for [`Self::process`] with empty input and
        /// [`Operation::Flush`]. Repeat it, waiting on `Pending`, until it
        /// reports [`ParallelEncoderStatus::NeedsInput`]. A flush ends the
        /// current segment early, which moves every later segment boundary.
        ///
        /// # Errors
        ///
        /// As [`Self::process`].
        ///
        /// # Examples
        ///
        /// ```
        /// use mzstd::compressor::parallel::{
        ///     ParallelCompressor, ParallelConfig, ParallelEncoderStatus, TaskCount,
        /// };
        /// use mzstd::Operation;
        ///
        /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
        /// let (mut session, queue) = compressor.start(Default::default(), TaskCount::ONE);
        /// let mut output = [0u8; 64];
        /// session.process(b"flushed", &mut output, Operation::Continue)?;
        /// // The frame header is ready at once; the segment is queued.
        /// assert_eq!(session.flush(&mut output)?.status, ParallelEncoderStatus::Pending);
        /// queue.next_task().unwrap().run();
        /// session.wait()?;
        /// let progress = session.flush(&mut output)?;
        /// assert_eq!(progress.status, ParallelEncoderStatus::NeedsInput);
        /// assert!(progress.produced > 0);
        /// # Ok::<(), Box<dyn std::error::Error>>(())
        /// ```
        pub fn flush(
            &mut self,
            output: &mut [u8],
        ) -> Result<ParallelProgress, ParallelEncodeError> {
            self.process(&[], output, Operation::Flush)
        }

        /// Ends the frame, without taking input.
        ///
        /// Shorthand for [`Self::process`] with empty input and
        /// [`Operation::End`]. Repeat it, waiting on `Pending`, until it
        /// reports [`ParallelEncoderStatus::Finished`].
        ///
        /// # Errors
        ///
        /// As [`Self::process`].
        ///
        /// # Examples
        ///
        /// ```
        /// use mzstd::compressor::parallel::{
        ///     ParallelCompressor, ParallelConfig, ParallelEncoderStatus, TaskCount,
        /// };
        ///
        /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
        /// let (mut session, _queue) = compressor.start(Default::default(), TaskCount::ONE);
        /// let mut output = [0u8; 64];
        /// // An empty frame is compressed inline; no task is queued.
        /// assert_eq!(session.finish(&mut output)?.status, ParallelEncoderStatus::Finished);
        /// assert!(session.is_finished());
        /// # Ok::<(), Box<dyn std::error::Error>>(())
        /// ```
        pub fn finish(
            &mut self,
            output: &mut [u8],
        ) -> Result<ParallelProgress, ParallelEncodeError> {
            self.process(&[], output, Operation::End)
        }

        /// Blocks until a queued task completes; returns at once when one has
        /// completed since the session last looked, or when none is queued
        /// or running.
        ///
        /// The caller's executors must be able to run tasks meanwhile.
        ///
        /// # Errors
        ///
        /// As [`Self::process`].
        ///
        /// # Examples
        ///
        /// ```
        /// use mzstd::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
        ///
        /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
        /// let (mut session, _queue) = compressor.start(Default::default(), TaskCount::ONE);
        /// session.wait()?; // Nothing is queued.
        /// # Ok::<(), Box<dyn std::error::Error>>(())
        /// ```
        pub fn wait(&mut self) -> Result<(), ParallelEncodeError> {
            self.stream.wait()
        }

        /// Whether the frame has been completed and delivered.
        #[must_use]
        pub const fn is_finished(&self) -> bool {
            self.stream.is_finished()
        }
    };
}

/// One parallel frame whose input arrives in chunks, borrowing its
/// compressor.
///
/// Created by [`ParallelCompressor::start`], which also returns the
/// [`ParallelTaskQueue`] its tasks go to. The session copies input into
/// fixed segments; each full segment is queued as a task, and at most
/// `window` parts are queued, running or waiting for delivery at once. When
/// the window is full the session takes no more input and reports
/// [`ParallelEncoderStatus::Pending`].
///
/// With a declared input size and without [`Operation::Flush`], the frame is
/// byte-identical to [`ParallelCompressor::prepare_slice`] over the same
/// bytes, however the input is chunked and whatever the window, including
/// the serial fallback for inputs that fit in one segment (which holds
/// without a declared size too). A flush ends the current segment early,
/// which moves every later segment boundary.
///
/// `process` never blocks. [`Self::wait`] blocks until a queued task
/// completes, so executors must be able to run while the caller waits.
/// Finishing, failing or dropping the session closes the queue, which ends
/// every executor's `for task in queue` loop; dropping it unfinished
/// abandons the frame and skips tasks that have not started.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::{
///     ParallelCompressor, ParallelConfig, ParallelEncoderStatus, SegmentSize, TaskCount,
/// };
/// use mzstd::{InputSize, Operation};
///
/// let config = ParallelConfig::default().with_segment_size(SegmentSize::MIN);
/// let mut compressor = ParallelCompressor::new(Default::default(), config)?;
/// let payload = b"streamed into parallel segments ".repeat(10_000);
///
/// let mut frame = Vec::new();
/// std::thread::scope(|scope| {
///     let stream = InputSize::Exact(payload.len() as u64).into();
///     let (mut session, queue) = compressor.start(stream, TaskCount::try_from(4)?);
///     for _ in 0..4 {
///         let queue = queue.clone();
///         scope.spawn(move || queue.into_iter().for_each(|task| task.run()));
///     }
///     drop(queue);
///
///     let mut buffer = [0u8; 4096];
///     let mut input = payload.as_slice();
///     loop {
///         let progress = session.process(input, &mut buffer, Operation::End)?;
///         input = &input[progress.consumed..];
///         frame.extend_from_slice(&buffer[..progress.produced]);
///         match progress.status {
///             ParallelEncoderStatus::Finished => break,
///             ParallelEncoderStatus::Pending => session.wait()?,
///             _ => {}
///         }
///     }
///     Ok::<(), Box<dyn std::error::Error>>(())
/// })?;
///
/// assert_eq!(frame, compressor.compress(&payload)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct ParallelEncoderSession<'c> {
    stream: Stream,
    compressor: PhantomData<&'c mut ParallelCompressor>,
}

impl ParallelEncoderSession<'_> {
    session_methods!();
}

/// One parallel frame whose input arrives in chunks, owning its compressor.
///
/// Created by [`ParallelCompressor::into_session`]. It runs the same state
/// machine as [`ParallelEncoderSession`], so the same calls produce the same
/// bytes, counts, statuses and errors, but carries no lifetime.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::{
///     ParallelCompressor, ParallelConfig, ParallelEncoderStatus, TaskCount,
/// };
/// use mzstd::Operation;
///
/// let compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
/// let (mut session, queue) = compressor.into_session(Default::default(), TaskCount::ONE);
/// let executor = std::thread::spawn(move || queue.into_iter().for_each(|task| task.run()));
///
/// let mut output = [0u8; 64];
/// let mut input = &b"owned parallel frame"[..];
/// loop {
///     let progress = session.process(input, &mut output, Operation::End)?;
///     input = &input[progress.consumed..];
///     match progress.status {
///         ParallelEncoderStatus::Finished => break,
///         ParallelEncoderStatus::Pending => session.wait()?,
///         _ => {}
///     }
/// }
/// let _compressor = session.into_compressor();
/// executor.join().unwrap();
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct ParallelEncoderSessionOwned {
    stream: Stream,
    compressor: ParallelCompressor,
}

impl ParallelEncoderSessionOwned {
    session_methods!();

    /// Returns the compressor, abandoning an unfinished frame: tasks that
    /// have not started are skipped and the queue is closed.
    #[must_use]
    pub fn into_compressor(self) -> ParallelCompressor {
        drop(self.stream);
        self.compressor
    }
}

/// The declared size of a frame, when it has one.
const fn declared(stream: StreamConfig) -> Option<u64> {
    match stream.input_size() {
        InputSize::Exact(size) => Some(size),
        InputSize::Unknown => None,
    }
}

impl ParallelCompressor {
    fn stream(&self, stream: StreamConfig, window: TaskCount) -> (Stream, ParallelTaskQueue) {
        let (stream, queue) = Stream::new(
            self.requested,
            declared(stream),
            self.parallel.segment_size().get(),
            window,
            self.pool.clone(),
        );
        (stream, ParallelTaskQueue { inner: queue })
    }

    /// Starts a frame whose input arrives in chunks, and returns the queue
    /// its tasks go to.
    ///
    /// `window` caps how many segments may be queued, running or waiting for
    /// delivery at once; with the segment being gathered and its overlap, it
    /// bounds the session's memory. No thread is created: run the queue's
    /// tasks on any executor. See [`ParallelEncoderSession`] for the
    /// protocol.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (session, queue) = compressor.start(Default::default(), TaskCount::try_from(2)?);
    /// assert!(!session.is_finished());
    /// assert_eq!(queue.queued(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn start(
        &mut self,
        stream: StreamConfig,
        window: TaskCount,
    ) -> (ParallelEncoderSession<'_>, ParallelTaskQueue) {
        let (stream, queue) = self.stream(stream, window);
        let session = ParallelEncoderSession {
            stream,
            compressor: PhantomData,
        };
        (session, queue)
    }

    /// Starts a frame whose input arrives in chunks, with a session that
    /// owns this compressor.
    ///
    /// As [`Self::start`], but the session carries no lifetime.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (session, _queue) = compressor.into_session(Default::default(), TaskCount::ONE);
    /// let _compressor = session.into_compressor();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn into_session(
        self,
        stream: StreamConfig,
        window: TaskCount,
    ) -> (ParallelEncoderSessionOwned, ParallelTaskQueue) {
        let (stream, queue) = self.stream(stream, window);
        let session = ParallelEncoderSessionOwned {
            stream,
            compressor: self,
        };
        (session, queue)
    }
}

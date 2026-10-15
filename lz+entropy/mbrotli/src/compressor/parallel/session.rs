//! Incremental parallel streams: input arrives in chunks, segments become tasks.
//!
//! [`ParallelCompressor::start`] returns a session and a [`ParallelTaskQueue`].
//! Every full segment the session accepts becomes a task on that queue, and the
//! caller's executors take and run them. The session delivers finished parts in
//! input order, so output can be written while later segments still encode.
use super::{ParallelCompressor, ParallelEncodeError, TaskCount, WaitStatus, core};
use crate::compressor::Operation;
use std::{mem::MaybeUninit, time::Duration};

/// What a parallel session needs next.
///
/// # Examples
///
/// ```
/// use mbrotli::compressor::parallel::ParallelEncoderStatus;
///
/// assert_ne!(ParallelEncoderStatus::Pending, ParallelEncoderStatus::NeedsInput);
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ParallelEncoderStatus {
    /// More source bytes are needed before anything else can happen.
    NeedsInput,
    /// Encoded bytes are waiting; call again with room to put them.
    NeedsOutput,
    /// Queued tasks must complete first: keep executors running, call
    /// [`ParallelEncoderSession::wait`], then call again.
    Pending,
    /// The stream is complete and the final bytes have been delivered.
    Finished,
}

/// What one parallel `process` call did.
///
/// `consumed` and `produced` are exact.
///
/// # Examples
///
/// ```
/// use mbrotli::compressor::parallel::{ParallelEncoderStatus, ParallelProgress};
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
    /// How many bytes were taken from the caller's input.
    pub consumed: usize,
    /// How many bytes were written into the caller's output.
    pub produced: usize,
    /// What the session needs next.
    pub status: ParallelEncoderStatus,
}

/// The tasks a parallel session queues, for the caller's executors to run.
///
/// Clone one handle per executor thread. [`Self::recv`] blocks until a task is
/// queued and returns `None` once the stream has finished, failed or been
/// dropped, so `for task in queue { task.run() }` is a complete worker loop.
///
/// Dropping every handle abandons the tasks still queued: the session then
/// fails with [`ParallelEncodeError::TaskQueueClosed`] or a task abandonment
/// instead of waiting forever. A session whose handles are alive but never
/// drained cannot make progress; on a single thread, run queued tasks with
/// [`Self::try_recv`] before waiting.
///
/// # Examples
///
/// ```
/// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
/// use mbrotli::{EncoderConfig, Operation};
///
/// let mut compressor = ParallelCompressor::new(EncoderConfig::default(), ParallelConfig::default())?;
/// let (mut session, queue) = compressor.start(TaskCount::ONE)?;
/// let mut output = [0u8; 64];
/// session.process(b"queued on one thread", &mut output, Operation::Finish)?;
/// while let Some(task) = queue.try_recv() {
///     task.run();
/// }
/// session.wait()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ParallelTaskQueue {
    inner: core::QueueHandle,
}
impl ParallelTaskQueue {
    /// Blocks until a task is queued; returns `None` once the stream has ended.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (session, queue) = compressor.start(TaskCount::ONE)?;
    /// drop(session);
    /// assert!(queue.recv().is_none());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn recv(&self) -> Option<super::OwnedParallelTask> {
        self.inner
            .recv()
            .map(|inner| super::ScopedParallelTask { inner })
    }
    /// Takes a queued task without blocking.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (_session, queue) = compressor.start(TaskCount::ONE)?;
    /// assert!(queue.try_recv().is_none());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn try_recv(&self) -> Option<super::OwnedParallelTask> {
        self.inner
            .try_recv()
            .map(|inner| super::ScopedParallelTask { inner })
    }
}
impl Clone for ParallelTaskQueue {
    /// Returns another handle to the same queue, for another executor.
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
impl IntoIterator for ParallelTaskQueue {
    type Item = super::OwnedParallelTask;
    type IntoIter = ParallelTasks;
    /// Yields queued tasks until the stream ends, blocking between them.
    fn into_iter(self) -> ParallelTasks {
        ParallelTasks { queue: self }
    }
}
impl std::fmt::Debug for ParallelTaskQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelTaskQueue").finish_non_exhaustive()
    }
}

/// Blocking iterator over a [`ParallelTaskQueue`], ending with the stream.
#[derive(Debug)]
pub struct ParallelTasks {
    queue: ParallelTaskQueue,
}
impl Iterator for ParallelTasks {
    type Item = super::OwnedParallelTask;
    fn next(&mut self) -> Option<Self::Item> {
        self.queue.recv()
    }
}

/// One incremental parallel Brotli stream.
///
/// Created by [`ParallelCompressor::start`], which also returns the
/// [`ParallelTaskQueue`] its tasks go to. The session copies input into fixed
/// segments; each full segment is queued as a task, and at most the requested
/// number of parts are queued, running or waiting for delivery at once. When
/// that window is full the session takes no more input and reports
/// [`ParallelEncoderStatus::Pending`].
///
/// Without [`Operation::Flush`], the stream is byte-identical to
/// [`ParallelCompressor::prepare_slice`] over the same bytes, however the input
/// is chunked and whatever the window, including the serial fallback for
/// small inputs. A flush ends the current segment early, which moves every
/// later segment boundary.
///
/// The session never blocks in `process`. [`Self::wait`] blocks until a queued
/// task completes, so executors must be able to run while the caller waits.
///
/// Finishing, failing or dropping the session closes the queue, which ends
/// every executor's `for task in queue` loop. Dropping it unfinished abandons
/// the stream and cancels tasks that have not started.
///
/// # Examples
///
/// ```
/// use mbrotli::compressor::parallel::{
///     ParallelCompressor, ParallelConfig, ParallelEncoderStatus, SegmentSize, TaskCount,
/// };
/// use mbrotli::{EncoderConfig, Operation, Quality};
///
/// let config = ParallelConfig::from(SegmentSize::try_from(64 << 10)?);
/// let mut compressor =
///     ParallelCompressor::new(EncoderConfig::default().with_quality(Quality::Q1), config)?;
/// let payload = b"streamed into parallel segments ".repeat(10_000);
///
/// let mut compressed = Vec::new();
/// std::thread::scope(|scope| {
///     let (mut session, queue) = compressor.start(TaskCount::try_from(4)?)?;
///     for _ in 0..4 {
///         let queue = queue.clone();
///         scope.spawn(move || queue.into_iter().for_each(|task| task.run()));
///     }
///     drop(queue);
///
///     let mut buffer = [0u8; 4096];
///     let mut input = payload.as_slice();
///     loop {
///         let progress = session.process(input, &mut buffer, Operation::Finish)?;
///         input = &input[progress.consumed..];
///         compressed.extend_from_slice(&buffer[..progress.produced]);
///         match progress.status {
///             ParallelEncoderStatus::Finished => break,
///             ParallelEncoderStatus::Pending => session.wait()?,
///             _ => {}
///         }
///     }
///     Ok::<(), Box<dyn std::error::Error>>(())
/// })?;
///
/// assert!(compressed.len() < payload.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ParallelEncoderSession<'c> {
    inner: core::BorrowedStream<'c>,
}
impl ParallelEncoderSession<'_> {
    /// Moves the stream forward by one step without blocking.
    ///
    /// Takes what fits the window from `input`, delivers finished parts into
    /// `output` in input order, and reports what it needs next. Statuses:
    ///
    /// - [`ParallelEncoderStatus::NeedsOutput`]: delivered bytes are waiting.
    /// - [`ParallelEncoderStatus::Pending`]: input is left over because the
    ///   window is full, or a `Flush` or `Finish` waits for queued tasks.
    /// - [`ParallelEncoderStatus::NeedsInput`]: everything this operation can
    ///   do is done. A `Process` call may still have tasks running.
    /// - [`ParallelEncoderStatus::Finished`]: the final byte was delivered;
    ///   later calls consume and produce nothing.
    ///
    /// A `Flush` reports `NeedsInput` only once everything accepted has been
    /// delivered. Once a `Finish` has queued the last part, later calls
    /// consume no input whatever their operation, and only deliver.
    ///
    /// # Errors
    ///
    /// Returns [`ParallelEncodeError::StreamFailed`] with the lowest failing
    /// segment's cause when a task, allocation or the queue fails, and on
    /// every later call.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{
    ///     ParallelCompressor, ParallelConfig, ParallelEncoderStatus, TaskCount,
    /// };
    /// use mbrotli::Operation;
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (mut session, queue) = compressor.start(TaskCount::ONE)?;
    /// let mut output = [0u8; 64];
    ///
    /// let progress = session.process(b"accepted", &mut output, Operation::Process)?;
    /// assert_eq!(progress.consumed, 8);
    /// assert_eq!(progress.status, ParallelEncoderStatus::NeedsInput);
    ///
    /// let progress = session.process(b"", &mut output, Operation::Finish)?;
    /// assert_eq!(progress.status, ParallelEncoderStatus::Pending);
    /// queue.try_recv().unwrap().run();
    /// session.wait()?;
    /// let progress = session.process(b"", &mut output, Operation::Finish)?;
    /// assert_eq!(progress.status, ParallelEncoderStatus::Finished);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn process(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: Operation,
    ) -> Result<ParallelProgress, ParallelEncodeError> {
        let core::BorrowedStream { parent, stream } = &mut self.inner;
        stream.process(parent, input, output, operation)
    }
    /// Moves the stream forward by one step, writing into uninitialized memory.
    ///
    /// Behaves exactly as [`Self::process`]: the same bytes, the same
    /// `consumed` and `produced` counts, the same statuses and errors, and
    /// calls of the two methods may be mixed on one stream. Use it to deliver
    /// straight into spare capacity, such as [`Vec::spare_capacity_mut`],
    /// without zeroing it first.
    ///
    /// On success exactly `output[..produced]` has been initialized, and no
    /// byte after it is written. After an error, treat none of `output` as
    /// initialized.
    ///
    /// # Errors
    ///
    /// As [`Self::process`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{
    ///     ParallelCompressor, ParallelConfig, ParallelEncoderStatus, TaskCount,
    /// };
    /// use mbrotli::Operation;
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (mut session, queue) = compressor.start(TaskCount::ONE)?;
    /// let mut compressed = Vec::with_capacity(64);
    /// let mut input = &b"delivered into spare capacity"[..];
    /// loop {
    ///     if compressed.len() == compressed.capacity() {
    ///         compressed.reserve(64);
    ///     }
    ///     let progress =
    ///         session.process_uninit(input, compressed.spare_capacity_mut(), Operation::Finish)?;
    ///     input = &input[progress.consumed..];
    ///     // SAFETY: `process_uninit` initialized the first `produced` spare bytes.
    ///     unsafe { compressed.set_len(compressed.len() + progress.produced) };
    ///     match progress.status {
    ///         ParallelEncoderStatus::Finished => break,
    ///         ParallelEncoderStatus::Pending => {
    ///             while let Some(task) = queue.try_recv() {
    ///                 task.run();
    ///             }
    ///             session.wait()?;
    ///         }
    ///         _ => {}
    ///     }
    /// }
    /// assert!(!compressed.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn process_uninit(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        operation: Operation,
    ) -> Result<ParallelProgress, ParallelEncodeError> {
        let core::BorrowedStream { parent, stream } = &mut self.inner;
        stream.process(parent, input, output, operation)
    }
    /// Makes everything accepted so far decodable, without taking input.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`Operation::Flush`]. Repeat it, waiting on `Pending`, until it
    /// reports [`ParallelEncoderStatus::NeedsInput`].
    ///
    /// # Errors
    ///
    /// As [`Self::process`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{
    ///     ParallelCompressor, ParallelConfig, ParallelEncoderStatus, TaskCount,
    /// };
    /// use mbrotli::Operation;
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (mut session, queue) = compressor.start(TaskCount::ONE)?;
    /// let mut output = [0u8; 64];
    /// session.process(b"flushed", &mut output, Operation::Process)?;
    /// assert_eq!(session.flush(&mut output)?.status, ParallelEncoderStatus::Pending);
    /// queue.try_recv().unwrap().run();
    /// session.wait()?;
    /// let progress = session.flush(&mut output)?;
    /// assert_eq!(progress.status, ParallelEncoderStatus::NeedsInput);
    /// assert!(progress.produced > 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn flush(&mut self, output: &mut [u8]) -> Result<ParallelProgress, ParallelEncodeError> {
        self.process(&[], output, Operation::Flush)
    }
    /// Terminates the stream, without taking input.
    ///
    /// Shorthand for [`Self::process`] with empty input and
    /// [`Operation::Finish`]. Repeat it, waiting on `Pending`, until it
    /// reports [`ParallelEncoderStatus::Finished`].
    ///
    /// # Errors
    ///
    /// As [`Self::process`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{
    ///     ParallelCompressor, ParallelConfig, ParallelEncoderStatus, TaskCount,
    /// };
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (mut session, _queue) = compressor.start(TaskCount::ONE)?;
    /// let mut output = [0u8; 64];
    /// // An empty stream is encoded inline; no task is queued.
    /// assert_eq!(session.finish(&mut output)?.status, ParallelEncoderStatus::Finished);
    /// assert!(session.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn finish(&mut self, output: &mut [u8]) -> Result<ParallelProgress, ParallelEncodeError> {
        self.process(&[], output, Operation::Finish)
    }
    /// Blocks until a queued task completes; returns at once when none is queued
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
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (mut session, _queue) = compressor.start(TaskCount::ONE)?;
    /// session.wait()?; // Nothing is queued.
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn wait(&mut self) -> Result<(), ParallelEncodeError> {
        let core::BorrowedStream { parent, stream } = &mut self.inner;
        stream.wait(parent, None).map(|_| ())
    }
    /// Waits up to `timeout` for a queued task to complete.
    ///
    /// [`WaitStatus::Ready`] means a completion was collected or nothing is
    /// queued; a timeout cancels nothing.
    ///
    /// # Errors
    ///
    /// As [`Self::process`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    /// use mbrotli::compressor::parallel::{
    ///     ParallelCompressor, ParallelConfig, TaskCount, WaitStatus,
    /// };
    /// use mbrotli::Operation;
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (mut session, _queue) = compressor.start(TaskCount::ONE)?;
    /// session.process(b"never run", &mut [], Operation::Finish)?;
    /// assert_eq!(session.wait_timeout(Duration::ZERO)?, WaitStatus::TimedOut);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn wait_timeout(&mut self, timeout: Duration) -> Result<WaitStatus, ParallelEncodeError> {
        let core::BorrowedStream { parent, stream } = &mut self.inner;
        stream.wait(parent, Some(timeout))
    }
    /// Returns whether the final byte has been delivered.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (session, _queue) = compressor.start(TaskCount::ONE)?;
    /// assert!(!session.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.inner.stream.is_finished()
    }
}
impl std::fmt::Debug for ParallelEncoderSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelEncoderSession")
            .field("finished", &self.is_finished())
            .finish_non_exhaustive()
    }
}

/// One incremental parallel stream that owns its [`ParallelCompressor`].
///
/// Created by [`ParallelCompressor::into_session`]. It runs the same state
/// machine as [`ParallelEncoderSession`], so the same calls produce the same
/// bytes, counts, statuses and errors, but carries no lifetime.
/// [`Self::into_compressor`] ends the stream as dropping a borrowed session
/// does and returns the compressor; dropping the owned session drops it too.
///
/// # Examples
///
/// ```
/// use mbrotli::compressor::parallel::{
///     ParallelCompressor, ParallelConfig, ParallelEncoderStatus, TaskCount,
/// };
/// use mbrotli::Operation;
///
/// let compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
/// let (mut session, queue) = compressor.into_session(TaskCount::ONE)?;
/// let worker = std::thread::spawn(move || queue.into_iter().for_each(|task| task.run()));
///
/// let mut output = [0u8; 64];
/// let mut input = &b"owned parallel stream"[..];
/// loop {
///     let progress = session.process(input, &mut output, Operation::Finish)?;
///     input = &input[progress.consumed..];
///     match progress.status {
///         ParallelEncoderStatus::Finished => break,
///         ParallelEncoderStatus::Pending => session.wait()?,
///         _ => {}
///     }
/// }
/// let _compressor = session.into_compressor();
/// worker.join().unwrap();
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ParallelEncoderSessionOwned {
    inner: core::OwnedStream,
}
impl ParallelEncoderSessionOwned {
    /// Behaves exactly as [`ParallelEncoderSession::process`].
    ///
    /// # Errors
    ///
    /// As [`ParallelEncoderSession::process`].
    pub fn process(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: Operation,
    ) -> Result<ParallelProgress, ParallelEncodeError> {
        let core::OwnedStream { parent, stream } = &mut self.inner;
        stream.process(parent, input, output, operation)
    }
    /// Behaves exactly as [`ParallelEncoderSession::process_uninit`].
    ///
    /// On success exactly `output[..produced]` has been initialized, and no
    /// byte after it is written. After an error, treat none of `output` as
    /// initialized.
    ///
    /// # Errors
    ///
    /// As [`ParallelEncoderSession::process`].
    pub fn process_uninit(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        operation: Operation,
    ) -> Result<ParallelProgress, ParallelEncodeError> {
        let core::OwnedStream { parent, stream } = &mut self.inner;
        stream.process(parent, input, output, operation)
    }
    /// Behaves exactly as [`ParallelEncoderSession::flush`].
    ///
    /// # Errors
    ///
    /// As [`ParallelEncoderSession::process`].
    pub fn flush(&mut self, output: &mut [u8]) -> Result<ParallelProgress, ParallelEncodeError> {
        self.process(&[], output, Operation::Flush)
    }
    /// Behaves exactly as [`ParallelEncoderSession::finish`].
    ///
    /// # Errors
    ///
    /// As [`ParallelEncoderSession::process`].
    pub fn finish(&mut self, output: &mut [u8]) -> Result<ParallelProgress, ParallelEncodeError> {
        self.process(&[], output, Operation::Finish)
    }
    /// Behaves exactly as [`ParallelEncoderSession::wait`].
    ///
    /// # Errors
    ///
    /// As [`ParallelEncoderSession::process`].
    pub fn wait(&mut self) -> Result<(), ParallelEncodeError> {
        let core::OwnedStream { parent, stream } = &mut self.inner;
        stream.wait(parent, None).map(|_| ())
    }
    /// Behaves exactly as [`ParallelEncoderSession::wait_timeout`].
    ///
    /// # Errors
    ///
    /// As [`ParallelEncoderSession::process`].
    pub fn wait_timeout(&mut self, timeout: Duration) -> Result<WaitStatus, ParallelEncodeError> {
        let core::OwnedStream { parent, stream } = &mut self.inner;
        stream.wait(parent, Some(timeout))
    }
    /// Returns whether the final byte has been delivered.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.inner.stream.is_finished()
    }
    /// Ends the stream and returns the compressor, ready for the next operation.
    ///
    /// An unfinished stream is abandoned: tasks that have not started are
    /// cancelled, the queue is closed, and workers already returned stay
    /// available under the compressor's retention policy.
    #[must_use]
    pub fn into_compressor(self) -> ParallelCompressor {
        ParallelCompressor {
            inner: self.inner.into_parent(),
        }
    }
}
impl std::fmt::Debug for ParallelEncoderSessionOwned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelEncoderSessionOwned")
            .field("finished", &self.is_finished())
            .finish_non_exhaustive()
    }
}

impl ParallelCompressor {
    /// Starts an incremental parallel stream and returns its task queue.
    ///
    /// `window` caps how many segments may be queued, running or waiting for
    /// delivery at once; with the filling segment, it bounds the stream's
    /// memory to about `window + 1` segments of input, staged output and
    /// worker workspace. No thread is created: run the queue's tasks on any
    /// executor. See [`ParallelEncoderSession`] for the protocol.
    ///
    /// # Errors
    ///
    /// Returns [`ParallelEncodeError::WorkerMemoryLimit`] when the window's
    /// conservative workspace estimate exceeds the configured aggregate memory
    /// limit, and [`ParallelEncodeError::SizeOverflow`] if that estimate
    /// overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (session, queue) = compressor.start(TaskCount::try_from(2)?)?;
    /// # drop((session, queue));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn start(
        &mut self,
        window: TaskCount,
    ) -> Result<(ParallelEncoderSession<'_>, ParallelTaskQueue), ParallelEncodeError> {
        let (stream, queue) = core::Stream::new(&self.inner, window)?;
        Ok((
            ParallelEncoderSession {
                inner: core::BorrowedStream {
                    parent: &mut self.inner,
                    stream,
                },
            },
            ParallelTaskQueue { inner: queue },
        ))
    }
    /// Starts an incremental parallel stream that owns this compressor.
    ///
    /// As [`Self::start`], but the session carries no lifetime.
    ///
    /// # Errors
    ///
    /// As [`Self::start`]; the compressor is dropped with the error.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (session, _queue) = compressor.into_session(TaskCount::ONE)?;
    /// let _compressor = session.into_compressor();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_session(
        self,
        window: TaskCount,
    ) -> Result<(ParallelEncoderSessionOwned, ParallelTaskQueue), ParallelEncodeError> {
        let (stream, queue) = core::Stream::new(&self.inner, window)?;
        Ok((
            ParallelEncoderSessionOwned {
                inner: core::OwnedStream {
                    parent: self.inner,
                    stream,
                },
            },
            ParallelTaskQueue { inner: queue },
        ))
    }
}

//! A transactional [`Write`] adapter over a parallel session.

use crate::compressor::Operation;
use crate::compressor::parallel::{
    ParallelEncoderSession, ParallelEncoderStatus, ParallelProgress,
};
use crate::io::FinishError;
use std::io::{Error, ErrorKind, Result, Write};

/// How much room a single pull from the session is offered.
///
/// A finished part is copied out in pulls of at most this size, so the outbox
/// stays bounded however large a segment's compressed form is.
const PULL_CHUNK: usize = 128 * 1024;

/// Where a writer is in the life of its stream.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum State {
    /// Accepting writes.
    Open,
    /// Termination has been requested; parts may still be encoding or pending.
    Finishing,
    /// Everything has been encoded and delivered, and the sink was flushed.
    Finished,
}

/// Compresses everything written to it in parallel into an inner writer.
///
/// Created by [`ParallelCompressor::writer`](super::super::ParallelCompressor::writer),
/// which also returns the [`ParallelTaskQueue`](super::super::ParallelTaskQueue)
/// its segment tasks go to. Each full segment written becomes a task; the
/// caller's executors run them while the writer keeps accepting input, and
/// finished parts reach the sink in input order.
///
/// A write blocks while the window is full, until an executor completes a
/// task, so executors must run on other threads. Dropping every queue handle
/// instead makes the next write that needs a task fail rather than block.
///
/// Without [`Write::flush`], the output is byte-identical to
/// [`ParallelCompressor::prepare_slice`](super::super::ParallelCompressor::prepare_slice)
/// over the same bytes, however they are split into writes. A flush ends the
/// current segment early, which moves every later segment boundary.
///
/// Like [`EncoderWriter`](crate::io::EncoderWriter), the adapter keeps every
/// compressed byte until the sink has taken it, so a sink that writes short,
/// is interrupted, would block or fails loses nothing. The stream is only
/// terminated by [`Self::try_finish`] or [`Self::finish`]; dropping the adapter
/// abandons it, cancels tasks that have not started, and closes the queue.
///
/// # Examples
///
/// ```
/// use mbrotli::compressor::parallel::{
///     BatchConfig, ParallelCompressor, ParallelConfig, SegmentSize, TaskCount,
/// };
/// use mbrotli::{EncoderConfig, Quality};
/// use std::io::Write;
///
/// let config = ParallelConfig::from(SegmentSize::try_from(64 << 10)?);
/// let mut compressor =
///     ParallelCompressor::new(EncoderConfig::default().with_quality(Quality::Q1), config)?;
/// let payload = b"written into parallel segments ".repeat(10_000);
///
/// let streamed = std::thread::scope(|scope| {
///     let (mut sink, queue) = compressor.writer(Vec::new(), TaskCount::try_from(4)?)?;
///     for _ in 0..4 {
///         let queue = queue.clone();
///         scope.spawn(move || queue.into_iter().for_each(|task| task.run()));
///     }
///     drop(queue);
///     for chunk in payload.chunks(1000) {
///         sink.write_all(chunk)?;
///     }
///     Ok::<_, Box<dyn std::error::Error>>(sink.finish().map_err(mbrotli::io::FinishError::into_error)?)
/// })?;
///
/// // The bytes the batch API produces for the same input.
/// let mut batch = compressor.prepare_slice(&payload, BatchConfig::auto(TaskCount::ONE))?;
/// batch.run_inline()?;
/// let mut expected = Vec::new();
/// batch.finish_into(&mut expected)?;
/// assert_eq!(streamed, expected);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ParallelEncoderWriter<'c, W: Write> {
    /// The stream being encoded. Boxed because the stream state is large and
    /// [`FinishError`] hands the whole adapter back by value.
    session: Box<ParallelEncoderSession<'c>>,
    /// Where compressed bytes go.
    sink: W,
    /// Compressed bytes produced but not yet taken by the sink.
    outbox: Vec<u8>,
    /// How much of [`ParallelEncoderWriter::outbox`] the sink has already taken.
    head: usize,
    /// Initialized storage is retained; only `head..end` is undelivered output.
    end: usize,
    /// Where the stream is.
    state: State,
}

impl<W: Write> std::fmt::Debug for ParallelEncoderWriter<'_, W> {
    /// Reports the stream's state and how much output is still undelivered.
    ///
    /// The sink has no `Debug` bound, so it is not shown.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelEncoderWriter")
            .field("state", &self.state)
            .field("undelivered", &(self.end - self.head))
            .finish_non_exhaustive()
    }
}

impl<'c, W: Write> ParallelEncoderWriter<'c, W> {
    /// Wraps `sink` around `session`.
    pub(in crate::compressor::parallel) fn new(
        session: ParallelEncoderSession<'c>,
        sink: W,
    ) -> Self {
        Self {
            session: Box::new(session),
            sink,
            outbox: Vec::new(),
            head: 0,
            end: 0,
            state: State::Open,
        }
    }

    /// Returns a shared reference to the inner writer.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (sink, _queue) = compressor.writer(Vec::new(), TaskCount::ONE)?;
    /// assert!(sink.get_ref().is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn get_ref(&self) -> &W {
        &self.sink
    }

    /// Returns a mutable reference to the inner writer.
    ///
    /// Writing to it directly would corrupt the compressed stream; this is for
    /// the sink's own controls, such as setting a socket option.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (mut sink, _queue) = compressor.writer(Vec::new(), TaskCount::ONE)?;
    /// sink.get_mut().reserve(1024);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn get_mut(&mut self) -> &mut W {
        &mut self.sink
    }

    /// Returns whether the stream has been terminated and fully delivered.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (mut sink, _queue) = compressor.writer(Vec::new(), TaskCount::ONE)?;
    /// assert!(!sink.is_finished());
    /// // An empty stream is encoded inline, so no executor is needed.
    /// sink.try_finish()?;
    /// assert!(sink.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn is_finished(&self) -> bool {
        matches!(self.state, State::Finished)
    }

    /// Terminates the stream, and may be called again until it succeeds.
    ///
    /// Blocks until every queued part has been encoded and delivered, so the
    /// queue's executors must be running. A sink failure part-way leaves the
    /// remaining bytes buffered, and the next call resumes at exactly the byte
    /// the sink stopped at; the terminator is written once.
    ///
    /// # Errors
    ///
    /// Propagates the sink's errors, and the session's converted into
    /// [`std::io::Error`] with the [`ParallelEncodeError`](super::super::ParallelEncodeError)
    /// as its source. Once it returns `Ok`, the stream is complete and the
    /// sink has been flushed.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    /// use std::io::Write;
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// std::thread::scope(|scope| {
    ///     let (mut sink, queue) = compressor.writer(Vec::new(), TaskCount::ONE)?;
    ///     scope.spawn(move || queue.into_iter().for_each(|task| task.run()));
    ///     sink.write_all(b"payload payload")?;
    ///     sink.try_finish()?;
    ///     // Retrying a finished stream is a no-op rather than a second terminator.
    ///     sink.try_finish()?;
    ///     assert!(!sink.get_ref().is_empty());
    ///     Ok::<(), std::io::Error>(())
    /// })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn try_finish(&mut self) -> Result<()> {
        if self.state == State::Finished {
            return Ok(());
        }
        self.drain()?;
        loop {
            let progress = self.pump(&[], Operation::Finish)?;
            self.state = State::Finishing;
            self.drain()?;
            if self.settle(progress)? {
                break;
            }
        }
        self.sink.flush()?;
        self.state = State::Finished;
        Ok(())
    }

    /// Terminates the stream and returns the inner writer.
    ///
    /// On failure the adapter comes back inside the error, so the caller can
    /// retry [`Self::try_finish`] once the sink is ready, or drop it and give
    /// up on the stream.
    ///
    /// # Errors
    ///
    /// Returns [`FinishError`] carrying both the failure and this adapter.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    /// use mbrotli::io::FinishError;
    /// use std::io::Write;
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let compressed = std::thread::scope(|scope| {
    ///     let (mut sink, queue) = compressor.writer(Vec::new(), TaskCount::ONE)?;
    ///     scope.spawn(move || queue.into_iter().for_each(|task| task.run()));
    ///     sink.write_all(b"payload payload")?;
    ///     sink.finish().map_err(FinishError::into_error)
    /// })?;
    /// assert!(!compressed.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn finish(mut self) -> std::result::Result<W, FinishError<Self>> {
        match self.try_finish() {
            Ok(()) => Ok(self.sink),
            Err(error) => Err(FinishError::from_parts(error, self)),
        }
    }

    /// Hands the sink every byte it will take, keeping the rest.
    ///
    /// The cursor is what makes this safe to retry: a short write, an
    /// interruption or an error leaves the exact unwritten suffix in place.
    fn drain(&mut self) -> Result<()> {
        while self.head < self.end {
            let remaining = self.outbox.get(self.head..self.end).unwrap_or_default();
            match self.sink.write(remaining) {
                Ok(0) => {
                    return Err(Error::new(
                        ErrorKind::WriteZero,
                        "the sink accepted none of the compressed stream",
                    ));
                }
                Ok(count) => self.head += count,
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        self.head = 0;
        self.end = 0;
        Ok(())
    }

    /// Pulls at most one bounded buffer; the caller drains before pulling again.
    fn pump(&mut self, input: &[u8], operation: Operation) -> Result<ParallelProgress> {
        debug_assert_eq!(self.end, 0);
        self.outbox.resize(PULL_CHUNK, 0);
        let progress = self.session.process(input, &mut self.outbox, operation)?;
        self.end = progress.produced;
        Ok(progress)
    }

    /// Acts on a drained flush or finish step; returns whether it is complete.
    ///
    /// Waits for a task only when the step delivered nothing: a step that did
    /// deliver may have more ready to go, and is simply repeated.
    fn settle(&mut self, progress: ParallelProgress) -> Result<bool> {
        match progress.status {
            ParallelEncoderStatus::NeedsInput | ParallelEncoderStatus::Finished => Ok(true),
            ParallelEncoderStatus::Pending if progress.produced == 0 => {
                self.session.wait()?;
                Ok(false)
            }
            ParallelEncoderStatus::Pending | ParallelEncoderStatus::NeedsOutput => Ok(false),
        }
    }
}

impl<W: Write> Write for ParallelEncoderWriter<'_, W> {
    /// Accepts as much of `buf` as the window will take.
    ///
    /// Pending output is delivered before any new input is accepted, so a sink
    /// that is failing reports the failure with nothing consumed. When the
    /// window is full, finished parts are delivered and the call blocks until
    /// an executor frees a slot. Accepted bytes are this adapter's
    /// responsibility: a sink failure after acceptance surfaces on a later
    /// call rather than alongside a count.
    ///
    /// # Errors
    ///
    /// Propagates the sink's errors from the drain that precedes acceptance,
    /// and the session's, including a queue whose every handle was dropped.
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        if self.state != State::Open {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "the compressed stream has already been finished",
            ));
        }
        // Deliver first: if the sink is unhappy, the caller keeps its bytes.
        self.drain()?;
        if buf.is_empty() {
            return Ok(0);
        }
        loop {
            let progress = self.pump(buf, Operation::Process)?;
            if progress.consumed != 0 {
                // Accepted bytes belong to us. A sink error must not ask the
                // caller to repeat them; surface it on the next drain instead.
                drop(self.drain());
                return Ok(progress.consumed);
            }
            // The window is full: deliver what is ready, and wait only when
            // nothing was.
            self.drain()?;
            if progress.produced == 0 {
                self.session.wait()?;
            }
        }
    }

    /// Makes everything written so far decodable, without ending the stream.
    ///
    /// Queues the partly filled segment, blocks until every accepted byte has
    /// been encoded and delivered, then flushes the sink. A flush moves every
    /// later segment boundary, so the output no longer matches the batch API
    /// for the same bytes. Flushing with nothing written since emits nothing.
    ///
    /// # Errors
    ///
    /// Propagates the sink's errors and the session's. Retryable: a failed
    /// flush leaves its bytes buffered and re-flushing resumes from there.
    fn flush(&mut self) -> Result<()> {
        if self.state != State::Open {
            return self.sink.flush();
        }
        self.drain()?;
        loop {
            let progress = self.pump(&[], Operation::Flush)?;
            self.drain()?;
            if self.settle(progress)? {
                break;
            }
        }
        self.sink.flush()
    }
}

//! A transactional [`Write`] adapter over a parallel session.

use alloc::boxed::Box;
use alloc::vec::Vec;
use std::io::{Error, ErrorKind, Result, Write};

use crate::compressor::parallel::{
    ParallelEncoderSession, ParallelEncoderStatus, ParallelProgress,
};
use crate::compressor::session::Operation;
use crate::io::FinishError;

/// Room offered to the session per pull.
///
/// A finished part is copied out in pulls of at most this size, so the
/// outbox stays bounded however large a segment's compressed form is.
const PULL_CHUNK: usize = 128 * 1024;

/// Where a writer is in the life of its frame.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum State {
    /// Accepting writes.
    Open,
    /// The end has been requested; parts may still be compressing.
    Finishing,
    /// Everything has been delivered, and the sink was flushed.
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
/// instead makes the write that needs a task fail rather than block.
///
/// With a declared input size and without [`Write::flush`], the frame is
/// byte-identical to
/// [`ParallelCompressor::prepare_slice`](super::super::ParallelCompressor::prepare_slice)
/// over the same bytes, however they are split into writes. A flush is
/// [`Operation::Flush`]: it ends the current segment early, which moves
/// every later segment boundary.
///
/// Like [`EncoderWriter`](crate::io::EncoderWriter), the adapter keeps every
/// compressed byte until the sink has taken it, so a sink that writes short,
/// is interrupted, would block or fails loses nothing; the next call carries
/// on. The frame is only ended by [`Self::try_finish`] or [`Self::finish`];
/// dropping the adapter abandons it, skips tasks that have not started, and
/// closes the queue.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::{ParallelCompressor, ParallelConfig, SegmentSize, TaskCount};
/// use mzstd::io::FinishError;
/// use mzstd::InputSize;
/// use std::io::Write;
///
/// let config = ParallelConfig::default().with_segment_size(SegmentSize::MIN);
/// let mut compressor = ParallelCompressor::new(Default::default(), config)?;
/// let payload = b"written into parallel segments ".repeat(10_000);
///
/// let frame = std::thread::scope(|scope| {
///     let stream = InputSize::Exact(payload.len() as u64).into();
///     let (mut sink, queue) = compressor.writer(Vec::new(), stream, TaskCount::try_from(4)?);
///     for _ in 0..4 {
///         let queue = queue.clone();
///         scope.spawn(move || queue.into_iter().for_each(|task| task.run()));
///     }
///     drop(queue);
///     for chunk in payload.chunks(1000) {
///         sink.write_all(chunk)?;
///     }
///     Ok::<_, Box<dyn std::error::Error>>(sink.finish().map_err(FinishError::into_error)?)
/// })?;
///
/// // The bytes the batch API produces for the same input.
/// assert_eq!(frame, compressor.compress(&payload)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ParallelEncoderWriter<'c, W: Write> {
    /// Boxed because [`FinishError`] hands the whole adapter back by value.
    session: Box<ParallelEncoderSession<'c>>,
    sink: W,
    /// Compressed bytes; `outbox[head..end]` is not yet taken by the sink.
    outbox: Vec<u8>,
    head: usize,
    end: usize,
    state: State,
}

impl<W: Write> std::fmt::Debug for ParallelEncoderWriter<'_, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelEncoderWriter")
            .field("session", &self.session)
            .field("undelivered", &(self.end - self.head))
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl<'c, W: Write> ParallelEncoderWriter<'c, W> {
    pub(super) fn new(session: ParallelEncoderSession<'c>, sink: W) -> Self {
        Self {
            session: Box::new(session),
            sink,
            outbox: Vec::new(),
            head: 0,
            end: 0,
            state: State::Open,
        }
    }

    /// The inner writer.
    #[must_use]
    pub const fn get_ref(&self) -> &W {
        &self.sink
    }

    /// The inner writer, mutably; writing to it directly corrupts the frame.
    pub const fn get_mut(&mut self) -> &mut W {
        &mut self.sink
    }

    /// Whether the frame has been ended, delivered and flushed.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.state == State::Finished
    }

    /// Ends the frame and flushes the sink, keeping the adapter so a failed
    /// attempt can be retried.
    ///
    /// Blocks until every queued part has been compressed and delivered, so
    /// the queue's executors must be running. A sink failure part-way leaves
    /// the remaining bytes buffered, and the next call resumes at exactly the
    /// byte the sink stopped at; the end of the frame is written once.
    ///
    /// # Errors
    ///
    /// Sink failures, after which the call may be repeated, and session
    /// failures, converted from [`ParallelEncodeError`](super::super::ParallelEncodeError).
    pub fn try_finish(&mut self) -> Result<()> {
        if self.state == State::Finished {
            return Ok(());
        }
        self.drain()?;
        loop {
            let progress = self.pump(&[], Operation::End)?;
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

    /// Ends the frame and returns the inner writer.
    ///
    /// # Errors
    ///
    /// A [`FinishError`] carrying the adapter, so the finalisation can be
    /// retried.
    pub fn finish(mut self) -> std::result::Result<W, FinishError<Self>> {
        match self.try_finish() {
            Ok(()) => Ok(self.sink),
            Err(error) => Err(FinishError::from_parts(error, self)),
        }
    }

    /// Hands the sink every byte it will take, keeping the rest.
    fn drain(&mut self) -> Result<()> {
        while self.head < self.end {
            match self.sink.write(&self.outbox[self.head..self.end]) {
                Ok(0) => {
                    return Err(Error::new(
                        ErrorKind::WriteZero,
                        "the sink accepted no bytes",
                    ));
                }
                Ok(n) => self.head += n,
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        self.head = 0;
        self.end = 0;
        Ok(())
    }

    /// Pulls at most one bounded buffer; the caller drains before pulling
    /// again.
    fn pump(&mut self, input: &[u8], operation: Operation) -> Result<ParallelProgress> {
        // Only the first pull zeroes the outbox; later pulls overwrite it.
        self.outbox.resize(PULL_CHUNK, 0);
        let progress = self.session.process(input, &mut self.outbox, operation)?;
        self.end = progress.produced;
        Ok(progress)
    }

    /// Acts on a drained flush or end step; returns whether it is complete.
    ///
    /// Waits for a task only when the step delivered nothing: a step that
    /// did deliver may have more ready, and is simply repeated.
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
    /// Pending output is delivered before any input is accepted, so a
    /// failing sink reports its failure with nothing consumed. While the
    /// window is full, finished parts are delivered and the call blocks
    /// until an executor frees a slot. Accepted bytes are the adapter's
    /// responsibility: a sink failure after acceptance surfaces on a later
    /// call rather than alongside a count.
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        if self.state != State::Open {
            return Err(Error::other("the frame has already been finished"));
        }
        self.drain()?;
        if buf.is_empty() {
            return Ok(0);
        }
        loop {
            let progress = self.pump(buf, Operation::Continue)?;
            if progress.consumed != 0 {
                // The caller must not repeat accepted bytes; a sink error
                // waits for the next drain.
                drop(self.drain());
                return Ok(progress.consumed);
            }
            self.drain()?;
            if progress.produced == 0 {
                self.session.wait()?;
            }
        }
    }

    /// Makes everything written so far decodable and flushes the sink.
    ///
    /// This is [`Operation::Flush`]: it queues the partly gathered segment
    /// and blocks until everything accepted has been compressed and
    /// delivered. Retryable: a failed flush leaves its bytes buffered.
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

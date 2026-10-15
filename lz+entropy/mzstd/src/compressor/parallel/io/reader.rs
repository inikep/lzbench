//! A [`Read`] adapter over a parallel session.

use alloc::vec::Vec;
use std::io::{ErrorKind, Read, Result};

use crate::compressor::io::EncoderReaderParts;
use crate::compressor::parallel::{ParallelEncoderSession, ParallelEncoderStatus};
use crate::compressor::session::Operation;

/// How many source bytes one refill asks for.
const FILL_CHUNK: usize = 128 * 1024;

/// Yields the parallel-compressed frame of everything read from an inner
/// reader.
///
/// Created by [`ParallelCompressor::reader`](super::super::ParallelCompressor::reader),
/// which also returns the [`ParallelTaskQueue`](super::super::ParallelTaskQueue)
/// its segment tasks go to. A read keeps pulling source bytes and queuing
/// full segments until the window is full, then blocks until the next part
/// in input order is ready, so executors must run on other threads. Dropping
/// every queue handle instead makes the read that needs a task fail rather
/// than block.
///
/// With a declared input size, the frame is byte-identical to
/// [`ParallelCompressor::prepare_slice`](super::super::ParallelCompressor::prepare_slice)
/// over the whole source, however the source splits its reads and whatever
/// the size of the caller's buffer. Compressed bytes are written straight
/// into the caller's slice. Dropping the adapter abandons the frame, skips
/// tasks that have not started, and closes the queue.
///
/// # Examples
///
/// ```
/// use mzstd::compressor::parallel::{ParallelCompressor, ParallelConfig, SegmentSize, TaskCount};
/// use mzstd::InputSize;
/// use std::io::Read;
///
/// let config = ParallelConfig::default().with_segment_size(SegmentSize::MIN);
/// let mut compressor = ParallelCompressor::new(Default::default(), config)?;
/// let payload = b"read through parallel segments ".repeat(10_000);
///
/// let frame = std::thread::scope(|scope| {
///     let stream = InputSize::Exact(payload.len() as u64).into();
///     let (mut source, queue) = compressor.reader(payload.as_slice(), stream, TaskCount::try_from(4)?);
///     for _ in 0..4 {
///         let queue = queue.clone();
///         scope.spawn(move || queue.into_iter().for_each(|task| task.run()));
///     }
///     drop(queue);
///     let mut frame = Vec::new();
///     source.read_to_end(&mut frame)?;
///     Ok::<_, Box<dyn std::error::Error>>(frame)
/// })?;
///
/// assert_eq!(frame, compressor.compress(&payload)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ParallelEncoderReader<'c, R: Read> {
    session: ParallelEncoderSession<'c>,
    source: R,
    /// Read buffer; `input[head..end]` holds source bytes read but not yet
    /// accepted by the session. Once sized it keeps its length, so a refill
    /// reads over the previous chunk instead of zeroing it again.
    input: Vec<u8>,
    head: usize,
    end: usize,
    /// Whether the source has reported its end.
    eof: bool,
}

impl<R: Read> std::fmt::Debug for ParallelEncoderReader<'_, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelEncoderReader")
            .field("session", &self.session)
            .field("buffered", &(self.end - self.head))
            .field("eof", &self.eof)
            .finish_non_exhaustive()
    }
}

impl<'c, R: Read> ParallelEncoderReader<'c, R> {
    pub(super) const fn new(session: ParallelEncoderSession<'c>, source: R) -> Self {
        Self {
            session,
            source,
            input: Vec::new(),
            head: 0,
            end: 0,
            eof: false,
        }
    }

    /// The inner reader.
    #[must_use]
    pub const fn get_ref(&self) -> &R {
        &self.source
    }

    /// The inner reader, mutably; reading from it directly loses bytes the
    /// frame needs.
    pub const fn get_mut(&mut self) -> &mut R {
        &mut self.source
    }

    /// Whether the frame has been completed and delivered.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.session.is_finished()
    }

    /// Ends the operation and returns the source with the bytes read from it
    /// that the session had not accepted, in source order. Bytes the session
    /// accepted belong to the abandoned frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (source, _queue) = compressor.reader(&b"payload"[..], Default::default(), TaskCount::ONE);
    /// let parts = source.into_parts();
    /// assert_eq!(parts.source, b"payload");
    /// assert!(parts.unread_input.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn into_parts(self) -> EncoderReaderParts<R> {
        EncoderReaderParts {
            unread_input: self.input[self.head..self.end].to_vec(),
            source: self.source,
        }
    }

    /// Reads the next stretch of source bytes, retrying an interruption.
    fn fill(&mut self) -> Result<()> {
        self.head = 0;
        self.end = 0;
        // Only the first fill zeroes the buffer; later reads overwrite it.
        self.input.resize(FILL_CHUNK, 0);
        loop {
            match self.source.read(&mut self.input) {
                Ok(0) => {
                    self.eof = true;
                    return Ok(());
                }
                Ok(n) => {
                    self.end = n.min(self.input.len());
                    return Ok(());
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
    }
}

impl<R: Read> Read for ParallelEncoderReader<'_, R> {
    /// Fills `buf` with the next compressed bytes.
    ///
    /// A zero-length destination reads nothing from the source. A source
    /// error propagates without losing or repeating anything already
    /// accepted, and a retry carries on where it stopped.
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        if buf.is_empty() || self.session.is_finished() {
            return Ok(0);
        }
        loop {
            if self.head == self.end && !self.eof {
                self.fill()?;
            }
            let operation = if self.eof {
                Operation::End
            } else {
                Operation::Continue
            };
            let progress =
                self.session
                    .process(&self.input[self.head..self.end], buf, operation)?;
            self.head += progress.consumed;
            if progress.produced > 0 || progress.status == ParallelEncoderStatus::Finished {
                return Ok(progress.produced);
            }
            if progress.status == ParallelEncoderStatus::Pending {
                self.session.wait()?;
            }
        }
    }
}

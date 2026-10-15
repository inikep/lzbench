//! A [`Read`] adapter over a parallel session.

use crate::compressor::Operation;
use crate::compressor::parallel::{ParallelEncoderSession, ParallelEncoderStatus};
use crate::io::EncoderReaderParts;
use std::io::{ErrorKind, Read, Result};

/// How many source bytes one refill asks for.
const FILL_CHUNK: usize = 64 * 1024;

/// Yields the parallel-compressed form of an inner reader.
///
/// Created by [`ParallelCompressor::reader`](super::super::ParallelCompressor::reader),
/// which also returns the [`ParallelTaskQueue`](super::super::ParallelTaskQueue)
/// its segment tasks go to. A read keeps pulling source bytes and queuing full
/// segments until the window is full, then blocks until the next part in input
/// order is ready, so executors must run on other threads. Dropping every
/// queue handle instead makes the read that needs a task fail rather than
/// block.
///
/// The output is byte-identical to
/// [`ParallelCompressor::prepare_slice`](super::super::ParallelCompressor::prepare_slice)
/// over the whole source, however the source splits its reads and whatever the
/// size of the caller's buffer. Compressed bytes are written straight into the
/// caller's slice.
///
/// Dropping the adapter abandons the stream, cancels tasks that have not
/// started, and closes the queue.
///
/// # Examples
///
/// ```
/// use mbrotli::compressor::parallel::{
///     ParallelCompressor, ParallelConfig, SegmentSize, TaskCount,
/// };
/// use mbrotli::{EncoderConfig, Quality};
/// use std::io::Read;
///
/// let config = ParallelConfig::from(SegmentSize::try_from(64 << 10)?);
/// let mut compressor =
///     ParallelCompressor::new(EncoderConfig::default().with_quality(Quality::Q1), config)?;
/// let payload = b"read through parallel segments ".repeat(10_000);
///
/// let compressed = std::thread::scope(|scope| {
///     let (mut source, queue) = compressor.reader(payload.as_slice(), TaskCount::try_from(4)?)?;
///     for _ in 0..4 {
///         let queue = queue.clone();
///         scope.spawn(move || queue.into_iter().for_each(|task| task.run()));
///     }
///     drop(queue);
///     let mut compressed = Vec::new();
///     source.read_to_end(&mut compressed)?;
///     Ok::<_, Box<dyn std::error::Error>>(compressed)
/// })?;
///
/// assert!(compressed.len() < payload.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ParallelEncoderReader<'c, R: Read> {
    /// The stream being encoded.
    session: ParallelEncoderSession<'c>,
    /// Where source bytes come from.
    source: R,
    /// Read buffer; `input[head..end]` holds source bytes read but not yet
    /// accepted by the session. Once sized it keeps its length, so a refill
    /// reads over the previous chunk instead of zeroing it again.
    input: Vec<u8>,
    /// How much of [`ParallelEncoderReader::input`] the session has taken.
    head: usize,
    /// End of the bytes the last read produced.
    end: usize,
    /// Whether the source has reported end of file.
    eof: bool,
    /// Whether the compressed stream has been terminated and delivered.
    finished: bool,
}

impl<R: Read> std::fmt::Debug for ParallelEncoderReader<'_, R> {
    /// Reports the buffered source bytes and whether the stream is done.
    ///
    /// The source has no `Debug` bound, so it is not shown.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelEncoderReader")
            .field("buffered", &(self.end - self.head))
            .field("eof", &self.eof)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl<'c, R: Read> ParallelEncoderReader<'c, R> {
    /// Wraps `source` around `session`.
    pub(in crate::compressor::parallel) const fn new(
        session: ParallelEncoderSession<'c>,
        source: R,
    ) -> Self {
        Self {
            session,
            source,
            input: Vec::new(),
            head: 0,
            end: 0,
            eof: false,
            finished: false,
        }
    }

    /// Returns a shared reference to the inner reader.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (source, _queue) = compressor.reader(&b"data"[..], TaskCount::ONE)?;
    /// assert_eq!(source.get_ref().len(), 4);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn get_ref(&self) -> &R {
        &self.source
    }

    /// Returns a mutable reference to the inner reader.
    ///
    /// Reading from it directly would lose bytes the compressed stream needs;
    /// this is for the source's own controls.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (mut source, _queue) = compressor.reader(&b"data"[..], TaskCount::ONE)?;
    /// assert_eq!(source.get_mut().len(), 4);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn get_mut(&mut self) -> &mut R {
        &mut self.source
    }

    /// Returns whether the compressed stream has been fully delivered.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    /// use std::io::Read;
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// // An empty source is encoded inline, so no executor is needed.
    /// let (mut source, _queue) = compressor.reader(&b""[..], TaskCount::ONE)?;
    /// assert!(!source.is_finished());
    /// source.read_to_end(&mut Vec::new())?;
    /// assert!(source.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn is_finished(&self) -> bool {
        self.finished
    }

    /// Takes the adapter apart, returning the source and what it had read.
    ///
    /// Source bytes read ahead of what the session has accepted come back in
    /// [`EncoderReaderParts::buffered_input`], in the order the source
    /// produced them. Bytes the session already accepted are part of the
    /// abandoned stream.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::compressor::parallel::{ParallelCompressor, ParallelConfig, TaskCount};
    ///
    /// let mut compressor = ParallelCompressor::new(Default::default(), ParallelConfig::default())?;
    /// let (source, _queue) = compressor.reader(&b"payload"[..], TaskCount::ONE)?;
    /// let parts = source.into_parts();
    /// assert_eq!(parts.inner.len(), 7);
    /// assert!(parts.buffered_input.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn into_parts(self) -> EncoderReaderParts<R> {
        let buffered_input = self
            .input
            .get(self.head..self.end)
            .unwrap_or_default()
            .to_vec();
        EncoderReaderParts {
            inner: self.source,
            buffered_input,
        }
    }

    /// Reads the next stretch of source bytes, retrying an interruption.
    fn fill(&mut self) -> Result<()> {
        self.head = 0;
        self.end = 0;
        // Only the first fill initialises the buffer; later reads overwrite it.
        self.input.resize(FILL_CHUNK, 0);
        loop {
            match self.source.read(&mut self.input) {
                Ok(0) => {
                    self.eof = true;
                    return Ok(());
                }
                Ok(count) => {
                    self.end = count.min(self.input.len());
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
    /// A zero-length destination reads nothing from the source. A source error
    /// propagates without losing or duplicating anything already accepted, and
    /// a retry picks up where it left off.
    ///
    /// # Errors
    ///
    /// Propagates the source's errors, and the session's converted into
    /// [`std::io::Error`] with the [`ParallelEncodeError`](super::super::ParallelEncodeError)
    /// as its source, including a queue whose every handle was dropped.
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        if buf.is_empty() || self.finished {
            return Ok(0);
        }
        loop {
            if self.head == self.end && !self.eof {
                self.fill()?;
            }
            let operation = if self.eof {
                Operation::Finish
            } else {
                Operation::Process
            };
            let progress = {
                let pending = self.input.get(self.head..self.end).unwrap_or_default();
                self.session.process(pending, buf, operation)?
            };
            self.head += progress.consumed;
            if self.head == self.end {
                self.head = 0;
                self.end = 0;
            }
            if progress.status == ParallelEncoderStatus::Finished {
                self.finished = true;
            }
            if progress.produced > 0 {
                return Ok(progress.produced);
            }
            if self.finished {
                return Ok(0);
            }
            if progress.status == ParallelEncoderStatus::Pending {
                self.session.wait()?;
            }
        }
    }
}

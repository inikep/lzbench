//! A transactional [`Write`] adapter over an encoder session.

use crate::compressor::session::{EncoderSession, EncoderStatus, Operation, Progress};
use crate::io::FinishError;
use std::io::{Error, ErrorKind, Result, Write};

/// How much room a single pull from the session is offered.
///
/// A meta-block's compressed form is usually well under this, so one pull
/// normally empties the encoder. Larger outputs are drained in bounded pulls;
/// the initialized buffer is reused even for single-byte input writes.
const PULL_CHUNK: usize = 128 * 1024;

/// Where a writer is in the life of its stream.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum State {
    /// Accepting writes.
    Open,
    /// The final meta-block has been encoded; its bytes may still be pending.
    Finishing,
    /// Everything has been encoded and delivered, and the sink was flushed.
    Finished,
}

/// Compresses everything written to it into an inner writer.
///
/// Every byte this adapter accepts is one it has taken responsibility for, and
/// every compressed byte it has produced is kept until the sink has actually
/// taken it. A sink that writes short, returns
/// [`ErrorKind::Interrupted`](std::io::ErrorKind::Interrupted),
/// [`ErrorKind::WouldBlock`](std::io::ErrorKind::WouldBlock) or an error of its
/// own loses nothing: the unwritten suffix stays exactly where it was, and the
/// next call carries on from there.
///
/// The stream is only terminated by [`EncoderWriter::try_finish`] or
/// [`EncoderWriter::finish`]. Dropping the adapter abandons the stream, which
/// no decoder will accept; `Drop` performs no I/O and cannot fail.
///
/// # Examples
///
/// ```
/// use mbrotli::io::FinishError;
/// use mbrotli::{Compressor, EncoderConfig, InputSize, Quality};
/// use std::io::Write;
///
/// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
/// let payload = b"chunk one chunk two ";
///
/// let streamed = {
///     let mut sink = encoder.writer(Vec::new(), InputSize::Exact(payload.len() as u64).into())?;
///     for chunk in payload.chunks(4) {
///         sink.write_all(chunk)?;
///     }
///     sink.finish().map_err(FinishError::into_error)?
/// };
///
/// // The same bytes the one-shot path produces, for the same declared size.
/// assert_eq!(streamed, encoder.compress(payload)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct EncoderWriter<'c, 'd, W: Write> {
    /// The stream being encoded.
    session: EncoderSession<'c, 'd>,
    /// Where compressed bytes go.
    sink: W,
    /// Compressed bytes produced but not yet taken by the sink.
    outbox: Vec<u8>,
    /// How much of [`EncoderWriter::outbox`] the sink has already taken.
    head: usize,
    /// Initialized storage is retained; only `head..end` is undelivered output.
    end: usize,
    /// Where the stream is.
    state: State,
}

impl<W: Write> std::fmt::Debug for EncoderWriter<'_, '_, W> {
    /// Reports the stream's state and how much output is still undelivered.
    ///
    /// Neither the sink nor the session is shown: the sink has no `Debug`
    /// bound, and the session is the compressor's business.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderWriter")
            .field("state", &self.state)
            .field("undelivered", &(self.end - self.head))
            .finish_non_exhaustive()
    }
}

impl<'c, 'd, W: Write> EncoderWriter<'c, 'd, W> {
    /// Wraps `sink` around `session`.
    pub(crate) fn new(session: EncoderSession<'c, 'd>, sink: W) -> Self {
        Self {
            session,
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
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q0))?;
    /// let sink = encoder.writer(Vec::new(), Default::default())?;
    ///
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
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q0))?;
    /// let mut sink = encoder.writer(Vec::new(), Default::default())?;
    ///
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
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q0))?;
    /// let mut sink = encoder.writer(Vec::new(), Default::default())?;
    ///
    /// assert!(!sink.is_finished());
    /// sink.try_finish()?;
    /// assert!(sink.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn is_finished(&self) -> bool {
        matches!(self.state, State::Finished)
    }

    /// Terminates the stream, and may be called again until it succeeds.
    ///
    /// The final meta-block is encoded once however many calls it takes to
    /// deliver: a sink failure part-way leaves the remaining bytes buffered,
    /// and the next call resumes at exactly the byte the sink stopped at. A
    /// second terminator is never written.
    ///
    /// # Errors
    ///
    /// Propagates the sink's errors and the encoder's. Once it returns `Ok`,
    /// the stream is complete and the sink has been flushed.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    /// use std::io::Write;
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
    /// let mut sink = encoder.writer(Vec::new(), Default::default())?;
    ///
    /// sink.write_all(b"payload payload")?;
    /// sink.try_finish()?;
    /// // Retrying a finished stream is a no-op rather than a second terminator.
    /// sink.try_finish()?;
    ///
    /// assert!(!sink.get_ref().is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn try_finish(&mut self) -> Result<()> {
        if self.state == State::Finished {
            return Ok(());
        }
        self.drain()?;
        while !self.session.is_finished() {
            self.pump(&[], Operation::Finish)?;
            self.state = State::Finishing;
            self.drain()?;
        }
        self.drain()?;
        self.sink.flush()?;
        self.state = State::Finished;
        Ok(())
    }

    /// Terminates the stream and returns the inner writer.
    ///
    /// On failure the adapter comes back inside the error, so the caller can
    /// retry [`EncoderWriter::try_finish`] once the sink is ready, or take the
    /// sink out and give up on the stream. Nothing is lost either way.
    ///
    /// # Errors
    ///
    /// Returns [`FinishError`] carrying both the failure and this adapter.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::io::FinishError;
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    /// use std::io::Write;
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
    /// let mut sink = encoder.writer(Vec::new(), Default::default())?;
    /// sink.write_all(b"payload payload")?;
    ///
    /// let compressed = sink.finish().map_err(FinishError::into_error)?;
    ///
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
            let remaining = &self.outbox[self.head..self.end];
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
    fn pump(&mut self, input: &[u8], operation: Operation) -> Result<Progress> {
        debug_assert_eq!(self.end, 0);
        self.outbox.resize(PULL_CHUNK, 0);
        let progress = self.session.process(input, &mut self.outbox, operation)?;
        self.end = progress.produced;
        Ok(progress)
    }
}

impl<W: Write> Write for EncoderWriter<'_, '_, W> {
    /// Accepts as much of `buf` as the encoder will take.
    ///
    /// Pending output is delivered before any new input is accepted, so a sink
    /// that is failing reports the failure with nothing consumed and the caller
    /// may retry the very same bytes. Once bytes have been accepted they are
    /// this adapter's responsibility, and a sink failure caused by encoding
    /// them surfaces on a later call rather than being reported alongside a
    /// count that would make the caller send them twice.
    ///
    /// # Errors
    ///
    /// Propagates the sink's errors from the drain that precedes acceptance,
    /// and the encoder's own.
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
            // Previously accepted input can still have pending compressed
            // output. Drain it before trying to accept any new input.
            self.drain()?;
        }
    }

    /// Makes everything written so far decodable, without ending the stream.
    ///
    /// Compresses whatever is buffered, realigns the stream to a byte boundary
    /// and flushes the sink, so a reader on the far end can decode every byte
    /// written up to this point. See [`Operation::Flush`] for what it costs.
    ///
    /// Flushing an already-flushed stream with nothing written since emits
    /// nothing, exactly as the reference does.
    ///
    /// # Errors
    ///
    /// Propagates the sink's errors and the encoder's. Retryable: a failed
    /// flush leaves its bytes buffered and re-flushing resumes from there.
    fn flush(&mut self) -> Result<()> {
        if self.state != State::Open {
            return self.sink.flush();
        }
        self.drain()?;
        loop {
            let progress = self.pump(&[], Operation::Flush)?;
            self.drain()?;
            if progress.status != EncoderStatus::NeedsOutput {
                break;
            }
        }
        self.sink.flush()
    }
}

//! A [`Write`] adapter over an encoder session.

use std::io::{Error, ErrorKind, Result, Write};

use crate::compressor::session::{EncoderSession, EncoderStatus, Operation};
use crate::io::FinishError;

/// Room offered to the session per pull.
const PULL_CHUNK: usize = 128 * 1024;

/// Compresses everything written to it into an inner writer.
///
/// Compressed bytes are kept until the sink has taken them, so a sink that
/// writes short or fails loses nothing; the next call carries on. The frame
/// is only terminated by [`Self::try_finish`] or [`Self::finish`]; dropping
/// the adapter abandons it.
///
/// # Examples
///
/// ```
/// use mzstd::Compressor;
/// use std::io::Write;
///
/// let mut encoder = Compressor::new(Default::default())?;
/// let mut writer = encoder.writer(Vec::new(), Default::default());
/// writer.write_all(b"written payload")?;
/// writer.flush()?;
/// let frame = writer.finish().map_err(mzstd::io::FinishError::into_error)?;
/// assert!(!frame.is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct EncoderWriter<'c, W: Write> {
    session: EncoderSession<'c>,
    sink: W,
    buffer: Vec<u8>,
    /// Bytes of `buffer` not yet taken by the sink.
    pending: core::ops::Range<usize>,
}

impl<W: Write> std::fmt::Debug for EncoderWriter<'_, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderWriter")
            .field("session", &self.session)
            .field("pending", &self.pending.len())
            .finish_non_exhaustive()
    }
}

impl<'c, W: Write> EncoderWriter<'c, W> {
    pub(crate) fn new(session: EncoderSession<'c>, sink: W) -> Self {
        Self {
            session,
            sink,
            buffer: vec![0; PULL_CHUNK],
            pending: 0..0,
        }
    }

    /// The inner writer.
    #[must_use]
    pub const fn get_ref(&self) -> &W {
        &self.sink
    }

    /// Writes every pending compressed byte to the sink.
    fn drain(&mut self) -> Result<()> {
        while !self.pending.is_empty() {
            match self.sink.write(&self.buffer[self.pending.clone()]) {
                Ok(0) => {
                    return Err(Error::new(
                        ErrorKind::WriteZero,
                        "the sink accepted no bytes",
                    ));
                }
                Ok(n) => self.pending.start += n,
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// Runs `operation` until it needs nothing more, delivering output.
    fn run(&mut self, mut input: &[u8], operation: Operation) -> Result<usize> {
        let mut consumed = 0;
        loop {
            self.drain()?;
            let progress = self
                .session
                .process(input, &mut self.buffer, operation)
                .map_err(Error::other)?;
            consumed += progress.consumed;
            input = &input[progress.consumed..];
            self.pending = 0..progress.produced;
            self.drain()?;
            match progress.status {
                EncoderStatus::NeedsOutput => {}
                EncoderStatus::Finished => return Ok(consumed),
                EncoderStatus::NeedsInput => {
                    if operation != Operation::Continue || input.is_empty() {
                        return Ok(consumed);
                    }
                }
            }
        }
    }

    /// Ends the frame and flushes the sink, keeping the adapter so a failed
    /// attempt can be retried.
    ///
    /// # Errors
    ///
    /// Sink failures, after which the call may be repeated, and encoder
    /// failures.
    pub fn try_finish(&mut self) -> Result<()> {
        if !self.session.is_finished() {
            self.run(&[], Operation::End)?;
        }
        self.drain()?;
        self.sink.flush()
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
}

impl<W: Write> Write for EncoderWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        if self.session.is_finished() {
            return Err(Error::other("the frame has already been finished"));
        }
        self.drain()?;
        self.run(buf, Operation::Continue)
    }

    /// Makes everything written so far decodable and flushes the sink.
    fn flush(&mut self) -> Result<()> {
        if !self.session.is_finished() {
            self.run(&[], Operation::Flush)?;
        }
        self.drain()?;
        self.sink.flush()
    }
}

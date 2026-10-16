use alloc::boxed::Box;
use std::io::{self, ErrorKind, Write};

use super::super::{DecodeError, DecodeOperation, DecoderSession, DecoderStatus};
use crate::finish_error::FinishError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Lifecycle {
    Accepting,
    Finishing,
    Finished,
    Failed,
}

/// A compressed-input writer that writes the content to a sink.
///
/// Created by [`super::super::Decompressor::writer`]. Call [`Self::finish`]
/// or [`Self::try_finish`] to declare the end of the input, validate it and
/// flush the sink; dropping the writer performs no I/O and can lose
/// buffered content. [`Write::flush`] delivers the content decoded so far
/// but does not end the input.
///
/// When `write` accepts some input and then hits a decode or sink error, it
/// returns the accepted count and reports the error on the next call. Sink
/// errors keep undelivered content for a retry; decode errors are final.
#[derive(Debug)]
pub struct DecoderWriter<'d, W> {
    session: DecoderSession<'d>,
    sink: W,
    buffer: Box<[u8]>,
    cursor: usize,
    filled: usize,
    lifecycle: Lifecycle,
    pending_error: Option<io::Error>,
}

impl<'d, W: Write> DecoderWriter<'d, W> {
    pub(super) fn new(session: DecoderSession<'d>, sink: W) -> Self {
        Self {
            session,
            sink,
            buffer: super::buffer(),
            cursor: 0,
            filled: 0,
            lifecycle: Lifecycle::Accepting,
            pending_error: None,
        }
    }

    /// Borrows the content sink.
    pub const fn get_ref(&self) -> &W {
        &self.sink
    }

    /// Mutably borrows the content sink; writing to it directly interleaves
    /// with buffered content.
    pub const fn get_mut(&mut self) -> &mut W {
        &mut self.sink
    }

    /// Whether finalization succeeded: input validated, content delivered
    /// and the sink flushed.
    pub fn is_finished(&self) -> bool {
        self.lifecycle == Lifecycle::Finished
    }

    /// Reports a deferred error, then delivers buffered content.
    fn check(&mut self) -> io::Result<()> {
        if let Some(error) = self.pending_error.take() {
            return Err(error);
        }
        self.drain()?;
        if self.lifecycle == Lifecycle::Failed {
            return Err(DecodeError::InvalidState.into());
        }
        Ok(())
    }

    /// Writes the buffered content to the sink.
    fn drain(&mut self) -> io::Result<()> {
        while self.cursor < self.filled {
            match self.sink.write(&self.buffer[self.cursor..self.filled]) {
                Ok(0) => return Err(io::Error::from(ErrorKind::WriteZero)),
                Ok(written) => self.cursor += written,
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        self.cursor = 0;
        self.filled = 0;
        Ok(())
    }

    /// Runs the decoder once into the empty buffer and delivers the result.
    fn pump(
        &mut self,
        input: &[u8],
        operation: DecodeOperation,
    ) -> io::Result<(usize, DecoderStatus)> {
        match self.session.process(input, &mut self.buffer, operation) {
            Ok(progress) => {
                self.filled = progress.produced;
                match self.drain() {
                    Err(error) if progress.consumed != 0 => {
                        self.pending_error = Some(error);
                        Ok((progress.consumed, progress.status))
                    }
                    Err(error) => Err(error),
                    Ok(()) => Ok((progress.consumed, progress.status)),
                }
            }
            Err(failure) => {
                self.filled = failure.produced;
                self.lifecycle = Lifecycle::Failed;
                let error = io::Error::from(failure.error);
                if failure.consumed != 0 {
                    self.pending_error = Some(error);
                    Ok((failure.consumed, DecoderStatus::NeedsInput))
                } else {
                    Err(error)
                }
            }
        }
    }

    /// Ends the input, delivers the remaining content and flushes the sink.
    ///
    /// Succeeds again once finished. After the first call no more input is
    /// accepted; a sink error can be retried by calling it again.
    ///
    /// # Errors
    ///
    /// Reports truncated or invalid input, exceeded limits and sink errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{DecodeError, Decompressor};
    /// use std::io::Write;
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
    /// let mut decoder = Decompressor::default();
    /// let mut writer = decoder.writer(Vec::new());
    /// writer.write_all(&frame[..6])?;
    /// let error = writer.try_finish().unwrap_err();
    /// let decode_error = error.get_ref().and_then(|e| e.downcast_ref::<DecodeError>());
    /// assert!(matches!(decode_error, Some(DecodeError::UnexpectedEndOfInput)));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn try_finish(&mut self) -> io::Result<()> {
        if self.lifecycle == Lifecycle::Finished {
            return Ok(());
        }
        if self.lifecycle == Lifecycle::Accepting {
            self.lifecycle = Lifecycle::Finishing;
        }
        self.check()?;
        while !self.session.is_finished() {
            self.pump(&[], DecodeOperation::Finish)?;
            self.check()?;
        }
        self.sink.flush()?;
        self.lifecycle = Lifecycle::Finished;
        Ok(())
    }

    /// Finishes as [`Self::try_finish`] and returns the sink.
    ///
    /// # Errors
    ///
    /// Returns the error together with the writer, which can retry a sink
    /// error.
    pub fn finish(mut self) -> Result<W, FinishError<Self>> {
        match self.try_finish() {
            Ok(()) => Ok(self.sink),
            Err(error) => Err(FinishError::from_parts(error, self)),
        }
    }
}

impl<W: Write> Write for DecoderWriter<'_, W> {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        if input.is_empty() {
            return Ok(0);
        }
        self.check()?;
        if self.lifecycle != Lifecycle::Accepting {
            return Err(DecodeError::InvalidState.into());
        }
        if self.session.is_finished() {
            return Err(DecodeError::TrailingData {
                offset: self.session.total_in(),
            }
            .into());
        }
        loop {
            let (consumed, status) = self.pump(input, DecodeOperation::Process)?;
            if consumed != 0 {
                return Ok(consumed);
            }
            if status == DecoderStatus::Finished {
                return Err(DecodeError::TrailingData {
                    offset: self.session.total_in(),
                }
                .into());
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        self.check()?;
        if self.lifecycle == Lifecycle::Finishing {
            return self.try_finish();
        }
        while !self.session.is_finished() {
            let (_, status) = self.pump(&[], DecodeOperation::Process)?;
            self.check()?;
            if status != DecoderStatus::NeedsOutput {
                break;
            }
        }
        self.sink.flush()
    }
}

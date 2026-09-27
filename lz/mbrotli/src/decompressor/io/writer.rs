use super::super::{DecodeError, DecodeOperation, DecoderSession, DecoderStatus};
use crate::io::FinishError;
use std::io::{self, ErrorKind, Write};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Lifecycle {
    Accepting,
    Finishing,
    Finished,
    Failed,
}

/// Compressed-input writer with a bounded outbox of decompressed payload.
///
/// Construct with [`super::super::Decompressor::writer`]. Call [`Self::finish`]
/// or [`Self::try_finish`] to validate EOF and deliver all pending output;
/// dropping the writer performs no I/O and can discard buffered payload.
/// Ordinary [`Write::flush`] does not declare EOF or detect truncated input.
///
/// `write` may accept a prefix before encountering a codec or sink error. It
/// returns the accepted count and defers that error to the next write, flush,
/// or finalization. Sink errors preserve pending output for retry; codec errors
/// are terminal. Output already delivered is not rolled back on failure.
#[derive(Debug)]
pub struct DecoderWriter<'d, 'dict, W> {
    session: DecoderSession<'d, 'dict>,
    writer: W,
    buffer: Vec<u8>,
    // The fixed 8 KiB outbox fits in u16; compact cursors keep FinishError<Self> small.
    cursor: u16,
    filled: u16,
    lifecycle: Lifecycle,
    pending_error: Option<io::Error>,
}

impl<'d, 'dict, W: Write> DecoderWriter<'d, 'dict, W> {
    pub(super) fn new(session: DecoderSession<'d, 'dict>, writer: W) -> Result<Self, DecodeError> {
        Ok(Self {
            session,
            writer,
            buffer: super::buffer()?,
            cursor: 0,
            filled: 0,
            lifecycle: Lifecycle::Accepting,
            pending_error: None,
        })
    }
    /// Borrows the payload sink.
    pub const fn get_ref(&self) -> &W {
        &self.writer
    }
    /// Mutably borrows the sink. Writing around buffering can corrupt output.
    pub const fn get_mut(&mut self) -> &mut W {
        &mut self.writer
    }
    /// Whether finalization, outbox delivery and sink flushing succeeded.
    pub const fn is_finished(&self) -> bool {
        matches!(self.lifecycle, Lifecycle::Finished)
    }

    fn check_error(&mut self) -> io::Result<()> {
        if let Some(error) = self.pending_error.take() {
            return Err(error);
        }
        // A failed codec can still own bytes produced before its error. Sink
        // delivery remains retryable, even though decoding itself is terminal.
        self.drain()?;
        if self.lifecycle == Lifecycle::Failed {
            return Err(DecodeError::InvalidState.into());
        }
        Ok(())
    }

    fn drain(&mut self) -> io::Result<()> {
        while self.cursor < self.filled {
            match self
                .writer
                .write(&self.buffer[usize::from(self.cursor)..usize::from(self.filled)])
            {
                Ok(0) => return Err(io::Error::from(ErrorKind::WriteZero)),
                Ok(written) => self.cursor += written as u16,
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
        self.cursor = 0;
        self.filled = 0;
        Ok(())
    }

    /// Drives pending codec output while preserving the sink outbox on failures.
    fn pump(
        &mut self,
        input: &[u8],
        operation: DecodeOperation,
    ) -> io::Result<(usize, DecoderStatus)> {
        match self.session.process(input, &mut self.buffer, operation) {
            Ok(progress) => {
                self.filled = progress.produced as u16;
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
                self.filled = failure.produced as u16;
                self.lifecycle = Lifecycle::Failed;
                let error = failure.error.into();
                if failure.consumed != 0 {
                    self.pending_error = Some(error);
                    Ok((failure.consumed, DecoderStatus::NeedsInput))
                } else {
                    Err(error)
                }
            }
        }
    }

    /// Declares EOF, drains output, and flushes the sink. Sink failures may retry.
    ///
    /// # Errors
    /// Reports incomplete or invalid input, codec resource errors, and sink
    /// failures. After this call, new compressed input is no longer accepted.
    /// A successful call is idempotent; subsequent calls return `Ok(())`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecoderConfig, Decompressor};
    /// use std::io::Write;
    /// let mut decoder = Decompressor::new(DecoderConfig::default())?;
    /// let mut writer = decoder.writer(Vec::new(), Default::default())?;
    /// writer.write_all(&[0x3b])?;
    /// assert!(!writer.is_finished()); // Sink finalization is still pending.
    /// writer.try_finish()?;
    /// writer.try_finish()?;
    /// assert!(writer.is_finished());
    /// assert!(writer.get_ref().is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn try_finish(&mut self) -> io::Result<()> {
        if self.lifecycle == Lifecycle::Finished {
            return Ok(());
        }
        if self.lifecycle == Lifecycle::Accepting {
            self.lifecycle = Lifecycle::Finishing;
        }
        self.check_error()?;
        self.drain()?;
        while !self.session.is_finished() {
            self.pump(&[], DecodeOperation::Finish)?;
        }
        self.writer.flush()?;
        self.lifecycle = Lifecycle::Finished;
        Ok(())
    }

    /// Finishes and returns the sink, preserving this adapter on failure.
    ///
    /// # Errors
    /// Wraps [`Self::try_finish`] failures with the recoverable adapter.
    /// A retained adapter can retry sink failures, but cannot repair invalid or
    /// incomplete compressed input. See [`super::super::Decompressor::writer`]
    /// for successful finalization and [`FinishError`] for sink-retry mechanics.
    pub fn finish(mut self) -> Result<W, FinishError<Self>> {
        match self.try_finish() {
            Ok(()) => Ok(self.writer),
            Err(error) => Err(FinishError::from_parts(error, self)),
        }
    }
}

impl<W: Write> Write for DecoderWriter<'_, '_, W> {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        if input.is_empty() {
            return Ok(0);
        }
        self.check_error()?;
        if self.lifecycle != Lifecycle::Accepting {
            return Err(DecodeError::InvalidState.into());
        }
        if self.session.is_finished() {
            return Err(DecodeError::TrailingData {
                offset: self.session.total_in(),
            }
            .into());
        }
        self.drain()?;
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
        self.check_error()?;
        self.drain()?;
        if self.lifecycle == Lifecycle::Finishing {
            return self.try_finish();
        }
        while !self.session.is_finished() {
            let (_, status) = self.pump(&[], DecodeOperation::Process)?;
            if status == DecoderStatus::NeedsInput {
                break;
            }
        }
        self.writer.flush()
    }
}

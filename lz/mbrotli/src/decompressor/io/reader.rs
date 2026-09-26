use super::super::{DecodeError, DecodeOperation, DecoderSession, DecoderStatus};
use std::io::{self, ErrorKind, Read};

/// Decompressed view of a compressed reader, with bounded read-ahead.
///
/// Construct with [`super::super::Decompressor::reader`]. In single-member mode,
/// reading to EOF stops at the first member, even if the source has more bytes.
/// Recover that suffix with [`Self::into_parts`]. If a codec failure occurs after
/// producing bytes, `read` returns those bytes first and reports the error on
/// the next nonempty read. Continue reading until EOF to validate the operation.
///
/// Source I/O errors preserve state; retry after handling a transient error.
/// Codec failures are terminal. Dropping the reader performs no I/O.
#[derive(Debug)]
pub struct DecoderReader<'d, 'dict, R> {
    session: DecoderSession<'d, 'dict>,
    reader: R,
    buffer: Vec<u8>,
    cursor: usize,
    filled: usize,
    eof: bool,
    pending_error: Option<io::Error>,
    failed: bool,
}

/// Source ownership and unconsumed read-ahead after dismantling a reader.
///
/// Read `unread_input` before resuming `reader` to preserve byte order. These
/// fields do not retain decoder state, so they cannot resume a partial decode.
/// See [`DecoderReader::into_parts`] for recovering a protocol suffix.
#[derive(Debug)]
pub struct DecoderReaderParts<R> {
    /// Original compressed source at its current position.
    pub reader: R,
    /// Read-ahead preceding the source's current position, in original order.
    pub unread_input: Vec<u8>,
    /// Whether the codec confirmed the operation's completion.
    pub finished: bool,
    /// Deferred error not yet surfaced by a nonempty read.
    pub pending_error: Option<io::Error>,
}

impl<'d, 'dict, R: Read> DecoderReader<'d, 'dict, R> {
    pub(super) fn new(session: DecoderSession<'d, 'dict>, reader: R) -> Result<Self, DecodeError> {
        Ok(Self {
            session,
            reader,
            buffer: super::buffer()?,
            cursor: 0,
            filled: 0,
            eof: false,
            pending_error: None,
            failed: false,
        })
    }
    /// Borrows the underlying compressed source.
    pub const fn get_ref(&self) -> &R {
        &self.reader
    }
    /// Mutably borrows the source. Reading around the adapter can corrupt state.
    pub const fn get_mut(&mut self) -> &mut R {
        &mut self.reader
    }
    /// Whether the codec confirmed completion without a deferred codec error.
    pub const fn is_finished(&self) -> bool {
        self.session.is_finished() && !self.failed
    }
    /// Returns the source and unread read-ahead without performing I/O.
    ///
    /// Ends the decoder session even when it is incomplete. Inspect `finished`
    /// and `pending_error` before treating decoded bytes as a validated result.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecoderConfig, Decompressor};
    /// use std::io::{Cursor, Read};
    /// let input = [0x3b, b'O', b'K']; // Empty member followed by protocol data.
    /// let mut decoder = Decompressor::new(DecoderConfig::default())?;
    /// let mut reader = decoder.reader(&input[..], Default::default())?;
    /// let mut payload = Vec::new();
    /// reader.read_to_end(&mut payload)?;
    /// assert!(payload.is_empty());
    /// let parts = reader.into_parts();
    /// assert!(parts.finished);
    /// assert!(parts.pending_error.is_none());
    /// let mut rest = Cursor::new(parts.unread_input).chain(parts.reader);
    /// let mut suffix = Vec::new();
    /// rest.read_to_end(&mut suffix)?;
    /// assert_eq!(suffix, b"OK");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_parts(mut self) -> DecoderReaderParts<R> {
        let finished = self.is_finished();
        self.buffer.truncate(self.filled);
        self.buffer.drain(..self.cursor);
        DecoderReaderParts {
            reader: self.reader,
            unread_input: self.buffer,
            finished,
            pending_error: self.pending_error,
        }
    }
}

impl<R: Read> Read for DecoderReader<'_, '_, R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        if let Some(error) = self.pending_error.take() {
            return Err(error);
        }
        if self.failed {
            return Err(DecodeError::InvalidState.into());
        }
        loop {
            let operation = if self.eof {
                DecodeOperation::Finish
            } else {
                DecodeOperation::Process
            };
            match self
                .session
                .process(&self.buffer[self.cursor..self.filled], output, operation)
            {
                Ok(progress) => {
                    self.cursor += progress.consumed;
                    if progress.produced != 0 || progress.status == DecoderStatus::Finished {
                        return Ok(progress.produced);
                    }
                }
                Err(failure) => {
                    self.cursor += failure.consumed;
                    self.failed = true;
                    let error = failure.error.into();
                    if failure.produced == 0 {
                        return Err(error);
                    }
                    self.pending_error = Some(error);
                    return Ok(failure.produced);
                }
            }
            // NeedsInput guarantees that the current compressed slice was accepted.
            loop {
                match self.reader.read(&mut self.buffer) {
                    Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                    Err(error) => return Err(error),
                    Ok(length) => {
                        self.cursor = 0;
                        self.filled = length;
                        self.eof = length == 0;
                        break;
                    }
                }
            }
        }
    }
}

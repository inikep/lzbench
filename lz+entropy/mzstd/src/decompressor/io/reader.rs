use alloc::boxed::Box;
use alloc::vec::Vec;
use std::io::{self, ErrorKind, Read};

use super::super::{DecodeError, DecodeOperation, DecoderSession, DecoderStatus};

/// Decompressed view of a compressed reader.
///
/// Created by [`super::super::Decompressor::reader`]. Reading to the end
/// validates every frame; in [`super::super::FrameMode::Single`] it stops
/// after the frame, and [`Self::into_parts`] returns the bytes read ahead
/// past it. When a decode error follows some output, `read` returns that
/// output first and the error on the next call. Source errors leave the
/// state intact, so a read can be retried; decode errors are final.
#[derive(Debug)]
pub struct DecoderReader<'d, R> {
    session: DecoderSession<'d>,
    source: R,
    buffer: Box<[u8]>,
    cursor: usize,
    filled: usize,
    eof: bool,
    pending_error: Option<io::Error>,
    failed: bool,
}

/// The source and its unconsumed read-ahead, after
/// [`DecoderReader::into_parts`].
///
/// Read `unread_input` before resuming `source` to keep byte order.
#[derive(Debug)]
pub struct DecoderReaderParts<R> {
    /// The compressed source at its current position.
    pub source: R,
    /// Bytes read from `source` but not consumed by the decoder.
    pub unread_input: Vec<u8>,
    /// Whether the operation completed and validated.
    pub finished: bool,
    /// An error not yet returned by `read`.
    pub pending_error: Option<io::Error>,
}

impl<'d, R: Read> DecoderReader<'d, R> {
    pub(super) fn new(session: DecoderSession<'d>, source: R) -> Self {
        Self {
            session,
            source,
            buffer: super::buffer(),
            cursor: 0,
            filled: 0,
            eof: false,
            pending_error: None,
            failed: false,
        }
    }

    /// Borrows the compressed source.
    pub const fn get_ref(&self) -> &R {
        &self.source
    }

    /// Mutably borrows the compressed source; reading from it directly
    /// corrupts the stream.
    pub const fn get_mut(&mut self) -> &mut R {
        &mut self.source
    }

    /// Whether every frame was decoded and validated.
    pub fn is_finished(&self) -> bool {
        self.session.is_finished() && !self.failed
    }

    /// Ends the operation and returns the source and its read-ahead.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{Decompressor, FrameMode};
    /// use std::io::{Cursor, Read};
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
    /// let input = [&frame[..], b"rest"].concat();
    /// let mut decoder = Decompressor::new(FrameMode::Single.into());
    /// let mut reader = decoder.reader(&input[..]);
    /// let mut content = Vec::new();
    /// reader.read_to_end(&mut content)?;
    /// assert_eq!(content, b"ok");
    /// let parts = reader.into_parts();
    /// assert!(parts.finished);
    /// let mut rest = Vec::new();
    /// Cursor::new(parts.unread_input).chain(parts.source).read_to_end(&mut rest)?;
    /// assert_eq!(rest, b"rest");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_parts(self) -> DecoderReaderParts<R> {
        let finished = self.is_finished();
        let unread_input = self.buffer[self.cursor..self.filled].to_vec();
        DecoderReaderParts {
            source: self.source,
            unread_input,
            finished,
            pending_error: self.pending_error,
        }
    }
}

impl<R: Read> Read for DecoderReader<'_, R> {
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
                    let error = io::Error::from(failure.error);
                    if failure.produced == 0 {
                        return Err(error);
                    }
                    self.pending_error = Some(error);
                    return Ok(failure.produced);
                }
            }
            // `NeedsInput`: everything buffered was consumed.
            loop {
                match self.source.read(&mut self.buffer) {
                    Err(error) if error.kind() == ErrorKind::Interrupted => {}
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

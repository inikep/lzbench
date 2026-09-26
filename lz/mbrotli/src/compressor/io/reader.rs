//! A [`Read`] adapter over an encoder session.

use crate::compressor::session::{EncoderSession, EncoderStatus, Operation};
use std::io::{Error, ErrorKind, Read, Result};

/// How many source bytes one refill asks for.
const FILL_CHUNK: usize = 64 * 1024;

/// Yields the compressed form of an inner reader.
///
/// Source bytes are pulled in as the encoder needs them and handed to it from a
/// cursor, so nothing is ever moved to the front of a buffer. Compressed bytes
/// are written straight into the caller's slice: there is no intermediate queue
/// between the encoder and the reader's destination.
///
/// # Examples
///
/// ```
/// use mbrotli::{Compressor, EncoderConfig, InputSize, Quality};
/// use std::io::Read;
///
/// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
/// let payload = b"payload payload payload";
///
/// let streamed = {
///     let stream = InputSize::Exact(payload.len() as u64).into();
///     let mut source = encoder.reader(&payload[..], stream)?;
///     let mut compressed = Vec::new();
///     source.read_to_end(&mut compressed)?;
///     compressed
/// };
///
/// // The same bytes the one-shot path produces, for the same declared size.
/// assert_eq!(streamed, encoder.compress(payload)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct EncoderReader<'c, 'd, R: Read> {
    /// The stream being encoded.
    session: EncoderSession<'c, 'd>,
    /// Where source bytes come from.
    source: R,
    /// Source bytes read but not yet accepted by the encoder.
    input: Vec<u8>,
    /// How much of [`EncoderReader::input`] the encoder has taken.
    head: usize,
    /// Whether the source has reported end of file.
    eof: bool,
    /// Whether the compressed stream has been terminated and delivered.
    finished: bool,
}

impl<R: Read> std::fmt::Debug for EncoderReader<'_, '_, R> {
    /// Reports the buffered source bytes and whether the stream is done.
    ///
    /// Neither the source nor the session is shown: the source has no `Debug`
    /// bound, and the session is the compressor's business.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderReader")
            .field("buffered", &(self.input.len() - self.head))
            .field("eof", &self.eof)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl<'c, 'd, R: Read> EncoderReader<'c, 'd, R> {
    /// Wraps `source` around `session`.
    pub(crate) fn new(session: EncoderSession<'c, 'd>, source: R) -> Self {
        Self {
            session,
            source,
            input: Vec::new(),
            head: 0,
            eof: false,
            finished: false,
        }
    }

    /// Returns a shared reference to the inner reader.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q0))?;
    /// let source = encoder.reader(&b"data"[..], Default::default())?;
    ///
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
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q0))?;
    /// let mut source = encoder.reader(&b"data"[..], Default::default())?;
    ///
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
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    /// use std::io::Read;
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q0))?;
    /// let mut source = encoder.reader(&b"data"[..], Default::default())?;
    /// let mut compressed = Vec::new();
    ///
    /// assert!(!source.is_finished());
    /// source.read_to_end(&mut compressed)?;
    /// assert!(source.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn is_finished(&self) -> bool {
        self.finished
    }

    /// Takes the adapter apart, returning the source and what it had read.
    ///
    /// A reader has to read ahead of what the encoder has accepted, so
    /// abandoning one would otherwise swallow the bytes in between. They come
    /// back here instead, in the order the source produced them, so a caller
    /// can hand them to whatever takes over.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    /// use std::io::Read;
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
    /// let payload = b"a payload the caller may want back".repeat(10);
    ///
    /// let mut source = encoder.reader(payload.as_slice(), Default::default())?;
    /// let mut head = [0u8; 4];
    /// source.read(&mut head)?;
    ///
    /// let parts = source.into_parts();
    /// // Whatever was read ahead of the encoder is handed back, never dropped.
    /// assert!(parts.buffered_input.len() <= payload.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn into_parts(self) -> EncoderReaderParts<R> {
        let buffered_input = self.input.get(self.head..).unwrap_or_default().to_vec();
        EncoderReaderParts {
            inner: self.source,
            buffered_input,
        }
    }

    /// Reads the next stretch of source bytes, retrying an interruption.
    fn fill(&mut self) -> Result<()> {
        self.input.clear();
        self.head = 0;
        loop {
            self.input.resize(FILL_CHUNK, 0);
            let outcome = self.source.read(&mut self.input);
            match outcome {
                Ok(0) => {
                    self.input.clear();
                    self.eof = true;
                    return Ok(());
                }
                Ok(count) => {
                    self.input.truncate(count);
                    return Ok(());
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => {
                    self.input.clear();
                }
                Err(error) => {
                    self.input.clear();
                    return Err(error);
                }
            }
        }
    }
}

impl<R: Read> Read for EncoderReader<'_, '_, R> {
    /// Fills `buf` with the next compressed bytes.
    ///
    /// A zero-length destination reads nothing from the source and initialises
    /// nothing. A source error propagates without losing or duplicating
    /// anything already encoded: the bytes it had not yet produced are simply
    /// not there yet, and a retry picks up where it left off.
    ///
    /// # Errors
    ///
    /// Propagates the source's errors and the encoder's.
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        if buf.is_empty() || self.finished {
            return Ok(0);
        }
        loop {
            if self.head == self.input.len() && !self.eof {
                self.fill()?;
            }
            let operation = if self.eof {
                Operation::Finish
            } else {
                Operation::Process
            };
            let progress = {
                let pending = self.input.get(self.head..).unwrap_or_default();
                self.session
                    .process(pending, buf, operation)
                    .map_err(Error::from)?
            };
            self.head += progress.consumed;
            if self.head == self.input.len() {
                self.input.clear();
                self.head = 0;
            }
            if progress.status == EncoderStatus::Finished {
                self.finished = true;
            }
            if progress.produced > 0 {
                return Ok(progress.produced);
            }
            if self.finished {
                return Ok(0);
            }
        }
    }
}

/// What an [`EncoderReader`] was made of, once it is taken apart.
///
/// # Examples
///
/// ```
/// use mbrotli::{Compressor, EncoderConfig, Quality};
///
/// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
/// let source = encoder.reader(&b"payload"[..], Default::default())?;
///
/// let parts = source.into_parts();
/// assert_eq!(parts.inner.len(), 7);
/// assert!(parts.buffered_input.is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct EncoderReaderParts<R> {
    /// The source the adapter was reading from.
    pub inner: R,
    /// Bytes read from the source that the encoder had not yet accepted.
    pub buffered_input: Vec<u8>,
}

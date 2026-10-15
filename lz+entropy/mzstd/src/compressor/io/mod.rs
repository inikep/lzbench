//! `std::io` adapters over compression sessions.

mod reader;
mod writer;

pub use reader::{EncoderReader, EncoderReaderParts};
pub use writer::EncoderWriter;

use std::io::{Read, Write};

use super::dictionary::PreparedDictionary;
use super::encoder::Compressor;
use super::error::EncodeError;
use super::session::StreamConfig;

impl Compressor {
    /// Compresses everything written to the returned adapter into `sink`.
    ///
    /// The frame ends only with [`EncoderWriter::finish`]; dropping the
    /// adapter abandons it.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::{Compressor, InputSize};
    /// use std::io::Write;
    ///
    /// let mut encoder = Compressor::new(Default::default())?;
    /// let payload = b"chunk one chunk two ";
    /// let mut sink = encoder.writer(Vec::new(), InputSize::Exact(payload.len() as u64).into());
    /// for chunk in payload.chunks(4) {
    ///     sink.write_all(chunk)?;
    /// }
    /// let frame = sink.finish().map_err(mzstd::io::FinishError::into_error)?;
    /// assert!(!frame.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn writer<W: Write>(&mut self, sink: W, stream: StreamConfig) -> EncoderWriter<'_, W> {
        EncoderWriter::new(self.start(stream), sink)
    }

    /// Reads `source` to its end and yields the compressed frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::Compressor;
    /// use std::io::Read;
    ///
    /// let mut encoder = Compressor::new(Default::default())?;
    /// let mut frame = Vec::new();
    /// encoder.reader(&b"pulled payload"[..], Default::default()).read_to_end(&mut frame)?;
    /// assert!(!frame.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reader<R: Read>(&mut self, source: R, stream: StreamConfig) -> EncoderReader<'_, R> {
        EncoderReader::new(self.start(stream), source)
    }

    /// [`Self::writer`] compressing with `dictionary`.
    ///
    /// # Errors
    ///
    /// As [`Compressor::start_with_dictionary`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DictionaryBuilder;
    /// use mzstd::Compressor;
    /// use std::io::Write;
    ///
    /// let dictionary = DictionaryBuilder::new(b"a common prefix").build()?;
    /// let mut encoder = Compressor::default();
    /// let mut sink = encoder.writer_with_dictionary(&dictionary, Vec::new(), Default::default())?;
    /// sink.write_all(b"a common prefix")?;
    /// let frame = sink.finish().map_err(mzstd::io::FinishError::into_error)?;
    /// let restored = mzstd::Decompressor::default().decompress_with_dictionary(&dictionary, &frame)?;
    /// assert_eq!(restored, b"a common prefix");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn writer_with_dictionary<'c, W: Write>(
        &'c mut self,
        dictionary: &'c PreparedDictionary,
        sink: W,
        stream: StreamConfig,
    ) -> Result<EncoderWriter<'c, W>, EncodeError> {
        Ok(EncoderWriter::new(
            self.start_with_dictionary(dictionary, stream)?,
            sink,
        ))
    }

    /// [`Self::reader`] compressing with `dictionary`.
    ///
    /// # Errors
    ///
    /// As [`Compressor::start_with_dictionary`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DictionaryBuilder;
    /// use mzstd::Compressor;
    /// use std::io::Read;
    ///
    /// let dictionary = DictionaryBuilder::new(b"a common prefix").build()?;
    /// let mut encoder = Compressor::default();
    /// let mut frame = Vec::new();
    /// encoder
    ///     .reader_with_dictionary(&dictionary, &b"a common prefix"[..], Default::default())?
    ///     .read_to_end(&mut frame)?;
    /// assert!(!frame.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reader_with_dictionary<'c, R: Read>(
        &'c mut self,
        dictionary: &'c PreparedDictionary,
        source: R,
        stream: StreamConfig,
    ) -> Result<EncoderReader<'c, R>, EncodeError> {
        Ok(EncoderReader::new(
            self.start_with_dictionary(dictionary, stream)?,
            source,
        ))
    }
}

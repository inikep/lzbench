//! `std::io` adapters over an encoder session.
//!
//! Both adapters are thin: they own buffering and the conventions `Read` and
//! `Write` impose, and hand every encoding decision to
//! [`EncoderSession`](super::EncoderSession). Two shapes of the same stream
//! therefore cannot disagree, because there is only one encoder underneath.

mod reader;
mod writer;

pub use crate::io::FinishError;
pub use reader::{EncoderReader, EncoderReaderParts};
pub use writer::EncoderWriter;

use super::dictionary::PreparedDictionary;
use super::encoder::Compressor;
use super::error::EncodeError;
use super::session::StreamConfig;
use std::io::{Read, Write};

impl Compressor {
    /// Wraps `writer` in an adapter that compresses everything written to it.
    ///
    /// The stream is only terminated by
    /// [`EncoderWriter::try_finish`](EncoderWriter::try_finish) or
    /// [`EncoderWriter::finish`](EncoderWriter::finish); `Write` has no closing
    /// hook, and a meta-block boundary need not land on a byte boundary.
    /// Dropping the adapter abandons the stream.
    ///
    /// # Errors
    ///
    /// As [`Compressor::start`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, InputSize, Quality};
    /// use std::io::Write;
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
    /// let payload = b"chunk one chunk two";
    ///
    /// let compressed = {
    ///     let stream = InputSize::Exact(payload.len() as u64).into();
    ///     let mut sink = encoder.writer(Vec::new(), stream)?;
    ///     sink.write_all(b"chunk one ")?;
    ///     sink.write_all(b"chunk two")?;
    ///     sink.finish().map_err(mbrotli::io::FinishError::into_error)?
    /// };
    ///
    /// assert!(!compressed.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn writer<W: Write>(
        &mut self,
        writer: W,
        stream: StreamConfig,
    ) -> Result<EncoderWriter<'_, 'static, W>, EncodeError> {
        let session = self.start(stream)?;
        Ok(EncoderWriter::new(session, writer))
    }

    /// Wraps `writer` in an adapter compressing against `dictionary`.
    ///
    /// # Errors
    ///
    /// As [`Compressor::start_with_dictionary`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    /// use std::io::Write;
    ///
    /// let dictionary = DictionaryBuilder::new().add_prefix(&b"a common prefix"[..]).build()?;
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
    ///
    /// let mut sink = encoder.writer_with_dictionary(&dictionary, Vec::new(), Default::default())?;
    /// sink.write_all(b"a common prefix")?;
    /// let compressed = sink.finish().map_err(mbrotli::io::FinishError::into_error)?;
    ///
    /// assert!(!compressed.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn writer_with_dictionary<'c, 'd, W: Write>(
        &'c mut self,
        dictionary: &'d PreparedDictionary,
        writer: W,
        stream: StreamConfig,
    ) -> Result<EncoderWriter<'c, 'd, W>, EncodeError> {
        let session = self.start_with_dictionary(dictionary, stream)?;
        Ok(EncoderWriter::new(session, writer))
    }

    /// Wraps `reader` in an adapter that yields the compressed stream.
    ///
    /// # Errors
    ///
    /// As [`Compressor::start`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{Compressor, EncoderConfig, InputSize, Quality};
    /// use std::io::Read;
    ///
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
    /// let payload = b"streamed payload";
    ///
    /// let mut source = encoder.reader(&payload[..], InputSize::Exact(payload.len() as u64).into())?;
    /// let mut compressed = Vec::new();
    /// source.read_to_end(&mut compressed)?;
    ///
    /// assert!(!compressed.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reader<R: Read>(
        &mut self,
        reader: R,
        stream: StreamConfig,
    ) -> Result<EncoderReader<'_, 'static, R>, EncodeError> {
        let session = self.start(stream)?;
        Ok(EncoderReader::new(session, reader))
    }

    /// Wraps `reader` in an adapter compressing against `dictionary`.
    ///
    /// # Errors
    ///
    /// As [`Compressor::start_with_dictionary`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::dictionary::DictionaryBuilder;
    /// use mbrotli::{Compressor, EncoderConfig, Quality};
    /// use std::io::Read;
    ///
    /// let dictionary = DictionaryBuilder::new().add_prefix(&b"a common prefix"[..]).build()?;
    /// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
    ///
    /// let mut source =
    ///     encoder.reader_with_dictionary(&dictionary, &b"a common prefix"[..], Default::default())?;
    /// let mut compressed = Vec::new();
    /// source.read_to_end(&mut compressed)?;
    ///
    /// assert!(!compressed.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reader_with_dictionary<'c, 'd, R: Read>(
        &'c mut self,
        dictionary: &'d PreparedDictionary,
        reader: R,
        stream: StreamConfig,
    ) -> Result<EncoderReader<'c, 'd, R>, EncodeError> {
        let session = self.start_with_dictionary(dictionary, stream)?;
        Ok(EncoderReader::new(session, reader))
    }
}

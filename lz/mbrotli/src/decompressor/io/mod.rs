//! Bounded synchronous adapters over the same decoder session as one-shot APIs.
//!
//! Each adapter owns an 8 KiB byte buffer. Source/sink errors preserve codec
//! progress, and dropping an adapter never performs I/O.
//!
//! Use [`Decompressor::reader`] to pull payload from compressed input, or
//! [`Decompressor::writer`] to push compressed input into a payload sink. In
//! single-member mode, reader EOF validates that member without rejecting
//! read-ahead. Writer finalization explicitly declares EOF and flushes the sink.
//! Decode errors are retained inside [`std::io::Error`] and can be inspected with `get_ref`
//! and `downcast_ref::<DecodeError>()`.

mod reader;
mod writer;

pub use reader::{DecoderReader, DecoderReaderParts};
pub use writer::DecoderWriter;

use super::{DecodeError, DecodeStreamConfig, Decompressor};
use crate::dictionary::DictionaryRef;
use std::io::{Read, Write};

const BUFFER_BYTES: usize = 8192;

fn buffer() -> Result<Vec<u8>, DecodeError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(BUFFER_BYTES)
        .map_err(|_| DecodeError::AllocationFailed)?;
    bytes.resize(BUFFER_BYTES, 0);
    Ok(bytes)
}

impl Decompressor {
    /// Wraps a compressed source and produces decompressed payload.
    ///
    /// # Errors
    /// Returns session-start or fallible adapter-buffer allocation errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecoderConfig, Decompressor};
    /// use std::io::Read;
    /// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
    /// let mut decoder = Decompressor::new(DecoderConfig::default())?;
    /// let mut reader = decoder.reader(&compressed[..], Default::default())?;
    /// let mut payload = Vec::new();
    /// reader.read_to_end(&mut payload)?;
    /// assert_eq!(payload, b"hello");
    /// assert!(reader.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reader<R: Read>(
        &mut self,
        reader: R,
        stream: DecodeStreamConfig,
    ) -> Result<DecoderReader<'_, 'static, R>, DecodeError> {
        DecoderReader::new(self.start(stream)?, reader)
    }
    /// Wraps a source using a dictionary borrowed for the reader's lifetime.
    ///
    /// Uses the same pull loop as [`Self::reader`]; the dictionary must match
    /// the encoder's attachments. See [`crate::dictionary::DecodeDictionary`]
    /// for construction and ownership.
    ///
    /// # Errors
    /// Returns session-start or fallible adapter-buffer allocation errors.
    pub fn reader_with_dictionary<'d, 'dict, R: Read>(
        &'d mut self,
        dictionary: impl Into<DictionaryRef<'dict>>,
        reader: R,
        stream: DecodeStreamConfig,
    ) -> Result<DecoderReader<'d, 'dict, R>, DecodeError> {
        DecoderReader::new(self.start_with_dictionary(dictionary, stream)?, reader)
    }
    /// Wraps a payload sink and accepts compressed input through `Write`.
    ///
    /// # Errors
    /// Returns session-start or fallible adapter-buffer allocation errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::{DecoderConfig, Decompressor};
    /// use std::io::Write;
    /// let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
    /// let mut decoder = Decompressor::new(DecoderConfig::default())?;
    /// let mut writer = decoder.writer(Vec::new(), Default::default())?;
    /// writer.write_all(&compressed)?;
    /// let payload = writer.finish().map_err(mbrotli::io::FinishError::into_error)?;
    /// assert_eq!(payload, b"hello");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn writer<W: Write>(
        &mut self,
        writer: W,
        stream: DecodeStreamConfig,
    ) -> Result<DecoderWriter<'_, 'static, W>, DecodeError> {
        DecoderWriter::new(self.start(stream)?, writer)
    }
    /// Wraps a sink using an independently borrowed external dictionary.
    ///
    /// Use [`DecoderWriter::finish`] to complete the operation, as in
    /// [`Self::writer`]. The dictionary must match the encoder's attachments;
    /// raw Brotli does not authenticate dictionary identity.
    ///
    /// # Errors
    /// Returns session-start or fallible adapter-buffer allocation errors.
    pub fn writer_with_dictionary<'d, 'dict, W: Write>(
        &'d mut self,
        dictionary: impl Into<DictionaryRef<'dict>>,
        writer: W,
        stream: DecodeStreamConfig,
    ) -> Result<DecoderWriter<'d, 'dict, W>, DecodeError> {
        DecoderWriter::new(self.start_with_dictionary(dictionary, stream)?, writer)
    }
}

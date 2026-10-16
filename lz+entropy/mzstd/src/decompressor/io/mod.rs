//! `std::io` adapters over a decoder session.
//!
//! [`Decompressor::reader`] wraps a compressed [`Read`] source and reads out
//! the decompressed content; [`Decompressor::writer`] accepts compressed
//! bytes through [`Write`] and writes the content to a sink. Each adapter
//! owns one fixed 8 KiB buffer. Decode errors travel inside
//! [`std::io::Error`]; recover them with `get_ref` and
//! `downcast_ref::<DecodeError>()`. A failed [`DecoderWriter::finish`]
//! returns the writer inside the crate's shared
//! [`FinishError`](crate::io::FinishError).
//!
//! [`DecodeError`]: super::DecodeError

mod reader;
mod writer;

pub use reader::{DecoderReader, DecoderReaderParts};
pub use writer::DecoderWriter;

use alloc::boxed::Box;
use std::io::{Read, Write};

use super::Decompressor;
use crate::dictionary::DictionaryRef;

/// Size of each adapter's buffer.
const BUFFER_SIZE: usize = 8 << 10;

/// Allocates an adapter buffer.
fn buffer() -> Box<[u8]> {
    alloc::vec![0; BUFFER_SIZE].into_boxed_slice()
}

impl Decompressor {
    /// Wraps a compressed source; reading yields the decompressed content.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::Decompressor;
    /// use std::io::Read;
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
    /// let mut decoder = Decompressor::default();
    /// let mut reader = decoder.reader(&frame[..]);
    /// let mut content = Vec::new();
    /// reader.read_to_end(&mut content)?;
    /// assert_eq!(content, b"ok");
    /// assert!(reader.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reader<R: Read>(&mut self, source: R) -> DecoderReader<'_, R> {
        DecoderReader::new(self.start(), source)
    }

    /// Wraps a content sink; writing compressed bytes writes their content.
    ///
    /// Call [`DecoderWriter::finish`] to validate the end of the input and
    /// flush the sink; dropping the writer does neither.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::Decompressor;
    /// use std::io::Write;
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
    /// let mut decoder = Decompressor::default();
    /// let mut writer = decoder.writer(Vec::new());
    /// writer.write_all(&frame)?;
    /// let content = writer.finish().map_err(mzstd::io::FinishError::into_error)?;
    /// assert_eq!(content, b"ok");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn writer<W: Write>(&mut self, sink: W) -> DecoderWriter<'_, W> {
        DecoderWriter::new(self.start(), sink)
    }

    /// Wraps a compressed source as [`Self::reader`] does, decoding every
    /// frame with `dictionary`, which stays borrowed with the reader.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DecodeDictionary;
    /// use mzstd::Decompressor;
    /// use std::io::Read;
    /// let dictionary = DecodeDictionary::try_from(&b"history"[..])?;
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
    /// let mut decoder = Decompressor::default();
    /// let mut content = Vec::new();
    /// decoder
    ///     .reader_with_dictionary(&frame[..], &dictionary)
    ///     .read_to_end(&mut content)?;
    /// assert_eq!(content, b"ok");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reader_with_dictionary<'d, R: Read>(
        &'d mut self,
        source: R,
        dictionary: impl Into<DictionaryRef<'d>>,
    ) -> DecoderReader<'d, R> {
        DecoderReader::new(self.start_with_dictionary(dictionary), source)
    }

    /// Wraps a content sink as [`Self::writer`] does, decoding every frame
    /// with `dictionary`, which stays borrowed with the writer.
    ///
    /// # Examples
    ///
    /// ```
    /// use mzstd::dictionary::DecodeDictionary;
    /// use mzstd::Decompressor;
    /// use std::io::Write;
    /// let dictionary = DecodeDictionary::try_from(&b"history"[..])?;
    /// let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x02, 0x11, 0x00, 0x00, b'o', b'k'];
    /// let mut decoder = Decompressor::default();
    /// let mut writer = decoder.writer_with_dictionary(Vec::new(), &dictionary);
    /// writer.write_all(&frame)?;
    /// let content = writer.finish().map_err(mzstd::io::FinishError::into_error)?;
    /// assert_eq!(content, b"ok");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn writer_with_dictionary<'d, W: Write>(
        &'d mut self,
        sink: W,
        dictionary: impl Into<DictionaryRef<'d>>,
    ) -> DecoderWriter<'d, W> {
        DecoderWriter::new(self.start_with_dictionary(dictionary), sink)
    }
}

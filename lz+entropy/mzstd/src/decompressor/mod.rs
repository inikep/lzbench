//! Zstandard (RFC 8878) decompression.
//!
//! A [`Decompressor`] owns reusable buffers and a [`DecoderConfig`]: the
//! [`FrameMode`] (all concatenated frames, or a single one) and the
//! [`DecodeLimits`] that bound the window, input and output of untrusted
//! data. It decodes complete input in one call, or incrementally through a
//! [`DecoderSession`] (borrowing the decoder) or [`DecoderSessionOwned`]
//! (owning it), and with `std` through the `io` adapters.
//!
//! Every Zstandard frame feature is supported: frames of raw, RLE and
//! compressed blocks, Huffman-coded literals in one or four streams,
//! predefined, RLE, FSE-compressed and repeated sequence tables, repeat
//! offsets, content sizes and content checksums, which are always verified,
//! and dictionaries, raw or in the Zstandard format
//! ([`crate::dictionary::DecodeDictionary`]), through the
//! `*_with_dictionary` methods. Skippable frames are skipped. A frame that
//! names a dictionary fails with [`DecodeError::DictionaryUnsupported`]
//! without one and with [`DecodeError::DictionaryMismatch`] with another.
//!
//! # Examples
//!
//! One-shot decoding, reusing the decoder:
//!
//! ```
//! use mzstd::Decompressor;
//! // "hello" as one raw block in a single-segment frame.
//! let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x05, 0x29, 0x00, 0x00, b'h', b'e', b'l', b'l', b'o'];
//! let mut decoder = Decompressor::default();
//! assert_eq!(decoder.decompress(&frame)?, b"hello");
//! assert_eq!(decoder.decompress(&frame)?, b"hello");
//! # Ok::<(), mzstd::DecodeError>(())
//! ```
//!
//! Limits for untrusted input:
//!
//! ```
//! use mzstd::{DecodeError, DecodeLimits, DecoderConfig, Decompressor};
//! let limits = DecodeLimits::default()
//!     .with_max_window_log(20)?
//!     .with_max_output_bytes(Some(4));
//! let mut decoder = Decompressor::new(DecoderConfig::default().with_limits(limits));
//! let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x05, 0x29, 0x00, 0x00, b'h', b'e', b'l', b'l', b'o'];
//! assert!(matches!(
//!     decoder.decompress(&frame),
//!     Err(DecodeError::OutputLimitExceeded { limit: 4 })
//! ));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod config;
mod core;
mod decoder;
mod error;
#[cfg(feature = "framed")]
pub mod framing;
mod session;

#[cfg(not(feature = "no_std"))]
pub mod io;

#[cfg(test)]
pub(crate) use self::core::test_dictionary;
pub(crate) use self::core::{DecoderEntropy, DictView};
pub use config::{DecodeLimits, DecoderConfig, FrameMode};
pub use decoder::{Decompressor, DecompressorBuilder};
pub use error::{DecodeConfigError, DecodeError, InvalidDataKind};
pub use session::{
    DecodeFailure, DecodeOperation, DecodeProgress, DecoderSession, DecoderSessionOwned,
    DecoderStatus,
};

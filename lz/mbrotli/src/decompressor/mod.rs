//! Incremental Brotli decompression and explicit resource policies.
//!
//! One decoder owns reusable workspace; each session borrows it exclusively,
//! or, as a [`DecoderSessionOwned`], takes ownership of it until handed back.
//! RAW prefixes and standard/large windows are available in every profile.
//! Serialized/custom dictionary extensions require `experimental`.
//!
//! # Examples
//!
//! A strict one-shot operation rejects trailing bytes and can reuse storage:
//! ```
//! use mbrotli::{DecoderConfig, Decompressor};
//! // An uncompressed Brotli meta-block containing "hello".
//! let compressed = [0x0b, 0x02, 0x80, b'h', b'e', b'l', b'l', b'o', 0x03];
//! let mut decoder = Decompressor::new(DecoderConfig::default())?;
//! assert_eq!(decoder.decompress(&compressed)?, b"hello");
//! assert_eq!(decoder.decompress(&compressed)?, b"hello");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Set explicit budgets for untrusted data. Defaults impose no numeric limits;
//! an extended header alone never allocates its declared window size:
//! ```
//! use mbrotli::{DecodeLimits, DecoderConfig, Decompressor, WindowLimit};
//! let limits = DecodeLimits::default()
//!     .with_max_input_bytes(Some(1 << 20))
//!     .with_max_output_bytes(Some(8 << 20))
//!     .with_max_workspace_bytes(Some(32 << 20));
//! let config = DecoderConfig::default()
//!     .with_window_limit(WindowLimit::standard(24)?)
//!     .with_limits(limits);
//! let mut decoder = Decompressor::new(config)?;
//! assert!(decoder.decompress(&[0x3b])?.is_empty());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! A session stops exactly at its first member, preserving protocol suffixes:
//! ```
//! use mbrotli::{DecodeOperation, DecodeStreamConfig, DecoderConfig, DecoderStatus, Decompressor};
//! let mut decoder = Decompressor::new(DecoderConfig::default())?;
//! let mut session = decoder.start(DecodeStreamConfig::default())?;
//! let progress = session.process(&[0x3b, 0xaa], &mut [], DecodeOperation::Process)?;
//! assert_eq!(progress.consumed, 1);
//! assert_eq!(progress.status, DecoderStatus::Finished);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod config;
mod core;
mod decoder;
mod error;
mod session;

#[cfg(not(feature = "no_std"))]
pub mod io;

pub use decoder::{Decompressor, DecompressorBuilder};
pub use session::{
    DecodeFailure, DecodeOperation, DecodeProgress, DecoderSession, DecoderSessionOwned,
    DecoderStatus,
};

pub use config::{
    DecodeLimits, DecodeStreamConfig, DecoderConfig, MemberMode, OutputSize, WindowLimit,
};
pub use error::{DecodeConfigError, DecodeError, InvalidDataKind};

/// Experimental structured container decoding.
#[cfg(feature = "experimental")]
pub mod framing;

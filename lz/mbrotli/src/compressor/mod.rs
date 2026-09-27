//! The compressor: configuration, the encoder itself, and everything it needs.
//!
//! The surface is four ideas that do not overlap:
//!
//! - [`EncoderConfig`] is the settings that hold for every stream — quality,
//!   window, block size, mode, distance layout, literal contexts.
//! - [`Compressor`] is a reusable encoder built from one of those. It owns the
//!   workspace, so reuse is what ordinary code gets rather than an advanced
//!   variant of it, and every encoding method takes `&mut self`.
//! - [`StreamConfig`] is what one stream knows about itself: how many bytes are
//!   coming, and where it starts.
//! - [`PreparedDictionary`](dictionary::PreparedDictionary) is immutable
//!   knowledge many compressors can share at once.
//!
//! [`EncoderSession`] is the state machine underneath all of it, and the
//! `io` adapters (available without `no_std`) are conveniences over that.

mod config;
pub(crate) mod core;
mod encoder;
mod error;
mod internal;
mod session;
mod shared;

pub mod dictionary;
#[cfg(feature = "experimental")]
pub mod framing;
#[cfg(not(feature = "no_std"))]
pub mod io;
#[cfg(not(feature = "no_std"))]
pub mod parallel;

pub use crate::{Backend, RetentionPolicy};
pub use config::{
    BlockBits, BlockSize, CompressionMode, ConfigError, DistanceParams, EncoderConfig,
    LiteralContextMode, Quality, SizeOverflow, Window, WindowEncoding,
};
pub use encoder::{Compressor, CompressorBuilder};
pub use error::EncodeError;
pub use session::{
    EncoderSession, EncoderSessionOwned, EncoderStatus, InputSize, Operation, Progress,
    StreamConfig,
};

// The `core` tree is written against the encoders' own parameter and error
// shapes. They stay reachable under their original names so that redesigning
// the public surface moved no code the bitstream depends on.
pub(crate) use internal::{
    BrotliCompressError, BrotliResult, CompressMode, CompressParams, DistanceCodes, QualityLevel,
};
// Named only by the encoders' own unit tests, which build a window directly
// rather than lowering one from the public `Window`.
#[cfg(test)]
pub(crate) use internal::{WindowBits, WindowOutOfRange};

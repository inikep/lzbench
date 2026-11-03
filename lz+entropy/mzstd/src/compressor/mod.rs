//! Zstandard compression.
//!
//! [`Compressor`] owns the match finder tables and reuses them between
//! frames. One-shot methods compress a whole input at once; an
//! [`EncoderSession`] compresses a frame incrementally; the `io` adapters
//! wrap a session in `std::io::Read` and `std::io::Write`.

mod config;
pub(crate) mod core;
pub(crate) mod dictionary;
mod encoder;
mod error;
#[cfg(feature = "framed")]
pub mod framing;
mod internal;
#[cfg(not(feature = "no_std"))]
pub mod io;
#[cfg(not(feature = "no_std"))]
pub mod parallel;
mod session;

pub use config::{
    CompressionParams, ConfigError, EncoderConfig, LdmParams, Level, LongDistanceMatching,
    Parameter, SizeOverflow, Strategy,
};
pub use encoder::{Compressor, CompressorBuilder};
pub use error::EncodeError;
pub use session::{
    EncoderSession, EncoderSessionOwned, EncoderStatus, InputSize, Operation, Progress,
    StreamConfig,
};

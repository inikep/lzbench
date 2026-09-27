//! Structured, incremental RFC 9841 decoding. Payload is provisional until completion.
//!
//! ```
//! use mbrotli::framing::{FramedDecompressor, FramedDecodeConfig, InputMode, OutputStructure};
//! let mut decoder = FramedDecompressor::new(
//!     FramedDecodeConfig::default().with_input_mode(InputMode::Auto))?;
//! let output = decoder.decompress(&[0x3b])?;
//! assert!(matches!(output.structure, OutputStructure::Raw));
//! assert!(output.resources[0].data.is_empty());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
mod config;
mod core;
mod error;
mod owner;
#[cfg(not(feature = "no_std"))]
mod reader;
#[cfg(not(feature = "no_std"))]
mod seek;
#[cfg(not(feature = "no_std"))]
pub use seek::*;
mod session;
mod types;
pub use config::*;
pub use error::*;
pub use owner::*;
#[cfg(not(feature = "no_std"))]
pub use reader::*;
pub use session::*;
pub use types::*;

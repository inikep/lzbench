//! Structured, incremental decoding of framed containers. Payload is
//! provisional until completion.
//!
//! ```
//! use mzstd::framing::{FramedDecompressor, FramedDecodeConfig, InputMode, OutputStructure};
//! let mut decoder = FramedDecompressor::new(
//!     FramedDecodeConfig::default().with_input_mode(InputMode::Auto))?;
//! // An empty Zstandard frame is raw input: one empty resource.
//! let output = decoder.decompress(&[0x28, 0xb5, 0x2f, 0xfd, 0x20, 0, 1, 0, 0])?;
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
mod session;
mod types;

pub use config::*;
pub use error::*;
pub use owner::*;
#[cfg(not(feature = "no_std"))]
pub use reader::*;
#[cfg(not(feature = "no_std"))]
pub use seek::*;
pub use session::*;
pub use types::*;

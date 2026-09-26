//! Streaming I/O adapters for compression and decompression.

#[cfg(feature = "compression")]
pub use crate::compressor::io::*;
#[cfg(feature = "decompression")]
pub use crate::decompressor::io::*;

pub use crate::finish_error::FinishError;

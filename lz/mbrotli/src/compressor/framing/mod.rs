//! Experimental structured container compression over one reusable workspace.
//!
//! Native sessions and borrowed one-shot inputs are available with alloc.
//! The std writer and reader adapt the same bounded framing engine.
//!
//! # Examples
//! ```
//! use mbrotli::framing::{FramedCompressor, FramedInput, FramedItem, FramedResource};
//! let items = [FramedItem::Resource(FramedResource::from(&b"hello"[..]))];
//! let mut encoder = FramedCompressor::new(Default::default())?;
//! let bytes = encoder.compress(FramedInput::from(items.as_slice()))?;
//! assert_eq!(&bytes[..4], &[0x91, 0x0a, 0x42, 0x52]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
mod config;
mod core;
mod error;
mod input;
mod owner;
#[cfg(not(feature = "no_std"))]
mod reader;
mod session;
#[cfg(not(feature = "no_std"))]
mod writer;
pub use crate::framing::{DictionaryId, DictionaryReference, MetadataField, MetadataKind};
pub use config::*;
pub use error::*;
pub use input::*;
pub use owner::*;
#[cfg(not(feature = "no_std"))]
pub use reader::*;
pub use session::*;
#[cfg(not(feature = "no_std"))]
pub use writer::*;

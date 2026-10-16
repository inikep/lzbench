//! Private Zstandard decoder implementation.
//!
//! The layers, from the bitstream up:
//!
//! - [`bits`]: backward bit reader for Huffman and FSE streams, forward
//!   reader for table descriptions;
//! - [`fse`]: FSE table descriptions, decoding tables and the Huffman
//!   weight decoder;
//! - [`huffman`]: Huffman tree descriptions and one- or four-stream literal
//!   decoding;
//! - [`literals`] and [`sequences`]: the two sections of a compressed block,
//!   including sequence execution against the history;
//! - [`block`]: block headers and the per-frame entropy state;
//! - [`dictionary`]: a dictionary's entropy tables and the borrowed view
//!   frames start from;
//! - [`frame`]: frame headers, limits, content size and checksum checks;
//! - [`history`]: the streaming output buffer that doubles as the window;
//! - [`stream`]: the resumable frame-sequencing state machine and the
//!   one-shot driver.
//!
//! Entropy-level modules report [`crate::decompressor::InvalidDataKind`];
//! the frame and stream layers turn those into
//! [`crate::decompressor::DecodeError`] alongside limit, checksum and
//! lifecycle errors.

mod bits;
mod block;
mod copy;
mod dictionary;
mod frame;
mod fse;
mod history;
mod huffman;
mod literals;
mod sequences;
mod stream;

#[cfg(test)]
pub(crate) use dictionary::tests::dictionary as test_dictionary;
pub(crate) use dictionary::{DecoderEntropy, DictView};
pub(crate) use stream::Stream;

use crate::decompressor::InvalidDataKind;

/// Why a block could not be regenerated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BlockError {
    /// The block is malformed.
    Invalid(InvalidDataKind),
    /// The output buffer ends before the block's output does; only a
    /// caller-sized buffer can run out.
    OutputFull,
}

impl From<InvalidDataKind> for BlockError {
    fn from(kind: InvalidDataKind) -> Self {
        Self::Invalid(kind)
    }
}

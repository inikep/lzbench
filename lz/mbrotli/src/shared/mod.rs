//! Private format data and codec primitives shared across the crate.
//!
//! Two layers live here. The lower one is dictated by RFC 7932 and by the
//! pinned reference encoder — the bit writer, the Huffman builders, the
//! match-length scan, the reference logarithms and the format tables — so it is
//! identical whichever quality asks for it. The upper one is the shape of a
//! compressed meta-block: commands, histograms, block splits, context modes,
//! the distance alphabet, the ring buffer, the static dictionary and the
//! writer that turns all of it into bytes.
//!
//! What is *not* here is any decision: which match to take, where to split, how
//! many contexts to model. Those belong to the quality that makes them, which
//! is what lets the fast, greedy and high-quality encoders share this layer
//! without depending on each other. The decoder uses the same wire constants,
//! context tables and built-in dictionary data directly from this module.

#[cfg(feature = "compression")]
pub(crate) mod bit_cost;
#[cfg(feature = "compression")]
pub(crate) mod bits;
#[cfg(feature = "compression")]
pub(crate) mod bitstream;
#[cfg(feature = "compression")]
pub(crate) mod block_split;
#[cfg(feature = "compression")]
pub(crate) mod command;
#[cfg(feature = "compression")]
pub(crate) mod constants;
#[cfg(feature = "decompression")]
pub(crate) mod decode_dictionary;
#[cfg(feature = "experimental")]
#[cfg(feature = "decompression")]
pub(crate) mod decode_serialized;
pub(crate) mod dictionary;
#[cfg(feature = "compression")]
pub(crate) mod distance;
#[cfg(feature = "compression")]
pub(crate) mod fast_log;
pub(crate) mod format;
#[cfg(feature = "compression")]
pub(crate) mod histogram;
#[cfg(feature = "compression")]
pub(crate) mod huffman;
#[cfg(feature = "compression")]
pub(crate) mod match_len;
#[cfg(feature = "compression")]
pub(crate) mod metablock;
#[cfg(feature = "compression")]
pub(crate) mod ringbuffer;
#[cfg(feature = "compression")]
pub(crate) mod score;
#[cfg(feature = "compression")]
pub(crate) mod tables;

//! Qualities 10 and 11: Zopfli backward references and high-quality
//! meta-blocks.
//!
//! Where the greedy encoder takes the best match it can see, this one searches
//! every match at every position and then solves for the cheapest sequence of
//! commands through the whole block. Two things follow from that. The match
//! finder has to report *all* matches, in order, which is what
//! [`h10`] does; and the cost of every command has to be a number, which is
//! what the cost model does — in `f32`, with the reference's exact operation
//! order, because the dynamic program compares those numbers.

pub(crate) mod block_splitter;
pub(crate) mod cluster;
pub(crate) mod cost;
pub(crate) mod encoder;
pub(crate) mod h10;
pub(crate) mod literal_cost;
pub(crate) mod metablock;
pub(crate) mod nodes;
pub(crate) mod params;
pub(crate) mod utf8;
pub(crate) mod zopfli;

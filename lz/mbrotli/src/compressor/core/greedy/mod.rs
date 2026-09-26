//! Qualities 3 to 9: greedy backward references and greedy meta-blocks.
//!
//! The search here is greedy with a one-byte lazy look-ahead, and the
//! meta-block is built in a single pass over the commands. Everything below
//! the decisions — commands, histograms, block splits, the bit writer — lives
//! in [`crate::shared`], which the high-quality encoder uses as well.

pub(crate) mod backward_references;
pub(crate) mod context_model;
pub(crate) mod encoder;
pub(crate) mod hashers;
pub(crate) mod metablock;
pub(crate) mod params;
pub(crate) mod split;

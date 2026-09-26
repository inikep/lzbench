//! The block partition of one symbol stream.
//!
//! Ports `BlockSplit` from `c/enc/block_splitter.h` of the pinned reference
//! (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! A meta-block splits each of its three symbol streams — literals, commands
//! and distances — into consecutive blocks, each tagged with a block type that
//! selects the prefix code it is coded with. How the boundaries are chosen is a
//! quality decision; the partition itself is shared, because the bit writer and
//! the histogram builders read the same shape whichever splitter produced it.

use alloc::vec::Vec;

/// Largest number of block types a meta-block may use.
pub(crate) const MAX_NUMBER_OF_BLOCK_TYPES: usize = 256;

/// The block boundaries and types one symbol stream was split into.
#[derive(Clone, Debug, Default)]
pub(crate) struct BlockSplit {
    /// Number of distinct block types.
    pub(crate) num_types: usize,
    /// Number of blocks.
    pub(crate) num_blocks: usize,
    /// Type of each block.
    pub(crate) types: Vec<u8>,
    /// Length of each block, in symbols.
    pub(crate) lengths: Vec<u32>,
}

impl BlockSplit {
    /// Sizes the type and length arrays for at most `max_num_blocks` blocks.
    pub(crate) fn reserve(&mut self, max_num_blocks: usize) {
        self.types.clear();
        self.lengths.clear();
        self.types.resize(max_num_blocks, 0);
        self.lengths.resize(max_num_blocks, 0);
        self.num_blocks = max_num_blocks;
    }
}

//! What a meta-block is organised into, and the last pass over its histograms.
//!
//! Ports `MetaBlockSplit` from `c/enc/metablock.h` and
//! `BrotliOptimizeHistograms` from `c/enc/metablock.c` of the pinned reference
//! (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! Two builders fill this in: the greedy one for qualities four to nine and the
//! high-quality one for qualities ten and eleven. The bit writer reads it
//! without caring which, so the shape and the histogram optimisation live here.

use alloc::vec::Vec;

use super::block_split::BlockSplit;
use super::constants::{NUM_COMMAND_SYMBOLS, NUM_LITERAL_SYMBOLS};
use super::histogram::{
    HistogramCommand, HistogramDistance, HistogramLiteral, optimize_huffman_counts_for_rle,
};

/// Bits of literal context the format allows (`BROTLI_LITERAL_CONTEXT_BITS`).
pub(crate) const LITERAL_CONTEXT_BITS: usize = 6;

/// Bits of distance context the format allows (`BROTLI_DISTANCE_CONTEXT_BITS`).
pub(crate) const DISTANCE_CONTEXT_BITS: usize = 2;

/// Everything the bit writer needs about how a meta-block is organised.
#[derive(Default)]
pub(crate) struct MetaBlockSplit {
    /// Blocks of the literal stream.
    pub(crate) literal_split: BlockSplit,
    /// Blocks of the command stream.
    pub(crate) command_split: BlockSplit,
    /// Blocks of the distance stream.
    pub(crate) distance_split: BlockSplit,
    /// Context to histogram map for literals, empty when there is none.
    pub(crate) literal_context_map: Vec<u32>,
    /// Context to histogram map for distances, empty when there is none.
    pub(crate) distance_context_map: Vec<u32>,
    /// Literal histograms, one per context of each block type.
    pub(crate) literal_histograms: Vec<HistogramLiteral>,
    /// Command histograms, one per block type.
    pub(crate) command_histograms: Vec<HistogramCommand>,
    /// Distance histograms, one per block type.
    pub(crate) distance_histograms: Vec<HistogramDistance>,
}

impl MetaBlockSplit {
    /// Resets semantic contents without dropping any reusable capacity.
    pub(crate) fn clear(&mut self) {
        for split in [
            &mut self.literal_split,
            &mut self.command_split,
            &mut self.distance_split,
        ] {
            split.num_types = 0;
            split.num_blocks = 0;
            split.types.clear();
            split.lengths.clear();
        }
        self.literal_context_map.clear();
        self.distance_context_map.clear();
        self.literal_histograms.clear();
        self.command_histograms.clear();
        self.distance_histograms.clear();
    }

    /// Counts the splits, context maps and histogram capacities.
    pub(crate) fn retained_bytes(&self) -> usize {
        let splits = [
            &self.literal_split,
            &self.command_split,
            &self.distance_split,
        ];
        splits
            .iter()
            .map(|split| split.types.capacity() + split.lengths.capacity() * size_of::<u32>())
            .sum::<usize>()
            + (self.literal_context_map.capacity() + self.distance_context_map.capacity())
                * size_of::<u32>()
            + self.literal_histograms.capacity() * size_of::<HistogramLiteral>()
            + self.command_histograms.capacity() * size_of::<HistogramCommand>()
            + self.distance_histograms.capacity() * size_of::<HistogramDistance>()
    }
}

/// Rounds every histogram so its prefix code codes well by run length.
///
/// Mirrors `BrotliOptimizeHistograms`, which qualities four and up apply before
/// the codes are built.
pub(crate) fn optimize_histograms(num_distance_codes: usize, mb: &mut MetaBlockSplit) {
    let mut good_for_rle = [0u8; NUM_COMMAND_SYMBOLS];
    for histogram in &mut mb.literal_histograms {
        optimize_huffman_counts_for_rle(
            NUM_LITERAL_SYMBOLS,
            &mut histogram.data,
            &mut good_for_rle,
        );
    }
    for histogram in &mut mb.command_histograms {
        optimize_huffman_counts_for_rle(
            NUM_COMMAND_SYMBOLS,
            &mut histogram.data,
            &mut good_for_rle,
        );
    }
    for histogram in &mut mb.distance_histograms {
        optimize_huffman_counts_for_rle(num_distance_codes, &mut histogram.data, &mut good_for_rle);
    }
}

//! Splitting a meta-block's symbols into blocks and gathering their histograms.
//!
//! Ports `BrotliBuildMetaBlockGreedy`, `MapStaticContexts` and
//! `BrotliOptimizeHistograms` from `c/enc/metablock.c` of the pinned reference
//! (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! The higher qualities cluster histograms after splitting; the greedy path
//! deliberately does not, which is what keeps it fast. Everything a meta-block
//! needs is produced in one pass over the commands.

use super::context_model::{ContextModel, context};
use super::split::{BlockSplitter, ContextBlockSplitter};
use crate::shared::command::Command;
use crate::shared::constants::{NUM_COMMAND_SYMBOLS, NUM_LITERAL_SYMBOLS};
use crate::shared::distance::NUM_HISTOGRAM_DISTANCE_SYMBOLS;
use crate::shared::metablock::{LITERAL_CONTEXT_BITS, MetaBlockSplit};

/// Smallest literal block, and the threshold a new literal type has to beat.
const LITERAL_MIN_BLOCK: usize = 512;
/// Entropy a literal block has to save to earn its own type.
const LITERAL_SPLIT_THRESHOLD: f64 = 400.0;
/// Smallest command block.
const COMMAND_MIN_BLOCK: usize = 1024;
/// Entropy a command block has to save to earn its own type.
const COMMAND_SPLIT_THRESHOLD: f64 = 500.0;
/// Smallest distance block.
const DISTANCE_MIN_BLOCK: usize = 512;
/// Entropy a distance block has to save to earn its own type.
const DISTANCE_SPLIT_THRESHOLD: f64 = 100.0;

/// Distance symbols the splitter estimates entropy over.
///
/// The histogram itself is wider, but only the first sixty-four symbols can
/// occur while the distance parameters are still the defaults, and the
/// reference measures exactly those.
const DISTANCE_SPLIT_ALPHABET: usize = 64;

/// Expands a static context map to one entry per block type and context.
///
/// Mirrors `MapStaticContexts`.
fn map_static_contexts(num_contexts: usize, static_map: &[u32; 64], mb: &mut MetaBlockSplit) {
    let num_types = mb.literal_split.num_types;
    mb.literal_context_map.clear();
    // One entry per raw context of every block type.
    mb.literal_context_map
        .resize(num_types << LITERAL_CONTEXT_BITS, 0);
    for (block_type, slots) in mb
        .literal_context_map
        .as_chunks_mut::<{ 1usize << LITERAL_CONTEXT_BITS }>()
        .0
        .iter_mut()
        .enumerate()
    {
        let offset = (block_type * num_contexts) as u32;
        for (slot, &context) in slots.iter_mut().zip(static_map.iter()) {
            *slot = offset + context;
        }
    }
}

/// Splits a meta-block into blocks and gathers their histograms.
///
/// Mirrors `BrotliBuildMetaBlockGreedy`. `pos` is the wrapped position of the
/// first literal, and `prev_byte`/`prev_byte2` are the two bytes before it, so
/// the first literal's context is the same one the decoder will compute.
#[cfg(test)]
pub(crate) fn build_meta_block_greedy(
    ringbuffer: &[u8],
    pos: usize,
    mask: usize,
    prev_byte: u8,
    prev_byte2: u8,
    model: ContextModel,
    commands: &[Command],
) -> MetaBlockSplit {
    let mut mb = MetaBlockSplit::default();
    build_meta_block_greedy_into(
        ringbuffer, pos, mask, prev_byte, prev_byte2, model, commands, &mut mb,
    );
    mb
}

/// Builds into retained split/histogram storage.
#[expect(
    clippy::too_many_arguments,
    reason = "the reference's window, contexts and commands plus caller-owned output"
)]
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
pub(crate) fn build_meta_block_greedy_into(
    ringbuffer: &[u8],
    pos: usize,
    mask: usize,
    prev_byte: u8,
    prev_byte2: u8,
    model: ContextModel,
    commands: &[Command],
    mb: &mut MetaBlockSplit,
) {
    mb.clear();
    let num_literals: usize = commands
        .iter()
        .map(|command| command.insert_len as usize)
        .sum();

    let num_contexts = model.num_contexts;
    let mut plain_literals = if num_contexts == 1 {
        Some(BlockSplitter::<NUM_LITERAL_SYMBOLS>::with_storage(
            NUM_LITERAL_SYMBOLS,
            LITERAL_MIN_BLOCK,
            LITERAL_SPLIT_THRESHOLD,
            num_literals,
            core::mem::take(&mut mb.literal_split),
            core::mem::take(&mut mb.literal_histograms),
        ))
    } else {
        None
    };
    let mut context_literals = if num_contexts == 1 {
        None
    } else {
        Some(ContextBlockSplitter::with_storage(
            NUM_LITERAL_SYMBOLS,
            num_contexts,
            LITERAL_MIN_BLOCK,
            LITERAL_SPLIT_THRESHOLD,
            num_literals,
            core::mem::take(&mut mb.literal_split),
            core::mem::take(&mut mb.literal_histograms),
        ))
    };
    let mut cmd_blocks = BlockSplitter::<NUM_COMMAND_SYMBOLS>::with_storage(
        NUM_COMMAND_SYMBOLS,
        COMMAND_MIN_BLOCK,
        COMMAND_SPLIT_THRESHOLD,
        commands.len(),
        core::mem::take(&mut mb.command_split),
        core::mem::take(&mut mb.command_histograms),
    );
    let mut dist_blocks = BlockSplitter::<NUM_HISTOGRAM_DISTANCE_SYMBOLS>::with_storage(
        DISTANCE_SPLIT_ALPHABET,
        DISTANCE_MIN_BLOCK,
        DISTANCE_SPLIT_THRESHOLD,
        commands.len(),
        core::mem::take(&mut mb.distance_split),
        core::mem::take(&mut mb.distance_histograms),
    );

    let static_map = model.map;
    let mut pos = pos;
    let mut prev_byte = prev_byte;
    let mut prev_byte2 = prev_byte2;
    for command in commands {
        cmd_blocks.add_symbol(usize::from(command.cmd_prefix));
        for _ in 0..command.insert_len {
            let literal = ringbuffer.get(pos & mask).copied().unwrap_or(0);
            match (&mut plain_literals, &mut context_literals, static_map) {
                (Some(splitter), _, _) => splitter.add_symbol(usize::from(literal)),
                (_, Some(splitter), Some(map)) => {
                    let raw = context(prev_byte, prev_byte2);
                    splitter.add_symbol(usize::from(literal), map[raw] as usize);
                }
                _ => {}
            }
            prev_byte2 = prev_byte;
            prev_byte = literal;
            pos += 1;
        }
        pos += command.copy_len() as usize;
        if command.copy_len() != 0 {
            prev_byte2 = ringbuffer.get((pos - 2) & mask).copied().unwrap_or(0);
            prev_byte = ringbuffer.get((pos - 1) & mask).copied().unwrap_or(0);
            if command.cmd_prefix >= 128 {
                dist_blocks.add_symbol(usize::from(command.distance_code()));
            }
        }
    }

    if let Some(mut splitter) = plain_literals {
        splitter.finish_block(true);
        mb.literal_split = splitter.split;
        mb.literal_histograms = splitter.histograms;
    }
    if let Some(mut splitter) = context_literals {
        splitter.finish_block(true);
        mb.literal_split = splitter.split;
        mb.literal_histograms = splitter.histograms;
    }
    cmd_blocks.finish_block(true);
    mb.command_split = cmd_blocks.split;
    mb.command_histograms = cmd_blocks.histograms;
    dist_blocks.finish_block(true);
    mb.distance_split = dist_blocks.split;
    mb.distance_histograms = dist_blocks.histograms;

    if let Some(map) = static_map
        && num_contexts > 1
    {
        map_static_contexts(num_contexts, map, mb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::distance::DistanceParams;
    use crate::shared::format::STATIC_CONTEXT_MAP_SIMPLE_UTF8;
    use crate::shared::metablock::optimize_histograms;
    use alloc::vec::Vec;

    /// Builds commands that just insert `data` with no copies.
    fn literal_commands(data: &[u8]) -> Vec<Command> {
        let mut commands = Vec::new();
        let mut remaining = data.len();
        while remaining > 0 {
            let take = remaining.min(1000);
            commands.push(Command::insert_only(take));
            remaining -= take;
        }
        commands
    }

    #[test]
    fn a_literal_only_block_gathers_every_literal() {
        let data: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        let commands = literal_commands(&data);
        let mb =
            build_meta_block_greedy(&data, 0, usize::MAX, 0, 0, ContextModel::SINGLE, &commands);
        let counted: usize = mb
            .literal_histograms
            .iter()
            .map(|histogram| histogram.total_count)
            .sum();
        assert_eq!(counted, data.len());
        assert!(mb.literal_context_map.is_empty());
        assert_eq!(mb.command_histograms.len(), mb.command_split.num_types);
    }

    #[test]
    fn a_context_model_produces_a_context_map() {
        let data: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        let commands = literal_commands(&data);
        let model = ContextModel {
            num_contexts: 2,
            map: Some(&STATIC_CONTEXT_MAP_SIMPLE_UTF8),
        };
        let mb = build_meta_block_greedy(&data, 0, usize::MAX, 0, 0, model, &commands);
        assert_eq!(
            mb.literal_context_map.len(),
            mb.literal_split.num_types << LITERAL_CONTEXT_BITS
        );
        assert_eq!(mb.literal_histograms.len(), mb.literal_split.num_types * 2);
        assert!(
            mb.literal_context_map
                .iter()
                .all(|&index| (index as usize) < mb.literal_histograms.len())
        );
    }

    #[test]
    fn distances_are_gathered_only_from_copies() {
        let dist = DistanceParams::default();
        let data = vec![b'x'; 4000];
        let mut commands = Vec::new();
        for _ in 0..100 {
            commands.push(Command::new(&dist, 10, 20, 0, 100));
        }
        commands.push(Command::insert_only(5));
        let mb =
            build_meta_block_greedy(&data, 0, usize::MAX, 0, 0, ContextModel::SINGLE, &commands);
        let distances: usize = mb
            .distance_histograms
            .iter()
            .map(|histogram| histogram.total_count)
            .sum();
        assert_eq!(distances, 100);
    }

    #[test]
    fn an_empty_command_list_still_produces_one_block_each() {
        let mb = build_meta_block_greedy(&[], 0, usize::MAX, 0, 0, ContextModel::SINGLE, &[]);
        assert_eq!(mb.literal_split.num_types, 1);
        assert_eq!(mb.command_split.num_types, 1);
        assert_eq!(mb.distance_split.num_types, 1);
    }

    #[test]
    fn optimising_histograms_never_silences_a_used_symbol() {
        let mut rng = 0x5EED_0001u64;
        let data: Vec<u8> = (0..20_000u32)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                ((rng >> 24) % 200) as u8
            })
            .collect();
        let commands = literal_commands(&data);
        let mut mb =
            build_meta_block_greedy(&data, 0, usize::MAX, 0, 0, ContextModel::SINGLE, &commands);
        let before: Vec<Vec<u32>> = mb
            .literal_histograms
            .iter()
            .map(|histogram| histogram.data.to_vec())
            .collect();
        optimize_histograms(64, &mut mb);
        for (histogram, original) in mb.literal_histograms.iter().zip(&before) {
            for (symbol, &count) in original.iter().enumerate() {
                assert!(
                    count == 0 || histogram.data[symbol] != 0,
                    "symbol {symbol} lost its code"
                );
            }
        }
    }

    #[test]
    fn optimising_histograms_respects_the_distance_alphabet() {
        let dist = DistanceParams::default();
        let data = vec![b'x'; 4000];
        let commands: Vec<Command> = (0..200)
            .map(|index| Command::new(&dist, 1, 20, 0, 20 + index))
            .collect();
        let mut mb =
            build_meta_block_greedy(&data, 0, usize::MAX, 0, 0, ContextModel::SINGLE, &commands);
        optimize_histograms(dist.alphabet_size_limit as usize, &mut mb);
        for histogram in &mb.distance_histograms {
            assert!(
                histogram.data[dist.alphabet_size_limit as usize..]
                    .iter()
                    .all(|&count| count == 0)
            );
        }
    }
}

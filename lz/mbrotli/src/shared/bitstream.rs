//! Writing a meta-block into the bit stream.
//!
//! Ports `c/enc/brotli_bit_stream.c` and `c/enc/block_encoder_inc.h` of the
//! pinned reference (`google/brotli` v1.2.0, commit `028fb5a`): the compressed
//! meta-block header, the block-switch codes, the context maps, the prefix
//! codes and the symbols themselves.
//!
//! Everything here is bit layout defined by RFC 7932, plus the reference's
//! choices where the format leaves room: which prefix code form is used for a
//! small alphabet, how the context map is run-length coded, and how block
//! switches are encoded.

use alloc::vec::Vec;

use super::bits::BitWriter;
use super::block_split::{BlockSplit, MAX_NUMBER_OF_BLOCK_TYPES};
use super::command::Command;
use super::constants::{NUM_COMMAND_SYMBOLS, NUM_LITERAL_SYMBOLS};
use super::distance::{DistanceParams, MAX_SIMPLE_DISTANCE_ALPHABET_SIZE};
use super::fast_log::log2_floor_non_zero;
use super::format::{ContextMode, NUM_BLOCK_LEN_SYMBOLS, PREFIX_CODE_RANGES};
use super::huffman::{
    HuffmanNode, build_and_store_huffman_tree, build_and_store_huffman_tree_fast,
};
use super::metablock::{DISTANCE_CONTEXT_BITS, LITERAL_CONTEXT_BITS, MetaBlockSplit};
use super::tables::{
    STATIC_COMMAND_CODE_BITS, STATIC_COMMAND_CODE_DEPTH, STATIC_DISTANCE_CODE_BITS,
    STATIC_DISTANCE_CODE_DEPTH,
};

/// Symbols a context map may use (`BROTLI_MAX_CONTEXT_MAP_SYMBOLS`).
const MAX_CONTEXT_MAP_SYMBOLS: usize = MAX_NUMBER_OF_BLOCK_TYPES + 16;

/// Block-type symbols a meta-block may use (`BROTLI_MAX_BLOCK_TYPE_SYMBOLS`).
const MAX_BLOCK_TYPE_SYMBOLS: usize = MAX_NUMBER_OF_BLOCK_TYPES + 2;

/// Bits a run-length symbol occupies in the packed context map.
const CONTEXT_MAP_SYMBOL_BITS: u32 = 9;

/// Returns the prefix code of a block length (`BlockLengthPrefixCode`).
fn block_length_prefix_code(len: u32) -> usize {
    let mut code = if len >= 177 {
        if len >= 753 { 20 } else { 14 }
    } else if len >= 41 {
        7
    } else {
        0
    };
    while code < NUM_BLOCK_LEN_SYMBOLS - 1 && len >= PREFIX_CODE_RANGES[code + 1].0 {
        code += 1;
    }
    code
}

/// Splits a block length into its prefix code and extra bits.
fn block_length_code(len: u32) -> (usize, u32, u32) {
    let code = block_length_prefix_code(len);
    let (offset, nbits) = PREFIX_CODE_RANGES[code];
    (code, nbits, len - offset)
}

/// Writes a number between zero and 255 (`StoreVarLenUint8`).
fn store_var_len_uint8(n: usize, w: &mut BitWriter) {
    if n == 0 {
        w.write(1, 0);
        return;
    }
    let nbits = log2_floor_non_zero(n);
    w.write(1, 1);
    w.write(3, u64::from(nbits));
    w.write(nbits, (n - (1usize << nbits)) as u64);
}

/// Splits a meta-block length into its nibble count and value (`BrotliEncodeMlen`).
fn encode_mlen(length: usize) -> (u64, u32, u64) {
    debug_assert!((1..=1 << 24).contains(&length));
    let lg = if length == 1 {
        1
    } else {
        log2_floor_non_zero(length - 1) + 1
    };
    let mnibbles = (if lg < 16 { 16 } else { lg + 3 }) / 4;
    (u64::from(mnibbles - 4), mnibbles * 4, (length - 1) as u64)
}

/// Writes the header of a compressed meta-block.
///
/// Mirrors `StoreCompressedMetaBlockHeader`.
pub(crate) fn store_compressed_meta_block_header(
    is_final_block: bool,
    length: usize,
    w: &mut BitWriter,
) {
    w.write(1, u64::from(is_final_block));
    if is_final_block {
        w.write(1, 0);
    }
    let (nibblesbits, nlenbits, lenbits) = encode_mlen(length);
    w.write(2, nibblesbits);
    w.write(nlenbits, lenbits);
    if !is_final_block {
        w.write(1, 0);
    }
}

/// Writes the header of an uncompressed meta-block.
///
/// Mirrors `BrotliStoreUncompressedMetaBlockHeader`. An uncompressed
/// meta-block can never be the last one, so the caller appends an empty final
/// block after it.
fn store_uncompressed_meta_block_header(length: usize, w: &mut BitWriter) {
    w.write(1, 0);
    let (nibblesbits, nlenbits, lenbits) = encode_mlen(length);
    w.write(2, nibblesbits);
    w.write(nlenbits, lenbits);
    w.write(1, 1);
}

/// Tracks the two most recent block types, to code the next one relatively.
///
/// Mirrors `BlockTypeCodeCalculator`.
#[derive(Copy, Clone, Debug)]
struct BlockTypeCodeCalculator {
    last_type: usize,
    second_last_type: usize,
}

impl BlockTypeCodeCalculator {
    /// Returns a calculator in the state the format assumes at a block start.
    const fn new() -> Self {
        Self {
            last_type: 1,
            second_last_type: 0,
        }
    }

    /// Returns the code for switching to `block_type`.
    fn next(&mut self, block_type: u8) -> usize {
        let block_type = usize::from(block_type);
        let type_code = if block_type == self.last_type + 1 {
            1
        } else if block_type == self.second_last_type {
            0
        } else {
            block_type + 2
        };
        self.second_last_type = self.last_type;
        self.last_type = block_type;
        type_code
    }
}

/// Prefix codes for block types and block lengths (`BlockSplitCode`).
struct BlockSplitCode {
    calculator: BlockTypeCodeCalculator,
    type_depths: [u8; MAX_BLOCK_TYPE_SYMBOLS],
    type_bits: [u16; MAX_BLOCK_TYPE_SYMBOLS],
    length_depths: [u8; NUM_BLOCK_LEN_SYMBOLS],
    length_bits: [u16; NUM_BLOCK_LEN_SYMBOLS],
}

impl BlockSplitCode {
    /// Returns an unpopulated code.
    fn new() -> Self {
        Self {
            calculator: BlockTypeCodeCalculator::new(),
            type_depths: [0; MAX_BLOCK_TYPE_SYMBOLS],
            type_bits: [0; MAX_BLOCK_TYPE_SYMBOLS],
            length_depths: [0; NUM_BLOCK_LEN_SYMBOLS],
            length_bits: [0; NUM_BLOCK_LEN_SYMBOLS],
        }
    }

    /// Writes one block switch (`StoreBlockSwitch`).
    fn store_block_switch(
        &mut self,
        block_len: u32,
        block_type: u8,
        is_first_block: bool,
        w: &mut BitWriter,
    ) {
        let typecode = self.calculator.next(block_type);
        if !is_first_block {
            w.write(
                u32::from(self.type_depths[typecode]),
                u64::from(self.type_bits[typecode]),
            );
        }
        let (lencode, nextra, extra) = block_length_code(block_len);
        w.write(
            u32::from(self.length_depths[lencode]),
            u64::from(self.length_bits[lencode]),
        );
        w.write(nextra, u64::from(extra));
    }
}

/// Builds and writes the block-type and block-length codes of a split.
///
/// Mirrors `BuildAndStoreBlockSplitCode`.
fn build_and_store_block_split_code(
    split: &BlockSplit,
    tree: &mut Vec<HuffmanNode>,
    code: &mut BlockSplitCode,
    w: &mut BitWriter,
) {
    let num_types = split.num_types;
    let mut type_histo = [0u32; MAX_BLOCK_TYPE_SYMBOLS];
    let mut length_histo = [0u32; NUM_BLOCK_LEN_SYMBOLS];
    let mut calculator = BlockTypeCodeCalculator::new();
    for index in 0..split.num_blocks {
        let type_code = calculator.next(split.types[index]);
        if index != 0 {
            type_histo[type_code] += 1;
        }
        length_histo[block_length_prefix_code(split.lengths[index])] += 1;
    }
    store_var_len_uint8(num_types - 1, w);
    if num_types > 1 {
        build_and_store_huffman_tree(
            &type_histo,
            num_types + 2,
            num_types + 2,
            tree,
            &mut code.type_depths,
            &mut code.type_bits,
            w,
        );
        build_and_store_huffman_tree(
            &length_histo,
            NUM_BLOCK_LEN_SYMBOLS,
            NUM_BLOCK_LEN_SYMBOLS,
            tree,
            &mut code.length_depths,
            &mut code.length_bits,
            w,
        );
        code.store_block_switch(split.lengths[0], split.types[0], true, w);
    }
}

/// Returns the index of `value` in `v`, or its length when absent.
fn index_of(v: &[u8], value: u8) -> usize {
    v.iter()
        .position(|&entry| entry == value)
        .unwrap_or(v.len())
}

/// Moves `v[index]` to the front, shifting everything before it up.
fn move_to_front(v: &mut [u8], index: usize) {
    if index < v.len() {
        v[..=index].rotate_right(1);
    }
}

/// Recodes a context map so that recently used indices become small.
///
/// Mirrors `MoveToFrontTransform`.
fn move_to_front_transform(input: &[u32], output: &mut Vec<u32>) {
    output.clear();
    if input.is_empty() {
        return;
    }
    let max_value = input.iter().copied().max().unwrap_or(0);
    debug_assert!(max_value < 256);
    let mut values = core::array::from_fn::<_, 256, _>(|index| index as u8);
    let mtf = &mut values[..=max_value as usize];
    for &value in input {
        let index = index_of(mtf, value as u8);
        output.push(index as u32);
        move_to_front(mtf, index);
    }
}

/// Replaces runs of zeros with run-length prefix codes.
///
/// Mirrors `RunLengthCodeZeros`. The low nine bits of an output word are the
/// symbol and the rest are its extra bits. Returns the number of symbols
/// produced and the run-length prefix actually used.
fn run_length_code_zeros(v: &mut Vec<u32>, max_run_length_prefix: u32) -> (usize, u32) {
    let in_size = v.len();
    let mut max_reps = 0u32;
    let mut index = 0usize;
    while index < in_size {
        while index < in_size && v[index] != 0 {
            index += 1;
        }
        let mut reps = 0u32;
        while index < in_size && v[index] == 0 {
            reps += 1;
            index += 1;
        }
        max_reps = max_reps.max(reps);
    }
    let max_prefix = if max_reps > 0 {
        log2_floor_non_zero(max_reps as usize)
    } else {
        0
    };
    let max_prefix = max_prefix.min(max_run_length_prefix);

    let mut out_size = 0usize;
    let mut index = 0usize;
    while index < in_size {
        debug_assert!(out_size <= index);
        if v[index] != 0 {
            v[out_size] = v[index] + max_prefix;
            index += 1;
            out_size += 1;
            continue;
        }
        let mut reps = 1u32;
        let mut k = index + 1;
        while k < in_size && v[k] == 0 {
            reps += 1;
            k += 1;
        }
        index += reps as usize;
        while reps != 0 {
            if reps < (2u32 << max_prefix) {
                let run_length_prefix = log2_floor_non_zero(reps as usize);
                let extra_bits = reps - (1u32 << run_length_prefix);
                v[out_size] = run_length_prefix + (extra_bits << CONTEXT_MAP_SYMBOL_BITS);
                out_size += 1;
                break;
            }
            let extra_bits = (1u32 << max_prefix) - 1;
            v[out_size] = max_prefix + (extra_bits << CONTEXT_MAP_SYMBOL_BITS);
            reps -= (2u32 << max_prefix) - 1;
            out_size += 1;
        }
    }
    v.truncate(out_size);
    (out_size, max_prefix)
}

/// Scratch buffers the context map encoder reuses.
struct ContextMapArena {
    histogram: [u32; MAX_CONTEXT_MAP_SYMBOLS],
    depths: [u8; MAX_CONTEXT_MAP_SYMBOLS],
    bits: [u16; MAX_CONTEXT_MAP_SYMBOLS],
    rle_symbols: Vec<u32>,
}

impl ContextMapArena {
    /// Returns empty scratch space.
    fn new() -> Self {
        Self {
            histogram: [0; MAX_CONTEXT_MAP_SYMBOLS],
            depths: [0; MAX_CONTEXT_MAP_SYMBOLS],
            bits: [0; MAX_CONTEXT_MAP_SYMBOLS],
            rle_symbols: Vec::new(),
        }
    }
}

/// Writes a context map (`EncodeContextMap`).
fn encode_context_map(
    arena: &mut ContextMapArena,
    context_map: &[u32],
    num_clusters: usize,
    tree: &mut Vec<HuffmanNode>,
    w: &mut BitWriter,
) {
    store_var_len_uint8(num_clusters - 1, w);
    if num_clusters == 1 {
        return;
    }

    move_to_front_transform(context_map, &mut arena.rle_symbols);
    let (num_rle_symbols, max_run_length_prefix) = run_length_code_zeros(&mut arena.rle_symbols, 6);
    arena.histogram.fill(0);
    let symbol_mask = (1u32 << CONTEXT_MAP_SYMBOL_BITS) - 1;
    for &symbol in &arena.rle_symbols {
        arena.histogram[(symbol & symbol_mask) as usize] += 1;
    }

    let use_rle = max_run_length_prefix > 0;
    w.write(1, u64::from(use_rle));
    if use_rle {
        w.write(4, u64::from(max_run_length_prefix - 1));
    }
    let alphabet = num_clusters + max_run_length_prefix as usize;
    build_and_store_huffman_tree(
        &arena.histogram,
        alphabet,
        alphabet,
        tree,
        &mut arena.depths,
        &mut arena.bits,
        w,
    );
    for index in 0..num_rle_symbols {
        let symbol = arena.rle_symbols[index] & symbol_mask;
        let extra = arena.rle_symbols[index] >> CONTEXT_MAP_SYMBOL_BITS;
        w.write(
            u32::from(arena.depths[symbol as usize]),
            u64::from(arena.bits[symbol as usize]),
        );
        if symbol > 0 && symbol <= max_run_length_prefix {
            w.write(symbol, u64::from(extra));
        }
    }
    // Inverse move-to-front is always used.
    w.write(1, 1);
}

/// Writes the context map that gives every block type its own histogram.
///
/// Mirrors `StoreTrivialContextMap`.
fn store_trivial_context_map(
    arena: &mut ContextMapArena,
    num_types: usize,
    context_bits: usize,
    tree: &mut Vec<HuffmanNode>,
    w: &mut BitWriter,
) {
    store_var_len_uint8(num_types - 1, w);
    if num_types <= 1 {
        return;
    }
    let repeat_code = context_bits - 1;
    let repeat_bits = (1usize << repeat_code) - 1;
    let alphabet_size = num_types + repeat_code;
    arena.histogram[..alphabet_size].fill(0);
    w.write(1, 1);
    w.write(4, (repeat_code - 1) as u64);
    arena.histogram[repeat_code] = num_types as u32;
    arena.histogram[0] = 1;
    for slot in arena
        .histogram
        .iter_mut()
        .take(alphabet_size)
        .skip(context_bits)
    {
        *slot = 1;
    }
    build_and_store_huffman_tree(
        &arena.histogram,
        alphabet_size,
        alphabet_size,
        tree,
        &mut arena.depths,
        &mut arena.bits,
        w,
    );
    for index in 0..num_types {
        let code = if index == 0 {
            0
        } else {
            index + context_bits - 1
        };
        w.write(u32::from(arena.depths[code]), u64::from(arena.bits[code]));
        w.write(
            u32::from(arena.depths[repeat_code]),
            u64::from(arena.bits[repeat_code]),
        );
        w.write(repeat_code as u32, repeat_bits as u64);
    }
    w.write(1, 1);
}

/// Writes the symbols of one stream, switching prefix codes at block borders.
///
/// Mirrors `BlockEncoder`.
struct BlockEncoder<'a> {
    histogram_length: usize,
    block_types: &'a [u8],
    block_lengths: &'a [u32],
    split_code: BlockSplitCode,
    block_ix: usize,
    block_len: usize,
    entropy_ix: usize,
    depths: &'a mut Vec<u8>,
    bits: &'a mut Vec<u16>,
}

impl<'a> BlockEncoder<'a> {
    /// Creates an encoder for the stream `split` describes.
    fn new(
        histogram_length: usize,
        split: &'a BlockSplit,
        depths: &'a mut Vec<u8>,
        bits: &'a mut Vec<u16>,
    ) -> Self {
        Self {
            histogram_length,
            block_types: &split.types,
            block_lengths: &split.lengths,
            split_code: BlockSplitCode::new(),
            block_ix: 0,
            block_len: if split.num_blocks == 0 {
                0
            } else {
                split.lengths[0] as usize
            },
            entropy_ix: 0,
            depths,
            bits,
        }
    }

    /// Builds and writes the code that describes this stream's block switches.
    fn build_and_store_block_switch_codes(
        &mut self,
        split: &BlockSplit,
        tree: &mut Vec<HuffmanNode>,
        w: &mut BitWriter,
    ) {
        build_and_store_block_split_code(split, tree, &mut self.split_code, w);
    }

    /// Builds and writes one prefix code per histogram.
    ///
    /// Mirrors `BuildAndStoreEntropyCodes`.
    fn build_and_store_entropy_codes<const N: usize>(
        &mut self,
        histograms: &[super::histogram::Histogram<N>],
        alphabet_size: usize,
        tree: &mut Vec<HuffmanNode>,
        w: &mut BitWriter,
    ) {
        let table_size = histograms.len() * self.histogram_length;
        self.depths.clear();
        self.depths.resize(table_size, 0);
        self.bits.clear();
        self.bits.resize(table_size, 0);
        for (index, histogram) in histograms.iter().enumerate() {
            let ix = index * self.histogram_length;
            build_and_store_huffman_tree(
                &histogram.data,
                self.histogram_length,
                alphabet_size,
                tree,
                &mut self.depths[ix..],
                &mut self.bits[ix..],
                w,
            );
        }
    }

    /// Opens the next block when the current one has run out.
    fn advance_block<F: FnOnce(u8) -> usize>(&mut self, entropy_ix: F, w: &mut BitWriter) {
        if self.block_len != 0 {
            return;
        }
        self.block_ix += 1;
        let block_len = self.block_lengths[self.block_ix];
        let block_type = self.block_types[self.block_ix];
        self.block_len = block_len as usize;
        self.entropy_ix = entropy_ix(block_type);
        self.split_code
            .store_block_switch(block_len, block_type, false, w);
    }

    /// Writes one symbol with the current block's code (`StoreSymbol`).
    fn store_symbol(&mut self, symbol: usize, w: &mut BitWriter) {
        let histogram_length = self.histogram_length;
        self.advance_block(|block_type| usize::from(block_type) * histogram_length, w);
        self.block_len -= 1;
        let ix = self.entropy_ix + symbol;
        w.write(u32::from(self.depths[ix]), u64::from(self.bits[ix]));
    }

    /// Writes one symbol in `context` (`StoreSymbolWithContext`).
    fn store_symbol_with_context(
        &mut self,
        symbol: usize,
        context: usize,
        context_map: &[u32],
        context_bits: usize,
        w: &mut BitWriter,
    ) {
        self.advance_block(|block_type| usize::from(block_type) << context_bits, w);
        self.block_len -= 1;
        let histo_ix = context_map[self.entropy_ix + context] as usize;
        let ix = histo_ix * self.histogram_length + symbol;
        w.write(u32::from(self.depths[ix]), u64::from(self.bits[ix]));
    }
}

/// Scratch state the meta-block writer reuses between blocks.
/// Default construction defers allocation until a code needs its buffers.
#[derive(Default)]
pub(crate) struct MetaBlockWriter {
    tree: Vec<HuffmanNode>,
    arena: Option<ContextMapArena>,
    literal_depth: Vec<u8>,
    literal_bits: Vec<u16>,
    command_depth: Vec<u8>,
    command_bits: Vec<u16>,
    distance_depth: Vec<u8>,
    distance_bits: Vec<u16>,
}

impl MetaBlockWriter {
    /// Ensures room for one literal code. The node pool sizes itself per
    /// build, so only the depth and bit tables are prepared here.
    fn prepare_literals(&mut self) {
        self.literal_depth.resize(NUM_LITERAL_SYMBOLS, 0);
        self.literal_bits.resize(NUM_LITERAL_SYMBOLS, 0);
    }

    /// Ensures room for the three codes used without block splitting.
    fn prepare_simple_codes(&mut self) {
        self.prepare_literals();
        self.command_depth.resize(NUM_COMMAND_SYMBOLS, 0);
        self.command_bits.resize(NUM_COMMAND_SYMBOLS, 0);
        self.distance_depth
            .resize(MAX_SIMPLE_DISTANCE_ALPHABET_SIZE, 0);
        self.distance_bits
            .resize(MAX_SIMPLE_DISTANCE_ALPHABET_SIZE, 0);
    }

    /// Counts the retained entropy and context-map scratch buffers.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.tree.capacity() * size_of::<HuffmanNode>()
            + self
                .arena
                .as_ref()
                .map_or(0, |arena| arena.rle_symbols.capacity() * size_of::<u32>())
            + self.literal_depth.capacity()
            + self.command_depth.capacity()
            + self.distance_depth.capacity()
            + (self.literal_bits.capacity()
                + self.command_bits.capacity()
                + self.distance_bits.capacity())
                * size_of::<u16>()
    }

    /// Writes a meta-block with block splitting and context modelling.
    ///
    /// Mirrors `BrotliStoreMetaBlock`.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors BrotliStoreMetaBlock, whose parameters are all needed"
    )]
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn store_meta_block(
        &mut self,
        input: &[u8],
        start_pos: usize,
        length: usize,
        mask: usize,
        prev_byte: u8,
        prev_byte2: u8,
        is_last: bool,
        context_mode: ContextMode,
        dist: &DistanceParams,
        commands: &[Command],
        mb: &MetaBlockSplit,
        w: &mut BitWriter,
    ) {
        let num_distance_symbols = dist.alphabet_size_max as usize;
        let num_effective_distance_symbols = dist.alphabet_size_limit as usize;

        store_compressed_meta_block_header(is_last, length, w);

        let mut literal_enc = BlockEncoder::new(
            NUM_LITERAL_SYMBOLS,
            &mb.literal_split,
            &mut self.literal_depth,
            &mut self.literal_bits,
        );
        let mut command_enc = BlockEncoder::new(
            NUM_COMMAND_SYMBOLS,
            &mb.command_split,
            &mut self.command_depth,
            &mut self.command_bits,
        );
        let mut distance_enc = BlockEncoder::new(
            num_effective_distance_symbols,
            &mb.distance_split,
            &mut self.distance_depth,
            &mut self.distance_bits,
        );

        literal_enc.build_and_store_block_switch_codes(&mb.literal_split, &mut self.tree, w);
        command_enc.build_and_store_block_switch_codes(&mb.command_split, &mut self.tree, w);
        distance_enc.build_and_store_block_switch_codes(&mb.distance_split, &mut self.tree, w);

        w.write(2, u64::from(dist.postfix_bits));
        w.write(4, u64::from(dist.num_direct >> dist.postfix_bits));
        for _ in 0..mb.literal_split.num_types {
            w.write(2, context_mode.code());
        }

        if mb.literal_context_map.is_empty() {
            store_trivial_context_map(
                self.arena.get_or_insert_with(ContextMapArena::new),
                mb.literal_histograms.len(),
                LITERAL_CONTEXT_BITS,
                &mut self.tree,
                w,
            );
        } else {
            encode_context_map(
                self.arena.get_or_insert_with(ContextMapArena::new),
                &mb.literal_context_map,
                mb.literal_histograms.len(),
                &mut self.tree,
                w,
            );
        }
        if mb.distance_context_map.is_empty() {
            store_trivial_context_map(
                self.arena.get_or_insert_with(ContextMapArena::new),
                mb.distance_histograms.len(),
                DISTANCE_CONTEXT_BITS,
                &mut self.tree,
                w,
            );
        } else {
            encode_context_map(
                self.arena.get_or_insert_with(ContextMapArena::new),
                &mb.distance_context_map,
                mb.distance_histograms.len(),
                &mut self.tree,
                w,
            );
        }

        literal_enc.build_and_store_entropy_codes(
            &mb.literal_histograms,
            NUM_LITERAL_SYMBOLS,
            &mut self.tree,
            w,
        );
        command_enc.build_and_store_entropy_codes(
            &mb.command_histograms,
            NUM_COMMAND_SYMBOLS,
            &mut self.tree,
            w,
        );
        distance_enc.build_and_store_entropy_codes(
            &mb.distance_histograms,
            num_distance_symbols,
            &mut self.tree,
            w,
        );

        let mut pos = start_pos;
        let mut prev_byte = prev_byte;
        let mut prev_byte2 = prev_byte2;
        for command in commands {
            command_enc.store_symbol(usize::from(command.cmd_prefix), w);
            let (nbits, bits) = command.extra_bits();
            w.write(nbits, bits);
            if mb.literal_context_map.is_empty() {
                for _ in 0..command.insert_len {
                    let literal = input.get(pos & mask).copied().unwrap_or(0);
                    literal_enc.store_symbol(usize::from(literal), w);
                    pos += 1;
                }
            } else {
                for _ in 0..command.insert_len {
                    let literal = input.get(pos & mask).copied().unwrap_or(0);
                    literal_enc.store_symbol_with_context(
                        usize::from(literal),
                        context_mode.context(prev_byte, prev_byte2),
                        &mb.literal_context_map,
                        LITERAL_CONTEXT_BITS,
                        w,
                    );
                    prev_byte2 = prev_byte;
                    prev_byte = literal;
                    pos += 1;
                }
            }
            pos += command.copy_len() as usize;
            if command.copy_len() != 0 {
                prev_byte2 = input.get((pos - 2) & mask).copied().unwrap_or(0);
                prev_byte = input.get((pos - 1) & mask).copied().unwrap_or(0);
                if command.cmd_prefix >= 128 {
                    let dist_code = usize::from(command.distance_code());
                    if mb.distance_context_map.is_empty() {
                        distance_enc.store_symbol(dist_code, w);
                    } else {
                        distance_enc.store_symbol_with_context(
                            dist_code,
                            command.distance_context(),
                            &mb.distance_context_map,
                            DISTANCE_CONTEXT_BITS,
                            w,
                        );
                    }
                    w.write(command.distance_extra_bits(), u64::from(command.dist_extra));
                }
            }
        }
        if is_last {
            w.jump_to_byte_boundary();
        }
    }

    /// Writes a meta-block whose command and distance codes may be static.
    ///
    /// Mirrors `BrotliStoreMetaBlockFast`, the storage quality two uses. Below
    /// a hundred and twenty-nine commands only the literal code is built from
    /// the data; the command and distance codes are the fixed ones the format
    /// defines, which costs ratio but saves two prefix-code builds and their
    /// descriptions. Above that the three codes are built the fast way, which
    /// orders leaves by count alone rather than running the full package-merge
    /// that [`MetaBlockWriter::store_meta_block_trivial`] uses.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors BrotliStoreMetaBlockFast, whose parameters are all needed"
    )]
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn store_meta_block_fast(
        &mut self,
        input: &[u8],
        start_pos: usize,
        length: usize,
        mask: usize,
        is_last: bool,
        dist: &DistanceParams,
        commands: &[Command],
        w: &mut BitWriter,
    ) {
        let num_distance_symbols = dist.alphabet_size_max;
        let distance_alphabet_bits = log2_floor_non_zero(num_distance_symbols as usize - 1) + 1;

        store_compressed_meta_block_header(is_last, length, w);

        // One literal, one command and one distance block type, and no context
        // map: thirteen zero bits.
        w.write(13, 0);

        if commands.len() <= MAX_COMMANDS_FOR_STATIC_CODES {
            // Only the literals are worth a code of their own at this size.
            let mut literals = [0u32; NUM_LITERAL_SYMBOLS];
            let mut pos = start_pos;
            let mut num_literals = 0usize;
            for command in commands {
                for _ in 0..command.insert_len {
                    let byte = usize::from(input.get(pos & mask).copied().unwrap_or(0));
                    literals[byte] += 1;
                    pos += 1;
                }
                num_literals += command.insert_len as usize;
                pos += command.copy_len() as usize;
            }
            self.prepare_literals();
            build_and_store_huffman_tree_fast(
                &mut self.tree,
                &literals,
                num_literals,
                8,
                &mut self.literal_depth,
                &mut self.literal_bits,
                w,
            );
            store_static_command_huffman_tree(w);
            store_static_distance_huffman_tree(w);
            store_data_with_huffman_codes(
                input,
                start_pos,
                mask,
                commands,
                &self.literal_depth,
                &self.literal_bits,
                &STATIC_COMMAND_CODE_DEPTH,
                &STATIC_COMMAND_CODE_BITS,
                &STATIC_DISTANCE_CODE_DEPTH,
                &STATIC_DISTANCE_CODE_BITS,
                w,
            );
        } else {
            self.prepare_simple_codes();
            let mut lit_histo = super::histogram::HistogramLiteral::default();
            let mut cmd_histo = super::histogram::HistogramCommand::default();
            let mut dist_histo = super::histogram::HistogramDistance::default();
            build_histograms(
                input,
                start_pos,
                mask,
                commands,
                &mut lit_histo,
                &mut cmd_histo,
                &mut dist_histo,
            );
            build_and_store_huffman_tree_fast(
                &mut self.tree,
                &lit_histo.data,
                lit_histo.total_count,
                8,
                &mut self.literal_depth,
                &mut self.literal_bits,
                w,
            );
            build_and_store_huffman_tree_fast(
                &mut self.tree,
                &cmd_histo.data,
                cmd_histo.total_count,
                10,
                &mut self.command_depth,
                &mut self.command_bits,
                w,
            );
            build_and_store_huffman_tree_fast(
                &mut self.tree,
                &dist_histo.data,
                dist_histo.total_count,
                distance_alphabet_bits,
                &mut self.distance_depth,
                &mut self.distance_bits,
                w,
            );
            store_data_with_huffman_codes(
                input,
                start_pos,
                mask,
                commands,
                &self.literal_depth,
                &self.literal_bits,
                &self.command_depth,
                &self.command_bits,
                &self.distance_depth,
                &self.distance_bits,
                w,
            );
        }

        if is_last {
            w.jump_to_byte_boundary();
        }
    }

    /// Writes a meta-block with one prefix code per stream.
    ///
    /// Mirrors `BrotliStoreMetaBlockTrivial`, the storage quality three uses.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors BrotliStoreMetaBlockTrivial, whose parameters are all needed"
    )]
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn store_meta_block_trivial(
        &mut self,
        input: &[u8],
        start_pos: usize,
        length: usize,
        mask: usize,
        is_last: bool,
        dist: &DistanceParams,
        commands: &[Command],
        w: &mut BitWriter,
    ) {
        self.prepare_simple_codes();
        let num_distance_symbols = dist.alphabet_size_max as usize;

        store_compressed_meta_block_header(is_last, length, w);

        let mut lit_histo = super::histogram::HistogramLiteral::default();
        let mut cmd_histo = super::histogram::HistogramCommand::default();
        let mut dist_histo = super::histogram::HistogramDistance::default();
        build_histograms(
            input,
            start_pos,
            mask,
            commands,
            &mut lit_histo,
            &mut cmd_histo,
            &mut dist_histo,
        );

        // One literal, one command and one distance block type, and no context
        // map: thirteen zero bits.
        w.write(13, 0);

        build_and_store_huffman_tree(
            &lit_histo.data,
            NUM_LITERAL_SYMBOLS,
            NUM_LITERAL_SYMBOLS,
            &mut self.tree,
            &mut self.literal_depth,
            &mut self.literal_bits,
            w,
        );
        build_and_store_huffman_tree(
            &cmd_histo.data,
            NUM_COMMAND_SYMBOLS,
            NUM_COMMAND_SYMBOLS,
            &mut self.tree,
            &mut self.command_depth,
            &mut self.command_bits,
            w,
        );
        build_and_store_huffman_tree(
            &dist_histo.data,
            MAX_SIMPLE_DISTANCE_ALPHABET_SIZE,
            num_distance_symbols,
            &mut self.tree,
            &mut self.distance_depth,
            &mut self.distance_bits,
            w,
        );
        store_data_with_huffman_codes(
            input,
            start_pos,
            mask,
            commands,
            &self.literal_depth,
            &self.literal_bits,
            &self.command_depth,
            &self.command_bits,
            &self.distance_depth,
            &self.distance_bits,
            w,
        );
        if is_last {
            w.jump_to_byte_boundary();
        }
    }
}

/// Largest command count that still gets the fixed command and distance codes.
///
/// Mirrors the `n_commands <= 128` test in `BrotliStoreMetaBlockFast`: below
/// this a code description costs more than the fixed code wastes.
const MAX_COMMANDS_FOR_STATIC_CODES: usize = 128;

/// Writes the description of the static command prefix code.
///
/// Mirrors `StoreStaticCommandHuffmanTree`, which ships the description as the
/// fifty-nine literal bits it encodes to rather than rebuilding it.
fn store_static_command_huffman_tree(w: &mut BitWriter) {
    w.write(56, 0x0092_6244_1630_7003);
    w.write(3, 0);
}

/// Writes the description of the static distance prefix code.
///
/// Mirrors `StoreStaticDistanceHuffmanTree`.
fn store_static_distance_huffman_tree(w: &mut BitWriter) {
    w.write(28, 0x0369_DC03);
}

/// Gathers the three histograms of a meta-block (`BuildHistograms`).
fn build_histograms(
    input: &[u8],
    start_pos: usize,
    mask: usize,
    commands: &[Command],
    lit_histo: &mut super::histogram::HistogramLiteral,
    cmd_histo: &mut super::histogram::HistogramCommand,
    dist_histo: &mut super::histogram::HistogramDistance,
) {
    let mut pos = start_pos;
    for command in commands {
        cmd_histo.add(usize::from(command.cmd_prefix));
        for _ in 0..command.insert_len {
            lit_histo.add(usize::from(input.get(pos & mask).copied().unwrap_or(0)));
            pos += 1;
        }
        pos += command.copy_len() as usize;
        if command.has_distance() {
            dist_histo.add(usize::from(command.distance_code()));
        }
    }
}

/// Writes every command with one fixed set of codes.
///
/// Mirrors `StoreDataWithHuffmanCodes`.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors StoreDataWithHuffmanCodes, whose parameters are all needed"
)]
fn store_data_with_huffman_codes(
    input: &[u8],
    start_pos: usize,
    mask: usize,
    commands: &[Command],
    lit_depth: &[u8],
    lit_bits: &[u16],
    cmd_depth: &[u8],
    cmd_bits: &[u16],
    dist_depth: &[u8],
    dist_bits: &[u16],
    w: &mut BitWriter,
) {
    let mut pos = start_pos;
    for command in commands {
        let cmd_code = usize::from(command.cmd_prefix);
        w.write(
            u32::from(cmd_depth[cmd_code]),
            u64::from(cmd_bits[cmd_code]),
        );
        let (nbits, bits) = command.extra_bits();
        w.write(nbits, bits);
        for _ in 0..command.insert_len {
            let literal = usize::from(input.get(pos & mask).copied().unwrap_or(0));
            w.write(u32::from(lit_depth[literal]), u64::from(lit_bits[literal]));
            pos += 1;
        }
        pos += command.copy_len() as usize;
        if command.has_distance() {
            let dist_code = usize::from(command.distance_code());
            w.write(
                u32::from(dist_depth[dist_code]),
                u64::from(dist_bits[dist_code]),
            );
            w.write(command.distance_extra_bits(), u64::from(command.dist_extra));
        }
    }
}

/// Writes the input verbatim as an uncompressed meta-block.
///
/// Mirrors `BrotliStoreUncompressedMetaBlock`. An uncompressed meta-block is
/// never final, so a final empty one is appended when the stream ends here.
pub(crate) fn store_uncompressed_meta_block(
    is_final_block: bool,
    input: &[u8],
    position: usize,
    mask: usize,
    len: usize,
    w: &mut BitWriter,
) {
    let mut masked_pos = position & mask;
    let mut len = len;
    store_uncompressed_meta_block_header(len, w);
    w.jump_to_byte_boundary();

    // `mask + 1` is the window size; a mask of every bit set means the caller
    // is handing over a flat buffer that cannot wrap.
    if let Some(window_size) = mask.checked_add(1)
        && masked_pos + len > window_size
    {
        let head = window_size - masked_pos;
        if let Some(slice) = input.get(masked_pos..masked_pos + head) {
            w.write_bytes(slice);
        }
        len -= head;
        masked_pos = 0;
    }
    if let Some(slice) = input.get(masked_pos..masked_pos + len) {
        w.write_bytes(slice);
    }
    w.prepare_storage();

    if is_final_block {
        w.write(1, 1);
        w.write(1, 1);
        w.jump_to_byte_boundary();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_static_blocks_allocate_only_literal_scratch_and_match_preallocated_storage() {
        let mut lazy = MetaBlockWriter::default();
        assert_eq!(lazy.retained_bytes(), 0);
        for length in [1, 2, 3, 4, 16, 128, 129, 256] {
            let data: Vec<u8> = (0..length).map(|i| i as u8).collect();
            let commands = [Command::insert_only(length)];
            let mut preallocated = MetaBlockWriter::default();
            preallocated.prepare_simple_codes();
            let expected = written(|writer| {
                preallocated.store_meta_block_fast(
                    &data,
                    0,
                    length,
                    usize::MAX,
                    true,
                    &DistanceParams::default(),
                    &commands,
                    writer,
                );
            });
            let actual = written(|writer| {
                lazy.store_meta_block_fast(
                    &data,
                    0,
                    length,
                    usize::MAX,
                    true,
                    &DistanceParams::default(),
                    &commands,
                    writer,
                );
            });
            assert_eq!(actual, expected);
            assert!(lazy.command_depth.is_empty());
            assert!(lazy.distance_depth.is_empty());
            assert!(lazy.arena.is_none());
            assert!(lazy.retained_bytes() < preallocated.retained_bytes());
        }
    }

    #[test]
    fn block_length_codes_cover_their_ranges() {
        for &(offset, nbits) in &PREFIX_CODE_RANGES {
            let last = offset + (1u32 << nbits) - 1;
            let (code, extra_bits, extra) = block_length_code(offset);
            assert_eq!(PREFIX_CODE_RANGES[code].0, offset);
            assert_eq!(extra_bits, nbits);
            assert_eq!(extra, 0);
            let (code, _, extra) = block_length_code(last.min(1 << 24));
            assert!(code < NUM_BLOCK_LEN_SYMBOLS);
            assert!(extra < (1u32 << PREFIX_CODE_RANGES[code].1));
        }
    }

    #[test]
    fn meta_block_lengths_use_the_narrowest_nibble_count() {
        assert_eq!(encode_mlen(1), (0, 16, 0));
        assert_eq!(encode_mlen(1 << 16), (0, 16, (1 << 16) - 1));
        assert_eq!(encode_mlen((1 << 16) + 1), (1, 20, 1 << 16));
        assert_eq!(encode_mlen(1 << 20), (1, 20, (1 << 20) - 1));
        assert_eq!(encode_mlen((1 << 20) + 1), (2, 24, 1 << 20));
        assert_eq!(encode_mlen(1 << 24), (2, 24, (1 << 24) - 1));
    }

    fn written(bits: impl FnOnce(&mut BitWriter)) -> (Vec<u8>, usize) {
        let mut storage = vec![0u8; 4096];
        let mut w = BitWriter::new(&mut storage, 0);
        bits(&mut w);
        let position = w.position();
        assert!(!w.overflowed());
        (storage, position)
    }

    #[test]
    fn var_len_uint8_round_trips_its_range() {
        for n in 0usize..256 {
            let (_, position) = written(|w| store_var_len_uint8(n, w));
            if n == 0 {
                assert_eq!(position, 1);
            } else {
                let nbits = log2_floor_non_zero(n);
                assert_eq!(position, 1 + 3 + nbits as usize);
            }
        }
    }

    #[test]
    fn block_type_codes_track_the_two_last_types() {
        let mut calculator = BlockTypeCodeCalculator::new();
        // The first switch to type 0 is "the second last type".
        assert_eq!(calculator.next(0), 0);
        // Type 1 follows type 0, so it is the "next type" code.
        assert_eq!(calculator.next(1), 1);
        // Back to 0: the second last type again.
        assert_eq!(calculator.next(0), 0);
        // A jump is spelled out as the type plus two.
        assert_eq!(calculator.next(5), 7);
    }

    #[test]
    fn move_to_front_brings_the_used_index_forward() {
        let mut v = vec![0u8, 1, 2, 3];
        move_to_front(&mut v, 2);
        assert_eq!(v, vec![2, 0, 1, 3]);
        move_to_front(&mut v, 0);
        assert_eq!(v, vec![2, 0, 1, 3]);
    }

    #[test]
    fn the_move_to_front_transform_rewards_repetition() {
        let mut output = Vec::new();
        move_to_front_transform(&[0, 0, 0, 1, 1, 0], &mut output);
        assert_eq!(output, vec![0, 0, 0, 1, 0, 1]);

        move_to_front_transform(&[], &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn zero_runs_are_replaced_by_prefix_codes() {
        let mut v = vec![0u32; 10];
        let (size, prefix) = run_length_code_zeros(&mut v, 6);
        assert_eq!(size, 1);
        assert_eq!(prefix, 3);
        assert_eq!(v[0] & 0x1FF, 3);
        assert_eq!(v[0] >> 9, 10 - 8);

        let mut v = vec![1u32, 0, 0, 2];
        let (size, prefix) = run_length_code_zeros(&mut v, 6);
        assert_eq!(prefix, 1);
        assert_eq!(size, 3);
        assert_eq!(v[0], 1 + prefix);
        assert_eq!(v[2], 2 + prefix);
    }

    #[test]
    fn a_map_without_zeros_is_left_alone() {
        let mut v = vec![1u32, 2, 3];
        let (size, prefix) = run_length_code_zeros(&mut v, 6);
        assert_eq!(prefix, 0);
        assert_eq!(size, 3);
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[test]
    fn an_uncompressed_meta_block_carries_its_bytes_verbatim() {
        let payload: Vec<u8> = (0..300u32).map(|i| (i % 251) as u8).collect();
        let mut storage = vec![0u8; 4096];
        let mut w = BitWriter::new(&mut storage, 0);
        store_uncompressed_meta_block(false, &payload, 0, usize::MAX, payload.len(), &mut w);
        assert!(!w.overflowed());
        let bytes = w.position() >> 3;
        assert!(bytes >= payload.len());
        // The payload appears byte aligned, after the header.
        let window = &storage[..bytes];
        assert!(
            window
                .windows(payload.len())
                .any(|slice| slice == payload.as_slice())
        );
    }

    #[test]
    fn a_final_uncompressed_meta_block_is_followed_by_an_empty_one() {
        let payload = vec![7u8; 16];
        let mut storage = vec![0u8; 256];
        let mut w = BitWriter::new(&mut storage, 0);
        store_uncompressed_meta_block(true, &payload, 0, usize::MAX, payload.len(), &mut w);
        assert_eq!(w.position() % 8, 0);
        assert!(!w.overflowed());
    }

    #[test]
    fn an_uncompressed_meta_block_wraps_around_the_ring_buffer() {
        let mut ring = vec![0u8; 16];
        for (index, byte) in ring.iter_mut().enumerate() {
            *byte = index as u8;
        }
        let mut storage = vec![0u8; 256];
        let mut w = BitWriter::new(&mut storage, 0);
        // Start near the end of the window, so the copy has to split.
        store_uncompressed_meta_block(false, &ring, 12, 15, 8, &mut w);
        assert!(!w.overflowed());
        let bytes = w.position() >> 3;
        let window = &storage[..bytes];
        assert!(
            window
                .windows(8)
                .any(|slice| slice == [12u8, 13, 14, 15, 0, 1, 2, 3])
        );
    }

    #[test]
    fn a_compressed_header_marks_the_last_block() {
        let (storage, position) = written(|w| store_compressed_meta_block_header(true, 100, w));
        // ISLAST, ISEMPTY, two nibble bits and sixteen length bits.
        assert_eq!(position, 1 + 1 + 2 + 16);
        assert_eq!(storage[0] & 1, 1);

        let (storage, position) = written(|w| store_compressed_meta_block_header(false, 100, w));
        // ISLAST, two nibble bits, sixteen length bits and ISUNCOMPRESSED.
        assert_eq!(position, 1 + 2 + 16 + 1);
        assert_eq!(storage[0] & 1, 0);
    }
}

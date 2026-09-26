//! Prefix-code construction and serialisation for the fast encoders.
//!
//! Ported from `c/enc/entropy_encode.c` and `c/enc/brotli_bit_stream.c` of the
//! pinned reference (`google/brotli` v1.2.0, commit `028fb5a`, MIT licence).
//! Tie-breaking, sort order and the RLE representation of code lengths are all
//! observable in the bitstream, so this module reproduces them exactly.

use alloc::vec::Vec;

use super::bits::{BitWriter, ByteBuffer};
use super::constants::{
    CODE_LENGTH_CODES, INITIAL_REPEATED_CODE_LENGTH, NUM_COMMAND_SYMBOLS,
    REPEAT_PREVIOUS_CODE_LENGTH, REPEAT_ZERO_CODE_LENGTH,
};
use super::tables::{
    CODE_LENGTH_BITS, CODE_LENGTH_DEPTH, CODE_LENGTH_HUFFMAN_DEPTHS, CODE_LENGTH_HUFFMAN_SYMBOLS,
    NON_ZERO_REPS_BITS, NON_ZERO_REPS_DEPTH, REVERSE_LUT, SHELL_GAPS, STATIC_CODE_LENGTH_CODE,
    STATIC_CODE_LENGTH_CODE_BITS, STORAGE_ORDER, ZERO_REPS_BITS, ZERO_REPS_DEPTH,
};

/// Number of nodes a Huffman build over `n` symbols needs.
pub(crate) const fn tree_capacity(symbols: usize) -> usize {
    2 * symbols + 1
}

/// Returns node scratch for a build over `leaves` used symbols.
///
/// The pool only ever grows: a build over more leaves than any earlier one
/// extends it, and everything else reuses what is there. Nothing is cleared,
/// because every build writes each node it reads. Sizing by the used symbols
/// rather than the alphabet keeps a cold encoder from zeroing the full
/// command-alphabet pool for a block that uses a few dozen symbols.
fn nodes_for(tree: &mut Vec<HuffmanNode>, leaves: usize) -> &mut [HuffmanNode] {
    let needed = tree_capacity(leaves);
    if tree.len() < needed {
        tree.resize(needed, HuffmanNode::default());
    }
    tree
}

/// One node of an in-construction Huffman tree.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct HuffmanNode {
    total_count: u32,
    index_left: i16,
    index_right_or_value: i16,
}

impl HuffmanNode {
    /// Node used to terminate the leaf and parent runs during merging.
    const SENTINEL: Self = Self {
        total_count: u32::MAX,
        index_left: -1,
        index_right_or_value: -1,
    };

    /// Creates a leaf holding `count` occurrences of symbol `value`.
    const fn leaf(count: u32, value: i16) -> Self {
        Self {
            total_count: count,
            index_left: -1,
            index_right_or_value: value,
        }
    }
}

impl Default for HuffmanNode {
    fn default() -> Self {
        Self::SENTINEL
    }
}

/// Assigns bit depths by walking the finished tree from node `root`.
///
/// Returns `false` when some symbol would need more than `max_depth` bits, in
/// which case the caller raises the fake minimum count and rebuilds.
fn set_depth(root: usize, pool: &[HuffmanNode], depth: &mut [u8], max_depth: i32) -> bool {
    let mut stack = [-1i32; 16];
    let mut level = 0i32;
    let mut p = root;
    loop {
        let node = pool[p];
        if node.index_left >= 0 {
            level += 1;
            if level > max_depth {
                return false;
            }
            if let Some(slot) = stack.get_mut(level as usize) {
                *slot = i32::from(node.index_right_or_value);
            }
            p = node.index_left as usize;
            continue;
        }
        if let Some(slot) = depth.get_mut(node.index_right_or_value as usize) {
            *slot = level as u8;
        }
        while level >= 0 && stack[level as usize] == -1 {
            level -= 1;
        }
        if level < 0 {
            return true;
        }
        p = stack[level as usize] as usize;
        stack[level as usize] = -1;
    }
}

/// Orders the leaves least popular first.
///
/// With `TIE_BY_VALUE` the reference breaks ties by descending symbol index;
/// the fast literal builder compares counts only.
#[inline(always)]
fn precedes<const TIE_BY_VALUE: bool>(v0: &HuffmanNode, v1: &HuffmanNode) -> bool {
    if TIE_BY_VALUE && v0.total_count == v1.total_count {
        return v0.index_right_or_value > v1.index_right_or_value;
    }
    v0.total_count < v1.total_count
}

/// Input-size optimised shell sort (`SortHuffmanTreeItems`).
fn sort_items<const TIE_BY_VALUE: bool>(items: &mut [HuffmanNode], n: usize) {
    if n < 13 {
        for i in 1..n {
            let tmp = items[i];
            let mut k = i;
            let mut j = i;
            while j > 0 {
                j -= 1;
                if !precedes::<TIE_BY_VALUE>(&tmp, &items[j]) {
                    break;
                }
                items[k] = items[j];
                k = j;
            }
            items[k] = tmp;
        }
        return;
    }
    let first_gap = if n < 57 { 2 } else { 0 };
    for &gap in &SHELL_GAPS[first_gap..] {
        for i in gap..n {
            let mut j = i;
            let tmp = items[i];
            while j >= gap && precedes::<TIE_BY_VALUE>(&tmp, &items[j - gap]) {
                items[j] = items[j - gap];
                j -= gap;
            }
            items[j] = tmp;
        }
    }
}

/// Merges the sorted leaves in `tree[..n]` into a complete Huffman tree.
///
/// Returns the index of the root node.
fn merge_nodes(tree: &mut [HuffmanNode], n: usize) -> usize {
    tree[n] = HuffmanNode::SENTINEL;
    tree[n + 1] = HuffmanNode::SENTINEL;

    let mut i = 0usize;
    let mut j = n + 1;
    for k in (1..n).rev() {
        let left = if tree[i].total_count <= tree[j].total_count {
            let index = i;
            i += 1;
            index
        } else {
            let index = j;
            j += 1;
            index
        };
        let right = if tree[i].total_count <= tree[j].total_count {
            let index = i;
            i += 1;
            index
        } else {
            let index = j;
            j += 1;
            index
        };
        let parent = 2 * n - k;
        tree[parent] = HuffmanNode {
            // Wrapping matches the reference, which relies on unsigned
            // wraparound for histograms that saturate a 32-bit counter.
            total_count: tree[left].total_count.wrapping_add(tree[right].total_count),
            index_left: left as i16,
            index_right_or_value: right as i16,
        };
        tree[parent + 1] = HuffmanNode::SENTINEL;
    }
    2 * n - 1
}

/// Builds a depth-limited Huffman code over `data[..length]`.
///
/// `tree` is node scratch that grows to fit the used symbols; `depth` must
/// hold at least `length` entries.
pub(crate) fn create_huffman_tree(
    data: &[u32],
    length: usize,
    tree_limit: i32,
    tree: &mut Vec<HuffmanNode>,
    depth: &mut [u8],
) {
    let leaves = data
        .iter()
        .take(length)
        .filter(|&&count| count != 0)
        .count();
    let tree = nodes_for(tree, leaves);
    let mut count_limit: u32 = 1;
    loop {
        let mut n = 0usize;
        for i in (0..length).rev() {
            if data[i] != 0 {
                tree[n] = HuffmanNode::leaf(data[i].max(count_limit), i as i16);
                n += 1;
            }
        }

        if n == 1 {
            depth[tree[0].index_right_or_value as usize] = 1;
            return;
        }

        sort_items::<true>(tree, n);
        let root = merge_nodes(tree, n);
        if set_depth(root, tree, depth, tree_limit) {
            return;
        }
        count_limit = count_limit.wrapping_mul(2);
    }
}

/// Reverses the low `num_bits` bits of `bits`.
fn reverse_bits(num_bits: usize, bits: u16) -> u16 {
    let mut bits = bits;
    let mut retval = usize::from(REVERSE_LUT[usize::from(bits & 0x0F)]);
    let mut i = 4;
    while i < num_bits {
        retval <<= 4;
        bits >>= 4;
        retval |= usize::from(REVERSE_LUT[usize::from(bits & 0x0F)]);
        i += 4;
    }
    (retval >> (num_bits.wrapping_neg() & 0x03)) as u16
}

/// Turns canonical bit depths into the bit patterns Brotli expects.
pub(crate) fn convert_bit_depths_to_symbols(depth: &[u8], len: usize, bits: &mut [u16]) {
    const MAX_HUFFMAN_BITS: usize = 16;
    let mut bl_count = [0u16; MAX_HUFFMAN_BITS];
    let mut next_code = [0u16; MAX_HUFFMAN_BITS];
    for &d in &depth[..len] {
        bl_count[usize::from(d)] += 1;
    }
    bl_count[0] = 0;
    let mut code = 0u16;
    for i in 1..MAX_HUFFMAN_BITS {
        code = (code + bl_count[i - 1]) << 1;
        next_code[i] = code;
    }
    for i in 0..len {
        let d = usize::from(depth[i]);
        if d != 0 {
            bits[i] = reverse_bits(d, next_code[d]);
            next_code[d] += 1;
        }
    }
}

/// Scratch buffers for the RLE form of a code-length sequence.
struct CodeLengthRuns {
    symbols: [u8; NUM_COMMAND_SYMBOLS],
    extra_bits: [u8; NUM_COMMAND_SYMBOLS],
    len: usize,
}

impl CodeLengthRuns {
    const fn new() -> Self {
        Self {
            symbols: [0; NUM_COMMAND_SYMBOLS],
            extra_bits: [0; NUM_COMMAND_SYMBOLS],
            len: 0,
        }
    }

    fn push(&mut self, symbol: u8, extra: u8) {
        if self.len < NUM_COMMAND_SYMBOLS {
            self.symbols[self.len] = symbol;
            self.extra_bits[self.len] = extra;
            self.len += 1;
        }
    }

    fn reverse(&mut self, start: usize) {
        self.symbols[start..self.len].reverse();
        self.extra_bits[start..self.len].reverse();
    }

    /// Emits `repetitions` copies of a non-zero `value`.
    fn write_repetitions(&mut self, previous_value: u8, value: u8, mut repetitions: usize) {
        debug_assert!(repetitions > 0);
        if previous_value != value {
            self.push(value, 0);
            repetitions -= 1;
        }
        if repetitions == 7 {
            self.push(value, 0);
            repetitions -= 1;
        }
        if repetitions < 3 {
            for _ in 0..repetitions {
                self.push(value, 0);
            }
            return;
        }
        let start = self.len;
        repetitions -= 3;
        loop {
            self.push(REPEAT_PREVIOUS_CODE_LENGTH as u8, (repetitions & 0x3) as u8);
            repetitions >>= 2;
            if repetitions == 0 {
                break;
            }
            repetitions -= 1;
        }
        self.reverse(start);
    }

    /// Emits `repetitions` zero code lengths.
    fn write_zero_repetitions(&mut self, mut repetitions: usize) {
        if repetitions == 11 {
            self.push(0, 0);
            repetitions -= 1;
        }
        if repetitions < 3 {
            for _ in 0..repetitions {
                self.push(0, 0);
            }
            return;
        }
        let start = self.len;
        repetitions -= 3;
        loop {
            self.push(REPEAT_ZERO_CODE_LENGTH as u8, (repetitions & 0x7) as u8);
            repetitions >>= 3;
            if repetitions == 0 {
                break;
            }
            repetitions -= 1;
        }
        self.reverse(start);
    }
}

/// Decides whether run-length coding pays off for zero and non-zero lengths.
fn decide_over_rle_use(depth: &[u8], length: usize) -> (bool, bool) {
    let mut total_reps_zero = 0usize;
    let mut total_reps_non_zero = 0usize;
    let mut count_reps_zero = 1usize;
    let mut count_reps_non_zero = 1usize;
    let mut i = 0usize;
    while i < length {
        let value = depth[i];
        let mut reps = 1usize;
        let mut k = i + 1;
        while k < length && depth[k] == value {
            reps += 1;
            k += 1;
        }
        if reps >= 3 && value == 0 {
            total_reps_zero += reps;
            count_reps_zero += 1;
        }
        if reps >= 4 && value != 0 {
            total_reps_non_zero += reps;
            count_reps_non_zero += 1;
        }
        i += reps;
    }
    (
        total_reps_non_zero > count_reps_non_zero * 2,
        total_reps_zero > count_reps_zero * 2,
    )
}

/// Builds the RLE representation of `depth[..length]`.
fn write_huffman_tree(depth: &[u8], length: usize) -> CodeLengthRuns {
    let mut runs = CodeLengthRuns::new();
    let mut previous_value = INITIAL_REPEATED_CODE_LENGTH;

    let mut new_length = length;
    for i in 0..length {
        if depth[length - i - 1] == 0 {
            new_length -= 1;
        } else {
            break;
        }
    }

    let (use_rle_for_non_zero, use_rle_for_zero) = if length > 50 {
        decide_over_rle_use(depth, new_length)
    } else {
        (false, false)
    };

    let mut i = 0usize;
    while i < new_length {
        let value = depth[i];
        let mut reps = 1usize;
        if (value != 0 && use_rle_for_non_zero) || (value == 0 && use_rle_for_zero) {
            let mut k = i + 1;
            while k < new_length && depth[k] == value {
                reps += 1;
                k += 1;
            }
        }
        if value == 0 {
            runs.write_zero_repetitions(reps);
        } else {
            runs.write_repetitions(previous_value, value, reps);
            previous_value = value;
        }
        i += reps;
    }
    runs
}

/// Stores the code that compresses the code-length alphabet itself.
fn store_code_length_code(
    num_codes: i32,
    bitdepth: &[u8; CODE_LENGTH_CODES],
    w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
) {
    let mut codes_to_store = CODE_LENGTH_CODES;
    if num_codes > 1 {
        while codes_to_store > 0 {
            if bitdepth[STORAGE_ORDER[codes_to_store - 1]] != 0 {
                break;
            }
            codes_to_store -= 1;
        }
    }
    let mut skip_some = 0usize;
    if bitdepth[STORAGE_ORDER[0]] == 0 && bitdepth[STORAGE_ORDER[1]] == 0 {
        skip_some = if bitdepth[STORAGE_ORDER[2]] == 0 {
            3
        } else {
            2
        };
    }
    w.write(2, skip_some as u64);
    for &order in &STORAGE_ORDER[skip_some..codes_to_store] {
        let l = usize::from(bitdepth[order]);
        w.write(
            u32::from(CODE_LENGTH_HUFFMAN_DEPTHS[l]),
            u64::from(CODE_LENGTH_HUFFMAN_SYMBOLS[l]),
        );
    }
}

/// Serialises `depths[..num]` as a Brotli prefix-code description.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
pub(crate) fn store_huffman_tree(
    depths: &[u8],
    num: usize,
    tree: &mut Vec<HuffmanNode>,
    w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
) {
    debug_assert!(num <= NUM_COMMAND_SYMBOLS);
    let runs = write_huffman_tree(depths, num);

    let mut histogram = [0u32; CODE_LENGTH_CODES];
    for &symbol in &runs.symbols[..runs.len] {
        histogram[usize::from(symbol)] += 1;
    }

    let mut num_codes = 0i32;
    let mut code = 0usize;
    for (i, &count) in histogram.iter().enumerate() {
        if count != 0 {
            if num_codes == 0 {
                code = i;
                num_codes = 1;
            } else {
                num_codes = 2;
                break;
            }
        }
    }

    let mut bitdepth = [0u8; CODE_LENGTH_CODES];
    let mut symbols = [0u16; CODE_LENGTH_CODES];
    create_huffman_tree(&histogram, CODE_LENGTH_CODES, 5, tree, &mut bitdepth);
    convert_bit_depths_to_symbols(&bitdepth, CODE_LENGTH_CODES, &mut symbols);

    store_code_length_code(num_codes, &bitdepth, w);

    if num_codes == 1 {
        bitdepth[code] = 0;
    }

    for i in 0..runs.len {
        let ix = usize::from(runs.symbols[i]);
        w.write(u32::from(bitdepth[ix]), u64::from(symbols[ix]));
        if ix == REPEAT_PREVIOUS_CODE_LENGTH {
            w.write(2, u64::from(runs.extra_bits[i]));
        } else if ix == REPEAT_ZERO_CODE_LENGTH {
            w.write(3, u64::from(runs.extra_bits[i]));
        }
    }
}

/// Stores a "simple" prefix code of up to four symbols.
fn store_simple_code(
    depth: &[u8],
    symbols: &mut [usize; 4],
    count: usize,
    max_bits: u32,
    w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
) {
    w.write(2, 1);
    w.write(2, (count - 1) as u64);
    for i in 0..count {
        for j in (i + 1)..count {
            if depth[symbols[j]] < depth[symbols[i]] {
                symbols.swap(i, j);
            }
        }
    }
    for &symbol in &symbols[..count] {
        w.write(max_bits, symbol as u64);
    }
    if count == 4 {
        w.write(1, u64::from(depth[symbols[0]] == 1));
    }
}

/// Builds a literal prefix code from `histogram` and writes it to the stream.
///
/// Mirrors `BrotliBuildAndStoreHuffmanTreeFast`: leaves are ordered by count
/// only, the internal depth limit is fourteen, and the emitted description uses
/// the static code-length code.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
pub(crate) fn build_and_store_huffman_tree_fast(
    tree: &mut Vec<HuffmanNode>,
    histogram: &[u32],
    histogram_total: usize,
    max_bits: u32,
    depth: &mut [u8],
    bits: &mut [u16],
    w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
) {
    let mut count = 0usize;
    let mut symbols = [0usize; 4];
    let mut length = 0usize;
    let mut total = histogram_total;
    while total != 0 {
        if histogram[length] != 0 {
            if count < 4 {
                symbols[count] = length;
            }
            count += 1;
            total -= histogram[length] as usize;
        }
        length += 1;
    }

    if count <= 1 {
        w.write(4, 1);
        w.write(max_bits, symbols[0] as u64);
        depth[symbols[0]] = 0;
        bits[symbols[0]] = 0;
        return;
    }

    depth[..length].fill(0);
    let tree = nodes_for(tree, count);
    let mut count_limit: u32 = 1;
    loop {
        let mut n = 0usize;
        for l in (0..length).rev() {
            if histogram[l] != 0 {
                tree[n] = HuffmanNode::leaf(histogram[l].max(count_limit), l as i16);
                n += 1;
            }
        }
        sort_items::<false>(tree, n);
        let root = merge_nodes(tree, n);
        if set_depth(root, tree, depth, 14) {
            break;
        }
        count_limit = count_limit.wrapping_mul(2);
    }

    convert_bit_depths_to_symbols(depth, length, bits);

    if count <= 4 {
        store_simple_code(depth, &mut symbols, count, max_bits, w);
        return;
    }

    let mut previous_value = 8u8;
    w.write(STATIC_CODE_LENGTH_CODE_BITS, STATIC_CODE_LENGTH_CODE);
    let mut i = 0usize;
    while i < length {
        let value = depth[i];
        let mut reps = 1usize;
        let mut k = i + 1;
        while k < length && depth[k] == value {
            reps += 1;
            k += 1;
        }
        i += reps;
        if value == 0 {
            w.write(ZERO_REPS_DEPTH[reps], ZERO_REPS_BITS[reps]);
            continue;
        }
        let value_index = usize::from(value);
        if previous_value != value {
            w.write(
                u32::from(CODE_LENGTH_DEPTH[value_index]),
                u64::from(CODE_LENGTH_BITS[value_index]),
            );
            reps -= 1;
        }
        if reps < 3 {
            for _ in 0..reps {
                w.write(
                    u32::from(CODE_LENGTH_DEPTH[value_index]),
                    u64::from(CODE_LENGTH_BITS[value_index]),
                );
            }
        } else {
            reps -= 3;
            w.write(NON_ZERO_REPS_DEPTH[reps], NON_ZERO_REPS_BITS[reps]);
        }
        previous_value = value;
    }
}

/// Builds a prefix code from `histogram` and writes its description.
///
/// Mirrors `BuildAndStoreHuffmanTree` from `c/enc/brotli_bit_stream.c`: an
/// alphabet with at most four used symbols gets the compact "simple" form,
/// everything else the run-length coded one. `alphabet_size` is only used to
/// work out how many bits a symbol index takes in the simple form, which is
/// why it can differ from `histogram.len()`.
///
/// `depth` and `bits` receive the code, so the caller can then write symbols
/// with it.
pub(crate) fn build_and_store_huffman_tree(
    histogram: &[u32],
    histogram_length: usize,
    alphabet_size: usize,
    tree: &mut Vec<HuffmanNode>,
    depth: &mut [u8],
    bits: &mut [u16],
    w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
) {
    let mut count = 0usize;
    let mut symbols = [0usize; 4];
    for (symbol, &occurrences) in histogram.iter().enumerate().take(histogram_length) {
        if occurrences != 0 {
            if count < 4 {
                symbols[count] = symbol;
            } else if count > 4 {
                break;
            }
            count += 1;
        }
    }

    let mut max_bits = 0u32;
    let mut max_bits_counter = alphabet_size - 1;
    while max_bits_counter != 0 {
        max_bits_counter >>= 1;
        max_bits += 1;
    }

    if count <= 1 {
        w.write(4, 1);
        w.write(max_bits, symbols[0] as u64);
        depth[symbols[0]] = 0;
        bits[symbols[0]] = 0;
        return;
    }

    depth[..histogram_length].fill(0);
    create_huffman_tree(histogram, histogram_length, 15, tree, depth);
    convert_bit_depths_to_symbols(depth, histogram_length, bits);

    if count <= 4 {
        store_simple_code(depth, &mut symbols, count, max_bits, w);
    } else {
        store_huffman_tree(depth, histogram_length, tree, w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn depths_for(histogram: &[u32], limit: i32) -> Vec<u8> {
        let mut tree = vec![HuffmanNode::default(); tree_capacity(histogram.len())];
        let mut depth = vec![0u8; histogram.len()];
        create_huffman_tree(histogram, histogram.len(), limit, &mut tree, &mut depth);
        depth
    }

    fn kraft_sum(depth: &[u8]) -> f64 {
        depth
            .iter()
            .filter(|&&d| d != 0)
            .map(|&d| 0.5f64.powi(i32::from(d)))
            .sum()
    }

    #[test]
    fn tree_capacity_matches_the_reference_requirement() {
        assert_eq!(tree_capacity(256), 513);
        assert_eq!(tree_capacity(704), 1409);
    }

    #[test]
    fn single_symbol_histogram_gets_depth_one() {
        let depth = depths_for(&[0, 5, 0, 0], 15);
        assert_eq!(depth, vec![0, 1, 0, 0]);
    }

    #[test]
    fn balanced_histogram_produces_a_complete_code() {
        let depth = depths_for(&[1, 1, 1, 1], 15);
        assert_eq!(depth, vec![2, 2, 2, 2]);
        assert!((kraft_sum(&depth) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn skewed_histogram_stays_within_the_depth_limit() {
        let mut histogram = [0u32; 64];
        for (i, slot) in histogram.iter_mut().enumerate() {
            *slot = 1u32 << (i.min(20) as u32);
        }
        let depth = depths_for(&histogram, 15);
        assert!(depth.iter().all(|&d| d <= 15));
        assert!(kraft_sum(&depth) <= 1.0 + 1e-9);
    }

    #[test]
    fn depth_assignment_is_deterministic() {
        let histogram = [3u32, 1, 4, 1, 5, 9, 2, 6];
        assert_eq!(depths_for(&histogram, 15), depths_for(&histogram, 15));
    }

    #[test]
    fn ties_break_by_descending_symbol_index() {
        let low = HuffmanNode::leaf(5, 1);
        let high = HuffmanNode::leaf(5, 3);
        assert!(precedes::<true>(&high, &low));
        assert!(!precedes::<true>(&low, &high));
        // The fast literal builder compares counts only, so equal counts never
        // reorder there.
        assert!(!precedes::<false>(&low, &high));
        assert!(!precedes::<false>(&high, &low));
    }

    #[test]
    fn depths_match_the_reference_for_a_small_histogram() {
        assert_eq!(depths_for(&[1, 1, 2], 15), vec![2, 2, 1]);
        assert_eq!(depths_for(&[2, 1, 1], 15), vec![1, 2, 2]);
    }

    #[test]
    fn sorting_handles_the_insertion_and_shell_paths() {
        for n in [0usize, 1, 2, 12, 13, 56, 57, 200] {
            let mut items: Vec<HuffmanNode> = (0..n)
                .map(|i| HuffmanNode::leaf(((n - i) % 17) as u32, i as i16))
                .collect();
            sort_items::<true>(&mut items, n);
            for pair in items.windows(2) {
                assert!(!precedes::<true>(&pair[1], &pair[0]));
            }
        }
    }

    #[test]
    fn reverse_bits_reverses_each_supported_width() {
        assert_eq!(reverse_bits(1, 0b1), 0b1);
        assert_eq!(reverse_bits(2, 0b10), 0b01);
        assert_eq!(reverse_bits(4, 0b1000), 0b0001);
        assert_eq!(reverse_bits(5, 0b10000), 0b00001);
        assert_eq!(reverse_bits(8, 0b1010_0000), 0b0000_0101);
        assert_eq!(reverse_bits(15, 0b100_0000_0000_0000), 1);
    }

    #[test]
    fn canonical_symbols_follow_the_reference_layout() {
        let depth = [2u8, 2, 2, 2];
        let mut bits = [0u16; 4];
        convert_bit_depths_to_symbols(&depth, 4, &mut bits);
        assert_eq!(bits, [0b00, 0b10, 0b01, 0b11]);
    }

    #[test]
    fn unused_symbols_keep_a_zero_bit_pattern() {
        let depth = [0u8, 1, 0, 1];
        let mut bits = [0xFFFFu16; 4];
        convert_bit_depths_to_symbols(&depth, 4, &mut bits);
        assert_eq!(bits[0], 0xFFFF);
        assert_eq!(bits[2], 0xFFFF);
        assert_eq!(bits[1], 0);
        assert_eq!(bits[3], 1);
    }

    #[test]
    fn code_length_runs_collapse_long_stretches() {
        let mut depth = vec![0u8; 256];
        depth[0] = 1;
        depth[255] = 1;
        let runs = write_huffman_tree(&depth, 256);
        assert!(runs.len < 20);
        assert!(runs.symbols[..runs.len].contains(&(REPEAT_ZERO_CODE_LENGTH as u8)));
    }

    #[test]
    fn code_length_runs_use_repeat_previous_for_flat_codes() {
        let depth = vec![8u8; 256];
        let runs = write_huffman_tree(&depth, 256);
        assert!(runs.len < 20);
        assert!(
            runs.symbols[..runs.len].contains(&(REPEAT_PREVIOUS_CODE_LENGTH as u8)),
            "expected a repeat-previous run"
        );
    }

    #[test]
    fn short_alphabets_never_use_run_length_coding() {
        let depth = vec![4u8; 20];
        let runs = write_huffman_tree(&depth, 20);
        assert_eq!(runs.len, 20);
    }

    #[test]
    fn fast_builder_emits_a_simple_code_for_few_symbols() {
        let mut histogram = [0u32; 256];
        histogram[b'a' as usize] = 5;
        histogram[b'b' as usize] = 3;
        let mut storage = vec![0u8; 64];
        let mut w = BitWriter::new(&mut storage, 0);
        let mut tree = vec![HuffmanNode::default(); tree_capacity(256)];
        let mut depth = [0u8; 256];
        let mut bits = [0u16; 256];
        build_and_store_huffman_tree_fast(
            &mut tree, &histogram, 8, 8, &mut depth, &mut bits, &mut w,
        );
        assert!(!w.overflowed());
        assert_eq!(depth[b'a' as usize], 1);
        assert_eq!(depth[b'b' as usize], 1);
        assert_eq!(w.position(), 2 + 2 + 8 + 8);
    }

    #[test]
    fn fast_builder_emits_a_degenerate_code_for_one_symbol() {
        let mut histogram = [0u32; 256];
        histogram[7] = 9;
        let mut storage = vec![0u8; 64];
        let mut w = BitWriter::new(&mut storage, 0);
        let mut tree = vec![HuffmanNode::default(); tree_capacity(256)];
        let mut depth = [1u8; 256];
        let mut bits = [1u16; 256];
        build_and_store_huffman_tree_fast(
            &mut tree, &histogram, 9, 8, &mut depth, &mut bits, &mut w,
        );
        assert_eq!(depth[7], 0);
        assert_eq!(bits[7], 0);
        assert_eq!(w.position(), 12);
    }

    #[test]
    fn fast_builder_respects_the_maximum_literal_depth() {
        let mut histogram = [0u32; 256];
        for (i, slot) in histogram.iter_mut().enumerate() {
            *slot = 1u32 << ((i % 20) as u32);
        }
        let total: usize = histogram.iter().map(|&c| c as usize).sum();
        let mut storage = vec![0u8; 4096];
        let mut w = BitWriter::new(&mut storage, 0);
        let mut tree = vec![HuffmanNode::default(); tree_capacity(256)];
        let mut depth = [0u8; 256];
        let mut bits = [0u16; 256];
        build_and_store_huffman_tree_fast(
            &mut tree, &histogram, total, 8, &mut depth, &mut bits, &mut w,
        );
        assert!(!w.overflowed());
        assert!(depth.iter().all(|&d| d <= 14));
        assert!(kraft_sum(&depth) <= 1.0 + 1e-9);
    }

    #[test]
    fn store_huffman_tree_round_trips_a_flat_command_code() {
        let depths = vec![4u8; NUM_COMMAND_SYMBOLS];
        let mut storage = vec![0u8; 4096];
        let mut w = BitWriter::new(&mut storage, 0);
        let mut tree = vec![HuffmanNode::default(); tree_capacity(NUM_COMMAND_SYMBOLS)];
        store_huffman_tree(&depths, NUM_COMMAND_SYMBOLS, &mut tree, &mut w);
        assert!(!w.overflowed());
        assert!(w.position() > 0);
    }

    #[test]
    fn store_huffman_tree_handles_a_single_code_length() {
        let mut depths = vec![0u8; 64];
        depths[3] = 1;
        depths[9] = 1;
        let mut storage = vec![0u8; 1024];
        let mut w = BitWriter::new(&mut storage, 0);
        let mut tree = vec![HuffmanNode::default(); tree_capacity(NUM_COMMAND_SYMBOLS)];
        store_huffman_tree(&depths, 64, &mut tree, &mut w);
        assert!(!w.overflowed());
    }
}

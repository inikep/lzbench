//! Huffman encoding of literals (`huf_compress.c`).
//!
//! Tree construction is a port of the reference's bucket sort, node
//! building and height limiting: ties between equally frequent symbols are
//! broken by those exact algorithms, and the resulting code lengths are part
//! of the compressed bytes.

use fearless_simd::Simd;
use fearless_simd_macros::simd;

use super::fse;
use super::hist;
use crate::Backend;
use crate::compressor::core::error::CoreError;
use crate::shared::format::highbit32;

/// Largest table log a table may use.
pub(crate) const TABLE_LOG_MAX: u32 = 12;
/// Table log used when none is requested.
const TABLE_LOG_DEFAULT: u32 = 11;
/// Largest literal value.
pub(crate) const SYMBOL_VALUE_MAX: u32 = 255;
/// Largest block a single Huffman call accepts.
const BLOCK_SIZE_MAX: usize = 128 * 1024;
/// Table log of the FSE table that compresses weights.
const MAX_FSE_TABLE_LOG_FOR_HUFF_HEADER: u32 = 6;
/// Sample size for the incompressibility pre-check.
const SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE: usize = 4096;
/// Sample ratio for the incompressibility pre-check.
const SUSPECT_INCOMPRESSIBLE_SAMPLE_RATIO: usize = 10;
/// Capacity of the scratch destination used to price a table header; the
/// reference uses what remains of its entropy workspace, far more than any
/// header needs.
const HEADER_TRIAL_CAPACITY: usize = 512;

/// How a previous table may be reused (`HUF_repeat`).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) enum Repeat {
    /// The previous table cannot be used.
    #[default]
    None,
    /// The previous table can be used after checking it covers the input.
    Check,
    /// The previous table can be used as is.
    Valid,
}

/// Flags of the reference `HUF_flags_e` this encoder honors.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct Flags {
    /// Search table logs for the smallest output.
    pub(crate) optimal_depth: bool,
    /// Prefer reusing the previous table for small inputs.
    pub(crate) prefer_repeat: bool,
    /// Sample the input's ends before a full histogram.
    pub(crate) suspect_uncompressible: bool,
}

/// A Huffman encoding table (`HUF_CElt[]` with its header).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CTable {
    /// Longest code length in the table.
    pub(crate) table_log: u32,
    /// Largest symbol the table describes.
    pub(crate) max_symbol_value: u32,
    /// Code length per symbol; `0` for absent symbols.
    pub(crate) nb_bits: [u8; 256],
    /// Code per symbol.
    pub(crate) value: [u16; 256],
}

impl Default for CTable {
    fn default() -> Self {
        Self {
            table_log: 0,
            max_symbol_value: 0,
            nb_bits: [0; 256],
            value: [0; 256],
        }
    }
}

/// A node of the construction tree (`nodeElt`).
#[derive(Copy, Clone, Debug, Default)]
struct Node {
    count: u32,
    parent: u16,
    byte: u8,
    nb_bits: u8,
}

/// Buckets of the sort (`RANK_POSITION_TABLE_SIZE`).
const RANK_POSITION_TABLE_SIZE: usize = 192;
const RANK_POSITION_MAX_COUNT_LOG: usize = 32;
const RANK_POSITION_LOG_BUCKETS_BEGIN: u32 =
    (RANK_POSITION_TABLE_SIZE - 1 - RANK_POSITION_MAX_COUNT_LOG - 1) as u32;
const RANK_POSITION_DISTINCT_COUNT_CUTOFF: u32 =
    RANK_POSITION_LOG_BUCKETS_BEGIN + highbit32(RANK_POSITION_LOG_BUCKETS_BEGIN);
/// Index of the first internal node.
const START_NODE: usize = SYMBOL_VALUE_MAX as usize + 1;

/// Construction scratch: the node array with its leading sentinel slot, so
/// `node(i)` is `nodes[i + 1]` and `node(-1)` the barrier.
struct Tree {
    nodes: [Node; TREE_SLOTS],
}

/// Node slots; a power of two, so a masked index needs no bounds check.
const TREE_SLOTS: usize = 2 * (SYMBOL_VALUE_MAX as usize + 1);

impl Tree {
    // Every node index is in `-1..TREE_SLOTS - 1`, so the mask changes
    // nothing and removes the bounds check from the tree loops.
    #[inline(always)]
    fn at(&self, index: isize) -> &Node {
        &self.nodes[(index + 1) as usize & (TREE_SLOTS - 1)]
    }

    #[inline(always)]
    fn at_mut(&mut self, index: isize) -> &mut Node {
        &mut self.nodes[(index + 1) as usize & (TREE_SLOTS - 1)]
    }
}

fn bucket_index(count: u32) -> u32 {
    if count < RANK_POSITION_DISTINCT_COUNT_CUTOFF {
        count
    } else {
        highbit32(count) + RANK_POSITION_LOG_BUCKETS_BEGIN
    }
}

/// Descending insertion sort of `nodes` (`HUF_insertionSort`).
fn insertion_sort(nodes: &mut [Node]) {
    for i in 1..nodes.len() {
        let key = nodes[i];
        let mut j = i as isize - 1;
        while j >= 0 && nodes[j as usize].count < key.count {
            nodes[(j + 1) as usize] = nodes[j as usize];
            j -= 1;
        }
        nodes[(j + 1) as usize] = key;
    }
}

/// Lomuto partition around the last element (`HUF_quickSortPartition`).
fn partition(nodes: &mut [Node], low: isize, high: isize) -> isize {
    let pivot = nodes[high as usize].count;
    let mut i = low - 1;
    for j in low..high {
        if nodes[j as usize].count > pivot {
            i += 1;
            nodes.swap(i as usize, j as usize);
        }
    }
    nodes.swap((i + 1) as usize, high as usize);
    i + 1
}

/// Descending quicksort with insertion sort for short runs
/// (`HUF_simpleQuickSort`).
fn quick_sort(nodes: &mut [Node], mut low: isize, mut high: isize) {
    const INSERTION_SORT_THRESHOLD: isize = 8;
    if high - low < INSERTION_SORT_THRESHOLD {
        if high > low {
            insertion_sort(&mut nodes[low as usize..=high as usize]);
        }
        return;
    }
    while low < high {
        let idx = partition(nodes, low, high);
        if idx - low < high - idx {
            quick_sort(nodes, low, idx - 1);
            low = idx + 1;
        } else {
            quick_sort(nodes, idx + 1, high);
            high = idx - 1;
        }
    }
}

/// Sorts symbols by decreasing count into `tree` (`HUF_sort`).
fn sort(tree: &mut Tree, count: &[u32], max_symbol_value: u32) {
    let symbols = max_symbol_value as usize + 1;
    let mut base = [0u16; RANK_POSITION_TABLE_SIZE];
    let mut curr = [0u16; RANK_POSITION_TABLE_SIZE];
    for &c in &count[..symbols] {
        base[bucket_index(c) as usize] += 1;
    }
    for n in (1..RANK_POSITION_TABLE_SIZE).rev() {
        base[n - 1] += base[n];
        curr[n - 1] = base[n - 1];
    }
    for (n, &c) in count[..symbols].iter().enumerate() {
        let r = bucket_index(c) as usize + 1;
        let pos = curr[r];
        curr[r] += 1;
        let node = tree.at_mut(pos as isize);
        node.count = c;
        node.byte = n as u8;
    }
    for n in RANK_POSITION_DISTINCT_COUNT_CUTOFF as usize..RANK_POSITION_TABLE_SIZE - 1 {
        let bucket_size = curr[n] as isize - base[n] as isize;
        let start = usize::from(base[n]);
        if bucket_size > 1 {
            let nodes = &mut tree.nodes[start + 1..];
            quick_sort(nodes, 0, bucket_size - 1);
        }
    }
}

/// Builds an unlimited-height tree from the sorted nodes and returns the
/// index of the last non-zero symbol (`HUF_buildTree`).
fn build_tree(tree: &mut Tree, max_symbol_value: u32) -> isize {
    let mut non_null_rank = max_symbol_value as isize;
    while tree.at(non_null_rank).count == 0 {
        non_null_rank -= 1;
    }
    let mut low_s = non_null_rank;
    let mut node_nb = START_NODE as isize;
    let node_root = node_nb + low_s - 1;
    let mut low_n = node_nb;
    tree.at_mut(node_nb).count = tree.at(low_s).count + tree.at(low_s - 1).count;
    tree.at_mut(low_s).parent = node_nb as u16;
    tree.at_mut(low_s - 1).parent = node_nb as u16;
    node_nb += 1;
    low_s -= 2;
    for n in node_nb..=node_root {
        tree.at_mut(n).count = 1 << 30;
    }
    tree.at_mut(-1).count = 1 << 31;

    while node_nb <= node_root {
        let n1 = if tree.at(low_s).count < tree.at(low_n).count {
            low_s -= 1;
            low_s + 1
        } else {
            low_n += 1;
            low_n - 1
        };
        let n2 = if tree.at(low_s).count < tree.at(low_n).count {
            low_s -= 1;
            low_s + 1
        } else {
            low_n += 1;
            low_n - 1
        };
        tree.at_mut(node_nb).count = tree.at(n1).count + tree.at(n2).count;
        tree.at_mut(n1).parent = node_nb as u16;
        tree.at_mut(n2).parent = node_nb as u16;
        node_nb += 1;
    }

    tree.at_mut(node_root).nb_bits = 0;
    let mut n = node_root - 1;
    while n >= START_NODE as isize {
        let parent = tree.at(n).parent as isize;
        tree.at_mut(n).nb_bits = tree.at(parent).nb_bits + 1;
        n -= 1;
    }
    for n in 0..=non_null_rank {
        let parent = tree.at(n).parent as isize;
        tree.at_mut(n).nb_bits = tree.at(parent).nb_bits + 1;
    }
    non_null_rank
}

/// Limits code lengths to `target_nb_bits` (`HUF_setMaxHeight`).
fn set_max_height(tree: &mut Tree, last_non_null: isize, target_nb_bits: u32) -> u32 {
    let largest_bits = u32::from(tree.at(last_non_null).nb_bits);
    if largest_bits <= target_nb_bits {
        return largest_bits;
    }
    let mut total_cost: i32 = 0;
    let base_cost: i32 = 1 << (largest_bits - target_nb_bits);
    let mut n = last_non_null;

    while u32::from(tree.at(n).nb_bits) > target_nb_bits {
        total_cost += base_cost - (1 << (largest_bits - u32::from(tree.at(n).nb_bits)));
        tree.at_mut(n).nb_bits = target_nb_bits as u8;
        n -= 1;
    }
    while u32::from(tree.at(n).nb_bits) == target_nb_bits {
        n -= 1;
    }
    total_cost >>= largest_bits - target_nb_bits;

    const NO_SYMBOL: u32 = 0xF0F0_F0F0;
    let mut rank_last = [NO_SYMBOL; TABLE_LOG_MAX as usize + 2];
    {
        let mut current_nb_bits = target_nb_bits;
        let mut pos = n;
        while pos >= 0 {
            let bits = u32::from(tree.at(pos).nb_bits);
            if bits < current_nb_bits {
                current_nb_bits = bits;
                rank_last[(target_nb_bits - current_nb_bits) as usize] = pos as u32;
            }
            pos -= 1;
        }
    }

    while total_cost > 0 {
        let mut nb_bits_to_decrease = highbit32(total_cost as u32) + 1;
        while nb_bits_to_decrease > 1 {
            let high_pos = rank_last[nb_bits_to_decrease as usize];
            let low_pos = rank_last[nb_bits_to_decrease as usize - 1];
            if high_pos == NO_SYMBOL {
                nb_bits_to_decrease -= 1;
                continue;
            }
            if low_pos == NO_SYMBOL {
                break;
            }
            let high_total = tree.at(high_pos as isize).count;
            let low_total = 2 * tree.at(low_pos as isize).count;
            if high_total <= low_total {
                break;
            }
            nb_bits_to_decrease -= 1;
        }
        while nb_bits_to_decrease <= TABLE_LOG_MAX
            && rank_last[nb_bits_to_decrease as usize] == NO_SYMBOL
        {
            nb_bits_to_decrease += 1;
        }
        total_cost -= 1 << (nb_bits_to_decrease - 1);
        let promoted = rank_last[nb_bits_to_decrease as usize] as isize;
        tree.at_mut(promoted).nb_bits += 1;

        if rank_last[nb_bits_to_decrease as usize - 1] == NO_SYMBOL {
            rank_last[nb_bits_to_decrease as usize - 1] = rank_last[nb_bits_to_decrease as usize];
        }
        if rank_last[nb_bits_to_decrease as usize] == 0 {
            rank_last[nb_bits_to_decrease as usize] = NO_SYMBOL;
        } else {
            rank_last[nb_bits_to_decrease as usize] -= 1;
            let pos = rank_last[nb_bits_to_decrease as usize] as isize;
            if u32::from(tree.at(pos).nb_bits) != target_nb_bits - nb_bits_to_decrease {
                rank_last[nb_bits_to_decrease as usize] = NO_SYMBOL;
            }
        }
    }

    while total_cost < 0 {
        if rank_last[1] == NO_SYMBOL {
            while u32::from(tree.at(n).nb_bits) == target_nb_bits {
                n -= 1;
            }
            tree.at_mut(n + 1).nb_bits -= 1;
            rank_last[1] = (n + 1) as u32;
            total_cost += 1;
            continue;
        }
        let pos = rank_last[1] as isize + 1;
        tree.at_mut(pos).nb_bits -= 1;
        rank_last[1] += 1;
        total_cost += 1;
    }
    target_nb_bits
}

impl CTable {
    /// Assigns codes from the finished tree (`HUF_buildCTableFromTree`).
    fn fill_from_tree(
        &mut self,
        tree: &Tree,
        non_null_rank: isize,
        max_symbol_value: u32,
        max_nb_bits: u32,
    ) {
        let mut nb_per_rank = [0u16; TABLE_LOG_MAX as usize + 1];
        let mut val_per_rank = [0u16; TABLE_LOG_MAX as usize + 1];
        let alphabet_size = max_symbol_value as usize + 1;
        for n in 0..=non_null_rank {
            nb_per_rank[usize::from(tree.at(n).nb_bits)] += 1;
        }
        let mut min: u16 = 0;
        for n in (1..=max_nb_bits as usize).rev() {
            val_per_rank[n] = min;
            min += nb_per_rank[n];
            min >>= 1;
        }
        for n in 0..alphabet_size {
            let node = tree.at(n as isize);
            self.nb_bits[usize::from(node.byte)] = node.nb_bits;
        }
        for n in 0..alphabet_size {
            let bits = usize::from(self.nb_bits[n]);
            // The reference advances the rank-0 counter too, but never stores
            // a code for a symbol without a length.
            self.value[n] = if bits == 0 { 0 } else { val_per_rank[bits] };
            val_per_rank[bits] = val_per_rank[bits].wrapping_add(1);
        }
        self.table_log = max_nb_bits;
        self.max_symbol_value = max_symbol_value;
    }

    /// Builds a length-limited table for `count` (`HUF_buildCTable_wksp`).
    ///
    /// Returns the longest code length.
    ///
    /// # Errors
    ///
    /// [`CoreError::MaxSymbolValueTooLarge`] above 255 and
    /// [`CoreError::Generic`] when the limit cannot be met.
    pub(crate) fn build(
        &mut self,
        count: &[u32],
        max_symbol_value: u32,
        max_nb_bits: u32,
    ) -> Result<u32, CoreError> {
        let max_nb_bits = if max_nb_bits == 0 {
            TABLE_LOG_DEFAULT
        } else {
            max_nb_bits
        };
        if max_symbol_value > SYMBOL_VALUE_MAX {
            return Err(CoreError::MaxSymbolValueTooLarge);
        }
        let mut tree = Tree {
            nodes: [Node::default(); TREE_SLOTS],
        };
        sort(&mut tree, count, max_symbol_value);
        let non_null_rank = build_tree(&mut tree, max_symbol_value);
        let max_nb_bits = set_max_height(&mut tree, non_null_rank, max_nb_bits);
        if max_nb_bits > TABLE_LOG_MAX {
            return Err(CoreError::Generic);
        }
        self.fill_from_tree(&tree, non_null_rank, max_symbol_value, max_nb_bits);
        Ok(max_nb_bits)
    }

    /// Estimated encoded size in bytes (`HUF_estimateCompressedSize`).
    pub(crate) fn estimate_compressed_size(&self, count: &[u32], max_symbol_value: u32) -> usize {
        let bits: usize = (0..=max_symbol_value as usize)
            .map(|s| usize::from(self.nb_bits[s]) * count[s] as usize)
            .sum();
        bits >> 3
    }

    /// Code length of `symbol`, `0` past the table's largest symbol
    /// (`HUF_getNbBitsFromCTable`).
    pub(crate) fn symbol_nb_bits(&self, symbol: u32) -> u32 {
        if symbol > self.max_symbol_value {
            return 0;
        }
        self.nb_bits
            .get(symbol as usize)
            .map_or(0, |&bits| u32::from(bits))
    }

    /// Whether every present symbol has a code (`HUF_validateCTable`).
    pub(crate) fn validate(&self, count: &[u32], max_symbol_value: u32) -> bool {
        if self.max_symbol_value < max_symbol_value {
            return false;
        }
        (0..=max_symbol_value as usize).all(|s| count[s] == 0 || self.nb_bits[s] != 0)
    }

    /// Writes the table description (`HUF_writeCTable_wksp`).
    ///
    /// # Errors
    ///
    /// [`CoreError::DstSizeTooSmall`] when `dst` is too small,
    /// [`CoreError::MaxSymbolValueTooLarge`] above 255, and
    /// [`CoreError::Generic`] when raw weights cannot describe the alphabet.
    pub(crate) fn write(
        &self,
        dst: &mut [u8],
        max_symbol_value: u32,
        huff_log: u32,
    ) -> Result<usize, CoreError> {
        if max_symbol_value > SYMBOL_VALUE_MAX {
            return Err(CoreError::MaxSymbolValueTooLarge);
        }
        let mut bits_to_weight = [0u8; TABLE_LOG_MAX as usize + 1];
        for (n, weight) in bits_to_weight
            .iter_mut()
            .enumerate()
            .take(huff_log as usize + 1)
            .skip(1)
        {
            *weight = (huff_log as usize + 1 - n) as u8;
        }
        let mut weights = [0u8; SYMBOL_VALUE_MAX as usize + 1];
        for n in 0..max_symbol_value as usize {
            weights[n] = bits_to_weight[usize::from(self.nb_bits[n])];
        }

        let Some((head, rest)) = dst.split_first_mut() else {
            return Err(CoreError::DstSizeTooSmall);
        };
        let h_size = fse::compress_small(
            rest,
            &weights[..max_symbol_value as usize],
            TABLE_LOG_MAX,
            MAX_FSE_TABLE_LOG_FOR_HUFF_HEADER,
        )?;
        if h_size > 1 && h_size < max_symbol_value as usize / 2 {
            *head = h_size as u8;
            return Ok(h_size + 1);
        }

        // Raw 4-bit weights.
        if max_symbol_value > 256 - 128 {
            return Err(CoreError::Generic);
        }
        let size = (max_symbol_value as usize).div_ceil(2) + 1;
        if size > dst.len() {
            return Err(CoreError::DstSizeTooSmall);
        }
        dst[0] = (128 + (max_symbol_value - 1)) as u8;
        weights[max_symbol_value as usize] = 0;
        for n in (0..max_symbol_value as usize).step_by(2) {
            dst[n / 2 + 1] = (weights[n] << 4) + weights[n + 1];
        }
        Ok(size)
    }
}

/// Bits of one container of [`HufWriter`].
const HUF_CONTAINER_BITS: u32 = 64;

/// A table's codes packed for encoding (`HUF_CElt`): the code in the top
/// `nb_bits` bits and the length in the low byte, so one load yields both.
///
/// Every symbol is packed; only those the table describes are meaningful.
#[inline(always)]
fn packed_codes(table: &CTable) -> [u64; 256] {
    let mut codes = [0u64; 256];
    // Branch-free over the whole alphabet, so the loop vectorizes: a symbol
    // without a code packs to garbage (its shift wraps), but the input
    // holds no such symbol and it is never written.
    for ((code, &nb_bits), &value) in codes.iter_mut().zip(&table.nb_bits).zip(&table.value) {
        let nb_bits = u32::from(nb_bits);
        *code = u64::from(value).wrapping_shl(HUF_CONTAINER_BITS.wrapping_sub(nb_bits))
            | u64::from(nb_bits);
    }
    codes
}

/// The Huffman stream writer (`HUF_CStream_t`).
///
/// Two containers fill from the top: the newest code lands in the high bits
/// and older ones shift down, so a flush writes the top `bit_pos` bits as a
/// little-endian forward stream. The second container lets the loop encode
/// two groups of symbols without a dependency between them; it is merged
/// into the first before each flush. The upper bits of `bit_pos` collect
/// noise from the "fast" adds (the whole packed code is added), and only its
/// low byte is ever read, as in the reference.
struct HufWriter<'a> {
    dst: &'a mut [u8],
    /// Next byte to write.
    pos: usize,
    /// The furthest `pos` may advance: `dst.len() - 8`.
    end: usize,
    container: [u64; 2],
    bit_pos: [u64; 2],
}

impl<'a> HufWriter<'a> {
    /// `HUF_initCStream`; `None` when `dst` cannot hold one container.
    fn new(dst: &'a mut [u8]) -> Option<Self> {
        if dst.len() <= 8 {
            return None;
        }
        let end = dst.len() - 8;
        Some(Self {
            dst,
            pos: 0,
            end,
            container: [0; 2],
            bit_pos: [0; 2],
        })
    }

    /// `HUF_addBits`: with `FAST`, the length byte is left in the low bits of
    /// the container, below every bit a flush reads.
    #[inline(always)]
    fn add<const IDX: usize, const FAST: bool>(&mut self, code: u64) {
        // The shift reads the low byte; lengths stay below 64.
        self.container[IDX] = self.container[IDX].wrapping_shr(code as u32);
        self.container[IDX] |= if FAST { code } else { code & !0xFF };
        self.bit_pos[IDX] = self.bit_pos[IDX].wrapping_add(if FAST { code } else { code & 0xFF });
    }

    /// `HUF_zeroIndex1`.
    #[inline(always)]
    fn zero_second(&mut self) {
        self.container[1] = 0;
        self.bit_pos[1] = 0;
    }

    /// `HUF_mergeIndex1`.
    #[inline(always)]
    fn merge_second(&mut self) {
        self.container[0] = self.container[0].wrapping_shr((self.bit_pos[1] & 0xFF) as u32);
        self.container[0] |= self.container[1];
        self.bit_pos[0] = self.bit_pos[0].wrapping_add(self.bit_pos[1]);
    }

    /// `HUF_flushBits`: writes the pending whole bytes of the first
    /// container. With `FAST` the destination is known to hold the whole
    /// stream (`HUF_tightCompressBound`), so the position never passes the
    /// end. Otherwise, past the end the bytes land at it, as the reference's
    /// clamped position puts them, and the overflow surfaces in
    /// [`Self::close`]: the position only grows, so once past the end it
    /// stays there. Writing at the smaller of the two also lets the store
    /// go without a bounds check.
    #[inline(always)]
    fn flush<const FAST: bool>(&mut self) {
        let nb_bits = (self.bit_pos[0] & 0xFF) as u32;
        let nb_bytes = (nb_bits >> 3) as usize;
        let bits = self.container[0].wrapping_shr(HUF_CONTAINER_BITS.wrapping_sub(nb_bits));
        self.bit_pos[0] &= 7;
        let at = self.pos.min(self.end);
        if let Some(window) = self.dst.get_mut(at..at + 8) {
            window.copy_from_slice(&bits.to_le_bytes());
        }
        self.pos += nb_bytes;
        debug_assert!(!FAST || self.pos <= self.end);
    }

    /// `HUF_closeCStream`: adds the end mark and returns the stream size, or
    /// `0` when the stream did not fit.
    fn close(mut self) -> usize {
        self.add::<0, false>((1 << (HUF_CONTAINER_BITS - 1)) | 1);
        self.flush::<false>();
        let nb_bits = self.bit_pos[0] & 0xFF;
        if self.pos >= self.end {
            return 0;
        }
        self.pos + usize::from(nb_bits > 0)
    }
}

/// `HUF_compress1X_usingCTable_internal_body_loop`: encodes `src` backward
/// in groups of `UNROLL` symbols, alternating containers between groups.
/// `PAIR` is `2 * UNROLL`.
#[inline(always)]
fn encode_loop<
    const UNROLL: usize,
    const PAIR: usize,
    const FAST_FLUSH: bool,
    const LAST_FAST: bool,
>(
    writer: &mut HufWriter<'_>,
    src: &[u8],
    codes: &[u64; 256],
) {
    debug_assert_eq!(PAIR, 2 * UNROLL);
    let code = |symbol: u8| codes[usize::from(symbol)];
    let mut n = src.len();
    let rem = n % UNROLL;
    if rem > 0 {
        for &symbol in src[n - rem..].iter().rev() {
            writer.add::<0, false>(code(symbol));
        }
        writer.flush::<FAST_FLUSH>();
        n -= rem;
    }
    if !n.is_multiple_of(PAIR) {
        let group = &src[n - UNROLL..n];
        for u in 1..UNROLL {
            writer.add::<0, true>(code(group[UNROLL - u]));
        }
        writer.add::<0, LAST_FAST>(code(group[0]));
        writer.flush::<FAST_FLUSH>();
        n -= UNROLL;
    }
    let (_, pairs) = src[..n].as_rchunks::<PAIR>();
    for pair in pairs.iter().rev() {
        for u in 1..UNROLL {
            writer.add::<0, true>(code(pair[PAIR - u]));
        }
        writer.add::<0, LAST_FAST>(code(pair[UNROLL]));
        writer.flush::<FAST_FLUSH>();
        writer.zero_second();
        for u in 1..UNROLL {
            writer.add::<1, true>(code(pair[UNROLL - u]));
        }
        writer.add::<1, LAST_FAST>(code(pair[0]));
        writer.merge_second();
        writer.flush::<FAST_FLUSH>();
    }
}

/// Encodes `src` as one stream with packed codes
/// (`HUF_compress1X_usingCTable_internal_body`), compiled for one SIMD
/// level (the reference selects its BMI2 build at run time the same way).
///
/// The loop shape follows the table log as in the reference; every shape
/// writes the same bits.
#[simd]
fn compress_1x_packed<S: Simd>(
    simd: S,
    dst: &mut [u8],
    src: &[u8],
    codes: &[u64; 256],
    table_log: u32,
) -> usize {
    if dst.len() < 8 {
        return 0;
    }
    let tight_bound = ((src.len() * table_log as usize) >> 3) + 8;
    let fits = dst.len() >= tight_bound;
    let Some(mut writer) = HufWriter::new(dst) else {
        return 0;
    };
    match table_log {
        _ if !fits || table_log > 11 => {
            encode_loop::<4, 8, false, false>(&mut writer, src, codes);
        }
        11 => encode_loop::<5, 10, true, false>(&mut writer, src, codes),
        10 => encode_loop::<5, 10, true, true>(&mut writer, src, codes),
        9 => encode_loop::<6, 12, true, false>(&mut writer, src, codes),
        8 => encode_loop::<7, 14, true, false>(&mut writer, src, codes),
        7 => encode_loop::<8, 16, true, false>(&mut writer, src, codes),
        _ => encode_loop::<9, 18, true, true>(&mut writer, src, codes),
    }
    writer.close()
}

/// Encodes `src` as one stream (`HUF_compress1X_usingCTable`).
///
/// Returns `0` when the stream does not fit.
pub(crate) fn compress_1x(dst: &mut [u8], src: &[u8], table: &CTable, backend: Backend) -> usize {
    fearless_simd::dispatch!(backend.0, simd => compress_1x_on(simd, dst, src, table))
}

/// [`compress_1x`] for one SIMD level, packing the codes in its feature
/// context.
#[inline(always)]
fn compress_1x_on<S: Simd>(simd: S, dst: &mut [u8], src: &[u8], table: &CTable) -> usize {
    let codes = packed_codes(table);
    compress_1x_packed(simd, dst, src, &codes, table.table_log)
}

/// Encodes `src` as four streams behind a jump table
/// (`HUF_compress4X_usingCTable`).
///
/// Returns `0` when the streams do not fit or the input is too short.
pub(crate) fn compress_4x(dst: &mut [u8], src: &[u8], table: &CTable, backend: Backend) -> usize {
    fearless_simd::dispatch!(backend.0, simd => compress_4x_on(simd, dst, src, table))
}

/// [`compress_4x`] for one SIMD level.
#[inline(always)]
fn compress_4x_on<S: Simd>(simd: S, dst: &mut [u8], src: &[u8], table: &CTable) -> usize {
    if dst.len() < 6 + 1 + 1 + 1 + 8 {
        return 0;
    }
    if src.len() < 12 {
        return 0;
    }
    let codes = packed_codes(table);
    let segment_size = src.len().div_ceil(4);
    let mut op = 6usize;
    let mut ip = 0usize;
    for stream in 0..4 {
        let end = if stream == 3 {
            src.len()
        } else {
            ip + segment_size
        };
        let size = compress_1x_packed(simd, &mut dst[op..], &src[ip..end], &codes, table.table_log);
        if size == 0 || size > 65535 {
            return 0;
        }
        if stream < 3 {
            dst[stream * 2..stream * 2 + 2].copy_from_slice(&(size as u16).to_le_bytes());
        }
        op += size;
        ip = end;
    }
    op
}

/// Number of streams a literals section uses.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum Streams {
    /// One stream.
    Single,
    /// Four streams with a jump table.
    Four,
}

/// Encodes with a known table and rejects outputs that save nothing
/// (`HUF_compressCTable_internal`).
fn compress_with_table(
    dst: &mut [u8],
    header: usize,
    src: &[u8],
    streams: Streams,
    table: &CTable,
    backend: Backend,
) -> usize {
    let target = &mut dst[header..];
    let size = match streams {
        Streams::Single => compress_1x(target, src, table, backend),
        Streams::Four => compress_4x(target, src, table, backend),
    };
    if size == 0 {
        return 0;
    }
    let total = header + size;
    if total >= src.len() - 1 {
        return 0;
    }
    total
}

/// Number of distinct symbols present (`HUF_cardinality`).
fn cardinality(count: &[u32], max_symbol_value: u32) -> u32 {
    count[..=max_symbol_value as usize]
        .iter()
        .filter(|&&c| c != 0)
        .count() as u32
}

/// Chooses a table log (`HUF_optimalTableLog`).
///
/// Without `optimal_depth` this is the FSE heuristic; with it, every log
/// from the alphabet's minimum up is built and priced, stopping once sizes
/// grow.
pub(crate) fn optimal_table_log(
    max_table_log: u32,
    src_size: usize,
    max_symbol_value: u32,
    count: &[u32],
    flags: Flags,
) -> u32 {
    debug_assert!(src_size > 1);
    if !flags.optimal_depth {
        return fse::optimal_table_log_internal(max_table_log, src_size, max_symbol_value, 1);
    }
    let symbol_cardinality = cardinality(count, max_symbol_value);
    let min_table_log = highbit32(symbol_cardinality) + 1;
    if min_table_log > max_table_log {
        return max_table_log;
    }
    let mut tree = Tree {
        nodes: [Node::default(); TREE_SLOTS],
    };
    sort(&mut tree, count, max_symbol_value);
    let non_null_rank = build_tree(&mut tree, max_symbol_value);
    let mut opt_size = usize::MAX - 1;
    let mut opt_log = max_table_log;
    let mut table = CTable::default();
    let mut header = [0u8; HEADER_TRIAL_CAPACITY];
    for guess in min_table_log..=max_table_log {
        // Height limiting changes only the leaves' lengths. The sorted
        // counts, parents and internal-node heights remain the unlimited
        // tree, so restore the leaf heights instead of rebuilding it.
        if guess != min_table_log {
            for n in 0..=non_null_rank {
                let parent = tree.at(n).parent as isize;
                tree.at_mut(n).nb_bits = tree.at(parent).nb_bits + 1;
            }
        }
        let max_bits = set_max_height(&mut tree, non_null_rank, guess);
        if max_bits > TABLE_LOG_MAX {
            continue;
        }
        if max_bits < guess && guess > min_table_log {
            break;
        }
        table.fill_from_tree(&tree, non_null_rank, max_symbol_value, max_bits);
        let Ok(h_size) = table.write(&mut header, max_symbol_value, max_bits) else {
            continue;
        };
        let new_size = table.estimate_compressed_size(count, max_symbol_value) + h_size;
        if new_size > opt_size + 1 {
            break;
        }
        if new_size < opt_size {
            opt_size = new_size;
            opt_log = guess;
        }
    }
    opt_log
}

/// What a literals compression asks of the Huffman encoder.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct Options {
    /// Largest symbol to expect; `0` means 255.
    pub(crate) max_symbol_value: u32,
    /// Largest table log; `0` means the default.
    pub(crate) huff_log: u32,
    /// Stream layout.
    pub(crate) streams: Streams,
    /// Reference flags.
    pub(crate) flags: Flags,
    /// The SIMD level the stream encoder runs with.
    pub(crate) backend: Backend,
}

/// Compresses `src`, possibly reusing `old_table` (`HUF_compress_internal`
/// through `HUF_compress{1,4}X_repeat`).
///
/// Returns `0` when the input is not compressible, `1` with the single
/// symbol in `dst[0]` when it is one repeated byte, and otherwise the size
/// of the table description plus the streams. `repeat` and `old_table` are
/// updated as in the reference: a fresh table replaces `old_table` and sets
/// `repeat` to [`Repeat::None`].
///
/// # Errors
///
/// [`CoreError::SrcSizeWrong`] above a block, and whatever histogram, tree
/// construction or header writing reports.
pub(crate) fn compress(
    dst: &mut [u8],
    src: &[u8],
    options: Options,
    old_table: &mut CTable,
    repeat: &mut Repeat,
) -> Result<usize, CoreError> {
    let Options {
        max_symbol_value,
        huff_log,
        streams,
        flags,
        backend,
    } = options;
    if src.is_empty() || dst.is_empty() {
        return Ok(0);
    }
    if src.len() > BLOCK_SIZE_MAX {
        return Err(CoreError::SrcSizeWrong);
    }
    if huff_log > TABLE_LOG_MAX {
        return Err(CoreError::TableLogTooLarge);
    }
    if max_symbol_value > SYMBOL_VALUE_MAX {
        return Err(CoreError::MaxSymbolValueTooLarge);
    }
    let mut max_symbol_value = if max_symbol_value == 0 {
        SYMBOL_VALUE_MAX
    } else {
        max_symbol_value
    };
    let huff_log = if huff_log == 0 {
        TABLE_LOG_DEFAULT
    } else {
        huff_log
    };

    if flags.prefer_repeat && *repeat == Repeat::Valid {
        return Ok(compress_with_table(
            dst, 0, src, streams, old_table, backend,
        ));
    }

    let mut count = [0u32; 256];
    if flags.suspect_uncompressible
        && src.len() >= SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE * SUSPECT_INCOMPRESSIBLE_SAMPLE_RATIO
    {
        let mut largest_total = 0usize;
        let mut max_begin = max_symbol_value;
        largest_total += hist::count(
            &mut count,
            &mut max_begin,
            &src[..SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE],
            false,
        )? as usize;
        let mut max_end = max_symbol_value;
        largest_total += hist::count(
            &mut count,
            &mut max_end,
            &src[src.len() - SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE..],
            false,
        )? as usize;
        if largest_total <= ((2 * SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE) >> 7) + 4 {
            return Ok(0);
        }
    }

    let check = max_symbol_value < 255;
    let largest = hist::count(&mut count, &mut max_symbol_value, src, check)?;
    if largest as usize == src.len() {
        dst[0] = src[0];
        return Ok(1);
    }
    if largest as usize <= (src.len() >> 7) + 4 {
        return Ok(0);
    }

    if *repeat == Repeat::Check && !old_table.validate(&count, max_symbol_value) {
        *repeat = Repeat::None;
    }
    if flags.prefer_repeat && *repeat != Repeat::None {
        return Ok(compress_with_table(
            dst, 0, src, streams, old_table, backend,
        ));
    }

    let huff_log = optimal_table_log(huff_log, src.len(), max_symbol_value, &count, flags);
    let mut table = CTable::default();
    let huff_log = table.build(&count, max_symbol_value, huff_log)?;

    let h_size = table.write(dst, max_symbol_value, huff_log)?;
    if *repeat != Repeat::None {
        let old_size = old_table.estimate_compressed_size(&count, max_symbol_value);
        let new_size = table.estimate_compressed_size(&count, max_symbol_value);
        if old_size <= h_size + new_size || h_size + 12 >= src.len() {
            return Ok(compress_with_table(
                dst, 0, src, streams, old_table, backend,
            ));
        }
    }
    if h_size + 12 >= src.len() {
        return Ok(0);
    }
    *repeat = Repeat::None;
    *old_table = table;
    Ok(compress_with_table(
        dst, h_size, src, streams, old_table, backend,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn literal_histograms() -> alloc::vec::Vec<alloc::vec::Vec<u32>> {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut out = alloc::vec::Vec::new();
        for alphabet in [2usize, 3, 17, 64, 128, 129, 200, 256] {
            for shape in 0..30 {
                let mut count = alloc::vec![0u32; alphabet];
                for c in count.iter_mut() {
                    let r = next();
                    *c = match shape % 5 {
                        0 => (r % 300) as u32,
                        1 => 1 << (r % 15),
                        2 => u32::from(r % 4 == 0) * (r % 9) as u32,
                        3 => 165 + (r % 3) as u32,
                        _ => (r % 2) as u32 * 40_000,
                    };
                }
                count[0] = count[0].max(2);
                count[alphabet - 1] = count[alphabet - 1].max(1);
                out.push(count);
            }
        }
        out
    }

    #[test]
    fn tables_match_the_reference() {
        for count in literal_histograms() {
            let max = count.len() as u32 - 1;
            // The encoder never asks for fewer bits than the alphabet needs.
            let floor = highbit32(cardinality(&count, max)) + 1;
            for max_bits in [6, 8, 11, 12].into_iter().filter(|&bits| bits >= floor) {
                let mut table = CTable::default();
                let ours = table.build(&count, max, max_bits);
                let reference = zstd_ffi::oracle::huf_build(&count, max_bits);
                match (ours, reference) {
                    (Ok(log), Ok(expected)) => {
                        assert_eq!(log, expected.table_log, "{count:?}");
                        assert_eq!(
                            &table.nb_bits[..count.len()],
                            &expected.nb_bits[..],
                            "{count:?}"
                        );
                        for s in 0..count.len() {
                            if expected.nb_bits[s] != 0 {
                                assert_eq!(
                                    table.value[s], expected.values[s],
                                    "{count:?} symbol {s}"
                                );
                            }
                        }
                    }
                    (Err(_), Err(_)) => {}
                    (ours, reference) => panic!("{count:?}: {ours:?} vs {reference:?}"),
                }
            }
        }
    }

    #[test]
    fn table_log_choice_matches_the_reference() {
        for count in literal_histograms() {
            let max = count.len() as u32 - 1;
            let total: usize = count.iter().map(|&c| c as usize).sum();
            for depth in [false, true] {
                let flags = Flags {
                    optimal_depth: depth,
                    ..Flags::default()
                };
                for max_log in [5, 6, 8, 11, 12] {
                    assert_eq!(
                        optimal_table_log(max_log, total, max, &count, flags),
                        zstd_ffi::oracle::huf_optimal_table_log(max_log, total, &count, depth),
                        "{count:?} depth {depth} max_log {max_log}"
                    );
                }
            }
        }
    }

    /// The straightforward one-symbol-at-a-time encoder the packed loops
    /// must reproduce bit for bit.
    fn compress_1x_baseline(dst: &mut [u8], src: &[u8], table: &CTable) -> usize {
        if dst.len() < 8 {
            return 0;
        }
        let Some(mut writer) = super::super::bits::BitWriter::new(dst) else {
            return 0;
        };
        let mut pending = 0u32;
        for &symbol in src.iter().rev() {
            let s = usize::from(symbol);
            let bits = u32::from(table.nb_bits[s]);
            writer.add_bits_fast(u64::from(table.value[s]), bits);
            pending += bits;
            if pending > 32 {
                writer.flush();
                pending = 0;
            }
        }
        writer.close()
    }

    #[test]
    fn packed_streams_match_the_baseline_encoder() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for alphabet in [2u64, 5, 40, 256] {
            for skew in 0..4u64 {
                let src: alloc::vec::Vec<u8> = (0..3000)
                    .map(|_| {
                        let r = next();
                        ((r % alphabet) >> (skew * (r >> 61))) as u8
                    })
                    .collect();
                let mut count = [0u32; 256];
                for &b in &src {
                    count[usize::from(b)] += 1;
                }
                // The encoder trims the alphabet and never asks for fewer
                // bits than it needs.
                let max = count.iter().rposition(|&c| c != 0).unwrap_or(0) as u32;
                let floor = highbit32(cardinality(&count, max)) + 1;
                for max_bits in floor.max(6)..=12 {
                    let mut table = CTable::default();
                    if table.build(&count, max, max_bits).is_err() {
                        continue;
                    }
                    for len in [0, 1, 4, 5, 9, 10, 17, 18, 19, 100, 1000, 3000] {
                        let src = &src[..len];
                        // Room for the tight bound, just below it, and too little.
                        let tight = ((len * table.table_log as usize) >> 3) + 8;
                        for capacity in [tight + 64, tight, tight - 1, 9, 8, 7] {
                            let mut theirs = alloc::vec![0u8; capacity];
                            let b = compress_1x_baseline(&mut theirs, src, &table);
                            for backend in Backend::available() {
                                let mut ours = alloc::vec![0u8; capacity];
                                let a = compress_1x(&mut ours, src, &table, backend);
                                assert_eq!(a, b, "len {len} cap {capacity} {backend}");
                                assert_eq!(ours[..a], theirs[..b]);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn two_symbols_get_one_bit_codes() {
        let mut table = CTable::default();
        let mut count = [0u32; 256];
        count[0] = 5;
        count[1] = 3;
        assert_eq!(table.build(&count, 1, 11), Ok(1));
        assert_eq!(table.nb_bits[0], 1);
        assert_eq!(table.nb_bits[1], 1);
        assert_ne!(table.value[0], table.value[1]);
    }

    #[test]
    fn code_lengths_past_the_largest_symbol_are_zero() {
        let mut table = CTable::default();
        let mut count = [0u32; 256];
        count[..3].copy_from_slice(&[4, 2, 1]);
        table.build(&count, 2, 11).expect("table");
        for symbol in 0..3 {
            assert_eq!(
                table.symbol_nb_bits(symbol),
                u32::from(table.nb_bits[symbol as usize])
            );
            assert_ne!(table.symbol_nb_bits(symbol), 0);
        }
        assert_eq!(table.symbol_nb_bits(3), 0);
        assert_eq!(table.symbol_nb_bits(255), 0);
        assert_eq!(table.symbol_nb_bits(4096), 0);
    }

    #[test]
    fn validation_requires_a_code_for_every_present_symbol() {
        let mut table = CTable::default();
        let mut count = [0u32; 256];
        count[..3].copy_from_slice(&[4, 2, 1]);
        table.build(&count, 2, 11).expect("table");
        assert!(table.validate(&count, 2));
        count[3] = 1;
        assert!(!table.validate(&count, 3));
    }

    #[test]
    fn tiny_or_empty_inputs_are_not_compressed() {
        let mut table = CTable::default();
        let mut repeat = Repeat::None;
        let mut dst = [0u8; 64];
        let options = Options {
            max_symbol_value: 255,
            huff_log: 11,
            streams: Streams::Single,
            flags: Flags::default(),
            backend: Backend::SCALAR,
        };
        let result = compress(&mut dst, &[], options, &mut table, &mut repeat);
        assert_eq!(result, Ok(0));
        let result = compress(&mut dst, &[9; 40], options, &mut table, &mut repeat);
        assert_eq!(result, Ok(1));
        assert_eq!(dst[0], 9);
        assert_eq!(compress_4x(&mut dst, &[1; 8], &table, Backend::SCALAR), 0);
        assert_eq!(
            compress_1x(&mut dst[..7], &[1; 8], &table, Backend::SCALAR),
            0
        );
    }
}

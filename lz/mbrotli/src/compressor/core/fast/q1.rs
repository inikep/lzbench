//! Quality 1: the fast two-pass encoder.
//!
//! Direct port of `BrotliCompressFragmentTwoPass` from
//! `c/enc/compress_fragment_two_pass.c` of the pinned reference
//! (`google/brotli` v1.2.0, commit `028fb5a`). The first pass stores commands
//! and literals in compact buffers, the second builds exact histograms from
//! them and writes the meta-block.
//!
//! # Read window
//!
//! As in quality 0, the scan reads short words rather than single bytes:
//!
//! * `ip_limit = input + min(block_size - MIN_MATCH, input_size - 16)` and
//!   `input + input_size == data.len()`, so `ip_limit + 16 <= data.len()`.
//! * `ip` never exceeds `ip_limit`, and a candidate never exceeds `ip + 1`
//!   (the repeat candidate with the initial `last_distance` of `-1`).
//! * Post-copy hash updates happen only while `ip < ip_limit`, and read at
//!   most from `ip - 5` through `ip + 6`. `ip` is at least `input + MIN_MATCH`
//!   at those points, so the lower end never underflows.

use alloc::boxed::Box;
use alloc::vec::Vec;

use fearless_simd::Simd;

use super::bits::{BitWriter, ByteBuffer};
use super::commands::one_pass::pack_literals;
use super::commands::two_pass::{
    emit_copy_len, emit_copy_len_last_distance, emit_distance, emit_insert_len,
};
use super::commands::{fast_log2, log2_floor_non_zero, store_meta_block_header};
use super::constants::{
    BLOCK_SAMPLE_RATE, HASH_MUL32, MAX_BACKWARD_DISTANCE, MIN_HASH_ENTRIES, NUM_COMMAND_SYMBOLS,
    NUM_LITERAL_SYMBOLS, Q1_BLOCK_SIZE, Q1_MAX_HASH_ENTRIES, Q1_MIN_MATCH_LARGE,
    Q1_MIN_MATCH_SMALL, Q1_MIN_RATIO, WINDOW_GAP,
};
use super::histogram;
use super::huffman::{
    HuffmanNode, build_and_store_huffman_tree_fast, convert_bit_depths_to_symbols,
    create_huffman_tree, store_huffman_tree,
};
use super::match_len::{current_window, load_u64_le, match_len_at};
use super::tables::{INSERT_OFFSET, NUM_EXTRA_BITS};
use super::workspace::TwoPassArena;

/// Hash table widths quality 1 is compiled for.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum TableBits {
    /// 256 entries.
    B8,
    /// 512 entries.
    B9,
    /// 1024 entries.
    B10,
    /// 2048 entries.
    B11,
    /// 4096 entries.
    B12,
    /// 8192 entries.
    B13,
    /// 16384 entries.
    B14,
    /// 32768 entries.
    B15,
    /// 65536 entries.
    B16,
    /// 131072 entries.
    B17,
}

impl TableBits {
    /// Selects the table width the reference would use for `input_size` bytes.
    pub(crate) fn for_input(input_size: usize) -> Self {
        let mut entries = MIN_HASH_ENTRIES;
        while entries < Q1_MAX_HASH_ENTRIES && entries < input_size {
            entries <<= 1;
        }
        match log2_floor_non_zero(entries) {
            8 => Self::B8,
            9 => Self::B9,
            10 => Self::B10,
            11 => Self::B11,
            12 => Self::B12,
            13 => Self::B13,
            14 => Self::B14,
            15 => Self::B15,
            16 => Self::B16,
            _ => Self::B17,
        }
    }

    /// Returns the number of entries of a table of this width.
    pub(crate) const fn entries(self) -> usize {
        1 << self.bits()
    }

    /// Returns the base-2 logarithm of the table width.
    pub(crate) const fn bits(self) -> usize {
        match self {
            Self::B8 => 8,
            Self::B9 => 9,
            Self::B10 => 10,
            Self::B11 => 11,
            Self::B12 => 12,
            Self::B13 => 13,
            Self::B14 => 14,
            Self::B15 => 15,
            Self::B16 => 16,
            Self::B17 => 17,
        }
    }

    /// Returns the fixed match length probed at this table width.
    pub(crate) const fn min_match(self) -> usize {
        if self.bits() <= 15 {
            Q1_MIN_MATCH_SMALL
        } else {
            Q1_MIN_MATCH_LARGE
        }
    }
}

/// Hashes the `MIN_MATCH` bytes at `position` into a table index.
#[cfg(test)]
fn hash<const TABLE_BITS: usize, const MIN_MATCH: usize>(data: &[u8], position: usize) -> usize {
    let word = load_u64_le(data, position);
    let mixed = (word << ((8 - MIN_MATCH) * 8)).wrapping_mul(HASH_MUL32 as u64);
    (mixed >> (64 - TABLE_BITS)) as usize
}

/// Hashes the bytes starting at `offset` of an already loaded machine word.
#[inline(always)]
fn hash_bytes_at_offset<const TABLE_BITS: usize, const MIN_MATCH: usize>(
    word: u64,
    offset: u32,
) -> usize {
    debug_assert!(offset as usize <= 8 - MIN_MATCH);
    let mixed = ((word >> (8 * offset)) << ((8 - MIN_MATCH) * 8)).wrapping_mul(HASH_MUL32 as u64);
    (mixed >> (64 - TABLE_BITS)) as usize
}

/// Tests the fixed `MIN_MATCH`-byte match predicate at two positions.
///
/// Both positions sit inside the read window described at the top of this
/// module, so a single word load per side is enough; masking the difference to
/// `MIN_MATCH` bytes is equivalent to the reference's staged comparison. Fewer
/// than eight readable bytes on either side reports no match, which the scan
/// never relies on.
#[inline(always)]
fn is_match<const MIN_MATCH: usize>(data: &[u8], left: usize, right: usize) -> bool {
    words_match::<MIN_MATCH>(load_u64_le(data, left), load_u64_le(data, right))
}

/// Tests the `MIN_MATCH`-byte match predicate on two already loaded words.
#[inline(always)]
const fn words_match<const MIN_MATCH: usize>(left: u64, right: u64) -> bool {
    let mask = (1u64 << (8 * MIN_MATCH)) - 1;
    (left ^ right) & mask == 0
}

/// Refreshes the hash table for the positions inside the copy just emitted.
///
/// Returns the hash slot of the current position, which the caller uses to look
/// up the next candidate.
///
/// The `min_match == 4` branch of the first update path in the pinned reference
/// hashes offsets `0, 1, 0` where the chained path uses `0, 1, 2`. That
/// asymmetry changes the command stream, so `FIRST_UPDATE` reproduces it rather
/// than "fixing" it.
#[inline(always)]
fn update_hashes_after_copy<
    const TABLE_BITS: usize,
    const ENTRIES: usize,
    const MIN_MATCH: usize,
    const FIRST_UPDATE: bool,
>(
    data: &[u8],
    table: &mut [i32; ENTRIES],
    ip: usize,
) -> usize {
    if MIN_MATCH == Q1_MIN_MATCH_SMALL {
        let word = load_u64_le(data, ip - 3);
        let current = hash_bytes_at_offset::<TABLE_BITS, MIN_MATCH>(word, 3);
        table[hash_bytes_at_offset::<TABLE_BITS, MIN_MATCH>(word, 0)] = (ip - 3) as i32;
        table[hash_bytes_at_offset::<TABLE_BITS, MIN_MATCH>(word, 1)] = (ip - 2) as i32;
        let last_offset = if FIRST_UPDATE { 0 } else { 2 };
        table[hash_bytes_at_offset::<TABLE_BITS, MIN_MATCH>(word, last_offset)] = (ip - 1) as i32;
        return current;
    }
    let word = load_u64_le(data, ip - 5);
    table[hash_bytes_at_offset::<TABLE_BITS, MIN_MATCH>(word, 0)] = (ip - 5) as i32;
    table[hash_bytes_at_offset::<TABLE_BITS, MIN_MATCH>(word, 1)] = (ip - 4) as i32;
    table[hash_bytes_at_offset::<TABLE_BITS, MIN_MATCH>(word, 2)] = (ip - 3) as i32;
    let word = load_u64_le(data, ip - 2);
    let current = hash_bytes_at_offset::<TABLE_BITS, MIN_MATCH>(word, 2);
    table[hash_bytes_at_offset::<TABLE_BITS, MIN_MATCH>(word, 0)] = (ip - 2) as i32;
    table[hash_bytes_at_offset::<TABLE_BITS, MIN_MATCH>(word, 1)] = (ip - 1) as i32;
    current
}

/// One block of a fragment, together with the tail that follows it.
struct Block<'a> {
    /// The whole fragment; positions are indices into it.
    data: &'a [u8],
    /// Offset of the block inside the fragment.
    input: usize,
    /// Length of the block.
    block_size: usize,
    /// Bytes remaining in the fragment from `input` onwards.
    input_size: usize,
}

/// Output of the first pass.
struct Pass1<'a> {
    /// Literal bytes, in emission order.
    literals: &'a mut Vec<u8>,
    /// Packed command words.
    commands: &'a mut Vec<u32>,
}

/// First pass: finds matches and records commands and literals.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
fn create_commands<
    S: Simd,
    const TABLE_BITS: usize,
    const ENTRIES: usize,
    const MIN_MATCH: usize,
    const INDEPENDENT: bool,
>(
    simd: S,
    block: &Block<'_>,
    table: &mut [i32; ENTRIES],
    out: &mut Pass1<'_>,
) {
    // Specialize the complete scan inside the selected SIMD feature context.
    simd.vectorize(
        #[inline(always)]
        || {
            let Block {
                data,
                input,
                block_size,
                input_size,
            } = *block;
            // The buffers are owned by locals for the duration of the scan:
            // behind the caller's references their headers live in memory the
            // compiler cannot tell apart from the table, so every push would
            // reload the length and capacity after each table store.
            let mut literal_buffer = core::mem::take(out.literals);
            let mut command_buffer = core::mem::take(out.commands);
            let (literals, commands) = (&mut literal_buffer, &mut command_buffer);
            let mut ip = input;
            let ip_end = input + block_size;
            let mut next_emit = input;
            let mut last_distance: i64 = -1;

            'scan: {
                if block_size < WINDOW_GAP {
                    break 'scan;
                }
                let len_limit = (block_size - MIN_MATCH).min(input_size - WINDOW_GAP);
                let ip_limit = input + len_limit;

                ip += 1;

                // The word at the next position is loaded once: it is hashed here and

                // compared against the candidates when the scan reaches it.

                let mut next_word = load_u64_le(data, ip);

                let mut next_hash = hash_bytes_at_offset::<TABLE_BITS, MIN_MATCH>(next_word, 0);
                loop {
                    let mut skip = 32u32;
                    let mut next_ip = ip;
                    let mut candidate;

                    'found: {
                        loop {
                            loop {
                                let slot = next_hash;
                                let ip_word = next_word;
                                let stride = (skip >> 5) as usize;
                                skip = skip.wrapping_add(1);
                                ip = next_ip;
                                next_ip = ip + stride;
                                if next_ip > ip_limit {
                                    break 'scan;
                                }
                                next_word = load_u64_le(data, next_ip);
                                next_hash =
                                    hash_bytes_at_offset::<TABLE_BITS, MIN_MATCH>(next_word, 0);

                                // The reference subtracts the last distance without
                                // a sign test and rejects a position at or past `ip`
                                // after comparing; the two tests are pure, so
                                // rejecting first reads nothing for the initial
                                // distance of minus one.
                                let repeated = (ip as i64 - last_distance) as usize;
                                if repeated < ip
                                    && words_match::<MIN_MATCH>(
                                        ip_word,
                                        load_u64_le(data, repeated),
                                    )
                                {
                                    table[slot] = ip as i32;
                                    candidate = repeated;
                                    break;
                                }

                                candidate = table[slot] as usize;
                                table[slot] = ip as i32;
                                if words_match::<MIN_MATCH>(ip_word, load_u64_le(data, candidate)) {
                                    break;
                                }
                            }
                            if ip - candidate > MAX_BACKWARD_DISTANCE {
                                continue;
                            }
                            break 'found;
                        }
                    }

                    let base = ip;
                    let matched = MIN_MATCH
                        + match_len_at(
                            simd,
                            data,
                            candidate + MIN_MATCH,
                            current_window(data, ip + MIN_MATCH, (ip_end - ip) - MIN_MATCH),
                        );
                    let distance = (base - candidate) as i64;
                    let insert = base - next_emit;
                    ip += matched;

                    emit_insert_len(insert, commands);
                    if let Some(block) = data.get(next_emit..next_emit + insert) {
                        literals.extend_from_slice(block);
                    }
                    if !INDEPENDENT && distance == last_distance {
                        commands.push(64);
                    } else {
                        emit_distance(distance as usize, commands);
                        last_distance = distance;
                    }
                    if INDEPENDENT {
                        emit_copy_len(matched - 2, commands);
                        emit_distance(distance as usize, commands);
                    } else {
                        emit_copy_len_last_distance(matched, commands);
                    }

                    next_emit = ip;
                    if ip >= ip_limit {
                        break 'scan;
                    }
                    candidate = {
                        let current =
                            update_hashes_after_copy::<TABLE_BITS, ENTRIES, MIN_MATCH, true>(
                                data, table, ip,
                            );
                        let candidate = table[current] as usize;
                        table[current] = ip as i32;
                        candidate
                    };

                    while ip - candidate <= MAX_BACKWARD_DISTANCE
                        && is_match::<MIN_MATCH>(data, ip, candidate)
                    {
                        let base = ip;
                        let matched = MIN_MATCH
                            + match_len_at(
                                simd,
                                data,
                                candidate + MIN_MATCH,
                                current_window(data, ip + MIN_MATCH, (ip_end - ip) - MIN_MATCH),
                            );
                        ip += matched;
                        last_distance = (base - candidate) as i64;
                        emit_copy_len(matched, commands);
                        emit_distance(last_distance as usize, commands);

                        next_emit = ip;
                        if ip >= ip_limit {
                            break 'scan;
                        }
                        let current =
                            update_hashes_after_copy::<TABLE_BITS, ENTRIES, MIN_MATCH, false>(
                                data, table, ip,
                            );
                        candidate = table[current] as usize;
                        table[current] = ip as i32;
                    }

                    ip += 1;

                    next_word = load_u64_le(data, ip);

                    next_hash = hash_bytes_at_offset::<TABLE_BITS, MIN_MATCH>(next_word, 0);
                }
            }

            debug_assert!(next_emit <= ip_end);
            if next_emit < ip_end {
                let insert = ip_end - next_emit;
                emit_insert_len(insert, commands);
                if let Some(block) = data.get(next_emit..ip_end) {
                    literals.extend_from_slice(block);
                }
            }
            *out.literals = literal_buffer;
            *out.commands = command_buffer;
        },
    );
}

/// Builds the command and distance prefix codes and stores them.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
fn build_and_store_command_prefix_code<const INDEPENDENT: bool>(
    histogram: &[u32; 128],
    depth: &mut [u8; 128],
    bits: &mut [u16; 128],
    tmp_depth: &mut [u8; NUM_COMMAND_SYMBOLS],
    tmp_bits: &mut [u16; 64],
    tree: &mut Vec<HuffmanNode>,
    w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
) {
    tmp_depth.fill(0);
    create_huffman_tree(&histogram[..64], 64, 15, tree, &mut depth[..64]);
    create_huffman_tree(&histogram[64..], 64, 14, tree, &mut depth[64..]);

    tmp_depth[..24].copy_from_slice(&depth[24..48]);
    tmp_depth[24..32].copy_from_slice(&depth[..8]);
    tmp_depth[32..40].copy_from_slice(&depth[48..56]);
    tmp_depth[40..48].copy_from_slice(&depth[8..16]);
    tmp_depth[48..56].copy_from_slice(&depth[56..64]);
    tmp_depth[56..64].copy_from_slice(&depth[16..24]);
    convert_bit_depths_to_symbols(tmp_depth, 64, tmp_bits);
    bits[..8].copy_from_slice(&tmp_bits[24..32]);
    bits[8..16].copy_from_slice(&tmp_bits[40..48]);
    bits[16..24].copy_from_slice(&tmp_bits[56..64]);
    bits[24..48].copy_from_slice(&tmp_bits[..24]);
    bits[48..56].copy_from_slice(&tmp_bits[32..40]);
    bits[56..64].copy_from_slice(&tmp_bits[48..56]);
    convert_bit_depths_to_symbols(&depth[64..], 64, &mut bits[64..]);

    tmp_depth[..64].fill(0);
    tmp_depth[..8].copy_from_slice(&depth[24..32]);
    tmp_depth[64..72].copy_from_slice(&depth[32..40]);
    tmp_depth[128..136].copy_from_slice(&depth[40..48]);
    tmp_depth[192..200].copy_from_slice(&depth[48..56]);
    tmp_depth[384..392].copy_from_slice(&depth[56..64]);
    for i in 0..8 {
        // Independent fragments use explicit copy-two (compact symbol 40).
        // Keep its depth at wire symbol 128: the unused insert-zero alias
        // would overwrite it and invalidate canonical ordering of short copies.
        if !INDEPENDENT || i != 0 {
            tmp_depth[128 + 8 * i] = depth[i];
        }
        tmp_depth[256 + 8 * i] = depth[8 + i];
        tmp_depth[448 + 8 * i] = depth[16 + i];
    }
    store_huffman_tree(tmp_depth, NUM_COMMAND_SYMBOLS, tree, w);
    store_huffman_tree(&depth[64..], 64, tree, w);
}

/// Second pass: builds exact prefix codes and replays the buffered commands.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
fn store_commands<const INDEPENDENT: bool>(
    arena: &mut TwoPassArena,
    literals: &[u8],
    commands: &[u32],
    w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
) {
    let TwoPassArena {
        lit_histo,
        lit_depth,
        lit_bits,
        cmd_histo,
        cmd_depth,
        cmd_bits,
        tmp_tree,
        tmp_depth,
        tmp_bits,
    } = arena;

    lit_histo.fill(0);
    cmd_depth.fill(0);
    cmd_bits.fill(0);
    cmd_histo.fill(0);

    histogram::accumulate(literals, lit_histo);
    build_and_store_huffman_tree_fast(
        tmp_tree,
        lit_histo,
        literals.len(),
        8,
        lit_depth,
        lit_bits,
        w,
    );

    for &command in commands {
        // Every code the first pass emits is below 128; masking says so to the
        // compiler and removes the bounds check from this counting loop.
        cmd_histo[(command & 0x7F) as usize] += 1;
    }
    cmd_histo[1] += 1;
    cmd_histo[2] += 1;
    cmd_histo[64] += 1;
    cmd_histo[84] += 1;
    build_and_store_command_prefix_code::<INDEPENDENT>(
        cmd_histo, cmd_depth, cmd_bits, tmp_depth, tmp_bits, tmp_tree, w,
    );

    let mut literal_index = 0usize;
    for &command in commands {
        debug_assert!(command & 0xFF < 128);
        let code = (command & 0x7F) as usize;
        let extra = command >> 8;
        // Code and extra bits fit in one call: at most fifteen plus
        // twenty-four bits, against the writer's limit of fifty-six.
        let width = u32::from(cmd_depth[code]);
        w.write(
            width + NUM_EXTRA_BITS[code],
            u64::from(cmd_bits[code]) | (u64::from(extra) << width),
        );
        if code < 24 {
            let insert = (INSERT_OFFSET[code] + extra) as usize;
            let Some(block) = literals.get(literal_index..literal_index + insert) else {
                return;
            };
            pack_literals(block, lit_depth, lit_bits, w);
            literal_index += insert;
        }
    }
    debug_assert_eq!(literal_index, literals.len());
}

/// Reference entropy estimate in bits (`BrotliBitsEntropy`).
fn bits_entropy(population: &[u32]) -> f64 {
    let mut sum = 0usize;
    let mut retval = 0f64;
    for &count in population {
        sum += count as usize;
        retval -= f64::from(count) * fast_log2(count as usize);
    }
    if sum != 0 {
        retval += sum as f64 * fast_log2(sum);
    }
    if retval < sum as f64 {
        // At least one bit per literal is needed.
        retval = sum as f64;
    }
    retval
}

/// Decides whether the block is worth compressing at all.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
fn should_compress(
    histogram: &mut [u32; NUM_LITERAL_SYMBOLS],
    input: &[u8],
    num_literals: usize,
) -> bool {
    let corpus_size = input.len() as f64;
    if (num_literals as f64) < Q1_MIN_RATIO * corpus_size {
        return true;
    }
    let max_total_bit_cost = corpus_size * 8.0 * Q1_MIN_RATIO / BLOCK_SAMPLE_RATE as f64;
    histogram.fill(0);
    histogram::accumulate_sampled(input, BLOCK_SAMPLE_RATE, histogram);
    bits_entropy(histogram) < max_total_bit_cost
}

/// Writes `input` as an uncompressed meta-block at the current position.
fn emit_uncompressed_meta_block(input: &[u8], w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>) {
    store_meta_block_header(input.len(), true, w);
    w.align();
    w.write_bytes(input);
}

/// Everything the two-pass encoder reuses between fragments.
pub(crate) struct TwoPassState {
    /// Histograms, prefix codes and node pool.
    pub(crate) arena: Box<TwoPassArena>,
    /// Packed commands produced by the first pass.
    pub(crate) commands: Vec<u32>,
    /// Literal bytes produced by the first pass.
    pub(crate) literals: Vec<u8>,
}

impl Default for TwoPassState {
    /// Creates state with buffers sized for the largest block.
    fn default() -> Self {
        Self {
            arena: Box::default(),
            commands: Vec::with_capacity(Q1_BLOCK_SIZE),
            literals: Vec::with_capacity(Q1_BLOCK_SIZE),
        }
    }
}

impl TwoPassState {
    /// Restores the state [`TwoPassState::default`] would produce.
    ///
    /// The arena is assigned through its `Box` and the two buffers are cleared
    /// rather than dropped, so every allocation survives into the next stream.
    pub(crate) fn reset(&mut self) {
        self.arena.reset();
        self.commands.clear();
        self.literals.clear();
    }
}

/// Compresses one fragment with the table width and match length baked in.
fn compress_fragment_impl<
    S: Simd,
    const TABLE_BITS: usize,
    const ENTRIES: usize,
    const MIN_MATCH: usize,
    const INDEPENDENT: bool,
>(
    simd: S,
    state: &mut TwoPassState,
    data: &[u8],
    table: &mut [i32; ENTRIES],
    w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
) {
    let TwoPassState {
        arena,
        commands,
        literals,
    } = state;
    let mut input = 0usize;
    let mut input_size = data.len();

    while input_size > 0 {
        let block_size = input_size.min(Q1_BLOCK_SIZE);
        commands.clear();
        literals.clear();
        create_commands::<S, TABLE_BITS, ENTRIES, MIN_MATCH, INDEPENDENT>(
            simd,
            &Block {
                data,
                input,
                block_size,
                input_size,
            },
            table,
            &mut Pass1 { literals, commands },
        );
        if should_compress(
            &mut arena.lit_histo,
            &data[input..input + block_size],
            literals.len(),
        ) {
            store_meta_block_header(block_size, false, w);
            // No block splits, no contexts.
            w.write(13, 0);
            store_commands::<INDEPENDENT>(arena, literals, commands, w);
        } else {
            // Few backward references and an entropy close to eight bits per
            // byte: emitting the block verbatim is about three times faster.
            emit_uncompressed_meta_block(&data[input..input + block_size], w);
        }
        input += block_size;
        input_size -= block_size;
    }
}

/// Compresses `data` as one or more meta-blocks at quality 1.
///
/// `table` must be zeroed and hold exactly `table_bits.entries()` entries.
pub(crate) fn compress_fragment<S: Simd, const INDEPENDENT: bool>(
    simd: S,
    state: &mut TwoPassState,
    data: &[u8],
    is_last: bool,
    table_bits: TableBits,
    table: &mut [i32],
    w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
) {
    let initial_position = w.position();

    // The table is handed on as an array reference of its compile-time
    // width: the hash is a `64 - TABLE_BITS` shift, so every slot it yields is
    // inside the array and no lookup carries a bounds check. A slice of the
    // same length does not give the compiler that proof once the scan loop
    // is compiled inside the backend-specific function.
    macro_rules! run {
        ($bits:literal, $min_match:literal) => {{
            debug_assert_eq!(table_bits.bits(), $bits);
            debug_assert_eq!(table_bits.min_match(), $min_match);
            if let Some(table) = table[..1 << $bits].first_chunk_mut::<{ 1 << $bits }>() {
                compress_fragment_impl::<S, $bits, { 1 << $bits }, $min_match, INDEPENDENT>(
                    simd, state, data, table, w,
                );
            }
        }};
    }
    match table_bits {
        TableBits::B8 => run!(8, 4),
        TableBits::B9 => run!(9, 4),
        TableBits::B10 => run!(10, 4),
        TableBits::B11 => run!(11, 4),
        TableBits::B12 => run!(12, 4),
        TableBits::B13 => run!(13, 4),
        TableBits::B14 => run!(14, 4),
        TableBits::B15 => run!(15, 4),
        TableBits::B16 => run!(16, 6),
        TableBits::B17 => run!(17, 6),
    }

    // Rewrite the fragment verbatim when compressing made it larger.
    if w.position() - initial_position > 31 + (data.len() << 3) {
        w.rewind(initial_position);
        emit_uncompressed_meta_block(data, w);
    }

    if is_last {
        w.write(1, 1); // is_last
        w.write(1, 1); // is_empty
        w.align();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_width_follows_the_reference_selection() {
        assert_eq!(TableBits::for_input(0), TableBits::B8);
        assert_eq!(TableBits::for_input(256), TableBits::B8);
        assert_eq!(TableBits::for_input(257), TableBits::B9);
        assert_eq!(TableBits::for_input(1 << 17), TableBits::B17);
        assert_eq!(TableBits::for_input(1 << 24), TableBits::B17);
    }

    #[test]
    fn minimum_match_length_switches_at_sixteen_bits() {
        assert_eq!(TableBits::B8.min_match(), 4);
        assert_eq!(TableBits::B15.min_match(), 4);
        assert_eq!(TableBits::B16.min_match(), 6);
        assert_eq!(TableBits::B17.min_match(), 6);
    }

    #[test]
    fn hash_matches_the_reference_formula() {
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let word = load_u64_le(&data, 0);
        for (bits, min_match, expected_shift) in [(13usize, 4usize, 32u32), (17, 6, 16)] {
            let expected = (((word << expected_shift).wrapping_mul(u64::from(HASH_MUL32)))
                >> (64 - bits)) as usize;
            let actual = match (bits, min_match) {
                (13, 4) => hash::<13, 4>(&data, 0),
                _ => hash::<17, 6>(&data, 0),
            };
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn hash_at_offset_matches_hashing_the_shifted_position() {
        let data = [9u8, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 12, 13, 14];
        let word = load_u64_le(&data, 0);
        for offset in 0..=4u32 {
            assert_eq!(
                hash_bytes_at_offset::<15, 4>(word, offset),
                hash::<15, 4>(&data, offset as usize)
            );
        }
        for offset in 0..=2u32 {
            assert_eq!(
                hash_bytes_at_offset::<17, 6>(word, offset),
                hash::<17, 6>(&data, offset as usize)
            );
        }
    }

    #[test]
    fn fixed_predicate_uses_the_configured_match_length() {
        // Padded so both windows keep the eight readable bytes the wide
        // predicate needs, exactly as the scan guarantees.
        let mut data = vec![1u8, 2, 3, 4, 9, 9, 1, 2, 3, 4, 8, 8];
        data.extend_from_slice(&[0u8; 16]);
        assert!(is_match::<4>(&data, 0, 6));
        assert!(!is_match::<6>(&data, 0, 6));
    }

    #[test]
    fn first_update_reproduces_the_reference_offset_quirk() {
        let data: Vec<u8> = (0..64u8).collect();
        let mut first = [0i32; 1 << 12];
        let mut chained = [0i32; 1 << 12];
        let ip = 20usize;
        let first_current =
            update_hashes_after_copy::<12, { 1 << 12 }, 4, true>(&data, &mut first, ip);
        let chained_current =
            update_hashes_after_copy::<12, { 1 << 12 }, 4, false>(&data, &mut chained, ip);
        let word = load_u64_le(&data, ip - 3);
        assert_eq!(first_current, chained_current);
        assert_ne!(first, chained, "the reference quirk must be preserved");

        let quirk_slot = hash_bytes_at_offset::<12, 4>(word, 0);
        assert_eq!(first[quirk_slot], (ip - 1) as i32);
        assert_eq!(chained[quirk_slot], (ip - 3) as i32);
    }

    #[test]
    fn six_byte_updates_touch_five_positions() {
        let data: Vec<u8> = (0..64u8).collect();
        let mut table = vec![0i32; 1 << 17];
        let ip = 20usize;
        let Some(table) = table.first_chunk_mut::<{ 1 << 17 }>() else {
            panic!("table holds 1 << 17 entries");
        };
        let current = update_hashes_after_copy::<17, { 1 << 17 }, 6, true>(&data, table, ip);
        let stored: Vec<i32> = table.iter().copied().filter(|&v| v != 0).collect();
        assert_eq!(stored.len(), 5);
        assert!(current < table.len());
    }

    #[test]
    fn entropy_of_a_flat_histogram_is_eight_bits_per_symbol() {
        let histogram = [4u32; 256];
        assert!((bits_entropy(&histogram) - 8.0 * 1024.0).abs() < 1e-6);
    }

    #[test]
    fn entropy_never_drops_below_one_bit_per_symbol() {
        let mut histogram = [0u32; 256];
        histogram[3] = 100;
        assert_eq!(bits_entropy(&histogram), 100.0);
    }

    #[test]
    fn should_compress_accepts_data_with_backward_references() {
        let mut histogram = [0u32; NUM_LITERAL_SYMBOLS];
        let input = vec![0u8; 4096];
        assert!(should_compress(&mut histogram, &input, 10));
    }

    #[test]
    fn should_compress_rejects_incompressible_literal_only_blocks() {
        let mut histogram = [0u32; NUM_LITERAL_SYMBOLS];
        let mut input = vec![0u8; 65_536];
        let mut state = 0x9E37_79B9u32;
        for byte in input.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *byte = (state >> 24) as u8;
        }
        let len = input.len();
        assert!(!should_compress(&mut histogram, &input, len));
    }

    #[test]
    fn should_compress_accepts_low_entropy_literal_only_blocks() {
        let mut histogram = [0u32; NUM_LITERAL_SYMBOLS];
        let input: Vec<u8> = (0..65_536).map(|i| (i % 3) as u8).collect();
        let len = input.len();
        assert!(should_compress(&mut histogram, &input, len));
    }
}

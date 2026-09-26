//! Quality 0: the fast one-pass encoder.
//!
//! Direct port of `BrotliCompressFragmentFast` from `c/enc/compress_fragment.c`
//! of the pinned reference (`google/brotli` v1.2.0, commit `028fb5a`). Matches
//! are found and written to the bitstream in the same pass, so the literal code
//! is only an approximation of the post-LZ77 statistics.
//!
//! The order of every observable decision — candidate preference, hash update
//! points, block merging and the uncompressed fallback — is preserved exactly,
//! because it determines both the emitted bytes and the compression ratio.
//!
//! # Read window
//!
//! The scan reads short words rather than single bytes, which is what makes it
//! fast. Those reads never fall off the end of the fragment:
//!
//! * `ip_limit = input + min(block_size - 5, input_size - 16)` and
//!   `input + input_size == data.len()`, so `ip_limit + 16 <= data.len()`.
//! * `ip` never exceeds `ip_limit`, and a candidate never exceeds `ip + 1`
//!   (the repeat candidate with the initial `last_distance` of `-1`).
//! * Post-copy hash updates happen only while `ip < ip_limit`, and read at
//!   most from `ip - 3` through `ip + 5`.
//!
//! The loaders return zero past the end anyway, so the invariant is a
//! performance and clarity statement rather than a soundness one.

use alloc::vec::Vec;

use fearless_simd::Simd;

use super::bits::{BitWriter, ByteBuffer};
use super::commands::one_pass::{
    emit_copy_len, emit_copy_len_last_distance, emit_distance, emit_insert_len, emit_literals,
    emit_long_insert_len,
};
use super::commands::{fast_log2, log2_floor_non_zero, store_meta_block_header};
use super::constants::{
    BLOCK_SAMPLE_RATE, HASH_MUL32, LITERAL_SAMPLE_RATE, MAX_BACKWARD_DISTANCE, MIN_HASH_ENTRIES,
    NUM_COMMAND_SYMBOLS, NUM_LITERAL_SYMBOLS, Q0_FIRST_BLOCK_SIZE, Q0_MAX_HASH_ENTRIES,
    Q0_MAX_MERGED_BLOCK_SIZE, Q0_MERGE_BLOCK_SIZE, Q0_MIN_MATCH, Q0_MIN_RATIO, SHORT_INSERT_LIMIT,
    WINDOW_GAP,
};
use super::histogram;
use super::huffman::{
    HuffmanNode, build_and_store_huffman_tree_fast, convert_bit_depths_to_symbols,
    create_huffman_tree, store_huffman_tree,
};
use super::match_len::{current_window, load_u64_le, match_len_at};
use super::tables::CMD_HISTO_SEED;
use super::workspace::OnePassArena;

/// Hash table widths quality 0 is compiled for.
///
/// The reference bakes the shift and the minimum match length into four
/// specialised implementations; only odd shifts are supported.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum TableBits {
    /// 512 entries.
    B9,
    /// 2048 entries.
    B11,
    /// 8192 entries.
    B13,
    /// 32768 entries.
    B15,
}

impl TableBits {
    /// Selects the table width the reference would use for `input_size` bytes.
    pub(crate) fn for_input(input_size: usize) -> Self {
        let mut entries = MIN_HASH_ENTRIES;
        while entries < Q0_MAX_HASH_ENTRIES && entries < input_size {
            entries <<= 1;
        }
        // Only odd shifts are supported by the fast one-pass path.
        if entries & 0x000A_AAAA == 0 {
            entries <<= 1;
        }
        match log2_floor_non_zero(entries) {
            9 => Self::B9,
            11 => Self::B11,
            13 => Self::B13,
            _ => Self::B15,
        }
    }

    /// Returns the number of entries of a table of this width.
    pub(crate) const fn entries(self) -> usize {
        1 << self.bits()
    }

    /// Returns the base-2 logarithm of the table width.
    pub(crate) const fn bits(self) -> usize {
        match self {
            Self::B9 => 9,
            Self::B11 => 11,
            Self::B13 => 13,
            Self::B15 => 15,
        }
    }
}

/// Hashes the five bytes at `position` into a table index.
#[cfg(test)]
fn hash<const TABLE_BITS: usize>(data: &[u8], position: usize) -> usize {
    let word = load_u64_le(data, position);
    let mixed = (word << 24).wrapping_mul(HASH_MUL32 as u64);
    (mixed >> (64 - TABLE_BITS)) as usize
}

/// Hashes bytes `offset..offset + 5` of an already loaded machine word.
#[inline(always)]
fn hash_bytes_at_offset<const TABLE_BITS: usize>(word: u64, offset: u32) -> usize {
    debug_assert!(offset <= 3);
    let mixed = ((word >> (8 * offset)) << 24).wrapping_mul(HASH_MUL32 as u64);
    (mixed >> (64 - TABLE_BITS)) as usize
}

/// Bits of a 64-bit word covering the fixed five-byte match predicate.
const MATCH_MASK: u64 = (1 << (8 * Q0_MIN_MATCH)) - 1;

/// Tests the fixed five-byte match predicate at two positions.
///
/// Both positions sit inside the read window described at the top of this
/// module, so a single word load per side is enough; comparing the low five
/// bytes of the difference is equivalent to the reference's four-byte compare
/// followed by a fifth byte compare. Fewer than eight readable bytes on either
/// side reports no match, which the scan never relies on.
#[inline(always)]
fn is_match(data: &[u8], left: usize, right: usize) -> bool {
    words_match(load_u64_le(data, left), load_u64_le(data, right))
}

/// Tests the five-byte match predicate on two already loaded words.
#[inline(always)]
const fn words_match(left: u64, right: u64) -> bool {
    (left ^ right) & MATCH_MASK == 0
}

/// Builds the approximate literal prefix code and stores it.
///
/// Returns the estimated encoding ratio in millibytes per literal, which drives
/// the uncompressed-mode heuristic.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
fn build_and_store_literal_prefix_code(
    histogram: &mut [u32; NUM_LITERAL_SYMBOLS],
    tree: &mut Vec<HuffmanNode>,
    input: &[u8],
    depths: &mut [u8; NUM_LITERAL_SYMBOLS],
    bits: &mut [u16; NUM_LITERAL_SYMBOLS],
    w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
) -> usize {
    histogram.fill(0);
    let histogram_total = if input.len() < (1 << 15) {
        histogram::accumulate(input, histogram);
        let mut total = input.len();
        for slot in histogram.iter_mut() {
            // The first eleven samples weigh triple to account for the
            // balancing effect of the LZ77 phase on the histogram.
            let adjust = 2 * (*slot).min(11);
            *slot += adjust;
            total += adjust as usize;
        }
        total
    } else {
        histogram::accumulate_sampled(input, LITERAL_SAMPLE_RATE, histogram);
        let mut total = input.len().div_ceil(LITERAL_SAMPLE_RATE);
        for slot in histogram.iter_mut() {
            // The extra one keeps symbols the sample missed at a non-zero
            // depth; the rest is the same triple weighting as above.
            let adjust = 1 + 2 * (*slot).min(11);
            *slot += adjust;
            total += adjust as usize;
        }
        total
    };

    build_and_store_huffman_tree_fast(tree, histogram, histogram_total, 8, depths, bits, w);

    let mut literal_ratio = 0usize;
    for (index, &count) in histogram.iter().enumerate() {
        if count != 0 {
            literal_ratio += count as usize * usize::from(depths[index]);
        }
    }
    (literal_ratio * 125) / histogram_total
}

/// Builds the command and distance prefix codes and stores them.
///
/// The fast path keeps the 64 command symbols in a permuted order that removes
/// branches from the emit helpers, so the code is built in that order and then
/// scattered into the full alphabet for serialisation.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
fn build_and_store_command_prefix_code(
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

    tmp_depth[..24].copy_from_slice(&depth[..24]);
    tmp_depth[24..32].copy_from_slice(&depth[40..48]);
    tmp_depth[32..40].copy_from_slice(&depth[24..32]);
    tmp_depth[40..48].copy_from_slice(&depth[48..56]);
    tmp_depth[48..56].copy_from_slice(&depth[32..40]);
    tmp_depth[56..64].copy_from_slice(&depth[56..64]);
    convert_bit_depths_to_symbols(tmp_depth, 64, tmp_bits);
    bits[..24].copy_from_slice(&tmp_bits[..24]);
    bits[24..32].copy_from_slice(&tmp_bits[32..40]);
    bits[32..40].copy_from_slice(&tmp_bits[48..56]);
    bits[40..48].copy_from_slice(&tmp_bits[24..32]);
    bits[48..56].copy_from_slice(&tmp_bits[40..48]);
    bits[56..64].copy_from_slice(&tmp_bits[56..64]);
    convert_bit_depths_to_symbols(&depth[64..], 64, &mut bits[64..]);

    tmp_depth[..64].fill(0);
    tmp_depth[..8].copy_from_slice(&depth[..8]);
    tmp_depth[64..72].copy_from_slice(&depth[8..16]);
    tmp_depth[128..136].copy_from_slice(&depth[16..24]);
    tmp_depth[192..200].copy_from_slice(&depth[24..32]);
    tmp_depth[384..392].copy_from_slice(&depth[32..40]);
    for i in 0..8 {
        tmp_depth[128 + 8 * i] = depth[40 + i];
        tmp_depth[256 + 8 * i] = depth[48 + i];
        tmp_depth[448 + 8 * i] = depth[56 + i];
    }
    store_huffman_tree(tmp_depth, NUM_COMMAND_SYMBOLS, tree, w);
    store_huffman_tree(&depth[64..], 64, tree, w);
}

/// Decides whether the next chunk should extend the current meta-block.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
fn should_merge_block(
    histogram: &mut [u32; NUM_LITERAL_SYMBOLS],
    data: &[u8],
    depths: &[u8; NUM_LITERAL_SYMBOLS],
) -> bool {
    histogram.fill(0);
    histogram::accumulate_sampled(data, BLOCK_SAMPLE_RATE, histogram);
    let total = data.len().div_ceil(BLOCK_SAMPLE_RATE);
    let mut r = (fast_log2(total) + 0.5) * total as f64 + 200.0;
    for (index, &count) in histogram.iter().enumerate() {
        r -= f64::from(count) * (f64::from(depths[index]) + fast_log2(count as usize));
    }
    r >= 0.0
}

/// Decides whether a large literal run is better left uncompressed.
const fn should_use_uncompressed_mode(
    metablock_start: usize,
    next_emit: usize,
    insertlen: usize,
    literal_ratio: usize,
) -> bool {
    let compressed = next_emit - metablock_start;
    if compressed * 50 > insertlen {
        return false;
    }
    literal_ratio > Q0_MIN_RATIO
}

/// Returns whether a fragment needs no match scan at all.
///
/// An empty fragment is just the closing block. Otherwise the serial
/// compressed header alone costs at least 19 + 13 bits plus the retained
/// command code, so below this bound the size guard necessarily rewrites the
/// fragment verbatim. Only a final fragment qualifies: a non-final one must
/// still train the command code for its successor, and an independent one
/// carries no retained code to compare against.
pub(crate) const fn stores_verbatim(
    cmd_code_numbits: usize,
    independent: bool,
    len: usize,
    is_last: bool,
) -> bool {
    is_last && (len == 0 || (!independent && len <= cmd_code_numbits / 8))
}

/// Writes a final fragment verbatim and closes the stream.
///
/// This is the whole output for a fragment [`stores_verbatim`] accepts, so
/// the caller needs neither an arena nor a hash table.
pub(crate) fn store_final_verbatim(data: &[u8], w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>) {
    if !data.is_empty() {
        emit_uncompressed_meta_block(data, 0, data.len(), w.position(), w);
    }
    w.write(1, 1); // is_last
    w.write(1, 1); // is_empty
    w.align();
}

/// Rewinds to `start_position` and rewrites `data[begin..end]` verbatim.
fn emit_uncompressed_meta_block(
    data: &[u8],
    begin: usize,
    end: usize,
    start_position: usize,
    w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
) {
    w.rewind(start_position);
    store_meta_block_header(end - begin, true, w);
    w.align();
    if let Some(block) = data.get(begin..end) {
        w.write_bytes(block);
    }
}

/// Compresses one fragment with a table width baked in.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
fn compress_fragment_impl<
    S: Simd,
    const TABLE_BITS: usize,
    const ENTRIES: usize,
    const INDEPENDENT: bool,
>(
    simd: S,
    arena: &mut OnePassArena,
    data: &[u8],
    is_last: bool,
    table: &mut [i32; ENTRIES],
    w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
) {
    // Specialize the complete scan inside the selected SIMD feature context.
    simd.vectorize(
        #[inline(always)]
        || {
            let OnePassArena {
                lit_depth,
                lit_bits,
                cmd_depth,
                cmd_bits,
                cmd_histo,
                cmd_code,
                cmd_code_numbits,
                tree,
                histogram,
                tmp_depth,
                tmp_bits,
            } = arena;

            // "next_emit" is the first byte not covered by a previous copy.
            let mut next_emit = 0usize;
            let mut metablock_start = 0usize;
            let mut input = 0usize;
            let mut input_size = data.len();
            let mut block_size = input_size.min(Q0_FIRST_BLOCK_SIZE);
            let mut total_block_size = block_size;
            let mut mlen_storage_ix = w.position() + 3;

            store_meta_block_header(block_size, false, w);
            // No block splits, no contexts.
            w.write(13, 0);

            let mut literal_ratio = build_and_store_literal_prefix_code(
                histogram,
                tree,
                &data[input..input + block_size],
                lit_depth,
                lit_bits,
                w,
            );

            if INDEPENDENT {
                // Explicit remainders of five/six-byte matches use copy lengths three
                // and four. The serial starter tree omits their symbols (17 and 18).
                cmd_histo.copy_from_slice(&CMD_HISTO_SEED);
                cmd_histo[17..19].fill(1);
                build_and_store_command_prefix_code(
                    cmd_histo, cmd_depth, cmd_bits, tmp_depth, tmp_bits, tree, w,
                );
            } else {
                // Store the pre-compressed command and distance prefix codes.
                let numbits = *cmd_code_numbits;
                let mut i = 0usize;
                while i + 7 < numbits {
                    w.write(8, u64::from(cmd_code[i >> 3]));
                    i += 8;
                }
                let tail = (numbits & 7) as u32;
                let mask = (1u64 << tail) - 1;
                w.write(tail, u64::from(cmd_code[numbits >> 3]) & mask);
            }

            'emit_commands: loop {
                // Gather fresh statistics for the prefix code of the next block.
                cmd_histo.copy_from_slice(&CMD_HISTO_SEED);
                if INDEPENDENT {
                    cmd_histo[17..19].fill(1);
                }

                let mut ip = input;
                let mut last_distance: i64 = -1;
                let ip_end = input + block_size;
                let mut restart_meta_block = false;

                'scan: {
                    if block_size < WINDOW_GAP {
                        break 'scan;
                    }
                    // The last block keeps a sixteen byte margin so every distance stays
                    // below the window size; other blocks only need the copy margin.
                    let len_limit = (block_size - Q0_MIN_MATCH).min(input_size - WINDOW_GAP);
                    let ip_limit = input + len_limit;

                    ip += 1;
                    // The word at the next position is loaded once: it is
                    // hashed here and compared against the candidates when
                    // the scan reaches it.
                    let mut next_word = load_u64_le(data, ip);
                    let mut next_hash = hash_bytes_at_offset::<TABLE_BITS>(next_word, 0);
                    loop {
                        // Heuristic match skipping: after 32 unproductive bytes look at
                        // every other byte, after 32 more at every third, and so on. A
                        // found match resets the stride.
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
                                    next_hash = hash_bytes_at_offset::<TABLE_BITS>(next_word, 0);

                                    // The reference subtracts the last distance
                                    // without a sign test and rejects a position at
                                    // or past `ip` after comparing; the two tests
                                    // are pure, so rejecting first reads nothing
                                    // for the initial distance of minus one.
                                    let repeated = (ip as i64 - last_distance) as usize;
                                    if repeated < ip
                                        && words_match(ip_word, load_u64_le(data, repeated))
                                    {
                                        table[slot] = ip as i32;
                                        candidate = repeated;
                                        break;
                                    }

                                    candidate = table[slot] as usize;
                                    table[slot] = ip as i32;
                                    if words_match(ip_word, load_u64_le(data, candidate)) {
                                        break;
                                    }
                                }
                                // Checked outside the hot loop to keep it tight.
                                if ip - candidate > MAX_BACKWARD_DISTANCE {
                                    continue;
                                }
                                break 'found;
                            }
                        }

                        // A five byte match at "ip"; the bytes in [next_emit, ip) still
                        // have to be emitted as literals.
                        let base = ip;
                        let matched = Q0_MIN_MATCH
                            + match_len_at(
                                simd,
                                data,
                                candidate + Q0_MIN_MATCH,
                                current_window(
                                    data,
                                    ip + Q0_MIN_MATCH,
                                    (ip_end - ip) - Q0_MIN_MATCH,
                                ),
                            );
                        let distance = (base - candidate) as i64;
                        let insert = base - next_emit;
                        ip += matched;

                        if insert < SHORT_INSERT_LIMIT {
                            emit_insert_len(insert, cmd_depth, cmd_bits, cmd_histo, w);
                        } else if should_use_uncompressed_mode(
                            metablock_start,
                            next_emit,
                            insert,
                            literal_ratio,
                        ) {
                            emit_uncompressed_meta_block(
                                data,
                                metablock_start,
                                base,
                                mlen_storage_ix - 3,
                                w,
                            );
                            input_size -= base - input;
                            input = base;
                            next_emit = input;
                            restart_meta_block = true;
                            break 'scan;
                        } else {
                            emit_long_insert_len(insert, cmd_depth, cmd_bits, cmd_histo, w);
                        }
                        emit_literals(data, next_emit, insert, lit_depth, lit_bits, w);

                        if !INDEPENDENT && distance == last_distance {
                            w.write(u32::from(cmd_depth[64]), u64::from(cmd_bits[64]));
                            cmd_histo[64] += 1;
                        } else {
                            emit_distance(distance as usize, cmd_depth, cmd_bits, cmd_histo, w);
                            last_distance = distance;
                        }
                        if INDEPENDENT {
                            // The insert command already copied two bytes. Encode the
                            // remaining copy explicitly, including its distance.
                            emit_copy_len(matched - 2, cmd_depth, cmd_bits, cmd_histo, w);
                            emit_distance(distance as usize, cmd_depth, cmd_bits, cmd_histo, w);
                        } else {
                            emit_copy_len_last_distance(matched, cmd_depth, cmd_bits, cmd_histo, w);
                        }

                        next_emit = ip;
                        if ip >= ip_limit {
                            break 'scan;
                        }
                        // Hash a few positions inside the copy before moving on; this
                        // costs little and improves the ratio noticeably.
                        {
                            let word = load_u64_le(data, ip - 3);
                            let current = hash_bytes_at_offset::<TABLE_BITS>(word, 3);
                            table[hash_bytes_at_offset::<TABLE_BITS>(word, 0)] = (ip - 3) as i32;
                            table[hash_bytes_at_offset::<TABLE_BITS>(word, 1)] = (ip - 2) as i32;
                            table[hash_bytes_at_offset::<TABLE_BITS>(word, 2)] = (ip - 1) as i32;
                            candidate = table[current] as usize;
                            table[current] = ip as i32;
                        }

                        while is_match(data, ip, candidate) {
                            // Another five byte match, with no literals in between.
                            let base = ip;
                            let matched = Q0_MIN_MATCH
                                + match_len_at(
                                    simd,
                                    data,
                                    candidate + Q0_MIN_MATCH,
                                    current_window(
                                        data,
                                        ip + Q0_MIN_MATCH,
                                        (ip_end - ip) - Q0_MIN_MATCH,
                                    ),
                                );
                            if ip - candidate > MAX_BACKWARD_DISTANCE {
                                break;
                            }
                            ip += matched;
                            last_distance = (base - candidate) as i64;
                            emit_copy_len(matched, cmd_depth, cmd_bits, cmd_histo, w);
                            emit_distance(
                                last_distance as usize,
                                cmd_depth,
                                cmd_bits,
                                cmd_histo,
                                w,
                            );

                            next_emit = ip;
                            if ip >= ip_limit {
                                break 'scan;
                            }
                            let word = load_u64_le(data, ip - 3);
                            let current = hash_bytes_at_offset::<TABLE_BITS>(word, 3);
                            table[hash_bytes_at_offset::<TABLE_BITS>(word, 0)] = (ip - 3) as i32;
                            table[hash_bytes_at_offset::<TABLE_BITS>(word, 1)] = (ip - 2) as i32;
                            table[hash_bytes_at_offset::<TABLE_BITS>(word, 2)] = (ip - 1) as i32;
                            candidate = table[current] as usize;
                            table[current] = ip as i32;
                        }

                        ip += 1;
                        next_word = load_u64_le(data, ip);
                        next_hash = hash_bytes_at_offset::<TABLE_BITS>(next_word, 0);
                    }
                }

                if !restart_meta_block {
                    debug_assert!(next_emit <= ip_end);
                    input += block_size;
                    input_size -= block_size;
                    block_size = input_size.min(Q0_MERGE_BLOCK_SIZE);

                    // Continue this meta-block instead of closing it with a last
                    // insert-only command, when that looks cheaper.
                    if input_size > 0
                        && total_block_size + block_size <= Q0_MAX_MERGED_BLOCK_SIZE
                        && should_merge_block(
                            histogram,
                            &data[input..input + block_size],
                            lit_depth,
                        )
                    {
                        total_block_size += block_size;
                        // Both the old and the new size use five nibbles.
                        w.update(20, (total_block_size - 1) as u32, mlen_storage_ix);
                        continue 'emit_commands;
                    }

                    if next_emit < ip_end {
                        let insert = ip_end - next_emit;
                        if insert < SHORT_INSERT_LIMIT {
                            emit_insert_len(insert, cmd_depth, cmd_bits, cmd_histo, w);
                            emit_literals(data, next_emit, insert, lit_depth, lit_bits, w);
                        } else if should_use_uncompressed_mode(
                            metablock_start,
                            next_emit,
                            insert,
                            literal_ratio,
                        ) {
                            emit_uncompressed_meta_block(
                                data,
                                metablock_start,
                                ip_end,
                                mlen_storage_ix - 3,
                                w,
                            );
                        } else {
                            emit_long_insert_len(insert, cmd_depth, cmd_bits, cmd_histo, w);
                            emit_literals(data, next_emit, insert, lit_depth, lit_bits, w);
                        }
                    }
                    next_emit = ip_end;
                }

                // Open a new meta-block when input is left.
                if input_size > 0 {
                    metablock_start = input;
                    block_size = input_size.min(Q0_FIRST_BLOCK_SIZE);
                    total_block_size = block_size;
                    mlen_storage_ix = w.position() + 3;
                    store_meta_block_header(block_size, false, w);
                    w.write(13, 0);
                    literal_ratio = build_and_store_literal_prefix_code(
                        histogram,
                        tree,
                        &data[input..input + block_size],
                        lit_depth,
                        lit_bits,
                        w,
                    );
                    build_and_store_command_prefix_code(
                        cmd_histo, cmd_depth, cmd_bits, tmp_depth, tmp_bits, tree, w,
                    );
                    continue 'emit_commands;
                }

                break 'emit_commands;
            }

            if !is_last {
                // Hand the statistics of this fragment to the next one.
                cmd_code[0] = 0;
                *cmd_code_numbits = 0;
                let mut code_writer = BitWriter::new(cmd_code, 0);
                build_and_store_command_prefix_code(
                    cmd_histo,
                    cmd_depth,
                    cmd_bits,
                    tmp_depth,
                    tmp_bits,
                    tree,
                    &mut code_writer,
                );
                *cmd_code_numbits = code_writer.position();
            }
        },
    );
}

/// Compresses `data` as one or more meta-blocks at quality 0.
///
/// `table` must be zeroed and hold exactly `table_bits.entries()` entries.
pub(crate) fn compress_fragment<S: Simd, const INDEPENDENT: bool>(
    simd: S,
    arena: &mut OnePassArena,
    data: &[u8],
    is_last: bool,
    table_bits: TableBits,
    table: &mut [i32],
    w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
) {
    let initial_position = w.position();

    if data.is_empty() {
        debug_assert!(is_last);
        store_final_verbatim(data, w);
        return;
    }
    if stores_verbatim(arena.cmd_code_numbits, INDEPENDENT, data.len(), is_last) {
        store_final_verbatim(data, w);
        return;
    }

    // The table is handed on as an array reference of its compile-time
    // width: the hash is a `64 - TABLE_BITS` shift, so every slot it yields is
    // inside the array and no lookup carries a bounds check. A slice of the
    // same length does not give the compiler that proof once the scan loop
    // is compiled inside the backend-specific function.
    macro_rules! run {
        ($bits:literal) => {{
            debug_assert_eq!(table_bits.bits(), $bits);
            if let Some(table) = table[..1 << $bits].first_chunk_mut::<{ 1 << $bits }>() {
                compress_fragment_impl::<S, $bits, { 1 << $bits }, INDEPENDENT>(
                    simd, arena, data, is_last, table, w,
                );
            }
        }};
    }
    match table_bits {
        TableBits::B9 => run!(9),
        TableBits::B11 => run!(11),
        TableBits::B13 => run!(13),
        TableBits::B15 => run!(15),
    }

    // Rewrite the fragment verbatim when compressing made it larger.
    if w.position() - initial_position > 31 + (data.len() << 3) {
        emit_uncompressed_meta_block(data, 0, data.len(), initial_position, w);
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

    fn check_small_fragments<S: Simd, const INDEPENDENT: bool>(simd: S) {
        for trained in [false, true] {
            for offset in 0..8 {
                for pattern in [0u8, 1, 2] {
                    for len in 1..=80 {
                        for is_last in [false, true] {
                            let mut actual_arena = OnePassArena::default();
                            let mut reference_arena = OnePassArena::default();
                            if trained {
                                let prefix = b"a small repeated prefix ".repeat(40);
                                for arena in [&mut actual_arena, &mut reference_arena] {
                                    let mut storage = Vec::new();
                                    compress_fragment_impl::<S, 11, 2048, INDEPENDENT>(
                                        simd,
                                        arena,
                                        &prefix,
                                        false,
                                        &mut [0; 2048],
                                        &mut BitWriter::append(&mut storage, 0),
                                    );
                                }
                            }
                            let data: Vec<u8> = (0..len)
                                .map(|i| match pattern {
                                    0 => b'a',
                                    1 => i as u8,
                                    _ => (i as u8).wrapping_mul(73) ^ (i >> 2) as u8,
                                })
                                .collect();
                            let seed = ((1u16 << offset) - 1) as u8;
                            let mut actual = vec![seed];
                            let mut reference = vec![seed];
                            let mut writer = BitWriter::append(&mut reference, offset);
                            compress_fragment_impl::<S, 9, 512, INDEPENDENT>(
                                simd,
                                &mut reference_arena,
                                &data,
                                is_last,
                                &mut [0; 512],
                                &mut writer,
                            );
                            if writer.position() - offset > 31 + (len << 3) {
                                emit_uncompressed_meta_block(&data, 0, len, offset, &mut writer);
                            }
                            if is_last {
                                writer.write(2, 3);
                                writer.align();
                            }
                            let expected_bits = writer.position();
                            let mut writer = BitWriter::append(&mut actual, offset);
                            compress_fragment::<S, INDEPENDENT>(
                                simd,
                                &mut actual_arena,
                                &data,
                                is_last,
                                TableBits::B9,
                                &mut [0; 512],
                                &mut writer,
                            );
                            assert_eq!(writer.position(), expected_bits);
                            let bytes = expected_bits.div_ceil(8);
                            assert_eq!(
                                actual[..bytes],
                                reference[..bytes],
                                "trained={trained} offset={offset} pattern={pattern} len={len} last={is_last}"
                            );
                            if !is_last {
                                assert_eq!(actual_arena.cmd_code, reference_arena.cmd_code);
                                assert_eq!(
                                    actual_arena.cmd_code_numbits,
                                    reference_arena.cmd_code_numbits
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn small_final_fragments_match_full_encoding_on_every_backend() {
        for backend in crate::Backend::available() {
            fearless_simd::dispatch!(backend.0, simd => {
                check_small_fragments::<_, false>(simd);
                check_small_fragments::<_, true>(simd);
            });
        }
    }

    #[test]
    fn table_width_follows_the_reference_selection() {
        assert_eq!(TableBits::for_input(0), TableBits::B9);
        assert_eq!(TableBits::for_input(256), TableBits::B9);
        assert_eq!(TableBits::for_input(512), TableBits::B9);
        assert_eq!(TableBits::for_input(513), TableBits::B11);
        assert_eq!(TableBits::for_input(2048), TableBits::B11);
        assert_eq!(TableBits::for_input(2049), TableBits::B13);
        assert_eq!(TableBits::for_input(8192), TableBits::B13);
        assert_eq!(TableBits::for_input(8193), TableBits::B15);
        assert_eq!(TableBits::for_input(1 << 24), TableBits::B15);
    }

    #[test]
    fn table_width_reports_matching_entry_counts() {
        assert_eq!(TableBits::B9.entries(), 512);
        assert_eq!(TableBits::B11.entries(), 2048);
        assert_eq!(TableBits::B13.entries(), 8192);
        assert_eq!(TableBits::B15.entries(), 32_768);
    }

    #[test]
    fn hash_matches_the_reference_formula() {
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let word = u64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8]);
        let expected = (((word << 24).wrapping_mul(u64::from(HASH_MUL32))) >> (64 - 13)) as usize;
        assert_eq!(hash::<13>(&data, 0), expected);
    }

    #[test]
    fn hash_at_offset_matches_hashing_the_shifted_position() {
        let data = [9u8, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 12];
        let word = load_u64_le(&data, 0);
        for offset in 0..=3u32 {
            assert_eq!(
                hash_bytes_at_offset::<15>(word, offset),
                hash::<15>(&data, offset as usize)
            );
        }
    }

    #[test]
    fn fixed_predicate_needs_five_equal_bytes() {
        // Padded so both windows keep the eight readable bytes the wide
        // predicate needs, exactly as the scan guarantees.
        let mut data = vec![1u8, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 6];
        data.extend_from_slice(&[0u8; 16]);
        assert!(is_match(&data, 0, 5));
        assert!(!is_match(&data, 0, 10));
    }

    #[test]
    fn fixed_predicate_ignores_bytes_past_the_fifth() {
        let mut data = vec![1u8, 2, 3, 4, 5, 0xAA, 1, 2, 3, 4, 5, 0xBB];
        data.extend_from_slice(&[0u8; 16]);
        assert!(is_match(&data, 0, 6));
    }

    #[test]
    fn uncompressed_heuristic_follows_the_reference_thresholds() {
        assert!(!should_use_uncompressed_mode(0, 100, 1000, 999));
        assert!(should_use_uncompressed_mode(0, 1, 1000, 999));
        assert!(!should_use_uncompressed_mode(0, 1, 1000, 980));
    }

    #[test]
    fn block_merge_follows_the_cost_of_the_current_literal_code() {
        let mut histogram = [0u32; NUM_LITERAL_SYMBOLS];
        let mut random = vec![0u8; 65_536];
        let mut state = 12_345u32;
        for byte in random.iter_mut() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *byte = (state >> 16) as u8;
        }

        // A code whose depths already match the data pays for itself.
        let matching = [8u8; NUM_LITERAL_SYMBOLS];
        assert!(should_merge_block(&mut histogram, &random, &matching));

        // A code that is far too deep for the data does not.
        let mismatched = [15u8; NUM_LITERAL_SYMBOLS];
        assert!(!should_merge_block(&mut histogram, &random, &mismatched));

        // A degenerate code over a single byte value always merges.
        let degenerate = [0u8; NUM_LITERAL_SYMBOLS];
        assert!(should_merge_block(
            &mut histogram,
            &[0u8; 65_536],
            &degenerate
        ));
    }
}

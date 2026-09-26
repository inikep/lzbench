//! Insert, copy and distance code mapping shared by both fast qualities.
//!
//! Quality 0 writes command symbols straight into the bitstream while quality 1
//! buffers them as packed words, so the two families of helpers live side by
//! side here rather than being duplicated across the encoders. Both are direct
//! ports of `EmitInsertLen`, `EmitCopyLen`, `EmitCopyLenLastDistance` and
//! `EmitDistance` from the pinned reference.

use super::bits::{BitWriter, ByteBuffer, MAX_BITS_PER_WRITE};
use super::constants::{LONG_INSERT_LIMIT, SHORT_INSERT_LIMIT};

pub(crate) use crate::shared::fast_log::{fast_log2, log2_floor_non_zero};

/// Writes the meta-block header for `len` bytes.
///
/// `len` must be in `1..=1 << 24`.
pub(crate) fn store_meta_block_header(
    len: usize,
    is_uncompressed: bool,
    w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
) {
    debug_assert!((1..=1 << 24).contains(&len));
    let nibbles: u32 = if len <= 1 << 16 {
        4
    } else if len <= 1 << 20 {
        5
    } else {
        6
    };
    w.write(1, 0);
    w.write(2, u64::from(nibbles - 4));
    w.write(nibbles * 4, (len - 1) as u64);
    w.write(1, u64::from(is_uncompressed));
}

/// Quality 0 command emitters, writing prefix codes directly to the stream.
pub(crate) mod one_pass {
    use super::{
        BitWriter, ByteBuffer, LONG_INSERT_LIMIT, MAX_BITS_PER_WRITE, log2_floor_non_zero,
    };

    /// Writes a prefix code and its extra bits as one call.
    ///
    /// Command codes are at most fifteen bits and extra fields at most
    /// twenty-four, so the pair always fits in the writer's 56 bit limit.
    /// Fusing them halves the number of stores on command-heavy data without
    /// changing a single emitted bit.
    #[inline(always)]
    fn write_code_and_extra(
        depth: u8,
        bits: u16,
        extra_bits: u32,
        extra: u64,
        w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
    ) {
        let width = u32::from(depth);
        w.write(width + extra_bits, u64::from(bits) | (extra << width));
    }

    /// Writes a prefix code, its extra bits and the distance-reuse symbol as
    /// one call.
    ///
    /// Two codes of at most fifteen bits plus twenty-four extra bits still fit
    /// in the writer's 56 bit limit.
    #[inline(always)]
    fn write_code_extra_and_reuse(
        depth: &[u8; 128],
        bits: &[u16; 128],
        code: usize,
        extra_bits: u32,
        extra: u64,
        w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
    ) {
        let width = u32::from(depth[code]);
        let with_extra = u64::from(bits[code]) | (extra << width);
        let width = width + extra_bits;
        w.write(
            width + u32::from(depth[64]),
            with_extra | (u64::from(bits[64]) << width),
        );
    }

    /// Masks a command symbol into the 128 entry fast alphabet.
    ///
    /// Every code these emitters compute is already below 128; spelling that
    /// out lets the compiler drop the bounds check on the three parallel
    /// arrays each command touches.
    #[inline(always)]
    const fn symbol(code: usize) -> usize {
        code & 127
    }

    /// Writes an insert length below [`SHORT_INSERT_LIMIT`].
    #[inline(always)]
    pub(crate) fn emit_insert_len(
        insertlen: usize,
        depth: &[u8; 128],
        bits: &[u16; 128],
        histo: &mut [u32; 128],
        w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
    ) {
        if insertlen < 6 {
            let code = symbol(insertlen + 40);
            w.write(u32::from(depth[code]), u64::from(bits[code]));
            histo[code] += 1;
        } else if insertlen < 130 {
            let tail = insertlen - 2;
            let nbits = log2_floor_non_zero(tail) - 1;
            let prefix = tail >> nbits;
            let code = symbol(((nbits as usize) << 1) + prefix + 42);
            write_code_and_extra(
                depth[code],
                bits[code],
                nbits,
                (tail - (prefix << nbits)) as u64,
                w,
            );
            histo[code] += 1;
        } else if insertlen < 2114 {
            let tail = insertlen - 66;
            let nbits = log2_floor_non_zero(tail);
            let code = symbol(nbits as usize + 50);
            write_code_and_extra(
                depth[code],
                bits[code],
                nbits,
                (tail - (1usize << nbits)) as u64,
                w,
            );
            histo[code] += 1;
        } else {
            write_code_and_extra(depth[61], bits[61], 12, (insertlen - 2114) as u64, w);
            histo[61] += 1;
        }
    }

    /// Writes an insert length of at least [`SHORT_INSERT_LIMIT`].
    #[inline(always)]
    pub(crate) fn emit_long_insert_len(
        insertlen: usize,
        depth: &[u8; 128],
        bits: &[u16; 128],
        histo: &mut [u32; 128],
        w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
    ) {
        if insertlen < LONG_INSERT_LIMIT {
            write_code_and_extra(depth[62], bits[62], 14, (insertlen - 6210) as u64, w);
            histo[62] += 1;
        } else {
            write_code_and_extra(
                depth[63],
                bits[63],
                24,
                (insertlen - LONG_INSERT_LIMIT) as u64,
                w,
            );
            histo[63] += 1;
        }
    }

    /// Writes a copy length that follows an explicit distance.
    #[inline(always)]
    pub(crate) fn emit_copy_len(
        copylen: usize,
        depth: &[u8; 128],
        bits: &[u16; 128],
        histo: &mut [u32; 128],
        w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
    ) {
        if copylen < 10 {
            let code = symbol(copylen + 14);
            w.write(u32::from(depth[code]), u64::from(bits[code]));
            histo[code] += 1;
        } else if copylen < 134 {
            let tail = copylen - 6;
            let nbits = log2_floor_non_zero(tail) - 1;
            let prefix = tail >> nbits;
            let code = symbol(((nbits as usize) << 1) + prefix + 20);
            write_code_and_extra(
                depth[code],
                bits[code],
                nbits,
                (tail - (prefix << nbits)) as u64,
                w,
            );
            histo[code] += 1;
        } else if copylen < 2118 {
            let tail = copylen - 70;
            let nbits = log2_floor_non_zero(tail);
            let code = symbol(nbits as usize + 28);
            write_code_and_extra(
                depth[code],
                bits[code],
                nbits,
                (tail - (1usize << nbits)) as u64,
                w,
            );
            histo[code] += 1;
        } else {
            write_code_and_extra(depth[39], bits[39], 24, (copylen - 2118) as u64, w);
            histo[39] += 1;
        }
    }

    /// Writes a copy length that reuses the last distance.
    #[inline(always)]
    pub(crate) fn emit_copy_len_last_distance(
        copylen: usize,
        depth: &[u8; 128],
        bits: &[u16; 128],
        histo: &mut [u32; 128],
        w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
    ) {
        if copylen < 12 {
            let code = symbol(copylen - 4);
            w.write(u32::from(depth[code]), u64::from(bits[code]));
            histo[code] += 1;
        } else if copylen < 72 {
            let tail = copylen - 8;
            let nbits = log2_floor_non_zero(tail) - 1;
            let prefix = tail >> nbits;
            let code = symbol(((nbits as usize) << 1) + prefix + 4);
            write_code_and_extra(
                depth[code],
                bits[code],
                nbits,
                (tail - (prefix << nbits)) as u64,
                w,
            );
            histo[code] += 1;
        } else if copylen < 136 {
            let tail = copylen - 8;
            let code = symbol((tail >> 5) + 30);
            write_code_extra_and_reuse(depth, bits, code, 5, (tail & 31) as u64, w);
            histo[code] += 1;
            histo[64] += 1;
        } else if copylen < 2120 {
            let tail = copylen - 72;
            let nbits = log2_floor_non_zero(tail);
            let code = symbol(nbits as usize + 28);
            write_code_extra_and_reuse(
                depth,
                bits,
                code,
                nbits,
                (tail - (1usize << nbits)) as u64,
                w,
            );
            histo[code] += 1;
            histo[64] += 1;
        } else {
            write_code_extra_and_reuse(depth, bits, 39, 24, (copylen - 2120) as u64, w);
            histo[39] += 1;
            histo[64] += 1;
        }
    }

    /// Writes a backward distance.
    #[inline(always)]
    pub(crate) fn emit_distance(
        distance: usize,
        depth: &[u8; 128],
        bits: &[u16; 128],
        histo: &mut [u32; 128],
        w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
    ) {
        let d = distance + 3;
        let nbits = log2_floor_non_zero(d) - 1;
        let prefix = (d >> nbits) & 1;
        let offset = (2 + prefix) << nbits;
        let code = symbol(2 * (nbits as usize - 1) + prefix + 80);
        write_code_and_extra(depth[code], bits[code], nbits, (d - offset) as u64, w);
        histo[code] += 1;
    }

    /// Shortest literal run for which the four-at-a-time loop pays off.
    ///
    /// Below this a run yields at most one quadruple, and the extra split costs
    /// more than the store it saves; the pair loop alone is faster.
    const QUAD_RUN_THRESHOLD: usize = 8;

    /// Writes `len` literal bytes starting at `start`.
    ///
    /// Literal codes are capped at a depth of fourteen bits by the tree
    /// builder, so four of them always fit in the writer's
    /// [`MAX_BITS_PER_WRITE`] budget, and two always fit with room to spare.
    /// Packing several codes per call cuts the number of stores without
    /// changing a single emitted bit.
    #[inline(always)]
    pub(crate) fn emit_literals(
        data: &[u8],
        start: usize,
        len: usize,
        depth: &[u8; 256],
        bits: &[u16; 256],
        w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
    ) {
        let Some(literals) = data.get(start..start + len) else {
            return;
        };
        // Quality 0 emits literals between matches, so run length varies by
        // corpus: text averages roughly three literals per run while
        // structured binary averages twenty. Long runs are packed four at a
        // time, short ones go straight to the pair loop, which is why the
        // split is gated on the run length rather than applied unconditionally.
        // The greedy packing quality 1 uses is a loss here either way: it pays
        // a per-literal branch that only earns its keep across a whole
        // meta-block of literals.
        let literals = if literals.len() >= QUAD_RUN_THRESHOLD {
            // `as_chunks` keeps the loop bounds-check free; the running widths
            // are a plain prefix sum over the four code lengths.
            let (quads, rest) = literals.as_chunks::<4>();
            for quad in quads {
                let first = usize::from(quad[0]);
                let second = usize::from(quad[1]);
                let third = usize::from(quad[2]);
                let fourth = usize::from(quad[3]);
                let first_width = u32::from(depth[first]);
                let second_width = first_width + u32::from(depth[second]);
                let third_width = second_width + u32::from(depth[third]);
                let value = u64::from(bits[first])
                    | (u64::from(bits[second]) << first_width)
                    | (u64::from(bits[third]) << second_width)
                    | (u64::from(bits[fourth]) << third_width);
                w.write(third_width + u32::from(depth[fourth]), value);
            }
            rest
        } else {
            literals
        };
        let (pairs, tail) = literals.as_chunks::<2>();
        for pair in pairs {
            let first = usize::from(pair[0]);
            let second = usize::from(pair[1]);
            let first_width = u32::from(depth[first]);
            let value = u64::from(bits[first]) | (u64::from(bits[second]) << first_width);
            w.write(first_width + u32::from(depth[second]), value);
        }
        for &literal in tail {
            let index = usize::from(literal);
            w.write(u32::from(depth[index]), u64::from(bits[index]));
        }
    }

    /// Emits `literals` through the literal prefix code, packed into whole
    /// bit-writer calls.
    ///
    /// Codes are accumulated until the next one would overflow the writer's 56
    /// bit limit, which turns a long literal run into a handful of stores. The
    /// per-literal branch pays for itself on the runs quality 1 replays, where
    /// a whole meta-block of literals arrives at once.
    #[inline(always)]
    pub(crate) fn pack_literals(
        literals: &[u8],
        depth: &[u8; 256],
        bits: &[u16; 256],
        w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
    ) {
        let mut value = 0u64;
        let mut width = 0u32;
        for &literal in literals {
            let index = usize::from(literal);
            let code_width = u32::from(depth[index]);
            if width + code_width > MAX_BITS_PER_WRITE {
                w.write(width, value);
                value = 0;
                width = 0;
            }
            value |= u64::from(bits[index]) << width;
            width += code_width;
        }
        w.write(width, value);
    }
}

/// Quality 1 command emitters, packing a code and its extra bits into a word.
pub(crate) mod two_pass {
    use super::{LONG_INSERT_LIMIT, SHORT_INSERT_LIMIT, log2_floor_non_zero};
    use alloc::vec::Vec;

    /// Appends the packed representation of an insert length.
    #[inline(always)]
    pub(crate) fn emit_insert_len(insertlen: usize, commands: &mut Vec<u32>) {
        let word = if insertlen < 6 {
            insertlen as u32
        } else if insertlen < 130 {
            let tail = insertlen - 2;
            let nbits = log2_floor_non_zero(tail) - 1;
            let prefix = tail >> nbits;
            let inscode = ((nbits as usize) << 1) + prefix + 2;
            let extra = tail - (prefix << nbits);
            (inscode as u32) | ((extra as u32) << 8)
        } else if insertlen < 2114 {
            let tail = insertlen - 66;
            let nbits = log2_floor_non_zero(tail);
            let code = nbits + 10;
            let extra = tail - (1usize << nbits);
            code | ((extra as u32) << 8)
        } else if insertlen < SHORT_INSERT_LIMIT {
            21 | (((insertlen - 2114) as u32) << 8)
        } else if insertlen < LONG_INSERT_LIMIT {
            22 | (((insertlen - SHORT_INSERT_LIMIT) as u32) << 8)
        } else {
            23 | (((insertlen - LONG_INSERT_LIMIT) as u32) << 8)
        };
        commands.push(word);
    }

    /// Appends the packed representation of a copy length.
    #[inline(always)]
    pub(crate) fn emit_copy_len(copylen: usize, commands: &mut Vec<u32>) {
        let word = if copylen < 10 {
            (copylen + 38) as u32
        } else if copylen < 134 {
            let tail = copylen - 6;
            let nbits = log2_floor_non_zero(tail) - 1;
            let prefix = tail >> nbits;
            let code = ((nbits as usize) << 1) + prefix + 44;
            let extra = tail - (prefix << nbits);
            (code as u32) | ((extra as u32) << 8)
        } else if copylen < 2118 {
            let tail = copylen - 70;
            let nbits = log2_floor_non_zero(tail);
            let code = nbits + 52;
            let extra = tail - (1usize << nbits);
            code | ((extra as u32) << 8)
        } else {
            63 | (((copylen - 2118) as u32) << 8)
        };
        commands.push(word);
    }

    /// Appends the packed copy length that reuses the last distance.
    #[inline(always)]
    pub(crate) fn emit_copy_len_last_distance(copylen: usize, commands: &mut Vec<u32>) {
        if copylen < 12 {
            commands.push((copylen + 20) as u32);
            return;
        }
        if copylen < 72 {
            let tail = copylen - 8;
            let nbits = log2_floor_non_zero(tail) - 1;
            let prefix = tail >> nbits;
            let code = ((nbits as usize) << 1) + prefix + 28;
            let extra = tail - (prefix << nbits);
            commands.push((code as u32) | ((extra as u32) << 8));
            return;
        }
        if copylen < 136 {
            let tail = copylen - 8;
            let code = (tail >> 5) + 54;
            commands.push((code as u32) | (((tail & 31) as u32) << 8));
        } else if copylen < 2120 {
            let tail = copylen - 72;
            let nbits = log2_floor_non_zero(tail);
            let code = nbits + 52;
            let extra = tail - (1usize << nbits);
            commands.push(code | ((extra as u32) << 8));
        } else {
            commands.push(63 | (((copylen - 2120) as u32) << 8));
        }
        commands.push(64);
    }

    /// Appends the packed representation of a backward distance.
    #[inline(always)]
    pub(crate) fn emit_distance(distance: usize, commands: &mut Vec<u32>) {
        let d = distance + 3;
        let nbits = log2_floor_non_zero(d) - 1;
        let prefix = (d >> nbits) & 1;
        let offset = (2 + prefix) << nbits;
        let code = 2 * (nbits as usize - 1) + prefix + 80;
        commands.push((code as u32) | (((d - offset) as u32) << 8));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::core::fast::tables::{INSERT_OFFSET, NUM_EXTRA_BITS};
    use alloc::vec::Vec;

    fn one_pass_code(emit: impl FnOnce(&mut [u32; 128], &mut BitWriter)) -> (usize, Vec<usize>) {
        let mut storage = vec![0u8; 64];
        let mut w = BitWriter::new(&mut storage, 0);
        let mut histo = [0u32; 128];
        emit(&mut histo, &mut w);
        let used: Vec<usize> = histo
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c != 0)
            .map(|(i, _)| i)
            .collect();
        (w.position(), used)
    }

    #[test]
    fn meta_block_header_picks_the_narrowest_nibble_count() {
        for (len, nibbles) in [
            (1usize, 4u32),
            (1 << 16, 4),
            ((1 << 16) + 1, 5),
            (1 << 20, 5),
            ((1 << 20) + 1, 6),
        ] {
            let mut storage = vec![0u8; 16];
            let mut w = BitWriter::new(&mut storage, 0);
            store_meta_block_header(len, false, &mut w);
            assert_eq!(w.position(), 1 + 2 + nibbles as usize * 4 + 1, "len {len}");
        }
    }

    #[test]
    fn one_pass_insert_codes_cover_every_boundary() {
        let depth = [4u8; 128];
        let bits = [0u16; 128];
        for (insert, code, extra) in [
            (0usize, 40usize, 0u32),
            (5, 45, 0),
            (6, 46, 1),
            (129, 55, 5),
            (130, 56, 6),
            (2113, 60, 10),
            (2114, 61, 12),
        ] {
            let (position, used) = one_pass_code(|histo, w| {
                one_pass::emit_insert_len(insert, &depth, &bits, histo, w);
            });
            assert_eq!(used, vec![code], "insert {insert}");
            assert_eq!(position, 4 + extra as usize, "insert {insert}");
        }
    }

    #[test]
    fn one_pass_long_insert_codes_cover_every_boundary() {
        let depth = [4u8; 128];
        let bits = [0u16; 128];
        for (insert, code, extra) in [
            (SHORT_INSERT_LIMIT, 62usize, 14u32),
            (LONG_INSERT_LIMIT - 1, 62, 14),
            (LONG_INSERT_LIMIT, 63, 24),
        ] {
            let (position, used) = one_pass_code(|histo, w| {
                one_pass::emit_long_insert_len(insert, &depth, &bits, histo, w);
            });
            assert_eq!(used, vec![code], "insert {insert}");
            assert_eq!(position, 4 + extra as usize);
        }
    }

    #[test]
    fn one_pass_copy_codes_cover_every_boundary() {
        let depth = [4u8; 128];
        let bits = [0u16; 128];
        for (copy, code) in [
            (4usize, 18usize),
            (9, 23),
            (10, 24),
            (133, 33),
            (134, 34),
            (2117, 38),
            (2118, 39),
        ] {
            let (_, used) = one_pass_code(|histo, w| {
                one_pass::emit_copy_len(copy, &depth, &bits, histo, w);
            });
            assert_eq!(used, vec![code], "copy {copy}");
        }
    }

    #[test]
    fn one_pass_last_distance_copy_codes_cover_every_boundary() {
        let depth = [4u8; 128];
        let bits = [0u16; 128];
        for (copy, expected) in [
            (4usize, vec![0usize]),
            (11, vec![7]),
            (12, vec![8]),
            (71, vec![15]),
            (72, vec![32, 64]),
            (135, vec![33, 64]),
            (136, vec![34, 64]),
            (2119, vec![38, 64]),
            (2120, vec![39, 64]),
        ] {
            let (_, mut used) = one_pass_code(|histo, w| {
                one_pass::emit_copy_len_last_distance(copy, &depth, &bits, histo, w);
            });
            used.sort_unstable();
            let mut expected = expected;
            expected.sort_unstable();
            assert_eq!(used, expected, "copy {copy}");
        }
    }

    #[test]
    fn one_pass_distance_codes_cover_every_boundary() {
        let depth = [4u8; 128];
        let bits = [0u16; 128];
        for (distance, code) in [(1usize, 80usize), (2, 80), (3, 81), (5, 82), (262_128, 111)] {
            let (_, used) = one_pass_code(|histo, w| {
                one_pass::emit_distance(distance, &depth, &bits, histo, w);
            });
            assert_eq!(used, vec![code], "distance {distance}");
        }
    }

    /// Emits one literal per bit-writer call, the shape both packers refine.
    fn literals_one_at_a_time(
        literals: &[u8],
        depth: &[u8; 256],
        bits: &[u16; 256],
        w: &mut BitWriter,
    ) {
        for &literal in literals {
            let index = usize::from(literal);
            w.write(u32::from(depth[index]), u64::from(bits[index]));
        }
    }

    /// Builds a literal code whose depths span the full one-to-fourteen range.
    fn literal_code() -> ([u8; 256], [u16; 256]) {
        let mut depth = [0u8; 256];
        let mut bits = [0u16; 256];
        for (index, (slot, code)) in depth.iter_mut().zip(bits.iter_mut()).enumerate() {
            *slot = 1 + (index % 14) as u8;
            *code = (index as u16) & ((1u16 << *slot) - 1);
        }
        (depth, bits)
    }

    #[test]
    fn packed_literals_match_one_code_per_call_at_every_run_length() {
        let (depth, bits) = literal_code();
        let data: Vec<u8> = (0..64u16).map(|i| (i * 7 % 256) as u8).collect();

        for len in 0..=data.len() {
            let mut packed_storage = vec![0u8; 256];
            let mut packed = BitWriter::new(&mut packed_storage, 0);
            one_pass::emit_literals(&data, 0, len, &depth, &bits, &mut packed);

            let mut plain_storage = vec![0u8; 256];
            let mut plain = BitWriter::new(&mut plain_storage, 0);
            literals_one_at_a_time(&data[..len], &depth, &bits, &mut plain);

            assert!(!packed.overflowed() && !plain.overflowed(), "length {len}");
            assert_eq!(packed.position(), plain.position(), "length {len}");
            let bytes = plain.position().div_ceil(8);
            assert_eq!(
                packed_storage[..bytes],
                plain_storage[..bytes],
                "length {len}"
            );
        }
    }

    #[test]
    fn packed_literals_stay_inside_the_writers_budget_at_the_deepest_codes() {
        // Four codes of the maximum depth are exactly the writer's limit, so
        // this is the widest call the quadruple loop can ever make.
        let depth = [14u8; 256];
        let mut bits = [0u16; 256];
        for (index, code) in bits.iter_mut().enumerate() {
            *code = (index as u16) & 0x3FFF;
        }
        let data: Vec<u8> = (0..16u8).map(|i| i.wrapping_mul(17)).collect();

        let mut storage = vec![0u8; 128];
        let mut w = BitWriter::new(&mut storage, 0);
        one_pass::emit_literals(&data, 0, data.len(), &depth, &bits, &mut w);
        assert!(!w.overflowed());
        assert_eq!(w.position(), data.len() * 14);
        assert_eq!(u32::from(depth[0]) * 4, MAX_BITS_PER_WRITE);
    }

    #[test]
    fn literal_runs_outside_the_input_emit_nothing() {
        let (depth, bits) = literal_code();
        let data = [1u8, 2, 3];
        let mut storage = vec![0u8; 32];
        let mut w = BitWriter::new(&mut storage, 0);
        one_pass::emit_literals(&data, 1, 99, &depth, &bits, &mut w);
        assert_eq!(w.position(), 0);
        assert!(!w.overflowed());
    }

    #[test]
    fn two_pass_insert_codes_decode_back_to_the_input_length() {
        for insert in [
            0usize, 1, 5, 6, 7, 129, 130, 2113, 2114, 6209, 6210, 22593, 22594, 100_000,
        ] {
            let mut commands = Vec::new();
            two_pass::emit_insert_len(insert, &mut commands);
            assert_eq!(commands.len(), 1);
            let code = (commands[0] & 0xFF) as usize;
            let extra = commands[0] >> 8;
            assert!(code < 24, "insert {insert} produced code {code}");
            assert!(extra < 1 << NUM_EXTRA_BITS[code], "insert {insert}");
            assert_eq!(
                INSERT_OFFSET[code] + extra,
                insert as u32,
                "insert {insert}"
            );
        }
    }

    #[test]
    fn two_pass_copy_codes_stay_inside_the_alphabet() {
        for copy in [4usize, 9, 10, 133, 134, 2117, 2118, 100_000] {
            let mut commands = Vec::new();
            two_pass::emit_copy_len(copy, &mut commands);
            assert_eq!(commands.len(), 1);
            let code = (commands[0] & 0xFF) as usize;
            assert!((24..64).contains(&code), "copy {copy} produced code {code}");
            assert!(commands[0] >> 8 < 1 << NUM_EXTRA_BITS[code]);
        }
    }

    #[test]
    fn two_pass_last_distance_copy_codes_append_the_reuse_symbol() {
        for (copy, words) in [
            (4usize, 1usize),
            (11, 1),
            (12, 1),
            (71, 1),
            (72, 2),
            (135, 2),
            (136, 2),
            (2119, 2),
            (2120, 2),
        ] {
            let mut commands = Vec::new();
            two_pass::emit_copy_len_last_distance(copy, &mut commands);
            assert_eq!(commands.len(), words, "copy {copy}");
            if words == 2 {
                assert_eq!(commands[1], 64);
            }
            let code = (commands[0] & 0xFF) as usize;
            assert!(code < 128);
            assert!(commands[0] >> 8 < 1 << NUM_EXTRA_BITS[code]);
        }
    }

    #[test]
    fn two_pass_distance_codes_stay_inside_the_alphabet() {
        for distance in [1usize, 2, 3, 5, 262_128] {
            let mut commands = Vec::new();
            two_pass::emit_distance(distance, &mut commands);
            let code = (commands[0] & 0xFF) as usize;
            assert!((80..128).contains(&code), "distance {distance}");
            assert!(commands[0] >> 8 < 1 << NUM_EXTRA_BITS[code]);
        }
    }
}

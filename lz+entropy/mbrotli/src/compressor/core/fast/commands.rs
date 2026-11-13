//! Insert, copy and distance code mapping shared by both fast qualities.
//!
//! Quality 0 writes command symbols straight into the bitstream while quality 1
//! buffers them as packed words, so the two families of helpers live side by
//! side here rather than being duplicated across the encoders. Both are direct
//! ports of `EmitInsertLen`, `EmitCopyLen`, `EmitCopyLenLastDistance` and
//! `EmitDistance` from the pinned reference.

use super::bits::{BitWriter, ByteBuffer};
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
    use super::super::bits::MAX_BITS_PER_WRITE;
    use super::{BitWriter, ByteBuffer, LONG_INSERT_LIMIT, log2_floor_non_zero};

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

    /// A command symbol and the number of extra bits that follow it.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub(crate) struct LengthCode {
        /// Symbol in the 128 entry fast alphabet.
        code: u8,
        /// Extra bits after the symbol.
        extra_bits: u8,
    }

    /// Builds the table of a length family whose direct codes end at
    /// `direct` and whose two-prefix codes take over from there.
    ///
    /// Below `direct` the symbol is `length + direct_base` with no extra
    /// bits; from there `tail = length - tail_bias` has `log2(tail) - 1`
    /// extra bits and the symbol `2 * nbits + (tail >> nbits) + prefix_base`.
    /// These are the first two branches of `EmitInsertLen`, `EmitCopyLen`
    /// and `EmitCopyLenLastDistance`, tabulated so that picking between them
    /// is a load rather than a branch on the length.
    const fn length_codes<const N: usize>(
        direct: usize,
        direct_base: isize,
        tail_bias: usize,
        prefix_base: usize,
    ) -> [LengthCode; N] {
        let mut table = [LengthCode {
            code: 0,
            extra_bits: 0,
        }; N];
        let mut length = 0;
        while length < N {
            table[length] = if length < direct {
                LengthCode {
                    // Lengths the encoder never emits wrap into the alphabet.
                    code: ((length as isize + direct_base) & 127) as u8,
                    extra_bits: 0,
                }
            } else {
                let tail = length - tail_bias;
                let nbits = log2_floor_non_zero(tail) - 1;
                LengthCode {
                    code: (((nbits as usize) << 1) + (tail >> nbits) + prefix_base) as u8,
                    extra_bits: nbits as u8,
                }
            };
            length += 1;
        }
        table
    }

    /// Insert lengths the tables cover (the first two branches of
    /// [`emit_insert_len`]).
    const TABLED_INSERTS: usize = 130;

    /// Copy lengths after an explicit distance the tables cover (the first
    /// two branches of [`emit_copy_len`]).
    const TABLED_COPIES: usize = 134;

    /// Copy lengths at the last distance the tables cover (the first two
    /// branches of [`emit_copy_len_last_distance`]).
    const TABLED_LAST_COPIES: usize = 72;

    /// Symbols of insert lengths below [`TABLED_INSERTS`].
    static INSERT_CODES: [LengthCode; TABLED_INSERTS] = length_codes(6, 40, 2, 42);

    /// Symbols of copy lengths below [`TABLED_COPIES`].
    static COPY_CODES: [LengthCode; TABLED_COPIES] = length_codes(10, 14, 6, 20);

    /// Symbols of last-distance copy lengths below [`TABLED_LAST_COPIES`].
    static LAST_COPY_CODES: [LengthCode; TABLED_LAST_COPIES] = length_codes(12, -4, 8, 4);

    /// Literals [`emit_command`] packs next to the insert symbol.
    const PACKED_LITERALS: usize = 4;

    /// Returns a tabled length's symbol, its width and its bits, extra bits
    /// included. `tail` is the length less the family's tail bias.
    ///
    /// The extra bits of a two-prefix code are the low `nbits` bits of the
    /// tail, which is `tail - (prefix << nbits)`; a direct code has none, so
    /// the mask is empty and a tail that wrapped below zero contributes
    /// nothing.
    #[inline(always)]
    fn tabled(
        entry: LengthCode,
        tail: usize,
        depth: &[u8; 128],
        bits: &[u16; 128],
    ) -> (usize, u32, u64) {
        let code = symbol(usize::from(entry.code));
        let extra_bits = u32::from(entry.extra_bits);
        let width = u32::from(depth[code]);
        let extra = tail as u64 & ((1u64 << extra_bits) - 1);
        (
            code,
            width + extra_bits,
            u64::from(bits[code]) | (extra << width),
        )
    }

    /// Returns the symbol, width and bits of a backward distance, extra bits
    /// included: what [`emit_distance`] writes.
    #[inline(always)]
    fn distance_symbol(distance: usize, depth: &[u8; 128], bits: &[u16; 128]) -> (usize, u32, u64) {
        let d = distance + 3;
        let nbits = log2_floor_non_zero(d) - 1;
        let prefix = (d >> nbits) & 1;
        let offset = (2 + prefix) << nbits;
        let code = symbol(2 * (nbits as usize - 1) + prefix + 80);
        let width = u32::from(depth[code]);
        (
            code,
            width + nbits,
            u64::from(bits[code]) | (((d - offset) as u64) << width),
        )
    }

    /// Writes a whole command of a serial fragment: the insert length, its
    /// literals, the distance (or the symbol that repeats the last one) and
    /// the copy length.
    ///
    /// Returns `false`, having written nothing, for a command the tables do
    /// not cover: an insert of [`TABLED_INSERTS`] or more, a copy of
    /// [`TABLED_LAST_COPIES`] or more, or literals too close to the end of
    /// `data` to load as a group. The caller then emits it field by field.
    ///
    /// The bits are those of [`emit_insert_len`], [`emit_literals`], the
    /// distance and [`emit_copy_len_last_distance`] in that order. What
    /// differs is how they are reached. Each length family's symbol comes
    /// from a table, so which of its code ranges a length falls in is not a
    /// branch; the distance and the last-distance symbol are selected, not
    /// branched on; and the first [`PACKED_LITERALS`] literal codes are
    /// computed for every command, the ones past the run masked to zero bits.
    /// On the Silesia corpus 72% of these commands insert at most four
    /// literals and 83% copy fewer than twelve bytes, so the lengths vary too
    /// much to predict and too little to be worth a loop. Fields that fit the
    /// writer's 56 bits together share a store: a typical command is one
    /// store where it was four.
    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "one command's fields and the code tables they are written with"
    )]
    pub(crate) fn emit_command(
        data: &[u8],
        start: usize,
        insertlen: usize,
        distance: usize,
        repeats_distance: bool,
        copylen: usize,
        lit_depth: &[u8; 256],
        lit_bits: &[u16; 256],
        depth: &[u8; 128],
        bits: &[u16; 128],
        histo: &mut [u32; 128],
        w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
    ) -> bool {
        let (Some(&insert), Some(&copy), Some(first)) = (
            INSERT_CODES.get(insertlen),
            LAST_COPY_CODES.get(copylen),
            data.get(start..)
                .and_then(|rest| rest.first_chunk::<PACKED_LITERALS>()),
        ) else {
            return false;
        };
        let (insert_code, insert_width, insert_value) =
            tabled(insert, insertlen.wrapping_sub(2), depth, bits);
        let (copy_code, copy_width, copy_value) =
            tabled(copy, copylen.wrapping_sub(8), depth, bits);
        let (explicit_code, explicit_width, explicit_value) =
            distance_symbol(distance, depth, bits);
        let (distance_code, distance_width, distance_value) = if repeats_distance {
            (64, u32::from(depth[64]), u64::from(bits[64]))
        } else {
            (explicit_code, explicit_width, explicit_value)
        };
        histo[insert_code] += 1;
        histo[distance_code] += 1;
        histo[copy_code] += 1;

        let mut literal_width = 0u32;
        let mut literal_value = 0u64;
        for (index, &literal) in first.iter().enumerate() {
            let keep = 0u32.wrapping_sub(u32::from(index < insertlen));
            let literal = usize::from(literal);
            literal_value |= u64::from(u32::from(lit_bits[literal]) & keep) << literal_width;
            literal_width += u32::from(lit_depth[literal]) & keep;
        }

        // At most 14 + 16 and 15 + 5 bits: always one write.
        let tail_width = distance_width + copy_width;
        let tail_value = distance_value | (copy_value << distance_width);
        let head_width = insert_width + literal_width;
        if insertlen <= PACKED_LITERALS && head_width + tail_width <= MAX_BITS_PER_WRITE {
            w.write(
                head_width + tail_width,
                insert_value | (literal_value << insert_width) | (tail_value << head_width),
            );
            return true;
        }
        w.write_pair(insert_width, insert_value, literal_width, literal_value);
        if insertlen > PACKED_LITERALS {
            emit_literals(
                data,
                start + PACKED_LITERALS,
                insertlen - PACKED_LITERALS,
                lit_depth,
                lit_bits,
                w,
            );
        }
        w.write(tail_width, tail_value);
        true
    }

    /// Writes a copy length and the explicit distance that follows it with
    /// one store: [`emit_copy_len`] then [`emit_distance`].
    ///
    /// Returns `false`, having written nothing, for a copy of
    /// [`TABLED_COPIES`] or more, which the caller emits field by field. A
    /// tabled copy is at most 15 + 5 bits and a distance at most 14 + 16, so
    /// the pair always fits the writer's 56 bits.
    #[inline(always)]
    pub(crate) fn emit_copy_and_distance(
        copylen: usize,
        distance: usize,
        depth: &[u8; 128],
        bits: &[u16; 128],
        histo: &mut [u32; 128],
        w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
    ) -> bool {
        let Some(&copy) = COPY_CODES.get(copylen) else {
            return false;
        };
        let (copy_code, copy_width, copy_value) =
            tabled(copy, copylen.wrapping_sub(6), depth, bits);
        let (distance_code, distance_width, distance_value) =
            distance_symbol(distance, depth, bits);
        histo[copy_code] += 1;
        histo[distance_code] += 1;
        w.write(
            copy_width + distance_width,
            copy_value | (distance_value << copy_width),
        );
        true
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

    /// Emits `literals` through the literal prefix code, four codes per
    /// bit-writer call.
    ///
    /// Literal codes are at most 14 bits deep (the fast builder's limit), so
    /// four fill at most the writer's 56 bits and the up to three left over
    /// fit in one more call. Fixed groups replace greedy packing, whose
    /// per-literal overflow test cost more than the calls it saved: binary
    /// input, where most bytes reach this path as literals, compresses about
    /// 10% faster at quality 1.
    #[inline(always)]
    pub(crate) fn pack_literals(
        literals: &[u8],
        depth: &[u8; 256],
        bits: &[u16; 256],
        w: &mut BitWriter<'_, impl ByteBuffer + ?Sized>,
    ) {
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
        let mut value = 0u64;
        let mut width = 0u32;
        for &literal in rest {
            let index = usize::from(literal);
            value |= u64::from(bits[index]) << width;
            width += u32::from(depth[index]);
        }
        w.write(width, value);
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn length_tables_hold_the_first_two_code_ranges_of_each_family() {
            // Evaluated here at run time; the statics were built by the
            // compiler from the same calls.
            let inserts: [LengthCode; TABLED_INSERTS] = length_codes(6, 40, 2, 42);
            let copies: [LengthCode; TABLED_COPIES] = length_codes(10, 14, 6, 20);
            let last_copies: [LengthCode; TABLED_LAST_COPIES] = length_codes(12, -4, 8, 4);
            assert_eq!(inserts, INSERT_CODES);
            assert_eq!(copies, COPY_CODES);
            assert_eq!(last_copies, LAST_COPY_CODES);

            // The ends of the direct and of the two-prefix ranges, as
            // `emit_insert_len`, `emit_copy_len` and
            // `emit_copy_len_last_distance` code them.
            let entry = |code, extra_bits| LengthCode { code, extra_bits };
            assert_eq!(inserts[0], entry(40, 0));
            assert_eq!(inserts[5], entry(45, 0));
            assert_eq!(inserts[6], entry(46, 1));
            assert_eq!(inserts[129], entry(55, 5));
            assert_eq!(copies[4], entry(18, 0));
            assert_eq!(copies[9], entry(23, 0));
            assert_eq!(copies[10], entry(24, 1));
            assert_eq!(copies[133], entry(33, 5));
            assert_eq!(last_copies[4], entry(0, 0));
            assert_eq!(last_copies[11], entry(7, 0));
            assert_eq!(last_copies[12], entry(8, 1));
            assert_eq!(last_copies[71], entry(15, 4));
        }
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
    use super::super::bits::MAX_BITS_PER_WRITE;
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

    /// Code tables with every depth the tree builders can produce.
    ///
    /// `deep` gives every symbol its longest code — fifteen bits for a
    /// command, fourteen for a distance or a literal — which is what pushes
    /// a whole command past one write.
    fn command_tables(deep: bool) -> ([u8; 128], [u16; 128], [u8; 256], [u16; 256]) {
        let depth: [u8; 128] = core::array::from_fn(|code| match (deep, code < 64) {
            (true, true) => 15,
            (true, false) => 14,
            (false, _) => (code % 9 + 1) as u8,
        });
        let bits: [u16; 128] =
            core::array::from_fn(|code| (code as u16 * 211 + 5) & ((1u16 << depth[code]) - 1));
        let lit_depth: [u8; 256] =
            core::array::from_fn(|literal| if deep { 14 } else { (literal % 12 + 1) as u8 });
        let lit_bits: [u16; 256] = core::array::from_fn(|literal| {
            (literal as u16 * 97 + 3) & ((1u16 << lit_depth[literal]) - 1)
        });
        (depth, bits, lit_depth, lit_bits)
    }

    /// Runs `emit` on a zeroed buffer and returns its bytes, the final
    /// position and the histogram.
    fn emitted(emit: impl FnOnce(&mut [u32; 128], &mut BitWriter)) -> (Vec<u8>, usize, [u32; 128]) {
        let mut storage = vec![0u8; 512];
        let mut histo = [0u32; 128];
        let position = {
            let mut w = BitWriter::new(&mut storage, 0);
            // An odd start exercises every shift.
            w.write(3, 5);
            emit(&mut histo, &mut w);
            assert!(!w.overflowed());
            w.position()
        };
        (storage, position, histo)
    }

    #[test]
    fn whole_commands_match_field_by_field_emission_on_both_sides_of_every_table() {
        let data: Vec<u8> = (0..400u32).map(|index| (index * 131 + 7) as u8).collect();
        let inserts: Vec<usize> = if cfg!(miri) {
            vec![0, 1, 4, 5, 6, 129, 130]
        } else {
            (0..=140).collect()
        };
        let copies: Vec<usize> = if cfg!(miri) {
            vec![5, 11, 12, 71, 72]
        } else {
            (4..=80).collect()
        };
        for deep in [false, true] {
            let (depth, bits, lit_depth, lit_bits) = command_tables(deep);
            for &insert in &inserts {
                for &copy in &copies {
                    for (distance, repeats) in [(1usize, false), (77, true), (262_128, false)] {
                        if cfg!(miri) && distance == 1 {
                            continue;
                        }
                        let start = 3;
                        let mut handled = false;
                        let actual = emitted(|histo, w| {
                            handled = one_pass::emit_command(
                                &data, start, insert, distance, repeats, copy, &lit_depth,
                                &lit_bits, &depth, &bits, histo, w,
                            );
                        });
                        assert_eq!(
                            handled,
                            insert < 130 && copy < 72,
                            "insert {insert}, copy {copy}"
                        );
                        if !handled {
                            // A refused command writes nothing.
                            assert_eq!(actual.1, 3, "insert {insert}, copy {copy}");
                            assert_eq!(actual.2, [0; 128]);
                            continue;
                        }
                        let expected = emitted(|histo, w| {
                            one_pass::emit_insert_len(insert, &depth, &bits, histo, w);
                            one_pass::emit_literals(&data, start, insert, &lit_depth, &lit_bits, w);
                            if repeats {
                                w.write(u32::from(depth[64]), u64::from(bits[64]));
                                histo[64] += 1;
                            } else {
                                one_pass::emit_distance(distance, &depth, &bits, histo, w);
                            }
                            one_pass::emit_copy_len_last_distance(copy, &depth, &bits, histo, w);
                        });
                        assert_eq!(
                            actual, expected,
                            "deep {deep}, insert {insert}, copy {copy}, distance {distance}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_whole_command_is_refused_when_its_literals_cannot_be_loaded_together() {
        let (depth, bits, lit_depth, lit_bits) = command_tables(false);
        let data = [1u8, 2, 3, 4, 5];
        let attempt = |start: usize| {
            let mut handled = false;
            let written = emitted(|histo, w| {
                handled = one_pass::emit_command(
                    &data, start, 1, 9, false, 5, &lit_depth, &lit_bits, &depth, &bits, histo, w,
                );
            });
            (handled, written.1)
        };
        let (handled, position) = attempt(1);
        assert!(handled);
        assert!(position > 3);
        // Three bytes are left: fewer than the group the emitter loads.
        assert_eq!(attempt(2), (false, 3));
        assert_eq!(attempt(6), (false, 3));
    }

    #[test]
    fn copies_with_a_distance_match_field_by_field_emission_on_both_sides_of_the_table() {
        let copies: Vec<usize> = if cfg!(miri) {
            vec![5, 9, 10, 133, 134]
        } else {
            (2..=140).collect()
        };
        for deep in [false, true] {
            let (depth, bits, _, _) = command_tables(deep);
            for &copy in &copies {
                for distance in [1usize, 2, 3, 5, 4093, 262_128] {
                    let mut handled = false;
                    let actual = emitted(|histo, w| {
                        handled = one_pass::emit_copy_and_distance(
                            copy, distance, &depth, &bits, histo, w,
                        );
                    });
                    assert_eq!(handled, copy < 134, "copy {copy}");
                    if !handled {
                        assert_eq!(actual.1, 3, "copy {copy}");
                        assert_eq!(actual.2, [0; 128]);
                        continue;
                    }
                    let expected = emitted(|histo, w| {
                        one_pass::emit_copy_len(copy, &depth, &bits, histo, w);
                        one_pass::emit_distance(distance, &depth, &bits, histo, w);
                    });
                    assert_eq!(
                        actual, expected,
                        "deep {deep}, copy {copy}, distance {distance}"
                    );
                }
            }
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

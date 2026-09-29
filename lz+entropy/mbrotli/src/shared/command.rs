//! Commands: one insert-and-copy pair, with its prefix codes precomputed.
//!
//! Ports `c/enc/command.h` and `c/enc/prefix.h` of the pinned reference
//! (`google/brotli` v1.2.0, commit `028fb5a`). A command stores the codes it
//! will be written with rather than the raw lengths, because the meta-block
//! builder histograms the codes long before the bit writer sees them.

use super::distance::{DistanceParams, NUM_DISTANCE_SHORT_CODES};
use super::fast_log::log2_floor_non_zero;
use super::format::{COPY_BASE, COPY_EXTRA, INS_BASE, INS_EXTRA};

/// Mask of the copy length inside [`Command::copy_len`].
const COPY_LEN_MASK: u32 = 0x1FF_FFFF;

/// Shift of the copy-length code delta inside [`Command::copy_len`].
const COPY_LEN_CODE_SHIFT: u32 = 25;

/// Mask of the distance code inside [`Command::dist_prefix`].
const DIST_CODE_MASK: u16 = 0x3FF;

/// Returns the prefix code of an insert length (`GetInsertLengthCode`).
#[inline(always)]
pub(crate) const fn insert_length_code(insertlen: usize) -> u16 {
    if insertlen < 6 {
        insertlen as u16
    } else if insertlen < 130 {
        let nbits = log2_floor_non_zero(insertlen - 2) - 1;
        ((nbits << 1) + ((insertlen as u32 - 2) >> nbits) + 2) as u16
    } else if insertlen < 2114 {
        (log2_floor_non_zero(insertlen - 66) + 10) as u16
    } else if insertlen < 6210 {
        21
    } else if insertlen < 22594 {
        22
    } else {
        23
    }
}

/// Copy-length classes below the final unbounded code. The table removes
/// logarithms from the repeated length-pricing loop in the Zopfli search.
const COPY_LENGTH_CODES: [u8; 2118] = {
    let mut codes = [0; 2118];
    let mut length = 2;
    let mut code = 0;
    while length < codes.len() {
        if length >= COPY_BASE[code + 1] as usize {
            code += 1;
        }
        codes[length] = code as u8;
        length += 1;
    }
    codes
};

/// Returns the prefix code of a copy length (`GetCopyLengthCode`).
#[inline(always)]
pub(crate) const fn copy_length_code(copylen: usize) -> u16 {
    if copylen < 10 {
        (copylen - 2) as u16
    } else if copylen < COPY_LENGTH_CODES.len() {
        COPY_LENGTH_CODES[copylen] as u16
    } else {
        23
    }
}

/// Folds an insert and a copy code into one command symbol.
///
/// Mirrors `CombineLengthCodes`, magic constant included: the table in RFC 7932
/// section 5 is encoded as a shifted bit pattern rather than spelled out.
#[inline(always)]
pub(crate) const fn combine_length_codes(
    inscode: u16,
    copycode: u16,
    use_last_distance: bool,
) -> u16 {
    let bits64 = (copycode & 0x7) | ((inscode & 0x7) << 3);
    if use_last_distance && inscode < 8 && copycode < 16 {
        if copycode < 8 { bits64 } else { bits64 | 64 }
    } else {
        let offset = 2u32 * ((copycode as u32 >> 3) + 3 * (inscode as u32 >> 3));
        let offset = (offset << 5) + 0x40 + ((0x52_0D40u32 >> offset) & 0xC0);
        (offset as u16) | bits64
    }
}

/// Returns the command symbol for an insert and copy length.
#[inline(always)]
pub(crate) const fn length_code(insertlen: usize, copylen: usize, use_last_distance: bool) -> u16 {
    combine_length_codes(
        insert_length_code(insertlen),
        copy_length_code(copylen),
        use_last_distance,
    )
}

/// Splits an intermediate distance code into a prefix code and extra bits.
///
/// Mirrors `PrefixEncodeCopyDistance`. The returned code packs the number of
/// extra bits in its top six bits and the symbol in its low ten.
#[inline(always)]
pub(crate) const fn prefix_encode_copy_distance(
    distance_code: usize,
    num_direct_codes: u32,
    postfix_bits: u32,
) -> (u16, u32) {
    if distance_code < (NUM_DISTANCE_SHORT_CODES + num_direct_codes) as usize {
        return (distance_code as u16, 0);
    }
    let dist = (1usize << (postfix_bits + 2))
        + (distance_code - NUM_DISTANCE_SHORT_CODES as usize - num_direct_codes as usize);
    let bucket = log2_floor_non_zero(dist) - 1;
    let postfix_mask = (1usize << postfix_bits) - 1;
    let postfix = dist & postfix_mask;
    let prefix = (dist >> bucket) & 1;
    let offset = (2 + prefix) << bucket;
    let nbits = bucket - postfix_bits;
    let code = ((nbits as usize) << 10)
        | (NUM_DISTANCE_SHORT_CODES as usize
            + num_direct_codes as usize
            + ((2 * (nbits as usize - 1) + prefix) << postfix_bits)
            + postfix);
    (code as u16, ((dist - offset) >> postfix_bits) as u32)
}

/// One insert-and-copy command, with its prefix codes already resolved.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct Command {
    /// Number of literals inserted before the copy.
    pub(crate) insert_len: u32,
    /// Copy length in the low twenty-five bits, code delta in the top seven.
    pub(crate) copy_len: u32,
    /// Extra bits of the distance code.
    pub(crate) dist_extra: u32,
    /// Command prefix symbol.
    pub(crate) cmd_prefix: u16,
    /// Distance code in the low ten bits, extra-bit count in the top six.
    pub(crate) dist_prefix: u16,
}

impl Command {
    /// Creates a command copying `copylen` bytes from `distance_code`.
    ///
    /// `copylen_code_delta` carries the difference between the length the
    /// decoder will reconstruct and the length actually copied, which only a
    /// static-dictionary match ever sets.
    ///
    /// Mirrors `InitCommand`: the distance prefix is computed as if there were
    /// no postfix or direct codes, and is recomputed later only if the
    /// meta-block ends up using different distance parameters.
    #[inline(always)]
    pub(crate) const fn new(
        dist: &DistanceParams,
        insertlen: usize,
        copylen: usize,
        copylen_code_delta: i32,
        distance_code: usize,
    ) -> Self {
        let delta = (copylen_code_delta as i8) as u8 as u32;
        // Values 32..=63 cannot describe a normal static-word length delta.
        // Reserve them for the absolute base-word length of long RFC 9841
        // transforms, retaining the reference layout for ordinary commands.
        #[cfg(feature = "experimental")]
        let delta = if copylen_code_delta < -64 {
            32 + (copylen as i64 + copylen_code_delta as i64) as u32
        } else {
            delta
        };
        let (dist_prefix, dist_extra) =
            prefix_encode_copy_distance(distance_code, dist.num_direct, dist.postfix_bits);
        let cmd_prefix = length_code(
            insertlen,
            (copylen as i64 + copylen_code_delta as i64) as usize,
            (dist_prefix & DIST_CODE_MASK) == 0,
        );
        Self {
            insert_len: insertlen as u32,
            copy_len: (copylen as u32) | (delta << COPY_LEN_CODE_SHIFT),
            dist_extra,
            cmd_prefix,
            dist_prefix,
        }
    }

    /// Creates the trailing command that only inserts literals.
    ///
    /// Mirrors `InitInsertCommand`: the copy length is zero but the code is
    /// four, and the distance code is the first non-short one so no distance
    /// symbol is ever written.
    #[inline(always)]
    pub(crate) const fn insert_only(insertlen: usize) -> Self {
        Self {
            insert_len: insertlen as u32,
            copy_len: 4 << COPY_LEN_CODE_SHIFT,
            dist_extra: 0,
            cmd_prefix: length_code(insertlen, 4, false),
            dist_prefix: NUM_DISTANCE_SHORT_CODES as u16,
        }
    }

    /// Returns the number of bytes this command copies.
    #[inline(always)]
    pub(crate) const fn copy_len(&self) -> u32 {
        self.copy_len & COPY_LEN_MASK
    }

    /// Returns the copy length the decoder reconstructs.
    #[inline(always)]
    pub(crate) const fn copy_len_code(&self) -> u32 {
        let modifier = self.copy_len >> COPY_LEN_CODE_SHIFT;
        #[cfg(feature = "experimental")]
        if modifier >= 32 && modifier <= 63 {
            return modifier - 32;
        }
        let delta = ((modifier | ((modifier & 0x40) << 1)) as u8) as i8 as i32;
        ((self.copy_len & COPY_LEN_MASK) as i32 + delta) as u32
    }

    /// Returns the distance code without its extra-bit count.
    #[inline(always)]
    pub(crate) const fn distance_code(&self) -> u16 {
        self.dist_prefix & DIST_CODE_MASK
    }

    /// Returns how many extra bits the distance code carries.
    #[inline(always)]
    pub(crate) const fn distance_extra_bits(&self) -> u32 {
        (self.dist_prefix >> 10) as u32
    }

    /// Returns whether this command emits a distance symbol at all.
    #[inline(always)]
    pub(crate) const fn has_distance(&self) -> bool {
        self.copy_len() != 0 && self.cmd_prefix >= 128
    }

    /// Returns the context a distance symbol is coded in.
    ///
    /// Mirrors `CommandDistanceContext`: short copies from a fresh distance get
    /// their own context, everything else shares the last one.
    #[inline(always)]
    pub(crate) const fn distance_context(&self) -> usize {
        let r = self.cmd_prefix >> 6;
        let c = self.cmd_prefix & 7;
        if (r == 0 || r == 2 || r == 4 || r == 7) && c <= 2 {
            c as usize
        } else {
            3
        }
    }

    /// Rebuilds the intermediate distance code this command was created from.
    ///
    /// Mirrors `CommandRestoreDistanceCode`, which the meta-block builder needs
    /// when it re-encodes distances under different parameters.
    #[inline(always)]
    pub(crate) const fn restore_distance_code(&self, dist: &DistanceParams) -> u32 {
        let dcode = (self.dist_prefix & DIST_CODE_MASK) as u32;
        if dcode < NUM_DISTANCE_SHORT_CODES + dist.num_direct {
            return dcode;
        }
        let nbits = (self.dist_prefix >> 10) as u32;
        let extra = self.dist_extra;
        let postfix_mask = (1u32 << dist.postfix_bits) - 1;
        let base = dcode - dist.num_direct - NUM_DISTANCE_SHORT_CODES;
        let hcode = base >> dist.postfix_bits;
        let lcode = base & postfix_mask;
        let offset = ((2 + (hcode & 1)) << nbits) - 4;
        ((offset + extra) << dist.postfix_bits) + lcode + dist.num_direct + NUM_DISTANCE_SHORT_CODES
    }

    /// Writes the extra bits of the insert and copy lengths.
    ///
    /// Mirrors `StoreCommandExtra`; both fields together never exceed the bit
    /// writer's single-call limit.
    #[inline(always)]
    pub(crate) fn extra_bits(&self) -> (u32, u64) {
        let copylen_code = self.copy_len_code();
        let inscode = insert_length_code(self.insert_len as usize);
        let copycode = copy_length_code(copylen_code as usize);
        let insnumextra = INS_EXTRA[inscode as usize];
        let insextraval = u64::from(self.insert_len - INS_BASE[inscode as usize]);
        let copyextraval = u64::from(copylen_code - COPY_BASE[copycode as usize]);
        (
            insnumextra + COPY_EXTRA[copycode as usize],
            (copyextraval << insnumextra) | insextraval,
        )
    }
}

/// Grows the last command over input that continues its copy.
///
/// Mirrors `ExtendLastCommand` of `c/enc/encode.c`. When a block boundary falls
/// in the middle of a repeat, the bytes after it would otherwise start a fresh
/// command; extending the previous one instead costs nothing and saves a
/// command. Every quality above one does this, so it belongs beside the command
/// rather than in one encoder.
///
/// `span` is the stretch the search is about to process; the bytes absorbed
/// here are removed from its front.
pub(crate) struct CommandExtension<'a> {
    pub(crate) command: &'a mut Command,
    pub(crate) lgwin: usize,
    pub(crate) dist: &'a DistanceParams,
    pub(crate) last_distance: i32,
    pub(crate) window: super::ringbuffer::Window<'a>,
    pub(crate) last_processed_pos: u64,
    pub(crate) attached: Option<&'a crate::compressor::core::rfc9841::context::SharedContextInner>,
    pub(crate) span: &'a mut super::ringbuffer::BlockSpan,
}

/// Extends a copy with the retained encoder's selected comparison kernel.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
#[inline(always)]
pub(crate) fn extend_last_command<S: fearless_simd::Simd>(simd: S, input: CommandExtension<'_>) {
    let CommandExtension {
        command,
        lgwin,
        dist,
        last_distance,
        window,
        last_processed_pos,
        attached,
        span,
    } = input;
    let super::ringbuffer::Window {
        data: ringbuffer,
        mask,
    } = window;
    let max_backward_distance = (1u64 << lgwin) - 16;
    let last_copy_len = u64::from(command.copy_len());
    let last_processed_pos = last_processed_pos - last_copy_len;
    let max_distance = last_processed_pos.min(max_backward_distance);
    let cmd_dist = u64::from(last_distance as u32);
    let distance_code = command.restore_distance_code(dist);

    if u64::from(distance_code) < u64::from(NUM_DISTANCE_SHORT_CODES)
        || u64::from(distance_code) - u64::from(NUM_DISTANCE_SHORT_CODES - 1) == cmd_dist
    {
        if cmd_dist <= max_distance {
            while span.bytes != 0 {
                let here = span.position as usize & mask;
                let there = (u64::from(span.position) - cmd_dist) as usize & mask;
                // Bound both contiguous slices at the physical wrap point.
                // Re-entering the loop handles either side wrapping first.
                let count = (span.bytes as usize)
                    .min(mask - here + 1)
                    .min(mask - there + 1);
                let matched =
                    super::match_len::find_match_length(simd, ringbuffer, here, there, count)
                        as u32;
                command.copy_len += matched;
                span.bytes -= matched;
                span.position += matched;
                if matched as usize != count {
                    break;
                }
            }
        } else {
            extend_into_prefix(
                command,
                attached,
                cmd_dist,
                max_distance,
                last_copy_len,
                ringbuffer,
                mask,
                span,
            );
        }
        // The copy length changed, so the command symbol has to be recomputed.
        // `ExtendLastCommand` reads the length-code delta as an unsigned field
        // here, which agrees with the signed reading everywhere it can occur:
        // only a dictionary match sets it, and never to a negative value.
        command.cmd_prefix = length_code(
            command.insert_len as usize,
            (command.copy_len & COPY_LEN_MASK) as usize
                + (command.copy_len >> COPY_LEN_CODE_SHIFT) as usize,
            command.distance_code() == 0,
        );
    }
}

/// Runs a copy that addresses the attached dictionary on past its command.
///
/// The distance is beyond the window, so it addresses the concatenated prefix
/// instead. `ExtendLastCommand` walks that concatenation across attachment
/// seams — unlike the *search*, which stops at the end of the attachment a
/// candidate was found in.
///
/// Does nothing when nothing is attached, when the distance lands past the
/// concatenation, or when the copy already reached back further than the
/// window boundary, which are the reference's three guards.
#[expect(
    clippy::too_many_arguments,
    reason = "the branch of ExtendLastCommand it mirrors needs all of them"
)]
fn extend_into_prefix(
    command: &mut Command,
    attached: Option<&crate::compressor::core::rfc9841::context::SharedContextInner>,
    cmd_dist: u64,
    max_distance: u64,
    last_copy_len: u64,
    ringbuffer: &[u8],
    mask: usize,
    span: &mut super::ringbuffer::BlockSpan,
) {
    let Some(context) = attached else {
        return;
    };
    let compound = context.total_size() as u64;
    let reach = cmd_dist - max_distance;
    if reach > compound || last_copy_len >= reach {
        return;
    }
    let sources = context.dictionaries().prefix();
    let mut address = compound - reach + last_copy_len;
    loop {
        let run = sources.run_from(address);
        if run.is_empty() {
            return;
        }
        for &byte in run {
            if span.bytes == 0 {
                return;
            }
            let here = usize::try_from(u64::from(span.position)).unwrap_or(0);
            let Some(&current) = ringbuffer.get(here & mask) else {
                return;
            };
            if current != byte {
                return;
            }
            command.copy_len += 1;
            span.bytes -= 1;
            span.position += 1;
            address += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extending_a_copy_matches_the_byte_scan_across_both_wrap_points() {
        use crate::compressor::Backend;
        use crate::compressor::core::dispatch;
        use crate::shared::ringbuffer::{BlockSpan, Window};

        let dist = DistanceParams::default();
        for backend in Backend::available() {
            let kernels = dispatch::select(backend.0);
            for position in [256, 257, 319, 383, 447, 510, 511] {
                for distance in [1, 4, 17, 63, 128] {
                    for mismatch in [0, 1, 7, 15, 16, 31, 32, 63, 127, 255] {
                        let mut data = [7; 256];
                        data[(position + mismatch) & 255] = 8;
                        let matched = (0..192)
                            .take_while(|&i| {
                                data[(position + i) & 255] == data[(position + i - distance) & 255]
                            })
                            .count();
                        let mut command = Command::new(&dist, 0, 4, 0, 0);
                        let mut span = BlockSpan {
                            position: position as u32,
                            bytes: 192,
                        };
                        kernels.extend(CommandExtension {
                            command: &mut command,
                            lgwin: 10,
                            dist: &dist,
                            last_distance: distance as i32,
                            window: Window {
                                data: &data,
                                mask: 255,
                            },
                            last_processed_pos: position as u64,
                            attached: None,
                            span: &mut span,
                        });
                        assert_eq!(command.copy_len(), 4 + matched as u32, "{backend:?}");
                        assert_eq!(span.bytes, 192 - matched as u32);
                        assert_eq!(span.position, (position + matched) as u32);
                    }
                }
            }
        }
    }

    #[test]
    fn insert_length_codes_cover_every_range() {
        assert_eq!(insert_length_code(0), 0);
        assert_eq!(insert_length_code(5), 5);
        assert_eq!(insert_length_code(6), 6);
        assert_eq!(insert_length_code(129), 15);
        assert_eq!(insert_length_code(130), 16);
        assert_eq!(insert_length_code(2113), 20);
        assert_eq!(insert_length_code(2114), 21);
        assert_eq!(insert_length_code(6209), 21);
        assert_eq!(insert_length_code(6210), 22);
        assert_eq!(insert_length_code(22593), 22);
        assert_eq!(insert_length_code(22594), 23);
    }

    #[test]
    fn insert_and_copy_codes_agree_with_their_bases() {
        for len in 0usize..3000 {
            let code = insert_length_code(len) as usize;
            let base = INS_BASE[code] as usize;
            let extra = INS_EXTRA[code];
            assert!(base <= len, "insert {len} below base of code {code}");
            assert!(
                (len - base) < (1usize << extra),
                "insert {len} outside code {code}"
            );
        }
        for len in 2usize..3000 {
            let code = copy_length_code(len) as usize;
            let base = COPY_BASE[code] as usize;
            let extra = COPY_EXTRA[code];
            assert!(base <= len, "copy {len} below base of code {code}");
            assert!(
                (len - base) < (1usize << extra),
                "copy {len} outside code {code}"
            );
        }
    }

    #[test]
    fn command_symbols_stay_inside_the_alphabet() {
        for insert in [0usize, 5, 6, 100, 2000, 30_000] {
            for copy in [2usize, 9, 10, 100, 2000, 30_000] {
                for last in [false, true] {
                    assert!(length_code(insert, copy, last) < 704);
                }
            }
        }
    }

    #[test]
    fn distance_prefixes_round_trip_through_restore() {
        for postfix_bits in 0u32..=3 {
            for direct in [0u32, 4, 8, 12] {
                let direct = direct << postfix_bits;
                if direct > 120 {
                    continue;
                }
                let dist = DistanceParams::new(postfix_bits, direct);
                for code in [16usize, 17, 100, 1000, 100_000, 1_000_000] {
                    let command = Command::new(&dist, 3, 5, 0, code);
                    // `Command::new` encodes with the given parameters, so the
                    // restore has to hand back exactly the same code.
                    assert_eq!(
                        command.restore_distance_code(&dist),
                        code as u32,
                        "npostfix {postfix_bits}, ndirect {direct}, code {code}"
                    );
                }
            }
        }
    }

    #[test]
    fn short_distance_codes_are_passed_through() {
        let dist = DistanceParams::default();
        for code in 0usize..16 {
            let (prefix, extra) = prefix_encode_copy_distance(code, 0, 0);
            assert_eq!(prefix, code as u16);
            assert_eq!(extra, 0);
        }
        let command = Command::new(&dist, 0, 4, 0, 0);
        assert_eq!(command.distance_code(), 0);
        assert_eq!(command.distance_extra_bits(), 0);
    }

    #[test]
    fn an_insert_only_command_copies_nothing() {
        let command = Command::insert_only(17);
        assert_eq!(command.insert_len, 17);
        assert_eq!(command.copy_len(), 0);
        assert_eq!(command.copy_len_code(), 4);
        assert!(!command.has_distance());
    }

    #[test]
    fn a_dictionary_match_carries_its_length_code_delta() {
        let dist = DistanceParams::default();
        let command = Command::new(&dist, 0, 5, 3, 100);
        assert_eq!(command.copy_len(), 5);
        assert_eq!(command.copy_len_code(), 8);

        let shorter = Command::new(&dist, 0, 9, -3, 100);
        assert_eq!(shorter.copy_len(), 9);
        assert_eq!(shorter.copy_len_code(), 6);
    }

    #[test]
    fn distance_contexts_match_the_reference_classes() {
        let dist = DistanceParams::default();
        // A copy of two bytes from the last distance is context zero.
        let short = Command::new(&dist, 0, 2, 0, 0);
        assert!(short.distance_context() <= 3);
        for insert in [0usize, 7, 40] {
            for copy in [2usize, 4, 20, 500] {
                let command = Command::new(&dist, insert, copy, 0, 20);
                assert!(command.distance_context() < 4);
            }
        }
    }

    #[test]
    fn extra_bits_reproduce_the_lengths() {
        let dist = DistanceParams::default();
        for insert in [0usize, 6, 130, 2114, 6210] {
            for copy in [2usize, 10, 134, 2118] {
                let command = Command::new(&dist, insert, copy, 0, 100);
                let (nbits, bits) = command.extra_bits();
                assert!(nbits <= 48);
                assert!(nbits == 64 || (bits >> nbits) == 0);
            }
        }
    }
}

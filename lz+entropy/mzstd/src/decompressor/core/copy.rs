//! Literal and match copies for sequence execution.
//!
//! The fast copies are the reference's wildcopies (`ZSTD_wildcopy`,
//! `ZSTD_overlapCopy8`): they copy whole 16- or 32-byte chunks and may write
//! up to [`WILDCOPY_OVERLENGTH`] bytes past the end of the copy, and read as
//! far past the end of their source. Callers use them only when both
//! buffers have that much room; every byte written past the end lies in the
//! region the next sequences overwrite. The exact copies serve the end of
//! the output buffer.
//!
//! A match whose offset is shorter than 16 bytes overlaps its own output.
//! With a byte shuffle ([`MatchCopy::Shuffle`]) the period is spread over a
//! 16-byte vector once and the vector is rotated for each further chunk;
//! without one ([`MatchCopy::Scalar`]) the reference's offset-spreading
//! eight-byte copies are used.

use fearless_simd::{Simd, SimdBase, u8x16};

/// Bytes a fast copy may write past its end, and read past its source's.
pub(crate) const WILDCOPY_OVERLENGTH: usize = 32;

/// Largest buffer the copies index; callers cut longer buffers down.
pub(crate) const INDEX_LIMIT: usize = u32::MAX as usize;

/// Masks an index to 32 bits.
///
/// Every index the copies see is below [`INDEX_LIMIT`], so the mask changes
/// nothing, but it lets the compiler see that `index + 32` cannot overflow,
/// which leaves one compare in front of each access instead of two.
#[inline(always)]
pub(crate) const fn idx(index: usize) -> usize {
    debug_assert!(index <= INDEX_LIMIT);
    index & INDEX_LIMIT
}

/// How overlapping short-offset matches are copied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MatchCopy {
    /// Eight-byte copies after spreading the offset, as the reference does.
    Scalar,
    /// A native byte shuffle builds and rotates a 16-byte period vector.
    Shuffle,
}

/// `PERIOD[offset][i] = i % offset`: spreads the first `offset` bytes of a
/// vector over all sixteen lanes.
const PERIOD: [[u8; 16]; 16] = {
    let mut table = [[0u8; 16]; 16];
    let mut offset = 1;
    while offset < 16 {
        let mut lane = 0;
        while lane < 16 {
            table[offset][lane] = (lane % offset) as u8;
            lane += 1;
        }
        offset += 1;
    }
    table
};

/// `ROTATE[offset][i] = (i + 16 % offset) % offset`: turns the period
/// vector of one chunk into the next chunk's.
const ROTATE: [[u8; 16]; 16] = {
    let mut table = [[0u8; 16]; 16];
    let mut offset = 1;
    while offset < 16 {
        let mut lane = 0;
        while lane < 16 {
            table[offset][lane] = ((lane + 16 % offset) % offset) as u8;
            lane += 1;
        }
        offset += 1;
    }
    table
};

/// Bytes added to the source after its first four bytes, per offset
/// below eight (`dec32table`).
const SPREAD_ADD: [usize; 8] = [0, 1, 2, 1, 4, 4, 4, 4];
/// Bytes then subtracted from the source, per offset below eight
/// (`dec64table`).
const SPREAD_SUB: [usize; 8] = [8, 8, 8, 7, 8, 9, 10, 11];

/// Copies a fixed-size chunk within `out`, reading the whole chunk before
/// writing it, as one vector load and one store.
#[inline(always)]
fn chunk<const N: usize>(out: &mut [u8], src: usize, dst: usize) {
    let (src, dst) = (idx(src), idx(dst));
    let value: [u8; N] = out[src..src + N].try_into().unwrap_or([0; N]);
    out[dst..dst + N].copy_from_slice(&value);
}

/// Copies `len` literals from `lits[lp..]` to `out[op..]` in 16-byte
/// chunks, first one unconditionally.
///
/// Requires `out.len() >= op + len + WILDCOPY_OVERLENGTH` and
/// `lits.len() >= lp + len + WILDCOPY_OVERLENGTH`.
#[inline(always)]
pub(crate) fn wildcopy_literals(out: &mut [u8], op: usize, lits: &[u8], lp: usize, len: usize) {
    let (op, lp) = (idx(op), idx(lp));
    out[op..op + 16].copy_from_slice(&lits[lp..lp + 16]);
    if len > 16 {
        let end = idx(op + len);
        let (mut dst, mut src) = (op + 16, lp + 16);
        while dst < end {
            let (d, s) = (idx(dst), idx(src));
            out[d..d + 32].copy_from_slice(&lits[s..s + 32]);
            dst += 32;
            src += 32;
        }
    }
}

/// Copies a match of `len` bytes from `offset` bytes back to `out[op..]`,
/// writing whole chunks.
///
/// Requires `1 <= offset <= op` and
/// `out.len() >= op + len + WILDCOPY_OVERLENGTH`.
#[inline(always)]
pub(crate) fn wildcopy_match<S: Simd>(
    simd: S,
    mode: MatchCopy,
    out: &mut [u8],
    op: usize,
    offset: usize,
    len: usize,
) {
    let end = op + len;
    let mut src = op - offset;
    let mut dst = op;
    if offset >= 32 {
        loop {
            chunk::<32>(out, src, dst);
            src += 32;
            dst += 32;
            if dst >= end {
                return;
            }
        }
    }
    if offset >= 16 {
        loop {
            chunk::<16>(out, src, dst);
            src += 16;
            dst += 16;
            if dst >= end {
                return;
            }
        }
    }
    match mode {
        MatchCopy::Shuffle => shuffle_short_match(simd, out, src, dst, offset, end),
        MatchCopy::Scalar => spread_short_match(out, src, dst, offset, end),
    }
}

/// Copies an overlapping match with `offset < 16` by spreading its period
/// over a vector and rotating it for each 16-byte chunk.
#[inline(always)]
fn shuffle_short_match<S: Simd>(
    simd: S,
    out: &mut [u8],
    src: usize,
    mut dst: usize,
    offset: usize,
    end: usize,
) {
    let offset = offset & 15;
    let source = u8x16::from_slice(simd, &out[src..src + 16]);
    let rotate = u8x16::from_slice(simd, &ROTATE[offset]);
    let mut period = simd.swizzle_dyn_u8x16(source, u8x16::from_slice(simd, &PERIOD[offset]));
    loop {
        out[dst..dst + 16].copy_from_slice(period.as_slice());
        dst += 16;
        if dst >= end {
            return;
        }
        period = simd.swizzle_dyn_u8x16(period, rotate);
    }
}

/// Copies an overlapping match with `offset < 16` the reference's way:
/// eight bytes that spread the offset to at least eight, then eight-byte
/// chunks.
#[inline(always)]
fn spread_short_match(out: &mut [u8], src: usize, dst: usize, offset: usize, end: usize) {
    let mut src = src;
    if offset < 8 {
        for k in 0..4 {
            out[dst + k] = out[src + k];
        }
        let shifted = src + SPREAD_ADD[offset];
        chunk::<4>(out, shifted, dst + 4);
        src = shifted + 8 - SPREAD_SUB[offset];
    } else {
        chunk::<8>(out, src, dst);
        src += 8;
    }
    let mut dst = dst + 8;
    while dst < end {
        chunk::<8>(out, src, dst);
        src += 8;
        dst += 8;
    }
}

/// Copies a match of `len` bytes from `offset` bytes back to `out[op..]`,
/// writing nothing past its end. Requires `1 <= offset <= op` and
/// `out.len() >= op + len`.
pub(crate) fn exact_match(out: &mut [u8], op: usize, offset: usize, len: usize) {
    let start = op - offset;
    if offset >= len {
        out.copy_within(start..start + len, op);
        return;
    }
    // `copied` stays a multiple of `offset` until the last step, so copying
    // from `start` repeats the period while the copied source doubles.
    let mut copied = 0;
    while copied < len {
        let step = (len - copied).min(offset + copied);
        out.copy_within(start..start + step, op + copied);
        copied += step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backend;
    use alloc::vec::Vec;
    use fearless_simd::dispatch;

    /// The reference result: a byte-by-byte overlapping copy.
    fn naive(prefix: &[u8], offset: usize, len: usize) -> Vec<u8> {
        let mut out = prefix.to_vec();
        for _ in 0..len {
            out.push(out[out.len() - offset]);
        }
        out
    }

    fn prefix() -> Vec<u8> {
        (0..64u8).map(|byte| byte.wrapping_mul(37) ^ 0x5a).collect()
    }

    #[test]
    fn wildcopied_matches_equal_byte_copies_for_every_offset_and_mode() {
        let prefix = prefix();
        for backend in Backend::available() {
            for mode in [MatchCopy::Scalar, MatchCopy::Shuffle] {
                for offset in 1..=prefix.len() {
                    for len in [3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100] {
                        let expected = naive(&prefix, offset, len);
                        let mut out = prefix.clone();
                        out.resize(prefix.len() + len + WILDCOPY_OVERLENGTH, 0xee);
                        dispatch!(backend.0, simd => wildcopy_match(simd, mode, &mut out, prefix.len(), offset, len));
                        assert_eq!(
                            out[..expected.len()],
                            expected[..],
                            "{backend} {mode:?} offset {offset} len {len}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn exact_matches_write_nothing_past_their_end() {
        let prefix = prefix();
        for offset in 1..=prefix.len() {
            for len in [1, 3, 8, 16, 17, 100] {
                let expected = naive(&prefix, offset, len);
                let mut out = prefix.clone();
                out.resize(prefix.len() + len + 8, 0xee);
                exact_match(&mut out, prefix.len(), offset, len);
                assert_eq!(out[..expected.len()], expected[..]);
                assert!(out[expected.len()..].iter().all(|&byte| byte == 0xee));
            }
        }
    }

    #[test]
    fn wildcopied_literals_equal_their_source() {
        let lits: Vec<u8> = (0..200u8).collect();
        for len in [0, 1, 15, 16, 17, 32, 33, 48, 49, 100] {
            let mut out = alloc::vec![0u8; 8 + len + WILDCOPY_OVERLENGTH];
            wildcopy_literals(&mut out, 8, &lits, 3, len);
            assert_eq!(out[8..8 + len], lits[3..3 + len]);
        }
    }

    #[test]
    fn shuffle_tables_spread_and_rotate_the_period() {
        for offset in 1..16 {
            for lane in 0..16 {
                assert_eq!(usize::from(PERIOD[offset][lane]), lane % offset);
                let next = (lane + 16) % offset;
                assert_eq!(
                    usize::from(PERIOD[offset][usize::from(ROTATE[offset][lane])]),
                    next
                );
            }
        }
    }
}

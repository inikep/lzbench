//! Tag rows of the row match finder (`ZSTD_row_getMatchMask`).
//!
//! A row holds `ROW` one-byte tags; slot 0 doubles as the row's head, the
//! slot written last. A search compares the whole row with its tag in one
//! vector compare and reads the result as a bit mask, rotated so that bit 0
//! is the head and bits follow the slots from newest to oldest — the order
//! the reference visits them in.

use fearless_simd::{Simd, SimdBase, SimdMask, u8x16, u8x32, u8x64};

/// Bit `i` set when the tag in slot `(head + i) % ROW` of `tags` equals
/// `tag` (`ZSTD_row_getMatchMask`), except for slot 0: it holds the head,
/// not a tag, and the reference's search skips it. Bits at and above `ROW`
/// are clear.
///
/// `ROW` is 16, 32 or 64; `head` is below `ROW`.
#[inline(always)]
pub(super) fn match_mask<S: Simd, const ROW: usize>(
    simd: S,
    tags: &[u8; ROW],
    tag: u8,
    head: u32,
) -> u64 {
    if let Some(lanes) = tags.first_chunk::<16>()
        && ROW == 16
    {
        let equal = u8x16::<S>::load_array_ref(simd, lanes).simd_eq(u8x16::splat(simd, tag));
        return u64::from((equal.to_bitmask() as u16 & !1).rotate_right(head));
    }
    if let Some(lanes) = tags.first_chunk::<32>()
        && ROW == 32
    {
        let equal = u8x32::<S>::load_array_ref(simd, lanes).simd_eq(u8x32::splat(simd, tag));
        return u64::from((equal.to_bitmask() as u32 & !1).rotate_right(head));
    }
    if let Some(lanes) = tags.first_chunk::<64>() {
        let equal = u8x64::<S>::load_array_ref(simd, lanes).simd_eq(u8x64::splat(simd, tag));
        return (equal.to_bitmask() & !1).rotate_right(head);
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backend;

    fn reference<const ROW: usize>(tags: &[u8; ROW], tag: u8, head: u32) -> u64 {
        (0..ROW)
            .filter(|&i| {
                let slot = (head as usize + i) % ROW;
                slot != 0 && tags[slot] == tag
            })
            .fold(0u64, |mask, i| mask | 1 << i)
    }

    fn check<const ROW: usize>(backend: Backend) {
        // Tags 0..=3 repeat, so every tag value appears at many slots, and
        // a full row of one tag sets every bit.
        let mut tags = [0u8; ROW];
        for (i, slot) in tags.iter_mut().enumerate() {
            *slot = (i * 7 % 4) as u8;
        }
        for head in 0..ROW as u32 {
            for tag in 0..=4u8 {
                let got =
                    fearless_simd::dispatch!(backend.0, simd => match_mask(simd, &tags, tag, head));
                assert_eq!(got, reference(&tags, tag, head), "{backend} row {ROW}");
            }
        }
        let full = [0xA5u8; ROW];
        let got = fearless_simd::dispatch!(backend.0, simd => match_mask(simd, &full, 0xA5, 3));
        assert_eq!(got, (u64::MAX >> (64 - ROW)) & !(1u64 << ((ROW - 3) % ROW)));
    }

    #[test]
    fn masks_equal_the_scalar_reference_on_every_backend() {
        for backend in Backend::available() {
            check::<16>(backend);
            check::<32>(backend);
            check::<64>(backend);
        }
    }
}

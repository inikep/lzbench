//! Tag rows of the row match finder (`ZSTD_row_getMatchMask`).
//!
//! A row holds `ROW` one-byte tags; slot 0 doubles as the row's head, the
//! slot written last. A search compares the whole row with its tag in one
//! vector compare and reads the result as a bit mask, rotated so that its
//! lowest group is the head and groups follow the slots from newest to
//! oldest — the order the reference visits them in.
//!
//! The mask has one bit per slot, except for 16-slot rows on NEON: NEON has
//! no byte-to-bit `movemask`, and the portable bitmask emulates one, so
//! there the compare narrows to four bits per slot (`vshrn` by four) and
//! one bit of each matching group is kept, as the reference's
//! `ZSTD_row_getNEONMask` does. Its 32- and 64-slot forms measured slower
//! than the portable bitmask and are not used.

#[cfg(target_arch = "aarch64")]
use fearless_simd::Level;
use fearless_simd::{Simd, SimdBase, SimdMask, u8x16, u8x32, u8x64};

/// The matching slots of a row: bit group `i` (bits from `i << shift` up to
/// the next group) holds one set bit when slot `(head + i) % ROW` matches.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) struct Matches {
    /// The groups, the head's lowest.
    bits: u64,
    /// `log2` of the group width.
    shift: u32,
}

impl Matches {
    /// The offset from the head of the newest matching slot not yet taken.
    #[inline(always)]
    pub(super) fn pop(&mut self) -> Option<usize> {
        if self.bits == 0 {
            return None;
        }
        let offset = (self.bits.trailing_zeros() >> self.shift) as usize;
        self.bits &= self.bits - 1;
        Some(offset)
    }
}

/// The slots of `tags` whose tag equals `tag` (`ZSTD_row_getMatchMask`),
/// except for slot 0: it holds the head, not a tag, and the reference's
/// search skips it.
///
/// `ROW` is 16, 32 or 64; `head` is below `ROW`.
#[inline(always)]
pub(super) fn match_mask<S: Simd, const ROW: usize>(
    simd: S,
    tags: &[u8; ROW],
    tag: u8,
    head: u32,
) -> Matches {
    if let Some(lanes) = tags.first_chunk::<16>()
        && ROW == 16
    {
        #[cfg(target_arch = "aarch64")]
        if let Level::Neon(neon) = simd.level() {
            let row = u8x16::<fearless_simd::Neon>::load_array_ref(neon, lanes).into();
            // Four bits per slot; slot 0 is dropped before the rotation and
            // one bit of each group kept after it.
            let groups = neon::nibble_mask(neon, row, tag) & !0xF;
            return Matches {
                bits: groups.rotate_right(head * 4) & 0x1111_1111_1111_1111,
                shift: 2,
            };
        }
        let equal = u8x16::<S>::load_array_ref(simd, lanes).simd_eq(u8x16::splat(simd, tag));
        return Matches {
            bits: u64::from((equal.to_bitmask() as u16 & !1).rotate_right(head)),
            shift: 0,
        };
    }
    let bits = if let Some(lanes) = tags.first_chunk::<32>()
        && ROW == 32
    {
        let equal = u8x32::<S>::load_array_ref(simd, lanes).simd_eq(u8x32::splat(simd, tag));
        u64::from((equal.to_bitmask() as u32 & !1).rotate_right(head))
    } else if let Some(lanes) = tags.first_chunk::<64>() {
        let equal = u8x64::<S>::load_array_ref(simd, lanes).simd_eq(u8x64::splat(simd, tag));
        (equal.to_bitmask() & !1).rotate_right(head)
    } else {
        0
    };
    Matches { bits, shift: 0 }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{
        uint8x16_t, vceqq_u8, vdupq_n_u8, vget_lane_u64, vreinterpret_u64_u8, vreinterpretq_u16_u8,
        vshrn_n_u16,
    };

    fearless_simd::kernel!(
        /// Four bits per lane of `row`, all set where the lane equals `tag`:
        /// the compare narrowed by `vshrn` by four.
        #[inline(always)]
        pub(super) fn nibble_mask(neon: Neon, row: uint8x16_t, tag: u8) -> u64 {
            let equal = vceqq_u8(row, vdupq_n_u8(tag));
            let narrowed = vshrn_n_u16::<4>(vreinterpretq_u16_u8(equal));
            vget_lane_u64::<0>(vreinterpret_u64_u8(narrowed))
        }
    );
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

    /// The offsets a mask yields, as one bit each, checking that they come
    /// out in visiting order and once each.
    fn offsets(mut matches: Matches) -> u64 {
        let mut compact = 0u64;
        let mut previous = None;
        while let Some(offset) = matches.pop() {
            assert!(previous < Some(offset), "offsets come in order, once each");
            previous = Some(offset);
            compact |= 1 << offset;
        }
        compact
    }

    fn mask<const ROW: usize>(backend: Backend, tags: &[u8; ROW], tag: u8, head: u32) -> u64 {
        offsets(fearless_simd::dispatch!(backend.0, simd => match_mask(simd, tags, tag, head)))
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
                assert_eq!(
                    mask(backend, &tags, tag, head),
                    reference(&tags, tag, head),
                    "{backend} row {ROW}"
                );
            }
        }
        let full = [0xA5u8; ROW];
        assert_eq!(
            mask(backend, &full, 0xA5, 3),
            (u64::MAX >> (64 - ROW)) & !(1u64 << ((ROW - 3) % ROW))
        );
    }

    #[test]
    fn masks_equal_the_scalar_reference_on_every_backend() {
        for backend in Backend::available() {
            check::<16>(backend);
            check::<32>(backend);
            check::<64>(backend);
        }
    }

    #[test]
    fn a_single_matching_slot_is_found_from_every_head() {
        fn each<const ROW: usize>(backend: Backend) {
            for slot in 0..ROW {
                let mut tags = [0u8; ROW];
                tags[slot] = 9;
                for head in 0..ROW as u32 {
                    assert_eq!(
                        mask(backend, &tags, 9, head),
                        reference(&tags, 9, head),
                        "{backend} row {ROW} slot {slot} head {head}"
                    );
                }
            }
        }
        for backend in Backend::available() {
            each::<16>(backend);
            each::<32>(backend);
            each::<64>(backend);
        }
    }
}

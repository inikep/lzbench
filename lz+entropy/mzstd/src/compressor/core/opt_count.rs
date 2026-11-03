//! Exact common-prefix lengths (`ZSTD_count`, `ZSTD_count_2segments`) for
//! the optimal parser's binary tree, with a SIMD scan for long runs.
//!
//! The reference compares eight bytes at a time with an XOR and a
//! trailing-zero count, then four, two and one. Every scan here returns the
//! same length — only the number of bytes compared per step differs — so the
//! parse, and every emitted byte, is independent of the backend.
//!
//! Most candidates of a tree walk differ within the first word, so that word
//! is settled inline; a run that survives it continues in [`count_tail`],
//! which compares two more words, then native vectors while the run keeps
//! going, then words, and last one word that ends at the limit.

use fearless_simd::{Simd, SimdBase, SimdMask, u8x16, u8x32, u8x64};
use fearless_simd_macros::simd;

/// Words [`count_ordered`] compares inline before calling the tail scan.
const INLINE_WORDS: usize = 2;

/// Bytes compared with 64-bit words before the vector loop is entered.
const SCALAR_PREFIX_BYTES: usize = 16;

/// Leading bytes `left` and `right` share; both windows have one length.
///
/// The first word is compared inline; the rest of a longer run is measured
/// by [`count_tail`] out of line, in the backend's feature context.
#[inline(always)]
pub(crate) fn count_windows<S: Simd>(simd: S, left: &[u8], right: &[u8]) -> usize {
    debug_assert_eq!(left.len(), right.len());
    if let (Some(left_word), Some(right_word)) = (left.first_chunk::<8>(), right.first_chunk::<8>())
    {
        let difference = u64::from_le_bytes(*left_word) ^ u64::from_le_bytes(*right_word);
        if difference != 0 {
            return difference.trailing_zeros() as usize >> 3;
        }
        return 8 + count_tail(simd, &left[8..], &right[8..]);
    }
    count_short(left, right)
}

/// `ZSTD_count` inside one segment: the bytes `data[cur..limit]` share with
/// `data[earlier..]`, where `earlier < cur`.
///
/// # Panics
///
/// Panics when `cur..limit` is not inside `data` or `earlier >= cur`; the
/// tree walk's index invariants rule both out.
#[inline(always)]
pub(crate) fn count<S: Simd>(
    simd: S,
    data: &[u8],
    cur: usize,
    earlier: usize,
    limit: usize,
) -> usize {
    let left = &data[cur..limit];
    let right = &data[earlier..earlier + left.len()];
    count_windows(simd, left, right)
}

/// The tree walk's comparison: how many bytes past the `known` equal ones
/// `current` shares with `data[start..]` (the earlier side, already
/// advanced by `known`: an earlier run may have begun in another segment),
/// and whether the earlier side's first differing byte is the smaller one
/// (`match[length] < ip[length]`), which the walk reads next. The flag is
/// `false` when the run reaches the end of `current`.
///
/// Most candidates differ within the first word: it is loaded from each
/// side with one length test apiece (offsets are 32-bit, so their sums
/// cannot overflow), and both bytes of the order come from the words
/// already loaded. Only a run that survives the word cuts the full
/// windows.
///
/// # Panics
///
/// Panics when `known` exceeds `current` or the earlier window is not
/// inside `data`; the tree walk's index invariants rule both out.
#[inline(always)]
pub(crate) fn count_ordered<S: Simd>(
    simd: S,
    current: &[u8],
    data: &[u8],
    start: u32,
    known: u32,
) -> (usize, bool) {
    let known = known as usize;
    let start = start as usize;
    // The rest is the uncommon case, which the hints keep out of the
    // walk's path: the registers the scan's call clobbers are then saved
    // around that call, not at every candidate.
    let words = (word_at(current, known), word_at(data, start));
    if let (Some(left_word), Some(right_word)) = words
        && !branches::unlikely(left_word == right_word)
    {
        return ordered_difference(left_word, right_word, 0);
    }
    // A first word that fit on both sides matched.
    let equal = if words.0.is_some() && words.1.is_some() {
        8
    } else {
        0
    };
    count_ordered_run(simd, current, data, start, known, equal)
}

/// The length and order of two words that differ, `equal` bytes into the
/// run.
#[inline(always)]
fn ordered_difference(left_word: u64, right_word: u64, equal: usize) -> (usize, bool) {
    // Read big-endian, the first differing byte is the most significant
    // one that differs: the words order as it does.
    let smaller = right_word.swap_bytes() < left_word.swap_bytes();
    let length = ((left_word ^ right_word).trailing_zeros() >> 3) as usize;
    (equal + length, smaller)
}

/// [`count_ordered`] past a first word that matched (`equal` is 8) or does
/// not fit (`equal` is 0).
///
/// Out of line: the walk then keeps its state in registers across the
/// call rather than saving it for a scan most candidates never reach.
#[inline(never)]
fn count_ordered_run<S: Simd>(
    simd: S,
    current: &[u8],
    data: &[u8],
    start: usize,
    known: usize,
    mut equal: usize,
) -> (usize, bool) {
    // Up to `INLINE_WORDS` words inline: they settle most runs of a tree
    // walk, and only a longer one pays for the call into the tail scan.
    while equal < INLINE_WORDS * 8 {
        let (Some(left_word), Some(right_word)) = (
            word_at(current, known + equal),
            word_at(data, start + equal),
        ) else {
            break;
        };
        if left_word != right_word {
            return ordered_difference(left_word, right_word, equal);
        }
        equal += 8;
    }
    let left = &current[known + equal..];
    let right = &data[start + equal..start + equal + left.len()];
    let matched = if equal == 0 {
        count_windows(simd, left, right)
    } else {
        count_tail(simd, left, right)
    };
    (equal + matched, earlier_is_smaller(left, right, matched))
}

/// The little-endian word at `at`, if eight bytes are there.
#[inline(always)]
fn word_at(bytes: &[u8], at: usize) -> Option<u64> {
    let word: &[u8; 8] = bytes.get(at..at + 8)?.try_into().ok()?;
    Some(u64::from_le_bytes(*word))
}

/// Whether `right[at] < left[at]`; `false` past the end of either.
#[inline(always)]
fn earlier_is_smaller(left: &[u8], right: &[u8], at: usize) -> bool {
    match (left.get(at), right.get(at)) {
        (Some(current), Some(earlier)) => earlier < current,
        _ => false,
    }
}

/// `ZSTD_count_2segments`: a run whose earlier side starts in the external
/// segment at offset `earlier` and continues at the start of `prefix` once
/// it reaches the end of `ext`.
///
/// `cur` and `limit` are offsets into `prefix`.
#[inline(always)]
pub(crate) fn count_2segments<S: Simd>(
    simd: S,
    prefix: &[u8],
    cur: usize,
    ext: &[u8],
    earlier: usize,
    limit: usize,
) -> usize {
    let in_ext = ext.len() - earlier;
    let v_end = (cur + in_ext).min(limit);
    let left = &prefix[cur..v_end];
    let right = &ext[earlier..earlier + left.len()];
    let matched = count_windows(simd, left, right);
    if matched != in_ext {
        return matched;
    }
    matched + count(simd, prefix, cur + matched, 0, limit)
}

/// The bytes after a matching first word, behind a call in `simd`'s
/// feature context.
///
/// The wrapper inlines into the caller, which calls the level's
/// feature-enabled entry directly; that entry is the call.
#[simd]
#[inline(always)]
fn count_tail<S: Simd>(simd: S, left: &[u8], right: &[u8]) -> usize {
    count_long(simd, left, right)
}

/// Counts the leading bytes two windows of the same length share: up to two
/// words, native vectors, words, then one word ending at the limit.
#[inline(always)]
fn count_long<S: Simd>(simd: S, left: &[u8], right: &[u8]) -> usize {
    let limit = left.len().min(right.len());
    let (left, right) = (&left[..limit], &right[..limit]);

    let prefix = limit.min(SCALAR_PREFIX_BYTES) & !7;
    let mut matched = count_words(&left[..prefix], &right[..prefix]);
    if matched < prefix {
        return matched;
    }

    let stride = native_vector_stride::<S>();
    let whole_vectors = (limit - matched) - (limit - matched) % stride;
    let vectored = count_native_vectors(simd, &left[matched..], &right[matched..]);
    matched += vectored;
    if vectored < whole_vectors {
        return matched;
    }

    let whole_words = (limit - matched) & !7;
    let tail = count_words(
        &left[matched..matched + whole_words],
        &right[matched..matched + whole_words],
    );
    matched += tail;
    if tail < whole_words || matched == limit {
        return matched;
    }

    // Fewer than eight bytes remain. A word ending at the limit overlaps
    // bytes already found equal, so its first difference is the tail's.
    if let (Some(left_last), Some(right_last)) = (left.last_chunk::<8>(), right.last_chunk::<8>()) {
        let difference = u64::from_le_bytes(*left_last) ^ u64::from_le_bytes(*right_last);
        return if difference == 0 {
            limit
        } else {
            limit - 8 + (difference.trailing_zeros() as usize >> 3)
        };
    }
    count_short(left, right)
}

/// Windows shorter than a word: four, two, then one byte, as the
/// reference's tail does.
#[inline(always)]
fn count_short(left: &[u8], right: &[u8]) -> usize {
    let limit = left.len().min(right.len());
    let mut matched = 0;
    if let (Some(a), Some(b)) = (left.first_chunk::<4>(), right.first_chunk::<4>())
        && a == b
    {
        matched = 4;
    }
    if let (Some(a), Some(b)) = (
        left.get(matched..matched + 2),
        right.get(matched..matched + 2),
    ) && a == b
    {
        matched += 2;
    }
    if matched < limit && left[matched] == right[matched] {
        matched += 1;
    }
    matched
}

/// Compares two windows eight bytes at a time; returns the leading equal
/// bytes, or the whole-word length when every word matched.
#[inline(always)]
fn count_words(left: &[u8], right: &[u8]) -> usize {
    let (left_words, _) = left.as_chunks::<8>();
    let (right_words, _) = right.as_chunks::<8>();
    let mut matched = 0usize;
    for (left_word, right_word) in left_words.iter().zip(right_words) {
        let difference = u64::from_le_bytes(*left_word) ^ u64::from_le_bytes(*right_word);
        if difference != 0 {
            return matched + (difference.trailing_zeros() as usize >> 3);
        }
        matched += 8;
    }
    matched
}

/// Defines a vector comparison loop for one vector width.
macro_rules! vector_scan {
    ($name:ident, $vector:ident, $lanes:literal) => {
        #[doc = concat!(
                    "Compares two windows ", stringify!($lanes), " bytes at a time; returns \
             the leading equal bytes, or the whole-vector length when every vector matched."
                )]
        #[inline(always)]
        fn $name<S: Simd>(simd: S, left: &[u8], right: &[u8]) -> usize {
            let (left_vectors, _) = left.as_chunks::<$lanes>();
            let (right_vectors, _) = right.as_chunks::<$lanes>();
            let mut matched = 0usize;
            for (left_lanes, right_lanes) in left_vectors.iter().zip(right_vectors) {
                let equal = $vector::<S>::load_array_ref(simd, left_lanes)
                    .simd_eq($vector::<S>::load_array_ref(simd, right_lanes));
                if equal.any_false() {
                    return matched + equal.to_bitmask().trailing_ones() as usize;
                }
                matched += $lanes;
            }
            matched
        }
    };
}

vector_scan!(count_vectors_16, u8x16, 16);
vector_scan!(count_vectors_32, u8x32, 32);
vector_scan!(count_vectors_64, u8x64, 64);

/// Bytes [`count_native_vectors`] advances per step on this backend: its
/// native vector width, or 16 bytes (which every backend provides) for any
/// other width.
#[inline(always)]
const fn native_vector_stride<S: Simd>() -> usize {
    match <S::u8s as SimdBase<S>>::LEN {
        32 => 32,
        64 => 64,
        _ => 16,
    }
}

/// The vector scan at the backend's native width, chosen at
/// monomorphisation time; it pairs with [`native_vector_stride`].
#[inline(always)]
fn count_native_vectors<S: Simd>(simd: S, left: &[u8], right: &[u8]) -> usize {
    match <S::u8s as SimdBase<S>>::LEN {
        32 => count_vectors_32(simd, left, right),
        64 => count_vectors_64(simd, left, right),
        _ => count_vectors_16(simd, left, right),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backend;
    use alloc::vec::Vec;
    use fearless_simd::dispatch;

    fn baseline(left: &[u8], right: &[u8]) -> usize {
        left.iter().zip(right).take_while(|(a, b)| a == b).count()
    }

    fn data(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 7 % 251) as u8).collect()
    }

    #[test]
    fn windows_agree_with_a_byte_scan_on_every_backend() {
        let base = data(400);
        // Miri interprets every lane, so under it only lengths next to a
        // 16-byte boundary run, with sparser differences: every vector
        // width, tail and full-length scan still occurs.
        let near_a_boundary = |len: &usize| !cfg!(miri) || matches!(len % 16, 0 | 1 | 15);
        let spacing = if cfg!(miri) { 4 } else { 37 };
        for backend in Backend::available() {
            for len in (0..300).filter(near_a_boundary) {
                for diff in (0..len).step_by(1 + len / spacing).chain([len]) {
                    let left = &base[..len];
                    let mut right = left.to_vec();
                    if diff < len {
                        right[diff] ^= 0x5a;
                    }
                    let got = dispatch!(backend.0, simd => count_windows(simd, left, &right));
                    assert_eq!(
                        got,
                        baseline(left, &right),
                        "{backend} len {len} diff {diff}"
                    );
                    let long = dispatch!(backend.0, simd => count_long(simd, left, &right));
                    assert_eq!(long, baseline(left, &right), "{backend} long {len} {diff}");
                }
            }
        }
    }

    #[test]
    fn segment_counts_stop_at_the_limit() {
        let bytes = b"abcabcabcabcabcabcabcabcabcabcabcabcabcX";
        for backend in Backend::available() {
            for limit in 3..bytes.len() {
                let got = dispatch!(backend.0, simd => count(simd, bytes, 3, 0, limit));
                assert_eq!(got, baseline(&bytes[3..limit], bytes), "{backend} {limit}");
            }
        }
    }

    #[test]
    fn ordered_counts_report_the_first_differing_bytes() {
        let mut bytes: Vec<u8> = (0..120).map(|i| (i % 40) as u8).collect();
        bytes.push(200);
        for backend in Backend::available() {
            for (cur, earlier) in [(40usize, 0usize), (80, 0), (80, 40)] {
                for limit in cur..=bytes.len() {
                    for flip in [None, Some(3usize), Some(9), Some(13), Some(21), Some(30)] {
                        let mut data = bytes.clone();
                        if let Some(at) = flip.filter(|at| cur + at < limit) {
                            let delta = if at.is_multiple_of(2) { 1 } else { 255 };
                            data[cur + at] = data[cur + at].wrapping_add(delta);
                        }
                        let known = (limit - cur).min(2);
                        let (length, smaller) = dispatch!(backend.0, simd => {
                            count_ordered(simd, &data[cur..limit], &data, (earlier + known) as u32, known as u32)
                        });
                        let length = length + known;
                        let expected = baseline(&data[cur..limit], &data[earlier..]);
                        assert_eq!(length, expected, "{backend} {cur} {earlier} {limit}");
                        let expected_order = cur + expected < limit
                            && data[earlier + expected] < data[cur + expected];
                        assert_eq!(smaller, expected_order, "{backend} {cur} {limit} {flip:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn two_segment_counts_continue_at_the_prefix_start() {
        // "ab" ends the external segment; the prefix begins with "ab" again.
        let ext = b"xyab";
        let prefix = b"ababababababababababababx";
        for backend in Backend::available() {
            let got = dispatch!(backend.0, simd => count_2segments(simd, prefix, 2, ext, 2, prefix.len()));
            assert_eq!(got, 22, "{backend}");
            // A difference inside the external segment stops the run there.
            let got = dispatch!(backend.0, simd => count_2segments(simd, prefix, 1, ext, 2, prefix.len()));
            assert_eq!(got, 0, "{backend}");
            // The limit cuts the run inside the external segment.
            let got = dispatch!(backend.0, simd => count_2segments(simd, prefix, 2, ext, 2, 3));
            assert_eq!(got, 1, "{backend}");
        }
    }

    #[test]
    fn short_windows_compare_four_two_and_one_bytes() {
        for len in 0..8 {
            let left: Vec<u8> = (0..len as u8).collect();
            for diff in 0..=len {
                let mut right = left.clone();
                if diff < len {
                    right[diff] = 0xff;
                }
                assert_eq!(count_short(&left, &right), baseline(&left, &right));
            }
        }
    }
}

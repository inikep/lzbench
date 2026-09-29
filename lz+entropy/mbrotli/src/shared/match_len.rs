//! Exact match-length scan, with a portable SIMD tail.
//!
//! The reference encoder compares eight bytes at a time with an XOR and a
//! trailing-zero count (`c/enc/find_match_length.h`). That is already very fast
//! for the short matches the fast qualities mostly find, so this port keeps a
//! scalar prefix and only widens to native SIMD vectors once a match has
//! already run past [`SCALAR_PREFIX_BYTES`]. The result is the exact same
//! length in every case, so the emitted bitstream never depends on the level.

use fearless_simd::{Simd, SimdBase, SimdMask, u8x16, u8x32, u8x64};

/// Bytes compared with scalar 64-bit loads before the SIMD loop is entered.
///
/// Short matches dominate quality 0 and 1, and entering a vector loop for them
/// costs more than it saves.
pub(crate) const SCALAR_PREFIX_BYTES: usize = 16;

/// Reads eight little-endian bytes at `offset`, or zero past the end.
///
/// Borrowing a fixed-size chunk instead of copying into a scratch array keeps
/// this to a single unaligned load in the generated code, while staying inside
/// safe Rust: the bounds test is the only thing that survives.
///
/// Every buffer index in this crate is below 2^32 (positions are 32-bit in
/// the format), so `offset` is masked to that width: it changes nothing for
/// a real index and lets the compiler see that `offset + 8` cannot overflow,
/// which leaves a single compare in front of the load.
#[inline(always)]
pub(crate) fn load_u64_le(data: &[u8], offset: usize) -> u64 {
    debug_assert!(offset <= u32::MAX as usize);
    let offset = offset & u32::MAX as usize;
    match data.get(offset..offset + 8) {
        Some(chunk) => u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])),
        None => 0,
    }
}

/// The bytes a search measures its candidates against.
///
/// The ring buffer from the searched position to the end of the input, cut
/// once per search so every candidate scan needs one bounds check, on its
/// own side. A position the buffer cannot hold in full leaves an empty
/// slice, which matches nothing — the answer [`find_match_length`] gives.
///
/// Both operands are masked to 32 bits, as in [`load_u64_le`]: every buffer
/// index and length is below 2^32, and the mask lets the compiler see that
/// the end cannot overflow or precede the start, so only the compare
/// against the buffer's length survives.
#[inline(always)]
pub(crate) fn current_window(data: &[u8], cur_ix_masked: usize, max_length: usize) -> &[u8] {
    debug_assert!(cur_ix_masked <= u32::MAX as usize && max_length <= u32::MAX as usize);
    let start = cur_ix_masked & u32::MAX as usize;
    data.get(start..start + (max_length & u32::MAX as usize))
        .unwrap_or_default()
}

/// Counts the leading bytes `data[prev_ix..]` shares with `cur`.
///
/// Zero when the buffer cannot hold a window as long as `cur` at `prev_ix`.
///
/// The first word is settled before any window is cut: most candidates a
/// matcher measures differ within eight bytes, and the reference's scan
/// answers those with one load per side, an XOR and a trailing-zero count.
/// Only a match that runs through the whole word pays for the slicing that
/// the longer scan of [`match_len_windows`] needs.
#[inline(always)]
pub(crate) fn match_len_at<S: Simd>(simd: S, data: &[u8], prev_ix: usize, cur: &[u8]) -> usize {
    let Some(left) = data.get(prev_ix..prev_ix + cur.len()) else {
        return 0;
    };
    if let (Some(left_word), Some(cur_word)) = (left.first_chunk::<8>(), cur.first_chunk::<8>()) {
        let difference = u64::from_le_bytes(*left_word) ^ u64::from_le_bytes(*cur_word);
        if difference != 0 {
            return difference.trailing_zeros() as usize >> 3;
        }
        return 8 + match_len_windows(simd, &left[8..], &cur[8..]);
    }
    match_len_windows(simd, left, cur)
}

/// [`match_len_at`] with the scan past the first word kept out of line.
///
/// The quick matchers probe several slots per position with one current
/// window. Inlined, the longer scan's setup does not depend on the slot, so
/// the compiler hoists it in front of the slot loop and pays for it — and for
/// spilling what it computed — at every position, although only about a
/// quarter of the candidates get past the first word. Out of line, a
/// position whose candidates all differ within eight bytes pays for nothing
/// but the first-word compares.
#[inline(always)]
pub(crate) fn match_len_at_outlined<S: Simd>(
    simd: S,
    data: &[u8],
    prev_ix: usize,
    cur: &[u8],
) -> usize {
    // Masked to 32 bits for the reason `current_window` gives.
    debug_assert!(prev_ix <= u32::MAX as usize && cur.len() <= u32::MAX as usize);
    let start = prev_ix & u32::MAX as usize;
    let Some(left) = data.get(start..start + (cur.len() & u32::MAX as usize)) else {
        return 0;
    };
    if let (Some(left_word), Some(cur_word)) = (left.first_chunk::<8>(), cur.first_chunk::<8>()) {
        let difference = u64::from_le_bytes(*left_word) ^ u64::from_le_bytes(*cur_word);
        if difference != 0 {
            return difference.trailing_zeros() as usize >> 3;
        }
        return 8 + match_len_tail(simd, &left[8..], &cur[8..]);
    }
    match_len_tail(simd, left, cur)
}

/// [`match_len_windows`] behind a call, back inside `simd`'s feature context.
#[inline(never)]
fn match_len_tail<S: Simd>(simd: S, left: &[u8], right: &[u8]) -> usize {
    simd.vectorize(
        #[inline(always)]
        move || match_len_windows(simd, left, right),
    )
}

/// Counts the leading bytes two windows of the same length share.
///
/// The reference's `FindMatchLengthWithLimit` compares whole words and then
/// single bytes. This scan compares up to two words, then native vectors
/// while the match keeps going — the long matches of repetitive input are
/// where a vector loop pays — then whole words, and last one more word that
/// ends at the limit: it overlaps bytes the word loop already found equal,
/// so its first differing byte is the first differing byte of the tail.
/// Only a window shorter than a word is compared byte by byte, which keeps
/// a byte loop the compiler would unroll into kilobytes out of the search.
/// Both windows were cut to the same length by the caller, so the chunked
/// iterations carry no check of their own.
#[inline(always)]
pub(crate) fn match_len_windows<S: Simd>(simd: S, left: &[u8], right: &[u8]) -> usize {
    let limit = left.len().min(right.len());
    let (left, right) = (&left[..limit], &right[..limit]);

    // Short matches dominate; two plain words settle most of them.
    let prefix = limit.min(SCALAR_PREFIX_BYTES) & !7;
    let mut matched = match_len_words(&left[..prefix], &right[..prefix]);
    if matched < prefix {
        return matched;
    }

    let stride = native_vector_stride::<S>();
    let whole_vectors = (limit - matched) - (limit - matched) % stride;
    let vectored = match_len_native_vectors(simd, &left[matched..], &right[matched..]);
    matched += vectored;
    if vectored < whole_vectors {
        return matched;
    }

    let whole_words = (limit - matched) & !7;
    let tail = match_len_words(
        &left[matched..matched + whole_words],
        &right[matched..matched + whole_words],
    );
    matched += tail;
    if tail < whole_words || matched == limit {
        return matched;
    }

    if let (Some(left_last), Some(right_last)) = (left.last_chunk::<8>(), right.last_chunk::<8>()) {
        let difference = u64::from_le_bytes(*left_last) ^ u64::from_le_bytes(*right_last);
        return if difference == 0 {
            limit
        } else {
            limit - 8 + (difference.trailing_zeros() as usize >> 3)
        };
    }
    let mut bytes = 0usize;
    while bytes < limit && left[bytes] == right[bytes] {
        bytes += 1;
    }
    bytes
}

/// Compares two equal-length windows eight bytes at a time.
///
/// Returns the number of leading equal bytes, or the length of the whole-word
/// prefix when every word matched. Iterating over fixed-size chunks keeps the
/// loop free of bounds checks without any unsafe code.
#[inline(always)]
fn match_len_words(left: &[u8], right: &[u8]) -> usize {
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

/// Defines a vector comparison loop for one concrete vector width.
///
/// `as_chunks` hands the loop `&[u8; LANES]` references, which are exactly the
/// array type the vector loads from, so `load_array_ref` takes them by
/// reference with neither a copy, a bounds check, nor a length assertion.
macro_rules! vector_scan {
    ($name:ident, $vector:ident, $lanes:literal) => {
        #[doc = concat!(
                            "Compares two windows ", stringify!($lanes), " bytes at a time.\n\n",
                            "Returns the number of leading equal bytes, or the length of the \
             whole-vector prefix when every vector matched."
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

vector_scan!(match_len_vectors_16, u8x16, 16);
vector_scan!(match_len_vectors_32, u8x32, 32);
vector_scan!(match_len_vectors_64, u8x64, 64);

/// Bytes [`match_len_native_vectors`] advances per step on this backend.
///
/// The caller uses this to work out how many bytes a fully matching scan would
/// have reported, so it has to be the stride the scan really takes rather than
/// the backend's lane count: a width with no vector loop degrades to a
/// byte-at-a-time scan, whose stride is one.
#[inline(always)]
const fn native_vector_stride<S: Simd>() -> usize {
    match <S::u8s as SimdBase<S>>::LEN {
        16 => 16,
        32 => 32,
        64 => 64,
        _ => 1,
    }
}

/// Runs the vector scan with the backend's native lane count baked in.
///
/// The match resolves at monomorphisation time, because `S::u8s::LEN` is a
/// constant for every backend; no branch survives into the generated code.
#[inline(always)]
fn match_len_native_vectors<S: Simd>(simd: S, left: &[u8], right: &[u8]) -> usize {
    match <S::u8s as SimdBase<S>>::LEN {
        16 => match_len_vectors_16(simd, left, right),
        32 => match_len_vectors_32(simd, left, right),
        64 => match_len_vectors_64(simd, left, right),
        // No supported backend has another width. Falling back to single bytes
        // rather than to whole words is what keeps the caller's "did every step
        // match?" test exact: it pairs with the stride of one that
        // `native_vector_stride` reports, whereas a word scan would round the
        // window down to a multiple of eight and under-report a shorter match.
        _ => match_len_bytes(left, right),
    }
}

/// Compares two windows byte by byte, stopping at the shorter one.
#[inline(always)]
fn match_len_bytes(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left_byte, right_byte)| left_byte == right_byte)
        .count()
}

/// Returns how many bytes of `data[left..]` and `data[right..]` agree.
///
/// The comparison stops after `limit` bytes; neither side is read beyond it.
/// A window that does not fit inside `data` yields zero rather than panicking.
#[inline(always)]
pub(crate) fn find_match_length<S: Simd>(
    simd: S,
    data: &[u8],
    left: usize,
    right: usize,
    limit: usize,
) -> usize {
    // Two plain comparisons bound both windows for the rest of the scan.
    let (Some(left_window), Some(right_window)) = (data.get(left..), data.get(right..)) else {
        return 0;
    };
    if limit > left_window.len() || limit > right_window.len() {
        return 0;
    }
    scan_windows(simd, &left_window[..limit], &right_window[..limit])
}

/// Returns how many leading bytes two windows share, at most `limit`.
///
/// Whole eight-byte words first, then single bytes — the scan the reference's
/// `FindMatchLengthWithLimit` makes — with the bounds established once so the
/// loops carry no checks. Scalar on purpose: its callers compare static
/// dictionary words of at most twenty-four bytes, where a vector loop would
/// not pay for its setup. [`common_prefix_len_simd`] is the wide variant.
#[inline(always)]
pub(crate) fn common_prefix_len(left: &[u8], right: &[u8], limit: usize) -> usize {
    let limit = limit.min(left.len()).min(right.len());
    let (left, right) = (&left[..limit], &right[..limit]);
    let whole_words = limit & !7;
    let matched = match_len_words(&left[..whole_words], &right[..whole_words]);
    if matched < whole_words {
        return matched;
    }
    matched + match_len_bytes(&left[whole_words..], &right[whole_words..])
}

/// Returns how many leading bytes two windows share, at most `limit`, with
/// the vector scan of [`find_match_length`] over long agreements.
///
/// For attached-prefix matches, which run as long as the window allows.
#[inline(always)]
pub(crate) fn common_prefix_len_simd<S: Simd>(
    simd: S,
    left: &[u8],
    right: &[u8],
    limit: usize,
) -> usize {
    let limit = limit.min(left.len()).min(right.len());
    scan_windows(simd, &left[..limit], &right[..limit])
}

/// Counts the leading equal bytes of two windows of the same length.
#[inline(always)]
fn scan_windows<S: Simd>(simd: S, left_window: &[u8], right_window: &[u8]) -> usize {
    let limit = left_window.len().min(right_window.len());
    let (left_window, right_window) = (&left_window[..limit], &right_window[..limit]);

    // Cheap scalar prefix. Most matches the fast qualities find are short, and
    // entering a vector loop for them costs more than it saves.
    let prefix = limit.min(SCALAR_PREFIX_BYTES) & !7;
    let mut matched = match_len_words(&left_window[..prefix], &right_window[..prefix]);
    if matched < prefix {
        return matched;
    }

    // Native-width vectors over the bulk of a long match.
    let stride = native_vector_stride::<S>();
    let whole_vectors = (limit - matched) - (limit - matched) % stride;
    let vectored =
        match_len_native_vectors(simd, &left_window[matched..], &right_window[matched..]);
    matched += vectored;
    if vectored < whole_vectors {
        return matched;
    }

    // Whole-word and single-byte tails.
    let whole_words = (limit - matched) & !7;
    let tail = match_len_words(
        &left_window[matched..matched + whole_words],
        &right_window[matched..matched + whole_words],
    );
    matched += tail;
    if tail < whole_words {
        return matched;
    }

    matched + match_len_bytes(&left_window[matched..], &right_window[matched..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use fearless_simd::{Level, dispatch};

    /// Straight-line reference implementation used as the oracle.
    fn baseline(data: &[u8], left: usize, right: usize, limit: usize) -> usize {
        (0..limit)
            .take_while(|&i| data[left + i] == data[right + i])
            .count()
    }

    fn measure(data: &[u8], left: usize, right: usize, limit: usize) -> usize {
        let level = Level::try_detect().unwrap_or_else(Level::baseline);
        dispatch!(level, simd => find_match_length(simd, data, left, right, limit))
    }

    fn measure_fallback(data: &[u8], left: usize, right: usize, limit: usize) -> usize {
        let level = Level::fallback();
        dispatch!(level, simd => find_match_length(simd, data, left, right, limit))
    }

    #[test]
    fn loads_read_little_endian_words_at_every_offset() {
        let data: Vec<u8> = (1..=32u8).collect();
        for offset in 0..=(data.len() - 8) {
            let mut expected = [0u8; 8];
            expected.copy_from_slice(&data[offset..offset + 8]);
            assert_eq!(load_u64_le(&data, offset), u64::from_le_bytes(expected));
        }
    }

    #[test]
    fn loads_work_at_the_exact_end_of_the_slice() {
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(load_u64_le(&data, 0), 0x0807_0605_0403_0201);
        assert_eq!(load_u64_le(&data, 1), 0);
    }

    #[test]
    fn loads_past_the_end_read_as_zero() {
        let data = [1u8, 2, 3];
        assert_eq!(load_u64_le(&data, 0), 0);
        assert_eq!(load_u64_le(&data, 3), 0);
        assert_eq!(load_u64_le(&data, 99), 0);
    }

    #[test]
    fn reports_every_mismatch_position() {
        let length = 300usize;
        for mismatch in 0..length {
            let mut data = vec![0u8; 2 * length];
            data[length + mismatch] = 1;
            let limit = length;
            assert_eq!(measure(&data, 0, length, limit), mismatch);
            assert_eq!(measure_fallback(&data, 0, length, limit), mismatch);
            assert_eq!(baseline(&data, 0, length, limit), mismatch);
        }
    }

    #[test]
    fn respects_every_limit() {
        let data = vec![7u8; 512];
        for limit in 0..=256 {
            assert_eq!(measure(&data, 0, 256, limit), limit);
            assert_eq!(measure_fallback(&data, 0, 256, limit), limit);
        }
    }

    #[test]
    fn matches_the_baseline_on_pseudo_random_data() {
        let mut data = vec![0u8; 4096];
        let mut state = 0x1234_5678u32;
        for byte in data.iter_mut() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *byte = (state >> 16) as u8 & 0x03;
        }
        for left in (0..1024).step_by(7) {
            for right in (1024..2048).step_by(13) {
                let limit = 512;
                let expected = baseline(&data, left, right, limit);
                assert_eq!(measure(&data, left, right, limit), expected);
                assert_eq!(measure_fallback(&data, left, right, limit), expected);
            }
        }
    }

    #[test]
    fn a_window_that_does_not_fit_the_input_reports_no_match() {
        let data = vec![0u8; 32];
        for (left, right, limit) in [(0usize, 16usize, 17usize), (24, 0, 9), (33, 0, 1)] {
            assert_eq!(measure(&data, left, right, limit), 0);
            assert_eq!(measure_fallback(&data, left, right, limit), 0);
        }
    }

    /// Runs `match_len_at` and `match_len_at_outlined` on one level.
    fn both_scans(level: Level, data: &[u8], prev_ix: usize, cur: &[u8]) -> (usize, usize) {
        dispatch!(level, simd => (
            match_len_at(simd, data, prev_ix, cur),
            match_len_at_outlined(simd, data, prev_ix, cur),
        ))
    }

    #[test]
    fn the_outlined_scan_agrees_with_the_inlined_one_on_every_level() {
        // Runs of equal bytes of every length, so matches end inside the
        // first word, right after it, inside the vector scan and at the limit.
        let mut data = Vec::new();
        for run in 0..80u8 {
            data.extend(core::iter::repeat_n(run % 3, usize::from(run)));
        }
        // Every level the host runs, and the scalar fallback.
        let levels: Vec<Level> = crate::backend::Backend::available()
            .into_iter()
            .map(|backend| backend.0)
            .collect();
        for level in levels {
            for cur_ix in (0..data.len()).step_by(5) {
                for max_length in [0, 1, 7, 8, 9, 16, 17, 63, 200] {
                    let cur = current_window(&data, cur_ix, max_length);
                    for prev_ix in (0..data.len() + 4).step_by(3) {
                        let (inlined, outlined) = both_scans(level, &data, prev_ix, cur);
                        assert_eq!(
                            inlined, outlined,
                            "prev {prev_ix} cur {cur_ix} max {max_length}"
                        );
                        if prev_ix + cur.len() <= data.len() {
                            let limit = cur.len();
                            assert_eq!(outlined, baseline(&data, prev_ix, cur_ix, limit));
                        } else {
                            assert_eq!(outlined, 0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_current_window_is_empty_unless_it_fits_the_buffer() {
        let data = [1u8, 2, 3, 4, 5];
        assert_eq!(current_window(&data, 1, 3), &[2, 3, 4]);
        assert_eq!(current_window(&data, 2, 3), &[3, 4, 5]);
        assert!(current_window(&data, 3, 3).is_empty());
        assert!(current_window(&data, 9, 0).is_empty());
        assert!(current_window(&data, 5, 0).is_empty());
    }

    /// The stride the caller reasons with has to be the one the scan takes.
    ///
    /// `find_match_length` decides "did every step match?" by comparing the
    /// scan's result against the window rounded down to a whole number of
    /// strides. A stride wider than the scan's own step would round the window
    /// up past what the scan can report and truncate a match that ran to the
    /// limit, so the two must agree for every backend.
    #[test]
    fn the_reported_stride_is_the_one_the_vector_scan_takes() {
        fn check<S: Simd>(simd: S) {
            let stride = native_vector_stride::<S>();
            assert!(matches!(stride, 1 | 16 | 32 | 64), "stride {stride}");
            let data = vec![0xCDu8; 4 * stride];
            let left = vec![0xCDu8; 4 * stride];
            assert_eq!(
                match_len_native_vectors(simd, &left, &data),
                4 * stride,
                "a fully matching window has to report every stride"
            );
            // One byte short of two strides: a scan stepping by `stride` sees
            // one whole step, and the caller's guard has to agree.
            let window = 2 * stride - 1;
            assert_eq!(
                match_len_native_vectors(simd, &left[..window], &data[..window]),
                window - window % stride
            );
        }
        dispatch!(Level::try_detect().unwrap_or_else(Level::baseline), simd => check(simd));
        dispatch!(Level::fallback(), simd => check(simd));
    }

    #[test]
    fn common_prefixes_stop_at_the_first_difference_or_the_shortest_bound() {
        let left: Vec<u8> = (0..100u8).collect();
        let mut right = left.clone();
        assert_eq!(common_prefix_len(&left, &right, 100), 100);
        assert_eq!(common_prefix_len(&left, &right, 37), 37);
        assert_eq!(common_prefix_len(&left[..20], &right, 100), 20);
        assert_eq!(common_prefix_len(&[], &right, 100), 0);
        for mismatch in [0usize, 3, 7, 8, 9, 15, 16, 31, 32, 63, 64, 99] {
            right[mismatch] ^= 1;
            assert_eq!(common_prefix_len(&left, &right, 100), mismatch);
            for level in [
                Level::try_detect().unwrap_or_else(Level::baseline),
                Level::fallback(),
            ] {
                let wide =
                    dispatch!(level, simd => common_prefix_len_simd(simd, &left, &right, 100));
                assert_eq!(wide, mismatch, "{mismatch}");
                let bounded =
                    dispatch!(level, simd => common_prefix_len_simd(simd, &left, &right[..50], 80));
                assert_eq!(bounded, mismatch.min(50), "{mismatch}");
            }
            right[mismatch] ^= 1;
        }
    }

    #[test]
    fn overlapping_ranges_extend_like_the_reference() {
        let data = vec![0xABu8; 64];
        assert_eq!(measure(&data, 0, 1, 63), 63);
        assert_eq!(measure_fallback(&data, 0, 1, 63), 63);
    }
}

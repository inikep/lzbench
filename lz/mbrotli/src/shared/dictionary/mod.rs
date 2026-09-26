//! Brotli's built-in static dictionary, and the shallow probe into it.
//!
//! Ports `SearchInStaticDictionary` and `TestStaticDictionaryItem` from
//! `c/enc/hash.h`, together with the word table of `c/common/dictionary.c` and
//! the four-byte-prefix hash of `c/enc/dictionary_hash.c`, all from the pinned
//! reference (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! Only the encoder side is needed here. A dictionary match is emitted as an
//! ordinary distance beyond the end of the window, and the decoder is the side
//! that applies the transform, so the transform table itself never has to be
//! carried: the encoder only computes which transform id a given prefix cut
//! corresponds to.
//!
//! Two searches live here. [`search`] is the shallow probe the greedy
//! qualities use, which asks for one best match; [`all_matches::find_all`] is
//! the exhaustive one qualities ten and eleven use, which reports the best word
//! at every length so the dynamic program can price them all.

#[cfg(feature = "compression")]
pub(crate) mod all_matches;
#[cfg(any(feature = "decompression", feature = "experimental"))]
pub(crate) mod transform;

#[cfg(feature = "compression")]
use crate::shared::constants::HASH_MUL32;
#[cfg(feature = "compression")]
use crate::shared::score::{SearchResult, backward_reference_score};

/// Longest match the static dictionary can produce
/// (`BROTLI_MAX_STATIC_DICTIONARY_MATCH_LEN`).
#[cfg(feature = "compression")]
pub(crate) const MAX_STATIC_DICTIONARY_MATCH_LEN: usize = 37;

/// Word data of the built-in dictionary (`kBrotliDictionaryData`).
///
/// Extracted verbatim from `c/common/dictionary_inc.h` of the pinned
/// reference; Google distributes it under the MIT licence, see
/// `brotli-ffi/vendor/brotli/LICENSE`.
pub(crate) static BUILTIN_WORDS: &[u8; 122_784] = include_bytes!("words.bin");

/// Hash table over four-byte word prefixes (`kStaticDictionaryHashWords`).
///
/// Stored little-endian, two bytes per bucket, from
/// `c/enc/dictionary_hash_inc.h` of the pinned reference.
#[cfg(feature = "compression")]
static HASH_WORDS: &[u8; 2 * NUM_HASH_BUCKETS] = include_bytes!("hash_words.bin");

/// Word lengths of the hash table (`kStaticDictionaryHashLengths`).
#[cfg(feature = "compression")]
static HASH_LENGTHS: &[u8; NUM_HASH_BUCKETS] = include_bytes!("hash_lengths.bin");

/// Number of buckets in the dictionary hash (`BROTLI_ENC_NUM_HASH_BUCKETS`).
#[cfg(feature = "compression")]
const NUM_HASH_BUCKETS: usize = 32_768;

/// Where the words of each length start inside [`BUILTIN_WORDS`].
pub(crate) const BUILTIN_OFFSETS_BY_LENGTH: [u32; 32] = [
    0, 0, 0, 0, 0, 4096, 9216, 21504, 35840, 44032, 53248, 63488, 74752, 87040, 93696, 100_864,
    104_704, 106_752, 108_928, 113_536, 115_968, 118_528, 119_872, 121_280, 122_016, 122_784,
    122_784, 122_784, 122_784, 122_784, 122_784, 122_784,
];

/// Base-2 logarithm of how many words of each length the dictionary holds.
pub(crate) const BUILTIN_SIZE_BITS_BY_LENGTH: [u8; 32] = [
    0, 0, 0, 0, 10, 10, 11, 11, 10, 10, 10, 10, 10, 9, 9, 8, 7, 7, 8, 7, 7, 6, 6, 5, 5, 0, 0, 0, 0,
    0, 0, 0,
];

/// How many prefix cuts have a dedicated transform (`kCutoffTransformsCount`).
#[cfg(feature = "compression")]
pub(super) const CUTOFF_TRANSFORMS_COUNT: usize = 10;

/// Packed transform id per prefix cut (`kCutoffTransforms`).
///
/// Six bits per cut, cut zero in the low bits.
#[cfg(feature = "compression")]
pub(super) const CUTOFF_TRANSFORMS: u64 = 0x071B_520A_DA2D_3200;

/// Returns the dictionary hash of the four bytes at the start of `data`.
///
/// Mirrors `Hash14`: the high bits carry the most mixing from the multiply, so
/// the key is taken from there.
#[inline(always)]
#[cfg(feature = "compression")]
fn hash14(data: &[u8]) -> usize {
    let word = match data.first_chunk::<4>() {
        Some(chunk) => u32::from_le_bytes(*chunk),
        None => 0,
    };
    (word.wrapping_mul(HASH_MUL32) >> (32 - 14)) as usize
}

#[cfg(feature = "compression")]
pub(super) use crate::shared::match_len::common_prefix_len;

/// Running statistics that decide whether probing is still worth it.
///
/// The reference keeps these on the hasher, so they persist for the whole
/// stream: once a hundred and twenty-eight lookups have gone by per match,
/// the encoder stops paying for the probe.
#[derive(Copy, Clone, Debug, Default)]
#[cfg(feature = "compression")]
pub(crate) struct DictionaryStats {
    lookups: usize,
    matches: usize,
}

#[cfg(feature = "compression")]
impl DictionaryStats {
    /// Permanently disables dictionary probing for an independent fragment.
    #[cfg(not(feature = "no_std"))]
    pub(crate) const DISABLED: Self = Self {
        lookups: 128,
        matches: 0,
    };

    /// Returns whether the dictionary has been paying for itself so far.
    const fn is_worth_probing(&self) -> bool {
        self.matches >= (self.lookups >> 7)
    }
}

#[cfg(feature = "experimental")]
#[expect(
    clippy::too_many_arguments,
    reason = "same inputs as the reference static dictionary query"
)]
#[cfg(feature = "compression")]
pub(crate) fn search_custom(
    dictionary: &crate::compressor::core::rfc9841::static_index::StaticCombination,
    stats: &mut DictionaryStats,
    data: &[u8],
    max_length: usize,
    base: usize,
    max_distance: usize,
    out: &mut SearchResult,
    shallow: bool,
) {
    if !stats.is_worth_probing() {
        return;
    }
    for offset in 0..if shallow { 1 } else { 2 } {
        stats.lookups += 1;
        if dictionary.probe(data, max_length, base, max_distance, offset, out) {
            stats.matches += 1;
        }
    }
    if dictionary.probe_extended(data, max_length, base, max_distance, out) {
        stats.matches += 1;
    }
}

/// Tests one hash bucket entry against the data at the current position.
///
/// Mirrors `TestStaticDictionaryItem`. Returns whether the entry produced a
/// match good enough to replace `out`.
#[cfg(feature = "compression")]
fn test_item(
    len: usize,
    word_idx: usize,
    data: &[u8],
    max_length: usize,
    max_backward: usize,
    max_distance: usize,
    out: &mut SearchResult,
) -> bool {
    if len > max_length {
        return false;
    }
    let Some(&size_bits) = BUILTIN_SIZE_BITS_BY_LENGTH.get(len) else {
        return false;
    };
    let Some(&offset) = BUILTIN_OFFSETS_BY_LENGTH.get(len) else {
        return false;
    };
    let offset = offset as usize + len * word_idx;
    let Some(word) = BUILTIN_WORDS.get(offset..offset + len) else {
        return false;
    };

    let matchlen = common_prefix_len(data, word, len);
    if matchlen + CUTOFF_TRANSFORMS_COUNT <= len || matchlen == 0 {
        return false;
    }
    let cut = len - matchlen;
    let transform_id = (cut << 2) + ((CUTOFF_TRANSFORMS >> (cut * 6)) & 0x3F) as usize;
    let backward = max_backward + 1 + word_idx + (transform_id << size_bits);
    if backward > max_distance {
        return false;
    }
    let score = backward_reference_score(matchlen, backward);
    if score < out.score {
        return false;
    }
    out.len = matchlen;
    out.len_code_delta = len as i32 - matchlen as i32;
    out.distance = backward;
    out.score = score;
    true
}

/// Probes the static dictionary for a match at the start of `data`.
///
/// `SHALLOW` limits the probe to the bucket of shorter words, which is what
/// the quick match finders use. It is a const parameter so each match finder
/// gets only its own probe, whether or not this search is inlined into it.
///
/// Mirrors `SearchInStaticDictionary`, including the point at which it gives up
/// on the dictionary entirely for this stream.
#[inline(always)]
#[cfg(feature = "compression")]
pub(crate) fn search<const SHALLOW: bool>(
    stats: &mut DictionaryStats,
    data: &[u8],
    max_length: usize,
    max_backward: usize,
    max_distance: usize,
    out: &mut SearchResult,
) {
    // The give-up test is inlined into the match finder, which calls this
    // at every position that found nothing; the probe itself is not. It
    // works on a copy of the result: handing the caller's out-of-line would
    // pin the search's result to memory for the whole search.
    if !stats.is_worth_probing() {
        return;
    }
    let mut probed = *out;
    if SHALLOW {
        // Most shallow probes land on an empty bucket. Answering those here
        // spares the out-of-line call its prologue and the spill of the
        // caller's live registers; the count is what the probe would have
        // added, so the give-up test sees the same statistics.
        let key = hash14(data) << 1;
        if HASH_LENGTHS.get(key).is_none_or(|&len| len == 0) {
            stats.lookups += 1;
            return;
        }
        probe_shallow(
            stats,
            key,
            data,
            max_length,
            max_backward,
            max_distance,
            &mut probed,
        );
    } else {
        probe_deep(
            stats,
            data,
            max_length,
            max_backward,
            max_distance,
            &mut probed,
        );
    }
    *out = probed;
}

/// Probes the shorter-word bucket `key`, which is known to hold a word.
#[inline(never)]
#[cfg(feature = "compression")]
fn probe_shallow(
    stats: &mut DictionaryStats,
    key: usize,
    data: &[u8],
    max_length: usize,
    max_backward: usize,
    max_distance: usize,
    out: &mut SearchResult,
) {
    probe_buckets::<1>(
        stats,
        key,
        data,
        max_length,
        max_backward,
        max_distance,
        out,
    );
}

/// Probes both buckets of `data`'s hash (`SearchInStaticDictionary` after its
/// give-up test).
#[inline(never)]
#[cfg(feature = "compression")]
fn probe_deep(
    stats: &mut DictionaryStats,
    data: &[u8],
    max_length: usize,
    max_backward: usize,
    max_distance: usize,
    out: &mut SearchResult,
) {
    probe_buckets::<2>(
        stats,
        hash14(data) << 1,
        data,
        max_length,
        max_backward,
        max_distance,
        out,
    );
}

/// Probes the `PROBES` buckets from `key` on.
#[inline(always)]
#[cfg(feature = "compression")]
fn probe_buckets<const PROBES: usize>(
    stats: &mut DictionaryStats,
    key: usize,
    data: &[u8],
    max_length: usize,
    max_backward: usize,
    max_distance: usize,
    out: &mut SearchResult,
) {
    for offset in 0..PROBES {
        let bucket = key + offset;
        stats.lookups += 1;
        let Some(&len) = HASH_LENGTHS.get(bucket) else {
            continue;
        };
        if len == 0 {
            continue;
        }
        let Some(chunk) = HASH_WORDS
            .get(2 * bucket..)
            .and_then(<[u8]>::first_chunk::<2>)
        else {
            continue;
        };
        let word_idx = usize::from(u16::from_le_bytes(*chunk));
        if test_item(
            usize::from(len),
            word_idx,
            data,
            max_length,
            max_backward,
            max_distance,
            out,
        ) {
            stats.matches += 1;
        }
    }
}

#[cfg(all(test, feature = "compression"))]
mod tests {
    use super::*;

    /// Shortest word the dictionary holds.
    const MIN_WORD_LENGTH: usize = 4;

    #[test]
    fn the_word_table_matches_the_reference_layout() {
        assert_eq!(BUILTIN_WORDS.len(), 122_784);
        assert_eq!(BUILTIN_OFFSETS_BY_LENGTH[25], 122_784);
        for len in MIN_WORD_LENGTH..=24 {
            let words = 1usize << BUILTIN_SIZE_BITS_BY_LENGTH[len];
            assert_eq!(
                BUILTIN_OFFSETS_BY_LENGTH[len] as usize + len * words,
                BUILTIN_OFFSETS_BY_LENGTH[len + 1] as usize,
                "length {len} does not tile its region"
            );
        }
    }

    #[test]
    fn the_first_words_are_the_reference_ones() {
        let start = BUILTIN_OFFSETS_BY_LENGTH[4] as usize;
        assert_eq!(&BUILTIN_WORDS[start..start + 8], b"timedown");
    }

    #[test]
    fn the_hash_table_has_one_length_and_word_per_bucket() {
        assert_eq!(HASH_LENGTHS.len(), NUM_HASH_BUCKETS);
        assert_eq!(HASH_WORDS.len(), 2 * NUM_HASH_BUCKETS);
        for bucket in 0..NUM_HASH_BUCKETS {
            let len = usize::from(HASH_LENGTHS[bucket]);
            if len == 0 {
                continue;
            }
            assert!((MIN_WORD_LENGTH..=24).contains(&len), "bucket {bucket}");
            let word = u16::from_le_bytes([HASH_WORDS[2 * bucket], HASH_WORDS[2 * bucket + 1]]);
            assert!(
                usize::from(word) < (1usize << BUILTIN_SIZE_BITS_BY_LENGTH[len]),
                "bucket {bucket} points past the words of length {len}"
            );
        }
    }

    /// Runs a probe with a window small enough that a match can still win.
    #[cfg(feature = "compression")]
    fn probe(data: &[u8], shallow: bool) -> (DictionaryStats, SearchResult) {
        let mut stats = DictionaryStats::default();
        let mut out = SearchResult::empty();
        if shallow {
            search::<true>(
                &mut stats,
                data,
                data.len(),
                1000,
                u32::MAX as usize,
                &mut out,
            );
        } else {
            search::<false>(
                &mut stats,
                data,
                data.len(),
                1000,
                u32::MAX as usize,
                &mut out,
            );
        }
        (stats, out)
    }

    #[test]
    fn a_dictionary_word_is_found_at_its_own_bytes() {
        let (_, out) = probe(b"time is a construct", false);
        assert!(out.is_match());
        assert_eq!(out.len, 4);
        assert!(out.distance > 1000);
    }

    #[test]
    fn a_longer_word_yields_a_longer_match() {
        let (_, out) = probe(b"timestamp and more", false);
        assert!(out.is_match());
        assert_eq!(out.len, 9);
        assert_eq!(out.len_code_delta, 0);
    }

    #[test]
    fn a_word_that_is_not_in_the_dictionary_is_not_matched() {
        let (_, out) = probe(b"\x00\x01\x02\x03\x04\x05\x06\x07", false);
        assert!(!out.is_match());
    }

    #[test]
    fn a_far_away_word_loses_to_the_minimum_score() {
        let mut stats = DictionaryStats::default();
        let mut out = SearchResult::empty();
        let data = b"time is a construct";
        search::<false>(
            &mut stats,
            data,
            data.len(),
            1 << 20,
            u32::MAX as usize,
            &mut out,
        );
        assert!(!out.is_match());
    }

    #[test]
    fn a_distance_beyond_the_limit_is_rejected() {
        let mut stats = DictionaryStats::default();
        let mut out = SearchResult::empty();
        let data = b"time is a construct";
        search::<false>(&mut stats, data, data.len(), 1000, 0, &mut out);
        assert!(!out.is_match());
    }

    #[test]
    fn probing_stops_once_it_has_not_been_paying_off() {
        let mut stats = DictionaryStats {
            lookups: 1 << 20,
            matches: 0,
        };
        let mut out = SearchResult::empty();
        let data = b"time is a construct";
        search::<false>(
            &mut stats,
            data,
            data.len(),
            1000,
            u32::MAX as usize,
            &mut out,
        );
        assert!(!out.is_match());
        assert_eq!(stats.lookups, 1 << 20);
    }

    #[test]
    fn a_shallow_probe_only_visits_one_bucket() {
        let (stats, _) = probe(b"time is a construct", true);
        assert_eq!(stats.lookups, 1);
        let (stats, _) = probe(b"time is a construct", false);
        assert_eq!(stats.lookups, 2);
    }

    #[test]
    fn a_shallow_probe_of_an_empty_bucket_counts_like_the_full_probe() {
        let word = (0u32..)
            .map(u32::to_le_bytes)
            .find(|bytes| HASH_LENGTHS[hash14(bytes) << 1] == 0)
            .unwrap_or_default();
        let data = [word.as_slice(), b"tail"].concat();
        let (stats, out) = probe(&data, true);
        assert_eq!(stats.lookups, 1);
        assert_eq!(stats.matches, 0);
        assert!(!out.is_match());

        let mut full = DictionaryStats::default();
        let mut full_out = SearchResult::empty();
        probe_buckets::<1>(
            &mut full,
            hash14(&data) << 1,
            &data,
            data.len(),
            1000,
            u32::MAX as usize,
            &mut full_out,
        );
        assert_eq!((full.lookups, full.matches), (stats.lookups, stats.matches));
        assert_eq!(full_out.score, out.score);
    }

    #[test]
    fn a_shallow_probe_of_a_filled_bucket_still_tests_its_word() {
        // The word a shallow bucket holds, followed by bytes that end it.
        let (len, word) = (0..NUM_HASH_BUCKETS)
            .step_by(2)
            .filter(|&key| HASH_LENGTHS[key] != 0)
            .map(|key| {
                let len = usize::from(HASH_LENGTHS[key]);
                let index = usize::from(u16::from_le_bytes([
                    HASH_WORDS[2 * key],
                    HASH_WORDS[2 * key + 1],
                ]));
                let offset = BUILTIN_OFFSETS_BY_LENGTH[len] as usize + len * index;
                (len, &BUILTIN_WORDS[offset..offset + len])
            })
            .find(|(_, word)| HASH_LENGTHS[hash14(word) << 1] != 0)
            .unwrap_or_default();
        let data = [word, b"\x00\x00\x00\x00"].concat();
        let (stats, out) = probe(&data, true);
        assert_eq!(stats.lookups, 1);
        assert_eq!(stats.matches, 1);
        assert_eq!(out.len, len);
    }

    #[test]
    fn common_prefixes_stop_at_the_first_mismatch_and_the_limit() {
        assert_eq!(common_prefix_len(b"abcdef", b"abcxyz", 6), 3);
        assert_eq!(common_prefix_len(b"abcdef", b"abcdef", 2), 2);
        assert_eq!(common_prefix_len(b"abc", b"abcdef", 6), 3);
        assert_eq!(common_prefix_len(b"", b"abc", 3), 0);
    }
}

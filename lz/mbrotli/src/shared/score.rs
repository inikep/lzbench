//! Scoring of backward references.
//!
//! Ports the scoring block of `c/enc/hash.h` from the pinned reference
//! (`google/brotli` v1.2.0, commit `028fb5a`). Every comparison the greedy
//! match finders make goes through these integers, so an "equivalent"
//! floating-point estimate would change which match is chosen and therefore the
//! output.
//!
//! This lives beside the static dictionary rather than inside `core::greedy`
//! because the dictionary probe scores its own candidates, and the dictionary is
//! shared. Qualities ten and eleven never use it: they price candidates in `f32`
//! through [`crate::compressor::core::hq::cost`] instead.

use super::fast_log::log2_floor_non_zero;

/// Score credited per byte of a copy (`BROTLI_LITERAL_BYTE_SCORE`).
const LITERAL_BYTE_SCORE: usize = 135;

/// Score charged per bit of the distance (`BROTLI_DISTANCE_BIT_PENALTY`).
const DISTANCE_BIT_PENALTY: usize = 30;

/// Offset that keeps a score positive under the maximal penalty.
///
/// `BROTLI_SCORE_BASE` is `DISTANCE_BIT_PENALTY * 8 * sizeof(size_t)`, which is
/// 1920 on the sixty-four-bit targets this crate supports.
pub(crate) const SCORE_BASE: usize = DISTANCE_BIT_PENALTY * 8 * size_of::<usize>();

/// Smallest score the search accepts as a match at all (`kMinScore`).
pub(crate) const MIN_SCORE: usize = SCORE_BASE + 100;

/// Returns the score of copying `copy_length` bytes from `offset` back.
///
/// The distance is charged its rounded bit length, which is what makes a long
/// match from far away lose to a slightly shorter one nearby.
pub(crate) const fn backward_reference_score(copy_length: usize, offset: usize) -> usize {
    SCORE_BASE + LITERAL_BYTE_SCORE * copy_length
        - DISTANCE_BIT_PENALTY * log2_floor_non_zero(offset) as usize
}

/// Returns the score of a copy that reuses a cached distance.
///
/// A cached distance costs almost nothing to encode, so it is credited a flat
/// bonus instead of the usual bit penalty.
pub(crate) const fn backward_reference_score_using_last_distance(copy_length: usize) -> usize {
    LITERAL_BYTE_SCORE * copy_length + SCORE_BASE + 15
}

/// Returns the penalty for using cache slot `index` rather than the first one.
///
/// The reference packs the table `{39, 41, 43, 45, ...}` into a shifted
/// constant; the low bit of the index is deliberately ignored, so a distance
/// and its neighbour share a penalty.
pub(crate) const fn backward_reference_penalty_using_last_distance(index: usize) -> usize {
    39 + ((0x1CA10usize >> (index & 0xE)) & 0xE)
}

/// One candidate the match finder is considering (`HasherSearchResult`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct SearchResult {
    /// Length of the match, in bytes actually copied.
    pub(crate) len: usize,
    /// Backward distance of the match.
    pub(crate) distance: usize,
    /// Score used to compare this match against the alternatives.
    pub(crate) score: usize,
    /// Difference between the coded length and [`SearchResult::len`].
    pub(crate) len_code_delta: i32,
}

impl SearchResult {
    /// Returns an empty result that only a real match can beat.
    pub(crate) const fn empty() -> Self {
        Self {
            len: 0,
            distance: 0,
            score: MIN_SCORE,
            len_code_delta: 0,
        }
    }

    /// Returns whether the search found anything worth emitting.
    pub(crate) const fn is_match(&self) -> bool {
        self.score > MIN_SCORE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_constants_match_the_reference() {
        assert_eq!(SCORE_BASE, 1920);
        assert_eq!(MIN_SCORE, 2020);
    }

    #[test]
    fn a_longer_match_scores_higher_at_the_same_distance() {
        assert!(backward_reference_score(5, 100) < backward_reference_score(6, 100));
    }

    #[test]
    fn a_nearer_match_scores_higher_at_the_same_length() {
        assert!(backward_reference_score(5, 1 << 20) < backward_reference_score(5, 4));
        assert_eq!(
            backward_reference_score(5, 4) - backward_reference_score(5, 8),
            DISTANCE_BIT_PENALTY
        );
    }

    #[test]
    fn a_cached_distance_beats_the_same_match_spelled_out() {
        assert!(
            backward_reference_score(4, 1024) < backward_reference_score_using_last_distance(4)
        );
    }

    #[test]
    fn cache_penalties_match_the_reference_table() {
        let expected = [
            39usize, 39, 43, 43, 39, 39, 47, 47, 49, 49, 41, 41, 51, 51, 45, 45,
        ];
        for (index, &value) in expected.iter().enumerate() {
            assert_eq!(
                backward_reference_penalty_using_last_distance(index),
                value,
                "cache slot {index}"
            );
        }
    }

    #[test]
    fn a_penalised_cached_score_never_underflows() {
        for index in 0usize..16 {
            let score = backward_reference_score_using_last_distance(2);
            assert!(score > backward_reference_penalty_using_last_distance(index));
        }
    }

    #[test]
    fn an_empty_result_is_not_a_match() {
        let empty = SearchResult::empty();
        assert!(!empty.is_match());
        assert_eq!(empty.score, MIN_SCORE);

        let found = SearchResult {
            score: MIN_SCORE + 1,
            ..empty
        };
        assert!(found.is_match());
    }
}

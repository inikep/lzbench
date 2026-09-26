//! Matching an attached prefix dictionary from inside a match finder.
//!
//! Ports `FindCompoundDictionaryMatch`, `LookupCompoundDictionaryMatch`,
//! `FindAllCompoundDictionaryMatches` and `LookupAllCompoundDictionaryMatches`
//! from `c/enc/hash.h` of the pinned reference (`google/brotli` v1.2.0, commit
//! `028fb5a`).
//!
//! # Addressing
//!
//! The attached dictionaries sit *behind* the start of the stream. A backward
//! distance `d` from position `p` addresses the ring buffer while
//! `d <= max_ring_buffer_distance`, and the dictionary beyond that: the
//! concatenated prefix ends where the stream begins, so
//! `d = max_ring_buffer_distance + total_size - address` for a logical
//! `address` into the concatenation. The reference writes that as one
//! `distance_offset` per attachment, which is what the two search functions
//! here take.
//!
//! # What the search does not do
//!
//! A candidate is measured inside the attachment it was found in and stops at
//! its end, even though the *addressing* spans every attachment. That is the
//! reference's behaviour, not an omission: only the extension of an already
//! emitted command runs on across the seam.
//!
//! Scalar on purpose. The chain is short, the candidates are scattered, and
//! the whole search runs once per position at qualities five and above; the
//! measured cost is in the chain walk, not the byte comparison.

use alloc::vec::Vec;

use fearless_simd::Simd;

use super::context::SharedContextInner;
use crate::shared::match_len::{common_prefix_len, common_prefix_len_simd};
use crate::shared::score::{
    SearchResult, backward_reference_penalty_using_last_distance, backward_reference_score,
    backward_reference_score_using_last_distance,
};

/// Shortest match the chain walk will accept, as the reference's `len >= 4`.
const MIN_CHAIN_MATCH: usize = 4;

/// Shortest match the cached-distance probe will accept.
const MIN_CACHED_MATCH: usize = 2;

/// Length the chain walk pre-filters four bytes ending at, at minimum.
const MIN_PREFILTER_LEN: usize = 3;

/// How many cached distances the probe tries, as the reference's four.
const CACHED_DISTANCES: usize = 4;

impl SharedContextInner {
    /// Returns the total number of attached prefix bytes (`total_size`).
    ///
    /// This is the reference's `gap`: the amount every dictionary-addressing
    /// distance is shifted by, and zero for a context with nothing attached.
    pub(crate) fn total_size(&self) -> usize {
        self.dictionaries().prefix().total_len() as usize
    }

    /// Improves `out` with the best attached-dictionary match at `cur_ix`.
    ///
    /// Mirrors `LookupCompoundDictionaryMatch`: every attachment is searched
    /// in attachment order, and each may only replace the incumbent by scoring
    /// strictly higher — so a tie leaves the earlier finder's match in place.
    ///
    /// `max_ring_buffer_distance` is the largest distance that still addresses
    /// the ring buffer, and `max_distance` the largest the distance alphabet
    /// can express.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors LookupCompoundDictionaryMatch, whose parameters are all needed"
    )]
    pub(crate) fn find_match<S: Simd>(
        &self,
        simd: S,
        data: &[u8],
        ring_buffer_mask: usize,
        distance_cache: &[i32],
        cur_ix: usize,
        max_length: usize,
        max_ring_buffer_distance: usize,
        max_distance: usize,
        out: &mut SearchResult,
    ) {
        // Entered here so the vector prefix scan stays inline whether or not
        // the caller's feature context reaches this far.
        simd.vectorize(
            #[inline(always)]
            || {
                let sources = self.dictionaries().prefix();
                // `max_ring_buffer_distance + 1 + total_size - 1`, written the
                // way the reference writes it: the distance of logical
                // address zero.
                let base_offset = max_ring_buffer_distance + self.total_size();
                for attachment in 0..sources.segment_count() {
                    let Some(index) = self.prepared_prefix(attachment) else {
                        continue;
                    };
                    let source = sources.segment(attachment);
                    let chunk_start = sources.segment_start(attachment) as usize;
                    find_in_attachment(
                        simd,
                        index,
                        source,
                        data,
                        ring_buffer_mask,
                        distance_cache,
                        cur_ix,
                        max_length,
                        base_offset - chunk_start,
                        max_distance,
                        out,
                    );
                }
            },
        );
    }

    /// Collects every attached-dictionary match longer than `min_length`.
    ///
    /// Mirrors `LookupAllCompoundDictionaryMatches`, which the high-quality
    /// encoder feeds into its dynamic program. Matches are appended to `found`
    /// as `(distance, length)` in the order the reference produces them, which
    /// is strictly increasing in length within one attachment; `min_length`
    /// rises to the last match found before the next attachment is searched.
    ///
    /// Returns how many were appended, at most `match_limit`.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors LookupAllCompoundDictionaryMatches, whose parameters are all needed"
    )]
    pub(crate) fn find_all_matches(
        &self,
        data: &[u8],
        ring_buffer_mask: usize,
        cur_ix: usize,
        min_length: usize,
        max_length: usize,
        max_ring_buffer_distance: usize,
        max_distance: usize,
        match_limit: usize,
        found: &mut Vec<(usize, usize)>,
    ) -> usize {
        let sources = self.dictionaries().prefix();
        let base_offset = max_ring_buffer_distance + self.total_size();
        let mut min_length = min_length;
        let mut total = 0usize;
        for attachment in 0..sources.segment_count() {
            if total == match_limit {
                break;
            }
            let Some(index) = self.prepared_prefix(attachment) else {
                continue;
            };
            let source = sources.segment(attachment);
            let chunk_start = sources.segment_start(attachment) as usize;
            total += find_all_in_attachment(
                index,
                source,
                data,
                ring_buffer_mask,
                cur_ix,
                min_length,
                max_length,
                base_offset - chunk_start,
                max_distance,
                match_limit - total,
                found,
            );
            if total == match_limit {
                break;
            }
            if let Some(&(_, length)) = found.last() {
                min_length = length;
            }
        }
        total
    }
}

/// Searches one attachment (`FindCompoundDictionaryMatch`).
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors FindCompoundDictionaryMatch, whose parameters are all needed"
)]
#[inline(always)]
fn find_in_attachment<S: Simd>(
    simd: S,
    index: &super::prepared::PreparedPrefix,
    source: &[u8],
    data: &[u8],
    ring_buffer_mask: usize,
    distance_cache: &[i32],
    cur_ix: usize,
    max_length: usize,
    distance_offset: usize,
    max_distance: usize,
    out: &mut SearchResult,
) {
    let source_size = source.len();
    // Distances at or below this address the ring buffer, not this attachment.
    let boundary = distance_offset.saturating_sub(source_size);
    let cur_ix_masked = cur_ix & ring_buffer_mask;
    let Some(target) = data.get(cur_ix_masked..) else {
        return;
    };
    let Some(head) = target
        .first_chunk::<8>()
        .map(|bytes| u64::from_le_bytes(*bytes))
    else {
        return;
    };

    let mut best_score = out.score;
    let mut best_len = out.len;

    for (rank, &cached) in distance_cache.iter().take(CACHED_DISTANCES).enumerate() {
        let distance = cached as usize;
        if distance <= boundary || distance > distance_offset {
            continue;
        }
        let offset = distance_offset - distance;
        let Some(candidate) = source.get(offset..) else {
            continue;
        };
        let limit = candidate.len().min(max_length);
        let length = common_prefix_len_simd(simd, candidate, target, limit);
        if length < MIN_CACHED_MATCH {
            continue;
        }
        let mut score = backward_reference_score_using_last_distance(length);
        if best_score >= score {
            continue;
        }
        if rank != 0 {
            score = score.saturating_sub(backward_reference_penalty_using_last_distance(rank));
        }
        if best_score < score {
            best_score = score;
            best_len = best_len.max(length);
            out.len = length;
            out.len_code_delta = 0;
            out.distance = distance;
            out.score = score;
        }
    }

    // Raised so the four-byte pre-filter below can always look back three.
    best_len = best_len.max(MIN_PREFILTER_LEN);

    for item in index.candidates(head) {
        let offset = item as usize;
        let distance = distance_offset - offset;
        if distance > max_distance {
            continue;
        }
        let Some(candidate) = source.get(offset..) else {
            continue;
        };
        let limit = candidate.len().min(max_length);
        if cur_ix_masked + best_len > ring_buffer_mask || best_len >= limit {
            continue;
        }
        // Compare the four bytes ending at `best_len`, which is what lets a
        // candidate that cannot beat the incumbent be rejected without a scan.
        let (Some(left), Some(right)) = (
            candidate
                .get(best_len - MIN_PREFILTER_LEN..)
                .and_then(<[u8]>::first_chunk::<4>),
            target
                .get(best_len - MIN_PREFILTER_LEN..)
                .and_then(<[u8]>::first_chunk::<4>),
        ) else {
            continue;
        };
        if left != right {
            continue;
        }
        let length = common_prefix_len_simd(simd, candidate, target, limit);
        if length < MIN_CHAIN_MATCH {
            continue;
        }
        let score = backward_reference_score(length, distance);
        if best_score < score {
            best_score = score;
            best_len = length;
            out.len = length;
            out.len_code_delta = 0;
            out.distance = distance;
            out.score = score;
        }
    }
}

/// Collects every improving match in one attachment.
///
/// Mirrors `FindAllCompoundDictionaryMatches`, which unlike its single-result
/// sibling probes no cached distances and reports every length improvement
/// rather than only the best.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors FindAllCompoundDictionaryMatches, whose parameters are all needed"
)]
fn find_all_in_attachment(
    index: &super::prepared::PreparedPrefix,
    source: &[u8],
    data: &[u8],
    ring_buffer_mask: usize,
    cur_ix: usize,
    min_length: usize,
    max_length: usize,
    distance_offset: usize,
    max_distance: usize,
    match_limit: usize,
    found: &mut Vec<(usize, usize)>,
) -> usize {
    if match_limit == 0 {
        return 0;
    }
    let cur_ix_masked = cur_ix & ring_buffer_mask;
    let Some(target) = data.get(cur_ix_masked..) else {
        return 0;
    };
    let Some(head) = target
        .first_chunk::<8>()
        .map(|bytes| u64::from_le_bytes(*bytes))
    else {
        return 0;
    };

    let mut best_len = min_length;
    let mut count = 0usize;
    for item in index.candidates(head) {
        let offset = item as usize;
        let distance = distance_offset - offset;
        if distance > max_distance {
            continue;
        }
        let Some(candidate) = source.get(offset..) else {
            continue;
        };
        let limit = candidate.len().min(max_length);
        if cur_ix_masked + best_len > ring_buffer_mask || best_len >= limit {
            continue;
        }
        // A single byte here, not the four the scoring search compares: this
        // one has no incumbent to beat, only a length to exceed.
        if candidate.get(best_len) != target.get(best_len) {
            continue;
        }
        let length = common_prefix_len(candidate, target, limit);
        if length > best_len {
            best_len = length;
            found.push((distance, length));
            count += 1;
            if count == match_limit {
                break;
            }
        }
    }
    count
}

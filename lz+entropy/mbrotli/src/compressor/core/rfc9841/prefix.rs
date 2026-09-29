//! The attached LZ77 prefix, and how a backward distance addresses it.
//!
//! RFC 9841 lets a caller attach up to fifteen dictionaries in front of a
//! stream. They are *not* concatenated in memory: each stays in the buffer the
//! caller handed over, and this module is what makes them behave as one byte
//! sequence anyway — cumulative offsets turn a logical address into a segment
//! and an offset inside it, and the match scan walks from one segment into the
//! next and then on into the stream's own history.
//!
//! Ports the addressing half of `CompoundDictionary` from
//! `c/enc/compound_dictionary.h` of the pinned reference (`google/brotli`
//! v1.2.0, commit `028fb5a`). The reference's own search stops at the end of
//! the segment a candidate was found in; the virtual concatenation modelled by
//! [`PrefixSources::match_length`] is wider than that on purpose, and the
//! difference is recorded as decision D6.
//!
//! The scan here is scalar, and deliberately its own rather than the vector
//! kernel `shared::match_len` gives the encoders. No encoder consults a
//! prefix dictionary yet, so there is no profile that would justify
//! vectorising it, and the repository's rule is to measure first.

use alloc::boxed::Box;
use alloc::vec::Vec;

/// Dictionaries one context may attach (`SHARED_BROTLI_MAX_COMPOUND_DICTS`).
pub(crate) const MAX_PREFIX_DICTIONARIES: usize = 15;

/// Largest single segment the prepared index can address.
///
/// A prepared index stores source offsets in a `u32` whose top bit marks the
/// end of a bucket chain, so an offset has thirty-one usable bits. The
/// reference truncates `source_size` to a `u32` without checking; refusing the
/// segment instead is what keeps a large dictionary from silently indexing its
/// own head.
pub(crate) const MAX_PREFIX_SEGMENT_BYTES: u64 = (1 << 31) - 1;

/// The attached dictionaries, in attachment order, and their logical offsets.
///
/// Attachment order *is* prefix order: the first attachment holds the oldest
/// bytes and the last one the bytes immediately before the stream's own
/// output, which is the order every backward distance is resolved against.
#[derive(Debug, Default)]
pub(crate) struct PrefixSources {
    /// Caller-owned dictionary bytes, oldest first.
    ///
    /// Ordinary owned storage: no reference counting, so a context is a single
    /// owner and moving it between threads costs nothing.
    segments: Box<[Box<[u8]>]>,
    /// Logical start of each segment, with the total length appended.
    ///
    /// Always one longer than `segments`, so `starts[i]..starts[i + 1]` is the
    /// logical span of segment `i` and `starts[len]` is the total.
    starts: Box<[u64]>,
}

impl PrefixSources {
    /// Builds the logical prefix from the segments in attachment order.
    ///
    /// The caller has already checked the count and the sizes; this only lays
    /// the cumulative offsets out, which cannot overflow because every segment
    /// length came from a slice that exists in memory.
    pub(crate) fn new(segments: Vec<Box<[u8]>>) -> Self {
        let mut starts = Vec::with_capacity(segments.len() + 1);
        let mut total = 0u64;
        starts.push(0);
        for segment in &segments {
            total += segment.len() as u64;
            starts.push(total);
        }
        Self {
            segments: segments.into_boxed_slice(),
            starts: starts.into_boxed_slice(),
        }
    }

    /// Returns how many dictionaries are attached, empty ones included.
    pub(crate) const fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Returns the total logical prefix length in bytes.
    pub(crate) fn total_len(&self) -> u64 {
        match self.starts.last() {
            Some(&total) => total,
            None => 0,
        }
    }

    /// Returns whether the prefix addresses no bytes at all.
    ///
    /// True both for a context with nothing attached and for one whose every
    /// attachment was empty: neither can ever be the target of a distance.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.total_len() == 0
    }

    /// Returns the bytes of one attachment.
    pub(crate) fn segment(&self, index: usize) -> &[u8] {
        match self.segments.get(index) {
            Some(segment) => segment,
            None => &[],
        }
    }

    /// Returns the logical address the given attachment starts at.
    pub(crate) fn segment_start(&self, index: usize) -> u64 {
        match self.starts.get(index) {
            Some(&start) => start,
            None => self.total_len(),
        }
    }

    /// Maps a logical address to its segment and the offset inside it.
    ///
    /// Returns `None` past the end of the prefix. A forward scan rather than a
    /// binary search: there are at most [`MAX_PREFIX_DICTIONARIES`] segments,
    /// so the whole offset table is one or two cache lines and a branchless
    /// walk beats the mispredictions a search over fifteen entries would cost.
    /// Empty attachments fall out of the same test, because their span is
    /// half-open and empty.
    pub(crate) fn locate(&self, logical: u64) -> Option<(usize, usize)> {
        if logical >= self.total_len() {
            return None;
        }
        for index in 0..self.segments.len() {
            let end = self.starts.get(index + 1).copied().unwrap_or(0);
            if logical < end {
                let start = self.starts.get(index).copied().unwrap_or(0);
                return Some((index, (logical - start) as usize));
            }
        }
        None
    }

    /// Returns the contiguous bytes from `logical` to the end of its segment.
    ///
    /// Empty past the end of the prefix, so a caller can walk segment by
    /// segment without a separate bounds test.
    pub(crate) fn run_from(&self, logical: u64) -> &[u8] {
        match self.locate(logical) {
            Some((index, offset)) => match self.segment(index).get(offset..) {
                Some(run) => run,
                None => &[],
            },
            None => &[],
        }
    }

    /// Returns the logical address a backward `distance` refers to.
    ///
    /// `max_backward` is the largest distance the ordinary sliding window can
    /// express at this position: distances `1..=max_backward` are ordinary
    /// history, and `max_backward + 1` is the prefix byte immediately before
    /// the stream, which is the *last* logical byte. Anything below or past
    /// the prefix range returns `None` rather than wrapping.
    pub(crate) fn address_of(&self, distance: u64, max_backward: u64) -> Option<u64> {
        let into_prefix = distance.checked_sub(max_backward)?;
        if into_prefix == 0 {
            return None;
        }
        self.total_len().checked_sub(into_prefix)
    }

    /// Returns the backward distance that addresses `logical`.
    ///
    /// The inverse of [`PrefixSources::address_of`]; `None` past the end of the
    /// prefix or when the distance would not fit a `u64`.
    pub(crate) fn distance_of(&self, logical: u64, max_backward: u64) -> Option<u64> {
        let into_prefix = self.total_len().checked_sub(logical)?;
        if into_prefix == 0 {
            return None;
        }
        max_backward.checked_add(into_prefix)
    }

    /// Returns how many bytes of `target` the prefix matches from `logical`.
    ///
    /// Models the virtual concatenation RFC 9841 allows a copy to run over:
    /// the prefix from `logical` to its end, then every following attachment,
    /// then `history` — the stream's own output from its start. The scan stops
    /// after `limit` bytes, at the first difference, or when both sources run
    /// out, and it never reads past any of them.
    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn match_length(
        &self,
        logical: u64,
        history: &[u8],
        target: &[u8],
        limit: usize,
    ) -> usize {
        let limit = limit.min(target.len());
        let total = self.total_len();
        let mut matched = 0usize;
        let mut address = logical;
        while matched < limit && address < total {
            let run = self.run_from(address);
            if run.is_empty() {
                break;
            }
            let window = limit - matched;
            let stepped = common_prefix_len(run, &target[matched..], window);
            matched += stepped;
            address += stepped as u64;
            if stepped < run.len().min(window) {
                return matched;
            }
        }
        if matched < limit {
            matched += common_prefix_len(history, &target[matched..], limit - matched);
        }
        matched
    }
}

#[cfg(any(test, feature = "diagnostics"))]
use crate::shared::match_len::common_prefix_len;

#[cfg(test)]
mod tests {
    use super::*;
    fn sources(segments: &[&[u8]]) -> PrefixSources {
        PrefixSources::new(
            segments
                .iter()
                .map(|segment| segment.to_vec().into_boxed_slice())
                .collect(),
        )
    }

    /// Straight-line scalar oracle for the virtual concatenation.
    ///
    /// Materialises what [`PrefixSources::match_length`] walks without
    /// materialising, which is exactly the property under test.
    fn oracle(
        segments: &[&[u8]],
        logical: u64,
        history: &[u8],
        target: &[u8],
        limit: usize,
    ) -> usize {
        let mut flat: Vec<u8> = segments.concat();
        let tail = flat.split_off(logical as usize);
        let mut stream = tail;
        stream.extend_from_slice(history);
        stream
            .iter()
            .zip(target)
            .take(limit)
            .take_while(|(left, right)| left == right)
            .count()
    }

    fn measure(
        segments: &[&[u8]],
        logical: u64,
        history: &[u8],
        target: &[u8],
        limit: usize,
    ) -> usize {
        sources(segments).match_length(logical, history, target, limit)
    }

    #[test]
    fn an_empty_prefix_addresses_nothing() {
        let sources = PrefixSources::default();
        assert_eq!(sources.segment_count(), 0);
        assert_eq!(sources.total_len(), 0);
        assert!(sources.is_empty());
        assert_eq!(sources.locate(0), None);
        assert!(sources.run_from(0).is_empty());
        assert!(sources.segment(0).is_empty());
        assert_eq!(sources.segment_start(0), 0);
        assert_eq!(sources.segment_start(7), 0);
        assert_eq!(sources.address_of(1, 0), None);
        assert_eq!(sources.distance_of(0, 0), None);
    }

    #[test]
    fn attachment_order_is_oldest_first() {
        let sources = sources(&[b"old", b"middle", b"new"]);
        assert_eq!(sources.segment_count(), 3);
        assert_eq!(sources.total_len(), 12);
        assert!(!sources.is_empty());
        assert_eq!(sources.segment_start(0), 0);
        assert_eq!(sources.segment_start(1), 3);
        assert_eq!(sources.segment_start(2), 9);
        assert_eq!(sources.segment_start(3), 12);
        assert_eq!(sources.locate(0), Some((0, 0)));
        assert_eq!(sources.locate(2), Some((0, 2)));
        assert_eq!(sources.locate(3), Some((1, 0)));
        assert_eq!(sources.locate(8), Some((1, 5)));
        assert_eq!(sources.locate(9), Some((2, 0)));
        assert_eq!(sources.locate(11), Some((2, 2)));
        assert_eq!(sources.locate(12), None);
        assert_eq!(sources.run_from(4), b"iddle");
        assert!(sources.run_from(12).is_empty());
    }

    #[test]
    fn empty_attachments_are_skipped_by_addressing() {
        let sources = sources(&[b"", b"ab", b"", b"cd", b""]);
        assert_eq!(sources.segment_count(), 5);
        assert_eq!(sources.total_len(), 4);
        assert_eq!(sources.locate(0), Some((1, 0)));
        assert_eq!(sources.locate(1), Some((1, 1)));
        assert_eq!(sources.locate(2), Some((3, 0)));
        assert_eq!(sources.locate(3), Some((3, 1)));
        assert_eq!(sources.locate(4), None);
        assert_eq!(sources.run_from(2), b"cd");
    }

    #[test]
    fn the_nearest_prefix_byte_is_the_shortest_prefix_distance() {
        let sources = sources(&[b"old", b"new"]);
        let max_backward = 100u64;
        // One past the ordinary window is the byte immediately before the
        // stream, which is the last logical byte.
        assert_eq!(sources.address_of(101, max_backward), Some(5));
        assert_eq!(sources.address_of(106, max_backward), Some(0));
        // Inside the ordinary window, and past the whole prefix.
        assert_eq!(sources.address_of(100, max_backward), None);
        assert_eq!(sources.address_of(1, max_backward), None);
        assert_eq!(sources.address_of(107, max_backward), None);
    }

    #[test]
    fn addressing_round_trips_through_the_distance() {
        let sources = sources(&[b"alpha", b"beta", b"gamma"]);
        let max_backward = 1 << 20;
        for logical in 0..sources.total_len() {
            let distance = sources
                .distance_of(logical, max_backward)
                .expect("inside the prefix");
            assert!(distance > max_backward);
            assert_eq!(sources.address_of(distance, max_backward), Some(logical));
        }
        assert_eq!(sources.distance_of(sources.total_len(), max_backward), None);
        assert_eq!(
            sources.distance_of(sources.total_len() + 1, max_backward),
            None
        );
    }

    #[test]
    fn distance_arithmetic_saturates_rather_than_wrapping() {
        let sources = sources(&[b"tail"]);
        assert_eq!(sources.address_of(0, u64::MAX), None);
        assert_eq!(sources.address_of(u64::MAX, u64::MAX), None);
        assert_eq!(sources.distance_of(0, u64::MAX), None);
    }

    #[test]
    fn a_match_inside_one_segment_stops_at_the_first_difference() {
        let segments: &[&[u8]] = &[b"the quick brown fox"];
        assert_eq!(measure(segments, 4, b"", b"quick brainy", 64), 8);
        assert_eq!(measure(segments, 4, b"", b"quick brainy", 3), 3);
        assert_eq!(measure(segments, 4, b"", b"xyz", 64), 0);
    }

    #[test]
    fn a_match_crosses_the_seam_between_two_attachments() {
        let segments: &[&[u8]] = &[b"abcdef", b"ghijkl"];
        // Ends one byte before the seam, exactly at it, and one past it.
        assert_eq!(measure(segments, 0, b"", b"abcde", 64), 5);
        assert_eq!(measure(segments, 0, b"", b"abcdef", 64), 6);
        assert_eq!(measure(segments, 0, b"", b"abcdefg", 64), 7);
        assert_eq!(measure(segments, 0, b"", b"abcdefghijkl", 64), 12);
        assert_eq!(measure(segments, 0, b"", b"abcdefgHijkl", 64), 7);
    }

    #[test]
    fn a_match_crosses_from_the_prefix_into_stream_history() {
        let segments: &[&[u8]] = &[b"abc", b"def"];
        let history = b"ghijkl";
        // Stops one byte before the prefix end, exactly at it, one past it,
        // and many bytes past it.
        assert_eq!(measure(segments, 0, history, b"abcde", 64), 5);
        assert_eq!(measure(segments, 0, history, b"abcdef", 64), 6);
        assert_eq!(measure(segments, 0, history, b"abcdefg", 64), 7);
        assert_eq!(measure(segments, 0, history, b"abcdefghijkl", 64), 12);
        // Both sources run out before the limit does.
        assert_eq!(measure(segments, 0, history, b"abcdefghijklm", 64), 12);
        assert_eq!(measure(segments, 3, history, b"defghi", 64), 6);
    }

    #[test]
    fn a_match_starting_past_the_prefix_reads_only_history() {
        let segments: &[&[u8]] = &[b"abc"];
        assert_eq!(measure(segments, 3, b"xyz", b"xyz", 64), 3);
        assert_eq!(measure(segments, 9, b"xyz", b"xyz", 64), 3);
    }

    #[test]
    fn the_scan_agrees_with_a_materialised_oracle_everywhere() {
        let segments: &[&[u8]] = &[
            b"the quick brown fox jumps over the lazy dog and keeps going",
            b"",
            b"the quick brown fox jumps over the lazy dog and stops here!!",
            b"tail",
        ];
        let history: Vec<u8> = (0..200u32)
            .map(|i| b"the quick brown fox "[(i % 20) as usize])
            .collect();
        let flat: Vec<u8> = segments.concat();
        let total = flat.len() as u64;
        for logical in 0..=total {
            for start in [0usize, 1, 7, 33, 58, 59, 60] {
                let mut target: Vec<u8> = flat[start.min(flat.len())..].to_vec();
                target.extend_from_slice(&history);
                for limit in [0usize, 1, 3, 8, 16, 17, 64, 300, 4096] {
                    assert_eq!(
                        measure(segments, logical, &history, &target, limit),
                        oracle(segments, logical, &history, &target, limit),
                        "logical {logical} start {start} limit {limit}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_word_scan_agrees_with_a_byte_by_byte_comparison() {
        let left: Vec<u8> = (0..64u8).collect();
        for shared in 0..=64usize {
            let mut right = left.clone();
            if shared < right.len() {
                right[shared] ^= 0xFF;
            }
            for limit in [0usize, 1, 7, 8, 9, 16, 31, 64, 128] {
                let expected = left
                    .iter()
                    .zip(&right)
                    .take(limit)
                    .take_while(|(a, b)| a == b)
                    .count();
                assert_eq!(
                    common_prefix_len(&left, &right, limit),
                    expected,
                    "shared {shared} limit {limit}"
                );
            }
        }
        // A window shorter than the other, and both empty.
        assert_eq!(common_prefix_len(&left[..3], &left, 64), 3);
        assert_eq!(common_prefix_len(&[], &left, 64), 0);
    }
}

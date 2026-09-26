//! The binary-tree match finder qualities ten and eleven search with.
//!
//! Ports `hash_to_binary_tree_inc.h` (H10) and the `BackwardMatch`
//! representation of `c/enc/hash.h` from the pinned reference
//! (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! Every hash bucket holds a binary tree of the sequences whose first four
//! bytes share a hash. The tree is ordered lexicographically by the bytes at
//! each position and is a max-heap by position, so one traversal both collects
//! every match worth considering, in increasing length order, and re-roots the
//! tree at the current position. That ordering is what the dynamic program
//! above relies on: it walks the matches expecting each to be longer than the
//! last.

use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use fearless_simd::{Simd, SimdBase, SimdMask, u8x32};

use crate::shared::constants::{HASH_MUL32, WINDOW_GAP};
use crate::shared::dictionary::MAX_STATIC_DICTIONARY_MATCH_LEN;
use crate::shared::dictionary::all_matches::{self, INVALID_MATCH};
use crate::shared::match_len::find_match_length;

/// Base-2 logarithm of the number of hash buckets (`BUCKET_BITS`).
const BUCKET_BITS: u32 = 17;

/// Number of hash buckets.
const BUCKET_SIZE: usize = 1 << BUCKET_BITS;

/// How many bytes two sequences are compared over (`MAX_TREE_COMP_LENGTH`).
pub(crate) const MAX_TREE_COMP_LENGTH: usize = 128;

/// How deep one traversal descends (`MAX_TREE_SEARCH_DEPTH`).
const MAX_TREE_SEARCH_DEPTH: usize = 64;

/// Bytes a candidate needs available to be hashed (`HashTypeLength`).
pub(crate) const HASH_TYPE_LENGTH: usize = 4;

/// Bytes a store needs available (`StoreLookahead`).
pub(crate) const STORE_LOOKAHEAD: usize = MAX_TREE_COMP_LENGTH;

/// Most matches one `FindAllMatches` call can return (`MAX_NUM_MATCHES_H10`).
///
/// Sixty-four from the short backward scan plus one per level of the tree. The
/// reference sizes a fixed buffer with it; the match arena here grows instead,
/// so it bounds only what one position can contribute.
const MAX_NUM_MATCHES: usize = 64 + MAX_TREE_SEARCH_DEPTH;

/// The reference's own figure, which the bound above has to reproduce.
const _: () = assert!(MAX_NUM_MATCHES == 128);

/// Positions the range store keeps densely before it starts striding.
const DENSE_TAIL: usize = 63;

/// Positions a range has to span before the sparse stride kicks in.
const SPARSE_THRESHOLD: usize = 512;

/// Stride the range store uses over the sparse prefix.
const SPARSE_STRIDE: usize = 8;

/// One candidate copy the dynamic program may take (`BackwardMatch`).
///
/// The length and the length *code* differ only for a static-dictionary match,
/// where the decoder reconstructs a transformed word longer or shorter than the
/// bytes actually matched. Packing them together keeps the match arena at eight
/// bytes an entry, which matters: quality eleven holds one per position of the
/// whole block.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct BackwardMatch {
    /// Backward distance of the copy.
    pub(crate) distance: u32,
    /// Length in the high twenty-seven bits, length code in the low five.
    length_and_code: u32,
}

impl BackwardMatch {
    /// Creates an ordinary backward reference (`InitBackwardMatch`).
    #[inline(always)]
    pub(crate) const fn new(distance: usize, length: usize) -> Self {
        Self {
            distance: distance as u32,
            length_and_code: (length as u32) << 5,
        }
    }

    /// Creates a static-dictionary reference (`InitDictionaryBackwardMatch`).
    ///
    /// A `length_code` equal to `length` is stored as zero, which is how the
    /// reference distinguishes "no separate code" from code zero — a length
    /// code of zero cannot occur, since the shortest dictionary word is four
    /// bytes.
    #[inline(always)]
    pub(crate) const fn dictionary(distance: usize, length: usize, length_code: usize) -> Self {
        let code = if length == length_code {
            0
        } else {
            length_code
        };
        Self {
            distance: distance as u32,
            length_and_code: ((length as u32) << 5) | (code as u32),
        }
    }

    /// Returns how many bytes this match copies (`BackwardMatchLength`).
    #[inline(always)]
    pub(crate) const fn length(&self) -> usize {
        (self.length_and_code >> 5) as usize
    }

    /// Returns the length the decoder reconstructs (`BackwardMatchLengthCode`).
    #[inline(always)]
    pub(crate) const fn length_code(&self) -> usize {
        let code = self.length_and_code & 31;
        if code != 0 {
            code as usize
        } else {
            self.length()
        }
    }
}

/// Binary-tree match finder (`HashToBinaryTree`, H10).
pub(crate) struct BinaryTreeMatcher {
    window_mask: usize,
    invalid_pos: u32,
    buckets: Vec<u32>,
    /// Two child links per window position: `2 * pos` left, `2 * pos + 1` right.
    ///
    /// Sized by [`BinaryTreeMatcher::prepare`]: a one-shot stream only ever
    /// stores positions below its length, so it needs no more nodes than
    /// that, and a window-sized forest for a sixteen-byte input would cost
    /// more to zero than the input costs to encode. The links of positions a
    /// stream never stored are never followed, so a forest kept from an
    /// earlier stream needs no clearing.
    forest: Vec<u32>,
}

/// Candidate positions the short scan filters per vector compare.
const SHORT_SCAN_LANES: usize = 32;

/// Examines the positions just before `cur_ix`, nearest first, recording every
/// two-byte agreement that extends further than the last (the opening loop of
/// `FindAllMatches`). Returns the longest length found, or one.
///
/// A candidate whose first two bytes disagree with the current position cannot
/// match at all, so while the candidates lie contiguously in the ring their
/// leading bytes are compared thirty-two at a time and only the lanes that
/// agree are measured. The agreeing lanes are still visited nearest first, and
/// the walk still stops at the first match of three bytes or more, so exactly
/// the matches of the reference's byte-by-byte walk are recorded in its order.
/// Candidates the vector loop cannot reach — a window that wraps around the
/// ring, or one within two bytes of the buffer's end — take that walk instead.
#[expect(
    clippy::too_many_arguments,
    reason = "the scan needs every bound FindAllMatches was given"
)]
#[inline(always)]
fn scan_recent_positions<S: Simd>(
    simd: S,
    data: &[u8],
    ring_buffer_mask: usize,
    cur_ix: usize,
    max_length: usize,
    max_backward: usize,
    short_scan: usize,
    matches: &mut Vec<BackwardMatch>,
) -> usize {
    let cur_ix_masked = cur_ix & ring_buffer_mask;
    let mut best_len = 1usize;
    // The distances the reference's `for (i = cur_ix - 1; i > stop; --i)`
    // visits before its backward limit stops it. At position zero `i` wraps,
    // and the limit of zero ends the walk before anything is read; here that
    // is simply an empty span.
    let span = short_scan
        .saturating_sub(1)
        .min(cur_ix.saturating_sub(1))
        .min(max_backward);
    let mut backward = 1usize;

    if let (Some(&first), Some(&second)) = (data.get(cur_ix_masked), data.get(cur_ix_masked + 1)) {
        let first = u8x32::splat(simd, first);
        let second = u8x32::splat(simd, second);
        while backward <= span && best_len <= 2 {
            // Lane `l` of the block holds the candidate at distance
            // `backward + 31 - l`, so the nearest candidate sits in the top
            // lane. A block that would start before the buffer — the window
            // has wrapped, or the position is too close to its start — is
            // left to the byte-by-byte walk below.
            let Some(base) = cur_ix_masked.checked_sub(backward + SHORT_SCAN_LANES - 1) else {
                break;
            };
            let (Some(head), Some(next)) = (
                data.get(base..)
                    .and_then(<[u8]>::first_chunk::<SHORT_SCAN_LANES>),
                data.get(base + 1..)
                    .and_then(<[u8]>::first_chunk::<SHORT_SCAN_LANES>),
            ) else {
                break;
            };
            let agree = u8x32::load_array_ref(simd, head).simd_eq(first)
                & u8x32::load_array_ref(simd, next).simd_eq(second);
            let mut lanes = agree.to_bitmask() as u32;
            // Distances past the span occupy the low lanes; drop them.
            let remaining = span - backward + 1;
            if remaining < SHORT_SCAN_LANES {
                lanes &= u32::MAX << (SHORT_SCAN_LANES - remaining);
            }
            while lanes != 0 && best_len <= 2 {
                let lane = (SHORT_SCAN_LANES - 1) - lanes.leading_zeros() as usize;
                lanes &= !(1u32 << lane);
                let distance = backward + (SHORT_SCAN_LANES - 1) - lane;
                let len = find_match_length(simd, data, base + lane, cur_ix_masked, max_length);
                if len > best_len {
                    best_len = len;
                    matches.push(BackwardMatch::new(distance, len));
                }
            }
            backward += SHORT_SCAN_LANES;
        }
    }

    // Whatever the vector loop could not cover, exactly as the reference
    // walks it: two bytes compared, then measured only when they agree.
    while backward <= span && best_len <= 2 {
        let prev_ix = cur_ix.wrapping_sub(backward) & ring_buffer_mask;
        if data.get(cur_ix_masked) == data.get(prev_ix)
            && data.get(cur_ix_masked + 1) == data.get(prev_ix + 1)
        {
            let len = find_match_length(simd, data, prev_ix, cur_ix_masked, max_length);
            if len > best_len {
                best_len = len;
                matches.push(BackwardMatch::new(backward, len));
            }
        }
        backward += 1;
    }
    best_len
}

impl BinaryTreeMatcher {
    /// Creates a matcher over a window of `1 << lgwin` bytes.
    ///
    /// The forest is allocated on the first [`BinaryTreeMatcher::prepare`],
    /// once the shape of the stream is known.
    pub(crate) fn new(lgwin: usize) -> Self {
        let window_mask = (1usize << lgwin) - 1;
        Self {
            window_mask,
            invalid_pos: (0u32).wrapping_sub(window_mask as u32),
            buckets: Vec::new(),
            forest: Vec::new(),
        }
    }

    /// Returns the bucket of the four bytes at `offset` (`HashBytes`).
    #[inline(always)]
    fn hash(data: &[u8], offset: usize) -> usize {
        let word = match data.get(offset..).and_then(<[u8]>::first_chunk::<4>) {
            Some(chunk) => u32::from_le_bytes(*chunk),
            None => 0,
        };
        (word.wrapping_mul(HASH_MUL32) >> (32 - BUCKET_BITS)) as usize
    }

    /// Returns the bytes this match finder keeps allocated.
    pub(crate) fn retained_bytes(&self) -> usize {
        (self.buckets.capacity() + self.forest.capacity()) * size_of::<u32>()
    }

    /// Empties every tree and makes sure the forest can hold the stream
    /// (`Prepare` plus the sizing `HashMemAllocInBytes` does).
    ///
    /// `one_shot` is the reference's `position == 0 && is_last`: the whole
    /// stream is the `input_size` bytes of this block, so only that many
    /// positions are ever stored. Any other stream may store every window
    /// position. The forest only grows, so a reused matcher keeps the largest
    /// forest it has needed.
    pub(crate) fn prepare(&mut self, one_shot: bool, input_size: usize) {
        // The first preparation writes the buckets once, as the reference's
        // uninitialised allocation plus fill does, rather than zeroing them
        // first.
        if self.buckets.is_empty() {
            self.buckets.resize(BUCKET_SIZE, self.invalid_pos);
        } else {
            self.buckets.fill(self.invalid_pos);
        }
        let window = self.window_mask + 1;
        let num_nodes = if one_shot {
            input_size.min(window)
        } else {
            window
        };
        let needed = 2 * num_nodes;
        if self.forest.len() < needed {
            // Take the forest from the allocator's zeroing path instead of
            // writing it. A stream that is not one final block reserves a
            // window of nodes whatever its length is — eight gibibytes at a
            // thirty-bit window — and reads almost none of them; the reference
            // allocates the same reservation without touching its pages, so
            // filling them here would cost what the reference never pays. The
            // old forest is released first, so the peak holds one of the two.
            // Its contents are already unreachable: every bucket has just been
            // reset to `invalid_pos`, and a node no chain reaches is never
            // read.
            drop(mem::take(&mut self.forest));
            self.forest = vec![0u32; needed];
        }
    }

    /// Returns the forest index of the left child of `pos`.
    #[inline(always)]
    const fn left(&self, pos: usize) -> usize {
        2 * (pos & self.window_mask)
    }

    /// Returns the forest index of the right child of `pos`.
    #[inline(always)]
    const fn right(&self, pos: usize) -> usize {
        2 * (pos & self.window_mask) + 1
    }

    /// Reads a forest link, treating an out-of-range index as empty.
    #[inline(always)]
    fn link(&self, index: usize) -> u32 {
        self.forest.get(index).copied().unwrap_or(self.invalid_pos)
    }

    /// Writes a forest link, ignoring an out-of-range index.
    #[inline(always)]
    fn set_link(&mut self, index: usize, value: u32) {
        if let Some(slot) = self.forest.get_mut(index) {
            *slot = value;
        }
    }

    /// Walks the tree at `cur_ix`, collecting matches and re-rooting it.
    ///
    /// Mirrors `StoreAndFindMatches`. `matches` is appended to only when
    /// `collect` is set; a plain store passes `false` and keeps the traversal
    /// purely for its side effect on the tree.
    ///
    /// The tree is only re-rooted when a full [`MAX_TREE_COMP_LENGTH`] bytes of
    /// lookahead are available, because the final sort order of a shorter
    /// sequence is not yet known.
    ///
    /// Must be called with strictly increasing `cur_ix`.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors StoreAndFindMatches, whose parameters are all needed"
    )]
    #[inline(always)]
    fn store_and_find_matches<S: Simd>(
        &mut self,
        simd: S,
        data: &[u8],
        cur_ix: usize,
        ring_buffer_mask: usize,
        max_length: usize,
        max_backward: usize,
        best_len: &mut usize,
        matches: &mut Vec<BackwardMatch>,
        collect: bool,
    ) {
        let cur_ix_masked = cur_ix & ring_buffer_mask;
        let max_comp_len = max_length.min(MAX_TREE_COMP_LENGTH);
        let should_reroot_tree = max_length >= MAX_TREE_COMP_LENGTH;
        let key = Self::hash(data, cur_ix_masked);

        let mut prev_ix = self.buckets.get(key).copied().unwrap_or(self.invalid_pos) as usize;
        // The rightmost node of the new root's left subtree, and the leftmost
        // node of its right subtree, both updated as the traversal descends.
        let mut node_left = self.left(cur_ix);
        let mut node_right = self.right(cur_ix);
        let mut best_len_left = 0usize;
        let mut best_len_right = 0usize;

        if should_reroot_tree && let Some(slot) = self.buckets.get_mut(key) {
            *slot = cur_ix as u32;
        }

        let mut depth_remaining = MAX_TREE_SEARCH_DEPTH;
        loop {
            let backward = cur_ix.wrapping_sub(prev_ix);
            let prev_ix_masked = prev_ix & ring_buffer_mask;
            if backward == 0 || backward > max_backward || depth_remaining == 0 {
                if should_reroot_tree {
                    let invalid = self.invalid_pos;
                    self.set_link(node_left, invalid);
                    self.set_link(node_right, invalid);
                }
                return;
            }

            // Both subtrees already agree with the current position over
            // `cur_len` bytes, so the comparison resumes from there.
            let cur_len = best_len_left.min(best_len_right);
            let len = cur_len
                + find_match_length(
                    simd,
                    data,
                    prev_ix_masked + cur_len,
                    cur_ix_masked + cur_len,
                    max_length - cur_len,
                );
            if collect && len > *best_len {
                *best_len = len;
                matches.push(BackwardMatch::new(backward, len));
            }
            if len >= max_comp_len {
                // The two sequences agree as far as the tree can tell them
                // apart, so the old node's children become the new root's.
                if should_reroot_tree {
                    let left = self.link(self.left(prev_ix));
                    let right = self.link(self.right(prev_ix));
                    self.set_link(node_left, left);
                    self.set_link(node_right, right);
                }
                return;
            }

            let cur_byte = data.get(cur_ix_masked + len).copied().unwrap_or(0);
            let prev_byte = data.get(prev_ix_masked + len).copied().unwrap_or(0);
            if cur_byte > prev_byte {
                // The candidate sorts before the current position, so it and
                // its left subtree belong to the new root's left subtree.
                best_len_left = len;
                if should_reroot_tree {
                    self.set_link(node_left, prev_ix as u32);
                }
                node_left = self.right(prev_ix);
                prev_ix = self.link(node_left) as usize;
            } else {
                best_len_right = len;
                if should_reroot_tree {
                    self.set_link(node_right, prev_ix as u32);
                }
                node_right = self.left(prev_ix);
                prev_ix = self.link(node_right) as usize;
            }
            depth_remaining -= 1;
        }
    }

    /// Collects every match at `cur_ix` worth considering (`FindAllMatches`).
    ///
    /// Matches are appended to `matches` with strictly increasing length; the
    /// static-dictionary matches come last, again in increasing length order.
    ///
    /// `short_scan` is sixteen at quality ten and sixty-four at quality eleven:
    /// how many recent positions are examined directly before the tree is
    /// consulted at all.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors FindAllMatches, whose parameters are all needed"
    )]
    pub(crate) fn find_all_matches<S: Simd>(
        &mut self,
        simd: S,
        data: &[u8],
        ring_buffer_mask: usize,
        cur_ix: usize,
        max_length: usize,
        max_backward: usize,
        dictionary_distance: usize,
        max_distance: usize,
        short_scan: usize,
        matches: &mut Vec<BackwardMatch>,
    ) -> usize {
        simd.vectorize(
            #[inline(always)]
            || {
                let start = matches.len();
                let cur_ix_masked = cur_ix & ring_buffer_mask;

                // A short backward scan first: nearby two-byte repeats are cheap to
                // find and the tree, which only indexes four-byte prefixes, misses
                // them entirely.
                let mut best_len = scan_recent_positions(
                    simd,
                    data,
                    ring_buffer_mask,
                    cur_ix,
                    max_length,
                    max_backward,
                    short_scan,
                    matches,
                );

                if best_len < max_length {
                    self.store_and_find_matches(
                        simd,
                        data,
                        cur_ix,
                        ring_buffer_mask,
                        max_length,
                        max_backward,
                        &mut best_len,
                        matches,
                        true,
                    );
                }

                // Static-dictionary words, which sit past every real distance.
                if dictionary_distance > max_distance {
                    return matches.len() - start;
                }
                let min_len = best_len.saturating_add(1).max(4);
                let mut found = [INVALID_MATCH; MAX_STATIC_DICTIONARY_MATCH_LEN + 1];
                if all_matches::find_all(
                    data.get(cur_ix_masked..).unwrap_or_default(),
                    min_len,
                    max_length,
                    &mut found,
                ) {
                    let max_len = max_length.min(MAX_STATIC_DICTIONARY_MATCH_LEN);
                    for length in min_len..=max_len {
                        let Some(&packed) = found.get(length) else {
                            continue;
                        };
                        if packed >= INVALID_MATCH {
                            continue;
                        }
                        let distance = dictionary_distance + (packed >> 5) as usize + 1;
                        if distance <= max_distance {
                            matches.push(BackwardMatch::dictionary(
                                distance,
                                length,
                                (packed & 31) as usize,
                            ));
                        }
                    }
                }
                matches.len() - start
            },
        )
    }

    /// Re-roots the tree at `ix` without collecting matches (`Store`).
    ///
    /// Requires `ix + MAX_TREE_COMP_LENGTH` bytes of the current block to be
    /// available, which the callers guarantee through their store bounds.
    pub(crate) fn store<S: Simd>(&mut self, simd: S, data: &[u8], mask: usize, ix: usize) {
        simd.vectorize(
            #[inline(always)]
            || {
                let max_backward = self.window_mask - WINDOW_GAP + 1;
                let mut best_len = 0usize;
                let mut discard = Vec::new();
                self.store_and_find_matches(
                    simd,
                    data,
                    ix,
                    mask,
                    MAX_TREE_COMP_LENGTH,
                    max_backward,
                    &mut best_len,
                    &mut discard,
                    false,
                );
            },
        );
    }

    /// Stores every position of `start..end`, sparsely at first (`StoreRange`).
    ///
    /// The reference keeps the most recent sixty-three positions dense and
    /// strides by eight over a long prefix: the older positions of a copy that
    /// has already been chosen are worth far less than the ones a following
    /// match might start at.
    pub(crate) fn store_range<S: Simd>(
        &mut self,
        simd: S,
        data: &[u8],
        mask: usize,
        start: usize,
        end: usize,
    ) {
        let mut dense_from = start;
        if start + DENSE_TAIL <= end {
            dense_from = end - DENSE_TAIL;
        }
        if start + SPARSE_THRESHOLD <= dense_from {
            let mut sparse = start;
            while sparse < dense_from {
                self.store(simd, data, mask, sparse);
                sparse += SPARSE_STRIDE;
            }
        }
        for ix in dense_from..end {
            self.store(simd, data, mask, ix);
        }
    }

    /// Stores the positions that span the previous block boundary.
    ///
    /// Mirrors `StitchToPreviousBlock`: their trees need bytes from both
    /// blocks, so they could not be built when the previous block was
    /// processed. The backward limit shrinks with the distance back to each
    /// position, so the traversal never reads window bytes the next block has
    /// already overwritten.
    pub(crate) fn stitch_to_previous_block<S: Simd>(
        &mut self,
        simd: S,
        num_bytes: usize,
        position: usize,
        data: &[u8],
        mask: usize,
    ) {
        if num_bytes < HASH_TYPE_LENGTH - 1 || position < MAX_TREE_COMP_LENGTH {
            return;
        }
        let start = position - MAX_TREE_COMP_LENGTH + 1;
        let end = position.min(start + num_bytes);
        for ix in start..end {
            let max_backward = self.window_mask - (WINDOW_GAP - 1).max(position - ix);
            let mut best_len = 0usize;
            let mut discard = Vec::new();
            self.store_and_find_matches(
                simd,
                data,
                ix,
                mask,
                MAX_TREE_COMP_LENGTH,
                max_backward,
                &mut best_len,
                &mut discard,
                false,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fearless_simd::{Level, dispatch};

    /// Drives the matcher over `0..end` the way the encoder does.
    ///
    /// Every position is queried, which also stores it, so afterwards the tree
    /// holds the whole prefix. Returns the matches found at `end`.
    fn walk(data: &[u8], end: usize, short_scan: usize) -> Vec<BackwardMatch> {
        walk_with(
            Level::try_detect().unwrap_or_else(Level::baseline),
            data,
            end,
            short_scan,
        )
    }

    /// As [`walk`], on a chosen SIMD backend.
    fn walk_with(level: Level, data: &[u8], end: usize, short_scan: usize) -> Vec<BackwardMatch> {
        let mut matcher = BinaryTreeMatcher::new(22);
        matcher.prepare(false, 0);
        let mut out = Vec::new();
        for position in 0..=end {
            out.clear();
            // A dictionary distance past every representable one keeps the
            // static dictionary out of these fixtures.
            dispatch!(level, simd => matcher.find_all_matches(
                simd,
                data,
                usize::MAX,
                position,
                data.len() - position,
                position,
                usize::MAX >> 1,
                0,
                short_scan,
                &mut out,
            ));
        }
        out
    }

    /// Returns the length of the match at `distance` from `cur_ix`.
    fn true_length(data: &[u8], cur_ix: usize, distance: usize) -> usize {
        let prev = cur_ix - distance;
        data[cur_ix..]
            .iter()
            .zip(&data[prev..])
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Returns the longest match at `cur_ix` the tree is obliged to find.
    ///
    /// Only source positions the range store would have kept are considered: a
    /// position is stored when it was reached with a full comparison length of
    /// lookahead, so the last stretch of the block is deliberately excluded.
    fn brute_force_longest(data: &[u8], cur_ix: usize, stored_end: usize) -> usize {
        let mut best = 0usize;
        for prev in 0..cur_ix.min(stored_end) {
            let length = true_length(data, cur_ix, cur_ix - prev);
            best = best.max(length);
        }
        best
    }

    /// Returns how many earlier positions share `cur_ix`'s four-byte prefix.
    ///
    /// The tree only descends [`MAX_TREE_SEARCH_DEPTH`] levels, so a bucket
    /// deeper than that may legitimately hide the longest match.
    fn prefix_occurrences(data: &[u8], cur_ix: usize, stored_end: usize) -> usize {
        let Some(prefix) = data.get(cur_ix..cur_ix + 4) else {
            return usize::MAX;
        };
        (0..cur_ix.min(stored_end))
            .filter(|&prev| data.get(prev..prev + 4) == Some(prefix))
            .count()
    }

    /// Checks the invariants `FindAllMatches` promises its caller.
    fn assert_well_formed(data: &[u8], cur_ix: usize, matches: &[BackwardMatch]) {
        for (index, m) in matches.iter().enumerate() {
            let distance = m.distance as usize;
            assert!(distance >= 1, "match {index} has distance zero");
            assert!(
                distance <= cur_ix,
                "match {index} reaches back past the start of the input"
            );
            assert!(
                true_length(data, cur_ix, distance) >= m.length(),
                "match {index} claims {} bytes it does not have",
                m.length()
            );
        }
        for pair in matches.windows(2) {
            assert!(
                pair[1].length() > pair[0].length(),
                "lengths were not strictly increasing: {matches:?}"
            );
        }
    }

    /// A payload whose second half repeats its first.
    fn repeated() -> Vec<u8> {
        let body: Vec<u8> = (0..300u32).map(|i| (i * 7 % 251) as u8 + 1).collect();
        let mut data = body.clone();
        data.extend_from_slice(&body);
        data.extend_from_slice(&[0u8; 256]);
        data
    }

    /// The reference's byte-by-byte short scan, as the oracle for the vector
    /// one: `FindAllMatches`'s opening loop, transcribed.
    fn byte_walk(
        data: &[u8],
        mask: usize,
        cur_ix: usize,
        max_length: usize,
        max_backward: usize,
        short_scan: usize,
    ) -> (usize, Vec<BackwardMatch>) {
        let simd = fearless_simd::Fallback::new();
        let cur_ix_masked = cur_ix & mask;
        let stop = cur_ix.saturating_sub(short_scan);
        let mut index = cur_ix.wrapping_sub(1);
        let mut best_len = 1usize;
        let mut out = Vec::new();
        while index > stop && best_len <= 2 {
            let backward = cur_ix.wrapping_sub(index);
            if backward > max_backward {
                break;
            }
            let prev_ix = index & mask;
            index = index.wrapping_sub(1);
            if data.get(cur_ix_masked) != data.get(prev_ix)
                || data.get(cur_ix_masked + 1) != data.get(prev_ix + 1)
            {
                continue;
            }
            let len = find_match_length(simd, data, prev_ix, cur_ix_masked, max_length);
            if len > best_len {
                best_len = len;
                out.push(BackwardMatch::new(backward, len));
            }
        }
        (best_len, out)
    }

    #[test]
    fn the_vector_short_scan_agrees_with_the_byte_walk_on_every_backend() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut random = |bound: u32| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as u32 % bound
        };
        // Two symbols make two-byte agreements common and longer ones rare,
        // so the scan visits many lanes; five symbols thin them out; the
        // periodic and constant inputs end the scan at its first candidate.
        let mut corpora = vec![
            (0..1100)
                .map(|_| b'a' + random(2) as u8)
                .collect::<Vec<u8>>(),
            (0..1100).map(|_| b'a' + random(5) as u8).collect(),
            (0..1100u32).map(|i| (i % 5) as u8 + b'A').collect(),
            vec![b'z'; 1100],
        ];
        corpora.push(repeated());

        for level in [
            Level::try_detect().unwrap_or_else(Level::baseline),
            Level::baseline(),
            Level::fallback(),
        ] {
            for data in &corpora {
                // No wrap, then a ring of 512 whose positions wrap around it.
                for mask in [usize::MAX, 511] {
                    for short_scan in [0, 1, 2, 16, 33, 64] {
                        for cur_ix in 0..data.len() - 8 {
                            let cur_ix_masked = cur_ix & mask;
                            let max_length = data.len() - cur_ix_masked - 1;
                            for max_backward in [cur_ix, cur_ix.min(40), 0] {
                                let expected = byte_walk(
                                    data,
                                    mask,
                                    cur_ix,
                                    max_length,
                                    max_backward,
                                    short_scan,
                                );
                                let mut found = Vec::new();
                                let best = dispatch!(level, simd => scan_recent_positions(
                                    simd, data, mask, cur_ix, max_length, max_backward,
                                    short_scan, &mut found,
                                ));
                                assert_eq!(
                                    (best, found),
                                    expected,
                                    "position {cur_ix}, mask {mask:#x}, scan {short_scan}, \
                                     backward limit {max_backward}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_match_packs_its_length_and_distance() {
        let plain = BackwardMatch::new(1234, 56);
        assert_eq!(plain.distance, 1234);
        assert_eq!(plain.length(), 56);
        // No separate code means the code is the length.
        assert_eq!(plain.length_code(), 56);
    }

    #[test]
    fn a_dictionary_match_keeps_a_separate_length_code() {
        let transformed = BackwardMatch::dictionary(9000, 12, 15);
        assert_eq!(transformed.length(), 12);
        assert_eq!(transformed.length_code(), 15);

        // An untransformed word stores no code and reports the length.
        let plain = BackwardMatch::dictionary(9000, 12, 12);
        assert_eq!(plain.length(), 12);
        assert_eq!(plain.length_code(), 12);
    }

    #[test]
    fn an_empty_tree_finds_only_what_the_short_scan_sees() {
        // With nothing stored, the tree contributes nothing; anything found is
        // the short backward scan's doing, and it never looks past its window.
        let data = repeated();
        let mut matcher = BinaryTreeMatcher::new(22);
        matcher.prepare(false, 0);
        let mut out = Vec::new();
        let level = Level::try_detect().unwrap_or_else(Level::baseline);
        dispatch!(level, simd => matcher.find_all_matches(
            simd, &data, usize::MAX, 300, data.len() - 300, 300,
            usize::MAX >> 1, 0, 64, &mut out,
        ));
        assert!(
            out.iter().all(|m| (m.distance as usize) <= 64),
            "the tree returned a match it never stored: {out:?}"
        );
    }

    #[test]
    fn a_repeat_is_found_at_its_full_length() {
        let data = repeated();
        let found = walk(&data, 300, 64);
        assert_well_formed(&data, 300, &found);
        let best = found.last().expect("the repeat was missed");
        assert_eq!((best.distance, best.length()), (300, 300));
    }

    #[test]
    fn the_longest_match_agrees_with_brute_force() {
        // Over a corpus of overlapping repeats the tree has to find, at every
        // position, exactly the longest match a full scan would — wherever the
        // reference's own bounds do not excuse it: the search depth, the
        // comparison length it sorts by, and the tail it never stores.
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        let payload = 4000usize;
        let mut data: Vec<u8> = Vec::new();
        while data.len() < payload {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            // An eight-symbol alphabet: four-byte matches are common, but
            // buckets stay shallow enough that the depth bound rarely bites.
            data.push(b'a' + ((rng >> 24) % 8) as u8);
        }
        data.extend_from_slice(&[0u8; 256]);
        let stored_end = payload - MAX_TREE_COMP_LENGTH;

        let mut matcher = BinaryTreeMatcher::new(22);
        matcher.prepare(false, 0);
        let mut out = Vec::new();
        let level = Level::try_detect().unwrap_or_else(Level::baseline);
        let mut checked = 0usize;
        for position in 0..payload {
            out.clear();
            dispatch!(level, simd => matcher.find_all_matches(
                simd, &data, usize::MAX, position, payload - position, position,
                usize::MAX >> 1, 0, 64, &mut out,
            ));
            assert_well_formed(&data, position, &out);

            let expected =
                brute_force_longest(&data[..payload], position, stored_end).min(payload - position);
            if !(4..=MAX_TREE_COMP_LENGTH).contains(&expected)
                || prefix_occurrences(&data, position, stored_end) > MAX_TREE_SEARCH_DEPTH
            {
                continue;
            }
            // At least as long as brute force, and — by `assert_well_formed`
            // above — genuinely present. The two together are the guarantee:
            // the tree cannot pass by under-reporting, and it cannot pass by
            // claiming a match it does not have. It may legitimately report
            // *more*, because the short backward scan also reaches positions
            // the range store never kept.
            let found = out.last().map_or(0, BackwardMatch::length);
            assert!(
                found >= expected,
                "position {position}: found {found}, brute force {expected}"
            );
            checked += 1;
        }
        assert!(checked > 500, "only {checked} positions were checked");
    }

    #[test]
    fn the_short_scan_finds_a_two_byte_repeat_the_tree_cannot() {
        // The tree indexes four-byte prefixes; a two-byte repeat is only
        // reachable through the short backward scan. Note the scan stops
        // *above* its lower bound, so the oldest position in range is never
        // examined — hence the leading filler byte.
        let mut data = b"Xabcdab".to_vec();
        data.extend_from_slice(&[0u8; 256]);
        let mut matcher = BinaryTreeMatcher::new(22);
        matcher.prepare(false, 0);
        let mut out = Vec::new();
        let level = Level::try_detect().unwrap_or_else(Level::baseline);
        dispatch!(level, simd => matcher.find_all_matches(
            simd, &data, usize::MAX, 5, data.len() - 5, 5,
            usize::MAX >> 1, 0, 64, &mut out,
        ));
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!((out[0].distance, out[0].length()), (4, 2));
    }

    #[test]
    fn the_short_scan_length_bounds_how_far_back_it_looks() {
        // A two-byte repeat thirty positions back is inside quality eleven's
        // sixty-four-position scan and outside quality ten's sixteen.
        // A filler byte first: the scan stops above its lower bound, so the
        // oldest position in range is never examined.
        let mut data = b"Xab".to_vec();
        data.extend_from_slice(&[b'z'; 27]);
        data.extend_from_slice(b"ab");
        data.extend_from_slice(&[0u8; 256]);
        let cur = 30usize;

        for (scan, expected) in [(16usize, 0usize), (64, 1)] {
            let mut matcher = BinaryTreeMatcher::new(22);
            matcher.prepare(false, 0);
            let mut out = Vec::new();
            let level = Level::try_detect().unwrap_or_else(Level::baseline);
            dispatch!(level, simd => matcher.find_all_matches(
                simd, &data, usize::MAX, cur, data.len() - cur, cur,
                usize::MAX >> 1, 0, scan, &mut out,
            ));
            assert_eq!(out.len(), expected, "scan {scan} found {out:?}");
        }
    }

    #[test]
    fn a_range_store_keeps_the_last_positions_dense() {
        // `StoreRange` stores only the final sixty-three positions of a range
        // shorter than the sparse threshold; the rest are skipped outright.
        let body: Vec<u8> = (0..400u32).map(|i| (i * 31 % 251) as u8 + 1).collect();
        let mut data = body.clone();
        data.extend_from_slice(&body);
        data.extend_from_slice(&[0u8; 256]);

        let mut matcher = BinaryTreeMatcher::new(22);
        matcher.prepare(false, 0);
        let level = Level::try_detect().unwrap_or_else(Level::baseline);
        dispatch!(level, simd => matcher.store_range(simd, &data, usize::MAX, 0, 400));

        // A position inside the dense tail is stored: querying its repeat
        // finds it at the right distance.
        let inside = 400 - 10;
        let mut out = Vec::new();
        dispatch!(level, simd => matcher.find_all_matches(
            simd, &data, usize::MAX, 400 + inside, data.len() - 400 - inside,
            400 + inside, usize::MAX >> 1, 0, 0, &mut out,
        ));
        assert!(
            out.iter().any(|m| m.distance == 400),
            "the dense tail was not stored: {out:?}"
        );

        // A position before the tail was skipped, so its repeat is invisible.
        let skipped = 10usize;
        let mut out = Vec::new();
        dispatch!(level, simd => matcher.find_all_matches(
            simd, &data, usize::MAX, 400 + skipped, data.len() - 400 - skipped,
            400 + skipped, usize::MAX >> 1, 0, 0, &mut out,
        ));
        for found in &out {
            assert_ne!(
                found.distance, 400,
                "a skipped position was stored anyway: {out:?}"
            );
        }
    }

    #[test]
    fn a_long_range_store_strides_over_its_prefix() {
        // Past the sparse threshold the prefix is stored every eighth
        // position, so a repeat of a stored one is found and its neighbour is
        // not.
        let body: Vec<u8> = (0..2000u32).map(|i| (i * 17 % 251) as u8 + 1).collect();
        let mut data = body.clone();
        data.extend_from_slice(&body);
        data.extend_from_slice(&[0u8; 256]);

        let mut matcher = BinaryTreeMatcher::new(22);
        matcher.prepare(false, 0);
        let level = Level::try_detect().unwrap_or_else(Level::baseline);
        dispatch!(level, simd => matcher.store_range(simd, &data, usize::MAX, 0, 2000));

        for (offset, stored) in [(800usize, true), (801, false)] {
            let mut out = Vec::new();
            dispatch!(level, simd => matcher.find_all_matches(
                simd, &data, usize::MAX, 2000 + offset, data.len() - 2000 - offset,
                2000 + offset, usize::MAX >> 1, 0, 0, &mut out,
            ));
            assert_eq!(
                out.iter().any(|m| m.distance == 2000),
                stored,
                "offset {offset} was {}stored",
                if stored { "not " } else { "" }
            );
        }
    }

    #[test]
    fn every_backend_finds_the_same_matches() {
        let data = repeated();
        let mut results = Vec::new();
        for level in [
            Level::try_detect().unwrap_or_else(Level::baseline),
            Level::baseline(),
            Level::fallback(),
        ] {
            results.push(walk_with(level, &data, 300, 64));
        }
        assert!(results.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn stitching_stores_the_positions_before_the_boundary() {
        let data = repeated();
        let level = Level::try_detect().unwrap_or_else(Level::baseline);

        // Nothing happens before the tree has a full comparison length of
        // history behind it.
        let mut early = BinaryTreeMatcher::new(22);
        early.prepare(false, 0);
        dispatch!(level, simd => early.stitch_to_previous_block(
            simd, 64, MAX_TREE_COMP_LENGTH - 1, &data, usize::MAX));
        let mut out = Vec::new();
        dispatch!(level, simd => early.find_all_matches(
            simd, &data, usize::MAX, 300, data.len() - 300, 300,
            usize::MAX >> 1, 0, 0, &mut out,
        ));
        assert!(out.is_empty(), "stitching ran too early: {out:?}");

        // With the boundary far enough in, the last comparison length of
        // positions is stored and their repeats become findable.
        let mut late = BinaryTreeMatcher::new(22);
        late.prepare(false, 0);
        dispatch!(level, simd => late.stitch_to_previous_block(
            simd, 300, 300, &data, usize::MAX));
        let mut out = Vec::new();
        // Position 250 is inside the stitched range, so its repeat at 550 is
        // reachable at distance 300.
        dispatch!(level, simd => late.find_all_matches(
            simd, &data, usize::MAX, 550, data.len() - 550, 550,
            usize::MAX >> 1, 0, 0, &mut out,
        ));
        assert!(
            out.iter().any(|m| m.distance == 300),
            "stitching stored nothing: {out:?}"
        );
    }

    #[test]
    fn the_tree_search_depth_is_bounded() {
        // Every position hashes to one bucket, so the tree degenerates into a
        // chain; the traversal must still stop after its fixed depth.
        let mut data = vec![b'q'; 4096];
        data.extend_from_slice(&[0u8; 256]);
        let found = walk(&data, 2048, 64);
        assert!(
            found.len() <= MAX_NUM_MATCHES,
            "collected {} matches",
            found.len()
        );
    }
}

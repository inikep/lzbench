//! The dynamic program's state: one node per byte, and the start-position queue.
//!
//! Ports `ZopfliNode` from `c/enc/backward_references_hq.h`, together with
//! `PosData` and `StartPosQueue` from `c/enc/backward_references_hq.c`, of the
//! pinned reference (`google/brotli` v1.2.0, commit `028fb5a`).
//!
//! A node records the cheapest way found so far of arriving at one byte of the
//! block. Its fields are packed the way the reference packs them — length and
//! length-code modifier in one word, insert length and short distance code in
//! another — because there is one node per input byte and their layout decides
//! how much of the block stays in cache.
//!
//! The union in the reference is a genuine one: `cost` is meaningful during the
//! forward pass, `shortcut` once the node has been evaluated, and `next` while
//! the path is traced back. Rust models it as a plain `u32` with `cost` read
//! and written through the bit pattern, so the three phases keep the reference's
//! exact aliasing without any `unsafe`.

/// The reference's stand-in for infinity (`kInfinity`), about `2^127`.
pub(crate) const INFINITY: f32 = 1.7e38;

/// Mask of the copy length inside [`ZopfliNode::length`].
const COPY_LEN_MASK: u32 = 0x01FF_FFFF;

/// Shift of the length-code modifier inside [`ZopfliNode::length`].
const LEN_CODE_SHIFT: u32 = 25;

/// Mask of the insert length inside [`ZopfliNode::dcode_insert_length`].
const INSERT_LEN_MASK: u32 = 0x07FF_FFFF;

/// Shift of the short distance code inside [`ZopfliNode::dcode_insert_length`].
const DCODE_SHIFT: u32 = 27;

/// Number of positions the start queue keeps.
pub(crate) const START_POS_QUEUE_SIZE: usize = 8;

/// The cheapest known way of reaching one byte of the block (`ZopfliNode`).
#[derive(Copy, Clone, Debug)]
pub(crate) struct ZopfliNode {
    /// Copy length in the low twenty-five bits, `len + 9 - len_code` above it.
    pub(crate) length: u32,
    /// Distance of the copy that arrives here.
    pub(crate) distance: u32,
    /// Insert length in the low twenty-seven bits, short distance code above.
    pub(crate) dcode_insert_length: u32,
    /// Cost, then shortcut, then next: see the module documentation.
    payload: u32,
}

impl Default for ZopfliNode {
    /// Returns the stub every node is initialised to (`BrotliInitZopfliNodes`).
    fn default() -> Self {
        Self {
            length: 1,
            distance: 0,
            dcode_insert_length: 0,
            payload: INFINITY.to_bits(),
        }
    }
}

impl ZopfliNode {
    /// Returns how many bytes the copy arriving here spans.
    #[inline(always)]
    pub(crate) const fn copy_length(&self) -> u32 {
        self.length & COPY_LEN_MASK
    }

    /// Returns the copy length the decoder reconstructs.
    #[inline(always)]
    pub(crate) const fn length_code(&self) -> u32 {
        let modifier = self.length >> LEN_CODE_SHIFT;
        #[cfg(feature = "experimental")]
        if modifier >= 96 {
            return modifier - 96;
        }
        self.copy_length() + 9 - modifier
    }

    /// Returns the distance of the copy arriving here.
    #[inline(always)]
    pub(crate) const fn copy_distance(&self) -> u32 {
        self.distance
    }

    /// Returns the distance code the copy is written with.
    ///
    /// A short code was stored one higher than its value, so zero means the
    /// distance is spelled out in full.
    #[inline(always)]
    pub(crate) const fn distance_code(&self) -> u32 {
        let short_code = self.dcode_insert_length >> DCODE_SHIFT;
        if short_code == 0 {
            self.copy_distance() + crate::shared::distance::NUM_DISTANCE_SHORT_CODES - 1
        } else {
            short_code - 1
        }
    }

    /// Returns how many literals precede the copy arriving here.
    #[inline(always)]
    pub(crate) const fn insert_length(&self) -> u32 {
        self.dcode_insert_length & INSERT_LEN_MASK
    }

    /// Returns how many input bytes the whole command consumes.
    #[inline(always)]
    pub(crate) const fn command_length(&self) -> u32 {
        self.copy_length() + self.insert_length()
    }

    /// Returns the cost of reaching this byte, during the forward pass.
    #[inline(always)]
    pub(crate) const fn cost(&self) -> f32 {
        f32::from_bits(self.payload)
    }

    /// Sets the cost of reaching this byte.
    #[inline(always)]
    pub(crate) const fn set_cost(&mut self, cost: f32) {
        self.payload = cost.to_bits();
    }

    /// Returns the position whose command supplies the next cached distance.
    #[inline(always)]
    pub(crate) const fn shortcut(&self) -> u32 {
        self.payload
    }

    /// Sets the distance shortcut, overwriting the cost.
    #[inline(always)]
    pub(crate) const fn set_shortcut(&mut self, shortcut: u32) {
        self.payload = shortcut;
    }

    /// Returns the offset to the next node on the chosen path.
    #[inline(always)]
    pub(crate) const fn next(&self) -> u32 {
        self.payload
    }

    /// Sets the offset to the next node on the chosen path.
    #[inline(always)]
    pub(crate) const fn set_next(&mut self, next: u32) {
        self.payload = next;
    }

    /// Records the command arriving at this node, which sits `len` bytes past
    /// `pos`: the copy of `len` bytes from `dist` back that follows the
    /// literals since `start_pos`.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors UpdateZopfliNode, whose parameters are all needed"
    )]
    #[inline(always)]
    pub(crate) fn record(
        &mut self,
        pos: usize,
        start_pos: usize,
        len: usize,
        len_code: usize,
        dist: usize,
        short_code: usize,
        cost: f32,
    ) {
        let modifier = (len + 9 - len_code) as u32;
        // The built-in dictionary never reaches 96. These upper tags
        // encode an absolute base length for long custom transforms.
        #[cfg(feature = "experimental")]
        let modifier = if modifier >= 96 {
            96 + len_code as u32
        } else {
            modifier
        };
        self.length = (len as u32) | (modifier << LEN_CODE_SHIFT);
        self.distance = dist as u32;
        self.dcode_insert_length =
            ((short_code as u32) << DCODE_SHIFT) | ((pos - start_pos) as u32);
        self.set_cost(cost);
    }
}

/// One candidate command start, with the distance cache it would carry.
#[derive(Copy, Clone, Debug)]
pub(crate) struct PosData {
    /// Position within the block the command would start at.
    pub(crate) pos: usize,
    /// The four cached distances in effect there.
    pub(crate) distance_cache: [i32; 4],
    /// How much cheaper reaching `pos` is than coding it all as literals.
    pub(crate) costdiff: f32,
    /// Cost of reaching `pos`.
    pub(crate) cost: f32,
}

impl Default for PosData {
    /// Returns a zeroed candidate.
    fn default() -> Self {
        Self {
            pos: 0,
            distance_cache: [0; 4],
            costdiff: 0.0,
            cost: 0.0,
        }
    }
}

/// The eight most promising command starts (`StartPosQueue`).
///
/// A ring of eight kept sorted by cost difference, cheapest first. Pushing is
/// an insertion into the sorted order rather than a re-sort, which is what
/// bounds the work per position.
pub(crate) struct StartPosQueue {
    queue: [PosData; START_POS_QUEUE_SIZE],
    idx: usize,
}

impl Default for StartPosQueue {
    /// Returns an empty queue (`InitStartPosQueue`).
    fn default() -> Self {
        Self {
            queue: [PosData::default(); START_POS_QUEUE_SIZE],
            idx: 0,
        }
    }
}

impl StartPosQueue {
    /// Empties the queue without releasing its storage.
    pub(crate) const fn clear(&mut self) {
        self.idx = 0;
    }

    /// Returns how many candidates the queue holds (`StartPosQueueSize`).
    pub(crate) const fn len(&self) -> usize {
        if self.idx < START_POS_QUEUE_SIZE {
            self.idx
        } else {
            START_POS_QUEUE_SIZE
        }
    }

    /// Adds `posdata`, keeping the queue sorted (`StartPosQueuePush`).
    pub(crate) const fn push(&mut self, posdata: PosData) {
        let start = !self.idx & (START_POS_QUEUE_SIZE - 1);
        self.idx += 1;
        let len = self.len();
        let mut here = start;
        // Costs are finite. Shift the cheaper prefix once; the untouched suffix
        // is sorted, and inserting before ties preserves the reference order.
        let mut offset = 1;
        while offset < len {
            let next = (start + offset) & (START_POS_QUEUE_SIZE - 1);
            if posdata.costdiff <= self.queue[next].costdiff {
                break;
            }
            self.queue[here] = self.queue[next];
            here = next;
            offset += 1;
        }
        self.queue[here] = posdata;
    }

    /// Returns the `k`th cheapest candidate (`StartPosQueueAt`).
    pub(crate) fn at(&self, k: usize) -> &PosData {
        &self.queue[k.wrapping_sub(self.idx) & (START_POS_QUEUE_SIZE - 1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Builds a candidate whose only interesting field is its cost difference.
    fn candidate(pos: usize, costdiff: f32) -> PosData {
        PosData {
            pos,
            costdiff,
            ..PosData::default()
        }
    }

    #[test]
    fn a_fresh_node_is_the_reference_stub() {
        let node = ZopfliNode::default();
        assert_eq!(node.length, 1);
        assert_eq!(node.distance, 0);
        assert_eq!(node.dcode_insert_length, 0);
        assert_eq!(node.cost(), INFINITY);
    }

    #[test]
    fn a_node_round_trips_its_packed_fields() {
        let mut nodes = vec![ZopfliNode::default(); 64];
        nodes[10 + 20].record(10, 4, 20, 20, 1234, 0, 7.5);
        let node = nodes[30];
        assert_eq!(node.copy_length(), 20);
        assert_eq!(node.length_code(), 20);
        assert_eq!(node.copy_distance(), 1234);
        assert_eq!(node.insert_length(), 6);
        assert_eq!(node.command_length(), 26);
        assert_eq!(node.cost(), 7.5);
    }

    #[test]
    fn a_dictionary_length_code_survives_the_packing() {
        // A transformed dictionary word codes as a different length from the
        // one it copies, in either direction.
        let mut nodes = vec![ZopfliNode::default(); 64];
        nodes[10].record(0, 0, 10, 14, 9000, 0, 1.0);
        assert_eq!(nodes[10].copy_length(), 10);
        assert_eq!(nodes[10].length_code(), 14);

        nodes[12].record(0, 0, 12, 9, 9000, 0, 1.0);
        assert_eq!(nodes[12].copy_length(), 12);
        assert_eq!(nodes[12].length_code(), 9);
    }

    #[test]
    fn a_short_distance_code_is_stored_one_higher() {
        let mut nodes = vec![ZopfliNode::default(); 64];
        // Short code zero — the last distance — is stored as one.
        nodes[5].record(0, 0, 5, 5, 40, 1, 1.0);
        assert_eq!(nodes[5].distance_code(), 0);
        // No short code means the distance is spelled out.
        nodes[6].record(0, 0, 6, 6, 40, 0, 1.0);
        assert_eq!(nodes[6].distance_code(), 40 + 15);
    }

    #[test]
    fn the_payload_serves_all_three_phases_in_turn() {
        let mut node = ZopfliNode::default();
        node.set_cost(12.25);
        assert_eq!(node.cost(), 12.25);
        node.set_shortcut(99);
        assert_eq!(node.shortcut(), 99);
        node.set_next(7);
        assert_eq!(node.next(), 7);
    }

    #[test]
    fn an_empty_queue_has_no_candidates() {
        let queue = StartPosQueue::default();
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn the_queue_returns_candidates_cheapest_first() {
        let mut queue = StartPosQueue::default();
        for (pos, costdiff) in [(0usize, 5.0f32), (1, 3.0), (2, 9.0), (3, 1.0)] {
            queue.push(candidate(pos, costdiff));
        }
        assert_eq!(queue.len(), 4);
        let order: Vec<usize> = (0..queue.len()).map(|k| queue.at(k).pos).collect();
        assert_eq!(order, vec![3, 1, 0, 2]);
    }

    #[test]
    fn the_queue_never_grows_past_eight_and_stays_sorted() {
        // Each push overwrites one ring slot, so which candidate is evicted
        // depends on where the swaps have moved things — the reference makes
        // no promise about that. What it does maintain is the size bound and
        // the ordering, which is all the search reads.
        let mut rng = 0x2468_ACE0_1357_9BDFu64;
        let mut queue = StartPosQueue::default();
        for pos in 0..200usize {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let costdiff = ((rng >> 40) as i32 - 8192) as f32 / 64.0;
            queue.push(candidate(pos, costdiff));

            assert!(queue.len() <= START_POS_QUEUE_SIZE);
            assert_eq!(queue.len(), (pos + 1).min(START_POS_QUEUE_SIZE));
            let costs: Vec<f32> = (0..queue.len()).map(|k| queue.at(k).costdiff).collect();
            assert!(
                costs.windows(2).all(|pair| pair[0] <= pair[1]),
                "queue out of order after {pos} pushes: {costs:?}"
            );
        }
    }

    #[test]
    fn queue_insertion_preserves_reference_ties_and_eviction_after_wraps() {
        let mut actual = StartPosQueue::default();
        let mut reference = StartPosQueue::default();
        let mut random = 0x2468_ACE0_1357_9BDFu64;
        for pos in 0..4096 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let costdiff = match pos % 8 {
                0 => 0.0,
                1 => -0.0,
                2 => f32::MIN,
                3 => f32::MAX,
                _ => (random % 17) as f32 - 8.0,
            };
            let incoming = PosData {
                pos,
                distance_cache: [pos as i32, 11, 15, 16],
                costdiff,
                cost: pos as f32,
            };
            actual.push(incoming);

            // The reference always visits the entire suffix with adjacent swaps.
            let start = !reference.idx & (START_POS_QUEUE_SIZE - 1);
            reference.idx += 1;
            reference.queue[start] = incoming;
            for offset in start..start + reference.len().saturating_sub(1) {
                let here = offset & (START_POS_QUEUE_SIZE - 1);
                let next = (offset + 1) & (START_POS_QUEUE_SIZE - 1);
                if reference.queue[here].costdiff > reference.queue[next].costdiff {
                    reference.queue.swap(here, next);
                }
            }
            for k in 0..actual.len() {
                let (left, right) = (actual.at(k), reference.at(k));
                assert_eq!(left.pos, right.pos, "position {pos}, rank {k}");
                assert_eq!(left.distance_cache, right.distance_cache);
                assert_eq!(left.costdiff.to_bits(), right.costdiff.to_bits());
                assert_eq!(left.cost.to_bits(), right.cost.to_bits());
            }
        }
    }

    #[test]
    fn a_full_queue_holds_eight_distinct_candidates() {
        let mut queue = StartPosQueue::default();
        for pos in 0..8usize {
            queue.push(candidate(pos, (8 - pos) as f32));
        }
        let mut seen: Vec<usize> = (0..queue.len()).map(|k| queue.at(k).pos).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn a_late_cheap_candidate_reaches_the_front() {
        let mut queue = StartPosQueue::default();
        for pos in 0..8usize {
            queue.push(candidate(pos, 10.0 + pos as f32));
        }
        queue.push(candidate(100, -1.0));
        assert_eq!(queue.at(0).pos, 100);
        assert_eq!(queue.len(), START_POS_QUEUE_SIZE);
    }

    #[test]
    fn clearing_forgets_every_candidate() {
        let mut queue = StartPosQueue::default();
        for pos in 0..5usize {
            queue.push(candidate(pos, pos as f32));
        }
        queue.clear();
        assert_eq!(queue.len(), 0);
        queue.push(candidate(42, 0.0));
        assert_eq!((queue.len(), queue.at(0).pos), (1, 42));
    }

    #[test]
    fn equal_costs_keep_the_newer_candidate_first() {
        // The swap is on a strict `>`, so an equal-cost newcomer stays where
        // it was inserted rather than sinking past its equals.
        let mut queue = StartPosQueue::default();
        queue.push(candidate(1, 4.0));
        queue.push(candidate(2, 4.0));
        assert_eq!(queue.at(0).pos, 2);
        assert_eq!(queue.at(1).pos, 1);
    }
}

//! The immutable hash index built over one attached prefix dictionary.
//!
//! Ports `CreatePreparedDictionary` and the bucket walk of
//! `FindCompoundDictionaryMatch` from `c/enc/compound_dictionary.c` and
//! `c/enc/hash.h` of the pinned reference (`google/brotli` v1.2.0, commit
//! `028fb5a`).
//!
//! The layout is the reference's, not a convenience: the bucket count, the
//! hash width and the per-bucket cap decide *which* candidates a search sees
//! and in *what order*, so they are wire-visible through the commands the
//! encoder goes on to emit. What this port does change is ownership — three
//! ordinary boxed slices instead of one flat allocation carved up by pointer
//! arithmetic — and that the build is a pure function of the source bytes, so
//! two contexts prepared from the same dictionary are byte-identical whatever
//! machine, backend or thread built them.

use alloc::boxed::Box;

/// Multiplier the prepared index hashes with (`kPreparedDictionaryHashMul64Long`).
const HASH_MUL64: u64 = 0x1FE3_5A7B_D357_9BD3;

/// Bytes hashed into one bucket key.
pub(crate) const HASH_INPUT_BYTES: usize = 8;

/// Bucket bits the reference starts from, before scaling for a big source.
const INITIAL_BUCKET_BITS: u32 = 17;

/// Slot bits the reference starts from; scaled in step with the buckets.
const INITIAL_SLOT_BITS: u32 = 7;

/// Bucket bits the reference stops scaling at.
const MAX_BUCKET_BITS: u32 = 22;

/// Source bytes each initial bucket is expected to cover.
const BYTES_PER_BUCKET: usize = 16;

/// Low bits of the eight source bytes that reach the multiply.
const HASH_BITS: u32 = 40;

/// Longest chain the index keeps for one bucket.
const BUCKET_LIMIT: u32 = 32;

/// Marks the last item of a bucket chain.
const CHAIN_END: u32 = 0x8000_0000;

/// Empty-bucket marker stored in the head table.
const NO_HEAD: u16 = 0xFFFF;

/// Largest offset a slot may reach before its chains have to be shortened.
const MAX_SLOT_SPAN: u32 = 0xFFFF;

/// A hash index over one attached dictionary.
///
/// Semantically immutable once built. Holds no reference to the bytes it was
/// built from: [`PreparedPrefix::candidates`] yields offsets, and the caller
/// resolves them against the segment it already owns, which is what lets the
/// dictionary bytes live in a separate allocation that may move.
#[derive(Debug, Default)]
pub(crate) struct PreparedPrefix {
    /// Bucket bits, and with them the hash shift.
    bucket_bits: u32,
    /// Slot bits, and with them the slot mask.
    slot_bits: u32,
    /// First item index of each slot.
    slot_offsets: Box<[u32]>,
    /// Offset of each bucket's chain inside its slot, or [`NO_HEAD`].
    heads: Box<[u16]>,
    /// Source offsets, chain by chain, with [`CHAIN_END`] on the last of each.
    items: Box<[u32]>,
}

impl PreparedPrefix {
    /// Builds the index for one dictionary segment.
    ///
    /// A segment shorter than [`HASH_INPUT_BYTES`] has no hashable position at
    /// all, so it yields the empty index rather than a quarter-mebibyte of
    /// empty buckets. Every other size follows the reference exactly, scaling
    /// the bucket and slot counts together until one bucket covers
    /// [`BYTES_PER_BUCKET`] source bytes or [`MAX_BUCKET_BITS`] is reached.
    ///
    /// # Panics
    ///
    /// Never: `source` shorter than [`MAX_PREFIX_SEGMENT_BYTES`] is the
    /// caller's precondition, checked before a context is built, and every
    /// index derived below is bounded by the tables it indexes.
    ///
    /// [`MAX_PREFIX_SEGMENT_BYTES`]: super::prefix::MAX_PREFIX_SEGMENT_BYTES
    pub(crate) fn new(source: &[u8]) -> Self {
        if source.len() < HASH_INPUT_BYTES {
            return Self::default();
        }
        let (bucket_bits, slot_bits) = shape_for(source.len());
        Self::with_shape(source, bucket_bits, slot_bits)
    }

    /// Returns how many source offsets the index holds.
    pub(crate) const fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Returns an upper bound on the *peak* memory [`PreparedPrefix::new`] uses.
    ///
    /// Checked before any table is built, so a context that cannot fit its
    /// allocation limit is refused rather than allocated and thrown away — and
    /// it is the peak rather than the result, because the build holds its
    /// scratch tables and the finished ones at the same time and it is the
    /// peak an attacker would aim at.
    ///
    /// The item table is bounded by one entry per hashable position, which is
    /// the most step 1 can chain; every other term is exact. The result is
    /// therefore also an upper bound on [`PreparedPrefix::allocated_size`].
    pub(crate) fn allocation_bound(source_size: usize) -> Option<u64> {
        let base = size_of::<Self>() as u64;
        if source_size < HASH_INPUT_BYTES {
            return Some(base);
        }
        let (bucket_bits, slot_bits) = shape_for(source_size);
        let positions = source_size as u64 - HASH_INPUT_BYTES as u64 + 1;
        let word = size_of::<u32>() as u64;

        // The finished index.
        let heads = (size_of::<u16>() as u64).checked_shl(bucket_bits)?;
        let slot_offsets = word.checked_shl(slot_bits)?;
        let items = positions.checked_mul(word)?;

        // The scratch step 1 and step 2 hold while step 3 fills the above:
        // one chain length and one chain head per bucket, one link per source
        // byte, and one limit and one cursor per slot.
        let chain_length = word.checked_shl(bucket_bits)?;
        let bucket_head = word.checked_shl(bucket_bits)?;
        let next = (source_size as u64).checked_mul(word)?;
        let slot_scratch = slot_offsets.checked_mul(2)?;

        [
            heads,
            slot_offsets,
            items,
            chain_length,
            bucket_head,
            next,
            slot_scratch,
        ]
        .into_iter()
        .try_fold(base, u64::checked_add)
    }

    /// Returns the bytes this index occupies.
    ///
    /// Reported through `SharedContext::allocated_size`, which is why it counts
    /// the heap behind the three tables rather than the size of the struct.
    pub(crate) const fn allocated_size(&self) -> usize {
        size_of::<Self>()
            + self.slot_offsets.len() * size_of::<u32>()
            + self.heads.len() * size_of::<u16>()
            + self.item_count() * size_of::<u32>()
    }

    /// Returns the bucket key for the eight bytes a candidate would start with.
    ///
    /// `head` is a little-endian load of those bytes, which is the form both
    /// the build loop and the search already have them in.
    pub(crate) const fn key(&self, head: u64) -> usize {
        if self.bucket_bits == 0 {
            return 0;
        }
        hash_key(head, self.bucket_bits)
    }

    /// Returns the source offsets to try for the eight bytes at `head`.
    ///
    /// Newest first, capped at [`BUCKET_LIMIT`] entries and possibly shorter
    /// where a slot had to give way to the sixteen-bit head offsets. The order
    /// is the reference's and does not depend on the backend.
    pub(crate) fn candidates(&self, head: u64) -> Candidates<'_> {
        let key = self.key(head);
        let Some(&head_offset) = self.heads.get(key) else {
            return Candidates { chain: &[] };
        };
        if head_offset == NO_HEAD {
            return Candidates { chain: &[] };
        }
        let slot = key & ((1usize << self.slot_bits) - 1);
        let start =
            self.slot_offsets.get(slot).copied().unwrap_or(0) as usize + head_offset as usize;
        Candidates {
            chain: self.items.get(start..).unwrap_or(&[]),
        }
    }

    /// Builds the index with an explicit shape, as the reference's inner form.
    fn with_shape(source: &[u8], bucket_bits: u32, slot_bits: u32) -> Self {
        let num_buckets = 1usize << bucket_bits;
        let num_slots = 1usize << slot_bits;
        let slot_mask = num_slots - 1;

        // Step 1: a "bloated" hasher — every position chained into its bucket,
        // newest first, with the chain length remembered but capped.
        let mut chain_length = vec![0u32; num_buckets];
        let mut bucket_head = vec![u32::MAX; num_buckets];
        let mut next = vec![u32::MAX; source.len()];
        for (offset, window) in source.windows(HASH_INPUT_BYTES).enumerate() {
            let head = match window.first_chunk::<HASH_INPUT_BYTES>() {
                Some(bytes) => u64::from_le_bytes(*bytes),
                None => continue,
            };
            let key = hash_key(head, bucket_bits);
            let Some(count) = chain_length.get_mut(key) else {
                continue;
            };
            if let (Some(slot), Some(&previous)) = (next.get_mut(offset), bucket_head.get(key)) {
                *slot = if *count == 0 { u32::MAX } else { previous };
            }
            if let Some(entry) = bucket_head.get_mut(key) {
                *entry = offset as u32;
            }
            *count = (*count + 1).min(BUCKET_LIMIT);
        }

        // Step 2: each slot gathers the buckets congruent to it, and shortens
        // their chains until every head offset inside it fits sixteen bits.
        let mut slot_limit = vec![BUCKET_LIMIT; num_slots];
        let mut slot_size = vec![0u32; num_slots];
        let mut total_items = 0usize;
        for slot in 0..num_slots {
            loop {
                let limit = slot_limit.get(slot).copied().unwrap_or(BUCKET_LIMIT);
                let mut count = 0u32;
                let mut overflow = false;
                for bucket in (slot..num_buckets).step_by(num_slots) {
                    if count >= MAX_SLOT_SPAN {
                        overflow = true;
                        break;
                    }
                    count += chain_length.get(bucket).copied().unwrap_or(0).min(limit);
                }
                if !overflow {
                    if let Some(size) = slot_size.get_mut(slot) {
                        *size = count;
                    }
                    total_items += count as usize;
                    break;
                }
                if let Some(entry) = slot_limit.get_mut(slot) {
                    *entry = entry.saturating_sub(1);
                }
            }
        }

        // Step 3: transfer to the "slim" index the search actually walks.
        let mut slot_offsets = vec![0u32; num_slots];
        let mut cursor = 0u32;
        for (offset, size) in slot_offsets.iter_mut().zip(slot_size.iter_mut()) {
            *offset = cursor;
            cursor += *size;
            *size = 0;
        }

        let mut heads = vec![NO_HEAD; num_buckets];
        let mut items = vec![0u32; total_items];
        for (bucket, (head_slot, &chained)) in heads.iter_mut().zip(chain_length.iter()).enumerate()
        {
            let slot = bucket & slot_mask;
            let limit = slot_limit.get(slot).copied().unwrap_or(BUCKET_LIMIT);
            let count = chained.min(limit) as usize;
            if count == 0 {
                continue;
            }
            let filled = slot_size.get(slot).copied().unwrap_or(0) as usize;
            *head_slot = filled as u16;
            let at = slot_offsets.get(slot).copied().unwrap_or(0) as usize + filled;
            if let Some(size) = slot_size.get_mut(slot) {
                *size += count as u32;
            }
            let mut position = bucket_head.get(bucket).copied().unwrap_or(u32::MAX);
            let Some(chain) = items.get_mut(at..at + count) else {
                continue;
            };
            for item in chain.iter_mut() {
                *item = position;
                position = next.get(position as usize).copied().unwrap_or(u32::MAX);
            }
            if let Some(last) = chain.last_mut() {
                *last |= CHAIN_END;
            }
        }

        Self {
            bucket_bits,
            slot_bits,
            slot_offsets: slot_offsets.into_boxed_slice(),
            heads: heads.into_boxed_slice(),
            items: items.into_boxed_slice(),
        }
    }
}

/// Returns the bucket key of eight source bytes under `bucket_bits`.
const fn hash_key(head: u64, bucket_bits: u32) -> usize {
    let hash_mask = u64::MAX >> (64 - HASH_BITS);
    let hash = (head & hash_mask).wrapping_mul(HASH_MUL64);
    (hash >> (64 - bucket_bits)) as usize
}

/// Returns the bucket and slot bits the reference picks for a source size.
///
/// Both grow together so that `bucket_bits - slot_bits` stays at ten, which is
/// what keeps a slot's chains inside the sixteen-bit head offsets step 2 has to
/// fit them into.
const fn shape_for(source_size: usize) -> (u32, u32) {
    let mut bucket_bits = INITIAL_BUCKET_BITS;
    let mut slot_bits = INITIAL_SLOT_BITS;
    let mut volume = BYTES_PER_BUCKET << INITIAL_BUCKET_BITS;
    while volume < source_size && bucket_bits < MAX_BUCKET_BITS {
        bucket_bits += 1;
        slot_bits += 1;
        volume <<= 1;
    }
    (bucket_bits, slot_bits)
}

/// The source offsets of one bucket chain, newest first.
pub(crate) struct Candidates<'a> {
    /// The remaining chain, still carrying its [`CHAIN_END`] marker.
    chain: &'a [u32],
}

impl Iterator for Candidates<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let (&item, rest) = self.chain.split_first()?;
        self.chain = if item & CHAIN_END == 0 { rest } else { &[] };
        Some(item & !CHAIN_END)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn corpus(len: usize) -> Vec<u8> {
        // Deliberately repetitive: identical eight-byte windows are what put
        // more than one item in a bucket.
        (0..len)
            .map(|i| b"the quick brown fox jumps over the lazy dog. "[i % 44])
            .collect()
    }

    /// A corpus with no repeated eight-byte window, so nothing is ever capped.
    fn varied(len: usize) -> Vec<u8> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 33) as u8
            })
            .collect()
    }

    /// Every offset the index will ever hand out, bucket by bucket.
    fn all_chains(index: &PreparedPrefix) -> Vec<Vec<u32>> {
        (0..index.heads.len())
            .map(|bucket| {
                let Some(&head) = index.heads.get(bucket) else {
                    return Vec::new();
                };
                if head == NO_HEAD {
                    return Vec::new();
                }
                let slot = bucket & ((1usize << index.slot_bits) - 1);
                let start = index.slot_offsets[slot] as usize + head as usize;
                Candidates {
                    chain: &index.items[start..],
                }
                .collect()
            })
            .collect()
    }

    #[test]
    fn a_source_too_short_to_hash_has_no_index() {
        for len in 0..HASH_INPUT_BYTES {
            let index = PreparedPrefix::new(&corpus(len));
            assert_eq!(index.item_count(), 0);
            assert_eq!(index.candidates(0).count(), 0);
            assert_eq!(index.key(0x0123_4567_89AB_CDEF), 0);
            assert!(index.allocated_size() > 0);
        }
    }

    #[test]
    fn every_hashable_position_is_indexed_once() {
        let source = varied(4096);
        let index = PreparedPrefix::new(&source);
        let mut seen: Vec<u32> = all_chains(&index).into_iter().flatten().collect();
        seen.sort_unstable();
        seen.dedup();
        // Every position with eight bytes left is reachable: no eight-byte
        // window repeats, so no bucket comes near the per-bucket cap.
        let hashable = source.len() - HASH_INPUT_BYTES + 1;
        assert_eq!(seen.len(), hashable);
        assert_eq!(seen.first().copied(), Some(0));
        assert_eq!(seen.last().copied(), Some(hashable as u32 - 1));
        assert_eq!(index.item_count(), hashable);
    }

    #[test]
    fn a_repeated_window_is_kept_only_up_to_the_bucket_cap() {
        // Forty-four distinct eight-byte windows, each occurring far more often
        // than the cap allows, so the index holds exactly the cap for each.
        let source = corpus(4096);
        let index = PreparedPrefix::new(&source);
        assert_eq!(index.item_count(), 44 * BUCKET_LIMIT as usize);
    }

    #[test]
    fn a_bucket_chain_is_newest_first_and_capped() {
        // The same eight bytes repeated far more often than the cap allows.
        let source = vec![b'z'; 4096];
        let index = PreparedPrefix::new(&source);
        let head = u64::from_le_bytes([b'z'; 8]);
        let chain: Vec<u32> = index.candidates(head).collect();
        assert_eq!(chain.len(), BUCKET_LIMIT as usize);
        let newest = (source.len() - HASH_INPUT_BYTES) as u32;
        assert_eq!(chain.first().copied(), Some(newest));
        assert!(chain.windows(2).all(|pair| pair[0] > pair[1]));
    }

    #[test]
    fn a_candidate_offset_really_starts_the_hashed_bytes() {
        let source = corpus(8192);
        let index = PreparedPrefix::new(&source);
        for start in (0..source.len() - HASH_INPUT_BYTES).step_by(37) {
            let head = u64::from_le_bytes(
                *source[start..]
                    .first_chunk::<HASH_INPUT_BYTES>()
                    .expect("eight bytes"),
            );
            let found = index.candidates(head).any(|offset| {
                source[offset as usize..offset as usize + 8] == source[start..start + 8]
            });
            assert!(found, "no candidate for position {start}");
        }
    }

    #[test]
    fn a_bucket_never_holds_a_candidate_starting_with_other_bytes() {
        let source = corpus(1024);
        let index = PreparedPrefix::new(&source);
        // Eight high bytes: a printable-ASCII corpus contains no such window,
        // so whatever this bucket holds is a hash collision, and a search has
        // to be able to tell that apart from a match by comparing the bytes.
        let mut collisions = 0usize;
        for offset in index.candidates(u64::MAX) {
            let start = offset as usize;
            assert_ne!(
                source[start..start + HASH_INPUT_BYTES],
                [0xFFu8; HASH_INPUT_BYTES],
                "position {start} cannot really start with eight high bytes"
            );
            collisions += 1;
        }
        // Empty or colliding, a bucket never exceeds the cap.
        assert!(collisions <= BUCKET_LIMIT as usize);
    }

    #[test]
    fn the_shape_scales_with_the_source_and_then_stops() {
        assert_eq!(shape_for(0), (17, 7));
        assert_eq!(shape_for(BYTES_PER_BUCKET << 17), (17, 7));
        assert_eq!(shape_for((BYTES_PER_BUCKET << 17) + 1), (18, 8));
        assert_eq!(shape_for(BYTES_PER_BUCKET << 21), (21, 11));
        assert_eq!(shape_for(usize::MAX), (MAX_BUCKET_BITS, 12));
    }

    /// The reference's own prepared index, copied out of its flat allocation.
    struct ReferenceIndex {
        bucket_bits: u32,
        slot_bits: u32,
        slot_offsets: Vec<u32>,
        heads: Vec<u16>,
        items: Vec<u32>,
    }

    /// The reference's own prepared index for the same source bytes.
    ///
    /// Calls `CreatePreparedDictionary` through the workspace shim and copies
    /// its three tables out. `None` when the reference refused to build one.
    fn reference_index(source: &[u8]) -> Option<ReferenceIndex> {
        let capacity = 1usize << 23;
        let mut bucket_bits = 0u32;
        let mut slot_bits = 0u32;
        let mut num_items = 0u32;
        let mut slot_offsets = vec![0u32; capacity];
        let mut heads = vec![0u16; capacity];
        let mut items = vec![0u32; capacity];
        // SAFETY: `source` is readable for its own length, the three shape
        // pointers address live locals, and the three table pointers each
        // address a vector of exactly `capacity` elements of the type the shim
        // writes there.
        let built = unsafe {
            google_brotli_ffi::mbrotli_shim_prepare_dictionary(
                source.as_ptr(),
                source.len(),
                capacity,
                &raw mut bucket_bits,
                &raw mut slot_bits,
                &raw mut num_items,
                slot_offsets.as_mut_ptr(),
                heads.as_mut_ptr(),
                items.as_mut_ptr(),
            )
        };
        if built != google_brotli_ffi::BROTLI_TRUE {
            return None;
        }
        slot_offsets.truncate(1usize << slot_bits);
        heads.truncate(1usize << bucket_bits);
        items.truncate(num_items as usize);
        Some(ReferenceIndex {
            bucket_bits,
            slot_bits,
            slot_offsets,
            heads,
            items,
        })
    }

    #[test]
    fn the_index_is_identical_to_the_c_reference() {
        let cases: [(&str, Vec<u8>); 6] = [
            ("english", corpus(4096)),
            ("varied", varied(4096)),
            ("zeros", vec![0u8; 4096]),
            ("eight-bytes", b"exactly!".to_vec()),
            ("nine-bytes", b"exactly!?".to_vec()),
            (
                "scaling",
                varied((BYTES_PER_BUCKET << INITIAL_BUCKET_BITS) + 1),
            ),
        ];
        for (name, source) in cases {
            let index = PreparedPrefix::new(&source);
            let reference = reference_index(&source).expect("the reference builds an index");
            assert_eq!(
                index.bucket_bits, reference.bucket_bits,
                "{name}: bucket bits"
            );
            assert_eq!(index.slot_bits, reference.slot_bits, "{name}: slot bits");
            assert_eq!(
                index.slot_offsets.as_ref(),
                reference.slot_offsets,
                "{name}: slots"
            );
            assert_eq!(index.heads.as_ref(), reference.heads, "{name}: heads");
            assert_eq!(index.items.as_ref(), reference.items, "{name}: items");
        }
    }

    #[test]
    fn preparation_is_a_pure_function_of_the_bytes() {
        let source = corpus(20_000);
        let first = PreparedPrefix::new(&source);
        let second = PreparedPrefix::new(&source);
        assert_eq!(first.bucket_bits, second.bucket_bits);
        assert_eq!(first.slot_bits, second.slot_bits);
        assert_eq!(first.slot_offsets, second.slot_offsets);
        assert_eq!(first.heads, second.heads);
        assert_eq!(first.items, second.items);
        assert_eq!(first.allocated_size(), second.allocated_size());
    }
}

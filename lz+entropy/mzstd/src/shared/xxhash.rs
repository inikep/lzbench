//! Streaming XXH64, the content checksum of the Zstandard frame format.
//!
//! RFC 8878 section 3.1.1 stores the low 32 bits of the XXH64 digest of the
//! decompressed content, computed with seed 0. Both codecs feed the hasher the
//! content in whatever pieces they have, so it accepts any split.

const PRIME_1: u64 = 0x9E37_79B1_85EB_CA87;
const PRIME_2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const PRIME_3: u64 = 0x1656_67B1_9E37_79F9;
const PRIME_4: u64 = 0x85EB_CA77_C2B2_AE63;
const PRIME_5: u64 = 0x27D4_EB2F_1656_67C5;

/// Incremental XXH64 state.
#[derive(Clone, Debug)]
pub(crate) struct Xxh64 {
    /// The four lane accumulators.
    lanes: [u64; 4],
    /// Bytes not yet folded into a 32-byte stripe.
    pending: [u8; 32],
    /// How many of `pending` are meaningful.
    pending_len: usize,
    /// Total bytes hashed.
    total: u64,
    /// The seed the lanes started from.
    seed: u64,
}

const fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(PRIME_2))
        .rotate_left(31)
        .wrapping_mul(PRIME_1)
}

const fn merge_round(acc: u64, value: u64) -> u64 {
    (acc ^ round(0, value))
        .wrapping_mul(PRIME_1)
        .wrapping_add(PRIME_4)
}

const fn avalanche(mut hash: u64) -> u64 {
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(PRIME_2);
    hash ^= hash >> 29;
    hash = hash.wrapping_mul(PRIME_3);
    hash ^ (hash >> 32)
}

fn read_u64(bytes: &[u8; 8]) -> u64 {
    u64::from_le_bytes(*bytes)
}

impl Xxh64 {
    /// A fresh hasher with `seed`.
    pub(crate) const fn new(seed: u64) -> Self {
        Self {
            lanes: [
                seed.wrapping_add(PRIME_1).wrapping_add(PRIME_2),
                seed.wrapping_add(PRIME_2),
                seed,
                seed.wrapping_sub(PRIME_1),
            ],
            pending: [0; 32],
            pending_len: 0,
            total: 0,
            seed,
        }
    }

    /// Restarts the hash with `seed`.
    pub(crate) fn reset(&mut self, seed: u64) {
        *self = Self::new(seed);
    }

    fn consume_stripe(&mut self, stripe: &[u8; 32]) {
        let (words, _) = stripe.as_chunks::<8>();
        for (lane, word) in self.lanes.iter_mut().zip(words) {
            *lane = round(*lane, read_u64(word));
        }
    }

    /// Hashes `input` after everything hashed so far.
    pub(crate) fn update(&mut self, mut input: &[u8]) {
        self.total = self.total.wrapping_add(input.len() as u64);
        if self.pending_len > 0 {
            let take = (32 - self.pending_len).min(input.len());
            self.pending[self.pending_len..self.pending_len + take].copy_from_slice(&input[..take]);
            self.pending_len += take;
            input = &input[take..];
            if self.pending_len < 32 {
                return;
            }
            let stripe = self.pending;
            self.consume_stripe(&stripe);
            self.pending_len = 0;
        }
        let (stripes, rest) = input.as_chunks::<32>();
        for stripe in stripes {
            self.consume_stripe(stripe);
        }
        self.pending[..rest.len()].copy_from_slice(rest);
        self.pending_len = rest.len();
    }

    /// The digest of everything hashed so far; the state is unchanged.
    pub(crate) fn digest(&self) -> u64 {
        let mut hash = if self.total >= 32 {
            let [v1, v2, v3, v4] = self.lanes;
            let mut hash = v1
                .rotate_left(1)
                .wrapping_add(v2.rotate_left(7))
                .wrapping_add(v3.rotate_left(12))
                .wrapping_add(v4.rotate_left(18));
            for lane in self.lanes {
                hash = merge_round(hash, lane);
            }
            hash
        } else {
            self.seed.wrapping_add(PRIME_5)
        };
        hash = hash.wrapping_add(self.total);

        let mut tail = &self.pending[..self.pending_len];
        while let Some((word, rest)) = tail.split_first_chunk::<8>() {
            hash ^= round(0, read_u64(word));
            hash = hash
                .rotate_left(27)
                .wrapping_mul(PRIME_1)
                .wrapping_add(PRIME_4);
            tail = rest;
        }
        if let Some((word, rest)) = tail.split_first_chunk::<4>() {
            hash ^= u64::from(u32::from_le_bytes(*word)).wrapping_mul(PRIME_1);
            hash = hash
                .rotate_left(23)
                .wrapping_mul(PRIME_2)
                .wrapping_add(PRIME_3);
            tail = rest;
        }
        for &byte in tail {
            hash ^= u64::from(byte).wrapping_mul(PRIME_5);
            hash = hash.rotate_left(11).wrapping_mul(PRIME_1);
        }
        avalanche(hash)
    }
}

#[cfg(test)]
mod tests {
    use super::Xxh64;

    fn one_shot(input: &[u8]) -> u64 {
        let mut hasher = Xxh64::new(0);
        hasher.update(input);
        hasher.digest()
    }

    #[test]
    fn known_vectors_match_the_reference() {
        assert_eq!(one_shot(b""), 0xEF46_DB37_51D8_E999);
        assert_eq!(one_shot(b"a"), 0xD24E_C4F1_A98C_6E5B);
        assert_eq!(one_shot(b"abc"), 0x44BC_2CF5_AD77_0999);
        assert_eq!(
            one_shot(b"Nobody inspects the spammish repetition"),
            0xFBCE_A83C_8A37_8BF1
        );
    }

    #[test]
    fn any_split_gives_the_one_shot_digest() {
        let input: alloc::vec::Vec<u8> = (0..200u32).map(|i| (i * 31 % 251) as u8).collect();
        let expected = one_shot(&input);
        for split in 0..input.len() {
            let mut hasher = Xxh64::new(0);
            hasher.update(&input[..split]);
            hasher.update(&input[split..]);
            assert_eq!(hasher.digest(), expected, "split {split}");
        }
        let mut bytewise = Xxh64::new(0);
        for byte in &input {
            bytewise.update(core::slice::from_ref(byte));
        }
        assert_eq!(bytewise.digest(), expected);
    }

    #[test]
    fn reset_restarts_with_a_seed() {
        let mut hasher = Xxh64::new(0);
        hasher.update(b"stale");
        hasher.reset(0);
        hasher.update(b"abc");
        assert_eq!(hasher.digest(), 0x44BC_2CF5_AD77_0999);
        let seeded = Xxh64::new(1);
        assert_ne!(seeded.digest(), Xxh64::new(0).digest());
    }
}

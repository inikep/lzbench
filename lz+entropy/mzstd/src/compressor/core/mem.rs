//! Index-addressed views of the window's bytes, match counting and the
//! match finders' hash functions (`ZSTD_count`, `ZSTD_hashPtr` and
//! friends).
//!
//! The reference addresses history through two base pointers: indices at or
//! above `dictLimit` live in the current segment (`base + index`), indices
//! between `lowLimit` and `dictLimit` in the previous one
//! (`dictBase + index`). A [`Mem`] is one of those segments as a slice that
//! remembers which index its first byte has, so the ported match finders keep
//! the reference's index arithmetic and every byte they read is bounds
//! checked against the segment it belongs to.

/// One contiguous segment of history, addressed by window index.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Mem<'a> {
    bytes: &'a [u8],
    start: u32,
}

impl<'a> Mem<'a> {
    /// A segment whose first byte has index `start`.
    pub(crate) const fn new(bytes: &'a [u8], start: u32) -> Self {
        Self { bytes, start }
    }

    /// The same bytes, addressed from `start`.
    pub(crate) const fn rebased(self, start: u32) -> Self {
        Self {
            bytes: self.bytes,
            start,
        }
    }

    /// Index of the first byte.
    pub(crate) const fn start(&self) -> u32 {
        self.start
    }

    /// Index one past the last byte.
    pub(crate) const fn end(&self) -> u32 {
        self.start + self.bytes.len() as u32
    }

    #[inline]
    fn offset(&self, index: u32) -> usize {
        (index - self.start) as usize
    }

    /// The byte at `index`.
    #[inline]
    pub(crate) fn byte(&self, index: u32) -> u8 {
        self.bytes[self.offset(index)]
    }

    /// The bytes from `from` to `to`.
    #[inline]
    pub(crate) fn slice(&self, from: u32, to: u32) -> &'a [u8] {
        &self.bytes[self.offset(from)..self.offset(to)]
    }

    #[inline]
    fn array<const N: usize>(&self, index: u32) -> [u8; N] {
        let at = self.offset(index);
        match self.bytes.get(at..at + N) {
            Some(bytes) => {
                let mut out = [0u8; N];
                out.copy_from_slice(bytes);
                out
            }
            None => panic!(
                "read of {N} bytes at index {index} outside {}..{}",
                self.start,
                self.end()
            ),
        }
    }

    /// Little-endian `u32` at `index`.
    #[inline]
    pub(crate) fn read32(&self, index: u32) -> u32 {
        u32::from_le_bytes(self.array(index))
    }

    /// Little-endian `u64` at `index`.
    #[inline]
    pub(crate) fn read64(&self, index: u32) -> u64 {
        u64::from_le_bytes(self.array(index))
    }

    /// Length of the common run of `self[ip..limit]` and `other[m..]`
    /// (`ZSTD_count`); `other` must hold at least as many bytes from `m`.
    #[inline]
    pub(crate) fn count_with(&self, ip: u32, other: &Mem<'_>, m: u32, limit: u32) -> usize {
        if ip >= limit {
            return 0;
        }
        let a = &self.bytes[self.offset(ip)..self.offset(limit)];
        let b = &other.bytes[other.offset(m)..];
        count_slices(a, b)
    }

    /// `ZSTD_count` within this segment.
    #[inline]
    pub(crate) fn count(&self, ip: u32, m: u32, limit: u32) -> usize {
        self.count_with(ip, self, m, limit)
    }
}

/// Length of the common prefix of `a` and `b` (`ZSTD_count` on slices).
///
/// The first word settles most candidates with one load per side, an XOR and
/// a trailing-zero count, as the reference does. Longer runs compare whole
/// words over fixed-size chunks, which carry no bounds checks of their own,
/// and finish with one word that ends at the limit: it overlaps bytes already
/// found equal, so its first difference is the first difference of the
/// tail. Only a run shorter than a word is compared byte by byte.
#[inline]
pub(crate) fn count_slices(a: &[u8], b: &[u8]) -> usize {
    count_slices_with(a, b, count_long)
}

/// The same exact count in the match finder's already-selected SIMD context.
/// The first word stays inline; only longer runs enter the feature-enabled scan.
#[inline]
pub(crate) fn count_slices_simd<S: fearless_simd::Simd>(simd: S, a: &[u8], b: &[u8]) -> usize {
    count_slices_with(a, b, |a, b| count_long_simd(simd, a, b))
}

#[inline(always)]
fn count_slices_with(a: &[u8], b: &[u8], long: impl FnOnce(&[u8], &[u8]) -> usize) -> usize {
    let limit = a.len().min(b.len());
    let (a, b) = (&a[..limit], &b[..limit]);
    let (Some(first_a), Some(first_b)) = (a.first_chunk::<8>(), b.first_chunk::<8>()) else {
        return a.iter().zip(b).take_while(|(x, y)| x == y).count();
    };
    let difference = u64::from_le_bytes(*first_a) ^ u64::from_le_bytes(*first_b);
    if difference != 0 {
        return difference.trailing_zeros() as usize >> 3;
    }
    long(a, b)
}

/// `ZSTD_count` for two runs of one buffer: the common prefix of
/// `src[ip..]` and `src[m..]`, as [`count_slices_simd`] counts it.
///
/// The match finders call it once per match, and most matches end within
/// their first word: that word is compared after a bounds check on each
/// side, before the slices the full count needs are built.
#[inline(always)]
pub(crate) fn count_at_simd<S: fearless_simd::Simd>(
    simd: S,
    src: &[u8],
    ip: usize,
    m: usize,
) -> usize {
    let (ip, m) = (ip & u32::MAX as usize, m & u32::MAX as usize);
    if let (Some(a), Some(b)) = (src.get(ip..ip + 8), src.get(m..m + 8))
        && let (Some(a), Some(b)) = (a.first_chunk::<8>(), b.first_chunk::<8>())
    {
        let difference = u64::from_le_bytes(*a) ^ u64::from_le_bytes(*b);
        if difference != 0 {
            return difference.trailing_zeros() as usize >> 3;
        }
    }
    count_slices_simd(simd, &src[ip..], &src[m..])
}

/// [`count_at_simd`] for a match `m` behind `ip`: the first eight bytes at
/// `m` lie inside `src` whenever those at `ip` do, so one test covers both
/// loads.
#[inline(always)]
pub(crate) fn count_behind_simd<S: fearless_simd::Simd>(
    simd: S,
    src: &[u8],
    ip: usize,
    m: usize,
) -> usize {
    let (ip, m) = (ip & u32::MAX as usize, m & u32::MAX as usize);
    if m < ip
        && let Some(a) = src.get(ip..ip + 8).and_then(<[u8]>::first_chunk::<8>)
        && let Some(b) = src.get(m..m + 8).and_then(<[u8]>::first_chunk::<8>)
    {
        let difference = u64::from_le_bytes(*a) ^ u64::from_le_bytes(*b);
        if difference != 0 {
            return difference.trailing_zeros() as usize >> 3;
        }
    }
    count_slices_simd(simd, &src[ip..], &src[m..])
}

/// Baseline callers keep their compact 32-byte loop. Keeping the scan out
/// of line avoids hoisting its setup into the first-word candidate check.
#[inline(never)]
fn count_long(a: &[u8], b: &[u8]) -> usize {
    count_long_chunks::<32>(a, b)
}

/// Inline the dispatch wrapper, while its feature-enabled entry stays out
/// of line. Auto-vectorization of the 128-byte XOR/OR reduction uses four
/// AVX2 registers and one combined test, without retaining per-vector masks.
#[fearless_simd_macros::simd]
#[inline(always)]
fn count_long_simd<S: fearless_simd::Simd>(simd: S, a: &[u8], b: &[u8]) -> usize {
    count_long_chunks::<128>(a, b)
}

/// Past an equal first word: words up to 32 bytes, large chunks, 32-byte
/// chunks, words, then an overlapping last word. The intermediate loop
/// handles both a mismatching large chunk and a tail shorter than one.
#[inline(always)]
fn count_long_chunks<const CHUNK: usize>(a: &[u8], b: &[u8]) -> usize {
    let word = |at: usize| -> Option<u64> {
        let x = a.get(at..at + 8)?;
        let y = b.get(at..at + 8)?;
        Some(u64::from_le_bytes(x.try_into().ok()?) ^ u64::from_le_bytes(y.try_into().ok()?))
    };
    let mut matched = 8usize;
    while matched < 32 {
        match word(matched) {
            Some(0) => matched += 8,
            Some(difference) => return matched + (difference.trailing_zeros() as usize >> 3),
            None => return count_tail(a, b, matched),
        }
    }
    if CHUNK > 32 {
        while let (Some(x), Some(y)) = (
            a.get(matched..matched + CHUNK)
                .and_then(|x| <&[u8; CHUNK]>::try_from(x).ok()),
            b.get(matched..matched + CHUNK)
                .and_then(|y| <&[u8; CHUNK]>::try_from(y).ok()),
        ) {
            if x.iter().zip(y).fold(0u8, |acc, (p, q)| acc | (p ^ q)) != 0 {
                break;
            }
            matched += CHUNK;
        }
    }
    while let (Some(x), Some(y)) = (
        a.get(matched..matched + 32)
            .and_then(|x| <&[u8; 32]>::try_from(x).ok()),
        b.get(matched..matched + 32)
            .and_then(|y| <&[u8; 32]>::try_from(y).ok()),
    ) {
        if x.iter().zip(y).fold(0u8, |acc, (p, q)| acc | (p ^ q)) != 0 {
            break;
        }
        matched += 32;
    }
    loop {
        match word(matched) {
            Some(0) => matched += 8,
            Some(difference) => return matched + (difference.trailing_zeros() as usize >> 3),
            None => return count_tail(a, b, matched),
        }
    }
}

/// The end of [`count_long`] once fewer than eight bytes remain after
/// `matched` equal ones: one word that ends at the limit overlaps bytes
/// already found equal, so its first difference is the tail's.
fn count_tail(a: &[u8], b: &[u8], matched: usize) -> usize {
    let limit = a.len();
    if matched >= limit {
        return limit;
    }
    match (a.last_chunk::<8>(), b.last_chunk::<8>()) {
        (Some(last_a), Some(last_b)) => {
            let difference = u64::from_le_bytes(*last_a) ^ u64::from_le_bytes(*last_b);
            if difference == 0 {
                limit
            } else {
                limit - 8 + (difference.trailing_zeros() as usize >> 3)
            }
        }
        _ => limit,
    }
}

/// The two segments of history visible to one block.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Window<'a> {
    /// Indices `[dictLimit, block end)`.
    pub(crate) prefix: Mem<'a>,
    /// Indices `[lowLimit, dictLimit)`; empty without an external dictionary.
    pub(crate) ext: Mem<'a>,
}

impl Window<'_> {
    /// The byte at window index `index`, from whichever segment holds it.
    #[inline]
    pub(crate) fn byte_at(&self, index: u32) -> u8 {
        if index < self.prefix.start() {
            self.ext.byte(index)
        } else {
            self.prefix.byte(index)
        }
    }

    /// The `u32` at window index `index`; it must not straddle the
    /// boundary between segments.
    #[inline]
    pub(crate) fn read32_at(&self, index: u32) -> u32 {
        if index < self.prefix.start() {
            self.ext.read32(index)
        } else {
            self.prefix.read32(index)
        }
    }

    /// Common length of the runs at window indices `a` and `b`, stopping
    /// before `limit` for `a`. Indices below the prefix read the external
    /// dictionary, and a run leaving it continues at the prefix start —
    /// the reference's `ZSTD_count` / `ZSTD_count_2segments` pair seen as
    /// one contiguous index space.
    pub(crate) fn count_virtual(&self, a: u32, b: u32, limit: u32) -> usize {
        let boundary = self.prefix.start();
        if a >= boundary && b >= boundary {
            return self.prefix.count(a, b, limit);
        }
        if a >= boundary {
            return self.count_2segments(a, b, limit, boundary, boundary);
        }
        let mut n = 0u32;
        while a + n < limit && self.byte_at(a + n) == self.byte_at(b + n) {
            n += 1;
        }
        n as usize
    }

    /// `ZSTD_count_2segments`: a match starting in `ext` at `m` continues
    /// into the prefix at `prefix_start` once it reaches `m_end`.
    pub(crate) fn count_2segments(
        &self,
        ip: u32,
        m: u32,
        i_end: u32,
        m_end: u32,
        prefix_start: u32,
    ) -> usize {
        let v_end = (ip + (m_end - m)).min(i_end);
        let match_length = self.prefix.count_with(ip, &self.ext, m, v_end);
        if m + match_length as u32 != m_end {
            return match_length;
        }
        match_length
            + self
                .prefix
                .count(ip + match_length as u32, prefix_start, i_end)
    }
}

const PRIME_3_BYTES: u32 = 506_832_829;
const PRIME_4_BYTES: u32 = 2_654_435_761;
const PRIME_5_BYTES: u64 = 889_523_592_379;
const PRIME_6_BYTES: u64 = 227_718_039_650_203;
const PRIME_7_BYTES: u64 = 58_295_818_150_454_627;
/// `prime8bytes`, also the rolling hash multiplier.
pub(crate) const PRIME_8_BYTES: u64 = 0xCF1B_BCDC_B7A5_6463;

/// `ZSTD_hash3`.
#[inline]
pub(crate) const fn hash3(u: u32, h: u32, s: u32) -> usize {
    ((((u << 8).wrapping_mul(PRIME_3_BYTES)) ^ s) >> (32 - h)) as usize
}

#[inline]
const fn hash4(u: u32, h: u32, s: u32) -> usize {
    ((u.wrapping_mul(PRIME_4_BYTES) ^ s) >> (32 - h)) as usize
}

#[inline]
const fn hash5(u: u64, h: u32, s: u64) -> usize {
    (((u << 24).wrapping_mul(PRIME_5_BYTES) ^ s) >> (64 - h)) as usize
}

#[inline]
const fn hash6(u: u64, h: u32, s: u64) -> usize {
    (((u << 16).wrapping_mul(PRIME_6_BYTES) ^ s) >> (64 - h)) as usize
}

#[inline]
const fn hash7(u: u64, h: u32, s: u64) -> usize {
    (((u << 8).wrapping_mul(PRIME_7_BYTES) ^ s) >> (64 - h)) as usize
}

#[inline]
const fn hash8(u: u64, h: u32, s: u64) -> usize {
    ((u.wrapping_mul(PRIME_8_BYTES) ^ s) >> (64 - h)) as usize
}

/// Little-endian `u32` at byte offset `at` of `src` (`MEM_readLE32`).
///
/// Offsets are below 2^32 (they are window indices minus a base), so `at`
/// is masked to that width: it changes nothing for a real offset and lets
/// the compiler see that `at + 4` cannot overflow, which leaves a single
/// compare against the length in front of the load.
///
/// # Panics
///
/// When the four bytes are not inside `src`.
#[inline(always)]
pub(crate) fn load32(src: &[u8], at: usize) -> u32 {
    let at = at & u32::MAX as usize;
    let mut word = [0u8; 4];
    word.copy_from_slice(&src[at..at + 4]);
    u32::from_le_bytes(word)
}

/// Little-endian `u64` at byte offset `at` of `src` (`MEM_readLE64`); see
/// [`load32`].
///
/// # Panics
///
/// When the eight bytes are not inside `src`.
#[inline(always)]
pub(crate) fn load64(src: &[u8], at: usize) -> u64 {
    let at = at & u32::MAX as usize;
    let mut word = [0u8; 8];
    word.copy_from_slice(&src[at..at + 8]);
    u64::from_le_bytes(word)
}

/// Words that stand in for a candidate outside the history, so its bytes
/// are loaded and compared without first branching on whether it is valid
/// (the reference's `ZSTD_match4Found_cmov` reads its `dummy` the same
/// way). The first byte of `DUMMIES[b]` has its low bit clear exactly when
/// `b` is 1, so the one picked by a word's low bit differs from that word.
const DUMMIES: [[u8; 8]; 2] = [
    [0x13, 0x35, 0x57, 0x79, 0x9b, 0xbd, 0xdf, 0xf1],
    [0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0],
];

/// The `N` bytes at offset `at` of `src`, or bytes that differ from `word`
/// when they are not all inside `src`, so comparing them with `word` tells
/// whether the candidate matches.
///
/// The choice is a select rather than a branch: a stale candidate is as
/// likely as a live one, so a branch on it would be mispredicted often.
#[inline(always)]
fn candidate<'a, const N: usize>(src: &'a [u8], at: u32, word: &[u8; N]) -> &'a [u8; N] {
    let at = at as usize;
    let candidate = at
        .checked_add(N)
        .and_then(|end| src.get(at..end))
        .and_then(|bytes| bytes.first_chunk::<N>());
    let low_bit = word.first().map_or(0, |byte| usize::from(byte & 1));
    let Some(dummy) = DUMMIES[low_bit].first_chunk::<N>() else {
        unreachable!("candidates are at most eight bytes");
    };
    core::hint::select_unpredictable(candidate.is_some(), candidate.unwrap_or(dummy), dummy)
}

/// Compares a four-byte candidate if the whole word is in `src`.
/// Match finders pass the searchable history and an index with its lower
/// bound subtracted using wrapping arithmetic. An older index becomes a
/// large offset, so the same range check rejects stale entries and guards
/// the word load. No bytes outside the history are read.
#[inline(always)]
pub(crate) fn matches32(src: &[u8], at: u32, word: u32) -> bool {
    let word = word.to_le_bytes();
    *candidate(src, at, &word) == word
}

/// [`matches32`] with branches: whether the four bytes at offset `at` lie
/// in `src`, then whether they equal `word`. On large inputs a candidate is
/// rarely outside the history and rarely matches, so both branches predict
/// well, and the comparison needs no stand-in word.
#[inline(always)]
pub(crate) fn matches32_branchy(src: &[u8], at: u32, word: u32) -> bool {
    let at = at as usize;
    match src.get(at..at + 4) {
        Some(bytes) => u32::from_le_bytes(bytes.try_into().unwrap_or_default()) == word,
        None => false,
    }
}

/// Eight-byte counterpart of [`matches32`].
#[inline(always)]
pub(crate) fn matches64(src: &[u8], at: u32, word: u64) -> bool {
    let word = word.to_le_bytes();
    *candidate(src, at, &word) == word
}

/// Hints that `src[at]` will be read soon (`PREFETCH_L1`). The pointer only
/// reaches the prefetch instruction, so an offset past the end is harmless.
#[inline(always)]
pub(crate) fn prefetch_l1(src: &[u8], at: usize) {
    branches::prefetch_read_data::<u8, 0>(src.as_ptr().wrapping_add(at));
}

/// The little-endian `u64` at offset `at` of `src`, or `None` when the
/// eight bytes are not all inside it.
///
/// The `dfast` search goes on while its next position is at most `ilimit`,
/// eight bytes before the end of `src`: that is while this load succeeds,
/// so one range test is the loop's exit test and the load's bounds check.
#[inline(always)]
pub(crate) fn try_load64(src: &[u8], at: usize) -> Option<u64> {
    let at = at & u32::MAX as usize;
    match src.get(at..at + 8) {
        Some(bytes) => bytes
            .first_chunk::<8>()
            .map(|word| u64::from_le_bytes(*word)),
        None => None,
    }
}

/// The nine bytes at offset `at` of a position the `fast` search may still
/// probe, or `None` once it may not: the position and the one after it,
/// both readable as whole words without further bounds tests
/// ([`window_word`]).
///
/// The search continues while `at + 1 < ilimit`, with `ilimit` eight bytes
/// before the end of `src`: that is while the ten bytes at `at` are inside
/// `src`. Testing that range is the loop's exit test and the window's bounds
/// check at once.
#[inline(always)]
pub(crate) fn probe_window(src: &[u8], at: usize) -> Option<&[u8; 9]> {
    let at = at & u32::MAX as usize;
    match src.get(at..at + 10) {
        Some(bytes) => bytes.first_chunk::<9>(),
        None => None,
    }
}

/// The [`probe_window`]s at offsets `at` and `at + gap` of `src`, or `None`
/// when the second is refused: both windows lie in one range, so a single
/// bounds check covers them.
#[inline(always)]
pub(crate) fn probe_pair(src: &[u8], at: usize, gap: usize) -> Option<(&[u8; 9], &[u8; 9])> {
    let (at, gap) = (at & u32::MAX as usize, gap & u32::MAX as usize);
    let pair = src.get(at..at + gap + 10)?;
    match (
        pair.first_chunk::<9>(),
        pair.get(gap..).and_then(|second| second.first_chunk::<9>()),
    ) {
        (Some(first), Some(second)) => Some((first, second)),
        _ => None,
    }
}

/// The little-endian word at `OFF` (0 or 1) of a [`probe_window`].
#[inline(always)]
pub(crate) fn window_word<const OFF: usize>(window: &[u8; 9]) -> u64 {
    let mut word = [0u8; 8];
    word.copy_from_slice(&window[OFF..OFF + 8]);
    u64::from_le_bytes(word)
}

/// `ZSTD_hashPtr` over a little-endian word already loaded from the
/// position: the match finders load eight bytes once per position and
/// derive every hash and compare at it from them. `MLS` outside `5..=8`
/// hashes the low four bytes.
#[inline(always)]
pub(crate) const fn hash_word<const MLS: u32>(word: u64, h_bits: u32) -> usize {
    match MLS {
        5 => hash5(word, h_bits, 0),
        6 => hash6(word, h_bits, 0),
        7 => hash7(word, h_bits, 0),
        8 => hash8(word, h_bits, 0),
        _ => hash4(word as u32, h_bits, 0),
    }
}

/// `ZSTD_hashPtr` over the bytes at offset `at` of `src`, with the match
/// length fixed at compile time as the reference's templated match finders
/// call it; `MLS` outside `5..=8` hashes four bytes.
#[inline(always)]
pub(crate) fn hash_at<const MLS: u32>(src: &[u8], at: usize, h_bits: u32) -> usize {
    match MLS {
        5 => hash5(load64(src, at), h_bits, 0),
        6 => hash6(load64(src, at), h_bits, 0),
        7 => hash7(load64(src, at), h_bits, 0),
        8 => hash8(load64(src, at), h_bits, 0),
        _ => hash4(load32(src, at), h_bits, 0),
    }
}

/// `ZSTD_hashPtrSalted` over the bytes at `index` of `mem`.
#[inline]
pub(crate) fn hash_salted(mem: &Mem<'_>, index: u32, h_bits: u32, mls: u32, salt: u64) -> usize {
    match mls {
        5 => hash5(mem.read64(index), h_bits, salt),
        6 => hash6(mem.read64(index), h_bits, salt),
        7 => hash7(mem.read64(index), h_bits, salt),
        8 => hash8(mem.read64(index), h_bits, salt),
        _ => hash4(mem.read32(index), h_bits, salt as u32),
    }
}

/// `ZSTD_hashPtr` over the bytes at `index` of `mem`.
#[inline]
pub(crate) fn hash_ptr(mem: &Mem<'_>, index: u32, h_bits: u32, mls: u32) -> usize {
    hash_salted(mem, index, h_bits, mls, 0)
}

/// `ZSTD_hash3Ptr` over the bytes at `index` of `mem`; the optimal parser
/// hashes through its own slice view, so only the tests use this one.
#[cfg(test)]
pub(crate) fn hash3_ptr(mem: &Mem<'_>, index: u32, h_bits: u32) -> usize {
    hash3(mem.read32(index), h_bits, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_words_reject_stale_indices_and_partial_tails() {
        let bytes: alloc::vec::Vec<_> = (0..32u8).collect();
        for start in 0..8 {
            for end in start..=bytes.len() {
                let history = &bytes[start..end];
                for index in 0..=40u32 {
                    let at = index.wrapping_sub(start as u32);
                    let word32 = [
                        index as u8,
                        (index + 1) as u8,
                        (index + 2) as u8,
                        (index + 3) as u8,
                    ];
                    let word64 = core::array::from_fn(|i| (index + i as u32) as u8);
                    assert_eq!(
                        matches32(history, at, u32::from_le_bytes(word32)),
                        index as usize >= start && index as usize + 4 <= end
                    );
                    assert_eq!(
                        matches64(history, at, u64::from_le_bytes(word64)),
                        index as usize >= start && index as usize + 8 <= end
                    );
                    assert!(!matches32(history, at, u32::MAX));
                    assert!(!matches64(history, at, u64::MAX));
                }
                for at in u32::MAX - 8..=u32::MAX {
                    assert!(!matches32(history, at, 0));
                    assert!(!matches64(history, at, 0));
                }
            }
        }
    }

    #[test]
    fn reads_are_little_endian_and_index_addressed() {
        let bytes = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let mem = Mem::new(&bytes, 100);
        assert_eq!(mem.byte(100), 1);
        assert_eq!(mem.read32(100), 0x0403_0201);
        assert_eq!(mem.read64(101), 0x0908_0706_0504_0302);
        assert_eq!(mem.slice(102, 104), &[3, 4]);
        assert_eq!((mem.start(), mem.end()), (100, 109));
    }

    #[test]
    #[should_panic(expected = "outside")]
    fn reading_past_a_segment_panics() {
        let bytes = [0u8; 4];
        Mem::new(&bytes, 0).read64(0);
    }

    #[test]
    fn counts_stop_at_the_first_difference_or_limit() {
        let bytes = b"abcabcabd";
        let mem = Mem::new(bytes, 0);
        assert_eq!(mem.count(3, 0, 9), 5);
        assert_eq!(mem.count(3, 0, 5), 2);
        assert_eq!(mem.count(9, 0, 9), 0);
    }

    #[test]
    fn two_segment_counts_continue_into_the_prefix() {
        let ext = b"xyab";
        let prefix = b"abab_";
        let window = Window {
            prefix: Mem::new(prefix, 10),
            ext: Mem::new(ext, 6),
        };
        // "ab" at ext index 8, then the prefix from index 10 continues "ab".
        assert_eq!(window.count_2segments(12, 8, 15, 10, 10), 2);
        assert_eq!(window.count_2segments(10, 8, 15, 10, 10), 4);
    }

    #[test]
    fn hashes_use_the_requested_width() {
        let bytes = [7u8; 16];
        let mem = Mem::new(&bytes, 0);
        for mls in 3..=8 {
            assert!(hash_ptr(&mem, 0, 12, mls) < 1 << 12);
        }
        assert!(hash3_ptr(&mem, 0, 10) < 1 << 10);
        assert_ne!(
            hash_salted(&mem, 0, 16, 5, 1 << 63),
            hash_salted(&mem, 0, 16, 5, 0)
        );
    }

    /// The byte-at-a-time count the word scan must reproduce.
    fn count_bytewise(a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b).take_while(|(x, y)| x == y).count()
    }

    #[test]
    fn word_counts_match_a_bytewise_scan_at_every_length_and_mismatch() {
        let base: alloc::vec::Vec<u8> = (0..200u8).map(|i| i.wrapping_mul(37)).collect();
        for len in 0..=base.len() {
            let a = &base[..len];
            // Equal throughout, and a difference at every position.
            assert_eq!(count_slices(a, &base), len);
            for at in 0..len {
                let mut b = base.clone();
                b[at] ^= 0x40;
                assert_eq!(
                    count_slices(a, &b),
                    count_bytewise(a, &b),
                    "len {len} at {at}"
                );
            }
            // The shorter side bounds the count.
            assert_eq!(count_slices(&base, a), len);
        }
    }

    #[test]
    fn simd_counts_match_bytewise_at_unaligned_chunk_boundaries() {
        use crate::Backend;

        let lengths: alloc::vec::Vec<usize> = (0..=65)
            .chain([127, 128, 129, 159, 160, 161, 255, 256, 257, 511, 512])
            .collect();
        let base: alloc::vec::Vec<u8> = (0..576).map(|i| (i * 37) as u8).collect();
        for backend in Backend::available() {
            for (left_offset, right_offset) in [(0, 1), (1, 0), (3, 7), (15, 31), (31, 63)] {
                for &len in &lengths {
                    let left = &base[left_offset..left_offset + len];
                    let mut storage = alloc::vec![0u8; right_offset + len + 1];
                    storage[right_offset..right_offset + len].copy_from_slice(left);
                    for diff in 0..=len {
                        if diff < len {
                            storage[right_offset + diff] ^= 0x80;
                        }
                        let right = &storage[right_offset..];
                        let expected = count_bytewise(left, right);
                        let got = fearless_simd::dispatch!(backend.0, simd => count_slices_simd(simd, left, right));
                        assert_eq!(got, expected, "{backend} len {len} diff {diff}");
                        let reversed = fearless_simd::dispatch!(backend.0, simd => count_slices_simd(simd, right, left));
                        assert_eq!(
                            reversed, expected,
                            "{backend} reversed len {len} diff {diff}"
                        );
                        if diff < len {
                            storage[right_offset + diff] ^= 0x80;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn offset_loads_are_little_endian() {
        let bytes = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(load32(&bytes, 1), 0x0504_0302);
        assert_eq!(load64(&bytes, 1), 0x0908_0706_0504_0302);
    }

    #[test]
    fn probe_windows_match_offset_loads_while_ten_bytes_remain() {
        let bytes: alloc::vec::Vec<u8> = (0..24u8).map(|i| i.wrapping_mul(53) ^ 0x3C).collect();
        for at in 0..=bytes.len() + 2 {
            let window = probe_window(&bytes, at);
            assert_eq!(window.is_some(), at + 10 <= bytes.len(), "at {at}");
            if let Some(window) = window {
                assert_eq!(window_word::<0>(window), load64(&bytes, at));
                assert_eq!(window_word::<1>(window), load64(&bytes, at + 1));
            }
            for gap in 0..=bytes.len() {
                let pair = probe_pair(&bytes, at, gap);
                assert_eq!(
                    pair.is_some(),
                    at + gap + 10 <= bytes.len(),
                    "at {at} gap {gap}"
                );
                if let Some((first, second)) = pair {
                    assert_eq!(window_word::<0>(first), load64(&bytes, at));
                    assert_eq!(window_word::<1>(first), load64(&bytes, at + 1));
                    assert_eq!(window_word::<0>(second), load64(&bytes, at + gap));
                    assert_eq!(window_word::<1>(second), load64(&bytes, at + gap + 1));
                }
            }
        }
        assert!(probe_window(&bytes, usize::MAX).is_none());
        assert!(probe_pair(&bytes, usize::MAX, usize::MAX).is_none());
    }

    #[test]
    fn checked_loads_stop_where_eight_bytes_no_longer_fit() {
        let bytes: alloc::vec::Vec<u8> = (0..20u8).map(|i| i.wrapping_mul(29) ^ 0xA5).collect();
        for at in 0..=bytes.len() + 2 {
            let word = try_load64(&bytes, at);
            assert_eq!(word.is_some(), at + 8 <= bytes.len(), "at {at}");
            if let Some(word) = word {
                assert_eq!(word, load64(&bytes, at));
            }
        }
        assert!(try_load64(&bytes, usize::MAX).is_none());
    }

    #[test]
    fn dummy_words_never_match_a_stale_candidate() {
        // A word equal to either stand-in, probed at an index outside the
        // history, must not match: the stand-in picked differs from it.
        let history = [0u8; 16];
        for dummy in DUMMIES {
            let word64 = u64::from_le_bytes(dummy);
            let word32 = u32::from_le_bytes([dummy[0], dummy[1], dummy[2], dummy[3]]);
            for at in [13, 16, 1000, u32::MAX - 3, u32::MAX] {
                assert!(!matches32(&history, at, word32), "at {at}");
                assert!(!matches64(&history, at, word64), "at {at}");
            }
        }
        // The same words in the history match where they are.
        let mut history = [0u8; 24];
        history[3..11].copy_from_slice(&DUMMIES[1]);
        history[11..19].copy_from_slice(&DUMMIES[0]);
        assert!(matches64(&history, 3, u64::from_le_bytes(DUMMIES[1])));
        assert!(matches64(&history, 11, u64::from_le_bytes(DUMMIES[0])));
        assert!(matches32(
            &history,
            11,
            u32::from_le_bytes([0x13, 0x35, 0x57, 0x79])
        ));
    }

    #[test]
    fn first_word_counts_match_the_slice_count_on_every_backend() {
        use crate::Backend;

        let mut bytes: alloc::vec::Vec<u8> = (0..96u8).map(|i| i % 7).collect();
        bytes.extend((0..64u8).map(|i| i.wrapping_mul(37)));
        for backend in Backend::available() {
            for ip in 0..=bytes.len() {
                for m in 0..=ip {
                    let expected = count_slices(&bytes[ip..], &bytes[m..]);
                    let got = fearless_simd::dispatch!(backend.0, simd => count_at_simd(simd, &bytes, ip, m));
                    assert_eq!(got, expected, "{backend} ip {ip} m {m}");
                }
            }
        }
    }

    #[test]
    fn counts_behind_equal_counts_at_every_boundary() {
        let backend = crate::Backend::available();
        let mut src: alloc::vec::Vec<u8> = (0..300u32).map(|i| (i % 7) as u8).collect();
        src[250] ^= 0xFF;
        for backend in backend {
            for (ip, m) in [
                (7, 0),
                (14, 7),
                (21, 0),
                (250, 243),
                (292, 285),
                (299, 292),
                (5, 5),
            ] {
                let behind = fearless_simd::dispatch!(backend.0, simd => count_behind_simd(simd, &src, ip, m));
                let at =
                    fearless_simd::dispatch!(backend.0, simd => count_at_simd(simd, &src, ip, m));
                assert_eq!(behind, at, "{backend} {ip} {m}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn offset_loads_past_the_end_panic() {
        load64(&[0u8; 8], 1);
    }

    #[test]
    fn word_and_offset_hashes_match_the_index_hash() {
        let bytes: alloc::vec::Vec<u8> = (0..32u8).map(|i| i.wrapping_mul(91) ^ 0x5A).collect();
        let mem = Mem::new(&bytes, 100);
        for at in 0..16usize {
            let index = 100 + at as u32;
            let word = load64(&bytes, at);
            assert_eq!(hash_word::<4>(word, 14), hash_ptr(&mem, index, 14, 4));
            assert_eq!(hash_word::<5>(word, 14), hash_ptr(&mem, index, 14, 5));
            assert_eq!(hash_word::<6>(word, 14), hash_ptr(&mem, index, 14, 6));
            assert_eq!(hash_word::<7>(word, 14), hash_ptr(&mem, index, 14, 7));
            assert_eq!(hash_word::<8>(word, 14), hash_ptr(&mem, index, 14, 8));
            assert_eq!(hash_at::<4>(&bytes, at, 13), hash_ptr(&mem, index, 13, 4));
            assert_eq!(hash_at::<5>(&bytes, at, 13), hash_ptr(&mem, index, 13, 5));
            assert_eq!(hash_at::<6>(&bytes, at, 13), hash_ptr(&mem, index, 13, 6));
            assert_eq!(hash_at::<7>(&bytes, at, 13), hash_ptr(&mem, index, 13, 7));
            assert_eq!(hash_at::<8>(&bytes, at, 13), hash_ptr(&mem, index, 13, 8));
        }
    }
}

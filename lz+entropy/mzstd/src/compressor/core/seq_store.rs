//! The sequence store: what a match finder produces for one block
//! (`SeqStore_t`, `ZSTD_storeSeq` and the repeat-offset helpers).
//!
//! Offsets travel as the reference's "offBase" sum type: `1..=3` name a
//! repeat offset, anything larger is a literal offset plus 3.

use fearless_simd::Simd;
use fearless_simd_macros::simd;

use crate::Backend;
use crate::shared::format::{LL_BITS, MIN_MATCH, ML_BITS, REP_NUM, highbit32};

/// Largest literal length code.
const MAX_LL: u8 = 35;
/// Largest match length code.
const MAX_ML: u8 = 52;

/// One sequence (`SeqDef`).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SeqDef {
    /// Offset sum type: repeat offset `1..=3`, or offset plus 3.
    pub(crate) off_base: u32,
    /// Literal length, modulo `0x10000` (see [`LongLength`]).
    pub(crate) lit_length: u16,
    /// Match length minus [`MIN_MATCH`], modulo `0x10000`.
    pub(crate) ml_base: u16,
}

/// Which length, if any, of one sequence exceeds 16 bits
/// (`ZSTD_longLengthType_e`).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) enum LongLength {
    #[default]
    None,
    /// The literal length at `pos` carries an extra `0x10000`.
    Literal(u32),
    /// The match length at `pos` carries an extra `0x10000`.
    Match(u32),
}

/// Records a block's exceptional long length, in the reference's order.
#[cold]
#[inline(never)]
fn mark_long_length(long_length: &mut LongLength, pos: u32, lit_length: usize, ml_base: usize) {
    if lit_length > 0xffff {
        debug_assert_eq!(*long_length, LongLength::None);
        *long_length = LongLength::Literal(pos);
    }
    if ml_base > 0xffff {
        debug_assert_eq!(*long_length, LongLength::None);
        *long_length = LongLength::Match(pos);
    }
}

/// `REPCODE_TO_OFFBASE`.
pub(crate) const fn repcode_to_offbase(repcode: u32) -> u32 {
    repcode
}

/// `OFFSET_TO_OFFBASE`.
pub(crate) const fn offset_to_offbase(offset: u32) -> u32 {
    offset + REP_NUM as u32
}

/// `OFFBASE_IS_REPCODE`.
pub(crate) const fn offbase_is_repcode(off_base: u32) -> bool {
    off_base >= 1 && off_base <= REP_NUM as u32
}

/// `OFFBASE_TO_OFFSET`.
pub(crate) const fn offbase_to_offset(off_base: u32) -> u32 {
    off_base - REP_NUM as u32
}

/// Updates repeat offsets after a sequence (`ZSTD_updateRep`).
pub(crate) const fn update_rep(rep: &mut [u32; REP_NUM], off_base: u32, ll0: bool) {
    if off_base > REP_NUM as u32 {
        rep[2] = rep[1];
        rep[1] = rep[0];
        rep[0] = offbase_to_offset(off_base);
    } else {
        let rep_code = off_base - 1 + ll0 as u32;
        if rep_code > 0 {
            let current = if rep_code == REP_NUM as u32 {
                rep[0].wrapping_sub(1)
            } else {
                rep[rep_code as usize]
            };
            rep[2] = if rep_code >= 2 { rep[1] } else { rep[2] };
            rep[1] = rep[0];
            rep[0] = current;
        }
    }
}

/// Repeat offsets after a sequence, leaving `rep` unchanged (`ZSTD_newRep`).
pub(crate) const fn new_rep(rep: &[u32; REP_NUM], off_base: u32, ll0: bool) -> [u32; REP_NUM] {
    let mut next = *rep;
    update_rep(&mut next, off_base, ll0);
    next
}

/// `ZSTD_LLcode`.
pub(crate) const fn ll_code(lit_length: u32) -> u8 {
    const LL_CODE: [u8; 64] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20,
        20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23,
        24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    ];
    if lit_length > 63 {
        (highbit32(lit_length) + 19) as u8
    } else {
        LL_CODE[lit_length as usize]
    }
}

/// `ZSTD_MLcode`, from `matchLength - MINMATCH`.
pub(crate) const fn ml_code(ml_base: u32) -> u8 {
    const ML_CODE: [u8; 128] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37,
        38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
        41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    ];
    if ml_base > 127 {
        (highbit32(ml_base) + 36) as u8
    } else {
        ML_CODE[ml_base as usize]
    }
}

/// Small literal lengths mapped to their code and extra-bit width with
/// one load. Larger lengths use their high bit for both values.
const LL_CODE_BITS: [u16; 64] = {
    let mut entries = [0; 64];
    let mut length = 0;
    while length < entries.len() {
        let code = ll_code(length as u32);
        entries[length] = code as u16 | ((LL_BITS[code as usize] as u16) << 8);
        length += 1;
    }
    entries
};

/// The matching map for `match_length - MIN_MATCH`.
const ML_CODE_BITS: [u16; 128] = {
    let mut entries = [0; 128];
    let mut length = 0;
    while length < entries.len() {
        let code = ml_code(length as u32);
        entries[length] = code as u16 | ((ML_BITS[code as usize] as u16) << 8);
        length += 1;
    }
    entries
};

#[inline(always)]
const fn ll_code_bits(length: u16) -> u32 {
    if length < 64 {
        LL_CODE_BITS[length as usize] as u32
    } else {
        let high = highbit32(length as u32);
        (high + 19) | (high << 8)
    }
}

#[inline(always)]
const fn ml_code_bits(length: u16) -> u32 {
    if length < 128 {
        ML_CODE_BITS[length as usize] as u32
    } else {
        let high = highbit32(length as u32);
        (high + 36) | (high << 8)
    }
}

/// The literal and match lengths of one sequence
/// (`ZSTD_getSequenceLength`).
#[cfg(test)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct SequenceLength {
    pub(crate) lit_length: u32,
    pub(crate) match_length: u32,
}

/// The three FSE codes of one sequence (`llCode`, `ofCode` and `mlCode`)
/// and the number of extra bits they take. Four bytes, a byte more than the
/// reference's three code arrays: a cold call's peak heap barely grows.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SeqCodes {
    pub(crate) ll: u8,
    pub(crate) of: u8,
    pub(crate) ml: u8,
    /// Number of extra bits of the sequence, at most 63.
    pub(crate) extra_bits: u8,
}

impl SeqCodes {
    /// The codes of a sequence, with `ll` and `ml` below
    /// [`CODE_COUNT_SLOTS`].
    #[inline(always)]
    fn new(ll: u32, of: u32, ml: u32) -> Self {
        let ll_bits = u32::from(LL_EXTRA_BITS[ll as usize & (CODE_COUNT_SLOTS - 1)]);
        let ml_bits = u32::from(ML_EXTRA_BITS[ml as usize & (CODE_COUNT_SLOTS - 1)]);
        Self {
            ll: ll as u8,
            of: of as u8,
            ml: ml as u8,
            extra_bits: (ll_bits + ml_bits + of) as u8,
        }
    }

    /// The extra bits of `seq`, whose codes these are, as the sequences
    /// bitstream takes them: literal length, match length and offset, in
    /// that order from the least significant bit, with nothing above them.
    #[inline(always)]
    pub(crate) fn extra(self, seq: &SeqDef) -> u64 {
        let ll_bits = u32::from(LL_EXTRA_BITS[usize::from(self.ll) & (CODE_COUNT_SLOTS - 1)]);
        let ml_bits = u32::from(ML_EXTRA_BITS[usize::from(self.ml) & (CODE_COUNT_SLOTS - 1)]);
        let of = u32::from(self.of) & 31;
        let low = |value: u32, bits: u32| u64::from(value) & ((1u64 << bits) - 1);
        low(u32::from(seq.lit_length), ll_bits)
            | low(u32::from(seq.ml_base), ml_bits) << ll_bits
            | low(seq.off_base, of) << (ll_bits + ml_bits)
    }
}

/// `bits` padded to [`CODE_COUNT_SLOTS`] entries, so a code masked to six
/// bits indexes it without a bounds check.
const fn padded_bits<const N: usize>(bits: [u8; N]) -> [u8; CODE_COUNT_SLOTS] {
    let mut out = [0u8; CODE_COUNT_SLOTS];
    let mut i = 0;
    while i < N {
        out[i] = bits[i];
        i += 1;
    }
    out
}

/// Extra bits per literal length code.
const LL_EXTRA_BITS: [u8; CODE_COUNT_SLOTS] = padded_bits(LL_BITS);
/// Extra bits per match length code.
const ML_EXTRA_BITS: [u8; CODE_COUNT_SLOTS] = padded_bits(ML_BITS);

/// Number of entries in each [`CodeCounts`] histogram: every code fits in
/// six bits, so a code masked to them indexes without a bounds check.
pub(crate) const CODE_COUNT_SLOTS: usize = 64;

/// The maximum sequence and literal counts of a full block whose matches
/// have at least four bytes.
pub(crate) const FULL_BLOCK_SEQUENCES: usize = 1 << 15;
pub(crate) const FULL_BLOCK_LITERALS: usize = 1 << 17;

/// A full block's initialized sequence and literal destinations. Positions
/// belong to the writer rather than the vectors, so a match does not update
/// vector metadata or test capacity.
pub(crate) struct SeqCursor<'a> {
    sequences: &'a mut [SeqDef; FULL_BLOCK_SEQUENCES],
    literals: &'a mut [u8; FULL_BLOCK_LITERALS + 16],
    long_length: &'a mut LongLength,
    sequence_count: usize,
    literal_count: usize,
}

impl SeqCursor<'_> {
    #[inline(always)]
    pub(crate) const fn nb_seq(&self) -> usize {
        self.sequence_count
    }

    /// A cursor over the same destinations with the same counts; write its
    /// [`Self::counts`] back with [`Self::set_counts`].
    #[inline(always)]
    pub(crate) const fn reborrow(&mut self) -> SeqCursor<'_> {
        SeqCursor {
            sequences: &mut *self.sequences,
            literals: &mut *self.literals,
            long_length: &mut *self.long_length,
            sequence_count: self.sequence_count,
            literal_count: self.literal_count,
        }
    }

    /// The numbers of sequences and literals written.
    #[inline(always)]
    pub(crate) const fn counts(&self) -> (usize, usize) {
        (self.sequence_count, self.literal_count)
    }

    /// Takes over the counts of a [`Self::reborrow`]ed cursor.
    #[inline(always)]
    pub(crate) const fn set_counts(&mut self, (sequences, literals): (usize, usize)) {
        self.sequence_count = sequences;
        self.literal_count = literals;
    }

    #[inline(always)]
    pub(crate) fn store_seq_only(&mut self, lit_length: usize, off_base: u32, match_length: usize) {
        debug_assert!(self.sequence_count < FULL_BLOCK_SEQUENCES);
        debug_assert!(match_length >= MIN_MATCH);
        let pos = self.sequence_count as u32;
        let ml_base = match_length - MIN_MATCH;
        if (lit_length | ml_base) > 0xffff {
            mark_long_length(self.long_length, pos, lit_length, ml_base);
        }
        self.sequences[self.sequence_count & (FULL_BLOCK_SEQUENCES - 1)] = SeqDef {
            off_base,
            lit_length: lit_length as u16,
            ml_base: ml_base as u16,
        };
        self.sequence_count += 1;
    }

    #[inline(always)]
    pub(crate) fn store_seq_from(
        &mut self,
        src: &[u8],
        anchor: usize,
        lit_length: usize,
        off_base: u32,
        match_length: usize,
    ) {
        let start = self.literal_count & (FULL_BLOCK_LITERALS - 1);
        let end = start + lit_length;
        debug_assert!(end <= FULL_BLOCK_LITERALS);
        match src.get(anchor..).and_then(<[u8]>::first_chunk::<16>) {
            Some(chunk) if lit_length <= 16 => {
                self.literals[start..start + 16].copy_from_slice(chunk);
            }
            _ => self.literals[start..end].copy_from_slice(&src[anchor..anchor + lit_length]),
        }
        self.literal_count = end;
        self.store_seq_only(lit_length, off_base, match_length);
    }
}

/// Histograms of a block's three code streams (`HIST_countFast_wksp` over
/// `llCode`, `ofCode` and `mlCode`), counted while the codes are made.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CodeCounts {
    pub(crate) ll: [u32; CODE_COUNT_SLOTS],
    pub(crate) of: [u32; CODE_COUNT_SLOTS],
    pub(crate) ml: [u32; CODE_COUNT_SLOTS],
}

impl Default for CodeCounts {
    fn default() -> Self {
        Self {
            ll: [0; CODE_COUNT_SLOTS],
            of: [0; CODE_COUNT_SLOTS],
            ml: [0; CODE_COUNT_SLOTS],
        }
    }
}

/// Writes into canonical vectors or a promoted full-block cursor.
pub(crate) trait SeqWrite {
    fn nb_seq(&self) -> usize;

    fn store_seq_only(&mut self, lit_length: usize, off_base: u32, match_length: usize);

    fn store_seq_wild(
        &mut self,
        src: &[u8],
        anchor: usize,
        lit_length: usize,
        off_base: u32,
        match_length: usize,
    );
}

impl SeqWrite for SeqStore {
    #[inline(always)]
    fn nb_seq(&self) -> usize {
        SeqStore::nb_seq(self)
    }

    #[inline(always)]
    fn store_seq_only(&mut self, lit_length: usize, off_base: u32, match_length: usize) {
        SeqStore::store_seq_only(self, lit_length, off_base, match_length);
    }

    #[inline(always)]
    fn store_seq_wild(
        &mut self,
        src: &[u8],
        anchor: usize,
        lit_length: usize,
        off_base: u32,
        match_length: usize,
    ) {
        self.push_literals_wild(src, anchor, lit_length);
        self.store_seq_only(lit_length, off_base, match_length);
    }
}

impl SeqWrite for SeqCursor<'_> {
    #[inline(always)]
    fn nb_seq(&self) -> usize {
        SeqCursor::nb_seq(self)
    }

    #[inline(always)]
    fn store_seq_only(&mut self, lit_length: usize, off_base: u32, match_length: usize) {
        SeqCursor::store_seq_only(self, lit_length, off_base, match_length);
    }

    #[inline(always)]
    fn store_seq_wild(
        &mut self,
        src: &[u8],
        anchor: usize,
        lit_length: usize,
        off_base: u32,
        match_length: usize,
    ) {
        self.store_seq_from(src, anchor, lit_length, off_base, match_length);
    }
}

/// A full block's sequence and literal destinations, zeroed once and kept
/// by a [`SeqStore`] across blocks.
#[derive(Clone, Debug)]
struct FullBlock {
    sequences: alloc::boxed::Box<[SeqDef; FULL_BLOCK_SEQUENCES]>,
    literals: alloc::boxed::Box<[u8; FULL_BLOCK_LITERALS + 16]>,
}

impl FullBlock {
    fn new() -> Self {
        let sequences = alloc::vec![SeqDef::default(); FULL_BLOCK_SEQUENCES].into_boxed_slice();
        let literals = alloc::vec![0; FULL_BLOCK_LITERALS + 16].into_boxed_slice();
        match (sequences.try_into(), literals.try_into()) {
            (Ok(sequences), Ok(literals)) => Self {
                sequences,
                literals,
            },
            _ => unreachable!("the vectors have the arrays' lengths"),
        }
    }
}

/// Sequences and literals of one block.
///
/// They are written to the vectors, except where a block's finder is
/// promoted to a full-block cursor: the block then lives in the kept
/// [`FullBlock`] destinations, so promotion does not zero a block's room
/// again per block.
#[derive(Clone, Debug, Default)]
pub(crate) struct SeqStore {
    /// Sequences in order.
    sequences: alloc::vec::Vec<SeqDef>,
    /// Literals of every sequence and the block's trailing literals.
    literals: alloc::vec::Vec<u8>,
    /// The full-block cursor's destinations, once a block was promoted.
    full: Option<FullBlock>,
    /// The current block's sequence and literal counts in [`Self::full`],
    /// when it was promoted.
    full_counts: Option<(usize, usize)>,
    /// Per-sequence codes, filled by [`SeqStore::fill_codes`]; only the
    /// first [`Self::nb_seq`] entries belong to the current block.
    pub(crate) codes: alloc::vec::Vec<SeqCodes>,
    /// Histograms of `codes`, filled with them.
    pub(crate) code_counts: CodeCounts,
    /// The single sequence whose length exceeds 16 bits, if any.
    pub(crate) long_length: LongLength,
    /// Capacity in sequences (`maxNbSeq`).
    pub(crate) max_nb_seq: usize,
    /// Capacity in literals (`maxNbLit`).
    pub(crate) max_nb_lit: usize,
}

impl SeqStore {
    /// A store holding `sequences` and `literals`, for blocks of at most
    /// `max_nb_seq` sequences and `max_nb_lit` literals.
    pub(crate) fn from_parts(
        sequences: alloc::vec::Vec<SeqDef>,
        literals: alloc::vec::Vec<u8>,
        long_length: LongLength,
        max_nb_seq: usize,
        max_nb_lit: usize,
    ) -> Self {
        Self {
            sequences,
            literals,
            long_length,
            max_nb_seq,
            max_nb_lit,
            ..Self::default()
        }
    }

    /// The block's sequences.
    #[inline]
    pub(crate) fn sequences(&self) -> &[SeqDef] {
        match (&self.full, self.full_counts) {
            (Some(full), Some((sequences, _))) => &full.sequences[..sequences],
            _ => &self.sequences,
        }
    }

    /// The block's sequences, to rewrite their offsets.
    #[inline]
    pub(crate) fn sequences_mut(&mut self) -> &mut [SeqDef] {
        match (&mut self.full, self.full_counts) {
            (Some(full), Some((sequences, _))) => &mut full.sequences[..sequences],
            _ => &mut self.sequences,
        }
    }

    /// The block's literals.
    #[inline]
    pub(crate) fn literals(&self) -> &[u8] {
        match (&self.full, self.full_counts) {
            (Some(full), Some((_, literals))) => &full.literals[..literals],
            _ => &self.literals,
        }
    }

    /// Runs a full-block finder over the kept full-block destinations,
    /// allocated and zeroed on the first promotion, after the entries the
    /// vectors already hold; the block's entries then stay there.
    #[inline(always)]
    pub(crate) fn with_full_block_cursor<R>(
        &mut self,
        run: impl FnOnce(&mut SeqCursor<'_>) -> R,
    ) -> R {
        debug_assert!(self.full_counts.is_none());
        let full = self.full.get_or_insert_with(FullBlock::new);
        let sequence_count = self.sequences.len();
        let literal_count = self.literals.len();
        full.sequences[..sequence_count].copy_from_slice(&self.sequences);
        full.literals[..literal_count].copy_from_slice(&self.literals);
        let mut cursor = SeqCursor {
            sequences: &mut full.sequences,
            literals: &mut full.literals,
            long_length: &mut self.long_length,
            sequence_count,
            literal_count,
        };
        let result = run(&mut cursor);
        self.full_counts = Some((SeqWrite::nb_seq(&cursor), cursor.literal_count));
        result
    }

    /// Sizes the store for blocks of `block_size` bytes with matches of at
    /// least `min_match` bytes (`ZSTD_maxNbSeq`).
    pub(crate) fn reserve(&mut self, block_size: usize, min_match: u32) {
        let divider = if min_match == 3 { 3 } else { 4 };
        self.max_nb_seq = block_size / divider;
        self.max_nb_lit = block_size;
        self.sequences.reserve(self.max_nb_seq);
        // Room for the 16-byte chunk `store_seq_from` writes past the end.
        self.literals.reserve(self.max_nb_lit + 16);
    }

    /// Bytes held between blocks.
    pub(crate) fn retained_bytes(&self) -> usize {
        let full = if self.full.is_some() {
            size_of::<[SeqDef; FULL_BLOCK_SEQUENCES]>() + FULL_BLOCK_LITERALS + 16
        } else {
            0
        };
        self.sequences.capacity() * core::mem::size_of::<SeqDef>()
            + self.literals.capacity()
            + self.codes.capacity() * core::mem::size_of::<SeqCodes>()
            + full
    }

    /// Empties the store for a new block (`ZSTD_resetSeqStore`).
    pub(crate) fn reset(&mut self) {
        self.sequences.clear();
        self.literals.clear();
        self.full_counts = None;
        self.long_length = LongLength::None;
    }

    /// Number of sequences.
    pub(crate) fn nb_seq(&self) -> usize {
        match self.full_counts {
            Some((sequences, _)) => sequences,
            None => self.sequences.len(),
        }
    }

    /// Appends `src[anchor..anchor + lit_length]`; a run of at most 16 bytes
    /// is appended as one 16-byte chunk straight from `src` when `src` holds
    /// it, with a bounds test (`16 <= src.len() - anchor`) that cannot
    /// overflow, and cut back.
    #[inline(always)]
    pub(crate) fn push_literals_wild(&mut self, src: &[u8], anchor: usize, lit_length: usize) {
        debug_assert!(self.literals.len() + lit_length <= self.max_nb_lit);
        let literals = &mut self.literals;
        match src.get(anchor..).and_then(<[u8]>::first_chunk::<16>) {
            Some(chunk) if lit_length <= 16 => {
                let start = literals.len();
                literals.extend_from_slice(chunk);
                literals.truncate(start + lit_length);
            }
            _ => literals.extend_from_slice(&src[anchor..anchor + lit_length]),
        }
    }

    /// Appends a sequence without its literals (`ZSTD_storeSeqOnly`).
    #[inline(always)]
    pub(crate) fn store_seq_only(&mut self, lit_length: usize, off_base: u32, match_length: usize) {
        debug_assert!(self.full_counts.is_none());
        debug_assert!(self.sequences.len() < self.max_nb_seq);
        debug_assert!(match_length >= MIN_MATCH);
        let pos = self.sequences.len() as u32;
        let ml_base = match_length - MIN_MATCH;
        if (lit_length | ml_base) > 0xffff {
            mark_long_length(&mut self.long_length, pos, lit_length, ml_base);
        }
        self.sequences.push(SeqDef {
            off_base,
            lit_length: lit_length as u16,
            ml_base: ml_base as u16,
        });
    }

    /// Appends a sequence and its literals (`ZSTD_storeSeq`).
    pub(crate) fn store_seq(&mut self, literals: &[u8], off_base: u32, match_length: usize) {
        debug_assert!(self.literals.len() + literals.len() <= self.max_nb_lit);
        self.literals.extend_from_slice(literals);
        self.store_seq_only(literals.len(), off_base, match_length);
    }

    /// [`Self::store_seq`] for the literals `src[anchor..anchor + lit_length]`,
    /// copied as `ZSTD_storeSeq` does: a run of at most 16 bytes is copied as
    /// one fixed 16-byte chunk (`ZSTD_copy16`) when `src` holds that many,
    /// and the excess is cut off again, so short runs cost no length-driven
    /// copy.
    #[inline(always)]
    pub(crate) fn store_seq_from(
        &mut self,
        src: &[u8],
        anchor: usize,
        lit_length: usize,
        off_base: u32,
        match_length: usize,
    ) {
        debug_assert!(self.literals.len() + lit_length <= self.max_nb_lit);
        match src.get(anchor..anchor + 16) {
            Some(chunk) if lit_length <= 16 => {
                let start = self.literals.len();
                let mut wild = [0u8; 16];
                wild.copy_from_slice(chunk);
                self.literals.extend_from_slice(&wild);
                self.literals.truncate(start + lit_length);
            }
            _ => self
                .literals
                .extend_from_slice(&src[anchor..anchor + lit_length]),
        }
        self.store_seq_only(lit_length, off_base, match_length);
    }

    /// Appends the block's trailing literals (`ZSTD_storeLastLiterals`),
    /// after a promoted block's in its full-block destination.
    pub(crate) fn store_last_literals(&mut self, literals: &[u8]) {
        match (&mut self.full, &mut self.full_counts) {
            (Some(full), Some((_, count))) => {
                full.literals[*count..*count + literals.len()].copy_from_slice(literals);
                *count += literals.len();
            }
            _ => self.literals.extend_from_slice(literals),
        }
    }

    /// Fills the codes, their extra bits and their histograms
    /// (`ZSTD_seqToCodes`, with the three `HIST_countFast_wksp` passes of
    /// `ZSTD_buildSequencesStatistics` folded into the same loop), compiled
    /// for `backend`: the extra bits take variable shifts and masks, which
    /// BMI2 does without the `cl` register.
    pub(crate) fn fill_codes(&mut self, backend: Backend) {
        fearless_simd::dispatch!(backend.0, simd => fill_codes_on(simd, self));
    }

    /// The full lengths of sequence `index` (`ZSTD_getSequenceLength`).
    #[cfg(test)]
    pub(crate) fn sequence_length(&self, index: usize) -> SequenceLength {
        let seq = self.sequences()[index];
        let mut lengths = SequenceLength {
            lit_length: u32::from(seq.lit_length),
            match_length: u32::from(seq.ml_base) + MIN_MATCH as u32,
        };
        match self.long_length {
            LongLength::Literal(pos) if pos as usize == index => lengths.lit_length += 0x10000,
            LongLength::Match(pos) if pos as usize == index => lengths.match_length += 0x10000,
            _ => {}
        }
        lengths
    }
}

/// [`SeqStore::fill_codes`] inside `simd`'s feature context.
#[simd]
fn fill_codes_on<S: Simd>(simd: S, store: &mut SeqStore) {
    let long_length = store.long_length;
    let SeqStore {
        sequences,
        full,
        full_counts,
        codes: code_store,
        code_counts: counts,
        ..
    } = store;
    let sequences: &[SeqDef] = match (full, *full_counts) {
        (Some(full), Some((n, _))) => &full.sequences[..n],
        _ => sequences,
    };
    let n = sequences.len();
    // Every entry below `n` is written before it is read: the buffer only
    // grows, and only its new tail is zeroed.
    if code_store.len() < n {
        code_store.resize(n, SeqCodes::default());
    }
    *counts = CodeCounts::default();
    for (seq, codes) in sequences.iter().zip(&mut code_store[..n]) {
        // Masked to the histograms' size in 32 bits; every code is below
        // it already.
        let ll_packed = ll_code_bits(seq.lit_length);
        let ml_packed = ml_code_bits(seq.ml_base);
        let ll = ll_packed & (CODE_COUNT_SLOTS as u32 - 1);
        let ml = ml_packed & (CODE_COUNT_SLOTS as u32 - 1);
        // `highbit32`; an offset base is never zero, so setting the low bit
        // changes nothing and spares the zero case.
        let of = (seq.off_base | 1).ilog2();
        counts.ll[ll as usize] += 1;
        counts.of[of as usize] += 1;
        counts.ml[ml as usize] += 1;
        *codes = SeqCodes {
            ll: ll as u8,
            of: of as u8,
            ml: ml as u8,
            // The low bytes sum to at most MAX_LL + MAX_ML < 256,
            // so they cannot carry into the combined width.
            extra_bits: (((ll_packed + ml_packed) >> 8) + of) as u8,
        };
    }
    let (pos, ll, ml) = match long_length {
        LongLength::Literal(pos) => (pos as usize, Some(MAX_LL), None),
        LongLength::Match(pos) => (pos as usize, None, Some(MAX_ML)),
        LongLength::None => return,
    };
    if let Some(codes) = code_store[..n].get_mut(pos) {
        let ll = ll.unwrap_or(codes.ll);
        let ml = ml.unwrap_or(codes.ml);
        counts.ll[usize::from(codes.ll)] -= 1;
        counts.ll[usize::from(ll)] += 1;
        counts.ml[usize::from(codes.ml)] -= 1;
        counts.ml[usize::from(ml)] += 1;
        *codes = SeqCodes::new(ll.into(), codes.of.into(), ml.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_length_codes_match_code_and_width_for_every_stored_length() {
        for length in 0..=u16::MAX {
            let ll = ll_code(u32::from(length));
            let ml = ml_code(u32::from(length));
            assert_eq!(
                ll_code_bits(length),
                u32::from(ll) | (u32::from(LL_BITS[usize::from(ll)]) << 8),
                "literal length {length}",
            );
            assert_eq!(
                ml_code_bits(length),
                u32::from(ml) | (u32::from(ML_BITS[usize::from(ml)]) << 8),
                "match base {length}",
            );
        }
        for ll in 0..=MAX_LL {
            for ml in 0..=MAX_ML {
                let ll_bits = u32::from(LL_BITS[usize::from(ll)]);
                let ml_bits = u32::from(ML_BITS[usize::from(ml)]);
                let ll_packed = u32::from(ll) | (ll_bits << 8);
                let ml_packed = u32::from(ml) | (ml_bits << 8);
                assert_eq!((ll_packed + ml_packed) >> 8, ll_bits + ml_bits);
            }
        }
    }

    #[test]
    fn full_block_cursor_matches_vector_stores_at_buffer_edges() {
        let src: alloc::vec::Vec<u8> = (0..FULL_BLOCK_LITERALS).map(|i| i as u8).collect();
        let mut cursor_store = SeqStore::default();
        let mut vector_store = SeqStore::default();
        cursor_store.reserve(FULL_BLOCK_LITERALS, 4);
        vector_store.reserve(FULL_BLOCK_LITERALS, 4);
        let cases = [
            (0, 0, 4),
            (7, 16, 4),
            (25, 17, 9),
            (FULL_BLOCK_LITERALS - 7, 7, 5),
        ];
        cursor_store.with_full_block_cursor(|cursor| {
            for (anchor, len, ml) in cases {
                cursor.store_seq_from(&src, anchor, len, 7, ml);
            }
        });
        for (anchor, len, ml) in cases {
            vector_store.store_seq_from(&src, anchor, len, 7, ml);
        }
        assert_eq!(cursor_store.sequences(), vector_store.sequences());
        assert_eq!(cursor_store.literals(), vector_store.literals());
        assert_eq!(cursor_store.long_length, vector_store.long_length);
        cursor_store.reset();
        cursor_store.with_full_block_cursor(|cursor| {
            for _ in 0..FULL_BLOCK_SEQUENCES {
                cursor.store_seq_only(0, 1, 4);
            }
        });
        assert_eq!(cursor_store.sequences().len(), FULL_BLOCK_SEQUENCES);
        assert!(cursor_store.literals().is_empty());
        for (ll, ml) in [(0x10000, 4), (0, 0x10000 + MIN_MATCH)] {
            cursor_store.reset();
            vector_store.reset();
            cursor_store.with_full_block_cursor(|cursor| cursor.store_seq_only(ll, 7, ml));
            vector_store.store_seq_only(ll, 7, ml);
            assert_eq!(cursor_store.sequences(), vector_store.sequences());
            assert_eq!(cursor_store.long_length, vector_store.long_length);
        }
    }

    #[test]
    fn promoted_blocks_reuse_the_kept_full_block_and_expose_only_their_own() {
        let src: alloc::vec::Vec<u8> = (0..FULL_BLOCK_LITERALS).map(|i| (i * 7) as u8).collect();
        let mut store = SeqStore::default();
        store.reserve(FULL_BLOCK_LITERALS, 4);
        let before = store.retained_bytes();
        store.with_full_block_cursor(|cursor| {
            for anchor in 0..1000 {
                cursor.store_seq_from(&src, anchor * 40, 20, 9, 4);
            }
        });
        let kept = store.full.as_ref().map(|full| full.sequences.as_ptr());
        assert_eq!(
            store.retained_bytes() - before,
            FULL_BLOCK_SEQUENCES * size_of::<SeqDef>() + FULL_BLOCK_LITERALS + 16
        );
        let cases = [
            (3usize, 2usize, 5usize),
            (100, 16, 4),
            (FULL_BLOCK_LITERALS - 5, 5, 6),
        ];
        let mut expected = SeqStore::default();
        expected.reserve(FULL_BLOCK_LITERALS, 4);
        for (anchor, len, ml) in cases {
            expected.store_seq_from(&src, anchor, len, 11, ml);
        }
        expected.store_last_literals(&src[..3]);
        // A block written through the vectors, then one promoted after its
        // first sequence: both expose exactly their own entries.
        for promoted in [false, true] {
            store.reset();
            assert!(store.sequences().is_empty() && store.literals().is_empty());
            let (first, rest) = cases.split_first().unwrap();
            store.store_seq_from(&src, first.0, first.1, 11, first.2);
            if promoted {
                store.with_full_block_cursor(|cursor| {
                    for &(anchor, len, ml) in rest {
                        cursor.store_seq_from(&src, anchor, len, 11, ml);
                    }
                });
            } else {
                for &(anchor, len, ml) in rest {
                    store.store_seq_from(&src, anchor, len, 11, ml);
                }
            }
            store.store_last_literals(&src[..3]);
            assert_eq!(store.sequences(), expected.sequences(), "{promoted}");
            assert_eq!(store.literals(), expected.literals(), "{promoted}");
            assert_eq!(store.nb_seq(), cases.len());
            store.sequences_mut()[0].off_base = 1;
            assert_eq!(store.sequences()[0].off_base, 1);
        }
        assert_eq!(
            store.full.as_ref().map(|full| full.sequences.as_ptr()),
            kept
        );
    }

    #[test]
    fn stores_from_parts_hold_exactly_those_parts() {
        let sequences = alloc::vec![SeqDef {
            off_base: 9,
            lit_length: 2,
            ml_base: 0
        }];
        let parts = SeqStore::from_parts(
            sequences.clone(),
            b"xy".to_vec(),
            LongLength::Match(0),
            7,
            8,
        );
        assert_eq!(
            (parts.sequences(), parts.literals()),
            (&sequences[..], &b"xy"[..])
        );
        assert_eq!(
            (parts.nb_seq(), parts.max_nb_seq, parts.max_nb_lit),
            (1, 7, 8)
        );
        assert_eq!(parts.long_length, LongLength::Match(0));
    }

    #[test]
    fn repeat_offsets_follow_the_format_rules() {
        let mut rep = [1, 4, 8];
        update_rep(&mut rep, offset_to_offbase(100), false);
        assert_eq!(rep, [100, 1, 4]);
        update_rep(&mut rep, repcode_to_offbase(2), false);
        assert_eq!(rep, [1, 100, 4]);
        update_rep(&mut rep, repcode_to_offbase(1), false);
        assert_eq!(rep, [1, 100, 4]);
        update_rep(&mut rep, repcode_to_offbase(3), true);
        assert_eq!(rep, [0, 1, 100]);
        assert_eq!(new_rep(&[5, 6, 7], repcode_to_offbase(1), true), [6, 5, 7]);
        assert!(offbase_is_repcode(3) && !offbase_is_repcode(4));
    }

    #[test]
    fn length_codes_match_the_format_tables() {
        assert_eq!(ll_code(0), 0);
        assert_eq!(ll_code(16), 16);
        assert_eq!(ll_code(64), 25);
        assert_eq!(ll_code(65535), 34);
        assert_eq!(ml_code(0), 0);
        assert_eq!(ml_code(127), 42);
        assert_eq!(ml_code(128), 43);
    }

    #[test]
    fn wild_literal_copies_store_what_exact_copies_store() {
        let src: alloc::vec::Vec<u8> = (0..64u8).collect();
        // Short runs far from the end take the 16-byte chunk; runs near
        // the end or longer than 16 bytes copy exactly.
        for anchor in [0usize, 10, 47, 48, 50, 60] {
            for lit_length in 0..=(src.len() - anchor).min(20) {
                let mut wild = SeqStore::default();
                wild.reserve(1 << 10, 4);
                wild.store_seq_from(&src, anchor, lit_length, offset_to_offbase(7), 5);
                wild.store_seq_from(&src, 0, 3, repcode_to_offbase(1), 4);
                let mut exact = SeqStore::default();
                exact.reserve(1 << 10, 4);
                exact.store_seq(&src[anchor..anchor + lit_length], offset_to_offbase(7), 5);
                exact.store_seq(&src[..3], repcode_to_offbase(1), 4);
                assert_eq!(wild.literals(), exact.literals(), "{anchor} {lit_length}");
                assert_eq!(wild.sequences(), exact.sequences());
            }
        }
    }

    #[test]
    fn padded_extra_bits_extend_the_format_tables_with_zeros() {
        // Evaluated at run time too, not only in the constants.
        assert_eq!(padded_bits(LL_BITS), LL_EXTRA_BITS);
        assert_eq!(LL_EXTRA_BITS[..LL_BITS.len()], LL_BITS);
        assert!(LL_EXTRA_BITS[LL_BITS.len()..].iter().all(|&b| b == 0));
        assert_eq!(ML_EXTRA_BITS[..ML_BITS.len()], ML_BITS);
        assert!(ML_EXTRA_BITS[ML_BITS.len()..].iter().all(|&b| b == 0));
    }

    #[test]
    fn codes_carry_the_extra_bits_in_stream_order() {
        let mut store = SeqStore::default();
        store.reserve(1 << 17, 4);
        // Literal length 18 (code 17, 1 bit), match length 3 + 35 (code 33,
        // 1 bit), offset 100 (code 6, 6 bits).
        store.store_seq_only(18, offset_to_offbase(100), 35 + MIN_MATCH);
        store.store_seq_only(0, repcode_to_offbase(1), MIN_MATCH);
        for backend in Backend::available() {
            store.fill_codes(backend);
            let first = store.codes[0];
            assert_eq!((first.ll, first.ml, first.of), (17, 33, 6));
            assert_eq!(first.extra_bits, 8);
            // Literal length 18 is its code's base (extra 0), match length
            // base 35 one above its code's (extra 1), then the offset.
            let off_extra = u64::from(offset_to_offbase(100)) - (1 << 6);
            assert_eq!(
                first.extra(&store.sequences()[0]),
                (off_extra << 2) | (1 << 1),
                "{backend}"
            );
            let second = store.codes[1];
            assert_eq!(
                (second.extra(&store.sequences()[1]), second.extra_bits),
                (0, 0)
            );
            assert_eq!(store.code_counts.of[0], 1);
            assert_eq!(store.code_counts.of[6], 1);
            assert_eq!(store.code_counts.ll.iter().sum::<u32>(), 2);
        }
    }

    #[test]
    fn long_lengths_are_flagged_and_restored() {
        let mut store = SeqStore::default();
        store.reserve(1 << 17, 4);
        store.store_seq_only(0x10005, offset_to_offbase(9), 4);
        store.store_seq(b"ab", repcode_to_offbase(1), MIN_MATCH + 1);
        store.fill_codes(Backend::default());
        assert_eq!(store.codes[0].ll, MAX_LL);
        assert_eq!(store.code_counts.ll[usize::from(MAX_LL)], 1);
        assert_eq!(store.code_counts.ll.iter().sum::<u32>(), 2);
        assert_eq!(store.codes[0].extra_bits, 16 + 3);
        assert_eq!(store.codes[0].extra(&store.sequences()[0]) & 0xFFFF, 5);
        assert_eq!(store.sequence_length(0).lit_length, 0x10005);
        let mut other = SeqStore::default();
        other.reserve(1 << 17, 4);
        other.store_seq(b"", repcode_to_offbase(1), 0x10000 + MIN_MATCH + 1);
        other.fill_codes(Backend::default());
        assert_eq!(other.codes[0].ml, MAX_ML);
        assert_eq!(other.code_counts.ml[usize::from(MAX_ML)], 1);
        assert_eq!(
            other.sequence_length(0).match_length,
            0x10000 + MIN_MATCH as u32 + 1
        );
        other.reset();
        assert_eq!(other.nb_seq(), 0);
    }
}

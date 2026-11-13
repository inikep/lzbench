//! Huffman tree descriptions and literal stream decoding (RFC 8878
//! section 4.2).
//!
//! [`HuffmanTable::read`] parses a tree description, either direct 4-bit
//! weights or an FSE-compressed weight series, completes the implied last
//! weight and builds a lookup table indexed by the next 12 bits of a
//! backward stream, whatever the tree's depth, so lookups need no mask.
//! [`HuffmanTable::decode_streams`] then decodes one stream or four
//! streams with a jump table.
//!
//! Four streams decode interleaved, as in the reference's fast loops
//! (`HUF_decompress4X1_usingDTable_internal_fast_c_loop` and its `X2`
//! twin): each stream is a [`Sentinel`] reader, the rounds every stream can
//! afford are counted up front, and each round refills all four and
//! decodes several symbols from each, so the four dependency chains
//! overlap. When the reference's timing model (`HUF_selectDecoder`) favours
//! it, a table of symbol pairs decodes up to two symbols per lookup. Each
//! stream then finishes on its own with the single-symbol table and the
//! reference's cautious loop, and must be consumed exactly.
//!
//! The reference accepts more: its fast four-stream path does not check
//! where a stream ends, and its pair decoder caps a stream's last read.
//! When the exact decoder rejects a section, [`reference`] decodes it
//! again as the reference does, with the decoder type ([`TableKind`]) the
//! reference would use, and its verdict stands.

mod reference;

use alloc::boxed::Box;

pub(crate) use self::reference::TableKind;
use super::ahead::WriteAhead;
use super::bits::{BitContainer, Parked, Sentinel};
use super::fse::{MAX_WEIGHTS, decode_weights};
use crate::decompressor::InvalidDataKind;
use crate::shared::format::{BLOCK_SIZE_MAX, LIT_HUF_LOG, highbit32};

/// Deepest code the reference decoder accepts.
///
/// RFC 8878 limits literal codes to [`LIT_HUF_LOG`] (11) bits, and encoders
/// never exceed it, but the reference decoder sizes its tables for one more
/// bit and accepts 12-bit trees; this decoder accepts exactly what it does.
pub(crate) const MAX_TABLE_LOG: u32 = LIT_HUF_LOG + 1;

/// Bits every lookup reads: the deepest tree's.
const LOOKUP_BITS: u32 = MAX_TABLE_LOG;
/// Rows of every table.
const TABLE_SIZE: usize = 1 << LOOKUP_BITS;
/// Shift that leaves the next [`LOOKUP_BITS`] bits of a [`Sentinel`].
const LOOKUP_SHIFT: u32 = 64 - LOOKUP_BITS;
/// Masks an index into [`TABLE_SIZE`].
const INDEX_MASK: usize = TABLE_SIZE - 1;

/// Bits a pair lookup reads: the reference's fast tables'
/// (`HUF_DECODER_FAST_TABLELOG`). Trees up to this depth decode pairs;
/// five lookups then read at most 55 bits, which one refill provides.
const PAIR_BITS: u32 = LIT_HUF_LOG;
/// Rows of the pair table.
const PAIR_SIZE: usize = 1 << PAIR_BITS;
/// Shift that leaves the next [`PAIR_BITS`] bits of a [`Sentinel`].
const PAIR_SHIFT: u32 = 64 - PAIR_BITS;
/// Masks an index into [`PAIR_SIZE`].
const PAIR_MASK: usize = PAIR_SIZE - 1;

/// One lookup row packs the code length in the low byte, so a
/// [`Sentinel`] shift by the whole row consumes exactly the code, and the
/// symbol in the high byte.
type Row = u16;

/// A pair lookup row, the reference's `HUF_DEltX2`: the bits the symbols
/// take in the low byte, so a [`Sentinel`] shift by the whole row consumes
/// them straight from the load, as single rows do; one or two symbols in
/// the next 16 bits, first one lower; and the symbol count in the top two
/// bits, where a shift extracts it without a mask and its type bounds the
/// stores it advances. Packed in one word, a row of one symbol and a row
/// continuing it add up to the row of both, which keeps the table build to
/// one add per row.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
struct PairRow(u32);

/// Where a [`PairRow`] keeps its symbols.
const PAIR_SYMBOLS_SHIFT: u32 = 8;
/// Where a [`PairRow`] keeps its symbol count.
const PAIR_COUNT_SHIFT: u32 = 30;

impl PairRow {
    /// Packs a row.
    const fn new(symbols: u16, nb_bits: u8, count: u8) -> Self {
        Self(
            nb_bits as u32
                | (symbols as u32) << PAIR_SYMBOLS_SHIFT
                | (count as u32) << PAIR_COUNT_SHIFT,
        )
    }

    /// The decoded symbols, first one in the low byte.
    const fn symbols(self) -> u16 {
        (self.0 >> PAIR_SYMBOLS_SHIFT) as u16
    }

    /// The bits the symbols take.
    const fn nb_bits(self) -> u8 {
        self.0 as u8
    }

    /// The number of symbols: one or two in a table, at most three by the
    /// type.
    const fn count(self) -> usize {
        (self.0 >> PAIR_COUNT_SHIFT) as usize
    }
}

/// One slot of a [`PairTable`], read with an atomic load.
///
/// The load is an ordinary one on every target, but one the compiler may
/// not vectorize. The four streams look their pairs up side by side, and
/// given AVX-512 the compiler otherwise packs the four lookups into a
/// gather, whose latency then sits on every stream's chain of dependent
/// lookups. No slot is written while a table is shared: [`Self::get_mut`]
/// needs the table exclusively.
#[cfg(target_has_atomic = "32")]
#[derive(Debug)]
#[repr(transparent)]
struct PairSlot(core::sync::atomic::AtomicU32);

#[cfg(target_has_atomic = "32")]
impl PairSlot {
    /// An empty slot.
    const fn new() -> Self {
        Self(core::sync::atomic::AtomicU32::new(0))
    }

    /// The row.
    #[inline(always)]
    fn get(&self) -> PairRow {
        PairRow(self.0.load(core::sync::atomic::Ordering::Relaxed))
    }

    /// The row's word, to write.
    #[inline(always)]
    fn get_mut(&mut self) -> &mut u32 {
        self.0.get_mut()
    }
}

/// One slot of a [`PairTable`] on targets without 32-bit atomics.
#[cfg(not(target_has_atomic = "32"))]
#[derive(Debug)]
#[repr(transparent)]
struct PairSlot(u32);

#[cfg(not(target_has_atomic = "32"))]
impl PairSlot {
    /// An empty slot.
    const fn new() -> Self {
        Self(0)
    }

    /// The row.
    #[inline(always)]
    const fn get(&self) -> PairRow {
        PairRow(self.0)
    }

    /// The row's word, to write.
    #[inline(always)]
    const fn get_mut(&mut self) -> &mut u32 {
        &mut self.0
    }
}

/// The pair rows of one tree.
#[derive(Debug)]
struct PairTable(Box<[PairSlot; PAIR_SIZE]>);

impl PairTable {
    /// A table of empty rows.
    fn new() -> Self {
        Self(Box::new([const { PairSlot::new() }; PAIR_SIZE]))
    }

    /// The row of the 11-bit `index`; the mask shows it in bounds and
    /// changes no index a lookup makes.
    #[inline(always)]
    fn row(&self, index: usize) -> PairRow {
        self.0[index & PAIR_MASK].get()
    }

    /// The slots, to build the table.
    fn slots_mut(&mut self) -> &mut [PairSlot; PAIR_SIZE] {
        &mut self.0
    }

    /// The rows in index order.
    fn rows(&self) -> impl Iterator<Item = PairRow> + '_ {
        self.0.iter().map(PairSlot::get)
    }
}

impl Clone for PairTable {
    fn clone(&self) -> Self {
        let mut copy = Self::new();
        for (slot, row) in copy.0.iter_mut().zip(self.rows()) {
            *slot.get_mut() = row.0;
        }
        copy
    }
}

impl PartialEq for PairTable {
    fn eq(&self, other: &Self) -> bool {
        self.rows().eq(other.rows())
    }
}

impl Eq for PairTable {}

/// One slot of a [`LockstepRows`] table: a [`Row`] read with an atomic
/// load, for the reason [`PairSlot`] is.
#[cfg(target_has_atomic = "16")]
#[derive(Debug)]
#[repr(transparent)]
struct RowSlot(core::sync::atomic::AtomicU16);

#[cfg(target_has_atomic = "16")]
impl RowSlot {
    /// An empty slot.
    const fn new() -> Self {
        Self(core::sync::atomic::AtomicU16::new(0))
    }

    /// The row.
    #[inline(always)]
    fn get(&self) -> Row {
        self.0.load(core::sync::atomic::Ordering::Relaxed)
    }

    /// The row, to write.
    #[inline(always)]
    fn get_mut(&mut self) -> &mut Row {
        self.0.get_mut()
    }
}

/// One slot of a [`LockstepRows`] table on targets without 16-bit
/// atomics.
#[cfg(not(target_has_atomic = "16"))]
#[derive(Debug)]
#[repr(transparent)]
struct RowSlot(Row);

#[cfg(not(target_has_atomic = "16"))]
impl RowSlot {
    /// An empty slot.
    const fn new() -> Self {
        Self(0)
    }

    /// The row.
    #[inline(always)]
    const fn get(&self) -> Row {
        self.0
    }

    /// The row, to write.
    #[inline(always)]
    const fn get_mut(&mut self) -> &mut Row {
        &mut self.0
    }
}

/// A copy of a tree's single-symbol rows for the loop that looks the four
/// streams up side by side ([`single_rounds_lockstep`]): the same rows,
/// behind loads the compiler cannot pack into a gather. Every other
/// reader keeps the plain rows.
#[derive(Debug)]
struct LockstepRows(Box<[RowSlot; TABLE_SIZE]>);

impl LockstepRows {
    /// A table of empty rows.
    fn new() -> Self {
        Self(Box::new([const { RowSlot::new() }; TABLE_SIZE]))
    }

    /// The row of the 12-bit `index`; the mask shows it in bounds and
    /// changes no index a lookup makes.
    #[inline(always)]
    fn row(&self, index: usize) -> Row {
        self.0[index & INDEX_MASK].get()
    }

    /// Copies `rows` in, through exclusive references, as plain words.
    fn copy_from(&mut self, rows: &[Row; TABLE_SIZE]) {
        for (slot, &row) in self.0.iter_mut().zip(rows) {
            *slot.get_mut() = row;
        }
    }

    /// The rows in index order.
    fn rows(&self) -> impl Iterator<Item = Row> + '_ {
        self.0.iter().map(RowSlot::get)
    }
}

impl Clone for LockstepRows {
    fn clone(&self) -> Self {
        let mut copy = Self::new();
        for (slot, row) in copy.0.iter_mut().zip(self.rows()) {
            *slot.get_mut() = row;
        }
        copy
    }
}

impl PartialEq for LockstepRows {
    fn eq(&self, other: &Self) -> bool {
        self.rows().eq(other.rows())
    }
}

impl Eq for LockstepRows {}

/// Packs a row.
const fn row(symbol: u8, nb_bits: u32) -> Row {
    nb_bits as Row | ((symbol as Row) << 8)
}

/// A row's code length.
const fn row_bits(row: Row) -> u32 {
    (row & 0xff) as u32
}

/// A row's symbol.
const fn row_symbol(row: Row) -> u8 {
    (row >> 8) as u8
}

/// A literal decoding table, and on demand a table of symbol pairs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HuffmanTable {
    /// Depth of the tree.
    log: u32,
    rows: Box<[Row; TABLE_SIZE]>,
    /// Pair rows for trees of at most [`PAIR_BITS`], allocated on first
    /// use.
    pairs: Option<PairTable>,
    /// Whether `pairs` describes the current tree.
    pairs_valid: bool,
    /// The rows again for the lockstep loop, allocated on first use.
    lockstep: Option<LockstepRows>,
    /// Whether `lockstep` describes the current tree.
    lockstep_valid: bool,
    /// The decoder the reference reads this tree for, which decides how it
    /// treats stream ends the exact decoder rejects.
    kind: TableKind,
}

impl Default for HuffmanTable {
    fn default() -> Self {
        Self {
            log: 0,
            rows: Box::new([0; TABLE_SIZE]),
            pairs: None,
            pairs_valid: false,
            lockstep: None,
            lockstep_valid: false,
            kind: TableKind::Single,
        }
    }
}

/// The reference's decoder timing model (`algoTime`): per compression
/// ratio bucket, the table build and per-256-symbol decode times of the
/// single-symbol and the pair decoder.
const ALGO_TIME: [[(u32, u32); 2]; 16] = [
    [(0, 0), (1, 1)],
    [(0, 0), (1, 1)],
    [(150, 216), (381, 119)],
    [(170, 205), (514, 112)],
    [(177, 199), (539, 110)],
    [(197, 194), (644, 107)],
    [(221, 192), (735, 107)],
    [(256, 189), (881, 106)],
    [(359, 188), (1167, 109)],
    [(582, 187), (1570, 114)],
    [(688, 187), (1712, 122)],
    [(825, 186), (1965, 136)],
    [(976, 185), (2131, 150)],
    [(1180, 186), (2070, 175)],
    [(1377, 185), (1731, 202)],
    [(1412, 185), (1695, 202)],
];

/// Whether the pair decoder is expected to be faster for `literals`
/// symbols in `compressed` bytes: the reference's `HUF_selectDecoder`.
const fn prefers_pairs(literals: usize, compressed: usize) -> bool {
    let bucket = if compressed >= literals {
        15
    } else {
        compressed * 16 / literals
    };
    let blocks = (literals >> 8) as u32;
    let [single, pair] = ALGO_TIME[bucket];
    let single = single.0 + single.1 * blocks;
    let pair = pair.0 + pair.1 * blocks;
    pair + (pair >> 5) < single
}

/// Decodes one symbol from a container.
#[inline(always)]
fn decode_symbol(rows: &[Row; TABLE_SIZE], bits: &mut BitContainer) -> u8 {
    let row = rows[bits.look_nonzero(LOOKUP_BITS) as usize & INDEX_MASK];
    bits.skip(row_bits(row));
    row_symbol(row)
}

/// Finishes one stream into `out`: the reference's `HUF_decodeStreamX1`,
/// four symbols per refill while the stream can be refilled, then the
/// remaining symbols from the container. The stream must then be consumed
/// exactly.
#[inline(always)]
fn decode_tail(
    rows: &[Row; TABLE_SIZE],
    mut bits: BitContainer,
    src: &[u8],
    out: &mut [u8],
) -> Result<(), InvalidDataKind> {
    let mut done = 0;
    if out.len() > 3 {
        let (quads, _) = out.as_chunks_mut::<4>();
        for quad in quads {
            if !bits.reload(src) {
                break;
            }
            for byte in quad.iter_mut() {
                *byte = decode_symbol(rows, &mut bits);
            }
            done += 4;
        }
        // The reference evaluates its refill before its bound check, so
        // one more refill follows the last full quad; refills never change
        // which bits are read.
        bits.reload(src);
    } else {
        bits.reload(src);
    }
    for byte in &mut out[done..] {
        *byte = decode_symbol(rows, &mut bits);
    }
    if bits.is_exhausted() {
        Ok(())
    } else {
        Err(InvalidDataKind::HuffmanStream)
    }
}

/// Continues an X2 stream four pair lookups per refill while there is room
/// for their stores, then uses the exact single-symbol tail for its end.
#[inline(always)]
fn decode_pair_tail(
    rows: &[Row; TABLE_SIZE],
    pairs: &PairTable,
    mut bits: BitContainer,
    src: &[u8],
    out: &mut [u8],
) -> Result<(), InvalidDataKind> {
    const LOOKUPS: usize = 4;
    const WINDOW: usize = 3 * (LOOKUPS - 1) + 2;
    let mut done = 0;
    while let Some(window) = out
        .get_mut(done..)
        .and_then(<[u8]>::first_chunk_mut::<WINDOW>)
    {
        // Near the start, reload places all remaining bits in this word.
        // Valid pairs consume only the symbols still promised by out;
        // corrupt overreads keep avail wrapped until decode_tail rejects.
        bits.reload(src);
        let mut written = 0;
        for _ in 0..LOOKUPS {
            let pair = pairs.row(bits.look_nonzero(PAIR_BITS) as usize);
            bits.skip(u32::from(pair.nb_bits()));
            window[written..written + 2].copy_from_slice(&pair.symbols().to_le_bytes());
            written += pair.count();
        }
        done += written;
    }
    decode_tail(rows, bits, src, &mut out[done..])
}

/// The four streams of a four-stream literals section, laid out in one
/// buffer, and their outputs, laid out in another.
struct FourStreams<'a> {
    /// The jump table's streams, back to back.
    src: &'a [u8],
    /// Where each stream starts in `src`, and where the next one does.
    starts: [usize; 5],
    /// The regenerated literals.
    out: &'a mut [u8],
    /// Where each stream's output starts in `out`, and the next one's.
    segments: [usize; 5],
}

/// How many rounds every sentinel can afford: each refill moves back at
/// most seven bytes and must stay seven bytes into its stream.
#[inline(always)]
fn affordable_rounds(sentinels: &[Sentinel; 4], starts: &[usize; 5]) -> usize {
    let mut rounds = usize::MAX;
    for (sentinel, &start) in sentinels.iter().zip(starts) {
        rounds = rounds.min(sentinel.remaining(start) / 7);
    }
    rounds
}

/// Refills one stream within `limit` and decodes `K` symbols into
/// `window`, or returns `false`, leaving the stream as it was, when `limit`
/// lies past the end of `src`.
#[inline(always)]
fn single_round<const K: usize>(
    rows: &[Row; TABLE_SIZE],
    (src, limit): (&[u8], usize),
    sentinel: &mut Sentinel,
    window: &mut [u8; K],
) -> bool {
    if !sentinel.reload_within(src, limit) {
        return false;
    }
    for byte in window {
        let row = rows[sentinel.peek(LOOKUP_SHIFT)];
        sentinel.skip(u32::from(row));
        *byte = row_symbol(row);
    }
    true
}

/// The `K`-byte windows of `segment` from `from` on.
#[inline(always)]
fn windows<const K: usize>(segment: &mut [u8], from: usize) -> core::slice::IterMut<'_, [u8; K]> {
    segment
        .get_mut(from..)
        .unwrap_or_default()
        .as_chunks_mut::<K>()
        .0
        .iter_mut()
}

/// Decodes `K` symbols per stream per round with single-symbol rows while
/// every stream can afford it, and returns how many symbols each stream
/// decoded; all four advance together.
///
/// Every round reads at most `K * log` bits after a refill that leaves at
/// most seven consumed, which the caller keeps within 56. Each stream
/// stores its symbols straight into a `K`-byte window of its segment, the
/// reference's `HUF_4X1_DECODE_SYMBOL`, so a round keeps no symbols in
/// registers. The windows come from walking the four segments, each cut
/// to the shortest one, in `K`-byte chunks, so they need no bounds checks.
/// Each complete round also prefetches part of the next block's output
/// (`ahead`).
#[inline(always)]
fn single_rounds<const K: usize>(
    rows: &[Row; TABLE_SIZE],
    streams: &mut FourStreams<'_>,
    sentinels: &mut [Sentinel; 4],
    ahead: &mut WriteAhead,
) -> [usize; 4] {
    let src = streams.src;
    let [b1, b2, b3, b4, end] = streams.segments;
    // The fourth segment is the shortest.
    let length = end - b4;
    let mut done = 0;
    let mut decoded = [0; 4];
    let mut s = *sentinels;
    let segments = streams.out.get_mut(b1..end).and_then(|out| {
        let (first, rest) = out.split_at_mut_checked(b2 - b1)?;
        let (second, rest) = rest.split_at_mut_checked(b3 - b2)?;
        let (third, fourth) = rest.split_at_mut_checked(b4 - b3)?;
        Some([
            first.get_mut(..length)?,
            second.get_mut(..length)?,
            third.get_mut(..length)?,
            fourth,
        ])
    });
    // Refills stay below this, which the round counts ensure; clamped to
    // it, they need no branches.
    let (Some([first, second, third, fourth]), Some(limit)) = (segments, src.len().checked_sub(8))
    else {
        return decoded;
    };
    let src = (src, limit);
    'rounds: loop {
        let rounds = affordable_rounds(&s, &streams.starts).min((length - done) / K);
        if rounds == 0 {
            break;
        }
        let windows = windows::<K>(first, done)
            .zip(windows::<K>(second, done))
            .zip(windows::<K>(third, done))
            .zip(windows::<K>(fourth, done))
            .take(rounds);
        let [s1, s2, s3, s4] = &mut s;
        for (((w1, w2), w3), w4) in windows {
            // Stopping between two streams leaves the earlier ones a round
            // ahead, which their counts record; the counts above never let
            // it happen.
            let stopped = if !single_round(rows, src, s1, w1) {
                0
            } else if !single_round(rows, src, s2, w2) {
                1
            } else if !single_round(rows, src, s3, w3) {
                2
            } else if !single_round(rows, src, s4, w4) {
                3
            } else {
                done += K;
                ahead.step();
                continue;
            };
            decoded
                .iter_mut()
                .take(stopped)
                .for_each(|count| *count = K);
            break 'rounds;
        }
    }
    *sentinels = s;
    decoded.map(|extra| done + extra)
}

/// The `K`-byte window of `segment` at `from`, if it has one.
#[inline(always)]
fn window_at<const K: usize>(segment: &mut [u8], from: usize) -> Option<&mut [u8; K]> {
    segment.get_mut(from..)?.first_chunk_mut::<K>()
}

/// [`single_rounds`] with the four streams in lockstep, the way the
/// reference's assembly loop decodes: the streams refill, then every step
/// looks one symbol up in each stream, so four chains of dependent
/// lookups overlap without the processor having to reach across a whole
/// stream's round for the next stream's.
///
/// A round keeps the four containers in registers. The stream positions,
/// read once a round, are [`Parked`] in memory, and the round's windows
/// are cut after the refill, so the refill holds nothing but the
/// containers. Each refill ends a basic block, and so does each half of a
/// lookup step: four like operations in one block are what the compiler
/// packs into vector registers, which puts every stream's chain through
/// them. A refill fails only when the clamp's limit is out of range; the
/// streams that did refill then hold a valid, merely refilled, state. The
/// rounds, their bounds and the decoded symbols are those of
/// [`single_rounds`]. Returns `None` for a row without bits, which no
/// complete tree has: the section is then invalid.
#[inline(always)]
fn single_rounds_lockstep<const K: usize>(
    rows: &LockstepRows,
    streams: &mut FourStreams<'_>,
    sentinels: &mut [Sentinel; 4],
    ahead: &mut WriteAhead,
) -> Option<[usize; 4]> {
    let src = streams.src;
    let [b1, b2, b3, b4, end] = streams.segments;
    // The fourth segment is the shortest.
    let length = end - b4;
    let mut done = 0;
    let segments = streams.out.get_mut(b1..end).and_then(|out| {
        let (first, rest) = out.split_at_mut_checked(b2 - b1)?;
        let (second, rest) = rest.split_at_mut_checked(b3 - b2)?;
        let (third, fourth) = rest.split_at_mut_checked(b4 - b3)?;
        Some([
            first.get_mut(..length)?,
            second.get_mut(..length)?,
            third.get_mut(..length)?,
            fourth,
        ])
    });
    let Some([first, second, third, fourth]) = segments else {
        return Some([0; 4]);
    };
    let [(mut c1, p1), (mut c2, p2), (mut c3, p3), (mut c4, p4)] =
        sentinels.map(Sentinel::into_parts);
    let parked = [p1, p2, p3, p4].map(Parked::new);
    let [p1, p2, p3, p4] = &parked;
    'rounds: loop {
        let now = [
            Sentinel::from_parts(c1, p1.get()),
            Sentinel::from_parts(c2, p2.get()),
            Sentinel::from_parts(c3, p3.get()),
            Sentinel::from_parts(c4, p4.get()),
        ];
        let rounds = affordable_rounds(&now, &streams.starts).min((length - done) / K);
        if rounds == 0 {
            break;
        }
        for _ in 0..rounds {
            if !Sentinel::reload_parked(&mut c1, p1, src)
                || !Sentinel::reload_parked(&mut c2, p2, src)
                || !Sentinel::reload_parked(&mut c3, p3, src)
                || !Sentinel::reload_parked(&mut c4, p4, src)
            {
                break 'rounds;
            }
            let (Some(w1), Some(w2), Some(w3), Some(w4)) = (
                window_at::<K>(first, done),
                window_at::<K>(second, done),
                window_at::<K>(third, done),
                window_at::<K>(fourth, done),
            ) else {
                break 'rounds;
            };
            for (((s1, s2), s3), s4) in w1.iter_mut().zip(w2).zip(w3).zip(w4) {
                let r1 = rows.row((c1 >> LOOKUP_SHIFT) as usize);
                let r2 = rows.row((c2 >> LOOKUP_SHIFT) as usize);
                c1 <<= r1 & 63;
                c2 <<= r2 & 63;
                *s1 = row_symbol(r1);
                *s2 = row_symbol(r2);
                if r2 == 0 {
                    return None;
                }
                let r3 = rows.row((c3 >> LOOKUP_SHIFT) as usize);
                let r4 = rows.row((c4 >> LOOKUP_SHIFT) as usize);
                c3 <<= r3 & 63;
                c4 <<= r4 & 63;
                *s3 = row_symbol(r3);
                *s4 = row_symbol(r4);
                if r4 == 0 {
                    return None;
                }
            }
            done += K;
            ahead.step();
        }
    }
    *sentinels = [
        Sentinel::from_parts(c1, p1.get()),
        Sentinel::from_parts(c2, p2.get()),
        Sentinel::from_parts(c3, p3.get()),
        Sentinel::from_parts(c4, p4.get()),
    ];
    Some([done; 4])
}

/// Pair lookups per stream per round.
const PAIR_LOOKUPS: usize = 5;
/// Bytes a round may write per stream: two per lookup.
const PAIR_ROUND_BYTES: usize = 2 * PAIR_LOOKUPS;
/// Each stream's window per round. A [`PairRow`] count is at most three by
/// its type, so the stores of a round provably end inside the window and
/// need no checks; the counts in a table are at most two.
const PAIR_WINDOW: usize = 3 * (PAIR_LOOKUPS - 1) + 2;

/// Decodes [`PAIR_LOOKUPS`] pair lookups per stream per round while every
/// stream can afford them and has room for the round's stores, and
/// returns how many symbols each stream decoded: the reference's
/// `HUF_decompress4X2_usingDTable_internal_fast_c_loop`.
///
/// Every round reads at most `5 * 11` bits after a refill that leaves at
/// most seven consumed. Each lookup stores two bytes and advances by the
/// symbols it decoded, as `HUF_4X2_DECODE_SYMBOL` does; a store's byte past
/// the symbols lies inside the stream's own segment, where the next lookup
/// or the tail overwrites it. The first three streams interleave their
/// lookups; the fourth is decoded separately to limit live registers while
/// retaining three independent lookup chains.
///
/// Refill positions and window offsets are clamped to the last ones that
/// fit their buffers. The round counts keep them below those anyway, so
/// the clamps change nothing, but they prove the loads and windows in
/// bounds with a compare and a conditional move instead of a branch each.
#[inline(always)]
fn pair_rounds(
    pairs: &PairTable,
    streams: &mut FourStreams<'_>,
    sentinels: &mut [Sentinel; 4],
) -> [usize; 4] {
    const {
        assert!(PAIR_LOOKUPS * PAIR_BITS as usize + 7 <= 63);
        assert!(PAIR_WINDOW >= PAIR_ROUND_BYTES);
    };
    let src = streams.src;
    let segments = streams.segments;
    let mut ops = [segments[0], segments[1], segments[2], segments[3]];
    let mut s = *sentinels;
    let out = &mut *streams.out;
    // Split the output once so each stream's clamped window is confined to
    // its own segment. Counts of at most three bound the unrolled stores.
    let [b1, b2, b3, b4, end] = segments;
    let Some(out) = out.get_mut(b1..end) else {
        return [0; 4];
    };
    let Some((first, rest)) = out.split_at_mut_checked(b2 - b1) else {
        return [0; 4];
    };
    let Some((second, rest)) = rest.split_at_mut_checked(b3 - b2) else {
        return [0; 4];
    };
    let Some((third, fourth)) = rest.split_at_mut_checked(b4 - b3) else {
        return [0; 4];
    };
    let Some(src_limit) = src.len().checked_sub(8) else {
        return [0; 4];
    };
    let [Some(l1), Some(l2), Some(l3), Some(l4)] =
        [first.len(), second.len(), third.len(), fourth.len()]
            .map(|len| len.checked_sub(PAIR_WINDOW))
    else {
        return [0; 4];
    };
    'rounds: loop {
        // Every position indexes the common input. Later streams may
        // refill across their starts; into_container restores their exact
        // remaining bits before the stream-local tails validate the ends.
        let mut rounds = affordable_rounds(&s, &[0; 5]);
        for (op, end) in ops.iter().zip(&segments[1..]) {
            rounds = rounds
                .min((end - op).saturating_sub(PAIR_WINDOW - PAIR_ROUND_BYTES) / PAIR_ROUND_BYTES);
        }
        if rounds == 0 {
            break;
        }
        let [s1, s2, s3, s4] = &mut s;
        for _ in 0..rounds {
            if !s1.reload_within(src, src_limit)
                || !s2.reload_within(src, src_limit)
                || !s3.reload_within(src, src_limit)
                || !s4.reload_within(src, src_limit)
            {
                break 'rounds;
            }
            let [a1, a2, a3, a4] = core::array::from_fn(|i| ops[i] - segments[i]);
            let Some(w1) = first
                .get_mut(a1.min(l1)..)
                .and_then(<[u8]>::first_chunk_mut::<PAIR_WINDOW>)
            else {
                break 'rounds;
            };
            let Some(w2) = second
                .get_mut(a2.min(l2)..)
                .and_then(<[u8]>::first_chunk_mut::<PAIR_WINDOW>)
            else {
                break 'rounds;
            };
            let Some(w3) = third
                .get_mut(a3.min(l3)..)
                .and_then(<[u8]>::first_chunk_mut::<PAIR_WINDOW>)
            else {
                break 'rounds;
            };
            let Some(w4) = fourth
                .get_mut(a4.min(l4)..)
                .and_then(<[u8]>::first_chunk_mut::<PAIR_WINDOW>)
            else {
                break 'rounds;
            };
            let (mut n1, mut n2, mut n3, mut n4) = (0, 0, 0, 0);
            for _ in 0..PAIR_LOOKUPS {
                let p1 = pairs.row(s1.peek(PAIR_SHIFT));
                let p2 = pairs.row(s2.peek(PAIR_SHIFT));
                let p3 = pairs.row(s3.peek(PAIR_SHIFT));
                s1.skip(p1.0);
                s2.skip(p2.0);
                s3.skip(p3.0);
                w1[n1..n1 + 2].copy_from_slice(&p1.symbols().to_le_bytes());
                w2[n2..n2 + 2].copy_from_slice(&p2.symbols().to_le_bytes());
                w3[n3..n3 + 2].copy_from_slice(&p3.symbols().to_le_bytes());
                n1 += p1.count();
                n2 += p2.count();
                n3 += p3.count();
            }
            for _ in 0..PAIR_LOOKUPS {
                let p4 = pairs.row(s4.peek(PAIR_SHIFT));
                s4.skip(p4.0);
                w4[n4..n4 + 2].copy_from_slice(&p4.symbols().to_le_bytes());
                n4 += p4.count();
            }
            for (op, count) in ops.iter_mut().zip([n1, n2, n3, n4]) {
                *op += count;
            }
        }
    }
    *sentinels = s;
    [
        ops[0] - segments[0],
        ops[1] - segments[1],
        ops[2] - segments[2],
        ops[3] - segments[3],
    ]
}

/// Positions in a literals section: below [`BLOCK_SIZE_MAX`], a power of
/// two.
const OUT_MASK: usize = BLOCK_SIZE_MAX - 1;
/// Bytes a literal buffer needs for [`pair_rounds_fixed`]: every position
/// a mask leaves, and a whole [`PAIR_WINDOW`] after it.
pub(crate) const FIXED_OUT: usize = BLOCK_SIZE_MAX + PAIR_WINDOW;

/// Decodes one pair lookup into `out` at `*op`, the reference's
/// `HUF_4X2_DECODE_SYMBOL`: two bytes are stored and `*op` advances by the
/// symbols decoded.
#[inline(always)]
fn pair_lookup(
    pairs: &PairTable,
    sentinel: &mut Sentinel,
    out: &mut [u8; FIXED_OUT],
    op: &mut usize,
) {
    let pair = pairs.row(sentinel.peek(PAIR_SHIFT));
    sentinel.skip(pair.0);
    out[*op..*op + 2].copy_from_slice(&pair.symbols().to_le_bytes());
    *op += pair.count();
}

/// [`pair_rounds`] over a buffer of [`FIXED_OUT`] bytes, as the reference
/// decodes: each stream keeps only its container and its output position
/// live, the first three streams interleave their lookups and the fourth
/// stream's interleave with the refills, and the fourth stream's position
/// counts the rounds.
///
/// Every round masks each position to [`OUT_MASK`], which a section's
/// positions never exceed. A round then advances a position by at most
/// `3 * PAIR_LOOKUPS` (the type of a count bounds it) before its last
/// store, so every store provably lies inside the buffer with no check.
/// The round counts keep each stream's stores inside its own segment, as
/// [`pair_rounds`] does. Each round also prefetches part of the next
/// block's output (`ahead`).
#[inline(always)]
fn pair_rounds_fixed(
    pairs: &PairTable,
    src: &[u8],
    out: &mut [u8; FIXED_OUT],
    segments: [usize; 5],
    sentinels: &mut [Sentinel; 4],
    ahead: &mut WriteAhead,
) -> [usize; 4] {
    const {
        assert!(PAIR_LOOKUPS * PAIR_BITS as usize + 7 <= 63);
        assert!(OUT_MASK + 1 + 3 * (PAIR_LOOKUPS - 1) + 2 <= FIXED_OUT);
    };
    let starts = [segments[0], segments[1], segments[2], segments[3]];
    let mut ops = starts;
    let mut s = *sentinels;
    let Some(src_limit) = src.len().checked_sub(8) else {
        return [0; 4];
    };
    'rounds: loop {
        let mut rounds = affordable_rounds(&s, &[0; 5]);
        for (op, end) in ops.iter().zip(&segments[1..]) {
            rounds = rounds
                .min((end - op).saturating_sub(PAIR_WINDOW - PAIR_ROUND_BYTES) / PAIR_ROUND_BYTES);
        }
        if rounds == 0 {
            break;
        }
        // Each lookup decodes at least one symbol, so the fourth stream
        // gets there within `rounds` rounds.
        let olimit = ops[3] + PAIR_LOOKUPS * rounds;
        let [s1, s2, s3, s4] = &mut s;
        let [mut o1, mut o2, mut o3, mut o4] = ops;
        loop {
            [o1, o2, o3, o4] = [o1, o2, o3, o4].map(|op| op & OUT_MASK);
            for _ in 0..PAIR_LOOKUPS {
                pair_lookup(pairs, s1, out, &mut o1);
                pair_lookup(pairs, s2, out, &mut o2);
                pair_lookup(pairs, s3, out, &mut o3);
            }
            pair_lookup(pairs, s4, out, &mut o4);
            pair_lookup(pairs, s4, out, &mut o4);
            let refilled = s1.reload_within(src, src_limit);
            pair_lookup(pairs, s4, out, &mut o4);
            let refilled = refilled & s2.reload_within(src, src_limit);
            pair_lookup(pairs, s4, out, &mut o4);
            let refilled = refilled & s3.reload_within(src, src_limit);
            pair_lookup(pairs, s4, out, &mut o4);
            let refilled = refilled & s4.reload_within(src, src_limit);
            ahead.step();
            if !refilled {
                ops = [o1, o2, o3, o4];
                break 'rounds;
            }
            if o4 >= olimit {
                break;
            }
        }
        ops = [o1, o2, o3, o4];
    }
    *sentinels = s;
    [
        ops[0] - starts[0],
        ops[1] - starts[1],
        ops[2] - starts[2],
        ops[3] - starts[3],
    ]
}

/// [`pair_rounds_fixed`] with the four streams in lockstep, the way the
/// reference's assembly loop decodes: every lookup step takes one pair
/// from each stream, so four chains of dependent lookups overlap, and the
/// streams refill together at the end of a round.
///
/// It is the loop for AVX-512 feature contexts. There the compiler packs
/// four streams' plain table loads into one gather, which is why
/// [`PairTable`] is read with atomic loads: they stay four scalar loads.
/// The four containers and the four output positions are the values a
/// round keeps on its chains; the stream positions, read once a round,
/// are [`Parked`] in memory so that they do not compete with the
/// containers for registers. Each round also prefetches part of the next
/// block's output (`ahead`). The rounds, their bounds and the decoded
/// symbols are those of [`pair_rounds_fixed`].
#[inline(always)]
fn pair_rounds_lockstep(
    pairs: &PairTable,
    src: &[u8],
    out: &mut [u8; FIXED_OUT],
    segments: [usize; 5],
    sentinels: &mut [Sentinel; 4],
    ahead: &mut WriteAhead,
) -> [usize; 4] {
    let starts = [segments[0], segments[1], segments[2], segments[3]];
    let mut ops = starts;
    let [(mut b1, p1), (mut b2, p2), (mut b3, p3), (mut b4, p4)] =
        sentinels.map(Sentinel::into_parts);
    let parked = [p1, p2, p3, p4].map(Parked::new);
    let [p1, p2, p3, p4] = &parked;
    // One pair of one stream: the reference's `HUF_4X2_DECODE_SYMBOL`.
    macro_rules! lookup {
        ($bits:ident, $op:ident) => {{
            let pair = pairs.row(($bits >> PAIR_SHIFT) as usize);
            $bits <<= pair.0 & 63;
            out[$op..$op + 2].copy_from_slice(&pair.symbols().to_le_bytes());
            $op += pair.count();
        }};
    }
    'rounds: loop {
        let now = [
            Sentinel::from_parts(b1, p1.get()),
            Sentinel::from_parts(b2, p2.get()),
            Sentinel::from_parts(b3, p3.get()),
            Sentinel::from_parts(b4, p4.get()),
        ];
        let mut rounds = affordable_rounds(&now, &[0; 5]);
        for (op, end) in ops.iter().zip(&segments[1..]) {
            rounds = rounds
                .min((end - op).saturating_sub(PAIR_WINDOW - PAIR_ROUND_BYTES) / PAIR_ROUND_BYTES);
        }
        if rounds == 0 {
            break;
        }
        let [mut o1, mut o2, mut o3, mut o4] = ops;
        for _ in 0..rounds {
            [o1, o2, o3, o4] = [o1, o2, o3, o4].map(|op| op & OUT_MASK);
            for _ in 0..PAIR_LOOKUPS {
                lookup!(b1, o1);
                lookup!(b2, o2);
                lookup!(b3, o3);
                lookup!(b4, o4);
            }
            ahead.step();
            // Each refill ends a basic block, so the four are not packed
            // into one gather.
            if !Sentinel::reload_parked(&mut b1, p1, src)
                || !Sentinel::reload_parked(&mut b2, p2, src)
                || !Sentinel::reload_parked(&mut b3, p3, src)
                || !Sentinel::reload_parked(&mut b4, p4, src)
            {
                ops = [o1, o2, o3, o4];
                break 'rounds;
            }
        }
        ops = [o1, o2, o3, o4];
    }
    *sentinels = [
        Sentinel::from_parts(b1, p1.get()),
        Sentinel::from_parts(b2, p2.get()),
        Sentinel::from_parts(b3, p3.get()),
        Sentinel::from_parts(b4, p4.get()),
    ];
    [
        ops[0] - starts[0],
        ops[1] - starts[1],
        ops[2] - starts[2],
        ops[3] - starts[3],
    ]
}

/// Slots for weights `0..=12` in a counting sort.
const WEIGHT_SLOTS: usize = 16;

/// Writes one run of `L` rows per symbol from `start`, returning where the
/// next run starts. Runs start at multiples of `L`, so the mask never
/// changes an index; it only shows the stores stay inside the table.
#[inline(always)]
fn fill_runs<const L: usize>(
    rows: &mut [Row; TABLE_SIZE],
    mut start: usize,
    symbols: &[u8],
    nb_bits: u32,
) -> usize {
    for &symbol in symbols {
        let at = start & (TABLE_SIZE - L);
        rows[at..at + L].copy_from_slice(&[row(symbol, nb_bits); L]);
        start += L;
    }
    start
}

/// [`fill_runs`] for runs of `len` rows, a power of two of at least 16,
/// written sixteen rows at a time.
#[inline(always)]
fn fill_long_runs(
    rows: &mut [Row; TABLE_SIZE],
    mut start: usize,
    symbols: &[u8],
    nb_bits: u32,
    len: usize,
) -> usize {
    for &symbol in symbols {
        let chunk = [row(symbol, nb_bits); 16];
        if let Some(run) = rows.get_mut(start..start + len) {
            run.as_chunks_mut::<16>().0.fill(chunk);
        }
        start += len;
    }
    start
}

impl HuffmanTable {
    /// Heap bytes the table holds.
    pub(crate) fn retained_bytes(&self) -> usize {
        size_of::<[Row; TABLE_SIZE]>()
            + self
                .pairs
                .as_ref()
                .map_or(0, |_| size_of::<[PairSlot; PAIR_SIZE]>())
            + self
                .lockstep
                .as_ref()
                .map_or(0, |_| size_of::<[RowSlot; TABLE_SIZE]>())
    }

    /// Copies the rows into the table the lockstep loop reads.
    fn build_lockstep(&mut self) {
        self.lockstep
            .get_or_insert_with(LockstepRows::new)
            .copy_from(&self.rows);
        self.lockstep_valid = true;
    }

    /// Parses a tree description at the start of `src` into this table,
    /// returning the bytes the description occupies. On error the table is
    /// left unusable and must be read again.
    pub(crate) fn read(&mut self, src: &[u8]) -> Result<usize, InvalidDataKind> {
        let invalid = InvalidDataKind::HuffmanTable;
        let (&header, body) = src.split_first().ok_or(invalid)?;
        let mut weights = [0u8; MAX_WEIGHTS + 1];
        let (count, used) = if header >= 128 {
            let count = usize::from(header) - 127;
            let bytes = count.div_ceil(2);
            let packed = body.get(..bytes).ok_or(invalid)?;
            for (pair, &byte) in weights.chunks_mut(2).zip(packed) {
                pair[0] = byte >> 4;
                if let Some(second) = pair.get_mut(1) {
                    *second = byte & 0x0f;
                }
            }
            (count, 1 + bytes)
        } else {
            let size = usize::from(header);
            let compressed = body.get(..size).ok_or(invalid)?;
            (decode_weights(compressed, &mut weights)?, 1 + size)
        };
        self.pairs_valid = false;
        self.lockstep_valid = false;
        self.fill(&mut weights, count)?;
        Ok(used)
    }

    /// Completes the implied last weight of `weights[..count]` and builds
    /// the single-symbol rows.
    fn fill(
        &mut self,
        weights: &mut [u8; MAX_WEIGHTS + 1],
        count: usize,
    ) -> Result<(), InvalidDataKind> {
        let invalid = InvalidDataKind::HuffmanTable;
        let mut rank_count = [0u32; MAX_TABLE_LOG as usize + 1];
        let mut total: u32 = 0;
        for &weight in &weights[..count] {
            if u32::from(weight) > MAX_TABLE_LOG {
                return Err(invalid);
            }
            rank_count[usize::from(weight)] += 1;
            total += (1 << weight) >> 1;
        }
        if total == 0 {
            return Err(invalid);
        }
        let log = highbit32(total) + 1;
        if log > MAX_TABLE_LOG {
            return Err(invalid);
        }
        let rest = (1u32 << log) - total;
        if !rest.is_power_of_two() {
            return Err(invalid);
        }
        let last = highbit32(rest) + 1;
        weights[count] = last as u8;
        rank_count[last as usize] += 1;
        // A complete prefix code has an even number, at least two, of the
        // longest codes.
        if rank_count[1] < 2 || rank_count[1] & 1 != 0 {
            return Err(invalid);
        }
        // Lowest weights (longest codes) take the lowest code values, in
        // symbol order within one weight. A code of `log + 1 - weight` bits
        // fills `1 << (weight - 1)` rows of a `log`-bit table, and the
        // 12-bit table repeats each of them `1 << (12 - log)` times.
        //
        // As the reference's `HUF_readDTableX1_wksp` does, the symbols are
        // first sorted by weight, so each weight's runs, which all have one
        // length, are written in fixed-size stores. A complete code starts
        // every code at a multiple of its run length.
        let spread = LOOKUP_BITS - log;
        let mut next = [0usize; WEIGHT_SLOTS];
        let mut position = 0;
        for (slot, &count) in next.iter_mut().zip(&rank_count) {
            *slot = position;
            position += count as usize;
        }
        let mut sorted = [0u8; MAX_WEIGHTS + 1];
        for (symbol, &weight) in weights[..=count].iter().enumerate() {
            let slot = &mut next[usize::from(weight) & (WEIGHT_SLOTS - 1)];
            sorted[*slot & MAX_WEIGHTS] = symbol as u8;
            *slot += 1;
        }
        let rows = &mut *self.rows;
        let mut start = 0;
        let mut first = rank_count[0] as usize;
        for weight in 1..=log {
            let count = rank_count[weight as usize] as usize;
            let Some(symbols) = sorted.get(first..first + count) else {
                return Err(invalid);
            };
            first += count;
            let nb_bits = log + 1 - weight;
            start = match weight - 1 + spread {
                0 => fill_runs::<1>(rows, start, symbols, nb_bits),
                1 => fill_runs::<2>(rows, start, symbols, nb_bits),
                2 => fill_runs::<4>(rows, start, symbols, nb_bits),
                3 => fill_runs::<8>(rows, start, symbols, nb_bits),
                len_log => fill_long_runs(rows, start, symbols, nb_bits, 1 << len_log),
            };
        }
        debug_assert_eq!(start, TABLE_SIZE);
        self.log = log;
        Ok(())
    }

    /// Builds the pair rows from the single-symbol rows, for a tree of at
    /// most [`PAIR_BITS`]; deeper trees decode without them.
    ///
    /// A row pairs the code at the top of its [`PAIR_BITS`]-bit index with
    /// the code that follows when both fit in the lookup; decoding then
    /// reads exactly the bits of the symbols it yields, as the
    /// single-symbol table does.
    ///
    /// A code of `b` bits fills a run of `1 << (11 - b)` rows, whose `j`th
    /// row continues with the code at row `j << b`. Every code of one
    /// length is therefore followed by the same series, built once per
    /// length (the reference's `HUF_fillDTableX2Level2` per rank); each row
    /// of the run is that series' row plus the first symbol's.
    fn build_pairs(&mut self) {
        if self.log > PAIR_BITS {
            return;
        }
        let rows = &*self.rows;
        // The single rows of an 11-bit index: a code of at most 11 bits
        // fills both 12-bit rows that share its top 11 bits.
        let single = |index: usize| rows[(index & PAIR_MASK) << (LOOKUP_BITS - PAIR_BITS)];
        // The slots are written through exclusive references, as plain
        // words: only the decoders' reads are atomic loads.
        let pairs = self.pairs.get_or_insert_with(PairTable::new).slots_mut();
        // The first run of each length is built from the rows; later runs
        // of that length copy it, swapping in their own first symbol.
        let mut series = (0, 0, 0);
        let mut index = 0;
        while index < PAIR_SIZE {
            let first = single(index);
            // Rows are built for codes of 1 to 11 bits.
            let first_bits = row_bits(first).clamp(1, PAIR_BITS);
            let len = 1 << (PAIR_BITS - first_bits);
            let end = (index + len).min(PAIR_SIZE);
            let own = PairRow::new(u16::from(row_symbol(first)), first_bits as u8, 1).0;
            let (series_bits, series_start, series_own) = series;
            if first_bits == series_bits && series_start + len <= index {
                let (built, rest) = pairs.split_at_mut(index);
                let delta = own.wrapping_sub(series_own);
                for (pair, model) in rest[..end - index]
                    .iter_mut()
                    .zip(&mut built[series_start..])
                {
                    *pair.get_mut() = model.get_mut().wrapping_add(delta);
                }
            } else {
                series = (first_bits, index, own);
                for (j, pair) in pairs[index..end].iter_mut().enumerate() {
                    let second = single(j << first_bits);
                    let second_bits = row_bits(second);
                    let next = if first_bits + second_bits <= PAIR_BITS {
                        PairRow::new(u16::from(row_symbol(second)) << 8, second_bits as u8, 1)
                    } else {
                        PairRow(0)
                    };
                    *pair.get_mut() = next.0 + own;
                }
            }
            index = end;
        }
        self.pairs_valid = true;
    }

    /// Decodes `out.len()` symbols from one stream, which must be consumed
    /// exactly.
    #[inline(always)]
    fn decode_stream(&self, src: &[u8], out: &mut [u8]) -> Result<(), InvalidDataKind> {
        let bits = BitContainer::new(src).ok_or(InvalidDataKind::HuffmanStream)?;
        decode_tail(&self.rows, bits, src, out)
    }

    /// Decodes `size` literals from one stream or four streams into the
    /// start of `buffer`, with `simd`'s target features. The pair rows are
    /// built on first use. A `buffer` of at least [`FIXED_OUT`] bytes lets
    /// the pair loop run over a buffer of a fixed size; the bytes after the
    /// literals are left as they were. The four-stream loops prefetch
    /// `ahead` for writing as they go, which changes no result.
    #[fearless_simd_macros::simd]
    pub(crate) fn decode_streams<S: fearless_simd::Simd>(
        &mut self,
        simd: S,
        src: &[u8],
        buffer: &mut [u8],
        size: usize,
        four_streams: bool,
        ahead: WriteAhead,
    ) -> Result<(), InvalidDataKind> {
        decode_with(
            &mut *self,
            src,
            buffer,
            size,
            four_streams,
            lockstep(simd),
            ahead,
        )
        .or_else(|_| self.decode_as_reference(src, buffer, size, four_streams))
    }

    /// Decodes literals as [`Self::decode_streams`] with a table that may
    /// not be modified, a dictionary's: its pair rows are used when
    /// [`Self::build_shared`] built them, and are never built here.
    #[fearless_simd_macros::simd]
    pub(crate) fn decode_streams_shared<S: fearless_simd::Simd>(
        &self,
        simd: S,
        src: &[u8],
        buffer: &mut [u8],
        size: usize,
        four_streams: bool,
        ahead: WriteAhead,
    ) -> Result<(), InvalidDataKind> {
        decode_with(self, src, buffer, size, four_streams, lockstep(simd), ahead)
            .or_else(|_| self.decode_as_reference(src, buffer, size, four_streams))
    }

    /// Decodes a section the exact decoder rejected as the reference
    /// would, whose verdict then stands. Out of line, so that the hot
    /// loops' code is unchanged.
    #[cold]
    #[inline(never)]
    fn decode_as_reference(
        &self,
        src: &[u8],
        buffer: &mut [u8],
        size: usize,
        four_streams: bool,
    ) -> Result<(), InvalidDataKind> {
        let out = buffer
            .get_mut(..size)
            .ok_or(InvalidDataKind::HuffmanStream)?;
        reference::decode(self, src, out, four_streams)
    }

    /// Records the decoder the reference reads this tree for; treeless
    /// blocks keep it.
    pub(crate) const fn set_kind(&mut self, kind: TableKind) {
        self.kind = kind;
    }

    /// The decoder the reference reads this tree for.
    #[cfg(test)]
    pub(crate) const fn kind(&self) -> TableKind {
        self.kind
    }

    /// Builds the pair rows and the lockstep loop's copy of the single
    /// rows ahead of time, for a table that is then shared.
    pub(crate) fn build_shared(&mut self) {
        self.build_pairs();
        self.build_lockstep();
    }
}

/// A table's rows as one decode reaches them: a table the decoder owns
/// builds its pair rows and its lockstep copy on first use, a shared one
/// only offers what was built before it was shared. Both decode the same
/// symbols.
trait TableAccess {
    /// The table.
    fn table(&self) -> &HuffmanTable;

    /// The pair rows of the current tree, if available.
    fn pairs(&mut self) -> Option<&PairTable>;

    /// The current tree's rows for the lockstep loop, if available.
    fn lockstep(&mut self) -> Option<&LockstepRows>;
}

impl TableAccess for &mut HuffmanTable {
    #[inline(always)]
    fn table(&self) -> &HuffmanTable {
        self
    }

    #[inline(always)]
    fn pairs(&mut self) -> Option<&PairTable> {
        if !self.pairs_valid {
            self.build_pairs();
        }
        if self.pairs_valid {
            self.pairs.as_ref()
        } else {
            None
        }
    }

    #[inline(always)]
    fn lockstep(&mut self) -> Option<&LockstepRows> {
        if !self.lockstep_valid {
            self.build_lockstep();
        }
        self.lockstep.as_ref()
    }
}

impl TableAccess for &HuffmanTable {
    #[inline(always)]
    fn table(&self) -> &HuffmanTable {
        self
    }

    #[inline(always)]
    fn pairs(&mut self) -> Option<&PairTable> {
        if self.pairs_valid {
            self.pairs.as_ref()
        } else {
            None
        }
    }

    #[inline(always)]
    fn lockstep(&mut self) -> Option<&LockstepRows> {
        if self.lockstep_valid {
            self.lockstep.as_ref()
        } else {
            None
        }
    }
}

/// Whether `simd`'s features include AVX-512, with which the fixed-buffer
/// pair loop decodes the four streams in lockstep
/// ([`pair_rounds_lockstep`]). Constant for each backend, so the test
/// folds away.
#[inline(always)]
fn lockstep<S: fearless_simd::Simd>(simd: S) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    return simd.level().as_avx512().is_some();
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = simd;
        false
    }
}

/// Decodes `out.len()` literals from one stream or four streams; `lockstep`
/// as for [`decode_four`].
#[inline(always)]
fn decode_with<T: TableAccess>(
    table: T,
    src: &[u8],
    buffer: &mut [u8],
    size: usize,
    four_streams: bool,
    lockstep: bool,
    ahead: WriteAhead,
) -> Result<(), InvalidDataKind> {
    let out = buffer
        .get_mut(..size)
        .ok_or(InvalidDataKind::HuffmanStream)?;
    if !four_streams {
        return table.table().decode_stream(src, out);
    }
    let pairs = prefers_pairs(size, src.len());
    decode_four(table, src, buffer, size, pairs, lockstep, ahead)
}

/// Decodes four streams after their jump table, with the pair rows in the
/// fast loops when `pairs` is set; `lockstep` decodes a fixed-size buffer's
/// pairs with the four streams in lockstep.
#[inline(always)]
fn decode_four<T: TableAccess>(
    mut table: T,
    src: &[u8],
    buffer: &mut [u8],
    size: usize,
    pairs: bool,
    lockstep: bool,
    mut ahead: WriteAhead,
) -> Result<(), InvalidDataKind> {
    let invalid = InvalidDataKind::HuffmanStream;
    let out_len = size;
    let (jump, streams) = src.split_first_chunk::<6>().ok_or(invalid)?;
    if streams.len() < 4 || out_len < 6 || out_len > buffer.len() {
        return Err(invalid);
    }
    let size = |at: usize| usize::from(u16::from_le_bytes([jump[at], jump[at + 1]]));
    let second = size(0);
    let third = second + size(2);
    let fourth = third + size(4);
    if fourth > streams.len() {
        return Err(invalid);
    }
    let segment = out_len.div_ceil(4);
    if 3 * segment > out_len {
        return Err(invalid);
    }
    let starts = [0, second, third, fourth, streams.len()];
    let segments = [0, segment, 2 * segment, 3 * segment, out_len];
    let stream = |index: usize| &streams[starts[index]..starts[index + 1]];
    let open = |index: usize| BitContainer::new(stream(index)).ok_or(invalid);
    let mut bits = [open(0)?, open(1)?, open(2)?, open(3)?];

    // The fast loops need every stream to refill a whole word.
    let mut done = [0usize; 4];
    if bits.iter().all(BitContainer::can_reload_fast) {
        let mut sentinels: [Sentinel; 4] = ::core::array::from_fn(|index| {
            bits[index].reload_fast(stream(index));
            Sentinel::from_container(bits[index], streams, starts[index])
        });
        // Without pair rows (a tree deeper than they serve, or a shared
        // table without them), the single-symbol loop decodes instead.
        let pairs = if pairs { table.pairs() } else { None };
        let fixed = match buffer.first_chunk_mut::<FIXED_OUT>() {
            Some(fixed) if out_len <= BLOCK_SIZE_MAX => Some(fixed),
            _ => None,
        };
        let mut four = FourStreams {
            src: streams,
            starts,
            out: &mut [],
            segments,
        };
        done = if let (Some(pairs), Some(fixed)) = (pairs, fixed) {
            if lockstep {
                pair_rounds_lockstep(pairs, streams, fixed, segments, &mut sentinels, &mut ahead)
            } else {
                pair_rounds_fixed(pairs, streams, fixed, segments, &mut sentinels, &mut ahead)
            }
        } else if let Some(pairs) = pairs {
            four.out = &mut buffer[..out_len];
            pair_rounds(pairs, &mut four, &mut sentinels)
        } else {
            four.out = &mut buffer[..out_len];
            // A table the decoder owns copies its rows for the lockstep
            // loop on first use; a shared one that has no copy decodes a
            // stream's round at a time.
            let short = table.table().log <= 11;
            match (table.lockstep(), short) {
                (Some(rows), true) => {
                    single_rounds_lockstep::<5>(rows, &mut four, &mut sentinels, &mut ahead)
                        .ok_or(invalid)?
                }
                (Some(rows), false) => {
                    single_rounds_lockstep::<4>(rows, &mut four, &mut sentinels, &mut ahead)
                        .ok_or(invalid)?
                }
                (None, short) => {
                    let single = table.table();
                    if short {
                        single_rounds::<5>(&single.rows, &mut four, &mut sentinels, &mut ahead)
                    } else {
                        single_rounds::<4>(&single.rows, &mut four, &mut sentinels, &mut ahead)
                    }
                }
            }
        };
        for ((container, sentinel), &start) in bits.iter_mut().zip(sentinels).zip(&starts) {
            *container = sentinel.into_container(streams, start);
        }
    }
    let table = table.table();
    let rows = &*table.rows;
    let tail_pairs = if pairs && table.pairs_valid {
        table.pairs.as_ref()
    } else {
        None
    };
    let out = &mut buffer[..out_len];
    for (index, (container, decoded)) in bits.into_iter().zip(done).enumerate() {
        let output = &mut out[segments[index] + decoded..segments[index + 1]];
        if let Some(pairs) = tail_pairs {
            decode_pair_tail(rows, pairs, container, stream(index), output)?;
        } else {
            decode_tail(rows, container, stream(index), output)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn read(src: &[u8]) -> Result<(HuffmanTable, usize), InvalidDataKind> {
        let mut table = HuffmanTable::default();
        let used = table.read(src)?;
        Ok((table, used))
    }

    /// The RFC 8878 example tree: A B C D E F with weights 4 3 2 0 1 1,
    /// as symbols 0..=5, described directly.
    pub(super) fn example_table() -> HuffmanTable {
        // Five listed weights (F is implied): header 127 + 5, then 4|3, 2|0, 1|_.
        let src = [132, 0x43, 0x20, 0x10];
        let (table, used) = read(&src).expect("valid tree");
        assert_eq!(used, 4);
        table
    }

    #[test]
    fn the_rfc_example_tree_assigns_canonical_codes() {
        let table = example_table();
        assert_eq!(table.log, 4);
        // E = 0000, F = 0001, C = 001x, B = 01xx, A = 1xxx.
        assert_eq!(table.rows[0 << (LOOKUP_BITS - table.log)], row(4, 4));
        assert_eq!(table.rows[1 << (LOOKUP_BITS - table.log)], row(5, 4));
        assert_eq!(table.rows[2 << (LOOKUP_BITS - table.log)], row(2, 3));
        assert_eq!(table.rows[4 << (LOOKUP_BITS - table.log)], row(1, 2));
        assert_eq!(table.rows[8 << (LOOKUP_BITS - table.log)], row(0, 1));
    }

    #[test]
    fn the_rfc_example_stream_decodes_with_canonical_codes() {
        let table = example_table();
        let mut out = [0u8; 4];
        table
            .decode_stream(&[0b0001_0000, 0b0000_1101], &mut out)
            .expect("valid stream");
        // The RFC labels the last two codes of its stream example the other
        // way round from its own code table; with E = 0000 and F = 0001, as
        // the reference decoder assigns them, the stream reads A B F E.
        assert_eq!(out, [0, 1, 5, 4]);
    }

    #[test]
    fn streams_must_be_consumed_exactly() {
        let table = example_table();
        let stream = [0b0001_0000, 0b0000_1101];
        let invalid = Err(InvalidDataKind::HuffmanStream);
        assert_eq!(table.decode_stream(&stream, &mut [0u8; 3]), invalid);
        assert_eq!(table.decode_stream(&stream, &mut [0u8; 5]), invalid);
        assert_eq!(table.decode_stream(&[0x10, 0x00], &mut [0u8; 1]), invalid);
    }

    #[test]
    fn four_streams_split_the_output_by_the_jump_table() {
        let mut table = example_table();
        // Each one-byte stream holds the marker and three bits: A = 1, then
        // B = 01, read from the top.
        let mut src = alloc::vec![1, 0, 1, 0, 1, 0];
        src.extend_from_slice(&[0b0000_1101; 4]);
        let mut out = [0u8; 8];
        table
            .decode_streams(
                fearless_simd::Fallback::new(),
                &src,
                &mut out,
                8,
                true,
                WriteAhead::NONE,
            )
            .expect("valid");
        assert_eq!(out, [0, 1, 0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn malformed_jump_tables_are_rejected() {
        let mut table = example_table();
        let mut out = [0u8; 8];
        let invalid = Err(InvalidDataKind::HuffmanStream);
        // No room for the jump table, a first stream past the end, fewer
        // than four stream bytes, and fewer than six literals.
        assert_eq!(
            table.decode_streams(
                fearless_simd::Fallback::new(),
                &[1, 0, 1, 0, 1],
                &mut out,
                8,
                true,
                WriteAhead::NONE
            ),
            invalid
        );
        let oversized = [9, 0, 1, 0, 1, 0, 1, 1, 1, 1];
        assert_eq!(
            table.decode_streams(
                fearless_simd::Fallback::new(),
                &oversized,
                &mut out,
                8,
                true,
                WriteAhead::NONE
            ),
            invalid
        );
        let short = [1, 0, 1, 0, 1, 0, 1, 1, 1];
        assert_eq!(
            table.decode_streams(
                fearless_simd::Fallback::new(),
                &short,
                &mut out,
                8,
                true,
                WriteAhead::NONE
            ),
            invalid
        );
        let streams = [1, 0, 1, 0, 1, 0, 1, 1, 1, 1];
        assert_eq!(
            table.decode_streams(
                fearless_simd::Fallback::new(),
                &streams,
                &mut [0u8; 5],
                5,
                true,
                WriteAhead::NONE
            ),
            invalid
        );
        // More literals than the buffer holds.
        assert_eq!(
            table.decode_streams(
                fearless_simd::Fallback::new(),
                &streams,
                &mut out,
                9,
                true,
                WriteAhead::NONE
            ),
            invalid
        );
    }

    #[test]
    fn invalid_weight_sets_are_rejected() {
        let invalid = Err(InvalidDataKind::HuffmanTable);
        // Empty description, truncated direct weights, truncated FSE weights.
        assert_eq!(read(&[]).map(|_| ()), invalid);
        assert_eq!(read(&[132, 0x43]).map(|_| ()), invalid);
        assert_eq!(read(&[5, 0, 0]).map(|_| ()), invalid);
        // All weights zero, and a weight above the maximum.
        assert_eq!(read(&[129, 0x00]).map(|_| ()), invalid);
        assert_eq!(read(&[129, 0xd0]).map(|_| ()), invalid);
        // Weights 2 and 2: total 4, so the implied weight is 3 and no
        // weight-1 codes exist.
        assert_eq!(read(&[129, 0x22]).map(|_| ()), invalid);
        // Weights 1, 1, 1 imply a fourth weight 1: valid. Weights 3 and 1
        // leave 3, which is not a power of two.
        assert!(read(&[130, 0x11, 0x10]).is_ok());
        assert_eq!(read(&[129, 0x31]).map(|_| ()), invalid);
        // Two weight-12 codes make a 13-bit tree.
        assert_eq!(read(&[129, 0xcc]).map(|_| ()), invalid);
    }

    #[test]
    fn twelve_bit_trees_are_accepted_like_the_reference() {
        // Weights 12, 11, ..., 1 and an implied 1: a complete 12-bit tree.
        let src = [139, 0xcb, 0xa9, 0x87, 0x65, 0x43, 0x21];
        let (table, used) = read(&src).expect("valid tree");
        assert_eq!(used, src.len());
        assert_eq!(table.log, MAX_TABLE_LOG);
        assert_eq!(table.rows.len(), 1 << 12);
    }

    /// Writes Huffman bitstreams the way the encoder does: codes appended
    /// low bits first, the last symbol written is the first one read.
    #[derive(Default)]
    struct StreamWriter {
        bytes: alloc::vec::Vec<u8>,
        acc: u64,
        filled: u32,
    }

    impl StreamWriter {
        fn put(&mut self, value: u64, width: u32) {
            self.acc |= value << self.filled;
            self.filled += width;
            while self.filled >= 8 {
                self.bytes.push(self.acc as u8);
                self.acc >>= 8;
                self.filled -= 8;
            }
        }

        /// Ends the stream with its marker bit and padding.
        fn finish(mut self) -> alloc::vec::Vec<u8> {
            self.put(1, 1);
            if self.filled > 0 {
                self.bytes.push(self.acc as u8);
            }
            self.bytes
        }
    }

    /// Each symbol's code and length, read back from the table's rows.
    fn codes(table: &HuffmanTable) -> [(u64, u32); 256] {
        let mut codes = [(0, 0); 256];
        for (index, &row) in table.rows.iter().enumerate().rev() {
            let length = row_bits(row);
            codes[usize::from(row_symbol(row))] =
                ((index >> (LOOKUP_BITS - length)) as u64, length);
        }
        codes
    }

    /// One stream holding `symbols`, decoded first to last.
    pub(super) fn encode(table: &HuffmanTable, symbols: &[u8]) -> alloc::vec::Vec<u8> {
        let codes = codes(table);
        let mut writer = StreamWriter::default();
        for &symbol in symbols.iter().rev() {
            let (code, length) = codes[usize::from(symbol)];
            writer.put(code, length);
        }
        writer.finish()
    }

    /// A jump table and four streams for `symbols`.
    pub(super) fn encode_four(table: &HuffmanTable, symbols: &[u8]) -> alloc::vec::Vec<u8> {
        // Three segments of a quarter rounded up; the fourth takes the rest.
        let segment = symbols.len().div_ceil(4);
        let streams: alloc::vec::Vec<_> = (0..4)
            .map(|index| {
                let start = (index * segment).min(symbols.len());
                let end = if index == 3 {
                    symbols.len()
                } else {
                    (start + segment).min(symbols.len())
                };
                encode(table, &symbols[start..end])
            })
            .collect();
        let mut out = alloc::vec::Vec::new();
        for stream in &streams[..3] {
            out.extend_from_slice(&(stream.len() as u16).to_le_bytes());
        }
        for stream in &streams {
            out.extend_from_slice(stream);
        }
        out
    }

    /// Symbols the table can code, in a skewed pseudo-random mix.
    pub(super) fn message(table: &HuffmanTable, len: usize, seed: u64) -> alloc::vec::Vec<u8> {
        let codes = codes(table);
        let alphabet: alloc::vec::Vec<u8> = (0..=255u8)
            .filter(|&s| codes[usize::from(s)].1 > 0)
            .collect();
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let pick = (state >> 33) as usize;
                // Favour the first symbols, which have the shortest codes.
                alphabet[(pick % alphabet.len()).min(pick % 3)]
            })
            .collect()
    }

    /// The weights 12, 11, ..., 1 and an implied 1: a complete 12-bit tree.
    pub(super) fn twelve_bit_table() -> HuffmanTable {
        read(&[139, 0xcb, 0xa9, 0x87, 0x65, 0x43, 0x21])
            .expect("valid")
            .0
    }

    /// Eight-bit codes for all 256 symbols, as direct weights of 1.
    pub(super) fn flat_table() -> HuffmanTable {
        let mut src = alloc::vec![128 + 127];
        src.extend_from_slice(&[0x11; 64]);
        read(&src).expect("valid").0
    }

    #[test]
    fn four_streams_decode_alike_on_every_path() {
        for mut table in [example_table(), twelve_bit_table(), flat_table()] {
            for len in [6, 7, 64, 333, 4_000, 20_000] {
                let symbols = message(&table, len, len as u64);
                let src = encode_four(&table, &symbols);
                for (pairs, lockstep) in [(false, false), (true, false), (true, true)] {
                    let mut out = alloc::vec![0xee; len];
                    decode_four(
                        &mut table,
                        &src,
                        &mut out,
                        len,
                        pairs,
                        lockstep,
                        WriteAhead::NONE,
                    )
                    .expect("valid");
                    assert_eq!(out, symbols, "log {} len {len} pairs {pairs}", table.log);
                    // A buffer of the fixed size decodes alike and keeps the
                    // bytes after the literals, in lockstep or three and one.
                    let mut fixed = alloc::vec![0xee; FIXED_OUT + 3];
                    decode_four(
                        &mut table,
                        &src,
                        &mut fixed,
                        len,
                        pairs,
                        lockstep,
                        WriteAhead::NONE,
                    )
                    .expect("valid");
                    assert_eq!(
                        fixed[..len],
                        symbols,
                        "fixed len {len} pairs {pairs} lockstep {lockstep}"
                    );
                    assert!(fixed[len..].iter().all(|&byte| byte == 0xee));
                }
                // A shared table decodes alike, with or without pair rows.
                let mut shared = table.clone();
                shared.pairs_valid = false;
                for prebuilt in [false, true] {
                    if prebuilt {
                        shared.build_shared();
                    }
                    for four in [true, false] {
                        let src = if four {
                            src.clone()
                        } else {
                            encode(&table, &symbols)
                        };
                        for room in [len, FIXED_OUT] {
                            let mut out = alloc::vec![0xee; room];
                            shared
                                .decode_streams_shared(
                                    fearless_simd::Fallback::new(),
                                    &src,
                                    &mut out,
                                    len,
                                    four,
                                    WriteAhead::NONE,
                                )
                                .expect("valid");
                            assert_eq!(out[..len], symbols, "prebuilt {prebuilt} four {four}");
                        }
                    }
                }
                let mut out = alloc::vec![0xee; len];
                table
                    .decode_streams(
                        fearless_simd::Fallback::new(),
                        &src,
                        &mut out,
                        len,
                        true,
                        WriteAhead::NONE,
                    )
                    .expect("valid");
                assert_eq!(out, symbols);
                let single = encode(&table, &symbols);
                let mut out = alloc::vec![0xee; len];
                table
                    .decode_streams(
                        fearless_simd::Fallback::new(),
                        &single,
                        &mut out,
                        len,
                        false,
                        WriteAhead::NONE,
                    )
                    .expect("valid");
                assert_eq!(out, symbols);
            }
        }
    }

    #[test]
    fn four_streams_decode_alike_on_every_backend() {
        for backend in crate::Backend::available() {
            let lockstep = fearless_simd::dispatch!(backend.0, simd => lockstep(simd));
            // AVX-512 decodes in lockstep; other levels may too, where the
            // build's baseline already has AVX-512.
            assert!(lockstep || backend.name() != "avx512");
            for mut table in [example_table(), twelve_bit_table(), flat_table()] {
                for len in [6, 333, 20_000] {
                    let symbols = message(&table, len, len as u64);
                    let src = encode_four(&table, &symbols);
                    let mut fixed = alloc::vec![0xee; FIXED_OUT + 3];
                    fearless_simd::dispatch!(backend.0, simd => table.decode_streams(
                        simd, &src, &mut fixed, len, true,
                        WriteAhead::NONE,
                    ))
                    .expect("valid");
                    assert_eq!(fixed[..len], symbols, "{backend} log {}", table.log);
                    assert!(fixed[len..].iter().all(|&byte| byte == 0xee));
                    table.build_shared();
                    let mut shared = alloc::vec![0xee; FIXED_OUT];
                    fearless_simd::dispatch!(backend.0, simd => table.decode_streams_shared(
                        simd, &src, &mut shared, len, true,
                        WriteAhead::NONE,
                    ))
                    .expect("valid");
                    assert_eq!(shared[..len], symbols, "{backend} shared");
                }
            }
        }
    }

    #[test]
    fn fast_paths_reject_streams_with_bits_left_or_missing() {
        let mut table = flat_table();
        let symbols = message(&table, 20_000, 3);
        let src = encode_four(&table, &symbols);
        for (pairs, lockstep) in [(false, false), (true, false), (true, true)] {
            // One symbol fewer leaves bits in every stream, one more reads
            // past the start of the last one.
            for len in [symbols.len() - 1, symbols.len() + 1] {
                for room in [len, FIXED_OUT] {
                    let mut out = alloc::vec![0; room];
                    assert_eq!(
                        decode_four(
                            &mut table,
                            &src,
                            &mut out,
                            len,
                            pairs,
                            lockstep,
                            WriteAhead::NONE
                        ),
                        Err(InvalidDataKind::HuffmanStream),
                        "pairs {pairs} lockstep {lockstep} len {len} room {room}"
                    );
                }
            }
        }
    }

    #[test]
    fn pair_tails_decode_exactly_at_output_boundaries() {
        for mut table in [example_table(), flat_table()] {
            table.build_pairs();
            let pairs = table.pairs.as_ref().expect("pair rows");
            for len in [0, 1, 2, 3, 7, 10, 11, 12, 15, 16, 17, 31, 32, 1000] {
                let symbols = message(&table, len, len as u64 + 1);
                let source = encode(&table, &symbols);
                let bits = BitContainer::new(&source).expect("terminated");
                let mut out = alloc::vec![0; len];
                decode_pair_tail(&table.rows, pairs, bits, &source, &mut out).expect("valid");
                assert_eq!(out, symbols, "log {} len {len}", table.log);
                if len > 0 {
                    let mut short = alloc::vec![0; len - 1];
                    assert_eq!(
                        decode_pair_tail(&table.rows, pairs, bits, &source, &mut short),
                        Err(InvalidDataKind::HuffmanStream)
                    );
                }
                let mut long = alloc::vec![0; len + 1];
                assert_eq!(
                    decode_pair_tail(&table.rows, pairs, bits, &source, &mut long),
                    Err(InvalidDataKind::HuffmanStream)
                );
            }
        }
    }

    #[test]
    fn lockstep_rows_copy_the_single_rows() {
        for mut table in [example_table(), twelve_bit_table(), flat_table()] {
            assert!(table.lockstep.is_none() && !table.lockstep_valid);
            let before = table.retained_bytes();
            table.build_lockstep();
            assert!(table.lockstep_valid);
            assert_eq!(table.retained_bytes(), before + 2 * TABLE_SIZE);
            let copy = table.lockstep.as_ref().expect("built");
            assert!(copy.rows().eq(table.rows.iter().copied()));
            for index in [0, 1, TABLE_SIZE / 2, TABLE_SIZE - 1] {
                assert_eq!(copy.row(index), table.rows[index]);
                // An index past the table is masked into it.
                assert_eq!(copy.row(index + TABLE_SIZE), table.rows[index]);
            }
            // Copies compare and clone by their rows.
            let clone = copy.clone();
            assert_eq!(&clone, copy);
            assert_ne!(clone, LockstepRows::new());
            assert_eq!(table.clone(), table);
            // An owned table hands the copy out, a new tree invalidates
            // it, and a shared table offers it only once it is built.
            let mut owned = &mut table;
            assert!(owned.lockstep().is_some());
            let src = [132, 0x43, 0x20, 0x10];
            table.read(&src).expect("valid");
            assert!(!table.lockstep_valid);
            let mut shared = &table;
            assert!(shared.lockstep().is_none());
            table.build_shared();
            let mut shared = &table;
            assert!(shared.lockstep().is_some() && shared.pairs().is_some());
        }
    }

    #[test]
    fn pair_tables_compare_and_clone_by_their_rows() {
        let mut table = example_table();
        table.build_pairs();
        let pairs = table.pairs.as_ref().expect("built");
        let clone = pairs.clone();
        assert_eq!(&clone, pairs);
        assert!(clone.rows().eq(pairs.rows()));
        assert_ne!(clone, PairTable::new());
        // An index past the table is masked into it.
        assert_eq!(pairs.row(5 + PAIR_SIZE), pairs.row(5));
    }

    #[test]
    fn a_row_without_bits_ends_the_lockstep_loop_as_invalid() {
        let table = flat_table();
        let symbols = message(&table, 4_000, 7);
        let src = encode_four(&table, &symbols);
        let (jump, streams) = src.split_first_chunk::<6>().expect("jump table");
        let size = |at: usize| usize::from(u16::from_le_bytes([jump[at], jump[at + 1]]));
        let starts = [
            0,
            size(0),
            size(0) + size(2),
            size(0) + size(2) + size(4),
            streams.len(),
        ];
        let sentinels = || -> [Sentinel; 4] {
            core::array::from_fn(|index| {
                let stream = &streams[starts[index]..starts[index + 1]];
                let mut bits = BitContainer::new(stream).expect("stream");
                bits.reload_fast(stream);
                Sentinel::from_container(bits, streams, starts[index])
            })
        };
        let segment = symbols.len().div_ceil(4);
        let segments = [0, segment, 2 * segment, 3 * segment, symbols.len()];
        let run = |rows: &LockstepRows| {
            let mut out = alloc::vec![0xeeu8; symbols.len()];
            let mut ahead = WriteAhead::NONE;
            let mut four = FourStreams {
                src: streams,
                starts,
                out: &mut out,
                segments,
            };
            let done = single_rounds_lockstep::<5>(rows, &mut four, &mut sentinels(), &mut ahead);
            (done, out)
        };
        // The tree's own rows decode whole rounds of every stream.
        let mut rows = LockstepRows::new();
        rows.copy_from(&table.rows);
        let (done, out) = run(&rows);
        let done = done.expect("valid rows");
        assert!(done[0] >= 5 && done.iter().all(|&count| count == done[0]));
        for (index, &count) in done.iter().enumerate() {
            let at = segments[index];
            assert!(
                out[at..at + count] == symbols[at..at + count],
                "stream {index}"
            );
        }
        // Rows without bits belong to no tree.
        assert_eq!(run(&LockstepRows::new()).0, None);
    }

    #[test]
    fn lockstep_loops_stop_at_a_missing_window_refill_or_row() {
        let mut table = flat_table();
        let symbols = message(&table, 4_000, 11);
        let src = encode_four(&table, &symbols);
        let (jump, streams) = src.split_first_chunk::<6>().expect("jump table");
        let size = |at: usize| usize::from(u16::from_le_bytes([jump[at], jump[at + 1]]));
        let starts = [
            0,
            size(0),
            size(0) + size(2),
            size(0) + size(2) + size(4),
            streams.len(),
        ];
        let sentinels = || -> [Sentinel; 4] {
            core::array::from_fn(|index| {
                let stream = &streams[starts[index]..starts[index + 1]];
                let mut bits = BitContainer::new(stream).expect("stream");
                bits.reload_fast(stream);
                Sentinel::from_container(bits, streams, starts[index])
            })
        };
        let segment = symbols.len().div_ceil(4);
        let segments = [0, segment, 2 * segment, 3 * segment, symbols.len()];
        let single = |rows: &LockstepRows, src: &[u8], room: usize| {
            let mut out = alloc::vec![0xeeu8; room];
            let mut ahead = WriteAhead::NONE;
            let mut four = FourStreams {
                src,
                starts,
                out: &mut out,
                segments,
            };
            let done = single_rounds_lockstep::<5>(rows, &mut four, &mut sentinels(), &mut ahead);
            (done, out)
        };
        let mut rows = LockstepRows::new();
        rows.copy_from(&table.rows);

        // An output shorter than its segments has no windows, and a source
        // that ends before the streams' positions refills none of them:
        // nothing decodes, and neither is an error of the loop's.
        for (src, room) in [(streams, symbols.len() - 1), (&streams[..4], symbols.len())] {
            let (done, out) = single(&rows, src, room);
            assert_eq!(done, Some([0; 4]), "room {room}");
            assert!(out.iter().all(|&byte| byte == 0xee));
        }

        // A row without bits under the fourth stream alone is caught by
        // the second half of a step.
        let index = |stream: usize| (sentinels()[stream].into_parts().0 >> LOOKUP_SHIFT) as usize;
        assert_ne!(index(1), index(3));
        let mut broken = table.rows.clone();
        broken[index(3)] = 0;
        rows.copy_from(&broken);
        assert_eq!(single(&rows, streams, symbols.len()).0, None);

        // The pair loop looks up before it refills: on the cut source it
        // decodes one round from the containers it was given and stops.
        table.build_pairs();
        let pairs = table.pairs.as_ref().expect("pair rows");
        let mut out = alloc::vec![0xeeu8; FIXED_OUT];
        let fixed = out.first_chunk_mut::<FIXED_OUT>().expect("fixed size");
        let mut ahead = WriteAhead::NONE;
        let done = pair_rounds_lockstep(
            pairs,
            &streams[..4],
            fixed,
            segments,
            &mut sentinels(),
            &mut ahead,
        );
        for (stream, &count) in done.iter().enumerate() {
            let at = segments[stream];
            assert!((PAIR_LOOKUPS..=PAIR_ROUND_BYTES).contains(&count));
            assert!(
                out[at..at + count] == symbols[at..at + count],
                "stream {stream}"
            );
        }

        // Empty slots read as rows without bits.
        assert_eq!(PairSlot::new().get(), PairRow(0));
        assert_eq!(RowSlot::new().get(), 0);
    }

    #[test]
    fn pair_rows_hold_two_symbols_when_both_codes_fit() {
        let mut table = example_table();
        table.build_pairs();
        let pairs = table.pairs.as_ref().expect("built");
        // A = 1, then A again: two symbols in two bits.
        let both = pairs.row(0b11 << (PAIR_BITS - 2));
        assert_eq!(both, PairRow::new(0, 2, 2));
        assert_eq!((both.symbols(), both.nb_bits(), both.count()), (0, 2, 2));
        // E = 0000, then A = 1: E in the low byte, A in the high one.
        let two = pairs.row(0b1 << (PAIR_BITS - 5));
        assert_eq!((two.symbols(), two.nb_bits(), two.count()), (4, 5, 2));
        assert!(table.pairs_valid);
        assert_eq!(table.retained_bytes(), 2 * TABLE_SIZE + 4 * PAIR_SIZE);
        // A new tree invalidates them.
        let src = [132, 0x43, 0x20, 0x10];
        table.read(&src).expect("valid");
        assert!(!table.pairs_valid);
    }

    /// The pair row of the 11-bit `index` by its definition, from the
    /// single rows.
    fn naive_pair(rows: &[Row; TABLE_SIZE], index: usize) -> PairRow {
        let first = rows[index << 1];
        let first_bits = row_bits(first);
        let second = rows[((index << first_bits) & PAIR_MASK) << 1];
        let second_bits = row_bits(second);
        let (first, second) = (u16::from(row_symbol(first)), u16::from(row_symbol(second)));
        if first_bits + second_bits <= PAIR_BITS {
            PairRow::new(first | second << 8, (first_bits + second_bits) as u8, 2)
        } else {
            PairRow::new(first, first_bits as u8, 1)
        }
    }

    #[test]
    fn pair_rows_follow_their_definition_for_every_tree_depth() {
        // Direct weights 1..=n for n symbols plus the implied one give a
        // tree of depth n + 1; uniform and lopsided trees cover the rest.
        let mut trees: alloc::vec::Vec<alloc::vec::Vec<u8>> =
            (1..=11).map(|n: u8| (1..=n).rev().collect()).collect();
        trees.push(alloc::vec![1; 127]);
        trees.push([alloc::vec![7, 6, 5], alloc::vec![1; 30]].concat());
        trees.push([alloc::vec![9], alloc::vec![2; 40], alloc::vec![1; 30]].concat());
        let mut checked = 0;
        for weights in trees {
            let mut src = alloc::vec![127 + weights.len() as u8];
            for pair in weights.chunks(2) {
                src.push(pair[0] << 4 | pair.get(1).copied().unwrap_or(0));
            }
            let Ok((mut table, _)) = read(&src) else {
                continue;
            };
            table.build_pairs();
            if table.log > PAIR_BITS {
                // Twelve-bit trees decode without pair rows.
                assert!(!table.pairs_valid && table.pairs.is_none());
                continue;
            }
            let pairs = table.pairs.as_ref().expect("built");
            for (index, pair) in pairs.rows().enumerate() {
                assert_eq!(
                    pair,
                    naive_pair(&table.rows, index),
                    "log {} index {index}",
                    table.log
                );
            }
            checked += 1;
        }
        assert!(checked >= 11, "{checked} trees");
    }

    #[test]
    fn the_timing_model_prefers_pairs_for_well_compressed_large_sections() {
        assert!(prefers_pairs(100_000, 30_000));
        assert!(!prefers_pairs(100, 99));
        assert!(!prefers_pairs(1_000, 2_000));
    }
}

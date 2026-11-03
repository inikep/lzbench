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

use alloc::boxed::Box;

use super::bits::{BitContainer, Sentinel};
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
    pairs: Option<Box<[PairRow; PAIR_SIZE]>>,
    /// Whether `pairs` describes the current tree.
    pairs_valid: bool,
}

impl Default for HuffmanTable {
    fn default() -> Self {
        Self {
            log: 0,
            rows: Box::new([0; TABLE_SIZE]),
            pairs: None,
            pairs_valid: false,
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
    pairs: &[PairRow; PAIR_SIZE],
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
            let pair = pairs[bits.look_nonzero(PAIR_BITS) as usize & PAIR_MASK];
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
#[inline(always)]
fn single_rounds<const K: usize>(
    rows: &[Row; TABLE_SIZE],
    streams: &mut FourStreams<'_>,
    sentinels: &mut [Sentinel; 4],
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
                continue;
            };
            for count in &mut decoded[..stopped] {
                *count = K;
            }
            break 'rounds;
        }
    }
    *sentinels = s;
    decoded.map(|extra| done + extra)
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
    pairs: &[PairRow; PAIR_SIZE],
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
                let p1 = pairs[s1.peek(PAIR_SHIFT)];
                let p2 = pairs[s2.peek(PAIR_SHIFT)];
                let p3 = pairs[s3.peek(PAIR_SHIFT)];
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
                let p4 = pairs[s4.peek(PAIR_SHIFT)];
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
    pairs: &[PairRow; PAIR_SIZE],
    sentinel: &mut Sentinel,
    out: &mut [u8; FIXED_OUT],
    op: &mut usize,
) {
    let pair = pairs[sentinel.peek(PAIR_SHIFT)];
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
/// [`pair_rounds`] does.
#[inline(always)]
fn pair_rounds_fixed(
    pairs: &[PairRow; PAIR_SIZE],
    src: &[u8],
    out: &mut [u8; FIXED_OUT],
    segments: [usize; 5],
    sentinels: &mut [Sentinel; 4],
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
            for rows in run.as_chunks_mut::<16>().0 {
                *rows = chunk;
            }
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
                .map_or(0, |_| size_of::<[PairRow; PAIR_SIZE]>())
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
        let pairs = self
            .pairs
            .get_or_insert_with(|| Box::new([PairRow::default(); PAIR_SIZE]));
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
                for (pair, model) in rest[..end - index].iter_mut().zip(&built[series_start..]) {
                    *pair = PairRow(model.0.wrapping_add(delta));
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
                    *pair = PairRow(next.0 + own);
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
    /// literals are left as they were.
    #[fearless_simd_macros::simd]
    pub(crate) fn decode_streams<S: fearless_simd::Simd>(
        &mut self,
        simd: S,
        src: &[u8],
        buffer: &mut [u8],
        size: usize,
        four_streams: bool,
    ) -> Result<(), InvalidDataKind> {
        decode_with(self, src, buffer, size, four_streams)
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
    ) -> Result<(), InvalidDataKind> {
        decode_with(self, src, buffer, size, four_streams)
    }

    /// Builds the pair rows ahead of time, for a table that is then shared.
    pub(crate) fn build_shared(&mut self) {
        self.build_pairs();
    }
}

/// A table's rows as one decode reaches them: a table the decoder owns
/// builds its pair rows on first use, a shared one only offers rows built
/// before it was shared. Both decode the same symbols.
trait TableAccess {
    /// The table.
    fn table(&self) -> &HuffmanTable;

    /// The pair rows of the current tree, if available.
    fn pairs(&mut self) -> Option<&[PairRow; PAIR_SIZE]>;
}

impl TableAccess for &mut HuffmanTable {
    #[inline(always)]
    fn table(&self) -> &HuffmanTable {
        self
    }

    #[inline(always)]
    fn pairs(&mut self) -> Option<&[PairRow; PAIR_SIZE]> {
        if !self.pairs_valid {
            self.build_pairs();
        }
        if self.pairs_valid {
            self.pairs.as_deref()
        } else {
            None
        }
    }
}

impl TableAccess for &HuffmanTable {
    #[inline(always)]
    fn table(&self) -> &HuffmanTable {
        self
    }

    #[inline(always)]
    fn pairs(&mut self) -> Option<&[PairRow; PAIR_SIZE]> {
        if self.pairs_valid {
            self.pairs.as_deref()
        } else {
            None
        }
    }
}

/// Decodes `out.len()` literals from one stream or four streams.
#[inline(always)]
fn decode_with<T: TableAccess>(
    table: T,
    src: &[u8],
    buffer: &mut [u8],
    size: usize,
    four_streams: bool,
) -> Result<(), InvalidDataKind> {
    let out = buffer
        .get_mut(..size)
        .ok_or(InvalidDataKind::HuffmanStream)?;
    if !four_streams {
        return table.table().decode_stream(src, out);
    }
    let pairs = prefers_pairs(size, src.len());
    decode_four(table, src, buffer, size, pairs)
}

/// Decodes four streams after their jump table, with the pair rows in the
/// fast loops when `pairs` is set.
#[inline(always)]
fn decode_four<T: TableAccess>(
    mut table: T,
    src: &[u8],
    buffer: &mut [u8],
    size: usize,
    pairs: bool,
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
            pair_rounds_fixed(pairs, streams, fixed, segments, &mut sentinels)
        } else if let Some(pairs) = pairs {
            four.out = &mut buffer[..out_len];
            pair_rounds(pairs, &mut four, &mut sentinels)
        } else {
            four.out = &mut buffer[..out_len];
            let single = table.table();
            if single.log <= 11 {
                single_rounds::<5>(&single.rows, &mut four, &mut sentinels)
            } else {
                single_rounds::<4>(&single.rows, &mut four, &mut sentinels)
            }
        };
        for ((container, sentinel), &start) in bits.iter_mut().zip(sentinels).zip(&starts) {
            *container = sentinel.into_container(streams, start);
        }
    }
    let table = table.table();
    let rows = &*table.rows;
    let tail_pairs = if pairs && table.pairs_valid {
        table.pairs.as_deref()
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

    fn read(src: &[u8]) -> Result<(HuffmanTable, usize), InvalidDataKind> {
        let mut table = HuffmanTable::default();
        let used = table.read(src)?;
        Ok((table, used))
    }

    /// The RFC 8878 example tree: A B C D E F with weights 4 3 2 0 1 1,
    /// as symbols 0..=5, described directly.
    fn example_table() -> HuffmanTable {
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
            .decode_streams(fearless_simd::Fallback::new(), &src, &mut out, 8, true)
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
                true
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
                true
            ),
            invalid
        );
        let short = [1, 0, 1, 0, 1, 0, 1, 1, 1];
        assert_eq!(
            table.decode_streams(fearless_simd::Fallback::new(), &short, &mut out, 8, true),
            invalid
        );
        let streams = [1, 0, 1, 0, 1, 0, 1, 1, 1, 1];
        assert_eq!(
            table.decode_streams(
                fearless_simd::Fallback::new(),
                &streams,
                &mut [0u8; 5],
                5,
                true
            ),
            invalid
        );
        // More literals than the buffer holds.
        assert_eq!(
            table.decode_streams(fearless_simd::Fallback::new(), &streams, &mut out, 9, true),
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
    fn encode(table: &HuffmanTable, symbols: &[u8]) -> alloc::vec::Vec<u8> {
        let codes = codes(table);
        let mut writer = StreamWriter::default();
        for &symbol in symbols.iter().rev() {
            let (code, length) = codes[usize::from(symbol)];
            writer.put(code, length);
        }
        writer.finish()
    }

    /// A jump table and four streams for `symbols`.
    fn encode_four(table: &HuffmanTable, symbols: &[u8]) -> alloc::vec::Vec<u8> {
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
    fn message(table: &HuffmanTable, len: usize, seed: u64) -> alloc::vec::Vec<u8> {
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
    fn twelve_bit_table() -> HuffmanTable {
        read(&[139, 0xcb, 0xa9, 0x87, 0x65, 0x43, 0x21])
            .expect("valid")
            .0
    }

    /// Eight-bit codes for all 256 symbols, as direct weights of 1.
    fn flat_table() -> HuffmanTable {
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
                for pairs in [false, true] {
                    let mut out = alloc::vec![0xee; len];
                    decode_four(&mut table, &src, &mut out, len, pairs).expect("valid");
                    assert_eq!(out, symbols, "log {} len {len} pairs {pairs}", table.log);
                    // A buffer of the fixed size decodes alike and keeps the
                    // bytes after the literals.
                    let mut fixed = alloc::vec![0xee; FIXED_OUT + 3];
                    decode_four(&mut table, &src, &mut fixed, len, pairs).expect("valid");
                    assert_eq!(fixed[..len], symbols, "fixed len {len} pairs {pairs}");
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
                                )
                                .expect("valid");
                            assert_eq!(out[..len], symbols, "prebuilt {prebuilt} four {four}");
                        }
                    }
                }
                let mut out = alloc::vec![0xee; len];
                table
                    .decode_streams(fearless_simd::Fallback::new(), &src, &mut out, len, true)
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
                    )
                    .expect("valid");
                assert_eq!(out, symbols);
            }
        }
    }

    #[test]
    fn fast_paths_reject_streams_with_bits_left_or_missing() {
        let mut table = flat_table();
        let symbols = message(&table, 20_000, 3);
        let src = encode_four(&table, &symbols);
        for pairs in [false, true] {
            // One symbol fewer leaves bits in every stream, one more reads
            // past the start of the last one.
            for len in [symbols.len() - 1, symbols.len() + 1] {
                for room in [len, FIXED_OUT] {
                    let mut out = alloc::vec![0; room];
                    assert_eq!(
                        decode_four(&mut table, &src, &mut out, len, pairs),
                        Err(InvalidDataKind::HuffmanStream),
                        "pairs {pairs} len {len} room {room}"
                    );
                }
            }
        }
    }

    #[test]
    fn pair_tails_decode_exactly_at_output_boundaries() {
        for mut table in [example_table(), flat_table()] {
            table.build_pairs();
            let pairs = table.pairs.as_deref().expect("pair rows");
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
    fn pair_rows_hold_two_symbols_when_both_codes_fit() {
        let mut table = example_table();
        table.build_pairs();
        let pairs = table.pairs.as_deref().expect("built");
        // A = 1, then A again: two symbols in two bits.
        let both = pairs[0b11 << (PAIR_BITS - 2)];
        assert_eq!(both, PairRow::new(0, 2, 2));
        assert_eq!((both.symbols(), both.nb_bits(), both.count()), (0, 2, 2));
        // E = 0000, then A = 1: E in the low byte, A in the high one.
        let two = pairs[0b1 << (PAIR_BITS - 5)];
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
            let pairs = table.pairs.as_deref().expect("built");
            for (index, &pair) in pairs.iter().enumerate() {
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

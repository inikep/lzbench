//! The reference decoder's verdict on literal streams that are not consumed
//! exactly: a port of the stream decoders of zstd 1.5.7's
//! `huf_decompress.c`, run only after the exact decoder has rejected a
//! section.
//!
//! RFC 8878 requires each Huffman stream to end exactly at its first bit,
//! and [`super::HuffmanTable::decode_streams`] checks that. The reference
//! is laxer in three places:
//!
//! - its fast four-stream path (`HUF_decompress4X{1,2}_usingDTable_internal_fast`,
//!   taken on 64-bit little-endian hosts for trees of at most 11 bits and
//!   streams of at least eight bytes) checks only that each stream
//!   regenerates its segment. A stream may stop short of its first bit or
//!   read on into the bytes before it, down to the jump table, and past
//!   them its container wraps, as `BIT_lookBitsFast` does;
//! - its pair decoder (`HUF_decodeLastSymbolX2`) caps the bits a stream's
//!   last symbol consumes at the container's end, so a stream whose last
//!   lookup is a pair ends exactly even when that pair reads past it;
//! - a treeless block decodes with the decoder type of the table it reuses
//!   (`HUF_decompress{1,4}X_usingDTable`): the last compressed block's
//!   choice, or pairs for a dictionary's table.
//!
//! Every section the exact decoder accepts, the reference accepts with the
//! same literals: an exactly consumed stream reads only its own bits, and
//! every path above decodes those bits to the same symbols. So the exact
//! decoder stays the only hot path, and this port, which follows the
//! reference's control flow statement for statement and reads the same
//! bits, decides only the sections it rejects.

use super::{HuffmanTable, LOOKUP_BITS, Row, TABLE_SIZE, prefers_pairs, row_bits, row_symbol};
use crate::decompressor::InvalidDataKind;
use crate::shared::format::highbit32;

/// Lookup bits of the reference's fast loops and its tables for trees of
/// at most that depth (`HUF_DECODER_FAST_TABLELOG`).
const FAST_LOG: u32 = 11;

/// The reference's `DTableDesc::tableType`: the decoder a table was read
/// for, which treeless blocks keep using.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum TableKind {
    /// `HUF_DEltX1` rows: one symbol per lookup.
    #[default]
    Single,
    /// `HUF_DEltX2` rows: up to two symbols per lookup.
    Pairs,
}

impl TableKind {
    /// The decoder the reference reads a `Compressed_Literals_Block`'s
    /// tree for: single symbols for one stream (`HUF_decompress1X1_DCtx_wksp`),
    /// and for four streams the choice of `HUF_selectDecoder` for `size`
    /// literals in a `compressed`-byte section, tree included.
    pub(crate) const fn for_block(four_streams: bool, size: usize, compressed: usize) -> Self {
        if four_streams && prefers_pairs(size, compressed) {
            Self::Pairs
        } else {
            Self::Single
        }
    }
}

/// `BIT_DStream_status`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Status {
    /// At least 57 bits are in the container.
    Unfinished,
    /// The container holds the stream's first bytes, not all consumed.
    EndOfBuffer,
    /// The container holds the stream's first bytes, all consumed.
    Completed,
    /// More bits were consumed than the stream holds.
    Overflow,
}

/// Loads the eight bytes at `src[at..]`; every caller's word is in bounds.
fn read64(src: &[u8], at: usize) -> u64 {
    src.get(at..)
        .and_then(<[u8]>::first_chunk::<8>)
        .map_or(0, |word| u64::from_le_bytes(*word))
}

/// `BIT_DStream_t` over a buffer: `ptr` and `start` index it.
#[derive(Clone, Copy, Debug)]
struct BitStream {
    container: u64,
    consumed: u32,
    ptr: usize,
    start: usize,
}

impl BitStream {
    /// `BIT_initDStream` over `src[start..start + len]`; `None` for an
    /// empty stream or a last byte without an end marker.
    fn new(src: &[u8], start: usize, len: usize) -> Option<Self> {
        let bytes = src.get(start..start.checked_add(len)?)?;
        let &last = bytes.last()?;
        if last == 0 {
            return None;
        }
        let marker = 8 - highbit32(u32::from(last));
        if len >= 8 {
            let ptr = start + len - 8;
            return Some(Self {
                container: read64(src, ptr),
                consumed: marker,
                ptr,
                start,
            });
        }
        let container = bytes
            .iter()
            .rev()
            .fold(0u64, |word, &byte| (word << 8) | u64::from(byte));
        Some(Self {
            container,
            consumed: marker + (8 - len as u32) * 8,
            ptr: start,
            start,
        })
    }

    /// `BIT_lookBitsFast`: past the container's end the shift wraps.
    const fn look(&self, count: u32) -> usize {
        ((self.container << (self.consumed & 63)) >> ((64 - count) & 63)) as usize
    }

    /// `BIT_skipBits`.
    const fn skip(&mut self, count: u32) {
        self.consumed = self.consumed.wrapping_add(count);
    }

    /// `BIT_reloadDStream_internal`.
    fn refill(&mut self, src: &[u8]) -> Status {
        self.ptr = self.ptr.saturating_sub((self.consumed >> 3) as usize);
        self.consumed &= 7;
        self.container = read64(src, self.ptr);
        Status::Unfinished
    }

    /// `BIT_reloadDStreamFast`.
    fn reload_fast(&mut self, src: &[u8]) -> Status {
        if self.ptr < self.start + 8 {
            return Status::Overflow;
        }
        self.refill(src)
    }

    /// `BIT_reloadDStream`.
    fn reload(&mut self, src: &[u8]) -> Status {
        if self.consumed > 64 {
            return Status::Overflow;
        }
        if self.ptr >= self.start + 8 {
            return self.refill(src);
        }
        if self.ptr == self.start {
            return if self.consumed < 64 {
                Status::EndOfBuffer
            } else {
                Status::Completed
            };
        }
        let mut bytes = (self.consumed >> 3) as usize;
        let mut status = Status::Unfinished;
        if self.ptr - self.start < bytes {
            bytes = self.ptr - self.start;
            status = Status::EndOfBuffer;
        }
        self.ptr -= bytes;
        self.consumed -= bytes as u32 * 8;
        self.container = read64(src, self.ptr);
        status
    }

    /// `BIT_endOfDStream`.
    const fn finished(&self) -> bool {
        self.ptr == self.start && self.consumed == 64
    }
}

/// The single-symbol row at the `log`-bit index `index`.
fn single(rows: &[Row; TABLE_SIZE], index: usize, log: u32) -> Row {
    rows[(index << (LOOKUP_BITS - log)) & (TABLE_SIZE - 1)]
}

/// A `HUF_DEltX2` row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Pair {
    /// The symbols, first one in the low byte; zero above a lone symbol.
    sequence: u16,
    /// The bits the symbols take.
    nb_bits: u32,
    /// The number of symbols, one or two.
    length: usize,
}

/// The pair row at the `log`-bit index `index`, as `HUF_fillDTableX2`
/// builds it: the code at the top of the index, followed by the next
/// code when both fit in `log` bits.
fn pair(rows: &[Row; TABLE_SIZE], index: usize, log: u32) -> Pair {
    let first = single(rows, index, log);
    let first_bits = row_bits(first);
    let rest = (index << first_bits) & ((1 << log) - 1);
    let second = single(rows, rest, log);
    let second_bits = row_bits(second);
    if first_bits + second_bits <= log {
        Pair {
            sequence: u16::from(row_symbol(first)) | (u16::from(row_symbol(second)) << 8),
            nb_bits: first_bits + second_bits,
            length: 2,
        }
    } else {
        Pair {
            sequence: u16::from(row_symbol(first)),
            nb_bits: first_bits,
            length: 1,
        }
    }
}

/// Stores the bytes of `sequence` that fit at `out[at..]`, as
/// `MEM_write16`; every store of the reference stays in bounds.
fn store2(out: &mut [u8], at: usize, sequence: u16) {
    for (byte, value) in out.iter_mut().skip(at).zip(sequence.to_le_bytes()) {
        *byte = value;
    }
}

/// Stores one byte, as the reference's in-bounds stores do.
fn store1(out: &mut [u8], at: usize, value: u8) {
    if let Some(byte) = out.get_mut(at) {
        *byte = value;
    }
}

/// The table, at the depth the reference decodes it with.
#[derive(Clone, Copy)]
struct Table<'a> {
    rows: &'a [Row; TABLE_SIZE],
    /// `DTableDesc::tableLog`: trees of at most [`FAST_LOG`] bits are
    /// decoded with [`FAST_LOG`]-bit lookups.
    log: u32,
}

impl Table<'_> {
    /// `HUF_decodeSymbolX1`.
    fn symbol(self, bits: &mut BitStream) -> u8 {
        let row = single(self.rows, bits.look(self.log), self.log);
        bits.skip(row_bits(row));
        row_symbol(row)
    }

    /// `HUF_decodeSymbolX2`: returns the symbols decoded.
    fn pair(self, bits: &mut BitStream, out: &mut [u8], at: usize) -> usize {
        let entry = pair(self.rows, bits.look(self.log), self.log);
        store2(out, at, entry.sequence);
        bits.skip(entry.nb_bits);
        entry.length
    }

    /// `HUF_decodeLastSymbolX2`.
    fn last_pair(self, bits: &mut BitStream, out: &mut [u8], at: usize) {
        let entry = pair(self.rows, bits.look(self.log), self.log);
        store1(out, at, entry.sequence as u8);
        if entry.length == 1 {
            bits.skip(entry.nb_bits);
        } else if bits.consumed < 64 {
            bits.skip(entry.nb_bits);
            bits.consumed = bits.consumed.min(64);
        }
    }

    /// `HUF_decodeStreamX1`: decodes `out[p..end]`.
    fn stream_single(self, bits: &mut BitStream, src: &[u8], out: &mut [u8], p: usize, end: usize) {
        let mut p = p;
        if end.saturating_sub(p) > 3 {
            while (bits.reload(src) == Status::Unfinished) & (p < end - 3) {
                for _ in 0..4 {
                    store1(out, p, self.symbol(bits));
                    p += 1;
                }
            }
        } else {
            bits.reload(src);
        }
        while p < end {
            store1(out, p, self.symbol(bits));
            p += 1;
        }
    }

    /// `HUF_decodeStreamX2`: decodes `out[p..end]`.
    fn stream_pairs(self, bits: &mut BitStream, src: &[u8], out: &mut [u8], p: usize, end: usize) {
        let mut p = p;
        if end.saturating_sub(p) >= 8 {
            let (lookups, room) = if self.log <= FAST_LOG { (5, 9) } else { (4, 7) };
            while (bits.reload(src) == Status::Unfinished) & (p < end - room) {
                for _ in 0..lookups {
                    p += self.pair(bits, out, p);
                }
            }
        } else {
            bits.reload(src);
        }
        if end.saturating_sub(p) >= 2 {
            while (bits.reload(src) == Status::Unfinished) & (p <= end - 2) {
                p += self.pair(bits, out, p);
            }
            while p <= end - 2 {
                p += self.pair(bits, out, p);
            }
        }
        if p < end {
            self.last_pair(bits, out, p);
        }
    }

    /// `HUF_decodeStreamX1` or `HUF_decodeStreamX2`.
    fn stream(
        self,
        kind: TableKind,
        bits: &mut BitStream,
        src: &[u8],
        out: &mut [u8],
        p: usize,
        end: usize,
    ) {
        match kind {
            TableKind::Single => self.stream_single(bits, src, out, p, end),
            TableKind::Pairs => self.stream_pairs(bits, src, out, p, end),
        }
    }
}

/// Decodes `out.len()` literals from one stream or four streams of `src`
/// exactly as the reference does with `table`, whose
/// [`HuffmanTable::kind`] is the reference's decoder type.
pub(super) fn decode(
    table: &HuffmanTable,
    src: &[u8],
    out: &mut [u8],
    four_streams: bool,
) -> Result<(), InvalidDataKind> {
    let decoder = Table {
        rows: &table.rows,
        log: table.log.max(FAST_LOG),
    };
    let kind = table.kind;
    let accepted = if four_streams {
        match fast(decoder, kind, src, out) {
            Fast::Decoded(accepted) => accepted,
            Fast::Unsupported => four(decoder, kind, src, out),
        }
    } else {
        one(decoder, kind, src, out)
    };
    if accepted {
        Ok(())
    } else {
        Err(InvalidDataKind::HuffmanStream)
    }
}

/// `HUF_decompress1X{1,2}_usingDTable_internal_body`.
fn one(table: Table<'_>, kind: TableKind, src: &[u8], out: &mut [u8]) -> bool {
    let Some(mut bits) = BitStream::new(src, 0, src.len()) else {
        return false;
    };
    table.stream(kind, &mut bits, src, out, 0, out.len());
    bits.finished()
}

/// The jump table of a four-stream section: the first three streams'
/// sizes, or `None` when `src` cannot hold them and a fourth stream
/// (`length4 > cSrcSize`).
fn jump_table(src: &[u8]) -> Option<[usize; 3]> {
    let (jump, _) = src.split_first_chunk::<6>()?;
    let size = |at: usize| usize::from(u16::from_le_bytes([jump[at], jump[at + 1]]));
    let sizes = [size(0), size(2), size(4)];
    (sizes.iter().sum::<usize>() + 6 <= src.len()).then_some(sizes)
}

/// `HUF_decompress4X{1,2}_usingDTable_internal_body`.
fn four(table: Table<'_>, kind: TableKind, src: &[u8], out: &mut [u8]) -> bool {
    let n = out.len();
    if src.len() < 10 || n < 6 {
        return false;
    }
    let Some([l1, l2, l3]) = jump_table(src) else {
        return false;
    };
    let starts = [6, 6 + l1, 6 + l1 + l2, 6 + l1 + l2 + l3];
    let lengths = [l1, l2, l3, src.len() - starts[3]];
    let segment = n.div_ceil(4);
    let ends = [segment, 2 * segment, 3 * segment, n];
    if ends[2] > n {
        return false;
    }
    let mut bits = [None; 4];
    for (slot, (&start, &len)) in bits.iter_mut().zip(starts.iter().zip(&lengths)) {
        *slot = BitStream::new(src, start, len);
    }
    let [Some(b1), Some(b2), Some(b3), Some(b4)] = bits else {
        return false;
    };
    let mut bits = [b1, b2, b3, b4];
    let mut ops = [0, segment, 2 * segment, 3 * segment];
    let (lookups, olimit) = match kind {
        TableKind::Single => (4, n - 3),
        TableKind::Pairs => (4, n - 7),
    };
    if n - ops[3] >= 8 {
        let mut unfinished = true;
        while unfinished && ops[3] < olimit {
            for (stream, op) in bits.iter_mut().zip(&mut ops) {
                for _ in 0..lookups {
                    match kind {
                        TableKind::Single => {
                            store1(out, *op, table.symbol(stream));
                            *op += 1;
                        }
                        TableKind::Pairs => *op += table.pair(stream, out, *op),
                    }
                }
            }
            unfinished = bits
                .iter_mut()
                .map(|stream| stream.reload_fast(src) == Status::Unfinished)
                .fold(true, |all, each| all & each);
        }
    }
    if ops[0] > ends[0] || ops[1] > ends[1] || ops[2] > ends[2] {
        return false;
    }
    for ((stream, &op), &end) in bits.iter_mut().zip(&ops).zip(&ends) {
        table.stream(kind, stream, src, out, op, end);
    }
    bits.iter().all(BitStream::finished)
}

/// The outcome of the fast four-stream path.
enum Fast {
    /// `HUF_DecompressFastArgs_init` declined; the general path decodes.
    Unsupported,
    /// The fast path ran: whether it accepted the streams.
    Decoded(bool),
}

/// `HUF_initFastDStream`: the word at `src[at..]` with a sentinel bit
/// below it, shifted past the end marker.
fn fast_container(src: &[u8], at: usize) -> u64 {
    let last = src.get(at + 7).copied().unwrap_or(0);
    let consumed = if last == 0 {
        0
    } else {
        8 - highbit32(u32::from(last))
    };
    (read64(src, at) | 1) << consumed
}

/// `HUF_decompress4X{1,2}_usingDTable_internal_fast` with the C loops.
fn fast(table: Table<'_>, kind: TableKind, src: &[u8], out: &mut [u8]) -> Fast {
    let n = out.len();
    // HUF_DecompressFastArgs_init.
    let Some((jump, _)) = src.split_first_chunk::<6>().filter(|_| src.len() >= 10) else {
        return Fast::Decoded(false);
    };
    if table.log != FAST_LOG {
        return Fast::Unsupported;
    }
    let size = |at: usize| usize::from(u16::from_le_bytes([jump[at], jump[at + 1]]));
    let [l1, l2, l3] = [size(0), size(2), size(4)];
    let l4 = src.len().wrapping_sub(l1 + l2 + l3 + 6);
    if l1 < 8 || l2 < 8 || l3 < 8 || l4 < 8 {
        return Fast::Unsupported;
    }
    if l4 > src.len() {
        return Fast::Decoded(false);
    }
    let iend = [6, 6 + l1, 6 + l1 + l2, 6 + l1 + l2 + l3];
    let mut ip = [iend[1] - 8, iend[2] - 8, iend[3] - 8, src.len() - 8];
    let segment = n.div_ceil(4);
    let mut op = [0, segment, 2 * segment, 3 * segment];
    if op[3] >= n {
        return Fast::Unsupported;
    }
    let mut bits = ip.map(|at| fast_container(src, at));
    match kind {
        TableKind::Single => fast_singles(table, src, out, &mut ip, &mut op, &mut bits),
        TableKind::Pairs => fast_pairs(table, src, out, &mut ip, &mut op, &mut bits),
    }
    // Finish the streams one by one.
    let mut segment_end = 0;
    for stream in 0..4 {
        segment_end = if segment <= n - segment_end {
            segment_end + segment
        } else {
            n
        };
        // HUF_initRemainingDStream.
        if op[stream] > segment_end || ip[stream] + 8 < iend[stream] {
            return Fast::Decoded(false);
        }
        let mut remaining = BitStream {
            container: read64(src, ip[stream]),
            consumed: bits[stream].trailing_zeros(),
            ptr: ip[stream],
            start: 0,
        };
        table.stream(kind, &mut remaining, src, out, op[stream], segment_end);
    }
    Fast::Decoded(true)
}

/// Whether a stream's position has crossed the previous one's, which ends
/// the fast loops.
fn crossed(ip: &[usize; 4]) -> bool {
    (1..4).any(|stream| ip[stream] < ip[stream - 1])
}

/// `HUF_4X1_RELOAD_STREAM` / `HUF_4X2_RELOAD_STREAM`'s refill.
fn fast_refill(src: &[u8], ip: &mut usize, bits: &mut u64) {
    let ctz = bits.trailing_zeros();
    *ip = ip.saturating_sub((ctz >> 3) as usize);
    *bits = (read64(src, *ip) | 1) << (ctz & 7);
}

/// `HUF_decompress4X1_usingDTable_internal_fast_c_loop`.
fn fast_singles(
    table: Table<'_>,
    src: &[u8],
    out: &mut [u8],
    ip: &mut [usize; 4],
    op: &mut [usize; 4],
    bits: &mut [u64; 4],
) {
    let n = out.len();
    loop {
        let iters = ((n - op[3]) / 5).min(ip[0] / 7);
        let olimit = op[3] + iters * 5;
        if op[3] == olimit || crossed(ip) {
            break;
        }
        while op[3] < olimit {
            for stream in 0..4 {
                for symbol in 0..5 {
                    let row = single(table.rows, (bits[stream] >> 53) as usize, FAST_LOG);
                    bits[stream] <<= row_bits(row) & 63;
                    store1(out, op[stream] + symbol, row_symbol(row));
                }
            }
            for stream in 0..4 {
                op[stream] += 5;
                fast_refill(src, &mut ip[stream], &mut bits[stream]);
            }
        }
    }
}

/// `HUF_decompress4X2_usingDTable_internal_fast_c_loop`.
fn fast_pairs(
    table: Table<'_>,
    src: &[u8],
    out: &mut [u8],
    ip: &mut [usize; 4],
    op: &mut [usize; 4],
    bits: &mut [u64; 4],
) {
    let oend = [op[1], op[2], op[3], out.len()];
    loop {
        let mut iters = ip[0] / 7;
        for stream in 0..4 {
            iters = iters.min((oend[stream] - op[stream]) / 10);
        }
        let olimit = op[3] + iters * 5;
        if op[3] == olimit || crossed(ip) {
            break;
        }
        while op[3] < olimit {
            for stream in 0..4 {
                for _ in 0..5 {
                    let entry = pair(table.rows, (bits[stream] >> 53) as usize, FAST_LOG);
                    store2(out, op[stream], entry.sequence);
                    bits[stream] <<= entry.nb_bits & 63;
                    op[stream] += entry.length;
                }
                fast_refill(src, &mut ip[stream], &mut bits[stream]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{
        encode, encode_four, example_table, flat_table, message, read, twelve_bit_table,
    };
    use super::*;
    use alloc::vec::Vec;

    /// `table` with the reference's decoder type `kind`.
    fn with_kind(mut table: HuffmanTable, kind: TableKind) -> HuffmanTable {
        table.set_kind(kind);
        table
    }

    /// Decodes `n` literals through the port.
    fn port(
        table: &HuffmanTable,
        src: &[u8],
        n: usize,
        four: bool,
    ) -> Result<Vec<u8>, InvalidDataKind> {
        let mut out = alloc::vec![0xee; n];
        decode(table, src, &mut out, four).map(|()| out)
    }

    /// Decodes `n` literals through the public entry, the exact decoder
    /// first.
    fn entry(
        table: &mut HuffmanTable,
        src: &[u8],
        n: usize,
        four: bool,
    ) -> Result<Vec<u8>, InvalidDataKind> {
        let mut out = alloc::vec![0xee; n];
        table
            .decode_streams(
                fearless_simd::Fallback::new(),
                src,
                &mut out,
                n,
                four,
                crate::decompressor::core::ahead::WriteAhead::NONE,
            )
            .map(|()| out)
    }

    /// The jump table and streams of a four-stream section.
    fn section(streams: &[&[u8]; 4]) -> Vec<u8> {
        let mut src = Vec::new();
        for stream in &streams[..3] {
            src.extend_from_slice(&(stream.len() as u16).to_le_bytes());
        }
        for stream in streams {
            src.extend_from_slice(stream);
        }
        src
    }

    #[test]
    fn exactly_consumed_sections_decode_as_the_exact_decoder_does() {
        for table in [example_table(), twelve_bit_table(), flat_table()] {
            for kind in [TableKind::Single, TableKind::Pairs] {
                let table = with_kind(table.clone(), kind);
                for len in [0, 1, 2, 6, 7, 9, 13, 64, 333, 4_000, 20_000] {
                    let symbols = message(&table, len, len as u64 + 5);
                    let label = alloc::format!("log {} {kind:?} len {len}", table.log);
                    let single = encode(&table, &symbols);
                    assert_eq!(
                        port(&table, &single, len, false),
                        Ok(symbols.clone()),
                        "{label}"
                    );
                    if len >= 6 {
                        let four = encode_four(&table, &symbols);
                        assert_eq!(port(&table, &four, len, true), Ok(symbols), "{label} four");
                    }
                }
            }
        }
    }

    #[test]
    fn the_fast_path_accepts_streams_with_bits_left() {
        // Each stream holds two symbols more than its segment: the fast
        // path takes each segment's symbols and ignores the rest.
        for (table, fast) in [(flat_table(), true), (twelve_bit_table(), false)] {
            for kind in [TableKind::Single, TableKind::Pairs] {
                let mut table = with_kind(table.clone(), kind);
                let segment = 40;
                let messages: Vec<Vec<u8>> = (0..4)
                    .map(|index| message(&table, segment + 2, index))
                    .collect();
                let streams: Vec<Vec<u8>> = messages
                    .iter()
                    .map(|symbols| encode(&table, symbols))
                    .collect();
                let src = section(&[&streams[0], &streams[1], &streams[2], &streams[3]]);
                let expected: Vec<u8> = messages
                    .iter()
                    .flat_map(|symbols| symbols[..segment].iter().copied())
                    .collect();
                let decoded = entry(&mut table, &src, 4 * segment, true);
                if fast {
                    assert_eq!(decoded, Ok(expected), "{kind:?}");
                } else {
                    // Twelve-bit trees take the general path, which checks
                    // every stream's end.
                    assert_eq!(decoded, Err(InvalidDataKind::HuffmanStream), "{kind:?}");
                }
            }
        }
    }

    #[test]
    fn the_fast_path_reads_on_into_the_bytes_before_a_stream() {
        // Four streams of eight zero bytes, which the fast path opens
        // without an end marker, and 20 symbols of four bits each: every
        // stream reads 16 bits past its start. E (0000) fills the zeros;
        // the first stream reads on into the jump table, 08 00 08 00 08
        // 00, whose fifth byte, 0000_1000, gives E then A.
        let zeros = [0u8; 8];
        let src = section(&[&zeros, &zeros, &zeros, &zeros]);
        let mut expected = alloc::vec![4u8; 80];
        expected[19] = 0;
        for kind in [TableKind::Single, TableKind::Pairs] {
            let mut table = with_kind(example_table(), kind);
            assert_eq!(
                entry(&mut table, &src, 80, true),
                Ok(expected.clone()),
                "{kind:?}"
            );
        }
    }

    #[test]
    fn reading_far_past_a_stream_start_is_caught_after_the_fast_loop() {
        // A long first stream lets the fast loop run eight rounds; 40
        // symbols of four bits then move the later streams' positions 20
        // bytes back, more than a word before their own starts.
        let long = [0u8; 64];
        let zeros = [0u8; 8];
        let src = section(&[&long, &zeros, &zeros, &zeros]);
        for kind in [TableKind::Single, TableKind::Pairs] {
            let mut table = with_kind(example_table(), kind);
            assert_eq!(
                entry(&mut table, &src, 160, true),
                Err(InvalidDataKind::HuffmanStream),
                "{kind:?}"
            );
            // With 17 symbols per stream, the later streams read only four
            // bits past their starts, all E.
            assert_eq!(
                entry(&mut table, &src, 68, true),
                Ok(alloc::vec![4; 68]),
                "{kind:?}"
            );
        }
    }

    #[test]
    fn a_last_pair_may_end_a_stream_past_its_bits() {
        // The stream holds `10`: A (1), then a lone 0. A pair lookup reads
        // A and E (0000) from the zero padding, and the reference caps the
        // bits consumed at the stream's end.
        let stream = [0b0000_0110u8];
        let mut single = with_kind(example_table(), TableKind::Single);
        assert_eq!(
            entry(&mut single, &stream, 1, false),
            Err(InvalidDataKind::HuffmanStream)
        );
        let mut pairs = with_kind(example_table(), TableKind::Pairs);
        assert_eq!(entry(&mut pairs, &stream, 1, false), Ok(alloc::vec![0]));
        // So does the general four-stream path, here for streams shorter
        // than a word: three of A A, then the same last stream.
        let aa = [0b0000_0111u8];
        let src = section(&[&aa, &aa, &aa, &stream]);
        assert_eq!(
            entry(&mut single, &src, 7, true),
            Err(InvalidDataKind::HuffmanStream)
        );
        assert_eq!(entry(&mut pairs, &src, 7, true), Ok(alloc::vec![0; 7]));
    }

    #[test]
    fn malformed_sections_stay_rejected() {
        let table = with_kind(flat_table(), TableKind::Pairs);
        let invalid = Err(InvalidDataKind::HuffmanStream);
        // Too short for a jump table and four streams; streams past the
        // section, with long first streams (the fast path) and short ones;
        // a stream without an end marker on the general path; an empty
        // single stream; fewer than six literals.
        assert_eq!(port(&table, &[1, 0, 1, 0, 1, 0, 1, 1, 1], 8, true), invalid);
        let mut past = section(&[&[1; 8], &[1; 8], &[1; 8], &[1; 8]]);
        past[4] = 9;
        assert_eq!(port(&table, &past, 8, true), invalid);
        assert_eq!(
            port(&table, &[9, 0, 1, 0, 1, 0, 1, 1, 1, 1], 8, true),
            invalid
        );
        assert_eq!(
            port(&table, &[1, 0, 1, 0, 1, 0, 1, 1, 1, 0], 8, true),
            invalid
        );
        assert_eq!(port(&table, &[], 1, false), invalid);
        assert_eq!(
            port(&table, &[1, 0, 1, 0, 1, 0, 1, 1, 1, 1], 5, true),
            invalid
        );
    }

    #[test]
    fn pair_rows_are_the_decoders_pair_rows() {
        for table in [example_table(), flat_table()] {
            let mut table = table;
            table.build_pairs();
            let built = table.pairs.as_ref().expect("pair rows");
            for (index, row) in built.rows().enumerate() {
                let entry = pair(&table.rows, index, FAST_LOG);
                assert_eq!(
                    (entry.sequence, entry.nb_bits, entry.length),
                    (row.symbols(), u32::from(row.nb_bits()), row.count()),
                    "log {} index {index}",
                    table.log
                );
            }
        }
        // Twelve-bit lookups pair codes up to twelve bits.
        let table = twelve_bit_table();
        let both = pair(&table.rows, 0b11 << 10, MAX_LOG_FOR_TESTS);
        assert_eq!((both.sequence, both.nb_bits, both.length), (0, 2, 2));
        let lone = pair(&table.rows, 0, MAX_LOG_FOR_TESTS);
        assert_eq!(lone.length, 1);
        assert_eq!(lone.nb_bits, 12);
    }

    /// The deepest table's lookup bits.
    const MAX_LOG_FOR_TESTS: u32 = LOOKUP_BITS;

    #[test]
    fn bit_streams_follow_the_reference_reader() {
        assert!(BitStream::new(&[], 0, 0).is_none());
        assert!(BitStream::new(&[1, 0], 0, 2).is_none());
        assert!(BitStream::new(&[1], 0, 2).is_none());
        // Three bytes: the missing five count as consumed.
        let short = BitStream::new(&[0xaa, 0xbb, 0x01], 0, 3).expect("marked");
        assert_eq!(short.consumed, 8 + 40);
        assert_eq!(short.container, 0x01bbaa);
        // Twelve bytes ending in 12 (five padding and marker bits):
        // refills move back by whole bytes, then cautiously to the start,
        // and report the end and overruns.
        let src: Vec<u8> = (1..=12).collect();
        let mut bits = BitStream::new(&src, 0, 12).expect("marked");
        assert_eq!((bits.ptr, bits.consumed), (4, 5));
        bits.skip(11);
        assert_eq!(bits.reload(&src), Status::Unfinished);
        assert_eq!((bits.ptr, bits.consumed), (2, 0));
        let mut bits = BitStream::new(&src, 0, 12).expect("marked");
        bits.skip(35);
        assert_eq!(bits.reload_fast(&src), Status::Overflow);
        assert_eq!(bits.reload(&src), Status::EndOfBuffer);
        assert_eq!((bits.ptr, bits.consumed), (0, 8));
        assert_eq!(bits.reload(&src), Status::EndOfBuffer);
        bits.skip(56);
        assert_eq!(bits.reload(&src), Status::Completed);
        assert!(bits.finished());
        bits.skip(1);
        assert_eq!(bits.reload(&src), Status::Overflow);
        assert!(!bits.finished());
        // A whole word away from the start, both refills move back.
        let long = [7u8; 20];
        let mut bits = BitStream::new(&long, 0, 20).expect("marked");
        assert_eq!((bits.ptr, bits.consumed), (12, 6));
        bits.skip(18);
        assert_eq!(bits.reload_fast(&long), Status::Unfinished);
        assert_eq!((bits.ptr, bits.consumed), (9, 0));
        bits.skip(16);
        assert_eq!(bits.reload(&long), Status::Unfinished);
        assert_eq!((bits.ptr, bits.consumed), (7, 0));
    }

    #[test]
    fn compressed_blocks_choose_the_reference_decoder() {
        assert_eq!(TableKind::default(), TableKind::Single);
        assert_eq!(
            TableKind::for_block(false, 100_000, 30_000),
            TableKind::Single
        );
        assert_eq!(
            TableKind::for_block(true, 100_000, 30_000),
            TableKind::Pairs
        );
        assert_eq!(TableKind::for_block(true, 100, 99), TableKind::Single);
        let (table, _) = read(&[132, 0x43, 0x20, 0x10]).expect("valid");
        assert_eq!(table.kind(), TableKind::Single);
    }
}

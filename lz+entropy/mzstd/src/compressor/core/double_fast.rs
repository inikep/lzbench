//! The `dfast` strategy (`zstd_double_fast.c`): a long (8-byte) hash table
//! in `hash_table` and a short one in `chain_table`.

use core::cell::Cell;

use fearless_simd::Simd;
use fearless_simd_macros::simd;

use super::cdict::TableLoad;
use super::fast::{DictBytes, rep_index_ok, tags_match, write_tagged_index};
use super::match_state::{BlockInput, DictMatchState, MatchState};
use super::mem::{
    count_at_simd, count_slices_simd, hash_ptr, hash_word, load32, load64, prefetch_l1, try_load64,
};
use super::params::SHORT_CACHE_TAG_BITS;
use super::seq_store::{SeqStore, offset_to_offbase, repcode_to_offbase};
use super::window::HASH_READ_SIZE;

/// `kSearchStrength`.
const SEARCH_STRENGTH: u32 = 8;

/// `ZSTD_fillDoubleHashTableForCCtx` with `ZSTD_dtlm_fast`.
pub(crate) fn fill_double_hash_table(ms: &mut MatchState, prefix: &super::mem::Mem<'_>, end: u32) {
    let h_bits_l = ms.cparams.hash_log;
    let h_bits_s = ms.cparams.chain_log;
    let mls = ms.cparams.min_match;
    let iend = i64::from(end) - i64::from(HASH_READ_SIZE);
    let mut ip = i64::from(ms.next_to_update);
    while ip + 2 <= iend {
        let curr = ip as u32;
        ms.chain_table[hash_ptr(prefix, curr, h_bits_s, mls)] = curr;
        ms.hash_table[hash_ptr(prefix, curr, h_bits_l, 8)] = curr;
        ip += 3;
    }
}

/// `ZSTD_fillDoubleHashTableForCCtx`: [`fill_double_hash_table`], and with
/// [`TableLoad::Full`] also the two positions after each indexed one in the
/// long table where their slot is still empty.
pub(crate) fn fill_double_hash_table_for(
    ms: &mut MatchState,
    prefix: &super::mem::Mem<'_>,
    end: u32,
    dtlm: TableLoad,
) {
    if dtlm == TableLoad::Fast {
        return fill_double_hash_table(ms, prefix, end);
    }
    let h_bits_l = ms.cparams.hash_log;
    let h_bits_s = ms.cparams.chain_log;
    let mls = ms.cparams.min_match;
    let iend = i64::from(end) - i64::from(HASH_READ_SIZE);
    let mut ip = i64::from(ms.next_to_update);
    while ip + 2 <= iend {
        let curr = ip as u32;
        for i in 0..3 {
            let small = hash_ptr(prefix, curr + i, h_bits_s, mls);
            let large = hash_ptr(prefix, curr + i, h_bits_l, 8);
            if i == 0 {
                ms.chain_table[small] = curr + i;
            }
            if i == 0 || ms.hash_table[large] == 0 {
                ms.hash_table[large] = curr + i;
            }
        }
        ip += 3;
    }
}

/// `ZSTD_fillDoubleHashTableForCDict` with `ZSTD_dtlm_full`: both tables
/// hold tagged indices (`ZSTD_writeTaggedIndex`); every third position
/// enters both, the ones in between the long table where its slot is empty.
pub(crate) fn fill_cdict_double_hash_table(
    ms: &mut MatchState,
    prefix: &super::mem::Mem<'_>,
    end: u32,
) {
    let h_bits_l = ms.cparams.hash_log + SHORT_CACHE_TAG_BITS;
    let h_bits_s = ms.cparams.chain_log + SHORT_CACHE_TAG_BITS;
    let mls = ms.cparams.min_match;
    let iend = i64::from(end) - i64::from(HASH_READ_SIZE);
    let mut ip = i64::from(ms.next_to_update);
    while ip + 2 <= iend {
        let curr = ip as u32;
        for i in 0..3 {
            let small = hash_ptr(prefix, curr + i, h_bits_s, mls);
            let large = hash_ptr(prefix, curr + i, h_bits_l, 8);
            if i == 0 {
                write_tagged_index(&mut ms.chain_table, small, curr + i);
            }
            if i == 0 || ms.hash_table[large >> SHORT_CACHE_TAG_BITS] == 0 {
                write_tagged_index(&mut ms.hash_table, large, curr + i);
            }
        }
        ip += 3;
    }
}

/// `ZSTD_compressBlock_doubleFast`: selects the variant compiled for the
/// minimum match length, as the reference's `ZSTD_GEN_DFAST_FN` templates
/// do.
pub(crate) fn compress_block(
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
) -> usize {
    fearless_simd::dispatch!(ms.backend.0, simd => match ms.cparams.min_match {
        5 => compress_block_mls::<_, 5>(simd, ms, seq_store, rep, block),
        6 => compress_block_mls::<_, 6>(simd, ms, seq_store, rep, block),
        7.. => compress_block_mls::<_, 7>(simd, ms, seq_store, rep, block),
        _ => compress_block_mls::<_, 4>(simd, ms, seq_store, rep, block),
    })
}

/// `ZSTD_compressBlock_doubleFast_noDict_generic` for one minimum match
/// length, compiled for one SIMD level.
#[simd]
fn compress_block_mls<S: Simd, const MLS: u32>(
    simd: S,
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
) -> usize {
    let cparams = ms.cparams;
    let prefix = block.window.prefix;
    let base = prefix.start();
    let prefix_lowest_index = ms.lowest_prefix_index(block.end, cparams.window_log);
    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];
    let mut offset_saved1 = 0u32;
    let mut offset_saved2 = 0u32;

    let mut ip = block.start;
    ip += u32::from(ip == prefix_lowest_index);
    {
        let window_low = ms.lowest_prefix_index(ip, cparams.window_log);
        let max_rep = ip - window_low;
        if offset_2 > max_rep {
            offset_saved2 = offset_2;
            offset_2 = 0;
        }
        if offset_1 > max_rep {
            offset_saved1 = offset_1;
            offset_1 = 0;
        }
    }

    let MatchState {
        hash_table: hash_long,
        chain_table: hash_small,
        ..
    } = ms;
    let tables = Tables {
        long: hash_long.as_mut_slice(),
        small: hash_small.as_mut_slice(),
    };
    let scan = |origin: u32| Scan {
        src: prefix.slice(origin, block.end),
        origin,
        prefix_lowest: (prefix_lowest_index - origin) as usize,
        start: (block.start - origin) as usize,
        h_bits_l: cparams.hash_log,
        h_bits_s: cparams.chain_log,
    };
    // Every repcode the search can use either is one of these two or was
    // found at a candidate inside the prefix, so when these two reach no
    // further back than the first position the block reads nothing below
    // the prefix: it is then searched as offsets from the prefix's start,
    // which spares the search loop the window's base and the history's
    // bounds.
    let in_prefix =
        block.start >= prefix_lowest_index && offset_1.max(offset_2) <= ip - prefix_lowest_index;
    let (literals, [rep_1, rep_2]) = if in_prefix {
        let scan = scan(prefix_lowest_index);
        let ip = (ip - prefix_lowest_index) as usize;
        let reps = [offset_1, offset_2];
        macro_rules! scan {
            ($hl:literal, $hs:literal) => {
                compress_scan_widths::<S, MLS, $hl, $hs>(simd, tables, seq_store, &scan, ip, reps)
            };
        }
        // The table widths of the default parameters (`hashLog`, `chainLog`
        // after their adjustment to the input's size) are compiled in with
        // their match lengths: levels 3 and 4 above 256 KiB, levels 2 and 3
        // up to 256 KiB and levels 3 and 4 up to 64 KiB.
        match (MLS, cparams.hash_log, cparams.chain_log) {
            (5, 14, 14) => scan!(14, 14),
            (4, 16, 16) => scan!(16, 16),
            (5, 17, 16) => scan!(17, 16),
            (5, 18, 18) => scan!(18, 18),
            (5, 16, 15) => scan!(16, 15),
            (4, 17, 16) => scan!(17, 16),
            _ => compress_scan::<S, MLS, true, 0, 0>(simd, tables, seq_store, &scan, ip, reps),
        }
    } else {
        let scan = scan(base);
        let ip = (ip - base) as usize;
        compress_scan::<S, MLS, false, 0, 0>(
            simd,
            tables,
            seq_store,
            &scan,
            ip,
            [offset_1, offset_2],
        )
    };
    offset_1 = rep_1;
    offset_2 = rep_2;

    offset_saved2 = if offset_saved1 != 0 && offset_1 != 0 {
        offset_saved1
    } else {
        offset_saved2
    };
    rep[0] = if offset_1 != 0 {
        offset_1
    } else {
        offset_saved1
    };
    rep[1] = if offset_2 != 0 {
        offset_2
    } else {
        offset_saved2
    };
    literals
}

/// The two hash tables a `dfast` block reads and fills.
struct Tables<'a> {
    /// Eight-byte hashes (`hashLong`).
    long: &'a mut [u32],
    /// `MLS`-byte hashes (`hashSmall`).
    small: &'a mut [u32],
}

/// The bytes one `dfast` block searches and the parameters it searches them
/// with.
struct Scan<'a> {
    /// The block and the history before it, from window index `origin` on.
    src: &'a [u8],
    /// Window index of `src[0]`: the reference's `base`, as an offset.
    origin: u32,
    /// Offset of the prefix's lowest valid index, below which no candidate
    /// is taken.
    prefix_lowest: usize,
    /// Offset of the block's first byte.
    start: usize,
    /// `hashLog`, the long table's hash width.
    h_bits_l: u32,
    /// `chainLog`, the short table's hash width.
    h_bits_s: u32,
}

/// The two hash tables of a `dfast` search as cells, with their widths:
/// `HL` and `HS` when compiled in (non-zero), else `h_bits_l` and
/// `h_bits_s`.
///
/// The slots are found by methods that are always inlined, rather than by
/// closures over the widths: when LLVM left a closure that calls them out
/// of line, the widths and table lengths it had captured were loaded from
/// memory again, the hash shifts became variable and the bounds checks came
/// back.
#[derive(Clone, Copy)]
struct Slots<'t, const MLS: u32, const HL: u32, const HS: u32> {
    /// Eight-byte hashes (`hashLong`), `1 << HL` entries when `HL` is set.
    long: &'t [Cell<u32>],
    /// `MLS`-byte hashes (`hashSmall`), `1 << HS` entries when `HS` is set.
    small: &'t [Cell<u32>],
    /// `hashLog`, used when `HL` is zero.
    h_bits_l: u32,
    /// `chainLog`, used when `HS` is zero.
    h_bits_s: u32,
}

impl<'t, const MLS: u32, const HL: u32, const HS: u32> Slots<'t, MLS, HL, HS> {
    /// The long table's slot for the eight bytes `word`. The hash is below
    /// the table's size (a power of two), so masking it with `len - 1`
    /// changes nothing and proves the index in bounds.
    #[inline(always)]
    fn long(self, word: u64) -> &'t Cell<u32> {
        let bits = if HL == 0 { self.h_bits_l } else { HL };
        &self.long[hash_word::<8>(word, bits) & (self.long.len() - 1)]
    }

    /// The short table's slot for the `MLS` bytes that `word` starts with.
    #[inline(always)]
    fn small(self, word: u64) -> &'t Cell<u32> {
        let bits = if HS == 0 { self.h_bits_s } else { HS };
        &self.small[hash_word::<MLS>(word, bits) & (self.small.len() - 1)]
    }

    /// The word at offset `p` of `src` and its long-table slot, or `None`
    /// when the eight bytes at `p` are not all inside `src` (`p > ilimit`);
    /// the two candidates of `p` are prefetched.
    ///
    /// After a match the search goes on from where it ends: these are loaded
    /// there before the match's backward extension and its store, so the
    /// next search's loads wait neither for the catch-up loop nor for the
    /// store's branches (2-3% less time on text).
    #[inline(always)]
    fn ahead(self, src: &[u8], base: u32, p: usize) -> Option<(u64, &'t Cell<u32>)> {
        let word = try_load64(src, p)?;
        let slot = self.long(word);
        // Only with the default widths: the run-time widths serve the
        // small inputs, whose tables stay in L1 (tiny text at level 3 took
        // 4% longer with the prefetches).
        if HL != 0 {
            prefetch_l1(src, slot.get().wrapping_sub(base) as usize);
            prefetch_l1(src, self.small(word).get().wrapping_sub(base) as usize);
        }
        Some((word, slot))
    }
}

/// [`compress_scan`] in the prefix with the hash widths `HL` and `HS`
/// compiled in: the hash shifts become immediates and the table bounds
/// constants, which frees four of the loop's registers.
///
/// Each width pair is its own function, out of line: inlined side by side
/// into one caller, LLVM merged the instances back into a single loop that
/// selects the widths at run time.
#[simd]
#[inline(never)]
fn compress_scan_widths<S: Simd, const MLS: u32, const HL: u32, const HS: u32>(
    simd: S,
    tables: Tables<'_>,
    seq_store: &mut SeqStore,
    scan: &Scan<'_>,
    ip: usize,
    reps: [u32; 2],
) -> (usize, [u32; 2]) {
    compress_scan::<S, MLS, true, HL, HS>(simd, tables, seq_store, scan, ip, reps)
}

/// The search and match loop of `ZSTD_compressBlock_doubleFast_noDict_generic`
/// from offset `ip` of `scan.src`, with the repcodes `reps`. Returns the
/// length of the trailing literals and the final repcodes.
///
/// `IN_PREFIX` states that `scan.prefix_lowest` is zero, so the candidates'
/// history is all of `src` and its lower bound the window index `origin`.
#[inline(always)]
fn compress_scan<S: Simd, const MLS: u32, const IN_PREFIX: bool, const HL: u32, const HS: u32>(
    simd: S,
    tables: Tables<'_>,
    seq_store: &mut SeqStore,
    scan: &Scan<'_>,
    mut ip: usize,
    reps: [u32; 2],
) -> (usize, [u32; 2]) {
    let Scan {
        src, origin: base, ..
    } = *scan;
    let h_bits_l = if HL == 0 { scan.h_bits_l } else { HL };
    let h_bits_s = if HS == 0 { scan.h_bits_s } else { HS };
    let prefix_lowest = if IN_PREFIX { 0 } else { scan.prefix_lowest };
    let prefix_lowest_index = base + prefix_lowest as u32;
    // The candidate load's bound also enforces the searchable lower limit.
    let history = &src[prefix_lowest..];
    let iend = src.len();
    let mut anchor = scan.start;
    // The reference's `ilimit` is `iend - HASH_READ_SIZE`: a position `p`
    // is at most `ilimit` exactly while the eight bytes at `p` are inside
    // `src`, so `try_load64` is that test throughout.
    // Offsets below 2^32, as `usize` so that the repcode tests and reads
    // share one register.
    let [mut offset_1, mut offset_2] = reps.map(|rep| rep as usize);
    let k_step_incr: usize = 1 << SEARCH_STRENGTH;

    let index = |p: usize| p as u32 + base;
    let read32 = |p: usize| load32(src, p);
    // The first word is compared inline (`count_at_simd`), without the two
    // slices the full count builds. With the loop inlined into
    // `compress_block_mls` this took 3-6% longer on text; since the
    // compiled-in widths moved the loop into its own function it saves
    // 1-3% (i7-13700KF, 2026-10-18).
    let count = |a: usize, b: usize| count_at_simd(simd, src, a, b);

    let Tables {
        long: hash_long,
        small: hash_small,
    } = tables;
    // Hashes are below the table sizes (`1 << hash_log`, `1 << chain_log`);
    // masking with `len - 1` changes nothing and, with the tables known to
    // be non-empty, proves every index in bounds. With a width compiled in,
    // the tables are first cut to that size, which makes the mask a
    // constant (a table is never smaller than its `1 << hash_log`).
    let hash_long = if HL == 0 {
        hash_long
    } else {
        &mut hash_long[..1 << HL]
    };
    let hash_small = if HS == 0 {
        hash_small
    } else {
        &mut hash_small[..1 << HS]
    };
    assert!(!hash_long.is_empty() && !hash_small.is_empty());
    // The search keeps references to the slots it will write rather than
    // their hashes, so a slot is bounds checked once, where its hash is
    // computed.
    let slots = Slots::<MLS, HL, HS> {
        long: Cell::from_mut(hash_long).as_slice_of_cells(),
        small: Cell::from_mut(hash_small).as_slice_of_cells(),
        h_bits_l,
        h_bits_s,
    };
    // A block whose first position is past `ilimit` is all literals: the
    // first `ip1 > ilimit` test would stop it too.
    let Some(mut word0) = try_load64(src, ip) else {
        return (iend - anchor, [offset_1 as u32, offset_2 as u32]);
    };
    let mut slot_l0 = slots.long(word0);

    'outer: loop {
        let mut step: usize = 1;
        let mut next_step = ip + k_step_incr;
        let mut ip1 = ip + step;
        // The eight bytes at `ip` and at `ip1`: every hash and compare at a
        // searched position reads them. The search goes on while
        // `ip1 <= ilimit`, that is while the eight bytes at `ip1` are inside
        // `src`, so loading them is the loop's exit test.
        let Some(mut word1) = try_load64(src, ip1) else {
            break 'outer;
        };
        let mut idxl0 = slot_l0.get();
        let mut curr: u32;

        enum Next<'t> {
            Stored(usize),
            Found {
                offset: usize,
                m_length: usize,
                slot_l1: &'t Cell<u32>,
            },
            SearchNextLong {
                idxs0: u32,
                idxl1: u32,
                slot_l1: &'t Cell<u32>,
            },
        }
        let next = loop {
            let slot_s0 = slots.small(word0);
            let idxs0 = slot_s0.get();
            curr = index(ip);
            slot_l0.set(curr);
            slot_s0.set(curr);

            if offset_1 > 0 && read32(ip + 1 - offset_1) == (word0 >> 8) as u32 {
                let m_length = count(ip + 1 + 4, ip + 1 + 4 - offset_1) + 4;
                ip += 1;
                break Next::Stored(m_length);
            }

            let slot_l1 = slots.long(word1);
            if super::mem::matches64(history, idxl0.wrapping_sub(prefix_lowest_index), word0) {
                let matchl0 = (idxl0 - base) as usize;
                let m_length = count(ip + 8, matchl0 + 8) + 8;
                break Next::Found {
                    offset: ip - matchl0,
                    m_length,
                    slot_l1,
                };
            }

            let idxl1 = slot_l1.get();
            if super::mem::matches32(
                history,
                idxs0.wrapping_sub(prefix_lowest_index),
                word0 as u32,
            ) {
                break Next::SearchNextLong {
                    idxs0,
                    idxl1,
                    slot_l1,
                };
            }

            if ip1 >= next_step {
                // `PREFETCH_L1`, as the reference does when the step grows.
                prefetch_l1(src, ip1 + 64);
                prefetch_l1(src, ip1 + 128);
                step += 1;
                next_step += k_step_incr;
            }
            ip = ip1;
            ip1 += step;
            slot_l0 = slot_l1;
            idxl0 = idxl1;
            word0 = word1;
            match try_load64(src, ip1) {
                Some(word) => word1 = word,
                None => break 'outer,
            }
        };

        // The match's forward end, the position the search goes on from.
        let end;
        let next_word = match next {
            Next::Stored(m_length) => {
                end = ip + m_length;
                let next_word = slots.ahead(src, base, end);
                store_seq_wild(
                    seq_store,
                    src,
                    anchor,
                    ip - anchor,
                    repcode_to_offbase(1),
                    m_length,
                );
                next_word
            }
            Next::Found {
                offset,
                mut m_length,
                slot_l1,
            } => {
                end = ip + m_length;
                let next_word = slots.ahead(src, base, end);
                let mut matchl0 = ip - offset;
                while ip > anchor && matchl0 > prefix_lowest && src[ip - 1] == src[matchl0 - 1] {
                    ip -= 1;
                    matchl0 -= 1;
                    m_length += 1;
                }
                offset_2 = offset_1;
                offset_1 = offset;
                if step < 4 {
                    slot_l1.set(index(ip1));
                }
                store_seq_wild(
                    seq_store,
                    src,
                    anchor,
                    ip - anchor,
                    offset_to_offbase(offset as u32),
                    m_length,
                );
                next_word
            }
            Next::SearchNextLong {
                idxs0,
                idxl1,
                slot_l1,
            } => {
                let mut matchs0 = (idxs0 - base) as usize;
                let mut m_length = count(ip + 4, matchs0 + 4) + 4;
                let mut offset = ip - matchs0;
                // This second long candidate uses a strict lower limit in C.
                if super::mem::matches64(
                    &history[1..],
                    idxl1.wrapping_sub(prefix_lowest_index + 1),
                    word1,
                ) {
                    let matchl1 = (idxl1 - base) as usize;
                    let l1len = count(ip1 + 8, matchl1 + 8) + 8;
                    if l1len > m_length {
                        ip = ip1;
                        m_length = l1len;
                        offset = ip - matchl1;
                        matchs0 = matchl1;
                    }
                }
                end = ip + m_length;
                let next_word = slots.ahead(src, base, end);
                while ip > anchor && matchs0 > prefix_lowest && src[ip - 1] == src[matchs0 - 1] {
                    ip -= 1;
                    matchs0 -= 1;
                    m_length += 1;
                }
                offset_2 = offset_1;
                offset_1 = offset;
                if step < 4 {
                    slot_l1.set(index(ip1));
                }
                store_seq_wild(
                    seq_store,
                    src,
                    anchor,
                    ip - anchor,
                    offset_to_offbase(offset as u32),
                    m_length,
                );
                next_word
            }
        };

        // _match_stored:
        ip = end;
        anchor = ip;
        // `ip <= ilimit`:
        if let Some((word, slot)) = next_word {
            (word0, slot_l0) = (word, slot);
            let index_to_insert = curr + 2;
            let insert_at = (index_to_insert - base) as usize;
            slots.long(load64(src, insert_at)).set(index_to_insert);
            slots.long(load64(src, ip - 2)).set(index(ip - 2));
            slots.small(load64(src, insert_at)).set(index_to_insert);
            slots.small(load64(src, ip - 1)).set(index(ip - 1));

            while offset_2 > 0 && read32(ip - offset_2) == word0 as u32 {
                let r_length = count(ip + 4, ip + 4 - offset_2) + 4;
                core::mem::swap(&mut offset_1, &mut offset_2);
                slots.small(word0).set(index(ip));
                slot_l0.set(index(ip));
                seq_store.store_seq_only(0, repcode_to_offbase(1), r_length);
                ip += r_length;
                anchor = ip;
                match try_load64(src, ip) {
                    Some(word) => (word0, slot_l0) = (word, slots.long(word)),
                    None => break 'outer,
                }
            }
        }
    }

    (iend - anchor, [offset_1 as u32, offset_2 as u32])
}

/// [`SeqStore::store_seq_from`] for the prefix search: a literal run of at
/// most 16 bytes is copied as the 16-byte chunk at `anchor` (`ZSTD_copy16`)
/// when `src` holds that many, and the excess is cut off again.
///
/// The chunk is appended straight from `src`, without the shared helper's
/// copy through the stack, and its bounds test is `16 <= src.len() - anchor`,
/// which cannot overflow.
#[inline(always)]
fn store_seq_wild(
    seq_store: &mut SeqStore,
    src: &[u8],
    anchor: usize,
    lit_length: usize,
    off_base: u32,
    match_length: usize,
) {
    seq_store.push_literals_wild(src, anchor, lit_length);
    seq_store.store_seq_only(lit_length, off_base, match_length);
}

/// `ZSTD_compressBlock_doubleFast_extDict`: selects the variant compiled for
/// the minimum match length, dispatched once per block.
pub(crate) fn compress_block_ext_dict(
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
) -> usize {
    fearless_simd::dispatch!(ms.backend.0, simd => match ms.cparams.min_match {
        5 => compress_block_ext_dict_mls::<_, 5>(simd, ms, seq_store, rep, block),
        6 => compress_block_ext_dict_mls::<_, 6>(simd, ms, seq_store, rep, block),
        7.. => compress_block_ext_dict_mls::<_, 7>(simd, ms, seq_store, rep, block),
        _ => compress_block_ext_dict_mls::<_, 4>(simd, ms, seq_store, rep, block),
    })
}

/// `ZSTD_compressBlock_doubleFast_extDict_generic` for one minimum match length,
/// compiled for one SIMD level.
#[simd]
fn compress_block_ext_dict_mls<S: Simd, const MLS: u32>(
    simd: S,
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
) -> usize {
    let cparams = ms.cparams;
    let h_bits_l = cparams.hash_log;
    let h_bits_s = cparams.chain_log;
    let window = block.window;
    let prefix = window.prefix;
    let ext = window.ext;
    let low_limit = ms.lowest_match_index(block.end, cparams.window_log);
    let dict_start_index = low_limit;
    let dict_limit = ms.window.dict_limit;
    let prefix_start_index = dict_limit.max(low_limit);
    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];

    if prefix_start_index == dict_start_index {
        return compress_block(ms, seq_store, rep, block);
    }

    // Prefix positions are offsets into `src`, whose first byte has window
    // index `base` (the reference's `base` pointer); candidates below the
    // prefix are read from the external segment by window index.
    let base = prefix.start();
    let src = prefix.slice(base, block.end);
    let istart = (block.start - base) as usize;
    let iend = src.len();
    // A block shorter than eight bytes never enters the search.
    let ilimit = iend.saturating_sub(8);
    let index = |p: usize| p as u32 + base;
    // Hashes are below the table sizes; the masks prove the indices in
    // bounds once the tables are known to be non-empty.
    assert!(!ms.hash_table.is_empty() && !ms.chain_table.is_empty());
    let (mask_l, mask_s) = (ms.hash_table.len() - 1, ms.chain_table.len() - 1);
    let hash_l = |p: usize| hash_word::<8>(load64(src, p), h_bits_l) & mask_l;
    let hash_s = |p: usize| hash_word::<MLS>(load64(src, p), h_bits_s) & mask_s;
    let read32 = |p: usize| load32(src, p);
    let read64 = |p: usize| load64(src, p);
    let in_dict = |idx: u32| idx < prefix_start_index;
    let any32 = |idx: u32| {
        if in_dict(idx) {
            ext.read32(idx)
        } else {
            load32(src, (idx - base) as usize)
        }
    };
    let any64 = |idx: u32| {
        if in_dict(idx) {
            ext.read64(idx)
        } else {
            load64(src, (idx - base) as usize)
        }
    };
    let any_byte = |idx: u32| {
        if in_dict(idx) {
            ext.byte(idx)
        } else {
            src[(idx - base) as usize]
        }
    };
    // Forward match length of offset `ip` against window index `m`, across
    // the segment boundary.
    let count_any = |ip: usize, m: u32| {
        if in_dict(m) {
            window.count_2segments(
                index(ip),
                m,
                block.end,
                prefix_start_index,
                prefix_start_index,
            )
        } else {
            count_slices_simd(simd, &src[ip..], &src[(m - base) as usize..])
        }
    };
    // Backward extension of a match at window index `m` for offset `ip`.
    let catch_up = |ip: &mut usize, m: &mut u32, m_length: &mut usize, anchor: usize, low: u32| {
        while *ip > anchor && *m > low && src[*ip - 1] == any_byte(*m - 1) {
            *ip -= 1;
            *m -= 1;
            *m_length += 1;
        }
    };

    let MatchState {
        hash_table: hash_long,
        chain_table: hash_small,
        ..
    } = ms;
    let hash_long = hash_long.as_mut_slice();
    let hash_small = hash_small.as_mut_slice();

    let mut ip = istart;
    let mut anchor = istart;
    while ip < ilimit {
        let h_small = hash_s(ip);
        let match_index = hash_small[h_small];
        let h_long = hash_l(ip);
        let match_long_index = hash_long[h_long];
        let curr = index(ip);
        let rep_index = (curr + 1).wrapping_sub(offset_1);
        hash_small[h_small] = curr;
        hash_long[h_long] = curr;

        let m_length;
        let overlap_ok = prefix_start_index.wrapping_sub(1).wrapping_sub(rep_index) >= 3;
        if overlap_ok
            && offset_1 <= curr + 1 - dict_start_index
            && any32(rep_index) == read32(ip + 1)
        {
            m_length = count_any(ip + 1 + 4, rep_index + 4) + 4;
            ip += 1;
            seq_store.store_seq_from(src, anchor, ip - anchor, repcode_to_offbase(1), m_length);
        } else if match_long_index > dict_start_index && any64(match_long_index) == read64(ip) {
            let low = if in_dict(match_long_index) {
                dict_start_index
            } else {
                prefix_start_index
            };
            let mut ml = count_any(ip + 8, match_long_index + 8) + 8;
            let offset = curr - match_long_index;
            let mut m = match_long_index;
            catch_up(&mut ip, &mut m, &mut ml, anchor, low);
            offset_2 = offset_1;
            offset_1 = offset;
            seq_store.store_seq_from(src, anchor, ip - anchor, offset_to_offbase(offset), ml);
            m_length = ml;
        } else if match_index > dict_start_index && any32(match_index) == read32(ip) {
            let h3 = hash_l(ip + 1);
            let match_index3 = hash_long[h3];
            hash_long[h3] = curr + 1;
            let offset;
            let mut ml;
            if match_index3 > dict_start_index && any64(match_index3) == read64(ip + 1) {
                let low = if in_dict(match_index3) {
                    dict_start_index
                } else {
                    prefix_start_index
                };
                ml = count_any(ip + 9, match_index3 + 8) + 8;
                ip += 1;
                offset = curr + 1 - match_index3;
                let mut m = match_index3;
                catch_up(&mut ip, &mut m, &mut ml, anchor, low);
            } else {
                let low = if in_dict(match_index) {
                    dict_start_index
                } else {
                    prefix_start_index
                };
                ml = count_any(ip + 4, match_index + 4) + 4;
                offset = curr - match_index;
                let mut m = match_index;
                catch_up(&mut ip, &mut m, &mut ml, anchor, low);
            }
            offset_2 = offset_1;
            offset_1 = offset;
            seq_store.store_seq_from(src, anchor, ip - anchor, offset_to_offbase(offset), ml);
            m_length = ml;
        } else {
            ip += ((ip - anchor) >> SEARCH_STRENGTH) + 1;
            continue;
        }

        ip += m_length;
        anchor = ip;
        if ip <= ilimit {
            let index_to_insert = curr + 2;
            let insert_at = (index_to_insert - base) as usize;
            hash_long[hash_l(insert_at)] = index_to_insert;
            hash_long[hash_l(ip - 2)] = index(ip - 2);
            hash_small[hash_s(insert_at)] = index_to_insert;
            hash_small[hash_s(ip - 1)] = index(ip - 1);

            while ip <= ilimit {
                let current2 = index(ip);
                let rep_index2 = current2.wrapping_sub(offset_2);
                let overlap_ok = prefix_start_index.wrapping_sub(1).wrapping_sub(rep_index2) >= 3;
                if overlap_ok
                    && offset_2 <= current2 - dict_start_index
                    && any32(rep_index2) == read32(ip)
                {
                    let rep_length2 = count_any(ip + 4, rep_index2 + 4) + 4;
                    core::mem::swap(&mut offset_1, &mut offset_2);
                    seq_store.store_seq_only(0, repcode_to_offbase(1), rep_length2);
                    hash_small[hash_s(ip)] = current2;
                    hash_long[hash_l(ip)] = current2;
                    ip += rep_length2;
                    anchor = ip;
                    continue;
                }
                break;
            }
        }
    }

    rep[0] = offset_1;
    rep[1] = offset_2;
    iend - anchor
}

/// `ZSTD_compressBlock_doubleFast_dictMatchState`: selects the variant
/// compiled for the minimum match length, dispatched once per block.
///
/// A block without an attached dictionary is compressed as
/// [`compress_block`] would.
pub(crate) fn compress_block_dict_match_state(
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
) -> usize {
    let Some(dms) = block.dms else {
        return compress_block(ms, seq_store, rep, block);
    };
    fearless_simd::dispatch!(ms.backend.0, simd => match ms.cparams.min_match {
        5 => compress_block_dms_mls::<_, 5>(simd, ms, seq_store, rep, block, &dms),
        6 => compress_block_dms_mls::<_, 6>(simd, ms, seq_store, rep, block, &dms),
        7.. => compress_block_dms_mls::<_, 7>(simd, ms, seq_store, rep, block, &dms),
        _ => compress_block_dms_mls::<_, 4>(simd, ms, seq_store, rep, block, &dms),
    })
}

/// Where the short candidate a `dfast` dictMatchState search settles on
/// lies.
enum ShortMatch {
    /// At this offset of the prefix.
    Prefix(usize),
    /// At this dictionary index.
    Dict(u32),
}

/// `ZSTD_compressBlock_doubleFast_dictMatchState_generic` for one minimum
/// match length, compiled for one SIMD level.
///
/// Positions are offsets into the prefix from its lowest valid index;
/// candidates below it are read from the dictionary.
#[simd]
fn compress_block_dms_mls<S: Simd, const MLS: u32>(
    simd: S,
    ms: &mut MatchState,
    seq_store: &mut SeqStore,
    rep: &mut [u32; 3],
    block: &BlockInput<'_>,
    dms: &DictMatchState<'_>,
) -> usize {
    let cparams = ms.cparams;
    let h_bits_l = cparams.hash_log;
    let h_bits_s = cparams.chain_log;
    // With a dictionary attached this is the context's `dictLimit`.
    let prefix_lowest_index = ms.lowest_prefix_index(block.end, cparams.window_log);
    let src = block.window.prefix.slice(prefix_lowest_index, block.end);
    let istart = (block.start - prefix_lowest_index) as usize;
    let iend = src.len();
    let index = |p: usize| p as u32 + prefix_lowest_index;
    let read32 = |p: usize| load32(src, p);
    let read64 = |p: usize| load64(src, p);
    let count = |a: usize, b: usize| count_at_simd(simd, src, a, b);
    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];

    let dict = DictBytes::new(dms, prefix_lowest_index);
    let dict_long = dms.ms.hash_table.as_slice();
    let dict_small = dms.ms.chain_table.as_slice();
    let dict_h_bits_l = dms.ms.cparams.hash_log + SHORT_CACHE_TAG_BITS;
    let dict_h_bits_s = dms.ms.cparams.chain_log + SHORT_CACHE_TAG_BITS;
    let dict_and_prefix_length = istart as u32 + dict.bytes.len() as u32;

    // Hashes are below the tables' sizes; masking with `len - 1` changes
    // nothing and, with the tables known to be non-empty, proves every
    // index in bounds.
    assert!(
        !ms.hash_table.is_empty()
            && !ms.chain_table.is_empty()
            && !dict_long.is_empty()
            && !dict_small.is_empty()
    );
    let (mask_l, mask_s) = (ms.hash_table.len() - 1, ms.chain_table.len() - 1);
    let (dict_mask_l, dict_mask_s) = (dict_long.len() - 1, dict_small.len() - 1);
    let MatchState {
        hash_table: hash_long,
        chain_table: hash_small,
        ..
    } = ms;
    let hash_long = hash_long.as_mut_slice();
    let hash_small = hash_small.as_mut_slice();
    let hash_l = |p: usize| hash_word::<8>(load64(src, p), h_bits_l) & mask_l;
    let hash_s = |p: usize| hash_word::<MLS>(load64(src, p), h_bits_s) & mask_s;
    let dict_entry_l = |hash_and_tag: usize| {
        let entry = dict_long[(hash_and_tag >> SHORT_CACHE_TAG_BITS) & dict_mask_l];
        (
            entry >> SHORT_CACHE_TAG_BITS,
            tags_match(entry, hash_and_tag),
        )
    };
    let dict_entry_s = |hash_and_tag: usize| {
        let entry = dict_small[(hash_and_tag >> SHORT_CACHE_TAG_BITS) & dict_mask_s];
        (
            entry >> SHORT_CACHE_TAG_BITS,
            tags_match(entry, hash_and_tag),
        )
    };
    // Whether the four bytes at the repcode candidate `rep_index`, in the
    // dictionary or the prefix, are `word`.
    let rep_matches = |rep_index: u32, word: u32| {
        if rep_index < prefix_lowest_index {
            super::mem::matches32(dict.bytes, dict.offset_of_context(rep_index), word)
        } else {
            super::mem::matches32(src, rep_index - prefix_lowest_index, word)
        }
    };
    // Forward length at `ip` of the repcode candidate `rep_index`.
    let rep_count = |ip: usize, rep_index: u32| {
        if rep_index < prefix_lowest_index {
            dict.count_2segments(simd, src, ip, dict.offset_of_context(rep_index) as usize)
        } else {
            count(ip, (rep_index - prefix_lowest_index) as usize)
        }
    };
    // Backward extension of a match at offset `m` of `history` for the
    // position `ip`.
    let catch_up =
        |ip: &mut usize, m: &mut usize, m_length: &mut usize, anchor: usize, history: &[u8]| {
            while *ip > anchor && *m > 0 && src[*ip - 1] == history[*m - 1] {
                *ip -= 1;
                *m -= 1;
                *m_length += 1;
            }
        };
    // Offset of the context index `curr` from the dictionary index `m`.
    let dict_offset = |curr: u32, m: u32| curr.wrapping_sub(m).wrapping_sub(dict.index_delta);

    let mut ip = istart;
    let mut anchor = istart;
    ip += usize::from(dict_and_prefix_length == 0);

    // The reference tests `ip < ilimit`, with `ilimit = iend - 8`.
    'outer: while ip + (HASH_READ_SIZE as usize) < iend {
        let word = read64(ip);
        let h2 = hash_word::<8>(word, h_bits_l) & mask_l;
        let h = hash_word::<MLS>(word, h_bits_s) & mask_s;
        let (dict_match_index_l, dict_tags_match_l) =
            dict_entry_l(hash_word::<8>(word, dict_h_bits_l));
        let (dict_match_index_s, dict_tags_match_s) =
            dict_entry_s(hash_word::<MLS>(word, dict_h_bits_s));
        let curr = index(ip);
        let match_index_l = hash_long[h2];
        let match_index_s = hash_small[h];
        let rep_index = (curr + 1).wrapping_sub(offset_1);
        hash_long[h2] = curr;
        hash_small[h] = curr;

        let m_length = 'stored: {
            if rep_index_ok(prefix_lowest_index, rep_index)
                && rep_matches(rep_index, read32(ip + 1))
            {
                let m_length = rep_count(ip + 1 + 4, rep_index + 4) + 4;
                ip += 1;
                seq_store.store_seq_from(src, anchor, ip - anchor, repcode_to_offbase(1), m_length);
                break 'stored m_length;
            }

            let (offset, m_length) = 'found: {
                if super::mem::matches64(src, match_index_l.wrapping_sub(prefix_lowest_index), word)
                {
                    let mut m = (match_index_l - prefix_lowest_index) as usize;
                    let mut m_length = count(ip + 8, m + 8) + 8;
                    let offset = (ip - m) as u32;
                    catch_up(&mut ip, &mut m, &mut m_length, anchor, src);
                    break 'found (offset, m_length);
                } else if dict_tags_match_l {
                    let m = dict.offset_of(dict_match_index_l);
                    if dict_match_index_l > dict.start_index
                        && super::mem::matches64(dict.bytes, m, word)
                    {
                        let mut m = m as usize;
                        let mut m_length = dict.count_2segments(simd, src, ip + 8, m + 8) + 8;
                        let offset = dict_offset(curr, dict_match_index_l);
                        catch_up(&mut ip, &mut m, &mut m_length, anchor, dict.bytes);
                        break 'found (offset, m_length);
                    }
                }

                let short = 'short: {
                    if match_index_s > prefix_lowest_index {
                        let m = (match_index_s - prefix_lowest_index) as usize;
                        if super::mem::matches32(src, m as u32, word as u32) {
                            break 'short ShortMatch::Prefix(m);
                        }
                    } else if dict_tags_match_s
                        && dict_match_index_s > dict.start_index
                        && super::mem::matches32(
                            dict.bytes,
                            dict.offset_of(dict_match_index_s),
                            word as u32,
                        )
                    {
                        break 'short ShortMatch::Dict(dict_match_index_s);
                    }
                    ip += ((ip - anchor) >> SEARCH_STRENGTH) + 1;
                    continue 'outer;
                };

                // _search_next_long:
                let word1 = read64(ip + 1);
                let hl3 = hash_word::<8>(word1, h_bits_l) & mask_l;
                let (dict_match_index_l3, dict_tags_match_l3) =
                    dict_entry_l(hash_word::<8>(word1, dict_h_bits_l));
                let match_index_l3 = hash_long[hl3];
                hash_long[hl3] = curr + 1;

                if super::mem::matches64(
                    src,
                    match_index_l3.wrapping_sub(prefix_lowest_index),
                    word1,
                ) {
                    let mut m = (match_index_l3 - prefix_lowest_index) as usize;
                    let mut m_length = count(ip + 9, m + 8) + 8;
                    ip += 1;
                    let offset = (ip - m) as u32;
                    catch_up(&mut ip, &mut m, &mut m_length, anchor, src);
                    break 'found (offset, m_length);
                } else if dict_tags_match_l3 {
                    let m = dict.offset_of(dict_match_index_l3);
                    if dict_match_index_l3 > dict.start_index
                        && super::mem::matches64(dict.bytes, m, word1)
                    {
                        let mut m = m as usize;
                        let mut m_length = dict.count_2segments(simd, src, ip + 1 + 8, m + 8) + 8;
                        ip += 1;
                        let offset = dict_offset(curr + 1, dict_match_index_l3);
                        catch_up(&mut ip, &mut m, &mut m_length, anchor, dict.bytes);
                        break 'found (offset, m_length);
                    }
                }

                // No long match at `ip + 1`: extend the short one.
                match short {
                    ShortMatch::Dict(dict_match_index) => {
                        let mut m = dict.offset_of(dict_match_index) as usize;
                        let mut m_length = dict.count_2segments(simd, src, ip + 4, m + 4) + 4;
                        let offset = dict_offset(curr, dict_match_index);
                        catch_up(&mut ip, &mut m, &mut m_length, anchor, dict.bytes);
                        (offset, m_length)
                    }
                    ShortMatch::Prefix(mut m) => {
                        let mut m_length = count(ip + 4, m + 4) + 4;
                        let offset = (ip - m) as u32;
                        catch_up(&mut ip, &mut m, &mut m_length, anchor, src);
                        (offset, m_length)
                    }
                }
            };

            // _match_found:
            offset_2 = offset_1;
            offset_1 = offset;
            seq_store.store_seq_from(
                src,
                anchor,
                ip - anchor,
                offset_to_offbase(offset),
                m_length,
            );
            m_length
        };

        // _match_stored:
        ip += m_length;
        anchor = ip;

        if ip + HASH_READ_SIZE as usize <= iend {
            // Complementary insertion, after the `ilimit` test because the
            // positions may lie past it.
            let index_to_insert = curr + 2;
            let insert_at = (index_to_insert - prefix_lowest_index) as usize;
            hash_long[hash_l(insert_at)] = index_to_insert;
            hash_long[hash_l(ip - 2)] = index(ip - 2);
            hash_small[hash_s(insert_at)] = index_to_insert;
            hash_small[hash_s(ip - 1)] = index(ip - 1);

            while ip + HASH_READ_SIZE as usize <= iend {
                let current2 = index(ip);
                let rep_index2 = current2.wrapping_sub(offset_2);
                if rep_index_ok(prefix_lowest_index, rep_index2)
                    && rep_matches(rep_index2, read32(ip))
                {
                    let rep_length2 = rep_count(ip + 4, rep_index2 + 4) + 4;
                    core::mem::swap(&mut offset_1, &mut offset_2);
                    seq_store.store_seq_only(0, repcode_to_offbase(1), rep_length2);
                    hash_small[hash_s(ip)] = current2;
                    hash_long[hash_l(ip)] = current2;
                    ip += rep_length2;
                    anchor = ip;
                    continue;
                }
                break;
            }
        }
    }

    rep[0] = offset_1;
    rep[1] = offset_2;
    iend - anchor
}

#[cfg(test)]
mod tests {
    use super::super::cctx::{Engine, RequestedParams, compress_bound};
    use super::super::cdict::AttachPref;
    use super::super::compress_once_into;
    use super::super::dict_tests::{inputs, request, trained};
    use super::super::fast::tests::{
        check_every_backend, check_stream, many_blocks, raw_dictionary, tiny_inputs,
    };
    use super::super::mem::{Mem, Window};
    use super::super::params::{ParamSwitch, Strategy, level_cparams};
    use super::*;
    use crate::Backend;
    use crate::dictionary::DictionaryContent;
    use alloc::vec;
    use zstd_ffi::{corpus, oracle};

    #[test]
    fn wild_literal_copies_store_what_the_shared_helper_stores() {
        // Runs up to 16 bytes take the chunk while it fits before the end
        // of `src`, longer runs and runs near the end the exact copy.
        let src: alloc::vec::Vec<u8> = (0..64u8).map(|byte| byte.wrapping_mul(37)).collect();
        for anchor in 0..src.len() {
            for lit_length in 0..=(src.len() - anchor).min(40) {
                let mut wild = SeqStore::default();
                let mut shared = SeqStore::default();
                wild.reserve(1 << 10, 4);
                shared.reserve(1 << 10, 4);
                wild.store_last_literals(&[0xAA]);
                shared.store_last_literals(&[0xAA]);
                store_seq_wild(&mut wild, &src, anchor, lit_length, 7, 5);
                shared.store_seq_from(&src, anchor, lit_length, 7, 5);
                assert_eq!(wild.literals(), shared.literals(), "{anchor} {lit_length}");
                assert_eq!(wild.literals()[1..], src[anchor..anchor + lit_length]);
                assert_eq!(
                    wild.sequences(),
                    shared.sequences(),
                    "{anchor} {lit_length}"
                );
            }
        }
    }

    /// Checks `Slots::<MLS, HL, HS>` over tables of `1 << bits_l` and
    /// `1 << bits_s` entries against the plain hashes, and its look-ahead
    /// load against `try_load64`.
    fn check_slots<const MLS: u32, const HL: u32, const HS: u32>(bits_l: u32, bits_s: u32) {
        let src = corpus::text(300, 7);
        let mut long = vec![0u32; 1 << bits_l];
        let mut small = vec![0u32; 1 << bits_s];
        for (i, slot) in long.iter_mut().enumerate() {
            *slot = i as u32;
        }
        for (i, slot) in small.iter_mut().enumerate() {
            *slot = !(i as u32);
        }
        let slots = Slots::<MLS, HL, HS> {
            long: Cell::from_mut(long.as_mut_slice()).as_slice_of_cells(),
            small: Cell::from_mut(small.as_mut_slice()).as_slice_of_cells(),
            h_bits_l: bits_l,
            h_bits_s: bits_s,
        };
        for p in 0..src.len() + 2 {
            let ahead = slots.ahead(&src, 3, p);
            let Some(word) = try_load64(&src, p) else {
                assert!(ahead.is_none(), "{p}");
                continue;
            };
            let long_slot = slots.long(word);
            assert_eq!(long_slot.get(), hash_word::<8>(word, bits_l) as u32);
            assert_eq!(
                slots.small(word).get(),
                !(hash_word::<MLS>(word, bits_s) as u32)
            );
            let Some((ahead_word, ahead_slot)) = ahead else {
                panic!("{p}: the look-ahead stopped before the end");
            };
            assert_eq!(ahead_word, word);
            assert!(core::ptr::eq(ahead_slot, long_slot));
        }
    }

    #[test]
    fn slots_hash_like_the_tables_with_compiled_and_run_time_widths() {
        check_slots::<5, 14, 14>(14, 14);
        check_slots::<4, 17, 16>(17, 16);
        check_slots::<6, 0, 0>(12, 11);
        check_slots::<4, 0, 0>(10, 9);
    }

    #[test]
    fn dfast_levels_attach_dictionaries_like_the_reference() {
        let (trained, records) = trained();
        let raw = raw_dictionary(&records);
        let inputs = inputs(&records);
        for level in 2..=4 {
            for pref in [AttachPref::ForceAttach, AttachPref::Default] {
                let requested = request(level, None, pref);
                check_every_backend(&requested, &trained, DictionaryContent::Auto, &inputs);
                check_every_backend(&requested, &raw, DictionaryContent::Auto, &inputs);
                check_every_backend(&requested, &trained, DictionaryContent::Raw, &inputs);
            }
        }
    }

    #[test]
    fn explicit_dfast_strategy_attaches_with_every_min_match() {
        let (trained, records) = trained();
        let raw = raw_dictionary(&records);
        let inputs = inputs(&records);
        for level in [-1, 1, 4] {
            for min_match in 4..=7 {
                let mut requested = request(level, Some(Strategy::DFast), AttachPref::ForceAttach);
                requested.cparams.min_match = min_match;
                check_every_backend(&requested, &trained, DictionaryContent::Auto, &inputs);
                check_every_backend(&requested, &raw, DictionaryContent::Raw, &inputs);
            }
        }
    }

    #[test]
    fn attached_dictionary_survives_and_expires_over_many_blocks() {
        let (trained, records) = trained();
        let big = many_blocks(&records);
        for window_log in [0, 17] {
            for level in [2, 4] {
                let mut requested = request(level, None, AttachPref::ForceAttach);
                requested.cparams.window_log = window_log;
                check_every_backend(&requested, &trained, DictionaryContent::Auto, &big);
            }
        }
    }

    #[test]
    fn tiny_inputs_attach_dictionaries_like_the_reference() {
        let (trained, records) = trained();
        let tiny = tiny_inputs(&records);
        for level in [2, 4] {
            for pref in [AttachPref::ForceAttach, AttachPref::Default] {
                let requested = request(level, None, pref);
                check_every_backend(&requested, &trained, DictionaryContent::Auto, &tiny);
            }
        }
    }

    #[test]
    fn streams_with_an_attached_dictionary_match_the_reference() {
        let (trained, records) = trained();
        let input = records[records.len() - 600..].concat();
        for level in [2, 4] {
            for chunk in [1_000, 7_001, 1 << 17] {
                let requested = request(level, None, AttachPref::Default);
                check_stream(&requested, &trained, DictionaryContent::Auto, &input, chunk);
            }
        }
    }

    #[test]
    fn large_inputs_compress_like_the_reference() {
        // Above 256 KiB, levels 3 and 4 use the large-input table widths
        // and fill several blocks of history.
        let input = [
            corpus::text(400_000, 11),
            corpus::binary(300_000, 12),
            corpus::random(60_000, 13),
            corpus::text(500_000, 14),
        ]
        .concat();
        for level in [3, 4] {
            let expected = oracle::compress(&input, level);
            let requested = RequestedParams {
                compression_level: level,
                ..RequestedParams::default()
            };
            for backend in Backend::available() {
                let mut engine = Engine::with_backend(backend);
                let mut dst = vec![0u8; compress_bound(input.len()).expect("bound")];
                let size = compress_once_into(&mut engine, &requested, None, &input, 1, &mut dst)
                    .expect("compresses");
                assert!(dst[..size] == expected[..], "{backend} level {level}");
            }
        }
    }

    #[test]
    fn every_compiled_width_pair_compresses_like_the_reference() {
        // One input size and level per `(min_match, hashLog, chainLog)`
        // that `compress_block_mls` compiles in, and one beside them that
        // takes the run-time widths.
        let cases = [
            (150_000, 2, (5, 14, 14)),
            (150_000, 3, (4, 16, 16)),
            (300_000, 3, (5, 17, 16)),
            (300_000, 4, (5, 18, 18)),
            (100_000, 3, (5, 16, 15)),
            (65_536, 4, (4, 17, 16)),
            (100_000, 4, (4, 17, 17)),
        ];
        for (size, level, widths) in cases {
            let cparams = level_cparams(level, size as u64);
            assert_eq!(
                cparams.strategy,
                Strategy::DFast,
                "{size} bytes, level {level}"
            );
            assert_eq!(
                (cparams.min_match, cparams.hash_log, cparams.chain_log),
                widths,
                "{size} bytes, level {level}"
            );
            let half = size / 2;
            let input = [corpus::text(half, 21), corpus::binary(size - half, 22)].concat();
            let expected = oracle::compress(&input, level);
            let requested = RequestedParams {
                compression_level: level,
                ..RequestedParams::default()
            };
            for backend in Backend::available() {
                let mut engine = Engine::with_backend(backend);
                let mut dst = vec![0u8; compress_bound(input.len()).expect("bound")];
                let size = compress_once_into(&mut engine, &requested, None, &input, 1, &mut dst)
                    .expect("compresses");
                assert!(
                    dst[..size] == expected[..],
                    "{backend} level {level} {widths:?}"
                );
            }
        }
    }

    #[test]
    fn without_a_dictionary_the_dict_match_state_entry_compresses_as_no_dict() {
        let input = corpus::text(20_000, 3);
        let cparams = level_cparams(3, input.len() as u64);
        let block = BlockInput {
            window: Window {
                prefix: Mem::new(&input, 2),
                ext: Mem::new(&[], 2),
            },
            start: 2,
            end: 2 + input.len() as u32,
            backend: Backend::default(),
            dms: None,
            symbol_costs: None,
        };
        let run = |compressor: super::super::match_state::BlockCompressor| {
            let mut ms = MatchState::new(cparams, ParamSwitch::Disable);
            let mut seq_store = SeqStore::default();
            seq_store.reserve(input.len(), cparams.min_match);
            let mut rep = [1, 4, 8];
            let literals = compressor(&mut ms, &mut seq_store, &mut rep, &block);
            (
                literals,
                rep,
                seq_store.sequences().to_vec(),
                seq_store.literals().to_vec(),
            )
        };
        let ours = run(compress_block_dict_match_state);
        assert!(!ours.2.is_empty());
        assert_eq!(ours, run(compress_block));
    }
}

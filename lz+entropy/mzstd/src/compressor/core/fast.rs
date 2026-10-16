//! The `fast` strategy (`zstd_fast.c`): one hash table, positions probed in
//! an interleaved pipeline with a step that grows over incompressible data.
//!
//! Positions are window indices held in `i64` so the reference's pointer
//! comparisons against `iend - 8` stay meaningful for blocks shorter than
//! eight bytes.

use core::cell::Cell;

use fearless_simd::Simd;
use fearless_simd_macros::simd;

use super::cdict::TableLoad;
use super::match_state::{BlockInput, DictMatchState, MatchState};
use super::mem::{
    common_suffix, count_at_simd, count_slices_simd, hash_at, hash_ptr, hash_word, load32, load64,
    probe_pair, probe_window, window_word,
};
use super::params::SHORT_CACHE_TAG_BITS;
use super::seq_store::{SeqStore, offset_to_offbase, repcode_to_offbase};
use super::window::HASH_READ_SIZE;

/// `kSearchStrength`.
const SEARCH_STRENGTH: u32 = 8;

/// `ZSTD_fillHashTableForCCtx` with `ZSTD_dtlm_fast`: indexes every third
/// position from `next_to_update` up to eight bytes before `end`.
pub(crate) fn fill_hash_table(ms: &mut MatchState, prefix: &super::mem::Mem<'_>, end: u32) {
    let h_bits = ms.cparams.hash_log;
    let mls = ms.cparams.min_match;
    let iend = i64::from(end) - i64::from(HASH_READ_SIZE);
    let mut ip = i64::from(ms.next_to_update);
    while ip + 3 < iend + 2 {
        let curr = ip as u32;
        ms.hash_table[hash_ptr(prefix, curr, h_bits, mls)] = curr;
        ip += 3;
    }
}

/// `ZSTD_fillHashTableForCCtx`: [`fill_hash_table`], and with
/// [`TableLoad::Full`] also the two positions after each indexed one where
/// their slot is still empty.
pub(crate) fn fill_hash_table_for(
    ms: &mut MatchState,
    prefix: &super::mem::Mem<'_>,
    end: u32,
    dtlm: TableLoad,
) {
    if dtlm == TableLoad::Fast {
        return fill_hash_table(ms, prefix, end);
    }
    let h_bits = ms.cparams.hash_log;
    let mls = ms.cparams.min_match;
    let iend = i64::from(end) - i64::from(HASH_READ_SIZE);
    let mut ip = i64::from(ms.next_to_update);
    while ip + 3 < iend + 2 {
        let curr = ip as u32;
        ms.hash_table[hash_ptr(prefix, curr, h_bits, mls)] = curr;
        for p in 1..3 {
            let hash = hash_ptr(prefix, curr + p, h_bits, mls);
            if ms.hash_table[hash] == 0 {
                ms.hash_table[hash] = curr + p;
            }
        }
        ip += 3;
    }
}

/// `ZSTD_fillHashTableForCDict` with `ZSTD_dtlm_full`: a dictionary's table
/// holds each index shifted past an 8-bit tag of its hash
/// (`ZSTD_writeTaggedIndex`), and indexes every third position plus the
/// ones in between whose slot is still empty.
pub(crate) fn fill_cdict_hash_table(ms: &mut MatchState, prefix: &super::mem::Mem<'_>, end: u32) {
    let h_bits = ms.cparams.hash_log + SHORT_CACHE_TAG_BITS;
    let mls = ms.cparams.min_match;
    let iend = i64::from(end) - i64::from(HASH_READ_SIZE);
    let mut ip = i64::from(ms.next_to_update);
    while ip + 3 < iend + 2 {
        let curr = ip as u32;
        write_tagged_index(
            &mut ms.hash_table,
            hash_ptr(prefix, curr, h_bits, mls),
            curr,
        );
        for p in 1..3 {
            let hash_and_tag = hash_ptr(prefix, curr + p, h_bits, mls);
            if ms.hash_table[hash_and_tag >> SHORT_CACHE_TAG_BITS] == 0 {
                write_tagged_index(&mut ms.hash_table, hash_and_tag, curr + p);
            }
        }
        ip += 3;
    }
}

/// `ZSTD_writeTaggedIndex`.
pub(crate) fn write_tagged_index(table: &mut [u32], hash_and_tag: usize, index: u32) {
    let tag = (hash_and_tag & ((1 << SHORT_CACHE_TAG_BITS) - 1)) as u32;
    table[hash_and_tag >> SHORT_CACHE_TAG_BITS] = (index << SHORT_CACHE_TAG_BITS) | tag;
}

/// `ZSTD_compressBlock_fast`: selects the variant compiled for the
/// minimum match length, as the reference's `ZSTD_GEN_FAST_FN` templates do.
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

/// `ZSTD_compressBlock_fast_noDict_generic` for one minimum match length,
/// compiled for one SIMD level.
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
    let prefix_start_index = ms.lowest_prefix_index(block.end, cparams.window_log);
    let mut rep_offset1 = rep[0];
    let mut rep_offset2 = rep[1];
    let mut offset_saved1 = 0u32;
    let mut offset_saved2 = 0u32;

    let mut ip0 = block.start;
    ip0 += u32::from(ip0 == prefix_start_index);
    {
        let window_low = ms.lowest_prefix_index(ip0, cparams.window_log);
        let max_rep = ip0 - window_low;
        if rep_offset2 > max_rep {
            offset_saved2 = rep_offset2;
            rep_offset2 = 0;
        }
        if rep_offset1 > max_rep {
            offset_saved1 = rep_offset1;
            rep_offset1 = 0;
        }
    }

    let table = ms.hash_table.as_mut_slice();
    let scan = |origin: u32| Scan {
        src: prefix.slice(origin, block.end),
        origin,
        prefix_start: (prefix_start_index - origin) as usize,
        start: (block.start - origin) as usize,
        hlog: cparams.hash_log,
        step_size: cparams.target_length as usize + usize::from(cparams.target_length == 0) + 1,
    };
    // Every repcode the search can use either is one of these two or was
    // found at a candidate inside the prefix, so when these two reach no
    // further back than the first probe the block reads nothing below the
    // prefix: it is then searched as offsets from the prefix's start, which
    // spares the search loop the window's base and the history's bounds.
    let in_prefix = block.start >= prefix_start_index
        && rep_offset1.max(rep_offset2) <= ip0 - prefix_start_index;
    let (literals, [offset_1, offset_2]) = if in_prefix {
        let scan = scan(prefix_start_index);
        let ip0 = (ip0 - prefix_start_index) as usize;
        let reps = [rep_offset1, rep_offset2];
        // A candidate is checked with branches once the history before the
        // block is several times the table's size: its entries are then
        // mostly inside the window, and both branches predict well. While
        // the table is still filling, a select spares the mispredictions.
        let warm = u64::from(block.start - prefix_start_index) >= 4 << cparams.hash_log;
        macro_rules! scan {
            ($hlog:literal) => {
                if warm {
                    compress_scan::<S, MLS, true, $hlog, true>(
                        simd, table, seq_store, &scan, ip0, reps,
                    )
                } else {
                    compress_scan::<S, MLS, true, $hlog, false>(
                        simd, table, seq_store, &scan, ip0, reps,
                    )
                }
            };
        }
        // The hash widths of the default parameters for large inputs
        // (negative levels, levels 1 and 2) are compiled in with their
        // match lengths: the hash shift and the table mask become
        // constants, which frees two of the loop's registers.
        match (MLS, cparams.hash_log) {
            (6, 13) => scan!(13),
            (7, 14) => scan!(14),
            (6, 16) => scan!(16),
            _ => scan!(0),
        }
    } else {
        let scan = scan(base);
        let ip0 = (ip0 - base) as usize;
        compress_scan::<S, MLS, false, 0, false>(
            simd,
            table,
            seq_store,
            &scan,
            ip0,
            [rep_offset1, rep_offset2],
        )
    };
    rep_offset1 = offset_1;
    rep_offset2 = offset_2;

    offset_saved2 = if offset_saved1 != 0 && rep_offset1 != 0 {
        offset_saved1
    } else {
        offset_saved2
    };
    rep[0] = if rep_offset1 != 0 {
        rep_offset1
    } else {
        offset_saved1
    };
    rep[1] = if rep_offset2 != 0 {
        rep_offset2
    } else {
        offset_saved2
    };
    literals
}

/// Whether the candidate at offset `at` of `history` starts with `word`:
/// with branches (`BRANCHY`) or with a select (see [`super::mem::matches32`]).
#[inline(always)]
fn candidate_matches<const BRANCHY: bool>(history: &[u8], at: u32, word: u32) -> bool {
    if BRANCHY {
        super::mem::matches32_branchy(history, at, word)
    } else {
        super::mem::matches32(history, at, word)
    }
}

/// The bytes one `fast` block searches and the parameters it searches them
/// with.
struct Scan<'a> {
    /// The block and the history before it, from window index `origin` on.
    src: &'a [u8],
    /// Window index of `src[0]`: the reference's `base`, as an offset.
    origin: u32,
    /// Offset of the prefix's lowest valid index, below which no candidate
    /// is taken.
    prefix_start: usize,
    /// Offset of the block's first byte.
    start: usize,
    /// `hashLog`, the table's hash width.
    hlog: u32,
    /// The reference's `stepSize`: the distance to `ip2` when a search
    /// starts.
    step_size: usize,
}

/// The search and match loop of `ZSTD_compressBlock_fast_noDict_generic`
/// from offset `ip0` of `scan.src`, with the repcodes `reps`. Returns the
/// length of the trailing literals and the final repcodes.
///
/// `IN_PREFIX` states that `scan.prefix_start` is zero, so the candidates'
/// history is all of `src` and its lower bound the window index `origin`.
#[inline(always)]
fn compress_scan<
    S: Simd,
    const MLS: u32,
    const IN_PREFIX: bool,
    const HLOG: u32,
    const BRANCHY: bool,
>(
    simd: S,
    table: &mut [u32],
    seq_store: &mut SeqStore,
    scan: &Scan<'_>,
    mut ip0: usize,
    reps: [u32; 2],
) -> (usize, [u32; 2]) {
    let Scan {
        src,
        origin: base,
        step_size,
        ..
    } = *scan;
    // `HLOG`, when not zero, is the table's hash width known at compile
    // time; the caller selects it only when `scan.hlog` equals it.
    let hlog = if HLOG == 0 { scan.hlog } else { HLOG };
    let prefix_start = if IN_PREFIX { 0 } else { scan.prefix_start };
    let prefix_start_index = base + prefix_start as u32;
    let history = &src[prefix_start..];
    let istart = scan.start;
    let iend = src.len();
    // A block shorter than HASH_READ_SIZE never enters the search: the
    // first `ip3 >= ilimit` test fails for any `ilimit <= 0`.
    let ilimit = iend.saturating_sub(HASH_READ_SIZE as usize);
    let index = |p: usize| p as u32 + base;
    let read32 = |p: usize| load32(src, p);
    let k_step_incr: usize = 1 << (SEARCH_STRENGTH - 1);

    let mut anchor = istart;
    let [mut rep_offset1, mut rep_offset2] = reps;

    // Hashes are below the table size (`1 << hash_log`); masking with
    // `len - 1` changes nothing and, with the table known to be non-empty,
    // proves every index in bounds.
    let mask = if HLOG == 0 {
        table.len().wrapping_sub(1)
    } else {
        (1 << HLOG) - 1
    };
    assert!(!table.is_empty() && table.len() > mask);
    // The search keeps references to the slots it will write rather than
    // their hashes, so a slot is bounds checked once, where its hash is
    // computed.
    let table = Cell::from_mut(table).as_slice_of_cells();
    let hash = |p: usize| hash_at::<MLS>(src, p, hlog) & mask;
    'start: loop {
        // The reference keeps `ip0..ip3`; at every probe `ip1` is `ip0 + 1`
        // and `ip3` is `ip2 + 1`, so only `ip0` and `ip2` are kept here,
        // which leaves the loop's registers to the values it needs.
        let mut step = step_size;
        let mut next_step = ip0 + k_step_incr;
        let mut ip2 = ip0 + step;
        // Every probe reads inside the nine bytes at `ip0` or at `ip2`
        // (`ip2 + 1 < ilimit` keeps both whole); `ip2`'s window becomes the
        // next `ip0`'s, so each step bound-checks one window, and that check
        // is the loop's exit test. The first two come from one check.
        let Some((mut w0, mut w2)) = probe_pair(src, ip0, step) else {
            break 'start;
        };
        let slot = |word: u64| &table[hash_word::<MLS>(word, hlog) & mask];
        let mut slot0 = slot(window_word::<0>(w0));
        let mut slot1 = slot(window_word::<1>(w0));
        let mut match_idx = slot0.get();
        let mut current0;

        // Which way the search found a match.
        enum Found {
            Rep,
            Offset,
        }
        let found = loop {
            // Probe `ip0` (and the repcode at `ip2`); `ip1` is `ip0 + 1`.
            let rval = read32(ip2 - rep_offset1 as usize);
            current0 = index(ip0);
            slot0.set(current0);
            if window_word::<0>(w2) as u32 == rval && rep_offset1 > 0 {
                slot1.set(current0 + 1);
                ip0 = ip2;
                break Found::Rep;
            }
            if candidate_matches::<BRANCHY>(
                history,
                match_idx.wrapping_sub(prefix_start_index),
                window_word::<0>(w0) as u32,
            ) {
                slot1.set(current0 + 1);
                break Found::Offset;
            }

            match_idx = slot1.get();
            slot0 = slot1;
            slot1 = slot(window_word::<0>(w2));
            ip0 += 1;

            // Probe `ip0` (the old `ip1`); `ip1` is now the old `ip2`.
            current0 = index(ip0);
            slot0.set(current0);
            if candidate_matches::<BRANCHY>(
                history,
                match_idx.wrapping_sub(prefix_start_index),
                window_word::<1>(w0) as u32,
            ) {
                if step <= 4 {
                    slot1.set(index(ip2));
                }
                break Found::Offset;
            }

            match_idx = slot1.get();
            slot0 = slot1;
            slot1 = slot(window_word::<1>(w2));
            ip0 = ip2;
            w0 = w2;
            ip2 = ip0 + step;
            if ip2 >= next_step {
                step += 1;
                next_step += k_step_incr;
            }
            match probe_window(src, ip2) {
                Some(window) => w2 = window,
                None => break 'start,
            }
        };

        let (offcode, mut m_length, match0);
        match found {
            Found::Rep => {
                let mut m0 = ip0 - rep_offset1 as usize;
                let back = usize::from(src[ip0 - 1] == src[m0 - 1]);
                ip0 -= back;
                m0 -= back;
                match0 = m0;
                offcode = repcode_to_offbase(1);
                m_length = back + 4;
            }
            Found::Offset => {
                let mut m0 = (match_idx - base) as usize;
                rep_offset2 = rep_offset1;
                rep_offset1 = (ip0 - m0) as u32;
                offcode = offset_to_offbase(rep_offset1);
                let back = common_suffix(&src[anchor..ip0], &src[prefix_start..m0]);
                ip0 -= back;
                m0 -= back;
                m_length = back + 4;
                match0 = m0;
            }
        }

        // _match:
        m_length += count_at_simd(simd, src, ip0 + m_length, match0 + m_length);
        seq_store.store_seq_from(src, anchor, ip0 - anchor, offcode, m_length);
        ip0 += m_length;
        anchor = ip0;

        if ip0 <= ilimit {
            let current2 = (current0 - base) as usize + 2;
            table[hash(current2)].set(current0 + 2);
            // The word two bytes back hashes `ip0 - 2` and holds the four
            // bytes at `ip0` that the first repcode check compares.
            let back2 = load64(src, ip0 - 2);
            table[hash_word::<MLS>(back2, hlog) & mask].set(index(ip0 - 2));
            let mut word = (back2 >> 16) as u32;
            if rep_offset2 > 0 {
                while word == read32(ip0 - rep_offset2 as usize) {
                    let r_length =
                        count_at_simd(simd, src, ip0 + 4, ip0 + 4 - rep_offset2 as usize) + 4;
                    core::mem::swap(&mut rep_offset1, &mut rep_offset2);
                    table[hash(ip0)].set(index(ip0));
                    ip0 += r_length;
                    seq_store.store_seq_only(0, repcode_to_offbase(1), r_length);
                    anchor = ip0;
                    if ip0 > ilimit {
                        break;
                    }
                    word = read32(ip0);
                }
            }
        }
    }

    (iend - anchor, [rep_offset1, rep_offset2])
}

/// `ZSTD_compressBlock_fast_extDict`: selects the variant compiled for
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

/// `ZSTD_compressBlock_fast_extDict_generic` for one minimum match length,
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
    let hlog = cparams.hash_log;
    let step_size = cparams.target_length as usize + usize::from(cparams.target_length == 0) + 1;
    let window = block.window;
    let prefix = window.prefix;
    let ext = window.ext;
    let end_index = block.end;
    let low_limit = ms.lowest_match_index(end_index, cparams.window_log);
    let dict_start_index = low_limit;
    let dict_limit = ms.window.dict_limit;
    let prefix_start_index = dict_limit.max(low_limit);
    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];
    let mut offset_saved1 = 0u32;
    let mut offset_saved2 = 0u32;

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
    // A block shorter than eight bytes never enters the search: the first
    // `ip3 >= ilimit` test fails for any `ilimit <= 0`.
    let ilimit = iend.saturating_sub(8);
    let index = |p: usize| p as u32 + base;
    // Hashes are below the table size; the mask proves the indices in
    // bounds once the table is known to be non-empty.
    assert!(!ms.hash_table.is_empty());
    let mask = ms.hash_table.len() - 1;
    let hash = |p: usize| hash_at::<MLS>(src, p, hlog) & mask;
    let read32 = |p: usize| load32(src, p);
    // Reads 4 bytes at a window index in either segment.
    let read_any = |idx: u32| {
        if idx < prefix_start_index {
            ext.read32(idx)
        } else {
            load32(src, (idx - base) as usize)
        }
    };
    let byte_any = |idx: u32| {
        if idx < prefix_start_index {
            ext.byte(idx)
        } else {
            src[(idx - base) as usize]
        }
    };
    // Forward match length of offset `ip` against window index `m`, across
    // the segment boundary when the match started (at `start`) in the
    // external segment.
    let count_any = |ip: usize, m: u32, start: u32| {
        if start < prefix_start_index {
            window.count_2segments(
                index(ip),
                m,
                end_index,
                prefix_start_index,
                prefix_start_index,
            )
        } else {
            count_slices_simd(simd, &src[ip..], &src[(m - base) as usize..])
        }
    };
    let k_step_incr: usize = 1 << (SEARCH_STRENGTH - 1);

    let mut ip0 = istart;
    let mut anchor = istart;
    {
        let curr = index(ip0);
        let max_rep = curr - dict_start_index;
        if offset_2 >= max_rep {
            offset_saved2 = offset_2;
            offset_2 = 0;
        }
        if offset_1 >= max_rep {
            offset_saved1 = offset_1;
            offset_1 = 0;
        }
    }

    let table = ms.hash_table.as_mut_slice();
    'start: loop {
        let mut step = step_size;
        let mut next_step = ip0 + k_step_incr;
        let mut ip1 = ip0 + 1;
        let mut ip2 = ip0 + step;
        let mut ip3 = ip2 + 1;
        if ip3 >= ilimit {
            break 'start;
        }
        let mut hash0 = hash(ip0);
        let mut hash1 = hash(ip1);
        let mut idx = table[hash0 & mask];
        let mut current0;

        enum Found {
            Rep(u32),
            Offset,
        }
        let found = loop {
            {
                let rep_index = index(ip2).wrapping_sub(offset_1);
                let rval = if prefix_start_index.wrapping_sub(rep_index) >= 4 && offset_1 > 0 {
                    read_any(rep_index)
                } else {
                    read32(ip2) ^ 1
                };
                current0 = index(ip0);
                table[hash0 & mask] = current0;
                if read32(ip2) == rval {
                    ip0 = ip2;
                    break Found::Rep(rep_index);
                }
            }
            {
                let mval = if idx >= dict_start_index {
                    read_any(idx)
                } else {
                    read32(ip0) ^ 1
                };
                if read32(ip0) == mval {
                    break Found::Offset;
                }
            }
            idx = table[hash1 & mask];
            hash0 = hash1;
            hash1 = hash(ip2);
            ip0 = ip1;
            ip1 = ip2;
            ip2 = ip3;
            current0 = index(ip0);
            table[hash0 & mask] = current0;
            {
                let mval = if idx >= dict_start_index {
                    read_any(idx)
                } else {
                    read32(ip0) ^ 1
                };
                if read32(ip0) == mval {
                    break Found::Offset;
                }
            }
            idx = table[hash1 & mask];
            hash0 = hash1;
            hash1 = hash(ip2);
            ip0 = ip1;
            ip1 = ip2;
            ip2 = ip0 + step;
            ip3 = ip1 + step;
            if ip2 >= next_step {
                step += 1;
                next_step += k_step_incr;
            }
            if ip3 >= ilimit {
                break 'start;
            }
        };

        let (offcode, mut m_length, mut match0): (u32, usize, u32);
        match found {
            Found::Rep(rep_index) => {
                let back = usize::from(src[ip0 - 1] == byte_any(rep_index - 1));
                ip0 -= back;
                match0 = rep_index - back as u32;
                offcode = repcode_to_offbase(1);
                m_length = back + 4;
            }
            Found::Offset => {
                let offset = current0 - idx;
                let low_match = if idx < prefix_start_index {
                    dict_start_index
                } else {
                    prefix_start_index
                };
                match0 = idx;
                offset_2 = offset_1;
                offset_1 = offset;
                offcode = offset_to_offbase(offset);
                m_length = 4;
                while ip0 > anchor && match0 > low_match && src[ip0 - 1] == byte_any(match0 - 1) {
                    ip0 -= 1;
                    match0 -= 1;
                    m_length += 1;
                }
            }
        }

        // _match:
        m_length += count_any(ip0 + m_length, match0 + m_length as u32, match0);
        seq_store.store_seq_from(src, anchor, ip0 - anchor, offcode, m_length);
        ip0 += m_length;
        anchor = ip0;

        if ip1 < ip0 {
            table[hash1 & mask] = index(ip1);
        }

        if ip0 <= ilimit {
            table[hash((current0 - base) as usize + 2)] = current0 + 2;
            table[hash(ip0 - 2)] = index(ip0 - 2);
            while ip0 <= ilimit {
                let rep_index2 = index(ip0).wrapping_sub(offset_2);
                let overlap_ok = prefix_start_index.wrapping_sub(1).wrapping_sub(rep_index2) >= 3;
                if overlap_ok && offset_2 > 0 && read_any(rep_index2) == read32(ip0) {
                    let rep_length2 = count_any(ip0 + 4, rep_index2 + 4, rep_index2) + 4;
                    core::mem::swap(&mut offset_1, &mut offset_2);
                    seq_store.store_seq_only(0, repcode_to_offbase(1), rep_length2);
                    table[hash(ip0)] = index(ip0);
                    ip0 += rep_length2;
                    anchor = ip0;
                    continue;
                }
                break;
            }
        }
    }

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
    iend - anchor
}

/// `ZSTD_compressBlock_fast_dictMatchState`: selects the variant compiled
/// for the minimum match length, dispatched once per block.
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

/// An attached dictionary's content as the fast and dfast dictMatchState
/// finders read it: the bytes from the dictionary's first valid index
/// (`dictStart`) to its end (`dictEnd`), seen just below the context's
/// prefix in the context's index space.
#[derive(Copy, Clone, Debug)]
pub(crate) struct DictBytes<'a> {
    /// `[dictStart, dictEnd)`.
    pub(crate) bytes: &'a [u8],
    /// The dictionary's own index of `bytes[0]` (`dictStartIndex`).
    pub(crate) start_index: u32,
    /// `dictIndexDelta`: a context index below the prefix is this much
    /// above the dictionary index of the same byte.
    pub(crate) index_delta: u32,
}

impl<'a> DictBytes<'a> {
    /// The dictionary `dms` attached below a prefix starting at context
    /// index `prefix_start_index`.
    pub(crate) fn new(dms: &DictMatchState<'a>, prefix_start_index: u32) -> Self {
        let content = dms.window.prefix;
        let start_index = dms.ms.window.dict_limit.max(content.start());
        let end_index = content.end();
        Self {
            bytes: content.slice(start_index, end_index),
            start_index,
            index_delta: prefix_start_index.wrapping_sub(end_index),
        }
    }

    /// The offset in [`Self::bytes`] of dictionary index `index`, wrapped
    /// to a large value below the dictionary's start.
    #[inline(always)]
    pub(crate) const fn offset_of(&self, index: u32) -> u32 {
        index.wrapping_sub(self.start_index)
    }

    /// The offset in [`Self::bytes`] of context index `index`, which lies
    /// below the prefix; wrapped to a large value below the dictionary.
    #[inline(always)]
    pub(crate) const fn offset_of_context(&self, index: u32) -> u32 {
        index
            .wrapping_sub(self.index_delta)
            .wrapping_sub(self.start_index)
    }

    /// `ZSTD_count_2segments` for a match at offset `m` of the dictionary
    /// against offset `ip` of `src`, the prefix from its first byte: a
    /// match reaching the dictionary's end continues at the prefix's start.
    #[inline(always)]
    pub(crate) fn count_2segments<S: Simd>(
        &self,
        simd: S,
        src: &[u8],
        ip: usize,
        m: usize,
    ) -> usize {
        let input = src.get(ip..).unwrap_or_default();
        let length = count_slices_simd(simd, input, self.bytes.get(m..).unwrap_or_default());
        if m + length != self.bytes.len() {
            return length;
        }
        length + count_slices_simd(simd, input.get(length..).unwrap_or_default(), src)
    }
}

/// `ZSTD_comparePackedTags`: whether the tag a tagged table entry carries
/// is the one of the hash that selected it.
#[inline(always)]
pub(crate) const fn tags_match(index_and_tag: u32, hash_and_tag: usize) -> bool {
    let mask = (1u32 << SHORT_CACHE_TAG_BITS) - 1;
    index_and_tag & mask == hash_and_tag as u32 & mask
}

/// `ZSTD_index_overlap_check`: a repcode candidate at `rep_index` is
/// either in the prefix or four whole bytes below it.
#[inline(always)]
pub(crate) const fn rep_index_ok(prefix_start_index: u32, rep_index: u32) -> bool {
    prefix_start_index.wrapping_sub(1).wrapping_sub(rep_index) >= 3
}

/// `ZSTD_compressBlock_fast_dictMatchState_generic` for one minimum match
/// length, compiled for one SIMD level.
///
/// Positions are offsets into the prefix, which starts at the context's
/// `dictLimit`; candidates below it are read from the dictionary.
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
    let hlog = cparams.hash_log;
    // Supports a step size of 0.
    let step_size = cparams.target_length as usize + usize::from(cparams.target_length == 0);
    let prefix_start_index = ms.window.dict_limit;
    let src = block.window.prefix.slice(prefix_start_index, block.end);
    let istart = (block.start - prefix_start_index) as usize;
    let iend = src.len();
    let index = |p: usize| p as u32 + prefix_start_index;
    let read32 = |p: usize| load32(src, p);
    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];

    let dict = DictBytes::new(dms, prefix_start_index);
    let dict_table = dms.ms.hash_table.as_slice();
    let dict_h_bits = dms.ms.cparams.hash_log + SHORT_CACHE_TAG_BITS;
    let dict_and_prefix_length = istart as u32 + dict.bytes.len() as u32;
    let k_step_incr: usize = 1 << SEARCH_STRENGTH;

    // Hashes are below the tables' sizes; masking with `len - 1` changes
    // nothing and, with the tables known to be non-empty, proves every
    // index in bounds.
    assert!(!ms.hash_table.is_empty() && !dict_table.is_empty());
    let mask = ms.hash_table.len() - 1;
    let dict_mask = dict_table.len() - 1;
    let table = ms.hash_table.as_mut_slice();
    let hash = |p: usize| hash_at::<MLS>(src, p, hlog) & mask;
    let dict_hash = |p: usize| hash_at::<MLS>(src, p, dict_h_bits);
    let dict_entry =
        |hash_and_tag: usize| dict_table[(hash_and_tag >> SHORT_CACHE_TAG_BITS) & dict_mask];
    // Whether the four bytes at the repcode candidate `rep_index`, in the
    // dictionary or the prefix, are `word`.
    let rep_matches = |rep_index: u32, word: u32| {
        if rep_index < prefix_start_index {
            super::mem::matches32(dict.bytes, dict.offset_of_context(rep_index), word)
        } else {
            super::mem::matches32(src, rep_index - prefix_start_index, word)
        }
    };
    // Forward length at `ip` of the repcode candidate `rep_index`.
    let rep_count = |ip: usize, rep_index: u32| {
        if rep_index < prefix_start_index {
            dict.count_2segments(simd, src, ip, dict.offset_of_context(rep_index) as usize)
        } else {
            count_at_simd(simd, src, ip, (rep_index - prefix_start_index) as usize)
        }
    };

    let mut ip0 = istart;
    let mut ip1 = ip0 + step_size;
    let mut anchor = istart;
    ip0 += usize::from(dict_and_prefix_length == 0);

    // The reference tests `ip1 <= ilimit`, with `ilimit = iend - 8`.
    'outer: while ip1 + HASH_READ_SIZE as usize <= iend {
        let m_length;
        let mut hash0 = hash(ip0);
        let dict_hash_and_tag0 = dict_hash(ip0);
        let mut dict_match_index_and_tag = dict_entry(dict_hash_and_tag0);
        let mut dict_tags_match = tags_match(dict_match_index_and_tag, dict_hash_and_tag0);
        let mut match_index = table[hash0];
        let mut curr = index(ip0);
        let mut step = step_size;
        let mut next_step = ip0 + k_step_incr;

        loop {
            let rep_index = (curr + 1).wrapping_sub(offset_1);
            let hash1 = hash(ip1);
            let dict_hash_and_tag1 = dict_hash(ip1);
            table[hash0] = curr;

            if rep_index_ok(prefix_start_index, rep_index)
                && rep_matches(rep_index, read32(ip0 + 1))
            {
                m_length = rep_count(ip0 + 1 + 4, rep_index + 4) + 4;
                ip0 += 1;
                seq_store.store_seq_from(
                    src,
                    anchor,
                    ip0 - anchor,
                    repcode_to_offbase(1),
                    m_length,
                );
                break;
            }

            if dict_tags_match {
                let dict_match_index = dict_match_index_and_tag >> SHORT_CACHE_TAG_BITS;
                let dict_offset = dict.offset_of(dict_match_index);
                // Dictionary matches are taken only where the regular
                // candidate is invalid, as the extDict parse would.
                if dict_match_index > dict.start_index
                    && super::mem::matches32(dict.bytes, dict_offset, read32(ip0))
                    && match_index <= prefix_start_index
                {
                    let offset = curr
                        .wrapping_sub(dict_match_index)
                        .wrapping_sub(dict.index_delta);
                    let mut dict_match = dict_offset as usize;
                    let mut length = dict.count_2segments(simd, src, ip0 + 4, dict_match + 4) + 4;
                    while ip0 > anchor
                        && dict_match > 0
                        && src[ip0 - 1] == dict.bytes[dict_match - 1]
                    {
                        ip0 -= 1;
                        dict_match -= 1;
                        length += 1;
                    }
                    offset_2 = offset_1;
                    offset_1 = offset;
                    seq_store.store_seq_from(
                        src,
                        anchor,
                        ip0 - anchor,
                        offset_to_offbase(offset),
                        length,
                    );
                    m_length = length;
                    break;
                }
            }

            if super::mem::matches32(
                src,
                match_index.wrapping_sub(prefix_start_index),
                read32(ip0),
            ) {
                let mut m = (match_index - prefix_start_index) as usize;
                let offset = (ip0 - m) as u32;
                let mut length = count_at_simd(simd, src, ip0 + 4, m + 4) + 4;
                while ip0 > anchor && m > 0 && src[ip0 - 1] == src[m - 1] {
                    ip0 -= 1;
                    m -= 1;
                    length += 1;
                }
                offset_2 = offset_1;
                offset_1 = offset;
                seq_store.store_seq_from(
                    src,
                    anchor,
                    ip0 - anchor,
                    offset_to_offbase(offset),
                    length,
                );
                m_length = length;
                break;
            }

            dict_match_index_and_tag = dict_entry(dict_hash_and_tag1);
            dict_tags_match = tags_match(dict_match_index_and_tag, dict_hash_and_tag1);
            match_index = table[hash1];

            if ip1 >= next_step {
                step += 1;
                next_step += k_step_incr;
            }
            ip0 = ip1;
            ip1 += step;
            if ip1 + HASH_READ_SIZE as usize > iend {
                break 'outer;
            }
            curr = index(ip0);
            hash0 = hash1;
        }

        ip0 += m_length;
        anchor = ip0;

        if ip0 + HASH_READ_SIZE as usize <= iend {
            // `curr + 2` may lie past `ilimit`, so it is inserted here.
            table[hash((curr - prefix_start_index) as usize + 2)] = curr + 2;
            table[hash(ip0 - 2)] = index(ip0 - 2);

            while ip0 + HASH_READ_SIZE as usize <= iend {
                let current2 = index(ip0);
                let rep_index2 = current2.wrapping_sub(offset_2);
                if rep_index_ok(prefix_start_index, rep_index2)
                    && rep_matches(rep_index2, read32(ip0))
                {
                    let rep_length2 = rep_count(ip0 + 4, rep_index2 + 4) + 4;
                    core::mem::swap(&mut offset_1, &mut offset_2);
                    seq_store.store_seq_only(0, repcode_to_offbase(1), rep_length2);
                    table[hash(ip0)] = current2;
                    ip0 += rep_length2;
                    anchor = ip0;
                    continue;
                }
                break;
            }
        }

        ip1 = ip0 + step_size;
    }

    rep[0] = offset_1;
    rep[1] = offset_2;
    iend - anchor
}

#[cfg(test)]
pub(crate) mod tests {
    use alloc::boxed::Box;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::super::cctx::{Engine, RequestedParams, compress_bound};
    use super::super::cdict::{AttachPref, CDict};
    use super::super::compress_once_into;
    use super::super::dict_tests::{dict_params, inputs, request, trained};
    use super::super::mem::{Mem, Window};
    use super::super::params::{ParamSwitch, Strategy, level_cparams};
    use super::super::stream::{EndOp, Stream};
    use super::*;
    use crate::Backend;
    use crate::dictionary::DictionaryContent;
    use zstd_ffi::corpus;
    use zstd_ffi::oracle::{self, Directive};

    /// Compresses every input with `dict` under `requested` on one reused
    /// engine per available backend and compares each frame with the
    /// reference's.
    pub(crate) fn check_every_backend(
        requested: &RequestedParams,
        dict: &[u8],
        content: DictionaryContent,
        inputs: &[Vec<u8>],
    ) {
        let cdict = CDict::new(Box::from(dict), content, requested).expect("prepares");
        let params = dict_params(requested, dict, content);
        let expected: Vec<Vec<u8>> = inputs
            .iter()
            .map(|input| oracle::compress2(input, &params).expect("oracle"))
            .collect();
        for backend in Backend::available() {
            let mut engine = Engine::with_backend(backend);
            for (n, (input, expected)) in inputs.iter().zip(&expected).enumerate() {
                let mut dst = vec![0u8; compress_bound(input.len()).expect("bound")];
                let size =
                    compress_once_into(&mut engine, requested, Some(&cdict), input, 7, &mut dst)
                        .expect("compresses");
                assert!(
                    dst[..size] == expected[..],
                    "{backend} input {n} ({} bytes), {requested:?}: {size} vs {} bytes",
                    input.len(),
                    expected.len()
                );
            }
        }
    }

    /// Streams `input` in chunks of `chunk` bytes with `dict` under
    /// `requested`, the size unknown, on every available backend, and
    /// compares with the reference's `ZSTD_compressStream2`.
    pub(crate) fn check_stream(
        requested: &RequestedParams,
        dict: &[u8],
        content: DictionaryContent,
        input: &[u8],
        chunk: usize,
    ) {
        let cdict = CDict::new(Box::from(dict), content, requested).expect("prepares");
        let mut steps: Vec<(&[u8], Directive)> = input
            .chunks(chunk)
            .map(|data| (data, Directive::Continue))
            .collect();
        steps.push((&[], Directive::End));
        let expected = oracle::compress_stream(&dict_params(requested, dict, content), &steps)
            .expect("oracle");
        for backend in Backend::available() {
            let mut engine = Engine::with_backend(backend);
            let mut state = Stream::new();
            let mut out = Vec::new();
            let mut buffer = vec![0u8; 1 << 17];
            for &(data, directive) in &steps {
                let op = match directive {
                    Directive::End => EndOp::End,
                    _ => EndOp::Continue,
                };
                let mut in_pos = 0;
                loop {
                    let step = state
                        .compress(
                            &mut engine,
                            requested,
                            Some(&cdict),
                            &mut buffer[..],
                            &data[in_pos..],
                            op,
                        )
                        .expect("stream");
                    in_pos += step.consumed;
                    out.extend_from_slice(&buffer[..step.produced]);
                    if in_pos == data.len() && (op != EndOp::End || step.remaining == 0) {
                        break;
                    }
                }
            }
            assert!(
                out == expected,
                "{backend} chunks of {chunk}, {requested:?}: {} vs {} bytes",
                out.len(),
                expected.len()
            );
        }
    }

    /// A raw-content dictionary: text and records unlike the trained one.
    pub(crate) fn raw_dictionary(records: &[Vec<u8>]) -> Vec<u8> {
        [corpus::text(20_000, 9), records[..100].concat()].concat()
    }

    /// Inputs of 0 to 40 bytes, below and around `HASH_READ_SIZE` and the
    /// minimum block size, cut from the records.
    pub(crate) fn tiny_inputs(records: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let record = records[records.len() - 1].as_slice();
        [0usize, 1, 3, 7, 8, 9, 10, 12, 15, 16, 17, 24, 33, 40]
            .into_iter()
            .map(|len| record[..len.min(record.len())].to_vec())
            .collect()
    }

    /// 300 KiB of records: several blocks, enough to pass a 128 KiB
    /// window beyond the dictionary.
    pub(crate) fn many_blocks(records: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let mut input = records[records.len() - 2400..].concat();
        input.truncate(300 << 10);
        vec![input]
    }

    #[test]
    fn fast_levels_attach_dictionaries_like_the_reference() {
        let (trained, records) = trained();
        let raw = raw_dictionary(&records);
        let inputs = inputs(&records);
        for level in -5..=1 {
            for pref in [AttachPref::ForceAttach, AttachPref::Default] {
                let requested = request(level, None, pref);
                check_every_backend(&requested, &trained, DictionaryContent::Auto, &inputs);
                check_every_backend(&requested, &raw, DictionaryContent::Auto, &inputs);
                check_every_backend(&requested, &trained, DictionaryContent::Raw, &inputs);
            }
        }
    }

    #[test]
    fn explicit_fast_strategy_attaches_with_every_min_match() {
        let (trained, records) = trained();
        let raw = raw_dictionary(&records);
        let inputs = inputs(&records);
        for level in [1, 3] {
            for min_match in 4..=7 {
                for target_length in [0, 1, 6] {
                    let mut requested =
                        request(level, Some(Strategy::Fast), AttachPref::ForceAttach);
                    requested.cparams.min_match = min_match;
                    requested.cparams.target_length = target_length;
                    check_every_backend(&requested, &trained, DictionaryContent::Auto, &inputs);
                    check_every_backend(&requested, &raw, DictionaryContent::Raw, &inputs);
                }
            }
        }
    }

    #[test]
    fn attached_dictionary_survives_and_expires_over_many_blocks() {
        let (trained, records) = trained();
        let big = many_blocks(&records);
        for window_log in [0, 17] {
            for level in [-3, 1] {
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
        for level in [-5, 1] {
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
        for level in [-2, 1] {
            for chunk in [1_000, 7_001, 1 << 17] {
                let requested = request(level, None, AttachPref::Default);
                check_stream(&requested, &trained, DictionaryContent::Auto, &input, chunk);
            }
        }
    }

    #[test]
    fn packed_tags_compare_only_the_low_bits() {
        assert!(tags_match(0x1234_56ab, 0x99ab));
        assert!(tags_match(0xab, 0xffff_ffab));
        assert!(!tags_match(0x1234_56ab, 0x99ac));
        assert!(!tags_match(0, 1));
    }

    #[test]
    fn repcode_candidates_must_not_straddle_the_prefix_start() {
        let prefix_start = 100;
        assert!(rep_index_ok(prefix_start, 96));
        assert!(!rep_index_ok(prefix_start, 97));
        assert!(!rep_index_ok(prefix_start, 99));
        assert!(rep_index_ok(prefix_start, 100));
        assert!(rep_index_ok(prefix_start, 150));
        assert!(rep_index_ok(prefix_start, 0));
    }

    /// A dictionary over `bytes` from index `start`, with `ms`'s tables.
    fn dictionary<'a>(bytes: &'a [u8], start: u32, ms: &'a mut MatchState) -> DictMatchState<'a> {
        ms.window.dict_limit = start;
        DictMatchState {
            ms,
            window: Window {
                prefix: Mem::new(bytes, start),
                ext: Mem::new(&[], start),
            },
        }
    }

    #[test]
    fn dictionary_bytes_map_context_indices_below_the_prefix() {
        let bytes = b"abcdefghij";
        let mut dict_ms = MatchState::new(level_cparams(1, 0), ParamSwitch::Disable);
        let dms = dictionary(bytes, 2, &mut dict_ms);
        // The dictionary ends at its index 12, just below context index 40.
        let dict = DictBytes::new(&dms, 40);
        assert_eq!(dict.bytes, bytes);
        assert_eq!(dict.start_index, 2);
        assert_eq!(dict.index_delta, 28);
        assert_eq!(dict.offset_of(2), 0);
        assert_eq!(dict.offset_of(11), 9);
        assert_eq!(dict.offset_of(1), u32::MAX);
        assert_eq!(dict.offset_of_context(30), 0);
        assert_eq!(dict.offset_of_context(39), 9);
        assert!(dict.offset_of_context(29) as usize >= bytes.len());
    }

    #[test]
    fn dictionary_matches_continue_into_the_prefix() {
        let bytes = b"xxxxabcd";
        let mut dict_ms = MatchState::new(level_cparams(1, 0), ParamSwitch::Disable);
        let dms = dictionary(bytes, 2, &mut dict_ms);
        let dict = DictBytes::new(&dms, 100);
        let simd = fearless_simd::Fallback::new();
        // "abcd" ends the dictionary, then the prefix's "efg" continues it.
        assert_eq!(dict.count_2segments(simd, b"efgQabcdefgZ", 4, 4), 7);
        // A mismatch inside the dictionary stops the count there.
        assert_eq!(dict.count_2segments(simd, b"efgQabXdefgZ", 4, 4), 2);
        // The input's end stops the count inside the dictionary.
        assert_eq!(dict.count_2segments(simd, b"ab", 0, 4), 2);
        // At the dictionary's end the count runs from the prefix start.
        assert_eq!(dict.count_2segments(simd, b"efgefh", 3, 8), 2);
    }

    #[test]
    fn large_inputs_take_the_default_table_widths_like_the_reference() {
        // Above 256 KiB, levels -5, -1, 1 and 2 select the compiled-in hash
        // widths; their first blocks check candidates with a select and,
        // once the history outgrows the table, later ones with branches.
        let input = [
            corpus::text(400_000, 11),
            corpus::binary(300_000, 12),
            corpus::random(60_000, 13),
            corpus::text(500_000, 14),
        ]
        .concat();
        for level in [-5, -1, 1, 2] {
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
    fn without_a_dictionary_the_dict_match_state_entry_compresses_as_no_dict() {
        let input = corpus::text(20_000, 3);
        let cparams = level_cparams(1, input.len() as u64);
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
            (literals, rep, seq_store.sequences, seq_store.literals)
        };
        let ours = run(compress_block_dict_match_state);
        assert!(!ours.2.is_empty());
        assert_eq!(ours, run(compress_block));
    }
}

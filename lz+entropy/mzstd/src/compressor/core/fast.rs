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
    count_at_simd, count_behind_simd, count_slices_simd, hash_at, hash_ptr, hash_word, load32,
    load64, prefetch_l1, probe_pair, probe_window, window_word,
};
use super::params::SHORT_CACHE_TAG_BITS;
use super::seq_store::{
    FULL_BLOCK_LITERALS, FULL_BLOCK_SEQUENCES, SeqCursor, SeqStore, SeqWrite, offset_to_offbase,
    repcode_to_offbase,
};
use super::window::{HASH_READ_SIZE, WINDOW_START_INDEX};

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
    compress_block_with_store::<S, MLS>(simd, ms, seq_store, rep, block)
}

#[inline(always)]
fn compress_block_with_store<S: Simd, const MLS: u32>(
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
            ($hlog:literal, $branchy:literal, $origin:expr) => {
                compress_scan_adaptive::<S, MLS, true, $hlog, $branchy, $origin>(
                    simd, table, seq_store, &scan, ip0, reps,
                )
            };
            ($hlog:literal, $branchy:literal) => {
                scan!($hlog, $branchy, 0)
            };
            ($hlog:literal) => {
                if warm {
                    scan!($hlog, true)
                } else {
                    scan!($hlog, false)
                }
            };
        }
        // The hash widths of the default parameters are compiled in with
        // their match lengths: the hash shift and the table mask become
        // constants, which frees two of the loop's registers. Large inputs
        // (negative levels, levels 1 and 2) get both candidate checks.
        // Inputs of at most 128 or 256 KiB (negative levels and level 1)
        // span one or two blocks, whose table is still filling or holds at
        // most twice the block's history: they take the select only, with
        // the window's first index compiled in as well, which a one-shot
        // frame's prefix starts at. That frees a third register and costs
        // one instance per pair. Any other prefix takes the generic loop.
        let first = prefix_start_index == WINDOW_START_INDEX;
        match (MLS, cparams.hash_log, warm, first) {
            (6, 13, ..) => scan!(13),
            (7, 14, ..) => scan!(14),
            (6, 16, ..) => scan!(16),
            (5, 12, _, true) => scan!(12, false, WINDOW_START_INDEX),
            (5, 13, _, true) => scan!(13, false, WINDOW_START_INDEX),
            (6, 14, _, true) => scan!(14, false, WINDOW_START_INDEX),
            _ => scan!(0),
        }
    } else {
        let scan = scan(base);
        let ip0 = (ip0 - base) as usize;
        compress_scan_adaptive::<S, MLS, false, 0, false, 0>(
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
/// with branches (`BRANCHY`), or without: the offset is clamped into the
/// history, so the load needs neither a branch nor a stand-in word, and a
/// clamped offset never counts as a match.
///
/// Both conditions are folded into one value tested once. Written as
/// `(loaded == word) & (at == clamped)`, LLVM split the range test back
/// into a branch on x86, mispredicted on every fourth to fifth probe of a
/// filling table: text at levels -5 to 2 took 5-17% longer.
#[inline(always)]
fn candidate_matches<const BRANCHY: bool>(history: &[u8], at: u32, word: u32) -> bool {
    if BRANCHY {
        return super::mem::matches32_branchy(history, at, word);
    }
    let at = at as usize;
    let Some(limit) = history.len().checked_sub(4) else {
        return false;
    };
    let clamped = at.min(limit);
    let bytes = history[clamped..]
        .first_chunk::<4>()
        .copied()
        .unwrap_or_default();
    let differs = u32::from_le_bytes(bytes) ^ word;
    (differs as usize | (at ^ clamped)) == 0
}

/// The bytes one `fast` block searches and the parameters it searches them
/// with.
#[derive(Clone, Copy)]
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

/// Starts with ordinary vector pushes. A dense first part promotes the
/// remaining search to initialized slices at an outer-loop boundary, after
/// the hash insertions and repeat-code chain have finished.
#[inline(always)]
fn compress_scan_adaptive<
    S: Simd,
    const MLS: u32,
    const IN_PREFIX: bool,
    const HLOG: u32,
    const BRANCHY: bool,
    const ORIGIN: u32,
>(
    simd: S,
    table: &mut [u32],
    store: &mut SeqStore,
    scan: &Scan<'_>,
    ip: usize,
    reps: [u32; 2],
) -> (usize, [u32; 2]) {
    let mut stopped = false;
    let eligible = scan.src.len() - scan.start == FULL_BLOCK_LITERALS
        && store.max_nb_seq == FULL_BLOCK_SEQUENCES
        && store.max_nb_lit >= FULL_BLOCK_LITERALS;
    if !eligible {
        return compress_scan::<S, SeqStore, MLS, IN_PREFIX, HLOG, BRANCHY, ORIGIN, false>(
            simd,
            table,
            store,
            scan,
            ip,
            reps,
            &mut stopped,
        );
    }
    let (remaining, reps) = compress_scan::<S, SeqStore, MLS, IN_PREFIX, HLOG, BRANCHY, ORIGIN, true>(
        simd,
        table,
        store,
        scan,
        ip,
        reps,
        &mut stopped,
    );
    if !stopped {
        return (remaining, reps);
    }
    let ip = scan.src.len() - remaining;
    let resume = Scan { start: ip, ..*scan };
    if remaining >= 1 << 16 {
        store.with_full_block_cursor(|cursor| {
            compress_cursor_scan::<S, MLS, IN_PREFIX, HLOG, BRANCHY, ORIGIN>(
                simd, table, cursor, &resume, ip, reps,
            )
        })
    } else {
        compress_scan::<S, SeqStore, MLS, IN_PREFIX, HLOG, BRANCHY, ORIGIN, false>(
            simd,
            table,
            store,
            &resume,
            ip,
            reps,
            &mut stopped,
        )
    }
}

/// The promoted loop has its own feature entry: the cursor callback may
/// otherwise remain out of line without inheriting its caller's ISA.
#[simd]
#[inline(always)]
fn compress_cursor_scan<
    S: Simd,
    const MLS: u32,
    const IN_PREFIX: bool,
    const HLOG: u32,
    const BRANCHY: bool,
    const ORIGIN: u32,
>(
    simd: S,
    table: &mut [u32],
    cursor: &mut SeqCursor<'_>,
    scan: &Scan<'_>,
    ip: usize,
    reps: [u32; 2],
) -> (usize, [u32; 2]) {
    let mut stopped = false;
    // A local cursor, whose address never escapes, keeps its counts in
    // registers: through `cursor` they would be stored on every sequence,
    // since the compiler cannot tell the stores into its slices apart from
    // them.
    let mut local = cursor.reborrow();
    let result = compress_scan::<S, SeqCursor<'_>, MLS, IN_PREFIX, HLOG, BRANCHY, ORIGIN, false>(
        simd,
        table,
        &mut local,
        scan,
        ip,
        reps,
        &mut stopped,
    );
    let counts = local.counts();
    cursor.set_counts(counts);
    result
}

/// The search and match loop of `ZSTD_compressBlock_fast_noDict_generic`
/// from offset `ip0` of `scan.src`, with the repcodes `reps`. Returns the
/// length of the trailing literals and the final repcodes.
///
/// `IN_PREFIX` states that `scan.prefix_start` is zero, so the candidates'
/// history is all of `src` and its lower bound the window index `origin`.
/// `HLOG` and `ORIGIN`, when not zero, are `scan.hlog` and `scan.origin`
/// compiled in; `BRANCHY` picks the candidate check
/// ([`candidate_matches`]).
#[inline(always)]
fn compress_scan<
    S: Simd,
    W: SeqWrite,
    const MLS: u32,
    const IN_PREFIX: bool,
    const HLOG: u32,
    const BRANCHY: bool,
    const ORIGIN: u32,
    const PAUSE: bool,
>(
    simd: S,
    table: &mut [u32],
    seq_store: &mut W,
    scan: &Scan<'_>,
    mut ip0: usize,
    reps: [u32; 2],
    stopped: &mut bool,
) -> (usize, [u32; 2]) {
    let Scan { src, step_size, .. } = *scan;
    // `ORIGIN`, when not zero, is `scan.origin` known at compile time; the
    // caller selects it only when the two are equal.
    debug_assert!(ORIGIN == 0 || ORIGIN == scan.origin);
    let base = if ORIGIN == 0 { scan.origin } else { ORIGIN };
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
                // `ip1` is `ip0 + 1`.
                prefetch_l1(src, ip0 + 1 + 64);
                prefetch_l1(src, ip0 + 1 + 128);
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
                // `ZSTD_compressBlock_fast`'s backward extension, as a loop
                // rather than over two cut slices: it rarely takes a step.
                let mut back = 0;
                while ip0 - back > anchor && m0 - back > prefix_start {
                    match (src.get(ip0 - back - 1), src.get(m0 - back - 1)) {
                        (Some(a), Some(b)) if a == b => back += 1,
                        _ => break,
                    }
                }
                ip0 -= back;
                m0 -= back;
                m_length = back + 4;
                match0 = m0;
            }
        }

        // _match:
        m_length += count_behind_simd(simd, src, ip0 + m_length, match0 + m_length);
        seq_store.store_seq_wild(src, anchor, ip0 - anchor, offcode, m_length);
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
        if PAUSE && seq_store.nb_seq() >= 1024 {
            *stopped = true;
            return (iend - anchor, [rep_offset1, rep_offset2]);
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

/// Whether the four bytes at offset `at` of `history`, which holds at
/// least one word, equal `word`. The offset is clamped to the last whole
/// word, so the load needs no branch, and a clamped offset never counts as
/// a match: unlike [`candidate_matches`], the history length is not tested
/// for a word on every probe.
///
/// # Panics
///
/// When `history` is shorter than four bytes.
#[inline(always)]
fn segment_matches(history: &[u8], at: u32, word: u32) -> bool {
    let at = at as usize;
    let clamped = at.min(history.len().wrapping_sub(4));
    let differs = load32(history, clamped) ^ word;
    (differs as usize | (at ^ clamped)) == 0
}

/// `ZSTD_count_2segments` for a match that starts in the external segment,
/// kept out of line so its loops do not share the extDict search loop's
/// register allocation (about 1% fewer instructions in that loop).
#[inline(never)]
fn count_from_ext(
    window: &super::mem::Window<'_>,
    ip: u32,
    m: u32,
    end: u32,
    prefix_start: u32,
) -> usize {
    window.count_2segments(ip, m, end, prefix_start, prefix_start)
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
    // Candidates below the prefix are searched in the external segment
    // from its lowest valid index on, the others in `src`. `locate` picks
    // the segment holding `idx` and its offset there with a select (the
    // reference's `idxBase` cmov): a probed entry is about as likely to sit
    // in either segment, so a branch on it is mispredicted often. An entry
    // below `dict_start_index` becomes a large offset, which the candidate
    // check rejects like the reference's `idx >= dictStartIndex` test.
    // Both segments hold a whole word, as `segment_matches` requires: an
    // external segment too short for one (whose words would all straddle
    // into the prefix) is replaced by `src` addressed from the prefix
    // start, where every index below the prefix is out of range.
    let ext_history = ext.slice(dict_start_index, prefix_start_index);
    let (ext_segment, ext_origin) = if ext_history.len() >= 4 {
        (ext_history, dict_start_index)
    } else {
        (src, prefix_start_index)
    };
    // Only the slice goes through the select. The origin is added as a
    // masked difference: given a select of both, LLVM turned the origin's
    // half back into a branch.
    let ext_shift = base.wrapping_sub(ext_origin);
    let locate = |idx: u32| {
        let in_ext = idx < prefix_start_index;
        let history = core::hint::select_unpredictable(in_ext, ext_segment, src);
        let shift = ext_shift & 0u32.wrapping_sub(u32::from(in_ext));
        (history, idx.wrapping_sub(base).wrapping_add(shift))
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
            count_from_ext(&window, index(ip), m, end_index, prefix_start_index)
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
                let (history, at) = locate(idx);
                if segment_matches(history, at, read32(ip0)) {
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
                let (history, at) = locate(idx);
                if segment_matches(history, at, read32(ip0)) {
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

        let (offcode, mut m_length, match0): (u32, usize, u32);
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
                offset_2 = offset_1;
                offset_1 = offset;
                offcode = offset_to_offbase(offset);
                // The match extends backwards within its own segment, down
                // to the segment's lowest valid index (offset 0 there).
                let (history, at) = locate(idx);
                let mut at = at as usize;
                let mut back = 0;
                while ip0 > anchor && at > 0 && src[ip0 - 1] == history[at - 1] {
                    ip0 -= 1;
                    at -= 1;
                    back += 1;
                }
                match0 = idx - back as u32;
                m_length = back + 4;
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

    #[test]
    fn segment_candidates_match_only_whole_words_inside_the_history() {
        let history = b"abcdefgh";
        let word = |at: usize| u32::from_le_bytes(history[at..at + 4].try_into().unwrap());
        assert!(segment_matches(history, 0, word(0)));
        assert!(segment_matches(history, 4, word(4)));
        assert!(!segment_matches(history, 4, word(3)));
        // Offsets past the last whole word are clamped onto it and rejected
        // even when the clamped word is the one sought.
        assert!(!segment_matches(history, 5, word(4)));
        assert!(!segment_matches(history, u32::MAX, word(4)));
        assert!(segment_matches(&history[..4], 0, word(0)));
    }

    #[test]
    fn ext_matches_continue_at_the_prefix_start() {
        // Indices 10..14 are the external segment, 14.. the prefix.
        let window = Window {
            prefix: Mem::new(b"cdxyzabcdQ", 14),
            ext: Mem::new(b"zzab", 10),
        };
        // "ab" at 12 continues as "cd" at 14; "abcd" at 19 matches four
        // bytes, then 'Q' differs from 'x'.
        assert_eq!(count_from_ext(&window, 19, 12, 24, 14), 4);
        // A run ending inside the external segment stops there.
        assert_eq!(count_from_ext(&window, 18, 10, 24, 14), 1);
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
    fn small_inputs_take_the_compiled_in_widths_like_the_reference() {
        // Up to 128 KiB the negative levels hash 5 bytes into 2^12 slots,
        // up to 256 KiB into 2^13 and level 1 6 bytes into 2^14; a frame
        // that starts the window takes those widths and its first index
        // compiled in, in both of its blocks. A reused engine continues the
        // window, so its second frame starts higher and takes the generic
        // loop: both must write the reference's frames.
        let inputs = [
            corpus::text(100_000, 21),
            corpus::text(200_000, 22),
            corpus::binary(250_000, 23),
            [corpus::random(70_000, 24), corpus::text(120_000, 25)].concat(),
        ];
        for level in [-5, -1, 1] {
            let requested = RequestedParams {
                compression_level: level,
                ..RequestedParams::default()
            };
            for input in &inputs {
                let expected = oracle::compress(input, level);
                for backend in Backend::available() {
                    let mut engine = Engine::with_backend(backend);
                    for frame in 0..2 {
                        let mut dst = vec![0u8; compress_bound(input.len()).expect("bound")];
                        let size = compress_once_into(
                            &mut engine,
                            &requested,
                            None,
                            input,
                            frame,
                            &mut dst,
                        )
                        .expect("compresses");
                        assert!(
                            dst[..size] == expected[..],
                            "{backend} level {level} {} bytes frame {frame}",
                            input.len()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn adaptive_cursor_matches_uninterrupted_vector_scan() {
        // This oracle never enters PAUSE or SeqCursor and calls no C code.
        // Compare parser decisions as well as the materialized sequences:
        // a different hash table or final repcode can change a later block.
        fn check<const IN_PREFIX: bool>(
            input: &[u8],
            start: usize,
            prefix_start: usize,
            block_size: usize,
            promotes: Option<bool>,
        ) {
            let logical = &input[..start + block_size];
            let scan = Scan {
                src: logical,
                origin: WINDOW_START_INDEX,
                prefix_start,
                start,
                hlog: 13,
                step_size: 2,
            };
            let ip = start + usize::from(start == prefix_start);
            let mut table = vec![0; 1 << scan.hlog];
            // Populate a preceding prefix as a streaming block would.
            for at in (prefix_start..start.saturating_sub(8)).step_by(3) {
                table[hash_at::<5>(logical, at, scan.hlog)] = scan.origin + at as u32;
            }
            let mut empty = SeqStore::default();
            empty.reserve(FULL_BLOCK_LITERALS, 5);
            for backend in Backend::available() {
                if let Some(promotes) = promotes {
                    let mut paused = empty.clone();
                    let mut paused_table = table.clone();
                    let mut stopped = false;
                    let (remaining, _) = fearless_simd::dispatch!(backend.0, simd =>
                        compress_scan::<_, SeqStore, 5, IN_PREFIX, 13, false, 0, true>(
                            simd, &mut paused_table, &mut paused, &scan, ip, [0, 0],
                            &mut stopped,
                        )
                    );
                    assert!(stopped, "{backend}: fixture must reach the pause");
                    assert!(paused.nb_seq() >= 1024, "{backend}");
                    assert_eq!(remaining >= 1 << 16, promotes, "{backend}: {remaining}");
                    assert_eq!(paused.max_nb_seq, FULL_BLOCK_SEQUENCES);
                    assert_eq!(paused.max_nb_lit, FULL_BLOCK_LITERALS);
                }
                let mut reference = empty.clone();
                let mut reference_table = table.clone();
                let mut stopped = false;
                let expected = fearless_simd::dispatch!(backend.0, simd =>
                    compress_scan::<_, SeqStore, 5, IN_PREFIX, 13, false, 0, false>(
                        simd, &mut reference_table, &mut reference, &scan, ip, [0, 0],
                        &mut stopped,
                    )
                );
                assert!(!stopped);
                let sequence_bytes: usize = (0..reference.nb_seq())
                    .map(|at| {
                        let lengths = reference.sequence_length(at);
                        (lengths.lit_length + lengths.match_length) as usize
                    })
                    .sum();
                assert_eq!(sequence_bytes + expected.0, block_size, "{backend}");

                let mut optimized = empty.clone();
                let mut optimized_table = table.clone();
                let actual = fearless_simd::dispatch!(backend.0, simd =>
                    compress_scan_adaptive::<_, 5, IN_PREFIX, 13, false, 0>(
                        simd, &mut optimized_table, &mut optimized, &scan, ip,
                        [0, 0],
                    )
                );
                assert_eq!(actual, expected, "{backend}");
                assert_eq!(optimized.sequences(), reference.sequences(), "{backend}");
                assert_eq!(optimized.literals(), reference.literals(), "{backend}");
                assert_eq!(optimized.long_length, reference.long_length, "{backend}");
                assert_eq!(optimized_table, reference_table, "{backend}");
            }
        }

        let alice = include_bytes!("../../../testdata/alice29.txt");
        // Dense from the start: promote after the first 1024 sequences.
        check::<true>(alice, 0, 0, FULL_BLOCK_LITERALS, Some(true));
        // A late dense region reaches 1024 after the 64 KiB cutoff and
        // resumes the vector writer instead of initializing a cursor.
        let mut late = corpus::random(70_000, 0x512c_0ffe);
        late.extend_from_slice(&alice[..FULL_BLOCK_LITERALS - late.len()]);
        check::<true>(&late, 0, 0, FULL_BLOCK_LITERALS, Some(false));
        // An incomplete block must retain the unpaused vector parser.
        check::<true>(alice, 0, 0, FULL_BLOCK_LITERALS - 1, None);
        // A later streaming block has readable history, a nonzero start,
        // and a clipped history lower bound; it can still use the cursor.
        let mut history = alice[..4096].to_vec();
        history.extend_from_slice(&alice[..FULL_BLOCK_LITERALS]);
        check::<false>(&history, 4096, 128, FULL_BLOCK_LITERALS, Some(true));
    }

    #[test]
    fn wild_literal_copies_store_what_the_shared_helper_stores() {
        // Runs up to 16 bytes take the chunk while it fits before the end
        // of `src`, longer runs and runs near the end the exact copy.
        let src: Vec<u8> = (0..64u8).map(|byte| byte.wrapping_mul(37)).collect();
        for anchor in 0..src.len() {
            for lit_length in 0..=(src.len() - anchor).min(40) {
                let mut wild = SeqStore::default();
                let mut shared = SeqStore::default();
                wild.reserve(1 << 10, 4);
                shared.reserve(1 << 10, 4);
                wild.store_last_literals(&[0xAA]);
                shared.store_last_literals(&[0xAA]);
                SeqWrite::store_seq_wild(&mut wild, &src, anchor, lit_length, 7, 5);
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

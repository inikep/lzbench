//! The sequences section of a compressed block
//! (`zstd_compress_sequences.c` and `ZSTD_buildSequencesStatistics`).

use fearless_simd::Simd;
use fearless_simd_macros::simd;

use super::bits::BitWriter;
use super::fse::{self, CState, CTable};
use super::{FseRepeat, FseTables};
use crate::Backend;
use crate::compressor::core::error::CoreError;
use crate::compressor::core::params::Strategy;
use crate::compressor::core::seq_store::{CODE_COUNT_SLOTS, SeqStore};
use crate::shared::format::{
    DEFAULT_MAX_OFF, EncodingType, LL_DEFAULT_NORM, LL_DEFAULT_NORM_LOG, LL_FSE_LOG, MAX_LL,
    MAX_ML, MAX_OFF, MAX_SEQ, ML_DEFAULT_NORM, ML_DEFAULT_NORM_LOG, ML_FSE_LOG, OF_DEFAULT_NORM,
    OF_DEFAULT_NORM_LOG, OFF_FSE_LOG,
};

/// `-log2(x / 256)` in 1/256 bits for `x` in `0..256`; `0` for `x == 0`.
const INVERSE_PROBABILITY_LOG256: [u32; 256] = [
    0, 2048, 1792, 1642, 1536, 1453, 1386, 1329, 1280, 1236, 1197, 1162, 1130, 1100, 1073, 1047,
    1024, 1001, 980, 960, 941, 923, 906, 889, 874, 859, 844, 830, 817, 804, 791, 779, 768, 756,
    745, 734, 724, 714, 704, 694, 685, 676, 667, 658, 650, 642, 633, 626, 618, 610, 603, 595, 588,
    581, 574, 567, 561, 554, 548, 542, 535, 529, 523, 517, 512, 506, 500, 495, 489, 484, 478, 473,
    468, 463, 458, 453, 448, 443, 438, 434, 429, 424, 420, 415, 411, 407, 402, 398, 394, 390, 386,
    382, 377, 373, 370, 366, 362, 358, 354, 350, 347, 343, 339, 336, 332, 329, 325, 322, 318, 315,
    311, 308, 305, 302, 298, 295, 292, 289, 286, 282, 279, 276, 273, 270, 267, 264, 261, 258, 256,
    253, 250, 247, 244, 241, 239, 236, 233, 230, 228, 225, 222, 220, 217, 215, 212, 209, 207, 204,
    202, 199, 197, 194, 192, 190, 187, 185, 182, 180, 178, 175, 173, 171, 168, 166, 164, 162, 159,
    157, 155, 153, 151, 149, 146, 144, 142, 140, 138, 136, 134, 132, 130, 128, 126, 123, 121, 119,
    117, 115, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 93, 91, 89, 87, 85, 83, 82, 80,
    78, 76, 74, 73, 71, 69, 67, 66, 64, 62, 61, 59, 57, 55, 54, 52, 50, 49, 47, 46, 44, 42, 41, 39,
    37, 36, 34, 33, 31, 30, 28, 26, 25, 23, 22, 20, 19, 17, 16, 14, 13, 11, 10, 8, 7, 5, 4, 2, 1,
];

/// The reference's `ERROR(GENERIC)` as a cost: larger than any real cost.
const COST_ERROR: usize = usize::MAX;

/// Static description of one sequence code alphabet.
#[derive(Copy, Clone, Debug)]
pub(crate) struct CodeKind {
    /// Accuracy log of a described table.
    pub(crate) fse_log: u32,
    /// Predefined distribution.
    pub(crate) default_norm: &'static [i16],
    /// Accuracy log of the predefined distribution.
    pub(crate) default_norm_log: u32,
    /// Largest code of the predefined distribution.
    pub(crate) default_max: u32,
    /// Largest code.
    pub(crate) max_code: u32,
}

/// Literal lengths.
pub(crate) const LITERAL_LENGTHS: CodeKind = CodeKind {
    fse_log: LL_FSE_LOG,
    default_norm: &LL_DEFAULT_NORM,
    default_norm_log: LL_DEFAULT_NORM_LOG,
    default_max: MAX_LL,
    max_code: MAX_LL,
};
/// Offset codes.
pub(crate) const OFFSETS: CodeKind = CodeKind {
    fse_log: OFF_FSE_LOG,
    default_norm: &OF_DEFAULT_NORM,
    default_norm_log: OF_DEFAULT_NORM_LOG,
    default_max: DEFAULT_MAX_OFF,
    max_code: MAX_OFF,
};
/// Match lengths.
pub(crate) const MATCH_LENGTHS: CodeKind = CodeKind {
    fse_log: ML_FSE_LOG,
    default_norm: &ML_DEFAULT_NORM,
    default_norm_log: ML_DEFAULT_NORM_LOG,
    default_max: MAX_ML,
    max_code: MAX_ML,
};

/// `ZSTD_useLowProbCount`.
const fn use_low_prob_count(nb_seq: usize) -> bool {
    nb_seq >= 2048
}

/// Header cost in bytes of describing `count` (`ZSTD_NCountCost`).
fn ncount_cost(count: &[u32], max: u32, nb_seq: usize, fse_log: u32) -> Result<usize, CoreError> {
    let mut wksp = [0u8; 512];
    let mut norm = [0i16; MAX_SEQ as usize + 1];
    let table_log = fse::optimal_table_log(fse_log, nb_seq, max);
    fse::normalize_count(
        &mut norm,
        table_log,
        count,
        nb_seq,
        max,
        use_low_prob_count(nb_seq),
    )?;
    fse::write_ncount(&mut wksp, &norm, max, table_log)
}

/// Entropy bound of `count` in bits (`ZSTD_entropyCost`).
fn entropy_cost(count: &[u32], max: u32, total: usize) -> usize {
    let mut cost: u32 = 0;
    for &c in &count[..=max as usize] {
        let mut norm = ((256 * c as usize) / total) as u32;
        if c != 0 && norm == 0 {
            norm = 1;
        }
        cost = cost.wrapping_add(c.wrapping_mul(INVERSE_PROBABILITY_LOG256[norm as usize]));
    }
    (cost >> 8) as usize
}

/// Cost in bits of `count` with an existing table (`ZSTD_fseBitCost`), or
/// [`COST_ERROR`] when the table cannot encode some present symbol.
pub(crate) fn fse_bit_cost(table: &CTable, count: &[u32], max: u32) -> usize {
    const ACCURACY_LOG: u32 = 8;
    if table.max_symbol_value < max {
        return COST_ERROR;
    }
    let table_log = table.table_log;
    let bad_cost = (table_log + 1) << ACCURACY_LOG;
    let mut cost = 0usize;
    for (s, &c) in count[..=max as usize].iter().enumerate() {
        let bit_cost = table.bit_cost(s as u32, ACCURACY_LOG);
        if c == 0 {
            continue;
        }
        if bit_cost >= bad_cost {
            return COST_ERROR;
        }
        cost += c as usize * bit_cost as usize;
    }
    cost >> ACCURACY_LOG
}

/// Cost in bits of `count` under the distribution `norm`
/// (`ZSTD_crossEntropyCost`).
pub(crate) fn cross_entropy_cost(
    norm: &[i16],
    accuracy_log: u32,
    count: &[u32],
    max: u32,
) -> usize {
    let shift = 8 - accuracy_log;
    let mut cost = 0usize;
    for s in 0..=max as usize {
        let norm_acc = if norm[s] != -1 { norm[s] as u32 } else { 1 };
        let norm256 = norm_acc << shift;
        cost += count[s] as usize * INVERSE_PROBABILITY_LOG256[norm256 as usize] as usize;
    }
    cost >> 8
}

/// Histogram of one code stream.
#[derive(Copy, Clone, Debug)]
pub(crate) struct CodeStats<'a> {
    pub(crate) count: &'a [u32],
    pub(crate) max: u32,
    pub(crate) most_frequent: usize,
    pub(crate) nb_seq: usize,
}

/// A code stream's mutable histogram, with what table construction needs of
/// the stream itself.
#[derive(Debug)]
struct Histogram<'a> {
    count: &'a mut [u32],
    max: u32,
    /// Number of codes in the stream.
    nb_seq: usize,
    /// The stream's last code.
    last: u8,
}

/// A code stream's histogram as `HIST_countFast_wksp` reports it: the
/// counts, the largest code present (`0` for none) and the largest count.
fn code_histogram(counts: &[u32; CODE_COUNT_SLOTS]) -> ([u32; CODE_COUNT_SLOTS], u32, usize) {
    let max = counts.iter().rposition(|&c| c != 0).unwrap_or(0);
    let most_frequent = counts.iter().copied().max().unwrap_or(0);
    (*counts, max as u32, most_frequent as usize)
}

/// Chooses how to encode one code stream (`ZSTD_selectEncodingType`).
pub(crate) fn select_encoding_type(
    repeat_mode: &mut FseRepeat,
    stats: CodeStats<'_>,
    kind: CodeKind,
    prev_table: &CTable,
    default_allowed: bool,
    strategy: Strategy,
) -> EncodingType {
    let CodeStats {
        count,
        max,
        most_frequent,
        nb_seq,
    } = stats;
    if most_frequent == nb_seq {
        *repeat_mode = FseRepeat::None;
        if default_allowed && nb_seq <= 2 {
            return EncodingType::Basic;
        }
        return EncodingType::Rle;
    }
    if strategy < Strategy::Lazy {
        if default_allowed {
            const STATIC_FSE_NB_SEQ_MAX: usize = 1000;
            let mult = 10 - strategy as usize;
            const BASE_LOG: usize = 3;
            let dynamic_fse_nb_seq_min = ((1usize << kind.default_norm_log) * mult) >> BASE_LOG;
            if *repeat_mode == FseRepeat::Valid && nb_seq < STATIC_FSE_NB_SEQ_MAX {
                return EncodingType::Repeat;
            }
            if nb_seq < dynamic_fse_nb_seq_min
                || most_frequent < (nb_seq >> (kind.default_norm_log - 1))
            {
                *repeat_mode = FseRepeat::None;
                return EncodingType::Basic;
            }
        }
    } else {
        let basic_cost = if default_allowed {
            cross_entropy_cost(kind.default_norm, kind.default_norm_log, count, max)
        } else {
            COST_ERROR
        };
        let repeat_cost = if *repeat_mode != FseRepeat::None {
            fse_bit_cost(prev_table, count, max)
        } else {
            COST_ERROR
        };
        let ncount = ncount_cost(count, max, nb_seq, kind.fse_log).unwrap_or(COST_ERROR);
        let compressed_cost = (ncount << 3).wrapping_add(entropy_cost(count, max, nb_seq));
        if basic_cost <= repeat_cost && basic_cost <= compressed_cost {
            *repeat_mode = FseRepeat::None;
            return EncodingType::Basic;
        }
        if repeat_cost <= compressed_cost {
            return EncodingType::Repeat;
        }
    }
    *repeat_mode = FseRepeat::Check;
    EncodingType::Compressed
}

/// Builds the table for the chosen encoding and writes its description
/// (`ZSTD_buildCTable`).
///
/// # Errors
///
/// Header writing and normalization failures, and
/// [`CoreError::DstSizeTooSmall`] when an RLE symbol does not fit.
fn build_ctable(
    dst: &mut [u8],
    next: &mut CTable,
    prev: &CTable,
    kind: CodeKind,
    encoding: EncodingType,
    histogram: Histogram<'_>,
) -> Result<usize, CoreError> {
    let Histogram {
        count,
        max,
        nb_seq,
        last,
    } = histogram;
    match encoding {
        EncodingType::Rle => {
            next.build_rle(max as u8);
            let Some(first) = dst.first_mut() else {
                return Err(CoreError::DstSizeTooSmall);
            };
            // Every code is the one present symbol, `max`.
            *first = max as u8;
            Ok(1)
        }
        EncodingType::Repeat => {
            next.clone_from(prev);
            Ok(0)
        }
        EncodingType::Basic => {
            next.build(kind.default_norm, kind.default_max, kind.default_norm_log);
            Ok(0)
        }
        EncodingType::Compressed => {
            let mut nb_seq_1 = nb_seq;
            let table_log = fse::optimal_table_log(kind.fse_log, nb_seq, max);
            let last = usize::from(last);
            if count[last] > 1 {
                count[last] -= 1;
                nb_seq_1 -= 1;
            }
            let mut norm = [0i16; MAX_SEQ as usize + 1];
            fse::normalize_count(
                &mut norm,
                table_log,
                count,
                nb_seq_1,
                max,
                use_low_prob_count(nb_seq_1),
            )?;
            let ncount_size = fse::write_ncount(dst, &norm, max, table_log)?;
            next.build(&norm, max, table_log);
            Ok(ncount_size)
        }
    }
}

/// Encoding choices of a block's three code streams
/// (`ZSTD_symbolEncodingTypeStats_t`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct EncodingStats {
    pub(crate) ll_type: EncodingType,
    pub(crate) of_type: EncodingType,
    pub(crate) ml_type: EncodingType,
    /// Bytes of table descriptions written.
    pub(crate) size: usize,
    /// Size of the last described table, for the 1.3.4 decoder workaround.
    pub(crate) last_count_size: usize,
}

/// Chooses encodings for the three code streams, builds their tables into
/// `next` and writes their descriptions (`ZSTD_buildSequencesStatistics`).
///
/// The store must hold at least one sequence; its codes are (re)computed
/// with `backend`.
///
/// # Errors
///
/// Whatever table construction reports.
pub(crate) fn build_statistics(
    store: &mut SeqStore,
    prev: &FseTables,
    next: &mut FseTables,
    dst: &mut [u8],
    strategy: Strategy,
    backend: Backend,
) -> Result<EncodingStats, CoreError> {
    let nb_seq = store.nb_seq();
    debug_assert!(nb_seq != 0);
    store.fill_codes(backend);
    let mut op = 0usize;
    let mut last_count_size = 0usize;
    let last = store.codes[nb_seq - 1];

    // Literal lengths.
    let (mut count, max, most_frequent) = code_histogram(&store.code_counts.ll);
    next.litlength_repeat = prev.litlength_repeat;
    let ll_type = select_encoding_type(
        &mut next.litlength_repeat,
        CodeStats {
            count: &count,
            max,
            most_frequent,
            nb_seq,
        },
        LITERAL_LENGTHS,
        &prev.litlength,
        true,
        strategy,
    );
    let size = build_ctable(
        &mut dst[op..],
        &mut next.litlength,
        &prev.litlength,
        LITERAL_LENGTHS,
        ll_type,
        Histogram {
            count: &mut count,
            max,
            nb_seq,
            last: last.ll,
        },
    )?;
    if ll_type == EncodingType::Compressed {
        last_count_size = size;
    }
    op += size;

    // Offsets.
    let (mut count, max, most_frequent) = code_histogram(&store.code_counts.of);
    let default_allowed = max <= DEFAULT_MAX_OFF;
    next.offcode_repeat = prev.offcode_repeat;
    let of_type = select_encoding_type(
        &mut next.offcode_repeat,
        CodeStats {
            count: &count,
            max,
            most_frequent,
            nb_seq,
        },
        OFFSETS,
        &prev.offcode,
        default_allowed,
        strategy,
    );
    let size = build_ctable(
        &mut dst[op..],
        &mut next.offcode,
        &prev.offcode,
        OFFSETS,
        of_type,
        Histogram {
            count: &mut count,
            max,
            nb_seq,
            last: last.of,
        },
    )?;
    if of_type == EncodingType::Compressed {
        last_count_size = size;
    }
    op += size;

    // Match lengths.
    let (mut count, max, most_frequent) = code_histogram(&store.code_counts.ml);
    next.matchlength_repeat = prev.matchlength_repeat;
    let ml_type = select_encoding_type(
        &mut next.matchlength_repeat,
        CodeStats {
            count: &count,
            max,
            most_frequent,
            nb_seq,
        },
        MATCH_LENGTHS,
        &prev.matchlength,
        true,
        strategy,
    );
    let size = build_ctable(
        &mut dst[op..],
        &mut next.matchlength,
        &prev.matchlength,
        MATCH_LENGTHS,
        ml_type,
        Histogram {
            count: &mut count,
            max,
            nb_seq,
            last: last.ml,
        },
    )?;
    if ml_type == EncodingType::Compressed {
        last_count_size = size;
    }
    op += size;

    Ok(EncodingStats {
        ll_type,
        of_type,
        ml_type,
        size: op,
        last_count_size,
    })
}

/// Writes the sequences bitstream (`ZSTD_encodeSequences`), backward from
/// the last sequence, with the loop compiled for `backend` (the reference
/// selects its BMI2 build at run time the same way).
///
/// # Errors
///
/// [`CoreError::DstSizeTooSmall`] when the stream does not fit, and
/// [`CoreError::Generic`] for a store without sequences.
pub(crate) fn encode(
    dst: &mut [u8],
    tables: &FseTables,
    store: &SeqStore,
    backend: Backend,
) -> Result<usize, CoreError> {
    fearless_simd::dispatch!(backend.0, simd => encode_on(simd, dst, tables, store))
}

/// Bits of extra values one sequence may add without a flush: the 64-bit
/// container holds up to 7 pending bits and the three states' bits.
const EXTRA_BITS_WITHOUT_FLUSH: u32 = 64 - 7 - (LL_FSE_LOG + ML_FSE_LOG + OFF_FSE_LOG);

/// [`encode`] inside `simd`'s feature context.
#[simd]
fn encode_on<S: Simd>(
    simd: S,
    dst: &mut [u8],
    tables: &FseTables,
    store: &SeqStore,
) -> Result<usize, CoreError> {
    let Some(mut writer) = BitWriter::new(dst) else {
        return Err(CoreError::DstSizeTooSmall);
    };
    let n = store.nb_seq();
    let Some((last, codes)) = store.codes[..n].split_last() else {
        return Err(CoreError::Generic);
    };
    let Some((last_seq, seqs)) = store.sequences[..n].split_last() else {
        return Err(CoreError::Generic);
    };

    let mut state_ml = CState::with_first_symbol(&tables.matchlength, u32::from(last.ml));
    let mut state_of = CState::with_first_symbol(&tables.offcode, u32::from(last.of));
    let mut state_ll = CState::with_first_symbol(&tables.litlength, u32::from(last.ll));
    // An empty container takes any sequence's extra bits.
    writer.add_bits_fast(last.extra(last_seq), u32::from(last.extra_bits));
    writer.flush();

    for (&codes, seq) in codes.iter().zip(seqs).rev() {
        state_of.encode(&mut writer, u32::from(codes.of));
        state_ml.encode(&mut writer, u32::from(codes.ml));
        state_ll.encode(&mut writer, u32::from(codes.ll));
        let extra_bits = u32::from(codes.extra_bits);
        let extra = codes.extra(seq);
        if extra_bits < EXTRA_BITS_WITHOUT_FLUSH {
            writer.add_bits_fast(extra, extra_bits);
        } else {
            // Flushes only move whole bytes out of the container, so where
            // they fall does not change the stream: this one makes room,
            // and the second one splits the at most 63 extra bits.
            writer.flush();
            writer.add_bits(extra, EXTRA_BITS_WITHOUT_FLUSH);
            writer.flush();
            writer.add_bits_fast(
                extra >> EXTRA_BITS_WITHOUT_FLUSH,
                extra_bits - EXTRA_BITS_WITHOUT_FLUSH,
            );
        }
        writer.flush();
    }

    state_ml.flush(&mut writer);
    state_of.flush(&mut writer);
    state_ll.flush(&mut writer);
    match writer.close() {
        0 => Err(CoreError::DstSizeTooSmall),
        size => Ok(size),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::core::seq_store::{offset_to_offbase, repcode_to_offbase};

    /// `ZSTD_encodeSequences_body` step by step, with every conditional
    /// flush of the reference: the oracle for the fused loop. Returns `0`
    /// when the stream does not fit.
    fn encode_reference(dst: &mut [u8], tables: &FseTables, store: &SeqStore) -> usize {
        use crate::shared::format::{LL_BITS, ML_BITS};
        let Some(mut writer) = BitWriter::new(dst) else {
            return 0;
        };
        let n = store.nb_seq();
        let (seqs, codes) = (&store.sequences[..n], &store.codes[..n]);
        let ll_bits = |code: u8| u32::from(LL_BITS[usize::from(code)]);
        let ml_bits = |code: u8| u32::from(ML_BITS[usize::from(code)]);
        let last = &codes[n - 1];
        let mut state_ml = CState::with_first_symbol(&tables.matchlength, u32::from(last.ml));
        let mut state_of = CState::with_first_symbol(&tables.offcode, u32::from(last.of));
        let mut state_ll = CState::with_first_symbol(&tables.litlength, u32::from(last.ll));
        writer.add_bits(u64::from(seqs[n - 1].lit_length), ll_bits(last.ll));
        writer.add_bits(u64::from(seqs[n - 1].ml_base), ml_bits(last.ml));
        writer.add_bits(u64::from(seqs[n - 1].off_base), u32::from(last.of));
        writer.flush();
        for (seq, codes) in seqs[..n - 1].iter().zip(&codes[..n - 1]).rev() {
            let (ll, ml, of) = (ll_bits(codes.ll), ml_bits(codes.ml), u32::from(codes.of));
            state_of.encode(&mut writer, u32::from(codes.of));
            state_ml.encode(&mut writer, u32::from(codes.ml));
            state_ll.encode(&mut writer, u32::from(codes.ll));
            if of + ml + ll >= 64 - 7 - (LL_FSE_LOG + ML_FSE_LOG + OFF_FSE_LOG) {
                writer.flush();
            }
            writer.add_bits(u64::from(seq.lit_length), ll);
            writer.add_bits(u64::from(seq.ml_base), ml);
            if of + ml + ll > 56 {
                writer.flush();
            }
            writer.add_bits(u64::from(seq.off_base), of);
            writer.flush();
        }
        state_ml.flush(&mut writer);
        state_of.flush(&mut writer);
        state_ll.flush(&mut writer);
        writer.close()
    }

    /// A store of `count` pseudo-random sequences whose lengths and offsets
    /// span every code, so the extra bits of one sequence range from none
    /// to the 63 that need a flush in between; `long` makes one length
    /// exceed 16 bits.
    fn random_store(seed: &mut u64, count: usize, long: Option<bool>) -> SeqStore {
        let mut next = move || {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *seed
        };
        let mut store = SeqStore::default();
        store.reserve(1 << 22, 3);
        for i in 0..count {
            let r = next();
            // Bit widths drawn first, so small and huge values are equally
            // likely.
            let ll = (r & 0xFFFF) as usize >> (r >> 16 & 15);
            let ml = 3 + ((r >> 20 & 0xFFFF) as usize >> (r >> 36 & 15));
            let off = ((r >> 40) as u32 | 1 << 23) >> (r >> 59);
            let off_base = if r >> 62 == 0 {
                repcode_to_offbase(1 + (r >> 8) as u32 % 3)
            } else {
                offset_to_offbase(off.max(1))
            };
            match long {
                Some(true) if i == count / 2 => store.store_seq_only(0x10000 + ll, off_base, ml),
                Some(false) if i == count / 2 => store.store_seq_only(ll, off_base, 0x10003 + ml),
                _ => store.store_seq_only(ll, off_base, ml),
            }
        }
        store
    }

    #[test]
    fn fused_encoding_matches_the_reference_loop() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for (count, long) in [
            (1usize, None),
            (2, None),
            (7, Some(true)),
            (300, None),
            (300, Some(false)),
            (5000, Some(true)),
        ] {
            let mut store = random_store(&mut seed, count, long);
            let prev = FseTables::default();
            let mut next = FseTables::default();
            let mut headers = [0u8; 1024];
            for strategy in [Strategy::Fast, Strategy::BtOpt] {
                build_statistics(
                    &mut store,
                    &prev,
                    &mut next,
                    &mut headers,
                    strategy,
                    Backend::default(),
                )
                .expect("statistics");
                assert!(
                    store.codes[..count]
                        .iter()
                        .any(|codes| u32::from(codes.extra_bits) >= EXTRA_BITS_WITHOUT_FLUSH)
                        || count < 300,
                    "the slow path is exercised"
                );
                let mut expected = alloc::vec![0u8; count * 16 + 64];
                let size = encode_reference(&mut expected, &next, &store);
                assert_ne!(size, 0);
                // Room to spare, just enough, and too little by one byte or
                // by everything.
                for capacity in [expected.len(), size + 8, size + 7, 9, 8] {
                    let mut want = alloc::vec![0u8; capacity];
                    let fits = encode_reference(&mut want, &next, &store);
                    for backend in Backend::available() {
                        let mut got = alloc::vec![0u8; capacity];
                        let result = encode(&mut got, &next, &store, backend);
                        if fits == 0 {
                            assert_eq!(result, Err(CoreError::DstSizeTooSmall), "{capacity}");
                        } else {
                            assert_eq!(result, Ok(fits), "{capacity} {backend}");
                            assert_eq!(got[..fits], want[..fits], "{capacity} {backend}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn an_empty_store_has_no_bitstream() {
        let store = SeqStore::default();
        let mut dst = [0u8; 64];
        assert_eq!(
            encode(&mut dst, &FseTables::default(), &store, Backend::default()),
            Err(CoreError::Generic)
        );
        assert_eq!(
            encode(
                &mut dst[..8],
                &FseTables::default(),
                &store,
                Backend::default()
            ),
            Err(CoreError::DstSizeTooSmall)
        );
    }

    #[test]
    fn a_single_code_is_rle_unless_basic_is_cheaper() {
        let count = [0, 5, 0];
        let stats = CodeStats {
            count: &count,
            max: 1,
            most_frequent: 5,
            nb_seq: 5,
        };
        let table = CTable::default();
        let mut repeat = FseRepeat::Check;
        let kind = select_encoding_type(
            &mut repeat,
            stats,
            LITERAL_LENGTHS,
            &table,
            true,
            Strategy::Fast,
        );
        assert_eq!(kind, EncodingType::Rle);
        assert_eq!(repeat, FseRepeat::None);
        let two = CodeStats {
            nb_seq: 2,
            most_frequent: 2,
            ..stats
        };
        assert_eq!(
            select_encoding_type(
                &mut repeat,
                two,
                LITERAL_LENGTHS,
                &table,
                true,
                Strategy::Fast
            ),
            EncodingType::Basic
        );
    }

    #[test]
    fn costs_reject_tables_that_cannot_encode_present_symbols() {
        let mut table = CTable::default();
        table.build(&LL_DEFAULT_NORM, MAX_LL, LL_DEFAULT_NORM_LOG);
        let mut count = [0u32; 64];
        count[..3].copy_from_slice(&[10, 5, 1]);
        assert_ne!(fse_bit_cost(&table, &count, 2), COST_ERROR);
        assert_eq!(fse_bit_cost(&table, &count, 40), COST_ERROR);
        let mut sparse = CTable::default();
        let mut norm = [0i16; 64];
        norm[0] = 32;
        sparse.build(&norm, 2, 5);
        assert_eq!(fse_bit_cost(&sparse, &count, 2), COST_ERROR);
        assert!(cross_entropy_cost(&LL_DEFAULT_NORM, LL_DEFAULT_NORM_LOG, &count, 2) > 0);
        assert!(entropy_cost(&count, 2, 16) > 0);
    }

    #[test]
    fn rle_tables_need_a_byte_of_room() {
        let mut next = CTable::default();
        let prev = CTable::default();
        let mut count = [0u32; 64];
        assert_eq!(
            build_ctable(
                &mut [],
                &mut next,
                &prev,
                LITERAL_LENGTHS,
                EncodingType::Rle,
                Histogram {
                    count: &mut count,
                    max: 3,
                    nb_seq: 1,
                    last: 3,
                },
            ),
            Err(CoreError::DstSizeTooSmall)
        );
    }
}

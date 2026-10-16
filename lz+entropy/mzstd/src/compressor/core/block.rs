//! One block: match finding into a sequence store, then entropy coding
//! (`ZSTD_buildSeqStore`, `ZSTD_entropyCompressSeqStore` and
//! `ZSTD_compressBlock_internal`).

use super::cctx::Engine;
use super::cdict::CDict;
use super::double_fast;
use super::entropy::Tables;
use super::entropy::literals::{self, LiteralsSettings, min_gain};
use super::entropy::sequences;
use super::error::CoreError;
use super::fast;
use super::lazy;
use super::ldm;
use super::match_state::{BlockCompressor, BlockInput};
use super::mem;
use super::opt;
use super::params::{ParamSwitch, Strategy};
use super::seq_store::SeqStore;
use super::window::Segments;
use crate::Backend;
use crate::shared::format::{BLOCK_HEADER_SIZE, LONG_NB_SEQ};

/// Blocks shorter than this are stored without an attempt
/// (`MIN_CBLOCK_SIZE + ZSTD_blockHeaderSize + 1 + 1`).
const MIN_BLOCK_TO_COMPRESS: usize = 2 + BLOCK_HEADER_SIZE + 1 + 1;
/// `SUSPECT_UNCOMPRESSIBLE_LITERAL_RATIO`.
const SUSPECT_UNCOMPRESSIBLE_LITERAL_RATIO: usize = 20;
/// Upper bound on the size of a compressed block worth turning into RLE.
const RLE_MAX_LENGTH: usize = 25;

/// Outcome of match finding (`ZSTD_BuildSeqStore_e`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum Build {
    Compress,
    NoCompress,
}

/// Where a block's history comes from (`ZSTD_dictMode_e`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum DictMode {
    /// One contiguous prefix (`ZSTD_noDict`).
    NoDict,
    /// A prefix and an older segment (`ZSTD_extDict`).
    ExtDict,
    /// A prefix and an attached dictionary (`ZSTD_dictMatchState`).
    DictMatchState,
}

/// The block compressor for a strategy and window shape
/// (`ZSTD_selectBlockCompressor`).
fn select_block_compressor(
    strategy: Strategy,
    row: bool,
    mode: DictMode,
) -> Result<BlockCompressor, CoreError> {
    use DictMode::{DictMatchState as Dms, ExtDict as Ext, NoDict};
    if row
        && matches!(
            strategy,
            Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2
        )
    {
        return Ok(match (strategy, mode) {
            (Strategy::Greedy, NoDict) => lazy::greedy_row,
            (Strategy::Greedy, Ext) => lazy::greedy_row_ext,
            (Strategy::Greedy, Dms) => lazy::greedy_row_dms,
            (Strategy::Lazy, NoDict) => lazy::lazy_row,
            (Strategy::Lazy, Ext) => lazy::lazy_row_ext,
            (Strategy::Lazy, Dms) => lazy::lazy_row_dms,
            (_, NoDict) => lazy::lazy2_row,
            (_, Ext) => lazy::lazy2_row_ext,
            (_, Dms) => lazy::lazy2_row_dms,
        });
    }
    Ok(match (strategy, mode) {
        (Strategy::Fast, NoDict) => fast::compress_block,
        (Strategy::Fast, Ext) => fast::compress_block_ext_dict,
        (Strategy::Fast, Dms) => fast::compress_block_dict_match_state,
        (Strategy::DFast, NoDict) => double_fast::compress_block,
        (Strategy::DFast, Ext) => double_fast::compress_block_ext_dict,
        (Strategy::DFast, Dms) => double_fast::compress_block_dict_match_state,
        (Strategy::Greedy, NoDict) => lazy::greedy,
        (Strategy::Greedy, Ext) => lazy::greedy_ext,
        (Strategy::Greedy, Dms) => lazy::greedy_dms,
        (Strategy::Lazy, NoDict) => lazy::lazy,
        (Strategy::Lazy, Ext) => lazy::lazy_ext,
        (Strategy::Lazy, Dms) => lazy::lazy_dms,
        (Strategy::Lazy2, NoDict) => lazy::lazy2,
        (Strategy::Lazy2, Ext) => lazy::lazy2_ext,
        (Strategy::Lazy2, Dms) => lazy::lazy2_dms,
        (Strategy::BtLazy2, NoDict) => lazy::btlazy2,
        (Strategy::BtLazy2, Ext) => lazy::btlazy2_ext,
        (Strategy::BtLazy2, Dms) => lazy::btlazy2_dms,
        (Strategy::BtOpt, NoDict) => opt::btopt,
        (Strategy::BtOpt, Ext) => opt::btopt_ext,
        (Strategy::BtOpt, Dms) => opt::btopt_dms,
        (Strategy::BtUltra, NoDict) => opt::btultra,
        (Strategy::BtUltra | Strategy::BtUltra2, Ext) => opt::btultra_ext,
        (Strategy::BtUltra | Strategy::BtUltra2, Dms) => opt::btultra_dms,
        (Strategy::BtUltra2, NoDict) => opt::btultra2,
    })
}

/// Runs the match finder over `[start, end)` (`ZSTD_buildSeqStore`).
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
pub(crate) fn build_seq_store(
    engine: &mut Engine,
    segments: &Segments<'_>,
    view: &mem::Window<'_>,
    start: u32,
    end: u32,
) -> Result<Build, CoreError> {
    let src_size = (end - start) as usize;
    if src_size < MIN_BLOCK_TO_COMPRESS {
        return Ok(Build::NoCompress);
    }
    engine.seq_store.reset();
    let ms = &mut engine.ms;
    ms.backend = engine.backend;
    ms.opt.literal_compression = engine.applied.literal_compression;

    // Limited update after a very long match.
    let curr = start;
    if curr > ms.next_to_update + 384 {
        ms.next_to_update = curr - 192.min(curr - ms.next_to_update - 384);
    }

    engine.next.rep = engine.prev.rep;
    let row = engine.applied.use_row_match_finder == ParamSwitch::Enable;
    // ZSTD_matchState_dictMode.
    let dms = if ms.dict_match_state {
        segments.cdict.map(CDict::match_state)
    } else {
        None
    };
    let mode = if ms.window.has_ext_dict() {
        DictMode::ExtDict
    } else if dms.is_some() {
        DictMode::DictMatchState
    } else {
        DictMode::NoDict
    };
    let compressor = select_block_compressor(engine.applied.cparams.strategy, row, mode)?;
    let block = BlockInput {
        window: *view,
        start,
        end,
        backend: engine.backend,
        dms: if mode == DictMode::DictMatchState {
            dms
        } else {
            None
        },
        symbol_costs: Some(&engine.prev.entropy),
    };
    let last_ll_size = if engine.applied.ldm.enable == ParamSwitch::Enable {
        let store = &mut engine.ldm_store;
        store.seq.clear();
        store.pos = 0;
        store.pos_in_sequence = 0;
        let block_addr = ms.window.base.add(i64::from(start));
        let ldm_start = engine.ldm.window.index_of(block_addr);
        let ldm_end = ldm_start + (end - start);
        engine.ldm.generate_sequences(
            segments,
            store,
            &engine.applied.ldm,
            ldm_start,
            ldm_end,
            engine.backend,
        );
        ldm::block_compress(
            store,
            ms,
            &mut engine.seq_store,
            &mut engine.next.rep,
            compressor,
            &block,
        )
    } else {
        compressor(ms, &mut engine.seq_store, &mut engine.next.rep, &block)
    };
    engine
        .seq_store
        .store_last_literals(view.prefix.slice(end - last_ll_size as u32, end));
    Ok(Build::Compress)
}

/// Writes the literals and sequences sections of `store`
/// (`ZSTD_entropyCompressSeqStore_internal`).
///
/// Returns `0` for the reference's "emit an uncompressed block instead".
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
fn entropy_compress_internal(
    store: &mut SeqStore,
    prev: &Tables,
    next: &mut Tables,
    engine_params: &super::cctx::AppliedParams,
    dst: &mut [u8],
    block_size: usize,
    backend: Backend,
) -> Result<usize, CoreError> {
    let strategy = engine_params.cparams.strategy;
    let nb_seq = store.nb_seq();
    let lit_size = store.literals.len();
    let suspect = nb_seq == 0 || lit_size / nb_seq >= SUSPECT_UNCOMPRESSIBLE_LITERAL_RATIO;
    let mut op = literals::try_compress(
        dst,
        &store.literals,
        &prev.huf,
        &mut next.huf,
        LiteralsSettings {
            strategy,
            disable_compression: engine_params.literals_compression_disabled(),
            suspect_uncompressible: suspect,
            backend,
        },
    )?;
    if op == 0 {
        // Raw literals alone already make this block too large to keep.
        // Avoid copying them and encoding sequences into a doomed result.
        // A short destination still follows the ordinary error path below.
        if lit_size >= block_size - min_gain(block_size, strategy) && dst.len() >= block_size {
            if nb_seq == 0 {
                next.fse.clone_from(&prev.fse);
            }
            return Ok(0);
        }
        op = literals::no_compress(dst, &store.literals)?;
    }

    if dst.len() - op < 3 + 1 {
        return Err(CoreError::DstSizeTooSmall);
    }
    if nb_seq < 128 {
        dst[op] = nb_seq as u8;
        op += 1;
    } else if nb_seq < LONG_NB_SEQ {
        dst[op] = ((nb_seq >> 8) + 0x80) as u8;
        dst[op + 1] = nb_seq as u8;
        op += 2;
    } else {
        dst[op] = 0xFF;
        dst[op + 1..op + 3].copy_from_slice(&((nb_seq - LONG_NB_SEQ) as u16).to_le_bytes());
        op += 3;
    }
    if nb_seq == 0 {
        next.fse.clone_from(&prev.fse);
        return Ok(op);
    }
    let seq_head = op;
    op += 1;
    let stats = sequences::build_statistics(
        store,
        &prev.fse,
        &mut next.fse,
        &mut dst[op..],
        strategy,
        backend,
    )?;
    dst[seq_head] =
        ((stats.ll_type as u8) << 6) + ((stats.of_type as u8) << 4) + ((stats.ml_type as u8) << 2);
    op += stats.size;
    let bitstream_size = sequences::encode(&mut dst[op..], &next.fse, store, backend)?;
    op += bitstream_size;
    if stats.last_count_size != 0 && stats.last_count_size + bitstream_size < 4 {
        return Ok(0);
    }
    Ok(op)
}

/// Entropy-codes `store` and decides whether the result is worth keeping
/// (`ZSTD_entropyCompressSeqStore`).
///
/// Returns `0` when the block should be stored uncompressed.
pub(crate) fn entropy_compress(
    store: &mut SeqStore,
    prev: &Tables,
    next: &mut Tables,
    params: &super::cctx::AppliedParams,
    dst: &mut [u8],
    block_size: usize,
    backend: Backend,
) -> Result<usize, CoreError> {
    let capacity = dst.len();
    let c_size =
        match entropy_compress_internal(store, prev, next, params, dst, block_size, backend) {
            Ok(0) => return Ok(0),
            Err(CoreError::DstSizeTooSmall) if block_size <= capacity => return Ok(0),
            other => other?,
        };
    let max_c_size = block_size - min_gain(block_size, params.cparams.strategy);
    if c_size >= max_c_size {
        return Ok(0);
    }
    Ok(c_size)
}

/// Whether every byte of `src` is the same (`ZSTD_isRLE`).
///
/// Equal adjacent slices mean every byte equals its predecessor. Slice
/// equality uses the platform's bulk comparison, including its vector
/// loop and short-input handling.
pub(crate) fn is_rle(src: &[u8]) -> bool {
    let Some((_, tail)) = src.split_first() else {
        return true;
    };
    tail == &src[..tail.len()]
}

/// Compresses one block body into `dst` (`ZSTD_compressBlock_internal`).
///
/// Returns `0` for "store uncompressed", `1` for an RLE block whose byte is
/// in `dst[0]`, and otherwise the compressed body size.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
pub(crate) fn compress_block_internal(
    engine: &mut Engine,
    segments: &Segments<'_>,
    view: &mem::Window<'_>,
    start: u32,
    end: u32,
    dst: &mut [u8],
    frame: bool,
) -> Result<usize, CoreError> {
    let src = view.prefix.slice(start, end);
    let c_size = match build_seq_store(engine, segments, view, start, end)? {
        Build::NoCompress => 0,
        Build::Compress => {
            let params = engine.applied;
            let mut c_size = entropy_compress(
                &mut engine.seq_store,
                &engine.prev.entropy,
                &mut engine.next.entropy,
                &params,
                dst,
                src.len(),
                engine.backend,
            )?;
            if frame && !engine.is_first_block && c_size < RLE_MAX_LENGTH && is_rle(src) {
                c_size = 1;
                dst[0] = src[0];
            }
            c_size
        }
    };
    if c_size > 1 {
        engine.confirm_block_state();
    }
    engine.demote_offcode_repeat();
    Ok(c_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rle_detection() {
        assert!(is_rle(&[3]));
        assert!(is_rle(&[3; 100]));
        assert!(!is_rle(&[3, 3, 4]));
        assert!(is_rle(&[]));
        // A difference in a whole chunk, in the tail, and at either end.
        let mut long = [7u8; 100];
        assert!(is_rle(&long));
        for at in [0, 1, 31, 32, 63, 64, 95, 96, 99] {
            long[at] = 8;
            assert!(!is_rle(&long), "{at}");
            long[at] = 7;
        }
    }
}

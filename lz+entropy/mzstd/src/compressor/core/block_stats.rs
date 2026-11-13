//! Entropy statistics of a block's sequence store (`ZSTD_buildBlockEntropyStats`
//! and its literals and sequences halves, from `zstd_compress.c`).
//!
//! Both the post-match block splitter ([`super::split_block`]) and the
//! superblock path ([`super::superblock`]) decide from these statistics:
//! which table each stream uses, and the table descriptions a first block
//! would write. They are built with the real table construction, error paths
//! included, so a decision is made exactly when the reference makes it.

use super::entropy::huf::{self, Flags};
use super::entropy::sequences;
use super::entropy::{FseTables, HufTables, Tables, hist};
use super::error::{CoreError, as_code};
use super::params::Strategy;
use super::seq_store::SeqStore;
use crate::Backend;
use crate::shared::format::EncodingType;

/// `ZSTD_MAX_HUF_HEADER_SIZE`.
pub(crate) const MAX_HUF_HEADER_SIZE: usize = 128;
/// `ZSTD_MAX_FSE_HEADERS_SIZE`.
pub(crate) const MAX_FSE_HEADERS_SIZE: usize = (53 * 9 + 36 * 9 + 32 * 8usize).div_ceil(8);

/// `ZSTD_hufCTablesMetadata_t`.
#[derive(Copy, Clone, Debug)]
pub(crate) struct HufMetadata {
    /// The literals block type the statistics chose (`hType`).
    pub(crate) kind: EncodingType,
    /// Size of the table description (`hufDesSize`).
    pub(crate) des_size: usize,
    /// The table description (`hufDesBuffer`); `des_size` bytes are valid.
    pub(crate) des_buffer: [u8; MAX_HUF_HEADER_SIZE],
}

impl HufMetadata {
    /// Metadata of a literals block without a table description.
    const fn without_table(kind: EncodingType) -> Self {
        Self {
            kind,
            des_size: 0,
            des_buffer: [0; MAX_HUF_HEADER_SIZE],
        }
    }
}

/// `ZSTD_fseCTablesMetadata_t`.
#[derive(Copy, Clone, Debug)]
pub(crate) struct FseMetadata {
    pub(crate) ll_type: EncodingType,
    pub(crate) of_type: EncodingType,
    pub(crate) ml_type: EncodingType,
    /// Size of the table descriptions (`fseTablesSize`).
    pub(crate) tables_size: usize,
    /// The table descriptions (`fseTablesBuffer`); `tables_size` bytes are
    /// valid.
    pub(crate) tables_buffer: [u8; MAX_FSE_HEADERS_SIZE],
    /// Size of the last `Compressed` table description (`lastCountSize`).
    pub(crate) last_count_size: usize,
}

/// `ZSTD_entropyCTablesMetadata_t`.
#[derive(Copy, Clone, Debug)]
pub(crate) struct EntropyMetadata {
    pub(crate) huf: HufMetadata,
    pub(crate) fse: FseMetadata,
}

/// `ZSTD_buildBlockEntropyStats_literals`.
///
/// # Errors
///
/// The error of the table construction or of its description, as the
/// reference returns it as `hufDesSize`.
fn build_literals_stats(
    src: &[u8],
    prev: &HufTables,
    next: &mut HufTables,
    disabled: bool,
    optimal_depth: bool,
) -> Result<HufMetadata, CoreError> {
    let src_size = src.len();
    next.clone_from(prev);
    let basic = HufMetadata::without_table(EncodingType::Basic);
    if disabled {
        return Ok(basic);
    }
    let min_lit_size = if prev.repeat == huf::Repeat::Valid {
        6
    } else {
        63
    };
    if src_size <= min_lit_size {
        return Ok(basic);
    }
    let mut count = [0u32; 256];
    let mut max_symbol_value = 255;
    let largest = hist::count(&mut count, &mut max_symbol_value, src, false)? as usize;
    if largest == src_size {
        return Ok(HufMetadata::without_table(EncodingType::Rle));
    }
    if largest <= (src_size >> 7) + 4 {
        return Ok(basic);
    }
    let mut repeat = prev.repeat;
    if repeat == huf::Repeat::Check && !prev.table.validate(&count, max_symbol_value) {
        repeat = huf::Repeat::None;
    }
    *next.table = huf::CTable::EMPTY;
    let flags = Flags {
        optimal_depth,
        ..Flags::default()
    };
    let huff_log = huf::optimal_table_log(11, src_size, max_symbol_value, &count, flags);
    let huff_log = next.table.build(&count, max_symbol_value, huff_log)?;
    let new_c_size = next
        .table
        .estimate_compressed_size(&count, max_symbol_value);
    let mut description = [0u8; MAX_HUF_HEADER_SIZE];
    let written = next
        .table
        .write(&mut description, max_symbol_value, huff_log);
    let h_size = as_code(written);
    if repeat != huf::Repeat::None {
        let old_c_size = prev
            .table
            .estimate_compressed_size(&count, max_symbol_value);
        if old_c_size < src_size
            && (old_c_size <= h_size.wrapping_add(new_c_size)
                || h_size.wrapping_add(12) >= src_size)
        {
            next.clone_from(prev);
            return Ok(HufMetadata::without_table(EncodingType::Repeat));
        }
    }
    if new_c_size.wrapping_add(h_size) >= src_size {
        next.clone_from(prev);
        return Ok(basic);
    }
    next.repeat = huf::Repeat::Check;
    Ok(HufMetadata {
        kind: EncodingType::Compressed,
        des_size: written?,
        des_buffer: description,
    })
}

/// `ZSTD_buildBlockEntropyStats_sequences`.
///
/// # Errors
///
/// The error of building a table.
fn build_sequences_stats(
    store: &mut SeqStore,
    prev: &FseTables,
    next: &mut FseTables,
    strategy: Strategy,
    backend: Backend,
) -> Result<FseMetadata, CoreError> {
    let mut metadata = FseMetadata {
        ll_type: EncodingType::Basic,
        of_type: EncodingType::Basic,
        ml_type: EncodingType::Basic,
        tables_size: 0,
        tables_buffer: [0; MAX_FSE_HEADERS_SIZE],
        last_count_size: 0,
    };
    if store.nb_seq() == 0 {
        next.litlength_repeat = super::entropy::FseRepeat::None;
        next.offcode_repeat = super::entropy::FseRepeat::None;
        next.matchlength_repeat = super::entropy::FseRepeat::None;
        return Ok(metadata);
    }
    let stats = sequences::build_statistics(
        store,
        prev,
        next,
        &mut metadata.tables_buffer,
        strategy,
        backend,
    )?;
    metadata.ll_type = stats.ll_type;
    metadata.of_type = stats.of_type;
    metadata.ml_type = stats.ml_type;
    metadata.tables_size = stats.size;
    metadata.last_count_size = stats.last_count_size;
    Ok(metadata)
}

/// Builds the entropy tables of `store` into `next` and the choices and
/// descriptions that go with them (`ZSTD_buildBlockEntropyStats`).
///
/// # Errors
///
/// The error of building a table, which the reference returns instead of
/// metadata.
pub(crate) fn build_block_entropy_stats(
    store: &mut SeqStore,
    prev: &Tables,
    next: &mut Tables,
    strategy: Strategy,
    literals_disabled: bool,
    backend: Backend,
) -> Result<EntropyMetadata, CoreError> {
    let huf = build_literals_stats(
        store.literals(),
        &prev.huf,
        &mut next.huf,
        literals_disabled,
        strategy >= Strategy::BtUltra,
    )?;
    let fse = build_sequences_stats(store, &prev.fse, &mut next.fse, strategy, backend)?;
    Ok(EntropyMetadata { huf, fse })
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::compressor::core::entropy::huf::Repeat;

    fn tables() -> (Tables, Tables) {
        (Tables::default(), Tables::default())
    }

    #[test]
    fn literals_that_cannot_gain_stay_raw_and_keep_the_previous_table() {
        let (prev, mut next) = tables();
        let tiny = [1u8; 63];
        let stats = build_literals_stats(&tiny, &prev.huf, &mut next.huf, false, false).unwrap();
        assert_eq!(stats.kind, EncodingType::Basic);
        assert_eq!(stats.des_size, 0);
        // A disabled literals compression is raw whatever the content.
        let skewed: Vec<u8> = (0..4000).map(|i| (i % 7) as u8).collect();
        let stats = build_literals_stats(&skewed, &prev.huf, &mut next.huf, true, false).unwrap();
        assert_eq!(stats.kind, EncodingType::Basic);
        // Flat content has no gain.
        let flat: Vec<u8> = (0..4096).map(|i| i as u8).collect();
        let stats = build_literals_stats(&flat, &prev.huf, &mut next.huf, false, false).unwrap();
        assert_eq!(stats.kind, EncodingType::Basic);
    }

    #[test]
    fn one_repeated_literal_is_rle_and_a_skewed_alphabet_gets_a_table() {
        let (prev, mut next) = tables();
        let stats =
            build_literals_stats(&[9u8; 500], &prev.huf, &mut next.huf, false, false).unwrap();
        assert_eq!(stats.kind, EncodingType::Rle);
        let skewed: Vec<u8> = (0..4000u32).map(|i| (i * i % 11) as u8).collect();
        let stats = build_literals_stats(&skewed, &prev.huf, &mut next.huf, false, false).unwrap();
        assert_eq!(stats.kind, EncodingType::Compressed);
        assert!(stats.des_size > 0 && stats.des_size <= MAX_HUF_HEADER_SIZE);
        assert_eq!(next.huf.repeat, Repeat::Check);
    }

    #[test]
    fn a_valid_previous_table_admits_much_shorter_literals() {
        let mut prev = Tables::default();
        let mut next = Tables::default();
        // Seven literals are too few to try, unless a table is waiting.
        let seven = [4u8; 7];
        let stats = build_literals_stats(&seven, &prev.huf, &mut next.huf, false, false).unwrap();
        assert_eq!(stats.kind, EncodingType::Basic);
        prev.huf.repeat = Repeat::Valid;
        let stats = build_literals_stats(&seven, &prev.huf, &mut next.huf, false, false).unwrap();
        assert_eq!(stats.kind, EncodingType::Rle);
    }

    #[test]
    fn a_table_that_fits_the_next_block_is_repeated() {
        let (prev, mut next) = tables();
        let skewed: Vec<u8> = (0..4000u32).map(|i| (i * i % 11) as u8).collect();
        build_literals_stats(&skewed, &prev.huf, &mut next.huf, false, false).unwrap();
        // The table just built is the previous one of the next block.
        let mut after = Tables::default();
        let stats = build_literals_stats(&skewed, &next.huf, &mut after.huf, false, false).unwrap();
        assert_eq!(stats.kind, EncodingType::Repeat);
        assert_eq!(stats.des_size, 0);
    }

    #[test]
    fn no_sequences_leave_every_stream_basic_and_unrepeatable() {
        let mut store = SeqStore::default();
        let prev = Tables::default();
        let mut next = Tables::default();
        let stats = build_block_entropy_stats(
            &mut store,
            &prev,
            &mut next,
            Strategy::Lazy,
            false,
            Backend::default(),
        )
        .unwrap();
        assert_eq!(stats.huf.kind, EncodingType::Basic);
        assert_eq!(stats.fse.ll_type, EncodingType::Basic);
        assert_eq!(stats.fse.of_type, EncodingType::Basic);
        assert_eq!(stats.fse.ml_type, EncodingType::Basic);
        assert_eq!(stats.fse.tables_size, 0);
        assert_eq!(stats.fse.last_count_size, 0);
    }
}

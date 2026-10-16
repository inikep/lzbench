//! Dictionaries prepared for compression (`ZSTD_CDict`) and the loading of
//! dictionary bytes into match finder tables (`ZSTD_compress_insertDictionary`,
//! `ZSTD_loadCEntropy`, `ZSTD_loadDictionaryContent`).
//!
//! A [`CDict`] owns a copy of the dictionary and the tables built over its
//! content with the parameters the reference's `ZSTD_CCtx_loadDictionary`
//! derives from the requested ones. A frame then either attaches it
//! (dictMatchState: the match finders read the dictionary's tables in
//! place) or copies its tables into the context, as `ZSTD_resetCCtx_usingCDict`
//! decides; see [`super::cctx::Engine::begin_with_dict`].
//!
//! [`load_dictionary_content`] also loads raw history straight into a
//! context's own tables (`ZSTD_dtlm_fast`), which the parallel encoder uses
//! for the overlap before a segment.

use alloc::boxed::Box;

use super::cctx::{BlockState, LdmParams, RequestedParams};
use super::double_fast;
use super::entropy::read::{read_ctable, read_ncount};
use super::entropy::{FseRepeat, huf};
use super::fast;
use super::lazy;
use super::ldm::LdmState;
use super::match_state::{DictMatchState, MatchState};
use super::opt;
use super::params::{
    CONTENT_SIZE_UNKNOWN, CParamMode, CParams, ParamSwitch, SHORT_CACHE_TAG_BITS, Strategy,
    cdict_indices_are_tagged,
};
use super::window::{Addr, CURRENT_MAX, HASH_READ_SIZE, Segment, Segments, WINDOW_START_INDEX};
use crate::dictionary::{DictionaryContent, DictionaryError, Layout};
use crate::shared::format::{
    LL_FSE_LOG, MAX_LL, MAX_ML, MAX_OFF, ML_FSE_LOG, OFF_FSE_LOG, REP_NUM, highbit32,
};

/// How many positions a table load indexes (`ZSTD_dictTableLoadMethod_e`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum TableLoad {
    /// Every position a finder would insert while compressing
    /// (`ZSTD_dtlm_fast`).
    Fast,
    /// Also the positions in between, where their slot is free
    /// (`ZSTD_dtlm_full`).
    Full,
}

/// Whose tables a load fills (`ZSTD_tableFillPurpose_e`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum FillPurpose {
    /// A compression context's tables.
    CCtx,
    /// A dictionary's tables, which fast and dfast tag.
    CDict,
}

/// When an attached dictionary is used in place, and when its tables are
/// copied (`ZSTD_dictAttachPref_e`).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub(crate) enum AttachPref {
    /// The reference's size heuristic (`ZSTD_dictDefaultAttach`).
    #[default]
    Default,
    /// Always attach (`ZSTD_dictForceAttach`).
    ForceAttach,
    /// Always copy the tables (`ZSTD_dictForceCopy`).
    ForceCopy,
    /// Load the dictionary's bytes into the context's own tables
    /// (`ZSTD_dictForceLoad`).
    ForceLoad,
}

/// Parameters a content load depends on (the parts of `ZSTD_CCtx_params`
/// `ZSTD_loadDictionaryContent` reads).
#[derive(Copy, Clone, Debug)]
pub(crate) struct LoadParams<'p> {
    /// Whether the row match finder is in use.
    pub(crate) use_row_match_finder: ParamSwitch,
    /// `forceWindow`: the loaded bytes obey the window like any input
    /// rather than staying reachable whatever the distance.
    pub(crate) force_window: bool,
    /// Long distance matching parameters, when it is enabled.
    pub(crate) ldm: Option<&'p LdmParams>,
}

/// `attachDictSizeCutoffs`: source sizes up to which a dictionary is
/// attached rather than copied.
const ATTACH_DICT_SIZE_CUTOFFS: [u64; 10] = [
    8 << 10,
    8 << 10,
    16 << 10,
    32 << 10,
    32 << 10,
    32 << 10,
    32 << 10,
    32 << 10,
    8 << 10,
    8 << 10,
];

/// A dictionary prepared for compression (`ZSTD_CDict`, created the way
/// `ZSTD_initLocalDict` creates one for `ZSTD_CCtx_loadDictionary`).
#[derive(Debug)]
pub(crate) struct CDict {
    /// The dictionary as given, header included (`dictContent`).
    bytes: Box<[u8]>,
    /// The ID frames name, `0` for none.
    pub(crate) dict_id: u32,
    /// How the bytes were read.
    content: DictionaryContent,
    /// Tables over the content, in the dictionary's own index space.
    pub(crate) ms: MatchState,
    /// Entropy tables and repeat offsets the first block inherits.
    pub(crate) block_state: BlockState,
    /// Whether the dictionary's lazy tables are rows.
    pub(crate) use_row_match_finder: ParamSwitch,
}

impl CDict {
    /// Prepares `bytes` for frames compressed with `requested`
    /// (`ZSTD_createCDict_advanced2` with `ZSTD_dlm_byCopy`).
    ///
    /// # Errors
    ///
    /// [`DictionaryError::NotZstdFormat`] when `content` demands the
    /// Zstandard format and the bytes are not in it, and
    /// [`DictionaryError::Corrupted`] for a malformed entropy section.
    pub(crate) fn new(
        bytes: Box<[u8]>,
        content: DictionaryContent,
        requested: &RequestedParams,
    ) -> Result<Self, DictionaryError> {
        let cparams = requested.cparams_for(
            CONTENT_SIZE_UNKNOWN,
            bytes.len() as u64,
            CParamMode::CreateCDict,
        );
        let use_row_match_finder =
            super::params::resolve_row_match_finder(requested.use_row_match_finder, &cparams);
        let mut ms = MatchState::empty(cparams, use_row_match_finder);
        ms.reset_for_dictionary(cparams, use_row_match_finder);
        let mut block_state = BlockState::default();
        block_state.reset();
        let mut dict = Self {
            bytes,
            dict_id: 0,
            content,
            ms,
            block_state,
            use_row_match_finder,
        };
        // Empty bytes are no dictionary, whatever the content type
        // (`ZSTD_CCtx_loadDictionary_advanced` with a size of zero).
        if dict.bytes.is_empty() {
            return Ok(dict);
        }
        let segments = Segments {
            buffer: &[],
            input: (&[], 0),
            dict: &dict.bytes,
            cdict: None,
        };
        let id = insert_dictionary(
            &mut dict.block_state,
            &mut dict.ms,
            None,
            &segments,
            content,
            requested.fparams.no_dict_id,
            &LoadParams {
                use_row_match_finder,
                force_window: false,
                ldm: None,
            },
            TableLoad::Full,
            FillPurpose::CDict,
        )?;
        dict.dict_id = id;
        Ok(dict)
    }

    /// The dictionary as given.
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// How the bytes were read.
    pub(crate) const fn content(&self) -> DictionaryContent {
        self.content
    }

    /// `dictContentSize`: the whole dictionary's size, header included.
    pub(crate) fn content_size(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// The parameters of the dictionary's tables.
    pub(crate) const fn cparams(&self) -> CParams {
        self.ms.cparams
    }

    /// `ZSTD_shouldAttachDict` for a frame of `pledged` bytes.
    pub(crate) fn should_attach(&self, pref: AttachPref, pledged: u64) -> bool {
        let cutoff = ATTACH_DICT_SIZE_CUTOFFS[self.ms.cparams.strategy as usize];
        (pledged <= cutoff || pledged == CONTENT_SIZE_UNKNOWN || pref == AttachPref::ForceAttach)
            && pref != AttachPref::ForceCopy
    }

    /// The parameter mode a frame of `pledged` bytes resolves with
    /// (`ZSTD_getCParamMode`).
    pub(crate) fn cparam_mode(&self, pref: AttachPref, pledged: u64) -> CParamMode {
        if self.should_attach(pref, pledged) {
            CParamMode::AttachDict
        } else {
            CParamMode::NoAttachDict
        }
    }

    /// The tables and bytes match finders read when the dictionary is
    /// attached.
    pub(crate) fn match_state(&self) -> DictMatchState<'_> {
        let segments = Segments {
            buffer: &[],
            input: (&[], 0),
            dict: &self.bytes,
            cdict: None,
        };
        DictMatchState {
            ms: &self.ms,
            window: self.ms.window.view(&segments, self.ms.window.end_index()),
        }
    }

    /// Bytes this dictionary keeps allocated.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.bytes.len()
            + 4 * (self.ms.hash_table.capacity() + self.ms.chain_table.capacity())
            + self.ms.tag_table.capacity()
            + core::mem::size_of::<BlockState>()
    }
}

/// `ZSTD_dictNCountRepeat`: a dictionary's table is valid only when it
/// gives every symbol up to `max_symbol` a non-zero probability.
fn dict_ncount_repeat(counts: &[i16], dict_max_symbol: u32, max_symbol: u32) -> FseRepeat {
    if dict_max_symbol < max_symbol {
        return FseRepeat::Check;
    }
    if counts[..=max_symbol as usize].contains(&0) {
        return FseRepeat::Check;
    }
    FseRepeat::Valid
}

/// `ZSTD_loadCEntropy`: reads the entropy section of a Zstandard-format
/// dictionary into `bs` and returns the size of the header (magic number,
/// ID and entropy section).
pub(crate) fn load_c_entropy(bs: &mut BlockState, dict: &[u8]) -> Result<usize, DictionaryError> {
    let corrupted = DictionaryError::Corrupted;
    let mut pos = 8;
    bs.entropy.huf.repeat = huf::Repeat::Check;
    {
        let (size, has_zero_weights) =
            read_ctable(&mut bs.entropy.huf.table, &dict[pos..]).map_err(|_| corrupted.clone())?;
        if !has_zero_weights && bs.entropy.huf.table.max_symbol_value == 255 {
            bs.entropy.huf.repeat = huf::Repeat::Valid;
        }
        pos += size;
    }

    let offcode = read_ncount(&dict[pos..], MAX_OFF).map_err(|_| corrupted.clone())?;
    if offcode.table_log > OFF_FSE_LOG {
        return Err(corrupted);
    }
    // Every offset symbol is filled, so the table has no gaps at its end.
    bs.entropy
        .fse
        .offcode
        .build(&offcode.counts, MAX_OFF, offcode.table_log);
    pos += offcode.size;

    let matchlength = read_ncount(&dict[pos..], MAX_ML).map_err(|_| corrupted.clone())?;
    if matchlength.table_log > ML_FSE_LOG {
        return Err(corrupted);
    }
    bs.entropy.fse.matchlength.build(
        &matchlength.counts,
        matchlength.max_symbol,
        matchlength.table_log,
    );
    bs.entropy.fse.matchlength_repeat =
        dict_ncount_repeat(&matchlength.counts, matchlength.max_symbol, MAX_ML);
    pos += matchlength.size;

    let litlength = read_ncount(&dict[pos..], MAX_LL).map_err(|_| corrupted.clone())?;
    if litlength.table_log > LL_FSE_LOG {
        return Err(corrupted);
    }
    bs.entropy
        .fse
        .litlength
        .build(&litlength.counts, litlength.max_symbol, litlength.table_log);
    bs.entropy.fse.litlength_repeat =
        dict_ncount_repeat(&litlength.counts, litlength.max_symbol, MAX_LL);
    pos += litlength.size;

    let reps = dict.get(pos..pos + 12).ok_or(corrupted.clone())?;
    for (rep, word) in bs.rep.iter_mut().zip(reps.as_chunks::<4>().0) {
        *rep = u32::from_le_bytes(*word);
    }
    pos += 12;

    let content_size = (dict.len() - pos) as u64;
    let mut offcode_max = MAX_OFF;
    if content_size <= u64::from(u32::MAX) - (128 << 10) {
        // The largest offset a frame must be able to code.
        let max_offset = content_size as u32 + (128 << 10);
        offcode_max = highbit32(max_offset);
    }
    bs.entropy.fse.offcode_repeat = dict_ncount_repeat(
        &offcode.counts,
        offcode.max_symbol,
        offcode_max.min(MAX_OFF),
    );
    if bs
        .rep
        .iter()
        .any(|&rep| rep == 0 || u64::from(rep) > content_size)
    {
        return Err(corrupted);
    }
    debug_assert_eq!(bs.rep.len(), REP_NUM);
    Ok(pos)
}

/// `ZSTD_compress_insertDictionary`: loads the dictionary segment of
/// `segments` into `bs` and `ms`, and returns the ID frames should name.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors ZSTD_compress_insertDictionary's parameter list"
)]
pub(crate) fn insert_dictionary(
    bs: &mut BlockState,
    ms: &mut MatchState,
    ldm: Option<&mut LdmState>,
    segments: &Segments<'_>,
    content: DictionaryContent,
    no_dict_id: bool,
    params: &LoadParams<'_>,
    dtlm: TableLoad,
    tfp: FillPurpose,
) -> Result<u32, DictionaryError> {
    let dict = segments.dict;
    let dict_addr = Addr::new(Segment::Dict, 0);
    if dict.len() < 8 {
        if content == DictionaryContent::Zstd {
            return Err(DictionaryError::NotZstdFormat);
        }
        return Ok(0);
    }
    bs.reset();
    match crate::dictionary::layout(dict, content)? {
        Layout::Raw => {
            load_dictionary_content(ms, ldm, segments, dict_addr, dict.len(), params, dtlm, tfp);
            Ok(0)
        }
        Layout::Zstd { id } => {
            let id = if no_dict_id { 0 } else { id };
            let header = load_c_entropy(bs, dict)?;
            load_dictionary_content(
                ms,
                None,
                segments,
                dict_addr.add(header as i64),
                dict.len() - header,
                params,
                dtlm,
                tfp,
            );
            Ok(id)
        }
    }
}

/// `ZSTD_loadDictionaryContent`: makes the `len` bytes at `src` history of
/// `ms` (and of `ldm`, when given and enabled) and indexes them.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors ZSTD_loadDictionaryContent's parameter list"
)]
pub(crate) fn load_dictionary_content(
    ms: &mut MatchState,
    ldm: Option<&mut LdmState>,
    segments: &Segments<'_>,
    src: Addr,
    len: usize,
    params: &LoadParams<'_>,
    dtlm: TableLoad,
    tfp: FillPurpose,
) {
    let cparams = ms.cparams;
    let iend = src.add(len as i64);
    let mut src = src;
    let mut len = len as u64;
    let ldm = ldm.zip(params.ldm);

    // Large dictionaries keep their suffix, so indices cannot overflow;
    // tagged tables leave 24 bits per index.
    let mut max_dict_size = u64::from(CURRENT_MAX - WINDOW_START_INDEX);
    if cdict_indices_are_tagged(&cparams) && tfp == FillPurpose::CDict {
        max_dict_size = max_dict_size.min(u64::from(
            (1u32 << (32 - SHORT_CACHE_TAG_BITS)) - WINDOW_START_INDEX,
        ));
    }
    if len > max_dict_size {
        src = iend.add(-(max_dict_size as i64));
        len = max_dict_size;
    }

    ms.window.update(src, len as usize, false);

    if let Some((state, ldm_params)) = ldm {
        state.window.update(src, len as usize, false);
        let start = state.window.index_of(src);
        let end = state.window.index_of(iend);
        let loaded_dict_end = if params.force_window { 0 } else { end };
        let view = state.window.view(segments, end);
        state.fill_hash_table(
            view.prefix.slice(start, end),
            start,
            loaded_dict_end,
            ldm_params,
        );
    }

    // Only the suffix the tables can reasonably index is inserted.
    let max_dict_size = 1u64 << (cparams.hash_log + 3).max(cparams.chain_log + 1).min(31);
    if len > max_dict_size {
        src = iend.add(-(max_dict_size as i64));
        len = max_dict_size;
    }

    let end = ms.window.index_of(iend);
    ms.next_to_update = ms.window.index_of(src);
    ms.loaded_dict_end = if params.force_window { 0 } else { end };
    ms.force_non_contiguous = false;

    if len <= u64::from(HASH_READ_SIZE) {
        return;
    }

    ms.overflow_correct_if_needed(ms.window.index_of(src), end);
    let end = ms.window.index_of(iend);
    let view = ms.window.view(segments, end);
    match cparams.strategy {
        Strategy::Fast => match tfp {
            FillPurpose::CDict => fast::fill_cdict_hash_table(ms, &view.prefix, end),
            FillPurpose::CCtx => fast::fill_hash_table_for(ms, &view.prefix, end, dtlm),
        },
        Strategy::DFast => match tfp {
            FillPurpose::CDict => double_fast::fill_cdict_double_hash_table(ms, &view.prefix, end),
            FillPurpose::CCtx => {
                double_fast::fill_double_hash_table_for(ms, &view.prefix, end, dtlm);
            }
        },
        Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 => {
            let row = params.use_row_match_finder == ParamSwitch::Enable;
            lazy::load_dictionary(ms, &view, end - HASH_READ_SIZE, row);
        }
        Strategy::BtLazy2 | Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => {
            opt::load_dictionary(ms, &view, end - HASH_READ_SIZE, end);
        }
    }
    ms.next_to_update = end;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::core::entropy::fse;
    use crate::compressor::core::entropy::huf::CTable;
    use alloc::vec::Vec;

    /// A normalized description of `symbols` equally likely symbols at
    /// `table_log`.
    fn flat_ncount(symbols: u32, table_log: u32) -> Vec<u8> {
        let mut norm = [0i16; 64];
        let total = 1i16 << table_log;
        for (s, slot) in norm.iter_mut().enumerate().take(symbols as usize) {
            *slot = total / symbols as i16 + i16::from((s as i16) < total % symbols as i16);
        }
        let mut out = [0u8; 128];
        let size = fse::write_ncount(&mut out, &norm, symbols - 1, table_log).expect("writes");
        out[..size].to_vec()
    }

    /// A Zstandard-format dictionary built from its sections.
    fn dictionary(of: &[u8], ml: &[u8], ll: &[u8], reps: [u32; 3], content: usize) -> Vec<u8> {
        let mut count = [3u32; 256];
        count[0] = 900;
        let mut table = CTable::default();
        let log = table.build(&count, 255, 11).expect("builds");
        let mut huffman = [0u8; 300];
        let size = table.write(&mut huffman, 255, log).expect("writes");
        let mut out = crate::dictionary::MAGIC_DICTIONARY.to_le_bytes().to_vec();
        out.extend_from_slice(&7u32.to_le_bytes());
        out.extend_from_slice(&huffman[..size]);
        out.extend_from_slice(of);
        out.extend_from_slice(ml);
        out.extend_from_slice(ll);
        for rep in reps {
            out.extend_from_slice(&rep.to_le_bytes());
        }
        out.extend((0..content).map(|i| i as u8));
        out
    }

    fn load(dict: &[u8]) -> Result<BlockState, DictionaryError> {
        let mut bs = BlockState::default();
        load_c_entropy(&mut bs, dict).map(|_| bs)
    }

    #[test]
    fn complete_tables_are_valid_and_partial_ones_need_checking() {
        let valid = dictionary(
            &flat_ncount(32, 6),
            &flat_ncount(53, 7),
            &flat_ncount(36, 7),
            [1, 4, 8],
            100,
        );
        let bs = load(&valid).expect("loads");
        assert_eq!(bs.entropy.huf.repeat, huf::Repeat::Valid);
        assert_eq!(bs.entropy.fse.matchlength_repeat, FseRepeat::Valid);
        assert_eq!(bs.entropy.fse.litlength_repeat, FseRepeat::Valid);
        assert_eq!(bs.entropy.fse.offcode_repeat, FseRepeat::Valid);
        assert_eq!(bs.rep, [1, 4, 8]);
        // Fewer symbols than the alphabet: usable after a check only.
        let partial = dictionary(
            &flat_ncount(10, 6),
            &flat_ncount(20, 7),
            &flat_ncount(30, 7),
            [1, 4, 8],
            100,
        );
        let bs = load(&partial).expect("loads");
        assert_eq!(bs.entropy.fse.matchlength_repeat, FseRepeat::Check);
        assert_eq!(bs.entropy.fse.litlength_repeat, FseRepeat::Check);
        assert_eq!(bs.entropy.fse.offcode_repeat, FseRepeat::Check);
        assert_eq!(dict_ncount_repeat(&[1, 0, 1], 2, 2), FseRepeat::Check);
        assert_eq!(dict_ncount_repeat(&[1, 1, 1], 2, 2), FseRepeat::Valid);
        assert_eq!(dict_ncount_repeat(&[1, 1, 1], 2, 5), FseRepeat::Check);
    }

    #[test]
    fn malformed_sections_are_corrupted() {
        let of = flat_ncount(32, 6);
        let ml = flat_ncount(53, 7);
        let ll = flat_ncount(36, 7);
        let corrupted: Result<(), _> = Err(DictionaryError::Corrupted);
        // A broken Huffman description.
        let mut broken = dictionary(&of, &ml, &ll, [1, 4, 8], 100);
        broken[8] = 0xFF;
        broken[9] = 0xFF;
        assert_eq!(load(&broken).err(), corrupted.clone().err());
        // Table logs beyond each table's limit, and an unreadable header.
        for (of, ml, ll) in [
            (flat_ncount(32, 9), ml.clone(), ll.clone()),
            (of.clone(), flat_ncount(53, 10), ll.clone()),
            (of.clone(), ml.clone(), flat_ncount(36, 10)),
            (alloc::vec![0x0F; 8], ml.clone(), ll.clone()),
            (of.clone(), alloc::vec![0x0F; 8], ll.clone()),
            (of.clone(), ml.clone(), alloc::vec![0x0F; 8]),
        ] {
            let dict = dictionary(&of, &ml, &ll, [1, 4, 8], 100);
            assert_eq!(load(&dict).err(), corrupted.clone().err());
        }
        // Repeat offsets: missing, zero, or beyond the content.
        let mut short = dictionary(&of, &ml, &ll, [1, 4, 8], 0);
        short.truncate(short.len() - 5);
        assert_eq!(load(&short).err(), corrupted.clone().err());
        for reps in [[0, 4, 8], [1, 4, 101]] {
            let dict = dictionary(&of, &ml, &ll, reps, 100);
            assert_eq!(load(&dict).err(), corrupted.clone().err());
        }
        assert!(load(&dictionary(&of, &ml, &ll, [1, 4, 100], 100)).is_ok());
    }
}

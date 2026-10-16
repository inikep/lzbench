//! Frames compressed with a dictionary against the reference's
//! `ZSTD_CCtx_loadDictionary` + `ZSTD_compress2` and `ZSTD_compressStream2`.

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::advanced_tests::oracle_params;
use super::cctx::{Engine, RequestedParams};
use super::cdict::{AttachPref, CDict};
use super::compress_once_into;
use super::params::Strategy;
use crate::dictionary::DictionaryContent;
use zstd_ffi::corpus;
use zstd_ffi::oracle::{self, Dictionary, Params};

/// A dictionary trained on JSON-like records, and the records.
pub(crate) fn trained() -> (Vec<u8>, Vec<Vec<u8>>) {
    let records = corpus::records(3000, 1);
    let samples: Vec<&[u8]> = records.iter().map(Vec::as_slice).collect();
    let dictionary = oracle::train_dictionary(&samples, 32 << 10).expect("trains");
    (dictionary, records)
}

/// Inputs shaped like the records: one record, a few, many.
pub(crate) fn inputs(records: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for count in [1usize, 3, 40, 300, 2500] {
        out.push(records[records.len() - count..].concat());
    }
    out.push(Vec::new());
    out.push(corpus::text(9_000, 5));
    out.push(corpus::binary(70_000, 6));
    out
}

fn content_type(content: DictionaryContent) -> i32 {
    match content {
        DictionaryContent::Auto => zstd_ffi::ZSTD_dct_auto,
        DictionaryContent::Raw => zstd_ffi::ZSTD_dct_rawContent,
        DictionaryContent::Zstd => zstd_ffi::ZSTD_dct_fullDict,
    }
}

fn pref_value(pref: AttachPref) -> i32 {
    match pref {
        AttachPref::Default => zstd_ffi::ZSTD_dictDefaultAttach,
        AttachPref::ForceAttach => zstd_ffi::ZSTD_dictForceAttach,
        AttachPref::ForceCopy => zstd_ffi::ZSTD_dictForceCopy,
        AttachPref::ForceLoad => zstd_ffi::ZSTD_dictForceLoad,
    }
}

/// The oracle parameters for `requested` with `dict` loaded.
pub(crate) fn dict_params(
    requested: &RequestedParams,
    dict: &[u8],
    content: DictionaryContent,
) -> Params {
    oracle_params(requested)
        .with(
            zstd_ffi::ZSTD_c_forceAttachDict,
            pref_value(requested.attach_dict_pref),
        )
        .with_dictionary(Dictionary {
            bytes: dict.to_vec(),
            content_type: content_type(content),
        })
}

/// Compresses every input with `dict` under `requested`, on a reused and
/// on a fresh engine, and compares with the reference.
pub(crate) fn check(
    requested: &RequestedParams,
    dict: &[u8],
    content: DictionaryContent,
    inputs: &[Vec<u8>],
) {
    let cdict = CDict::new(Box::from(dict), content, requested).expect("prepares");
    let mut reused = Engine::new();
    let params = dict_params(requested, dict, content);
    for (index, input) in inputs.iter().enumerate() {
        let expected = oracle::compress2(input, &params).expect("oracle");
        for engine in [&mut reused, &mut Engine::new()] {
            let mut dst = alloc::vec![0u8; super::cctx::compress_bound(input.len()).unwrap()];
            let size = compress_once_into(engine, requested, Some(&cdict), input, 7, &mut dst)
                .expect("compresses");
            assert!(
                dst[..size] == expected[..],
                "input {index} ({} bytes), {requested:?}: {size} vs {} bytes",
                input.len(),
                expected.len()
            );
        }
    }
}

pub(crate) fn request(level: i32, strategy: Option<Strategy>, pref: AttachPref) -> RequestedParams {
    let mut requested = RequestedParams {
        compression_level: level,
        attach_dict_pref: pref,
        ..RequestedParams::default()
    };
    requested.cparams.strategy = strategy;
    requested
}

#[test]
fn fast_and_dfast_copy_and_load_dictionaries_like_the_reference() {
    let (trained, records) = trained();
    let raw = corpus::text(20_000, 9);
    let inputs = inputs(&records);
    for (level, strategy) in [
        (-5, None),
        (-1, None),
        (1, None),
        (2, Some(Strategy::Fast)),
        (3, Some(Strategy::DFast)),
        (4, Some(Strategy::DFast)),
    ] {
        for pref in [AttachPref::ForceCopy, AttachPref::ForceLoad] {
            let requested = request(level, strategy, pref);
            check(&requested, &trained, DictionaryContent::Auto, &inputs);
            check(&requested, &raw, DictionaryContent::Auto, &inputs);
            check(&requested, &trained, DictionaryContent::Raw, &inputs);
        }
    }
}

#[test]
fn every_level_attaches_copies_and_loads_raw_dictionaries_like_the_reference() {
    let (_, records) = trained();
    let raw = [corpus::text(20_000, 9), records[..200].concat()].concat();
    let inputs = inputs(&records);
    for level in -3..=22 {
        for pref in [
            AttachPref::ForceAttach,
            AttachPref::ForceCopy,
            AttachPref::ForceLoad,
        ] {
            let requested = request(level, None, pref);
            check(&requested, &raw, DictionaryContent::Auto, &inputs);
        }
    }
}

#[test]
fn every_level_attaches_copies_and_loads_trained_dictionaries_like_the_reference() {
    let (trained, records) = trained();
    let inputs = inputs(&records);
    for level in -3..=22 {
        for pref in [
            AttachPref::Default,
            AttachPref::ForceAttach,
            AttachPref::ForceCopy,
            AttachPref::ForceLoad,
        ] {
            let requested = request(level, None, pref);
            check(&requested, &trained, DictionaryContent::Auto, &inputs);
        }
    }
}

#[test]
fn empty_and_tiny_dictionaries_behave_like_the_reference() {
    let (_, records) = trained();
    let inputs = inputs(&records);
    for level in [-1, 3, 9, 19] {
        for pref in [
            AttachPref::Default,
            AttachPref::ForceCopy,
            AttachPref::ForceLoad,
        ] {
            let requested = request(level, None, pref);
            for dict in [&b""[..], b"abc", b"seven b", b"eight by"] {
                check(&requested, dict, DictionaryContent::Auto, &inputs);
                check(&requested, dict, DictionaryContent::Raw, &inputs);
            }
            // Empty bytes are no dictionary even when the format is required.
            check(&requested, b"", DictionaryContent::Zstd, &inputs);
        }
    }
}

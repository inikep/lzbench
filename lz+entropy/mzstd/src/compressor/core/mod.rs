//! Private compressor implementation: a port of the reference encoder.

pub(crate) mod block;
pub(crate) mod cctx;
pub(crate) mod cdict;
#[cfg(test)]
mod dict_tests;
pub(crate) mod double_fast;
pub(crate) mod entropy;
pub(crate) mod error;
pub(crate) mod fast;
pub(crate) mod lazy;
mod lazy_box;
pub(crate) mod ldm;
pub(crate) mod ldm_geartab;
pub(crate) mod match_state;
pub(crate) mod mem;
pub(crate) mod opt;
mod opt_count;
pub(crate) mod output;
pub(crate) mod params;
pub(crate) mod presplit;
pub(crate) mod seq_store;
pub(crate) mod split_block;
pub(crate) mod stream;
pub(crate) mod window;

#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;
use core::mem::MaybeUninit;

use cctx::{Engine, RequestedParams};
use cdict::CDict;
use error::CoreError;
use window::{Addr, Segment, Segments};

/// Begins a frame of `pledged` bytes with the dictionary `segments` carry,
/// if any (`ZSTD_CCtx_init_compressStream2` through
/// `ZSTD_compressBegin_internal`).
#[inline(always)]
pub(crate) fn begin_frame(
    engine: &mut Engine,
    requested: &RequestedParams,
    pledged: u64,
    segments: &Segments<'_>,
) -> Result<(), CoreError> {
    // An empty dictionary is no dictionary (`ZSTD_CCtx_loadDictionary`
    // with a size of zero clears it).
    match segments.cdict.filter(|cdict| cdict.content_size() > 0) {
        None => {
            engine.begin(requested.resolve(pledged), pledged);
            Ok(())
        }
        Some(cdict) => {
            let pref = requested.attach_dict_pref;
            let applied = requested.resolve_for(
                pledged,
                cdict.content_size(),
                cdict.cparam_mode(pref, pledged),
            );
            engine.begin_with_dict(applied, pledged, cdict, pref, segments)
        }
    }
}

/// Compresses `input` as one frame into `dst` the way `ZSTD_compress2`
/// does: the whole input is known up front and read in place, and `dst` is
/// the reference's destination with its exact capacity.
#[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
#[inline]
pub(crate) fn compress_once_into(
    engine: &mut Engine,
    requested: &RequestedParams,
    dict: Option<&CDict>,
    input: &[u8],
    input_id: u64,
    dst: &mut [u8],
) -> Result<usize, CoreError> {
    compress_once_to(
        engine,
        requested,
        dict,
        input,
        input_id,
        &mut output::SliceOutput(dst),
    )
}

/// [`compress_once_into`] into memory that may be uninitialized, zeroed just
/// ahead of each block; on success exactly `dst[..size]` is initialized.
pub(crate) fn compress_once_uninit(
    engine: &mut Engine,
    requested: &RequestedParams,
    dict: Option<&CDict>,
    input: &[u8],
    input_id: u64,
    dst: &mut [MaybeUninit<u8>],
) -> Result<usize, CoreError> {
    compress_once_to(
        engine,
        requested,
        dict,
        input,
        input_id,
        &mut output::UninitOutput::from(dst),
    )
}

#[inline(always)]
fn compress_once_to<O: output::Output>(
    engine: &mut Engine,
    requested: &RequestedParams,
    dict: Option<&CDict>,
    input: &[u8],
    input_id: u64,
    dst: &mut O,
) -> Result<usize, CoreError> {
    let pledged = input.len() as u64;
    let segments = Segments::input(input, input_id, dict);
    begin_frame(engine, requested, pledged, &segments)?;
    let src = Addr::new(Segment::Input(input_id), 0);
    engine.compress_end_to(&segments, dst, src, input.len())
}

/// The same frame pipeline with output initialized one block at a time.
pub(crate) fn compress_once_append(
    engine: &mut Engine,
    requested: &RequestedParams,
    dict: Option<&CDict>,
    input: &[u8],
    input_id: u64,
    dst: &mut Vec<u8>,
    bound: usize,
) -> Result<usize, CoreError> {
    let start = dst.len();
    let limit = start.checked_add(bound).ok_or(CoreError::DstSizeTooSmall)?;
    dst.reserve(bound);
    let pledged = input.len() as u64;
    let segments = Segments::input(input, input_id, dict);
    begin_frame(engine, requested, pledged, &segments)?;
    let src = Addr::new(Segment::Input(input_id), 0);
    engine.compress_end_to(
        &segments,
        &mut output::VecOutput {
            buffer: dst,
            start,
            limit,
        },
        src,
        input.len(),
    )
}

/// [`compress_once_into`] into a fresh buffer of `compressBound` bytes.
#[cfg(test)]
pub(crate) fn compress_once(
    engine: &mut Engine,
    requested: &RequestedParams,
    input: &[u8],
    input_id: u64,
) -> Result<Vec<u8>, CoreError> {
    let capacity = cctx::compress_bound(input.len()).ok_or(CoreError::SrcSizeWrong)?;
    let mut dst = vec![0u8; capacity];
    let size = compress_once_into(engine, requested, None, input, input_id, &mut dst)?;
    dst.truncate(size);
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_level(level: i32, inputs: &[(&str, alloc::vec::Vec<u8>)]) {
        let mut engine = Engine::new();
        for (name, input) in inputs {
            let requested = RequestedParams {
                compression_level: level,
                ..RequestedParams::default()
            };
            let ours = compress_once(&mut engine, &requested, input, 1).expect("compress");
            let expected =
                zstd_ffi::oracle::compress2(input, &zstd_ffi::oracle::Params::level(level))
                    .expect("oracle");
            assert!(
                ours == expected,
                "level {level}, input {name} ({} bytes): {} vs {} bytes",
                input.len(),
                ours.len(),
                expected.len()
            );
        }
    }

    #[test]
    fn fast_levels_match_the_reference() {
        let inputs = zstd_ffi::corpus::standard();
        for level in [1, 2, 3, 4, -1, -2, -5, -100, -131072] {
            check_level(level, &inputs);
        }
    }

    /// The fast and dfast finders and the entropy coders dispatch on the
    /// engine's backend: every backend, scalar included, writes the
    /// reference's frames, and a reused engine (whose tables keep stale
    /// entries) writes them too.
    #[test]
    fn fast_and_dfast_levels_match_the_reference_on_every_backend() {
        let inputs = zstd_ffi::corpus::standard();
        for backend in crate::Backend::available() {
            let mut engine = Engine::with_backend(backend);
            for level in [-5, 1, 2, 3, 4] {
                for (name, input) in &inputs {
                    let requested = RequestedParams {
                        compression_level: level,
                        ..RequestedParams::default()
                    };
                    let ours = compress_once(&mut engine, &requested, input, 1).expect("compress");
                    let expected =
                        zstd_ffi::oracle::compress2(input, &zstd_ffi::oracle::Params::level(level))
                            .expect("oracle");
                    assert!(ours == expected, "{backend} level {level}, input {name}");
                }
            }
        }
    }

    #[test]
    fn optimal_levels_match_the_reference() {
        let inputs = zstd_ffi::corpus::standard();
        for level in 16..=22 {
            check_level(level, &inputs);
        }
    }

    #[test]
    fn lazy_levels_match_the_reference() {
        let inputs = zstd_ffi::corpus::standard();
        for level in 4..=15 {
            check_level(level, &inputs);
        }
    }
}

#[cfg(test)]
pub(crate) mod advanced_tests {
    use super::cctx::{FrameParams, LdmParams, RequestedParams};
    use super::params::{CParamOverrides, ParamSwitch, Strategy};
    use super::*;
    use zstd_ffi::corpus;
    use zstd_ffi::oracle::Params;
    use zstd_ffi::*;

    fn switch_value(switch: ParamSwitch) -> i32 {
        match switch {
            ParamSwitch::Auto => 0,
            ParamSwitch::Enable => 1,
            ParamSwitch::Disable => 2,
        }
    }

    /// The oracle parameters equivalent to `requested`.
    pub(crate) fn oracle_params(requested: &RequestedParams) -> Params {
        let mut params = Params::level(requested.compression_level);
        let o = requested.cparams;
        for (param, value) in [
            (ZSTD_c_windowLog, o.window_log),
            (ZSTD_c_hashLog, o.hash_log),
            (ZSTD_c_chainLog, o.chain_log),
            (ZSTD_c_searchLog, o.search_log),
            (ZSTD_c_minMatch, o.min_match),
            (ZSTD_c_targetLength, o.target_length),
        ] {
            if value != 0 {
                params = params.with(param, value as i32);
            }
        }
        if let Some(strategy) = o.strategy {
            params = params.with(ZSTD_c_strategy, strategy.value() as i32);
        }
        params = params
            .with(ZSTD_c_checksumFlag, i32::from(requested.fparams.checksum))
            .with(
                ZSTD_c_contentSizeFlag,
                i32::from(requested.fparams.content_size),
            );
        params = params.with(
            ZSTD_c_enableLongDistanceMatching,
            switch_value(requested.ldm.enable),
        );
        for (param, value) in [
            (ZSTD_c_ldmHashLog, requested.ldm.hash_log),
            (ZSTD_c_ldmMinMatch, requested.ldm.min_match_length),
            (ZSTD_c_ldmBucketSizeLog, requested.ldm.bucket_size_log),
            (ZSTD_c_ldmHashRateLog, requested.ldm.hash_rate_log),
        ] {
            if value != 0 {
                params = params.with(param, value as i32);
            }
        }
        params
    }

    fn random_request(rng: &mut corpus::Rng) -> RequestedParams {
        let mut pick = |n: u64| rng.next_u64() % n;
        let level = pick(33) as i32 - 10;
        let mut overrides = CParamOverrides::default();
        if pick(2) == 0 {
            overrides.window_log = 10 + pick(14) as u32;
        }
        if pick(2) == 0 {
            overrides.hash_log = 6 + pick(17) as u32;
        }
        if pick(2) == 0 {
            overrides.chain_log = 6 + pick(17) as u32;
        }
        if pick(2) == 0 {
            overrides.search_log = 1 + pick(9) as u32;
        }
        if pick(2) == 0 {
            overrides.min_match = 3 + pick(5) as u32;
        }
        if pick(2) == 0 {
            overrides.target_length = pick(1000) as u32;
        }
        if pick(2) == 0 {
            overrides.strategy = Strategy::from_value(1 + pick(9) as u32);
        }
        let ldm_enable = match pick(3) {
            0 => ParamSwitch::Auto,
            1 => ParamSwitch::Enable,
            _ => ParamSwitch::Disable,
        };
        let mut ldm = LdmParams {
            enable: ldm_enable,
            ..LdmParams::default()
        };
        if pick(3) == 0 {
            ldm.min_match_length = 4 + pick(120) as u32;
        }
        if pick(3) == 0 {
            ldm.hash_log = 6 + pick(16) as u32;
        }
        if pick(3) == 0 {
            ldm.bucket_size_log = 1 + pick(8) as u32;
        }
        if pick(3) == 0 {
            ldm.hash_rate_log = pick(10) as u32;
        }
        RequestedParams {
            compression_level: level,
            cparams: overrides,
            fparams: FrameParams {
                content_size: pick(2) == 0,
                checksum: pick(2) == 0,
                no_dict_id: false,
            },
            ldm,
            ..RequestedParams::default()
        }
    }

    /// A repeat far longer than a block makes long distance matching cut
    /// its sequences at every block boundary.
    #[test]
    fn long_distance_matches_spanning_blocks_match_the_reference() {
        let unit = corpus::random(600_000, 41);
        let input = [
            &unit[..],
            &unit[..],
            &corpus::text(50_000, 42)[..],
            &unit[..],
        ]
        .concat();
        let mut engine = Engine::new();
        for strategy in [
            Strategy::Fast,
            Strategy::DFast,
            Strategy::Lazy2,
            Strategy::BtOpt,
        ] {
            let mut requested = RequestedParams::default();
            requested.cparams.strategy = Some(strategy);
            requested.cparams.window_log = 22;
            requested.ldm.enable = ParamSwitch::Enable;
            let expected =
                zstd_ffi::oracle::compress2(&input, &oracle_params(&requested)).expect("oracle");
            let ours = compress_once(&mut engine, &requested, &input, 0).expect("compresses");
            assert!(ours == expected, "{strategy:?}");
            assert!(ours.len() < unit.len() + 60_000, "{strategy:?}");
        }
    }

    /// A number from a build-time variable, or `default` when it is unset
    /// or not a number.
    fn env_number(value: Option<&str>, default: u64) -> u64 {
        match value {
            Some(text) => text.parse().unwrap_or(default),
            None => default,
        }
    }

    #[test]
    fn build_time_numbers_fall_back_to_their_default() {
        assert_eq!(env_number(None, 7), 7);
        assert_eq!(env_number(Some("12"), 7), 12);
        assert_eq!(env_number(Some("twelve"), 7), 7);
    }

    #[test]
    fn random_advanced_parameters_match_the_reference() {
        let inputs = [
            corpus::text(300_000, 1),
            corpus::binary(200_000, 2),
            corpus::mixed(400_000, 3),
            corpus::text(5_000, 4),
            corpus::random(70_000, 5),
        ];
        let mut rng = corpus::Rng::new(env_number(option_env!("MZSTD_SEED"), 0xDEAD_BEEF));
        let mut engine = Engine::new();
        let mut append_engine = Engine::new();
        let default_cases = if cfg!(debug_assertions) { 30 } else { 300 };
        let cases = env_number(option_env!("MZSTD_CASES"), default_cases) as usize;
        for case in 0..cases {
            let requested = random_request(&mut rng);
            let input = &inputs[case % inputs.len()];
            let expected = zstd_ffi::oracle::compress2(input, &oracle_params(&requested));
            let ours = compress_once(&mut engine, &requested, input, case as u64);
            let mut appended = vec![0xa5; 7];
            let appended_result = compress_once_append(
                &mut append_engine,
                &requested,
                None,
                input,
                case as u64,
                &mut appended,
                cctx::compress_bound(input.len()).unwrap(),
            )
            .map(|size| appended[7..7 + size].to_vec());
            assert_eq!(&appended[..7], &[0xa5; 7]);
            assert_eq!(
                appended_result, ours,
                "Vec vs slice, case {case}: {requested:?}"
            );
            match (ours, expected) {
                (Ok(ours), Ok(expected)) => assert!(
                    ours == expected,
                    "case {case}: {requested:?}, input {} bytes: {} vs {} bytes",
                    input.len(),
                    ours.len(),
                    expected.len()
                ),
                (Err(error), Ok(_)) => panic!("case {case}: {requested:?}: we failed with {error}"),
                (Ok(_), Err(error)) => {
                    panic!("case {case}: {requested:?}: reference failed with {error}")
                }
                (Err(_), Err(_)) => {}
            }
        }
    }
}

#[cfg(test)]
mod stream_tests {
    use super::advanced_tests::oracle_params;
    use super::cctx::{Engine, RequestedParams};
    use super::params::Strategy;
    use super::stream::{EndOp, Stream};
    use super::*;
    use zstd_ffi::corpus;
    use zstd_ffi::oracle::Directive;

    /// Streams `steps` the way `zstd_ffi::oracle::compress_stream` does:
    /// a fresh 128 KiB output buffer per call, repeated until each step's
    /// input is consumed and its directive completed.
    fn stream(
        engine: &mut Engine,
        requested: &RequestedParams,
        steps: &[(&[u8], EndOp)],
    ) -> Vec<u8> {
        let mut state = Stream::new();
        let mut out = Vec::new();
        let mut chunk = vec![0u8; 1 << 17];
        let mut run = |data: &[u8], op: EndOp, state: &mut Stream| {
            let mut in_pos = 0;
            loop {
                let step = state
                    .compress(engine, requested, None, &mut chunk[..], &data[in_pos..], op)
                    .expect("stream");
                in_pos += step.consumed;
                out.extend_from_slice(&chunk[..step.produced]);
                let done = match op {
                    EndOp::Continue => in_pos == data.len(),
                    EndOp::Flush | EndOp::End => step.remaining == 0,
                };
                if done {
                    break;
                }
            }
        };
        for &(data, op) in steps {
            run(data, op, &mut state);
        }
        if steps.last().is_none_or(|&(_, op)| op != EndOp::End) {
            run(&[], EndOp::End, &mut state);
        }
        out
    }

    fn directive(op: EndOp) -> Directive {
        match op {
            EndOp::Continue => Directive::Continue,
            EndOp::Flush => Directive::Flush,
            EndOp::End => Directive::End,
        }
    }

    fn schedules(
        input: &[u8],
        rng: &mut corpus::Rng,
    ) -> Vec<Vec<(core::ops::Range<usize>, EndOp)>> {
        let mut all = Vec::new();
        // Whole input at once, continue then end, and end in one call.
        all.push(vec![(0..input.len(), EndOp::End)]);
        all.push(vec![(0..input.len(), EndOp::Continue)]);
        // Random chunks with occasional flushes.
        for _ in 0..3 {
            let mut steps = Vec::new();
            let mut pos = 0;
            while pos < input.len() {
                let len = (rng.next_u64() % 200_000) as usize + 1;
                let end = (pos + len).min(input.len());
                let op = match rng.next_u64() % 6 {
                    0 => EndOp::Flush,
                    _ => EndOp::Continue,
                };
                steps.push((pos..end, op));
                pos = end;
            }
            all.push(steps);
        }
        all
    }

    #[test]
    fn streamed_frames_match_the_reference() {
        let inputs = [
            corpus::text(700_000, 21),
            corpus::mixed(600_000, 22),
            corpus::binary(150_000, 23),
            corpus::text(3_000, 24),
            Vec::new(),
        ];
        let mut rng = corpus::Rng::new(77);
        let mut engine = Engine::new();
        let levels: &[i32] = if cfg!(debug_assertions) {
            &[1, 3]
        } else {
            &[-3, 1, 2, 3, 5, 6, 9, 12, 13, 16, 18, 19, 22]
        };
        for &level in levels {
            for (i, input) in inputs.iter().enumerate() {
                for (pledge, window_log) in [(false, 0u32), (true, 0), (false, 17), (false, 20)] {
                    let mut requested = RequestedParams {
                        compression_level: level,
                        ..RequestedParams::default()
                    };
                    requested.cparams.window_log = window_log;
                    requested.fparams.checksum = level % 2 == 0;
                    for steps in schedules(input, &mut rng) {
                        let ours_steps: Vec<(&[u8], EndOp)> = steps
                            .iter()
                            .map(|(r, op)| (&input[r.clone()], *op))
                            .collect();
                        let theirs_steps: Vec<(&[u8], Directive)> = ours_steps
                            .iter()
                            .map(|&(d, op)| (d, directive(op)))
                            .collect();
                        let mut params = oracle_params(&requested);
                        let mut state_requested = requested;
                        let mut stream_state_pledge = None;
                        if pledge {
                            params = params.with_pledged(input.len() as u64);
                            stream_state_pledge = Some(input.len() as u64);
                        }
                        let theirs = zstd_ffi::oracle::compress_stream(&params, &theirs_steps)
                            .expect("oracle");
                        let ours = match stream_state_pledge {
                            None => stream(&mut engine, &state_requested, &ours_steps),
                            Some(size) => {
                                state_requested.src_size_hint = 0;
                                stream_pledged(&mut engine, &state_requested, size, &ours_steps)
                            }
                        };
                        assert!(
                            ours == theirs,
                            "level {level} input {i} pledge {pledge} wlog {window_log} steps {steps:?}: {} vs {}",
                            ours.len(),
                            theirs.len()
                        );
                    }
                }
            }
        }
    }

    /// Windows of 16 KiB and less resolve greedy, lazy and lazy2 to the
    /// hash chain finder rather than the row one, so a stream that wraps
    /// such a window runs their extDict variants.
    #[test]
    fn small_window_streams_run_every_ext_dict_finder_like_the_reference() {
        let input = [corpus::text(150_000, 31), corpus::mixed(150_000, 32)].concat();
        let steps: Vec<(&[u8], EndOp)> = input
            .chunks(10_007)
            .map(|chunk| (chunk, EndOp::Continue))
            .collect();
        let theirs_steps: Vec<(&[u8], Directive)> = steps
            .iter()
            .map(|&(data, op)| (data, directive(op)))
            .collect();
        let mut engine = Engine::new();
        for strategy in [
            Strategy::Fast,
            Strategy::DFast,
            Strategy::Greedy,
            Strategy::Lazy,
            Strategy::Lazy2,
            Strategy::BtLazy2,
            Strategy::BtOpt,
        ] {
            for window_log in [10, 12, 14] {
                let mut requested = RequestedParams::default();
                requested.cparams.strategy = Some(strategy);
                requested.cparams.window_log = window_log;
                let theirs =
                    zstd_ffi::oracle::compress_stream(&oracle_params(&requested), &theirs_steps)
                        .expect("oracle");
                let ours = stream(&mut engine, &requested, &steps);
                assert!(ours == theirs, "{strategy:?} window {window_log}");
            }
        }
    }

    /// Every minimum match length of the fast and dfast extDict finders,
    /// on every backend, over streams that wrap a small window.
    #[test]
    fn fast_ext_dict_finders_match_the_reference_on_every_backend() {
        let input = [corpus::text(60_000, 41), corpus::binary(60_000, 42)].concat();
        let steps: Vec<(&[u8], EndOp)> = input
            .chunks(7_001)
            .map(|chunk| (chunk, EndOp::Continue))
            .collect();
        let theirs_steps: Vec<(&[u8], Directive)> = steps
            .iter()
            .map(|&(data, op)| (data, directive(op)))
            .collect();
        for strategy in [Strategy::Fast, Strategy::DFast] {
            for min_match in 4..=7 {
                let mut requested = RequestedParams::default();
                requested.cparams.strategy = Some(strategy);
                requested.cparams.window_log = 12;
                requested.cparams.min_match = min_match;
                let theirs =
                    zstd_ffi::oracle::compress_stream(&oracle_params(&requested), &theirs_steps)
                        .expect("oracle");
                for backend in crate::Backend::available() {
                    let mut engine = Engine::with_backend(backend);
                    let ours = stream(&mut engine, &requested, &steps);
                    assert!(
                        ours == theirs,
                        "{strategy:?} min match {min_match} {backend}"
                    );
                }
            }
        }
    }

    /// The noDict variants for every minimum match length, on every
    /// backend.
    #[test]
    fn fast_finders_match_the_reference_for_every_min_match_and_backend() {
        let input = [corpus::text(200_000, 43), corpus::binary(100_000, 44)].concat();
        for strategy in [Strategy::Fast, Strategy::DFast] {
            for min_match in 4..=7 {
                let mut requested = RequestedParams::default();
                requested.cparams.strategy = Some(strategy);
                requested.cparams.min_match = min_match;
                let theirs = zstd_ffi::oracle::compress2(&input, &oracle_params(&requested))
                    .expect("oracle");
                for backend in crate::Backend::available() {
                    let mut engine = Engine::with_backend(backend);
                    let ours = compress_once(&mut engine, &requested, &input, 1).expect("ours");
                    assert!(
                        ours == theirs,
                        "{strategy:?} min match {min_match} {backend}"
                    );
                }
            }
        }
    }

    fn stream_pledged(
        engine: &mut Engine,
        requested: &RequestedParams,
        size: u64,
        steps: &[(&[u8], EndOp)],
    ) -> Vec<u8> {
        let mut state = Stream::new();
        state.set_pledged_src_size(size).expect("pledge");
        let mut out = Vec::new();
        let mut chunk = vec![0u8; 1 << 17];
        let mut run = |data: &[u8], op: EndOp, state: &mut Stream| {
            let mut in_pos = 0;
            loop {
                let step = state
                    .compress(engine, requested, None, &mut chunk[..], &data[in_pos..], op)
                    .expect("stream");
                in_pos += step.consumed;
                out.extend_from_slice(&chunk[..step.produced]);
                let done = match op {
                    EndOp::Continue => in_pos == data.len(),
                    EndOp::Flush | EndOp::End => step.remaining == 0,
                };
                if done {
                    break;
                }
            }
        };
        for &(data, op) in steps {
            run(data, op, &mut state);
        }
        if steps.last().is_none_or(|&(_, op)| op != EndOp::End) {
            run(&[], EndOp::End, &mut state);
        }
        out
    }
}

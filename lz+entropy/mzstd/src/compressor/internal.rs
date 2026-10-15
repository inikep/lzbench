//! Lowering of the public configuration into the core's parameter shape.
//!
//! The core is written against the reference's `ZSTD_CCtx_params`; keeping
//! the translation in one place lets the public surface change without
//! touching anything the compressed bytes depend on.

use super::config::{EncoderConfig, LongDistanceMatching};
use super::core::cctx::{FrameParams, LdmParams, RequestedParams};
use super::core::params::{CParamOverrides, ParamSwitch};

/// The reference parameters equivalent to `config`.
pub(crate) fn requested_params(config: &EncoderConfig) -> RequestedParams {
    let p = config.params();
    let cparams = CParamOverrides {
        window_log: p.window_log().unwrap_or(0),
        chain_log: p.chain_log().unwrap_or(0),
        hash_log: p.hash_log().unwrap_or(0),
        search_log: p.search_log().unwrap_or(0),
        min_match: p.min_match().unwrap_or(0),
        target_length: p.target_length().unwrap_or(0),
        strategy: p.strategy().map(super::config::Strategy::to_core),
    };
    let ldm = match config.long_distance_matching() {
        LongDistanceMatching::Auto => LdmParams::default(),
        LongDistanceMatching::Disabled => LdmParams {
            enable: ParamSwitch::Disable,
            ..LdmParams::default()
        },
        LongDistanceMatching::Enabled(params) => LdmParams {
            enable: ParamSwitch::Enable,
            hash_log: params.hash_log().unwrap_or(0),
            bucket_size_log: params.bucket_size_log().unwrap_or(0),
            min_match_length: params.min_match().unwrap_or(0),
            hash_rate_log: params.hash_rate_log().unwrap_or(0),
            window_log: 0,
        },
    };
    RequestedParams {
        compression_level: config.level().get(),
        cparams,
        fparams: FrameParams {
            content_size: config.content_size(),
            checksum: config.checksum(),
            no_dict_id: false,
        },
        src_size_hint: config.size_hint().unwrap_or(0),
        ldm,
        ..RequestedParams::default()
    }
}

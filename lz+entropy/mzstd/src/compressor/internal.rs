//! Lowering of the public configuration into the core's parameter shape.
//!
//! The core is written against the reference's `ZSTD_CCtx_params`; keeping
//! the translation in one place lets the public surface change without
//! touching anything the compressed bytes depend on.

use super::config::{EncoderConfig, Strategy};
use super::core::cctx::{FrameParams, LdmParams, RequestedParams};
use super::core::params::{CParamOverrides, TARGET_C_BLOCK_SIZE_BOUNDS};

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
        strategy: p.strategy().map(Strategy::to_core),
    };
    let ldm = config.ldm_params();
    RequestedParams {
        compression_level: config.level().get(),
        cparams,
        fparams: FrameParams {
            content_size: config.content_size(),
            checksum: config.checksum(),
            no_dict_id: !config.dictionary_id(),
            format: config.format(),
        },
        src_size_hint: config.size_hint().unwrap_or(0),
        literal_compression: config.literal_compression().to_core(),
        ldm: LdmParams {
            enable: config.ldm_switch().to_core(),
            hash_log: ldm.hash_log().unwrap_or(0),
            bucket_size_log: ldm.bucket_size_log().unwrap_or(0),
            min_match_length: ldm.min_match().unwrap_or(0),
            hash_rate_log: ldm.hash_rate_log().unwrap_or(0),
            window_log: 0,
        },
        post_block_splitter: config.post_block_splitter().to_core(),
        pre_block_splitter_level: i32::from(config.block_splitter_level()),
        max_block_size: config.max_block_size().unwrap_or(0) as usize,
        // `ZSTD_CCtxParams_setParameter` raises a target to the minimum.
        target_c_block_size: config.target_compressed_block_size().map_or(0, |size| {
            (size.get() as usize).max(TARGET_C_BLOCK_SIZE_BOUNDS.lower as usize)
        }),
        use_row_match_finder: config.row_match_finder().to_core(),
        attach_dict_pref: config.dictionary_attach().to_core(),
        force_window: config.force_max_window(),
    }
}

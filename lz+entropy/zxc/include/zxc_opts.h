/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * ZXC - High-performance lossless compression
 * Copyright (c) Bertrand Lebonnois and contributors.
 */

/**
 * @file zxc_opts.h
 * @brief Compression and decompression parameters.
 *
 * The parameter identifiers set on a context with zxc_cctx_set_param() and
 * zxc_dctx_set_param() (@ref zxc_cparam_t, @ref zxc_dparam_t), their bounds,
 * and @ref zxc_progress_callback_t. A configured context carries these
 * settings into every public API: one-shot buffer, multi-threaded @c FILE*
 * streaming, push streaming, block. No need to include it directly: the API
 * header you actually use (@c zxc_buffer.h, @c zxc_stream.h, @c zxc_pstream.h,
 * ...) pulls it in.
 */

#ifndef ZXC_OPTS_H
#define ZXC_OPTS_H

#include <stddef.h>
#include <stdint.h>

#include "zxc_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Progress callback function type.
 * Called from the writer thread after each block, or batch of blocks, is processed.
 * @param[in] bytes_processed Input bytes processed so far.
 * @param[in] bytes_total     Total input bytes to process (0 if unknown, e.g. stdin).
 * @param[in] user_data       Context pointer passed through from the API call.
 * @note Keep it fast and non-blocking: no heavy I/O, no mutex.
 */
typedef void (*zxc_progress_callback_t)(uint64_t bytes_processed, uint64_t bytes_total,
                                        const void* user_data);

/**
 * @brief Compression parameters, set on a @c zxc_cctx with zxc_cctx_set_param().
 *
 * Every parameter has a default, which the value 0 restores (zxc_cctx_reset()
 * restores them all). Any other value must sit within the bounds reported by
 * zxc_cparam_bounds(). Identifiers are stable: new parameters are appended,
 * and an existing value never changes meaning. The dictionary and the
 * progress callback are pointers, so they have their own setters,
 * zxc_cctx_set_dict() and zxc_cctx_set_progress().
 */
typedef enum {
    /** Compression level, 1 to 7 (@ref zxc_compression_level_t); default
     *  @ref ZXC_LEVEL_DEFAULT. */
    ZXC_C_LEVEL = 1,
    /** Block size in bytes, a power of two in [@ref ZXC_BLOCK_SIZE_MIN,
     *  @ref ZXC_BLOCK_SIZE_MAX]; default @ref ZXC_BLOCK_SIZE_DEFAULT. */
    ZXC_C_BLOCK_SIZE = 2,
    /** Workers of the @c FILE* pipeline, at most @ref ZXC_MAX_THREADS; the
     *  reader and writer come on top. 0 (default): online processors, capped
     *  at @ref ZXC_MAX_THREADS, minus one; at least one, at most 511. The
     *  buffer, block and push APIs are single-threaded and ignore it. */
    ZXC_C_THREADS = 3,
    /** 1 appends the block checksums and the archive digest; default 0. */
    ZXC_C_CHECKSUM = 4,
    /** 1 appends a seek table for random-access decompression; default 0. */
    ZXC_C_SEEKABLE = 5
} zxc_cparam_t;

/**
 * @brief Decompression parameters, set on a @c zxc_dctx with zxc_dctx_set_param().
 *
 * Same rules as @ref zxc_cparam_t, so 0 restores the default and other values
 * must sit within zxc_dparam_bounds(). The dictionary and the progress callback
 * go through zxc_dctx_set_dict() and zxc_dctx_set_progress().
 */
typedef enum {
    /** Workers of the @c FILE* pipeline, at most @ref ZXC_MAX_THREADS; the
     *  reader and writer come on top. 0 (default): online processors, capped
     *  at @ref ZXC_MAX_THREADS, minus one unless the workers read a regular
     *  file themselves; at least one, at most 511 or 512. */
    ZXC_D_THREADS = 1,
    /** 1 verifies the block checksums and the archive digest when the archive
     *  carries them; default 0, no verification. */
    ZXC_D_CHECKSUM = 2
} zxc_dparam_t;

/**
 * @brief Reports the accepted range of a compression parameter.
 *
 * Lets a caller, or a binding, validate a value before setting it, and tells
 * whether this library knows @p param at all. The range is necessary, not
 * always sufficient: @ref ZXC_C_BLOCK_SIZE must also be a power of two, which
 * only zxc_cctx_set_param() checks.
 *
 * @param[in]  param Parameter identifier.
 * @param[out] min   Lowest accepted value; may be NULL.
 * @param[out] max   Highest accepted value; may be NULL.
 * @return @ref ZXC_OK, or @ref ZXC_ERROR_BAD_PARAM for an unknown @p param.
 */
ZXC_EXPORT int zxc_cparam_bounds(zxc_cparam_t param, int64_t* min, int64_t* max);

/**
 * @brief Reports the accepted range of a decompression parameter.
 * @see zxc_cparam_bounds
 */
ZXC_EXPORT int zxc_dparam_bounds(zxc_dparam_t param, int64_t* min, int64_t* max);

#ifdef __cplusplus
}
#endif

#endif  // ZXC_OPTS_H

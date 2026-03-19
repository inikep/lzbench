/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_buffer.h
 * @brief Buffer-based (single-shot) compression and decompression API.
 *
 * This header exposes the simplest way to use ZXC: pass an entire input buffer
 * and receive the result in a single output buffer.  All functions in this
 * header are single-threaded and blocking.
 *
 * @par Typical usage
 * @code
 * // Compress
 * size_t bound = zxc_compress_bound(src_size);
 * void *dst    = malloc(bound);
 * zxc_compress_opts_t opts = { .level = ZXC_LEVEL_DEFAULT, .checksum = 1 };
 * int64_t csize = zxc_compress(src, src_size, dst, bound, &opts);
 *
 * // Decompress
 * uint64_t orig = zxc_get_decompressed_size(dst, csize);
 * void *out     = malloc(orig);
 * zxc_decompress_opts_t dopts = { .checksum = 1 };
 * int64_t dsize = zxc_decompress(dst, csize, out, orig, &dopts);
 * @endcode
 *
 * @see zxc_stream.h  for the streaming (multi-threaded) API.
 * @see zxc_sans_io.h for the low-level sans-I/O building blocks.
 */

#ifndef ZXC_BUFFER_H
#define ZXC_BUFFER_H

#include <stddef.h>
#include <stdint.h>

#include "zxc_export.h"
#include "zxc_stream.h" /* zxc_compress_opts_t, zxc_decompress_opts_t */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup library_info Library Information
 * @brief Runtime-queryable library metadata.
 *
 * These functions allow callers (including filesystem integrations)
 * to discover the supported compression level range and library version at
 * runtime, without relying on compile-time constants alone.
 * @{
 */

/**
 * @brief Returns the minimum supported compression level.
 *
 * Currently returns @ref ZXC_LEVEL_FASTEST (1).
 *
 * @return Minimum compression level value.
 */
ZXC_EXPORT int zxc_min_level(void);

/**
 * @brief Returns the maximum supported compression level.
 *
 * Currently returns @ref ZXC_LEVEL_COMPACT (5).
 *
 * @return Maximum compression level value.
 */
ZXC_EXPORT int zxc_max_level(void);

/**
 * @brief Returns the default compression level.
 *
 * Currently returns @ref ZXC_LEVEL_DEFAULT (3).
 *
 * @return Default compression level value.
 */
ZXC_EXPORT int zxc_default_level(void);

/**
 * @brief Returns the human-readable library version string.
 *
 * The returned pointer is a compile-time constant and must not be freed.
 * Example: "0.9.1".
 *
 * @return Null-terminated version string.
 */
ZXC_EXPORT const char* zxc_version_string(void);

/** @} */ /* end of library_info */

/**
 * @defgroup buffer_api Buffer API
 * @brief Single-shot, buffer-based compression and decompression.
 * @{
 */

/**
 * @brief Calculates the maximum theoretical compressed size for a given input.
 *
 * Useful for allocating output buffers before compression.
 * Accounts for file headers, block headers, and potential expansion
 * of incompressible data.
 *
 * @param[in] input_size Size of the input data in bytes.
 *
 * @return           Maximum required buffer size in bytes.
 */
ZXC_EXPORT uint64_t zxc_compress_bound(const size_t input_size);

/**
 * @brief Compresses a data buffer using the ZXC algorithm.
 *
 * This version uses standard size_t types and void pointers.
 * It executes in a single thread (blocking operation).
 * It writes the ZXC file header followed by compressed blocks.
 *
 * @param[in] src          Pointer to the source buffer.
 * @param[in] src_size     Size of the source data in bytes.
 * @param[out] dst          Pointer to the destination buffer.
 * @param[in] dst_capacity Maximum capacity of the destination buffer.
 * @param[in] opts         Compression options (NULL uses all defaults).
 *                         Only @c level, @c block_size, and @c checksum are used.
 *
 * @return The number of bytes written to dst (>0 on success),
 *         or a negative zxc_error_t code (e.g., ZXC_ERROR_DST_TOO_SMALL) on failure.
 */
ZXC_EXPORT int64_t zxc_compress(const void* src, const size_t src_size, void* dst,
                                const size_t dst_capacity, const zxc_compress_opts_t* opts);

/**
 * @brief Decompresses a ZXC compressed buffer.
 *
 * This version uses standard size_t types and void pointers.
 * It executes in a single thread (blocking operation).
 * It expects a valid ZXC file header followed by compressed blocks.
 *
 * @param[in] src          Pointer to the source buffer containing compressed data.
 * @param[in] src_size      Size of the compressed data in bytes.
 * @param[out] dst          Pointer to the destination buffer.
 * @param[in] dst_capacity  Capacity of the destination buffer.
 * @param[in] opts          Decompression options (NULL uses all defaults).
 *                          Only @c checksum is used.
 *
 * @return The number of bytes written to dst (>0 on success),
 *         or a negative zxc_error_t code (e.g., ZXC_ERROR_CORRUPT_DATA) on failure.
 */
ZXC_EXPORT int64_t zxc_decompress(const void* src, const size_t src_size, void* dst,
                                  const size_t dst_capacity, const zxc_decompress_opts_t* opts);

/**
 * @brief Returns the decompressed size stored in a ZXC compressed buffer.
 *
 * This function reads the file footer to extract the original uncompressed size
 * without performing any decompression. Useful for allocating output buffers.
 *
 * @param[in] src       Pointer to the compressed data buffer.
 * @param[in] src_size  Size of the compressed data in bytes.
 *
 * @return The original uncompressed size in bytes, or 0 if the buffer is invalid
 *         or too small to contain a valid ZXC archive.
 */
ZXC_EXPORT uint64_t zxc_get_decompressed_size(const void* src, const size_t src_size);

/* ========================================================================= */
/*  Block-Level API (no file framing)                                        */
/* ========================================================================= */

/**
 * @defgroup block_api Block API
 * @brief Single-block compression/decompression without file framing.
 *
 * These functions compress or decompress a single independent block, producing
 * only the block header (8 bytes) + compressed payload + optional checksum (4 bytes).
 * No file header, EOF block, or footer is written.
 *
 * This API is designed for filesystem integrations where the filesystem manages its own block
 * indexing and each block is compressed independently.
 *
 * @par Typical usage
 * @code
 * // Compress a single filesystem block
 * zxc_cctx* cctx = zxc_create_cctx(NULL);
 * zxc_compress_opts_t opts = { .level = 3 };
 * size_t bound = zxc_compress_block_bound(block_size);
 * void *dst = malloc(bound);
 * int64_t csize = zxc_compress_block(cctx, block, block_size, dst, bound, &opts);
 *
 * // Decompress
 * zxc_dctx* dctx = zxc_create_dctx();
 * int64_t dsize = zxc_decompress_block(dctx, dst, csize, out, block_size, NULL);
 *
 * zxc_free_cctx(cctx);
 * zxc_free_dctx(dctx);
 * @endcode
 * @{
 */

/* Forward declarations for context types (defined below). */
typedef struct zxc_cctx_s zxc_cctx;
typedef struct zxc_dctx_s zxc_dctx;

/**
 * @brief Returns the maximum compressed size for a single block.
 *
 * Unlike zxc_compress_bound(), this does NOT include file header,
 * EOF block, or footer overhead.  Use this to size the destination
 * buffer for zxc_compress_block().
 *
 * @param[in] input_size Size of the uncompressed block in bytes.
 * @return Upper bound on compressed block size, or 0 on overflow.
 */
ZXC_EXPORT uint64_t zxc_compress_block_bound(size_t input_size);

/**
 * @brief Compresses a single block without file framing.
 *
 * Output format: @c block_header(8B) + payload + optional @c checksum(4B).
 * The output can be decompressed with zxc_decompress_block().
 *
 * @param[in,out] cctx         Reusable compression context.
 * @param[in]     src          Source data.
 * @param[in]     src_size     Source data size in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of the destination buffer
 *                             (use zxc_compress_block_bound() to size).
 * @param[in]     opts         Compression options, or NULL for defaults.
 *                             Only @c level, @c block_size, and
 *                             @c checksum_enabled are used.
 *
 * @return Compressed block size in bytes (> 0) on success,
 *         or a negative @ref zxc_error_t code on failure.
 */
ZXC_EXPORT int64_t zxc_compress_block(zxc_cctx* cctx, const void* src, size_t src_size, void* dst,
                                      size_t dst_capacity, const zxc_compress_opts_t* opts);

/**
 * @brief Decompresses a single block produced by zxc_compress_block().
 *
 * @param[in,out] dctx         Reusable decompression context.
 * @param[in]     src          Compressed block data.
 * @param[in]     src_size     Compressed data size in bytes.
 * @param[out]    dst          Destination buffer for decompressed data.
 * @param[in]     dst_capacity Capacity of the destination buffer (must be
 *                             at least the original uncompressed size).
 * @param[in]     opts         Decompression options (NULL for defaults).
 *                             Only @c checksum_enabled is used.
 *
 * @return Decompressed size in bytes (> 0) on success,
 *         or a negative @ref zxc_error_t code on failure.
 */
ZXC_EXPORT int64_t zxc_decompress_block(zxc_dctx* dctx, const void* src, size_t src_size, void* dst,
                                        size_t dst_capacity, const zxc_decompress_opts_t* opts);

/** @} */ /* end of block_api */

/* ========================================================================= */
/*  Reusable Context API (opaque, heap-allocated)                            */
/* ========================================================================= */

/**
 * @defgroup context_api Reusable Context API
 * @brief Opaque, reusable compression / decompression contexts.
 *
 * This API eliminates per-call allocation overhead by letting callers retain
 * a context across multiple operations.  The internal layout is hidden behind
 * an opaque pointer.
 *
 * @{
 */

/* --- Compression context ------------------------------------------------- */

/**
 * @brief Creates a new reusable compression context.
 *
 * When @p opts is non-NULL the context pre-allocates all internal buffers
 * using the supplied level, block_size, and checksum_enabled settings.
 * When @p opts is NULL, allocation is deferred to the first call to
 * zxc_compress_cctx().
 *
 * The returned context must be freed with zxc_free_cctx().
 *
 * @param[in] opts  Compression options for eager init, or NULL for lazy init.
 * @return Pointer to the new context, or @c NULL on allocation failure.
 */
ZXC_EXPORT zxc_cctx* zxc_create_cctx(const zxc_compress_opts_t* opts);

/**
 * @brief Frees a compression context and all associated resources.
 *
 * It is safe to pass @c NULL; the call is a no-op in that case.
 *
 * @param[in] cctx Context to free.
 */
ZXC_EXPORT void zxc_free_cctx(zxc_cctx* cctx);

/**
 * @brief Compresses data using a reusable context.
 *
 * Identical to zxc_compress() but reuses the internal buffers from @p cctx,
 * avoiding per-call malloc/free overhead.  The context automatically
 * re-initializes when block_size or level changes between calls.
 *
 * Options are **sticky**: settings passed via @p opts are remembered and
 * reused on subsequent calls where @p opts is NULL.  The initial sticky
 * values come from the @p opts passed to zxc_create_cctx().
 *
 * @param[in,out] cctx         Reusable compression context.
 * @param[in]     src          Source data.
 * @param[in]     src_size     Source data size in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of the destination buffer.
 * @param[in]     opts         Compression options, or NULL to reuse
 *                             settings from create / last call.
 *
 * @return Compressed size in bytes (> 0) on success,
 *         or a negative @ref zxc_error_t code on failure.
 */
ZXC_EXPORT int64_t zxc_compress_cctx(zxc_cctx* cctx, const void* src, size_t src_size, void* dst,
                                     size_t dst_capacity, const zxc_compress_opts_t* opts);

/* --- Decompression context ----------------------------------------------- */

/**
 * @brief Creates a new reusable decompression context.
 *
 * @return Pointer to the new context, or @c NULL on allocation failure.
 */
ZXC_EXPORT zxc_dctx* zxc_create_dctx(void);

/**
 * @brief Frees a decompression context and all associated resources.
 *
 * It is safe to pass @c NULL.
 *
 * @param[in] dctx Context to free.
 */
ZXC_EXPORT void zxc_free_dctx(zxc_dctx* dctx);

/**
 * @brief Decompresses data using a reusable context.
 *
 * Identical to zxc_decompress() but reuses buffers from @p dctx.
 *
 * @param[in,out] dctx         Reusable decompression context.
 * @param[in]     src          Compressed data.
 * @param[in]     src_size     Compressed data size in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of the destination buffer.
 * @param[in]     opts         Decompression options (NULL for defaults).
 *
 * @return Decompressed size in bytes (> 0) on success,
 *         or a negative @ref zxc_error_t code on failure.
 */
ZXC_EXPORT int64_t zxc_decompress_dctx(zxc_dctx* dctx, const void* src, size_t src_size, void* dst,
                                       size_t dst_capacity, const zxc_decompress_opts_t* opts);

/** @} */ /* end of context_api */
/** @} */ /* end of buffer_api */

#ifdef __cplusplus
}
#endif

#endif  // ZXC_BUFFER_H
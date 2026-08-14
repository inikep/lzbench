/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_buffer.h
 * @brief One-shot buffer compression and decompression.
 *
 * The simplest way to use ZXC: one input buffer in, one output buffer out.
 * Everything here is single-threaded and blocking.
 *
 * @par Typical usage
 * @code
 * // Compress
 * size_t bound = zxc_compress_bound(src_size);
 * void *dst    = malloc(bound);
 * zxc_compress_opts_t opts = { .level = ZXC_LEVEL_DEFAULT, .checksum_enabled = 1 };
 * int64_t csize = zxc_compress(src, src_size, dst, bound, &opts);
 *
 * // Decompress
 * uint64_t orig = zxc_get_decompressed_size(dst, csize);
 * void *out     = malloc(orig);
 * zxc_decompress_opts_t dopts = { .checksum_enabled = 1 };
 * int64_t dsize = zxc_decompress(dst, csize, out, orig, &dopts);
 * @endcode
 *
 * @see zxc_stream.h  multi-threaded @c FILE* streaming.
 * @see zxc_pstream.h single-threaded push streaming.
 */

#ifndef ZXC_BUFFER_H
#define ZXC_BUFFER_H

#include <stddef.h>
#include <stdint.h>

#include "zxc_export.h"
#include "zxc_opts.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup library_info Library Information
 * @brief Runtime-queryable library metadata.
 *
 * Lets callers (filesystem integrations, bindings) discover the supported
 * level range and the version at runtime instead of from compile-time
 * constants alone.
 * @{
 */

/**
 * @brief Minimum supported compression level.
 * @return @ref ZXC_LEVEL_FASTEST (1).
 */
ZXC_EXPORT int zxc_min_level(void);

/**
 * @brief Maximum supported compression level.
 * @return @ref ZXC_LEVEL_ULTRA (7).
 */
ZXC_EXPORT int zxc_max_level(void);

/**
 * @brief Default compression level.
 * @return @ref ZXC_LEVEL_DEFAULT (3).
 */
ZXC_EXPORT int zxc_default_level(void);

/**
 * @brief Library version string, "MAJOR.MINOR.PATCH" (e.g. "0.13.1").
 * @return Null-terminated compile-time constant; do not free.
 */
ZXC_EXPORT const char* zxc_version_string(void);

/** @} */ /* end of library_info */

/**
 * @defgroup buffer_api Buffer API
 * @brief Single-shot, buffer-based compression and decompression.
 * @{
 */

/**
 * @brief Maximum compressed size for an input of @p input_size bytes.
 *
 * Covers the file header, block headers, and worst-case expansion of
 * incompressible data. Use it to size the destination buffer.
 *
 * @param[in] input_size Input size in bytes.
 * @return Required destination capacity in bytes.
 */
ZXC_EXPORT uint64_t zxc_compress_bound(const size_t input_size);

/**
 * @brief Compresses a buffer into a complete ZXC archive.
 *
 * Writes the file header followed by compressed blocks. Single-threaded and
 * blocking, so @c n_threads and the progress callback in @p opts are ignored.
 *
 * @param[in]  src          Source buffer.
 * @param[in]  src_size     Source size in bytes.
 * @param[out] dst          Destination buffer.
 * @param[in]  dst_capacity Capacity of @p dst.
 * @param[in]  opts         Compression options, or NULL for defaults.
 *                           @c n_threads and the progress callback are ignored
 *                          (this call is single-threaded and blocking).
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 * @note Levels above @ref ZXC_LEVEL_ULTRA are silently clamped to it;
 *       levels <= 0 select @ref ZXC_LEVEL_DEFAULT.
 *
 * @return Bytes written to @p dst (> 0), or a negative @ref zxc_error_t
 *         (e.g. @ref ZXC_ERROR_DST_TOO_SMALL).
 */
ZXC_EXPORT int64_t zxc_compress(const void* src, const size_t src_size, void* dst,
                                const size_t dst_capacity, const zxc_compress_opts_t* opts);

/**
 * @brief Decompresses a complete ZXC archive.
 *
 * Expects a valid file header followed by compressed blocks. Single-threaded
 * and blocking, so @c n_threads and the progress callback in @p opts are
 * ignored.
 *
 * @param[in]  src          Compressed buffer.
 * @param[in]  src_size     Compressed size in bytes.
 * @param[out] dst          Destination buffer.
 * @param[in]  dst_capacity Capacity of @p dst.
 * @param[in]  opts         Decompression options, or NULL for defaults.
 *                          @c n_threads and the progress callback are ignored
 *                         (this call is single-threaded and blocking).
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Bytes written to @p dst (> 0), or a negative @ref zxc_error_t
 *         (e.g. @ref ZXC_ERROR_CORRUPT_DATA).
 */
ZXC_EXPORT int64_t zxc_decompress(const void* src, const size_t src_size, void* dst,
                                  const size_t dst_capacity, const zxc_decompress_opts_t* opts);

/**
 * @brief Buffer size required for an in-place decompression of @p src.
 *
 * Reads @p src's header and footer only (no decoding) and returns
 * `decompressed_size` plus the one-block and wild-copy safety margin
 * @ref zxc_decompress_inplace needs. Allocate that much, place the
 * @c comp_size-byte archive flush-right in it, then decode.
 *
 * @param[in] src       Compressed archive (only header + footer are read).
 * @param[in] src_size  Archive size in bytes.
 * @return Required buffer size in bytes, or 0 if @p src is not a valid archive.
 */
ZXC_EXPORT size_t zxc_decompress_inplace_bound(const void* src, const size_t src_size);

/**
 * @brief Decompresses inside a single caller-owned buffer.
 *
 * The archive must sit **flush-right** in @p buffer (its last @p comp_size
 * bytes). Decoding runs left-to-right from @c buffer[0]; as long as
 * @p buffer_capacity is at least @ref zxc_decompress_inplace_bound, the write
 * cursor provably never overtakes the read cursor. One allocation instead of
 * two, which is what makes this worthwhile on memory-constrained targets
 * (embedded, FOTA, firmware).
 *
 * @note @p buffer is both input and output: on success its first @c N bytes
 *       hold the decompressed data (@c N = the return value).
 *
 * @param[in,out] buffer          Work buffer holding the flush-right archive.
 * @param[in]     buffer_capacity Total size of @p buffer in bytes.
 * @param[in]     comp_size       Size of the compressed archive in bytes.
 * @param[in]     opts            Decompression options, or NULL for defaults.
 * @return Decompressed size (> 0), 0 for an empty frame, or a negative
 *         @ref zxc_error_t (@ref ZXC_ERROR_DST_TOO_SMALL if the margin is missing).
 */
ZXC_EXPORT int64_t zxc_decompress_inplace(void* buffer, const size_t buffer_capacity,
                                          const size_t comp_size,
                                          const zxc_decompress_opts_t* opts);

/**
 * @brief Reads the original size from an archive footer, without decoding.
 *
 * The footer is untrusted input, so the value is checked for plausibility
 * against the archive size (each block costs at least a block header and
 * decodes to at most one block): a forged footer claiming an absurd size
 * returns 0 rather than driving an oversized allocation.
 *
 * @param[in] src       Compressed buffer.
 * @param[in] src_size  Compressed size in bytes.
 *
 * @return Original uncompressed size in bytes, or 0 if the buffer is invalid,
 *         too small, or carries an implausible footer value.
 */
ZXC_EXPORT uint64_t zxc_get_decompressed_size(const void* src, const size_t src_size);

/**
 * @brief Reads the dictionary ID from an archive header, without decoding.
 *
 * @param[in] src       Compressed buffer.
 * @param[in] src_size  Compressed size in bytes.
 * @return Dictionary ID, or 0 if no dictionary is required or the buffer is invalid.
 */
ZXC_EXPORT uint32_t zxc_get_dict_id(const void* src, size_t src_size);

/* ========================================================================= */
/*  Block-Level API (no file framing)                                        */
/* ========================================================================= */

/**
 * @defgroup block_api Block API
 * @brief Single-block compression/decompression without file framing.
 *
 * Each call handles one independent block and emits only
 * @c block_header(8B) + payload + optional @c checksum(4B): no file header,
 * EOF block, or footer. Meant for filesystem integrations that do their own
 * block indexing and compress each block independently.
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
/** @brief Opaque reusable compression context (see @ref zxc_create_cctx). */
typedef struct zxc_cctx_s zxc_cctx;
/** @brief Opaque reusable decompression context (see @ref zxc_create_dctx). */
typedef struct zxc_dctx_s zxc_dctx;

/**
 * @brief Maximum compressed size for a single block.
 *
 * Unlike zxc_compress_bound(), this excludes file header, EOF block, and
 * footer overhead. Use it to size the destination of zxc_compress_block().
 *
 * @param[in] input_size Uncompressed block size in bytes
 *                       (must be <= @ref ZXC_BLOCK_SIZE_MAX).
 * @return Upper bound on the compressed block size, or 0 if @p input_size is
 *         out of range for the Block API or would overflow.
 */
ZXC_EXPORT uint64_t zxc_compress_block_bound(size_t input_size);

/**
 * @brief Minimum @c dst_capacity zxc_decompress_block() needs for a block of
 *        @p uncompressed_size bytes.
 *
 * The decoder uses speculative (wild-copy) writes on its fast path, so it
 * needs a tail pad beyond the declared size. Passing exactly
 * @p uncompressed_size forces the slow tail path and may trip
 * @ref ZXC_ERROR_OVERFLOW on some inputs; the value returned here always
 * enables the fast path.
 *
 * @param[in] uncompressed_size Original block size in bytes
 *                              (must be <= @ref ZXC_BLOCK_SIZE_MAX).
 * @return Minimum @c dst_capacity, or 0 if @p uncompressed_size is out of
 *         range for the Block API or would overflow.
 */
ZXC_EXPORT uint64_t zxc_decompress_block_bound(const size_t uncompressed_size);

/**
 * @brief Compresses a single block without file framing.
 *
 * Output is @c block_header(8B) + payload + optional @c checksum(4B), readable
 * by zxc_decompress_block(). One format-conformant block per call: @p src_size
 * must not exceed @ref ZXC_BLOCK_SIZE_MAX (2 MiB). For larger payloads use the
 * frame API (zxc_compress) or the streaming API (zxc_cstream_*), which chunk
 * transparently.
 *
 * @param[in,out] cctx         Reusable compression context.
 * @param[in]     src          Source data.
 * @param[in]     src_size     Source size in bytes, in [1, @ref ZXC_BLOCK_SIZE_MAX].
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst (see zxc_compress_block_bound()).
 * @param[in]     opts         Compression options, or NULL for defaults. Only
 *                             @c level, @c block_size and @c checksum_enabled
 *                             are used.
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Compressed block size (> 0), or a negative @ref zxc_error_t.
 *         @ref ZXC_ERROR_BAD_BLOCK_SIZE if @p src_size exceeds
 *         @ref ZXC_BLOCK_SIZE_MAX; @ref ZXC_ERROR_BAD_LEVEL on a static
 *         context for a level raise its workspace cannot accommodate (levels
 *         above @ref ZXC_LEVEL_ULTRA are otherwise silently clamped).
 */
ZXC_EXPORT int64_t zxc_compress_block(zxc_cctx* cctx, const void* src, size_t src_size, void* dst,
                                      size_t dst_capacity, const zxc_compress_opts_t* opts);

/**
 * @brief Decompresses a single block produced by zxc_compress_block().
 *
 * One format-conformant block per call: @p dst_capacity must not exceed
 * @ref ZXC_BLOCK_SIZE_MAX + @ref ZXC_DECOMPRESS_TAIL_PAD (what
 * zxc_decompress_block_bound() returns for the maximum block size). For
 * payloads from the frame or streaming APIs, use zxc_decompress instead.
 *
 * @param[in,out] dctx         Reusable decompression context.
 * @param[in]     src          Compressed block.
 * @param[in]     src_size     Compressed size in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst: at least the original
 *                             uncompressed size, at most
 *                             @ref ZXC_BLOCK_SIZE_MAX +
 *                             @ref ZXC_DECOMPRESS_TAIL_PAD.
 * @param[in]     opts         Decompression options, or NULL for defaults.
 *                             Only @c checksum_enabled is used.
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Decompressed size (> 0), or a negative @ref zxc_error_t;
 *         @ref ZXC_ERROR_BAD_BLOCK_SIZE if @p dst_capacity exceeds the
 *         per-block limit.
 */
ZXC_EXPORT int64_t zxc_decompress_block(zxc_dctx* dctx, const void* src, size_t src_size, void* dst,
                                        size_t dst_capacity, const zxc_decompress_opts_t* opts);

/**
 * @brief Decompresses a single block into an exactly-sized destination.
 *
 * Same semantics as zxc_decompress_block(), except @p dst_capacity may equal
 * the uncompressed size, with no @c ZXC_DECOMPRESS_TAIL_PAD margin. For
 * integrations whose destination cannot be oversized (say, an exactly-sized
 * page-aligned region). "In place" here means a tightly-sized destination, not
 * an overlapping @p src / @p dst (see the note below).
 *
 * Slightly slower than zxc_decompress_block() since it gives up the wild-copy
 * overshoot the fast decoder relies on; output is bit-identical. RAW blocks
 * forward straight to zxc_decompress_block(), only GLO/GHI take the
 * strict-tail path.
 *
 * @param[in,out] dctx         Reusable decompression context.
 * @param[in]     src          Compressed block.
 * @param[in]     src_size     Compressed size in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst: at least the original
 *                             uncompressed size, at most
 *                             @ref ZXC_BLOCK_SIZE_MAX (no tail-pad margin
 *                             needed, unlike zxc_decompress_block).
 * @param[in]     opts         Decompression options, or NULL for defaults.
 *                             Only @c checksum_enabled is used.
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Decompressed size (> 0), or a negative @ref zxc_error_t;
 *         @ref ZXC_ERROR_BAD_BLOCK_SIZE if @p dst_capacity >
 *         @ref ZXC_BLOCK_SIZE_MAX.
 */
ZXC_EXPORT int64_t zxc_decompress_block_safe(zxc_dctx* dctx, const void* src, const size_t src_size,
                                             void* dst, const size_t dst_capacity,
                                             const zxc_decompress_opts_t* opts);

/**
 * @brief Estimates peak compression memory for a given block size and level.
 *
 * Totals everything @ref zxc_compress_block reserves for a @p src_size block:
 * per-chunk working buffers (chain table, literals, sequence/token/offset/extras),
 * the fixed hash tables, and cache-line padding. At @p level >= 6 it also counts
 * the `opt_scratch` region (~8.125 x @p src_size) used by the price-based optimal
 * parser, which is lazy-allocated on the first level-6 call and then reused for
 * the lifetime of the cctx. Scales roughly linearly with @p src_size.
 *
 * @param[in] src_size Uncompressed block size in bytes.
 * @param[in] level    Compression level (1..7). Levels <= 5 share the same
 *                     persistent footprint; levels >= 6 add the optimal-parser
 *                     scratch.
 * @return Estimated peak cctx memory in bytes, or 0 if @p src_size is 0.
 */
ZXC_EXPORT uint64_t zxc_estimate_cctx_size(size_t src_size, int level);

/** @} */ /* end of block_api */

/* ========================================================================= */
/*  Reusable Context API (opaque, heap-allocated)                            */
/* ========================================================================= */

/**
 * @defgroup context_api Reusable Context API
 * @brief Opaque, reusable compression / decompression contexts.
 *
 * Keeping a context across calls removes the per-call allocation overhead.
 * The internal layout stays hidden behind an opaque pointer.
 *
 * @{
 */

/* --- Compression context ------------------------------------------------- */

/**
 * @brief Creates a reusable compression context.
 *
 * With a non-NULL @p opts the context pre-allocates its buffers from the given
 * level, block_size and checksum_enabled; with NULL, allocation is deferred to
 * the first zxc_compress_cctx() call. Levels above @ref ZXC_LEVEL_ULTRA are
 * silently clamped. Free with zxc_free_cctx().
 *
 * @param[in] opts  Options for eager init, or NULL for lazy init.
 * @return New context, or @c NULL on allocation failure or an invalid block
 *         size (not a power of two in range).
 */
ZXC_EXPORT zxc_cctx* zxc_create_cctx(const zxc_compress_opts_t* opts);

/**
 * @brief Frees a compression context and its resources. @c NULL is a no-op.
 *
 * @param[in] cctx Context to free.
 */
ZXC_EXPORT void zxc_free_cctx(zxc_cctx* cctx);

/**
 * @brief Compresses data using a reusable context.
 *
 * Like zxc_compress(), but reuses @p cctx's buffers instead of allocating per
 * call. The context re-initialises itself when block_size changes, or when a
 * level raise into @ref ZXC_LEVEL_DENSITY needs the optimal-parser scratch
 * that lower-level inits skip. On a static (caller-workspace) context such a
 * raise returns @ref ZXC_ERROR_BAD_LEVEL instead, since the workspace cannot
 * grow.
 *
 * Options are **sticky**: values passed in @p opts are remembered and reused
 * on later calls that pass NULL, starting from those given to
 * zxc_create_cctx(). Levels above @ref ZXC_LEVEL_ULTRA are silently clamped.
 *
 * @param[in,out] cctx         Reusable compression context.
 * @param[in]     src          Source data.
 * @param[in]     src_size     Source size in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst.
 * @param[in]     opts         Options, or NULL to reuse the sticky settings.
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Compressed size (> 0), or a negative @ref zxc_error_t.
 */
ZXC_EXPORT int64_t zxc_compress_cctx(zxc_cctx* cctx, const void* src, size_t src_size, void* dst,
                                     size_t dst_capacity, const zxc_compress_opts_t* opts);

/* --- Decompression context ----------------------------------------------- */

/**
 * @brief Creates a reusable decompression context.
 *
 * @return New context, or @c NULL on allocation failure.
 */
ZXC_EXPORT zxc_dctx* zxc_create_dctx(void);

/**
 * @brief Frees a decompression context and its resources. @c NULL is a no-op.
 *
 * @param[in] dctx Context to free.
 */
ZXC_EXPORT void zxc_free_dctx(zxc_dctx* dctx);

/**
 * @brief Decompresses data using a reusable context.
 *
 * Like zxc_decompress(), but reuses @p dctx's buffers.
 *
 * @param[in,out] dctx         Reusable decompression context.
 * @param[in]     src          Compressed data.
 * @param[in]     src_size     Compressed size in bytes.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst.
 * @param[in]     opts         Decompression options, or NULL for defaults.
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Decompressed size (> 0), or a negative @ref zxc_error_t.
 */
ZXC_EXPORT int64_t zxc_decompress_dctx(zxc_dctx* dctx, const void* src, size_t src_size, void* dst,
                                       size_t dst_capacity, const zxc_decompress_opts_t* opts);

/* ========================================================================= */
/*  Static Context API (caller-allocated workspace)                          */
/* ========================================================================= */

/**
 * @defgroup static_context_api Static Context API
 * @brief Caller-allocated, fixed-footprint compression / decompression
 *        contexts.
 *
 * Mirrors the Reusable Context API, but the whole context (handle + persistent
 * buffers) lives in one buffer the caller allocates and owns. Required wherever
 * the library must not touch the host allocator on the hot path: Linux kernel
 * filesystems (one workspace per mount, from @c vmalloc / @c kmalloc up front),
 * heapless embedded targets (`.bss` or stack workspace), sandboxed runtimes on
 * a fixed memory budget.
 *
 * The trade-off: the workspace is pinned to one @c block_size and @c level at
 * init time and cannot grow afterwards, so a workload mixing block sizes must
 * size for the largest one up front.
 *
 * @par Typical usage
 * @code
 * size_t ws_sz = zxc_static_cctx_workspace_size(64 * 1024, ZXC_LEVEL_DEFAULT);
 * void *ws = aligned_alloc(64, ws_sz);                   // or kmalloc, vmalloc, .bss
 * zxc_compress_opts_t opts = { .level = ZXC_LEVEL_DEFAULT, .block_size = 64 * 1024 };
 * zxc_cctx *cctx = zxc_init_static_cctx(ws, ws_sz, &opts);
 *
 * for (each block) zxc_compress_cctx(cctx, src, n, dst, cap, NULL);
 *
 * // zxc_free_cctx is a no-op on a static cctx; the caller owns @c ws.
 * free(ws);
 * @endcode
 * @{
 */

/**
 * @brief Exact size of a static compression workspace.
 *
 * Sums the opaque @ref zxc_cctx wrapper and every persistent sub-buffer the
 * library partitions out of it (hash tables, chain table, sequence buffers,
 * literal scratch, plus the optimal-parser scratch at
 * @ref ZXC_LEVEL_DENSITY). The workspace must be at least cache-line aligned,
 * so round up for @c posix_memalign / @c aligned_alloc.
 *
 * @param[in] block_size  Block size in bytes (power of two in
 *                        [@ref ZXC_BLOCK_SIZE_MIN, @ref ZXC_BLOCK_SIZE_MAX]).
 * @param[in] level       Compression level (1..7); levels at or above
 *                        @ref ZXC_LEVEL_DENSITY add the optimal-parser
 *                        scratch (~8.125 x block_size).
 * @return Workspace size in bytes, or 0 if either argument is invalid.
 */
ZXC_EXPORT size_t zxc_static_cctx_workspace_size(const size_t block_size, const int level);

/**
 * @brief Initialises a compression context inside a caller-supplied workspace.
 *
 * @p workspace_size must be at least @ref zxc_static_cctx_workspace_size for
 * the same @c block_size and @c level. The workspace must be cache-line
 * (64-byte) aligned and must outlive the returned handle. The caller owns it;
 * @ref zxc_free_cctx is a no-op on this handle.
 *
 * @par Locked parameters
 * @c block_size, @c level and @c checksum_enabled are pinned at init time. A
 * later @ref zxc_compress_cctx call asking for a different @c block_size
 * returns @ref ZXC_ERROR_BAD_BLOCK_SIZE without re-initialising. A different
 * @c level / @c checksum_enabled is honoured per call without re-partitioning,
 * except a raise into @ref ZXC_LEVEL_DENSITY on a workspace carved below it:
 * the optimal-parser scratch is absent, so the call returns
 * @ref ZXC_ERROR_BAD_LEVEL.
 *
 * @param[in,out] workspace       Caller-allocated buffer, cache-line aligned.
 * @param[in]     workspace_size  Capacity of @p workspace in bytes.
 * @param[in]     opts            Must be non-NULL, with @c block_size and
 *                                @c level set explicitly to size the workspace
 *                                correctly.
 * @return Handle pointing inside @p workspace, or @c NULL if the workspace is
 *         too small or the options are invalid.
 */
ZXC_EXPORT zxc_cctx* zxc_init_static_cctx(void* workspace, const size_t workspace_size,
                                          const zxc_compress_opts_t* opts);

/**
 * @brief Exact size of a static decompression workspace.
 *
 * Unlike the compression variant this is independent of the archive's level:
 * @c lit_buffer is always provisioned worst-case, because the decoder cannot
 * know a block's literal encoding until it reads that block's header.
 *
 * @param[in] block_size  Largest block size the decoder will meet (same
 *                        constraints as everywhere else).
 * @return Workspace size in bytes, or 0 if @p block_size is invalid.
 */
ZXC_EXPORT size_t zxc_static_dctx_workspace_size(const size_t block_size);

/**
 * @brief Initialises a decompression context inside a caller-supplied workspace.
 *
 * @p workspace_size must be at least @ref zxc_static_dctx_workspace_size for
 * the same @p block_size. The workspace must be cache-line aligned and must
 * outlive the returned handle. The caller owns it; @ref zxc_free_dctx is a
 * no-op on this handle.
 *
 * @par Locked block size
 * @p block_size is pinned at init time: an archive whose header declares a
 * different @c block_size is rejected with @ref ZXC_ERROR_BAD_BLOCK_SIZE.
 *
 * @param[in,out] workspace       Caller-allocated buffer, cache-line aligned.
 * @param[in]     workspace_size  Capacity of @p workspace in bytes.
 * @param[in]     block_size      Block size the decoder will accept.
 * @return Handle pointing inside @p workspace, or @c NULL if the workspace is
 *         too small or @p block_size is invalid.
 */
ZXC_EXPORT zxc_dctx* zxc_init_static_dctx(void* workspace, const size_t workspace_size,
                                          const size_t block_size);

/** @} */ /* end of static_context_api */
/** @} */ /* end of context_api */
/** @} */ /* end of buffer_api */

#ifdef __cplusplus
}
#endif

#endif  // ZXC_BUFFER_H

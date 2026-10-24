/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) Bertrand Lebonnois and contributors.
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
 * // Compress, the simple form: one level, every other parameter at its default
 * size_t bound = zxc_compress_bound(src_size);
 * void *dst    = malloc(bound);
 * int64_t csize = zxc_compress(src, src_size, dst, bound, ZXC_LEVEL_DEFAULT);
 *
 * // Decompress
 * uint64_t orig = zxc_get_decompressed_size(dst, csize);
 * void *out     = malloc(orig);
 * int64_t dsize = zxc_decompress(dst, csize, out, orig);
 *
 * // Anything else goes through a configured context (Reusable Context API)
 * zxc_cctx* cctx = zxc_create_cctx();
 * zxc_cctx_set_param(cctx, ZXC_C_CHECKSUM, 1);
 * csize = zxc_compress_cctx(cctx, src, src_size, dst, bound);
 * zxc_free_cctx(cctx);
 * @endcode
 *
 * @see zxc_stream.h  multi-threaded @c FILE* streaming.
 * @see zxc_pstream.h single-threaded push streaming.
 */

#ifndef ZXC_BUFFER_H
#define ZXC_BUFFER_H

#include <stddef.h>
#include <stdint.h>

#include "zxc_constants.h"
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
 * @brief Library version string, "MAJOR.MINOR.PATCH".
 * @return Null-terminated compile-time constant; do not free.
 */
ZXC_EXPORT const char* zxc_version_string(void);

/**
 * @brief Library version as one integer, @ref ZXC_VERSION_NUMBER of the loaded library.
 *
 * Compare with the header's @ref ZXC_VERSION_NUMBER to detect a runtime
 * library older than the one compiled against.
 * @return MAJOR * 10000 + MINOR * 100 + PATCH.
 */
ZXC_EXPORT unsigned zxc_version_number(void);

/**
 * @brief Release code name of the loaded library, @ref ZXC_VERSION_CODENAME.
 * @return Null-terminated compile-time constant; do not free.
 */
ZXC_EXPORT const char* zxc_version_codename(void);

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
 * @return Required destination capacity in bytes, or 0 when that capacity
 *         would not fit a @c size_t on this platform (no buffer could hold it).
 */
ZXC_EXPORT size_t zxc_compress_bound(const size_t input_size);

/**
 * @brief Compresses a buffer into a complete ZXC archive, at @p level.
 *
 * The simple form: one level, every other parameter at its default (512 KB
 * blocks, no checksum, no seek table, no dictionary). For anything else,
 * configure a context and call zxc_compress_cctx(). Writes the file header
 * followed by compressed blocks; single-threaded and blocking.
 *
 * @param[in]  src          Source buffer; may be NULL when @p src_size is 0.
 * @param[in]  src_size     Source size in bytes; 0 writes the empty archive
 *                          (file header + EOF block + footer).
 * @param[out] dst          Destination buffer.
 * @param[in]  dst_capacity Capacity of @p dst.
 * @param[in]  level        Compression level, 1 to 7 (@ref zxc_compression_level_t);
 *                          0 selects @ref ZXC_LEVEL_DEFAULT.
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Bytes written to @p dst (> 0), or a negative @ref zxc_error_t code:
 *         @ref ZXC_ERROR_BAD_LEVEL for any other @p level,
 *         @ref ZXC_ERROR_DST_TOO_SMALL, ...
 */
ZXC_EXPORT int64_t zxc_compress(const void* src, const size_t src_size, void* dst,
                                const size_t dst_capacity, int level);

/**
 * @brief Decompresses a complete ZXC archive.
 *
 * The simple form: no checksum verification and no dictionary. For either,
 * configure a context and call zxc_decompress_dctx(). Expects a valid file
 * header followed by compressed blocks; single-threaded and blocking.
 *
 * @par Concatenated frames
 * @p src may hold several frames back to back; they decode in order into
 * @p dst. Anything else after a frame is @ref ZXC_ERROR_CORRUPT_DATA.
 *
 * @par Asking without a destination
 * A NULL @p dst, or a @p dst_capacity of 0, decodes nothing and reports what
 * the archive holds: 0 for a well-formed empty one, @ref ZXC_ERROR_DST_TOO_SMALL
 * when it stores a payload, its own error otherwise. Success is never handed
 * out early: a 0 here means a call with a destination would have returned 0
 * too, checksum and dictionary binding included. Not the reverse, since nothing
 * is decoded: an archive that stores a payload reports
 * @ref ZXC_ERROR_DST_TOO_SMALL even where a decode would name the actual fault.
 * Caller errors outrank all of it, before anything is read from @p src, a
 * source too short to hold a frame included: a NULL @p dst with a non-zero
 * @p dst_capacity is @ref ZXC_ERROR_NULL_INPUT.
 *
 * @param[in]  src          Compressed buffer.
 * @param[in]  src_size     Compressed size in bytes.
 * @param[out] dst          Destination buffer, or NULL to probe (see above).
 * @param[in]  dst_capacity Capacity of @p dst.
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Bytes written to @p dst, 0 for an empty archive, or a negative
 *         @ref zxc_error_t (e.g. @ref ZXC_ERROR_CORRUPT_DATA).
 */
ZXC_EXPORT int64_t zxc_decompress(const void* src, const size_t src_size, void* dst,
                                  const size_t dst_capacity);

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
 * cursor provably never overtakes the read cursor; an archive whose output would
 * reach unread input (padding, forged block sizes) is refused with
 * @ref ZXC_ERROR_CORRUPT_DATA first. One allocation instead of
 * two, which is what makes this worthwhile on memory-constrained targets
 * (embedded, FOTA, firmware).
 *
 * @note The decode context is heap-allocated for the call, with no checksum
 *       verification and no dictionary; for either, or to supply your own
 *       context, see zxc_decompress_inplace_dctx().
 *
 * @note @p buffer is both input and output: on success its first @c N bytes
 *       hold the decompressed data (@c N = the return value).
 *
 * @param[in,out] buffer          Work buffer holding the flush-right archive.
 * @param[in]     buffer_capacity Total size of @p buffer in bytes.
 * @param[in]     comp_size       Size of the compressed archive in bytes.
 * @return Decompressed size (> 0), 0 for an empty frame, or a negative
 *         @ref zxc_error_t (@ref ZXC_ERROR_DST_TOO_SMALL if the margin is missing).
 */
ZXC_EXPORT int64_t zxc_decompress_inplace(void* buffer, const size_t buffer_capacity,
                                          const size_t comp_size);

/**
 * @brief Reads the original size from the archive footers, without decoding.
 *
 * Sums the sizes the footers store, walking the frames back from the end. Each
 * frame is checked like zxc_get_frame_info(), its size against what its bytes
 * can hold: a forged footer returns 0 rather than drive an oversized allocation.
 *
 * @param[in] src       Compressed buffer.
 * @param[in] src_size  Compressed size in bytes.
 *
 * @return Original uncompressed size in bytes, or 0 if the buffer is invalid,
 *         too small, or carries an implausible footer value.
 */
ZXC_EXPORT uint64_t zxc_get_decompressed_size(const void* src, const size_t src_size);

/**
 * @brief Reads the dictionary ID an archive requires, from its header.
 *
 * Also reported as zxc_frame_info_t::dict_id. For the ID of a `.zxd` file see
 * zxc_dict_read_id(); to compute one from content, zxc_dict_compute_id().
 *
 * @param[in] src       Compressed buffer.
 * @param[in] src_size  Compressed size in bytes.
 * @return Dictionary ID, or 0 if no dictionary is required or the buffer is invalid.
 */
ZXC_EXPORT uint32_t zxc_get_dict_id(const void* src, size_t src_size);

/**
 * @brief What a frame's header and footer declare, read without decoding.
 *
 * Filled by zxc_get_frame_info(), zxc_get_last_frame_info() and their FILE*
 * twins; pass them `sizeof(zxc_frame_info_t)`. Fields are only ever added at
 * the end: zero the struct first to read the ones this library does not know as 0.
 */
typedef struct {
    uint64_t decompressed_size; /**< Source bytes the frame decodes to. */
    uint64_t compressed_size;   /**< Compressed bytes of the frame, footer included. */
    uint64_t digest;            /**< Archive digest; 0 when @c has_checksum is 0. */
    size_t block_size;          /**< Block size, 4 KB to 2 MB. */
    uint32_t dict_id;           /**< Dictionary the frame needs; 0 for none. */
    uint8_t format_version;     /**< Format version of the frame. */
    uint8_t has_checksum;       /**< 1 when blocks carry checksums and the footer a digest. */
    uint8_t has_seek_table;     /**< 1 when a seek table precedes the footer. */
} zxc_frame_info_t;

/**
 * @brief Reads a frame's header and footer, without decoding.
 *
 * Validates the header as a decoder would, parses the footer back from the end
 * of @p src, and checks that the frame spans all of @p src and that its stored
 * size is reachable. The blocks are not read, so a frame that passes may still
 * fail to decode.
 *
 * @param[in]  src        Compressed buffer.
 * @param[in]  src_size   Compressed size in bytes.
 * @param[out] info       Filled on success, untouched otherwise.
 * @param[in]  info_size  `sizeof(*info)` as the caller compiled it.
 * @return @ref ZXC_OK, or a negative @ref zxc_error_t (e.g.
 *         @ref ZXC_ERROR_BAD_MAGIC, @ref ZXC_ERROR_CORRUPT_DATA).
 */
ZXC_EXPORT int zxc_get_frame_info(const void* src, size_t src_size, zxc_frame_info_t* info,
                                  size_t info_size);

/**
 * @brief Reads the last frame of @p src, without decoding.
 *
 * Same checks as zxc_get_frame_info(), but the frame starts at
 * `src_size - info->compressed_size`, so concatenated frames walk back:
 *
 * @code
 * size_t n = src_size;
 * while (n > 0) {
 *     if (zxc_get_last_frame_info(src, n, &info, sizeof(info)) != ZXC_OK) break;
 *     n -= info.compressed_size;  // the frame before ends here
 * }
 * @endcode
 *
 * @param[in]  src        Compressed buffer.
 * @param[in]  src_size   Bytes of @p src the frame ends.
 * @param[out] info       Filled on success, untouched otherwise.
 * @param[in]  info_size  `sizeof(*info)` as the caller compiled it.
 * @return @ref ZXC_OK, or a negative @ref zxc_error_t.
 */
ZXC_EXPORT int zxc_get_last_frame_info(const void* src, size_t src_size, zxc_frame_info_t* info,
                                       size_t info_size);

/**
 * @brief Returns `sizeof(zxc_frame_info_t)` as compiled into the library.
 *
 * Layout guard for bindings that mirror the struct by hand instead of
 * compiling against this header: comparing the mirrored size against this at
 * load time turns a silent layout drift into an explicit error.
 */
ZXC_EXPORT size_t zxc_frame_info_size(void);

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
 * zxc_cctx* cctx = zxc_create_cctx();
 * zxc_cctx_set_param(cctx, ZXC_C_LEVEL, 3);
 * size_t bound = zxc_compress_block_bound(block_size);
 * void *dst = malloc(bound);
 * int64_t csize = zxc_compress_block(cctx, block, block_size, dst, bound);
 *
 * // Decompress
 * zxc_dctx* dctx = zxc_create_dctx();
 * int64_t dsize = zxc_decompress_block(dctx, dst, csize, out, block_size);
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
ZXC_EXPORT size_t zxc_compress_block_bound(size_t input_size);

/**
 * @brief Minimum @c dst_capacity zxc_decompress_block() needs for a block of
 *        @p uncompressed_size bytes.
 *
 * The decoder uses speculative (wild-copy) writes on its fast path, so it
 * needs a tail pad beyond the declared size: with this value a dictionary-free
 * block decodes straight into @c dst on a heap context; without it, through a
 * bounce and a copy. A dictionary block always bounces, needing the dictionary
 * and the payload contiguous, and so does an over-sized static context.
 *
 * @param[in] uncompressed_size Original block size in bytes
 *                              (must be <= @ref ZXC_BLOCK_SIZE_MAX).
 * @return Minimum @c dst_capacity, or 0 if @p uncompressed_size is out of
 *         range for the Block API or would overflow.
 */
ZXC_EXPORT size_t zxc_decompress_block_bound(const size_t uncompressed_size);

/**
 * @brief Compresses a single block without file framing.
 *
 * Output is @c block_header(8B) + payload + optional @c checksum(4B), readable
 * by zxc_decompress_block(). The checksum covers the uncompressed bytes, seeded
 * with position 0. One format-conformant block per call: @p src_size
 * must not exceed @ref ZXC_BLOCK_SIZE_MAX (2 MiB). For larger payloads use the
 * frame API (zxc_compress) or the streaming API (zxc_cstream_*), which chunk
 * transparently.
 *
 * @param[in,out] cctx         Reusable compression context.
 * @param[in]     src          Source data.
 * @param[in]     src_size     Source size in bytes, in [1, @ref ZXC_BLOCK_SIZE_MAX].
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst (see zxc_compress_block_bound()).
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Compressed block size (> 0), or a negative @ref zxc_error_t;
 *         @ref ZXC_ERROR_BAD_BLOCK_SIZE if @p src_size exceeds
 *         @ref ZXC_BLOCK_SIZE_MAX, or on a static context its pinned block size.
 */
ZXC_EXPORT int64_t zxc_compress_block(zxc_cctx* cctx, const void* src, size_t src_size, void* dst,
                                      size_t dst_capacity);

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
 * @param[in]     dst_capacity At least the uncompressed size, at most
 *                             @ref ZXC_BLOCK_SIZE_MAX +
 *                             @ref ZXC_DECOMPRESS_TAIL_PAD. At
 *                             zxc_decompress_block_bound() a dictionary-free
 *                             block on a heap context decodes straight into
 *                             @p dst, else possibly through a bounce.
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Decompressed size (> 0), or a negative @ref zxc_error_t;
 *         @ref ZXC_ERROR_BAD_BLOCK_SIZE if @p dst_capacity exceeds the
 *         per-block limit. Static context: the carved block is the effective
 *         capacity; a larger block is @ref ZXC_ERROR_BAD_BLOCK_SIZE while it
 *         fits the workspace margin and fails like a too-small destination
 *         beyond it. On error @p dst holds whatever the decode wrote, all of it
 *         on @ref ZXC_ERROR_BAD_CHECKSUM; its previous contents do not survive.
 */
ZXC_EXPORT int64_t zxc_decompress_block(zxc_dctx* dctx, const void* src, size_t src_size, void* dst,
                                        size_t dst_capacity);

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
 * forward straight to zxc_decompress_block(), only SPL/PAK take the
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
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Decompressed size (> 0), or a negative @ref zxc_error_t;
 *         @ref ZXC_ERROR_BAD_BLOCK_SIZE if @p dst_capacity >
 *         @ref ZXC_BLOCK_SIZE_MAX. Static context: same bound and codes as
 *         zxc_decompress_block(), a larger @p dst_capacity accepted alike. On
 *         error @p dst holds whatever the decode wrote, all of it on
 *         @ref ZXC_ERROR_BAD_CHECKSUM.
 */
ZXC_EXPORT int64_t zxc_decompress_block_safe(zxc_dctx* dctx, const void* src, const size_t src_size,
                                             void* dst, const size_t dst_capacity);

/**
 * @brief Estimates peak compression memory for a given block size and level.
 *
 * Totals everything @ref zxc_compress_block reserves for a @p src_size block:
 * per-chunk working buffers (chain table, literals, sequence/token/offset/extras),
 * the fixed hash tables and match-split histograms, and cache-line padding. At
 * @p level >= 6 it also counts the `opt_scratch` region: ~8.125 x @p src_size for
 * the optimal parser. Lazy-allocated on the first level-6 call, then kept for the cctx's lifetime.
 *
 * @param[in] src_size Uncompressed block size in bytes.
 * @param[in] level    Compression level (1..7). Levels <= 5 share the same
 *                     persistent footprint; levels >= 6 add the optimal-parser
 *                     scratch.
 * @return Estimated peak cctx memory in bytes, or 0 if @p src_size is 0.
 */
ZXC_EXPORT size_t zxc_estimate_cctx_size(size_t src_size, int level);

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
 * @brief Creates a reusable compression context, every parameter at its default.
 *
 * Only the handle is allocated here; the working buffers come on the first
 * call, sized from the parameters then in force. Configure it with
 * zxc_cctx_set_param(), zxc_cctx_set_dict() and zxc_cctx_set_progress(); the
 * settings stay until changed or zxc_cctx_reset(). Free with zxc_free_cctx().
 *
 * @return New context, or @c NULL on allocation failure.
 */
ZXC_EXPORT zxc_cctx* zxc_create_cctx(void);

/**
 * @brief Frees a compression context and its resources. @c NULL is a no-op.
 *
 * @param[in] cctx Context to free.
 */
ZXC_EXPORT void zxc_free_cctx(zxc_cctx* cctx);

/**
 * @brief Sets one parameter on @p cctx; 0 restores its default.
 *
 * Takes effect on the next call. A value outside zxc_cparam_bounds(), or an
 * unknown @p param, is refused and the parameter keeps its value
 * (@ref ZXC_ERROR_BAD_LEVEL, @ref ZXC_ERROR_BAD_BLOCK_SIZE or
 * @ref ZXC_ERROR_BAD_PARAM). On a static context
 * (zxc_init_static_cctx()) the block size is pinned, a level raise into
 * @ref ZXC_LEVEL_DENSITY is refused when the workspace was carved below it,
 * and 0 restores the pinned block size and level.
 *
 * @param[in,out] cctx  Context to configure.
 * @param[in]     param Parameter identifier.
 * @param[in]     value New value, or 0 for the default.
 * @return @ref ZXC_OK, or a negative @ref zxc_error_t.
 */
ZXC_EXPORT int zxc_cctx_set_param(zxc_cctx* cctx, zxc_cparam_t param, int64_t value);

/**
 * @brief Reads one parameter of @p cctx.
 *
 * @param[in]  cctx  Context.
 * @param[in]  param Parameter identifier.
 * @param[out] value Receives the value in force.
 * @return @ref ZXC_OK, @ref ZXC_ERROR_BAD_PARAM for an unknown @p param, or
 *         @ref ZXC_ERROR_NULL_INPUT.
 */
ZXC_EXPORT int zxc_cctx_get_param(const zxc_cctx* cctx, zxc_cparam_t param, int64_t* value);

/**
 * @brief Attaches a pre-trained dictionary to @p cctx; a NULL @p dict detaches.
 *
 * The context keeps its own copy of @p dict and @p dict_huf: the caller's
 * buffers are free once this returns. @p dict_huf, when non-NULL, is the
 * shared literal table of @ref ZXC_HUF_TABLE_SIZE bytes (zxc_dict_huf()), used
 * at levels 6-7 and part of the archive's dict_id binding.
 *
 * @param[in,out] cctx      Context to configure.
 * @param[in]     dict      Dictionary content, or NULL for none.
 * @param[in]     dict_size Size of @p dict in bytes, at most @ref ZXC_DICT_SIZE_MAX.
 * @param[in]     dict_huf  Shared literal table, or NULL.
 * @return @ref ZXC_OK; @ref ZXC_ERROR_DICT_TOO_LARGE above the maximum;
 *         @ref ZXC_ERROR_DICT_UNSUPPORTED on a static context, which has no
 *         room for the dictionary prefix; @ref ZXC_ERROR_MEMORY when the copy
 *         cannot be allocated. A refused call leaves the attached dictionary.
 */
ZXC_EXPORT int zxc_cctx_set_dict(zxc_cctx* cctx, const void* dict, size_t dict_size,
                                 const void* dict_huf);

/**
 * @brief Sets the progress callback; NULL @p cb disables it.
 *
 * Honoured by zxc_stream_compress() only: the buffer, block and push APIs
 * are synchronous and never call it.
 *
 * @param[in,out] cctx      Context to configure.
 * @param[in]     cb        Callback, or NULL.
 * @param[in]     user_data Pointer handed back to @p cb.
 * @return @ref ZXC_OK, or @ref ZXC_ERROR_NULL_INPUT.
 */
ZXC_EXPORT int zxc_cctx_set_progress(zxc_cctx* cctx, zxc_progress_callback_t cb, void* user_data);

/**
 * @brief Restores every parameter to its default, detaches the dictionary
 *        and the callback. The working buffers are kept.
 *
 * On a static context the pinned block size and level are the defaults.
 *
 * @param[in,out] cctx Context to reset.
 * @return @ref ZXC_OK, or @ref ZXC_ERROR_NULL_INPUT.
 */
ZXC_EXPORT int zxc_cctx_reset(zxc_cctx* cctx);

/**
 * @brief Compresses data using a reusable context.
 *
 * Like zxc_compress(), but with the context's parameters (level, block size,
 * checksum, seek table, dictionary) and its buffers, which are re-carved only
 * when the block size or the dictionary changes, or when the level rises into
 * @ref ZXC_LEVEL_DENSITY for the first time. A static (caller-workspace)
 * context cannot grow: such a raise returns @ref ZXC_ERROR_BAD_LEVEL there.
 * @ref ZXC_C_SEEKABLE appends the seek table without allocating, static
 * contexts included.
 *
 * @param[in,out] cctx         Configured compression context.
 * @param[in]     src          Source data; may be NULL when @p src_size is 0.
 * @param[in]     src_size     Source size in bytes; 0 writes the empty archive,
 *                             as zxc_compress() does, without carving the
 *                             encoder workspace.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst.
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Compressed size (> 0), or a negative @ref zxc_error_t.
 */
ZXC_EXPORT int64_t zxc_compress_cctx(zxc_cctx* cctx, const void* src, size_t src_size, void* dst,
                                     size_t dst_capacity);

/* --- Decompression context ----------------------------------------------- */

/**
 * @brief Creates a reusable decompression context.
 *
 * Only the handle is allocated here: the working buffers are sized from the
 * archive header, so they come on the first decode. Configure it with
 * zxc_dctx_set_param(), zxc_dctx_set_dict() and zxc_dctx_set_progress(). Free
 * with zxc_free_dctx().
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
 * @brief Sets one parameter on @p dctx; 0 restores its default.
 *
 * Takes effect on the next call. A value outside zxc_dparam_bounds() is
 * @ref ZXC_ERROR_BAD_PARAM, as is an unknown @p param; the parameter keeps
 * its value.
 *
 * @param[in,out] dctx  Context to configure.
 * @param[in]     param Parameter identifier.
 * @param[in]     value New value, or 0 for the default.
 * @return @ref ZXC_OK, or a negative @ref zxc_error_t.
 */
ZXC_EXPORT int zxc_dctx_set_param(zxc_dctx* dctx, zxc_dparam_t param, int64_t value);

/**
 * @brief Reads one parameter of @p dctx.
 * @see zxc_cctx_get_param
 */
ZXC_EXPORT int zxc_dctx_get_param(const zxc_dctx* dctx, zxc_dparam_t param, int64_t* value);

/**
 * @brief Attaches the dictionary the archives were compressed with; NULL detaches.
 *
 * Copies the bytes, as zxc_cctx_set_dict() does. @p dict_huf must be the table
 * used at compression time when the archive carries one; the header's dict_id
 * is checked against the pair, @ref ZXC_ERROR_DICT_MISMATCH otherwise.
 *
 * @param[in,out] dctx      Context to configure.
 * @param[in]     dict      Dictionary content, or NULL for none.
 * @param[in]     dict_size Size of @p dict in bytes, at most @ref ZXC_DICT_SIZE_MAX.
 * @param[in]     dict_huf  Shared literal table, or NULL.
 * @return @ref ZXC_OK; @ref ZXC_ERROR_DICT_TOO_LARGE above the maximum;
 *         @ref ZXC_ERROR_DICT_UNSUPPORTED on a static context;
 *         @ref ZXC_ERROR_MEMORY when the copy cannot be allocated.
 */
ZXC_EXPORT int zxc_dctx_set_dict(zxc_dctx* dctx, const void* dict, size_t dict_size,
                                 const void* dict_huf);

/**
 * @brief Sets the progress callback; NULL @p cb disables it.
 *
 * Honoured by zxc_stream_decompress() only.
 *
 * @param[in,out] dctx      Context to configure.
 * @param[in]     cb        Callback, or NULL.
 * @param[in]     user_data Pointer handed back to @p cb.
 * @return @ref ZXC_OK, or @ref ZXC_ERROR_NULL_INPUT.
 */
ZXC_EXPORT int zxc_dctx_set_progress(zxc_dctx* dctx, zxc_progress_callback_t cb, void* user_data);

/**
 * @brief Restores every parameter to its default, detaches the dictionary
 *        and the callback. The working buffers are kept.
 *
 * @param[in,out] dctx Context to reset.
 * @return @ref ZXC_OK, or @ref ZXC_ERROR_NULL_INPUT.
 */
ZXC_EXPORT int zxc_dctx_reset(zxc_dctx* dctx);

/**
 * @brief Decompresses data using a reusable context.
 *
 * Like zxc_decompress(), dictionary options included, but reuses @p dctx's
 * buffers; the shared literal table is rebuilt only when it changes between
 * calls. A static context returns @ref ZXC_ERROR_DICT_UNSUPPORTED for any
 * dictionary.
 *
 * @par Asking without a destination
 * Same probe as zxc_decompress(), answered under this context's rules: a static
 * context still rejects a foreign block size and a dictionary-bound archive.
 *
 * @param[in,out] dctx         Reusable decompression context.
 * @param[in]     src          Compressed data.
 * @param[in]     src_size     Compressed size in bytes.
 * @param[out]    dst          Destination buffer, or NULL to probe (see above).
 * @param[in]     dst_capacity Capacity of @p dst.
 *
 * @note @p src and @p dst must not overlap (same contract as memcpy).
 *
 * @return Decompressed size, 0 for an empty archive, or a negative
 *         @ref zxc_error_t.
 */
ZXC_EXPORT int64_t zxc_decompress_dctx(zxc_dctx* dctx, const void* src, size_t src_size, void* dst,
                                       size_t dst_capacity);

/**
 * @brief In-place decompression with a caller-supplied context.
 *
 * Same contract and error codes as zxc_decompress_inplace(), with @p dctx as
 * the decode context. With a static context (zxc_init_static_dctx()) the call
 * allocates nothing.
 *
 * A static context rejects a foreign block size
 * (@ref ZXC_ERROR_BAD_BLOCK_SIZE) and any dictionary
 * (@ref ZXC_ERROR_DICT_UNSUPPORTED).
 *
 * @code
 * static uint8_t ws[...] __attribute__((aligned(64)));  // zxc_static_dctx_workspace_size(bs)
 * zxc_dctx* d = zxc_init_static_dctx(ws, sizeof(ws), bs);
 * // buf: zxc_decompress_inplace_bound() bytes, archive flush-right
 * int64_t n = zxc_decompress_inplace_dctx(d, buf, cap, archive_size);
 * @endcode
 *
 * @param[in,out] dctx            Decompression context (static or heap).
 * @param[in,out] buffer          Work buffer holding the flush-right archive.
 * @param[in]     buffer_capacity Total size of @p buffer in bytes.
 * @param[in]     comp_size       Size of the compressed archive in bytes.
 * @return Decompressed size (> 0), 0 for an empty frame, or a negative
 *         @ref zxc_error_t.
 */
ZXC_EXPORT int64_t zxc_decompress_inplace_dctx(zxc_dctx* dctx, void* buffer,
                                               const size_t buffer_capacity,
                                               const size_t comp_size);

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
 * zxc_cctx *cctx = zxc_init_static_cctx(ws, ws_sz, 64 * 1024, ZXC_LEVEL_DEFAULT);
 *
 * for (each block) zxc_compress_cctx(cctx, src, n, dst, cap);
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
 * @par Pinned parameters
 * The workspace is carved for @p block_size and @p level, so
 * zxc_cctx_set_param() refuses another @ref ZXC_C_BLOCK_SIZE
 * (@ref ZXC_ERROR_BAD_BLOCK_SIZE) and a raise into @ref ZXC_LEVEL_DENSITY
 * when @p level was below it (@ref ZXC_ERROR_BAD_LEVEL, as the optimal-parser
 * scratch is absent). Lower levels, checksum and seek table are honoured
 * without re-partitioning. Dictionaries are rejected by zxc_cctx_set_dict():
 * @ref ZXC_ERROR_DICT_UNSUPPORTED.
 *
 * @param[in,out] workspace       Caller-allocated buffer, cache-line aligned.
 * @param[in]     workspace_size  Capacity of @p workspace in bytes.
 * @param[in]     block_size      Block size the workspace was sized for.
 * @param[in]     level           Level the workspace was sized for, 1 to 7.
 * @return Handle pointing inside @p workspace, or @c NULL if the workspace is
 *         too small or a parameter is invalid.
 */
ZXC_EXPORT zxc_cctx* zxc_init_static_cctx(void* workspace, const size_t workspace_size,
                                          const size_t block_size, const int level);

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
 * different @c block_size is rejected with @ref ZXC_ERROR_BAD_BLOCK_SIZE; a
 * block larger than it never decodes through the block API (see
 * zxc_decompress_block() for the codes).
 *
 * @par No dictionary
 * Any dictionary is rejected with @ref ZXC_ERROR_DICT_UNSUPPORTED, since the
 * workspace has no room for the prefix.
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

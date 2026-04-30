/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_seekable.h
 * @brief Seekable compression and random-access decompression API.
 *
 * This header provides functions to produce seekable ZXC archives and to
 * decompress arbitrary byte ranges without reading the entire file.
 *
 * A seekable archive embeds a Seek Table block (block_type = @c ZXC_BLOCK_SEK)
 * after the EOF block, recording the compressed size of every block.
 * The table is detected at read time by deriving @c num_blocks from the file
 * footer's total decompressed size and the header's block size, then seeking
 * backward to validate the SEK block header.
 * Standard (non-seekable) decompressors ignore the seek table entirely.
 *
 * @par Creating a seekable archive
 * @code
 * zxc_compress_opts_t opts = { .level = 3, .seekable = 1 };
 * int64_t csize = zxc_compress(src, src_size, dst, dst_cap, &opts);
 * @endcode
 *
 * @par Random-access decompression
 * @code
 * zxc_seekable* s = zxc_seekable_open(compressed, csize);
 * int64_t n = zxc_seekable_decompress_range(s, out, out_cap, offset, len);
 * zxc_seekable_free(s);
 * @endcode
 */

#ifndef ZXC_SEEKABLE_H
#define ZXC_SEEKABLE_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "zxc_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup seekable_api Seekable API
 * @brief Random-access compression and decompression.
 * @{
 */

/* ========================================================================= */
/*  Seekable Reader (Random-Access Decompression)                            */
/* ========================================================================= */

/**
 * @brief Opaque handle for a seekable ZXC archive.
 *
 * Created by zxc_seekable_open() or zxc_seekable_open_file().
 * Must be freed with zxc_seekable_free().
 */
typedef struct zxc_seekable_s zxc_seekable;

/**
 * @brief Opens a seekable archive from a memory buffer.
 *
 * Parses the seek table from the end of the buffer and builds the internal
 * block index.  The buffer must remain valid for the lifetime of the handle.
 *
 * @param[in] src       Pointer to the compressed data.
 * @param[in] src_size  Size of the compressed data in bytes.
 * @return Handle on success, or @c NULL if the buffer does not contain a
 *         valid seekable archive (e.g. missing seek block, bad block type).
 */
ZXC_EXPORT zxc_seekable* zxc_seekable_open(const void* src, const size_t src_size);

/**
 * @brief Opens a seekable archive from a FILE*.
 *
 * The file must be seekable (not stdin/pipe).  The current file position
 * is saved and restored after parsing the seek table.  The FILE* must
 * remain open for the lifetime of the handle.
 *
 * @param[in] f  File opened in "rb" mode.
 * @return Handle on success, or @c NULL on error.
 */
ZXC_EXPORT zxc_seekable* zxc_seekable_open_file(FILE* f);

/**
 * @brief Returns the total number of blocks in the seekable archive.
 *
 * @param[in] s  Seekable handle.
 * @return Number of data blocks (excluding EOF).
 */
ZXC_EXPORT uint32_t zxc_seekable_get_num_blocks(const zxc_seekable* s);

/**
 * @brief Returns the total decompressed size of the seekable archive.
 *
 * @param[in] s  Seekable handle.
 * @return Total decompressed size in bytes.
 */
ZXC_EXPORT uint64_t zxc_seekable_get_decompressed_size(const zxc_seekable* s);

/**
 * @brief Returns the compressed size of a specific block.
 *
 * This is the "on-disk" size including block header, payload, and optional
 * per-block checksum.
 *
 * @param[in] s          Seekable handle.
 * @param[in] block_idx  Zero-based block index.
 * @return Compressed block size, or 0 if @p block_idx is out of range.
 */
ZXC_EXPORT uint32_t zxc_seekable_get_block_comp_size(const zxc_seekable* s,
                                                     const uint32_t block_idx);

/**
 * @brief Returns the decompressed size of a specific block.
 *
 * @param[in] s          Seekable handle.
 * @param[in] block_idx  Zero-based block index.
 * @return Decompressed block size, or 0 if @p block_idx is out of range.
 */
ZXC_EXPORT uint32_t zxc_seekable_get_block_decomp_size(const zxc_seekable* s,
                                                       const uint32_t block_idx);

/**
 * @brief Decompresses an arbitrary byte range from the original data.
 *
 * Only the blocks overlapping [@p offset, @p offset + @p len) are read and
 * decompressed.  This is the core random-access primitive.
 *
 * @param[in,out] s            Seekable handle.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst (must be >= @p len).
 * @param[in]     offset       Byte offset into the original uncompressed data.
 * @param[in]     len          Number of bytes to decompress.
 * @return Number of bytes written to @p dst (== @p len on success),
 *         or a negative @ref zxc_error_t code on failure.
 */
ZXC_EXPORT int64_t zxc_seekable_decompress_range(zxc_seekable* s, void* dst,
                                                 const size_t dst_capacity, const uint64_t offset,
                                                 const size_t len);

/**
 * @brief Multi-threaded variant of zxc_seekable_decompress_range().
 *
 * Decompresses blocks in parallel using a fork-join thread pool.  Each worker
 * thread owns its own decompression context and reads compressed data via
 * @c pread() (POSIX) or @c ReadFile() (Windows) for lock-free concurrent I/O.
 *
 * Falls back to single-threaded mode when @p n_threads <= 1 or when the
 * requested range spans a single block.
 *
 * @param[in,out] s            Seekable handle.
 * @param[out]    dst          Destination buffer.
 * @param[in]     dst_capacity Capacity of @p dst (must be >= @p len).
 * @param[in]     offset       Byte offset into the original uncompressed data.
 * @param[in]     len          Number of bytes to decompress.
 * @param[in]     n_threads    Number of worker threads (0 = auto-detect CPU cores).
 * @return Number of bytes written to @p dst (== @p len on success),
 *         or a negative @ref zxc_error_t code on failure.
 */
ZXC_EXPORT int64_t zxc_seekable_decompress_range_mt(zxc_seekable* s, void* dst,
                                                    const size_t dst_capacity,
                                                    const uint64_t offset, const size_t len,
                                                    int n_threads);

/**
 * @brief Frees a seekable handle and all associated resources.
 *
 * Safe to call with @c NULL.
 *
 * @param[in] s  Handle to free.
 */
ZXC_EXPORT void zxc_seekable_free(zxc_seekable* s);

/* ========================================================================= */
/*  Seek Table Writer (low-level)                                            */
/* ========================================================================= */

/**
 * @brief Writes a seek table to the destination buffer.
 *
 * This is a low-level helper used internally by the seekable compression
 * paths.  It writes: block_header(8) + N entries(4 each).
 * Each entry stores only @c comp_size; decompressed sizes are derived at
 * read time from the file header's block_size.
 *
 * @param[out] dst             Destination buffer.
 * @param[in]  dst_capacity    Capacity of @p dst in bytes.
 * @param[in]  comp_sizes      Array of compressed block sizes.
 * @param[in]  num_blocks      Number of blocks.
 * @return Number of bytes written, or a negative @ref zxc_error_t on failure.
 */
ZXC_EXPORT int64_t zxc_write_seek_table(uint8_t* dst, const size_t dst_capacity,
                                        const uint32_t* comp_sizes, const uint32_t num_blocks);

/**
 * @brief Returns the encoded size of a seek table for the given block count.
 *
 * @param[in] num_blocks     Number of blocks.
 * @return Total byte size of the seek table.
 */
ZXC_EXPORT size_t zxc_seek_table_size(const uint32_t num_blocks);

/** @} */ /* end of seekable_api */

#ifdef __cplusplus
}
#endif

#endif /* ZXC_SEEKABLE_H */

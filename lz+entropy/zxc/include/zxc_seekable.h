/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) Bertrand Lebonnois and contributors.
 */

/**
 * @file zxc_seekable.h
 * @brief Seekable compression and random-access decompression API.
 *
 * Produce seekable ZXC archives, and decompress arbitrary byte ranges out of
 * them without reading the whole file.
 *
 * A seekable archive carries a Seek Table block (block_type = @c ZXC_BLOCK_SEK)
 * after the EOF block: groups of 64 blocks, each its first block's offset then
 * every block's on-disk size. Readers derive @c num_blocks from the footer and
 * read a group when one of its blocks is accessed: open costs the same at any
 * block count and no table stays in memory. A header flag announces it; plain
 * decompressors skip it and refuse a tail that disagrees.
 *
 * Seekable archives concatenated read as one: blocks and offsets run across the
 * frames, at most 2^20 of them. Every frame must carry a table; block sizes,
 * checksums and dictionaries may differ.
 *
 * No field holds the block count, so nothing caps it but the footer's 64-bit
 * size: counts and block indices are 64-bit here. Writing a table still costs
 * four bytes of memory per block, which is what limits a producer today.
 *
 * The table is not authenticated: a consistent forgery can point a block at
 * another of the same size. Only checksums verified through
 * @ref zxc_seekable_set_checksum bind a block to its position.
 *
 * This header is freestanding: only @c <stddef.h>, @c <stdint.h> and the rest
 * of the ZXC public API, no @c <stdio.h>, so kernel-space and other
 * freestanding environments can include it. The @c FILE*-based
 * @ref zxc_seekable_open_file entry point lives in @c zxc_stream.h.
 *
 * @par Creating a seekable archive
 * @code
 * zxc_compress_opts_t opts = { .level = 3, .seekable = 1 };
 * int64_t csize = zxc_compress(src, src_size, dst, dst_cap, &opts);
 * @endcode
 *
 * @par Random-access decompression (buffer-backed)
 * @code
 * zxc_seekable* s = zxc_seekable_open(compressed, csize);
 * int64_t n = zxc_seekable_decompress_range(s, out, out_cap, offset, len);
 * zxc_seekable_free(s);
 * @endcode
 *
 * @par Random-access decompression (custom storage)
 * @code
 * zxc_reader_t r = { .read_at = my_read_at, .ctx = my_state, .size = total };
 * zxc_seekable* s = zxc_seekable_open_reader(&r);
 * @endcode
 */

#ifndef ZXC_SEEKABLE_H
#define ZXC_SEEKABLE_H

#include <stddef.h>
#include <stdint.h>

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
 * Created by zxc_seekable_open(), zxc_seekable_open_reader(), or
 * zxc_seekable_open_file() (see @c zxc_stream.h). Free with
 * zxc_seekable_free().
 */
typedef struct zxc_seekable_s zxc_seekable;

/**
 * @brief Opens a seekable archive from a memory buffer.
 *
 * Locates and validates the seek table at the end of the buffer; entries are
 * read from @p src on access. @p src must stay valid for the lifetime of the handle.
 * An empty seekable archive opens with 0 blocks: any non-empty range is then
 * @ref ZXC_ERROR_SRC_TOO_SMALL.
 *
 * @param[in] src       Pointer to the compressed data.
 * @param[in] src_size  Size of the compressed data in bytes.
 * @return Handle on success, or @c NULL if the buffer does not contain a
 *         valid seekable archive (e.g. missing seek block, bad block type).
 */
ZXC_EXPORT zxc_seekable* zxc_seekable_open(const void* src, const size_t src_size);

/**
 * @brief Storage-agnostic reader interface for seekable archives.
 *
 * Plugs any backend behind the seekable reader: mmap, HTTP range requests, S3,
 * a custom VFS, kernel @c vfs_read. Positional reads only, no seeking state.
 *
 * @par Thread safety
 * @c read_at MUST be safe to call concurrently under
 * zxc_seekable_decompress_range_mt(), or when calls on one handle overlap: all
 * of them read through it, zxc_seekable_get_block_comp_size() included.
 *
 * @par Lifetime
 * @c ctx and the backing storage must both outlive the zxc_seekable handle
 * (until zxc_seekable_free()).
 */
typedef struct {
    /**
     * @brief Reads exactly @c len bytes at @c offset into @c dst.
     *
     * @param[in,out] ctx     Opaque user context (forwarded from zxc_reader_t::ctx).
     * @param[out]    dst     Destination buffer.
     * @param[in]     len     Number of bytes to read.
     * @param[in]     offset  Byte offset from the start of the archive.
     * @return Bytes read (@c == @c len on success), or a negative
     *         @ref zxc_error_t. The seekable reader treats a short read as an
     *         error.
     */
    int64_t (*read_at)(void* ctx, void* dst, size_t len, uint64_t offset);

    /** @brief Opaque user context passed unchanged to @c read_at. */
    void* ctx;

    /** @brief Total size of the compressed archive in bytes. */
    uint64_t size;
} zxc_reader_t;

/**
 * @brief Opens a seekable archive through a user-supplied reader.
 *
 * Two reads per frame at open (at most 2^20 frames), plus the last frame's
 * EOF/SEK headers; per range, its table groups and every block decoded. This is the entry point for
 * backing the seekable API with any storage that does positional reads (mmap, HTTP, S3, a kernel
 * file descriptor).
 *
 * @param[in] r  Reader interface (must remain valid for the handle lifetime).
 * @return Handle (0 blocks if the archive is empty), or @c NULL on error.
 */
ZXC_EXPORT zxc_seekable* zxc_seekable_open_reader(const zxc_reader_t* r);

/**
 * @brief Returns the total number of blocks in the seekable archive.
 *
 * The count is derived from the footer, not stored, so it is bounded by the
 * archive size and not by any field: 64 bits here.
 *
 * @param[in] s  Seekable handle.
 * @return Number of data blocks (excluding EOF).
 */
ZXC_EXPORT uint64_t zxc_seekable_get_num_blocks(const zxc_seekable* s);

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
 * The on-disk size: block header + payload + optional per-block checksum, from
 * the block's seek table group: one read per call. Checked against bounds only:
 * a forged size within them comes back as is.
 *
 * @param[in] s          Seekable handle.
 * @param[in] block_idx  Zero-based block index.
 * @return Compressed block size, or 0 if @p block_idx is out of range or its
 *         group is unreadable or invalid.
 */
ZXC_EXPORT uint32_t zxc_seekable_get_block_comp_size(const zxc_seekable* s,
                                                     const uint64_t block_idx);

/**
 * @brief Returns the decompressed size of a specific block.
 *
 * @param[in] s          Seekable handle.
 * @param[in] block_idx  Zero-based block index.
 * @return Decompressed block size, or 0 if @p block_idx is out of range.
 */
ZXC_EXPORT uint32_t zxc_seekable_get_block_decomp_size(const zxc_seekable* s,
                                                       const uint64_t block_idx);

/**
 * @brief Decompresses an arbitrary byte range of the original data.
 *
 * The core random-access primitive: only the blocks overlapping
 * [@p offset, @p offset + @p len) are read and decompressed.
 *
 * @par Checksums
 * Per-block checksums are **not** verified unless
 * @ref zxc_seekable_set_checksum was called with a non-zero argument; a
 * mismatch then returns @ref ZXC_ERROR_BAD_CHECKSUM. Without it, a forged seek
 * table can return another block's bytes with no error.
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
 * Decompresses blocks in parallel over a fork-join pool. Each worker owns its
 * decompression context and reads through @c pread() (POSIX) or @c ReadFile()
 * (Windows), so the I/O stays lock-free.
 *
 * Falls back to the single-threaded path when @p n_threads <= 1, the range fits
 * in one block, or it spans more than 2^32 blocks (16 TiB of @p dst). Same
 * checksum rule as @ref zxc_seekable_decompress_range.
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

/**
 * @brief Turns per-block checksum verification on or off.
 *
 * Off by default, as in the frame API. No effect without checksums in the
 * archive. Applies from the next call, on both paths. Checksums are seeded with
 * each block's position, so this also binds blocks to their index, which the seek
 * table cannot.
 *
 * @param[in,out] s       Seekable handle.
 * @param[in]     enabled Non-zero to verify, 0 to skip.
 * @return @ref ZXC_OK, or @ref ZXC_ERROR_NULL_INPUT if @p s is NULL.
 */
ZXC_EXPORT int zxc_seekable_set_checksum(zxc_seekable* s, int enabled);

/**
 * @brief Attaches a pre-trained dictionary to a seekable handle.
 *
 * Content and table are copied, so the caller may free them once this returns.
 * Call it once per dictionary the frames use; a range through a frame whose
 * dictionary is missing returns @ref ZXC_ERROR_DICT_REQUIRED.
 *
 * @param[in] s         Seekable handle.
 * @param[in] dict      Dictionary content.
 * @param[in] dict_size Size in bytes (max ZXC_DICT_SIZE_MAX).
 * @param[in] dict_huf  Shared literal Huffman table (128 bytes, see
 *                      zxc_dict_huf()), or NULL if the archive was compressed
 *                      without one. Must be the compression-time table: the
 *                      archive's dict_id binds the (dict, table) pair. A pair
 *                      no frame uses: @ref ZXC_ERROR_DICT_MISMATCH, on this call,
 *                      unless no frame needs a dictionary: then it is ignored.
 * @return @ref ZXC_OK on success, or a negative @ref zxc_error_t code.
 */
ZXC_EXPORT int zxc_seekable_set_dict(zxc_seekable* s, const void* dict, size_t dict_size,
                                     const void* dict_huf);

/* ========================================================================= */
/*  Seek Table Writer (low-level)                                            */
/* ========================================================================= */

/**
 * @brief Writes a seek table to the destination buffer.
 *
 * Low-level helper used by the seekable compression paths. Layout is
 * block_header(8) + groups of 64 blocks, each a u64 anchor (its first block's
 * offset) then one u32 on-disk size per block; the header's size field holds
 * the groups' byte size folded to 32 bits, exact below 4 GiB.
 *
 * @param[out] dst             Destination buffer.
 * @param[in]  dst_capacity    Capacity of @p dst in bytes.
 * @param[in]  comp_sizes      Array of compressed block sizes.
 * @param[in]  num_blocks      Number of blocks.
 * @return Number of bytes written, or a negative @ref zxc_error_t on failure.
 */
ZXC_EXPORT int64_t zxc_write_seek_table(uint8_t* dst, const size_t dst_capacity,
                                        const uint32_t* comp_sizes, const uint64_t num_blocks);

/**
 * @brief Returns the encoded size of a seek table for the given block count.
 *
 * @param[in] num_blocks     Number of blocks.
 * @return Total byte size of the seek table, or 0 when it does not fit @c size_t.
 */
ZXC_EXPORT size_t zxc_seek_table_size(const uint64_t num_blocks);

/** @} */ /* end of seekable_api */

#ifdef __cplusplus
}
#endif

#endif /* ZXC_SEEKABLE_H */

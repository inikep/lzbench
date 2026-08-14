/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_dict.h
 * @brief Pre-trained dictionary API for ZXC compression.
 *
 * Train, save, load and identify dictionaries, which buy back compression
 * ratio on small, similar payloads. A dictionary is raw byte content that
 * prefills the LZ77 window at the start of every block, so the compressor
 * starts out already knowing the patterns instead of waiting for them to
 * show up in the input. Dictionaries live in `.zxd` files and are referenced
 * by a 32-bit ID in the ZXC file header.
 *
 * @code
 * // Train a dictionary from a corpus of JSON samples
 * void* dict_buf = malloc(32768);
 * int64_t dict_sz = zxc_train_dict(samples, sizes, n, dict_buf, 32768);
 *
 * // Train the shared literal Huffman table on the same corpus
 * uint8_t huf[ZXC_HUF_TABLE_SIZE];
 * zxc_train_dict_huf(samples, sizes, n, dict_buf, dict_sz, huf);
 *
 * // Save to .zxd file (content + table)
 * void* zxd = malloc(zxc_dict_save_bound(dict_sz));
 * int64_t zxd_sz = zxc_dict_save(dict_buf, dict_sz, huf, zxd, ...);
 *
 * // Use for compression
 * zxc_compress_opts_t opts = {
 *     .level = 6, .dict = dict_buf, .dict_size = dict_sz, .dict_huf = huf };
 * zxc_compress(src, src_size, dst, dst_capacity, &opts);
 * @endcode
 */

#ifndef ZXC_DICT_H
#define ZXC_DICT_H

#include <stddef.h>
#include <stdint.h>

#include "zxc_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup dict Dictionary
 * @brief Pre-trained dictionary training, serialization, and identification.
 * @{
 */

/**
 * @brief Computes the dictionary ID for the given content and optional table.
 *
 * A deterministic 32-bit hash, stored in the ZXC file header so the decoder can
 * check it was handed the right dictionary. With @p huf_lengths NULL it hashes
 * the content alone (the content-only dictionary of the buffer API); with a
 * table it binds the pair as `hash(table, seed = hash(content))`, which is what
 * `.zxd` files and archives compressed with a shared table carry.
 *
 * @param[in] dict        Pointer to dictionary content.
 * @param[in] dict_size   Size in bytes.
 * @param[in] huf_lengths Shared literal Huffman table (@ref ZXC_HUF_TABLE_SIZE
 *                        bytes), or NULL for a content-only ID.
 * @return 32-bit dictionary ID. Returns 0 if @p dict is NULL or @p dict_size is 0.
 */
ZXC_EXPORT uint32_t zxc_dict_id(const void* dict, size_t dict_size, const void* huf_lengths);

/**
 * @brief Loads and validates a `.zxd` dictionary file from a memory buffer.
 *
 * Zero-copy: on success @p content_out and @p huf_out point into @p buf, which
 * must stay alive as long as they are in use. One call gives everything needed
 * to (de)compress with the dictionary, feed @p content_out / @p huf_out
 * straight to the @c dict / @c dict_huf option fields.
 *
 * @param[in]  buf              Buffer containing the .zxd file.
 * @param[in]  buf_size         Size of @p buf in bytes.
 * @param[out] content_out      Receives a pointer to the dictionary content.
 * @param[out] content_size_out Receives the content size in bytes.
 * @param[out] huf_out          Receives a pointer to the 128-byte shared Huffman
 *                              table (may be NULL if not needed).
 * @param[out] dict_id_out      Receives the dictionary ID (may be NULL).
 * @return @ref ZXC_OK on success, or a negative @ref zxc_error_t code.
 */
ZXC_EXPORT int zxc_dict_load(const void* buf, size_t buf_size, const void** content_out,
                             size_t* content_size_out, const void** huf_out, uint32_t* dict_id_out);

/**
 * @brief Serializes dictionary content and its shared Huffman table to `.zxd`.
 *
 * The 128-byte packed code-lengths table (from zxc_train_dict_huf()) is
 * mandatory and follows the content. The stored dict_id covers both, so
 * archives compressed with this dictionary are bound to the exact pair.
 *
 * @param[in]  content       Raw dictionary content.
 * @param[in]  content_size  Size of @p content in bytes (max ZXC_DICT_SIZE_MAX).
 * @param[in]  huf_lengths   128-byte packed Huffman code lengths (required).
 * @param[out] buf           Output buffer for the .zxd file.
 * @param[in]  buf_capacity  Capacity of @p buf (see zxc_dict_save_bound()).
 * @return Number of bytes written on success, or a negative @ref zxc_error_t code.
 */
ZXC_EXPORT int64_t zxc_dict_save(const void* content, size_t content_size, const void* huf_lengths,
                                 void* buf, size_t buf_capacity);

/**
 * @brief Returns the `.zxd` file size for a given content size.
 *
 * @param[in] content_size Size of the dictionary content.
 * @return Total .zxd file size (header + content).
 */
ZXC_EXPORT size_t zxc_dict_save_bound(size_t content_size);

/**
 * @brief Returns the dictionary ID stored in a `.zxd` file buffer.
 *
 * Reads the header's dict_id field only; the rest of the file is not validated.
 *
 * @param[in] buf       Buffer containing the .zxd file.
 * @param[in] buf_size  Size of @p buf in bytes.
 * @return Dictionary ID, or 0 if @p buf is too small or the magic word does
 *         not match.
 */
ZXC_EXPORT uint32_t zxc_dict_get_id(const void* buf, size_t buf_size);

/**
 * @brief Trains a dictionary from a corpus of samples.
 *
 * Picks the byte sequences that maximize LZ77 match coverage across the
 * samples. The content can go straight into zxc_compress_opts_t::dict, or
 * through zxc_dict_save().
 *
 * @param[in]  samples        Array of pointers to sample buffers.
 * @param[in]  sample_sizes   Array of sample sizes in bytes.
 * @param[in]  n_samples      Number of samples.
 * @param[out] dict_buf       Output buffer for trained dictionary content.
 * @param[in]  dict_capacity  Capacity of @p dict_buf (max ZXC_DICT_SIZE_MAX).
 * @return Size of the trained dictionary on success, or a negative
 *         @ref zxc_error_t code.
 */
ZXC_EXPORT int64_t zxc_train_dict(const void* const* samples, const size_t* sample_sizes,
                                  size_t n_samples, void* dict_buf, size_t dict_capacity);

/**
 * @brief Trains the shared literal Huffman table for an already-trained dictionary.
 *
 * Compresses the samples with @p dict and derives canonical Huffman code
 * lengths from the real post-LZ literal distribution. Embed the resulting
 * 128-byte packed table in a `.zxd` via zxc_dict_save(), or pass it through the
 * `dict_huf` option field. Blocks whose literals compress better with the
 * shared table drop their own 128-byte table header, which is decisive at
 * small block sizes.
 *
 * @param[in]  samples         Array of pointers to sample buffers (typically
 *                             the same corpus used for zxc_train_dict()).
 * @param[in]  sample_sizes    Array of sample sizes in bytes.
 * @param[in]  n_samples       Number of samples.
 * @param[in]  dict            Trained dictionary content.
 * @param[in]  dict_size       Dictionary content size in bytes.
 * @param[out] huf_lengths_out Receives the 128-byte packed code-lengths table.
 * @return @ref ZXC_OK on success, or a negative @ref zxc_error_t code.
 */
ZXC_EXPORT int zxc_train_dict_huf(const void* const* samples, const size_t* sample_sizes,
                                  size_t n_samples, const void* dict, size_t dict_size,
                                  uint8_t* huf_lengths_out);

/**
 * @brief One-call dictionary creation: content + shared table, serialized to
 *        ready-to-write `.zxd` bytes.
 *
 * Runs zxc_train_dict(), then zxc_train_dict_huf() on the trained content, then
 * zxc_dict_save() into @p zxd_buf. Size @p zxd_capacity from the dictionary you
 * expect, or use zxc_dict_save_bound(ZXC_DICT_SIZE_MAX) for a safe upper bound.
 * The three primitives stay available for the cases this shortcut does not
 * cover: content-only dictionaries, retraining just the table, externally
 * sourced content.
 *
 * @param[in]  samples       Array of pointers to sample buffers.
 * @param[in]  sample_sizes  Array of sample sizes in bytes.
 * @param[in]  n_samples     Number of samples.
 * @param[out] zxd_buf       Output buffer for the `.zxd` file.
 * @param[in]  zxd_capacity  Capacity of @p zxd_buf.
 * @return Number of `.zxd` bytes written on success, or a negative
 *         @ref zxc_error_t code.
 */
ZXC_EXPORT int64_t zxc_dict_train(const void* const* samples, const size_t* sample_sizes,
                                  size_t n_samples, void* zxd_buf, size_t zxd_capacity);

/**
 * @brief Returns a pointer to the shared Huffman table inside a `.zxd` buffer.
 *
 * Zero-copy: the returned pointer aims into @p buf and lives as long as it does.
 *
 * @param[in] buf       Buffer containing the .zxd file.
 * @param[in] buf_size  Size of @p buf in bytes.
 * @return Pointer to the 128-byte packed code-lengths table, or NULL if @p buf
 *         is not a valid `.zxd` file or carries no table.
 */
ZXC_EXPORT const void* zxc_dict_huf(const void* buf, size_t buf_size);

/** @} */ /* end of dict */

#ifdef __cplusplus
}
#endif

#endif /* ZXC_DICT_H */

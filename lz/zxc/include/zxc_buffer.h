/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ZXC_BUFFER_H
#define ZXC_BUFFER_H

#include <stddef.h>
#include <stdint.h>

#include "zxc_export.h"

/*
 * ============================================================================
 * ZXC Compression Library - Public API (Buffer-Based)
 * ============================================================================
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
 * It writes the ZXC file header followed by compressed blocks
 *
 * @param[in] src          Pointer to the source buffer.
 * @param[in] src_size     Size of the source data in bytes.
 * @param[out] dst          Pointer to the destination buffer.
 * @param[in] dst_capacity Maximum capacity of the destination buffer.
 * @param[in] level        Compression level (e.g., ZXC_LEVEL_BALANCED).
 * @param[in] checksum_enabled Flag indicating whether to verify the checksum of the
 * data (1 to enable, 0 to disable).
 *
 * @return The number of bytes written to dst, or 0 if the destination buffer
 * is too small or an error occurred.
 */
ZXC_EXPORT size_t zxc_compress(const void* src, const size_t src_size, void* dst,
                               const size_t dst_capacity, const int level,
                               const int checksum_enabled);

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
 * @param[in] checksum_enabled Flag indicating whether to verify the checksum of the
 * data (1 to enable, 0 to disable).
 *
 * @return The number of bytes written to dst, or 0 if decompression fails
 * (invalid header, corruption, or destination too small).
 */
ZXC_EXPORT size_t zxc_decompress(const void* src, const size_t src_size, void* dst,
                                 const size_t dst_capacity, const int checksum_enabled);

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

#endif  // ZXC_BUFFER_H
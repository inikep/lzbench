/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_stream.h
 * @brief Multi-threaded streaming compression and decompression API.
 *
 * This header provides the streaming driver that reads from a @c FILE* input and
 * writes compressed (or decompressed) output to a @c FILE*.  Internally the
 * driver uses an asynchronous Producer-Consumer pipeline via a ring buffer to
 * separate I/O from CPU-intensive work:
 *
 * 1. **Reader thread**  - reads chunks from `f_in`.
 * 2. **Worker threads** - compress/decompress chunks in parallel.
 * 3. **Writer thread**  - orders the results and writes them to `f_out`.
 *
 * @see zxc_buffer.h  for the simple one-shot buffer API.
 * @see zxc_sans_io.h for low-level sans-I/O building blocks.
 */

#ifndef ZXC_STREAM_H
#define ZXC_STREAM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "zxc_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup stream_api Streaming API
 * @brief Multi-threaded, FILE*-based compression and decompression.
 * @{
 */

/**
 * @brief Progress callback function type.
 *
 * This callback is invoked periodically during compression/decompression to report
 * progress. It is called from the writer thread after each block is processed.
 *
 * @param[in] bytes_processed Total input bytes processed so far.
 * @param[in] bytes_total     Total input bytes to process (0 if unknown, e.g., stdin).
 * @param[in] user_data       User-provided context pointer (passed through from API call).
 *
 * @note The callback should be fast and non-blocking. Avoid heavy I/O or mutex locks.
 */
typedef void (*zxc_progress_callback_t)(uint64_t bytes_processed, uint64_t bytes_total,
                                        const void* user_data);

/**
 * @brief Options for streaming compression.
 *
 * Zero-initialise for safe defaults: level 0 maps to ZXC_LEVEL_DEFAULT (3),
 * block_size 0 maps to ZXC_BLOCK_SIZE_DEFAULT (256 KB), n_threads 0 means
 * auto-detect, and all other fields are disabled.
 *
 * @code
 * zxc_compress_opts_t opts = { .level = ZXC_LEVEL_COMPACT };
 * zxc_stream_compress(f_in, f_out, &opts);
 * @endcode
 */
typedef struct {
    int n_threads;        /**< Worker thread count (0 = auto-detect CPU cores). */
    int level;            /**< Compression level 1-5 (0 = default, ZXC_LEVEL_DEFAULT). */
    size_t block_size;    /**< Block size in bytes (0 = default 256 KB). Must be power of 2, [4KB -
                             2MB]. */
    int checksum_enabled; /**< 1 to enable per-block and global checksums, 0 to disable. */
    zxc_progress_callback_t progress_cb; /**< Optional progress callback (NULL to disable). */
    void* user_data;                     /**< User context pointer passed to progress_cb. */
} zxc_compress_opts_t;

/**
 * @brief Options for streaming decompression.
 *
 * Zero-initialise for safe defaults.
 *
 * @code
 * zxc_decompress_opts_t opts = { .checksum = 1 };
 * zxc_stream_decompress(f_in, f_out, &opts);
 * @endcode
 */
typedef struct {
    int n_threads;        /**< Worker thread count (0 = auto-detect CPU cores). */
    int checksum_enabled; /**< 1 to verify per-block and global checksums, 0 to skip. */
    zxc_progress_callback_t progress_cb; /**< Optional progress callback (NULL to disable). */
    void* user_data;                     /**< User context pointer passed to progress_cb. */
} zxc_decompress_opts_t;

/**
 * @brief Compresses data from an input stream to an output stream.
 *
 * This function sets up a multi-threaded pipeline:
 * 1. Reader Thread: Reads chunks from f_in.
 * 2. Worker Threads: Compress chunks in parallel (LZ77 + Bitpacking).
 * 3. Writer Thread: Orders the processed chunks and writes them to f_out.
 *
 * @param[in] f_in   Input file stream (must be opened in "rb" mode).
 * @param[out] f_out  Output file stream (must be opened in "wb" mode).
 * @param[in] opts   Compression options (NULL uses all defaults).
 *
 * @return Total compressed bytes written, or a negative zxc_error_t code (e.g.,
 * ZXC_ERROR_IO) if an error occurred.
 */
ZXC_EXPORT int64_t zxc_stream_compress(FILE* f_in, FILE* f_out, const zxc_compress_opts_t* opts);

/**
 * @brief Decompresses data from an input stream to an output stream.
 *
 * Uses the same pipeline architecture as compression to maximize throughput.
 *
 * @param[in] f_in   Input file stream (must be opened in "rb" mode).
 * @param[out] f_out  Output file stream (must be opened in "wb" mode).
 * @param[in] opts   Decompression options (NULL uses all defaults).
 *
 * @return Total decompressed bytes written, or a negative zxc_error_t code (e.g.,
 * ZXC_ERROR_BAD_HEADER) if an error occurred.
 */
ZXC_EXPORT int64_t zxc_stream_decompress(FILE* f_in, FILE* f_out,
                                         const zxc_decompress_opts_t* opts);

/**
 * @brief Returns the decompressed size stored in a ZXC compressed file.
 *
 * This function reads the file footer to extract the original uncompressed size
 * without performing any decompression. The file position is restored after reading.
 *
 * @param[in] f_in  Input file stream (must be opened in "rb" mode).
 *
 * @return The original uncompressed size in bytes, or a negative zxc_error_t code (e.g.,
 * ZXC_ERROR_BAD_MAGIC) if the file is invalid or an I/O error occurred.
 */
ZXC_EXPORT int64_t zxc_stream_get_decompressed_size(FILE* f_in);

/** @} */ /* end of stream_api */

#ifdef __cplusplus
}
#endif

#endif  // ZXC_STREAM_H
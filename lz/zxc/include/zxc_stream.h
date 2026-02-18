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
 * Extended variants (`_ex` suffix) accept an optional progress callback that
 * is invoked from the writer thread after each block.
 *
 * @see zxc_buffer.h  for the simple one-shot buffer API.
 * @see zxc_sans_io.h for low-level sans-I/O building blocks.
 */

#ifndef ZXC_STREAM_H
#define ZXC_STREAM_H

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
 * @brief Compresses data from an input stream to an output stream.
 *
 * This function sets up a multi-threaded pipeline:
 * 1. Reader Thread: Reads chunks from f_in.
 * 2. Worker Threads: Compress chunks in parallel (LZ77 + Bitpacking).
 * 3. Writer Thread: Orders the processed chunks and writes them to f_out.
 *
 * @param[in] f_in      Input file stream (must be opened in "rb" mode).
 * @param[out] f_out     Output file stream (must be opened in "wb" mode).
 * @param[in] n_threads Number of worker threads to spawn (0 = auto-detect number of
 * CPU cores).
 * @param[in] level     Compression level (1-5, see @ref zxc_compression_level_t).
 * @param[in] checksum_enabled  If non-zero, enables checksum verification for data
 * integrity.
 *
 * @return          Total compressed bytes written, or a negative zxc_error_t code (e.g.,
 * ZXC_ERROR_IO) if an error occurred.
 */
ZXC_EXPORT int64_t zxc_stream_compress(FILE* f_in, FILE* f_out, const int n_threads,
                                       const int level, const int checksum_enabled);

/**
 * @brief Decompresses data from an input stream to an output stream.
 *
 * Uses the same pipeline architecture as compression to maximize throughput.
 *
 * @param[in] f_in      Input file stream (must be opened in "rb" mode).
 * @param[out] f_out     Output file stream (must be opened in "wb" mode).
 * @param[in] n_threads Number of worker threads to spawn (0 = auto-detect number of
 * CPU cores).
 * @param[in] checksum_enabled  If non-zero, enables checksum verification for data
 * integrity.
 *
 * @return          Total decompressed bytes written, or a negative zxc_error_t code (e.g.,
 * ZXC_ERROR_BAD_HEADER) if an error occurred.
 */
ZXC_EXPORT int64_t zxc_stream_decompress(FILE* f_in, FILE* f_out, const int n_threads,
                                         const int checksum_enabled);

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
 * @brief Compresses data from an input stream to an output stream (with progress callback).
 *
 * Extended version of zxc_stream_compress() that accepts an optional progress callback.
 *
 * @param[in] f_in             Input file stream (must be opened in "rb" mode).
 * @param[out] f_out           Output file stream (must be opened in "wb" mode).
 * @param[in] n_threads        Number of worker threads to spawn (0 = auto-detect).
 * @param[in] level            Compression level (1-5, see @ref zxc_compression_level_t).
 * @param[in] checksum_enabled If non-zero, enables checksum verification.
 * @param[in] progress_cb      Optional progress callback (NULL to disable).
 * @param[in] user_data        User context pointer passed to progress callback.
 *
 * @return          Total compressed bytes written, or a negative zxc_error_t code if an error
 * occurred.
 */
ZXC_EXPORT int64_t zxc_stream_compress_ex(FILE* f_in, FILE* f_out, const int n_threads,
                                          const int level, const int checksum_enabled,
                                          zxc_progress_callback_t progress_cb, void* user_data);

/**
 * @brief Decompresses data from an input stream to an output stream (with progress callback).
 *
 * Extended version of zxc_stream_decompress() that accepts an optional progress callback.
 *
 * @param[in] f_in             Input file stream (must be opened in "rb" mode).
 * @param[out] f_out           Output file stream (must be opened in "wb" mode).
 * @param[in] n_threads        Number of worker threads to spawn (0 = auto-detect).
 * @param[in] checksum_enabled If non-zero, enables checksum verification.
 * @param[in] progress_cb      Optional progress callback (NULL to disable).
 * @param[in] user_data        User context pointer passed to progress callback.
 *
 * @return          Total decompressed bytes written, or a negative zxc_error_t code if an error
 * occurred.
 */
ZXC_EXPORT int64_t zxc_stream_decompress_ex(FILE* f_in, FILE* f_out, const int n_threads,
                                            const int checksum_enabled,
                                            zxc_progress_callback_t progress_cb, void* user_data);

/** @} */ /* end of stream_api */

#ifdef __cplusplus
}
#endif

#endif  // ZXC_STREAM_H
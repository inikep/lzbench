/*
 * Copyright (c) 2025-2026, Bertrand Lebonnois
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef ZXC_STREAM_H
#define ZXC_STREAM_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ============================================================================
 * ZXC Compression Library - Public Streaming Driver API
 * ============================================================================
 * This driver uses an asynchronous pipeline architecture (Producer-Consumer)
 * via a Ring Buffer to separate I/O operations from CPU-intensive compression
 * tasks.
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
 * @param[in] level     Compression level (1-9).
 * @param[in] checksum_enabled  If non-zero, enables checksum verification for data
 * integrity.
 *
 * @return          Total compressed bytes written, or -1 if an error occurred.
 */
int64_t zxc_stream_compress(FILE* f_in, FILE* f_out, int n_threads, int level,
                            int checksum_enabled);

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
 * @return          Total decompressed bytes written, or -1 if an error
 * occurred.
 */
int64_t zxc_stream_decompress(FILE* f_in, FILE* f_out, int n_threads, int checksum_enabled);

#ifdef __cplusplus
}
#endif

#endif  // ZXC_STREAM_H
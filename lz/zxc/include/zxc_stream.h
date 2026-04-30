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
 * @see zxc_pstream.h for single-threaded push-based streaming.
 */

#ifndef ZXC_STREAM_H
#define ZXC_STREAM_H

#include <stdint.h>
#include <stdio.h>

#include "zxc_export.h"
#include "zxc_opts.h"

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
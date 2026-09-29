/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_stream.h
 * @brief @c FILE*-flavored variants of the ZXC API.
 *
 * Every public entry point that needs @c <stdio.h> lives here, so kernel and
 * freestanding builds can just leave this header out and use @c zxc_buffer.h,
 * @c zxc_pstream.h and the storage-agnostic part of @c zxc_seekable.h.
 *
 * Two subsystems:
 *
 * 1. **Multi-threaded streaming driver**: @c FILE* in, @c FILE* out. A ring
 *    buffer keeps I/O off the CPU-bound work, producer-consumer style:
 *      - reader thread: reads chunks from @c f_in;
 *      - worker threads: compress/decompress chunks in parallel;
 *      - writer thread: reorders the results and writes them to @c f_out.
 *    See @ref zxc_stream_compress, @ref zxc_stream_decompress,
 *    @ref zxc_stream_get_decompressed_size.
 *
 * 2. **Seekable @c FILE* open helper**: wraps a @c FILE* into a thread-safe
 *    @c pread / @c ReadFile-backed @ref zxc_reader_t and hands it to
 *    @ref zxc_seekable_open_reader. See @ref zxc_seekable_open_file.
 *
 * @see zxc_buffer.h   for the simple one-shot buffer API.
 * @see zxc_pstream.h  for single-threaded push-based streaming.
 * @see zxc_seekable.h for the storage-agnostic seekable reader.
 */

#ifndef ZXC_STREAM_H
#define ZXC_STREAM_H

#include <stdint.h>
#include <stdio.h>

#include "zxc_export.h"
#include "zxc_opts.h"
#include "zxc_seekable.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup stream_api Streaming API
 * @brief Multi-threaded, FILE*-based compression and decompression.
 * @{
 */

/**
 * @brief Compresses one file stream into another.
 *
 * Runs the multi-threaded pipeline: a reader thread pulls chunks from @p f_in,
 * worker threads compress them in parallel (LZ77 + bitpacking), and a writer
 * thread reorders and writes them to @p f_out.
 *
 * @param[in]  f_in   Input stream, opened in "rb" mode.
 * @param[out] f_out  Output stream, opened in "wb" mode.
 * @param[in]  opts   Compression options, or NULL for defaults.
 *
 * @return Total compressed bytes written, or a negative @ref zxc_error_t
 *         (e.g. @ref ZXC_ERROR_IO).
 */
ZXC_EXPORT int64_t zxc_stream_compress(FILE* f_in, FILE* f_out, const zxc_compress_opts_t* opts);

/**
 * @brief Decompresses one file stream into another.
 *
 * Same pipeline as compression, for the same throughput reasons.
 *
 * @param[in]  f_in   Input stream, opened in "rb" mode.
 * @param[out] f_out  Output stream, opened in "wb" mode.
 * @param[in]  opts   Decompression options, or NULL for defaults.
 *
 * @return Total decompressed bytes written, or a negative @ref zxc_error_t
 *         (e.g. @ref ZXC_ERROR_BAD_HEADER).
 */
ZXC_EXPORT int64_t zxc_stream_decompress(FILE* f_in, FILE* f_out,
                                         const zxc_decompress_opts_t* opts);

/**
 * @brief Reads the original size from a ZXC file's footer, without decoding.
 *
 * The file position is restored afterwards.
 *
 * @param[in] f_in  Input stream, opened in "rb" mode.
 *
 * @return Original uncompressed size in bytes, or a negative @ref zxc_error_t
 *         (e.g. @ref ZXC_ERROR_BAD_MAGIC) on an invalid file or an I/O error.
 */
ZXC_EXPORT int64_t zxc_stream_get_decompressed_size(FILE* f_in);

/* ========================================================================= */
/*  Seekable FILE* open helper                                               */
/* ========================================================================= */

/**
 * @brief Opens a seekable archive from a @c FILE*.
 *
 * Builds a @ref zxc_reader_t doing thread-safe positional reads (@c pread on
 * POSIX, @c ReadFile + @c OVERLAPPED on Windows) on @p f's descriptor, then
 * delegates to @ref zxc_seekable_open_reader. The current file position is
 * saved and restored, and @p f must stay open for the lifetime of the handle.
 *
 * It lives here with the other @c FILE* entry points rather than in
 * @c zxc_seekable.h, which stays freestanding (kernel-includable).
 *
 * @param[in] f  File opened in @c "rb" mode (must be seekable, not a pipe).
 * @return Handle on success (free with @ref zxc_seekable_free), or @c NULL
 *         on error.
 */
ZXC_EXPORT zxc_seekable* zxc_seekable_open_file(FILE* f);

/** @} */ /* end of stream_api */

#ifdef __cplusplus
}
#endif

#endif  // ZXC_STREAM_H
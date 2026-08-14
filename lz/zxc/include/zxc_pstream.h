/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_pstream.h
 * @brief Push-based, single-threaded streaming compression and decompression.
 *
 * The counterpart of the @c FILE*-based @ref zxc_stream_compress / @ref
 * zxc_stream_decompress. Those own the pipeline (read until EOF, write until
 * done); here the caller stays in control: feed input chunks when they arrive,
 * drain output chunks when ready, finalise on demand. Nothing blocks.
 *
 * That is what you want inside a callback-driven library, an asynchronous
 * event loop, a non-seeking network protocol (HTTP chunked transfer, gRPC,
 * your own binary protocol), or any pipeline with no @c FILE* to block on.
 *
 * One context, one thread at a time. To compress a single file end-to-end
 * across threads, use @ref zxc_stream_compress instead.
 *
 * @par Compression usage
 * @code
 * zxc_compress_opts_t opts = { .level = 3, .checksum_enabled = 1 };
 * zxc_cstream* cs = zxc_cstream_create(&opts);
 *
 * uint8_t in_buf[64*1024], out_buf[64*1024];
 * zxc_outbuf_t out = { out_buf, sizeof out_buf, 0 };
 *
 * ssize_t n;
 * while ((n = read_some(in_buf, sizeof in_buf)) > 0) {
 *     zxc_inbuf_t in = { in_buf, (size_t)n, 0 };
 *     while (in.pos < in.size) {
 *         int64_t r = zxc_cstream_compress(cs, &out, &in);
 *         if (r < 0) goto fatal;
 *         if (out.pos > 0) { write_to_sink(out_buf, out.pos); out.pos = 0; }
 *     }
 * }
 *
 * int64_t pending;
 * do {
 *     pending = zxc_cstream_end(cs, &out);
 *     if (pending < 0) goto fatal;
 *     if (out.pos > 0) { write_to_sink(out_buf, out.pos); out.pos = 0; }
 * } while (pending > 0);
 *
 * zxc_cstream_free(cs);
 * @endcode
 *
 * @see zxc_stream.h  for the multi-threaded @c FILE*-based pipeline.
 * @see zxc_buffer.h  for one-shot in-memory compression.
 */

#ifndef ZXC_PSTREAM_H
#define ZXC_PSTREAM_H

#include <stddef.h>
#include <stdint.h>

#include "zxc_export.h"
#include "zxc_opts.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup pstream Push Streaming API
 * @brief Caller-driven, single-threaded streaming compression and decompression.
 * @{
 */

/**
 * @brief Input buffer descriptor for push streaming.
 *
 * The caller fills @c src and sets @c size; the library advances @c pos as it
 * consumes input. Do not touch @c pos between calls.
 */
typedef struct {
    const void* src; /**< Caller-owned input bytes. */
    size_t size;     /**< Total bytes available in @c src. */
    size_t pos;      /**< Bytes already consumed by the library (in/out). */
} zxc_inbuf_t;

/**
 * @brief Output buffer descriptor for push streaming.
 *
 * The caller provides a writable region of capacity @c size at @c dst. The
 * library writes from @c dst+pos and advances @c pos by what it produced; the
 * caller drains @c [dst, dst+pos) and resets @c pos to 0 between rounds (or
 * grows @c size).
 *
 * Treat the ENTIRE region @c [dst+pos, dst+size) as scratch: when the
 * remaining capacity allows it the decoder writes blocks straight into it with
 * speculative (wild-copy) stores, so bytes past the final @c pos are
 * unspecified after a call. Never leave live data inside the declared capacity.
 */
typedef struct {
    void* dst;   /**< Caller-owned output region. */
    size_t size; /**< Total capacity available at @c dst. */
    size_t pos;  /**< Bytes already produced by the library (in/out). */
} zxc_outbuf_t;

/* Opaque streaming contexts. */
/** @brief Opaque push-model compression stream (see @ref zxc_cstream_create). */
typedef struct zxc_cstream_s zxc_cstream;
/** @brief Opaque push-model decompression stream (see @ref zxc_dstream_create). */
typedef struct zxc_dstream_s zxc_dstream;

/* ===== Compression =================================================== */

/**
 * @brief Creates a push compression stream.
 *
 * @p opts is copied into the context and may be freed or reused afterwards.
 *
 * Only @c level, @c block_size and @c checksum_enabled are honoured, and they
 * must be valid (an unsupported @c block_size fails creation); levels above
 * @ref ZXC_LEVEL_ULTRA are clamped. @c n_threads is ignored, this API being
 * single-threaded, see @ref zxc_stream_compress for the multi-threaded
 * @c FILE* pipeline. Dictionary options are rejected outright: the push-stream
 * format carries no dict_id, so @c dict / @c dict_size / @c dict_huf fail
 * creation rather than being silently dropped.
 *
 * @param[in] opts  Compression options, or @c NULL for all defaults.
 * @return Context to release with @ref zxc_cstream_free, or @c NULL on
 *         allocation failure or invalid options.
 */
ZXC_EXPORT zxc_cstream* zxc_cstream_create(const zxc_compress_opts_t* opts);

/**
 * @brief Releases a compression stream and all internal buffers.
 *
 * Safe to call with @c NULL (no-op).
 *
 * @param[in] cs  Stream returned by @ref zxc_cstream_create.
 */
ZXC_EXPORT void zxc_cstream_free(zxc_cstream* cs);

/**
 * @brief Pushes input bytes into the stream and drains compressed output.
 *
 * Reads from @c in->src at @c in->pos, writes to @c out->dst at @c out->pos,
 * advancing both as data flows. Each call goes as far as the two buffers allow:
 *
 * - emits the file header on the first invocation (16 B);
 * - copies input into the internal block accumulator;
 * - compresses one block into @p out whenever the accumulator fills;
 * - returns once @p in is fully consumed *and* nothing is pending, or as soon
 *   as @p out has no room left.
 *
 * Fully reentrant: if @p out fills mid-block, the next call picks up where this
 * one stopped. Calling with @c in->size == in->pos drains only.
 *
 * @par Errors
 * Errors are sticky: once one is returned, @ref zxc_cstream_compress and
 * @ref zxc_cstream_end keep returning the same code and do no further work.
 * Only @ref zxc_cstream_free is safe from there.
 *
 * @param[in,out] cs   Compression stream.
 * @param[in,out] out  Output descriptor; @c pos is advanced by produced bytes.
 * @param[in,out] in   Input descriptor;  @c pos is advanced by consumed bytes.
 *
 * @return @c 0 @p in fully consumed and nothing pending in staging;
 *         @c >0 bytes still pending, drain @p out and call again with the same
 *         (or new) input;
 *         @c <0 a @ref zxc_error_t code.
 */
ZXC_EXPORT int64_t zxc_cstream_compress(zxc_cstream* cs, zxc_outbuf_t* out, zxc_inbuf_t* in);

/**
 * @brief Finalises the stream: flushes pending data, writes EOF block + footer.
 *
 * Required after the last @ref zxc_cstream_compress call to end up with a valid
 * ZXC file. Reentrant like it: if @p out fills first, it returns a positive
 * count and the caller drains and calls again.
 *
 * Once it returns @c 0 the stream is DONE and any further call returns
 * @c ZXC_ERROR_NULL_INPUT; release it with @ref zxc_cstream_free.
 *
 * @param[in,out] cs   Compression stream.
 * @param[in,out] out  Output descriptor.
 *
 * @return @c 0 finalised, the file is now valid;
 *         @c >0 bytes still pending, drain @p out and call again;
 *         @c <0 a @ref zxc_error_t code.
 */
ZXC_EXPORT int64_t zxc_cstream_end(zxc_cstream* cs, zxc_outbuf_t* out);

/**
 * @brief Suggested input chunk size for best throughput.
 *
 * The configured block size (512 KB by default).
 *
 * @param[in] cs  Compression stream.
 * @return Suggested @c in_buf capacity in bytes, or 0 if @p cs is @c NULL.
 */
ZXC_EXPORT size_t zxc_cstream_in_size(const zxc_cstream* cs);

/**
 * @brief Suggested output chunk size that never triggers a partial drain.
 *
 * Holds one full compressed block plus framing overhead. Smaller outputs work,
 * at the cost of an extra drain loop.
 *
 * @param[in] cs  Compression stream.
 * @return Suggested @c out_buf capacity in bytes, or 0 if @p cs is @c NULL.
 */
ZXC_EXPORT size_t zxc_cstream_out_size(const zxc_cstream* cs);

/* ===== Decompression ================================================= */

/**
 * @brief Creates a push decompression stream.
 *
 * @p opts is copied into the context. Only @c checksum_enabled is honoured: it
 * decides whether per-block and global checksums are verified when present.
 * Dictionary options are rejected outright (the push-stream format carries no
 * dict_id), so @c dict / @c dict_size / @c dict_huf fail creation rather than
 * being silently ignored.
 *
 * @param[in] opts  Decompression options, or @c NULL for defaults.
 * @return Context to release with @ref zxc_dstream_free, or @c NULL on
 *         allocation failure or dictionary options in @p opts.
 */
ZXC_EXPORT zxc_dstream* zxc_dstream_create(const zxc_decompress_opts_t* opts);

/**
 * @brief Releases a decompression stream.  Safe to call with @c NULL.
 *
 * @param[in] ds  Stream returned by @ref zxc_dstream_create.
 */
ZXC_EXPORT void zxc_dstream_free(zxc_dstream* ds);

/**
 * @brief Pushes compressed input and drains decompressed output.
 *
 * Drives a parser state machine: file header -> per-block (header + payload +
 * optional checksum) -> EOF block -> optional SEK block -> file footer. Each
 * call goes as far as @p in and @p out allow.
 *
 * @par End of stream
 * Validating the file footer puts the stream in DONE state; later calls return
 * @c 0 and produce nothing, even with bytes left in @p in. Those trailing bytes
 * are ignored, and @c in->pos tells the caller how much real data was consumed.
 *
 * @par Errors
 * Sticky: once a negative code comes back, every later call returns it too.
 *
 * @param[in,out] ds   Decompression stream.
 * @param[in,out] out  Output descriptor; @c pos advanced by produced bytes.
 * @param[in,out] in   Input descriptor;  @c pos advanced by consumed bytes.
 *
 * @return @c >0 decompressed bytes written into @p out by this call;
 *         @c 0 stream complete (DONE), or no progress possible and more input
 *         is needed;
 *         @c <0 a @ref zxc_error_t code.
 */
ZXC_EXPORT int64_t zxc_dstream_decompress(zxc_dstream* ds, zxc_outbuf_t* out, zxc_inbuf_t* in);

/**
 * @brief Reports whether the decoder has fully consumed a valid stream.
 *
 * True only once the parser has reached the file footer **and** validated it.
 * That is how a caller done feeding input detects truncation: if
 * @ref zxc_dstream_decompress returns @c 0 with no output and this returns
 * @c 0, the input ended early.
 *
 * @param[in] ds  Decompression stream.
 * @return @c 1 if DONE, @c 0 otherwise (including errored).
 */
ZXC_EXPORT int zxc_dstream_finished(const zxc_dstream* ds);

/**
 * @brief Suggested input chunk size for the decompressor.
 *
 * @param[in] ds  Decompression stream.
 * @return Suggested @c in_buf capacity in bytes, or 0 if @p ds is @c NULL.
 */
ZXC_EXPORT size_t zxc_dstream_in_size(const zxc_dstream* ds);

/**
 * @brief Suggested output chunk size for the decompressor.
 *
 * Sized to hold at least one full decompressed block.
 *
 * @param[in] ds  Decompression stream.
 * @return Suggested @c out_buf capacity in bytes, or 0 if @p ds is @c NULL.
 */
ZXC_EXPORT size_t zxc_dstream_out_size(const zxc_dstream* ds);

/** @} */ /* end of pstream */

#ifdef __cplusplus
}
#endif

#endif /* ZXC_PSTREAM_H */

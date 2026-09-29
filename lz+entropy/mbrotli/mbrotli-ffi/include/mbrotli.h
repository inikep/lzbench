/*
 * mbrotli C ABI: one-shot Brotli compression and decompression.
 *
 * Link against libmbrotli_ffi (static or shared) built from the mbrotli-ffi
 * crate. Every function is thread-safe and keeps no state between calls.
 */
#ifndef MBROTLI_H
#define MBROTLI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    /* Success; *output_len holds the number of bytes written. */
    MBROTLI_OK = 0,
    /* Null output_len, null buffer with a non-zero length, a length above
     * PTRDIFF_MAX, overlapping arguments, or quality/lgwin out of range. */
    MBROTLI_INVALID_PARAMETER = 1,
    /* The result does not fit in the output buffer. */
    MBROTLI_OUTPUT_TOO_SMALL = 2,
    /* Corrupt, truncated or trailing input, allocation failure, or any other
     * failure. */
    MBROTLI_ERROR = 3
} mbrotli_result;

/*
 * Compresses input into output as one complete Brotli stream.
 *
 * quality: 0..=11 (11 is densest). lgwin: window bits, 10..=24 (22 default).
 * On entry *output_len is the capacity of output; on return it is the number
 * of bytes written on MBROTLI_OK and 0 otherwise (a null or misaligned
 * output_len is left untouched). mbrotli_compress_bound(input_len) bytes are
 * always enough. input may be NULL when input_len is 0.
 *
 * The output is byte-identical to BrotliEncoderCompress(quality, lgwin,
 * BROTLI_MODE_GENERIC, ...), including its fallback to an uncompressed stream
 * when compression would exceed mbrotli_compress_bound(input_len).
 */
mbrotli_result mbrotli_compress(
    const uint8_t *input,
    size_t input_len,
    uint8_t *output,
    size_t *output_len,
    int quality,
    int lgwin
);

/*
 * Decompresses exactly one complete Brotli stream from input into output.
 *
 * On entry *output_len is the capacity of output; on return it is the number
 * of bytes written on MBROTLI_OK and 0 otherwise. On MBROTLI_OUTPUT_TOO_SMALL
 * output holds as much of the data as fits; decoding stops there, so a stream
 * corrupt only past that point is reported this way too, and MBROTLI_ERROR
 * with a larger buffer. Bytes after the stream are an error.
 */
mbrotli_result mbrotli_decompress(
    const uint8_t *input,
    size_t input_len,
    uint8_t *output,
    size_t *output_len
);

/*
 * Largest output mbrotli_compress can produce for input_len bytes at any
 * quality and window, or 0 if that bound does not fit in size_t. Equal to
 * BrotliEncoderMaxCompressedSize(input_len).
 */
size_t mbrotli_compress_bound(size_t input_len);

#ifdef __cplusplus
}
#endif

#endif /* MBROTLI_H */

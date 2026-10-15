/*
 * mzstd C ABI: one-shot Zstandard compression and decompression.
 *
 * Link against libmzstd_ffi (static or shared) built from the mzstd-ffi
 * crate. Every function is thread-safe and keeps no state between calls.
 */
#ifndef MZSTD_H
#define MZSTD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    /* Success; *output_len holds the number of bytes written. */
    MZSTD_OK = 0,
    /* Null output_len, null buffer with a non-zero length, a length above
     * PTRDIFF_MAX, overlapping arguments, or a level out of range. */
    MZSTD_INVALID_PARAMETER = 1,
    /* The result does not fit in the output buffer. */
    MZSTD_OUTPUT_TOO_SMALL = 2,
    /* Corrupt or truncated input, a checksum or content size mismatch, a
     * frame that needs a dictionary, allocation failure, or any other
     * failure. */
    MZSTD_ERROR = 3
} mzstd_result;

/*
 * Compresses input into output as one Zstandard frame.
 *
 * level: -131072..=22; 0 selects the default, 3. Unlike ZSTD_compress, a
 * level out of range is rejected rather than clamped.
 * On entry *output_len is the capacity of output; on return it is the number
 * of bytes written on MZSTD_OK and 0 otherwise (a null or misaligned
 * output_len is left untouched). mzstd_compress_bound(input_len) bytes are
 * always enough. input may be NULL when input_len is 0.
 *
 * The frame is byte-identical to ZSTD_compress(level) from zstd 1.5.7, and
 * a smaller output fails exactly when ZSTD_compress fails: at least 18 bytes
 * (the largest frame header) and some slack past the frame are needed.
 */
mzstd_result mzstd_compress(const uint8_t *input, size_t input_len,
                            uint8_t *output, size_t *output_len,
                            int level);

/*
 * Decompresses every frame in input into output, skipping skippable frames.
 *
 * On entry *output_len is the capacity of output; on return it is the number
 * of bytes written on MZSTD_OK and 0 otherwise. An empty input decodes to
 * nothing. Windows up to 2 GiB are accepted; output is zeroed before
 * decoding.
 */
mzstd_result mzstd_decompress(const uint8_t *input, size_t input_len,
                              uint8_t *output, size_t *output_len);

/*
 * The largest frame mzstd_compress writes for input_len bytes, equal to
 * ZSTD_compressBound; 0 when it does not fit in a size_t.
 */
size_t mzstd_compress_bound(size_t input_len);

#ifdef __cplusplus
}
#endif

#endif /* MZSTD_H */

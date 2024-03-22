#ifndef TAMP_DECOMPRESSOR_H
#define TAMP_DECOMPRESSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "common.h"

/* Externally, do not directly edit ANY of these attributes */
typedef struct {
    unsigned char *window;
    uint32_t bit_buffer;

    /* Conf attributes */
    uint32_t conf_window:4;   // number of window bits
    uint32_t conf_literal:4;  // number of literal bits
    //uint32_t conf_use_custom_dictionary:1;  // Not used past initialization.

    uint32_t bit_buffer_pos:6;
    uint32_t min_pattern_size:2;
    uint32_t window_pos:15;
    uint32_t configured:1;  // Whether or not conf has been properly set

    uint32_t skip_bytes:4;  // Skip this many decompressed bytes (from previous output-buffer-limited decompression).
} TampDecompressor;

/**
 * @brief Read tamp header and populate configuration.
 *
 * Don't invoke if setting conf to NULL in tamp_decompressor_init.
 *
 * @param[out] conf Configuration read from header
 * @param[in] data Tamp compressed data stream.
 */
tamp_res tamp_decompressor_read_header(TampConf *conf, const unsigned char *input, size_t input_size, size_t *input_consumed_size);

/**
 * @brief Initialize decompressor object.
 *
 *
 *
 * @param[in,out] TampDecompressor object to perform decompression with.
 * @param[in] conf Compressor configuration. Set to NULL to perform an implicit header read.
 * @param[in] window Pre-allocated window buffer. Size must agree with conf->window.
 *                   If conf.use_custom_dictionary is true, then the window must be
 *                   externally initialized and be at least as big as conf->window.
 */
tamp_res tamp_decompressor_init(TampDecompressor *decompressor, const TampConf *conf, unsigned char *window);

/**
 * @brief Decompress an input stream of data.
 *
 * Input data is **not** guaranteed to be consumed.  Imagine if a 6-byte sequence has been encoded, and
 * tamp_decompressor_decompress is called multiple times with a 2-byte output buffer:
 *
 *     1.  On the 1st call, a few input bytes may be consumed, filling the internal input buffer.
 *         The first 2 bytes of the 6-byte output sequence are returned.
 *         The internal input buffer remains full.
 *     2.  On the 2nd call, no input bytes are consumed since the internal input buffer is still full.
 *         The {3, 4} bytes of the 6-byte output sequence are returned.
 *         The internal input buffer remains full.
 *     3.  On the 3rd call, no input bytes are consumed since the internal input buffer is still full.
 *         The {5, 6} bytes of the 6-byte output sequence are returned.
 *         The input buffer is no longer full since this sequence has now been fully decoded.
 *     4.  On the 4th call, more input bytes are consumed, potentially filling the internal input buffer.
 *         It is not strictly necessary for the internal input buffer to be full to further decode the output.
 *         There simply has to be enough to decode a token/literal.
 *         If there is not enough bits in the internal input buffer, then TAMP_INPUT_EXHAUSTED will be returned.
 *
 * @param[in,out] TampDecompressor object to perform decompression with.
 * @param[out] output Pointer to a pre-allocated buffer to hold the output decompressed data.
 * @param[in] output_size Size of the pre-allocated buffer. Will decompress up-to this many bytes.
 * @param[out] output_written_size Number of bytes written to output. May be NULL.
 * @param[in] input Pointer to the compressed input data.
 * @param[in] input_size Number of bytes in input data.
 * @param[out] input_consumed_size Number of bytes of input data consumed. May be NULL.
 *
 * @return Tamp Status Code. In cases of success, will return TAMP_INPUT_EXHAUSTED or TAMP_OUTPUT_FULL, in lieu of TAMP_OK.
 */
tamp_res tamp_decompressor_decompress(
        TampDecompressor *decompressor,
        unsigned char *output,
        size_t output_size,
        size_t *output_written_size,
        const unsigned char *input,
        size_t input_size,
        size_t *input_consumed_size
        );

#ifdef __cplusplus
}
#endif

#endif

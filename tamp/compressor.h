#ifndef TAMP_COMPRESSOR_H
#define TAMP_COMPRESSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "common.h"

/* Externally, do not directly edit ANY of these attributes */
typedef struct TampCompressor{
    /* nicely aligned attributes */
    unsigned char *window;
    unsigned char input[16];
    uint32_t bit_buffer;

    /* Conf attributes */
    uint32_t conf_window:4;   // number of window bits
    uint32_t conf_literal:4;  // number of literal bits
    uint32_t conf_use_custom_dictionary:1;  // Use a custom initialized dictionary.

    /* Other small attributes */
    uint32_t window_pos:15;
    uint32_t bit_buffer_pos:6;
    uint32_t min_pattern_size:2;

    uint32_t input_size:5;
    uint32_t input_pos:4;
} TampCompressor;


/**
 * @brief Initialize Tamp Compressor object.
 *
 * @param[out] compressor Object to initialize.
 * @param[in] conf Compressor configuration. Set to NULL for default (window=10, literal=8).
 * @param[in] window Pre-allocated window buffer. Size must agree with conf->window.
 *                   If conf.use_custom_dictionary is true, then the window must be
 *                   externally initialized.
 *
 * @return Tamp Status Code. Returns TAMP_INVALID_CONF if an invalid conf state is provided.
 */
tamp_res tamp_compressor_init(TampCompressor *compressor, const TampConf *conf, unsigned char *window);

/**
 * @brief Sink data into input buffer.
 *
 * @param[in,out] compressor TampCompressor object to perform compression with.
 * @param[in] input Pointer to the input data to be sinked into compressor.
 * @param[in] input_size Size of input.
 * @param[out] consumed_size Number of bytes of input consumed. May be NULL.
 */
void tamp_compressor_sink(
        TampCompressor *compressor,
        const unsigned char *input,
        size_t input_size,
        size_t *consumed_size
        );

/**
 * @brief Run a single compression iteration on the internal input buffer.
 *
 * The most that will ever be written to output in a single invocation is:
 *
 *     (1 + 8 + WINDOW_BITS + 7) // 8
 *
 * or more simply:
 *
 *     (16 + WINDOW_BITS) // 8
 *
 * where // represents floor-division.  Explanation:
 *      * 1 - is_literal bit
 *      * 8 - maximum huffman code length
 *      * WINDOW_BITS - The number of bits to represent the match index. By default, 10.
 *      * 7 - The internal bit buffer may have up to 7 bits from a previous invocation.
 *      * // 8 - Floor divide by 8 to get bytes; the upto remaining 7 bits remain in the internal output bit buffer.
 *
 * A reasonable 4-byte output buffer should be able to handle any compressor configuration.
 *
 * @param[in,out] compressor TampCompressor object to perform compression with.
 * @param[out] output Pointer to a pre-allocated buffer to hold the output compressed data.
 * @param[in] output_size Size of the pre-allocated buffer. Will compress up-to this many bytes.
 * @param[out] output_written_size Number of bytes written to output. May be NULL.
 *
 * @return Tamp Status Code. Can return TAMP_OK, TAMP_OUTPUT_FULL, or TAMP_EXCESS_BITS.
 */
tamp_res tamp_compressor_compress_poll(
        TampCompressor *compressor,
        unsigned char *output,
        size_t output_size,
        size_t *output_written_size
        );

/**
 * @brief Completely flush the internal bit buffer. Makes output "complete".
 *
 * At a maximum, the compressor will have 16 bytes in it's input buffer.
 * The worst-case compression scenario would use `literal + 1` bits per input byte.
 * This means that for the typical `literal=8` scenario, the output buffer size
 * should be 18 bytes long. If `write_token=true`, then the output buffer size should
 * be 20 bytes long to absolutely guarantee a complete flush.
 *
 * @param[in,out] compressor TampCompressor object to flush.
 * @param[out] output Pointer to a pre-allocated buffer to hold the output compressed data.
 * @param[in] output_size Size of the pre-allocated buffer. Will compress up-to this many bytes.
 * @param[out] output_written_size Number of bytes written to output. May be NULL.
 * @param[in] write_token Write the FLUSH token, if appropriate. Set to true if you want to continue using the compressor. Set to false if you are done with the compressor, usually at the end of a stream.
 *
 * @return Tamp Status Code. Can return TAMP_OK, or TAMP_OUTPUT_FULL.
 */
tamp_res tamp_compressor_flush(
                TampCompressor *compressor,
                unsigned char *output,
                size_t output_size,
                size_t *output_written_size,
                bool write_token
                );

/**
 * @brief Compress a chunk of data until input or output buffer is exhausted.
 *
 * @param[in,out] compressor TampCompressor object to perform compression with.
 * @param[out] output Pointer to a pre-allocated buffer to hold the output compressed data.
 * @param[in] output_size Size of the pre-allocated buffer. Will compress up-to this many bytes.
 * @param[out] output_written_size Number of bytes written to output. May be NULL.
 * @param[in] input Pointer to the input data to be compressed.
 * @param[in] input_size Number of bytes in input data.
 * @param[out] input_consumed_size Number of bytes of input data consumed. May be NULL.
 *
 * @return Tamp Status Code. Can return TAMP_OK, TAMP_OUTPUT_FULL, or TAMP_EXCESS_BITS.
 */
tamp_res tamp_compressor_compress(
        TampCompressor *compressor,
        unsigned char *output,
        size_t output_size,
        size_t *output_written_size,
        const unsigned char *input,
        size_t input_size,
        size_t *input_consumed_size
        );

/**
 * @brief Compress a chunk of data until input or output buffer is exhausted.
 *
 * If the output buffer is full, buffer flushing will not be performed and TAMP_OUTPUT_FULL will be returned.
 * May be called again with an appropriately updated pointers and sizes.
 *
 * @param[in,out] compressor TampCompressor object to perform compression with.
 * @param[out] output Pointer to a pre-allocated buffer to hold the output compressed data.
 * @param[in] output_size Size of the pre-allocated buffer. Will compress up-to this many bytes.
 * @param[out] output_written_size Number of bytes written to output. May be NULL.
 * @param[in] input Pointer to the input data to be compressed.
 * @param[in] input_size Number of bytes in input data.
 * @param[out] input_consumed_size Number of bytes of input data consumed. May be NULL.
 *
 * @return Tamp Status Code. Can return TAMP_OK, TAMP_OUTPUT_FULL, or TAMP_EXCESS_BITS.
 */
tamp_res tamp_compressor_compress_and_flush(
        TampCompressor *compressor,
        unsigned char *output,
        size_t output_size,
        size_t *output_written_size,
        const unsigned char *input,
        size_t input_size,
        size_t *input_consumed_size,
        bool write_token
        );


#ifdef __cplusplus
}
#endif

#endif

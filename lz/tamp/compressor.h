#ifndef TAMP_COMPRESSOR_H
#define TAMP_COMPRESSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "common.h"

/* Externally, do not directly edit ANY of these attributes.
 * Fields are ordered by access frequency for cache efficiency.
 */
typedef struct TampCompressor {
#if TAMP_ESP32  // Avoid bitfields for speed.
    /* HOT: accessed every iteration of the compression loop */
    unsigned char *window;    // Pointer to window buffer
    uint32_t bit_buffer;      // Bit buffer for output (32 bits)
    uint32_t window_pos;      // Current position in window (15 bits used)
    uint32_t bit_buffer_pos;  // Bits currently in bit_buffer (6 bits used)
    uint32_t input_size;      // Bytes in input buffer (5 bits used; 0-16)
    uint32_t input_pos;       // Current position in input buffer (4 bits used; 0-15)
    unsigned char input[16];  // Input ring buffer

    /* WARM: read frequently, often cached in locals */
    uint8_t min_pattern_size;  // Minimum pattern size (2 bits used; 2 or 3)
    TampConf conf;
#else   // Use bitfields for reduced memory-usage
    /* HOT: accessed every iteration of the compression loop */
    unsigned char *window;    // Pointer to window buffer
    uint32_t bit_buffer;      // Bit buffer for output (32 bits)
    uint16_t window_pos;      // Current position in window (15 bits used)
    uint8_t bit_buffer_pos;   // Bits currently in bit_buffer (6 bits used)
    uint8_t input_size;       // Bytes in input buffer (5 bits used; 0-16)
    uint8_t input_pos;        // Current position in input buffer (4 bits used; 0-15)
    unsigned char input[16];  // Input ring buffer

    /* WARM: read frequently, often cached in locals */
    uint8_t min_pattern_size;  // Minimum pattern size (2 or 3)
    TampConf conf;
#endif  // TAMP_ESP32

    /* Fields interleaved to avoid internal padding when both LAZY_MATCHING and EXTENDED_COMPRESS enabled */
#if TAMP_LAZY_MATCHING
    int16_t cached_match_index;  // Lazy matching cache
#endif
#if TAMP_EXTENDED_COMPRESS
    uint16_t extended_match_position;  // Window position for extended match
#endif
#if TAMP_LAZY_MATCHING
    uint8_t cached_match_size;
#endif
#if TAMP_EXTENDED_COMPRESS
    uint8_t rle_count;             // Current RLE run length (max 241)
    uint8_t extended_match_count;  // Current extended match size (max ~134)
#endif
} TampCompressor;

/**
 * @brief Initialize Tamp Compressor object.
 *
 * When conf->append is true, writes a FLUSH token to the internal bit buffer
 * instead of a header, enabling append-after-reboot on streams with
 * dictionary_reset enabled. Requires dictionary_reset=true and
 * use_custom_dictionary=false.
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
 * Copies bytes from `input` to the internal input buffer until the internal
 * input buffer is full, or the supplied input is exhausted.
 *
 * Somewhere between 0 and 16 bytes will be copied from the input.
 *
 * This is a computationally cheap/fast function.
 *
 * @param[in,out] compressor TampCompressor object to perform compression with.
 * @param[in] input Pointer to the input data to be sinked into compressor.
 * @param[in] input_size Size of input.
 * @param[out] consumed_size Number of bytes of input consumed. May be NULL.
 */
void tamp_compressor_sink(TampCompressor *compressor, const unsigned char *input, size_t input_size,
                          size_t *consumed_size);

/**
 * @brief Run a single compression iteration on the internal input buffer.
 *
 * This is a computationally intensive function.
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
 *      * 7 - The internal bit buffer may have up to 7 bits from a previous invocation. See NOTE
 * below.
 *      * // 8 - Floor divide by 8 to get bytes; the upto remaining 7 bits remain in the internal
 * output bit buffer.
 *
 * NOTE: Unintuitively, tamp_compressor_poll partially flushes (flushing multiples of 8-bits) the
 * internal output bit buffer at the **beginning** of the function call (not the end). This means
 * that a **previous** tamp_compressor_poll call may have placed up to (16 + WINDOW_BITS) bits in
 * the internal output bit buffer. The partial flush at the beginning of tamp_compressor_poll clears
 * as many whole-bytes as possible from this buffer. After this flush, there remains up to 7 bits,
 * to which the current call's compressed token/literal is added to.
 *
 * A 3-byte output buffer should be able to handle any compressor configuration.
 *
 * @param[in,out] compressor TampCompressor object to perform compression with.
 * @param[out] output Pointer to a pre-allocated buffer to hold the output compressed data.
 * @param[in] output_size Size of the pre-allocated output buffer.
 * @param[out] output_written_size Number of bytes written to output. May be NULL.
 *
 * @return Tamp Status Code. Can return TAMP_OK, TAMP_OUTPUT_FULL, or TAMP_EXCESS_BITS.
 */
TAMP_NOINLINE tamp_res tamp_compressor_poll(TampCompressor *compressor, unsigned char *output, size_t output_size,
                                            size_t *output_written_size);
// backwards compatibility for old naming
#define tamp_compressor_compress_poll tamp_compressor_poll

/**
 * @brief Check if the compressor's input buffer is full.
 *
 * @param[in] compressor TampCompressor object to check.
 *
 * @return true if the compressor is full, false otherwise.
 */
bool tamp_compressor_full(const TampCompressor *compressor);

/**
 * @brief Completely flush the internal bit buffer. Makes output "complete".
 *
 * The following table contains the most number of bytes that could be flushed in a worst-case
 * scenario:
 *
 * +---------------------+--------------------+-------------------------------------------+------------------------------------------+
 * | Literal Size (Bits) | Window Size (Bits) | Max Output Size write_token=false (Bytes) | Max
 * Output Size write_token=true (Bytes) |
 * +=====================+====================+===========================================+==========================================+
 * | 5                   | 8                  | 15                                        | 16 |
 * +---------------------+--------------------+-------------------------------------------+------------------------------------------+
 * | 5                   | 9-15               | 16                                        | 17 |
 * +---------------------+--------------------+-------------------------------------------+------------------------------------------+
 * | 6                   | 8                  | 17                                        | 18 |
 * +---------------------+--------------------+-------------------------------------------+------------------------------------------+
 * | 6                   | 9-15               | 18                                        | 19 |
 * +---------------------+--------------------+-------------------------------------------+------------------------------------------+
 * | 7                   | 8                  | 19                                        | 20 |
 * +---------------------+--------------------+-------------------------------------------+------------------------------------------+
 * | 7                   | 9-15               | 20                                        | 21 |
 * +---------------------+--------------------+-------------------------------------------+------------------------------------------+
 * | 8                   | 8                  | 21                                        | 22 |
 * +---------------------+--------------------+-------------------------------------------+------------------------------------------+
 * | 8                   | 9-15               | 22                                        | 23 |
 * +---------------------+--------------------+-------------------------------------------+------------------------------------------+
 *
 * @param[in,out] compressor TampCompressor object to flush.
 * @param[out] output Pointer to a pre-allocated buffer to hold the output compressed data.
 * @param[in] output_size Size of the pre-allocated output buffer.
 * @param[out] output_written_size Number of bytes written to output. May be NULL.
 * @param[in] write_token Write the FLUSH token, if appropriate. Set to true if you want to continue
 * using the compressor. Set to false if you are done with the compressor, usually at the end of
 * compression.
 *
 * @return Tamp Status Code. Can return TAMP_OK, or TAMP_OUTPUT_FULL.
 */
tamp_res tamp_compressor_flush(TampCompressor *compressor, unsigned char *output, size_t output_size,
                               size_t *output_written_size, bool write_token);

/**
 * @brief Reset the compressor dictionary and internal state.
 *
 * Writes a double-FLUSH token sequence to signal dictionary re-initialization
 * to the decompressor, then resets the window and all internal compression state.
 *
 * The compressor continues to use the same configuration (window, literal, etc.)
 * but starts fresh with a re-initialized dictionary.
 *
 * The compressor must have been initialized with conf->dictionary_reset set.
 * This causes the header to include the more_header flag, which old decompressors
 * will reject rather than silently producing corrupt output.
 *
 * @param[in,out] compressor TampCompressor object to reset.
 * @param[out] output Pointer to a pre-allocated buffer to hold output data.
 * @param[in] output_size Size of the pre-allocated output buffer.
 * @param[out] output_written_size Number of bytes written to output. May be NULL.
 *
 * @return Tamp Status Code. Can return TAMP_OK, TAMP_OUTPUT_FULL, or
 *         TAMP_INVALID_CONF if dictionary_reset was not set at init.
 */
tamp_res tamp_compressor_reset_dictionary(TampCompressor *compressor, unsigned char *output, size_t output_size,
                                          size_t *output_written_size);

/**
 * Callback-variant of tamp_compressor_compress.
 *
 * @param[in] callback User-provided function to be called every compression-cycle.
 *                     Receives (user_data, input_bytes_consumed, total_input_size).
 * @param[in,out] user_data Passed along to callback.
 */
tamp_res tamp_compressor_compress_cb(TampCompressor *compressor, unsigned char *output, size_t output_size,
                                     size_t *output_written_size, const unsigned char *input, size_t input_size,
                                     size_t *input_consumed_size, tamp_callback_t callback, void *user_data);

/**
 * @brief Compress a chunk of data until input or output buffer is exhausted.
 *
 * @param[in,out] compressor TampCompressor object to perform compression with.
 * @param[out] output Pointer to a pre-allocated buffer to hold the output compressed data.
 * @param[in] output_size Size of the pre-allocated output buffer.
 * @param[out] output_written_size Number of bytes written to output. May be NULL.
 * @param[in] input Pointer to the input data to be compressed.
 * @param[in] input_size Number of bytes in input data.
 * @param[out] input_consumed_size Number of bytes of input data consumed. May be NULL.
 *
 * @return Tamp Status Code. Can return TAMP_OK, TAMP_OUTPUT_FULL, or TAMP_EXCESS_BITS.
 */
TAMP_ALWAYS_INLINE tamp_res tamp_compressor_compress(TampCompressor *compressor, unsigned char *output,
                                                     size_t output_size, size_t *output_written_size,
                                                     const unsigned char *input, size_t input_size,
                                                     size_t *input_consumed_size) {
    return tamp_compressor_compress_cb(compressor, output, output_size, output_written_size, input, input_size,
                                       input_consumed_size, NULL, NULL);
}

/**
 * Callback-variant of tamp_compressor_compress_and_flush.
 *
 * @param[in] callback User-provided function to be called every compression-cycle.
 *                     A final callback is issued after flushing to signal completion.
 * @param[in,out] user_data Passed along to callback.
 */
tamp_res tamp_compressor_compress_and_flush_cb(TampCompressor *compressor, unsigned char *output, size_t output_size,
                                               size_t *output_written_size, const unsigned char *input,
                                               size_t input_size, size_t *input_consumed_size, bool write_token,
                                               tamp_callback_t callback, void *user_data);

/**
 * @brief Compress a chunk of data until input or output buffer is exhausted.
 *
 * If the output buffer is full, buffer flushing will not be performed and TAMP_OUTPUT_FULL will be
 * returned. May be called again with an appropriately updated pointers and sizes.
 *
 * @param[in,out] compressor TampCompressor object to perform compression with.
 * @param[out] output Pointer to a pre-allocated buffer to hold the output compressed data.
 * @param[in] output_size Size of the pre-allocated output buffer.
 * @param[out] output_written_size Number of bytes written to output. May be NULL.
 * @param[in] input Pointer to the input data to be compressed.
 * @param[in] input_size Number of bytes in input data.
 * @param[out] input_consumed_size Number of bytes of input data consumed. May be NULL.
 *
 * @return Tamp Status Code. Can return TAMP_OK, TAMP_OUTPUT_FULL, or TAMP_EXCESS_BITS.
 */
TAMP_ALWAYS_INLINE tamp_res tamp_compressor_compress_and_flush(TampCompressor *compressor, unsigned char *output,
                                                               size_t output_size, size_t *output_written_size,
                                                               const unsigned char *input, size_t input_size,
                                                               size_t *input_consumed_size, bool write_token) {
    return tamp_compressor_compress_and_flush_cb(compressor, output, output_size, output_written_size, input,
                                                 input_size, input_consumed_size, write_token, NULL, NULL);
}

#if TAMP_STREAM
/**
 * @brief Compress data from input source to output destination using callbacks.
 *
 * High-level function that reads from an input source, compresses the data,
 * and writes to an output destination using user-provided I/O callbacks.
 * Works with any I/O backend (stdio, littlefs, fatfs, UART, etc.).
 *
 * Uses an internal **stack-allocated** work buffer sized by TAMP_STREAM_WORK_BUFFER_SIZE
 * (default 32 bytes). For better compression performance, increase this via
 * compiler flag: -DTAMP_STREAM_WORK_BUFFER_SIZE=256
 *
 * Example with stdio:
 * @code
 * FILE *in = fopen("input.bin", "rb");
 * FILE *out = fopen("output.tamp", "wb");
 * unsigned char window[1024];
 * TampCompressor compressor;
 * tamp_compressor_init(&compressor, NULL, window);
 * tamp_compress_stream(
 *     &compressor,             // compressor: initialized compressor
 *     tamp_stream_stdio_read,  // read_cb: reads uncompressed input
 *     in,                      // read_handle: passed to read_cb
 *     tamp_stream_stdio_write, // write_cb: writes compressed output
 *     out,                     // write_handle: passed to write_cb
 *     NULL,                    // input_consumed_size: out, bytes read
 *     NULL,                    // output_written_size: out, bytes written
 *     NULL,                    // callback: progress callback
 *     NULL                     // user_data: passed to callback
 * );
 * fclose(in);
 * fclose(out);
 * @endcode
 *
 * @param[in,out] compressor Initialized TampCompressor (via tamp_compressor_init).
 * @param[in] read_cb Callback to read uncompressed input data.
 * @param[in] read_handle Opaque handle passed to read_cb (e.g., input FILE*).
 * @param[in] write_cb Callback to write compressed output data.
 * @param[in] write_handle Opaque handle passed to write_cb (e.g., output FILE*).
 * @param[out] input_consumed_size Total input bytes read. May be NULL.
 * @param[out] output_written_size Total compressed bytes written. May be NULL.
 * @param[in] callback Optional progress callback invoked periodically. May be NULL.
 *                     Receives (user_data, input_bytes_consumed, 0).
 * @param[in] user_data User data passed to progress callback.
 *
 * @return TAMP_OK on success, or an error code:
 *         - TAMP_READ_ERROR: read_cb returned a negative value
 *         - TAMP_WRITE_ERROR: write_cb returned a negative value or incomplete write
 *         - Other tamp_res error codes from compression
 */
tamp_res tamp_compress_stream(TampCompressor *compressor, tamp_read_t read_cb, void *read_handle, tamp_write_t write_cb,
                              void *write_handle, size_t *input_consumed_size, size_t *output_written_size,
                              tamp_callback_t callback, void *user_data);
#endif /* TAMP_STREAM */

#ifdef __cplusplus
}
#endif

#endif

#ifndef TAMP_DECOMPRESSOR_H
#define TAMP_DECOMPRESSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "common.h"

/* Externally, do not directly edit ANY of these attributes.
 * Fields are ordered by access frequency for cache efficiency.
 */
typedef struct {
    /* HOT: accessed every iteration of the decompression loop. */
    unsigned char *window;  // Pointer to window buffer
    uint32_t bit_buffer;    // Bit buffer for reading compressed data (32 bits)
    uint16_t window_pos;    // Current position in window (15 bits)

    /* Union allows single zero-check in main loop instead of two separate checks. */
#if TAMP_EXTENDED_DECOMPRESS
    union {
        struct {
            uint8_t bit_buffer_pos;  // Bits currently in bit_buffer (6 bits needed)
            uint8_t token_state;     // 0=none, 1=RLE, 2=ext match, 3=ext match fresh (2 bits used)
        };
        uint16_t pos_and_state;  // Combined for fast 16-bit zero-check
    };
#else
    union {
        uint8_t bit_buffer_pos;  // Bits currently in bit_buffer (6 bits needed)
        uint8_t pos_and_state;   // Alias for consistent access in main loop
    };
#endif
#if TAMP_EXTENDED_DECOMPRESS
    uint16_t pending_window_offset;  // Saved window_offset for extended match output-full resume
    uint16_t pending_match_size;     // Saved match_size for extended match resume
#endif

    /* WARM: read once at start of decompress, cached in locals */
    uint8_t conf_window : 4;            // Window bits from config
    uint8_t conf_literal : 4;           // Literal bits from config
    uint8_t min_pattern_size : 2;       // Minimum pattern size, 2 or 3
    uint8_t conf_extended : 1;          // Extended format enabled (from header)
    uint8_t conf_dictionary_reset : 1;  // Stream may contain double-FLUSH dictionary resets (from header byte 1 bit [0]
                                        // / more_header)

    /* COLD: rarely accessed (init or edge cases).
     * Bitfields save space; add new cold fields here. */
    union {
        uint8_t skip_bytes;           // After configured: output-buffer-limited resumption (v2 needs >4 bits)
        uint8_t stashed_header_byte;  // Before configured: first header byte when waiting for second
    };
    uint8_t window_bits_max : 4;    // Max window bits buffer can hold
    uint8_t configured : 1;         // Whether config has been set (authoritative "header complete" flag)
    uint8_t header_bytes_read : 2;  // Before configured: header bytes consumed so far
    uint8_t last_was_flush : 1;     // Previous token was FLUSH (for double-FLUSH dictionary reset detection)
} TampDecompressor;

/**
 * @brief Read tamp header and populate configuration.
 *
 * Don't invoke if setting conf to NULL in tamp_decompressor_init.
 *
 * @param[out] conf Configuration read from header
 * @param[in] data Tamp compressed data.
 */
tamp_res tamp_decompressor_read_header(TampConf *conf, const unsigned char *input, size_t input_size,
                                       size_t *input_consumed_size);

/**
 * @brief Initialize decompressor object.
 *
 * @param[in,out] decompressor TampDecompressor object to perform decompression with.
 * @param[in] conf Decompressor configuration. Set to NULL to perform an implicit header read.
 * @param[in] window Pre-allocated window buffer.
 * @param[in] window_bits Number of window bits the buffer can accommodate (8-15).
 *                        Buffer must be at least (1 << window_bits) bytes.
 *                        When conf is NULL (implicit header read), the header's window size
 *                        is validated against this value.
 *
 * @return TAMP_OK on success, TAMP_INVALID_CONF if window_bits is invalid or too small.
 */
tamp_res tamp_decompressor_init(TampDecompressor *decompressor, const TampConf *conf, unsigned char *window,
                                uint8_t window_bits);

/**
 * Callback-variant of tamp_decompressor_decompress.
 *
 * @param[in] callback User-provided function to be called every decompression-cycle.
 *                     Receives (user_data, input_bytes_consumed, total_input_size).
 * @param[in,out] user_data Passed along to callback.
 */
tamp_res tamp_decompressor_decompress_cb(TampDecompressor *decompressor, unsigned char *output, size_t output_size,
                                         size_t *output_written_size, const unsigned char *input, size_t input_size,
                                         size_t *input_consumed_size, tamp_callback_t callback, void *user_data);

/**
 * @brief Decompress input data into an output buffer.
 *
 * Input data is **not** guaranteed to be consumed.  Imagine if a 6-byte sequence has been encoded,
 * and tamp_decompressor_decompress is called multiple times with a 2-byte output buffer:
 *
 *     1.  On the 1st call, a few input bytes may be consumed, filling the internal input buffer.
 *         The first 2 bytes of the 6-byte output sequence are returned.
 *         The internal input buffer remains full.
 *     2.  On the 2nd call, no input bytes are consumed since the internal input buffer is still
 * full. The {3, 4} bytes of the 6-byte output sequence are returned. The internal input buffer
 * remains full.
 *     3.  On the 3rd call, no input bytes are consumed since the internal input buffer is still
 * full. The {5, 6} bytes of the 6-byte output sequence are returned. The input buffer is no longer
 * full since this sequence has now been fully decoded.
 *     4.  On the 4th call, more input bytes are consumed, potentially filling the internal input
 * buffer. It is not strictly necessary for the internal input buffer to be full to further decode
 * the output. There simply has to be enough to decode a token/literal. If there is not enough bits
 * in the internal input buffer, then TAMP_INPUT_EXHAUSTED will be returned.
 *
 * @param[in,out] TampDecompressor object to perform decompression with.
 * @param[out] output Pointer to a pre-allocated buffer to hold the output decompressed data.
 * @param[in] output_size Size of the pre-allocated buffer. Will decompress up-to this many bytes.
 * @param[out] output_written_size Number of bytes written to output. May be NULL.
 * @param[in] input Pointer to the compressed input data.
 * @param[in] input_size Number of bytes in input data.
 * @param[out] input_consumed_size Number of bytes of input data consumed. May be NULL.
 *
 * @return Tamp Status Code. In cases of success, will return TAMP_INPUT_EXHAUSTED or
 * TAMP_OUTPUT_FULL, in lieu of TAMP_OK.
 */
TAMP_ALWAYS_INLINE tamp_res tamp_decompressor_decompress(TampDecompressor *decompressor, unsigned char *output,
                                                         size_t output_size, size_t *output_written_size,
                                                         const unsigned char *input, size_t input_size,
                                                         size_t *input_consumed_size) {
    return tamp_decompressor_decompress_cb(decompressor, output, output_size, output_written_size, input, input_size,
                                           input_consumed_size, NULL, NULL);
}

#if TAMP_STREAM
/**
 * @brief Decompress data from input source to output destination using callbacks.
 *
 * High-level function that reads compressed data, decompresses it,
 * and writes to an output destination using user-provided I/O callbacks.
 * Works with any I/O backend (stdio, littlefs, fatfs, UART, etc.).
 *
 * Uses an internal **stack-allocated** work buffer sized by TAMP_STREAM_WORK_BUFFER_SIZE
 * (default 32 bytes). For better decompression performance, increase this via
 * compiler flag: -DTAMP_STREAM_WORK_BUFFER_SIZE=256
 *
 * Example with littlefs:
 * @code
 * lfs_file_t in, out;
 * lfs_file_open(&lfs, &in, "data.tamp", LFS_O_RDONLY);
 * lfs_file_open(&lfs, &out, "data.bin", LFS_O_WRONLY | LFS_O_CREAT);
 * unsigned char window[1024];
 * TampDecompressor decompressor;
 * tamp_decompressor_init(&decompressor, NULL, window, 10);
 * tamp_decompress_stream(
 *     &decompressor,         // decompressor: initialized decompressor
 *     tamp_stream_lfs_read,  // read_cb: reads compressed input
 *     &in,                   // read_handle: passed to read_cb
 *     tamp_stream_lfs_write, // write_cb: writes decompressed output
 *     &out,                  // write_handle: passed to write_cb
 *     NULL,                  // input_consumed_size: out, bytes read
 *     NULL,                  // output_written_size: out, bytes written
 *     NULL,                  // callback: progress callback
 *     NULL                   // user_data: passed to callback
 * );
 * lfs_file_close(&lfs, &in);
 * lfs_file_close(&lfs, &out);
 * @endcode
 *
 * @param[in,out] decompressor Initialized TampDecompressor (via tamp_decompressor_init).
 *                             When initialized with conf=NULL, the header will be read
 *                             from the stream automatically.
 * @param[in] read_cb Callback to read compressed input data.
 * @param[in] read_handle Opaque handle passed to read_cb.
 * @param[in] write_cb Callback to write decompressed output data.
 * @param[in] write_handle Opaque handle passed to write_cb.
 * @param[out] input_consumed_size Total compressed bytes read. May be NULL.
 * @param[out] output_written_size Total decompressed bytes written. May be NULL.
 * @param[in] callback Optional progress callback invoked periodically. May be NULL.
 *                     Receives (user_data, input_bytes_consumed, 0).
 * @param[in] user_data User data passed to progress callback.
 *
 * @return TAMP_OK on success (stream fully decompressed), or an error code:
 *         - TAMP_READ_ERROR: read_cb returned a negative value
 *         - TAMP_WRITE_ERROR: write_cb returned a negative value or incomplete write
 *         - TAMP_OOB: Corrupt/malicious data attempted out-of-bounds access
 *         - Other tamp_res error codes from decompression
 */
tamp_res tamp_decompress_stream(TampDecompressor *decompressor, tamp_read_t read_cb, void *read_handle,
                                tamp_write_t write_cb, void *write_handle, size_t *input_consumed_size,
                                size_t *output_written_size, tamp_callback_t callback, void *user_data);
#endif /* TAMP_STREAM */

#ifdef __cplusplus
}
#endif

#endif

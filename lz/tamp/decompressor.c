#include "decompressor.h"

#include "common.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define FLUSH 14

#if TAMP_EXTENDED_DECOMPRESS
/* Token state for extended decode suspend/resume (2 bits).
 * TOKEN_RLE and TOKEN_EXT_MATCH_FRESH are arranged so that:
 *     token_state = match_size - (TAMP_RLE_SYMBOL - 1)
 * maps TAMP_RLE_SYMBOL (12) -> 1 and TAMP_EXTENDED_MATCH_SYMBOL (13) -> 2.
 */
#define TOKEN_NONE 0
#define TOKEN_RLE 1
#define TOKEN_EXT_MATCH_FRESH 2
#define TOKEN_EXT_MATCH 3 /* Resume: have match_size, need window_offset */
#endif

/**
 * Huffman lookup table indexed by 7 bits (after first "1" bit consumed).
 * Upper 4 bits = additional bits to consume, lower 4 bits = symbol (14 = FLUSH).
 *
 * Note: A 64-byte table with special-cased symbol 1 was tried but was ~10% slower
 * and only saved 8 bytes in final firmware due to added branch logic.
 */
static const uint8_t HUFFMAN_TABLE[128] = {
    50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,  50,  85,  85,  85, 85, 122, 123, 104, 104, 86, 86,
    86, 86, 93, 93, 93, 93, 68, 68, 68, 68, 68, 68, 68, 68, 105, 105, 124, 126, 87, 87, 87,  87,  51,  51,  51, 51,
    51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 17, 17, 17,  17,  17,  17,  17, 17, 17,  17,  17,  17,  17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,  17,  17,  17,  17, 17, 17,  17,  17,  17,  17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,  17,  17,  17,  17, 17, 17,  17,  17,  17};

/**
 * @brief Decode huffman symbol + optional trailing bits from bit buffer.
 *
 * Modifies bit_buffer and bit_buffer_pos in place. Caller is responsible
 * for committing to decompressor state if needed.
 *
 * @param bit_buffer Pointer to bit buffer (modified in place)
 * @param bit_buffer_pos Pointer to bit position (modified in place)
 * @param trailing_bits Number of trailing bits to read (0, 3, or 4)
 * @param result Output: (huffman << trailing_bits) + trailing (max 239 for trailing_bits=4)
 * @return TAMP_OK on success, TAMP_INPUT_EXHAUSTED if more bits needed
 */
static tamp_res decode_huffman(uint32_t* bit_buffer, uint8_t* bit_buffer_pos, uint8_t trailing_bits, uint8_t* result) {
    /* Need at least 1 bit for huffman, plus trailing bits */
    if (TAMP_UNLIKELY(*bit_buffer_pos < 1 + trailing_bits)) return TAMP_INPUT_EXHAUSTED;

    /* Decode huffman symbol */
    int8_t huffman_value;
    (*bit_buffer_pos)--;
    if (TAMP_LIKELY((*bit_buffer >> 31) == 0)) {
        /* Symbol 0: code "0" */
        *bit_buffer <<= 1;
        huffman_value = 0;
    } else {
        /* All other symbols: use 128-entry table indexed by next 7 bits */
        *bit_buffer <<= 1;
        uint8_t code = HUFFMAN_TABLE[*bit_buffer >> (32 - 7)];
        uint8_t bit_len = code >> 4;
        if (TAMP_UNLIKELY(*bit_buffer_pos < bit_len + trailing_bits)) return TAMP_INPUT_EXHAUSTED;
        *bit_buffer <<= bit_len;
        *bit_buffer_pos -= bit_len;
        huffman_value = code & 0xF;
    }

    /* Read trailing bits (skip if trailing_bits==0 to avoid undefined shift) */
    if (trailing_bits) {
        uint8_t trailing = *bit_buffer >> (32 - trailing_bits);
        *bit_buffer <<= trailing_bits;
        *bit_buffer_pos -= trailing_bits;
        *result = (huffman_value << trailing_bits) + trailing;
    } else {
        *result = huffman_value;
    }

    return TAMP_OK;
}

#if TAMP_EXTENDED_DECOMPRESS

/**
 * @brief Decode RLE token and write repeated bytes to output.
 *
 * RLE format: huffman(count_high) + trailing_bits(count_low)
 * rle_count = (count_high << 4) + count_low + 2
 */
static tamp_res decode_rle(TampDecompressor* d, unsigned char** output, const unsigned char* output_end,
                           size_t* output_written_size) {
    uint8_t rle_count; /* max 241: (14 << 4) + 15 + 2 */
    uint8_t skip = d->skip_bytes;

    if (skip > 0) {
        /* Resume from output-full: rle_count saved in pending_window_offset */
        rle_count = d->pending_window_offset;
    } else {
        /* Fresh decode */
        uint32_t bit_buffer = d->bit_buffer;
        uint8_t bit_buffer_pos = d->bit_buffer_pos;
        uint8_t raw;
        tamp_res res = decode_huffman(&bit_buffer, &bit_buffer_pos, TAMP_LEADING_RLE_BITS, &raw);
        if (res != TAMP_OK) return res;
        d->bit_buffer = bit_buffer;
        d->bit_buffer_pos = bit_buffer_pos;
        rle_count = raw + 2;
    }

    /* Get the byte to repeat (last written byte) */
    uint16_t prev_pos = (d->window_pos - 1) & ((1u << d->conf_window) - 1);
    uint8_t symbol = d->window[prev_pos];

    /* Calculate how many to write this call */
    uint8_t remaining_count = rle_count - skip;
    size_t output_space = output_end - *output;
    uint8_t to_write;

    if (TAMP_UNLIKELY(remaining_count > output_space)) {
        /* Partial write - save state for resume */
        to_write = output_space;
        d->skip_bytes = skip + to_write;
        d->token_state = TOKEN_RLE;
        d->pending_window_offset = rle_count;
    } else {
        /* Complete write */
        to_write = remaining_count;
        d->skip_bytes = 0;
        d->token_state = TOKEN_NONE;
    }

    /* Write repeated bytes to output */
    TAMP_MEMSET(*output, symbol, to_write);
    *output += to_write;
    *output_written_size += to_write;

    /* Update window only on first chunk (skip==0).
     * Write up to TAMP_RLE_MAX_WINDOW or until end of buffer (no wrap). */
    if (skip == 0) {
        const uint16_t window_size = 1u << d->conf_window;
        uint16_t remaining = window_size - d->window_pos;
        uint8_t window_write = MIN(MIN(rle_count, TAMP_RLE_MAX_WINDOW), remaining); /* max 8 */
        for (uint8_t i = 0; i < window_write; i++) {
            d->window[d->window_pos++] = symbol;
        }
        d->window_pos &= (window_size - 1);
    }

    return (d->token_state == TOKEN_NONE) ? TAMP_OK : TAMP_OUTPUT_FULL;
}

/**
 * @brief Decode extended match token and copy from window to output.
 *
 * NEW FORMAT: huffman(size_high) + trailing_bits(size_low) + window_offset
 * match_size = (size_high << 3) + size_low + min_pattern_size + 12
 *
 * State machine:
 * - Fresh: decode huffman+trailing, then window_offset
 * - TOKEN_EXT_MATCH: have match_size, need window_offset
 * - Output-full resume (skip > 0): have both match_size and window_offset
 */
static tamp_res decode_extended_match(TampDecompressor* d, unsigned char** output, const unsigned char* output_end,
                                      size_t* output_written_size) {
    const uint8_t conf_window = d->conf_window;
    uint16_t window_offset;
    uint8_t match_size; /* max 134: (14<<3)+7 + 3 + 12 */
    uint8_t skip = d->skip_bytes;

    if (skip > 0) {
        /* Resume from output-full: both values saved */
        window_offset = d->pending_window_offset;
        match_size = d->pending_match_size;
    } else if (d->token_state == TOKEN_EXT_MATCH) {
        /* Resume: have match_size, need window_offset */
        match_size = d->pending_match_size;

        if (TAMP_UNLIKELY(d->bit_buffer_pos < conf_window)) return TAMP_INPUT_EXHAUSTED;
        window_offset = d->bit_buffer >> (32 - conf_window);
        d->bit_buffer <<= conf_window;
        d->bit_buffer_pos -= conf_window;
    } else {
        /* Fresh decode: huffman+trailing first, then window_offset */
        uint32_t bit_buffer = d->bit_buffer;
        uint8_t bit_buffer_pos = d->bit_buffer_pos;
        uint8_t raw;
        tamp_res res = decode_huffman(&bit_buffer, &bit_buffer_pos, TAMP_LEADING_EXTENDED_MATCH_BITS, &raw);
        if (res != TAMP_OK) return res;
        match_size = raw + d->min_pattern_size + 12;

        /* Now decode window_offset */
        if (TAMP_UNLIKELY(bit_buffer_pos < conf_window)) {
            /* Save match_size and return */
            d->bit_buffer = bit_buffer;
            d->bit_buffer_pos = bit_buffer_pos;
            d->token_state = TOKEN_EXT_MATCH;
            d->pending_match_size = match_size;
            return TAMP_INPUT_EXHAUSTED;
        }
        window_offset = bit_buffer >> (32 - conf_window);
        bit_buffer <<= conf_window;
        bit_buffer_pos -= conf_window;
        d->bit_buffer = bit_buffer;
        d->bit_buffer_pos = bit_buffer_pos;
    }

    /* Security check: validate window bounds */
    const uint32_t window_size = (1u << conf_window);
    if (TAMP_UNLIKELY((uint32_t)window_offset >= window_size ||
                      (uint32_t)window_offset + (uint32_t)match_size > window_size)) {
        return TAMP_OOB;
    }

    /* Calculate how many to write this call */
    uint8_t remaining_count = match_size - skip;
    size_t output_space = output_end - *output;
    uint8_t to_write;

    if (TAMP_UNLIKELY(remaining_count > output_space)) {
        /* Partial write - save state for resume */
        to_write = output_space;
        d->skip_bytes = skip + output_space;
        d->token_state = TOKEN_EXT_MATCH; /* Reuse for output-full */
        d->pending_window_offset = window_offset;
        d->pending_match_size = match_size;
    } else {
        /* Complete write */
        to_write = remaining_count;
        d->skip_bytes = 0;
        d->token_state = TOKEN_NONE;
    }

    /* Copy from window to output */
    uint16_t src_offset = window_offset + skip;
    for (uint8_t i = 0; i < to_write; i++) {
        *(*output)++ = d->window[src_offset + i];
    }
    *output_written_size += to_write;

    /* Update window only on complete decode.
     * Write up to end of buffer (no wrap), matching RLE behavior. */
    if (d->token_state == TOKEN_NONE) {
        uint16_t wp = d->window_pos;
        uint16_t remaining = window_size - wp;
        uint8_t window_write = (match_size < remaining) ? match_size : remaining;
        tamp_window_copy(d->window, &wp, window_offset, window_write, window_size - 1);
        d->window_pos = wp;
    }

    return (d->token_state == TOKEN_NONE) ? TAMP_OK : TAMP_OUTPUT_FULL;
}
#endif /* TAMP_EXTENDED_DECOMPRESS */

tamp_res tamp_decompressor_read_header(TampConf* conf, const unsigned char* input, size_t input_size,
                                       size_t* input_consumed_size) {
    if (input_consumed_size) (*input_consumed_size) = 0;
    if (input_size == 0) return TAMP_INPUT_EXHAUSTED;

    // Validate all header bytes before mutating conf.
    size_t header_size = 1 + (input[0] & 0x1);
    if (input_size < header_size) return TAMP_INPUT_EXHAUSTED;
    // All bits in byte 2 are reserved for future use; reject if any are set.
    if (header_size >= 2 && input[1]) return TAMP_INVALID_CONF;

    conf->window = ((input[0] >> 5) & 0x7) + 8;
    conf->literal = ((input[0] >> 3) & 0x3) + 5;
    conf->use_custom_dictionary = ((input[0] >> 2) & 0x1);
    conf->extended = ((input[0] >> 1) & 0x1);
    // more_header (byte 1 bit 0) implies dictionary_reset.
    conf->dictionary_reset = input[0] & 0x1;

    if (input_consumed_size) (*input_consumed_size) += header_size;

    return TAMP_OK;
}

/**
 * Populate the rest of the decompressor structure after the following fields have been populated:
 *   * window
 *   * window_bits_max
 */
static TAMP_OPTIMIZE_SIZE tamp_res tamp_decompressor_populate_from_conf(TampDecompressor* decompressor,
                                                                        uint8_t conf_window, uint8_t conf_literal,
                                                                        uint8_t conf_use_custom_dictionary,
                                                                        uint8_t conf_extended,
                                                                        uint8_t conf_dictionary_reset) {
    if (conf_window < 8 || conf_window > 15) return TAMP_INVALID_CONF;
    if (conf_literal < 5 || conf_literal > 8) return TAMP_INVALID_CONF;
    if (conf_window > decompressor->window_bits_max) return TAMP_INVALID_CONF;
    if (!conf_use_custom_dictionary)
        tamp_initialize_dictionary(decompressor->window, (size_t)1 << conf_window, conf_extended ? conf_literal : 8);

    decompressor->conf_window = conf_window;
    decompressor->conf_literal = conf_literal;
    decompressor->min_pattern_size = tamp_compute_min_pattern_size(conf_window, conf_literal);
    decompressor->configured = true;
    decompressor->conf_extended = conf_extended;
    decompressor->conf_dictionary_reset = conf_dictionary_reset;
#if !TAMP_EXTENDED_DECOMPRESS
    if (conf_extended) return TAMP_INVALID_CONF;  // Extended stream but extended support not compiled in
#endif

    return TAMP_OK;
}

tamp_res tamp_decompressor_init(TampDecompressor* decompressor, const TampConf* conf, unsigned char* window,
                                uint8_t window_bits) {
    tamp_res res = TAMP_OK;

    // Validate window_bits parameter
    if (window_bits < 8 || window_bits > 15) return TAMP_INVALID_CONF;

    TAMP_MEMSET(decompressor, 0, sizeof(TampDecompressor));
    decompressor->window = window;
    decompressor->window_bits_max = window_bits;
    if (conf) {
        res = tamp_decompressor_populate_from_conf(decompressor, conf->window, conf->literal,
                                                   conf->use_custom_dictionary, conf->extended, conf->dictionary_reset);
    }

    return res;
}

/**
 * @brief Refill bit buffer from input stream.
 *
 * Consumes bytes from input until bit_buffer has at least 25 bits or input is exhausted.
 *
 * NOTE: NOINLINE saves ~192 bytes on armv6m but causes ~10% decompression
 * speed regression. Keep this inlined for performance.
 */
static inline void refill_bit_buffer(TampDecompressor* d, const unsigned char** input, const unsigned char* input_end,
                                     size_t* input_consumed_size) {
    while (*input != input_end && d->bit_buffer_pos <= 24) {
        d->bit_buffer_pos += 8;
        d->bit_buffer |= (uint32_t) * (*input) << (32 - d->bit_buffer_pos);
        (*input)++;
        (*input_consumed_size)++;
    }
}

#if TAMP_HAS_GCC_OPTIMIZE
#pragma GCC push_options
#pragma GCC optimize("-fno-tree-pre")
#endif
tamp_res tamp_decompressor_decompress_cb(TampDecompressor* decompressor, unsigned char* output, size_t output_size,
                                         size_t* output_written_size, const unsigned char* input, size_t input_size,
                                         size_t* input_consumed_size, tamp_callback_t callback, void* user_data) {
    size_t input_consumed_size_proxy;
    size_t output_written_size_proxy;
    tamp_res res;
    const unsigned char* input_end = input + input_size;
    const unsigned char* output_end = output + output_size;

    if (!output_written_size) output_written_size = &output_written_size_proxy;
    if (!input_consumed_size) input_consumed_size = &input_consumed_size_proxy;

    *input_consumed_size = 0;
    *output_written_size = 0;

    if (TAMP_UNLIKELY(!decompressor->configured)) {
        // Try reading header directly from input. read_header handles
        // variable-length headers (1-2 bytes based on more_headers bit).
        // If the first byte indicates a 2-byte header but only 1 byte is
        // available, stash it and return INPUT_EXHAUSTED.
        size_t header_consumed;
        TampConf conf;
        if (TAMP_UNLIKELY(decompressor->header_bytes_read)) {
            // Second call: prepend stashed first byte.
            unsigned char header_buf[2] = {decompressor->stashed_header_byte, 0};
            if (input != input_end) header_buf[1] = *input;
            res = tamp_decompressor_read_header(&conf, header_buf, 1 + (input != input_end), &header_consumed);
            if (res != TAMP_OK) return res;
            // First byte was already consumed in prior call; only count new bytes.
            size_t new_consumed = header_consumed - 1;
            input += new_consumed;
            (*input_consumed_size) += new_consumed;
        } else {
            res = tamp_decompressor_read_header(&conf, input, input_end - input, &header_consumed);
            if (res == TAMP_INPUT_EXHAUSTED && input != input_end) {
                // Have first byte but need second; stash and retry next call.
                decompressor->stashed_header_byte = *input;
                decompressor->header_bytes_read = 1;
                (*input_consumed_size)++;
                return TAMP_INPUT_EXHAUSTED;
            }
            if (res != TAMP_OK) return res;
            input += header_consumed;
            (*input_consumed_size) += header_consumed;
        }

        res = tamp_decompressor_populate_from_conf(decompressor, conf.window, conf.literal, conf.use_custom_dictionary,
                                                   conf.extended, conf.dictionary_reset);
        if (res != TAMP_OK) return res;
        decompressor->skip_bytes = 0;  // Clear stale stashed_header_byte (shares union storage)
    }

    // Cache bitfield values in local variables for faster access
    const uint8_t conf_window = decompressor->conf_window;
    const uint8_t conf_literal = decompressor->conf_literal;
    const uint8_t min_pattern_size = decompressor->min_pattern_size;

    const uint16_t window_mask = (1 << conf_window) - 1;
#if TAMP_EXTENDED_DECOMPRESS
    const bool extended_enabled = decompressor->conf_extended;
#endif

    while (input != input_end || decompressor->pos_and_state) {
        if (TAMP_UNLIKELY(output == output_end)) return TAMP_OUTPUT_FULL;

        // Populate the bit buffer
        refill_bit_buffer(decompressor, &input, input_end, input_consumed_size);

#if TAMP_EXTENDED_DECOMPRESS
        /* Handle extended tokens - either resuming or fresh from match_size detection below. */
        if (TAMP_UNLIKELY(decompressor->token_state)) {
        extended_dispatch:
            if (decompressor->token_state == TOKEN_RLE) {
                res = decode_rle(decompressor, &output, output_end, output_written_size);
            } else {
                res = decode_extended_match(decompressor, &output, output_end, output_written_size);
            }
            if (res == TAMP_INPUT_EXHAUSTED) {
                uint8_t old_bit_pos = decompressor->bit_buffer_pos;
                refill_bit_buffer(decompressor, &input, input_end, input_consumed_size);
                /* If we couldn't get more bits and input is exhausted, stop.
                 * Otherwise the loop would run forever with token_state set. */
                if (decompressor->bit_buffer_pos == old_bit_pos && input == input_end) {
                    return TAMP_INPUT_EXHAUSTED;
                }
                continue;
            }
            if (res != TAMP_OK) return res;
            continue;
        }
#endif  // TAMP_EXTENDED_DECOMPRESS

        if (TAMP_UNLIKELY(decompressor->bit_buffer_pos == 0)) return TAMP_INPUT_EXHAUSTED;

        // Hint that patterns are more likely than literals
        if (TAMP_UNLIKELY(decompressor->bit_buffer >> 31)) {
            // is literal
            if (TAMP_UNLIKELY(decompressor->last_was_flush)) decompressor->last_was_flush = 0;
            if (TAMP_UNLIKELY(decompressor->bit_buffer_pos < (1 + conf_literal))) return TAMP_INPUT_EXHAUSTED;
            decompressor->bit_buffer <<= 1;  // shift out the is_literal flag

            // Copy literal to output
            *output = decompressor->bit_buffer >> (32 - conf_literal);
            decompressor->bit_buffer <<= conf_literal;
            decompressor->bit_buffer_pos -= (1 + conf_literal);

            // Update window
            decompressor->window[decompressor->window_pos] = *output;
            decompressor->window_pos = (decompressor->window_pos + 1) & window_mask;

            output++;
            (*output_written_size)++;
        } else {
            // is token; attempt a decode
            /* copy the bit buffers so that we can abort at any time */
            uint32_t bit_buffer = decompressor->bit_buffer;
            uint16_t window_offset;
            uint16_t window_offset_skip;
            uint8_t bit_buffer_pos = decompressor->bit_buffer_pos;
            int8_t match_size;
            int8_t match_size_skip;

            // shift out the is_literal flag
            bit_buffer <<= 1;
            bit_buffer_pos--;

            uint8_t match_size_u8;
            if (decode_huffman(&bit_buffer, &bit_buffer_pos, 0, &match_size_u8) != TAMP_OK) return TAMP_INPUT_EXHAUSTED;
            match_size = match_size_u8;

            if (TAMP_UNLIKELY(match_size == FLUSH)) {
                // flush bit_buffer to the nearest byte and skip the remainder of decoding
                decompressor->bit_buffer = bit_buffer << (bit_buffer_pos & 7);
                decompressor->bit_buffer_pos =
                    bit_buffer_pos & ~7;  // Round bit_buffer_pos down to nearest multiple of 8.
                if (decompressor->conf_dictionary_reset && decompressor->last_was_flush) {
                    // Double-FLUSH: reset dictionary.
                    decompressor->window_pos = 0;
                    tamp_initialize_dictionary(decompressor->window, (size_t)1 << conf_window,
                                               decompressor->conf_extended ? conf_literal : 8);
                }
                decompressor->last_was_flush = 1;
                continue;
            }

            if (TAMP_UNLIKELY(decompressor->last_was_flush)) decompressor->last_was_flush = 0;

#if TAMP_EXTENDED_DECOMPRESS
            /* Check for extended symbols (RLE=12, extended match=13).
             * Convert match_size to token_state via subtraction (see TOKEN_* defines). */
            if (TAMP_UNLIKELY(extended_enabled && match_size >= TAMP_RLE_SYMBOL)) {
                decompressor->bit_buffer = bit_buffer;
                decompressor->bit_buffer_pos = bit_buffer_pos;
                decompressor->token_state = match_size - (TAMP_RLE_SYMBOL - 1);
                goto extended_dispatch;
            }
#endif  // TAMP_EXTENDED_DECOMPRESS

            if (TAMP_UNLIKELY(bit_buffer_pos < conf_window)) {
                // There are not enough bits to decode window offset
                return TAMP_INPUT_EXHAUSTED;
            }
            match_size += min_pattern_size;
            window_offset = bit_buffer >> (32 - conf_window);

            // Security check: validate that the pattern reference (offset + size) does not
            // exceed window bounds. Malicious compressed data could craft out-of-bounds
            // references to read past the window buffer, potentially leaking memory.
            // Cast to uint32_t prevents signed integer overflow.
            const uint32_t window_size = (1u << conf_window);
            if (TAMP_UNLIKELY((uint32_t)window_offset >= window_size ||
                              (uint32_t)window_offset + (uint32_t)match_size > window_size)) {
                return TAMP_OOB;
            }

            // Apply skip_bytes
            match_size_skip = match_size - decompressor->skip_bytes;
            window_offset_skip = window_offset + decompressor->skip_bytes;

            // Check if we are output-buffer-limited, and if so to set skip_bytes.
            // Next tamp_decompressor_decompress_cb we will re-decode the same
            // token, and skip the first skip_bytes of it.
            // Otherwise, update the decompressor buffers
            size_t remaining = output_end - output;
            if (TAMP_UNLIKELY((uint8_t)match_size_skip > remaining)) {
                decompressor->skip_bytes += remaining;
                match_size_skip = remaining;
            } else {
                decompressor->skip_bytes = 0;
                decompressor->bit_buffer = bit_buffer << conf_window;
                decompressor->bit_buffer_pos = bit_buffer_pos - conf_window;
            }

            // Copy pattern to output
            for (uint8_t i = 0; i < match_size_skip; i++) {
                *output++ = decompressor->window[window_offset_skip + i];
            }
            (*output_written_size) += match_size_skip;

            if (TAMP_LIKELY(decompressor->skip_bytes == 0)) {
                uint16_t wp = decompressor->window_pos;
                tamp_window_copy(decompressor->window, &wp, window_offset, match_size, window_mask);
                decompressor->window_pos = wp;
            }
        }
        if (TAMP_UNLIKELY(callback && (res = callback(user_data, *input_consumed_size, input_size))))
            return (tamp_res)res;
    }
    return TAMP_INPUT_EXHAUSTED;
}
#if TAMP_HAS_GCC_OPTIMIZE
#pragma GCC pop_options
#endif

#if TAMP_STREAM

TAMP_OPTIMIZE_SIZE tamp_res tamp_decompress_stream(TampDecompressor* decompressor, tamp_read_t read_cb,
                                                   void* read_handle, tamp_write_t write_cb, void* write_handle,
                                                   size_t* input_consumed_size, size_t* output_written_size,
                                                   tamp_callback_t callback, void* user_data) {
    size_t input_consumed_size_proxy, output_written_size_proxy;
    if (!input_consumed_size) input_consumed_size = &input_consumed_size_proxy;
    if (!output_written_size) output_written_size = &output_written_size_proxy;
    *input_consumed_size = 0;
    *output_written_size = 0;

    unsigned char input_buffer[TAMP_STREAM_WORK_BUFFER_SIZE / 2];
    unsigned char output_buffer[TAMP_STREAM_WORK_BUFFER_SIZE / 2];
    const size_t input_buffer_size = sizeof(input_buffer);
    const size_t output_buffer_size = sizeof(output_buffer);

    size_t input_pos = 0;
    size_t input_available = 0;
    bool eof_reached = false;

    while (1) {
        if (input_available == 0 && !eof_reached) {
            int bytes_read = read_cb(read_handle, input_buffer, input_buffer_size);
            if (TAMP_UNLIKELY(bytes_read < 0)) return TAMP_READ_ERROR;
            eof_reached = (bytes_read == 0);
            input_pos = 0;
            input_available = bytes_read;
            *input_consumed_size += bytes_read;
        }

        size_t chunk_consumed, chunk_written;

        tamp_res res = tamp_decompressor_decompress(decompressor, output_buffer, output_buffer_size, &chunk_written,
                                                    input_buffer + input_pos, input_available, &chunk_consumed);
        if (TAMP_UNLIKELY(res < TAMP_OK)) return res;

        input_pos += chunk_consumed;
        input_available -= chunk_consumed;

        if (TAMP_LIKELY(chunk_written > 0)) {
            int bytes_written = write_cb(write_handle, output_buffer, chunk_written);
            if (TAMP_UNLIKELY(bytes_written < 0 || (size_t)bytes_written != chunk_written)) {
                return TAMP_WRITE_ERROR;
            }
            *output_written_size += chunk_written;
        }

        if (TAMP_UNLIKELY(res == TAMP_INPUT_EXHAUSTED && eof_reached)) break;

        if (TAMP_UNLIKELY(callback)) {
            int cb_res = callback(user_data, *input_consumed_size, 0);
            if (TAMP_UNLIKELY(cb_res)) return (tamp_res)cb_res;
        }
    }

    return TAMP_OK;
}

#endif /* TAMP_STREAM */

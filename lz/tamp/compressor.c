#include "compressor.h"

#include <stdbool.h>
#include <stdlib.h>

#include "common.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define BUILD_BUG_ON(condition) ((void)sizeof(char[1 - 2 * !!(condition)]))

#define MAX_PATTERN_SIZE (compressor->min_pattern_size + 13)
#define WINDOW_SIZE (1 << compressor->conf_window)
// 0xF because sizeof(TampCompressor.input) == 16;
#define input_add(offset) ((compressor->input_pos + offset) & 0xF)
#define read_input(offset) (compressor->input[input_add(offset)])
#define IS_LITERAL_FLAG (1 << compressor->conf_literal)

#define FLUSH_CODE (0xAB)

// encodes [min_pattern_bytes, min_pattern_bytes + 13] pattern lengths
static const uint8_t huffman_codes[] = {0x0, 0x3, 0x8, 0xb, 0x14, 0x24, 0x26, 0x2b, 0x4b, 0x54, 0x94, 0x95, 0xaa, 0x27};
// These bit lengths pre-add the 1 bit for the 0-value is_literal flag.
static const uint8_t huffman_bits[] = {0x2, 0x3, 0x5, 0x5, 0x6, 0x7, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0x9, 0x7};

static inline void write_to_bit_buffer(TampCompressor *compressor, uint32_t bits, uint8_t n_bits) {
    compressor->bit_buffer_pos += n_bits;
    compressor->bit_buffer |= bits << (32 - compressor->bit_buffer_pos);
}

/**
 * @brief Partially flush the internal bit buffer.
 *
 * Up to 7 bits may remain in the internal bit buffer.
 */
static inline tamp_res partial_flush(TampCompressor *compressor, unsigned char *output, size_t output_size,
                                     size_t *output_written_size) {
    for (*output_written_size = output_size; compressor->bit_buffer_pos >= 8 && output_size;
         output_size--, compressor->bit_buffer_pos -= 8, compressor->bit_buffer <<= 8)
        *output++ = compressor->bit_buffer >> 24;
    *output_written_size -= output_size;
    return (compressor->bit_buffer_pos >= 8) ? TAMP_OUTPUT_FULL : TAMP_OK;
}

inline bool tamp_compressor_full(TampCompressor *compressor) {
    return compressor->input_size == sizeof(compressor->input);
}

#if TAMP_ESP32
extern void find_best_match(TampCompressor *compressor, uint16_t *match_index, uint8_t *match_size);
#else
/**
 * @brief Find the best match for the current input buffer.
 *
 * @param[in,out] compressor TampCompressor object to perform search on.
 * @param[out] match_index  If match_size is 0, this value is undefined.
 * @param[out] match_size Size of best found match.
 */
static inline void find_best_match(TampCompressor *compressor, uint16_t *match_index, uint8_t *match_size) {
    *match_size = 0;

    if (TAMP_UNLIKELY(compressor->input_size < compressor->min_pattern_size)) return;

    const uint16_t first_second = (read_input(0) << 8) | read_input(1);
    const uint16_t window_size_minus_1 = WINDOW_SIZE - 1;
    const uint8_t max_pattern_size = MIN(compressor->input_size, MAX_PATTERN_SIZE);

    uint16_t window_rolling_2_byte = compressor->window[0];

    for (uint16_t window_index = 0; window_index < window_size_minus_1; window_index++) {
        window_rolling_2_byte <<= 8;
        window_rolling_2_byte |= compressor->window[window_index + 1];
        if (TAMP_LIKELY(window_rolling_2_byte != first_second)) {
            continue;
        }

        // Found 2-byte match, now extend the match
        uint8_t match_len = 2;

        // Extend match byte by byte with optimized bounds checking
        for (uint8_t i = 2; i < max_pattern_size; i++) {
            if (TAMP_UNLIKELY((window_index + i) > window_size_minus_1)) break;

            if (TAMP_LIKELY(compressor->window[window_index + i] != read_input(i))) break;
            match_len = i + 1;
        }

        // Update best match if this is better
        if (TAMP_UNLIKELY(match_len > *match_size)) {
            *match_size = match_len;
            *match_index = window_index;
            // Early termination if we found the maximum possible match
            if (TAMP_UNLIKELY(*match_size == max_pattern_size)) return;
        }
    }
}

#endif

#if TAMP_LAZY_MATCHING
/**
 * @brief Check if writing a single byte will overlap with a future match section.
 *
 * @param[in] write_pos Position where the single byte will be written.
 * @param[in] match_index Index in window where the match starts.
 * @param[in] match_size Size of the match to validate.
 * @return true if no overlap (match is safe), false if there's overlap.
 */
static inline bool validate_no_match_overlap(uint16_t write_pos, uint16_t match_index, uint8_t match_size) {
    // Check if write position falls within the match range [match_index, match_index + match_size - 1]
    return write_pos < match_index || write_pos >= match_index + match_size;
}
#endif

tamp_res tamp_compressor_init(TampCompressor *compressor, const TampConf *conf, unsigned char *window) {
    const TampConf conf_default = {
        .window = 10,
        .literal = 8,
        .use_custom_dictionary = false,
#if TAMP_LAZY_MATCHING
        .lazy_matching = false,
#endif
    };
    if (!conf) conf = &conf_default;
    if (conf->window < 8 || conf->window > 15) return TAMP_INVALID_CONF;
    if (conf->literal < 5 || conf->literal > 8) return TAMP_INVALID_CONF;

    for (uint8_t i = 0; i < sizeof(TampCompressor); i++)  // Zero-out the struct
        ((unsigned char *)compressor)[i] = 0;

    compressor->conf_literal = conf->literal;
    compressor->conf_window = conf->window;
    compressor->conf_use_custom_dictionary = conf->use_custom_dictionary;
#if TAMP_LAZY_MATCHING
    compressor->conf_lazy_matching = conf->lazy_matching;
#endif

    compressor->window = window;
    compressor->min_pattern_size = tamp_compute_min_pattern_size(conf->window, conf->literal);

#if TAMP_LAZY_MATCHING
    compressor->cached_match_index = -1;  // Initialize cache as invalid
#endif

    if (!compressor->conf_use_custom_dictionary) tamp_initialize_dictionary(window, (1 << conf->window));

    // Write header to bit buffer
    write_to_bit_buffer(compressor, conf->window - 8, 3);
    write_to_bit_buffer(compressor, conf->literal - 5, 2);
    write_to_bit_buffer(compressor, conf->use_custom_dictionary, 1);
    write_to_bit_buffer(compressor, 0, 1);  // Reserved
    write_to_bit_buffer(compressor, 0, 1);  // No more header bytes

    return TAMP_OK;
}

tamp_res tamp_compressor_poll(TampCompressor *compressor, unsigned char *output, size_t output_size,
                              size_t *output_written_size) {
    tamp_res res;
    const uint16_t window_mask = (1 << compressor->conf_window) - 1;
    size_t output_written_size_proxy;

    if (!output_written_size) output_written_size = &output_written_size_proxy;
    *output_written_size = 0;

    if (TAMP_UNLIKELY(compressor->input_size == 0)) return TAMP_OK;

    {
        // Make sure there's enough room in the bit buffer.
        size_t flush_bytes_written;
        res = partial_flush(compressor, output, output_size, &flush_bytes_written);
        (*output_written_size) += flush_bytes_written;
        if (TAMP_UNLIKELY(res != TAMP_OK)) return res;
        output_size -= flush_bytes_written;
        output += flush_bytes_written;
    }

    if (TAMP_UNLIKELY(output_size == 0)) return TAMP_OUTPUT_FULL;

    uint8_t match_size = 0;
    uint16_t match_index = 0;

#if TAMP_LAZY_MATCHING
    if (compressor->conf_lazy_matching) {
        // Check if we have a cached match from lazy matching
        if (TAMP_UNLIKELY(compressor->cached_match_index >= 0)) {
            match_index = compressor->cached_match_index;
            match_size = compressor->cached_match_size;
            compressor->cached_match_index = -1;  // Clear cache after using
        } else {
            find_best_match(compressor, &match_index, &match_size);
        }
    } else {
        find_best_match(compressor, &match_index, &match_size);
    }
#else
    find_best_match(compressor, &match_index, &match_size);
#endif

#if TAMP_LAZY_MATCHING
    if (compressor->conf_lazy_matching) {
        // Lazy matching: if we have a good match, check if position i+1 has a better match
        if (match_size >= compressor->min_pattern_size && match_size <= 8 && compressor->input_size > match_size + 2) {
            // Temporarily advance input position to check next position
            compressor->input_pos = input_add(1);
            compressor->input_size--;

            uint8_t next_match_size = 0;
            uint16_t next_match_index = 0;
            find_best_match(compressor, &next_match_index, &next_match_size);

            // Restore input position
            compressor->input_pos = input_add(-1);
            compressor->input_size++;

            // If next position has a better match, and the match doesn't overlap with the literal we are writing, emit
            // literal and cache the next match
            if (next_match_size > match_size &&
                validate_no_match_overlap(compressor->window_pos, next_match_index, next_match_size)) {
                // Write LITERAL at current position
                match_size = 1;
                unsigned char c = read_input(0);
                if (TAMP_UNLIKELY(c >> compressor->conf_literal)) {
                    return TAMP_EXCESS_BITS;
                }
                write_to_bit_buffer(compressor, IS_LITERAL_FLAG | c, compressor->conf_literal + 1);
            } else {
                // Use current match, clear cache
                compressor->cached_match_index = -1;
                uint8_t huffman_index = match_size - compressor->min_pattern_size;
                write_to_bit_buffer(compressor, huffman_codes[huffman_index], huffman_bits[huffman_index]);
                write_to_bit_buffer(compressor, match_index, compressor->conf_window);
            }
        } else if (TAMP_UNLIKELY(match_size < compressor->min_pattern_size)) {
            // Write LITERAL
            compressor->cached_match_index = -1;  // Clear cache
            match_size = 1;
            unsigned char c = read_input(0);
            if (TAMP_UNLIKELY(c >> compressor->conf_literal)) {
                return TAMP_EXCESS_BITS;
            }
            write_to_bit_buffer(compressor, IS_LITERAL_FLAG | c, compressor->conf_literal + 1);
        } else {
            // Write TOKEN
            compressor->cached_match_index = -1;  // Clear cache
            uint8_t huffman_index = match_size - compressor->min_pattern_size;
            write_to_bit_buffer(compressor, huffman_codes[huffman_index], huffman_bits[huffman_index]);
            write_to_bit_buffer(compressor, match_index, compressor->conf_window);
        }
    } else
#endif
    {
        // Non-lazy matching path
        if (TAMP_UNLIKELY(match_size < compressor->min_pattern_size)) {
            // Write LITERAL
            match_size = 1;
            unsigned char c = read_input(0);
            if (TAMP_UNLIKELY(c >> compressor->conf_literal)) {
                return TAMP_EXCESS_BITS;
            }
            write_to_bit_buffer(compressor, IS_LITERAL_FLAG | c, compressor->conf_literal + 1);
        } else {
            // Write TOKEN
            uint8_t huffman_index = match_size - compressor->min_pattern_size;
            write_to_bit_buffer(compressor, huffman_codes[huffman_index], huffman_bits[huffman_index]);
            write_to_bit_buffer(compressor, match_index, compressor->conf_window);
        }
    }
    // Populate Window
    for (uint8_t i = 0; i < match_size; i++) {
        compressor->window[compressor->window_pos] = read_input(0);
        compressor->window_pos = (compressor->window_pos + 1) & window_mask;
        compressor->input_pos = input_add(1);
    }
    compressor->input_size -= match_size;

    return TAMP_OK;
}

void tamp_compressor_sink(TampCompressor *compressor, const unsigned char *input, size_t input_size,
                          size_t *consumed_size) {
    size_t consumed_size_proxy;
    if (TAMP_LIKELY(consumed_size))
        *consumed_size = 0;
    else
        consumed_size = &consumed_size_proxy;

    for (size_t i = 0; i < input_size; i++) {
        if (TAMP_UNLIKELY(tamp_compressor_full(compressor))) break;
        compressor->input[input_add(compressor->input_size)] = input[i];
        compressor->input_size += 1;
        (*consumed_size)++;
    }
}

tamp_res tamp_compressor_compress_cb(TampCompressor *compressor, unsigned char *output, size_t output_size,
                                     size_t *output_written_size, const unsigned char *input, size_t input_size,
                                     size_t *input_consumed_size, tamp_callback_t callback, void *user_data) {
    tamp_res res;
    size_t input_consumed_size_proxy, output_written_size_proxy;
    size_t total_input_size = input_size;

    if (TAMP_LIKELY(output_written_size))
        *output_written_size = 0;
    else
        output_written_size = &output_written_size_proxy;

    if (TAMP_LIKELY(input_consumed_size))
        *input_consumed_size = 0;
    else
        input_consumed_size = &input_consumed_size_proxy;

    while (input_size > 0 && output_size > 0) {
        {
            // Sink Data into input buffer.
            size_t consumed;
            tamp_compressor_sink(compressor, input, input_size, &consumed);
            input += consumed;
            input_size -= consumed;
            (*input_consumed_size) += consumed;
        }
        if (TAMP_LIKELY(tamp_compressor_full(compressor))) {
            // Input buffer is full and ready to start compressing.
            size_t chunk_output_written_size;
            res = tamp_compressor_poll(compressor, output, output_size, &chunk_output_written_size);
            output += chunk_output_written_size;
            output_size -= chunk_output_written_size;
            (*output_written_size) += chunk_output_written_size;
            if (TAMP_UNLIKELY(res != TAMP_OK)) return res;
            if (TAMP_UNLIKELY(callback && (res = callback(user_data, *output_written_size, total_input_size))))
                return (tamp_res)res;
        }
    }
    return TAMP_OK;
}

tamp_res tamp_compressor_flush(TampCompressor *compressor, unsigned char *output, size_t output_size,
                               size_t *output_written_size, bool write_token) {
    tamp_res res;
    size_t chunk_output_written_size;
    size_t output_written_size_proxy;

    if (!output_written_size) output_written_size = &output_written_size_proxy;
    *output_written_size = 0;

    while (compressor->input_size) {
        // Compress the remainder of the input buffer.
        res = tamp_compressor_poll(compressor, output, output_size, &chunk_output_written_size);
        (*output_written_size) += chunk_output_written_size;
        if (TAMP_UNLIKELY(res != TAMP_OK)) return res;
        output_size -= chunk_output_written_size;
        output += chunk_output_written_size;
    }

    // Perform partial flush to see if we need a FLUSH token (check if output buffer in not empty),
    // and to subsequently make room for the FLUSH token.
    res = partial_flush(compressor, output, output_size, &chunk_output_written_size);
    output_size -= chunk_output_written_size;
    (*output_written_size) += chunk_output_written_size;
    output += chunk_output_written_size;
    if (TAMP_UNLIKELY(res != TAMP_OK)) return res;

    // Check if there's enough output buffer space
    if (compressor->bit_buffer_pos) {
        if (output_size == 0) {
            return TAMP_OUTPUT_FULL;
        }
        if (write_token) {
            if (output_size < 2) return TAMP_OUTPUT_FULL;
            write_to_bit_buffer(compressor, FLUSH_CODE, 9);
        }
    }

    // Flush the remainder of the output bit-buffer
    while (compressor->bit_buffer_pos) {
        *output = compressor->bit_buffer >> 24;
        output++;
        compressor->bit_buffer <<= 8;
        compressor->bit_buffer_pos -= MIN(compressor->bit_buffer_pos, 8);
        output_size--;
        (*output_written_size)++;
    }

    return TAMP_OK;
}

tamp_res tamp_compressor_compress_and_flush_cb(TampCompressor *compressor, unsigned char *output, size_t output_size,
                                               size_t *output_written_size, const unsigned char *input,
                                               size_t input_size, size_t *input_consumed_size, bool write_token,
                                               tamp_callback_t callback, void *user_data) {
    tamp_res res;
    size_t flush_size;
    size_t output_written_size_proxy;

    if (!output_written_size) output_written_size = &output_written_size_proxy;

    res = tamp_compressor_compress_cb(compressor, output, output_size, output_written_size, input, input_size,
                                      input_consumed_size, callback, user_data);
    if (TAMP_UNLIKELY(res != TAMP_OK)) return res;

    res = tamp_compressor_flush(compressor, output + *output_written_size, output_size - *output_written_size,
                                &flush_size, write_token);

    (*output_written_size) += flush_size;

    if (TAMP_UNLIKELY(res != TAMP_OK)) return res;

    return TAMP_OK;
}

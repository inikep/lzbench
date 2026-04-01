#include "compressor.h"

#include <stdbool.h>
#include <stdlib.h>

#include "common.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define BUILD_BUG_ON(condition) ((void)sizeof(char[1 - 2 * !!(condition)]))

#if TAMP_EXTENDED_COMPRESS
// Extended max pattern: min_pattern_size + 11 + EXTENDED_MATCH_MAX_EXTRA
// = min_pattern_size + 11 + (14 << 3) + 7 + 1 = min_pattern_size + 131
#define MAX_PATTERN_SIZE_EXTENDED (compressor->min_pattern_size + 11 + EXTENDED_MATCH_MAX_EXTRA)
#define MAX_PATTERN_SIZE (compressor->conf.extended ? MAX_PATTERN_SIZE_EXTENDED : (compressor->min_pattern_size + 13))
#else
#define MAX_PATTERN_SIZE (compressor->min_pattern_size + 13)
#endif
#define WINDOW_SIZE (1 << compressor->conf.window)
// 0xF because sizeof(TampCompressor.input) == 16;
#define input_add(offset) ((compressor->input_pos + offset) & 0xF)
#define read_input(offset) (compressor->input[input_add(offset)])
#define IS_LITERAL_FLAG (1 << compressor->conf.literal)

#define FLUSH_CODE (0xAB)

// Internal return value for poll_extended_handling: signals caller to
// proceed with normal pattern matching rather than returning immediately.
#define TAMP_POLL_CONTINUE ((tamp_res)127)

// encodes [min_pattern_bytes, min_pattern_bytes + 14] pattern lengths (14 = FLUSH pattern, used in secondary reads)
static const uint8_t huffman_codes[] = {0x0,  0x3,  0x8,  0xb,  0x14, 0x24, 0x26, 0x2b,
                                        0x4b, 0x54, 0x94, 0x95, 0xaa, 0x27, 0xab};
// These bit lengths pre-add the 1 bit for the 0-value is_literal flag.
static const uint8_t huffman_bits[] = {0x2, 0x3, 0x5, 0x5, 0x6, 0x7, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0x9, 0x7, 0x09};

#if TAMP_EXTENDED_COMPRESS
#define RLE_MAX_COUNT ((14 << 4) + 15 + 2)            // 241
#define EXTENDED_MATCH_MAX_EXTRA ((14 << 3) + 7 + 1)  // 120

// Minimum output buffer space required for extended match token.
// Extended match: symbol (7 bits) + extended huffman (11 bits) + window pos (15 bits) = 33 bits.
// With 7 bits in bit buffer, need up to 40 bits = 5 bytes. Add 1 byte margin.
// Pre-checking prevents OUTPUT_FULL mid-token, which would corrupt bit_buffer on retry.
#define EXTENDED_MATCH_MIN_OUTPUT_BYTES 6
#endif

static TAMP_NOINLINE void write_to_bit_buffer(TampCompressor* compressor, uint32_t bits, uint8_t n_bits) {
    compressor->bit_buffer_pos += n_bits;
    compressor->bit_buffer |= bits << (32 - compressor->bit_buffer_pos);
}

/**
 * @brief Partially flush the internal bit buffer.
 *
 * Flushes complete bytes from the bit buffer. Up to 7 bits may remain.
 *
 * @param[in,out] compressor Compressor state.
 * @param[in,out] output Output buffer pointer (updated on return).
 * @param[in,out] output_size Available space (updated on return).
 * @param[in,out] output_written_size Bytes written (accumulated).
 * @return TAMP_OK on success, TAMP_OUTPUT_FULL if output buffer is too small.
 */
static TAMP_NOINLINE TAMP_OPTIMIZE_SIZE tamp_res partial_flush(TampCompressor* compressor, unsigned char** output,
                                                               size_t* output_size, size_t* output_written_size) {
    while (compressor->bit_buffer_pos >= 8 && *output_size) {
        *(*output)++ = compressor->bit_buffer >> 24;
        (*output_size)--;
        (*output_written_size)++;
        compressor->bit_buffer_pos -= 8;
        compressor->bit_buffer <<= 8;
    }
    return (compressor->bit_buffer_pos >= 8) ? TAMP_OUTPUT_FULL : TAMP_OK;
}

inline bool tamp_compressor_full(const TampCompressor* compressor) {
    return compressor->input_size == sizeof(compressor->input);
}

/*
 * Platform-specific find_best_match implementations:
 *
 * 1. TAMP_ESP32: External implementation in espidf/tamp/compressor_esp32.cpp
 *
 * 2. Desktop 64-bit (x86_64, aarch64, Windows 64-bit):
 *    Included from compressor_find_match_desktop.c - uses bit manipulation
 *    and 64-bit loads for parallel match detection
 *
 * 3. Embedded/Default (Cortex-M0/M0+, other 32-bit):
 *    Defined below - single-byte-first comparison, safe for all architectures
 *
 * Set TAMP_USE_EMBEDDED_MATCH=1 to force the embedded implementation on desktop
 * (useful for testing the embedded code path on CI).
 */

#if TAMP_ESP32
extern void find_best_match(TampCompressor* compressor, uint16_t* match_index, uint8_t* match_size);

#elif (defined(__x86_64__) || defined(__aarch64__) || defined(_M_X64) || defined(_M_ARM64)) && !TAMP_USE_EMBEDDED_MATCH
#include "compressor_find_match_desktop.c"

#else
/**
 * @brief Find the best match for the current input buffer.
 *
 * Embedded/32-bit implementation: uses single-byte-first comparison (faster on simple cores).
 *
 * @param[in,out] compressor TampCompressor object to perform search on.
 * @param[out] match_index  If match_size is 0, this value is undefined.
 * @param[out] match_size Size of best found match.
 */
static TAMP_NOINLINE void find_best_match(TampCompressor* compressor, uint16_t* match_index, uint8_t* match_size) {
    *match_size = 0;

    if (TAMP_UNLIKELY(compressor->input_size < compressor->min_pattern_size)) return;

    const uint8_t first_byte = read_input(0);
    const uint8_t second_byte = read_input(1);
    const uint32_t window_size_minus_1 = WINDOW_SIZE - 1;
    const uint8_t max_pattern_size = MIN(compressor->input_size, MAX_PATTERN_SIZE);
    const unsigned char* window = compressor->window;

    for (uint32_t window_index = 0; window_index < window_size_minus_1; window_index++) {
        if (TAMP_LIKELY(window[window_index] != first_byte)) {
            continue;
        }
        if (TAMP_LIKELY(window[window_index + 1] != second_byte)) {
            continue;
        }

        // Found 2-byte match, now extend the match
        uint8_t match_len = 2;

        // Extend match byte by byte with optimized bounds checking
        for (uint8_t i = 2; i < max_pattern_size; i++) {
            if (TAMP_UNLIKELY((window_index + i) > window_size_minus_1)) break;

            if (TAMP_LIKELY(window[window_index + i] != read_input(i))) break;
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

TAMP_OPTIMIZE_SIZE tamp_res tamp_compressor_init(TampCompressor* compressor, const TampConf* conf,
                                                 unsigned char* window) {
    const TampConf conf_default = {
        .window = 10,
        .literal = 8,
        .use_custom_dictionary = false,
#if TAMP_LAZY_MATCHING
        .lazy_matching = false,
#endif
#if TAMP_EXTENDED_COMPRESS
        .extended = true,  // Default to extended format
#endif
    };
    if (!conf) conf = &conf_default;
    if (conf->window < 8 || conf->window > 15) return TAMP_INVALID_CONF;
    if (conf->literal < 5 || conf->literal > 8) return TAMP_INVALID_CONF;
    if (conf->append && (!conf->dictionary_reset || conf->use_custom_dictionary)) return TAMP_INVALID_CONF;
#if !TAMP_EXTENDED_COMPRESS
    if (conf->extended) return TAMP_INVALID_CONF;  // Extended requested but not compiled in
#endif

    TAMP_MEMSET(compressor, 0, sizeof(TampCompressor));

    compressor->conf = *conf;  // Single struct copy
    compressor->window = window;
    compressor->min_pattern_size = tamp_compute_min_pattern_size(conf->window, conf->literal);

#if TAMP_LAZY_MATCHING
    compressor->cached_match_index = -1;  // Initialize cache as invalid
#endif

    if (!conf->use_custom_dictionary)
        tamp_initialize_dictionary(window, (1 << conf->window), conf->extended ? conf->literal : 8);

    if (conf->append) {
        // Write a FLUSH token (9 bits) padded to 16 bits. Combined with the
        // previous stream's trailing FLUSH, this triggers a dictionary reset.
        write_to_bit_buffer(compressor, FLUSH_CODE, 9);
        compressor->bit_buffer_pos = 16;
    } else {
        // Header byte 1: [window:3][literal:2][use_custom_dictionary:1][extended:1][more_headers:1]
        uint8_t header = ((conf->window - 8) << 5) | ((conf->literal - 5) << 3) | (conf->use_custom_dictionary << 2) |
                         (conf->extended << 1) | conf->dictionary_reset;
        write_to_bit_buffer(compressor, header, 8);
        // Header byte 2 (if present): reserved bits, all zero from memset.
        compressor->bit_buffer_pos += conf->dictionary_reset ? 8 : 0;
    }

    return TAMP_OK;
}

#if TAMP_EXTENDED_COMPRESS
/**
 * @brief Write extended huffman encoding (huffman + trailing bits).
 *
 * Used for both RLE count and extended match size encoding.
 *
 * @param[in,out] compressor Compressor with bit buffer.
 * @param[in] value The value to encode.
 * @param[in] trailing_bits Number of trailing bits (3 for extended match, 4 for RLE).
 */
static TAMP_NOINLINE TAMP_OPTIMIZE_SIZE void write_extended_huffman(TampCompressor* compressor, uint8_t value,
                                                                    uint8_t trailing_bits) {
    uint8_t code_index = value >> trailing_bits;
    // Write huffman code (without literal flag) + trailing bits in one call
    write_to_bit_buffer(compressor, (huffman_codes[code_index] << trailing_bits) | (value & ((1 << trailing_bits) - 1)),
                        (huffman_bits[code_index] - 1) + trailing_bits);
}

/**
 * @brief Get the last byte written to the window.
 *
 * NOINLINE: called from 3 sites; outlining saves ~44 bytes on armv6m.
 */
static TAMP_NOINLINE TAMP_OPTIMIZE_SIZE uint8_t get_last_window_byte(TampCompressor* compressor) {
    uint16_t prev_pos = (compressor->window_pos - 1) & ((1 << compressor->conf.window) - 1);
    return compressor->window[prev_pos];
}

/**
 * @brief Search for extended match continuation using implicit pattern comparison.
 *
 * Searches for pattern: window[current_pos:current_pos+current_count] + input[0...]
 * starting from current_pos. Returns the longest match found (which may be at
 * current_pos itself if O(1) extension works, or at a different position).
 *
 * NOINLINE + Os: Called only during extended match continuation (rare path).
 * Outlining saves ~100 bytes in poll on armv6m.
 *
 * @param[in] compressor TampCompressor object
 * @param[in] current_pos Current match position in window (also search start)
 * @param[in] current_count Current match length
 * @param[out] new_pos Position of found match (only valid if new_count > current_count)
 * @param[out] new_count Length of found match
 */
static TAMP_NOINLINE TAMP_OPTIMIZE_SIZE void find_extended_match(TampCompressor* compressor, uint16_t current_pos,
                                                                 uint8_t current_count, uint16_t* new_pos,
                                                                 uint8_t* new_count) {
    // Preconditions (guaranteed by caller):
    // - input_size > 0
    // - current_pos + current_count < WINDOW_SIZE
    // - current_count < MAX_PATTERN_SIZE
    *new_count = 0;
    const unsigned char* window = compressor->window;
    const uint16_t window_size = WINDOW_SIZE;
    const uint8_t max_pattern = MIN(current_count + compressor->input_size, MAX_PATTERN_SIZE);
    const uint8_t extend_byte = read_input(0);

    for (uint16_t cand = current_pos; cand + current_count + 1 <= window_size; cand++) {
        // Check extension byte first (most discriminating)
        if (window[cand + current_count] != extend_byte) continue;

        // Check if current_count bytes match (at cand==current_pos, compares with self)
        uint8_t i = 0;
        while (i < current_count && window[cand + i] == window[current_pos + i]) i++;
        if (i < current_count) continue;

        // Found a match - extend as far as possible
        const uint8_t cand_max = MIN(max_pattern, window_size - cand);
        uint8_t match_len = current_count + 1;
        for (i = current_count + 1; i < cand_max; i++) {
            if (window[cand + i] != read_input(i - current_count)) break;
            match_len = i + 1;
        }

        if (match_len > *new_count) {
            *new_count = match_len;
            *new_pos = cand;
            if (match_len == max_pattern) return;
        }
    }
}

/**
 * @brief Write RLE token to bit buffer and update window.
 *
 * @param[in,out] compressor Compressor state.
 * @param[in] count Number of repeated bytes (must be >= 2).
 */
static TAMP_NOINLINE void write_rle_token(TampCompressor* compressor, uint8_t count) {
    const uint16_t window_mask = (1 << compressor->conf.window) - 1;
    uint8_t symbol = get_last_window_byte(compressor);

    // Write RLE symbol (12) with literal flag
    // Note: symbols 12 and 13 are at indices 12 and 13 in huffman table (not offset by min_pattern_size)
    write_to_bit_buffer(compressor, huffman_codes[TAMP_RLE_SYMBOL], huffman_bits[TAMP_RLE_SYMBOL]);
    // Write extended huffman for count-2
    write_extended_huffman(compressor, count - 2, TAMP_LEADING_RLE_BITS);

    // Write up to TAMP_RLE_MAX_WINDOW bytes to window (or until buffer end, no wrap)
    uint16_t remaining = WINDOW_SIZE - compressor->window_pos;
    uint8_t window_write = MIN(MIN(count, TAMP_RLE_MAX_WINDOW), remaining);
    for (uint8_t i = 0; i < window_write; i++) {
        compressor->window[compressor->window_pos] = symbol;
        compressor->window_pos = (compressor->window_pos + 1) & window_mask;
    }
}

/**
 * @brief Write extended match token to bit buffer and update window.
 *
 * Token format: symbol (7 bits) + extended_huffman (up to 11 bits) + window_pos (up to 15 bits)
 * Total: up to 33 bits. We flush after symbol+huffman (18 bits max) to ensure window_pos fits.
 *
 * @param[in,out] compressor Compressor state.
 * @param[in,out] output Output buffer pointer (updated on return).
 * @param[in,out] output_size Available space (updated on return).
 * @param[in,out] output_written_size Bytes written (accumulated).
 * @return TAMP_OK on success, TAMP_OUTPUT_FULL if output buffer is too small.
 */
#if TAMP_HAS_GCC_OPTIMIZE
#pragma GCC push_options
#pragma GCC optimize("-fno-reorder-blocks")
#endif
static TAMP_NOINLINE tamp_res write_extended_match_token(TampCompressor* compressor, unsigned char** output,
                                                         size_t* output_size, size_t* output_written_size) {
    // Pre-check output space to prevent OUTPUT_FULL mid-token (would corrupt bit_buffer)
    if (TAMP_UNLIKELY(*output_size < EXTENDED_MATCH_MIN_OUTPUT_BYTES)) return TAMP_OUTPUT_FULL;

    const uint16_t window_mask = (1 << compressor->conf.window) - 1;
    const uint8_t count = compressor->extended_match_count;
    const uint16_t position = compressor->extended_match_position;
    tamp_res res;

    // Write symbol (7 bits) + extended huffman (up to 11 bits) = 18 bits max
    // With ≤7 bits already in buffer, total ≤25 bits - fits in 32-bit buffer
    write_to_bit_buffer(compressor, huffman_codes[TAMP_EXTENDED_MATCH_SYMBOL],
                        huffman_bits[TAMP_EXTENDED_MATCH_SYMBOL]);
    write_extended_huffman(compressor, count - compressor->min_pattern_size - 11 - 1, TAMP_LEADING_EXTENDED_MATCH_BITS);

    // Flush to make room for window position (up to 15 bits)
    res = partial_flush(compressor, output, output_size, output_written_size);
    if (TAMP_UNLIKELY(res != TAMP_OK)) return res;

    // Write window position - with ≤7 bits remaining, up to 22 bits total - fits
    write_to_bit_buffer(compressor, position, compressor->conf.window);

    // Final flush
    res = partial_flush(compressor, output, output_size, output_written_size);
    if (TAMP_UNLIKELY(res != TAMP_OK)) return res;

    // Write to window (up to end of buffer, no wrap)
    uint16_t remaining = WINDOW_SIZE - compressor->window_pos;
    uint8_t window_write = MIN(count, remaining);
    tamp_window_copy(compressor->window, &compressor->window_pos, position, window_write, window_mask);

    compressor->extended_match_count = 0;  // Position reset not needed - only read when count > 0

    return TAMP_OK;
}
#if TAMP_HAS_GCC_OPTIMIZE
#pragma GCC pop_options
#endif

/**
 * @brief Handle all extended-specific logic in poll (match continuation + RLE).
 *
 * NOINLINE + Os: Extended paths are rarely executed. Outlining from poll saves
 * significant code size on register-constrained Cortex-M0+ where the compiler
 * otherwise spills heavily to stack (~48 bytes saved on armv6m).
 *
 * @return TAMP_OK if fully handled (caller should return TAMP_OK),
 *         TAMP_POLL_CONTINUE if caller should proceed to normal pattern matching,
 *         other tamp_res on error.
 */
static TAMP_NOINLINE TAMP_OPTIMIZE_SIZE tamp_res poll_extended_handling(TampCompressor* compressor,
                                                                        unsigned char** output, size_t* output_size,
                                                                        size_t* output_written_size) {
    // Handle extended match continuation
    if (compressor->extended_match_count) {
        const uint8_t max_ext_match = compressor->min_pattern_size + 11 + EXTENDED_MATCH_MAX_EXTRA;

        while (compressor->input_size > 0) {
            const uint16_t current_pos = compressor->extended_match_position;
            const uint8_t current_count = compressor->extended_match_count;

            if (current_pos + current_count >= WINDOW_SIZE || current_count >= max_ext_match) {
                return write_extended_match_token(compressor, output, output_size, output_written_size);
            }

            uint16_t new_pos;
            uint8_t new_count;
            find_extended_match(compressor, current_pos, current_count, &new_pos, &new_count);

            if (new_count > current_count) {
                uint8_t extra_bytes = new_count - current_count;
                compressor->extended_match_position = new_pos;
                compressor->extended_match_count = new_count;
                compressor->input_pos = input_add(extra_bytes);
                compressor->input_size -= extra_bytes;
                continue;
            }

            return write_extended_match_token(compressor, output, output_size, output_written_size);
        }
        return TAMP_OK;
    }

    // Handle RLE accumulation
    uint8_t last_byte = get_last_window_byte(compressor);

    uint8_t rle_available = 0;
    while (rle_available < compressor->input_size && compressor->rle_count + rle_available < RLE_MAX_COUNT &&
           compressor->input[input_add(rle_available)] == last_byte) {
        rle_available++;
    }

    uint8_t total_rle = compressor->rle_count + rle_available;
    bool rle_ended = (rle_available < compressor->input_size) || (total_rle >= RLE_MAX_COUNT);

    if (!rle_ended && total_rle > 0) {
        compressor->rle_count = total_rle;
        compressor->input_pos = input_add(rle_available);
        compressor->input_size -= rle_available;
        return TAMP_OK;
    }

    if (total_rle >= 2) {
        if (total_rle == rle_available && total_rle <= 6) {
            uint16_t pattern_index;
            uint8_t pattern_size;
            find_best_match(compressor, &pattern_index, &pattern_size);

            if (pattern_size > total_rle) {
                compressor->rle_count = 0;
                return TAMP_POLL_CONTINUE;  // Proceed to pattern matching
            }
        }

        compressor->input_pos = input_add(rle_available);
        compressor->input_size -= rle_available;
        write_rle_token(compressor, total_rle);
        compressor->rle_count = 0;
        return TAMP_OK;
    }

    if (total_rle == 1) compressor->rle_count = 0;
    return TAMP_POLL_CONTINUE;  // Proceed to pattern matching
}
#endif  // TAMP_EXTENDED_COMPRESS

#if TAMP_HAS_GCC_OPTIMIZE
#pragma GCC push_options
#pragma GCC optimize("-fno-schedule-insns2")
#endif
TAMP_NOINLINE tamp_res tamp_compressor_poll(TampCompressor* compressor, unsigned char* output, size_t output_size,
                                            size_t* output_written_size) {
    tamp_res res;
    // Cache bitfield values for faster access in hot path
    const uint8_t conf_window = compressor->conf.window;
    const uint8_t conf_literal = compressor->conf.literal;
    const uint16_t window_mask = (1 << conf_window) - 1;
    size_t output_written_size_proxy;

    if (!output_written_size) output_written_size = &output_written_size_proxy;
    *output_written_size = 0;

    if (TAMP_UNLIKELY(compressor->input_size == 0)) return TAMP_OK;

    // Make sure there's enough room in the bit buffer.
    res = partial_flush(compressor, &output, &output_size, output_written_size);
    if (TAMP_UNLIKELY(res != TAMP_OK)) return res;

    if (TAMP_UNLIKELY(output_size == 0)) return TAMP_OUTPUT_FULL;

    uint8_t match_size = 0;
    uint16_t match_index = 0;

#if TAMP_EXTENDED_COMPRESS
    if (TAMP_UNLIKELY(compressor->conf.extended)) {
        // Handle extended match continuation + RLE (outlined for code size)
        res = poll_extended_handling(compressor, &output, &output_size, output_written_size);
        if (res != TAMP_POLL_CONTINUE) return res;
        // TAMP_POLL_CONTINUE: proceed to pattern matching below
    }
#endif  // TAMP_EXTENDED_COMPRESS

#if TAMP_LAZY_MATCHING
    if (compressor->conf.lazy_matching) {
        // Check if we have a cached match from lazy matching
        if (TAMP_UNLIKELY(compressor->cached_match_index >= 0)) {
            match_index = compressor->cached_match_index;
            match_size = compressor->cached_match_size;
            compressor->cached_match_index = -1;  // Clear cache after using
        } else {
            find_best_match(compressor, &match_index, &match_size);
        }

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
                // Force literal at current position, cache next match
                compressor->cached_match_index = next_match_index;
                compressor->cached_match_size = next_match_size;
                match_size = 0;  // Will trigger literal write below
            } else {
                compressor->cached_match_index = -1;
                // Note: No V2 extended match check here - we're in the match_size <= 8 branch,
                // so extended matches (which require match_size > min_pattern_size + 11) are impossible.
            }
        } else {
            compressor->cached_match_index = -1;  // Clear cache
        }
    } else {
        find_best_match(compressor, &match_index, &match_size);
    }
#else
    find_best_match(compressor, &match_index, &match_size);
#endif

    // Shared token/literal writing logic
    if (TAMP_UNLIKELY(match_size < compressor->min_pattern_size)) {
        // Write LITERAL
        match_size = 1;
        unsigned char c = read_input(0);
        if (TAMP_UNLIKELY(c >> conf_literal)) {
            return TAMP_EXCESS_BITS;
        }
        write_to_bit_buffer(compressor, (1 << conf_literal) | c, conf_literal + 1);
    } else {
#if TAMP_EXTENDED_COMPRESS
        // Extended: Start extended match continuation
        if (compressor->conf.extended && match_size > compressor->min_pattern_size + 11) {
            compressor->extended_match_count = match_size;
            compressor->extended_match_position = match_index;
            // Consume matched bytes from input
            compressor->input_pos = input_add(match_size);
            compressor->input_size -= match_size;
            // Return - continuation code at start of poll will try to extend or emit
            return TAMP_OK;
        }
#endif  // TAMP_EXTENDED_COMPRESS
        // Write TOKEN (huffman code + window position)
        uint8_t huffman_index = match_size - compressor->min_pattern_size;
        write_to_bit_buffer(compressor, (huffman_codes[huffman_index] << conf_window) | match_index,
                            huffman_bits[huffman_index] + conf_window);
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
#if TAMP_HAS_GCC_OPTIMIZE
#pragma GCC pop_options
#endif

void tamp_compressor_sink(TampCompressor* compressor, const unsigned char* input, size_t input_size,
                          size_t* consumed_size) {
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

TAMP_OPTIMIZE_SIZE tamp_res tamp_compressor_compress_cb(TampCompressor* compressor, unsigned char* output,
                                                        size_t output_size, size_t* output_written_size,
                                                        const unsigned char* input, size_t input_size,
                                                        size_t* input_consumed_size, tamp_callback_t callback,
                                                        void* user_data) {
    tamp_res res;
    size_t input_consumed_size_proxy = 0, output_written_size_proxy = 0;
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
            if (TAMP_UNLIKELY(callback && (res = callback(user_data, *input_consumed_size, total_input_size))))
                return (tamp_res)res;
        }
    }
    return TAMP_OK;
}

#if TAMP_HAS_GCC_OPTIMIZE
#pragma GCC push_options
#pragma GCC optimize("-fno-tree-pre")
#endif
tamp_res tamp_compressor_flush(TampCompressor* compressor, unsigned char* output, size_t output_size,
                               size_t* output_written_size, bool write_token) {
    tamp_res res;
    size_t chunk_output_written_size;
    size_t output_written_size_proxy;

    if (!output_written_size) output_written_size = &output_written_size_proxy;
    *output_written_size = 0;

flush_check:
    // Flush pending bits before checking for more work
    chunk_output_written_size = 0;
    res = partial_flush(compressor, &output, &output_size, output_written_size);
    if (TAMP_UNLIKELY(res != TAMP_OK)) return res;

    if (TAMP_LIKELY(compressor->input_size)) {
        res = tamp_compressor_poll(compressor, output, output_size, &chunk_output_written_size);
    }
#if TAMP_EXTENDED_COMPRESS
    else if (compressor->conf.extended && compressor->rle_count >= 1) {
        if (compressor->rle_count == 1) {
            // Single byte - write as literal (can't use RLE token for count < 2)
            uint8_t literal = get_last_window_byte(compressor);
            write_to_bit_buffer(compressor, IS_LITERAL_FLAG | literal, compressor->conf.literal + 1);

            // Write to window
            const uint16_t window_mask = (1 << compressor->conf.window) - 1;
            compressor->window[compressor->window_pos] = literal;
            compressor->window_pos = (compressor->window_pos + 1) & window_mask;
        } else {
            // count >= 2: write as RLE token
            write_rle_token(compressor, compressor->rle_count);
        }
        compressor->rle_count = 0;
    } else if (compressor->conf.extended && compressor->extended_match_count) {
        res = write_extended_match_token(compressor, &output, &output_size, output_written_size);
    }
#endif  // TAMP_EXTENDED_COMPRESS
    else {
        goto flush_done;
    }
    (*output_written_size) += chunk_output_written_size;
    if (TAMP_UNLIKELY(res != TAMP_OK)) return res;
    output_size -= chunk_output_written_size;
    output += chunk_output_written_size;
    goto flush_check;

flush_done:
    // At this point, up to 7 bits may remain in the compressor->bit_buffer
    // The output buffer may have 0 bytes remaining.
    if (write_token && (compressor->bit_buffer_pos || compressor->conf.dictionary_reset)) {
        // We don't want to write the FLUSH token to the bit_buffer unless
        // we are confident that it'll wind up in the output buffer
        // in THIS function call.
        // Otherwise, if we wind up with a TAMP_OUTPUT_FULL result, we could
        // end up accidentally writing multiple FLUSH tokens.
        if (TAMP_UNLIKELY(output_size < 2)) return TAMP_OUTPUT_FULL;
        write_to_bit_buffer(compressor, FLUSH_CODE, 9);
    }

    // At this point, up to 16 bits may remain in the compressor->bit_buffer
    // The output buffer may have 0 bytes remaining.

    // Flush whole bytes, then write trailing partial byte
    res = partial_flush(compressor, &output, &output_size, output_written_size);
    if (compressor->bit_buffer_pos) {
        if (TAMP_UNLIKELY(output_size == 0)) return TAMP_OUTPUT_FULL;
        *output = compressor->bit_buffer >> 24;
        (*output_written_size)++;
        compressor->bit_buffer_pos = 0;
        compressor->bit_buffer = 0;
    }

    return res;
}
#if TAMP_HAS_GCC_OPTIMIZE
#pragma GCC pop_options
#endif

TAMP_OPTIMIZE_SIZE tamp_res tamp_compressor_compress_and_flush_cb(TampCompressor* compressor, unsigned char* output,
                                                                  size_t output_size, size_t* output_written_size,
                                                                  const unsigned char* input, size_t input_size,
                                                                  size_t* input_consumed_size, bool write_token,
                                                                  tamp_callback_t callback, void* user_data) {
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

    // Final callback to signal 100% completion (compress_cb's last callback may
    // have reported less than input_size if the final sink didn't trigger a poll).
    if (TAMP_UNLIKELY(callback)) {
        int cb_res = callback(user_data, input_size, input_size);
        if (TAMP_UNLIKELY(cb_res)) return (tamp_res)cb_res;
    }

    return TAMP_OK;
}

TAMP_OPTIMIZE_SIZE tamp_res tamp_compressor_reset_dictionary(TampCompressor* compressor, unsigned char* output,
                                                             size_t output_size, size_t* output_written_size) {
    if (!compressor->conf.dictionary_reset) return TAMP_INVALID_CONF;

    tamp_res res;
    size_t output_written_size_proxy;

    if (!output_written_size) output_written_size = &output_written_size_proxy;
    *output_written_size = 0;

    // Write 2 FLUSH tokens for the double-FLUSH reset signal. The first drains
    // any pending data. dictionary_reset being set guarantees flush() writes a
    // FLUSH token even when bit_buffer_pos is 0.
    for (uint8_t i = 0; i < 2; i++) {
        size_t flush_written_size;
        res = tamp_compressor_flush(compressor, output, output_size, &flush_written_size, true);
        *output_written_size += flush_written_size;
        if (TAMP_UNLIKELY(res != TAMP_OK)) return res;
        output += flush_written_size;
        output_size -= flush_written_size;
    }

    // Re-initialize compressor, then discard the header it writes to the bit buffer.
    // Copy conf because tamp_compressor_init zeroes the struct before reading it.
    TampConf conf = compressor->conf;
    conf.use_custom_dictionary = false;
    res = tamp_compressor_init(compressor, &conf, compressor->window);
    compressor->bit_buffer = 0;
    compressor->bit_buffer_pos = 0;

    return res;
}

#if TAMP_STREAM

TAMP_OPTIMIZE_SIZE tamp_res tamp_compress_stream(TampCompressor* compressor, tamp_read_t read_cb, void* read_handle,
                                                 tamp_write_t write_cb, void* write_handle, size_t* input_consumed_size,
                                                 size_t* output_written_size, tamp_callback_t callback,
                                                 void* user_data) {
    size_t input_consumed_size_proxy, output_written_size_proxy;
    if (!input_consumed_size) input_consumed_size = &input_consumed_size_proxy;
    if (!output_written_size) output_written_size = &output_written_size_proxy;
    *input_consumed_size = 0;
    *output_written_size = 0;

    unsigned char input_buffer[TAMP_STREAM_WORK_BUFFER_SIZE / 2];
    unsigned char output_buffer[TAMP_STREAM_WORK_BUFFER_SIZE / 2];

    // Main compression loop
    while (1) {
        int bytes_read = read_cb(read_handle, input_buffer, sizeof(input_buffer));
        if (TAMP_UNLIKELY(bytes_read < 0)) return TAMP_READ_ERROR;
        if (bytes_read == 0) break;

        *input_consumed_size += bytes_read;

        size_t input_pos = 0;
        while (input_pos < (size_t)bytes_read) {
            size_t chunk_consumed, chunk_written;

            tamp_res res = tamp_compressor_compress(compressor, output_buffer, sizeof(output_buffer), &chunk_written,
                                                    input_buffer + input_pos, bytes_read - input_pos, &chunk_consumed);
            if (TAMP_UNLIKELY(res < TAMP_OK)) return res;

            input_pos += chunk_consumed;

            if (TAMP_LIKELY(chunk_written > 0)) {
                int bytes_written = write_cb(write_handle, output_buffer, chunk_written);
                if (TAMP_UNLIKELY(bytes_written < 0 || (size_t)bytes_written != chunk_written)) {
                    return TAMP_WRITE_ERROR;
                }
                *output_written_size += chunk_written;
            }
        }

        if (TAMP_UNLIKELY(callback)) {
            int cb_res = callback(user_data, *input_consumed_size, 0);
            if (TAMP_UNLIKELY(cb_res)) return (tamp_res)cb_res;
        }
    }

    // Flush remaining data
    while (1) {
        size_t chunk_written;
        tamp_res res = tamp_compressor_flush(compressor, output_buffer, sizeof(output_buffer), &chunk_written, false);
        if (TAMP_UNLIKELY(res < TAMP_OK)) return res;

        if (TAMP_LIKELY(chunk_written > 0)) {
            int bytes_written = write_cb(write_handle, output_buffer, chunk_written);
            if (TAMP_UNLIKELY(bytes_written < 0 || (size_t)bytes_written != chunk_written)) {
                return TAMP_WRITE_ERROR;
            }
            *output_written_size += chunk_written;
        }

        if (res == TAMP_OK) break;
    }

    return TAMP_OK;
}

#endif /* TAMP_STREAM */

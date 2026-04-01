/**
 * @file compressor_find_match_desktop.c
 * @brief Desktop/64-bit optimized find_best_match implementation.
 *
 * NOTE: This file is #include'd by compressor.c, not compiled separately.
 *
 * This file is included by compressor.c when compiling for desktop 64-bit platforms
 * (x86_64, aarch64, Windows 64-bit). It uses bit manipulation and 64-bit loads
 * for parallel match detection.
 *
 * Requirements:
 *   - Little-endian byte order
 *   - 64-bit compiler intrinsics (__builtin_ctzll or _BitScanForward64)
 */

#include <string.h>  // for memcpy (portable unaligned loads)

// MSVC compatibility for count trailing zeros
#if defined(_MSC_VER)
#include <intrin.h>
static inline int tamp_ctzll(uint64_t value) {
    unsigned long index;
    _BitScanForward64(&index, value);
    return (int)index;
}
#else
#define tamp_ctzll(v) __builtin_ctzll(v)
#endif

// Detect zero bytes in a 64-bit word (classic bit manipulation trick)
// Note: Can have false positives when a zero byte is followed by 0x01-0x7F due to borrow propagation
#define HAS_ZERO_BYTE(v) (((v)-0x0101010101010101ULL) & ~(v) & 0x8080808080808080ULL)

// Helper macro to extend a match and update best match
// Uses pre-computed input_word_ext and input_bytes to avoid repeated read_input() calls
#define EXTEND_MATCH(idx)                                                           \
    do {                                                                            \
        uint8_t match_len = 2;                                                      \
        uint8_t i = 2;                                                              \
        int extend_done = 0;                                                        \
                                                                                    \
        /* Try 8-byte comparison for bytes 2-9 using pre-computed input_word_ext */ \
        if (max_pattern_size >= 10 && (idx + 9) <= window_size_minus_1) {           \
            uint64_t window_word;                                                   \
            memcpy(&window_word, window + idx + 2, sizeof(uint64_t));               \
            uint64_t diff = window_word ^ input_word_ext;                           \
            if (diff == 0) {                                                        \
                match_len = 10;                                                     \
                i = 10;                                                             \
            } else {                                                                \
                match_len = 2 + (tamp_ctzll(diff) >> 3);                            \
                extend_done = 1;                                                    \
            }                                                                       \
        }                                                                           \
                                                                                    \
        /* Byte-by-byte for remainder using pre-loaded input_bytes */               \
        if (!extend_done) {                                                         \
            for (; i < max_pattern_size; i++) {                                     \
                if (TAMP_UNLIKELY((idx + i) > window_size_minus_1)) break;          \
                if (TAMP_LIKELY(window[idx + i] != input_bytes[i])) break;          \
                match_len = i + 1;                                                  \
            }                                                                       \
        }                                                                           \
                                                                                    \
        if (TAMP_UNLIKELY(match_len > *match_size)) {                               \
            *match_size = match_len;                                                \
            *match_index = idx;                                                     \
            if (TAMP_UNLIKELY(*match_size == max_pattern_size)) return;             \
        }                                                                           \
    } while (0)

/**
 * @brief Find the best match for the current input buffer.
 *
 * Desktop/64-bit implementation: uses bit manipulation to detect multiple 2-byte matches
 * simultaneously within 64-bit words.
 *
 * @param[in,out] compressor TampCompressor object to perform search on.
 * @param[out] match_index  If match_size is 0, this value is undefined.
 * @param[out] match_size Size of best found match.
 */
static inline void find_best_match(TampCompressor *compressor, uint16_t *match_index, uint8_t *match_size) {
    *match_size = 0;

    if (TAMP_UNLIKELY(compressor->input_size < compressor->min_pattern_size)) return;

    const uint16_t window_size_minus_1 = WINDOW_SIZE - 1;
    const uint8_t max_pattern_size = MIN(compressor->input_size, MAX_PATTERN_SIZE);
    const unsigned char *window = compressor->window;

    // Pre-load input bytes into linear array to avoid repeated modular arithmetic
    uint8_t input_bytes[16];
    for (uint8_t i = 0; i < max_pattern_size && i < 16; i++) {
        input_bytes[i] = read_input(i);
    }

    // Little-endian format to match direct memory loads
    const uint16_t first_second = input_bytes[0] | (input_bytes[1] << 8);

    // Broadcast patterns for detecting first_byte and second_byte in parallel
    const uint64_t first_pattern = 0x0101010101010101ULL * input_bytes[0];
    const uint64_t second_pattern = 0x0101010101010101ULL * input_bytes[1];

    // Pre-compute 64-bit input word for extension (bytes 2-9)
    const uint64_t input_word_ext = (uint64_t)input_bytes[2] | ((uint64_t)input_bytes[3] << 8) |
                                    ((uint64_t)input_bytes[4] << 16) | ((uint64_t)input_bytes[5] << 24) |
                                    ((uint64_t)input_bytes[6] << 32) | ((uint64_t)input_bytes[7] << 40) |
                                    ((uint64_t)input_bytes[8] << 48) | ((uint64_t)input_bytes[9] << 56);

    uint16_t window_index = 0;

    // Main loop: check 14 positions per iteration using two 64-bit loads
    // Uses bit manipulation to find all 2-byte matches simultaneously
    for (; window_index + 15 < window_size_minus_1; window_index += 14) {
        uint64_t word1, word2;
        memcpy(&word1, window + window_index, sizeof(uint64_t));
        memcpy(&word2, window + window_index + 7, sizeof(uint64_t));

        // Find 2-byte matches in word1 (positions 0-6)
        // XOR with broadcast patterns to find matching bytes, then detect zeros
        uint64_t first_zeros1 = HAS_ZERO_BYTE(word1 ^ first_pattern);
        uint64_t second_zeros1 = HAS_ZERO_BYTE(word1 ^ second_pattern);
        // A 2-byte match at position i requires first_byte at i AND second_byte at i+1
        uint64_t matches1 = first_zeros1 & (second_zeros1 >> 8);

        // Process all matches found in word1
        // Note: HAS_ZERO_BYTE can have false positives, so verify before extending
        while (matches1) {
            int byte_pos = tamp_ctzll(matches1) >> 3;
            uint16_t candidate;
            memcpy(&candidate, window + window_index + byte_pos, sizeof(uint16_t));
            if (TAMP_UNLIKELY(candidate == first_second)) {
                EXTEND_MATCH(window_index + byte_pos);
            }
            matches1 &= matches1 - 1;  // Clear lowest set bit
        }

        // Find 2-byte matches in word2 (positions 7-13)
        uint64_t first_zeros2 = HAS_ZERO_BYTE(word2 ^ first_pattern);
        uint64_t second_zeros2 = HAS_ZERO_BYTE(word2 ^ second_pattern);
        uint64_t matches2 = first_zeros2 & (second_zeros2 >> 8);

        while (matches2) {
            int byte_pos = tamp_ctzll(matches2) >> 3;
            uint16_t candidate;
            memcpy(&candidate, window + window_index + 7 + byte_pos, sizeof(uint16_t));
            if (TAMP_UNLIKELY(candidate == first_second)) {
                EXTEND_MATCH(window_index + 7 + byte_pos);
            }
            matches2 &= matches2 - 1;
        }
    }

    // Handle remaining positions
    for (; window_index < window_size_minus_1; window_index++) {
        uint16_t candidate;
        memcpy(&candidate, window + window_index, sizeof(uint16_t));
        if (TAMP_LIKELY(candidate != first_second)) {
            continue;
        }
        EXTEND_MATCH(window_index);
    }
}

#undef EXTEND_MATCH
#undef HAS_ZERO_BYTE
#if !defined(_MSC_VER)
#undef tamp_ctzll
#endif

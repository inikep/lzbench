#ifndef TAMP_COMMON_H
#define TAMP_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

/* Compiler branch optimizations */
#if defined(__clang__) || (defined(__GNUC__) && (__GNUC__ > 2))
#define TAMP_LIKELY(c) (__builtin_expect(!!(c), 1))
#define TAMP_UNLIKELY(c) (__builtin_expect(!!(c), 0))
#else
#define TAMP_LIKELY(c) (c)
#define TAMP_UNLIKELY(c) (c)
#endif

enum {
    /* Normal/Recoverable status >= 0 */
    TAMP_OK = 0,
    TAMP_OUTPUT_FULL = 1,  // Wasn't able to complete action due to full output buffer.
    TAMP_INPUT_EXHAUSTED = 2, // Wasn't able to complete action due to exhausted input buffer.

    /* Error codes < 0 */
    TAMP_ERROR = -1,  // Generic error
    TAMP_EXCESS_BITS = -2,  // Provided symbol has more bits than conf->literal
    TAMP_INVALID_CONF = -3,  // Invalid configuration parameters.
};
typedef int8_t tamp_res;

typedef struct TampConf {
    uint16_t window:4;   // number of window bits
    uint16_t literal:4;  // number of literal bits
    uint16_t use_custom_dictionary:1;  // Use a custom initialized dictionary.
} TampConf;

/**
 * @brief Pre-populate a window buffer with common characters.
 *
 * @param[out] buffer Populated output buffer.
 * @param[in] size Size of output buffer in bytes.
 */
void tamp_initialize_dictionary(unsigned char *buffer, size_t size);

/**
 * @brief Compute the minimum viable pattern size given window and literal config parameters.
 *
 * @param[in] window Number of window bits. Valid values are [8, 15].
 * @param[in] literal Number of literal bits. Valid values are [5, 8].
 *
 * @return The minimum pattern size in bytes. Either 2 or 3.
 */
int8_t tamp_compute_min_pattern_size(uint8_t window, uint8_t literal);

#ifdef __cplusplus
}
#endif

#endif

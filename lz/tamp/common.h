#ifndef TAMP_COMMON_H
#define TAMP_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#if ESP_PLATFORM
// (External) code #including this header MUST use the SAME TAMP_ESP32 setting that is used when
// building this lib!
#include "sdkconfig.h"
#endif

/* Should the ESP32-optimized variant be built? */
#ifdef CONFIG_TAMP_ESP32  // CONFIG_... from Kconfig takes precedence
#if CONFIG_TAMP_ESP32
#define TAMP_ESP32 1
#else
#define TAMP_ESP32 0
#endif
#endif

#ifndef TAMP_ESP32  // If not set via Kconfig, and not otherwise -D_efined, default TAMP_ESP32 to
                    // compatible version.
#define TAMP_ESP32 0
#endif

/* Compiler branch optimizations */
#if defined(__clang__) || (defined(__GNUC__) && (__GNUC__ > 2))
#define TAMP_LIKELY(c) (__builtin_expect(!!(c), 1))
#define TAMP_UNLIKELY(c) (__builtin_expect(!!(c), 0))
#else
#define TAMP_LIKELY(c) (c)
#define TAMP_UNLIKELY(c) (c)
#endif

#if defined(_MSC_VER)
#define TAMP_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define TAMP_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define TAMP_ALWAYS_INLINE inline
#endif

enum {
    /* Normal/Recoverable status >= 0 */
    TAMP_OK = 0,
    TAMP_OUTPUT_FULL = 1,      // Wasn't able to complete action due to full output buffer.
    TAMP_INPUT_EXHAUSTED = 2,  // Wasn't able to complete action due to exhausted input buffer.

    /* Error codes < 0 */
    TAMP_ERROR = -1,         // Generic error
    TAMP_EXCESS_BITS = -2,   // Provided symbol has more bits than conf->literal
    TAMP_INVALID_CONF = -3,  // Invalid configuration parameters.
};
typedef int8_t tamp_res;

typedef struct TampConf {
    uint16_t window : 4;                 // number of window bits
    uint16_t literal : 4;                // number of literal bits
    uint16_t use_custom_dictionary : 1;  // Use a custom initialized dictionary.
#if TAMP_LAZY_MATCHING
    uint16_t lazy_matching : 1;  // use Lazy Matching (spend 50-75% more CPU for around 0.5-2.0% better compression.)
                                 // only effects compression operations.
#endif
} TampConf;

/**
 * User-provied callback to be invoked after each compression cycle in the higher-level API.
 * @param[in,out] user_data Arbitrary user-provided data.
 * @param[in] bytes_processed Number of input bytes consumed so far.
 * @param[in] total_bytes Total number of input bytes.
 *
 * @return Some error code. If non-zero, abort current compression and return the value.
 *         For clarity, is is recommend to avoid already-used tamp_res values.
 *         e.g. start custom error codes at 100.
 */
typedef int (*tamp_callback_t)(void *user_data, size_t bytes_processed, size_t total_bytes);

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

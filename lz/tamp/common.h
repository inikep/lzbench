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
// cppcheck-suppress missingInclude
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

/* Per-function optimize attributes and #pragma GCC push/pop_options require
 * GCC on a target that supports them. Xtensa GCC does not. */
#if defined(__GNUC__) && !defined(__clang__) && !defined(__XTENSA__)
#define TAMP_HAS_GCC_OPTIMIZE 1
#else
#define TAMP_HAS_GCC_OPTIMIZE 0
#endif

#if defined(_MSC_VER)
#define TAMP_ALWAYS_INLINE __forceinline
#define TAMP_NOINLINE __declspec(noinline)
#define TAMP_OPTIMIZE_SIZE /* not supported */
#elif defined(__GNUC__) && !defined(__clang__)
#define TAMP_ALWAYS_INLINE inline __attribute__((always_inline))
#define TAMP_NOINLINE __attribute__((noinline))
#if TAMP_HAS_GCC_OPTIMIZE
#define TAMP_OPTIMIZE_SIZE __attribute__((optimize("Os")))
#else
#define TAMP_OPTIMIZE_SIZE
#endif
#elif defined(__clang__)
#define TAMP_ALWAYS_INLINE inline __attribute__((always_inline))
#define TAMP_NOINLINE __attribute__((noinline))
#define TAMP_OPTIMIZE_SIZE /* clang doesn't support per-function optimize */
#else
#define TAMP_ALWAYS_INLINE inline
#define TAMP_NOINLINE
#define TAMP_OPTIMIZE_SIZE
#endif

/* TAMP_USE_MEMSET: Use libc memset (default: 1).
 * Set to 0 for environments without libc (e.g. MicroPython native modules).
 * When disabled, uses a volatile loop that prevents GCC from emitting a
 * memset call at the cost of inhibiting store coalescing. */
#ifndef TAMP_USE_MEMSET
#define TAMP_USE_MEMSET 1
#endif

#if TAMP_USE_MEMSET
#include <string.h>
#define TAMP_MEMSET(dst, val, n) memset((dst), (val), (n))
#else
#define TAMP_MEMSET(dst, val, n)                                                     \
    do {                                                                             \
        volatile unsigned char *_tamp_p = (volatile unsigned char *)(dst);           \
        for (size_t _tamp_i = 0; _tamp_i < (n); _tamp_i++) _tamp_p[_tamp_i] = (val); \
    } while (0)
#endif

/* Include stream API (tamp_compress_stream, tamp_decompress_stream).
 * Enabled by default. Disable with -DTAMP_STREAM=0 to save ~2.8KB.
 */
#ifndef TAMP_STREAM
#define TAMP_STREAM 1
#endif

/* Work buffer size for stream API functions.
 * The buffer is allocated on the stack and split in half for input/output.
 * Larger values reduce I/O callback invocations, improving decompression speed.
 * Default of 32 bytes is safe for constrained stacks; 256+ bytes recommended
 * for better performance when stack space permits.
 * Override via compiler flag: -DTAMP_STREAM_WORK_BUFFER_SIZE=256
 */
#ifndef TAMP_STREAM_WORK_BUFFER_SIZE
#define TAMP_STREAM_WORK_BUFFER_SIZE 32
#endif

/* Extended format support (RLE, extended match).
 * Enabled by default. Disable to save code size on minimal builds.
 *
 * TAMP_EXTENDED is the master switch (default: 1).
 * TAMP_EXTENDED_COMPRESS and TAMP_EXTENDED_DECOMPRESS default to TAMP_EXTENDED,
 * but can be individually overridden for compressor-only or decompressor-only builds.
 */
#ifndef TAMP_EXTENDED
#define TAMP_EXTENDED 1
#endif
#ifndef TAMP_EXTENDED_DECOMPRESS
#define TAMP_EXTENDED_DECOMPRESS TAMP_EXTENDED
#endif
#ifndef TAMP_EXTENDED_COMPRESS
#define TAMP_EXTENDED_COMPRESS TAMP_EXTENDED
#endif

/* Extended encoding constants */
#if TAMP_EXTENDED_DECOMPRESS || TAMP_EXTENDED_COMPRESS
#define TAMP_RLE_SYMBOL 12
#define TAMP_EXTENDED_MATCH_SYMBOL 13
#define TAMP_LEADING_EXTENDED_MATCH_BITS 3
#define TAMP_LEADING_RLE_BITS 4
#define TAMP_RLE_MAX_WINDOW 8
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
    TAMP_OOB = -4,           // Out-of-bounds access detected in compressed data.
                             // Indicates malicious or corrupted input data attempting to
                             // reference memory outside the decompressor window buffer.

    /* Stream I/O error codes */
    TAMP_IO_ERROR = -10,     // Generic I/O error from read/write callback
    TAMP_READ_ERROR = -11,   // Read callback returned error
    TAMP_WRITE_ERROR = -12,  // Write callback returned error

    /* [100, 127] and [-128, -100] are reserved for user-defined callback codes.
     * tamp_callback_t returns are truncated to int8_t; use these ranges to
     * avoid collisions with library status codes. */
};
typedef int8_t tamp_res;

typedef struct TampConf {
    uint16_t window : 4;                 // number of window bits
    uint16_t literal : 4;                // number of literal bits
    uint16_t use_custom_dictionary : 1;  // Use a custom initialized dictionary.
    uint16_t extended : 1;               // Extended format (RLE, extended match). Read from header bit [1].
    uint16_t dictionary_reset : 1;  // Stream may contain double-FLUSH dictionary resets. Implied by header byte 1 bit
                                    // [0] (more_header).
    uint16_t append : 1;            // Initialize for appending to an existing stream (FLUSH instead of header).
#if TAMP_LAZY_MATCHING
    uint16_t lazy_matching : 1;  // use Lazy Matching (spend 50-75% more CPU for around 0.5-2.0% better compression.)
                                 // only effects compression operations.
#endif
} TampConf;

/**
 * User-provided callback to be invoked periodically by the higher-level API.
 *
 * The callback fires once per compression/decompression cycle (i.e., once per
 * encoded or decoded token). For the stream API, it fires once per read-chunk.
 *
 * In all contexts, bytes_processed tracks input bytes consumed and total_bytes
 * is the total input size (or 0 if unknown). This allows computing a meaningful
 * progress percentage as bytes_processed / total_bytes.
 *
 * Non-stream API (total_bytes is known):
 *   tamp_compressor_compress_cb:   (input_consumed, total_input_size)
 *   tamp_decompressor_decompress_cb: (input_consumed, total_input_size)
 *
 * Stream API (total_bytes is 0; input size unknown):
 *   tamp_compress_stream:   (input_consumed, 0)
 *   tamp_decompress_stream: (input_consumed, 0)
 *
 * @param[in,out] user_data Arbitrary user-provided data.
 * @param[in] bytes_processed Input bytes consumed so far.
 * @param[in] total_bytes Total input size, or 0 if unknown (stream API).
 *
 * @return 0 to continue, or non-zero to abort. The return value is truncated
 *         to tamp_res (int8_t) and propagated to the caller. Use values in
 *         [100, 127] or [-128, -100] for custom codes to avoid collisions.
 */
typedef int (*tamp_callback_t)(void *user_data, size_t bytes_processed, size_t total_bytes);

/**
 * Stream read callback type for file/stream-based operations.
 *
 * Should behave like fread(): read up to `size` bytes into `buffer`.
 * Returns plain int (not tamp_res) for compatibility with standard I/O functions.
 * The stream API translates negative returns to TAMP_READ_ERROR.
 *
 * @param[in] handle User-provided handle (e.g., FILE*, lfs_file_t*, FIL*)
 * @param[out] buffer Buffer to read data into
 * @param[in] size Maximum number of bytes to read
 *
 * @return Number of bytes actually read (0 for EOF), or negative (e.g., -1) on error
 */
typedef int (*tamp_read_t)(void *handle, unsigned char *buffer, size_t size);

/**
 * Stream write callback type for file/stream-based operations.
 *
 * Should behave like fwrite(): write `size` bytes from `buffer`.
 * Returns plain int (not tamp_res) for compatibility with standard I/O functions.
 * The stream API treats negative returns or incomplete writes (fewer bytes than requested)
 * as TAMP_WRITE_ERROR. Chunks are small (at most TAMP_STREAM_WORK_BUFFER_SIZE/2 bytes),
 * so writing the full amount is expected.
 *
 * @param[in] handle User-provided handle (e.g., FILE*, lfs_file_t*, FIL*)
 * @param[in] buffer Buffer containing data to write
 * @param[in] size Number of bytes to write
 *
 * @return `size` on success, or negative (e.g., -1) on error
 */
typedef int (*tamp_write_t)(void *handle, const unsigned char *buffer, size_t size);

/*******************************************************************************
 * Built-in I/O handlers for common sources/sinks.
 *
 * Enable the ones you need by defining the appropriate macro in your build
 * system (e.g., -DTAMP_STREAM_STDIO=1):
 *
 *   - TAMP_STREAM_MEMORY  : Memory buffers (always available, no dependencies)
 *   - TAMP_STREAM_STDIO   : Standard C FILE* (POSIX, ESP-IDF VFS, etc.)
 *   - TAMP_STREAM_LITTLEFS: LittleFS filesystem
 *   - TAMP_STREAM_FATFS   : FatFs (ChaN's FAT filesystem)
 ******************************************************************************/

/* Memory buffer I/O */
#if TAMP_STREAM_MEMORY

/**
 * @brief Reader state for memory buffer input.
 *
 * Example:
 * @code
 * const unsigned char compressed_data[] = {...};
 * TampMemReader reader = {
 *     .data = compressed_data,
 *     .size = sizeof(compressed_data),
 *     .pos = 0
 * };
 * tamp_decompress_stream(tamp_stream_mem_read, &reader, ...);
 * @endcode
 */
typedef struct TampMemReader {
    const unsigned char *data; /**< Pointer to input data */
    size_t size;               /**< Total size of input data */
    size_t pos;                /**< Current read position (initialize to 0) */
} TampMemReader;

/**
 * @brief Writer state for memory buffer output.
 *
 * Example:
 * @code
 * unsigned char output[4096];
 * TampMemWriter writer = {
 *     .data = output,
 *     .capacity = sizeof(output),
 *     .pos = 0
 * };
 * tamp_compress_stream(..., tamp_stream_mem_write, &writer, ...);
 * // writer.pos contains bytes written
 * @endcode
 */
typedef struct TampMemWriter {
    unsigned char *data; /**< Pointer to output buffer */
    size_t capacity;     /**< Total capacity of output buffer */
    size_t pos;          /**< Current write position (initialize to 0) */
} TampMemWriter;

/**
 * @brief Read callback for memory buffers.
 * @param handle Pointer to TampMemReader.
 */
int tamp_stream_mem_read(void *handle, unsigned char *buffer, size_t size);

/**
 * @brief Write callback for memory buffers.
 * @param handle Pointer to TampMemWriter.
 * @return Bytes written, or -1 if buffer would overflow.
 */
int tamp_stream_mem_write(void *handle, const unsigned char *buffer, size_t size);

#endif /* TAMP_STREAM_MEMORY */

/* POSIX / Standard C stdio (FILE*) */
#if TAMP_STREAM_STDIO

/**
 * @brief Read callback for stdio FILE*.
 * @param handle FILE* opened for reading.
 */
int tamp_stream_stdio_read(void *handle, unsigned char *buffer, size_t size);

/**
 * @brief Write callback for stdio FILE*.
 * @param handle FILE* opened for writing.
 */
int tamp_stream_stdio_write(void *handle, const unsigned char *buffer, size_t size);

#endif /* TAMP_STREAM_STDIO */

/* LittleFS */
#if TAMP_STREAM_LITTLEFS

#include "lfs.h"

/**
 * @brief Bundle struct for LittleFS file operations.
 *
 * LittleFS API requires both the filesystem context and file handle.
 */
typedef struct TampLfsFile {
    lfs_t *lfs;       /**< Pointer to mounted LittleFS instance */
    lfs_file_t *file; /**< Pointer to opened file handle */
} TampLfsFile;

/**
 * @brief Read callback for LittleFS.
 * @param handle Pointer to TampLfsFile.
 */
int tamp_stream_lfs_read(void *handle, unsigned char *buffer, size_t size);

/**
 * @brief Write callback for LittleFS.
 * @param handle Pointer to TampLfsFile.
 */
int tamp_stream_lfs_write(void *handle, const unsigned char *buffer, size_t size);

#endif /* TAMP_STREAM_LITTLEFS */

/* FatFs (ChaN's FAT Filesystem) */
#if TAMP_STREAM_FATFS

#include "ff.h"

/**
 * @brief Read callback for FatFs.
 * @param handle Pointer to FIL (FatFs file object).
 */
int tamp_stream_fatfs_read(void *handle, unsigned char *buffer, size_t size);

/**
 * @brief Write callback for FatFs.
 * @param handle Pointer to FIL (FatFs file object).
 */
int tamp_stream_fatfs_write(void *handle, const unsigned char *buffer, size_t size);

#endif /* TAMP_STREAM_FATFS */

/**
 * @brief Pre-populate a window buffer with common characters.
 *
 * Uses a per-literal-size seed table so the dictionary only contains bytes that
 * are valid and useful for the given configuration:
 *   - literal=7,8: common english text/markup characters (" \0 0 e i > t o < a n s \\n r / .")
 *   - literal=5,6: common english letters (" etaoinshrdlcumw") downshifted to the target bit width
 *
 * For v1 backwards compatibility, callers should pass literal=8 when the
 * extended header flag is not set, regardless of the configured literal value.
 *
 * @param[out] buffer Populated output buffer.
 * @param[in] size Size of output buffer in bytes.
 * @param[in] literal Number of literal bits (5-8). Selects the appropriate seed character table.
 */
void tamp_initialize_dictionary(unsigned char *buffer, size_t size, uint8_t literal);

/**
 * @brief Compute the minimum viable pattern size given window and literal config parameters.
 *
 * @param[in] window Number of window bits. Valid values are [8, 15].
 * @param[in] literal Number of literal bits. Valid values are [5, 8].
 *
 * @return The minimum pattern size in bytes. Either 2 or 3.
 */
int8_t tamp_compute_min_pattern_size(uint8_t window, uint8_t literal);

/**
 * @brief Copy pattern from window to window, updating window_pos.
 *
 * Handles potential overlap between source and destination regions by
 * copying backwards when the destination would "catch up" to the source.
 *
 * IMPORTANT: Caller must validate that (window_offset + match_size) does not
 * exceed window bounds before calling this function. This function assumes
 * window_offset and match_size are pre-validated and does not perform
 * bounds checking on source reads.
 *
 * @param window Circular buffer (size must be power of 2)
 * @param window_pos Current write position (updated by this function)
 * @param window_offset Source position to copy from
 * @param match_size Number of bytes to copy
 * @param window_mask Bitmask for wrapping (window_size - 1)
 */
void tamp_window_copy(unsigned char *window, uint16_t *window_pos, uint16_t window_offset, uint8_t match_size,
                      uint16_t window_mask);

#ifdef __cplusplus
}
#endif

#endif

#include "common.h"

#if TAMP_STREAM && TAMP_STREAM_WORK_BUFFER_SIZE < 4
#error "TAMP_STREAM_WORK_BUFFER_SIZE must be at least 4 bytes"
#endif

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#if TAMP_STREAM_STDIO
#include <stdio.h>
#endif

/* Per-literal-size seed tables.  All 16 entries must be unique and fit within
 * (1 << literal) - 1.  literal=7,8 share the original table (all < 0x80). */
// clang-format off
static const unsigned char common_characters_8[] = {' ', 0, '0', 'e', 'i', '>', 't', 'o',
                                                    '<', 'a', 'n', 's', '\n', 'r', '/', '.'};
/* Common English characters, downshifted to 6 bits */
static const unsigned char common_characters_6[] = {' ' & 0x3F, 'e' & 0x3F, 't' & 0x3F, 'a' & 0x3F, 'o' & 0x3F, 'i' & 0x3F, 'n' & 0x3F, 's' & 0x3F,
                                                    'h' & 0x3F, 'r' & 0x3F, 'd' & 0x3F, 'l' & 0x3F, 'c' & 0x3F, 'u' & 0x3F, 'm' & 0x3F, 'w' & 0x3F};
/* Common English characters, downshifted to 5 bits */
static const unsigned char common_characters_5[] = {' ' & 0x1F, 'e' & 0x1F, 't' & 0x1F, 'a' & 0x1F, 'o' & 0x1F, 'i' & 0x1F, 'n' & 0x1F, 's' & 0x1F,
                                                    'h' & 0x1F, 'r' & 0x1F, 'd' & 0x1F, 'l' & 0x1F, 'c' & 0x1F, 'u' & 0x1F, 'm' & 0x1F, 'w' & 0x1F};
// clang-format on

static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

TAMP_OPTIMIZE_SIZE void tamp_initialize_dictionary(unsigned char *buffer, size_t size, uint8_t literal) {
    uint32_t seed = 3758097560;  // This was experimentally discovered with tools/find_seed.py
    uint32_t randbuf = 0;
    const unsigned char *chars;
    if (literal <= 5)
        chars = common_characters_5;
    else if (literal <= 6)
        chars = common_characters_6;
    else
        chars = common_characters_8;
    for (size_t i = 0; i < size; i++) {
        if (TAMP_UNLIKELY((i & 0x7) == 0)) randbuf = xorshift32(&seed);
        buffer[i] = chars[randbuf & 0x0F];
        randbuf >>= 4;
    }
}

TAMP_OPTIMIZE_SIZE int8_t tamp_compute_min_pattern_size(uint8_t window, uint8_t literal) {
    return 2 + (window > (10 + ((literal - 5) << 1)));
}

void tamp_window_copy(unsigned char *window, uint16_t *window_pos, uint16_t window_offset, uint8_t match_size,
                      uint16_t window_mask) {
    /* Calculate distance from source to destination in circular buffer.
     * src_to_dst = (dst - src) & mask gives the forward distance. */
    const uint16_t src_to_dst = (*window_pos - window_offset) & window_mask;

    /* Critical overlap case: destination is AHEAD of source and they overlap.
     * When dst > src by less than match_size, a forward copy corrupts data because
     * we write to positions before reading from them.
     *
     * Example: src=100, dst=105, match_size=8
     *   - Forward copy at i=5 would read window[105], but we already overwrote it at i=0!
     *   - Must copy in REVERSE order (end to start) to read source bytes before overwriting.
     */
    if (TAMP_UNLIKELY(src_to_dst < match_size && src_to_dst > 0)) {
        /* Copy in reverse order: start from last byte, work backwards to first byte.
         * This ensures we read all overlapping source bytes before they're overwritten.
         * Destination wraps via mask; source doesn't need wrapping (pre-validated bounds). */
        for (uint8_t i = match_size; i-- > 0;) {
            window[(*window_pos + i) & window_mask] = window[window_offset + i];
        }
        *window_pos = (*window_pos + match_size) & window_mask;
    } else {
        for (uint8_t i = 0; i < match_size; i++) {
            window[*window_pos] = window[window_offset + i];
            *window_pos = (*window_pos + 1) & window_mask;
        }
    }
}

/*******************************************************************************
 * Built-in I/O handler implementations
 ******************************************************************************/

#if TAMP_STREAM_MEMORY

int tamp_stream_mem_read(void *handle, unsigned char *buffer, size_t size) {
    TampMemReader *r = (TampMemReader *)handle;
    size_t available = r->size - r->pos;
    size_t to_read = (size < available) ? size : available;
    if (to_read > (size_t)INT_MAX) to_read = (size_t)INT_MAX;
    memcpy(buffer, r->data + r->pos, to_read);
    r->pos += to_read;
    return (int)to_read;
}

int tamp_stream_mem_write(void *handle, const unsigned char *buffer, size_t size) {
    TampMemWriter *w = (TampMemWriter *)handle;
    size_t available = w->capacity - w->pos;
    if (size > available) return -1;
    if (size > (size_t)INT_MAX) return -1;
    memcpy(w->data + w->pos, buffer, size);
    w->pos += size;
    return (int)size;
}

#endif /* TAMP_STREAM_MEMORY */

#if TAMP_STREAM_STDIO

int tamp_stream_stdio_read(void *handle, unsigned char *buffer, size_t size) {
    FILE *f = (FILE *)handle;
    size_t bytes_read = fread(buffer, 1, size, f);
    if (bytes_read == 0 && ferror(f)) return -1;
    return (int)bytes_read;
}

int tamp_stream_stdio_write(void *handle, const unsigned char *buffer, size_t size) {
    FILE *f = (FILE *)handle;
    size_t bytes_written = fwrite(buffer, 1, size, f);
    if (bytes_written < size && ferror(f)) return -1;
    return (int)bytes_written;
}

#endif /* TAMP_STREAM_STDIO */

#if TAMP_STREAM_LITTLEFS

int tamp_stream_lfs_read(void *handle, unsigned char *buffer, size_t size) {
    TampLfsFile *f = (TampLfsFile *)handle;
    lfs_ssize_t result = lfs_file_read(f->lfs, f->file, buffer, size);
    return (int)result;
}

int tamp_stream_lfs_write(void *handle, const unsigned char *buffer, size_t size) {
    TampLfsFile *f = (TampLfsFile *)handle;
    lfs_ssize_t result = lfs_file_write(f->lfs, f->file, buffer, size);
    return (int)result;
}

#endif /* TAMP_STREAM_LITTLEFS */

#if TAMP_STREAM_FATFS

int tamp_stream_fatfs_read(void *handle, unsigned char *buffer, size_t size) {
    UINT bytes_read;
    FRESULT res = f_read((FIL *)handle, buffer, (UINT)size, &bytes_read);
    if (res != FR_OK) return -1;
    return (int)bytes_read;
}

int tamp_stream_fatfs_write(void *handle, const unsigned char *buffer, size_t size) {
    UINT bytes_written;
    FRESULT res = f_write((FIL *)handle, buffer, (UINT)size, &bytes_written);
    if (res != FR_OK) return -1;
    return (int)bytes_written;
}

#endif /* TAMP_STREAM_FATFS */

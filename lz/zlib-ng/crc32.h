/* crc32.h -- crc32 folding interface
 * Copyright (C) 2021 Nathan Moinvaziri
 * For conditions of distribution and use, see copyright notice in zlib.h
 */
#ifndef CRC32_H_
#define CRC32_H_

/* sizeof(__m128i) * (4 folds) */
#define CRC32_FOLD_BUFFER_SIZE (16 * 4)

/* Size thresholds for Chorba algorithm variants */
#define CHORBA_LARGE_THRESHOLD (sizeof(z_word_t) * 64 * 1024)
#define CHORBA_MEDIUM_UPPER_THRESHOLD 32768
#define CHORBA_MEDIUM_LOWER_THRESHOLD 8192
#define CHORBA_SMALL_THRESHOLD_64BIT 72
#define CHORBA_SMALL_THRESHOLD_32BIT 80

Z_INTERNAL uint32_t crc32_braid_internal(uint32_t c, const uint8_t *buf, size_t len);

typedef struct crc32_fold_s {
    uint8_t fold[CRC32_FOLD_BUFFER_SIZE];
    uint32_t value;
} crc32_fold;

#endif

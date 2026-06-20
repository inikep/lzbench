// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_DECOMPRESS_TOKENIZE2TO1_H
#define ZSTRONG_DECOMPRESS_TOKENIZE2TO1_H

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h> // size_t
#include <stdint.h> // uintx

/* ZS_tokenize2to1_decode():
 * reform an array of 2-bytes tokens into @dst16
 * from an array of 1-byte indexes in @srcIndex8
 * translated using the @alphabet16 map.
 *
 * @return : nb of 2-bytes tokens regenerated (must be == @srcIndex8Size)
 *           note : since it's the nb of 2-bytes tokens, the actual occupation
 *                  size in bytes is 2x this amount.
 *
 * Conditions : all buffers must exist (!=NULL) and be valid
 *              dst16Capacity must be >= srcIndex8Size
 *
 * Note : If, for example, the alphabet size is == 100,
 *        but one of the indexes has a value > 100 (yet necessarily <= 255)
 *        this erroneous situation will only be detected in debug mode.
 */

size_t ZS_tokenize2to1_decode(
        uint16_t* dst16,
        size_t dst16Capacity,
        const uint8_t* srcIndex8,
        size_t srcIndex8Size,
        const uint16_t* alphabet16,
        size_t alphabet16Size);

#if defined(__cplusplus)
}
#endif

#endif // ZSTRONG_COMPRESS_TRANSFORMS_DELTA_ENCODE_H

// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_DECOMPRESS_TOKENIZE4TO2_H
#define ZSTRONG_DECOMPRESS_TOKENIZE4TO2_H

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h> // size_t
#include <stdint.h> // uintx

/* ZS_tokenize4to2_decode():
 * reform an array of 4-bytes tokens into @dst32
 * from an array of 2-byte indexes in @srcIndex16
 * translated using the @alphabet32 map.
 *
 * @return : nb of 4-bytes tokens regenerated (must be == @srcIndex16Size)
 *           note : since it's the nb of 4-bytes tokens, the actual occupation
 * size in bytes is 4x this amount.
 *
 * Conditions : all buffers must exist (!=NULL) and be valid
 *              dst32Capacity must be >= srcIndex16Size
 *
 * Note : assume input to be correct;
 *        is not protected vs malicious inputs (yet) (well, just an assert())
 */

size_t ZS_tokenize4to2_decode(
        uint32_t* dst32,
        size_t dst32Capacity,
        const uint16_t* srcIndex16,
        size_t srcIndex16Size,
        const uint32_t* alphabet32,
        size_t alphabet32Size);

#if defined(__cplusplus)
}
#endif

#endif // ZSTRONG_DECOMPRESS_TOKENIZE4TO2_H

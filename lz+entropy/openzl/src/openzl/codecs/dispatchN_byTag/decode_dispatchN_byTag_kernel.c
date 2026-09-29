// Copyright (c) Meta Platforms, Inc. and affiliates.

// Implementation note :
// raw transform == minimal dependency
#include "decode_dispatchN_byTag_kernel.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/numeric_operations.h" // NUMOP_*

#include <assert.h>
#include <string.h> // memcpy

static size_t sumArrayST(const size_t array[], size_t arraySize)
{
    size_t total = 0;
    for (size_t n = 0; n < arraySize; n++) {
        total += array[n];
    }
    return total;
}

/* Implementation notes :
 * This is a first generic implementation.
 * - It requires all inputs to be valid (see assert() below)
 *   In which case, it's necessarily successful.
 * - A different design would be to validate some or all input conditions,
 *   returning an error if input validation fails.
 *   This topic will probably come back, especially during fuzzer tests.
 * - The content of @srcs[] input array is modified by this function.
 *   This is documented, but this could be surprising for users.
 *   An alternate strategy could be to copy this array and modify the copy.
 *   But this requires to allocate space for this array.
 *   This could be done on stack for small array, but wouldn't be generic.
 *   We try to avoid any kind of dynamic memory allocation in raw transforms.
 *   It seems preferable to require this copy operation on the user side,
 *   (aka the Decoder Interface) where dynamic Allocation is available.
 * - These memcpy() invocation use variable sizes.
 *   If input consists of a few large segments, that's fine, no big deal.
 *   But if it consists of a lot of small data,
 *   for example in the 1-8 bytes ranges,
 *   then the overhead will be significant.
 *   More optimized copy strategies could be implemented if need be.
 */
size_t ZL_dispatchN_byTag_decode(
        void* restrict dst,
        size_t dstCapacity,
        const void* restrict srcs[],
        size_t nbSrcs,
        const size_t* restrict segmentSizes,
        const uint16_t* restrict bufIndex,
        size_t nbSegments)
{
    /* input validation */
    size_t const dstSize = sumArrayST(segmentSizes, nbSegments);
    assert(dstSize <= dstCapacity);
    if (dstCapacity) {
        assert(dst != NULL);
    }
    assert(srcs != NULL);
    for (size_t n = 0; n < nbSrcs; n++) {
        assert(srcs[n] != NULL);
    }
    assert(nbSrcs < 1 << 16);
    assert(NUMOP_underLimitU16(bufIndex, nbSegments, (uint16_t)nbSrcs));

    for (size_t n = 0; n < nbSegments; n++) {
        size_t const segSize = segmentSizes[n];
        uint16_t const tag   = bufIndex[n];
        switch (segSize) {
            case 1:
                ZL_write8(dst, ZL_read8(srcs[tag]));
                break;
            case 2:
                ZL_write16(dst, ZL_read16(srcs[tag]));
                break;
            case 4:
                ZL_write32(dst, ZL_read32(srcs[tag]));
                break;
            case 8:
                ZS_write64(dst, ZL_read64(srcs[tag]));
                break;
            default:
                memcpy(dst, srcs[tag], segSize);
        }
        dst       = (char*)dst + segSize;
        srcs[tag] = (const char*)srcs[tag] + segSize;
    }
    return dstSize;
}

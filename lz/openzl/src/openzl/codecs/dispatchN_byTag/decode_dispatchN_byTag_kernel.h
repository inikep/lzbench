// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_DISPATCHN_BYTAG_DECODE_DISPATCHN_BYTAG_KERNEL_H
#define ZSTRONG_TRANSFORMS_DISPATCHN_BYTAG_DECODE_DISPATCHN_BYTAG_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // uintX_t

#if defined(__cplusplus)
extern "C" {
#endif

/* ZL_dispatchN_byTag_decode():
 * join segments of bytes from buffers in @src[]
 * entangling them into a single buffer @dst
 * following order instructions from @indexBuffer.
 *
 * CONDITIONS :
 * - all values in @bufIndex must be < @nbSrcs
 * - @dstCapacity must be large enough, aka >= sum(@segmentSizes[])
 *
 * SIDE EFFECTS :
 *  @return : size written into @dst (<= @dstCapacity)
 *  Arrays @srcs will be modified, with pointers placed at the end
 *
 * Note : this function currently doesn't fail if its conditions are respected.
 * That might prove difficult to guarantee, notably for @bufIndex.
 * In which case, it would be possible to return an error,
 * by returning a value > dstCapacity.
 */

size_t ZL_dispatchN_byTag_decode(
        void* restrict dst,
        size_t dstCapacity,
        const void* restrict srcs[],
        size_t nbSrcs,
        const size_t* restrict segmentSizes,
        const uint16_t* restrict tags,
        size_t nbSegments);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif

// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_DISPATCH_BY_TAG_DECODE_DISPATCH_BY_TAG_KERNEL_H
#define ZSTRONG_TRANSFORMS_DISPATCH_BY_TAG_DECODE_DISPATCH_BY_TAG_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // uintX_t

#if defined(__cplusplus)
extern "C" {
#endif

#define JOINBY_NB_SRCS_MAX 16

/**
 * Join all values from buffers in @src[]
 * entangling them into a single buffer @dst
 * following order instructions from @indexBuffer.
 * @return : total nb of elts written into @dst (== sum(@nbElts))
 *
 * Reverse of ZS_DispatchByTag_encode() operation.
 *
 * CONDITIONS :
 * - @nbSrcs <= JOINBY_NB_SRCS_MAX
 * - all values in @indexBuffer must be < @nbSrcs
 * - size of @indexBuffer array must == sum(@nbElts[])
 * - @dstCapacity must be large enough, aka >= sum(@nbElts[]) * eltSize
 */

size_t ZS_DispatchByTag_decode(
        void* restrict dst,
        size_t dstCapacity,
        const void* restrict srcs[],
        const size_t nbElts[],
        size_t nbSrcs,
        size_t eltSize,
        const uint8_t* restrict indexBuffer);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif

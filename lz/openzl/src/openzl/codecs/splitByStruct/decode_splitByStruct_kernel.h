// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_SPLITBYSTRUCT_DECODE_SPLITBYSTRUCT_KERNEL_H
#define ZSTRONG_TRANSFORMS_SPLITBYSTRUCT_DECODE_SPLITBYSTRUCT_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // uintX_t

#if defined(__cplusplus)
extern "C" {
#endif

/* ZS_dispatchArrayFixedSizeStruct_decode():
 * Reverse of ZS_dispatchArrayFixedSizeStruct() operation.
 *
 * join all values from inputs in @src[]
 * entangling them into a single buffer @dst,
 * taking one field per input @src, one at a time, in order.
 * It follows that all inputs must contain the same nb of fields.
 * @return : total nb of bytes written into @dst (== nbFields * structSize)
 *
 * CONDITIONS :
 * - @dstCapacity must be large enough, aka >= nbFields * structSize
 * - all inputs are valid and contain as many elements as advertised
 */

size_t ZS_dispatchArrayFixedSizeStruct_decode(
        void* restrict dst,
        size_t dstCapacity,
        const void* restrict srcs[],
        const size_t fieldSizes[],
        size_t nbFields,
        const size_t nbElts);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif

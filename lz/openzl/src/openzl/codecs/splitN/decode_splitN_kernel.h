// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_SPLITN_DECODE_SPLITN_KERNEL_H
#define ZSTRONG_TRANSFORMS_SPLITN_DECODE_SPLITN_KERNEL_H

#include <stddef.h> // size_t

#if defined(__cplusplus)
extern "C" {
#endif

/* ZS_appendN():
 * append content of buffers in @src[]
 * into a single destination buffer @dst
 * following order of @src[].
 * @return : total size written into @dst (== sum(@srcSizes[]))
 *
 * Reverse of ZS_splitN() operation.
 *
 * CONDITIONS :
 * - @dstCapacity must be large enough, aka >= sum(@srcSizes[])
 * - all buffers with a size > 0 must be valid, aka non NULL
 */

size_t ZS_appendN(
        void* restrict dst,
        size_t dstCapacity,
        const void* restrict srcs[],
        const size_t srcSizes[],
        size_t nbSrcs);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // ZSTRONG_TRANSFORMS_SPLITN_DECODE_SPLITN_KERNEL_H

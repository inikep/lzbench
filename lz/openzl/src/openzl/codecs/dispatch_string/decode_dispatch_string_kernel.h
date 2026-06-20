// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_DISPATCH_STRING_DECODE_DISPATCH_STRING_KERNEL_H
#define ZSTRONG_TRANSFORMS_DISPATCH_STRING_DECODE_DISPATCH_STRING_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // uint8_t, uint32_t

#if defined(__cplusplus)
#    define restrict // not a keyword in C++
extern "C" {
#endif

/**
 * Joins the values from @srcBuffers into a single buffer @dst following the
 * order given in @inputIndices. The reverse of ZL_DispatchString_encode().
 *
 * Assumptions:
 * - @dst is valid and large enough to contain the concatenation of all
 * @srcBuffers, and an additional 32 bytes of padding
 * - @dstStrLens is large enough to fit all the values in @srcStrLens
 * - @dstNbStrs is the size of @dstStrLens, equal to the sum of @srcNbStrs[i]
 * - @srcBuffers is valid and contains @nbSrcs pointers, each to a buffer s.t.
 * @srcBuffers[i] contains enough bytes for the sum of the string lengths
 * defined in @srcStrLens[i]
 * - @srcStrLens is valid and contains @nbSrcs pointers, each to an array of
 * positive values s.t. @srcStrLen[i] has at least as many elements as required
 * by @inputIndices
 * - @srcNbStrs[i] is the number of such elements in @srcStrLens[i], as required
 * by @inputIndices
 * - All values in @inputIndices are in range [0, @nbSrcs)
 *
 * N.B.:
 *   It is valid to provide a completely empty @srcBuffers (aka @nbSrcs == 0).
 *   However, @inputIndices must then not contain any values (aka @dstNbStrs ==
 *   0) since any values would automatically fall outside the permitted range.
 *
 *   The converse is not true. @inputIndices may be empty without @srcBuffers
 *   being empty. In this case, the kernel will essentially no-op and @dst will
 *   be empty.
 */
void ZL_DispatchString_decode(
        void* restrict dst,
        uint32_t dstStrLens[],
        size_t dstNbStrs,
        const uint8_t nbSrcs,
        const char* const* restrict srcBuffers,
        const uint32_t* const* restrict srcStrLens,
        const size_t srcNbStrs[],
        const uint8_t inputIndices[]);

void ZL_DispatchString_decode16(
        void* restrict dst,
        uint32_t dstStrLens[],
        size_t dstNbStrs,
        const uint16_t nbSrcs,
        const char* const* restrict srcBuffers,
        const uint32_t* const* restrict srcStrLens,
        const size_t srcNbStrs[],
        const uint16_t inputIndices[]);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif

// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_DISPATCH_STRING_ENCODE_DISPATCH_STRING_KERNEL_H
#define ZSTRONG_TRANSFORMS_DISPATCH_STRING_ENCODE_DISPATCH_STRING_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // uint32_t

#if defined(__cplusplus)
#    define restrict // not a keyword in C++
extern "C" {
#endif

/**
 * Dispatches input @src containing @nbStrs strings, each of size @srcStrLens[i]
 * into @nbDsts buffers, whose starting positions are stored in @dstBuffers.
 * Dispatch is controlled by @outputIndices, which is presumed valid and
 * contains at least @nbStrs values between 0 and @nbDsts - 1, inclusive.
 *
 * Assumptions:
 * - @srcStrLens is valid and contains at least @nbStrs positive values
 * - @src contains enough bytes for the sum of the string lengths defined in
 * @srcStrLens
 * - @dstBuffers is valid and contains @nbDsts pointers, each to a buffer which
 * is large enough to contain the concatenation of all strings assigned to it by
 * @outputIndices, plus 32 bytes of padding.
 * - @dstStrLens is valid and contains @nbDsts pointers, each to an array which
 * is large enough to record all the string sizes assigned to it by
 * @outputIndices
 * - @dstNbStrs contains space for @nbDsts values
 * - All values in @outputIndices are in range [0, @nbDsts)
 * - @outputIndices contains at least @nbStrs values
 *
 * N.B.:
 *   It is valid to provide a completely empty @dstBuffers (aka @nbDsts == 0).
 *   However, @outputIndices must then not contain any values (aka @nbStrs == 0)
 *   since any values would automatically fall outside the permitted range.
 *
 *   The converse is not true. @outputIndices may be empty without @dstBuffers
 *   being empty. In this case, the kernel will essentially no-op and all
 *   @dstBuffers[i] will be empty.
 */
void ZL_DispatchString_encode(
        uint8_t nbDsts,
        void** restrict dstBuffers,
        uint32_t** restrict dstStrLens,
        size_t dstNbStrs[],
        const void* restrict src,
        const uint32_t srcStrLens[],
        const size_t nbStrs,
        const uint8_t outputIndices[]);

// 16-bit dispatch indices
void ZL_DispatchString_encode16(
        uint16_t nbDsts,
        void** restrict dstBuffers,
        uint32_t** restrict dstStrLens,
        size_t dstNbStrs[],
        const void* restrict src,
        const uint32_t srcStrLens[],
        const size_t nbStrs,
        const uint16_t outputIndices[]);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif

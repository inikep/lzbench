// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_BITPACK_COMMON_BITPACK_KERNEL_H
#define ZSTRONG_TRANSFORMS_BITPACK_COMMON_BITPACK_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // uintX_t

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Packs each value in @src using @nbBits each.
 *
 * @pre @dstCapacity >= ZS_bitpackEncodeBound(nbElts, nbBits)
 * @pre @nbBits <= 8 * sizeof(@src[0]) && @nbBits >= 0
 *
 * @param dst destination buffer of size @dstCapacity.
 * @param nbBits nbBits to use to encode each value.
 *
 * @returns The nb of bytes written into @dst.
 */
size_t ZS_bitpackEncode32(
        void* dst,
        size_t dstCapacity,
        uint32_t const* src,
        size_t nbElts,
        int nbBits);
/// @see ZS_bitpackEncode32
size_t ZS_bitpackEncode16(
        void* dst,
        size_t dstCapacity,
        uint16_t const* src,
        size_t nbElts,
        int nbBits);
/// @see ZS_bitpackEncode32
size_t ZS_bitpackEncode8(
        void* dst,
        size_t dstCapacity,
        uint8_t const* src,
        size_t nbElts,
        int nbBits);
/// @see ZS_bitpackEncode32
size_t ZS_bitpackEncode64(
        void* dst,
        size_t dstCapacity,
        const uint64_t* src,
        size_t nbElts,
        int nbBits);

/// Generic version - dispatches based on @eltWidth.
size_t ZS_bitpackEncode(
        void* dst,
        size_t dstCapacity,
        void const* src,
        size_t nbElts,
        size_t eltWidth,
        int nbBits);

/**
 * @returns The minimum destination buffer capacity to ensure
 * that ZS_bitpackEncode*() will be succeed.
 */
size_t ZS_bitpackEncodeBound(size_t nbElts, int nbBits);

/**
 * Checks if the data in the source buffer can be legally bitpacked.
 * @returns 1 on success and 0 otherwise
 */
int ZS_bitpackEncodeVerify(
        const void* src,
        size_t nbElts,
        size_t eltWidth,
        int nbBits);

/**
 * Unpacks @nbElts values of @nbBits each from @src to @dst.
 *
 * @pre @srcCapacity >= (@nbElts * @nbBits * 8 + 7) / 8
 * @pre @nbBits <= sizeof(@dst[0]) * 8 && @nbBits >= 0
 *
 * @param dst destination buffer of size @nbElts.
 *
 * @returns Number of bytes decoded from src.
 */
size_t ZS_bitpackDecode32(
        uint32_t* dst,
        size_t nbElts,
        void const* src,
        size_t srcCapacity,
        int nbBits);
/// @see ZS_bitpackDeocde32
size_t ZS_bitpackDecode16(
        uint16_t* dst,
        size_t nbElts,
        void const* src,
        size_t srcCapacity,
        int nbBits);
/// @see ZS_bitpackDeocde32
size_t ZS_bitpackDecode8(
        uint8_t* dst,
        size_t nbElts,
        void const* src,
        size_t srcCapacity,
        int nbBits);
/// @see ZS_bitpackDeocde32
size_t ZS_bitpackDecode64(
        uint64_t* dst,
        size_t nbElts,
        void const* src,
        size_t srcCapacity,
        int nbBits);

/// Generic version - dispatches based on @eltWidth.
size_t ZS_bitpackDecode(
        void* dst,
        size_t nbElts,
        size_t eltWidth,
        void const* src,
        size_t srcCapacity,
        int nbBits);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // ZSTRONG_TRANSFORMS_BITPACK_COMMON_BITPACK_KERNEL_H

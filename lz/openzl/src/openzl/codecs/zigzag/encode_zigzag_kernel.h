// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * \file
 *
 * This file defines the internal implementation for a zigzag
 * transformation.
 */

#ifndef ZSTRONG_TRANSFORMS_ZIGZAG_ENCODE_ZIGZAG_KERNEL_H
#define ZSTRONG_TRANSFORMS_ZIGZAG_ENCODE_ZIGZAG_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // xintXX_t

#if defined(__cplusplus)
extern "C" {
#endif

/* raw transforms (transportable).
 * Required conditions :
 * if (nbElts>0), dst and src must be non-NULL
 * dst & src must be valid (already allocated, aligned and sized accordingly)
 **/
void ZL_zigzagEncode8(uint8_t* dst, const int8_t* src, size_t nbElts);
void ZL_zigzagEncode16(uint16_t* dst, const int16_t* src, size_t nbElts);
void ZL_zigzagEncode32(uint32_t* dst, const int32_t* src, size_t nbElts);
void ZL_zigzagEncode64(uint64_t* dst, const int64_t* src, size_t nbElts);
void ZL_zigzagEncode(
        void* dst,
        const void* src,
        size_t nbElts,
        size_t eltWidth);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // ZSTRONG_TRANSFORMS_ZIGZAG_ENCODE_ZIGZAG_KERNEL_H

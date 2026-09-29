// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * \file
 *
 * This file defines the internal implementation for a zigzag
 * transformation.
 */

#ifndef ZSTRONG_TRANSFORMS_ZIGZAG_DECODE_ZIGZAG_KERNEL_H
#define ZSTRONG_TRANSFORMS_ZIGZAG_DECODE_ZIGZAG_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // xintXX_t

#if defined(__cplusplus)
extern "C" {
#endif

/* raw transforms (transportable).
 * Required conditions :
 * dst & src must be valid (already allocated, aligned and sized accordingly)
 * if (nbElts>0), dst and src must be non-NULL
 **/
void ZL_zigzagDecode8(int8_t* dst, const uint8_t* src, size_t nbElts);
void ZL_zigzagDecode16(int16_t* dst, const uint16_t* src, size_t nbElts);
void ZL_zigzagDecode32(int32_t* dst, const uint32_t* src, size_t nbElts);
void ZL_zigzagDecode64(int64_t* dst, const uint64_t* src, size_t nbElts);
void ZL_zigzagDecode(
        void* dst,
        const void* src,
        size_t nbElts,
        size_t eltWidth);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif

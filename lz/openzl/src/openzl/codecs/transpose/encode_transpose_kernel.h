// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * \file
 *
 * This file defines the internal implementation for a transposition
 * transformation.
 */

#ifndef ZSTRONG_TRANSFORMS_TRANSPOSE_ENCODE_TRANSPOSE_KERNEL_H
#define ZSTRONG_TRANSFORMS_TRANSPOSE_ENCODE_TRANSPOSE_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // uint8_t
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/**
 * This function encodes a transposition transformation, which re-orders the
 * bytes in the stream. This is used on integers or other fixed-size objects
 * whose n-th bytes are perhaps more correlated with each other than they are
 * with the adjacent bytes in their individual objects.
 *
 * For example, the most-significant bytes of a sequence of integers might be
 * much better correlated with each other (because they're all '\x00') than they
 * are with their least-significant bytes.
 *
 * Examples:
 * - ZS_transposeEncode(dst, "12345678", 2, 4) -> "15263748"
 * - ZS_transposeEncode(dst, "12345678", 4, 2) -> "13572468"
 *
 * Conditions:
 *   eltWidth > 1
 *   src & dst are nbElts * eltWidth bytes large
 */
void ZS_transposeEncode(
        void* dst,
        const void* src,
        size_t nbElts,
        size_t eltWidth);

/**
 * Similar to above except the `eltWidth` dst buffers are each their own
 * pointer. Each dst buffer must be `nbElts` bytes large.
 *
 * Examples:
 * - ZS_splitTransposeEncode(dst, "12345678", 2, 4) -> ["15", "26", "37", "48"]
 * - ZS_splitTransposeEncode(dst, "12345678", 4, 2) -> ["1357", "2468"]
 * Conditions:
 *   eltWidth >= 1
 *   src is nbElts * eltWidth bytes large
 *   There are eltWidth dst buffers, each nbElts bytes large
 *   @p dst and @p src do not overlap
 */
void ZS_splitTransposeEncode(
        uint8_t** dst,
        void const* src,
        size_t nbElts,
        size_t eltWidth);

ZL_END_C_DECLS

#endif // ZSTRONG_TRANSFORMS_TRANSPOSE_ENCODE_TRANSPOSE_KERNEL_H

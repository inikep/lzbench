// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * \file
 *
 * This file defines the decoder for a transposition transformation.
 */

#ifndef ZSTRONG_TRANSFORMS_TRANSPOSE_DECODE_TRANSPOSE_KERNEL_H
#define ZSTRONG_TRANSFORMS_TRANSPOSE_DECODE_TRANSPOSE_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // uint8_t
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/**
 * This function decodes a transposition transformation, which re-orders the
 * bytes in the stream. This is used on integers or other fixed-size objects
 * whose n-th bytes are perhaps more correlated with each other than they are
 * with the adjacent bytes in their individual objects.
 *
 * For example, the most-significant bytes of a sequence of integers might be
 * much better correlated with each other (because they're all '\x00') than they
 * are with their least-significant bytes.
 *
 * Examples:
 * - ZS_transposeDecode(dst, "15263748", 2, 4) -> "12345678"
 * - ZS_transposeDecode(dst, "13572468", 4, 2) -> "12345678"
 */
void ZS_transposeDecode(
        void* dst,
        const void* src,
        size_t nbElts,
        size_t eltWidth);

/// This function is the same as above but doesn't assume that each row of
/// the transpose is adjacent.
/// Examples:
/// - ZS_splitTransposeDecode(dst, ["15", "26", "37", "38"], 2, 4) -> "12345678"
/// - ZS_splitTransposeDecode(dst, ["1357", "2467"], 4, 2) -> "12345678"
void ZS_splitTransposeDecode(
        void* dst,
        uint8_t const** src,
        size_t nbElts,
        size_t eltWidth);

ZL_END_C_DECLS

#endif // ZSTRONG_TRANSFORMS_TRANSPOSE_DECODE_TRANSPOSE_KERNEL_H

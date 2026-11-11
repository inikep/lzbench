// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * \file
 *
 * This file defines the internal implementation for a constant encoding
 * transformation
 */

#ifndef ZSTRONG_TRANSFORMS_CONSTANT_ENCODE_CONSTANT_KERNEL_H
#define ZSTRONG_TRANSFORMS_CONSTANT_ENCODE_CONSTANT_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // uint8_t

#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/**
 * Verifies that an input stream is "constant", that is all elements are an
 * instance of a single, repeated token. A stream is also not "constant" if it
 * contains no elements. It returns 0 if the stream is not constant and 1 if the
 * stream is constant
 *
 * Conditions:
 * - @p src is @p nbElts * @p eltWidth bytes large
 *
 * @param src The stream buffer to be checked
 * @param nbElts The total number of elements in the stream
 * @param eltWidth The width of each element in bytes
 */
uint8_t ZS_isConstantStream(
        const uint8_t* const src,
        size_t const nbElts,
        size_t const eltWidth);

/**
 * Compresses an input stream using constant encoding, which reduces a stream
 * of a single, repeated token to a single instance of that token and the number
 * of instances of that token in the transform header
 *
 * Examples:
 * - ZS_encodeConstant(dst, "aaa", 3) => dst: "a", nbElts: 3
 * - ZS_encodeConstant(dst, "appappapp", 3) => dst: "app", nbElts: 3
 *
 * Conditions:
 * - @p dst is @p eltWidth bytes large
 * - @p src is 'nbElts * @p eltWidth' bytes large
 * - @p src is a "constant" stream (can be checked with `ZS_isConstantStream`)
 * - @p eltWidth >= 1
 *
 * @p dst The destination buffer for the compressed result to be written to
 * @p src The source buffer for the original, uncompressed data
 * @p eltWidth The width of each element in @p src
 */
void ZS_encodeConstant(
        uint8_t* const dst,
        const uint8_t* const src,
        size_t const eltWidth);

ZL_END_C_DECLS

#endif // ZSTRONG_TRANSFORMS_CONSTANT_ENCODE_CONSTANT_KERNEL_H

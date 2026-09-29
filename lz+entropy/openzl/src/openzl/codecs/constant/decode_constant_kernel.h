// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * \file
 *
 * This file defines the internal implementation for a constant decoding
 * transformation
 */

#ifndef ZSTRONG_COMPRESS_TRANSFORMS_CONSTANT_DECODE_H
#define ZSTRONG_COMPRESS_TRANSFORMS_CONSTANT_DECODE_H

#include <stddef.h> // size_t
#include <stdint.h> // uint8_t

#include "openzl/shared/portability.h"
#include "openzl/zl_data.h"

ZL_BEGIN_C_DECLS

/**
 * Decompresses an input stream that has been transformed by constant encoding,
 * which reduces a stream of a single, repeated token to a single instance of
 * that token and the number of instances of that token in the transform header
 *
 * Examples:
 * - ZS_decodeConstant(dst, 3, "a", 1) => dst: "aaa"
 * - ZS_decodeConstant(dst, 3, "app", 3) => dst: "appappapp"
 *
 * Conditions:
 * - @p dst must be @p dstNbElts * @p eltWidth bytes large
 * - @p src must be @p eltWidth bytes large
 * - @p dstNbElts and @p eltWidth >= 1
 * - @p eltBuffer is an allocated buffer of size MAX(32, eltWidth)
 *
 * @p dst The destination buffer for the decompressed result to be written to
 * @p dstNbElts The number of elements to write to the destination buffer
 * @p src The source buffer that contains a single instance of the token to copy
 * @p eltWidth The width of the token in @p src
 */
void ZS_decodeConstant(
        uint8_t* const dst,
        size_t const dstNbElts,
        const uint8_t* const src,
        size_t const eltWidth,
        void* eltBuffer);

ZL_END_C_DECLS

#endif // ZSTRONG_COMPRESS_TRANSFORMS_CONSTANT_DECODE_H

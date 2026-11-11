// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * \file
 *
 * This file defines the internal implementation for prefix decoding
 */

#ifndef ZSTRONG_COMPRESS_TRANSFORMS_PREFIX_DECODE_H
#define ZSTRONG_COMPRESS_TRANSFORMS_PREFIX_DECODE_H

#include <stddef.h> // size_t
#include <stdint.h> // uint*_t

#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/portability.h"
#include "openzl/zl_data.h"

ZL_BEGIN_C_DECLS

/**
 * Calculates the size of the original, unencoded source buffer. Can be used to
 * specify the size of the destination buffer for `ZS_decodePrefix`
 */
size_t ZS_calcOriginalPrefixSize(
        const uint32_t* matchSizes,
        size_t eltWidthsSum,
        size_t nbElts);

/**
 * Decompresses an input stream that has been transformed by prefix encoding
 *
 * Note: best used for sorted, overlapping strings
 *
 * Examples:
 * - ZS_decodePrefix(out, fieldSizes, ["a", "pp", "le", "s"], 4, [1, 2, 2, 1],
 * [0, 1, 3, 5]) => out: ["a", "app", "apple", "apples"], fieldSizes: [1, 3, 5,
 * 6]
 * - ZS_decodePrefix(out, fieldSizes, ["app", "", ""], 3, [3, 0, 0], [0, 3, 3])
 * => out: ["app", "app", "app"], fieldSizes: [3, 3, 3]
 *
 * Conditions:
 * - @p suffixes is at least sum( @p eltWidths ) bytes large
 * - @p out is at least sum( @p eltWidths ) + sum( @p matchSizes ) bytes large
 * - @p suffixes, @p fieldSizes, @p matchSizes, @p src, @p eltWidths, all have
 * @p nbElts positions
 *
 * @param out The buffer that will contain each decoded element
 * @param fieldSizes The width of each decoded element in @p out
 * @param suffixes The buffer that contains the unmatched portion of each
 * element
 * @param nbElts The number of elements in all buffers
 * @param eltWidths The width of each element in @p suffixes
 * @param matchSizes The width of each matched prefix
 */
ZL_Report ZS_decodePrefix(
        uint8_t* out,
        uint32_t* fieldSizes,
        const uint8_t* suffixes,
        size_t nbElts,
        const uint32_t* eltWidths,
        const uint32_t* matchSizes);

ZL_END_C_DECLS

#endif // ZSTRONG_COMPRESS_TRANSFORMS_PREFIX_DECODE_H

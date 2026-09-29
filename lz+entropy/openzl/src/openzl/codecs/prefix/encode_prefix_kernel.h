// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * \file
 *
 * This file defines the internal implementation for prefix encoding
 */

#ifndef ZSTRONG_TRANSFORMS_PREFIX_ENCODE_PREFIX_KERNEL_H
#define ZSTRONG_TRANSFORMS_PREFIX_ENCODE_PREFIX_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // uint*_t

#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/**
 * Compresses an input stream using prefix encoding, which transforms a stream
 * into a stream of suffixes and a stream of match lengths
 *
 * Note: best used for sorted, overlapping strings (see examples)
 *
 * Examples:
 * - ZS_encodePrefix(suffixes, fieldSizes, matchLengths, ["a", "app", "apple",
 * "apples"], 4, [1, 3, 5, 6], 15) => suffixes: ["a", "pp", "le", "s"],
 * fieldSizes: [1, 2, 2, 1], matchLengths: [0, 1, 3, 5]
 * - ZS_encodePrefix(suffixes, fieldSizes, matchLengths, ["app", "app", "app"],
 * 3, [3, 3, 3]) => suffixes: ["app", "", ""], fieldSizes: [3, 0, 0],
 * matchLengths: [0, 3, 3]
 *
 * Conditions:
 * - @p suffixes and @p src are at least sum( @p fieldSizesSum ) bytes large
 * - @p suffixes, @p fieldSizes, @p matchSizes, @p src, @p eltWidths, all have
 * @p nbElts positions
 * - @p fieldSizesSum = sum( @p eltWidths )
 *
 * @param suffixes The segments of each element that no match was found for
 * @param fieldSizes The size of each suffix
 * @param matchSizes The size of each match found
 * @param src The source buffer for the original, uncompressed data
 * @param nbElts The number of elements in @p src
 * @param eltWidths The width of each element in @p src
 * @param fieldSizesSum The sum of all elements in @p eltWidths
 */
void ZS_encodePrefix(
        uint8_t* suffixes,
        uint32_t* fieldSizes,
        uint32_t* matchSizes,
        const uint8_t* src,
        size_t nbElts,
        const uint32_t* eltWidths,
        size_t fieldSizesSum);

ZL_END_C_DECLS

#endif // ZSTRONG_TRANSFORMS_PREFIX_ENCODE_PREFIX_KERNEL_H

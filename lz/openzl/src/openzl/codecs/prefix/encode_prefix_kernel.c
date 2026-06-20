// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/prefix/encode_prefix_kernel.h"
#include <string.h>
#include "openzl/codecs/common/copy.h"
#include "openzl/codecs/common/count.h"

/**
 * Calculates the maximum matching prefix size between an element and its
 * predecessor. The first element never has a match
 *
 * Examples:
 * - ZS_getPrefixMatchSize("app", 3, "apple", 5) => 3
 * - ZS_getPrefixMatchSize("app", 3, "bag", 3) => 0
 *
 * Conditions:
 * @p prevSize == the size of @p prev
 * @p currSize == the size of @p curr
 *
 * @param prev The start of the previous element
 * @param prevSize The size of the previous element
 * @param curr The start of the current element
 * @param currSize The size of the current element
 */
ZL_FORCE_INLINE uint32_t ZS_getPrefixMatchSize(
        const uint8_t* const prev,
        const uint32_t prevSize,
        const uint8_t* const curr,
        const uint32_t currSize,
        const uint8_t* const endPtr)
{
    uint32_t const maxMatchLen = ZL_MIN(prevSize, currSize);
    return (uint32_t)ZS_countBound(curr, prev, curr + maxMatchLen, endPtr);
}

static void ZS_encodePrefix_fallback(
        uint32_t* const fieldSizes,
        uint32_t* const matchSizes,
        const uint32_t* const eltWidths,
        size_t const nbElts,
        const uint8_t* const rEndPtr,
        const uint8_t* prevEltPtr,
        const uint8_t* currEltPtr,
        uint8_t* currSuffix,
        uint32_t prevEltWidth)
{
    for (size_t i = 0; i < nbElts; ++i) {
        uint32_t const currEltWidth = eltWidths[i];
        uint32_t const matchSize    = ZS_getPrefixMatchSize(
                prevEltPtr, prevEltWidth, currEltPtr, currEltWidth, rEndPtr);
        uint32_t const unmatchedSize = currEltWidth - matchSize;
        const uint8_t* nextRPtr      = currEltPtr + matchSize;

        memcpy(currSuffix, nextRPtr, unmatchedSize);
        matchSizes[i] = matchSize;
        fieldSizes[i] = unmatchedSize;

        prevEltPtr = currEltPtr;
        currEltPtr += currEltWidth;
        currSuffix += unmatchedSize;

        prevEltWidth = currEltWidth;
    }
}

void ZS_encodePrefix(
        uint8_t* const suffixes,
        uint32_t* const fieldSizes,
        uint32_t* const matchSizes,
        const uint8_t* const src,
        size_t const nbElts,
        const uint32_t* const restrict eltWidths,
        size_t const fieldSizesSum)
{
    size_t nbWildcopies = nbElts;
    for (size_t sum = 0; nbWildcopies > 0 && sum < ZS_WILDCOPY_OVERLENGTH;
         --nbWildcopies) {
        sum += eltWidths[nbWildcopies - 1];
    }

    const uint8_t* const rEndPtr = src + fieldSizesSum;
    const uint8_t* prevEltPtr    = src;
    const uint8_t* currEltPtr    = src;
    uint8_t* currSuffix          = suffixes;
    uint32_t prevEltWidth        = 0;

    for (size_t i = 0; i < nbWildcopies; ++i) {
        uint32_t const currEltWidth = eltWidths[i];
        uint32_t const matchSize    = ZS_getPrefixMatchSize(
                prevEltPtr, prevEltWidth, currEltPtr, currEltWidth, rEndPtr);
        uint32_t const unmatchedSize = currEltWidth - matchSize;
        const uint8_t* nextRPtr      = currEltPtr + matchSize;

        ZS_wildcopy(currSuffix, nextRPtr, unmatchedSize, ZS_wo_no_overlap);
        matchSizes[i] = matchSize;
        fieldSizes[i] = unmatchedSize;

        prevEltPtr = currEltPtr;
        currEltPtr += currEltWidth;
        currSuffix += unmatchedSize;

        prevEltWidth = currEltWidth;
    }

    ZS_encodePrefix_fallback(
            fieldSizes + nbWildcopies,
            matchSizes + nbWildcopies,
            eltWidths + nbWildcopies,
            nbElts - nbWildcopies,
            rEndPtr,
            prevEltPtr,
            currEltPtr,
            currSuffix,
            prevEltWidth);
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZS_COMMON_COUNT_H
#define ZS_COMMON_COUNT_H

#include "openzl/common/debug.h"
#include "openzl/shared/bits.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"
#include "openzl/shared/utils.h"

ZL_INLINE size_t ZS_nbCommonBytes(size_t val)
{
    ZL_ASSERT(val != 0);
    if (ZL_isLittleEndian()) {
        if (ZL_64bits()) {
            return (size_t)(ZL_ctz64(val) >> 3);
        } else { /* 32 bits */
            return (size_t)(ZL_ctz32((uint32_t)val) >> 3);
        }
    } else { /* Big Endian CPU */
        if (ZL_64bits()) {
            return (size_t)(ZL_clz64(val) >> 3);
        } else { /* 32 bits */
            return (size_t)(ZL_clz32((uint32_t)val) >> 3);
        }
    }
}

ZL_INLINE size_t ZS_nbCommonBytesBack(size_t val)
{
    ZL_ASSERT(val != 0);
    if (ZL_isLittleEndian()) {
        if (ZL_64bits()) {
            return (size_t)(ZL_clz64(val) >> 3);
        } else { /* 32 bits */
            return (size_t)(ZL_clz32((uint32_t)val) >> 3);
        }
    } else { /* Big Endian CPU */
        if (ZL_64bits()) {
            return (size_t)(ZL_ctz64(val) >> 3);
        } else { /* 32 bits */
            return (size_t)(ZL_ctz32((uint32_t)val) >> 3);
        }
    }
}

ZL_INLINE size_t ZS_count(
        const uint8_t* pIn,
        const uint8_t* pMatch,
        const uint8_t* const pInLimit)
{
    const uint8_t* const pStart       = pIn;
    const uint8_t* const pInLoopLimit = pInLimit - (sizeof(size_t) - 1);

    if (pIn < pInLoopLimit) {
        {
            size_t const diff = ZL_readST(pMatch) ^ ZL_readST(pIn);
            if (diff)
                return ZS_nbCommonBytes(diff);
        }
        pIn += sizeof(size_t);
        pMatch += sizeof(size_t);
        while (pIn < pInLoopLimit) {
            size_t const diff = ZL_readST(pMatch) ^ ZL_readST(pIn);
            if (!diff) {
                pIn += sizeof(size_t);
                pMatch += sizeof(size_t);
                continue;
            }
            pIn += ZS_nbCommonBytes(diff);
            return (size_t)(pIn - pStart);
        }
    }
    if (ZL_64bits() && (pIn < (pInLimit - 3))
        && (ZL_read32(pMatch) == ZL_read32(pIn))) {
        pIn += 4;
        pMatch += 4;
    }
    if ((pIn < (pInLimit - 1)) && (ZL_read16(pMatch) == ZL_read16(pIn))) {
        pIn += 2;
        pMatch += 2;
    }
    if ((pIn < pInLimit) && (*pMatch == *pIn))
        pIn++;
    return (size_t)(pIn - pStart);
}

ZL_INLINE size_t ZS_countBack(
        const uint8_t* pIn,
        const uint8_t* pMatch,
        uint8_t const* pInLimit,
        const uint8_t* const pLowLimit)
{
    ZL_ASSERT_LT(pMatch, pIn);
    size_t const st                      = sizeof(size_t);
    const uint8_t* const pStart          = pIn;
    const uint8_t* const pMatchLoopLimit = pLowLimit + st;
    size_t const maxLength               = (size_t)(pStart - pInLimit);

    if (pMatch >= pMatchLoopLimit) {
        {
            size_t const diff = ZL_readST(pMatch - st) ^ ZL_readST(pIn - st);
            if (diff)
                return ZL_MIN(ZS_nbCommonBytesBack(diff), maxLength);
        }
        pIn -= st;
        pMatch -= st;
        if (pIn <= pInLimit)
            return maxLength;
        while (pMatch >= pMatchLoopLimit) {
            size_t const diff = ZL_readST(pMatch - st) ^ ZL_readST(pIn - st);
            if (!diff) {
                pIn -= st;
                pMatch -= st;
                if (pIn <= pInLimit)
                    return maxLength;
                continue;
            }
            pIn -= ZS_nbCommonBytesBack(diff);
            return ZL_MIN((size_t)(pStart - pIn), maxLength);
        }
    }
    while (pMatch > pLowLimit && pMatch[-1] == pIn[-1]) {
        --pMatch;
        --pIn;
    }
    return ZL_MIN((size_t)(pStart - pIn), maxLength);
}

/** ZS_count_2segments() :
 *  can count match length with `ip` & `match` in 2 different segments.
 *  convention : on reaching mEnd, match count continue starting from iStart
 */
ZL_INLINE size_t ZS_count2segments(
        const uint8_t* ip,
        const uint8_t* match,
        const uint8_t* iEnd,
        const uint8_t* mEnd,
        const uint8_t* iStart)
{
    const uint8_t* const vEnd = ZL_MIN(ip + (mEnd - match), iEnd);
    size_t const matchLength  = ZS_count(ip, match, vEnd);
    if (match + matchLength != mEnd)
        return matchLength;
    return matchLength + ZS_count(ip + matchLength, iStart, iEnd);
}

/**
 * Fast counts past the end of either buffers ( @p pBound ) up to @p pEnd
 *
 * Conditions:
 * - @p pIn >= @p pMatch
 * - @p pEnd >= @p pBound
 * - @p pBound is @p pIn offset by the maximum possible match size
 * - All memory between @p pMatch and @p pEnd is safe to read
 *
 * @param pIn The pointer to the forward buffer
 * @param pMatch The pointer to the buffer to compare to
 * @param pBound The pointer to the bound at which we've reached the maximum
 * match length
 * @param pEnd The pointer to the end of the buffer that holds both @p pIn and
 * @p pMatch (i.e., the first non-read safe memory region)
 */
ZL_INLINE size_t ZS_countBound(
        const uint8_t* const pIn,
        const uint8_t* const pMatch,
        const uint8_t* const pBound,
        const uint8_t* const pEnd)
{
    ZL_ASSERT_GE(pIn, pMatch);
    ZL_ASSERT_GE(pEnd, pBound);
    ZL_ASSERT_GE(pBound, pIn);
    const uint8_t* pInNext    = pIn;
    const uint8_t* pMatchNext = pMatch;
    const uint8_t* const pEndLoopLimit =
            ZL_MIN(pEnd - (sizeof(size_t) - 1), pBound);
    if (pInNext < pEndLoopLimit) {
        {
            size_t const diff = ZL_readST(pInNext) ^ ZL_readST(pMatchNext);
            if (diff) {
                return ZL_MIN(ZS_nbCommonBytes(diff), (size_t)(pBound - pIn));
            }
        }
        pInNext += sizeof(size_t);
        pMatchNext += sizeof(size_t);
        while (pInNext < pEndLoopLimit) {
            size_t const diff = ZL_readST(pInNext) ^ ZL_readST(pMatchNext);
            if (!diff) {
                pInNext += sizeof(size_t);
                pMatchNext += sizeof(size_t);
                continue;
            }
            pInNext += ZS_nbCommonBytes(diff);
            return (size_t)ZL_MIN(pBound - pIn, pInNext - pIn);
        }
    }
    if (ZL_64bits() && (pInNext < (pBound - 3))
        && (ZL_read32(pInNext) == ZL_read32(pMatchNext))) {
        pInNext += 4;
        pMatchNext += 4;
    }
    if ((pInNext < (pBound - 1))
        && (ZL_read16(pInNext) == ZL_read16(pMatchNext))) {
        pInNext += 2;
        pMatchNext += 2;
    }
    if ((pInNext < pBound) && (*pInNext == *pMatchNext)) {
        pInNext++;
    }
    return (size_t)ZL_MIN(pInNext - pIn, pBound - pIn);
}

#endif // ZS_COMMON_COUNT_H

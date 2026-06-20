// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/common/window.h"

#include "openzl/shared/utils.h"

static void logWindow(ZS_window const* window, char const* prefix)
{
    ZL_LOG(BLOCK,
           "%s: extDict=[%p, %p) prefix=[%p, %p) lowLimit=%u dictLimit=%u",
           prefix,
           (void const*)(window->dictBase + window->lowLimit),
           (void const*)(window->dictBase + window->dictLimit),
           (void const*)(window->base + window->dictLimit),
           (void const*)(window->nextSrc),
           window->lowLimit,
           window->dictLimit);
}

static uint8_t const base[1] = { 0 };

int ZS_window_init(ZS_window* window, uint32_t maxDist, uint32_t minDictSize)
{
    window->base        = base;
    window->dictBase    = base;
    window->dictLimit   = 1;
    window->lowLimit    = 1;
    window->nextSrc     = base + 1;
    window->maxDist     = maxDist;
    window->minDictSize = minDictSize;
    return 0;
}

void ZS_window_clear(ZS_window* window)
{
    ptrdiff_t const endT = window->nextSrc - window->base;
    uint32_t const end   = (uint32_t)endT;
    ZL_ASSERT((ptrdiff_t)end == endT);

    window->dictLimit = end;
    window->lowLimit  = end;
}

bool ZS_window_hasExtDict(ZS_window const* window)
{
    return window->lowLimit < window->dictLimit;
}

uint32_t ZS_window_maxIndex(void)
{
    return (3u << 30) + (1u << 29);
}

uint32_t ZS_window_maxChunkSize(void)
{
    return (uint32_t)-1 - ZS_window_maxIndex();
}

bool ZS_window_needOverflowCorrection(
        ZS_window const* window,
        uint8_t const* srcEnd)
{
    ZL_ASSERT(srcEnd <= window->nextSrc);
    ptrdiff_t const currentT = srcEnd - window->base;
    /* Must not overflow since we overflow correct every max chunk size */
    uint32_t const current = (uint32_t)currentT;
    ZL_ASSERT(currentT == (ptrdiff_t)current);
    return current > ZS_window_maxIndex();
}

uint32_t ZS_window_correctOverflow(
        ZS_window* window,
        uint32_t cycleLog,
        uint8_t const* src)
{
    /* preemptive overflow correction:
     * 1. correction is large enough:
     *    lowLimit > (3<<29) ==> current > 3<<29 + 1<<windowLog
     *    1<<windowLog <= newCurrent < 1<<chainLog + 1<<windowLog
     *
     *    current - newCurrent
     *    > (3<<29 + 1<<windowLog) - (1<<windowLog + 1<<chainLog)
     *    > (3<<29) - (1<<chainLog)
     *    > (3<<29) - (1<<30)             (NOTE: chainLog <= 30)
     *    > 1<<29
     *
     * 2. (ip+ZSTD_CHUNKSIZE_MAX - cctx->base) doesn't overflow:
     *    After correction, current is less than (1<<chainLog + 1<<windowLog).
     *    In 64-bit mode we are safe, because we have 64-bit ptrdiff_t.
     *    In 32-bit mode we are safe, because (chainLog <= 29), so
     *    ip+ZSTD_CHUNKSIZE_MAX - cctx->base < 1<<32.
     * 3. (cctx->lowLimit + 1<<windowLog) < 1<<32:
     *    windowLog <= 31 ==> 3<<29 + 1<<windowLog < 7<<29 < 1<<32.
     */
    uint32_t const cycleMask  = (1U << cycleLog) - 1;
    uint32_t const current    = (uint32_t)(src - window->base);
    uint32_t const newCurrent = (current & cycleMask) + window->maxDist;
    uint32_t const correction = current - newCurrent;
    ZL_ASSERT((window->maxDist & cycleMask) == 0);
    ZL_ASSERT(current > newCurrent);
    /* Loose bound, should be around 1<<29 (see above) */
    ZL_ASSERT(correction > 1 << 28);

    window->base += correction;
    window->dictBase += correction;
    window->dictLimit -= correction;
    window->lowLimit -= correction;

    ZL_LOG(OBJ,
           "Correction of 0x%x bytes to lowLimit=0x%x",
           correction,
           window->dictLimit);
    return correction;
}

uint32_t ZS_window_prefixPtrToIdx(ZS_window const* window, uint8_t const* ptr)
{
    ZL_ASSERT(ptr >= window->base && ptr <= window->nextSrc);
    return (uint32_t)(ptr - window->base);
}

bool ZS_window_indexIsValid(ZS_window const* window, uint32_t index)
{
    uint32_t const minIndex = window->lowLimit;
    uint32_t const maxIndex = ZS_window_prefixPtrToIdx(window, window->nextSrc);
    ZL_ASSERT(maxIndex <= ZS_window_maxIndex());
    return index >= minIndex && index < maxIndex;
}

ZS_continuity
ZS_window_update(ZS_window* window, uint8_t const* src, size_t srcSize)
{
    // NOTE: srcSize may be >= 4GB!
    ZL_LOG(BLOCK,
           "ZS_window_update(window=%p, src=%p, %llu)",
           (void const*)window,
           (void const*)src,
           (unsigned long long)srcSize);
    logWindow(window, "Old window");

    // Check if blocks follow each other
    ZS_continuity const continuity =
            src == window->nextSrc ? ZS_c_contiguous : ZS_c_newSegment;

    if (continuity == ZS_c_newSegment) {
        ZL_LOG(BLOCK, "ZS_window_update: New segment");
        // The current prefix becomes the new extDict.
        ptrdiff_t const newDictLimitT = window->nextSrc - window->base;
        uint32_t newDictLimit         = (uint32_t)newDictLimitT;
        ZL_ASSERT(newDictLimitT == (ptrdiff_t)newDictLimit); // No overflow

        window->lowLimit  = window->dictLimit;
        window->dictLimit = (uint32_t)newDictLimit;
        window->dictBase  = window->base;
        window->base      = src - newDictLimit;

        // Clear the extDict if it is below the minDictSize.
        if (window->dictLimit - window->lowLimit < window->minDictSize) {
            ZL_LOG(BLOCK, "ZS_window_update: ExtDict too small => clearing");
            window->lowLimit = window->dictLimit;
        }
    }
    window->nextSrc = src + srcSize;
    ZL_ASSERT(window->nextSrc >= src); // No overflow

    // Check if the prefix and the extDict overlap. If they do then
    // increase lowLimit until they no longer overlap.
    uint8_t const* const dictBegin = window->dictBase + window->lowLimit;
    uint8_t const* const dictEnd   = window->dictBase + window->dictLimit;
    if ((src + srcSize > dictBegin) && (src < dictEnd)) {
        ptrdiff_t const highInputIdx = (src + srcSize) - window->dictBase;
        uint32_t const lowLimitMax =
                (uint32_t)ZL_MIN(highInputIdx, (ptrdiff_t)window->dictLimit);
        window->lowLimit = lowLimitMax;
        ZL_LOG(BLOCK, "Overlapping extDict and input => increasing lowLimit");
    }
    logWindow(window, "Updated window");

    return continuity;
}

void ZS_window_moveSuffix(
        ZS_window* window,
        uint8_t const* src,
        size_t suffixSize)
{
    ZL_ASSERT(src != window->nextSrc);
    window->nextSrc -= suffixSize;
    ZS_continuity const continuity = ZS_window_update(window, src, suffixSize);
    ZL_ASSERT(continuity == ZS_c_newSegment);
    ZL_ASSERT(suffixSize <= window->dictLimit - window->lowLimit);
    (void)continuity;
}

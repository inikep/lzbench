// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZS_COMMON_WINDOW_H
#define ZS_COMMON_WINDOW_H

#include "openzl/common/debug.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

typedef struct {
    /// The end of the prefix. If the next source starts here we continue the
    /// prefix buffer, otherwise the prefix becomes the second buffer.
    uint8_t const* nextSrc;
    /// The base of the prefix buffer.
    uint8_t const* base;
    /// The base of the second buffer.
    uint8_t const* dictBase;
    /// All indices >= the prefix limit are relative to base.
    uint32_t dictLimit;
    /// All indices >= secondLimit and < prefixLimit are relative to dictBase.
    /// All indices below secondLimit are invalid.
    uint32_t lowLimit;
    /// Maximum distance allowed by this window.
    uint32_t maxDist;
    /// Minimum size of the extDict.
    uint32_t minDictSize;
} ZS_window;

/// Initializes an empty window.
int ZS_window_init(ZS_window* window, uint32_t maxDist, uint32_t minDictSize);

/// Invalidate all indices in the window.
void ZS_window_clear(ZS_window* window);

/// Returns true iff the window has a second buffer.
bool ZS_window_hasExtDict(ZS_window const* window);

uint32_t ZS_window_maxIndex(void);

uint32_t ZS_window_maxChunkSize(void);

/**
 * ZS_window_needOverflowCorrection():
 * Returns non-zero if the indices are getting too large and need overflow
 * protection before we reach srcEnd. srcEnd must be within the prefix.
 */
bool ZS_window_needOverflowCorrection(
        ZS_window const* window,
        uint8_t const* srcEnd);

/**
 * ZS_window_correctOverflow():
 * Reduces the indices to protect from index overflow.
 * Returns the correction made to the indices, which must be applied to every
 * stored index.
 *
 * The least significant cycleLog bits of the indices must remain the same,
 * which may be 0. Every index up to maxDist in the past must be valid.
 * NOTE: (maxDist & cycleMask) must be zero.
 */
uint32_t ZS_window_correctOverflow(
        ZS_window* window,
        uint32_t cycleLog,
        uint8_t const* src);

typedef enum {
    ZS_c_contiguous,
    ZS_c_newSegment,
} ZS_continuity;

/**
 * ZS_window_update():
 * Updates the window by appending [src, src + srcSize) to the window.
 * If it is not contiguous, the current prefix becomes the extDict, and we
 * forget about the extDict. Handles overlap of the prefix and extDict.
 * Returns non-zero if the segment is contiguous.
 */
ZS_continuity
ZS_window_update(ZS_window* window, uint8_t const* src, size_t srcSize);

/**
 * Logically moves the suffix bytes to the new src pointer.
 * NOTE: Does not actually memcpy the bytes.
 */
void ZS_window_moveSuffix(
        ZS_window* window,
        uint8_t const* src,
        size_t suffixSize);

ZL_INLINE uint32_t
ZS_window_getLowestMatchIndex(ZS_window const* window, uint32_t current)
{
    uint32_t const lowestValid  = window->lowLimit;
    uint32_t const withinWindow = (current - lowestValid > window->maxDist)
            ? current - window->maxDist
            : lowestValid;
    return withinWindow;
}

bool ZS_window_indexIsValid(ZS_window const* window, uint32_t index);

ZL_INLINE uint8_t const* ZS_window_idxToPtr(
        ZS_window const* window,
        uint32_t index)
{
    ZL_ASSERT(ZS_window_indexIsValid(window, index));
    uint8_t const* const base =
            index >= window->dictLimit ? window->base : window->dictBase;
    return base + index;
}

uint32_t ZS_window_prefixPtrToIdx(ZS_window const* window, uint8_t const* ptr);

ZL_END_C_DECLS

#endif // ZS_COMMON_WINDOW_H

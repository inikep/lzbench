// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "openzl/codecs/common/window.h"

namespace {
// This is never accessed. We just do pointer math.
uint8_t const* data = (uint8_t const*)0xdeadbeef;

TEST(WindowTest, init)
{
    ZS_window window;
    ASSERT_EQ(ZS_window_init(&window, 42, 350), 0);
    // NULL pointer arithimitic is undefined
    ASSERT_NE(window.base, nullptr);
    ASSERT_NE(window.dictBase, nullptr);
    // 0 must be out of bounds for reproducibility
    ASSERT_GT(window.dictLimit, 0u);
    ASSERT_GT(window.lowLimit, 0u);
    // ExtDict and prefix must both be empty
    ASSERT_EQ(window.lowLimit, window.dictLimit);
    ASSERT_EQ(window.base + 1, window.nextSrc);
    // Correctly sets the variables
    ASSERT_EQ(window.maxDist, 42u);
    ASSERT_EQ(window.minDictSize, 350u);
}

TEST(WindowTest, clear)
{
    ZS_window window;
    ASSERT_EQ(ZS_window_init(&window, 42, 42), 0);
    ZS_window prev = window;
    ZS_window_clear(&window);
    ASSERT_EQ(window.dictLimit, prev.dictLimit);
    ASSERT_EQ(window.lowLimit, prev.lowLimit);
    prev = window;
    // Not contiguous
    ASSERT_EQ(ZS_window_update(&window, data, 6), ZS_c_newSegment);
    ZS_window_clear(&window);
    ASSERT_EQ(window.dictLimit, prev.dictLimit + 6);
    ASSERT_EQ(window.lowLimit, prev.lowLimit + 6);
}

TEST(WindowTest, hasExtDict)
{
    ZS_window window;
    // No buffers
    ASSERT_EQ(ZS_window_init(&window, 42, 0), 0);
    ASSERT_FALSE(ZS_window_hasExtDict(&window));
    // Just prefix
    ASSERT_EQ(ZS_window_update(&window, data, 10), ZS_c_newSegment);
    ASSERT_FALSE(ZS_window_hasExtDict(&window));
    // Has an ext dict
    ASSERT_EQ(ZS_window_update(&window, data + 20, 10), ZS_c_newSegment);
    ASSERT_TRUE(ZS_window_hasExtDict(&window));
    // 0 buffers
    ZS_window_clear(&window);
    ASSERT_FALSE(ZS_window_hasExtDict(&window));
}

TEST(WindowTest, maxIndexAndChunkSize)
{
    uint64_t const maxIndex     = ZS_window_maxIndex();
    uint64_t const maxChunkSize = ZS_window_maxChunkSize();
    ASSERT_LE(maxIndex + maxChunkSize, (uint64_t)(uint32_t)-1);
}

TEST(WindowTest, needOverflowCorrection)
{
    ZS_window window;
    ASSERT_EQ(ZS_window_init(&window, 42, 42), 0);
    ZS_window_update(&window, data, ZS_window_maxIndex() + 1);
    for (uint32_t i = 0; i < ZS_window_maxIndex();
         i += ZS_window_maxChunkSize()) {
        ASSERT_FALSE(ZS_window_needOverflowCorrection(&window, data + i));
    }
    ASSERT_TRUE(ZS_window_needOverflowCorrection(
            &window, data + ZS_window_maxIndex()));
}

static void
testCorrectOverflow(uint32_t offset, uint32_t cycleLog, uint32_t maxDist)
{
    ZS_window window;
    ASSERT_EQ(ZS_window_init(&window, maxDist, 0), 0);
    uint64_t const kMax = 1ull << 35;
    ZS_window_update(&window, data, kMax);
    uint64_t bytes           = offset;
    uint32_t const chunkSize = ZS_window_maxChunkSize();
    for (uint32_t idx = offset; bytes < kMax;
         idx += chunkSize, bytes += chunkSize) {
        uint8_t const* ptr = window.base + idx;
        if (ZS_window_needOverflowCorrection(&window, ptr)) {
            uint32_t const correction = ZS_window_correctOverflow(
                    &window, cycleLog, window.base + idx);
            uint32_t const cycleMask = (1u << cycleLog) - 1;
            ASSERT_EQ(idx & cycleMask, (idx - correction) & cycleMask);
            idx -= correction;
            ptr = window.base + idx;
            ASSERT_GE(correction, 1u << 28);
            ASSERT_FALSE(ZS_window_needOverflowCorrection(&window, ptr));
        }
    }
}

TEST(WindowTest, correctOverflow)
{
    // Test many different scenarios for overflow correction
    for (uint32_t offsetLog = 0; offsetLog <= 28; ++offsetLog) {
        ASSERT_LE(1u << offsetLog, ZS_window_maxChunkSize());
        for (uint32_t cycleLog = 0; cycleLog <= 30; ++cycleLog) {
            for (uint32_t windowLog = 0; windowLog <= 31; ++windowLog) {
                if (windowLog < cycleLog)
                    continue;
                testCorrectOverflow(1u << offsetLog, cycleLog, 1u << windowLog);
            }
        }
    }
}

TEST(WindowTest, updateSmallExtDict)
{
    ZS_window window;
    ASSERT_EQ(ZS_window_init(&window, 42, 8), 0);

    ASSERT_EQ(ZS_window_update(&window, data, 8), ZS_c_newSegment);
    ASSERT_FALSE(ZS_window_hasExtDict(&window));

    ASSERT_EQ(ZS_window_update(&window, data + 100, 8 - 1), ZS_c_newSegment);
    ASSERT_TRUE(ZS_window_hasExtDict(&window));

    ASSERT_EQ(ZS_window_update(&window, data + 200, 8), ZS_c_newSegment);
    ASSERT_FALSE(ZS_window_hasExtDict(&window));
}

TEST(WindowTest, updateOverlap)
{
    ZS_window window;
    ASSERT_EQ(ZS_window_init(&window, 42, 0), 0);
    ZS_window const init = window;
    ASSERT_EQ(ZS_window_update(&window, data, 10), ZS_c_newSegment);
    ASSERT_EQ(ZS_window_update(&window, data + 10, 10), ZS_c_contiguous);
    ASSERT_EQ(ZS_window_update(&window, data + 20, 10), ZS_c_contiguous);
    ASSERT_FALSE(ZS_window_hasExtDict(&window));
    ASSERT_EQ(window.lowLimit, init.lowLimit);

    ASSERT_EQ(ZS_window_update(&window, data, 10), ZS_c_newSegment);
    ASSERT_TRUE(ZS_window_hasExtDict(&window));
    ASSERT_EQ(window.dictLimit, init.dictLimit + 30);
    ASSERT_EQ(window.lowLimit, init.lowLimit + 10);

    ASSERT_EQ(ZS_window_update(&window, data + 10, 10), ZS_c_contiguous);
    ASSERT_TRUE(ZS_window_hasExtDict(&window));
    ASSERT_EQ(window.dictLimit, init.dictLimit + 30);
    ASSERT_EQ(window.lowLimit, init.lowLimit + 20);

    ASSERT_EQ(ZS_window_update(&window, data + 20, 10), ZS_c_contiguous);
    ASSERT_FALSE(ZS_window_hasExtDict(&window));
    ASSERT_EQ(window.dictLimit, init.dictLimit + 30);
    ASSERT_EQ(window.lowLimit, init.lowLimit + 30);
}

TEST(WindowTest, getLowestMatchIndex)
{
    ZS_window window;
    ASSERT_EQ(ZS_window_init(&window, 100, 0), 0);
    uint32_t const lowLimit = window.lowLimit;
    ASSERT_EQ(ZS_window_getLowestMatchIndex(&window, lowLimit + 10), lowLimit);
    ASSERT_EQ(ZS_window_getLowestMatchIndex(&window, lowLimit + 100), lowLimit);
    ASSERT_EQ(
            ZS_window_getLowestMatchIndex(&window, lowLimit + 101),
            lowLimit + 1);
    ASSERT_EQ(
            ZS_window_getLowestMatchIndex(&window, lowLimit + 200),
            lowLimit + 100);
}

TEST(WindowTest, indexIsValid)
{
    ZS_window window;
    ASSERT_EQ(ZS_window_init(&window, 100, 0), 0);
    ASSERT_FALSE(ZS_window_indexIsValid(&window, 0));
    ZS_window_update(&window, data + 1000, 100);
    ZS_window_update(&window, data + 2100, 100);
    ZS_window_update(&window, data + 2000, 100);

    uint32_t const start = (uint32_t)(data + 2100 - window.dictBase);
    ASSERT_GT(start, 0u);
    ASSERT_FALSE(ZS_window_indexIsValid(&window, start - 1));
    ASSERT_FALSE(ZS_window_indexIsValid(&window, 0));
    for (uint32_t i = start; i < start + 200; ++i) {
        ASSERT_TRUE(ZS_window_indexIsValid(&window, i));
    }
    ASSERT_FALSE(ZS_window_indexIsValid(&window, start + 200));
}

TEST(WindowTest, moveSuffix)
{
    ZS_window window;
    ASSERT_EQ(ZS_window_init(&window, 100, 0), 0);
    ASSERT_EQ(ZS_window_update(&window, data, 100), ZS_c_newSegment);
    ASSERT_FALSE(ZS_window_hasExtDict(&window));
    ASSERT_EQ(window.lowLimit, 1u);
    ASSERT_EQ(window.dictLimit, 1u);
    ASSERT_EQ(window.nextSrc, data + 100);

    ZS_window_moveSuffix(&window, data + 200, 10);
    ASSERT_TRUE(ZS_window_hasExtDict(&window));
    ASSERT_EQ(window.lowLimit, 1u);
    ASSERT_EQ(window.dictLimit, 91u);
    ASSERT_EQ(window.nextSrc, data + 210);
}

} // namespace

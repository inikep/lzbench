// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "openzl/codecs/common/count.h"

namespace {

TEST(CountTest, nbCommonBytes)
{
    uint8_t const x[8] = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07 };
    uint8_t y[8]       = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07 };

    y[3] = 0x13;
    ASSERT_EQ(ZS_nbCommonBytes(ZL_readST(x) ^ ZL_readST(y)), 3u);

    y[2] = 0x03;
    ASSERT_EQ(ZS_nbCommonBytes(ZL_readST(x) ^ ZL_readST(y)), 2u);
    y[3] = 0x03;
    ASSERT_EQ(ZS_nbCommonBytes(ZL_readST(x) ^ ZL_readST(y)), 2u);

    y[1] = 0xff;
    ASSERT_EQ(ZS_nbCommonBytes(ZL_readST(x) ^ ZL_readST(y)), 1u);

    y[0] = 0x10;
    ASSERT_EQ(ZS_nbCommonBytes(ZL_readST(x) ^ ZL_readST(y)), 0u);
}

TEST(CountTest, count)
{
    // Try different segment sizes
    for (size_t size = 1; size < 32; ++size) {
        std::vector<uint8_t> x(size);
        std::iota(x.begin(), x.end(), 0);
        std::vector<uint8_t> y;
        // Try each possible diff byte including identical
        for (size_t diff = 0; diff <= x.size(); ++diff) {
            y = x;
            if (diff < y.size()) {
                y[diff] = 0xff;
            }
            ASSERT_EQ(ZS_count(x.data(), y.data(), x.data() + x.size()), diff);
        }
    }
}

TEST(CountTest, count2segments)
{
    // Try different segment sizes
    for (size_t size = 1; size < 32; ++size) {
        std::vector<uint8_t> x(size);
        std::iota(x.begin(), x.end(), 0);
        std::vector<uint8_t> y;
        // Try each possible diff byte including identical
        for (size_t diff = 0; diff <= x.size(); ++diff) {
            y = x;
            if (diff < y.size()) {
                y[diff] = 0xff;
            }
            // Break the match up into 2 segments of every possible size
            for (size_t split = 1; split <= y.size(); ++split) {
                std::vector<uint8_t> y1(split);
                std::vector<uint8_t> y2(x.size() + y.size() - split);

                memcpy(y1.data(), y.data(), split);
                memcpy(y2.data(), y.data() + split, y.size() - split);
                uint8_t* yx = y2.data() + y.size() - split;
                memcpy(yx, x.data(), x.size());

                ASSERT_EQ(
                        ZS_count2segments(
                                yx,
                                y1.data(),
                                y2.data() + y2.size(),
                                y1.data() + y1.size(),
                                y2.data()),
                        diff);
            }
        }
    }
}

TEST(CountTest, countBoundTest)
{
    // Try different segment sizes
    for (uint8_t size = 1; size <= 32; ++size) {
        std::vector<uint8_t> x(size * 2);
        std::iota(x.begin(), x.end() - size, 0);
        std::iota(x.begin() + size, x.end(), 0);

        // Try each possible diff byte including identical
        for (uint8_t diff = 0; diff <= size; ++diff) {
            if (diff < size) {
                x[diff + size] = 0xff;
            }

            // Try each possible bound
            for (size_t bound = 1; bound <= size; ++bound) {
                size_t const res = ZS_countBound(
                        x.data() + size,
                        x.data(),
                        x.data() + size + bound,
                        x.data() + x.size());
                size_t const actual = std::min<size_t>(diff, bound);
                ASSERT_EQ(res, actual);
            }

            // Reset diff byte
            if (diff < size) {
                x[diff + size] = diff;
            }
        }
    }
}

} // namespace

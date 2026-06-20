// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "openzl/shared/overflow.h"

namespace {
template <typename Int>
std::vector<Int> generatedInts()
{
    size_t const nbInts = 100;
    std::vector<Int> data;
    std::mt19937 gen(0xdeadbeef);
    std::uniform_int_distribution<Int> dist;

    data.reserve(nbInts);
    for (size_t i = 0; i < nbInts; ++i) {
        data.push_back(dist(gen));
    }
    return data;
}

template <typename Int, typename F1, typename F2>
void fallbackTest(F1 f1, F2 f2)
{
    auto ints = generatedInts<Int>();
    for (size_t x = 0; x < ints.size(); ++x) {
        for (size_t y = 0; y < ints.size(); ++y) {
            Int r1, r2;
            ASSERT_EQ(f1(ints[x], ints[y], &r1), f2(ints[x], ints[y], &r2));
            ASSERT_EQ(r1, r2);
        }
    }
}

TEST(OverflowTest, MulU32)
{
    uint32_t r;
    ASSERT_FALSE(ZL_overflowMulU32(5, 10, &r));
    ASSERT_EQ(r, uint32_t(5 * 10));

    ASSERT_FALSE(ZL_overflowMulU32(50000, 85899, &r));
    ASSERT_EQ(r, (uint32_t)50000 * (uint32_t)(85899));

    ASSERT_TRUE(ZL_overflowMulU32(50000, 85900, &r));

    uint32_t const x = ((uint32_t)1 << 16);
    ASSERT_TRUE(ZL_overflowMulU32(x, x, &r));
    ASSERT_EQ(r, uint32_t(0));
    ASSERT_TRUE(ZL_overflowMulU32_fallback(x, x, &r));
    ASSERT_EQ(r, uint32_t(0));

    fallbackTest<uint32_t>(ZL_overflowMulU32, ZL_overflowMulU32_fallback);
}

TEST(OverflowTest, MulU64)
{
    uint64_t r;
    ASSERT_FALSE(ZL_overflowMulU64(5, 10, &r));
    ASSERT_EQ(r, uint64_t(5 * 10));

    uint64_t const x = ((uint64_t)1 << 32) - 1;
    ASSERT_FALSE(ZL_overflowMulU64(x, x, &r));
    ASSERT_EQ(r, x * x);

    ASSERT_TRUE(ZL_overflowMulU64(x + 1, x + 1, &r));
    ASSERT_EQ(r, uint64_t(0));
    ASSERT_TRUE(ZL_overflowMulU64_fallback(x + 1, x + 1, &r));
    ASSERT_EQ(r, uint64_t(0));

    fallbackTest<uint64_t>(ZL_overflowMulU64, ZL_overflowMulU64_fallback);
}

TEST(OverflowTest, MulST)
{
    size_t r;
    ASSERT_FALSE(ZL_overflowMulST(5, 10, &r));
    ASSERT_EQ(r, size_t(5 * 10));

    size_t const shift = sizeof(size_t) * 4;
    size_t const x     = ((size_t)1 << shift) - 1;
    ASSERT_FALSE(ZL_overflowMulST(x, x, &r));
    ASSERT_EQ(r, x * x);

    ASSERT_TRUE(ZL_overflowMulST(x + 1, x + 1, &r));
    ASSERT_EQ(r, size_t(0));
    ASSERT_TRUE(ZL_overflowMulST_fallback(x + 1, x + 1, &r));
    ASSERT_EQ(r, size_t(0));

    fallbackTest<size_t>(ZL_overflowMulST, ZL_overflowMulST_fallback);
}

TEST(OverflowTest, AddU32)
{
    uint32_t r;
    ASSERT_FALSE(ZL_overflowAddU32(5, 10, &r));
    ASSERT_EQ(r, uint32_t(5 + 10));

    auto const max  = std::numeric_limits<uint32_t>::max();
    auto const half = max / 2 + 1;

    ASSERT_FALSE(ZL_overflowAddU32(half, half - 1, &r));
    ASSERT_EQ(r, max);
    ASSERT_FALSE(ZL_overflowAddU32_fallback(half, half - 1, &r));
    ASSERT_EQ(r, max);

    ASSERT_TRUE(ZL_overflowAddU32(half, half, &r));
    ASSERT_EQ(r, uint32_t(0));
    ASSERT_TRUE(ZL_overflowAddU32_fallback(half, half, &r));
    ASSERT_EQ(r, uint32_t(0));

    fallbackTest<uint32_t>(ZL_overflowAddU32, ZL_overflowAddU32_fallback);
}

TEST(OverflowTest, AddU64)
{
    uint64_t r;
    ASSERT_FALSE(ZL_overflowAddU64(5, 10, &r));
    ASSERT_EQ(r, uint64_t(5 + 10));

    auto const max  = std::numeric_limits<uint64_t>::max();
    auto const half = max / 2 + 1;

    ASSERT_FALSE(ZL_overflowAddU64(half, half - 1, &r));
    ASSERT_EQ(r, max);
    ASSERT_FALSE(ZL_overflowAddU64_fallback(half, half - 1, &r));
    ASSERT_EQ(r, max);

    ASSERT_TRUE(ZL_overflowAddU64(half, half, &r));
    ASSERT_EQ(r, uint64_t(0));
    ASSERT_TRUE(ZL_overflowAddU64_fallback(half, half, &r));
    ASSERT_EQ(r, uint64_t(0));

    fallbackTest<uint64_t>(ZL_overflowAddU64, ZL_overflowAddU64_fallback);
}

TEST(OverflowTest, AddST)
{
    size_t r;
    ASSERT_FALSE(ZL_overflowAddST(5, 10, &r));
    ASSERT_EQ(r, size_t(5 + 10));

    auto const max  = std::numeric_limits<size_t>::max();
    auto const half = max / 2 + 1;

    ASSERT_FALSE(ZL_overflowAddST(half, half - 1, &r));
    ASSERT_EQ(r, max);
    ASSERT_FALSE(ZL_overflowAddST_fallback(half, half - 1, &r));
    ASSERT_EQ(r, max);

    ASSERT_TRUE(ZL_overflowAddST(half, half, &r));
    ASSERT_EQ(r, size_t(0));
    ASSERT_TRUE(ZL_overflowAddST_fallback(half, half, &r));
    ASSERT_EQ(r, size_t(0));

    fallbackTest<size_t>(ZL_overflowAddST, ZL_overflowAddST_fallback);
}

} // namespace

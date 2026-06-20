// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <vector>

#include "openzl/codecs/divide_by/decode_divide_by_kernel.h"
#include "openzl/codecs/divide_by/encode_divide_by_kernel.h"

namespace {

template <typename T>
class DivideByKernelTest : public testing::Test {};

using DivideTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(DivideByKernelTest, DivideTypes, );

TYPED_TEST(DivideByKernelTest, testDivideBy2)
{
    std::vector<TypeParam> input    = { 16, 32, 0, 8 };
    std::vector<TypeParam> expected = { 8, 16, 0, 4 };
    std::vector<TypeParam> output(4, 0);
    uint64_t divisor = 2;
    ZS_divideByEncode(
            output.data(),
            input.data(),
            input.size(),
            divisor,
            sizeof(TypeParam));
    EXPECT_EQ(expected, output);
}

TYPED_TEST(DivideByKernelTest, EmptyRoundTrip)
{
    uint64_t divisor = 1;
    ZS_divideByEncode(nullptr, nullptr, 0, divisor, sizeof(TypeParam));
    ZS_divideByDecode(nullptr, nullptr, 0, divisor, sizeof(TypeParam));
}

TYPED_TEST(DivideByKernelTest, OneRoundTrip)
{
    uint32_t inputLength = sizeof(TypeParam) == 1
            ? 30
            : 1000; // Use smaller input length for uint8_t
    uint64_t divisor     = 7;
    std::vector<TypeParam> input(inputLength, 0);
    std::vector<TypeParam> output(inputLength, 0);
    std::vector<TypeParam> recovered(inputLength, 0);
    for (uint32_t i = 0; i < inputLength; i++) {
        input[i] = i * divisor;
    }
    ZS_divideByEncode(
            output.data(),
            input.data(),
            input.size(),
            divisor,
            sizeof(TypeParam));
    ZS_divideByDecode(
            recovered.data(),
            output.data(),
            output.size(),
            divisor,
            sizeof(TypeParam));
    EXPECT_EQ(input, recovered);
}

TYPED_TEST(DivideByKernelTest, testDivideBy1RoundTrip)
{
    uint32_t inputLength = 200;
    uint64_t divisor     = 1;
    std::vector<TypeParam> input(inputLength, 0);
    std::vector<TypeParam> output(inputLength, 0);
    std::vector<TypeParam> recovered(inputLength, 0);
    for (uint32_t i = 0; i < inputLength; i++) {
        input[i] = i;
    }
    ZS_divideByEncode(
            output.data(),
            input.data(),
            input.size(),
            divisor,
            sizeof(TypeParam));
    ZS_divideByDecode(
            recovered.data(),
            output.data(),
            output.size(),
            divisor,
            sizeof(TypeParam));
    EXPECT_EQ(input, recovered);
}

TYPED_TEST(DivideByKernelTest, testDivideByLimits)
{
    uint32_t inputLength = 8;
    uint64_t divisor     = std::numeric_limits<TypeParam>::max();
    std::vector<TypeParam> input(inputLength, 0);
    std::vector<TypeParam> output(inputLength, 0);
    std::vector<TypeParam> recovered(inputLength, 0);
    for (uint32_t i = 0; i < inputLength / 2; i++) {
        input[i] = std::numeric_limits<TypeParam>::max();
    }
    ZS_divideByEncode(
            output.data(),
            input.data(),
            input.size(),
            divisor,
            sizeof(TypeParam));
    ZS_divideByDecode(
            recovered.data(),
            output.data(),
            output.size(),
            divisor,
            sizeof(TypeParam));
    EXPECT_EQ(input, recovered);
}

TYPED_TEST(DivideByKernelTest, testDivideByLargeNumerator)
{
    uint32_t inputLength = 20;
    uint64_t divisor     = 3;
    TypeParam valueBase  = static_cast<TypeParam>(1)
            << (sizeof(TypeParam) * 8 - 2);
    std::vector<TypeParam> input(inputLength, 0);
    std::vector<TypeParam> output(inputLength, 0);
    std::vector<TypeParam> recovered(inputLength, 0);
    for (uint32_t i = 0; i < inputLength; i++) {
        input[i] = (valueBase + i) * divisor;
    }
    ZS_divideByEncode(
            output.data(),
            input.data(),
            input.size(),
            divisor,
            sizeof(TypeParam));
    ZS_divideByDecode(
            recovered.data(),
            output.data(),
            output.size(),
            divisor,
            sizeof(TypeParam));
    EXPECT_EQ(input, recovered);
}

} // namespace

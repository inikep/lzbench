// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "openzl/codecs/transpose/decode_transpose_kernel.h"
#include "openzl/codecs/transpose/encode_transpose_kernel.h"

namespace {

std::pair<std::string, std::string> makeBeforeAndAfter(
        size_t size,
        size_t stride)
{
    EXPECT_FALSE(size % stride);
    std::string before;
    std::string after;
    before.reserve(size);
    after.reserve(size);
    for (size_t i = 0; i < size; i++) {
        before += (char)i;
    }
    for (size_t i = 0; i < stride; i++) {
        for (size_t j = 0; j < size / stride; j++) {
            after += (char)(i + j * stride);
        }
    }

    return std::make_pair(std::move(before), std::move(after));
}

TEST(TransposeKernelTest, testEncodeCombos)
{
    for (size_t stride = 2; stride < 200; stride++) {
        for (int strides :
             { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 1000 }) {
            auto size              = stride * (size_t)strides;
            auto [input, expected] = makeBeforeAndAfter(size, stride);
            std::string output(size, '\0');
            ZS_transposeEncode(
                    &output[0], input.data(), input.size() / stride, stride);
            output.resize(size);
            EXPECT_EQ(output, expected);
        }
    }
}

TEST(TransposeKernelTest, testDecodeCombos)
{
    for (size_t stride = 2; stride < 200; stride++) {
        for (int strides :
             { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 1000 }) {
            auto size              = stride * (size_t)strides;
            auto [expected, input] = makeBeforeAndAfter(size, stride);
            std::string output(size, '\0');
            ZS_transposeDecode(
                    &output[0], input.data(), input.size() / stride, stride);
            EXPECT_EQ(output, expected);
        }
    }
}

TEST(TransposeKernelTest, testEncodeCombosSplit)
{
    for (size_t eltWidth = 1; eltWidth < 200; eltWidth++) {
        for (int nbElts : { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 1000 }) {
            auto size              = eltWidth * (size_t)nbElts;
            auto [input, expected] = makeBeforeAndAfter(size, eltWidth);
            std::string output(size, '\0');
            std::vector<uint8_t*> outputs;
            outputs.push_back((uint8_t*)&output[0]);
            for (size_t i = 1; i < eltWidth; ++i) {
                outputs.push_back(outputs.back() + nbElts);
            }
            ZS_splitTransposeEncode(
                    outputs.data(),
                    input.data(),
                    input.size() / eltWidth,
                    eltWidth);
            output.resize(size);
            EXPECT_EQ(output, expected);
        }
    }
}

TEST(TransposeKernelTest, testDecodeCombosSplit)
{
    for (size_t eltWidth = 1; eltWidth < 200; ++eltWidth) {
        for (int nbElts : { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 1000 }) {
            auto size              = eltWidth * (size_t)nbElts;
            auto [expected, input] = makeBeforeAndAfter(size, eltWidth);
            std::string output(size, '\0');
            std::vector<uint8_t const*> inputs;
            inputs.push_back((uint8_t const*)input.data());
            for (size_t i = 1; i < eltWidth; ++i) {
                inputs.push_back(inputs.back() + nbElts);
            }
            ZS_splitTransposeDecode(
                    &output[0], inputs.data(), (size_t)nbElts, eltWidth);
            EXPECT_EQ(output, expected);
        }
    }
}

} // namespace

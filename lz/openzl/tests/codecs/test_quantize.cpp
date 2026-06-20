// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "tests/codecs/test_codec.h"

using namespace ::testing;

namespace openzl {
namespace tests {

class QuantizeTest : public CodecTest {
   public:
    void testQuantize(
            NodeID node,
            const Input& input,
            const Input& codes,
            const Input* bits)
    {
        testCodec(node, input, { &codes, bits }, ZL_TYPED_INPUT_VERSION_MIN);
    }

    void testQuantize(
            NodeID node,
            const std::vector<uint32_t>& input,
            const std::vector<uint8_t>& codes)
    {
        testQuantize(
                node,
                Input::refNumeric(poly::span<const uint32_t>(input)),
                Input::refNumeric(poly::span<const uint8_t>(codes)),
                nullptr);
    }

    void testQuantize(
            NodeID node,
            const std::vector<uint32_t>& input,
            const std::vector<uint8_t>& codes,
            const std::vector<uint8_t>& bits)
    {
        auto bitsInput = Input::refSerial(bits.data(), bits.size());
        testQuantize(
                node,
                Input::refNumeric(poly::span<const uint32_t>(input)),
                Input::refNumeric(poly::span<const uint8_t>(codes)),
                &bitsInput);
    }

    void testQuantize(NodeID node, const std::vector<uint32_t>& input)
    {
        testCodec(
                node,
                Input::refNumeric(poly::span<const uint32_t>(input)),
                { nullptr, nullptr },
                ZL_TYPED_INPUT_VERSION_MIN);
    }
};

TEST_F(QuantizeTest, TestQuantizeOffsets)
{
    ASSERT_THROW(testQuantize(ZL_NODE_QUANTIZE_OFFSETS, { 0 }), Exception);
    testQuantize(ZL_NODE_QUANTIZE_OFFSETS, { 1 });
    testQuantize(
            ZL_NODE_QUANTIZE_OFFSETS,
            { 1, 2, 3, 4, 5 },
            { 0, 1, 1, 2, 2 },
            { 18 });
}

TEST_F(QuantizeTest, TestQuantizeLengths)
{
    testQuantize(
            ZL_NODE_QUANTIZE_LENGTHS,
            { 0, 1, 2, 3, 4, 15 },
            { 0, 1, 2, 3, 4, 15 },
            {});
    testQuantize(
            ZL_NODE_QUANTIZE_LENGTHS,
            { 16, 17, 18, 19 },
            { 16, 16, 16, 16 },
            { 0x10, 0x32 });
}
} // namespace tests
} // namespace openzl

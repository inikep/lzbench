// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <numeric>

#include <gtest/gtest.h>

#include "openzl/openzl.hpp"
#include "openzl/shared/mem.h"

using namespace ::testing;

namespace openzl {
namespace tests {
template <typename T>
void byteswap(std::vector<T>& data)
{
    for (auto& v : data) {
        if (sizeof(T) == 2) {
            v = ZL_swap16(v);
        } else if (sizeof(T) == 4) {
            v = ZL_swap32(v);
        } else if (sizeof(T) == 8) {
            v = ZL_swap64(v);
        }
    }
}

template <typename T>
void littleEndian(std::vector<T>& data)
{
    if (ZL_isLittleEndian()) {
        return;
    }
    byteswap(data);
}

template <typename T>
void bigEndian(std::vector<T>& data)
{
    if (!ZL_isLittleEndian()) {
        return;
    }
    byteswap(data);
}

class NumericConversionTest : public ::testing::Test {
   public:
    // Test on a delta+constant backend graph that will only succeed if the
    // numbers are converted correctly
    void testConversionToNum(NodeID node, const Input& input)
    {
        Compressor compressor;
        auto graph = compressor.buildStaticGraph(
                node, { nodes::DeltaInt()(compressor, graphs::Constant{}()) });
        compressor.selectStartingGraph(graph);
        compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        CCtx cctx;
        cctx.refCompressor(compressor);
        auto compressed   = cctx.compressOne(input);
        auto decompressed = DCtx().decompressOne(compressed);
        EXPECT_EQ(input, decompressed);
    }

    template <typename T>
    void testSerialConversionToNum(NodeID node, const std::vector<T>& input)
    {
        testConversionToNum(
                node, Input::refSerial(input.data(), input.size() * sizeof(T)));
    }

    template <typename T>
    void testStructConversionToNum(NodeID node, const std::vector<T>& input)
    {
        testConversionToNum(
                node, Input::refStruct(input.data(), sizeof(T), input.size()));
    }
};

TEST_F(NumericConversionTest, TestWorks)
{
    std::vector<uint16_t> src(1000, 0);
    std::iota(src.begin(), src.end(), 0);

    littleEndian(src);
    ASSERT_THROW(
            testSerialConversionToNum(ZL_NODE_CONVERT_SERIAL_TO_NUM_BE16, src),
            Exception);
}

TEST_F(NumericConversionTest, ConvertSerialToNum8)
{
    std::vector<uint8_t> src(1000, 0);
    std::iota(src.begin(), src.end(), 0);

    testSerialConversionToNum(ZL_NODE_CONVERT_SERIAL_TO_NUM8, src);
    testStructConversionToNum(ZL_NODE_CONVERT_STRUCT_TO_NUM_LE, src);
    testStructConversionToNum(ZL_NODE_CONVERT_STRUCT_TO_NUM_BE, src);

    ASSERT_EQ(
            nodes::ConvertSerialToNum8().baseNode(),
            ZL_NODE_CONVERT_SERIAL_TO_NUM8);
    ASSERT_EQ(
            nodes::ConvertStructToNumLE().baseNode(),
            ZL_NODE_CONVERT_STRUCT_TO_NUM_LE);
    ASSERT_EQ(
            nodes::ConvertStructToNumBE().baseNode(),
            ZL_NODE_CONVERT_STRUCT_TO_NUM_BE);
}

TEST_F(NumericConversionTest, ConvertToNum16)
{
    std::vector<uint16_t> src(1000, 0);
    std::iota(src.begin(), src.end(), 0);

    littleEndian(src);
    testSerialConversionToNum(ZL_NODE_CONVERT_SERIAL_TO_NUM_LE16, src);
    testStructConversionToNum(ZL_NODE_CONVERT_STRUCT_TO_NUM_LE, src);

    bigEndian(src);
    testSerialConversionToNum(ZL_NODE_CONVERT_SERIAL_TO_NUM_BE16, src);
    testStructConversionToNum(ZL_NODE_CONVERT_STRUCT_TO_NUM_BE, src);

    ASSERT_EQ(
            nodes::ConvertSerialToNumLE16().baseNode(),
            ZL_NODE_CONVERT_SERIAL_TO_NUM_LE16);
    ASSERT_EQ(
            nodes::ConvertSerialToNumBE16().baseNode(),
            ZL_NODE_CONVERT_SERIAL_TO_NUM_BE16);
}

TEST_F(NumericConversionTest, ConvertToNum32)
{
    std::vector<uint32_t> src(1000, 0);
    std::iota(src.begin(), src.end(), 0);

    littleEndian(src);
    testSerialConversionToNum(ZL_NODE_CONVERT_SERIAL_TO_NUM_LE32, src);
    testStructConversionToNum(ZL_NODE_CONVERT_STRUCT_TO_NUM_LE, src);

    bigEndian(src);
    testSerialConversionToNum(ZL_NODE_CONVERT_SERIAL_TO_NUM_BE32, src);
    testStructConversionToNum(ZL_NODE_CONVERT_STRUCT_TO_NUM_BE, src);

    ASSERT_EQ(
            nodes::ConvertSerialToNumLE32().baseNode(),
            ZL_NODE_CONVERT_SERIAL_TO_NUM_LE32);
    ASSERT_EQ(
            nodes::ConvertSerialToNumBE32().baseNode(),
            ZL_NODE_CONVERT_SERIAL_TO_NUM_BE32);
}

TEST_F(NumericConversionTest, ConvertToNum64)
{
    std::vector<uint64_t> src(1000, 0);
    std::iota(src.begin(), src.end(), 0);

    littleEndian(src);
    testSerialConversionToNum(ZL_NODE_CONVERT_SERIAL_TO_NUM_LE64, src);
    testStructConversionToNum(ZL_NODE_CONVERT_STRUCT_TO_NUM_LE, src);

    bigEndian(src);
    testSerialConversionToNum(ZL_NODE_CONVERT_SERIAL_TO_NUM_BE64, src);
    testStructConversionToNum(ZL_NODE_CONVERT_STRUCT_TO_NUM_BE, src);

    ASSERT_EQ(
            nodes::ConvertSerialToNumLE64().baseNode(),
            ZL_NODE_CONVERT_SERIAL_TO_NUM_LE64);
    ASSERT_EQ(
            nodes::ConvertSerialToNumBE64().baseNode(),
            ZL_NODE_CONVERT_SERIAL_TO_NUM_BE64);
}

} // namespace tests
} // namespace openzl

// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <zstd.h>

#include "openzl/compress/private_nodes.h"
#include "openzl/openzl.hpp"
#include "tests/codecs/test_codec.h"

namespace openzl {
namespace tests {
namespace {
} // namespace

class ZstdTest : public CodecTest {
   public:
    void testZstd(NodeID node, const std::string& input)
    {
        testCodec(
                node,
                Input::refSerial(input),
                { nullptr },
                ZL_MIN_FORMAT_VERSION);
    }

    void testZstd(
            NodeID node,
            const std::string& input,
            const std::vector<uint8_t>& compressed,
            int minFormatVersion,
            int maxFormatVersion)
    {
        auto c = Input::refSerial(compressed.data(), compressed.size());
        testCodec(
                node,
                Input::refSerial(input),
                { &c },
                minFormatVersion,
                maxFormatVersion);
    }

    std::string testZstdRoundTrip(NodeID node, const std::string& input)
    {
        auto graph = compressor_.buildStaticGraph(node, { graphs::Store{}() });
        compressor_.selectStartingGraph(graph);
        return testRoundTrip(input);
    }
};

TEST_F(ZstdTest, ContextReuseWithDifferentFormatVersions)
{
    CCtx cctx;
    DCtx dctx;

    compressor_.selectStartingGraph(ZL_GRAPH_ZSTD);

    for (int rep = 0; rep < 2; ++rep) {
        for (int formatVersion = ZL_MIN_FORMAT_VERSION;
             formatVersion <= ZL_MAX_FORMAT_VERSION;
             ++formatVersion) {
            cctx.refCompressor(compressor_);
            cctx.setParameter(CParam::FormatVersion, formatVersion);
            auto input        = std::string(1000, 'a') + std::string(1000, 'b');
            auto compressed   = cctx.compressSerial(input);
            auto roundTripped = dctx.decompressSerial(compressed);
            EXPECT_EQ(roundTripped, input);
        }
    }
}

TEST_F(ZstdTest, FormatVersionUpTo8)
{
    // These versions includes the zstd magic number
    compressor_.setParameter(CParam::CompressionLevel, 1);
    testZstd(
            ZL_NODE_ZSTD,
            std::string(1000, 'a'),
            { 0x01, 0x28, 0xb5, 0x2f, 0xfd, 0x60, 0xe8, 0x02, 0x4d, 0x00,
              0x00, 0x10, 0x61, 0x61, 0x01, 0x00, 0xe3, 0x2b, 0x80, 0x05 },
            ZL_MIN_FORMAT_VERSION,
            8);
}

TEST_F(ZstdTest, FormatVersionAtLeast9)
{
    // These versions includes the zstd magic number
    compressor_.setParameter(CParam::CompressionLevel, 1);
    testZstd(
            ZL_NODE_ZSTD,
            std::string(1000, 'a'),
            { 0x01,
              0x60,
              0xe8,
              0x02,
              0x4d,
              0x00,
              0x00,
              0x10,
              0x61,
              0x61,
              0x01,
              0x00,
              0xe3,
              0x2b,
              0x80,
              0x05 },
            9,
            ZL_MAX_FORMAT_VERSION);
}

TEST_F(ZstdTest, SettingCompressionLevelWorks)
{
    std::string input =
            "hello world helworllloellohelworldhello world world hello llloheworld";
    compressor_.setParameter(CParam::CompressionLevel, 1);

    auto zstdWithLevel = [this](int level) {
        LocalParams localParams;
        localParams.addIntParam(ZSTD_c_compressionLevel, level);
        return compressor_.parameterizeNode(
                ZL_NODE_ZSTD, { .localParams = { localParams } });
    };

    auto zstd1  = zstdWithLevel(1);
    auto zstd19 = zstdWithLevel(19);

    testZstd(zstd1, input);
    testZstd(zstd19, input);

    compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);

    ASSERT_LT(
            testZstdRoundTrip(zstd19, input).size(),
            testZstdRoundTrip(zstd1, input).size());
}

} // namespace tests
} // namespace openzl

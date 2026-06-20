// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <string>

#include <gtest/gtest.h>

#include "openzl/common/errors_internal.h"
#include "openzl/zl_common_types.h"
#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_decompress.h"
#include "openzl/zl_version.h"

using namespace ::testing;

namespace {

class DecompressParamsTest : public Test {
   protected:
    size_t compress(
            void* dst,
            size_t dstCapacity,
            void const* src,
            size_t srcSize,
            ZL_GraphID graph)
    {
        auto cctx = ZL_CCtx_create();

        ZL_REQUIRE_SUCCESS(ZL_CCtx_setParameter(
                cctx, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION));
        ZL_REQUIRE_SUCCESS(
                ZL_CCtx_selectStartingGraphID(cctx, NULL, graph, NULL));

        auto const report =
                ZL_CCtx_compress(cctx, dst, dstCapacity, src, srcSize);
        ZL_REQUIRE_SUCCESS(report);

        ZL_CCtx_free(cctx);

        return ZL_validResult(report);
    }
};

TEST_F(DecompressParamsTest, StickyParams)
{
    std::string data(1000, 'a');
    std::string compressed(ZL_compressBound(data.size()), '\0');
    // Get zstrong compressed data
    auto cSize = compress(
            compressed.data(),
            compressed.size(),
            data.data(),
            data.size(),
            ZL_GRAPH_CONSTANT);

    auto dctx = ZL_DCtx_create();

    // Without sticky parameters we expect the default value after decompress
    ZL_REQUIRE_SUCCESS(ZL_DCtx_setParameter(
            dctx, ZL_DParam_checkCompressedChecksum, ZL_TernaryParam_disable));
    std::string decompressed(1000, '\0');
    ZL_REQUIRE_SUCCESS(ZL_DCtx_decompress(
            dctx,
            decompressed.data(),
            decompressed.size(),
            compressed.data(),
            cSize));
    EXPECT_EQ(ZL_DCtx_getParameter(dctx, ZL_DParam_checkCompressedChecksum), 0);

    // With sticky parameters we expect the value we set after decompress
    ZL_REQUIRE_SUCCESS(
            ZL_DCtx_setParameter(dctx, ZL_DParam_stickyParameters, 1));
    ZL_REQUIRE_SUCCESS(ZL_DCtx_setParameter(
            dctx, ZL_DParam_checkCompressedChecksum, ZL_TernaryParam_disable));
    ZL_REQUIRE_SUCCESS(ZL_DCtx_decompress(
            dctx,
            decompressed.data(),
            decompressed.size(),
            compressed.data(),
            cSize));
    EXPECT_EQ(
            ZL_DCtx_getParameter(dctx, ZL_DParam_checkCompressedChecksum),
            ZL_TernaryParam_disable);
    ZL_DCtx_free(dctx);
}

TEST_F(DecompressParamsTest, CheckCompressedChecksum)
{
    std::string data(1000, 'a');
    std::string compressed(ZL_compressBound(data.size()), '\0');
    auto cSize = compress(
            compressed.data(),
            compressed.size(),
            data.data(),
            data.size(),
            ZL_GRAPH_CONSTANT);
    // Corrupt the compressed checksum
    // Warning: this test implies knowledge of the frame format, which can
    // evolve over time. This is brittle!
    compressed[cSize - 4] ^= 0x01;

    auto dctx = ZL_DCtx_create();
    std::string decompressed(1000, '\0');

    // Expect decompression to fail by default
    auto report = ZL_DCtx_decompress(
            dctx,
            decompressed.data(),
            decompressed.size(),
            compressed.data(),
            cSize);
    EXPECT_TRUE(ZL_isError(report));

    // Expect decompression to succeed with checksum disabled
    ZL_REQUIRE_SUCCESS(ZL_DCtx_setParameter(
            dctx, ZL_DParam_checkCompressedChecksum, ZL_TernaryParam_disable));
    ZL_REQUIRE_SUCCESS(ZL_DCtx_decompress(
            dctx,
            decompressed.data(),
            decompressed.size(),
            compressed.data(),
            cSize));

    // Expect decompression to fail with checksum enabled
    ZL_REQUIRE_SUCCESS(ZL_DCtx_setParameter(
            dctx, ZL_DParam_checkCompressedChecksum, ZL_TernaryParam_enable));
    report = ZL_DCtx_decompress(
            dctx,
            decompressed.data(),
            decompressed.size(),
            compressed.data(),
            cSize);
    EXPECT_TRUE(ZL_isError(report));
    ZL_DCtx_free(dctx);
}

TEST_F(DecompressParamsTest, CheckContentChecksum)
{
    std::string data(1000, 'a');
    std::string compressed(ZL_compressBound(data.size()), '\0');
    auto cSize = compress(
            compressed.data(),
            compressed.size(),
            data.data(),
            data.size(),
            ZL_GRAPH_CONSTANT);

    // Corrupt the content checksum and recalculate the compressed checksum
    // Warning: this test implies knowledge of the frame format, which can
    // evolve over time. This is brittle!
    compressed[cSize - 7] ^= 0x01;

    auto dctx = ZL_DCtx_create();
    std::string decompressed(1000, '\0');

    // Disable the compressed checksum check
    ZL_REQUIRE_SUCCESS(ZL_DCtx_setParameter(
            dctx, ZL_DParam_checkCompressedChecksum, ZL_TernaryParam_disable));

    // Expect decompression to fail by default
    auto report = ZL_DCtx_decompress(
            dctx,
            decompressed.data(),
            decompressed.size(),
            compressed.data(),
            cSize);
    EXPECT_TRUE(ZL_isError(report));

    // Expect decompression to succeed with both checksums disabled
    ZL_REQUIRE_SUCCESS(ZL_DCtx_setParameter(
            dctx, ZL_DParam_checkContentChecksum, ZL_TernaryParam_disable));
    ZL_REQUIRE_SUCCESS(ZL_DCtx_setParameter(
            dctx, ZL_DParam_checkCompressedChecksum, ZL_TernaryParam_disable));
    ZL_REQUIRE_SUCCESS(ZL_DCtx_decompress(
            dctx,
            decompressed.data(),
            decompressed.size(),
            compressed.data(),
            cSize));

    // Expect decompression to fail with just content checksum enabled
    ZL_REQUIRE_SUCCESS(ZL_DCtx_setParameter(
            dctx, ZL_DParam_checkContentChecksum, ZL_TernaryParam_enable));
    ZL_REQUIRE_SUCCESS(ZL_DCtx_setParameter(
            dctx, ZL_DParam_checkCompressedChecksum, ZL_TernaryParam_disable));
    report = ZL_DCtx_decompress(
            dctx,
            decompressed.data(),
            decompressed.size(),
            compressed.data(),
            cSize);
    EXPECT_TRUE(ZL_isError(report));
    ZL_DCtx_free(dctx);
}
} // namespace

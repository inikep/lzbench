// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdio>
#include <string>

#include <gtest/gtest.h>

#include "openzl/common/assertion.h"
#include "openzl/decompress/decode_frameheader.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/xxhash.h"
#include "openzl/zl_common_types.h" // ZL_TernaryParam_enable, ZL_TernaryParam_disable
#include "openzl/zl_compress.h"
#include "openzl/zl_decompress.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_reflection.h"

using namespace ::testing;

namespace {

struct ChecksumFlags {
    bool contentChecksum;
    bool compressedChecksum;
};

std::string compress(std::string_view src, ChecksumFlags flags)
{
    ZL_CCtx* cctx = ZL_CCtx_create();

    ZL_REQUIRE_SUCCESS(ZL_CCtx_setParameter(
            cctx, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION));

    ZL_REQUIRE_SUCCESS(ZL_CCtx_setParameter(
            cctx,
            ZL_CParam_contentChecksum,
            flags.contentChecksum ? ZL_TernaryParam_enable
                                  : ZL_TernaryParam_disable));
    ZL_REQUIRE_SUCCESS(ZL_CCtx_setParameter(
            cctx,
            ZL_CParam_compressedChecksum,
            flags.compressedChecksum ? ZL_TernaryParam_enable
                                     : ZL_TernaryParam_disable));

    std::string dst;
    dst.resize(ZL_compressBound(src.size()));
    ZL_Report const ret = ZL_CCtx_compress(
            cctx, dst.data(), dst.size(), src.data(), src.size());
    ZL_REQUIRE_SUCCESS(ret);

    dst.resize(ZL_validResult(ret));

    ZL_CCtx_free(cctx);

    return dst;
}

std::pair<ZL_Report, std::string> decompress(std::string_view src)
{
    std::string out;
    ZL_Report const decompressedSize =
            ZL_getDecompressedSize(src.data(), src.size());
    ZL_REQUIRE_SUCCESS(decompressedSize);
    out.resize(ZL_validResult(decompressedSize));
    return { ZL_decompress(out.data(), out.size(), src.data(), src.size()),
             std::move(out) };
}

struct Checksums {
    uint32_t realContentHash;
    uint32_t realCompressedHash;
    uint32_t frameContentHash;
    uint32_t frameCompressedHash;
};

Checksums getChecksums(
        std::string_view uncompressed,
        std::string_view compressed,
        ChecksumFlags flags)
{
    Checksums checksums = { 0, 0, 0, 0 };
    ZL_FrameInfo* const fi =
            ZL_FrameInfo_create(compressed.data(), compressed.size());
    assert(fi != NULL);
    ZL_Report const vReport = ZL_FrameInfo_getFormatVersion(fi);
    assert(!ZL_isError(vReport));
    size_t version = ZL_validResult(vReport);
    /* @note (@cyan): this test is brittle, it depends on knowledge of the wire
     * format, which can evolve over time */
    size_t contentPos = compressed.size() - 4 - (flags.compressedChecksum * 4)
            - (version >= 21);
    size_t compressedPos = compressed.size() - 4 - (version >= 21);
    size_t fhSize        = FrameInfo_frameHeaderSize(fi);
    assert(fhSize <= compressedPos);
    if (flags.compressedChecksum) {
        checksums.frameCompressedHash =
                ZL_readCE32(compressed.data() + compressedPos);
        checksums.realCompressedHash = (uint32_t)XXH3_64bits(
                compressed.data() + fhSize, compressedPos - fhSize);
    }
    if (flags.contentChecksum) {
        checksums.frameContentHash =
                ZL_readCE32(compressed.data() + contentPos);
        checksums.realContentHash =
                (uint32_t)XXH3_64bits(uncompressed.data(), uncompressed.size());
    }
    ZL_FrameInfo_free(fi);
    return checksums;
}

} // namespace

TEST(ChecksumTest, SuccessBothEnabled)
{
    std::string src =
            "hello world, I am some data to compress, hello world hello";
    src.resize(src.size() + 100);

    ChecksumFlags flags      = { true, true };
    auto const compressed    = compress(src, flags);
    auto const [report, out] = decompress(compressed);
    ASSERT_EQ(out, src);
    ZL_REQUIRE_SUCCESS(report);

    auto const checksums = getChecksums(src, compressed, flags);
    ASSERT_EQ(checksums.frameCompressedHash, checksums.realCompressedHash);
    ASSERT_EQ(checksums.frameContentHash, checksums.frameContentHash);
}

TEST(ChecksumTest, SuccessOnlyCompressed)
{
    std::string src =
            "hello world, I am some data to compress, hello world hello";
    src.resize(src.size() + 100);

    ChecksumFlags flags      = { false, false };
    flags.compressedChecksum = true;
    auto const compressed    = compress(src, flags);
    auto const [report, out] = decompress(compressed);
    ZL_REQUIRE_SUCCESS(report);
    ASSERT_EQ(out, src);

    auto const checksums = getChecksums(src, compressed, flags);
    ASSERT_EQ(checksums.frameCompressedHash, checksums.realCompressedHash);
}

TEST(ChecksumTest, SuccessOnlyContent)
{
    std::string src =
            "hello world, I am some data to compress, hello world hello";
    src.resize(src.size() + 100);

    ChecksumFlags flags      = { false, false };
    flags.contentChecksum    = true;
    auto const compressed    = compress(src, flags);
    auto const [report, out] = decompress(compressed);
    ZL_REQUIRE_SUCCESS(report);
    ASSERT_EQ(out, src);

    auto const checksums = getChecksums(src, compressed, flags);
    ASSERT_EQ(checksums.frameContentHash, checksums.frameContentHash);
}

TEST(ChecksumTest, FailureBothEnabled)
{
    std::string src =
            "hello world, I am some data to compress, hello world hello";
    src.resize(src.size() + 100);

    ChecksumFlags flags = { true, true };
    auto compressed     = compress(src, flags);

    unsigned const version = ZL_MAX_FORMAT_VERSION;
    // @note (@cyan): this test code is brittle, it presumes knowledge of the
    // wire format, which can change overtime
    size_t const contentHashPos    = 8 + (version >= 21);
    size_t const compressedHashPos = 4 + (version >= 21);

    {
        // Only content hash is bad
        auto badContentHash = compressed;
        badContentHash[badContentHash.size() - contentHashPos] =
                (char)(badContentHash[badContentHash.size() - contentHashPos]
                       ^ 1);

        {
            // the compressed checksum includes the content checksum, so, in
            // order to remain valid, the compressed checksum must be
            // recalculated after modifying the content checksum
            auto const checksums = getChecksums(src, badContentHash, flags);
            memcpy(badContentHash.data() + badContentHash.size()
                           - compressedHashPos,
                   &checksums.realCompressedHash,
                   4);
        }

        auto const [report, out] = decompress(badContentHash);
        ASSERT_EQ(
                ZL_RES_error(report)._code, ZL_ErrorCode_contentChecksumWrong);

        auto const checksums = getChecksums(src, badContentHash, flags);
        ASSERT_EQ(checksums.frameCompressedHash, checksums.realCompressedHash);
        ASSERT_NE(checksums.frameContentHash, checksums.realContentHash);
    }
    {
        // Only compressed hash is bad
        auto badCompressedHash = compressed;
        badCompressedHash[badCompressedHash.size() - compressedHashPos] =
                (char)(badCompressedHash
                               [badCompressedHash.size() - compressedHashPos]
                       ^ 1);

        auto const [report, out] = decompress(badCompressedHash);
        ASSERT_EQ(
                ZL_RES_error(report)._code,
                ZL_ErrorCode_compressedChecksumWrong);

        auto const checksums = getChecksums(src, badCompressedHash, flags);
        ASSERT_NE(checksums.frameCompressedHash, checksums.realCompressedHash);
        ASSERT_EQ(checksums.frameContentHash, checksums.realContentHash);
    }

    // Both hashes are bad
    // @note: since compressed checksum includes the content checksum,
    // modifying the content checksum is enough to make the compressed one
    // wrong.
    compressed[compressed.size() - contentHashPos] =
            (char)(compressed[compressed.size() - contentHashPos] ^ 1);

    auto const [report, out] = decompress(compressed);
    ASSERT_EQ(ZL_RES_error(report)._code, ZL_ErrorCode_compressedChecksumWrong);

    auto const checksums = getChecksums(src, compressed, flags);
    ASSERT_NE(checksums.frameCompressedHash, checksums.realCompressedHash);
    ASSERT_NE(checksums.frameContentHash, checksums.realContentHash);
}

TEST(ChecksumTest, FailureOnlyCompressed)
{
    std::string src =
            "hello world, I am some data to compress, hello world hello";
    src.resize(src.size() + 100);

    ChecksumFlags flags      = { false, false };
    flags.compressedChecksum = true;
    auto compressed          = compress(src, flags);

    compressed[compressed.size() - 4] = compressed[compressed.size() - 4] ^ 1;

    auto const [report, out] = decompress(compressed);
    ASSERT_EQ(ZL_RES_error(report)._code, ZL_ErrorCode_compressedChecksumWrong);

    auto const checksums = getChecksums(src, compressed, flags);
    ASSERT_NE(checksums.frameCompressedHash, checksums.realCompressedHash);
}

TEST(ChecksumTest, FailureOnlyContent)
{
    std::string src =
            "hello world, I am some data to compress, hello world hello";
    src.resize(src.size() + 100);

    ChecksumFlags flags   = { false, false };
    flags.contentChecksum = true;
    auto compressed       = compress(src, flags);

    compressed[compressed.size() - 4] = compressed[compressed.size() - 4] ^ 1;

    auto const [report, out] = decompress(compressed);
    ASSERT_EQ(ZL_RES_error(report)._code, ZL_ErrorCode_contentChecksumWrong);

    auto const checksums = getChecksums(src, compressed, flags);
    ASSERT_NE(checksums.frameContentHash, checksums.realContentHash);
}

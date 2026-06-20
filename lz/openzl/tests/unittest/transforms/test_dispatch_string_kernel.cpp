// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <numeric> // accumulate

#include "openzl/codecs/dispatch_string/common_dispatch_string.h"
#include "openzl/codecs/dispatch_string/decode_dispatch_string_kernel.h"
#include "openzl/codecs/dispatch_string/encode_dispatch_string_kernel.h"

using namespace ::testing;

namespace { // anonymous namespace

const std::string text =
        "O glaube, mein Herz, o glaube: "
        "Es geht dir nichts verloren! "
        "Dein ist, ja dein, was du gesehnt, "
        "Dein, was du geliebt, was du gestritten! "
        "O glaube: Du wardst nicht umsonst geboren! "
        "Hast nicht umsonst gelebt, gelitten! "
        "Was entstanden ist, das muß vergehen! "
        "Was vergangen, auferstehen! "
        "Hör auf zu beben! "
        "Bereite dich zu leben!";

class DispatchStringKernelTest : public ::testing::Test {
   protected:
    void SetUp() override
    {
        src        = text.c_str();
        size_t ptr = 0;
        std::vector<uint32_t> sizes;
        while (1) {
            const auto idx = text.find(' ', ptr + 1);
            if (idx == std::string::npos) {
                sizes.push_back(text.size() - ptr);
                break;
            } else {
                sizes.push_back(idx - ptr);
                ptr = idx;
            }
        }
        std::cout << "setup {";
        for (auto i : sizes) {
            std::cout << i << " ";
        }
        std::cout << "}" << std::endl;
        nbStrs     = sizes.size();
        srcStrLens = (uint32_t*)malloc(nbStrs * sizeof(uint32_t));
        for (size_t i = 0; i < nbStrs; ++i) {
            srcStrLens[i] = sizes[i];
        }
    }

    void TearDown() override
    {
        free(srcStrLens);
    }

    void lateSetUp(uint16_t nbDsts, uint16_t* indices)
    {
        std::cout << "late" << std::endl;
        std::cout << nbStrs << std::endl;
        for (auto i = 0u; i < nbStrs; ++i) {
            std::cout << (int)indices[i] << ' ';
        }
        std::cout << std::endl;
        expectedDstStrLens = { nbDsts, std::vector<uint32_t>() };
        expectedDstBuffers = { nbDsts, "" };
        size_t currStrPtr  = 0;
        for (auto i = 0u; i < nbStrs; ++i) {
            const auto strLen = srcStrLens[i];
            std::cout << strLen << " ";
            expectedDstStrLens[indices[i]].push_back(strLen);
            expectedDstBuffers[indices[i]] += text.substr(currStrPtr, strLen);
            currStrPtr += strLen;
        }
        std::cout << std::endl;
        std::cout << expectedDstBuffers[0] << std::endl;
        dstBuffers = (void**)calloc(nbDsts, sizeof(dstBuffers[0]));
        dstStrLens = (uint32_t**)calloc(nbDsts, sizeof(dstStrLens[0]));
        dstSizes   = (size_t*)calloc(nbDsts, sizeof(dstSizes[0]));
        for (auto i = 0; i < nbDsts; ++i) {
            dstStrLens[i] = (uint32_t*)malloc(
                    expectedDstStrLens[i].size() * sizeof(dstStrLens[0][0]));
            const auto sum = std::accumulate(
                    expectedDstStrLens[i].begin(),
                    expectedDstStrLens[i].end(),
                    0lu);
            std::cout << "sum: " << sum << std::endl;
            dstBuffers[i] = (void*)malloc(sum + ZL_DISPATCH_STRING_BLK_SIZE);
        }
        // workaround for strict aliasing rules
        dstBuffersChar = (char**)calloc(nbDsts, sizeof(dstBuffers[0]));
        for (auto i = 0; i < nbDsts; ++i) {
            dstBuffersChar[i] = (char*)dstBuffers[i];
        }
    }

    void earlyTearDown(uint16_t nbDsts)
    {
        free(dstSizes);
        for (auto i = 0; i < nbDsts; ++i) {
            free(dstStrLens[i]);
            free(dstBuffers[i]);
        }
        free(dstStrLens);
        free(dstBuffersChar);
        free(dstBuffers);
    }

    void** dstBuffers;
    char** dstBuffersChar;
    uint32_t** dstStrLens;
    size_t* dstSizes;

    const void* src;
    uint32_t* srcStrLens;
    uint32_t nbStrs;

    std::vector<std::string> expectedDstBuffers;
    std::vector<std::vector<uint32_t>> expectedDstStrLens;
};

TEST_F(DispatchStringKernelTest, RoundtripOne)
{
    std::vector<uint16_t> indices(nbStrs, 0);
    lateSetUp(1, indices.data());

    // encode
    ZL_DispatchString_encode16(
            1,
            dstBuffers,
            dstStrLens,
            dstSizes,
            src,
            srcStrLens,
            nbStrs,
            indices.data());
    EXPECT_EQ(dstSizes[0], nbStrs);
    for (auto i = 0u; i < expectedDstStrLens[0].size(); ++i) {
        EXPECT_EQ(expectedDstStrLens[0][i], dstStrLens[0][i]);
    }
    EXPECT_EQ(
            expectedDstBuffers[0],
            std::string((char*)dstBuffers[0], expectedDstBuffers[0].size()));

    // decode
    std::vector<char> roundtripDst(
            text.size() + ZL_DISPATCH_STRING_BLK_SIZE, 0);
    std::vector<uint32_t> roundtripDstStrLens(nbStrs, 0);
    ZL_DispatchString_decode16(
            roundtripDst.data(),
            roundtripDstStrLens.data(),
            nbStrs,
            1,
            dstBuffersChar,
            dstStrLens,
            dstSizes,
            indices.data());

    // strip padding
    EXPECT_EQ(
            text,
            std::string(
                    roundtripDst.data(),
                    roundtripDst.size() - ZL_DISPATCH_STRING_BLK_SIZE));
    for (auto i = 0u; i < nbStrs; ++i) {
        EXPECT_EQ(roundtripDstStrLens[i], srcStrLens[i]);
    }

    earlyTearDown(1);
}

TEST_F(DispatchStringKernelTest, RoundtripMany)
{
    const size_t maxSplits = 16;
    std::vector<uint16_t> indices(nbStrs);
    for (auto i = 0u; i < nbStrs; ++i) {
        indices[i] = (uint16_t)i % maxSplits;
    }
    lateSetUp(maxSplits, indices.data());

    // encode
    ZL_DispatchString_encode16(
            maxSplits,
            dstBuffers,
            dstStrLens,
            dstSizes,
            src,
            srcStrLens,
            nbStrs,
            indices.data());
    for (auto i = 0u; i < expectedDstStrLens.size(); ++i) {
        EXPECT_EQ(dstSizes[i], expectedDstStrLens[i].size());
        for (auto j = 0u; j < expectedDstStrLens[i].size(); ++j) {
            EXPECT_EQ(expectedDstStrLens[i][j], dstStrLens[i][j]);
        }
    }
    for (auto i = 0u; i < expectedDstBuffers.size(); ++i) {
        EXPECT_EQ(
                expectedDstBuffers[i],
                std::string(
                        (char*)dstBuffers[i], expectedDstBuffers[i].size()));
    }

    // decode
    std::vector<char> roundtripDst(
            text.size() + ZL_DISPATCH_STRING_BLK_SIZE, 0);
    std::vector<uint32_t> roundtripDstStrLens(nbStrs, 0);
    ZL_DispatchString_decode16(
            roundtripDst.data(),
            roundtripDstStrLens.data(),
            nbStrs,
            maxSplits,
            dstBuffersChar,
            dstStrLens,
            dstSizes,
            indices.data());

    // strip padding
    EXPECT_EQ(
            text,
            std::string(
                    roundtripDst.data(),
                    roundtripDst.size() - ZL_DISPATCH_STRING_BLK_SIZE));
    for (auto i = 0u; i < nbStrs; ++i) {
        EXPECT_EQ(roundtripDstStrLens[i], srcStrLens[i]);
    }

    earlyTearDown(16);
}

} // anonymous namespace

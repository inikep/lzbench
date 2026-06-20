// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gtest/gtest.h>

#include "openzl/openzl.hpp"
#include "openzl/zl_compressor.h"

namespace openzl::tests {

class InterleaveTest : public ::testing::Test {
   public:
    void roundtrip(const std::vector<openzl::Input>& inputs)
    {
        auto compressor    = openzl::Compressor();
        auto compressorPtr = compressor.get();
        const auto gid     = ZL_Compressor_registerStaticGraph_fromNode1o(
                compressorPtr, ZL_NODE_INTERLEAVE_STRING, ZL_GRAPH_STORE);
        ASSERT_NE(gid.gid, ZL_GRAPH_ILLEGAL.gid);
        auto rep = ZL_Compressor_selectStartingGraphID(compressorPtr, gid);
        ASSERT_FALSE(ZL_isError(rep));
        auto cctx = openzl::CCtx();
        cctx.setParameter(openzl::CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        cctx.refCompressor(compressor);
        auto comped = cctx.compress(inputs);

        auto dctx        = openzl::DCtx();
        const auto regen = dctx.decompress(comped);
        ASSERT_EQ(regen.size(), inputs.size());
        for (size_t i = 0; i < regen.size(); ++i) {
            ASSERT_EQ(regen[i], inputs[i]);
        }
    }

    void roundtripCompressionMayFail(const std::vector<openzl::Input>& inputs)
    {
        auto compressor    = openzl::Compressor();
        auto compressorPtr = compressor.get();
        const auto gid     = ZL_Compressor_registerStaticGraph_fromNode1o(
                compressorPtr, ZL_NODE_INTERLEAVE_STRING, ZL_GRAPH_STORE);
        ASSERT_NE(gid.gid, ZL_GRAPH_ILLEGAL.gid);
        auto rep = ZL_Compressor_selectStartingGraphID(compressorPtr, gid);
        ASSERT_FALSE(ZL_isError(rep));
        auto cctx = openzl::CCtx();
        cctx.setParameter(openzl::CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        cctx.refCompressor(compressor);
        std::string comped;
        try {
            comped = cctx.compress(inputs);
        } catch (openzl::Exception&) {
            return;
        }
        auto dctx        = openzl::DCtx();
        const auto regen = dctx.decompress(comped);
        ASSERT_EQ(regen.size(), inputs.size());
        for (size_t i = 0; i < regen.size(); ++i) {
            ASSERT_EQ(regen[i], inputs[i]);
        }
    }
};

} // namespace openzl::tests

// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <benchmark/benchmark.h>
#include <math.h>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

#include "benchmark/e2e/e2e_zstrong_utils.h"

#include "openzl/common/assertion.h"
#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h" // ZL_Compressor_registerStaticGraph_fromPipelineNodes1o
#include "openzl/zl_ctransform.h" // ZL_PipeEncoderDesc
#include "openzl/zl_decompress.h" // ZL_decompress
#include "openzl/zl_errors.h"
#include "openzl/zl_opaque_types.h"
#include "openzl/zl_public_nodes.h"
#include "openzl/zl_reflection.h"

namespace zstrong::bench::e2e {

using namespace zstrong::bench::e2e::utils;

/**
 * ZstrongCompressor:
 * A base class for Zstrong based compressors that support compression and
 * decompression benchmarking. All one needs to do to create a new compressor is
 * inherit it an implement and `getGraph` and `name`.
 */
class ZstrongCompressor {
   private:
    /**
     * configureGraph:
     * Gets a @cgraph, configures it and returns the starting GraphID for
     * compression.
     */
    virtual ZL_GraphID configureGraph(ZL_Compressor* cgraph) = 0;

    /**
     * registerDTransforms:
     * Registers custom transforms' decoders to the DCtx.
     */
    virtual void registerDTransforms(ZL_DCtx* dctx)
    {
        (void)dctx;
    }

    /**
     * getGraph:
     * Creates a new CGraph, uses `configureGraph` to configure it and returns
     * the created unique pointer to the user.
     */
    CGraph_unique getGraph();

    /**
     * dctx_:
     * The decompressor context which is reused across decompressions.
     */
    DCTX_unique dctx_;

    /**
     * cctx_:
     * The compressor context which is reused across compressions.
     */
    CCTX_unique cctx_;

   public:
    ZstrongCompressor() : dctx_(createDCTX()), cctx_(createCCTX()) {}
    virtual ~ZstrongCompressor() = default;

    /**
     * compress:
     * Compresses src into output.
     */
    void compress(const std::string_view src, std::vector<uint8_t>& output);
    void compress(
            const std::vector<uint8_t>& src,
            std::vector<uint8_t>& output);

    /**
     * decompress:
     * Decompresses src into output.
     */
    void decompress(const std::string_view src, std::vector<uint8_t>& output);
    void decompress(
            const std::vector<uint8_t>& src,
            std::vector<uint8_t>& output);

    /**
     * roundtrip:
     * Verifies roundtrip compression of src, raises an exception if roundtrip
     * fails, returns size of compressed data.
     */
    size_t roundtrip(const std::string_view src);

    /**
     * benchCompression:
     * Benchmarks compression using given benchmarking state.
     * `state` shouldn't be used outside of this function.
     * Adds the following metrics:
     * - Size - src size
     * - CompressedSize - compressed src size
     * - CompressionRatio - compression ratio of src
     * - bytes_per_second - compression speed
     */
    virtual void benchCompression(
            benchmark::State& state,
            const std::string_view src);
    void benchCompressions(
            benchmark::State& state,
            const std::vector<std::string_view>& srcs);

    /**
     * benchDecompression:
     * Benchmarks decompression using given benchmarking state.
     * `state` shouldn't be used outside of this function.
     * Adds the following metrics:
     * - Size - src size
     * - CompressedSize - compressed src size
     * - CompressionRatio - compression ratio of src
     * - bytes_per_second - decompression speed
     */
    virtual void benchDecompression(
            benchmark::State& state,
            const std::string_view src);
    void benchDecompressions(
            benchmark::State& state,
            const std::vector<std::string_view>& srcs);

    virtual std::string name() = 0;
};

/**
 * ZstrongCompressorStandard:
 * A simpled Zstrong compressor that only executes one standard graph given by
 * `gid`.
 */
class ZstrongCompressorStandard : public ZstrongCompressor {
   private:
    ZL_GraphID _gid;
    std::string _name;
    ZL_GraphID configureGraph(ZL_Compressor* cgraph) override
    {
        (void)cgraph;
        return _gid;
    }

   public:
    ZstrongCompressorStandard(ZL_GraphID gid, std::string name)
            : _gid(gid), _name(name)
    {
    }
    ~ZstrongCompressorStandard() = default;
    std::string name() override
    {
        return _name;
    }
};

class ZstrongCompressorNode : public ZstrongCompressor {
   public:
    ZstrongCompressorNode(
            std::string name,
            size_t eltWidth,
            std::function<ZL_NodeID(ZL_Compressor* cgraph)> nodeGen)
            : name_(std::move(name)), eltWidth_(eltWidth), nodeGen_(nodeGen)
    {
    }

   private:
    ZL_GraphID configureGraph(ZL_Compressor* cgraph) override
    {
        ZL_NodeID nid          = nodeGen_(cgraph);
        size_t const nbOutputs = ZL_Compressor_Node_getNumOutcomes(cgraph, nid);
        std::vector<ZL_GraphID> dsts(nbOutputs, ZL_GRAPH_STORE);
        auto gid = ZL_Compressor_registerStaticGraph_fromNode(
                cgraph, nid, dsts.data(), dsts.size());
        return addConversionFromSerial(cgraph, gid, eltWidth_);
    }

    std::string name() override
    {
        return name_;
    }

    std::string name_;
    size_t eltWidth_;
    std::function<ZL_NodeID(ZL_Compressor* cgraph)> nodeGen_;
};

class ZstrongCompressorStandardNode : public ZstrongCompressorNode {
   public:
    ZstrongCompressorStandardNode(
            ZL_NodeID nid,
            std::string name,
            size_t eltWidth)
            : ZstrongCompressorNode(
                      std::move(name),
                      eltWidth,
                      [nid](ZL_Compressor*) { return nid; })
    {
    }
};

class ZstrongStringCompressor : public ZstrongCompressor {
   private:
    ZL_SetStringLensParserFn parser_;
    ZL_SetStringLensInstructions fieldSizes_;

    ZL_GraphID configureGraph(ZL_Compressor* cgraph) override final
    {
        ZL_NodeID stringConvNode =
                ZL_Compressor_registerConvertSerialToStringNode(
                        cgraph, parser_, &fieldSizes_);
        ZL_GraphID outGraph = configureStringGraph(cgraph);
        return ZL_Compressor_registerStaticGraph_fromNode1o(
                cgraph, stringConvNode, outGraph);
    }

   public:
    explicit ZstrongStringCompressor(ZL_SetStringLensInstructions fieldSizes)
            : fieldSizes_(fieldSizes)
    {
        parser_ = [](ZL_SetStringLensState* state,
                     const ZL_Input* in) -> ZL_SetStringLensInstructions {
            (void)in;
            ZL_ASSERT_NN(state);
            auto fs = (const ZL_SetStringLensInstructions*)
                    ZL_SetStringLensState_getOpaquePtr(state);
            return *fs;
        };
    }

    virtual ZL_GraphID configureStringGraph(ZL_Compressor* cgraph) = 0;
};

/**
 * A Zstrong compressor for variable-sized fields. Hooks up a serial -> String
 * conversion node to the graph defined by `node_`, `dstGids_`, and `nbGids_`
 */
class ZstrongStringStandardNodeCompressor : public ZstrongStringCompressor {
   private:
    ZL_NodeID node_;
    std::string name_;

   public:
    ZstrongStringStandardNodeCompressor(
            ZL_NodeID node,
            const std::string& name,
            ZL_SetStringLensInstructions fieldSizes)
            : ZstrongStringCompressor(fieldSizes), node_(node), name_(name)
    {
    }

    std::string name() override
    {
        return name_;
    }

    ZL_GraphID configureStringGraph(ZL_Compressor* cgraph) override
    {
        ZL_NodeID nid          = node_;
        size_t const nbOutputs = ZL_Compressor_Node_getNumOutcomes(cgraph, nid);
        std::vector<ZL_GraphID> dsts(nbOutputs, ZL_GRAPH_STORE);
        return ZL_Compressor_registerStaticGraph_fromNode(
                cgraph, nid, dsts.data(), dsts.size());
    }
};

} // namespace zstrong::bench::e2e

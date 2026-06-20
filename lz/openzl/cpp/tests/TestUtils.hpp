// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>

#include <gtest/gtest.h>

#include "openzl/openzl.hpp"

namespace openzl {

inline std::string
testRoundTrip(CCtx& cctx, DCtx& dctx, poly::span<const Input> inputs)
{
    auto compressed   = cctx.compress(inputs);
    auto decompressed = dctx.decompress(compressed);
    EXPECT_EQ(decompressed.size(), inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        EXPECT_EQ(decompressed[i], inputs[i]);
    }
    return compressed;
}

inline std::string testRoundTrip(CCtx& cctx, DCtx& dctx, const Input& input)
{
    return testRoundTrip(cctx, dctx, { &input, 1 });
}

inline std::string testRoundTrip(CCtx& cctx, poly::span<const Input> inputs)
{
    DCtx dctx;
    return testRoundTrip(cctx, dctx, inputs);
}

inline std::string testRoundTrip(CCtx& cctx, const Input& input)
{
    return testRoundTrip(cctx, { &input, 1 });
}

inline std::string testRoundTrip(
        const Compressor& compressor,
        DCtx& dctx,
        poly::span<const Input> inputs)
{
    CCtx cctx;
    if (compressor.getParameter(CParam::FormatVersion) == 0) {
        cctx.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    }
    cctx.refCompressor(compressor);
    return testRoundTrip(cctx, dctx, inputs);
}

inline std::string
testRoundTrip(const Compressor& compressor, DCtx& dctx, const Input& input)
{
    return testRoundTrip(compressor, dctx, { &input, 1 });
}

inline std::string testRoundTrip(
        const Compressor& compressor,
        poly::span<const Input> inputs)
{
    DCtx dctx;
    return testRoundTrip(compressor, dctx, inputs);
}

inline std::string testRoundTrip(
        const Compressor& compressor,
        const Input& input)
{
    return testRoundTrip(compressor, { &input, 1 });
}

class RunNodeThenGraphFunctionGraph : public FunctionGraph {
   public:
    RunNodeThenGraphFunctionGraph(NodeID node, GraphID graph)
            : node_(node), graph_(graph)
    {
    }

    RunNodeThenGraphFunctionGraph()
            : RunNodeThenGraphFunctionGraph(ZL_NODE_ILLEGAL, ZL_GRAPH_ILLEGAL)
    {
    }

    enum class Params {
        NodeParam  = 0,
        GraphParam = 1,
    };

    FunctionGraphDescription functionGraphDescription() const override
    {
        FunctionGraphDescription desc = {
            .name                = "RunNodeThenGraph",
            .inputTypeMasks      = { TypeMask::Any },
            .lastInputIsVariable = true,
            .customGraphs        = { graph_ },
            .customNodes         = { node_ },
        };
        return desc;
    }

    void graph(GraphState& state) const override
    {
        auto nodes  = state.customNodes();
        auto graphs = state.customGraphs();
        auto nodeIdx =
                state.getLocalIntParam(int(Params::NodeParam)).value_or(0);
        auto graphIdx =
                state.getLocalIntParam(int(Params::GraphParam)).value_or(0);
        auto out = Edge::runMultiInputNode(state.edges(), nodes[nodeIdx]);
        Edge::setMultiInputDestination(out, graphs[graphIdx]);
    }

   private:
    NodeID node_;
    poly::optional<LocalParams> params_;
    GraphID graph_;
};

class NoOpCustomEncoder : public CustomEncoder {
   public:
    NoOpCustomEncoder(unsigned id, std::string name, Type type)
            : id_(id), name_(std::move(name)), type_(type)
    {
    }

    virtual void preEncodeHook(EncoderState& encoder) const {}

    SimpleCodecDescription simpleCodecDescription() const override
    {
        return SimpleCodecDescription{ .id          = id_,
                                       .name        = name_,
                                       .inputType   = type_,
                                       .outputTypes = { type_ } };
    }

    void encode(EncoderState& encoder) const override
    {
        preEncodeHook(encoder);
        auto& input = encoder.inputs()[0];
        auto output =
                encoder.createOutput(0, input.numElts(), input.eltWidth());
        memcpy(output.ptr(), input.ptr(), input.contentSize());
        output.commit(input.numElts());
    }

   private:
    unsigned id_;
    std::string name_;
    Type type_;
};

class NoOpCustomDecoder : public CustomDecoder {
   public:
    NoOpCustomDecoder(unsigned id, std::string name, Type type)
            : id_(id), name_(std::move(name)), type_(type)
    {
    }

    virtual void preDecodeHook(DecoderState& decoder) const {}

    SimpleCodecDescription simpleCodecDescription() const override
    {
        return SimpleCodecDescription{ .id          = id_,
                                       .name        = name_,
                                       .inputType   = type_,
                                       .outputTypes = { type_ } };
    }

    void decode(DecoderState& decoder) const override
    {
        preDecodeHook(decoder);
        auto& input = decoder.singletonInputs()[0];
        auto output =
                decoder.createOutput(0, input.numElts(), input.eltWidth());
        memcpy(output.ptr(), input.ptr(), input.contentSize());
        output.commit(input.numElts());
    }

   private:
    unsigned id_;
    std::string name_;
    Type type_;
};

} // namespace openzl

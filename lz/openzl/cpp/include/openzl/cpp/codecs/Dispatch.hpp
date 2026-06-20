// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_dispatch.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/FunctionGraph.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
class DispatchSerial : public Node {
   public:
    static constexpr NodeMetadata<1, 2, 1> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Serial } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric,
                                              .name = "tags" },
                              OutputMetadata{ .type = Type::Numeric,
                                              .name = "sizes" } },
        .variableOutputs  = { OutputMetadata{ .type = Type::Serial,
                                              .name = "dispatched" } },
        .description =
                "Dispatch serial data into one of the `dispatched` variable outputs "
                "according to the `Instructions`."
    };

    struct Instructions {
        poly::span<const unsigned> segmentTags;
        poly::span<const size_t> segmentSizes;
        unsigned numTags;
    };

    explicit DispatchSerial(const Instructions& instructions)
            : instructions_(instructions)
    {
    }

    NodeID baseNode() const override
    {
        throw Exception("DispatchSerial: Can only call run()");
    }

    Edge::RunNodeResult run(Edge& edge) const override
    {
        if (instructions_.segmentSizes.size()
            != instructions_.segmentTags.size()) {
            throw Exception(
                    "Bad instructions: Must have same number of sizes & tags");
        }
        ZL_DispatchInstructions inst = {
            .segmentSizes = instructions_.segmentSizes.data(),
            .tags         = instructions_.segmentTags.data(),
            .nbSegments   = instructions_.segmentTags.size(),
            .nbTags       = instructions_.numTags,
        };
        auto edges = unwrap(ZL_Edge_runDispatchNode(edge.get(), &inst));
        return Edge::convert(edges);
    }

    ~DispatchSerial() override = default;

   private:
    Instructions instructions_;
};

class DispatchString : public Node {
   public:
    static constexpr NodeMetadata<1, 1, 1> metadata = {
        .inputs           = { InputMetadata{ .type = Type::String } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric,
                                              .name = "tags" } },
        .variableOutputs  = { OutputMetadata{ .type = Type::String,
                                              .name = "dispatched" } },
        .description =
                "Dispatch serial data into one of the `dispatched` variable outputs "
                "according to the `tags`"
    };

    DispatchString(poly::span<const uint16_t> tags, int numTags)
            : tags_(tags), numTags_(numTags)
    {
    }

    NodeID baseNode() const override
    {
        throw Exception("DispatchString: Can only call run()");
    }

    Edge::RunNodeResult run(Edge& edge) const override
    {
        if (tags_.size() != edge.getInput().numElts()) {
            throw Exception(
                    "DispatchString requires the same number of tags as strings");
        }
        auto edges = unwrap(ZL_Edge_runDispatchStringNode(
                edge.get(), numTags_, tags_.data()));
        return Edge::convert(edges);
    }

   private:
    poly::span<const uint16_t> tags_;
    int numTags_;
};

} // namespace nodes
} // namespace openzl

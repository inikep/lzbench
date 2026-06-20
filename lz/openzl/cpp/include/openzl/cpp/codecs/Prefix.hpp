// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_prefix.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
class Prefix : public Node {
   public:
    static inline constexpr NodeID node = ZL_NODE_PREFIX;

    static constexpr NodeMetadata<1, 2> metadata = {
        .inputs = { InputMetadata{ .type = Type::String, .name = "strings" } },
        .singletonOutputs = { OutputMetadata{ .type = Type::String,
                                              .name = "suffixes" },
                              OutputMetadata{ .type = Type::Numeric,
                                              .name = "prefix lengths" } },
        .description = "Remove shared prefixes between consecutive elements",
    };

    Prefix() = default;

    NodeID baseNode() const override
    {
        return node;
    }

    GraphID operator()(
            Compressor& compressor,
            GraphID suffixes,
            GraphID prefixLengths) const
    {
        return buildGraph(
                compressor,
                std::initializer_list<GraphID>{ suffixes, prefixLengths });
    }

    ~Prefix() override = default;

    NodeID operator()() const
    {
        return node;
    }

    Edge::RunNodeResult operator()(Edge& edge) const
    {
        return edge.runNode(node);
    }
};
} // namespace nodes
} // namespace openzl

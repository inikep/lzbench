// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_parse_int.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
class ParseInt : public SimplePipeNode<ParseInt> {
   public:
    static constexpr NodeID node = ZL_NODE_PARSE_INT;

    static constexpr NodeMetadata<1, 1> metadata = {
        .inputs           = { InputMetadata{ .type = Type::String,
                                             .name = "ascii int64s" } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric,
                                              .name = "int64s" } },
        .description      = "Parse ASCII integers into int64_t",
    };

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

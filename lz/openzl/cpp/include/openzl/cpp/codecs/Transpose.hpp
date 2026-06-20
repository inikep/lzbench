// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_transpose.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
class TransposeSplit : public SimplePipeNode<TransposeSplit> {
   public:
    static constexpr NodeID node = ZL_NODE_TRANSPOSE_SPLIT;

    static constexpr NodeMetadata<1, 0, 1> metadata = {
        .inputs          = { InputMetadata{ .type = Type::Struct } },
        .variableOutputs = { OutputMetadata{ .type = Type::Serial,
                                             .name = "lanes" } },
        .description =
                "Transpose the input structs into their lanes, and produce one output per lane"
    };
};
} // namespace nodes
} // namespace openzl

// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_delta.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
class DeltaInt : public SimplePipeNode<DeltaInt> {
   public:
    static constexpr NodeID node = ZL_NODE_DELTA_INT;

    static constexpr NodeMetadata<1, 1> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Numeric } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric,
                                              .name = "deltas" } },
        .description =
                "Output the deltas between each int in the input. "
                "The first value is written into the header."
    };
};
} // namespace nodes
} // namespace openzl

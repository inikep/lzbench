// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_range_pack.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
class RangePack : public SimplePipeNode<RangePack> {
   public:
    static inline constexpr NodeID node = ZL_NODE_RANGE_PACK;

    static constexpr NodeMetadata<1, 1> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Numeric } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric } },
        .description =
                "Subtract the minimum value and pack into the smallest possible "
                "integer width",
    };
};
} // namespace nodes
} // namespace openzl

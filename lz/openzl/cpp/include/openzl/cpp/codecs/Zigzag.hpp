// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_zigzag.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
struct Zigzag : public SimplePipeNode<Zigzag> {
   public:
    static constexpr NodeID node = ZL_NODE_ZIGZAG;

    static constexpr NodeMetadata<1, 1> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Numeric } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric } },
        .description      = "Zigzag encode the input values",
    };
};
} // namespace nodes
} // namespace openzl

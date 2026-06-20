// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_dedup.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
class DedupNumeric : public SimplePipeNode<DedupNumeric> {
   public:
    static constexpr NodeID node = ZL_NODE_DEDUP_NUMERIC;

    static constexpr NodeMetadata<1, 1> metadata = {
        .inputs              = { InputMetadata{ .type = Type::Numeric,
                                                .name = "duplicated" } },
        .singletonOutputs    = { OutputMetadata{ .type = Type::Numeric,
                                                 .name = "deduped" } },
        .lastInputIsVariable = true,
        .description =
                "Takes N numeric inputs containing exactly the same data & outputs a single copy",
    };
};

} // namespace nodes
} // namespace openzl

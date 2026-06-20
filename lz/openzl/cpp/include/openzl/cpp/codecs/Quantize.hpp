// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_entropy.h"
#include "openzl/codecs/zl_quantize.h"
#include "openzl/codecs/zl_store.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
class QuantizeOffsets : public Node {
   public:
    static constexpr NodeID node = ZL_NODE_QUANTIZE_OFFSETS;

    static constexpr NodeMetadata<1, 2> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Numeric } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric,
                                              .name = "codes" },
                              OutputMetadata{ .type = Type::Serial,
                                              .name = "extra_bits" } },
        .description =
                "Quantize uint32_t values != 0 using a power-of-2 scheme",
    };

    QuantizeOffsets() = default;

    NodeID baseNode() const override
    {
        return node;
    }

    GraphID operator()(
            Compressor& compressor,
            GraphID codes     = ZL_GRAPH_FSE,
            GraphID extraBits = ZL_GRAPH_STORE) const
    {
        return buildGraph(
                compressor, std::initializer_list<GraphID>{ codes, extraBits });
    }

    ~QuantizeOffsets() override = default;
};

class QuantizeLengths : public Node {
   public:
    static constexpr NodeID node = ZL_NODE_QUANTIZE_LENGTHS;

    static constexpr NodeMetadata<1, 2> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Numeric } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric,
                                              .name = "codes" },
                              OutputMetadata{ .type = Type::Serial,
                                              .name = "extra_bits" } },
        .description =
                "Quantize uint32_t values giving small values a unique code and large values a code based on their log2",
    };

    QuantizeLengths() = default;

    NodeID baseNode() const override
    {
        return node;
    }

    GraphID operator()(
            Compressor& compressor,
            GraphID codes     = ZL_GRAPH_FSE,
            GraphID extraBits = ZL_GRAPH_STORE) const
    {
        return buildGraph(
                compressor, std::initializer_list<GraphID>{ codes, extraBits });
    }

    ~QuantizeLengths() override = default;
};
} // namespace nodes
} // namespace openzl

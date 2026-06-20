// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_float_deconstruct.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
namespace detail {
template <typename NodeT>
class FloatDeconstructNode : public Node {
   public:
    FloatDeconstructNode() = default;

    NodeID baseNode() const override
    {
        return NodeT::node;
    }

    GraphID
    operator()(Compressor& compressor, GraphID signFrac, GraphID exponent) const
    {
        return buildGraph(
                compressor,
                std::initializer_list<GraphID>{ signFrac, exponent });
    }

    ~FloatDeconstructNode() override = default;
};
} // namespace detail

class Float32Deconstruct
        : public detail::FloatDeconstructNode<Float32Deconstruct> {
   public:
    static constexpr NodeID node = ZL_NODE_FLOAT32_DECONSTRUCT;

    static constexpr NodeMetadata<1, 2> metadata = {
        .inputs = { InputMetadata{ .type = Type::Numeric, .name = "floats" } },
        .singletonOutputs = { OutputMetadata{
                                      .type = Type::Struct,
                                      .name = "sign+fraction bits (24-bits)" },
                              OutputMetadata{
                                      .type = Type::Serial,
                                      .name = "exponent bits (8-bits)" } },
        .description      = "Separate float exponents from sign+fraction"
    };
};

class BFloat16Deconstruct
        : public detail::FloatDeconstructNode<BFloat16Deconstruct> {
   public:
    static constexpr NodeID node = ZL_NODE_BFLOAT16_DECONSTRUCT;

    static constexpr NodeMetadata<1, 2> metadata = {
        .inputs = { InputMetadata{ .type = Type::Numeric, .name = "floats" } },
        .singletonOutputs = { OutputMetadata{
                                      .type = Type::Struct,
                                      .name = "sign+fraction bits (8-bits)" },
                              OutputMetadata{
                                      .type = Type::Serial,
                                      .name = "exponent bits (8-bits)" } },
        .description      = "Separate float exponents from sign+fraction"
    };
};

class Float16Deconstruct
        : public detail::FloatDeconstructNode<Float16Deconstruct> {
   public:
    static constexpr NodeID node = ZL_NODE_FLOAT16_DECONSTRUCT;

    static constexpr NodeMetadata<1, 2> metadata = {
        .inputs = { InputMetadata{ .type = Type::Numeric, .name = "floats" } },
        .singletonOutputs = { OutputMetadata{
                                      .type = Type::Struct,
                                      .name = "sign+fraction bits (11-bits)" },
                              OutputMetadata{
                                      .type = Type::Serial,
                                      .name = "exponent bits (5-bits)" } },
        .description      = "Separate float exponents from sign+fraction"
    };
};

} // namespace nodes
} // namespace openzl

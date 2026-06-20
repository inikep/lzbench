// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <initializer_list>

#include "openzl/codecs/zl_concat.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/Exception.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
namespace detail {
constexpr NodeMetadata<1, 2> concatMetadata(Type type)
{
    return {
        .inputs = { InputMetadata{ .type = type } },
        .singletonOutputs = {
            OutputMetadata{ .type = Type::Numeric, .name = "input lengths"},
            OutputMetadata{ .type = type, .name = "concatenated" },
        },
        .lastInputIsVariable = true,
        .description = "Concatenate all inputs into a single output"
    };
}

template <unsigned kNode, Type kType>
class ConcatNode : public Node {
   public:
    static constexpr NodeID node = NodeID{ kNode };

    static constexpr NodeMetadata<1, 2> metadata = {
        .inputs = { InputMetadata{ .type = kType } },
        .singletonOutputs = {
            OutputMetadata{ .type = Type::Numeric, .name = "input lengths"},
            OutputMetadata{ .type = kType, .name = "concatenated" },
        },
        .lastInputIsVariable = true,
        .description = "Concatenate all inputs into a single output"
    };

    ConcatNode() = default;

    NodeID baseNode() const override
    {
        return node;
    }

    GraphID operator()(
            Compressor& compressor,
            GraphID inputLengths,
            GraphID concatenated)
    {
        return buildGraph(
                compressor,
                std::initializer_list<GraphID>{ inputLengths, concatenated });
    }

    ~ConcatNode() override = default;
};
} // namespace detail

class ConcatSerial
        : public detail::ConcatNode<ZL_NODE_CONCAT_SERIAL.nid, Type::Serial> {};

class ConcatStruct
        : public detail::ConcatNode<ZL_NODE_CONCAT_STRUCT.nid, Type::Struct> {};

class ConcatNumeric
        : public detail::ConcatNode<ZL_NODE_CONCAT_NUMERIC.nid, Type::Numeric> {
};

class ConcatString
        : public detail::ConcatNode<ZL_NODE_CONCAT_STRING.nid, Type::String> {};

class Concat : public Node {
   public:
    explicit Concat(Type type) : baseNode_(getNode(type)) {}

    NodeID baseNode() const override
    {
        return baseNode_;
    }

    GraphID operator()(
            Compressor& compressor,
            GraphID inputLengths,
            GraphID concatenated) const
    {
        return buildGraph(
                compressor,
                std::initializer_list<GraphID>{ inputLengths, concatenated });
    }

    ~Concat() override = default;

   private:
    NodeID getNode(Type type) const
    {
        switch (type) {
            case Type::Serial:
                return ConcatSerial::node;
            case Type::Struct:
                return ConcatStruct::node;
            case Type::Numeric:
                return ConcatNumeric::node;
            case Type::String:
                return ConcatString::node;
            default:
                throw Exception("Concat: Bad type");
        }
    }

    NodeID baseNode_;
};
} // namespace nodes
} // namespace openzl

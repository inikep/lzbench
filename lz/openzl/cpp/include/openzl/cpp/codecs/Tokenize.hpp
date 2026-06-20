// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_tokenize.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
namespace detail {
inline NodeParameters tokenizeParams(bool sort)
{
    NodeParameters params;
    if (sort) {
        params.localParams = LocalParams{};
        params.localParams->addIntParam(ZL_TOKENIZE_SORT_PID, 1);
    }
    return params;
}

template <unsigned kNode, Type kType>
class TokenizeNode : public Node {
   public:
    static constexpr NodeID node = NodeID{ kNode };

    static constexpr NodeMetadata<1, 2> metadata = {
      .inputs = {InputMetadata{.type = kType}},
      .singletonOutputs = {
          OutputMetadata{.type = kType, .name = "alphabet"},
          OutputMetadata{.type = Type::Numeric, .name = "indices"},
      },
      .description = "Tokenize the input struct into an alphabet of unique values and indices into that alphabet",
    };
    TokenizeNode() = default;

    NodeID baseNode() const override
    {
        return node;
    }

    GraphID
    operator()(Compressor& compressor, GraphID alphabet, GraphID indices) const
    {
        return buildGraph(
                compressor,
                std::initializer_list<GraphID>{ alphabet, indices });
    }

    ~TokenizeNode() override = default;
};
} // namespace detail

class TokenizeStruct
        : public detail::
                  TokenizeNode<ZL_NODE_TOKENIZE_STRUCT.nid, Type::Struct> {};

class TokenizeNumeric
        : public detail::
                  TokenizeNode<ZL_NODE_TOKENIZE_NUMERIC.nid, Type::Numeric> {
   public:
    explicit TokenizeNumeric(bool sort = false) : sort_(sort) {}

    poly::optional<NodeParameters> parameters() const override
    {
        return detail::tokenizeParams(sort_);
    }

    ~TokenizeNumeric() override = default;

   private:
    bool sort_;
};

class TokenizeString
        : public detail::
                  TokenizeNode<ZL_NODE_TOKENIZE_STRING.nid, Type::String> {
   public:
    explicit TokenizeString(bool sort = false) : sort_(sort) {}

    poly::optional<NodeParameters> parameters() const override
    {
        return detail::tokenizeParams(sort_);
    }

    ~TokenizeString() override = default;

   private:
    bool sort_;
};

class Tokenize : public Node {
   public:
    explicit Tokenize(Type type, bool sort = false)
            : baseNode_(getNode(type)), sort_(sort)
    {
    }

    NodeID baseNode() const override
    {
        return baseNode_;
    }

    poly::optional<NodeParameters> parameters() const override
    {
        return detail::tokenizeParams(sort_);
    }

    GraphID
    operator()(Compressor& compressor, GraphID alphabet, GraphID indices) const
    {
        return buildGraph(
                compressor,
                std::initializer_list<GraphID>{ alphabet, indices });
    }

    ~Tokenize() override = default;

   private:
    static NodeID getNode(Type type)
    {
        switch (type) {
            case Type::Struct:
                return TokenizeStruct::node;
            case Type::Numeric:
                return TokenizeNumeric::node;
            case Type::String:
                return TokenizeString::node;
            case Type::Serial:
            default:
                throw std::runtime_error("Unsupported type for Tokenize");
        }
    }

    NodeID baseNode_;
    bool sort_;
};
} // namespace nodes
} // namespace openzl

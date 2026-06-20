// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_divide_by.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
class DivideBy : public SimplePipeNode<DivideBy> {
   public:
    static constexpr NodeID node = ZL_NODE_DIVIDE_BY;

    static constexpr NodeMetadata<1, 1> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Numeric } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric } },
        .description =
                "Divide the input by the given divisor or the GCD if none is provided",
    };

    DivideBy() {}
    explicit DivideBy(uint64_t divisor) : divisor_(divisor) {}
    explicit DivideBy(poly::optional<uint64_t> divisor)
            : divisor_(std::move(divisor))
    {
    }

    NodeID operator()() const
    {
        return node;
    }

    poly::optional<NodeParameters> parameters() const override
    {
        if (divisor_.has_value()) {
            LocalParams params;
            params.addCopyParam(ZL_DIVIDE_BY_PID, divisor_.value());
            return NodeParameters{ .localParams = std::move(params) };
        } else {
            return poly::nullopt;
        }
    }

    ~DivideBy() override = default;

   private:
    poly::optional<uint64_t> divisor_;
};

} // namespace nodes
} // namespace openzl

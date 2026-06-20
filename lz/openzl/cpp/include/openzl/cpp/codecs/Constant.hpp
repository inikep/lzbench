// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_constant.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Graph.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"

namespace openzl {
namespace graphs {

class Constant : public SimpleGraph<Constant> {
   public:
    static constexpr GraphID graph = ZL_GRAPH_CONSTANT;

    static constexpr GraphMetadata<1> metadata = {
        .inputs = { InputMetadata{ .typeMask =
                                           TypeMask::Serial | TypeMask::Struct,
                                   .name = "constant data" } },
        .description =
                "Encode a constant input as a singleton value and size pair"
    };

    Constant() = default;

    ~Constant() override = default;
};

} // namespace graphs
} // namespace openzl

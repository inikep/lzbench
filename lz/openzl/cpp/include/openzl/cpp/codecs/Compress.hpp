// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_generic.h"
#include "openzl/cpp/codecs/Graph.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"

namespace openzl {
namespace graphs {

class Compress : public SimpleGraph<Compress> {
   public:
    static constexpr GraphID graph = ZL_GRAPH_COMPRESS_GENERIC;

    static constexpr GraphMetadata<1> metadata = {
        .inputs              = { InputMetadata{ .typeMask = TypeMask::Any } },
        .lastInputIsVariable = true,
        .description = "Compress the inputs using a generic compression backend"
    };

    Compress() = default;

    ~Compress() override = default;
};
} // namespace graphs
} // namespace openzl

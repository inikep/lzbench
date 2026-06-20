// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_store.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Graph.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"

namespace openzl {
namespace graphs {
class Store : public SimpleGraph<Store> {
   public:
    static constexpr GraphID graph = ZL_GRAPH_STORE;

    static constexpr GraphMetadata<1> metadata = {
        .inputs              = { InputMetadata{ .typeMask = TypeMask::Any } },
        .lastInputIsVariable = true,
        .description = "Store the input streams into the compressed frame"
    };
};
} // namespace graphs
} // namespace openzl

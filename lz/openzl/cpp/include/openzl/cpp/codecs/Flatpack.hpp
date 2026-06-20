// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_flatpack.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Graph.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"

namespace openzl {
namespace graphs {

class Flatpack : public SimpleGraph<Flatpack> {
   public:
    static constexpr GraphID graph = ZL_GRAPH_FLATPACK;

    static constexpr GraphMetadata<1> metadata = {
        .inputs      = { InputMetadata{ .typeMask = TypeMask::Serial } },
        .description = "Tokenize + bitpack"
    };
};
} // namespace graphs
} // namespace openzl

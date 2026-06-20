// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_bitpack.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Graph.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"

namespace openzl {
namespace graphs {

class Bitpack : public SimpleGraph<Bitpack> {
   public:
    static constexpr GraphID graph = ZL_GRAPH_BITPACK;

    static constexpr GraphMetadata<1> metadata = {
        .inputs = { InputMetadata{ .typeMask =
                                           TypeMask::Serial | TypeMask::Numeric,
                                   .name = "ints" } },
        .description =
                "Bitpacks ints into the smallest number of bits possible",
    };

    Bitpack() = default;

    ~Bitpack() override = default;
}; // namespace graphs

} // namespace graphs
} // namespace openzl

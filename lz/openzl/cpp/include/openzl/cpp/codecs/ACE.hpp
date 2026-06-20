// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_ace.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Graph.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"

namespace openzl {
namespace graphs {

class ACE : public Graph {
   public:
    static constexpr GraphMetadata<1> metadata = {
        .inputs = { InputMetadata{ .typeMask = TypeMask::Any } },
        .description =
                "Placeholder graph for the Automated Compressor Explorer (ACE) to replace",
    };

    constexpr ACE() : ACE(ZL_GRAPH_COMPRESS_GENERIC) {}
    constexpr explicit ACE(GraphID baseGraph) : baseGraph_(baseGraph) {}

    GraphID baseGraph() const override
    {
        return baseGraph_;
    }

    poly::optional<GraphParameters> parameters() const override
    {
        return GraphParameters{ .name = "zl.ace" };
    }

    GraphID operator()(Compressor& compressor) const
    {
        return compressor.unwrap(ZL_Compressor_buildACEGraphWithDefault2(
                compressor.get(), baseGraph_));
    }

    ~ACE() override = default;

   private:
    GraphID baseGraph_;
};
} // namespace graphs
} // namespace openzl

// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_sddl.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Graph.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/poly/StringView.hpp"

namespace openzl {
namespace graphs {

class SDDL : public Graph {
   public:
    static constexpr GraphID graph = ZL_GRAPH_SDDL;

    static constexpr GraphMetadata<1> metadata = {
        .inputs = { InputMetadata{ .typeMask = TypeMask::Serial } },
        .description =
                "Graph that runs the Simple Data Description Language over the input to decompose the input stream into a number of output streams. Must be given a description and successor. Refer to the SDDL documentation for usage instructions.",
    };

    explicit SDDL(poly::string_view description, GraphID successor)
            : description_(description), successor_(successor)
    {
    }

    GraphID baseGraph() const override
    {
        return graph;
    }

    poly::optional<GraphParameters> parameters() const override
    {
        LocalParams lp;
        lp.addCopyParam(
                ZL_SDDL_DESCRIPTION_PID,
                description_.data(),
                description_.size());
        return GraphParameters{ .customGraphs = { { successor_ } },
                                .localParams  = std::move(lp) };
    }

    ~SDDL() override = default;

   private:
    poly::string_view description_;
    GraphID successor_;
};

} // namespace graphs
} // namespace openzl

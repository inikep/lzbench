// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_brute_force_selector.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/FunctionGraph.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"

namespace openzl {
namespace graphs {

struct BruteForce {
    // TODO(terrelln): Make serializable & expose the GraphID & parameters
    // static constexpr GraphID graph = ZL_GRAPH_BRUTE_FORCE;

    // // TODO(terrelln): Work with multi-input graphs
    // static constexpr GraphMetadata<1> metadata = {
    //     .inputs = { InputMetadata{ .typeMask = TypeMask::Any } },
    //     .description =
    //             "Try each successor graph and choose the one that produces "
    //             "the smallest compressed size"
    // };

    // GraphID operator()() const
    // {
    //     return graph;
    // }

    // GraphParameters params(poly::span<const GraphID> successors) const;

    // GraphID operator()(
    //         Compressor& compressor,
    //         poly::span<const GraphID> successors) const
    // {
    //     return compressor.parameterizeGraph(graph, params(successors));
    // }

    // void operator()(Edge& edge, poly::span<const GraphID> successors) const
    // {
    //     edge.setDestination(graph, params(successors));
    // }
};
inline constexpr BruteForce brute_force;

} // namespace graphs
} // namespace openzl

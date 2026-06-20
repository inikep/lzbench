// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_merge_sorted.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
class MergeSorted : public SimplePipeNode<MergeSorted> {
   public:
    static constexpr NodeID node = ZL_NODE_MERGE_SORTED;

    static constexpr NodeMetadata<1, 2> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Numeric,
                                             .name = "sorted u32 runs" } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric,
                                              .name = "bitset" },
                              OutputMetadata{
                                      .type = Type::Numeric,
                                      .name = "strictly increasing u32s" } },
        .description =
                "Merge <= 64 sorted u32 runs into a bitset telling whether "
                "the i'th run has the next value, and the sorted list of "
                "unique u32 values",
    };
}; // namespace nodes
} // namespace nodes

namespace graphs {
class MergeSorted {
    // TODO(terrelln): Make serializable & expose the GraphID & parameters
    // static inline constexpr GraphID graph = ZL_GRAPH_MERGE_SORTED;

    // GraphID operator()() const
    // {
    //     return graph;
    // }

    // GraphParameters params(GraphID bitsetGraph, GraphID mergedGraph, GraphID
    // backupGraph) const;

    // GraphID operator()(
    //         Compressor& compressor,
    //         GraphID bitsetGraph, GraphID mergedGraph, GraphID backupGraph)
    //         const
    // {
    //     return compressor.parameterizeGraph(graph, params(bitsetGraph,
    //     mergedGraph, backupGraph));
    // }

    // void operator()(Edge& edge, GraphID bitsetGraph, GraphID mergedGraph,
    // GraphID backupGraph) const
    // {
    //     edge.setDestination(graph, params(bitsetGraph, mergedGraph,
    //     backupGraph));
    // }
};
inline constexpr MergeSorted merge_sorted;
} // namespace graphs
} // namespace openzl

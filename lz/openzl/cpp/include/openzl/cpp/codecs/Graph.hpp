// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/FunctionGraph.hpp"

namespace openzl {
namespace graphs {

/**
 * Base class for operating with graphs in OpenZL.
 *
 * The constructor of the class should take in any parameters needed to build
 * the base graph, including a `Compressor&` if needed.
 *
 * Then, subclasses must overload `baseGraph()` and if needed `parameters()`.
 * The rest of the helper methods on the class allow parameterizing graphs
 * during graph construction, and using the graph within a @ref FunctionGraph.
 */
class Graph {
   public:
    /**
     * @returns The @ref GraphID of the graph built in the context of the @p
     * compressor.
     */
    virtual GraphID parameterize(Compressor& compressor) const
    {
        const auto params = parameters();
        if (params.has_value()) {
            return compressor.parameterizeGraph(baseGraph(), *params);
        } else {
            return baseGraph();
        }
    }

    /**
     * Sets the destination of @p edge to this graph, to be used in the context
     * of a @ref FunctionGraph.
     */
    virtual void setDestination(Edge& edge) const
    {
        setMultiInputDestination({ &edge, 1 });
    }

    /**
     * Sets the destination of @p edges to this graph, to be used in the context
     * of a @ref FunctionGraph.
     */
    virtual void setMultiInputDestination(poly::span<Edge> edges) const
    {
        Edge::setMultiInputDestination(edges, baseGraph(), parameters());
    }

    /// This is an alias for @ref parameterize
    GraphID operator()(Compressor& compressor) const
    {
        return parameterize(compressor);
    }

    /**
     * @returns The @ref GraphID of the base graph component.
     *
     * @warning This is the base GraphID, and does not represent a configured
     * component. Using it directly may not work as expected.
     */
    virtual GraphID baseGraph() const = 0;

    /**
     * @returns The parameters to use to parameterize the graph, if any.
     */
    virtual poly::optional<GraphParameters> parameters() const
    {
        return poly::nullopt;
    }

    virtual ~Graph() = default;
};

template <typename GraphT>
class SimpleGraph : public Graph {
   public:
    GraphID baseGraph() const override
    {
        return GraphT::graph;
    }

    using Graph::operator();

    GraphID operator()() const
    {
        return baseGraph();
    }
};

} // namespace graphs
} // namespace openzl

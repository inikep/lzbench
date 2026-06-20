// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/FunctionGraph.hpp"

namespace openzl {
namespace nodes {

/**
 * Base class for operating with nodes in OpenZL.
 *
 * The constructor of the class should take in any parameters needed to build
 * the base node, including a `Compressor&` if needed.
 *
 * Then, subclasses must overload `baseNode()` and if needed `parameters()`.
 * The rest of the helper methods on the class allow building graphs,
 * parameterizing nodes during graph construction, and using the graph within a
 * @ref FunctionGraph.
 */
class Node {
   public:
    /**
     * Run the node on @p edge.
     *
     * @returns The outputs of the node.
     */
    virtual Edge::RunNodeResult run(Edge& edge) const
    {
        return runMultiInput({ &edge, 1 });
    }

    /**
     * Run the node on @p edges.
     *
     * @returns The outputs of the node.
     */
    virtual Edge::RunNodeResult runMultiInput(poly::span<Edge> edges) const
    {
        return Edge::runMultiInputNode(edges, baseNode(), parameters());
    }

    /**
     * Builds a static graph composed of the node followed by passing each
     * output to the corresponding successor in @p successors.
     *
     * @returns The @ref GraphID of the resulting graph.
     */
    virtual GraphID buildGraph(
            Compressor& compressor,
            poly::span<const GraphID> successors) const
    {
        return compressor.buildStaticGraph(
                parameterize(compressor), successors);
    }

    /**
     * Parameterizes the node in the context of the @p compressor to construct a
     * @ref NodeID.
     */
    virtual NodeID parameterize(Compressor& compressor) const
    {
        const auto params = parameters();
        if (params.has_value()) {
            return compressor.parameterizeNode(baseNode(), *params);
        } else {
            return baseNode();
        }
    }

    /**
     * @returns The @ref NodeID of the base node.
     *
     * @warning This is the base NodeID, and does not represent a configured
     * component. Using it directly may not work as expected.
     */
    virtual NodeID baseNode() const = 0;

    /**
     * @returns The parameters to use to parameterize the node, if any.
     */
    virtual poly::optional<NodeParameters> parameters() const
    {
        return poly::nullopt;
    }

    // Nodes are expected to override operator() for graph building with their
    // intended successors. This cannot be a virtual method because each node
    // has their own number of successors.
    // GraphID operator()(Compressor& compressor, GraphID... successors) const;

    virtual ~Node() = default;
};

template <typename NodeT>
class SimplePipeNode : public Node {
   public:
    SimplePipeNode() = default;

    NodeID baseNode() const override
    {
        return NodeT::node;
    }

    GraphID operator()(Compressor& compressor, GraphID successor) const
    {
        return buildGraph(compressor, { &successor, 1 });
    }

    ~SimplePipeNode() override = default;
};

} // namespace nodes
} // namespace openzl

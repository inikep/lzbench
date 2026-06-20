// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <nanobind/nanobind.h>

#include "openzl/ext.hpp"

namespace openzl {
namespace py {
void registerNodesModule(nanobind::module_& m);

class PyNodeBase : public nb::intrusive_base {
   public:
    virtual std::vector<nb::ref<PyEdge>> run(nb::ref<PyEdge> edge) const
    {
        return runMultiInput({ std::move(edge) });
    }

    virtual std::vector<nb::ref<PyEdge>> runMultiInput(
            std::vector<nb::ref<PyEdge>> edges) const
    {
        auto params = parameters();
        if (params.has_value()) {
            return PyEdge::runMultiInputNode(
                    std::move(edges),
                    baseNode(),
                    std::move(params->name),
                    std::move(params->localParams));
        } else {
            return PyEdge::runMultiInputNode(
                    std::move(edges), baseNode(), poly::nullopt, poly::nullopt);
        }
    }

    virtual GraphID buildGraph(
            nb::ref<PyCompressor> compressor,
            const std::vector<GraphID>& successors) const
    {
        auto params = parameters();
        if (params.has_value()) {
            return compressor->buildStaticGraph(
                    baseNode(),
                    successors,
                    std::move(params->name),
                    std::move(params->localParams));
        } else {
            return compressor->buildStaticGraph(
                    baseNode(), successors, poly::nullopt, poly::nullopt);
        }
    }

    virtual NodeID parameterize(nb::ref<PyCompressor> compressor) const
    {
        auto params = parameters();
        if (params.has_value()) {
            return compressor->parameterizeNode(
                    baseNode(),
                    std::move(params->name),
                    std::move(params->localParams));
        } else {
            return baseNode();
        }
    }

    virtual NodeID baseNode() const = 0;

    virtual poly::optional<NodeParameters> parameters() const
    {
        return poly::nullopt;
    }

    virtual ~PyNodeBase() = default;
};
} // namespace py
} // namespace openzl

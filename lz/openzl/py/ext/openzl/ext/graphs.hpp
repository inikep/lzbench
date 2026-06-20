// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <nanobind/nanobind.h>

#include "openzl/ext.hpp"

namespace openzl {
namespace py {
void registerGraphsModule(nanobind::module_& m);

class PyGraphBase : public nanobind::intrusive_base {
   public:
    virtual GraphID parameterize(nb::ref<PyCompressor> compressor) const
    {
        auto params = parameters();
        if (params.has_value()) {
            return compressor->parameterizeGraph(
                    baseGraph(),
                    std::move(params->name),
                    std::move(params->customGraphs),
                    std::move(params->customNodes),
                    std::move(params->localParams));
        } else {
            return baseGraph();
        }
    }

    GraphID operator()(nb::ref<PyCompressor> compressor) const
    {
        return parameterize(std::move(compressor));
    }

    virtual void setDestination(nb::ref<PyEdge> edge) const
    {
        setMultiInputDestination({ std::move(edge) });
    }

    virtual void setMultiInputDestination(
            std::vector<nb::ref<PyEdge>> edges) const
    {
        auto params = parameters();
        if (params.has_value()) {
            PyEdge::setMultiInputDestination(
                    std::move(edges),
                    baseGraph(),
                    std::move(params->name),
                    std::move(params->customGraphs),
                    std::move(params->customNodes),
                    std::move(params->localParams));
        } else {
            PyEdge::setMultiInputDestination(
                    std::move(edges),
                    baseGraph(),
                    poly::nullopt,
                    poly::nullopt,
                    poly::nullopt,
                    poly::nullopt);
        }
    }

    virtual GraphID baseGraph() const = 0;

    virtual poly::optional<GraphParameters> parameters() const
    {
        return poly::nullopt;
    }

    virtual ~PyGraphBase() = default;
};
} // namespace py
} // namespace openzl

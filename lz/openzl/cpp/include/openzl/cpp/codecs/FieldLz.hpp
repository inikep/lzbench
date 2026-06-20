// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_field_lz.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/LocalParams.hpp"
#include "openzl/cpp/codecs/Graph.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace graphs {

class FieldLz : public Graph {
   public:
    static constexpr GraphID graph = ZL_GRAPH_FIELD_LZ;

    static constexpr GraphMetadata<1> metadata = {
        .inputs = { InputMetadata{ .typeMask = TypeMask::Struct
                                           | TypeMask::Numeric } },
        .description =
                "Compress the struct inputs using the FieldLZ codec with the default graphs",
    };

    struct Parameters {
        /// Optionally override the compression level
        poly::optional<int> compressionLevel;
        /// Optionally override the backend literals graph
        poly::optional<GraphID> literalsGraph;
        /// Optionally override the backend tokens graph
        poly::optional<GraphID> tokensGraph;
        /// Optionally override the backend offsets graph
        poly::optional<GraphID> offsetsGraph;
        /// Optionally override the backend extraLiteralLengths graph
        poly::optional<GraphID> extraLiteralLengthsGraph;
        /// Optionally override the backend extraMatchLengths graph
        poly::optional<GraphID> extraMatchLengthsGraph;
    };

    FieldLz() {}
    explicit FieldLz(Parameters p) : params_(std::move(p)) {}
    explicit FieldLz(int compressionLevel)
            : FieldLz(Parameters{ .compressionLevel = compressionLevel })
    {
    }

    GraphID baseGraph() const override
    {
        return graph;
    }

    poly::optional<GraphParameters> parameters() const override
    {
        if (!params_.has_value()) {
            return poly::nullopt;
        }

        LocalParams lp;

        if (params_->compressionLevel.has_value()) {
            lp.addIntParam(
                    ZL_FIELD_LZ_COMPRESSION_LEVEL_OVERRIDE_PID,
                    params_->compressionLevel.value());
        }

        std::vector<GraphID> graphs;
        auto addGraph = [&](int key, poly::optional<GraphID> g) {
            if (g.has_value()) {
                lp.addIntParam(key, int(graphs.size()));
                graphs.push_back(g.value());
            }
        };
        addGraph(
                ZL_FIELD_LZ_LITERALS_GRAPH_OVERRIDE_INDEX_PID,
                params_->literalsGraph);
        addGraph(
                ZL_FIELD_LZ_TOKENS_GRAPH_OVERRIDE_INDEX_PID,
                params_->tokensGraph);
        addGraph(
                ZL_FIELD_LZ_OFFSETS_GRAPH_OVERRIDE_INDEX_PID,
                params_->offsetsGraph);
        addGraph(
                ZL_FIELD_LZ_EXTRA_LITERAL_LENGTHS_GRAPH_OVERRIDE_INDEX_PID,
                params_->extraLiteralLengthsGraph);
        addGraph(
                ZL_FIELD_LZ_EXTRA_MATCH_LENGTHS_GRAPH_OVERRIDE_INDEX_PID,
                params_->extraMatchLengthsGraph);

        return GraphParameters{
            .customGraphs = std::move(graphs),
            .localParams  = std::move(lp),
        };
    }

    ~FieldLz() override = default;

   private:
    poly::optional<Parameters> params_{};
};
} // namespace graphs

namespace nodes {
class FieldLz : public Node {
   public:
    static constexpr NodeID node = ZL_NODE_FIELD_LZ;

    static constexpr NodeMetadata<1, 5> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Struct } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Struct,
                                              .name = "literals" },
                              OutputMetadata{ .type = Type::Struct,
                                              .name = "tokens (2-bytes)" },
                              OutputMetadata{ .type = Type::Numeric,
                                              .name = "offsets" },
                              OutputMetadata{ .type = Type::Numeric,
                                              .name = "extra literal lengths" },
                              OutputMetadata{ .type = Type::Numeric,
                                              .name = "extra match lengths" } },
        .description      = "Run an LZ compression that matches whole structs",
    };

    FieldLz() {}
    explicit FieldLz(int compressionLevel) : compressionLevel_(compressionLevel)
    {
    }
    explicit FieldLz(poly::optional<int> compressionLevel)
            : compressionLevel_(std::move(compressionLevel))
    {
    }

    NodeID baseNode() const override
    {
        return node;
    }

    poly::optional<NodeParameters> parameters() const override
    {
        poly::optional<LocalParams> lp{};
        if (compressionLevel_.has_value()) {
            lp.emplace();
            lp->addIntParam(
                    ZL_FIELD_LZ_COMPRESSION_LEVEL_OVERRIDE_PID,
                    *compressionLevel_);
        }
        return NodeParameters{ .name        = "field_lz_with_level",
                               .localParams = std::move(lp) };
    }

    /**
     * Helper method to build a graph, since this is the most common operation,
     * and the one that benefits most from brevity, since it is often nested.
     */
    GraphID operator()(
            Compressor& compressor,
            GraphID literals,
            GraphID tokens,
            GraphID offsets,
            GraphID extraLiteralLengths,
            GraphID extraMatchLengths) const
    {
        return buildGraph(
                compressor,
                std::initializer_list<GraphID>{ literals,
                                                tokens,
                                                offsets,
                                                extraLiteralLengths,
                                                extraMatchLengths });
    }

    ~FieldLz() override = default;

   private:
    poly::optional<int> compressionLevel_;
};
} // namespace nodes
} // namespace openzl

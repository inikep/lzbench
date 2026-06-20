// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>

#include <folly/dynamic.h>

#include "tools/zstrong_cpp.h"

namespace zstrong {
namespace {
constexpr std::string_view kNameKey                = "name";
constexpr std::string_view kSuccessorsKey          = "successors";
constexpr std::string_view kIntParamsKey           = "int_params";
constexpr std::string_view kGenericBinaryParamsKey = "binary_params";
constexpr std::string_view kGenericStringParamsKey = "string_params";
constexpr std::string_view kGlobalParamsKey        = "global_params";
} // namespace

/// Creates a Zstrong graph based on data stored in a JSON-like
/// folly::dynamic. Allows the graph to reference named custom
/// transforms/graphs/selectors that can be passed to the constructor,
/// in addition to all standard transforms/graphs/selectors.
class JsonGraph : public Graph {
   public:
    explicit JsonGraph(
            folly::dynamic graph,
            ZL_Type inputType                            = ZL_Type_serial,
            std::optional<TransformMap> customTransforms = std::nullopt,
            std::optional<GraphMap> customGraphs         = std::nullopt,
            std::optional<SelectorMap> customSelectors   = std::nullopt);

    ZL_GraphID registerGraph(ZL_Compressor& cgraph) const override;

    void registerGraph(ZL_DCtx& dctx) const override;

    ZL_Type inputType() const override
    {
        return inputType_;
    }

   private:
    ZL_GraphID registerGraph(
            ZL_Compressor& cgraph,
            std::unordered_map<std::string_view, ZL_GraphID> const&
                    customGraphs,
            folly::dynamic const& graph) const;

    folly::dynamic graph_;
    ZL_Type inputType_;
    TransformMap customTransforms_;
    GraphMap customGraphs_;
    SelectorMap customSelectors_;
};

struct ExtractedStream {
    ZL_Type type;
    size_t nbElts;
    size_t eltWidth;
    std::string_view data;
};
/// Extracts individual streams from the format written by the "extract"
/// "selector".
std::vector<ExtractedStream> splitExtractedStreams(std::string_view data);

} // namespace zstrong

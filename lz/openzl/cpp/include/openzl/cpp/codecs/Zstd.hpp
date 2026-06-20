// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <unordered_map>

#include "openzl/codecs/zl_zstd.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Graph.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"

namespace openzl {
namespace graphs {
struct Zstd : public Graph {
   public:
    static constexpr GraphID graph = ZL_GRAPH_ZSTD;

    static constexpr GraphMetadata<1> metadata = {
        .inputs      = { InputMetadata{ .typeMask = TypeMask::Serial } },
        .description = "Zstd compress the input data",
    };

    Zstd() {}
    explicit Zstd(int compressionLevel);
    explicit Zstd(std::unordered_map<int, int> zstdParams)
            : zstdParams_(std::move(zstdParams))
    {
    }

    GraphID baseGraph() const override
    {
        return graph;
    }

    poly::optional<GraphParameters> parameters() const override
    {
        if (!zstdParams_.has_value()) {
            return poly::nullopt;
        }
        LocalParams lp;
        for (const auto& [key, value] : *zstdParams_) {
            lp.addIntParam(key, value);
        }
        return GraphParameters{ .localParams = std::move(lp) };
    }

    ~Zstd() override = default;

   private:
    poly::optional<std::unordered_map<int, int>> zstdParams_;
};
} // namespace graphs
} // namespace openzl

// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include "custom_transforms/tulip_v2/decode_tulip_v2.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_public_nodes.h"

namespace zstrong::tulip_v2 {

// Specify the successors, or leave unset to use the default graph.
struct TulipV2Successors {
    std::optional<ZL_GraphID> floatFeatures;
    std::optional<ZL_GraphID> idListFeatures;
    std::optional<ZL_GraphID> idListListFeatures;
    std::optional<ZL_GraphID> floatListFeatures;
    std::optional<ZL_GraphID> idScoreListFeatures;
    std::optional<ZL_GraphID> everythingElse;
};

std::vector<std::pair<Tag, size_t>> parseTulipV2(std::string_view input);

ZL_GraphID createTulipV2SuccessorSelector(
        ZL_Compressor* cgraph,
        TulipV2Successors const& successors,
        unsigned idRangeBegin,
        unsigned idRangeEnd);

/// Creates the TulipV2 graph & registers custom transforms needed.
/// Registers custom transforms beginning at @p idRangeBegin and using ids up to
/// @p idRangeEnd. The same @p idRangeBegin must be used for both compressors &
/// decompressors. Consumes IDs in order.
///
/// @returns The next free ID. We consumed [idRangeBegin, returnValue).
ZL_GraphID createTulipV2Graph(
        ZL_Compressor* cgraph,
        TulipV2Successors const& successors,
        unsigned idrangeBegin,
        unsigned idRangeEnd);

/// @returns the raw tulip_v2 node, for testing purposes
ZL_NodeID createTulipV2Node(ZL_Compressor* cgraph);

ZL_GraphID featureIDsGraph(ZL_Compressor* cgraph);

} // namespace zstrong::tulip_v2

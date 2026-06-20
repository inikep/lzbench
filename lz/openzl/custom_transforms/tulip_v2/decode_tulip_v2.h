// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/zl_dtransform.h"

namespace zstrong::tulip_v2 {
/// NOTE: These must remain stable!
enum class Tag {
    FloatFeatures       = 0,
    IdListFeatures      = 1,
    IdListListFeatures  = 2,
    FloatListFeatures   = 3,
    IdScoreListFeatures = 4,
    EverythingElse      = 5,
    NumTags,
};

/// We reserve up to this many IDs for custom transforms.
/// NOTE: Do not rely on this to be stable!
unsigned constexpr kNumCustomTransforms = 5;

/// Registers custom transforms beginning at @p idRangeBegin and using ids up to
/// @p idRangeEnd. The same @p idRangeBegin must be used for both compressors &
/// decompressors. Consumes IDs in order.
///
/// @returns The next free ID. We consumed [idRangeBegin, returnValue).
ZL_Report registerCustomTransforms(
        ZL_DCtx* dctx,
        unsigned idRangeBegin,
        unsigned idRangeEnd);
} // namespace zstrong::tulip_v2

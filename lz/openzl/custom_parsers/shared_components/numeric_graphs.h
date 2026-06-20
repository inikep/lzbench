// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/zl_compressor.h"

namespace openzl::custom_parsers {

// Note: can separate out into multiple files if needed. Currently the
// successors are unbenchmarked.
ZL_GraphID ZL_Compressor_registerRangePack(ZL_Compressor* compressor);
ZL_GraphID ZL_Compressor_registerTokenizeSorted(ZL_Compressor* compressor);
ZL_GraphID ZL_Compressor_registerRangePackZstd(ZL_Compressor* compressor);
ZL_GraphID ZL_Compressor_registerDeltaFieldLZ(ZL_Compressor* compressor);

} // namespace openzl::custom_parsers

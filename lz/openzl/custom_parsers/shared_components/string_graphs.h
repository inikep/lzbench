// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <string>
#include "openzl/zl_opaque_types.h"

#include "openzl/zl_compressor.h"

namespace openzl::custom_parsers {

ZL_GraphID ZL_Compressor_registerStringTokenize(ZL_Compressor* compressor);
ZL_GraphID registerNullAwareDispatch(
        ZL_Compressor* compressor,
        const std::string& name,
        const std::array<ZL_GraphID, 3>& successors);

} // namespace openzl::custom_parsers

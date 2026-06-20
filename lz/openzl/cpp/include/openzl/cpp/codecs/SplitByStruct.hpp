// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_split_by_struct.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/FunctionGraph.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"

namespace openzl {
namespace nodes {
struct SplitByStruct {
    // TODO(terrelln): Switch SplitByStruct to a function graph. It currently
    // does some hacky things in the implementation that are no longer necessary
    // now that function graphs exist.
};
constexpr SplitByStruct split_by_struct;
} // namespace nodes

namespace graphs {
struct SplitByStruct {
    // TODO(terrelln): Switch SplitByStruct to a function graph. It currently
    // does some hacky things in the implementation that are no longer necessary
    // now that function graphs exist.
};
inline constexpr SplitByStruct split_by_struct;
} // namespace graphs
} // namespace openzl

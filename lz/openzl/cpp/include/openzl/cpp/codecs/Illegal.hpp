// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_illegal.h"
#include "openzl/cpp/codecs/Graph.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
class Illegal : public SimplePipeNode<Illegal> {
   public:
    static inline constexpr NodeID node = ZL_NODE_ILLEGAL;
};
} // namespace nodes

namespace graphs {
class Illegal : public SimpleGraph<Illegal> {
   public:
    static inline constexpr GraphID graph = ZL_GRAPH_ILLEGAL;
};
} // namespace graphs
} // namespace openzl

// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/cpp/detail/Portability.hpp"

#if ZL_CPP_HAS_OPTIONAL
#    include <optional>
#endif

namespace openzl {
namespace poly {
#if ZL_CPP_HAS_OPTIONAL
using std::nullopt;
using std::nullopt_t;
using std::optional;
#else
#    error "TODO"
#endif
} // namespace poly
} // namespace openzl

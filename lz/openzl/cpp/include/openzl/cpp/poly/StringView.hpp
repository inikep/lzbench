// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/cpp/detail/Portability.hpp"

#if ZL_CPP_HAS_STRING_VIEW
#    include <string_view>
#endif

namespace openzl {
namespace poly {
#if ZL_CPP_HAS_STRING_VIEW
using std::string_view;
#else
#    error "TODO"
#endif
} // namespace poly
} // namespace openzl

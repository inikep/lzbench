// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <type_traits>

#include "openzl/cpp/detail/Portability.hpp"

namespace openzl {
namespace poly {
namespace impl {
template <typename...>
using void_t = void;
}

#if ZL_CPP_HAS_VOID_T
using std::void_t;
#else
using impl::void_t;
#endif
} // namespace poly
} // namespace openzl

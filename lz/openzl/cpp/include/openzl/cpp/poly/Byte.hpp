// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>

#include "openzl/cpp/detail/Portability.hpp"

namespace openzl {
namespace poly {
namespace impl {
enum class byte : unsigned char {};
}

#if ZL_CPP_HAS_BYTE
using std::byte;
#else
using impl::byte;
#endif

} // namespace poly
} // namespace openzl

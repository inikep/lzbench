// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>

#include "openzl/cpp/detail/Portability.hpp"

#if ZL_CPP_HAS_SOURCE_LOCATION
#    include <source_location>
#endif

namespace openzl {
namespace poly {
namespace impl {
struct source_location {
   public:
    static source_location current() noexcept
    {
        return source_location{};
    }
    std::uint32_t line() const noexcept
    {
        return 0;
    }
    std::uint32_t column() const noexcept
    {
        return 0;
    }
    const char* file_name() const noexcept
    {
        return "";
    }
    const char* function_name() const noexcept
    {
        return "";
    }
};
} // namespace impl
#if ZL_CPP_HAS_SOURCE_LOCATION
using std::source_location;
#else
using impl::source_location;
#endif
} // namespace poly
} // namespace openzl

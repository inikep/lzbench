// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/cpp/Config.hpp"

#ifdef __has_include
#    define ZL_CPP_HAS_INCLUDE(x) __has_include(x)
#else
#    define ZL_CPP_HAS_INCLUDE(x) 0
#endif

#if __cplusplus >= 202002L || ZL_CPP_HAS_INCLUDE(<version>)
#    include <version>
#else
// see https://en.cppreference.com/w/cpp/header/version.html
#    include <ciso646>
#endif

namespace openzl {
namespace detail {

#if __cplusplus >= 202002L
#    define ZL_CPP_HAS_CPP20 1
#else
#    define ZL_CPP_HAS_CPP20 0
#endif

#if __cplusplus >= 201703L
#    define ZL_CPP_HAS_CPP17 1
#else
#    define ZL_CPP_HAS_CPP17 0
#endif

#if __cplusplus >= 201402L
#    define ZL_CPP_HAS_CPP14 1
#else
#    define ZL_CPP_HAS_CPP14 0
#endif

#if ZL_CPP_HAS_CPP17                        \
        || (defined(__cpp_inline_variables) \
            && __cpp_inline_variables >= 201606L)
#    define ZL_CPP_INLINE_VARIABLE inline
#else
#    define ZL_CPP_INLINE_VARIABLE
#endif

#if !ZL_CPP_CONFIG_FORCE_POLYFILL && defined(__cpp_lib_source_location) \
        && __cpp_lib_source_location >= 201907L
#    define ZL_CPP_HAS_SOURCE_LOCATION 1
constexpr bool kHasSourceLocation = true;
#else
#    define ZL_CPP_HAS_SOURCE_LOCATION 0
constexpr bool kHasSourceLocation = false;
#endif

#if !ZL_CPP_CONFIG_FORCE_POLYFILL && defined(__cpp_lib_span) \
        && __cpp_lib_span >= 202002L
#    define ZL_CPP_HAS_SPAN 1
constexpr bool kHasSpan = true;
#else
#    define ZL_CPP_HAS_SPAN 0
constexpr bool kHasSpan = false;
#endif

#if !ZL_CPP_CONFIG_FORCE_POLYFILL              \
        && (ZL_CPP_HAS_CPP17                   \
            || (defined(__cpp_lib_string_view) \
                && __cpp_lib_string_view >= 201606L))
#    define ZL_CPP_HAS_STRING_VIEW 1
constexpr bool kHasStringView = true;
#else
#    define ZL_CPP_HAS_STRING_VIEW 0
constexpr bool kHasStringView = false;
#endif

#if !ZL_CPP_CONFIG_FORCE_POLYFILL \
        && (ZL_CPP_HAS_CPP17      \
            || (defined(__cpp_lib_optional) && __cpp_lib_optional >= 201606L))
#    define ZL_CPP_HAS_OPTIONAL 1
constexpr bool kHasOptional = true;
#else
#    define ZL_CPP_HAS_OPTIONAL 0
constexpr bool kHasOptional = false;
#endif

#if !ZL_CPP_CONFIG_FORCE_POLYFILL \
        && (ZL_CPP_HAS_CPP17      \
            || (defined(__cpp_lib_byte) && __cpp_lib_byte >= 201603L))
#    define ZL_CPP_HAS_BYTE 1
#else
#    define ZL_CPP_HAS_BYTE 0
#endif

#if !ZL_CPP_CONFIG_FORCE_POLYFILL                             \
        && (ZL_CPP_HAS_CPP17                                  \
            || (defined(__cpp_lib_nonmember_container_access) \
                && __cpp_lib_nonmember_container_access >= 201411L))
#    define ZL_CPP_HAS_NONMEMBER_CONTAINER_ACCESS 1
#else
#    define ZL_CPP_HAS_NONMEMBER_CONTAINER_ACCESS 0
#endif

#if !ZL_CPP_CONFIG_FORCE_POLYFILL \
        && (ZL_CPP_HAS_CPP17      \
            || (defined(__cpp_lib_void_t) && __cpp_lib_void_t >= 201411L))
#    define ZL_CPP_HAS_VOID_T 1
#else
#    define ZL_CPP_HAS_VOID_T 0
#endif

} // namespace detail
} // namespace openzl

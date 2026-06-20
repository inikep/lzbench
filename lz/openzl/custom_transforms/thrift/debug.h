// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/Range.h>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace zstrong::thrift {

// Should never be set "true" in fbcode master
constexpr bool kEnableDebugMode = false;

inline std::string bytesAsHex(const uint8_t* src, std::size_t srcSize)
{
    if (srcSize == 0) {
        return "[empty]";
    }

    std::stringstream ss;
    for (size_t i = 0; i < srcSize; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0')
           << static_cast<int>(src[i]);
        if (i < srcSize - 1) {
            ss << " ";
        }
    }
    return ss.str();
}

inline std::string bytesAsHex(folly::ByteRange range)
{
    return bytesAsHex(range.data(), range.size());
}

inline std::string bytesAsHex(const std::vector<uint8_t>& bytes)
{
    return bytesAsHex(bytes.data(), bytes.size());
}

#define debug(...)                  \
    do {                            \
        if (kEnableDebugMode) {     \
            debugImpl(__VA_ARGS__); \
        }                           \
    } while (0)

template <typename... Args>
inline void debugImpl(std::string_view fmt, Args... args)
{
    if (kEnableDebugMode) {
        // See https://stackoverflow.com/a/68675384/3154996
        std::cerr << fmt::vformat(fmt, fmt::make_format_args(args...))
                  << std::endl;
    }
}

} // namespace zstrong::thrift

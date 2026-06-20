// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/cpp/codecs/Zstd.hpp"

#include <zstd.h>

namespace openzl {
namespace graphs {

Zstd::Zstd(int compressionLevel)
        : Zstd({ { ZSTD_c_compressionLevel, compressionLevel } })
{
}

} // namespace graphs
} // namespace openzl

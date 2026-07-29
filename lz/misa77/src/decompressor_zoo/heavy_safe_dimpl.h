// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "format.h"
#include "misa77/misa77.h"
#include "util.h"

#include <cstdint>
#include <cstring>

namespace misa77
{
    // Returns number of bytes written to `dst`, and 0 on failure.
    // `isa_lib` is ISA-dependent.
    template <class isa_lib>
    uint64_t heavy_safe_dimpl(const uint8_t* __restrict src,
                              uint64_t src_size,
                              uint8_t* __restrict dst,
                              uint64_t dst_cap)
    {
        using namespace heavy;
        // support will be added in the future
        // for now, you must use the unsafe decompressor for the heavy mode
        (void)src, (void)src_size, (void)dst, (void)dst_cap;
        return 0;
    }
} // namespace misa77

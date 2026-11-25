// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "misa77/misa77.h"

#include <cstdint>

namespace misa77
{
    // ISA-specialized compress entry points. Each is defined in a per-target TU in src/isa/,
    // compiled with that ISA's flags (target_avx2.cpp with -mavx2, target_sse2.cpp /
    // target_portable.cpp at baseline, target_neon.cpp at armv8-a), and selected at runtime
    // by compress() (see compress.cpp).
    uint64_t compress_avx2(const uint8_t* __restrict src,
                           uint64_t src_size,
                           uint8_t* __restrict dst,
                           uint64_t dst_cap,
                           config cfg);
    uint64_t compress_sse2(const uint8_t* __restrict src,
                           uint64_t src_size,
                           uint8_t* __restrict dst,
                           uint64_t dst_cap,
                           config cfg);
    uint64_t compress_neon(const uint8_t* __restrict src,
                           uint64_t src_size,
                           uint8_t* __restrict dst,
                           uint64_t dst_cap,
                           config cfg);
    uint64_t compress_portable(const uint8_t* __restrict src,
                               uint64_t src_size,
                               uint8_t* __restrict dst,
                               uint64_t dst_cap,
                               config cfg);
} // namespace misa77

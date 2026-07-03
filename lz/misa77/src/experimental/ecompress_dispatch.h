// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "misa77/experimental.h"

#include <cstdint>

namespace misa77
{
    // ISA-specialized compress entry points. Each is defined in a per-target TU in
    // src/experimental/isa/, compiled with that ISA's flags, and selected at runtime by compress()
    // (see compress.cpp).
    namespace experimental
    {
        uint64_t compress_tuned_avx2(const uint8_t* __restrict src,
                                     uint64_t src_size,
                                     uint8_t* __restrict dst,
                                     uint64_t dst_cap,
                                     const param& given);
        uint64_t compress_tuned_sse2(const uint8_t* __restrict src,
                                     uint64_t src_size,
                                     uint8_t* __restrict dst,
                                     uint64_t dst_cap,
                                     const param& given);
        uint64_t compress_tuned_portable(const uint8_t* __restrict src,
                                         uint64_t src_size,
                                         uint8_t* __restrict dst,
                                         uint64_t dst_cap,
                                         const param& given);
    } // namespace experimental
} // namespace misa77

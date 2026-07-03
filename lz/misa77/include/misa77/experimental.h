// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include <cstdint>

namespace misa77
{
    namespace experimental
    {
        // Be careful with possible overflow
        class param
        {
        public:
            bool use_default = false;
            uint64_t size = 0;
            uint64_t block = 0;
            uint64_t short4_7 = 0;
            uint64_t short8_15 = 0;
            uint64_t lit7 = 0;
            uint64_t lit17 = 0;
            uint64_t lit33 = 0;
        };

        // Returns number of bytes written to `dst`, and 0 on failure.
        // PRECONDITION: `src_size <= max_src_size`
        uint64_t compress_tuned(const uint8_t* __restrict src,
                                uint64_t src_size,
                                uint8_t* __restrict dst,
                                uint64_t dst_cap,
                                const param& given);

        // Chooses a good compression parameter vector for `src` by sampling a small portion of it.
        // - `option = 0` gives significant decompress throughput gains with moderate ratio costs.
        // - `option = 1` gives moderate decompress throughput gains with tiny ratio costs.
        param suggest_homogeneous(const uint8_t* __restrict src,
                                  uint64_t src_size,
                                  uint8_t* __restrict dst,
                                  uint64_t dst_cap,
                                  uint8_t option);

        // Returns number of bytes written to `dst`, and 0 on failure.
        // PERFORMANCE ON HETEROGENEOUS DATA CAN VARY WILDLY, KEEP THAT IN MIND
        // PRECONDITION: `src_size <= max_src_size`
        // - `option = 0` gives significant decompress throughput gains with moderate ratio costs
        // - `option = 1` gives moderate decompress throughput gains with tiny ratio costs
        uint64_t adaptive_compress(const uint8_t* __restrict src,
                                   uint64_t src_size,
                                   uint8_t* __restrict dst,
                                   uint64_t dst_cap,
                                   uint8_t option);

        // Returns number of bytes written to `dst`, and 0 on failure.
        // PRECONDITION: `src_size <= max_src_size`
        // YOLO
        // (will improve decomp throughput a lot on most data distributions, and cost some ratio)
        uint64_t yolo_compress(const uint8_t* __restrict src,
                               uint64_t src_size,
                               uint8_t* __restrict dst,
                               uint64_t dst_cap);
    } // namespace experimental
} // namespace misa77
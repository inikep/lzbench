// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#include "compress_dispatch.h"
#include "misa77/misa77.h"

#include <cstdint>

namespace misa77
{
    // Upper-bound on compressed size (in bytes) for any input of size `src_size` bytes.
    // Use to size destination buffer.
    // PRECONDITION: `src_size <= max_src_size`
    uint64_t compress_bound(uint64_t src_size, config cfg)
    {
        if (cfg.level < cfg.heavy_lb)
            return uint64_t(8)   // uncompressed size
                   + uint64_t(8) // number of trailing raw literals (>= `literal_suffix`)
                   + src_size + (src_size / uint64_t(255)) +
                   uint64_t(16); // compressed data length upper-bound
        else
            return uint64_t(8) + uint64_t(8) + src_size + (src_size / uint64_t(32)) + uint64_t(64);
    }

    // Returns number of bytes written to `dst`, and 0 on failure.
    // Resolves the best finder path once per call, depending on the ISA.
    // PRECONDITION: `src_size <= max_src_size`
    uint64_t compress(const uint8_t* __restrict src,
                      uint64_t src_size,
                      uint8_t* __restrict dst,
                      uint64_t dst_cap,
                      config cfg)
    {
        if (src_size > max_src_size)
            return 0;

#if defined(__x86_64__)
        if (__builtin_cpu_supports("avx2"))
            return compress_avx2(src, src_size, dst, dst_cap, cfg);
        return compress_sse2(src, src_size, dst, dst_cap, cfg);
#elif defined(__aarch64__)
        // NEON is guaranteed on Aarch64
        return compress_neon(src, src_size, dst, dst_cap, cfg);
#else
        return compress_portable(src, src_size, dst, dst_cap, cfg);
#endif
    }
} // namespace misa77

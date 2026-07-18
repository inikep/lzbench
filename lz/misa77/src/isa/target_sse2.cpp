// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

// Baseline for x86-64 (SSE2 support is guaranteed in the ISA).

#include "compress_dispatch.h"
#include "compressor_zoo/default_compress_impl.h"
#include "compressor_zoo/loose_compress_impl.h"
#include "decompress_dispatch.h"
#include "decompressor_zoo/safe_decompress_impl.h"
#include "decompressor_zoo/unsafe_decompress_impl.h"
#include "isa/lib_sse2.h"

#include <cstdint>

namespace misa77
{
    uint64_t compress_sse2(const uint8_t* __restrict src,
                           uint64_t src_size,
                           uint8_t* __restrict dst,
                           uint64_t dst_cap,
                           config cfg)
    {
        if (cfg.level == 0)
            return loose_compress_impl<lib_sse2>(src, src_size, dst, dst_cap);
        else if (cfg.level == 1)
            return default_compress_impl<lib_sse2>(src, src_size, dst, dst_cap);
        return 0;
    }

    uint64_t decompress_sse2(const uint8_t* __restrict src,
                             uint64_t src_size,
                             uint8_t* __restrict dst,
                             uint64_t dst_cap,
                             dconfig dcfg)
    {
        if (dcfg.safe)
            return safe_decompress_impl<lib_sse2>(src, src_size, dst, dst_cap);
        return unsafe_decompress_impl<lib_sse2>(src, src_size, dst, dst_cap);
    }

} // namespace misa77

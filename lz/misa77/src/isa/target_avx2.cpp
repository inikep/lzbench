// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

// AVX2 TU.
// Built only on x86 targets and the implementations herein are chosen at dispatch iff the cpu has
// AVX2 support.

#include "compress_dispatch.h"
#include "compressor_zoo/default_compress_impl.h"
#include "compressor_zoo/heavy_compress_impl.h"
#include "compressor_zoo/loose_compress_impl.h"
#include "decompress_dispatch.h"
#include "decompressor_zoo/heavy_safe_decompress_impl.h"
#include "decompressor_zoo/heavy_unsafe_decompress_impl.h"
#include "decompressor_zoo/safe_decompress_impl.h"
#include "decompressor_zoo/unsafe_decompress_impl.h"
#include "isa/lib_avx2.h"

#include <cstdint>

namespace misa77
{
    uint64_t compress_avx2(const uint8_t* __restrict src,
                           uint64_t src_size,
                           uint8_t* __restrict dst,
                           uint64_t dst_cap,
                           config cfg)
    {
        if (cfg.level == 0)
            return loose_compress_impl<lib_avx2>(src, src_size, dst, dst_cap);
        else if (cfg.level == 1)
            return default_compress_impl<lib_avx2>(src, src_size, dst, dst_cap);
        else if (cfg.level == 2)
            return heavy_compress_impl<lib_avx2>(src, src_size, dst, dst_cap);
        return 0;
    }

    uint64_t decompress_avx2(const uint8_t* __restrict src,
                             uint64_t src_size,
                             uint8_t* __restrict dst,
                             uint64_t dst_cap,
                             dconfig dcfg)
    {
        if ((uint64_t(1) << format_bit) & src[7])
        {
            // heavy
            if (dcfg.safe)
                return heavy_safe_decompress_impl<lib_avx2>(src, src_size, dst, dst_cap);
            return heavy_unsafe_decompress_impl<lib_avx2>(src, src_size, dst, dst_cap);
        }

        // light
        if (dcfg.safe)
            return safe_decompress_impl<lib_avx2>(src, src_size, dst, dst_cap);
        return unsafe_decompress_impl<lib_avx2>(src, src_size, dst, dst_cap);
    }
} // namespace misa77

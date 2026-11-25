// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

// AVX2 TU.
// Built only on x86 targets and the implementations herein are chosen at dispatch iff the cpu has
// AVX2 support.

#include "compress_dispatch.h"
#include "compressor_zoo/entropy_optimal_cimpl.h"
#include "compressor_zoo/heavy_optimal_cimpl.h"
#include "compressor_zoo/light_blitz_cimpl.h"
#include "compressor_zoo/light_keen_cimpl.h"
#include "compressor_zoo/light_loose_cimpl.h"
#include "compressor_zoo/light_optimal_cimpl.h"
#include "compressor_zoo/light_swift_cimpl.h"
#include "decompress_dispatch.h"
#include "decompressor_zoo/entropy_safe_dimpl.h"
#include "decompressor_zoo/heavy_safe_dimpl.h"
#include "decompressor_zoo/light_safe_dimpl.h"
#include "format.h"
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
        if (cfg.level == -1)
            return light_blitz_cimpl<lib_avx2>(src, src_size, dst, dst_cap);
        else if (cfg.level == 0)
            return light_swift_cimpl<lib_avx2>(src, src_size, dst, dst_cap);
        else if (cfg.level == 1)
            return light_loose_cimpl<lib_avx2>(src, src_size, dst, dst_cap);
        else if (cfg.level == 2)
            return light_keen_cimpl<lib_avx2>(src, src_size, dst, dst_cap);
        else if (cfg.level == 3)
            return light_optimal_cimpl<lib_avx2>(src, src_size, dst, dst_cap);
        else if (cfg.level == 4)
            return heavy_optimal_cimpl<lib_avx2>(src, src_size, dst, dst_cap);
        else if (cfg.level == 5)
            return entropy_optimal_cimpl<lib_avx2>(src, src_size, dst, dst_cap);
        return 0;
    }

    uint64_t decompress_avx2(const uint8_t* __restrict src,
                             uint64_t src_size,
                             uint8_t* __restrict dst,
                             uint64_t dst_cap)
    {
        const uint8_t format = src[7] & format_mask;
        if (format == format_light)
            return light_safe_dimpl<lib_avx2>(src, src_size, dst, dst_cap);
        if (format == format_heavy)
            return heavy_safe_dimpl<lib_avx2>(src, src_size, dst, dst_cap);
        if (format == format_entropy)
            return entropy_safe_dimpl<lib_avx2>(src, src_size, dst, dst_cap);
        return 0;
    }
} // namespace misa77

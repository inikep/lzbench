// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

// AVX2 helper policy (256-bit).
//
// CONTRACT: the intrinsics below are AVX2, so this header MUST only be included
// from a TU compiled with -mavx2.

#include "decompressor_zoo/entropy_safe_kernels/kernel.h"
#include "decompressor_zoo/heavy_safe_kernels/kernel.h"
#include "decompressor_zoo/light_safe_kernels/kernel.h"

#include <cstdint>
#include <immintrin.h>

namespace misa77
{
    class lib_avx2
    {
    public:
        // Copies `src[0, vector_width)` to `src[dis, dis + vector_width)`.
        // It's guaranteed that `dis >= vector_width`, so there's no aliasing.
        [[gnu::always_inline]]
        static inline void cyccpy(uint8_t* src, uint64_t dis)
        {
            __m256i reg = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(src + dis), reg);
        }

        // Copies `src[0, vector_width)` to `src[dis, dis + vector_width)`.
        // We can have `dis = 0` here!!!
        // We need this because `dis = 0` is sometimes convenient but the memcpy in the portable
        // policy becomes UB
        [[gnu::always_inline]]
        static inline void safe_cyccpy(uint8_t* src, uint64_t dis)
        {
            return cyccpy(src, dis);
        }

        // Copies `src[0, vector_width)` to `dst[0, vector_width)`.
        // It's guaranteed that the two ranges don't alias.
        [[gnu::always_inline]]
        static inline void copy32(uint8_t* __restrict dst, const uint8_t* __restrict src)
        {
            __m256i reg = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), reg);
        }

        using vec = __m256i;

        // Return internal representation of `src[0, vector_width)`.
        [[gnu::always_inline]]
        static inline vec loadvec(const uint8_t* src)
        {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
        }

        // First differing byte index (0-indexed) between `reg1` and `reg2`.
        // If there's no such index, return `vector_width`.
        [[gnu::always_inline]]
        static inline uint64_t lcp(const vec& reg1, const vec& reg2)
        {
            const __m256i eq = _mm256_cmpeq_epi8(reg1, reg2);
            const uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(eq));
            const uint32_t diff = ~mask;

// __builtin_ctz(0) is UB, but tzcnt is defined for 0 and conveniently returns 32
#if defined(__BMI__)
            return static_cast<uint64_t>(_tzcnt_u32(diff));
#else
            return diff ? static_cast<uint64_t>(__builtin_ctz(diff)) : 32;
#endif
        }

        // AVX2 kernels for light and heavy
#if defined(MISA77_ASM_KERNELS)
        static constexpr bool has_light_kernel = true;
        static constexpr bool has_heavy_kernel = true;

        static uint8_t* light_kernel(const uint8_t* src,
                                     uint8_t* dst,
                                     const uint8_t* literals,
                                     uint8_t* out_limit) noexcept
        {
            return misa77_light_safe_kernel_avx2(src, dst, literals, out_limit);
        }

        static uint64_t heavy_kernel(heavy::kernel_io* io) noexcept
        {
            return misa77_heavy_safe_kernel_avx2(io);
        }
#else
        static constexpr bool has_light_kernel = false;
        static constexpr bool has_heavy_kernel = false;
#endif

        // AVX2 kernels for the entropy format
#if defined(MISA77_ASM_KERNELS)
        static constexpr bool has_entropy_kernel = true;
        static constexpr bool has_entropy_merge_kernel = true;

        static uint64_t entropy_kernel(entropy::kernel_io* io, uint64_t variant) noexcept
        {
            return misa77_entropy_safe_kernel_avx2(io, variant);
        }

        static void entropy_huffman_kernel(huffman_decoder::kernel_io* io) noexcept
        {
            misa77_entropy_huffman_kernel_avx2(io);
        }

        static void entropy_delta_kernel(uint8_t* b, uint8_t* e, uint64_t stride) noexcept
        {
            misa77_entropy_delta_kernel_avx2(b, e, stride);
        }

        static int64_t entropy_merge_kernel(entropy::merge_io* io) noexcept
        {
            return misa77_entropy_merge_kernel_avx2(io);
        }
#else
        static constexpr bool has_entropy_kernel = false;
        static constexpr bool has_entropy_merge_kernel = false;
#endif
    };
} // namespace misa77

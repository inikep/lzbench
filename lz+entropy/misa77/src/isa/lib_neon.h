// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

// ARM helper policy.
//
// CONTRACT: NEON (AdvSIMD) is guaranteed by the AArch64 ISA, so this header only
// requires the including TU to be an AArch64 baseline build (-march=armv8-a).

#include "decompressor_zoo/entropy_safe_kernels/kernel.h"
#include "decompressor_zoo/heavy_safe_kernels/kernel.h"
#include "decompressor_zoo/light_safe_kernels/kernel.h"
#include "format.h" // vector_width

#include <arm_neon.h>
#include <cstdint>

namespace misa77
{
    class lib_neon
    {
    public:
        // Copies `src[0, vector_width)` to `src[dis, dis + vector_width)`.
        // It's guaranteed that `dis >= vector_width`, so there's no aliasing.
        [[gnu::always_inline]]
        static inline void cyccpy(uint8_t* src, uint64_t dis)
        {
            uint8x16_t reg1 = vld1q_u8(src);
            uint8x16_t reg2 = vld1q_u8(src + vector_width / 2);
            vst1q_u8(src + dis, reg1);
            vst1q_u8(src + dis + vector_width / 2, reg2);
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
            uint8x16_t reg1 = vld1q_u8(src);
            uint8x16_t reg2 = vld1q_u8(src + vector_width / 2);
            vst1q_u8(dst, reg1);
            vst1q_u8(dst + vector_width / 2, reg2);
        }

        struct vec
        {
            uint8x16_t lo, hi;
        };

        // Return internal representation of `src[0, vector_width)`.
        [[gnu::always_inline]]
        static inline vec loadvec(const uint8_t* src)
        {
            return vec{vld1q_u8(src), vld1q_u8(src + vector_width / 2)};
        }

        // First differing byte index (0-indexed) between `reg1` and `reg2`.
        // If there's no such index, return `vector_width`.
        [[gnu::always_inline]]
        static inline uint64_t lcp(const vec& reg1, const vec& reg2)
        {
            // thanks to danlark1 on HN: https://news.ycombinator.com/item?id=48925428
            const uint8x16_t eq0 = vceqq_u8(reg1.lo, reg2.lo);
            const uint8x16_t eq1 = vceqq_u8(reg1.hi, reg2.hi);
            const uint64_t mask0 =
                vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(eq0), 4)), 0);
            const uint64_t mask1 =
                vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(eq1), 4)), 0);
            // __builtin_ctzll(0) is UB, so the all-equal halves must be tested explicitly
            const uint64_t diff0 = ~mask0;
            if (diff0)
                return static_cast<uint64_t>(__builtin_ctzll(diff0)) >> 2;
            const uint64_t diff1 = ~mask1;
            if (diff1)
                return (vector_width / 2) + (static_cast<uint64_t>(__builtin_ctzll(diff1)) >> 2);
            return vector_width;
        }

        // NEON kernels for light and heavy
#if defined(MISA77_ASM_KERNELS)
        static constexpr bool has_light_kernel = true;
        static constexpr bool has_heavy_kernel = true;

        static uint8_t* light_kernel(const uint8_t* src,
                                     uint8_t* dst,
                                     const uint8_t* literals,
                                     uint8_t* out_limit) noexcept
        {
            return misa77_light_safe_kernel_neon(src, dst, literals, out_limit);
        }

        static uint64_t heavy_kernel(heavy::kernel_io* io) noexcept
        {
            return misa77_heavy_safe_kernel_neon(io);
        }
#else
        static constexpr bool has_light_kernel = false;
        static constexpr bool has_heavy_kernel = false;
#endif

        // NEON kernels for the entropy format
#if defined(MISA77_ASM_KERNELS)
        static constexpr bool has_entropy_kernel = true;
        static constexpr bool has_entropy_merge_kernel = true;

        static uint64_t entropy_kernel(entropy::kernel_io* io, uint64_t variant) noexcept
        {
            return misa77_entropy_safe_kernel_neon(io, variant);
        }

        static void entropy_huffman_kernel(huffman_decoder::kernel_io* io) noexcept
        {
            misa77_entropy_huffman_kernel_neon(io);
        }

        static void entropy_delta_kernel(uint8_t* b, uint8_t* e, uint64_t stride) noexcept
        {
            misa77_entropy_delta_kernel_neon(b, e, stride);
        }

        static int64_t entropy_merge_kernel(entropy::merge_io* io) noexcept
        {
            return misa77_entropy_merge_kernel_neon(io);
        }
#else
        static constexpr bool has_entropy_kernel = false;
        static constexpr bool has_entropy_merge_kernel = false;
#endif
    };
} // namespace misa77

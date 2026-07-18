// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

// ARM NEON (2x128-bit) copy/finder policy.
//
// CONTRACT: NEON (AdvSIMD) is guaranteed by the AArch64 ISA, so this header only
// requires the including TU to be an AArch64 baseline build (-march=armv8-a).

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
    };
} // namespace misa77

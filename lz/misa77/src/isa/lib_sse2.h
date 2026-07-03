// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

// SSE2 (2x128-bit) copy/finder policy.
//
// CONTRACT: SSE2 is guaranteed by the x86-64 ISA, so this header only requires
// the including TU to be a genuine x86-64 baseline build (-march=x86-64).

#include "format.h" // vector_width

#include <cstdint>
#include <immintrin.h>

namespace misa77
{
    class lib_sse2
    {
    public:
        // Copies `src[0, vector_width)` to `src[dis, dis + vector_width)`.
        // It's guaranteed that `dis >= vector_width`, so there's no aliasing.
        [[gnu::always_inline]]
        static inline void cyccpy(uint8_t* src, uint64_t dis)
        {
            __m128i reg1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
            __m128i reg2 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + vector_width / 2));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(src + dis), reg1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(src + dis + vector_width / 2), reg2);
        }

        // Copies `src[0, vector_width)` to `dst[0, vector_width)`.
        // It's guaranteed that the two ranges don't alias.
        [[gnu::always_inline]]
        static inline void copy32(uint8_t* __restrict dst, const uint8_t* __restrict src)
        {
            __m128i reg1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
            __m128i reg2 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + vector_width / 2));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), reg1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + vector_width / 2), reg2);
        }

        struct vec
        {
            __m128i lo, hi;
        };

        // Return internal representation of `src[0, vector_width)`.
        [[gnu::always_inline]]
        static inline vec loadvec(const uint8_t* src)
        {
            return vec{_mm_loadu_si128(reinterpret_cast<const __m128i*>(src)),
                       _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + vector_width / 2))};
        }

        // First differing byte index (0-indexed) between `reg1` and `reg2`.
        // If there's no such index, return `vector_width`.
        [[gnu::always_inline]]
        static inline uint64_t lcp(const vec& reg1, const vec& reg2)
        {
            const __m128i eq0 = _mm_cmpeq_epi8(reg1.lo, reg2.lo);
            const __m128i eq1 = _mm_cmpeq_epi8(reg1.hi, reg2.hi);
            const uint32_t mask0 = static_cast<uint32_t>(_mm_movemask_epi8(eq0));
            const uint32_t mask1 = static_cast<uint32_t>(_mm_movemask_epi8(eq1));
            const uint32_t diff = ~(mask0 | (mask1 << 16));
// __builtin_ctz(0) is UB, but tzcnt is defined for 0 and conveniently returns 32
#if defined(__BMI__)
            return static_cast<uint64_t>(_tzcnt_u32(diff));
#else
            return diff ? static_cast<uint64_t>(__builtin_ctz(diff)) : 32;
#endif
        }
    };
} // namespace misa77

// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

// AVX2 copy/finder policy (256-bit).
//
// CONTRACT: the intrinsics below are AVX2, so this header MUST only be included
// from a TU compiled with -mavx2.

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
    };
} // namespace misa77

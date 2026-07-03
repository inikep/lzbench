// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

// Portable copy/finder policy.
//
// CONTRACT: no ISA requirement, safe to include from any TU.

#include "format.h" // vector_width
#include "util.h"   // loadu8

#include <array>
#include <cstdint>
#include <cstring>

namespace misa77
{
    // The compiler should auto-vectorize the `memcpy` calls in here w.r.t. the target architecture,
    // whenever possible.
    class lib_portable
    {
    public:
        // Copies `src[0, vector_width)` to `src[dis, dis + vector_width)`
        // It's guaranteed that `dis >= vector_width`, so there's no aliasing.
        [[gnu::always_inline]]
        static inline void cyccpy(uint8_t* src, uint64_t dis)
        {
            memcpy(src + dis, src, vector_width);
        }

        // Copies `src[0, vector_width)` to `dst[0, vector_width)`.
        // It's guaranteed that the two ranges don't alias.
        [[gnu::always_inline]]
        static inline void copy32(uint8_t* __restrict dst, const uint8_t* __restrict src)
        {
            memcpy(dst, src, vector_width);
        }

        // Any internal representation of 32 bytes of contiguous data is fine.
        using vec = std::array<uint8_t, vector_width>;

        // Return internal representation of `src[0, vector_width)`.
        [[gnu::always_inline]]
        static inline vec loadvec(const uint8_t* src)
        {
            vec ret;
            std::memcpy(ret.data(), src, vector_width);
            return ret;
        }

        // First differing byte index (0-indexed) between `reg1` and `reg2`.
        // If there's no such index, return `vector_width`.
        // Little-endian assumed:
        // ctz of the XOR points at the lowest differing bit, hence the lowest byte.
        [[gnu::always_inline]]
        static inline uint64_t lcp(const vec& reg1, const vec& reg2)
        {
            for (uint64_t off = 0; off < vector_width; off += 8)
            {
                const uint64_t x = loadu8(reg1.data() + off) ^ loadu8(reg2.data() + off);
                if (x)
                    return off + (static_cast<uint64_t>(__builtin_ctzll(x)) >> 3);
            }
            return vector_width;
        }
    };
} // namespace misa77

// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "format.h"
#include "misa77/misa77.h"
#include "util.h"

#include <cstdint>
#include <cstring>

namespace misa77
{
    template <bool cond, class isa_lib>
    uint64_t heavy_unsafe_decompress_inner(const uint8_t* __restrict src,
                                           uint64_t src_size,
                                           uint8_t* __restrict dst,
                                           uint64_t dst_cap)
    {
        using namespace heavy;

        const uint64_t original_size = decompressed_size(src);

        if (dst_cap < original_size or src_size < 8)
            return 0;

        bool check = ((1 << format_bit) & src[7]);
        if (!check)
            return 0;

        // Small source
        if (original_size <= small_lim)
        {
            if (original_size > 0)
                memcpy(dst, src + uint64_t(8), original_size);
            return original_size;
        }

        const uint64_t literal_suffix_cnt = loadu8(src + 8);
        const uint8_t* control = src + 16;
        const uint8_t* literals = src + src_size - literal_suffix_cnt;
        uint8_t* out = dst;

        uint32_t token = loadu4(control);
        while (control < literals)
        {
            uint64_t lit_len = token >> 26;
            uint64_t match_len = LUT.len_of[(token >> 20) & 63];
            uint64_t dis = (token & dis_mask) + min_dis;
            control += 4;

            // should almost never be taken with the heavy format :)
            if (lit_len == lit_lim) [[unlikely]]
            {
                uint64_t pot_add = *control;
                lit_len += pot_add;
                ++control;
                if (pot_add == 255) [[unlikely]]
                {
                    while (*control == 255) [[unlikely]]
                        lit_len += 255, ++control;
                    lit_len += *control, ++control;
                }
            }

            literals -= lit_len;
            if (lit_len > dec_literal_copy) [[unlikely]]
            {
                isa_lib::copy32(out, literals);
                if (lit_len > vector_width) [[unlikely]]
                    memcpy(out + vector_width, literals + vector_width, lit_len - vector_width);
            }
            else
            {
                // Unconditional copy, safe because we have `dec_literal_copy` bytes of
                // breathing room at the end in src and dst due to `literal_suffix`
                memcpy(out, literals, dec_literal_copy);
            }
            out += lit_len;

            // load this early
            token = loadu4(control);

            uint8_t* match_at = out - dis;

            isa_lib::cyccpy(match_at, dis);
            if constexpr (cond)
            {
                if (match_len > vector_width) [[unlikely]] // ummmm
                {
                    isa_lib::cyccpy(match_at + vector_width, dis);
                    if (match_len > vector_width + vector_width) [[unlikely]]
                        for (uint64_t off = 64; off < match_len; off += 32)
                            isa_lib::cyccpy(match_at + off, dis);
                }
            }
            else
            {
                isa_lib::cyccpy(match_at + vector_width, dis);
                if (match_len > vector_width + vector_width) [[unlikely]]
                    for (uint64_t off = 64; off < match_len; off += 32)
                        isa_lib::cyccpy(match_at + off, dis);
            }
            out += match_len;
        }

        if (uint64_t(out - dst) != original_size - literal_suffix_cnt)
            return 0;
        memcpy(out, src + src_size - literal_suffix_cnt, literal_suffix_cnt);
        return original_size;
    }

    // Returns number of bytes written to `dst`, and 0 on failure.
    // `isa_lib` is ISA-dependent.
    template <class isa_lib>
    uint64_t heavy_unsafe_dimpl(const uint8_t* __restrict src,
                                uint64_t src_size,
                                uint8_t* __restrict dst,
                                uint64_t dst_cap)
    {
        using namespace heavy;

        const uint64_t original_size = decompressed_size(src);

        if (dst_cap < original_size or src_size < 8)
            return 0;

        bool check = ((1 << format_bit) & src[7]);
        if (!check)
            return 0;

        const bool check2 = ((1 << cond_bit) & src[7]);

        if (check2)
            return heavy_unsafe_decompress_inner<true, isa_lib>(src, src_size, dst, dst_cap);
        return heavy_unsafe_decompress_inner<false, isa_lib>(src, src_size, dst, dst_cap);
    }
} // namespace misa77

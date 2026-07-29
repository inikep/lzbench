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
    // Returns number of bytes written to `dst`, and 0 on failure.
    // `isa_lib` is ISA-dependent.
    template <class isa_lib>
    uint64_t light_unsafe_dimpl(const uint8_t* __restrict src,
                                uint64_t src_size,
                                uint8_t* __restrict dst,
                                uint64_t dst_cap)
    {
        using namespace light;

        const uint64_t original_size = loadu8(src);

        if (dst_cap < original_size)
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

        // Token loop with safe overwrites in the destination buffer and safe overreads from source.
        while (control < literals)
        {
            // Overread is safe here
            uint8_t token = control[0];
            uint64_t lit_len = token >> 5;
            uint64_t match_len = (token & uint8_t(0x1F)) + min_match_len - 1;

            uint16_t dis_small = loadu2(control + 1);
            uint32_t dis = dis_small + hashtab_lag + 1;

            control += 3;

            if (lit_len == 7) [[unlikely]]
            {
                uint64_t pot_add = *control;
                lit_len += pot_add;
                ++control;

                constexpr uint64_t block = 255;
                if (pot_add == block) [[unlikely]]
                {
                    while (*control == block) [[unlikely]]
                        lit_len += block, ++control;
                    lit_len += *control, ++control;
                }
            }

            literals -= lit_len;
            if (lit_len > dec_literal_copy) [[unlikely]]
            {
                isa_lib::copy32(out, literals);

                if (lit_len > vector_width) [[unlikely]]
                {
                    memcpy(out + vector_width, literals + vector_width, lit_len - vector_width);
                }
            }
            else
            {
                // Unconditional copy, safe because we have `dec_literal_copy` bytes of
                // breathing room at the end in src and dst due to `literal_suffix`
                memcpy(out, literals, dec_literal_copy);
            }
            out += lit_len;

            // When we enter cyccpy, we're guaranteed to have `literal_suffix` >= vector_width
            // bytes of breathing room in the destination buffer as the last `literal_suffix`
            // bytes are literals
            isa_lib::cyccpy(out - dis, dis);
            out += match_len;
        }

        if (uint64_t(out - dst) != original_size - literal_suffix_cnt)
            return 0;

        // Literal suffix at the end
        memcpy(out, src + src_size - literal_suffix_cnt, literal_suffix_cnt);
        out += literal_suffix_cnt;

        return uint64_t(out - dst);
    }
} // namespace misa77

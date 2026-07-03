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
    uint64_t decompress_impl(const uint8_t* __restrict src,
                             uint64_t src_size,
                             uint8_t* __restrict dst,
                             uint64_t dst_cap)
    {
        const uint64_t original_size = decompressed_size(src);

        if (dst_cap < original_size)
            return 0;

        // Small source
        if (original_size <= small_lim)
        {
            if (original_size > 0)
                memcpy(dst, src + uint64_t(8), original_size);
            return original_size;
        }

        // Left and right pointers in the source buffer
        uint64_t lpos = 0, rpos = src_size;

        // Read the original size
        lpos += 8;

        // Size of literal suffix
        uint64_t literal_suffix_cnt = loadu8(src + lpos);
        lpos += 8;
        rpos -= literal_suffix_cnt;

        // Position in the destination buffer
        uint64_t dpos = 0;

        // Initial loop with safe overwrites in the destination buffer and safe overreads from
        // source buffer
        while (lpos < rpos)
        {
            // Overread is safe here
            uint8_t token = src[lpos];
            uint64_t lit_len = token >> 5;
            uint64_t match_len = (token & uint8_t(0x1F)) + min_match_len - 1;

            uint16_t dis_small = loadu2(src + lpos + 1);
            uint32_t dis = dis_small + hashtab_lag + 1;

            lpos += 3;

            if (lit_len == 7) [[unlikely]]
            {
                uint64_t pot_add = src[lpos];
                lit_len += pot_add;
                lpos++;

                constexpr uint64_t block = 255;
                if (pot_add == block) [[unlikely]]
                {
                    while (src[lpos] == block) [[unlikely]]
                        lit_len += block, ++lpos;
                    lit_len += src[lpos], ++lpos;
                }
            }

            if (lit_len > dec_literal_copy) [[unlikely]]
            {
                rpos -= lit_len;

                isa_lib::copy32(dst + dpos, src + rpos);

                if (lit_len > vector_width) [[unlikely]]
                {
                    memcpy(dst + (dpos + vector_width),
                           src + (rpos + vector_width),
                           lit_len - vector_width);
                }
            }
            else
            {
                // Unconditional copy, safe because we have `dec_literal_copy` bytes of
                // breathing room at the end in src and dst due to `literal_suffix`
                rpos -= lit_len;
                memcpy(dst + dpos, src + rpos, dec_literal_copy);
            }
            dpos += lit_len;

            // When we enter cyccpy, we're guaranteed to have `literal_suffix` >= vector_width
            // bytes of breathing room in the destination buffer as the last `literal_suffix`
            // bytes are literals
            isa_lib::cyccpy(dst + (dpos - dis), dis);
            dpos += match_len;
        }

        if (dpos != original_size - literal_suffix_cnt)
            return 0;

        // Literal suffix at the end
        memcpy(dst + (original_size - literal_suffix_cnt),
               src + (src_size - literal_suffix_cnt),
               literal_suffix_cnt);
        dpos += literal_suffix_cnt;

        return dpos;
    }
} // namespace misa77
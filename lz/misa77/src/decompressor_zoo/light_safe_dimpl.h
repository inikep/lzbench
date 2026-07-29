// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "format.h"
#include "util.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace misa77
{
    // Returns number of bytes written to `dst`, and 0 on failure.
    // `isa_lib` is ISA-dependent.
    template <class isa_lib>
    uint64_t light_safe_dimpl(const uint8_t* __restrict src,
                              uint64_t src_size,
                              uint8_t* __restrict dst,
                              uint64_t dst_cap)
    {
        using namespace light;

        // Header fields must exist
        if (src_size < 8) [[unlikely]]
            return 0;

        const uint64_t original_size = loadu8(src);

        if (dst_cap < original_size)
            return 0;

        // Small source
        if (original_size <= small_lim)
        {
            // Must not over-read here
            if (src_size < 8 + original_size) [[unlikely]]
                return 0;
            if (original_size > 0)
                memcpy(dst, src + uint64_t(8), original_size);
            return original_size;
        }

        // Extra headers for non-small source
        if (src_size < 16) [[unlikely]]
            return 0;

        const uint64_t literal_suffix_cnt = loadu8(src + 8);

        // The raw suffix must fit in both buffers (for our unconditional over-reads and writes to
        // be safe), and must be >= literal_suffix
        if (literal_suffix_cnt < literal_suffix or literal_suffix_cnt > src_size - 16 or
            literal_suffix_cnt > original_size) [[unlikely]]
            return 0;

        const uint8_t* control = src + 16;
        const uint8_t* literals = src + src_size - literal_suffix_cnt;
        uint8_t* out = dst;

        // The loop may produce at most this much output (we put the raw suffix after this)
        const uint64_t loop_limit = original_size - literal_suffix_cnt;
        uint8_t* const out_limit = dst + loop_limit;

        constexpr uint64_t dis_ceil = dis_lim + hashtab_lag; // max representable dis

        uint8_t* const prefix_end = dst + std::min(dis_ceil, loop_limit);

        // Fully guarded step for an initial small prefix (match distance is dangerous here), and
        // the tail
        auto guarded_step = [&]() -> bool
        {
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
                    // Bounded, shouldn't just keep reading ahead as it finds 255s
                    while (*control == block) [[unlikely]]
                    {
                        lit_len += block, ++control;
                        if (control >= literals) [[unlikely]]
                            return false;
                    }
                    lit_len += *control, ++control;
                }
            }

            // no src underflow.
            if (lit_len > uint64_t(literals - src)) [[unlikely]]
                return false;

            // bound this token's dst writes
            if (uint64_t(out - dst) + lit_len + min_match_len > loop_limit) [[unlikely]]
                return false;

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
                memcpy(out, literals, dec_literal_copy);
            }
            out += lit_len;

            // match source must not start before dst[0].
            if (dis > uint64_t(out - dst)) [[unlikely]]
                return false;

            isa_lib::cyccpy(out - dis, dis);
            out += match_len;
            return true;
        };

        // Prefix: first dis_ceil output bytes.
        while (control < literals and out < prefix_end)
            if (!guarded_step()) [[unlikely]]
                return 0;

        // The fast loop below must never begin a token within red_slack bytes of
        // out_limit.
        constexpr uint64_t red_slack = 8;

        // A non-extras token has lit_len <= 6 and starts at out <= out_limit - red_slack
        // cyccpy write must stay inside the suffix slack past out_limit
        static_assert(6 + vector_width - (red_slack + 1) <= literal_suffix);
        // a match code 31 must not advance `out` past one-past-the-end of a dst buffer
        static_assert(6 + (31 + min_match_len - 1) - (red_slack + 1) <= literal_suffix);

        uint8_t* const red = dst + (loop_limit > red_slack ? loop_limit - red_slack : uint64_t(0));

        // Fast main loop: no per-token guards outside the extras branch.
        while (control < literals and out < red)
        {
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
                    // Bounded, shouldn't just keep reading ahead as it finds 255s
                    while (*control == block) [[unlikely]]
                    {
                        lit_len += block, ++control;
                        if (control >= literals) [[unlikely]]
                            return 0;
                    }
                    lit_len += *control, ++control;
                }

                if (lit_len > uint64_t(literals - src)) [[unlikely]]
                    return 0;
#if defined(__clang__)
                // Equivalent to the check below, rewritten in terms of `red`
                if (lit_len - (red_slack - min_match_len) > uint64_t(red - out)) [[unlikely]]
                    return 0;
#else
                if (lit_len + min_match_len > uint64_t(out_limit - out)) [[unlikely]]
                    return 0;
#endif
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
                memcpy(out, literals, dec_literal_copy);
            }
            out += lit_len;

            isa_lib::cyccpy(out - dis, dis);
            out += match_len;
        }

        // The last few tokens
        while (control < literals)
            if (!guarded_step()) [[unlikely]]
                return 0;

        if (out != out_limit)
            return 0;

        // Literal suffix at the end
        memcpy(out, src + src_size - literal_suffix_cnt, literal_suffix_cnt);
        return loop_limit + literal_suffix_cnt;
    }
} // namespace misa77

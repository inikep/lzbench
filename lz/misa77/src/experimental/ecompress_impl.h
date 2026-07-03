// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "format.h"
#include "misa77/experimental.h"
#include "misa77/misa77.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace misa77
{
    namespace experimental
    {
        namespace
        {
            static_assert(max_match_len >= 32);

            constexpr uint32_t hash_top = 16;
            constexpr uint32_t hash_siz = 1 << hash_top;
            constexpr uint32_t hash_mul = 2654435761;

            // Hash 4 bytes to an integer in `[0, 1 << (hash_top))`.
            [[gnu::always_inline]]
            inline uint32_t hash4(const uint32_t val)
            {
                return (val * hash_mul) >> (32 - hash_top);
            }

            // Assuming that rem contains top 3 bits of the token, emits bytes specifying `n`.
            [[gnu::always_inline]]
            inline void emit_length_extras(uint8_t* dst, uint64_t& dlpos, uint64_t n, uint8_t rem)
            {
                if (rem != uint64_t(7)) [[likely]]
                    return;

                // Write this unconditionally first, if it's wrong we'll just overwrite
                dst[dlpos] = static_cast<uint8_t>(n);

                constexpr uint64_t block = 255;
                if (n >= block) [[unlikely]]
                {
                    while (n >= block) [[unlikely]]
                        dst[dlpos] = block, ++dlpos, n -= block;
                    dst[dlpos] = static_cast<uint8_t>(n);
                }

                ++dlpos;
            }

            // Size of region in bytes
            constexpr uint64_t region_width = 1 << 16;
            static_assert(region_width == dis_lim);

            constexpr uint64_t hashtab_wid = 16;
            constexpr uint64_t cand_lookahead = 8;
            constexpr uint64_t inf = std::numeric_limits<uint64_t>::max();

            [[gnu::always_inline]]
            // returns block cost - match length cost
            inline uint64_t l_cost(const uint32_t& lit_len, const param& given)
            {
                uint64_t extra = (lit_len < 7) ? 0 : (lit_len - 7) / 255 + 1;
                uint64_t add_cost = given.block + given.size * (3 + extra + lit_len);
                uint64_t lit_cost = (lit_len >= 7) * given.lit7 + (lit_len >= 17) * given.lit17 +
                                    (lit_len >= 33) * given.lit33;
                return add_cost + lit_cost;
            }

            [[gnu::always_inline]]
            // returns match length cost
            inline uint64_t m_cost(const uint32_t& match_len, const param& given)
            {
                if (match_len < 8)
                    return given.short4_7;
                if (match_len < 16)
                    return given.short8_15;
                return 0;
            }

        } // namespace

        template <class isa_lib>
        uint64_t compress_tuned_impl(const uint8_t* __restrict src,
                                     uint64_t src_size,
                                     uint8_t* __restrict dst,
                                     uint64_t dst_cap,
                                     const param& given)
        {
            if (given.use_default)
                return compress(src, src_size, dst, dst_cap);

            if (compress_bound(src_size) > dst_cap)
                return 0;

            // Left pointer in the destination buffer (metadata and control bytes)
            uint64_t dlpos = 0;

            // Right pointer in the destination buffer (literal bytes)
            // We've written to `[drpos, dst_cap)` at any given point of time
            uint64_t drpos = dst_cap;

            storeu8(dst, src_size);
            dlpos += 8;

            // Small source
            if (src_size <= small_lim)
            {
                if (src_size != 0)
                    memcpy(dst + dlpos, src, src_size);
                dlpos += src_size;
                return dlpos;
            }

            uint64_t literal_suffix_pos = dlpos;
            dlpos += 8;

            // Ensure that the last `literal_suffix` bytes will be literals
            uint64_t match_end_limit = (src_size < literal_suffix ? 0 : src_size - literal_suffix);

            // Beginning of the pending literal window
            uint64_t lit = 0;

            // `[0, hpos)` represents the range we've written into the hashtable
            uint64_t hpos = 0;

            std::vector<std::array<uint16_t, hashtab_wid>> hashtab(hash_siz);
            std::vector<uint8_t> hashtab_idx(hash_siz);

            // We reuse buffers across regions
            std::vector<uint8_t> match_len(region_width);
            std::vector<uint16_t> match_pos(region_width);

            // dp[u] = optimum cost for [u, region_width)
            std::vector<uint64_t> dp(region_width);
            // If optimal transition for dp[u] is choosing match start at u + x with length l, we
            // store ([top 16 bits = x] | [bottom 16 bits = l])
            std::vector<uint32_t> dp_forward(region_width);

            uint16_t seen_head = 0;
            // Bottom 16 bits of location
            std::vector<uint16_t> seen(cand_lookahead);
            // Optimal match length to take here
            std::vector<uint16_t> seen_ml(cand_lookahead);
            // Suffix dp value when we take optimal match length seen_ml
            std::vector<uint64_t> seen_dp(cand_lookahead);

            for (uint64_t region_lb = 0; region_lb < match_end_limit; region_lb += region_width)
            {
                const uint64_t region_rb =
                    std::min(region_lb + region_width, match_end_limit); // exclusive

                // Reset every region
                seen_head = 0;
                std::fill(seen.begin(), seen.end(), 0);
                std::fill(seen_dp.begin(), seen_dp.end(), 0);
                std::fill(seen_ml.begin(), seen_ml.end(), 0);

                // Compute match_len and match_pos from left to right
                bool match_found = false;
                for (uint64_t pos = region_lb; pos < region_rb; pos++)
                {
                    // Insert into hash structure (table for now, might swap out for chain walking
                    // later)
                    constexpr uint64_t batch = 8;
                    while (pos >= hpos + hashtab_lag + batch) [[unlikely]]
                    {
#pragma GCC unroll batch
                        for (uint64_t i = 0; i < batch; i++)
                        {
                            uint32_t hsh = hash4(loadu4(src + hpos + i));
                            hashtab[hsh][hashtab_idx[hsh]] =
                                uint16_t(hpos + i); // We just store the lowest 2 bytes
                            hashtab_idx[hsh] =
                                (hashtab_idx[hsh] == hashtab_wid - 1
                                     ? uint8_t(0)
                                     : hashtab_idx[hsh] +
                                           1); // Just cycle the pointer in the ring buffer
                        }
                        hpos += batch;
                    }

                    uint16_t idx_into = pos & (region_width - 1);

                    uint32_t val = loadu4(src + pos);
                    uint32_t hsh = hash4(val);

                    match_len[idx_into] = match_pos[idx_into] = 0;

                    // Get best possible match, trim it so that it's within bounds
                    if (pos > hashtab_lag and pos + min_match_len <= region_rb) [[likely]]
                    {
                        typename isa_lib::vec reg = isa_lib::loadvec(src + pos);
// MLP
#pragma GCC unroll hashtab_wid
                        for (uint8_t i = 0; i < hashtab_wid; i++)
                        {
                            uint16_t d = uint16_t(pos - hashtab[hsh][i] - hashtab_lag - 1);

                            static_assert(max_match_len == hashtab_lag);
                            // Guaranteed to lie within `[pos - 2^16 - hashtab_lag, pos -
                            // hashtab_lag)`
                            uint64_t ilst = (pos - max_match_len - 1 - d);

                            typename isa_lib::vec ireg = isa_lib::loadvec(src + ilst);
                            uint32_t imatch_len = isa_lib::lcp(reg, ireg);
                            match_pos[idx_into] =
                                (imatch_len > match_len[idx_into] ? hashtab[hsh][i]
                                                                  : match_pos[idx_into]);
                            match_len[idx_into] =
                                (imatch_len > match_len[idx_into] ? imatch_len
                                                                  : match_len[idx_into]);
                        }
                        match_len[idx_into] =
                            std::min(static_cast<uint64_t>(match_len[idx_into]), region_rb - pos);
                        match_found = match_found or (match_len[idx_into] >= min_match_len);
                    }
                }

                if (!match_found)
                {
                    // nothing to do here
                    continue;
                }

                // Compute dp from right to left
                for (uint64_t pos = region_rb; (pos--) > region_lb;)
                {
                    uint16_t idx_into = pos & (region_width - 1);
                    dp[idx_into] = inf;
                    dp_forward[idx_into] = 0;

                    // Insert ourselves into the lookahead ring first, and attempt a transition
                    if (match_len[idx_into] >= min_match_len)
                    {
                        seen[seen_head] = idx_into;
                        seen_dp[seen_head] = inf;
                        for (uint64_t m = min_match_len; m <= match_len[idx_into]; m++)
                        {
                            uint64_t s_cost_here =
                                m_cost(m, given) + (pos + m == region_rb ? 0 : dp[idx_into + m]);
                            if (s_cost_here < seen_dp[seen_head])
                            {
                                seen_dp[seen_head] = s_cost_here;
                                seen_ml[seen_head] = m;
                            }
                        }

                        // Transition with 0 lit length
                        uint64_t lit_len_here = idx_into - idx_into;
                        uint64_t l_cost_here = l_cost(lit_len_here, given);
                        uint64_t m_cost_here = seen_dp[seen_head];
                        if (dp[idx_into] > l_cost_here + m_cost_here)
                        {
                            // Yay
                            dp[idx_into] = l_cost_here + m_cost_here;
                            dp_forward[idx_into] = (lit_len_here << 16) | (seen_ml[seen_head]);
                        }

                        seen_head = (seen_head + 1 == cand_lookahead ? 0 : seen_head + 1);
                    }

                    for (uint16_t i = 0; i < cand_lookahead; i++)
                    {
                        uint16_t nxt = seen[i];
                        if (nxt > idx_into) [[likely]]
                        {
                            // Attempt transition
                            uint64_t lit_len_here = nxt - idx_into;
                            uint64_t l_cost_here = l_cost(lit_len_here, given);
                            uint64_t m_cost_here = seen_dp[i];
                            if (dp[idx_into] > l_cost_here + m_cost_here)
                            {
                                // Yay
                                dp[idx_into] = l_cost_here + m_cost_here;
                                dp_forward[idx_into] = (lit_len_here << 16) | (seen_ml[i]);
                            }
                        }
                    }

                    // Assuming sensible weights, this only happens when there's no match ahead, so
                    // we can just pretend that this is the end
                    if (dp[idx_into] == inf)
                        dp[idx_into] = 0;
                }

                // Reconstruct optimal sequence from left to right and emit stream
                uint64_t pos = region_lb;
                while (pos < region_rb)
                {
                    uint16_t idx_into = pos & (region_width - 1);

                    uint32_t trans = dp_forward[idx_into];
                    uint64_t match_start = pos + (trans >> 16);
                    uint16_t match_length = trans ^ ((trans >> 16) << 16);

                    // Incomplete block at the end in this region
                    if (match_length == 0)
                        break;

                    uint64_t norm_match_len = match_length - min_match_len + 1;

                    uint16_t d =
                        uint16_t(match_start - match_pos[match_start & (region_width - 1)] -
                                 hashtab_lag - 1);
                    uint64_t match_from = (match_start - max_match_len - 1 - d);

                    // `src[lit, match_start)` are the literals
                    uint64_t lit_len = match_start - lit;
                    uint8_t lrem = std::min(uint64_t(7), lit_len);
                    dst[dlpos] = static_cast<uint8_t>((lrem << 5) | (norm_match_len));
                    lit_len -= lrem;
                    ++dlpos;

                    // 2 Distance bytes
                    uint64_t dis = match_start - match_from;
                    uint16_t dbytes = dis - hashtab_lag - 1;
                    storeu2(dst + dlpos, dbytes);
                    dlpos += 2;

                    // Extra literal length bytes
                    emit_length_extras(dst, dlpos, lit_len, lrem);

                    // Literal bytes
                    if (lit < match_start)
                    {
                        uint64_t lit_cnt = match_start - lit;
                        drpos -= lit_cnt;
                        memcpy(dst + drpos, src + lit, lit_cnt);
                    }

                    pos = match_start + match_length;
                    lit = pos;
                }
            }

            // Close the gap between the two streams by moving the literal stream to the left
            if (drpos < dst_cap)
            {
                memmove(dst + dlpos, dst + drpos, dst_cap - drpos);
                dlpos += dst_cap - drpos;
            }

            uint64_t literal_suffix_cnt = src_size - lit;
            storeu8(dst + literal_suffix_pos, literal_suffix_cnt);
            memcpy(dst + dlpos, src + lit, literal_suffix_cnt);
            dlpos += literal_suffix_cnt;

            return dlpos;
        }
    } // namespace experimental
} // namespace misa77
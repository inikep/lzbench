// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "format.h"
#include "misa77/misa77.h"
#include "suffix/sais.h"
#include "util.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace misa77
{
    namespace light_optimal_detail
    {
        using namespace light;

        // 384 KB blocks, empirically derived.
        constexpr uint64_t block_size = uint64_t(3) << 17;

        // segments can extend past block boundaries
        constexpr uint64_t pad_len = max_match_len + 1;

        constexpr uint64_t dp_inf = std::numeric_limits<uint64_t>::max();

        [[gnu::always_inline]]
        inline uint64_t lit_extras(uint64_t run)
        {
            return run < lit_lim ? 0 : 1 + (run - lit_lim) / 255;
        }

        // 3-level bitset that maintains the positions of the alive suffixes inside the match
        // window
        struct liveset
        {
            std::vector<uint64_t> l0, l1, l2;

            // live iff sa[r] < lim
            void build(const int32_t* sa, uint64_t m, uint32_t lim)
            {
                l0.assign((m + 63) / 64, 0);
                l1.assign((l0.size() + 63) / 64, 0);
                l2.assign((l1.size() + 63) / 64, 0);
                for (uint64_t w = 0; w < l0.size(); ++w)
                {
                    const uint64_t base = w << 6;
                    const uint64_t top = std::min<uint64_t>(64, m - base);
                    uint64_t bits = 0;
                    for (uint64_t j = 0; j < top; ++j)
                        bits |= uint64_t(uint32_t(sa[base + j]) < lim) << j;
                    l0[w] = bits;
                }
                for (uint64_t w = 0; w < l0.size(); ++w)
                    if (l0[w])
                        l1[w >> 6] |= uint64_t(1) << (w & 63);
                for (uint64_t w = 0; w < l1.size(); ++w)
                    if (l1[w])
                        l2[w >> 6] |= uint64_t(1) << (w & 63);
            }

            void set(uint64_t i)
            {
                l0[i >> 6] |= uint64_t(1) << (i & 63);
                l1[i >> 12] |= uint64_t(1) << ((i >> 6) & 63);
                l2[i >> 18] |= uint64_t(1) << ((i >> 12) & 63);
            }

            void clear(uint64_t i)
            {
                if ((l0[i >> 6] &= ~(uint64_t(1) << (i & 63))) == 0)
                    if ((l1[i >> 12] &= ~(uint64_t(1) << ((i >> 6) & 63))) == 0)
                        l2[i >> 18] &= ~(uint64_t(1) << ((i >> 12) & 63));
            }

            // largest live index < i, or -1
            int64_t prev(uint64_t i) const
            {
                uint64_t w = i >> 6;
                uint64_t mask = (i & 63) ? (uint64_t(1) << (i & 63)) - 1 : 0;
                if (uint64_t v = l0[w] & mask)
                    return int64_t((w << 6) + 63 - uint64_t(__builtin_clzll(v)));
                uint64_t w1 = w >> 6;
                uint64_t m1 = (w & 63) ? (uint64_t(1) << (w & 63)) - 1 : 0;
                if (uint64_t v = l1[w1] & m1)
                {
                    w = (w1 << 6) + 63 - uint64_t(__builtin_clzll(v));
                    return int64_t((w << 6) + 63 - uint64_t(__builtin_clzll(l0[w])));
                }
                uint64_t w2 = w1 >> 6;
                uint64_t m2 = (w1 & 63) ? (uint64_t(1) << (w1 & 63)) - 1 : 0;
                for (int64_t t = int64_t(w2); t >= 0; --t)
                {
                    uint64_t v2 = l2[t] & (t == int64_t(w2) ? m2 : ~uint64_t(0));
                    if (!v2)
                        continue;
                    uint64_t a = (uint64_t(t) << 6) + 63 - uint64_t(__builtin_clzll(v2));
                    uint64_t b = (a << 6) + 63 - uint64_t(__builtin_clzll(l1[a]));
                    return int64_t((b << 6) + 63 - uint64_t(__builtin_clzll(l0[b])));
                }
                return -1;
            }

            // smallest live index > i, or -1
            int64_t next(uint64_t i) const
            {
                uint64_t w = i >> 6;
                uint64_t mask = (i & 63) == 63 ? 0 : ~((uint64_t(2) << (i & 63)) - 1);
                if (uint64_t v = l0[w] & mask)
                    return int64_t((w << 6) + uint64_t(__builtin_ctzll(v)));
                uint64_t w1 = w >> 6;
                uint64_t m1 = (w & 63) == 63 ? 0 : ~((uint64_t(2) << (w & 63)) - 1);
                if (uint64_t v = l1[w1] & m1)
                {
                    w = (w1 << 6) + uint64_t(__builtin_ctzll(v));
                    return int64_t((w << 6) + uint64_t(__builtin_ctzll(l0[w])));
                }
                uint64_t w2 = w1 >> 6;
                uint64_t m2 = (w1 & 63) == 63 ? 0 : ~((uint64_t(2) << (w1 & 63)) - 1);
                for (uint64_t t = w2; t < l2.size(); ++t)
                {
                    uint64_t v2 = l2[t] & (t == w2 ? m2 : ~uint64_t(0));
                    if (!v2)
                        continue;
                    uint64_t a = (t << 6) + uint64_t(__builtin_ctzll(v2));
                    uint64_t b = (a << 6) + uint64_t(__builtin_ctzll(l1[a]));
                    return int64_t((b << 6) + uint64_t(__builtin_ctzll(l0[b])));
                }
                return -1;
            }
        };
    } // namespace light_optimal_detail

    // Returns number of bytes written to `dst`, and 0 on failure.
    // `isa_lib` is ISA-dependent.
    template <class isa_lib>
    uint64_t light_optimal_cimpl(const uint8_t* __restrict src,
                                 uint64_t src_size,
                                 uint8_t* __restrict dst,
                                 uint64_t dst_cap)
    {
        using namespace light;
        using namespace light_optimal_detail;

        if (compress_bound(src_size, config()) > dst_cap)
            return 0;

        auto lcp_long = [&](const uint8_t* a, const uint8_t* b, uint32_t max_len) -> uint32_t
        {
            uint32_t i = 0;
            while (i + vector_width <= max_len)
            {
                uint32_t len = isa_lib::lcp(isa_lib::loadvec(a + i), isa_lib::loadvec(b + i));
                i += len;
                if (len < vector_width)
                    return i;
            }
            while (i < max_len and a[i] == b[i])
                ++i;
            return i;
        };

        // Left pointer in the destination buffer (metadata and control bytes)
        uint64_t dlpos = 0;

        // Right pointer in the destination buffer (literal bytes)
        // We've written to `[drpos, dst_cap)` at any given point of time
        uint64_t drpos = dst_cap;

        // The top byte stays clear, which is what marks the stream as light
        storeu8(dst, src_size);
        dlpos += 8;

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

        auto emit_token = [&](uint64_t match_start, uint32_t match_len, uint32_t match_dis)
        {
            uint64_t lit_len = match_start - lit;
            uint32_t lrem = lit_len < lit_lim ? uint32_t(lit_len) : uint32_t(lit_lim);
            dst[dlpos] = uint8_t((lrem << 5) | (match_len - min_match_len + 1));
            ++dlpos;
            storeu2(dst + dlpos, uint16_t(match_dis - min_dis));
            dlpos += 2;
            if (lrem == lit_lim)
            {
                uint64_t e = lit_len - lit_lim;
                while (e >= 255)
                {
                    dst[dlpos++] = 255;
                    e -= 255;
                }
                dst[dlpos++] = uint8_t(e);
            }
            if (lit_len)
            {
                drpos -= lit_len;
                memcpy(dst + drpos, src + lit, lit_len);
            }
            lit = match_start + match_len;
        };

        // Per-block state, reused across blocks
        sais_chan sorter_chan;
        std::vector<int32_t> sa, rank;
        liveset live;
        std::vector<uint8_t> max_len;    // longest in-window match at each block position
        std::vector<uint32_t> match_dis; // its distance
        std::vector<uint64_t> dp;        // exact cost of the cheapest parse ending at boundary i

        struct arrival
        {
            uint32_t dis;
            uint32_t len;
            uint64_t lit_run; // literals immediately before the match
        };
        std::vector<arrival> arr;

        struct token_rec
        {
            uint64_t match_start;
            uint32_t len;
            uint32_t dis;
        };
        std::vector<token_rec> block_tokens;

        // Last committed parse boundary and its exact total cost
        uint64_t qstar = 0;
        uint64_t qstar_cost = 16; // header

        for (uint64_t bs = 0; bs < src_size; bs += block_size)
        {
            const uint64_t be = std::min(bs + block_size, src_size);
            const uint64_t blen = be - bs;

            // Suffix array and rank over [history | block | pad]
            const uint64_t seg0 = bs > max_dis ? bs - max_dis : 0;
            const uint64_t seg_end = std::min(src_size, be + pad_len);
            const uint32_t m = uint32_t(seg_end - seg0);
            sa.resize(m);
            rank.resize(m);
            sorter_chan.sa(src + seg0, m, sa.data(), rank.data());

            // Longest in-window match per position, clipped at the block end
            max_len.assign(blen, 0);
            match_dis.assign(blen, 0);

            // Matches must end within the block AND BEFORE the raw suffix
            const uint64_t hard = std::min(be, match_end_limit);
            const uint32_t init_lim =
                bs >= min_dis and bs - min_dis >= seg0 ? uint32_t(bs - min_dis - seg0 + 1) : 0;
            live.build(sa.data(), m, init_lim);

            uint64_t carry_len = 0;
            uint32_t carry_dis = 0;
            for (uint64_t p = bs; p < be; ++p)
            {
                // Slide the source window [p - max_dis, p - min_dis]
                if (p >= min_dis and p - min_dis >= seg0)
                    live.set(uint64_t(rank[p - min_dis - seg0]));
                if (p > max_dis and p - max_dis - 1 >= seg0)
                    live.clear(uint64_t(rank[p - max_dis - 1 - seg0]));

                if (p + min_match_len > hard)
                    continue;
                const uint32_t limit = uint32_t(std::min<uint64_t>(max_match_len, hard - p));

                // Skip the queries, make use of carry
                if (carry_len >= limit)
                {
                    max_len[p - bs] = uint8_t(limit);
                    match_dis[p - bs] = carry_dis;
                    --carry_len;
                    continue;
                }

                const uint64_t r = uint64_t(rank[p - seg0]);
                uint32_t best = 0;
                uint32_t best_dis = 0;
                auto consider = [&](int64_t q)
                {
                    if (q < 0)
                        return;
                    const uint64_t s = seg0 + uint64_t(sa[q]);
                    const uint32_t l = lcp_long(src + p, src + s, limit);
                    if (l > best)
                    {
                        best = l;
                        best_dis = uint32_t(p - s);
                    }
                };
                consider(live.prev(r));
                consider(live.next(r));

                if (best >= min_match_len)
                {
                    max_len[p - bs] = uint8_t(best);
                    match_dis[p - bs] = best_dis;
                    if (best >= limit)
                    {
                        // Extend past the clamp to seed the carry for upcoming positions
                        const uint64_t room = src_size - (p + limit);
                        const uint32_t ext = lcp_long(src + p + limit,
                                                      src + (p - best_dis) + limit,
                                                      uint32_t(std::min<uint64_t>(room, dis_lim)));
                        carry_len = uint64_t(limit) + ext - 1;
                        carry_dis = best_dis;
                    }
                    else
                    {
                        carry_len = best - 1;
                        carry_dis = best_dis;
                    }
                }
                else
                    carry_len -= uint64_t(carry_len > 0);
            }

            // Exact DP over boundaries [bs, be]
            dp.assign(blen + 1, dp_inf);
            arr.resize(blen + 1);

            // G-state = cheapest known "boundary + all literals since" arrival
            uint64_t g_run = bs - qstar;
            uint64_t g_cost = qstar_cost + g_run + lit_extras(g_run);

            // Next run length at which the extras byte count grows
            uint64_t g_next = g_run < lit_lim ? lit_lim : g_run + 255 - (g_run - lit_lim) % 255;
            if (bs == 0)
                dp[0] = qstar_cost;
            for (uint64_t p = bs; p <= be; ++p)
            {
                const uint64_t i = p - bs;
                if (p > bs)
                {
                    ++g_run;
                    if (g_run >= g_next) [[unlikely]]
                    {
                        ++g_cost;
                        g_next = g_run + 255;
                    }
                    ++g_cost;
                    if (dp[i] <= g_cost)
                    {
                        g_cost = dp[i];
                        g_run = 0;
                        g_next = lit_lim;
                    }
                }
                const uint32_t lmax = i < blen ? max_len[i] : 0;
                if (lmax >= min_match_len)
                {
                    // Every token is 3 bytes regardless of length, and every length up to
                    // `lmax` is representable, so these are `lmax - min_match_len + 1` edges
                    // of equal weight
                    const uint64_t c = g_cost + 3;
                    for (uint32_t len = min_match_len; len <= lmax; ++len)
                    {
                        const uint64_t q = i + len;
                        if (c < dp[q])
                        {
                            dp[q] = c;
                            arr[q] = {match_dis[i], len, g_run};
                        }
                    }
                }
            }

            // Commit at the seam
            uint64_t commit;
            uint64_t commit_cost;
            if (be < src_size)
            {
                commit = be - g_run;
                commit_cost = g_cost - g_run - lit_extras(g_run);
            }
            else
            {
                // Final block, so minimize cost + terminal literals
                // Raw suffix obviously doesn't contribute
                uint64_t best_total = qstar_cost + (src_size - qstar);
                commit = qstar;
                commit_cost = qstar_cost;
                for (uint64_t p = bs; p <= be; ++p)
                    if (dp[p - bs] != dp_inf and dp[p - bs] + (src_size - p) < best_total)
                    {
                        best_total = dp[p - bs] + (src_size - p);
                        commit = p;
                        commit_cost = dp[p - bs];
                    }
            }

            // backtrack commit, then emit forward...
            block_tokens.clear();
            for (uint64_t i = commit; i > qstar;)
            {
                const arrival& a = arr[i - bs];
                const uint64_t next = i - a.len - a.lit_run;
                if (a.len < min_match_len or (next < bs and next != qstar))
                    return 0; // sommething went wrong
                block_tokens.push_back({i - a.len, a.len, a.dis});
                i = next;
            }
            for (uint64_t k = block_tokens.size(); k-- > 0;)
                emit_token(block_tokens[k].match_start, block_tokens[k].len, block_tokens[k].dis);

            qstar = commit;
            qstar_cost = commit_cost;
        }

        // Close the gap between the two streams by moving the literal stream to the left
        if (drpos < dst_cap)
        {
            memmove(dst + dlpos, dst + drpos, dst_cap - drpos);
            dlpos += dst_cap - drpos;
        }

        // Just deal with the last few literals (guaranteed to be `>= literal_suffix >=
        // vector_width` bytes)
        uint64_t literal_suffix_cnt = src_size - lit;
        storeu8(dst + literal_suffix_pos, literal_suffix_cnt);
        memcpy(dst + dlpos, src + lit, literal_suffix_cnt);
        dlpos += literal_suffix_cnt;

        return dlpos;
    }
} // namespace misa77

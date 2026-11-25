// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "util.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

// Binary-tree match finder.
// Stores the Pareto frontier of (length, distance) over the window, lengths strictly increasing.

namespace misa77
{
    namespace bt_finder
    {
        struct match
        {
            uint32_t len, dis;
        };

        // Most frontier entries one position can produce (one per compared length, + 2)
        inline constexpr uint32_t max_frontier = 1100;

        inline uint32_t lcp8(const uint8_t* a, const uint8_t* b, uint32_t max_len)
        {
            uint32_t i = 0;
            while (i + 8 <= max_len)
            {
                const uint64_t x = loadu8(a + i) ^ loadu8(b + i);
                if (x)
                    return i + (uint32_t(__builtin_ctzll(x)) >> 3);
                i += 8;
            }
            while (i < max_len and a[i] == b[i])
                ++i;
            return i;
        }

        struct finder
        {
            static constexpr uint32_t nil = 0xFFFFFFFFu;
            // Matches are compared up to this length; the longest is then extended on its own
            static constexpr uint32_t cmp_max = 512;
            static constexpr uint32_t depth = 128;
            static constexpr uint32_t max_hash_bits = 20;
            // Behind a node closer than `min_dis`, walk this many nodes of its far subtree (they
            // would otherwise be hidden)
            static constexpr uint32_t side_walk = 8;
            // ring of the last positions each node replaced (an equal node closer than `min_dis`
            // hides the farther one it replaced); 64 > `min_dis`
            static constexpr uint32_t eq_ring = 64;

            const uint8_t* b = nullptr;
            uint32_t n = 0, min_dis = 1;
            uint64_t max_win = 0;
            std::vector<uint32_t> head, head2, son;
            uint32_t hash_bits = 0, wrap_mask = 0;
            bool wrap = false;
            uint32_t next_ins = 0, h2_next = 0;
            // a match longer than `cmp_max` is carried to the next position
            uint32_t carry = 0, carry_dis = 0;
            uint32_t eq[eq_ring];

            uint32_t idx(uint32_t p) const
            {
                return wrap ? (p & wrap_mask) : p;
            }

            uint32_t hash(uint32_t p) const
            {
                return ((loadu4(b + p) & 0xFFFFFFu) * 0x9E3779B1u) >> (32 - hash_bits);
            }

            void init(const uint8_t* base, uint32_t size, uint64_t window, uint32_t min_distance)
            {
                b = base, n = size, min_dis = min_distance;
                max_win = std::min<uint64_t>(window, size);
                uint32_t lg = 0;
                while ((uint64_t(1) << lg) < size)
                    ++lg;
                hash_bits = std::min<uint32_t>(max_hash_bits, std::max<uint32_t>(14, lg + 1));
                head.assign(uint64_t(1) << hash_bits, nil);
                head2.assign(65536, nil);
                if (max_win + 1 >= n)
                {
                    wrap = false;
                    son.resize(uint64_t(n) * 2 + 2);
                }
                else
                {
                    wrap = true;
                    uint32_t cyc = 1;
                    while (cyc < max_win + 2)
                        cyc <<= 1;
                    wrap_mask = cyc - 1;
                    son.resize(uint64_t(cyc) * 2);
                }
                next_ins = 0, h2_next = 0;
                carry = 0;
            }

            // Inserts `p` and, if `report`, appends its frontier to `out`, all in one descent
            uint32_t descend(uint32_t p, match* out, bool report)
            {
                eq[p & (eq_ring - 1)] = nil;
                if (n - p < 4)
                    return 0;
                const uint32_t lim = std::min(cmp_max, n - p);
                uint32_t k = 0, best = 2;
                if (report)
                {
                    // 2-byte matches: the newest one at `>= min_dis`
                    while (h2_next + min_dis <= p)
                    {
                        head2[loadu2(b + h2_next)] = h2_next;
                        ++h2_next;
                    }
                    const uint32_t c = head2[loadu2(b + p)];
                    if (c != nil and p - c <= max_win)
                    {
                        const uint32_t l = lcp8(b + p, b + c, lim);
                        out[k++] = {l, p - c};
                        best = std::max(best, l);
                    }
                }
                const uint32_t h = hash(p);
                uint32_t cur = head[h];
                head[h] = p;
                uint32_t* ptr0 = &son[uint64_t(idx(p)) * 2 + 1];
                uint32_t* ptr1 = &son[uint64_t(idx(p)) * 2];
                uint32_t len0 = 0, len1 = 0, d = depth;
                while (true)
                {
                    if (cur == nil or p - cur > max_win or d-- == 0)
                    {
                        *ptr0 = *ptr1 = nil;
                        break;
                    }
                    uint32_t* pair = &son[uint64_t(idx(cur)) * 2];
                    uint32_t len = std::min(len0, len1);
                    if (b[cur + len] == b[p + len])
                        len += 1 + lcp8(b + cur + len + 1, b + p + len + 1, lim - len - 1);
                    if (report and len > best)
                    {
                        if (p - cur >= min_dis)
                            out[k++] = {len, p - cur}, best = len;
                        else if (len < lim)
                        {
                            const bool smaller = b[cur + len] < b[p + len];
                            uint32_t y = pair[smaller ? 0 : 1], steps = side_walk;
                            const uint32_t lb = std::min(len0, len1);
                            while (y != nil and p - y <= max_win and steps-- != 0)
                            {
                                const uint32_t l = lb + lcp8(b + y + lb, b + p + lb, lim - lb);
                                if (l > best and p - y >= min_dis and l <= len)
                                    out[k++] = {l, p - y}, best = l;
                                y = son[uint64_t(idx(y)) * 2 + (smaller ? 1 : 0)];
                            }
                        }
                    }
                    if (len == lim)
                    {
                        *ptr1 = pair[0];
                        *ptr0 = pair[1];
                        eq[p & (eq_ring - 1)] = cur;
                        if (report and p - cur < min_dis and lim > best)
                        {
                            uint32_t e = cur;
                            while (e != nil and p - e < min_dis)
                                e = eq[e & (eq_ring - 1)];
                            if (e != nil and p - e <= max_win)
                                out[k++] = {lim, p - e}, best = lim;
                        }
                        break;
                    }
                    if (b[cur + len] < b[p + len])
                    {
                        *ptr1 = cur;
                        ptr1 = pair + 1;
                        cur = *ptr1;
                        len1 = len;
                    }
                    else
                    {
                        *ptr0 = cur;
                        ptr0 = pair;
                        cur = *ptr0;
                        len0 = len;
                    }
                }
                if (k > 0 and out[k - 1].len == lim and lim < n - p)
                    out[k - 1].len += lcp8(b + p + lim, b + p + lim - out[k - 1].dis, n - p - lim);
                return k;
            }

            // The frontier of `p` (positions must come in increasing order)
            uint32_t step(uint32_t p, match* out)
            {
                while (next_ins < p)
                    descend(next_ins++, nullptr, false);
                next_ins = p + 1;
                if (carry > cmp_max)
                {
                    descend(p, nullptr, false);
                    --carry;
                    out[0] = {carry, carry_dis};
                    return 1;
                }
                const uint32_t k = descend(p, out, true);
                carry = 0;
                if (k > 0 and out[k - 1].len > cmp_max)
                    carry = out[k - 1].len, carry_dis = out[k - 1].dis;
                return k;
            }

            // Frontiers of [bs, be): those of position `bs + i` are `mm[moff[i], moff[i + 1])`
            void find_block(uint32_t bs,
                            uint32_t be,
                            std::vector<uint32_t>& moff,
                            std::vector<match>& mm)
            {
                moff.clear();
                mm.clear();
                moff.reserve(be - bs + 1);
                match tmp[max_frontier];
                for (uint32_t p = bs; p < be; ++p)
                {
                    moff.push_back(uint32_t(mm.size()));
                    const uint32_t k = step(p, tmp);
                    mm.insert(mm.end(), tmp, tmp + k);
                }
                moff.push_back(uint32_t(mm.size()));
            }
        };
    } // namespace bt_finder
} // namespace misa77

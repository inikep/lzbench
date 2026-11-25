// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "entropy/bt_finder.h"
#include "util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Cheap size estimate of a DELTA cell under one stride: a greedy LZ parse priced in modelled
// bits, over the block's already-chosen cells as history.

namespace misa77
{
    namespace stride_estimator
    {
        // order-0 entropy in bits of a byte histogram of `n` bytes
        inline double hist_bits(const uint32_t* h, uint32_t n)
        {
            if (n == 0)
                return 0;
            double b = 0;
            for (int i = 0; i < 256; ++i)
                if (h[i])
                    b -= double(h[i]) * std::log2(double(h[i]) / double(n));
            return b;
        }

        struct estimator
        {
            static constexpr uint32_t depth = 8;
            static constexpr uint32_t window = 262144;
            // shortest distance the format allows
            static constexpr uint32_t floor = 32;
            static constexpr double cmd_bits = 6.0;
            // the hash chains over the committed cells / over the candidate cell
            static constexpr int committed_bits = 16;
            static constexpr int candidate_bits = 15;

            // the block: committed cells, then the candidate being priced
            std::vector<uint8_t> t;
            std::vector<int32_t> head_c, prev_c, head_s, prev_s;
            // head_s[h] is valid only if gen_s[h] == gen (saves a fill per call)
            std::vector<uint32_t> gen_s;
            uint32_t gen = 0;
            // lg[d] = log2(d), built once
            static const double* lg_table()
            {
                static const std::vector<double> v = []
                {
                    std::vector<double> x(uint64_t(window) + 1);
                    for (uint32_t d = 1; d <= window; ++d)
                        x[d] = std::log2(double(d));
                    return x;
                }();
                return v.data();
            }
            // repeat distance
            uint32_t r0 = 32; // after the committed cells
            uint32_t cr0 = 0; // after the last priced candidate

            static uint32_t hash(uint32_t v, int bits)
            {
                return (v * 2654435761u) >> (32 - bits);
            }

            void reset(uint32_t size)
            {
                t.assign(uint64_t(size) + 64, 0);
                head_c.assign(uint64_t(1) << committed_bits, -1);
                prev_c.assign(size, -1);
                r0 = 32;
            }

            // Bits of `t[p0, p1)` (already filled with the candidate), history = `t[0, p0)`
            // Stops early (returning >= `bound`) once the price reaches `bound`.
            double cost(uint32_t p0, uint32_t p1, double bound = 1e300)
            {
                if (head_s.size() != (uint64_t(1) << candidate_bits))
                    head_s.assign(uint64_t(1) << candidate_bits, -1),
                        gen_s.assign(uint64_t(1) << candidate_bits, 0), gen = 0;
                if (++gen == 0)
                    std::fill(gen_s.begin(), gen_s.end(), 0u), gen = 1;
                const double* const lg = lg_table();
                prev_s.assign(p1 - p0, -1);

                // literal price during the parse: order-0 of the candidate
                uint32_t hseg[256] = {};
                for (uint32_t i = p0; i < p1; ++i)
                    ++hseg[t[i]];
                const double lc = std::max(0.5, hist_bits(hseg, p1 - p0) / double(p1 - p0));

                uint32_t hl[256] = {}, hsub[256] = {};
                uint32_t nl = 0;
                double bits = 0;
                uint32_t a = r0;
                uint32_t run = 0;
                auto mlen = [&](uint32_t p, uint32_t q) -> uint32_t
                { return bt_finder::lcp8(&t[p], &t[q], p1 - p); };
                auto ins = [&](uint32_t p)
                {
                    if (p + 4 > p1)
                        return;
                    const uint32_t h = hash(loadu4(&t[p]), candidate_bits);
                    prev_s[p - p0] = gen_s[h] == gen ? head_s[h] : -1;
                    head_s[h] = int32_t(p), gen_s[h] = gen;
                };

                uint32_t p = p0;
                while (p < p1)
                {
                    uint32_t bl = 0, bd = 0;
                    double bg = 0;
                    int bk = -1; // 0 = repeat, 1 = new
                    if (a <= p)
                    {
                        const uint32_t l = mlen(p, p - a);
                        const double g = l * lc - cmd_bits;
                        if (l and g > bg)
                            bg = g, bl = l, bk = 0;
                    }
                    if (p + 4 <= p1)
                    {
                        const uint32_t v = loadu4(&t[p]);
                        uint32_t dep = depth;
                        const uint32_t hv = hash(v, candidate_bits);
                        for (int32_t q = gen_s[hv] == gen ? head_s[hv] : -1; q >= 0 and dep;
                             q = prev_s[uint32_t(q) - p0], --dep)
                        {
                            const uint32_t d = p - uint32_t(q);
                            if (d < floor or loadu4(&t[uint32_t(q)]) != v)
                                continue;
                            const uint32_t l = mlen(p, uint32_t(q));
                            const double g = l * lc - cmd_bits - 1.5 - lg[d];
                            if (l >= 3 and g > bg)
                                bg = g, bl = l, bd = d, bk = 1;
                        }
                        dep = depth;
                        for (int32_t q = head_c[hash(v, committed_bits)]; q >= 0 and dep;
                             q = prev_c[uint32_t(q)], --dep)
                        {
                            const uint32_t d = p - uint32_t(q);
                            if (d > window)
                                break;
                            if (d < floor or loadu4(&t[uint32_t(q)]) != v)
                                continue;
                            const uint32_t l = mlen(p, uint32_t(q));
                            const double g = l * lc - cmd_bits - 1.5 - lg[d];
                            if (l >= 3 and g > bg)
                                bg = g, bl = l, bd = d, bk = 1;
                        }
                    }
                    if (bk < 0)
                    {
                        ++hl[t[p]];
                        ++hsub[uint8_t(t[p] - (a <= p ? t[p - a] : 0))];
                        ++nl;
                        ++run;
                        ins(p);
                        ++p;
                        continue;
                    }
                    bits += cmd_bits + (run > 15 ? double((run - 1) / 15) * cmd_bits : 0);
                    if (bits >= bound)
                        return bits;
                    run = 0;
                    if (bk == 1)
                        bits += 1.5 + lg[bd], a = bd;
                    for (uint32_t k = 0; k < bl; ++k)
                        ins(p + k);
                    p += bl;
                }
                if (run)
                    bits += cmd_bits * double((run + 14) / 15);
                const double lb = std::min(hist_bits(hl, nl), hist_bits(hsub, nl));
                cr0 = a;
                return bits + lb;
            }

            // `t[p0, p1)` becomes history
            void commit(uint32_t p0, uint32_t p1)
            {
                for (uint32_t p = p0; p + 4 <= p1; ++p)
                {
                    const uint32_t h = hash(loadu4(&t[p]), committed_bits);
                    prev_c[p] = head_c[h];
                    head_c[h] = int32_t(p);
                }
            }
        };
    } // namespace stride_estimator
} // namespace misa77

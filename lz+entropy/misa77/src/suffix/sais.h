// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace misa77
{
    class sais_chan
    {
    public:
        static constexpr int32_t emp = -1;

        // prefetch offsets
        static constexpr uint32_t pd = 64; // gather
        static constexpr uint32_t pd2 = 5; // cursors

        // working memory
        std::vector<int32_t> s32;       // widened input
        std::vector<uint32_t> st_pool;  // (symbol << 1 | type)
        std::vector<uint32_t> cnt_pool; // frequency prefix sums
        std::vector<uint32_t> lms_pool; // lms indices
        std::vector<int32_t> nm;        // for naming, (n+1)/2 slots
        std::vector<uint32_t> bkt;      // bucket cursors, shared across levels

        template <typename T> void sa(const T* s, uint32_t n, int32_t* sa)
        {
            static_assert(std::is_integral_v<T>);
            static_assert(sizeof(T) <= sizeof(uint32_t));

            if (n == 0)
                return;
            if (s32.size() < n)
                s32.resize(n);

            uint32_t mn = *std::min_element(s, s + n);
            uint32_t mx = *std::max_element(s, s + n);
            for (uint32_t i = 0; i < n; i++)
                s32[i] = int32_t(s[i] - mn);

            core(s32.data(), sa, n, mx - mn + 1, 0, 0, 0);
        }

        // sa + inverse permutation
        void sa(const uint8_t* s, uint32_t n, int32_t* sa, int32_t* rank)
        {
            this->sa(s, n, sa);

            const int32_t* sap = sa;
            int32_t* rk = rank;
            for (uint32_t i = 0; i < n; i++)
            {
                if (i + pd < n)
                    __builtin_prefetch(rk + sap[i + pd], 1);
                rk[sap[i]] = int32_t(i);
            }
        }

    private:
        // s[i] in [0, k)
        // st, cnt, lms are given offsets so they're disjoint across recursion levels
        // bkt is shared
        void core(const int32_t* s,
                  int32_t* sa,
                  uint32_t n,
                  uint32_t k,
                  uint64_t st_off,
                  uint64_t cnt_off,
                  uint64_t lms_off)
        {
            if (n == 0)
                return;
            if (n == 1)
            {
                sa[0] = 0;
                return;
            }

            if (st_pool.size() < st_off + n)
                st_pool.resize(st_off + n);
            if (cnt_pool.size() < cnt_off + k + 1)
                cnt_pool.resize(cnt_off + k + 1);
            if (lms_pool.size() < lms_off + n / 2 + 1)
                lms_pool.resize(lms_off + n / 2 + 1);
            if (bkt.size() < uint64_t(k) + 1)
                bkt.resize(uint64_t(k) + 1);
            if (nm.size() < (uint64_t(n) + 1) / 2)
                nm.resize((uint64_t(n) + 1) / 2);

            uint32_t* st = st_pool.data() + st_off;
            uint32_t* cnt = cnt_pool.data() + cnt_off;
            uint32_t* lms = lms_pool.data() + lms_off;
            uint32_t* bk = bkt.data();

            // classify types, compute frequencies, identify lms
            uint32_t lcnt = 0;
            {
                std::fill(cnt, cnt + k + 1, 0);
                st[n - 1] = uint32_t(s[n - 1]) << 1;
                cnt[s[n - 1]]++;
                for (uint32_t i = n - 1, t = 0; i-- > 0;)
                {
                    cnt[s[i]]++;

                    const uint32_t tn = t;
                    t = (s[i] < s[i + 1] ? 1 : (s[i] > s[i + 1] ? 0 : t));
                    st[i] = (uint32_t(s[i]) << 1) | t;

                    // i + 1 is lms iff type[i+1] = S and type[i] = L
                    lms[lcnt] = i + 1;
                    lcnt += tn & (t ^ 1);
                }

                uint32_t acc = 0;
                for (uint32_t c = 0; c <= k; c++)
                {
                    const uint32_t x = cnt[c];
                    cnt[c] = acc;
                    acc += x;
                }
            }

            auto cursors = [&]() -> void { std::copy(cnt, cnt + k + 1, bk); };

            auto induce = [&]() -> void
            {
                // induce L, left to right
                cursors();
                sa[bk[st[n - 1] >> 1]++] = int32_t(n) - 1;
                for (uint32_t i = 0; i < n; i++)
                {
                    if (i + pd < n)
                        __builtin_prefetch(st + sa[i + pd] - 1);
                    if (i + pd2 < n)
                    {
                        const int32_t pj = sa[i + pd2];
                        const uint32_t pidx = pj > 0 ? uint32_t(pj) - 1 : 0;
                        __builtin_prefetch(bk + (st[pidx] >> 1));
                    }
                    const int32_t j = sa[i];
                    if (j > 0)
                    {
                        const uint32_t v = st[j - 1];
                        if (!(v & 1))
                            sa[bk[v >> 1]++] = j - 1;
                    }
                }
                // induce S, right to left
                cursors();
                for (uint32_t i = n; i-- > 0;)
                {
                    if (i >= pd)
                        __builtin_prefetch(st + sa[i - pd] - 1);
                    if (i >= pd2)
                    {
                        const int32_t pj = sa[i - pd2];
                        const uint32_t pidx = pj > 0 ? uint32_t(pj) - 1 : 0;
                        __builtin_prefetch(bk + (st[pidx] >> 1) + 1);
                    }
                    const int32_t j = sa[i];
                    if (j > 0)
                    {
                        const uint32_t v = st[j - 1];
                        if (v & 1)
                            sa[--bk[(v >> 1) + 1]] = j - 1;
                    }
                }
            };

            // stage 1: induce and sort the lms substrings
            std::fill(sa, sa + n, emp);
            if (lcnt == 0)
            {
                induce();
                return;
            }
            cursors();

            // place lms substrings into the SA
            for (uint32_t r = 0; r < lcnt; r++)
            {
                const uint32_t i = lms[r];
                sa[--bk[(st[i] >> 1) + 1]] = int32_t(i);
            }
            induce();

            // store sorted (up to lms boundaries) lms substrings
            {
                uint32_t w = 0;
                for (uint32_t i = 0; i < n; i++)
                {
                    if (i + pd < n)
                        __builtin_prefetch(st + sa[i + pd] - 1);
                    const int32_t j = sa[i];
                    if (j > 0 and (st[j] & 1) and !(st[j - 1] & 1))
                        sa[w++] = j;
                }
            }

            // label sorted lms substrings
            int32_t* names = nm.data();
            int32_t prev = sa[0];
            uint32_t label = 1;
            names[uint32_t(prev) >> 1] = 0;
            for (uint32_t i = 1; i < lcnt; i++)
            {
                if (i + 1 < lcnt)
                    __builtin_prefetch(st + sa[i + 1]);
                const int32_t cur = sa[i];
                bool eq = true;
                for (uint32_t d = 0;; d++)
                {
                    const uint32_t px = uint32_t(prev) + d, qx = uint32_t(cur) + d;
                    if (px == n or qx == n or st[px] != st[qx])
                    {
                        eq = false;
                        break;
                    }
                    if (d > 0)
                    {
                        // lms end
                        const bool pe = (st[px] & 1) and !(st[px - 1] & 1);
                        const bool qe = (st[qx] & 1) and !(st[qx - 1] & 1);
                        if (pe and qe)
                            break;
                        if (pe != qe)
                        {
                            eq = false;
                            break;
                        }
                    }
                }
                label += !eq;
                names[uint32_t(cur) >> 1] = int32_t(label - 1);
                prev = cur;
            }

            // reduced string at sa[n - lcnt...n), text order = lms[] reversed
            int32_t* rs = sa + (n - lcnt);
            for (uint32_t i = 0; i < lcnt; i++)
                rs[i] = names[lms[lcnt - 1 - i] >> 1];

            if (label < lcnt)
                core(rs, sa, lcnt, label, st_off + n, cnt_off + k + 1, lms_off + n / 2 + 1);
            else
                for (uint32_t i = 0; i < lcnt; i++)
                    sa[rs[i]] = int32_t(i);

            // recursion may have reallocated the pools
            st = st_pool.data() + st_off;
            cnt = cnt_pool.data() + cnt_off;
            lms = lms_pool.data() + lms_off;
            bk = bkt.data();

            // final induction
            std::fill(sa + lcnt, sa + n, emp);
            cursors();
            for (uint32_t i = lcnt; i-- > 0;)
            {
                const int32_t j = int32_t(lms[lcnt - 1 - uint32_t(sa[i])]);
                sa[i] = emp;
                sa[--bk[(st[j] >> 1) + 1]] = j;
            }
            induce();
        }
    };
} // namespace misa77

// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "format.h"
#include "suffix/sais.h"
#include "util.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace misa77
{
    namespace heavy_optimal_detail
    {
        using namespace heavy;

        // Output bytes per block. Tokens never cross a block, so it is also the parse unit.
        constexpr uint64_t block_size = uint64_t(1) << 18;
        static_assert(block_size <= max_block_output);

        // `compressed_size_bound` needs every block but the last to be this large.
        static_assert(block_size >= min_block_output);

        // F reaches 16 MiB, but we only use 1 MiB of it (not worth leaving L2).
        constexpr uint64_t far_window = uint64_t(1) << 20;
        static_assert(far_window <= max_dis(kind::F));

        // The suffix array is built over [history | segment], so sorting costs
        // (segment + history) / segment per input byte. Barely affects the matches found
        // (only those near a segment's end).
        constexpr uint64_t segment_size = uint64_t(1) << 20;
        static_assert(segment_size % block_size == 0);

        // Parse costs are in 1/256 byte.
        constexpr uint32_t unit = 256;
        // Decode time is directly proportional to token count, so a token costs an extra byte on
        // top of its own bytes. The +1 breaks ties between "one more token" and "one more byte"
        // towards fewer tokens.
        constexpr uint32_t token_cost = unit + 1;
        // Matches this close read bytes that were only just written (store forwarding stalls)
        constexpr uint64_t near_dis = 64;
        constexpr uint32_t near_cost = 2 * unit;

        // A block's parse costs <= 3 units + 1 per byte (all single literals) plus one token
        constexpr uint32_t cost_inf = std::numeric_limits<uint32_t>::max() / 2;
        static_assert(block_size * (3 * unit + 1) + 64 * unit < cost_inf);

        // Matches are measured up to `carry_max`. Inside a match of `>= reuse_min` the next
        // positions reuse it (shifted by one) instead of searching again. Reusing shorter ones
        // isn't free: a fresh search may find the same length at another distance, and the
        // repeats of the parse depend on which one is kept.
        constexpr uint64_t carry_max = uint64_t(1) << 16;
        constexpr uint64_t reuse_min = 1024;

        // New-distance kinds, largest window first
        constexpr uint32_t window_cnt = 3;
        constexpr kind window_kinds[window_cnt] = {kind::F, kind::N, kind::S};

        constexpr uint32_t window_of(kind k)
        {
            return k == kind::F ? 0 : k == kind::N ? 1 : 2;
        }

        constexpr uint64_t window_reach(uint32_t w)
        {
            return window_kinds[w] == kind::F ? far_window : max_dis(window_kinds[w]);
        }

        constexpr uint32_t short_len = uint32_t(max_small_len);
        constexpr uint32_t max_lit = uint32_t(max_small_lit);

        // The parse relies on only L codes and R codes with no literals being big.
        constexpr bool parse_shape_ok(const entry (&cb)[codebook_size])
        {
            for (const entry& e : cb)
            {
                if (e.k == kind::L)
                    continue;
                if (e.lit > max_lit)
                    return false;
                if (e.len > short_len and not(e.k == kind::R and e.lit == 0))
                    return false;
            }
            return true;
        }
        static_assert(parse_shape_ok(codebook_a) and parse_shape_ok(codebook_s));

        // insert (v, c) into the ascending list (vals, codes) of size n
        template <class V>
        constexpr void insert(V* vals, uint8_t* codes, uint32_t& n, uint32_t v, uint8_t c)
        {
            uint32_t j = n++;
            while (j > 0 and vals[j - 1] > v)
            {
                vals[j] = vals[j - 1], codes[j] = codes[j - 1];
                --j;
            }
            vals[j] = V(v), codes[j] = c;
        }

        // A codebook, rearranged for the parse
        struct parse_book
        {
            // L codes, ascending literal count
            uint32_t l_cnt = 0;
            uint32_t l_lit[codebook_size] = {};
            uint8_t l_code[codebook_size] = {};

            // R codes with `len <= short_len`, per lit, ascending len
            uint32_t r_cnt[max_lit + 1] = {};
            uint8_t r_len[max_lit + 1][short_len] = {};
            uint8_t r_code[max_lit + 1][short_len] = {};
            // bit l: some such code has lit l
            uint64_t r_lits = 0;
            bool has_r = false;

            // R codes with `len > short_len` (all have lit 0), ascending len
            uint32_t big_cnt = 0;
            uint32_t big_len[codebook_size] = {};
            uint8_t big_code[codebook_size] = {};

            // S / N / F codes per window: the distinct lengths (ascending), and per length the
            // set of literal counts that have a code with it (bit l = lit l)
            uint32_t m_cnt[window_cnt] = {};
            uint8_t m_len[window_cnt][short_len] = {};
            uint32_t lits[window_cnt][short_len + 2] = {};
            uint8_t code[window_cnt][max_lit + 1][short_len + 1] = {};
            bool has_window[window_cnt] = {};

            // Shift dominance (see the parse): the lits `l >= 1` of len m for which (l - 1, m + 1)
            // exists (better to take the latter as fewer literals), and lit 0 of len m if (0, m +
            // 1) exists
            uint32_t shifted[window_cnt][short_len + 1] = {};
            uint32_t shifted0[window_cnt][short_len + 1] = {};
            // per R lit `l >= 1`: bit m set if (l, m) and (l - 1, m + 1) both exist
            uint64_t r_shifted[max_lit + 1] = {};

            // largest distance the finder needs
            uint64_t history = 0;

            constexpr explicit parse_book(const entry (&cb)[codebook_size])
            {
                for (uint32_t c = 0; c < codebook_size; ++c)
                {
                    const entry& e = cb[c];
                    if (e.k == kind::L)
                        insert(l_lit, l_code, l_cnt, e.lit, uint8_t(c));
                    else if (e.k == kind::R and e.len > short_len)
                        insert(big_len, big_code, big_cnt, e.len, uint8_t(c));
                    else if (e.k == kind::R)
                    {
                        insert(r_len[e.lit], r_code[e.lit], r_cnt[e.lit], e.len, uint8_t(c));
                        r_lits |= uint64_t(1) << e.lit;
                        has_r = true;
                    }
                    else
                    {
                        const uint32_t w = window_of(e.k);
                        code[w][e.lit][e.len] = uint8_t(c);
                        lits[w][e.len] |= uint32_t(1) << e.lit;
                    }
                }
                for (uint32_t w = 0; w < window_cnt; ++w)
                {
                    for (uint32_t len = 1; len <= short_len; ++len)
                        if (lits[w][len] != 0)
                        {
                            m_len[w][m_cnt[w]++] = uint8_t(len);
                            shifted[w][len] = (lits[w][len + 1] << 1) & lits[w][len];
                            shifted0[w][len] = lits[w][len + 1] & lits[w][len] & 1;
                        }
                    has_window[w] = m_cnt[w] > 0;
                    if (has_window[w])
                        history = std::max(history, window_reach(w));
                }
                for (uint32_t l = 1; l <= max_lit; ++l)
                    for (uint32_t j = 0; j < r_cnt[l]; ++j)
                        for (uint32_t i = 0; i < r_cnt[l - 1]; ++i)
                            if (r_len[l - 1][i] == r_len[l][j] + 1)
                                r_shifted[l] |= uint64_t(1) << r_len[l][j];
            }
        };

        inline constexpr parse_book book_a{codebook_a};
        inline constexpr parse_book book_s{codebook_s};

        // bit i set iff `a[i] != b[i]`, for i < 48
        // PRECONDITION: 48 readable bytes at both
        [[gnu::always_inline]]
        inline uint64_t mismatch48(const uint8_t* a, const uint8_t* b)
        {
            constexpr uint64_t low7 = 0x7F7F7F7F7F7F7F7FULL;
            constexpr uint64_t gather = 0x0102040810204080ULL;
            uint64_t m = 0;
            for (uint32_t j = 0; j < 6; ++j)
            {
                const uint64_t x = loadu8(a + 8 * j) ^ loadu8(b + 8 * j);
                // top bit of each byte = that byte is non-zero
                const uint64_t hi = (((x & low7) + low7) | x) & ~low7;
                m |= ((hi >> 7) * gather >> 56) << (8 * j);
            }
            return m;
        }

        // 3-level bitset that maintains the ranks of the suffixes inside a match window
        struct liveset
        {
            std::vector<uint64_t> l0, l1, l2;

            // live iff `lo <= sa[r] <= hi`
            void build(const int32_t* sa, uint64_t m, int64_t lo, int64_t hi)
            {
                l0.assign((m + 63) / 64, 0);
                l1.assign((l0.size() + 63) / 64, 0);
                l2.assign((l1.size() + 63) / 64, 0);
                for (uint64_t r = 0; r < m; ++r)
                    if (sa[r] >= lo and sa[r] <= hi)
                        set(r);
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

        // One distance class of the finder
        struct window
        {
            uint64_t reach; // largest distance
            liveset live;   // ranks of the suffixes starting in [p - reach, p - min_dis]
            // a match of `carry` bytes at distance `carry_dis` is known at the next position
            uint64_t carry = 0;
            uint32_t carry_dis = 0;
            std::vector<uint8_t> len;  // per block position: longest match (<= short_len)
            std::vector<uint32_t> dis; // its distance
        };

        // Best known way to reach a block position = the token that ends there.
        // Packed so that a relaxation is two cmovs and not an unpredictable branch.
        struct arrival
        {
            // start (block-relative, bits 0-23) | code (24-31) | dis (32-63, 0 for L)
            uint64_t bits;

            static arrival of(uint64_t from, uint8_t code, uint32_t dis)
            {
                return {from | uint64_t(code) << 24 | uint64_t(dis) << 32};
            }
            uint64_t from() const
            {
                return bits & 0xFFFFFF;
            }
            uint8_t code() const
            {
                return uint8_t(bits >> 24);
            }
            uint32_t dis() const
            {
                return uint32_t(bits >> 32);
            }
        };
        static_assert(block_size < (uint64_t(1) << 24));

        struct token_rec
        {
            uint8_t code;
            uint32_t dis;
        };

        // Writes the blocks to `dst`, returns number of bytes written.
        // `T` and `B` must come from the same codebook.
        // PRECONDITION: `src_size > small_lim`
        template <class isa_lib, const decode_tables& T, const parse_book& B>
        uint64_t heavy_optimal_actual_cimpl(const uint8_t* __restrict src,
                                            uint64_t src_size,
                                            uint8_t* __restrict dst)
        {
            auto lcp_long = [&](const uint8_t* a, const uint8_t* b, uint64_t max_len) -> uint64_t
            {
                uint64_t i = 0;
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

            // Ensure that the last `literal_suffix` bytes will be literals
            const uint64_t match_end_limit = src_size - literal_suffix;

            uint8_t* out = dst;

            // Finder state
            sais_chan sorter_chan;
            std::vector<int32_t> sa, rank;

            // A window for each of S/N/F, used for match-finding
            window win[window_cnt];
            for (uint32_t w = 0; w < window_cnt; ++w)
                win[w].reach = window_reach(w);

            // sized once for the largest segment
            const uint64_t sa_cap = std::min(src_size, segment_size + B.history);
            sa.reserve(sa_cap);
            rank.reserve(sa_cap);

            // Parse state, per block position
            std::vector<uint32_t> cost;   // cheapest parse ending here
            std::vector<arrival> arr;     // its last token
            std::vector<uint32_t> rep_at; // its repeat distance
            std::vector<token_rec> block_tokens;

            // Repeat distance held by the decoder at the start of the next block (0 = none yet)
            uint32_t rep = 0;

            // Length of the match at `p` with distance `d`, capped. Consecutive positions with
            // the same `d` are answered from the previous answer (long runs).
            uint64_t memo_p = std::numeric_limits<uint64_t>::max();
            uint32_t memo_dis = 0, memo_len = 0, memo_cap = 0;
            auto rep_len = [&](uint64_t p, uint32_t d, uint32_t cap) -> uint32_t
            {
                uint32_t known = 0;
                if (memo_p + 1 == p and memo_dis == d and memo_len > 0)
                    known = std::min(memo_len - 1, cap);
                uint32_t len = known;
                // the run at `p - 1` hit its cap, so it may go on
                if (known == 0 or memo_len >= memo_cap)
                    len += uint32_t(lcp_long(src + p + known, src + p + known - d, cap - known));
                memo_p = p, memo_dis = d, memo_len = len, memo_cap = cap;
                return len;
            };

            // Each segment handles several blocks
            for (uint64_t ss = 0; ss < src_size; ss += segment_size)
            {
                const uint64_t se = std::min(ss + segment_size, src_size);

                // Suffix array and rank over [history | segment]
                const uint64_t seg0 = ss > B.history ? ss - B.history : 0;
                const uint32_t m = uint32_t(se - seg0);
                sa.resize(m);
                rank.resize(m);
                sorter_chan.sa(src + seg0, m, sa.data(), rank.data());

                for (uint32_t k = 0; k < window_cnt; ++k)
                    if (B.has_window[k])
                    {
                        window& w = win[k];
                        w.live.build(sa.data(),
                                     m,
                                     int64_t(ss) - int64_t(w.reach) - int64_t(seg0),
                                     int64_t(ss) - int64_t(min_dis) - int64_t(seg0));
                        w.carry = 0;
                    }

                for (uint64_t bs = ss; bs < se; bs += block_size)
                {
                    const uint64_t be = std::min(bs + block_size, se);
                    const uint64_t blen = be - bs;

                    // Tokens must end within the block AND BEFORE the raw suffix
                    const uint64_t hard = std::min(be, match_end_limit);
                    const uint64_t lim = hard > bs ? hard - bs : 0;

                    // Longest match per window at every block position, capped at `short_len`.
                    // It is at one of the two rank-neighbours of `p` among the live suffixes.
                    for (window& w : win)
                    {
                        w.len.assign(blen, 0);
                        w.dis.assign(blen, 0);
                    }
                    for (uint64_t p = bs; p < be; ++p)
                    {
                        // Slide the source windows [p - reach, p - min_dis]
                        if (p > ss)
                            for (uint32_t k = 0; k < window_cnt; ++k)
                                if (B.has_window[k])
                                {
                                    window& w = win[k];
                                    if (p >= seg0 + min_dis)
                                        w.live.set(uint64_t(rank[p - min_dis - seg0]));
                                    if (p >= seg0 + w.reach + 1)
                                        w.live.clear(uint64_t(rank[p - w.reach - 1 - seg0]));
                                }

                        if (p >= hard)
                            continue;
                        const uint64_t limit = std::min<uint64_t>(short_len, hard - p);

                        // All neighbours first, then their positions, then the compares (MLP).
                        const uint64_t r = uint64_t(rank[p - seg0]);
                        bool search[window_cnt];
                        int64_t q[2 * window_cnt];
                        for (uint32_t k = 0; k < window_cnt; ++k)
                        {
                            search[k] = B.has_window[k] and
                                        win[k].carry < std::max<uint64_t>(limit, reuse_min);
                            q[2 * k] = search[k] ? win[k].live.prev(r) : -1;
                            q[2 * k + 1] = search[k] ? win[k].live.next(r) : -1;
                        }
                        uint64_t pos[2 * window_cnt];
                        for (uint32_t j = 0; j < 2 * window_cnt; ++j)
                            pos[j] = q[j] >= 0 ? seg0 + uint64_t(sa[uint64_t(q[j])]) : 0;
                        uint64_t len[2 * window_cnt];
                        const uint64_t cap = std::min(match_end_limit - p, carry_max);
                        for (uint32_t j = 0; j < 2 * window_cnt; ++j)
                        {
                            // windows often share a neighbour
                            uint32_t t = 0;
                            while (t < j and pos[t] != pos[j])
                                ++t;
                            if (q[j] < 0)
                                len[j] = 0;
                            else if (t < j)
                                len[j] = len[t];
                            else
                                len[j] = lcp_long(src + p, src + pos[j], cap);
                        }

                        const uint64_t i = p - bs;
                        for (uint32_t k = 0; k < window_cnt; ++k)
                        {
                            window& w = win[k];
                            if (search[k])
                            {
                                w.carry = 0, w.carry_dis = 0;
                                for (uint32_t j = 2 * k; j < 2 * k + 2; ++j)
                                {
                                    // on equal length, take the closer one
                                    const uint32_t d = uint32_t(p - pos[j]);
                                    if (len[j] > w.carry or
                                        (len[j] == w.carry and len[j] > 0 and d < w.carry_dis))
                                        w.carry = len[j], w.carry_dis = d;
                                }
                            }
                            if (w.carry >= 2)
                            {
                                w.len[i] = uint8_t(std::min(w.carry, limit));
                                w.dis[i] = w.carry_dis;
                            }
                            w.carry -= uint64_t(w.carry > 0);
                        }
                    }

                    // Shortest path over the block positions [0, lim], where an edge is a token.
                    // Exact, except for some suboptimality on repeat tokens.
                    //
                    // Position `p` is visited once every token ending at it has been relaxed.
                    // S / N / F tokens are relaxed when `p` is where their match starts: the
                    // token itself starts at `p - lit`, so each length takes the cheapest lit.
                    // L and R tokens are relaxed when `p` is where the token starts: R needs
                    // the repeat distance there, and L has no match.
                    //
                    // Shift dominance (exact): a code (l, m) whose match also covers the byte
                    // before it is the code (l - 1, m + 1) one position earlier from the same
                    // start, one literal cheaper, so it is skipped if that code exists.
                    cost.assign(lim + 1, cost_inf);
                    arr.resize(lim + 1);
                    rep_at.resize(lim + 1);
                    cost[0] = 0;
                    arr[0] = arrival::of(0, 0, 0);
                    memo_p = std::numeric_limits<uint64_t>::max();

                    // On equal cost the later relaxation wins (which decides the repeat distance
                    // that the position keeps; either choice is a wash on average).
                    auto relax = [&](uint64_t x, uint32_t c, arrival a)
                    {
                        const bool take = c <= cost[x];
                        cost[x] = take ? c : cost[x];
                        arr[x].bits = take ? a.bits : arr[x].bits;
                    };

                    for (uint64_t p = 0; p < lim; ++p)
                    {
                        const uint64_t room = lim - p;
                        const uint32_t cp = cost[p];

                        // Repeat distance after the best path to `p`: a match sets it, an L token
                        // keeps the one at its start.
                        uint32_t r = rep;
                        if (p > 0)
                            r = arr[p].dis() != 0 ? arr[p].dis() : rep_at[arr[p].from()];
                        rep_at[p] = r;

                        // New distances, matched at `p`.
                        // pre[a] = cheapest start among lits 0..a (a token with l literals starts
                        // at `p - l` and pays for them), pre_lit[a] = its lit
                        uint32_t pre[max_lit + 1];
                        uint32_t pre_lit[max_lit + 1];
                        bool gathered = false;

                        // smallest window (cheapest offset) first
                        for (uint32_t w = window_cnt; w-- > 0;)
                        {
                            if (not B.has_window[w])
                                continue;
                            const uint32_t ml = uint32_t(std::min<uint64_t>(win[w].len[p], room));
                            if (ml < B.m_len[w][0])
                                continue;

                            if (not gathered)
                            {
                                for (uint32_t l = 0; l <= max_lit; ++l)
                                {
                                    const uint32_t c = l <= p ? cost[p - l] + l * unit : cost_inf;
                                    const bool take = l == 0 or c < pre[l - 1];
                                    pre[l] = take ? c : pre[l - 1];
                                    pre_lit[l] = take ? l : pre_lit[l - 1];
                                }
                                gathered = true;
                            }

                            const uint32_t d = win[w].dis[p];

                            // A smaller window with the same match has the same codes one offset
                            // byte cheaper, so only the (lit, len) it lacks are worth relaxing.
                            uint32_t same = 0;
                            for (uint32_t v = w + 1; v < window_cnt; ++v)
                                if (B.has_window[v] and win[v].dis[p] == d and
                                    win[v].len[p] == win[w].len[p])
                                    same |= uint32_t(1) << v;

                            const uint32_t extra = unit + token_cost +
                                                   offset_bytes(window_kinds[w]) * unit +
                                                   (d < near_dis ? near_cost : 0);

                            // for shift dominance
                            uint32_t prev_len = 0;
                            if (p > 0 and win[w].dis[p - 1] == d)
                                prev_len =
                                    std::min<uint32_t>(win[w].len[p - 1], uint32_t(room) + 1);
                            const bool prev_cheaper = p > 0 and cost[p - 1] <= cp;

                            for (uint32_t j = 0; j < B.m_cnt[w] and B.m_len[w][j] <= ml; ++j)
                            {
                                const uint32_t len = B.m_len[w][j];
                                uint32_t mask = B.lits[w][len];
                                for (uint32_t v = w + 1; v < window_cnt; ++v)
                                    if ((same >> v) & 1)
                                        mask &= ~B.lits[v][len];
                                if (len + 1 <= prev_len)
                                    mask &= ~(B.shifted[w][len] |
                                              (prev_cheaper ? B.shifted0[w][len] : 0));
                                if (mask == 0)
                                    continue;

                                // lits 0..a-1 all have this len, so one prefix-min lookup
                                const uint32_t a = uint32_t(__builtin_ctz(~mask));
                                uint32_t best = a > 0 ? pre[a - 1] : cost_inf;
                                uint32_t best_lit = a > 0 ? pre_lit[a - 1] : 0;
                                for (uint32_t s = (mask >> a) << a; s != 0; s &= s - 1)
                                {
                                    const uint32_t l = uint32_t(__builtin_ctz(s));
                                    const uint32_t c = l <= p ? cost[p - l] + l * unit : cost_inf;
                                    if (c < best)
                                        best = c, best_lit = l;
                                }
                                if (best < cost_inf)
                                    relax(p + len,
                                          best + extra,
                                          arrival::of(p - best_lit, B.code[w][best_lit][len], d));
                            }
                        }

                        // control byte
                        const uint32_t base = cp + unit + token_cost;

                        // Literal runs
                        for (uint32_t j = 0; j < B.l_cnt and B.l_lit[j] <= room; ++j)
                            relax(p + B.l_lit[j],
                                  base + B.l_lit[j] * unit,
                                  arrival::of(p, B.l_code[j], 0));

                        // Repeats
                        if (r == 0 or not B.has_r)
                            continue;

                        // one compare gives the run of matching bytes at every `p + l`
                        const uint64_t ap = bs + p;
                        uint64_t miss = 0;
                        if (ap + 48 <= src_size)
                            miss = mismatch48(src + ap, src + ap - r);
                        else
                        {
                            for (uint64_t t = 0; t < 48; ++t)
                                if (ap + t >= src_size or src[ap + t] != src[ap + t - r])
                                    miss |= uint64_t(1) << t;
                        }

                        const uint32_t rep_base = base + (r < near_dis ? near_cost : 0);

                        // only lits whose first match byte matches can have a repeat
                        const uint64_t room_lits =
                            room > max_lit ? ~uint64_t(0) : (uint64_t(1) << room) - 1;
                        for (uint64_t cand = ~miss & B.r_lits & room_lits; cand != 0;
                             cand &= cand - 1)
                        {
                            const uint32_t l = uint32_t(__builtin_ctzll(cand));
                            const uint64_t run =
                                (miss >> l) ? uint64_t(__builtin_ctzll(miss >> l)) : 48 - l;
                            const uint64_t ml = std::min<uint64_t>(run, room - l);
                            const uint32_t c = rep_base + l * unit;

                            // shift dominance
                            uint64_t dom = 0;
                            if (l > 0 and ((miss >> (l - 1)) & 1) == 0)
                                dom = B.r_shifted[l];

                            for (uint32_t j = 0; j < B.r_cnt[l] and B.r_len[l][j] <= ml; ++j)
                                if (((dom >> B.r_len[l][j]) & 1) == 0)
                                    relax(p + l + B.r_len[l][j],
                                          c,
                                          arrival::of(p, B.r_code[l][j], r));
                        }

                        // big repeats, only if all 48 bytes matched
                        if (B.big_cnt > 0 and (miss & ((uint64_t(1) << 48) - 1)) == 0 and
                            room >= B.big_len[0])
                        {
                            const uint32_t cap =
                                uint32_t(std::min<uint64_t>(room, B.big_len[B.big_cnt - 1]));
                            const uint32_t ml = rep_len(ap, r, cap);
                            for (uint32_t j = 0; j < B.big_cnt and B.big_len[j] <= ml; ++j)
                                relax(p + B.big_len[j], rep_base, arrival::of(p, B.big_code[j], r));
                        }
                    }

                    // Any reachable position can end the tokens, the rest of the block is
                    // trailing literals.
                    uint64_t commit = 0;
                    uint64_t best_total = std::numeric_limits<uint64_t>::max();
                    for (uint64_t x = 0; x <= lim; ++x)
                        if (cost[x] < cost_inf and
                            uint64_t(cost[x]) + (blen - x) * unit < best_total)
                        {
                            best_total = uint64_t(cost[x]) + (blen - x) * unit;
                            commit = x;
                        }

                    // backtrack commit, then emit forward...
                    block_tokens.clear();
                    uint64_t off_cnt = 0;
                    for (uint64_t x = commit; x > 0; x = arr[x].from())
                    {
                        block_tokens.push_back({arr[x].code(), arr[x].dis()});
                        off_cnt += T.nb[arr[x].code()] & ~uint64_t(big_flag);
                    }
                    uint64_t tok_cnt = block_tokens.size();

                    // every byte that isn't covered by a match
                    uint64_t lit_cnt = blen;
                    for (const token_rec& t : block_tokens)
                        lit_cnt -= T.full_len[t.code];

                    // `compressed_size_bound` needs: streams <= output. Else, a block of raw
                    // literals. The decoder then never sees these tokens, so `rep` must not move.
                    const bool raw = tok_cnt + off_cnt + lit_cnt > blen;
                    if (raw)
                    {
                        block_tokens.clear();
                        tok_cnt = 0, off_cnt = 0, lit_cnt = blen;
                    }

                    auto store_cnt = [&](uint64_t pos, uint64_t cnt)
                    {
                        storeu2(out + pos, uint16_t(cnt));
                        out[pos + 2] = uint8_t(cnt >> 16);
                    };
                    store_cnt(tok_cnt_pos, tok_cnt);
                    store_cnt(off_cnt_pos, off_cnt);
                    store_cnt(lit_cnt_pos, lit_cnt);

                    uint8_t* cp = out + block_header_size;
                    uint8_t* op = cp + tok_cnt;
                    uint8_t* lp = op + off_cnt;
                    out = lp + lit_cnt;

                    uint64_t pos = bs;
                    for (uint64_t k = block_tokens.size(); k-- > 0;)
                    {
                        const token_rec& t = block_tokens[k];
                        const uint64_t lit = T.full_lit[t.code];
                        const uint64_t ob = T.nb[t.code] & ~uint64_t(big_flag);

                        *cp++ = t.code;
                        memcpy(lp, src + pos, lit);
                        lp += lit;
                        if (ob > 0)
                        {
                            const uint64_t v = t.dis - min_dis;
                            for (uint64_t b = 0; b < ob; ++b)
                                op[b] = uint8_t(v >> (8 * b));
                            op += ob;
                            rep = t.dis;
                        }
                        pos += lit + T.full_len[t.code];
                    }

                    // trailing literals
                    memcpy(lp, src + pos, be - pos);
                }
            }

            return uint64_t(out - dst);
        }
    } // namespace heavy_optimal_detail

    // Returns number of bytes written to `dst`, and 0 on failure.
    // `isa_lib` is ISA-dependent.
    template <class isa_lib>
    uint64_t heavy_optimal_cimpl(const uint8_t* __restrict src,
                                 uint64_t src_size,
                                 uint8_t* __restrict dst,
                                 uint64_t dst_cap)
    {
        using namespace heavy;

        if (compressed_size_bound(src_size) > dst_cap)
            return 0;

        const uint8_t flags = src_size <= codebook_s_lim ? flags_s : flags_a;
        storeu8(dst, src_size | (uint64_t(flags) << flags_shift));

        if (src_size <= small_lim)
        {
            if (src_size != 0)
                memcpy(dst + header_size, src, src_size);
            return header_size + src_size;
        }

        using namespace heavy_optimal_detail;

        if (flags == flags_s)
            return header_size + heavy_optimal_actual_cimpl<isa_lib, tables_s, book_s>(
                                     src, src_size, dst + header_size);
        return header_size + heavy_optimal_actual_cimpl<isa_lib, tables_a, book_a>(
                                 src, src_size, dst + header_size);
    }
} // namespace misa77

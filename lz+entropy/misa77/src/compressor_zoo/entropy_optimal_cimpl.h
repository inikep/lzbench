// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "entropy/bt_finder.h"
#include "entropy/huffman_encoder.h"
#include "entropy/stride_estimator.h"
#include "format.h"
#include "util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace misa77
{
    namespace entropy_optimal_detail
    {
        using namespace entropy;
        using namespace entropy::huffman;
        using bt_finder::match;

        // The parse prices a command with bits + lambda * modelled decode cycles
        // lambda is in bits per 1000 cycles.
        inline constexpr uint32_t default_lambda = 48;

        // Prices are in 1/64 bit
        inline constexpr uint32_t unit = 64;
        inline constexpr uint32_t cost_inf = 0x3FFFFFFF;

        // modelled decode cycles
        inline constexpr double cyc_cmd = 8.0;
        inline constexpr double cyc_huf = 1.15; // per Huffman symbol
        inline constexpr double huf_setup = 1800;
        inline constexpr double cyc_merge = 0.5; // per new offset (planes + flags -> distance)
        inline constexpr double cyc_far = 12.0;
        inline constexpr uint32_t far_dis = uint32_t(1) << 18; // cache misses beyond this
        // first parse of a frame: per command, the Huffman decode it will cost
        inline constexpr double cyc_sym0 = 1.3;

        // bits of a symbol a Huffman table does not have yet
        inline constexpr double unseen_bits = 12.0;
        // first parse: base bits of an N command
        inline constexpr double n_base_bits = 4.5;

        // parse search
        inline constexpr uint32_t arrivals = 3;   // per position, distinct by repeat distance
        inline constexpr uint32_t nice_len = 256; // a match this long is taken outright
        inline constexpr uint32_t all_lens =
            32; // every shorter length is priced, beyond: the longest

        // The encoder's window. The format allows `max_dis`, but 2^23 - 2 fills the finder's ring
        // of 2^23 slots exactly.
        inline constexpr uint32_t enc_window = (uint32_t(1) << 23) - 2;
        static_assert(enc_window <= max_dis);

        // The finder's positions are 32-bit (to better fit in memory), so it sees the input through
        // views: `view_span` bytes of blocks, after `view_history` bytes that only refill its
        // window.
        inline constexpr uint64_t view_span = uint64_t(1) << 31;
        inline constexpr uint64_t view_history = uint64_t(1) << 23;
        static_assert(view_span % block_size == 0 and view_history % block_size == 0);
        static_assert(view_history >= enc_window);
        static_assert(view_history + view_span + block_size <= UINT32_MAX);

        // A new distance needs a match of `>= 3` (format), 4 if it takes 3 offset bytes
        inline constexpr uint32_t n3_min_len = 4;

        // Price tables are exact up to these, then extended with the longest unit
        inline constexpr uint32_t run_tab = 30;
        inline constexpr uint32_t len_tab = 128;
        static_assert(run_tab >= 2 * max_lit and len_tab >= 2 * max_match_len);

        // A block whose coded form saves less than this fraction is stored (we don't wanna trash
        // decode in incompressible regions)
        inline constexpr double store_gain = 0.05;

        // Each block's commands come from rounds of (prices from the last round's streams ->
        // parse). A round's prior says what an unseen command costs: `flat` = x bits, `factored`
        // = an estimate from the seen commands' kinds, literals and lengths + x bits. Only the
        // last round keeps more than one arrival.
        enum class prior_kind
        {
            flat,
            factored,
        };
        struct prior
        {
            prior_kind t;
            double x;
        };
        inline constexpr prior rounds[] = {{prior_kind::factored, 1}, {prior_kind::flat, 12}};
        inline constexpr uint32_t round_cnt = sizeof(rounds) / sizeof(rounds[0]);

        // DELTA: per cell, the stride whose estimate beats plain by this fraction
        inline constexpr double delta_gain = 0.15;
        inline constexpr uint32_t strides[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32};
        // only strides whose order-0 entropy is at most this much worse (bits per byte) are tried
        inline constexpr double stride_margin = 0.5;
        // plain bytes before the block that the block's own finder sees
        inline constexpr uint32_t local_history = 32;
        // and after it
        inline constexpr uint32_t local_lookahead = 1024;

        // parse kinds: N1..N3 share the N commands and pay for their width in the offset planes
        // and flags
        enum pkind : uint32_t
        {
            k_r0,
            k_n1,
            k_n2,
            k_n3,
            kind_cnt,
        };
        inline constexpr uint32_t min_len[kind_cnt] = {1, 3, 3, n3_min_len};
        inline constexpr uint8_t k_lit = 255;

        // bytes of a stored offset `v`
        constexpr uint32_t offset_bytes(uint32_t v)
        {
            return v < 256 ? 1 : v < 65536 ? 2 : 3;
        }

        constexpr uint32_t kind_of(uint32_t d)
        {
            return k_n1 + offset_bytes(d - uint32_t(min_dis)) - 1;
        }

        // The codebook with N expanded into N1..N3 (symbol order = cmd order)
        struct pbook
        {
            static constexpr uint32_t max_cnt = 3 * codebook_size;
            uint32_t cnt = 0;
            uint8_t lit[max_cnt] = {}, len[max_cnt] = {}, kind[max_cnt] = {}, cmd[max_cnt] = {};
            int16_t sym[kind_cnt][max_lit + 1][max_match_len + 1] = {};

            constexpr pbook()
            {
                for (auto& a : sym)
                    for (auto& b : a)
                        for (auto& x : b)
                            x = -1;
                for (uint32_t c = 0; c < codebook_size; ++c)
                {
                    const entry& e = codebook[c];
                    const uint32_t k0 = e.k == kind::R ? k_r0 : k_n1;
                    const uint32_t k1 = e.k == kind::R ? k_r0 : k_n3;
                    for (uint32_t k = k0; k <= k1; ++k)
                    {
                        lit[cnt] = uint8_t(e.lit), len[cnt] = uint8_t(e.len);
                        kind[cnt] = uint8_t(k), cmd[cnt] = uint8_t(c);
                        sym[k][e.lit][e.len] = int16_t(cnt);
                        ++cnt;
                    }
                }
            }
        };
        inline constexpr pbook pb{};
        // `codebook_wellformed` guarantees every literal-only and every continuation command
        inline constexpr uint32_t sym_lit15 = uint32_t(pb.sym[k_r0][max_lit][0]);
        inline constexpr uint32_t sym_cont = uint32_t(pb.sym[k_r0][0][max_match_len]);

        // Command bytes, literals, offset planes and flags of one block
        struct streams
        {
            std::vector<uint8_t> cmd;
            std::vector<uint8_t> lit[2]; // plain, SUB
            std::vector<uint8_t> o1, o2[2], o3[3];
            std::vector<uint8_t> f1, f2; // one byte (0 / 1) per flag
            uint32_t tok_hist[kind_cnt][len_tab + 1];

            void clear()
            {
                cmd.clear(), lit[0].clear(), lit[1].clear(), o1.clear(), f1.clear(), f2.clear();
                for (auto& v : o2)
                    v.clear();
                for (auto& v : o3)
                    v.clear();
                memset(tok_hist, 0, sizeof tok_hist);
            }
        };

        inline void hist256(const std::vector<uint8_t>& v, uint32_t* h)
        {
            memset(h, 0, 256 * sizeof(uint32_t));
            for (uint8_t x : v)
                h[x]++;
        }

        inline void pack_flags(const std::vector<uint8_t>& f, std::vector<uint8_t>& out)
        {
            out.assign((f.size() + 7) / 8, 0);
            for (uint64_t i = 0; i < f.size(); ++i)
                out[i >> 3] |= uint8_t(f[i] << (i & 7));
        }

        // Shortest streams are always raw
        inline constexpr uint64_t huf_min = 32;

        // Modelled coding of a byte stream: raw or Huffman, whichever has the lower bytes +
        // lambda * cycles
        struct choice
        {
            bool huf;
            double j;
            uint8_t len[256];
            double bytes;
        };

        // Prices of the current round. Each `cmd[s]` / byte entry is bits * unit + cycles.
        struct costs
        {
            uint32_t cmd[pbook::max_cnt];
            uint32_t lit[256];
            bool sub = false;
            uint32_t o1[256], o2[2][256], o3[3][256];
            uint32_t f1[2], f2[2]; // per flag value
            uint32_t far = 0, merge = 0;
            // cheapest literal-only commands for a run / continuations for a length, and the
            // first command of that split
            uint32_t L[run_tab + 1], C[len_tab + 1];
            uint8_t Lj[run_tab + 1], Cj[len_tab + 1];
            // cheapest commands for (kind, match length, literal run), and the symbol carrying
            // the match
            uint32_t M[kind_cnt][len_tab + 1][run_tab + 1];
            uint16_t Ms[kind_cnt][len_tab + 1][run_tab + 1];
            // mean extra price of a match after a literal run, over this round's tokens
            uint32_t Ap[run_tab + 1];
            uint32_t c15 = 0, cmaxc = 0;
            double w[kind_cnt][len_tab + 1];

            void derive()
            {
                L[0] = 0, Lj[0] = 0;
                for (uint32_t r = 1; r <= run_tab; ++r)
                {
                    L[r] = cost_inf;
                    for (uint32_t j = 1; j <= std::min(r, uint32_t(max_lit)); ++j)
                        if (pb.sym[k_r0][j][0] >= 0 and L[r - j] + cmd[pb.sym[k_r0][j][0]] < L[r])
                            L[r] = L[r - j] + cmd[pb.sym[k_r0][j][0]], Lj[r] = uint8_t(j);
                }
                C[0] = 0, Cj[0] = 0;
                for (uint32_t m = 1; m <= len_tab; ++m)
                {
                    C[m] = cost_inf;
                    for (uint32_t j = 1; j <= std::min(m, uint32_t(max_match_len)); ++j)
                        if (pb.sym[k_r0][0][j] >= 0 and C[m - j] + cmd[pb.sym[k_r0][0][j]] < C[m])
                            C[m] = C[m - j] + cmd[pb.sym[k_r0][0][j]], Cj[m] = uint8_t(j);
                }
                c15 = cmd[sym_lit15], cmaxc = cmd[sym_cont];
                for (uint32_t k = 0; k < kind_cnt; ++k)
                    for (uint32_t len = 0; len <= len_tab; ++len)
                        for (uint32_t r = 0; r <= run_tab; ++r)
                            M[k][len][r] = cost_inf, Ms[k][len][r] = 0;
                // T[k][a][len] = cheapest command of kind k with a literals + its continuations
                // covering a match of len, Ts = that command (the first on ties)
                static_assert(run_tab >= max_lit);
                uint32_t T[kind_cnt][max_lit + 1][len_tab + 1];
                uint16_t Ts[kind_cnt][max_lit + 1][len_tab + 1];
                for (auto& x : T)
                    for (auto& y : x)
                        for (auto& z : y)
                            z = cost_inf;
                for (uint32_t s = 0; s < pb.cnt; ++s)
                {
                    const uint32_t a = pb.lit[s], m = pb.len[s], k = pb.kind[s];
                    if (m == 0)
                        continue;
                    const uint32_t c = cmd[s];
                    for (uint32_t len = m; len <= len_tab; ++len)
                    {
                        const uint32_t x = c + C[len - m];
                        if (x < T[k][a][len])
                            T[k][a][len] = x, Ts[k][a][len] = uint16_t(s);
                    }
                }
                for (uint32_t k = 0; k < kind_cnt; ++k)
                    for (uint32_t len = 1; len <= len_tab; ++len)
                        for (uint32_t r = 0; r <= run_tab; ++r)
                        {
                            uint32_t best = cost_inf, bs = 0;
                            for (uint32_t a = 0, e = std::min(r, uint32_t(max_lit)); a <= e; ++a)
                            {
                                if (T[k][a][len] >= cost_inf)
                                    continue;
                                const uint32_t x = T[k][a][len] + L[r - a], s = Ts[k][a][len];
                                if (x < best or (x == best and s < bs))
                                    best = x, bs = s;
                            }
                            if (best < cost_inf)
                                M[k][len][r] = best, Ms[k][len][r] = uint16_t(bs);
                        }
                double tw = 0;
                for (uint32_t k = 0; k < kind_cnt; ++k)
                    for (uint32_t len = 1; len <= len_tab; ++len)
                        if (M[k][len][0] < cost_inf)
                            tw += w[k][len];
                for (uint32_t r = 0; r <= run_tab; ++r)
                {
                    double a = 0;
                    if (tw > 0)
                        for (uint32_t k = 0; k < kind_cnt; ++k)
                            for (uint32_t len = 1; len <= len_tab; ++len)
                                if (w[k][len] > 0 and M[k][len][0] < cost_inf)
                                    a += w[k][len] * (double(M[k][len][r]) - double(M[k][len][0]));
                    Ap[r] = tw > 0 ? uint32_t(std::max(0.0, a / tw)) : 0;
                }
            }

            // literal-only commands for a run of `x`
            uint32_t Lc(uint32_t x) const
            {
                if (x <= run_tab)
                    return L[x];
                const uint32_t q = (x - (max_lit + 1)) / max_lit;
                return L[x - max_lit * q] + q * c15;
            }

            // continuations for `y` more match bytes
            uint32_t Cc(uint32_t y) const
            {
                if (y <= len_tab)
                    return C[y];
                const uint32_t q = (y - (len_tab - max_match_len) - 1) / max_match_len;
                return C[y - max_match_len * q] + q * cmaxc;
            }

            uint32_t mcost(uint32_t k, uint32_t len, uint32_t r) const
            {
                if (len <= len_tab and r <= run_tab)
                    return M[k][len][r];
                const uint32_t lt = std::min(len, len_tab), rt = std::min(r, run_tab);
                if (M[k][lt][rt] >= cost_inf)
                    return cost_inf;
                const uint32_t s = Ms[k][lt][rt];
                return cmd[s] + Lc(r - pb.lit[s]) + Cc(len - pb.len[s]);
            }

            uint32_t apend(uint32_t r) const
            {
                if (r <= run_tab)
                    return Ap[r];
                const uint32_t q = (r - (max_lit + 1)) / max_lit;
                return Ap[r - max_lit * q] + q * c15;
            }

            uint32_t offcost(uint32_t k, uint32_t d) const
            {
                d -= uint32_t(min_dis);
                if (k == k_n1)
                    return o1[d] + f1[0] + merge;
                if (k == k_n2)
                    return o2[0][d & 255] + o2[1][d >> 8] + f1[1] + f2[0] + merge;
                return o3[0][d & 255] + o3[1][(d >> 8) & 255] + o3[2][d >> 16] + f1[1] + f2[1] +
                       merge;
            }
        };

        // Best ways to reach a block position
        struct arrival
        {
            uint32_t key, cost; // key = cost + the expected price of a pending literal run
            uint32_t r0;        // repeat distance
            uint32_t litrun;
            uint32_t mlen, mdist;
            uint8_t slot, kind; // previous arrival
        };

        struct token
        {
            uint32_t lits, len, dist;
            uint8_t kind;
        };

        template <class isa_lib>
        uint32_t lcp(const uint8_t* a, const uint8_t* b, uint32_t max_match_len)
        {
            uint32_t i = 0;
            while (i + vector_width <= max_match_len)
            {
                const uint32_t l =
                    uint32_t(isa_lib::lcp(isa_lib::loadvec(a + i), isa_lib::loadvec(b + i)));
                i += l;
                if (l < vector_width)
                    return i;
            }
            while (i < max_match_len and a[i] == b[i])
                ++i;
            return i;
        }

        template <class isa_lib> struct encoder
        {
            double lambda;
            double lamc; // cycles -> price
            costs co;
            bool have_costs = false;

            std::vector<arrival> A;
            std::vector<uint8_t> cnt;
            std::vector<uint32_t> wc; // worst key at a full position (else max)
            uint32_t cur_k = 1;
            std::vector<token> toks;
            streams st;
            std::vector<uint8_t> tmp;
            // the last round's streams of the previous block, per view
            streams carry_plain, carry_delta;
            bool have_delta = false;

            // DELTA view of the current block: plain history, the block with its proposed cells
            // transformed. `bprop` = stride per cell, `lb` = the block's local segment, `wseg` =
            // view bytes [wlo, be + 64)
            stride_estimator::estimator de;
            std::vector<uint8_t> bprop, lb, wseg, cur_strides;
            std::vector<uint32_t> npc; // first transformed position at or after each cell
            uint32_t wlo = 0;
            bool cur_delta = false;
            bt_finder::finder lx, fp;
            std::vector<uint32_t> moff_plain, moff_delta;
            std::vector<match> mm_plain, mm_delta;
            match tmp_l[bt_finder::max_frontier], tmp_f[bt_finder::max_frontier],
                tmp_s[bt_finder::max_frontier];
            std::vector<uint8_t> cbuf, wbuf;

            explicit encoder(uint32_t lam)
                : lambda(double(lam)), lamc(double(lam) * 8.0 / 1000.0 * unit)
            {
            }

            uint32_t cyc(double c) const
            {
                return uint32_t(c * lamc + 0.5);
            }

            choice choose_bytes(const uint32_t* h, uint64_t n) const
            {
                choice c{};
                c.huf = false;
                c.bytes = double(n);
                c.j = double(n);
                if (n < huf_min)
                    return c;
                uint8_t len[256];
                const uint64_t bits = huffman_encoder::code_lengths(h, len);
                uint32_t present = 0;
                for (uint32_t a = 0; a < 256; ++a)
                    present += len[a] != 0;
                const double tb =
                    std::min(double(huffman_encoder::flat_table_size),
                             double(1 + huffman_encoder::bitmap_bytes) + (present + 1) / 2);
                // + half a byte of padding per bit string
                const double hb = tb + double(bits) / 8.0 + double(payload_split / 2) +
                                  double(huffman_encoder::size_fields);
                const double hj = hb + lambda * (cyc_huf * double(n) + huf_setup) / 1000.0;
                if (hj < c.j)
                {
                    c.huf = true, c.j = hj, c.bytes = hb;
                    memcpy(c.len, len, 256);
                }
                return c;
            }

            void prior_costs()
            {
                const double kb[kind_cnt] = {3.5, n_base_bits, n_base_bits, n_base_bits};
                for (uint32_t s = 0; s < pb.cnt; ++s)
                {
                    const uint32_t ll = pb.lit[s], ml = pb.len[s], k = pb.kind[s];
                    const double b = kb[k] + (ll ? 1.5 + 1.2 * std::log2(double(ll)) : 0.3) +
                                     (ml ? 0.8 + 1.3 * std::log2(double(ml)) : 0.0);
                    co.cmd[s] = uint32_t(std::min(b, 13.0) * unit) + cyc(cyc_cmd + cyc_sym0);
                }
                for (uint32_t a = 0; a < 256; ++a)
                {
                    co.lit[a] = 8 * unit;
                    co.o1[a] = co.o2[0][a] = co.o2[1][a] = 8 * unit;
                    co.o3[0][a] = co.o3[1][a] = co.o3[2][a] = 8 * unit;
                }
                co.f1[0] = co.f1[1] = co.f2[0] = co.f2[1] = unit;
                co.sub = false;
                co.far = cyc(cyc_far), co.merge = cyc(cyc_merge);
                for (uint32_t k = 0; k < kind_cnt; ++k)
                    for (uint32_t l = 0; l <= len_tab; ++l)
                        co.w[k][l] = (l >= 3 and l <= 12) ? 1.0 : 0.0;
                co.derive();
            }

            void byte_costs(const choice& c, uint32_t* out) const
            {
                if (c.huf)
                {
                    const uint32_t cy = cyc(cyc_huf);
                    for (uint32_t a = 0; a < 256; ++a)
                        out[a] = uint32_t((c.len[a] ? double(c.len[a]) : unseen_bits) * unit) + cy;
                }
                else
                    for (uint32_t a = 0; a < 256; ++a)
                        out[a] = 8 * unit;
            }

            // modelled coded bits per flag, flat
            void flag_costs(const std::vector<uint8_t>& f, uint32_t* out)
            {
                const uint64_t n = f.size();
                std::vector<uint8_t> pk;
                pack_flags(f, pk);
                uint32_t hb[256];
                hist256(pk, hb);
                const choice c = choose_bytes(hb, pk.size());
                if (not c.huf or n == 0)
                {
                    out[0] = out[1] = unit;
                    return;
                }
                out[0] = out[1] = uint32_t(std::min(1.0, 8.0 * c.bytes / double(n)) * unit);
            }

            void costs_from_streams(const prior pr)
            {
                uint32_t h[codebook_size] = {};
                for (uint8_t s : st.cmd)
                    h[s]++;
                const uint32_t cy = cyc(cyc_cmd + cyc_huf);
                uint8_t len[codebook_size] = {};
                huffman_encoder::code_lengths(h, len);
                double fp[codebook_size];
                for (uint32_t s = 0; s < codebook_size; ++s)
                    fp[s] = pr.t == prior_kind::flat ? pr.x : unseen_bits;
                if (pr.t == prior_kind::factored)
                {
                    constexpr uint32_t nk = 2;
                    double hk[nk] = {}, hl[nk][256] = {}, hm[nk][256] = {}, tot = 0;
                    for (uint32_t s = 0; s < codebook_size; ++s)
                    {
                        if (not h[s])
                            continue;
                        const entry& e = codebook[s];
                        const uint32_t k = uint32_t(e.k);
                        hk[k] += h[s], hl[k][e.lit] += h[s], hm[k][e.len] += h[s], tot += h[s];
                    }
                    for (uint32_t s = 0; s < codebook_size; ++s)
                    {
                        if (h[s])
                            continue;
                        const entry& e = codebook[s];
                        const uint32_t k = uint32_t(e.k);
                        const double b = std::log2((tot + 1) / (hk[k] + 0.5)) +
                                         std::log2((hk[k] + 1) / (hl[k][e.lit] + 0.5)) +
                                         std::log2((hk[k] + 1) / (hm[k][e.len] + 0.5)) + pr.x;
                        fp[s] = std::clamp(b, 2.0, unseen_bits);
                    }
                }
                uint32_t c1[codebook_size];
                for (uint32_t s = 0; s < codebook_size; ++s)
                    c1[s] = uint32_t((len[s] ? double(len[s]) : fp[s]) * unit) + cy;
                for (uint32_t s = 0; s < pb.cnt; ++s)
                    co.cmd[s] = c1[pb.cmd[s]];

                uint32_t hb[256];
                choice cl[2];
                for (int m = 0; m < 2; ++m)
                {
                    hist256(st.lit[m], hb);
                    cl[m] = choose_bytes(hb, st.lit[m].size());
                }
                const bool sub = cl[1].j < cl[0].j;
                co.sub = sub;
                byte_costs(cl[sub], co.lit);
                hist256(st.o1, hb), byte_costs(choose_bytes(hb, st.o1.size()), co.o1);
                for (int i = 0; i < 2; ++i)
                    hist256(st.o2[i], hb), byte_costs(choose_bytes(hb, st.o2[i].size()), co.o2[i]);
                for (int i = 0; i < 3; ++i)
                    hist256(st.o3[i], hb), byte_costs(choose_bytes(hb, st.o3[i].size()), co.o3[i]);
                flag_costs(st.f1, co.f1);
                flag_costs(st.f2, co.f2);
                for (uint32_t k = 0; k < kind_cnt; ++k)
                    for (uint32_t l = 0; l <= len_tab; ++l)
                        co.w[k][l] = double(st.tok_hist[k][l]);
                co.derive();
            }

            [[gnu::always_inline]] inline void ins(uint32_t q,
                                                   uint32_t key,
                                                   uint32_t cost,
                                                   uint32_t r0,
                                                   uint32_t litrun,
                                                   uint32_t mlen,
                                                   uint32_t mdist,
                                                   uint8_t slot,
                                                   uint8_t kind)
            {
                if (key >= wc[q])
                    return;
                arrival* s = &A[uint64_t(q) * cur_k];
                const uint32_t n = cnt[q];
                uint32_t at = n;
                for (uint32_t j = 0; j < n; ++j)
                    if (s[j].r0 == r0)
                    {
                        if (key >= s[j].key)
                            return;
                        at = j;
                        break;
                    }
                if (at == n)
                {
                    if (n < cur_k)
                        cnt[q] = uint8_t(n + 1);
                    else
                    {
                        at = 0;
                        for (uint32_t j = 1; j < n; ++j)
                            if (s[j].key > s[at].key)
                                at = j;
                    }
                }
                s[at] = {key, cost, r0, litrun, mlen, mdist, slot, kind};
                if (cnt[q] == cur_k)
                {
                    uint32_t w = s[0].key;
                    for (uint32_t j = 1; j < cur_k; ++j)
                        w = std::max(w, s[j].key);
                    wc[q] = w;
                }
            }

            // match tokens of lengths [lo, hi] from arrival `a` at block position `i`
            [[gnu::always_inline]] inline void relax(uint32_t i,
                                                     const arrival& a,
                                                     uint8_t slot,
                                                     uint32_t kind,
                                                     uint32_t d,
                                                     uint32_t lo,
                                                     uint32_t hi,
                                                     uint32_t offc,
                                                     bool only_longest)
            {
                if (lo > hi)
                    return;
                if (not only_longest)
                    for (uint32_t l = lo, e = std::min(hi, all_lens); l <= e; ++l)
                    {
                        const uint32_t m = co.mcost(kind, l, a.litrun);
                        if (m >= cost_inf)
                            continue;
                        const uint32_t c = a.cost + m + offc;
                        ins(i + l, c, c, d, 0, l, d, slot, uint8_t(kind));
                    }
                if (hi > all_lens or only_longest)
                {
                    const uint32_t m = co.mcost(kind, hi, a.litrun);
                    if (m < cost_inf)
                    {
                        const uint32_t c = a.cost + m + offc;
                        ins(i + hi, c, c, d, 0, hi, d, slot, uint8_t(kind));
                    }
                }
            }

            uint32_t new_offc(uint32_t kind, uint32_t d) const
            {
                return co.offcost(kind, d) + (d > far_dis ? co.far : 0);
            }

            // Byte at absolute position p is `v[p - vo]`. Returns false if the block end is
            // unreachable (cannot happen: every visited position reaches the next one).
            bool parse(const uint8_t* v,
                       uint32_t vo,
                       uint32_t bs,
                       uint32_t be,
                       const std::vector<uint32_t>& moff,
                       const std::vector<match>& mm,
                       uint32_t r0in)
            {
                const uint32_t nb = be - bs;
                A.resize(uint64_t(nb + 1) * cur_k);
                cnt.assign(nb + 1, 0);
                wc.assign(nb + 1, 0xFFFFFFFFu);
                A[0] = {0, 0, r0in, 0, 0, 0, 0, k_lit};
                cnt[0] = 1;
                uint32_t skip_until = 0;
                const bool sub = co.sub;
                for (uint32_t i = 0; i < nb; ++i)
                {
                    const uint32_t c = cnt[i];
                    if (not c or i < skip_until)
                        continue;
                    const uint32_t p = bs + i;
                    const uint8_t* const at = v + (p - vo);
                    const arrival* const s = &A[uint64_t(i) * cur_k];
                    const uint32_t maxl = be - p;
                    uint32_t best = 0;
                    for (uint32_t j = 1; j < c; ++j)
                        if (s[j].key < s[best].key)
                            best = j;
                    const match* const m0 = mm.data() + moff[i];
                    const match* const m1 = mm.data() + moff[i + 1];
                    uint32_t l0[arrivals];
                    uint32_t longest = m1 > m0 ? std::min(m1[-1].len, maxl) : 0;
                    for (uint32_t j = 0; j < c; ++j)
                    {
                        l0[j] = lcp<isa_lib>(at, at - s[j].r0, maxl);
                        longest = std::max(longest, l0[j]);
                    }

                    if (longest >= nice_len)
                    {
                        uint32_t mn = longest;
                        for (uint32_t j = 0; j < c; ++j)
                            if (l0[j] >= nice_len)
                            {
                                relax(i, s[j], uint8_t(j), k_r0, s[j].r0, l0[j], l0[j], 0, true);
                                mn = std::min(mn, l0[j]);
                            }
                        if (m1 > m0)
                        {
                            const uint32_t len = std::min(m1[-1].len, maxl), d = m1[-1].dis;
                            const arrival& a = s[best];
                            if (len >= nice_len and d != a.r0)
                            {
                                const uint32_t kind = kind_of(d);
                                relax(i,
                                      a,
                                      uint8_t(best),
                                      kind,
                                      d,
                                      len,
                                      len,
                                      new_offc(kind, d),
                                      true);
                                mn = std::min(mn, len);
                            }
                        }
                        skip_until = i + mn;
                        continue;
                    }

                    for (uint32_t j = 0; j < c; ++j)
                    {
                        const arrival& a = s[j];
                        const uint32_t lc =
                            sub ? co.lit[uint8_t(at[0] - at[-int64_t(a.r0)])] : co.lit[at[0]];
                        const uint32_t cost = a.cost + lc;
                        ins(i + 1,
                            cost + co.apend(a.litrun + 1),
                            cost,
                            a.r0,
                            a.litrun + 1,
                            0,
                            0,
                            uint8_t(j),
                            k_lit);
                    }
                    for (uint32_t j = 0; j < c; ++j)
                    {
                        const arrival a = s[j];
                        if (l0[j] >= min_len[k_r0])
                            relax(i, a, uint8_t(j), k_r0, a.r0, min_len[k_r0], l0[j], 0, false);
                    }
                    {
                        const arrival a = s[best];
                        uint32_t prev = 0;
                        for (const match* m = m0; m < m1; ++m)
                        {
                            const uint32_t len = std::min(m->len, maxl), d = m->dis;
                            if (len <= prev)
                                continue;
                            const uint32_t kind = kind_of(d);
                            const uint32_t lo = std::max(prev + 1, min_len[kind]);
                            prev = len;
                            if (d == a.r0 or d > enc_window)
                                continue;
                            relax(i, a, uint8_t(best), kind, d, lo, len, new_offc(kind, d), false);
                        }
                    }
                }

                if (not cnt[nb])
                    return false;
                uint32_t bj = 0, bc = 0xFFFFFFFFu;
                for (uint32_t j = 0; j < cnt[nb]; ++j)
                {
                    const arrival& a = A[uint64_t(nb) * cur_k + j];
                    const uint32_t c = a.cost + co.Lc(a.litrun);
                    if (c < bc)
                        bc = c, bj = j;
                }
                toks.clear();
                uint32_t i = nb, j = bj, trail = 0;
                bool seen = false;
                while (i > 0)
                {
                    const arrival& a = A[uint64_t(i) * cur_k + j];
                    if (a.kind == k_lit)
                    {
                        if (seen)
                            toks.back().lits++;
                        else
                            ++trail;
                        i -= 1, j = a.slot;
                    }
                    else
                    {
                        toks.push_back({0, a.mlen, a.mdist, a.kind});
                        seen = true;
                        i -= a.mlen, j = a.slot;
                    }
                }
                std::reverse(toks.begin(), toks.end());
                toks.push_back({trail, 0, 0, k_lit});
                return true;
            }

            // The parse's tokens as streams; `r0` = repeat distance at the block start, updated
            void emit(const uint8_t* v, uint32_t vo, uint32_t bs, uint32_t& r0)
            {
                st.clear();
                uint32_t p = bs;
                auto put = [&](uint32_t s) { st.cmd.push_back(pb.cmd[s]); };
                auto lit_emit = [&](uint32_t x)
                {
                    if (x > run_tab)
                    {
                        const uint32_t q = (x - (max_lit + 1)) / max_lit;
                        for (uint32_t k = 0; k < q; ++k)
                            put(sym_lit15);
                        x -= max_lit * q;
                    }
                    while (x)
                    {
                        const uint32_t j = co.Lj[x];
                        put(uint32_t(pb.sym[k_r0][j][0]));
                        x -= j;
                    }
                };
                auto rest_emit = [&](uint32_t y)
                {
                    if (y > len_tab)
                    {
                        const uint32_t q = (y - (len_tab - max_match_len) - 1) / max_match_len;
                        for (uint32_t k = 0; k < q; ++k)
                            put(sym_cont);
                        y -= max_match_len * q;
                    }
                    while (y)
                    {
                        const uint32_t j = co.Cj[y];
                        put(uint32_t(pb.sym[k_r0][0][j]));
                        y -= j;
                    }
                };
                for (const token& t : toks)
                {
                    for (uint32_t q = p; q < p + t.lits; ++q)
                    {
                        st.lit[0].push_back(v[q - vo]);
                        st.lit[1].push_back(uint8_t(v[q - vo] - v[q - vo - r0]));
                    }
                    if (t.len == 0)
                    {
                        lit_emit(t.lits);
                        break;
                    }
                    const uint32_t lt = std::min(t.len, len_tab), rt = std::min(t.lits, run_tab);
                    const uint32_t s = co.Ms[t.kind][lt][rt];
                    lit_emit(t.lits - pb.lit[s]);
                    put(s);
                    const uint32_t d = t.dist, o = d - uint32_t(min_dis);
                    if (t.kind == k_n1)
                        st.o1.push_back(uint8_t(o)), st.f1.push_back(0);
                    else if (t.kind == k_n2)
                    {
                        st.o2[0].push_back(uint8_t(o)), st.o2[1].push_back(uint8_t(o >> 8));
                        st.f1.push_back(1), st.f2.push_back(0);
                    }
                    else if (t.kind == k_n3)
                    {
                        st.o3[0].push_back(uint8_t(o)), st.o3[1].push_back(uint8_t(o >> 8));
                        st.o3[2].push_back(uint8_t(o >> 16));
                        st.f1.push_back(1), st.f2.push_back(1);
                    }
                    rest_emit(t.len - pb.len[s]);
                    if (t.kind != k_r0)
                        r0 = d;
                    st.tok_hist[t.kind][std::min(t.len, len_tab)]++;
                    p += t.lits + t.len;
                }
            }

            // Raw or Huffman, whichever is smaller after lambda * decode cycles
            bool put_bytes(const std::vector<uint8_t>& s,
                           std::vector<uint8_t>& out,
                           double* j_out = nullptr)
            {
                const uint64_t n = s.size();
                double best = double(n);
                bool huf = false;
                uint64_t hs = 0;
                if (n >= huf_min)
                {
                    tmp.resize(huffman_encoder::bound(n));
                    hs = huffman_encoder::encode(tmp.data(), s.data(), n);
                    const double j =
                        double(hs) + lambda * (cyc_huf * double(n) + huf_setup) / 1000.0;
                    if (hs < n and j < best)
                        best = j, huf = true;
                }
                if (huf)
                    out.insert(out.end(), tmp.begin(), tmp.begin() + int64_t(hs));
                else
                    out.insert(out.end(), s.begin(), s.end());
                if (j_out)
                    *j_out = best;
                return huf;
            }

            void write_block(std::vector<uint8_t>& dst)
            {
                std::vector<uint8_t> pay[stream_cnt];
                bool huf[stream_cnt] = {};
                huf[s_cmd] = put_bytes(st.cmd, pay[s_cmd]);
                bool sub;
                {
                    std::vector<uint8_t> a, b;
                    double ja = 0, jb = 0;
                    const bool ha = put_bytes(st.lit[0], a, &ja);
                    const bool hb = put_bytes(st.lit[1], b, &jb);
                    sub = jb < ja;
                    huf[s_lit] = sub ? hb : ha;
                    pay[s_lit] = sub ? b : a;
                }
                huf[s_o1] = put_bytes(st.o1, pay[s_o1]);
                huf[s_o2l] = put_bytes(st.o2[0], pay[s_o2l]);
                huf[s_o2h] = put_bytes(st.o2[1], pay[s_o2h]);
                huf[s_o3l] = put_bytes(st.o3[0], pay[s_o3l]);
                huf[s_o3m] = put_bytes(st.o3[1], pay[s_o3m]);
                huf[s_o3h] = put_bytes(st.o3[2], pay[s_o3h]);
                {
                    std::vector<uint8_t> pk;
                    pack_flags(st.f1, pk), huf[s_f1] = put_bytes(pk, pay[s_f1]);
                    pack_flags(st.f2, pk), huf[s_f2] = put_bytes(pk, pay[s_f2]);
                }

                auto put_cnt = [&](uint64_t v)
                {
                    for (uint64_t b = 0; b < cnt_bytes; ++b)
                        dst.push_back(uint8_t(v >> (8 * b)));
                };
                dst.push_back(uint8_t((sub ? mode_sub : 0) | (cur_delta ? mode_delta : 0)));
                put_cnt(st.cmd.size()), put_cnt(st.lit[0].size()), put_cnt(st.o1.size());
                put_cnt(st.o2[0].size()), put_cnt(st.o3[0].size());
                uint32_t coded = 0;
                for (uint32_t s = 0; s < stream_cnt; ++s)
                    coded |= uint32_t(huf[s]) << s;
                dst.push_back(uint8_t(coded)), dst.push_back(uint8_t(coded >> 8));
                if (cur_delta)
                    dst.insert(dst.end(), cur_strides.begin(), cur_strides.end());
                for (uint32_t s = 0; s < stream_cnt; ++s)
                    if (huf[s])
                        for (uint64_t b = 0; b < coded_size_bytes; ++b)
                            dst.push_back(uint8_t(pay[s].size() >> (8 * b)));
                for (uint32_t s = 0; s < stream_cnt; ++s)
                    dst.insert(dst.end(), pay[s].begin(), pay[s].end());
            }

            // Per cell of [bs, be), the stride with the lowest estimate, kept if it beats plain
            // by `delta_gain`. Into `bprop`; returns whether any cell is transformed.
            bool propose_block(const uint8_t* src, uint32_t bs, uint32_t be)
            {
                const uint32_t bn = be - bs;
                const uint32_t c0 = bs / uint32_t(cell_size), c1 = (be - 1) / uint32_t(cell_size);
                bprop.assign(c1 - c0 + 1, 0);
                de.reset(bn);
                bool any = false;
                for (uint32_t c = c0; c <= c1; ++c)
                {
                    const uint32_t p0 = std::max(c * uint32_t(cell_size), bs) - bs;
                    const uint32_t p1 = std::min((c + 1) * uint32_t(cell_size), be) - bs;
                    auto fill = [&](uint32_t sd)
                    {
                        for (uint32_t i = p0; i < p1; ++i)
                            de.t[i] = sd ? uint8_t(src[bs + i] - src[bs + i - sd]) : src[bs + i];
                    };
                    uint32_t h[256];
                    auto ent = [&](uint32_t sd)
                    {
                        memset(h, 0, sizeof h);
                        for (uint32_t i = p0; i < p1; ++i)
                            ++h[sd ? uint8_t(src[bs + i] - src[bs + i - sd]) : src[bs + i]];
                        return stride_estimator::hist_bits(h, p1 - p0) / double(p1 - p0);
                    };
                    uint32_t cand[sizeof(strides) / sizeof(strides[0])], ncand = 0;
                    const double h0 = ent(0);
                    for (uint32_t sd : strides)
                        if (ent(sd) <= h0 + stride_margin)
                            cand[ncand++] = sd;
                    if (ncand == 0)
                    {
                        fill(0);
                        de.commit(p0, p1);
                        continue;
                    }

                    fill(0);
                    const double plain = de.cost(p0, p1);
                    uint32_t rr0 = de.cr0, bsd = 0;
                    // a stride must beat this; then the best so far
                    double bc = plain * (1 - delta_gain);
                    for (uint32_t ci = 0; ci < ncand; ++ci)
                    {
                        const uint32_t sd = cand[ci];
                        fill(sd);
                        const double v = de.cost(p0, p1, bc);
                        if (v < bc)
                            bc = v, bsd = sd, rr0 = de.cr0;
                    }
                    fill(bsd);
                    de.commit(p0, p1);
                    de.r0 = rr0;
                    bprop[c - c0] = uint8_t(bsd);
                    any |= bsd != 0;
                }
                return any;
            }

            // first transformed position >= x of the block (positions before bs are plain)
            uint32_t next_prop(uint32_t x, uint32_t bs, uint32_t be) const
            {
                if (x < bs)
                    x = bs;
                if (x >= be)
                    return be;
                const uint32_t k = x / uint32_t(cell_size) - bs / uint32_t(cell_size);
                return bprop[k] ? x : npc[k + 1];
            }

            // The DELTA view's bytes (`wseg`) and matches. Its finder covers the block only, plus
            // the plain finder's matches cut where target or source enters a transformed cell.
            void build_view(const uint8_t* src, uint32_t n, uint32_t bs, uint32_t be)
            {
                const uint32_t cs = uint32_t(cell_size);
                const uint32_t c0 = bs / cs, nk = uint32_t(bprop.size());
                npc.assign(nk + 1, be);
                for (uint32_t k = nk; k-- > 0;)
                    npc[k] = bprop[k] ? std::max((c0 + k) * cs, bs) : npc[k + 1];

                const uint32_t seg0 = bs >= local_history ? bs - local_history : 0;
                const uint32_t segend = std::min<uint32_t>(n, be + local_lookahead);
                lb.resize(uint64_t(segend - seg0) + 64);
                memcpy(lb.data(), src + seg0, segend - seg0);
                for (uint32_t k = 0; k < nk; ++k)
                    if (const uint32_t sd = bprop[k])
                        for (uint32_t i = std::max((c0 + k) * cs, bs),
                                      e = std::min((c0 + k + 1) * cs, be);
                             i < e;
                             ++i)
                            lb[i - seg0] = uint8_t(src[i] - src[i - sd]);

                const uint32_t win = enc_window;
                wlo = bs > win + 64 ? bs - win - 64 : 0;
                const uint32_t whi = std::min<uint32_t>(n, be + 64);
                wseg.resize(uint64_t(whi - wlo) + 64);
                memcpy(wseg.data(), src + wlo, bs - wlo);
                memcpy(wseg.data() + (bs - wlo), lb.data() + (bs - seg0), be - bs);
                memcpy(wseg.data() + (be - wlo), src + be, whi - be);

                lx.init(lb.data(), segend - seg0, enc_window, uint32_t(min_dis));
                const bool far = seg0 > 0;
                moff_delta.clear(), mm_delta.clear();
                moff_delta.reserve(be - bs + 1);
                for (uint32_t i = 0; i < be - bs; ++i)
                {
                    moff_delta.push_back(uint32_t(mm_delta.size()));
                    const uint32_t p = bs + i;
                    uint32_t nl = 0;
                    const uint32_t kl = lx.step(p - seg0, tmp_s);
                    for (uint32_t k = 0; k < kl; ++k)
                    {
                        const match m = tmp_s[k];
                        const uint32_t q = p - m.dis;
                        // the first block's prefix sits in front of a transformed first cell
                        if (q < bs and q >= c0 * cs and bprop[0])
                            continue;
                        tmp_l[nl++] = m;
                    }
                    uint32_t nf = 0;
                    if (far)
                    {
                        const uint32_t cap_t = next_prop(p, bs, be) - p;
                        if (cap_t >= 3)
                            for (uint32_t k = moff_plain[i]; k < moff_plain[i + 1]; ++k)
                            {
                                const match m = mm_plain[k];
                                const uint32_t q = p - m.dis;
                                const uint32_t len =
                                    std::min({m.len, cap_t, next_prop(q, bs, be) - q});
                                if (len >= 3)
                                    tmp_f[nf++] = {len, m.dis};
                            }
                    }
                    // merge by distance, keep the frontier
                    uint32_t a = 0, b = 0, prev = 0;
                    while (a < nl or b < nf)
                    {
                        const match m = (b >= nf or (a < nl and tmp_l[a].dis <= tmp_f[b].dis))
                                            ? tmp_l[a++]
                                            : tmp_f[b++];
                        if (m.len > prev)
                            mm_delta.push_back(m), prev = m.len;
                    }
                }
                moff_delta.push_back(uint32_t(mm_delta.size()));
            }

            // Writes the blocks at `out` and returns their end, nullptr on failure. Every block is
            // at most its length + 1 bytes. PRECONDITION: `n > small_lim`
            uint8_t* encode(const uint8_t* src, uint64_t n, uint8_t* out)
            {
                uint32_t r0 = uint32_t(initial_dis);
                const uint64_t body_end = n - suffix_size;
                const uint32_t bsz = uint32_t(block_size);
                A.reserve((std::min<uint64_t>(bsz, n) + 1) * arrivals);
                for (uint64_t at = prefix_size; at < body_end;)
                {
                    // this view's blocks are [at, stop); the finder inserts [base, at) on its way
                    const uint64_t stop = std::min((at / view_span + 1) * view_span, body_end);
                    const uint64_t base = at >= view_history ? at - view_history : 0;
                    const uint8_t* const s = src + base;
                    const uint32_t vn = uint32_t(std::min<uint64_t>(n, stop + block_size) - base);
                    const uint32_t vend = uint32_t(stop - base);
                    fp.init(s, vn, enc_window, uint32_t(min_dis));
                    for (uint32_t bs = uint32_t(at - base); bs < vend;)
                    {
                        const uint32_t be = std::min((bs / bsz + 1) * bsz, vend);
                        const bool dblk = propose_block(s, bs, be);
                        fp.find_block(bs, be, moff_plain, mm_plain);
                        const bool first = not have_costs;
                        uint64_t best_size = ~uint64_t(0);
                        uint32_t best_r0 = r0;
                        if (dblk)
                            build_view(s, vn, bs, be);

                        // plain, then DELTA; keep the smaller block
                        for (uint32_t view = 0; view < (dblk ? 2u : 1u); ++view)
                        {
                            const uint8_t* const v = view ? wseg.data() : s;
                            const uint32_t vo = view ? wlo : 0;
                            const std::vector<uint32_t>& moff = view ? moff_delta : moff_plain;
                            const std::vector<match>& mm = view ? mm_delta : mm_plain;
                            cur_delta = view == 1;
                            if (cur_delta)
                                cur_strides = bprop;
                            if (first)
                            {
                                prior_costs();
                                cur_k = 1;
                                if (not parse(v, vo, bs, be, moff, mm, r0))
                                    return nullptr;
                                uint32_t a = r0;
                                emit(v, vo, bs, a);
                            }
                            else
                                st = (view and have_delta) ? carry_delta : carry_plain;
                            for (uint32_t r = 0; r < round_cnt; ++r)
                            {
                                const bool last = r + 1 == round_cnt;
                                costs_from_streams(rounds[r]);
                                cur_k = last ? arrivals : 1;
                                if (not parse(v, vo, bs, be, moff, mm, r0))
                                    return nullptr;
                                uint32_t a = r0;
                                emit(v, vo, bs, a);
                                if (not last)
                                    continue;
                                cbuf.clear();
                                write_block(cbuf);
                                if (cbuf.size() < best_size)
                                {
                                    best_size = cbuf.size(), best_r0 = a;
                                    wbuf.swap(cbuf);
                                }
                                if (view)
                                    carry_delta = st, have_delta = true;
                                else
                                    carry_plain = st;
                            }
                        }
                        cur_delta = false;
                        have_costs = true;

                        // a coded block larger than stored would break `compressed_size_bound`
                        const uint64_t len = be - bs;
                        bool store = wbuf.size() >= len + 1;
                        if (not store)
                            store =
                                double(len) + 1 - double(wbuf.size()) < store_gain * double(len);
                        if (store)
                        {
                            // the decoder's repeat distance does not move
                            *out++ = mode_stored;
                            memcpy(out, s + bs, len);
                            out += len;
                        }
                        else
                        {
                            memcpy(out, wbuf.data(), wbuf.size());
                            out += wbuf.size();
                            r0 = best_r0;
                        }
                        bs = be;
                    }
                    at = stop;
                }
                return out;
            }
        };
    } // namespace entropy_optimal_detail

    // Returns number of bytes written to `dst`, and 0 on failure.
    // `isa_lib` is ISA-dependent. `lambda` trades size for decode speed (larger = faster).
    template <class isa_lib>
    uint64_t entropy_optimal_cimpl(const uint8_t* __restrict src,
                                   uint64_t src_size,
                                   uint8_t* __restrict dst,
                                   uint64_t dst_cap,
                                   uint32_t lambda = entropy_optimal_detail::default_lambda)
    {
        using namespace entropy;
        using namespace entropy::huffman;
        using namespace entropy_optimal_detail;

        if (compressed_size_bound(src_size) > dst_cap)
            return 0;

        storeu8(dst, src_size | (uint64_t(flags) << flags_shift));
        if (src_size <= small_lim)
        {
            if (src_size != 0)
                memcpy(dst + header_size, src, src_size);
            return header_size + src_size;
        }

        // ~150 KB of tables, off the stack
        const auto enc = std::make_unique<encoder<isa_lib>>(lambda);
        uint8_t* out = enc->encode(src, src_size, dst + header_size);
        if (out == nullptr)
            return 0;
        memcpy(out, src, prefix_size);
        out += prefix_size;
        memcpy(out, src + src_size - suffix_size, suffix_size);
        out += suffix_size;
        return uint64_t(out - dst);
    }
} // namespace misa77

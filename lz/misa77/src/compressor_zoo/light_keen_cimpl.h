// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "format.h"
#include "misa77/misa77.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace misa77
{
    namespace light_keen_detail
    {
        using namespace light;

        constexpr uint32_t hash_top = 16;
        constexpr uint32_t hash_siz = 1 << hash_top;
        constexpr uint32_t hash_mul = 2654435761;

        // Size of ring buffer per hash in the hashtable. Wider than other hashtable modes to
        // lengthen matches, which improves ratio and decode (lesser tokens).
        inline constexpr uint64_t hashtab_wid = 20;

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
            if (rem != uint8_t(7)) [[likely]]
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

        // A match must clear this bar to be taken on sight (shorter ones only get emitted through
        // the fire rule below)
        constexpr uint64_t accept_len = 6;
        static_assert(accept_len >= min_match_len);

        // We ideally want to keep literal length `<= fire_at`
        constexpr uint64_t fire_at = 6;

        // Don't look further ahead once you have a match `>= la_gate`
        constexpr uint64_t la_gate = 16;

        // Once you have match `>= la_pate`, only keep looking if you get gains
        constexpr uint64_t la_pate = 8;

        // When you've gone `p` hashtab searches without a match, advance the pointer by `p >>
        // skip_shift`
        constexpr uint64_t skip_shift = 6;

        // Regime counter: an adaptive value in `[0, regime_cap]`
        // Emits with literal length `>= lit_lim` have a vote of `+3`, others `-1`
        // Above the threshold the accept bar falls back to `min_match_len`: extras-heavy data needs
        // its short matches, both for ratio and for the predictable token pattern the decoder's
        // branches learn
        inline constexpr int64_t regime_cap = 64;
        inline constexpr int64_t regime_threshold = 32;

    } // namespace light_keen_detail

    // Returns number of bytes written to `dst`, and 0 on failure.
    // `isa_lib` is ISA-dependent.
    template <class isa_lib>
    uint64_t light_keen_cimpl(const uint8_t* __restrict src,
                              uint64_t src_size,
                              uint8_t* __restrict dst,
                              uint64_t dst_cap)
    {
        using namespace light;
        using namespace light_keen_detail;

        static_assert(max_match_len >= 32);

        if (compress_bound(src_size, config()) > dst_cap)
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

        // `[lit, pos]` corresponds to the new range from the src buffer we're going to be
        // turning into a lit + match pair
        uint64_t pos = 0;

        // `[0, hpos)` represents the range we've written into the hashtable
        uint64_t hpos = 0;

        uint64_t miss_run = 0;

        std::vector<std::array<uint16_t, hashtab_wid>> hashtab(hash_siz);
        std::vector<uint8_t> hashtab_idx(hash_siz);

        // Last remembered below-threshold match
        uint64_t cand_pos = 0, cand_len = 0, cand_lst = 0;

        // Adaptive variable that reflects values of lit_len we've been getting for recent blocks
        int64_t regime = 0;

        while (pos + max_match_len <= match_end_limit)
        {
            // Reduce branch misses, `batch = 8` performs well empirically
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
                             : hashtab_idx[hsh] + 1); // Just cycle the pointer in the ring buffer
                }
                hpos += batch;
            }

            uint32_t val = loadu4(src + pos);
            uint32_t hsh = hash4(val);

            uint64_t lst = 0;
            uint64_t match_len = 0;

            typename isa_lib::vec reg = isa_lib::loadvec(src + pos);

            if (pos > hashtab_lag) [[likely]]
            {
// MLP
#pragma GCC unroll hashtab_wid
                for (uint8_t i = 0; i < hashtab_wid; i++)
                {
                    uint16_t d = uint16_t(pos - hashtab[hsh][i] - hashtab_lag - 1);

                    // Guaranteed to lie within `[pos - 2^16 - hashtab_lag, pos - hashtab_lag)`
                    uint64_t ilst = (pos - max_match_len - 1 - d);

                    typename isa_lib::vec ireg = isa_lib::loadvec(src + ilst);
                    uint32_t imatch_len = isa_lib::lcp(reg, ireg);
                    lst = (imatch_len > match_len ? ilst : lst);
                    match_len = (imatch_len > match_len ? imatch_len : match_len);
                }
            }

            // We've inserted stuff into the hashtable assuming this value of `pos`, so we cannot
            // probe it for any starting position behind this one going forward. This is
            // important as `pos` is regressed ahead.
            uint64_t pos_safe_bound = pos;

            bool accept =
                (match_len >= (regime >= regime_threshold ? uint64_t(min_match_len) : accept_len));

            if (accept)
            {
                for (uint64_t npos = pos + 1;
                     npos <= pos + lookahead and npos + max_match_len <= match_end_limit and
                     match_len < la_gate;
                     npos++)
                {
                    uint64_t nlst = 0;
                    uint64_t nmatch_len = 0;

                    uint32_t val = loadu4(src + npos);
                    uint32_t hsh = hash4(val);
                    typename isa_lib::vec reg = isa_lib::loadvec(src + npos);
#pragma GCC unroll hashtab_wid
                    for (uint8_t i = 0; i < hashtab_wid; i++)
                    {
                        uint16_t d = uint16_t(npos - hashtab[hsh][i] - hashtab_lag - 1);

                        // Guaranteed to lie within `[npos - 2^16 - hashtab_lag, npos -
                        // hashtab_lag)`
                        uint64_t ilst = (npos - max_match_len - 1 - d);

                        typename isa_lib::vec ireg = isa_lib::loadvec(src + ilst);
                        uint32_t imatch_len = isa_lib::lcp(reg, ireg);

                        nlst = (imatch_len > nmatch_len ? ilst : nlst);
                        nmatch_len = (imatch_len > nmatch_len ? imatch_len : nmatch_len);
                    }

                    bool improved = (nmatch_len > match_len);
                    pos = (improved ? npos : pos);
                    lst = (improved ? nlst : lst);
                    match_len = (improved ? nmatch_len : match_len);

                    if (!improved and match_len >= la_pate)
                        break;
                }
            }
            else
            {
                uint64_t pend = pos - lit;

                if (pend >= fire_at)
                {
                    // Prefer the one ending later between this position and the last
                    // remembered one
                    if (cand_len != 0 and
                        (match_len < min_match_len or cand_pos + cand_len >= pos + match_len))
                    {
                        pos = cand_pos;
                        lst = cand_lst;
                        match_len = cand_len;
                    }

                    // Note that if `accept` becomes false here, pos isn't pushed back by the
                    // above branch
                    accept = (match_len >= min_match_len);
                }
                else if (match_len >= min_match_len and
                         (cand_len == 0 or pos + match_len >= cand_pos + cand_len))
                {
                    // This position becomes the new candidate
                    cand_pos = pos;
                    cand_len = match_len;
                    cand_lst = lst;
                }
            }

            if (accept)
            {
                // Extend the match backwards (note that `dis` in `(hashtab_lag, 2^16 +
                // hashtab_lag]` holds).
                // Extending `pos` backwards here is safe because we're not looking into the
                // hashtable.
                while (pos > lit and lst > 0 and match_len < max_match_len and
                       src[pos - 1] == src[lst - 1])
                {
                    --pos;
                    --lst;
                    ++match_len;
                }

                uint64_t norm_match_len = match_len - min_match_len + 1;

                // `src[lit, pos)` are the literals
                uint64_t lit_len = pos - lit;

                // regime vote
                regime += (lit_len >= lit_lim ? 3 : -1);
                regime = std::clamp<int64_t>(regime, 0, regime_cap);

                // Token Byte
                uint8_t lrem = std::min(uint64_t(7), lit_len);
                dst[dlpos] = static_cast<uint8_t>((lrem << 5) | (norm_match_len));
                lit_len -= lrem;
                ++dlpos;

                // 2 Distance bytes
                uint64_t dis = pos - lst;
                uint16_t dbytes = dis - hashtab_lag - 1;
                storeu2(dst + dlpos, dbytes);
                dlpos += 2;

                // Extra literal length bytes
                emit_length_extras(dst, dlpos, lit_len, lrem);

                // Literal bytes
                if (lit < pos)
                {
                    uint64_t lit_cnt = pos - lit;
                    drpos -= lit_cnt;
                    memcpy(dst + drpos, src + lit, lit_cnt);
                }

                pos += match_len;
                lit = pos;
                pos = std::max(pos, pos_safe_bound);
                cand_len = 0;
                miss_run = 0;
            }
            else
            {
                pos += 1 + (miss_run >> skip_shift);
                ++miss_run;
            }
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

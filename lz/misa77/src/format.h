// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "codebook.h"

#include <cstdint>

namespace misa77
{
    // Header (both formats): LE u64 = size | flags << flags_shift
    inline constexpr uint64_t header_size = 8;
    inline constexpr uint64_t flags_shift = 56;
    inline constexpr uint64_t size_mask = (uint64_t(1) << flags_shift) - 1;

    // Low `format_bits` bits of flags = format id
    inline constexpr uint8_t format_bits = 2;
    inline constexpr uint8_t format_mask = (1 << format_bits) - 1;
    inline constexpr uint8_t format_light = 0;
    inline constexpr uint8_t format_heavy = 1;

    // Width in bytes of several unconditional operations we perform through isa libs in both
    // formats.
    inline constexpr uint64_t vector_width = 32;

    // light format, used by levels in [config::min_level, config::heavy_lb)
    namespace light
    {
        // (`raw file size <= small_lim`) => small mode, we just write the raw bytes.
        inline constexpr uint32_t small_lim = 32;

        // When we find a match, we check `lookahead` positions ahead for a potentially better
        // match.
        inline constexpr uint64_t lookahead = 2;

        // The distance in bytes by which the start of the active hashtable window lags the pointer.
        // Don't set this too high, as there were a lot of matches for dis in [32, 512].
        inline constexpr uint64_t hashtab_lag = 32;
        static_assert(hashtab_lag >= vector_width);

        // If we're asked to copy a match after having processed `c` bytes of raw data, then the
        // match will begin in `[c - dis_lim - hashtab_lag, c - hashtab_lag)`.
        inline constexpr uint64_t dis_lim = (uint32_t(1) << 16);

        // We have exactly 2 bytes to indicate distance.
        static_assert(dis_lim <= (1 << 16));

        // The two distance bytes hold `dis - min_dis`, so the window is exactly [min_dis, max_dis].
        inline constexpr uint32_t min_dis = uint32_t(hashtab_lag + 1);
        inline constexpr uint32_t max_dis = uint32_t(hashtab_lag + dis_lim);
        static_assert(max_dis - min_dis == dis_lim - 1);
        static_assert(dis_lim - 1 <= UINT16_MAX);

        inline constexpr uint32_t min_match_len = 4;

        // `min_match_len >= 4` is needed for the compression bound to hold.
        static_assert(min_match_len >= 4);

        inline constexpr uint32_t max_match_len = 32;

        // One `cyccpy` call should deal with the entire match.
        static_assert(max_match_len <= vector_width);

        // Every length in [min_match_len, max_match_len] should be exactly representable.
        static_assert(max_match_len - min_match_len + 1 <= (uint32_t(1) << 5) - 1);

        inline constexpr uint64_t lit_lim = 7;
        static_assert(lit_lim == (uint64_t(1) << 3) - 1);

        // Lowerbound for the number of literals in the final raw suffix.
        // This MUST be >= `vector_width` because:
        // 1. During decompression, `cyccpy` performs unconditional writes in the destination
        // buffer, and we're using this suffix as padding that we can safely overwrite!!!
        // 2. During decompression, we perform unconditional reads from the source buffer, and this
        // literal suffix acts as padding that we can safely "over-read" into!!!
        inline constexpr uint32_t literal_suffix = 32;
        static_assert(literal_suffix >= vector_width);

        // Need enough source bytes to be able to guarantee that we can produce the appropriate raw
        // literal suffix.
        static_assert(literal_suffix <= small_lim + 1);

        // Unconditionally copied number of literal bytes in decompressor.
        inline constexpr uint64_t dec_literal_copy = 16;
        static_assert(literal_suffix >= dec_literal_copy);
    } // namespace light

    // heavy format, used by levels in [config::heavy_lb, config::max_level]
    namespace heavy
    {
        // flags

        // Codebook id field, right above the format id. IDs 2 and 3 are unassigned rn (will
        // probably use 2 for an entropy level).
        inline constexpr uint8_t codebook_shift = format_bits;

        // Every other flags bit must be 0 for the heavy format, so these are the ONLY valid flags
        // here.
        inline constexpr uint8_t flags_a = format_heavy;
        inline constexpr uint8_t flags_s = uint8_t(flags_a | (1 << codebook_shift));
        static_assert(flags_a == 0x01 and flags_s == 0x05);

        // (`raw file size <= small_lim`) => small mode, we just write the raw bytes.
        inline constexpr uint64_t small_lim = 32;

        // block: [u24 tok_cnt][u24 off_cnt][u24 lit_cnt] cmd | off | lit

        inline constexpr uint64_t tok_cnt_pos = 0;
        inline constexpr uint64_t off_cnt_pos = 3;
        inline constexpr uint64_t lit_cnt_pos = 6;
        inline constexpr uint64_t cnt_bytes = 3;
        inline constexpr uint64_t cnt_lim = uint64_t(1) << (8 * cnt_bytes);
        inline constexpr uint64_t block_header_size = lit_cnt_pos + cnt_bytes;
        static_assert(off_cnt_pos == tok_cnt_pos + cnt_bytes and
                      lit_cnt_pos == off_cnt_pos + cnt_bytes);
        static_assert(block_header_size == 9);

        // Upper-bound on the output bytes of one block. Every token outputs >= 1 byte (asserted on
        // the codebooks below) and carries <= 3 offset bytes, so this bounds all three counts.
        // This restriction doesn't really have much to do with the decompressor, it just
        // makes compression's implementation convenient now that there are three streams.
        inline constexpr uint64_t max_block_output = uint64_t(1) << 20;
        static_assert(offset_bytes(kind::F) * max_block_output < cnt_lim);

        // tokens

        // dis = min_dis + v, where v is the token's offset bytes (LE).
        // `min_dis > vector_width` => a 32-byte match copy only reads bytes that are already
        // final, so `cyccpy` never overlaps with what it writes.
        inline constexpr uint64_t min_dis = 33;
        static_assert(min_dis > vector_width);

        // A stream starts with no repeat distance, so R is invalid before the first S / N / F.

        constexpr uint64_t max_dis(kind k)
        {
            return k == kind::S or k == kind::N or k == kind::F
                       ? min_dis + (uint64_t(1) << (8 * offset_bytes(k))) - 1
                       : 0;
        }
        static_assert(max_dis(kind::S) == 288 and max_dis(kind::N) == 65568);
        static_assert(max_dis(kind::F) == (uint64_t(1) << 24) + 32);

        // Lowerbound for the number of literals in the final raw suffix.
        // This MUST be >= `vector_width` because:
        // 1. During decompression, `cyccpy` performs unconditional writes in the destination
        // buffer, and we're using this suffix as padding that we can safely overwrite!!!
        // 2. During decompression, we perform unconditional reads from the source buffer, and this
        // literal suffix acts as padding that we can safely "over-read" into!!!
        // i.e. every token ends at `<= n - literal_suffix`. These literals may span blocks.
        inline constexpr uint32_t literal_suffix = 32;
        static_assert(literal_suffix >= vector_width);

        // Need enough source bytes to be able to guarantee that we can produce the appropriate raw
        // literal suffix.
        static_assert(literal_suffix <= small_lim + 1);

        // Unconditionally copied number of literal bytes in decompressor.
        inline constexpr uint64_t dec_literal_copy = 16;
        static_assert(literal_suffix >= dec_literal_copy);

        // Unconditionally loaded number of offset bytes in decompressor.
        inline constexpr uint64_t offset_load = 4;
        static_assert(literal_suffix >= offset_load);
        static_assert(offset_load >= offset_bytes(kind::F));

        // Largest match length copied by the single unconditional `cyccpy`.
        inline constexpr uint64_t max_small_len = vector_width;
        static_assert(max_small_len <= vector_width);

        // Per small token: most literal / offset bytes consumed, most output produced. The safe
        // decoder sizes its batches with these.
        inline constexpr uint64_t max_small_lit = dec_literal_copy;
        inline constexpr uint64_t max_offset_bytes = offset_bytes(kind::F);
        inline constexpr uint64_t max_small_out = max_small_lit + max_small_len;

        // A code that doesn't fit the unconditional copies is "big", and takes the rare branch.
        constexpr bool is_big(const entry& e)
        {
            return e.lit > dec_literal_copy or e.len > max_small_len;
        }

        // decode tables, derived from a codebook at compile time

        // nb[c] = offset bytes | (big ? big_flag : 0)
        inline constexpr uint8_t big_flag = 0x80;
        static_assert(max_offset_bytes < big_flag);

        struct decode_tables
        {
            // hot loop
            alignas(64) uint8_t nb[codebook_size];    // offset bytes, and big flag if big code
            alignas(64) uint8_t lit[codebook_size];   // 0 if big, lit length
            alignas(64) uint8_t len[codebook_size];   // 0 if big, match length
            alignas(64) uint32_t mask[codebook_size]; // 2^(8 * offset bytes) - 1, extracts offset

            // rare paths
            uint32_t full_lit[codebook_size];
            uint32_t full_len[codebook_size];
            kind k[codebook_size];

            // largest distance any code of this table can express
            uint64_t max_dis;

            constexpr explicit decode_tables(const entry (&cb)[codebook_size])
                : nb(), lit(), len(), mask(), full_lit(), full_len(), k(), max_dis(0)
            {
                for (uint32_t c = 0; c < codebook_size; ++c)
                {
                    const entry& e = cb[c];
                    const uint32_t ob = offset_bytes(e.k);
                    const bool big = is_big(e);
                    nb[c] = uint8_t(ob | (big ? big_flag : 0));
                    lit[c] = uint8_t(big ? 0 : e.lit);
                    len[c] = uint8_t(big ? 0 : e.len);
                    mask[c] = uint32_t((uint64_t(1) << (8 * ob)) - 1);
                    full_lit[c] = e.lit;
                    full_len[c] = e.len;
                    k[c] = e.k;
                    if (heavy::max_dis(e.k) > max_dis)
                        max_dis = heavy::max_dis(e.k);
                }
            }
        };

        // rules every codebook must satisfy

        // Every token must output > 0 bytes, so a block has at most as many tokens as output
        // bytes: `tok_cnt <= max_block_output`.
        constexpr bool entries_wellformed(const entry (&cb)[codebook_size])
        {
            for (const entry& e : cb)
            {
                if (e.k == kind::L)
                {
                    // literals only
                    if (e.lit == 0 or e.len != 0)
                        return false;
                }
                else
                {
                    // always a match
                    if (e.len == 0)
                        return false;
                }
                if (e.lit > max_block_output or e.len > max_block_output)
                    return false;
            }
            return true;
        }

        static_assert(entries_wellformed(codebook_a) and entries_wellformed(codebook_s));

        constexpr bool entries_distinct(const entry (&cb)[codebook_size])
        {
            for (uint32_t i = 0; i < codebook_size; ++i)
                for (uint32_t j = i + 1; j < codebook_size; ++j)
                    if (cb[i].k == cb[j].k and cb[i].lit == cb[j].lit and cb[i].len == cb[j].len)
                        return false;
            return true;
        }

        // eh, just so we don't unknowingly waste codes
        static_assert(entries_distinct(codebook_a) and entries_distinct(codebook_s));

        constexpr bool
        has_entry(const entry (&cb)[codebook_size], kind k, uint32_t lit, uint32_t len)
        {
            for (const entry& e : cb)
                if (e.k == k and e.lit == lit and e.len == len)
                    return true;
            return false;
        }

        constexpr bool has_kind(const entry (&cb)[codebook_size], kind k)
        {
            for (const entry& e : cb)
                if (e.k == k)
                    return true;
            return false;
        }

        // These codes allow us to express every run and match length ~cheaply.
        // Both tables must have all of them.

        // power-of-two lit lengths for L codes
        inline constexpr uint32_t max_lit_code = uint32_t(1) << 15;
        // power-of-two match lengths for R codes
        inline constexpr uint32_t min_rep_code = uint32_t(1) << 4;
        inline constexpr uint32_t max_rep_code = uint32_t(1) << 16;

        constexpr bool is_composable(const entry (&cb)[codebook_size])
        {
            for (uint32_t lit = 1; lit <= max_lit_code; lit *= 2)
                if (not has_entry(cb, kind::L, lit, 0))
                    return false;
            for (uint32_t len = min_rep_code; len <= max_rep_code; len *= 2)
                if (not has_entry(cb, kind::R, 0, len))
                    return false;
            return true;
        }

        static_assert(is_composable(codebook_a) and is_composable(codebook_s));

        inline constexpr decode_tables tables_a{codebook_a};
        inline constexpr decode_tables tables_s{codebook_s};

        // Table S has no 3-byte offsets, and is meant for small inputs.
        // (S and A break even on ratio and decode speed at about 256 KiB)
        inline constexpr uint64_t codebook_s_lim = uint64_t(1) << 18;
        static_assert(not has_kind(codebook_s, kind::F));
        static_assert(tables_s.max_dis == max_dis(kind::N));
        static_assert(tables_a.max_dis == max_dis(kind::F));

        // PRECONDITION: `flags` is `flags_a` or `flags_s`
        constexpr const decode_tables& tables_of(uint8_t flags)
        {
            return flags == flags_s ? tables_s : tables_a;
        }

        // compression bound

        // Every valid stream satisfies both of:
        // 1. a block's three streams never take more bytes than the block outputs (always
        // possible, as a block can just be raw literals).
        // 2. every block but the last outputs `>= min_block_output` bytes.
        // Decoders don't need to check either.
        inline constexpr uint64_t min_block_output = uint64_t(1) << 16;
        static_assert(min_block_output <= max_block_output);

        // input + stream header + one block header per block (<= n / min_block_output non-last
        // blocks, and the last one). Small mode takes `header_size + n`, which is less.
        constexpr uint64_t compressed_size_bound(uint64_t n)
        {
            return n + header_size + block_header_size * (n / min_block_output + 1);
        }
    } // namespace heavy
} // namespace misa77

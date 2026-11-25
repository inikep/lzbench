// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "format.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

// Encoder side of the entropy format's Huffman payload:
// [table][`payload_split` bit strings][u16 size of each but the last]

namespace misa77
{
    namespace huffman_encoder
    {
        using namespace entropy;
        using namespace entropy::huffman;

        inline constexpr uint32_t symbol_cnt = 256;
        // sparse table: [mode][presence bitmap][one nibble per present symbol]
        inline constexpr uint64_t bitmap_bytes = symbol_cnt / 8;
        // flat table: [mode][one nibble per symbol]
        inline constexpr uint64_t flat_table_size = 1 + symbol_cnt / 2;
        inline constexpr uint64_t size_fields = (payload_split - 1) * split_size_bytes;

        // Largest payload for `n` symbols
        constexpr uint64_t bound(uint64_t n)
        {
            return flat_table_size + size_fields + (n * max_code_len + 7) / 8 + payload_split;
        }

        // Optimal code lengths limited to `max_code_len` (package-merge).
        // Absent symbols have `len[a] = 0`, and if there's only one symbol, it gets length 1.
        // Returns the total number of code bits.
        inline uint64_t code_lengths(const uint32_t* cnt, uint8_t* len)
        {
            struct item
            {
                uint64_t w;
                int a, b; // children in the previous level's list (b < 0: leaf of symbol a)
            };
            std::vector<int> syms;
            for (int a = 0; a < int(symbol_cnt); ++a)
            {
                len[a] = 0;
                if (cnt[a])
                    syms.push_back(a);
            }
            const uint64_t m = syms.size();
            if (m == 0)
                return 0;
            if (m == 1)
            {
                len[syms[0]] = 1;
                return cnt[syms[0]];
            }
            std::sort(syms.begin(),
                      syms.end(),
                      [&](int x, int y) { return cnt[x] != cnt[y] ? cnt[x] < cnt[y] : x < y; });

            std::vector<std::vector<item>> lv(max_code_len);
            for (uint32_t l = 0; l < max_code_len; ++l)
            {
                std::vector<item>& cur = lv[l];
                std::vector<item> pk;
                if (l > 0)
                    for (uint64_t i = 0; i + 1 < lv[l - 1].size(); i += 2)
                        pk.push_back({lv[l - 1][i].w + lv[l - 1][i + 1].w, int(i), int(i + 1)});
                uint64_t i = 0, j = 0;
                while (i < m or j < pk.size())
                {
                    if (j >= pk.size() or (i < m and cnt[syms[i]] <= pk[j].w))
                        cur.push_back({cnt[syms[i]], syms[i], -1}), ++i;
                    else
                        cur.push_back(pk[j++]);
                }
            }

            std::vector<std::pair<uint32_t, int>> todo;
            for (uint64_t i = 0; i < 2 * m - 2; ++i)
                todo.push_back({max_code_len - 1, int(i)});
            while (not todo.empty())
            {
                const auto [l, i] = todo.back();
                todo.pop_back();
                const item& it = lv[l][uint64_t(i)];
                if (it.b < 0)
                    len[it.a]++;
                else
                    todo.push_back({l - 1, it.a}), todo.push_back({l - 1, it.b});
            }

            uint64_t bits = 0;
            for (int a : syms)
                bits += uint64_t(cnt[a]) * len[a];
            return bits;
        }

        // Canonical code in (length, symbol) order, stored bit-reversed (decoders read LSB first)
        inline void codewords(const uint8_t* len, uint16_t* code)
        {
            uint32_t c = 0;
            for (uint32_t l = 1; l <= max_code_len; ++l)
            {
                for (uint32_t s = 0; s < symbol_cnt; ++s)
                    if (len[s] == l)
                    {
                        uint32_t r = 0;
                        for (uint32_t b = 0; b < l; ++b)
                            r |= ((c >> b) & 1) << (l - 1 - b);
                        code[s] = uint16_t(r);
                        ++c;
                    }
                c <<= 1;
            }
        }

        // Returns its size
        inline uint64_t write_table(uint8_t* dst, const uint8_t* len)
        {
            uint32_t present = 0;
            for (uint32_t a = 0; a < symbol_cnt; ++a)
                present += len[a] != 0;
            const uint64_t nibbles = (present + 1) / 2;
            if (bitmap_bytes + nibbles < flat_table_size - 1)
            {
                dst[0] = table_sparse;
                memset(dst + 1, 0, bitmap_bytes + nibbles);
                uint32_t j = 0;
                for (uint32_t a = 0; a < symbol_cnt; ++a)
                    if (len[a])
                    {
                        dst[1 + (a >> 3)] |= uint8_t(1u << (a & 7));
                        dst[1 + bitmap_bytes + (j >> 1)] |= uint8_t(len[a] << (4 * (j & 1)));
                        ++j;
                    }
                return 1 + bitmap_bytes + nibbles;
            }
            dst[0] = table_flat;
            memset(dst + 1, 0, flat_table_size - 1);
            for (uint32_t a = 0; a < symbol_cnt; ++a)
                dst[1 + (a >> 1)] |= uint8_t(len[a] << (4 * (a & 1)));
            return flat_table_size;
        }

        // Symbols are coded back to front.
        // Decode index j is `s[n - 1 - j]`, in bit string `j % payload_split`.
        // Returns bytes written.
        inline uint64_t
        write_streams(uint8_t* dst, const uint8_t* s, uint64_t n, const uint8_t* len)
        {
            uint16_t code[symbol_cnt] = {};
            codewords(len, code);
            uint64_t pos = 0, sz[payload_split];
            for (uint32_t k = 0; k < payload_split; ++k)
            {
                const uint64_t start = pos;
                uint64_t acc = 0;
                uint32_t nb = 0;
                for (uint64_t j = k; j < n; j += payload_split)
                {
                    const uint8_t a = s[n - 1 - j];
                    acc |= uint64_t(code[a]) << nb;
                    nb += len[a];
                    while (nb >= 8)
                        dst[pos++] = uint8_t(acc), acc >>= 8, nb -= 8;
                }
                if (nb)
                    dst[pos++] = uint8_t(acc);
                sz[k] = pos - start;
            }
            for (uint32_t k = 0; k + 1 < payload_split; ++k)
                for (uint32_t b = 0; b < split_size_bytes; ++b)
                    dst[pos++] = uint8_t(sz[k] >> (8 * b));
            return pos;
        }

        // `dst` needs `bound(n)` bytes. Returns bytes written.
        inline uint64_t encode(uint8_t* dst, const uint8_t* s, uint64_t n)
        {
            uint32_t cnt[symbol_cnt] = {};
            for (uint64_t i = 0; i < n; ++i)
                cnt[s[i]]++;
            uint8_t len[symbol_cnt];
            code_lengths(cnt, len);
            const uint64_t tb = write_table(dst, len);
            return tb + write_streams(dst + tb, s, n, len);
        }
    } // namespace huffman_encoder
} // namespace misa77

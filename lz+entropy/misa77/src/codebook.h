// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include <cstdint>

namespace misa77
{
    namespace heavy
    {
        enum class kind : uint8_t
        {
            L, // literals only
            R, // match at the repeat distance
            S, // new distance, 1 offset byte
            N, // new distance, 2 offset bytes
            F, // new distance, 3 offset bytes
        };

        struct entry
        {
            kind k;
            uint32_t lit;
            uint32_t len;
        };

        constexpr uint32_t offset_bytes(kind k)
        {
            return k == kind::S ? 1 : k == kind::N ? 2 : k == kind::F ? 3 : 0;
        }

        inline constexpr uint32_t codebook_size = 256;

        // index = cmd byte

        // clang-format off

        // general
        inline constexpr entry codebook_a[codebook_size] = {
            {kind::L, 1, 0}, {kind::L, 2, 0}, {kind::L, 4, 0}, {kind::L, 8, 0}, // 0..3
            {kind::L, 16, 0}, {kind::L, 32, 0}, {kind::L, 64, 0}, {kind::L, 128, 0}, // 4..7
            {kind::L, 256, 0}, {kind::L, 512, 0}, {kind::L, 1024, 0}, {kind::L, 2048, 0}, // 8..11
            {kind::L, 4096, 0}, {kind::L, 8192, 0}, {kind::L, 16384, 0}, {kind::L, 32768, 0}, // 12..15
            {kind::R, 0, 16}, {kind::R, 0, 32}, {kind::R, 0, 64}, {kind::R, 0, 128}, // 16..19
            {kind::R, 0, 256}, {kind::R, 0, 512}, {kind::R, 0, 1024}, {kind::R, 0, 2048}, // 20..23
            {kind::R, 0, 4096}, {kind::R, 0, 8192}, {kind::R, 0, 16384}, {kind::R, 0, 32768}, // 24..27
            {kind::R, 0, 65536}, {kind::R, 1, 3}, {kind::N, 0, 8}, {kind::N, 0, 9}, // 28..31
            {kind::N, 0, 32}, {kind::N, 0, 7}, {kind::N, 0, 6}, {kind::R, 1, 6}, // 32..35
            {kind::N, 0, 10}, {kind::F, 0, 11}, {kind::F, 0, 10}, {kind::N, 1, 5}, // 36..39
            {kind::F, 0, 9}, {kind::F, 0, 12}, {kind::N, 0, 11}, {kind::R, 1, 7}, // 40..43
            {kind::R, 1, 15}, {kind::F, 0, 8}, {kind::N, 0, 5}, {kind::F, 1, 7}, // 44..47
            {kind::N, 0, 12}, {kind::F, 0, 32}, {kind::F, 0, 13}, {kind::N, 1, 7}, // 48..51
            {kind::N, 3, 5}, {kind::F, 0, 7}, {kind::R, 1, 11}, {kind::S, 0, 32}, // 52..55
            {kind::N, 0, 13}, {kind::F, 1, 11}, {kind::N, 1, 6}, {kind::R, 1, 4}, // 56..59
            {kind::F, 0, 14}, {kind::F, 1, 9}, {kind::R, 1, 19}, {kind::N, 2, 6}, // 60..63
            {kind::R, 2, 5}, {kind::R, 2, 6}, {kind::S, 0, 7}, {kind::R, 1, 5}, // 64..67
            {kind::N, 0, 14}, {kind::N, 1, 9}, {kind::N, 1, 8}, {kind::S, 0, 6}, // 68..71
            {kind::S, 0, 5}, {kind::N, 0, 15}, {kind::F, 0, 6}, {kind::R, 2, 14}, // 72..75
            {kind::F, 0, 15}, {kind::R, 2, 3}, {kind::N, 2, 5}, {kind::S, 0, 8}, // 76..79
            {kind::R, 2, 10}, {kind::R, 1, 16}, {kind::N, 2, 7}, {kind::N, 1, 11}, // 80..83
            {kind::F, 3, 7}, {kind::N, 0, 16}, {kind::F, 1, 8}, {kind::R, 1, 14}, // 84..87
            {kind::R, 2, 4}, {kind::F, 1, 10}, {kind::N, 1, 10}, {kind::F, 0, 16}, // 88..91
            {kind::N, 5, 5}, {kind::R, 4, 3}, {kind::N, 2, 8}, {kind::F, 2, 6}, // 92..95
            {kind::S, 0, 4}, {kind::S, 0, 9}, {kind::N, 3, 7}, {kind::F, 1, 12}, // 96..99
            {kind::N, 2, 10}, {kind::N, 0, 17}, {kind::N, 3, 6}, {kind::S, 0, 10}, // 100..103
            {kind::F, 1, 6}, {kind::S, 1, 7}, {kind::N, 15, 5}, {kind::R, 1, 23}, // 104..107
            {kind::R, 1, 17}, {kind::N, 0, 18}, {kind::F, 0, 17}, {kind::R, 1, 8}, // 108..111
            {kind::F, 1, 15}, {kind::N, 7, 5}, {kind::N, 2, 9}, {kind::S, 1, 6}, // 112..115
            {kind::F, 1, 13}, {kind::N, 0, 19}, {kind::R, 3, 5}, {kind::F, 2, 8}, // 116..119
            {kind::F, 3, 9}, {kind::S, 0, 11}, {kind::R, 1, 13}, {kind::S, 1, 5}, // 120..123
            {kind::F, 2, 9}, {kind::F, 2, 7}, {kind::N, 0, 20}, {kind::R, 1, 10}, // 124..127
            {kind::R, 2, 7}, {kind::F, 2, 10}, {kind::N, 3, 8}, {kind::F, 0, 18}, // 128..131
            {kind::S, 1, 11}, {kind::N, 3, 9}, {kind::N, 4, 5}, {kind::F, 5, 7}, // 132..135
            {kind::S, 0, 12}, {kind::N, 4, 6}, {kind::N, 1, 12}, {kind::N, 0, 21}, // 136..139
            {kind::F, 3, 8}, {kind::R, 1, 9}, {kind::S, 1, 4}, {kind::R, 5, 3}, // 140..143
            {kind::R, 3, 4}, {kind::S, 0, 13}, {kind::N, 0, 22}, {kind::N, 13, 5}, // 144..147
            {kind::F, 0, 19}, {kind::N, 2, 11}, {kind::R, 3, 3}, {kind::F, 2, 11}, // 148..151
            {kind::S, 2, 6}, {kind::S, 10, 14}, {kind::R, 1, 12}, {kind::R, 2, 8}, // 152..155
            {kind::N, 1, 13}, {kind::N, 11, 5}, {kind::N, 1, 15}, {kind::F, 1, 14}, // 156..159
            {kind::N, 0, 24}, {kind::N, 4, 7}, {kind::F, 15, 5}, {kind::N, 0, 23}, // 160..163
            {kind::F, 3, 10}, {kind::F, 0, 20}, {kind::S, 3, 5}, {kind::N, 9, 5}, // 164..167
            {kind::F, 2, 12}, {kind::S, 0, 14}, {kind::R, 2, 9}, {kind::N, 4, 8}, // 168..171
            {kind::N, 5, 7}, {kind::N, 2, 14}, {kind::F, 3, 6}, {kind::R, 6, 10}, // 172..175
            {kind::R, 3, 6}, {kind::S, 1, 8}, {kind::F, 3, 11}, {kind::N, 14, 5}, // 176..179
            {kind::S, 0, 15}, {kind::N, 0, 25}, {kind::N, 0, 26}, {kind::N, 2, 12}, // 180..183
            {kind::N, 3, 10}, {kind::F, 2, 14}, {kind::F, 0, 21}, {kind::N, 0, 27}, // 184..187
            {kind::R, 2, 13}, {kind::S, 0, 16}, {kind::N, 0, 28}, {kind::R, 2, 11}, // 188..191
            {kind::R, 5, 11}, {kind::F, 4, 8}, {kind::N, 1, 14}, {kind::S, 1, 15}, // 192..195
            {kind::N, 3, 13}, {kind::F, 0, 22}, {kind::N, 4, 12}, {kind::N, 2, 13}, // 196..199
            {kind::N, 5, 6}, {kind::F, 2, 13}, {kind::F, 3, 13}, {kind::S, 2, 5}, // 200..203
            {kind::S, 2, 7}, {kind::N, 3, 11}, {kind::R, 2, 15}, {kind::F, 1, 5}, // 204..207
            {kind::F, 4, 7}, {kind::F, 0, 23}, {kind::R, 1, 18}, {kind::R, 14, 2}, // 208..211
            {kind::R, 2, 12}, {kind::F, 3, 12}, {kind::S, 2, 4}, {kind::N, 6, 5}, // 212..215
            {kind::F, 4, 9}, {kind::F, 4, 6}, {kind::N, 0, 29}, {kind::N, 4, 9}, // 216..219
            {kind::N, 0, 31}, {kind::F, 13, 5}, {kind::N, 0, 30}, {kind::N, 6, 6}, // 220..223
            {kind::N, 14, 4}, {kind::N, 3, 12}, {kind::S, 0, 17}, {kind::F, 4, 10}, // 224..227
            {kind::R, 2, 16}, {kind::F, 0, 24}, {kind::F, 3, 5}, {kind::S, 1, 9}, // 228..231
            {kind::F, 1, 16}, {kind::F, 11, 5}, {kind::R, 1, 31}, {kind::N, 1, 32}, // 232..235
            {kind::R, 4, 4}, {kind::F, 7, 7}, {kind::R, 7, 1}, {kind::F, 0, 26}, // 236..239
            {kind::F, 0, 25}, {kind::N, 4, 10}, {kind::S, 2, 10}, {kind::S, 3, 7}, // 240..243
            {kind::S, 0, 18}, {kind::R, 3, 8}, {kind::N, 12, 4}, {kind::R, 3, 7}, // 244..247
            {kind::N, 1, 16}, {kind::R, 3, 9}, {kind::R, 9, 3}, {kind::F, 0, 27}, // 248..251
            {kind::S, 0, 19}, {kind::N, 2, 15}, {kind::R, 0, 31}, {kind::F, 5, 9}, // 252..255
        };

        // for small inputs, see `codebook_s_lim` in format.h (no F)
        inline constexpr entry codebook_s[codebook_size] = {
            {kind::L, 1, 0}, {kind::L, 2, 0}, {kind::L, 4, 0}, {kind::L, 8, 0}, // 0..3
            {kind::L, 16, 0}, {kind::L, 32, 0}, {kind::L, 64, 0}, {kind::L, 128, 0}, // 4..7
            {kind::L, 256, 0}, {kind::L, 512, 0}, {kind::L, 1024, 0}, {kind::L, 2048, 0}, // 8..11
            {kind::L, 4096, 0}, {kind::L, 8192, 0}, {kind::L, 16384, 0}, {kind::L, 32768, 0}, // 12..15
            {kind::R, 0, 16}, {kind::R, 0, 32}, {kind::R, 0, 64}, {kind::R, 0, 128}, // 16..19
            {kind::R, 0, 256}, {kind::R, 0, 512}, {kind::R, 0, 1024}, {kind::R, 0, 2048}, // 20..23
            {kind::R, 0, 4096}, {kind::R, 0, 8192}, {kind::R, 0, 16384}, {kind::R, 0, 32768}, // 24..27
            {kind::R, 0, 65536}, {kind::R, 1, 3}, {kind::N, 0, 7}, {kind::N, 0, 8}, // 28..31
            {kind::N, 0, 6}, {kind::N, 1, 5}, {kind::N, 0, 9}, {kind::N, 1, 7}, // 32..35
            {kind::N, 0, 32}, {kind::N, 0, 5}, {kind::R, 1, 6}, {kind::N, 0, 10}, // 36..39
            {kind::N, 3, 5}, {kind::R, 1, 7}, {kind::N, 1, 6}, {kind::N, 0, 11}, // 40..43
            {kind::N, 2, 6}, {kind::N, 1, 8}, {kind::N, 2, 5}, {kind::N, 1, 9}, // 44..47
            {kind::S, 0, 5}, {kind::N, 0, 12}, {kind::R, 1, 4}, {kind::R, 1, 11}, // 48..51
            {kind::N, 2, 7}, {kind::S, 0, 6}, {kind::R, 1, 15}, {kind::N, 3, 6}, // 52..55
            {kind::S, 0, 7}, {kind::N, 3, 7}, {kind::S, 0, 32}, {kind::R, 1, 5}, // 56..59
            {kind::N, 5, 5}, {kind::S, 0, 4}, {kind::R, 2, 5}, {kind::N, 2, 8}, // 60..63
            {kind::N, 0, 13}, {kind::R, 2, 6}, {kind::R, 2, 3}, {kind::N, 4, 5}, // 64..67
            {kind::R, 1, 19}, {kind::N, 0, 14}, {kind::N, 1, 11}, {kind::N, 4, 6}, // 68..71
            {kind::N, 2, 10}, {kind::S, 0, 8}, {kind::R, 2, 10}, {kind::N, 1, 10}, // 72..75
            {kind::N, 0, 15}, {kind::N, 2, 9}, {kind::N, 7, 5}, {kind::S, 1, 7}, // 76..79
            {kind::N, 3, 8}, {kind::R, 2, 14}, {kind::R, 2, 4}, {kind::N, 4, 7}, // 80..83
            {kind::S, 1, 5}, {kind::N, 15, 5}, {kind::S, 1, 6}, {kind::N, 3, 9}, // 84..87
            {kind::N, 0, 16}, {kind::S, 1, 4}, {kind::R, 4, 3}, {kind::S, 0, 9}, // 88..91
            {kind::N, 4, 8}, {kind::N, 5, 7}, {kind::N, 5, 6}, {kind::N, 6, 5}, // 92..95
            {kind::S, 0, 10}, {kind::R, 1, 16}, {kind::N, 0, 17}, {kind::R, 1, 14}, // 96..99
            {kind::R, 1, 8}, {kind::N, 0, 18}, {kind::N, 9, 5}, {kind::N, 6, 6}, // 100..103
            {kind::S, 3, 5}, {kind::N, 0, 19}, {kind::S, 2, 6}, {kind::R, 5, 3}, // 104..107
            {kind::N, 11, 5}, {kind::N, 2, 11}, {kind::S, 0, 11}, {kind::N, 1, 12}, // 108..111
            {kind::S, 2, 4}, {kind::R, 1, 23}, {kind::N, 3, 10}, {kind::N, 0, 20}, // 112..115
            {kind::S, 1, 11}, {kind::R, 1, 10}, {kind::N, 4, 9}, {kind::R, 2, 7}, // 116..119
            {kind::S, 2, 5}, {kind::R, 3, 5}, {kind::N, 13, 5}, {kind::S, 0, 12}, // 120..123
            {kind::N, 1, 13}, {kind::N, 7, 6}, {kind::R, 1, 9}, {kind::N, 9, 6}, // 124..127
            {kind::S, 3, 4}, {kind::N, 8, 5}, {kind::N, 0, 21}, {kind::R, 3, 3}, // 128..131
            {kind::R, 1, 13}, {kind::N, 2, 12}, {kind::N, 1, 15}, {kind::N, 4, 12}, // 132..135
            {kind::N, 3, 11}, {kind::N, 6, 7}, {kind::S, 4, 4}, {kind::R, 1, 17}, // 136..139
            {kind::S, 1, 8}, {kind::N, 2, 14}, {kind::N, 5, 8}, {kind::S, 0, 13}, // 140..143
            {kind::N, 9, 7}, {kind::N, 0, 22}, {kind::N, 4, 10}, {kind::S, 2, 7}, // 144..147
            {kind::R, 6, 10}, {kind::R, 14, 2}, {kind::N, 7, 7}, {kind::S, 10, 14}, // 148..151
            {kind::R, 2, 8}, {kind::N, 1, 14}, {kind::N, 8, 6}, {kind::R, 2, 9}, // 152..155
            {kind::N, 2, 13}, {kind::N, 0, 24}, {kind::S, 0, 14}, {kind::N, 0, 4}, // 156..159
            {kind::R, 1, 12}, {kind::N, 3, 13}, {kind::N, 16, 5}, {kind::N, 14, 5}, // 160..163
            {kind::N, 0, 23}, {kind::S, 0, 15}, {kind::S, 3, 7}, {kind::N, 3, 12}, // 164..167
            {kind::N, 14, 6}, {kind::R, 2, 13}, {kind::R, 2, 15}, {kind::S, 3, 6}, // 168..171
            {kind::N, 10, 5}, {kind::R, 3, 4}, {kind::N, 5, 9}, {kind::S, 1, 15}, // 172..175
            {kind::N, 6, 8}, {kind::N, 0, 26}, {kind::N, 1, 4}, {kind::S, 1, 3}, // 176..179
            {kind::N, 4, 4}, {kind::N, 14, 4}, {kind::N, 0, 25}, {kind::R, 3, 6}, // 180..183
            {kind::N, 4, 11}, {kind::N, 12, 20}, {kind::S, 15, 3}, {kind::S, 0, 16}, // 184..187
            {kind::S, 4, 6}, {kind::N, 10, 6}, {kind::N, 8, 7}, {kind::S, 4, 5}, // 188..191
            {kind::N, 0, 27}, {kind::N, 12, 4}, {kind::S, 1, 9}, {kind::N, 16, 4}, // 192..195
            {kind::R, 2, 11}, {kind::N, 12, 5}, {kind::R, 2, 12}, {kind::R, 2, 16}, // 196..199
            {kind::R, 4, 4}, {kind::N, 2, 4}, {kind::S, 5, 5}, {kind::S, 5, 4}, // 200..203
            {kind::S, 2, 8}, {kind::N, 6, 10}, {kind::S, 2, 10}, {kind::N, 3, 4}, // 204..207
            {kind::N, 0, 28}, {kind::R, 9, 3}, {kind::S, 5, 7}, {kind::N, 7, 8}, // 208..211
            {kind::R, 5, 11}, {kind::N, 1, 16}, {kind::N, 6, 9}, {kind::N, 2, 15}, // 212..215
            {kind::R, 15, 1}, {kind::S, 6, 4}, {kind::S, 0, 17}, {kind::R, 1, 18}, // 216..219
            {kind::S, 13, 3}, {kind::N, 1, 19}, {kind::N, 15, 4}, {kind::S, 1, 10}, // 220..223
            {kind::N, 15, 6}, {kind::N, 13, 6}, {kind::S, 5, 3}, {kind::N, 0, 29}, // 224..227
            {kind::R, 7, 1}, {kind::S, 7, 5}, {kind::N, 8, 8}, {kind::N, 5, 10}, // 228..231
            {kind::R, 5, 7}, {kind::N, 0, 30}, {kind::N, 0, 31}, {kind::N, 1, 17}, // 232..235
            {kind::R, 6, 2}, {kind::S, 3, 8}, {kind::S, 0, 18}, {kind::R, 1, 21}, // 236..239
            {kind::S, 2, 14}, {kind::N, 11, 6}, {kind::S, 4, 7}, {kind::S, 11, 3}, // 240..243
            {kind::R, 2, 17}, {kind::R, 11, 3}, {kind::R, 1, 31}, {kind::N, 16, 6}, // 244..247
            {kind::N, 12, 6}, {kind::N, 15, 3}, {kind::N, 4, 13}, {kind::N, 8, 4}, // 248..251
            {kind::S, 2, 9}, {kind::S, 0, 19}, {kind::N, 13, 4}, {kind::N, 5, 11}, // 252..255
        };

        // clang-format on
    } // namespace heavy

    namespace entropy
    {
        enum class kind : uint8_t
        {
            R, // match at the last distance (or no match)
            N, // new distance
        };

        struct entry
        {
            kind k;
            uint32_t lit;
            uint32_t len;
        };

        inline constexpr uint32_t codebook_size = 256;

        // index = cmd byte

        // clang-format off

        inline constexpr entry codebook[codebook_size] = {
            {kind::R, 1, 0}, {kind::R, 2, 0}, {kind::R, 3, 0}, {kind::R, 4, 0}, // 0..3
            {kind::R, 5, 0}, {kind::R, 6, 0}, {kind::R, 7, 0}, {kind::R, 8, 0}, // 4..7
            {kind::R, 9, 0}, {kind::R, 10, 0}, {kind::R, 11, 0}, {kind::R, 12, 0}, // 8..11
            {kind::R, 13, 0}, {kind::R, 14, 0}, {kind::R, 15, 0}, {kind::R, 0, 1}, // 12..15
            {kind::R, 1, 1}, {kind::R, 2, 1}, {kind::R, 3, 1}, {kind::R, 4, 1}, // 16..19
            {kind::R, 5, 1}, {kind::R, 6, 1}, {kind::R, 7, 1}, {kind::R, 8, 1}, // 20..23
            {kind::R, 9, 1}, {kind::R, 11, 1}, {kind::R, 12, 1}, {kind::R, 13, 1}, // 24..27
            {kind::R, 15, 1}, {kind::R, 0, 2}, {kind::R, 1, 2}, {kind::R, 2, 2}, // 28..31
            {kind::R, 3, 2}, {kind::R, 4, 2}, {kind::R, 6, 2}, {kind::R, 10, 2}, // 32..35
            {kind::R, 0, 3}, {kind::R, 1, 3}, {kind::R, 2, 3}, {kind::R, 3, 3}, // 36..39
            {kind::R, 5, 3}, {kind::R, 9, 3}, {kind::R, 0, 4}, {kind::R, 1, 4}, // 40..43
            {kind::R, 2, 4}, {kind::R, 3, 4}, {kind::R, 4, 4}, {kind::R, 8, 4}, // 44..47
            {kind::R, 12, 4}, {kind::R, 0, 5}, {kind::R, 1, 5}, {kind::R, 2, 5}, // 48..51
            {kind::R, 3, 5}, {kind::R, 7, 5}, {kind::R, 11, 5}, {kind::R, 0, 6}, // 52..55
            {kind::R, 1, 6}, {kind::R, 2, 6}, {kind::R, 3, 6}, {kind::R, 4, 6}, // 56..59
            {kind::R, 0, 7}, {kind::R, 1, 7}, {kind::R, 2, 7}, {kind::R, 5, 7}, // 60..63
            {kind::R, 0, 8}, {kind::R, 1, 8}, {kind::R, 4, 8}, {kind::R, 8, 8}, // 64..67
            {kind::R, 0, 9}, {kind::R, 1, 9}, {kind::R, 3, 9}, {kind::R, 0, 10}, // 68..71
            {kind::R, 1, 10}, {kind::R, 2, 10}, {kind::R, 0, 11}, {kind::R, 1, 11}, // 72..75
            {kind::R, 15, 11}, {kind::R, 0, 12}, {kind::R, 1, 12}, {kind::R, 4, 12}, // 76..79
            {kind::R, 15, 12}, {kind::R, 0, 13}, {kind::R, 1, 13}, {kind::R, 2, 13}, // 80..83
            {kind::R, 3, 13}, {kind::R, 0, 14}, {kind::R, 1, 14}, {kind::R, 2, 14}, // 84..87
            {kind::R, 0, 15}, {kind::R, 1, 15}, {kind::R, 0, 16}, {kind::R, 0, 17}, // 88..91
            {kind::R, 1, 17}, {kind::R, 0, 18}, {kind::R, 1, 18}, {kind::R, 2, 18}, // 92..95
            {kind::R, 0, 19}, {kind::R, 1, 19}, {kind::R, 2, 19}, {kind::R, 0, 20}, // 96..99
            {kind::R, 1, 20}, {kind::R, 2, 20}, {kind::R, 0, 21}, {kind::R, 1, 21}, // 100..103
            {kind::R, 2, 21}, {kind::R, 3, 21}, {kind::R, 0, 22}, {kind::R, 2, 22}, // 104..107
            {kind::R, 3, 22}, {kind::R, 4, 22}, {kind::R, 0, 23}, {kind::R, 1, 23}, // 108..111
            {kind::R, 3, 23}, {kind::R, 0, 24}, {kind::R, 2, 24}, {kind::R, 4, 24}, // 112..115
            {kind::R, 0, 25}, {kind::R, 3, 25}, {kind::R, 0, 26}, {kind::R, 2, 26}, // 116..119
            {kind::R, 0, 27}, {kind::R, 1, 27}, {kind::R, 0, 28}, {kind::R, 0, 29}, // 120..123
            {kind::R, 0, 30}, {kind::R, 1, 30}, {kind::R, 2, 30}, {kind::R, 0, 31}, // 124..127
            {kind::R, 1, 31}, {kind::R, 0, 32}, {kind::R, 4, 32}, {kind::N, 0, 3}, // 128..131
            {kind::N, 1, 3}, {kind::N, 2, 3}, {kind::N, 3, 3}, {kind::N, 4, 3}, // 132..135
            {kind::N, 5, 3}, {kind::N, 6, 3}, {kind::N, 7, 3}, {kind::N, 8, 3}, // 136..139
            {kind::N, 9, 3}, {kind::N, 10, 3}, {kind::N, 11, 3}, {kind::N, 12, 3}, // 140..143
            {kind::N, 13, 3}, {kind::N, 14, 3}, {kind::N, 15, 3}, {kind::N, 0, 4}, // 144..147
            {kind::N, 1, 4}, {kind::N, 2, 4}, {kind::N, 3, 4}, {kind::N, 4, 4}, // 148..151
            {kind::N, 5, 4}, {kind::N, 6, 4}, {kind::N, 7, 4}, {kind::N, 8, 4}, // 152..155
            {kind::N, 9, 4}, {kind::N, 10, 4}, {kind::N, 11, 4}, {kind::N, 12, 4}, // 156..159
            {kind::N, 13, 4}, {kind::N, 0, 5}, {kind::N, 1, 5}, {kind::N, 2, 5}, // 160..163
            {kind::N, 3, 5}, {kind::N, 4, 5}, {kind::N, 5, 5}, {kind::N, 7, 5}, // 164..167
            {kind::N, 11, 5}, {kind::N, 12, 5}, {kind::N, 0, 6}, {kind::N, 1, 6}, // 168..171
            {kind::N, 2, 6}, {kind::N, 3, 6}, {kind::N, 4, 6}, {kind::N, 6, 6}, // 172..175
            {kind::N, 10, 6}, {kind::N, 0, 7}, {kind::N, 1, 7}, {kind::N, 2, 7}, // 176..179
            {kind::N, 3, 7}, {kind::N, 4, 7}, {kind::N, 5, 7}, {kind::N, 0, 8}, // 180..183
            {kind::N, 1, 8}, {kind::N, 2, 8}, {kind::N, 3, 8}, {kind::N, 4, 8}, // 184..187
            {kind::N, 8, 8}, {kind::N, 0, 9}, {kind::N, 1, 9}, {kind::N, 2, 9}, // 188..191
            {kind::N, 3, 9}, {kind::N, 0, 10}, {kind::N, 1, 10}, {kind::N, 2, 10}, // 192..195
            {kind::N, 3, 10}, {kind::N, 4, 10}, {kind::N, 0, 11}, {kind::N, 1, 11}, // 196..199
            {kind::N, 2, 11}, {kind::N, 3, 11}, {kind::N, 0, 12}, {kind::N, 1, 12}, // 200..203
            {kind::N, 2, 12}, {kind::N, 3, 12}, {kind::N, 4, 12}, {kind::N, 0, 13}, // 204..207
            {kind::N, 1, 13}, {kind::N, 2, 13}, {kind::N, 3, 13}, {kind::N, 0, 14}, // 208..211
            {kind::N, 1, 14}, {kind::N, 2, 14}, {kind::N, 0, 15}, {kind::N, 1, 15}, // 212..215
            {kind::N, 2, 15}, {kind::N, 0, 16}, {kind::N, 1, 16}, {kind::N, 4, 16}, // 216..219
            {kind::N, 8, 16}, {kind::N, 0, 17}, {kind::N, 1, 17}, {kind::N, 3, 17}, // 220..223
            {kind::N, 0, 18}, {kind::N, 0, 19}, {kind::N, 1, 19}, {kind::N, 0, 20}, // 224..227
            {kind::N, 10, 20}, {kind::N, 11, 20}, {kind::N, 12, 20}, {kind::N, 0, 21}, // 228..231
            {kind::N, 1, 21}, {kind::N, 3, 21}, {kind::N, 0, 22}, {kind::N, 4, 22}, // 232..235
            {kind::N, 0, 23}, {kind::N, 1, 23}, {kind::N, 0, 24}, {kind::N, 2, 24}, // 236..239
            {kind::N, 4, 24}, {kind::N, 0, 25}, {kind::N, 1, 25}, {kind::N, 3, 25}, // 240..243
            {kind::N, 0, 26}, {kind::N, 2, 26}, {kind::N, 0, 27}, {kind::N, 1, 27}, // 244..247
            {kind::N, 0, 28}, {kind::N, 0, 29}, {kind::N, 0, 30}, {kind::N, 2, 30}, // 248..251
            {kind::N, 1, 31}, {kind::N, 0, 32}, {kind::N, 1, 32}, {kind::N, 4, 32}, // 252..255
        };

        // clang-format on
    } // namespace entropy
} // namespace misa77

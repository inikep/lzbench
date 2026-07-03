// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "misa77/experimental.h"

#include <array>
#include <cstdint>

namespace misa77
{
    namespace experimental
    {
        namespace params
        {
            // BASE is the weight of the size param, other weights are implicitly scaled relative to
            // it (well, they were when I ran an offline search)
            constexpr uint64_t BASE = 1024;

            // param aggregate order:
            // {use_default, size, block, short4_7, short8_15, lit7, lit17, lit33}

            // Default compressor
            inline constexpr param p_default = {true, 0, 0, 0, 0, 0, 0, 0};

            // Tight tier (get >= default decode speed, might slightly worsen ratio)
            inline constexpr param tight_general = {false, BASE, 2500, 0, 0, 11253, 25502, 0};
            inline constexpr param tight_short = {false, BASE, 0, 1076, 732, 3393, 37798, 27218};
            inline constexpr param tight_deeplit = {false, BASE, 2065, 0, 206, 4617, 21109, 41277};
            inline constexpr param tight_guarded = {false, BASE, 4096, 0, 0, 16384, 65536, 131072};
            inline constexpr param tight_lean = {false, BASE, 0, 540, 0, 6012, 0, 0};

            // Loose tier (very likely to produce significantly greater decode speed than default,
            // moderate ratio loss)
            inline constexpr param loose_general = {false, BASE, 5292, 0, 0, 13367, 0, 130367};
            inline constexpr param loose_champion = {false, BASE, 7254, 264, 0, 32768, 0, 129083};
            inline constexpr param loose_blockonly = {false, BASE, 3381, 219, 0, 0, 0, 0};
            inline constexpr param loose_longlit = {false, BASE, 4474, 0, 1147, 2514, 0, 96248};

            // Both tiers search through all 10 candidates
            inline constexpr std::array<param, 10> all_candidates = {{p_default,
                                                                      tight_general,
                                                                      tight_short,
                                                                      tight_deeplit,
                                                                      tight_guarded,
                                                                      tight_lean,
                                                                      loose_general,
                                                                      loose_champion,
                                                                      loose_blockonly,
                                                                      loose_longlit}};
        } // namespace params
    } // namespace experimental
} // namespace misa77

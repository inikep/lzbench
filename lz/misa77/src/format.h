// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include <cstdint>

namespace misa77
{
    // Width in bytes of several unconditional operations we perform.
    inline constexpr uint64_t vector_width = 32;

    // (`raw file size <= small_lim`) => small mode, we just write the raw bytes.
    inline constexpr uint32_t small_lim = 32;

    // When we find a match, we check `lookahead` positions ahead for a potentially better match.
    inline constexpr uint64_t lookahead = 2;

    // The distance in bytes by which the start of the active hashtable window lags the pointer.
    // Don't set this too high, as there were a lot of matches for dis in [32, 512].
    inline constexpr uint64_t hashtab_lag = 32;
    static_assert(hashtab_lag >= vector_width);

    // If we're asked to copy a match after having processed `c` bytes of raw data, then the match
    // will begin in `[c - dis_lim - hashtab_lag, c - hashtab_lag)`.
    inline constexpr uint64_t dis_lim = (uint32_t(1) << 16);

    // We have exactly 2 bytes to indicate distance.
    static_assert(dis_lim <= (1 << 16));

    inline constexpr uint32_t min_match_len = 4;

    // `min_match_len >= 4` is needed for the compression bound to hold.
    static_assert(min_match_len >= 4);

    inline constexpr uint32_t max_match_len = 32;

    // One `cyccpy` call should deal with the entire match.
    static_assert(max_match_len <= vector_width);

    // Lowerbound for the number of literals in the final raw suffix.
    // This MUST be >= `vector_width` because:
    // 1. During decompression, `cyccpy` performs unconditional writes in the destination buffer,
    // and we're using this suffix as padding that we can safely overwrite!!!
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
} // namespace misa77
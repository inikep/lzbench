// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include <cstdint>

namespace misa77
{
    // `1e17`
    inline constexpr uint64_t max_src_size = 100'000'000'000'000'000;

    class config
    {
    public:
        // Only integers in [0, max_level] are valid levels
        static constexpr uint8_t max_level = 1;
        static constexpr uint8_t default_level = 1;
        static_assert(default_level <= max_level);

        uint8_t level;
        config() : level(default_level) {}
        explicit config(uint8_t l) : level(l) {}
    };

    // Upper-bound on compressed size (in bytes) for any input of size `src_size` bytes.
    // Use to size destination buffer.
    // PRECONDITION: `src_size <= max_src_size`
    uint64_t compress_bound(uint64_t src_size);

    // Returns number of bytes written to `dst`, and 0 on failure.
    // `cfg.level` selects the compressor to be used (all levels conform to the same format).
    // PRECONDITION: `src_size <= max_src_size` and `0 <= cfg.level <= config::max_level`
    uint64_t compress(const uint8_t* __restrict src,
                      uint64_t src_size,
                      uint8_t* __restrict dst,
                      uint64_t dst_cap,
                      config cfg = config());

    // Returns the exact size (in bytes) of the file that the given compressed file will be
    // decompressed to.
    // Only call this on buffers that were compressed in the misa77 format, as it
    // requires the size of the buffer to be >= 8.
    uint64_t decompressed_size(const uint8_t* src);

    // Minimum size of buffer required to decompress a compressed file that corresponds to an
    // original file of `src_size` bytes.
    // Usage: decompressed_buffer_bound(decompressed_size(src))
    uint64_t decompressed_buffer_bound(uint64_t src_size);

    // Returns number of bytes written to `dst`, and 0 on failure.
    // PRECONDITION: `src` and `src_size` must correspond to a valid misa77 stream.
    // `decompress` does not validate input, and malformed/malicious data can lead to UB or worse.
    uint64_t decompress(const uint8_t* __restrict src,
                        uint64_t src_size,
                        uint8_t* __restrict dst,
                        uint64_t dst_cap);
} // namespace misa77
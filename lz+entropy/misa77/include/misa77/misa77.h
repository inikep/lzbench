// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include <cstdint>

// Library version
#define MISA77_VERSION_MAJOR 0
#define MISA77_VERSION_MINOR 8
#define MISA77_VERSION_PATCH 0
#define MISA77_VERSION_NUMBER                                                                      \
    (MISA77_VERSION_MAJOR * 10000 + MISA77_VERSION_MINOR * 100 + MISA77_VERSION_PATCH)
#define MISA77_STR_HELPER(x) #x
#define MISA77_STR(x) MISA77_STR_HELPER(x)
#define MISA77_VERSION_STR                                                                         \
    MISA77_STR(MISA77_VERSION_MAJOR)                                                               \
    "." MISA77_STR(MISA77_VERSION_MINOR) "." MISA77_STR(MISA77_VERSION_PATCH)

namespace misa77
{
    // `1e15`
    inline constexpr uint64_t max_src_size = 1'000'000'000'000'000;

    // compressor config
    class config
    {
    public:
        // Only integers in [min_level, max_level] are valid levels
        static constexpr int8_t min_level = -1;
        static constexpr int8_t max_level = 5;
        static constexpr int8_t default_level = 1;
        static_assert(min_level <= default_level and default_level <= max_level);

        // Light levels in [min_level, heavy_lb), heavy levels in [heavy_lb, entropy_lb), entropy
        // levels in [entropy_lb, max_level]
        static constexpr int8_t heavy_lb = 4;
        static constexpr int8_t entropy_lb = 5;

        int8_t level;
        config() : level(default_level) {}
        explicit config(int8_t l) : level(l) {}
    };

    // Upper-bound on compressed size (in bytes) for any input of size `src_size` bytes.
    // Use to size destination buffer. The bound depends on the format `cfg.level` emits, so
    // pass the same `cfg` you will pass to `compress`.
    // PRECONDITION: `src_size <= max_src_size`
    uint64_t compress_bound(uint64_t src_size, config cfg);

    // Returns number of bytes written to `dst`, and 0 on failure.
    // `cfg.level` selects the compressor to be used. Levels below `config::heavy_lb` emit the
    // light format, levels in [`config::heavy_lb`, `config::entropy_lb`) emit the heavy format,
    // and levels at or above `config::entropy_lb` emit the entropy format.
    // PRECONDITION: `src_size <= max_src_size` and
    // `config::min_level <= cfg.level <= config::max_level`
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
    // If the stream in `src` doesn't correspond to a valid misa77 stream, the function terminates
    // gracefully (no UB).
    // `src_size` must be the exact size of the compressed stream (what `compress` returned), and
    // not just the size of the buffer that holds it.
    uint64_t decompress(const uint8_t* __restrict src,
                        uint64_t src_size,
                        uint8_t* __restrict dst,
                        uint64_t dst_cap);
} // namespace misa77

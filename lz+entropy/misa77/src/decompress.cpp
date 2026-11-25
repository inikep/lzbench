// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#include "decompress_dispatch.h"
#include "format.h"
#include "misa77/misa77.h"
#include "util.h"

#include <cstdint>

#if defined(__x86_64__)
#include <cpuid.h>
#endif

namespace misa77
{
#if defined(__x86_64__)
    // `__builtin_cpu_supports("lzcnt")` is rejected by clang <= 14 and Apple clang
    static bool has_lzcnt()
    {
        unsigned a, b, c, d;
        return __get_cpuid(0x80000001, &a, &b, &c, &d) and (c & (1u << 5));
    }
#endif

    // sanity check
    static_assert(vector_width == 32);
    static_assert(light::hashtab_lag >= vector_width);
    static_assert(vector_width >= light::max_match_len);
    static_assert(light::literal_suffix >= light::dec_literal_copy);

    // Returns the exact size (in bytes) of the file that the given compressed file will be
    // decompressed to.
    // Only call this on buffers that were compressed in the misa77 format, as it
    // requires the size of the buffer to be >= 8.
    uint64_t decompressed_size(const uint8_t* src)
    {
        return loadu8(src) & 0x00FF'FFFF'FFFF'FFFF;
    }

    // Minimum size of buffer required to decompress a compressed file that corresponds to an
    // original file of `src_size` bytes.
    // Usage: decompressed_buffer_bound(decompressed_size(src))
    uint64_t decompressed_buffer_bound(uint64_t src_size)
    {
        return src_size;
    }

    // Returns number of bytes written to `dst`, and 0 on failure.
    // Resolves the best decode path once per call, depending on the ISA.
    // If the stream in `src` doesn't correspond to a valid misa77 stream, the function terminates
    // gracefully (no UB).
    uint64_t decompress(const uint8_t* __restrict src,
                        uint64_t src_size,
                        uint8_t* __restrict dst,
                        uint64_t dst_cap)
    {
        // no valid stream is shorter than 8 bytes (smallest is an empty file's bare header).
        if (src_size < 8)
            return 0;

#if defined(__x86_64__)
        // the entropy format's AVX2 kernels also need BMI1, BMI2 and LZCNT
        const bool entropy = (src[7] & format_mask) == format_entropy;
        if (__builtin_cpu_supports("avx2") and
            (not entropy or (__builtin_cpu_supports("bmi") and __builtin_cpu_supports("bmi2") and
                             has_lzcnt())))
            return decompress_avx2(src, src_size, dst, dst_cap);
        return decompress_sse2(src, src_size, dst, dst_cap);
#elif defined(__aarch64__)
        // NEON is guaranteed on Aarch64
        return decompress_neon(src, src_size, dst, dst_cap);
#else
        return decompress_portable(src, src_size, dst, dst_cap);
#endif
    }

} // namespace misa77
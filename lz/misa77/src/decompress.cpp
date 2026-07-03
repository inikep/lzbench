// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#include "decompress_dispatch.h"
#include "format.h"
#include "misa77/misa77.h"
#include "util.h"

#include <cstdint>

namespace misa77
{
    static_assert(vector_width == 32);
    static_assert(hashtab_lag >= vector_width);
    static_assert(vector_width >= max_match_len);
    static_assert(literal_suffix >= dec_literal_copy);

    // Returns the exact size (in bytes) of the file that the given compressed file will be
    // decompressed to.
    // Only call this on buffers that were compressed in the misa77 format, as it
    // requires the size of the buffer to be >= 8.
    uint64_t decompressed_size(const uint8_t* src)
    {
        return loadu8(src);
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
    // PRECONDITION: `src` and `src_size` must correspond to a valid misa77 stream.
    // `decompress` does not validate input, and malformed/malicious data can lead to UB or worse.
    uint64_t decompress(const uint8_t* __restrict src,
                        uint64_t src_size,
                        uint8_t* __restrict dst,
                        uint64_t dst_cap)
    {
#if defined(__x86_64__)
        if (__builtin_cpu_supports("avx2"))
            return decompress_avx2(src, src_size, dst, dst_cap);
        return decompress_sse2(src, src_size, dst, dst_cap);
#else
        return decompress_portable(src, src_size, dst, dst_cap);
#endif
    }

} // namespace misa77
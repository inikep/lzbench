// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "format.h"

#include <cstddef>
#include <cstdint>

namespace misa77::heavy
{
    // Layout is hard-coded in x86_64.inc and aarch64.inc.
    struct kernel_io
    {
        const uint8_t* cp;      //  0 cmd cursor
        const uint8_t* cp_end;  //  8
        const uint8_t* op;      // 16 offset cursor
        const uint8_t* op_end;  // 24
        const uint8_t* lp;      // 32 literal cursor
        const uint8_t* lp_end;  // 40
        const uint8_t* src_end; // 48
        uint8_t* out;           // 56
        uint8_t* out_lim;       // 64 dst + n - literal_suffix
        uint8_t* dst;           // 72
        uint64_t chk_end;       // 80 address: dst + max_dis of the codebook
        const void* tables;     // 88 kernel_tables
        uint64_t vneg;          // 96 min_dis - dis (min_dis = no distance yet)
        uint64_t scratch[5];    // 104 kernel-private
    };
    static_assert(offsetof(kernel_io, vneg) == 96 and sizeof(kernel_io) == 144);

    struct alignas(64) kernel_tables
    {
        // mask | big << 31 | ob << 32 | lit << 40 | (lit + len) << 48
        uint64_t pack[codebook_size];
        uint32_t full_lit[codebook_size];
        uint32_t full_len[codebook_size];

        constexpr explicit kernel_tables(const decode_tables& T) : pack(), full_lit(), full_len()
        {
            for (uint32_t c = 0; c < codebook_size; ++c)
            {
                const bool big = T.nb[c] & big_flag;
                pack[c] = uint64_t(T.mask[c]) | (big ? uint64_t(1) << 31 : 0) |
                          uint64_t(T.nb[c] & (big_flag - 1)) << 32 | uint64_t(T.lit[c]) << 40 |
                          uint64_t(T.lit[c] + T.len[c]) << 48;
                full_lit[c] = T.full_lit[c];
                full_len[c] = T.full_len[c];
            }
        }
    };
    static_assert(offsetof(kernel_tables, full_lit) == 2048 and
                  offsetof(kernel_tables, full_len) == 3072);

    inline constexpr kernel_tables kernel_tables_a{tables_a};
    inline constexpr kernel_tables kernel_tables_s{tables_s};

    // what the kernels assume of a codebook
    constexpr bool kernel_codebook_ok(const decode_tables& T)
    {
        for (uint32_t c = 0; c < codebook_size; ++c)
        {
            const uint32_t ob = T.nb[c] & (big_flag - 1);
            if (T.nb[c] & big_flag)
            {
                // no offset bytes; literals or a match, not both; copied 32 bytes at a time
                if (ob != 0 or (T.full_lit[c] != 0 and T.full_len[c] != 0) or
                    T.full_lit[c] % vector_width != 0 or T.full_len[c] % vector_width != 0)
                    return false;
            }
            // a new distance always comes with a match
            else if (T.lit[c] > dec_literal_copy or T.len[c] > max_small_len or ob >= offset_load or
                     (ob != 0 and T.len[c] == 0))
                return false;
        }
        return T.max_dis <= (uint64_t(1) << 24) + vector_width;
    }
    static_assert(kernel_codebook_ok(tables_a) and kernel_codebook_ok(tables_s));

    // the kernel table of codebook `T`
    constexpr const kernel_tables& kernel_tables_for(const decode_tables& T)
    {
        return &T == &tables_s ? kernel_tables_s : kernel_tables_a;
    }

    // the kernel is only used when src and dst are at or above this address
    inline constexpr uint64_t kernel_min_addr = uint64_t(1) << 16;
} // namespace misa77::heavy

#if defined(MISA77_ASM_KERNELS) && defined(__x86_64__)
// Token loops in x86_64.inc, one block per call.
// Return 0 = reject, 1 = block's tokens done, 2 = decode one token with the checked step.
// Disable C++ name mangling.
extern "C"
{
    uint64_t misa77_heavy_safe_kernel_avx2(misa77::heavy::kernel_io* io) noexcept;
    uint64_t misa77_heavy_safe_kernel_sse2(misa77::heavy::kernel_io* io) noexcept;
}
#elif defined(MISA77_ASM_KERNELS) && defined(__aarch64__)
// Token loop in aarch64.inc, one block per call. Same return values.
extern "C"
{
    uint64_t misa77_heavy_safe_kernel_neon(misa77::heavy::kernel_io* io) noexcept;
}
#endif

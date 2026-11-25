// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "entropy/huffman_decoder.h"
#include "format.h"

#include <cstddef>
#include <cstdint>

namespace misa77::entropy
{
    // Per command byte: {lit, lit + len}. The kind is `c >= first_new`.
    struct alignas(64) command_table
    {
        uint8_t pack[2 * codebook_size];

        constexpr explicit command_table(const entry (&cb)[codebook_size]) : pack()
        {
            for (uint32_t c = 0; c < codebook_size; ++c)
            {
                pack[2 * c] = uint8_t(cb[c].lit);
                pack[2 * c + 1] = uint8_t(cb[c].lit + cb[c].len);
            }
        }
    };
    inline constexpr command_table commands{codebook};

    static_assert(max_lit + max_match_len < 256);
    // FIRST_NEW in x86_64.inc and aarch64.inc
    static_assert(first_new == 131);

    // Most bytes a command moves `out` by; a command writes less than this far past its start.
    inline constexpr uint64_t max_cmd_out = max_lit + max_match_len;
    static_assert(max_cmd_out == 47 and max_cmd_out < suffix_size);

    // Layout is hard-coded in x86_64.inc and aarch64.inc.
    struct kernel_io
    {
        const uint8_t* cmd;     //  0
        const uint8_t* cmd_end; //  8
        const int32_t* off;     // 16 negated distances, one per N command
        const uint8_t* lit;     // 24 literal cursor
        uint8_t* out;           // 32
        uint8_t* out_end;       // 40 the block's end
        const uint8_t* dst;     // 48
        int64_t r0;             // 56 negated repeat distance
        const void* table;      // 64 command_table
    };
    static_assert(offsetof(kernel_io, table) == 64 and sizeof(kernel_io) == 72);

    // kernel variant bits
    inline constexpr uint64_t kernel_sub = 1;   // SUB block
    inline constexpr uint64_t kernel_check = 2; // check every match source against `dst`

    // expand.mask[f][i] = rank of bit i among the set bits of f, or 0x80 (selects zero)
    struct alignas(64) expand_table
    {
        uint8_t mask[256][8];
        uint8_t cnt[256];

        constexpr expand_table() : mask(), cnt()
        {
            for (uint32_t f = 0; f < 256; ++f)
            {
                uint8_t k = 0;
                for (uint32_t i = 0; i < 8; ++i)
                    mask[f][i] = ((f >> i) & 1) ? k++ : uint8_t(0x80);
                cnt[f] = k;
            }
        }
    };
    inline constexpr expand_table expand{};
    static_assert(offsetof(expand_table, cnt) == 2048);

    // Layout is hard-coded in x86_64.inc and aarch64.inc.
    struct merge_io
    {
        const uint8_t* plane[6]; //   0 o1, o2l, o2h, o3l, o3m, o3h
        const uint8_t* f1;       //  48
        const uint8_t* f2;       //  56
        uint64_t off_cnt;        //  64
        uint64_t wide_cnt;       //  72 2- and 3-byte offsets
        uint8_t* tmp[3];         //  80 working memory, `merge_tmp_size` bytes in all
        int32_t* out;            // 104 `off_cnt + merge_out_pad` entries
        const void* expand;      // 112 expand_table
        uint64_t o3_cnt;         // 120 3-byte offsets
    };
    static_assert(offsetof(merge_io, out) == 104 and sizeof(merge_io) == 128);

    // what the merge kernels write past the offsets, in entries
    inline constexpr uint64_t merge_out_pad = 16;
    // working memory of the merge kernels for `wide_cnt` wide offsets
    constexpr uint64_t merge_tmp_size(uint64_t wide_cnt)
    {
        return 3 * (wide_cnt + 8) + 32;
    }

    // the kernels are only used when dst is at or above this address
    inline constexpr uint64_t kernel_min_addr = uint64_t(1) << 16;
} // namespace misa77::entropy

#if defined(MISA77_ASM_KERNELS) && defined(__x86_64__)
// Kernels, along with contracts, can be found in x86_64.inc.
// The AVX2 ones also need BMI1, BMI2 and LZCNT.
// Disable C++ name mangling.
extern "C"
{
    // Process all commands of one block (the core LZ loop). Return 0 = reject, 1 = done.
    uint64_t misa77_entropy_safe_kernel_avx2(misa77::entropy::kernel_io* io,
                                             uint64_t variant) noexcept;
    uint64_t misa77_entropy_safe_kernel_sse2(misa77::entropy::kernel_io* io,
                                             uint64_t variant) noexcept;
    // Whole rounds of Huffman symbols.
    void misa77_entropy_huffman_kernel_avx2(misa77::huffman_decoder::kernel_io* io) noexcept;
    void misa77_entropy_huffman_kernel_sse2(misa77::huffman_decoder::kernel_io* io) noexcept;
    // x[i] += x[i - stride] over [b, e)
    void misa77_entropy_delta_kernel_avx2(uint8_t* b, uint8_t* e, uint64_t stride) noexcept;
    void misa77_entropy_delta_kernel_sse2(uint8_t* b, uint8_t* e, uint64_t stride) noexcept;
    // Offset planes -> negated distances. Returns a lower bound of the smallest.
    int64_t misa77_entropy_merge_kernel_avx2(misa77::entropy::merge_io* io) noexcept;
    int64_t misa77_entropy_merge_kernel_sse2(misa77::entropy::merge_io* io) noexcept;
}
#endif

#if defined(MISA77_ASM_KERNELS) && defined(__aarch64__)
// Kernels in aarch64.inc; contracts there. ARMv8.0-A + NEON only.
extern "C"
{
    uint64_t misa77_entropy_safe_kernel_neon(misa77::entropy::kernel_io* io,
                                             uint64_t variant) noexcept;
    void misa77_entropy_huffman_kernel_neon(misa77::huffman_decoder::kernel_io* io) noexcept;
    void misa77_entropy_delta_kernel_neon(uint8_t* b, uint8_t* e, uint64_t stride) noexcept;
    int64_t misa77_entropy_merge_kernel_neon(misa77::entropy::merge_io* io) noexcept;
}
#endif

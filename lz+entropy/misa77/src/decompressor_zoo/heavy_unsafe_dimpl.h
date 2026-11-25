// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

// NOTE: Support for unsafe decompression was removed in 0.8.0!!!
// The following code is only preserved as a simple illustration of the hot loop of the decoder for
// the 0.7.0 heavy format.

#include "format.h"
#include "misa77/misa77.h"
#include "util.h"

#include <cstdint>
#include <cstring>

namespace misa77
{
    namespace heavy_unsafe_detail
    {
        using namespace heavy;

        template <class isa_lib, const decode_tables& T>
        uint64_t heavy_unsafe_actual_dimpl(const uint8_t* __restrict src,
                                           uint64_t original_size,
                                           uint8_t* __restrict dst)
        {
            const uint8_t* block = src;
            uint8_t* out = dst;
            uint8_t* out_end = out + original_size;
            uint64_t v = uint64_t(0) - min_dis;

            while (out < out_end)
            {
                const uint64_t tok_cnt = loadu4(block + tok_cnt_pos) & (cnt_lim - 1);
                const uint64_t off_cnt = loadu4(block + off_cnt_pos) & (cnt_lim - 1);
                const uint64_t lit_cnt = loadu4(block + lit_cnt_pos) & (cnt_lim - 1);

                // controls
                const uint8_t* cp = block + block_header_size;
                const uint8_t* cp_end = cp + tok_cnt;

                // offsets
                const uint8_t* op = cp + tok_cnt;

                // literals
                const uint8_t* lp = op + off_cnt;

                const uint8_t* nblock = lp + lit_cnt;

                while (cp < cp_end)
                {
                    // control byte
                    const uint64_t c = *cp;
                    cp++;

                    const uint64_t nb = T.nb[c];

                    // big token
                    if (nb & big_flag) [[unlikely]]
                    {
                        const uint64_t lit = T.full_lit[c];
                        const uint64_t len = T.full_len[c];

                        for (uint64_t i = 0; i < lit; i += vector_width)
                            isa_lib::copy32(out + i, lp + i);
                        out += lit, lp += lit;

                        const uint32_t m = T.mask[c];
                        // gcc won't emit a cmov for the plain v = (m ? loadu4(op) & m : v)
                        v = (loadu4(op) & m) | (v & (m ? 0 : ~uint64_t(0)));
                        op += nb ^ big_flag;

                        const uint64_t dis = v + min_dis;
                        for (uint64_t i = 0; i < len; i += vector_width)
                            isa_lib::copy32(out + i, out + i - dis);
                        out += len;

                        continue;
                    }
                    // else unconditional copies cover both literals and match!!

                    // unconditional for lits
                    memcpy(out, lp, dec_literal_copy);
                    const uint64_t lit = T.lit[c];
                    out += lit, lp += lit;

                    // distance: always load 4 offset bytes, keep old one or update
                    // (GCC makes this a cmov, but turns a plain `m ? new : v` into a branch)
                    const uint32_t m = T.mask[c];
                    v = (loadu4(op) & m) | (v & (m ? 0 : ~uint64_t(0)));
                    op += nb;

                    // unconditional for match
                    // (`v` is 64-bit so that `out - dis` folds `min_dis` into the load address)
                    const uint64_t dis = v + min_dis;
                    isa_lib::safe_cyccpy(out - dis, dis);
                    out += T.len[c];
                }

                if (lp < nblock)
                {
                    uint64_t left = uint64_t(nblock - lp);
                    memcpy(out, lp, left);
                    out += left;
                }

                block = nblock;
            }

            return uint64_t(out - dst);
        }
    } // namespace heavy_unsafe_detail

    // Returns number of bytes written to `dst`, and 0 on failure.
    // `isa_lib` is ISA-dependent.
    template <class isa_lib>
    uint64_t heavy_unsafe_dimpl(const uint8_t* __restrict src,
                                [[maybe_unused]] uint64_t src_size,
                                uint8_t* __restrict dst,
                                uint64_t dst_cap)
    {
        using namespace heavy;

        const uint64_t header = loadu8(src);
        const uint64_t original_size = header & size_mask;
        const uint8_t flags = uint8_t(header >> flags_shift);

        // invalid flag
        if (flags != flags_a and flags != flags_s)
            return 0;

        if (dst_cap < original_size)
            return 0;

        // Small source
        if (original_size <= small_lim)
        {
            if (original_size > 0)
                memcpy(dst, src + header_size, original_size);
            return original_size;
        }

        using namespace heavy_unsafe_detail;

        if (flags == flags_s)
            return heavy_unsafe_actual_dimpl<isa_lib, tables_s>(
                src + header_size, original_size, dst);
        return heavy_unsafe_actual_dimpl<isa_lib, tables_a>(src + header_size, original_size, dst);
    }
} // namespace misa77

// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "format.h"
#include "util.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace misa77
{
    namespace heavy_safe_detail
    {
        using namespace heavy;

        // `src` points at the first block, `src_end` is one past the last readable byte.
        // PRECONDITION: `original_size > small_lim`
        template <class isa_lib, const decode_tables& T>
        uint64_t heavy_safe_actual_dimpl(const uint8_t* __restrict src,
                                         const uint8_t* src_end,
                                         uint64_t original_size,
                                         uint8_t* __restrict dst)
        {
            const uint8_t* block = src;
            uint8_t* out = dst;
            uint8_t* const out_end = out + original_size;
            // every token must end at `<= out_lim`
            uint8_t* const out_lim = out_end - literal_suffix;
            uint64_t v = uint64_t(0) - min_dis;

            while (out < out_end)
            {
                const uint64_t space = uint64_t(src_end - block);
                if (space < block_header_size) [[unlikely]]
                    return 0;

                // a 4-byte load of the last count could over-read by 1
                const auto load_cnt = [](const uint8_t* ptr)
                { return uint64_t(loadu2(ptr)) | (uint64_t(ptr[2]) << 16); };
                const uint64_t tok_cnt = load_cnt(block + tok_cnt_pos);
                const uint64_t off_cnt = load_cnt(block + off_cnt_pos);
                const uint64_t lit_cnt = load_cnt(block + lit_cnt_pos);

                // the three streams must lie in the source
                if (space - block_header_size < tok_cnt + off_cnt + lit_cnt) [[unlikely]]
                    return 0;

                // controls
                const uint8_t* cp = block + block_header_size;
                const uint8_t* const cp_end = cp + tok_cnt;

                // offsets
                const uint8_t* op = cp_end;
                const uint8_t* const op_end = op + off_cnt;

                // literals
                const uint8_t* lp = op_end;
                const uint8_t* const lp_end = lp + lit_cnt;

                while (cp < cp_end)
                {
                    // The batch below only keeps `lp` and `op` inside the source, so they may
                    // have crossed their own stream ends.
                    if (lp > lp_end or op > op_end) [[unlikely]]
                        return 0;

                    // Number of small tokens that can run with no checks on the cursors:
                    // each reads `<= dec_literal_copy` literal bytes, loads `offset_load` offset
                    // bytes, and writes `<= max_small_out` bytes past `out`.
                    uint64_t batch = uint64_t(cp_end - cp);
                    batch = std::min(batch, uint64_t(src_end - lp) / dec_literal_copy);
                    batch = std::min(batch, uint64_t(src_end - op) / offset_load);
                    batch = std::min(batch,
                                     out < out_lim ? uint64_t(out_lim - out) / max_small_out : 0);
                    // No distance yet (`dis = 0`): the exact step rejects repeat matches until
                    // then, so the batch never has to.
                    if (v + min_dis == 0)
                        batch = 0;

                    const uint8_t* const batch_start = cp;
                    const uint8_t* const batch_end = cp + batch;

                    while (cp < batch_end)
                    {
                        const uint64_t c = *cp;
                        const uint64_t nb = T.nb[c];
                        ++cp;

                        // Big token: rounded copies as in the unsafe decoder, checked here, then a
                        // new batch. The budget already covers its 4-byte offset load.
                        if (nb & big_flag) [[unlikely]]
                        {
                            const uint64_t lit = T.full_lit[c];
                            const uint64_t len = T.full_len[c];
                            const uint64_t lit_read =
                                (lit + vector_width - 1) & ~(vector_width - 1);
                            if (lit + len > uint64_t(out_lim - out) or
                                lit_read > uint64_t(src_end - lp)) [[unlikely]]
                                return 0;

                            for (uint64_t i = 0; i < lit; i += vector_width)
                                isa_lib::copy32(out + i, lp + i);
                            out += lit, lp += lit;

                            const uint32_t m = T.mask[c];
                            v = (loadu4(op) & m) | (v & (m ? 0 : ~uint64_t(0)));
                            op += nb ^ big_flag;

                            const uint64_t dis = v + min_dis;
                            if (dis > uint64_t(out - dst)) [[unlikely]]
                                return 0;
                            for (uint64_t i = 0; i < len; i += vector_width)
                                isa_lib::copy32(out + i, out + i - dis);
                            out += len;
                            break;
                        }

                        memcpy(out, lp, dec_literal_copy);
                        const uint64_t lit = T.lit[c];
                        out += lit, lp += lit;

                        const uint32_t m = T.mask[c];
                        v = (loadu4(op) & m) | (v & (m ? 0 : ~uint64_t(0)));
                        op += nb;

                        // the match source must not start before `dst`
                        const uint64_t dis = v + min_dis;
                        if (dis > uint64_t(out - dst)) [[unlikely]]
                            return 0;

                        isa_lib::cyccpy(out - dis, dis);
                        out += T.len[c];
                    }

                    if (cp != batch_start)
                        continue;

                    // Exact step: one token of any kind, everything checked, exact reads.
                    // Here `lp <= lp_end` and `op <= op_end`.
                    const uint64_t c = *cp;
                    ++cp;

                    const uint64_t lit = T.full_lit[c];
                    const uint64_t len = T.full_len[c];
                    const uint64_t ob = T.nb[c] & ~uint64_t(big_flag);

                    if (lit > uint64_t(lp_end - lp) or ob > uint64_t(op_end - op)) [[unlikely]]
                        return 0;
                    if (out > out_lim or lit + len > uint64_t(out_lim - out)) [[unlikely]]
                        return 0;

                    memcpy(out, lp, lit);
                    out += lit, lp += lit;

                    if (ob > 0)
                    {
                        v = 0;
                        for (uint64_t i = 0; i < ob; ++i)
                            v |= uint32_t(op[i]) << (8 * i);
                        op += ob;
                    }

                    if (len > 0)
                    {
                        const uint64_t dis = v + min_dis;
                        if (dis == 0 or dis > uint64_t(out - dst)) [[unlikely]]
                            return 0;

                        // over-writes by < vector_width, and the token ends at `<= out_lim`
                        for (uint64_t i = 0; i < len; i += vector_width)
                            isa_lib::copy32(out + i, out + i - dis);
                        out += len;
                    }
                }

                if (lp > lp_end or op > op_end) [[unlikely]]
                    return 0;

                // trailing literals
                const uint64_t left = uint64_t(lp_end - lp);
                if (left > uint64_t(out_end - out)) [[unlikely]]
                    return 0;
                if (left > 0)
                    memcpy(out, lp, left);
                out += left;

                block = lp_end;
            }

            return original_size;
        }
    } // namespace heavy_safe_detail

    // Returns number of bytes written to `dst`, and 0 on failure.
    // `isa_lib` is ISA-dependent.
    template <class isa_lib>
    uint64_t heavy_safe_dimpl(const uint8_t* __restrict src,
                              uint64_t src_size,
                              uint8_t* __restrict dst,
                              uint64_t dst_cap)
    {
        using namespace heavy;

        // Header fields must exist
        if (src_size < header_size) [[unlikely]]
            return 0;

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
            // Must not over-read here
            if (src_size < header_size + original_size) [[unlikely]]
                return 0;
            if (original_size > 0)
                memcpy(dst, src + header_size, original_size);
            return original_size;
        }

        using namespace heavy_safe_detail;

        if (flags == flags_s)
            return heavy_safe_actual_dimpl<isa_lib, tables_s>(
                src + header_size, src + src_size, original_size, dst);
        return heavy_safe_actual_dimpl<isa_lib, tables_a>(
            src + header_size, src + src_size, original_size, dst);
    }
} // namespace misa77

// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "decompressor_zoo/entropy_safe_kernels/kernel.h"
#include "entropy/huffman_decoder.h"
#include "format.h"
#include "util.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>

namespace misa77
{
    namespace entropy_safe_detail
    {
        using namespace entropy;

        // Readable bytes kept after every decoded stream in scratch (the merge kernels load 16 at a
        // time).
        inline constexpr uint64_t stream_pad = 16;
        // Entries kept after the offsets (the commands read one, the merge kernels write some).
        inline constexpr uint64_t off_pad = merge_out_pad;
        // `scratch` sizes the planes assuming this
        static_assert(min_new_tok_match_len >= max_offset_bytes);

        // Working memory of one call. Sized to the largest block of the input once.
        struct scratch
        {
            // `memory` is allocated, and everything below is just a subsegment of it
            void* memory = nullptr;
            huffman_decoder::decode_table* table = nullptr;
            uint8_t* cmd = nullptr;
            uint8_t* lit = nullptr;
            // offset planes and flag streams
            uint8_t* planes = nullptr;
            int32_t* off = nullptr;
            // working memory of the merge kernels
            uint8_t* merge_tmp = nullptr;

            scratch() = default;
            scratch(const scratch&) = delete;
            scratch& operator=(const scratch&) = delete;
            ~scratch()
            {
                ::operator delete(memory);
            }

            // `cap` = largest block length. False if there is no memory.
            bool allocate(uint64_t cap)
            {
                const auto round = [](uint64_t v) { return (v + 63) & ~uint64_t(63); };
                // a block's planes hold <= cap bytes in all, its two flag streams <= cap / 8
                const uint64_t max_off_cnt = cap / min_new_tok_match_len;
                const uint64_t table_size = round(sizeof(huffman_decoder::decode_table));
                const uint64_t cmd_size = round(cap + stream_pad);
                const uint64_t lit_size = round(cap + dec_literal_copy + stream_pad);
                const uint64_t planes_size = round(cap + cap / 8 + stream_cnt * stream_pad);
                const uint64_t off_size = round((max_off_cnt + off_pad) * sizeof(int32_t));
                const uint64_t tmp_size = round(merge_tmp_size(max_off_cnt));
                // Aligned by hand: an over-aligned request of this size goes to the kernel and
                // back on every call with glibc, a plain one is reused.
                constexpr uint64_t align = 64;
                memory = ::operator new(table_size + cmd_size + lit_size + planes_size + off_size +
                                            tmp_size + align,
                                        std::nothrow);
                if (memory == nullptr)
                    return false;
                uint8_t* at = static_cast<uint8_t*>(memory);
                at += (align - reinterpret_cast<uintptr_t>(at) % align) % align;
                table = new (at) huffman_decoder::decode_table;
                at += table_size;
                cmd = at, at += cmd_size;
                lit = at, at += lit_size;
                planes = at, at += planes_size;
                off = reinterpret_cast<int32_t*>(at), at += off_size;
                merge_tmp = at;
                return true;
            }
        };

        [[gnu::always_inline]]
        inline uint64_t load_cnt(const uint8_t* ptr)
        {
            return uint64_t(loadu2(ptr)) | (uint64_t(ptr[2]) << 16);
        }

        // set bits in array `p[0, size)`
        inline uint64_t popcount(const uint8_t* p, uint64_t size)
        {
            const auto count = [](uint64_t v)
            {
                v -= (v >> 1) & 0x5555'5555'5555'5555;
                v = (v & 0x3333'3333'3333'3333) + ((v >> 2) & 0x3333'3333'3333'3333);
                v = (v + (v >> 4)) & 0x0F0F'0F0F'0F0F'0F0F;
                return (v * 0x0101'0101'0101'0101) >> 56;
            };
            uint64_t total = 0;
            uint64_t i = 0;
            for (; i + 8 <= size; i += 8)
                total += count(loadu8(p + i));
            uint64_t last = 0;
            for (uint64_t k = 0; i + k < size; ++k)
                last |= uint64_t(p[i + k]) << (8 * k);
            return total + count(last);
        }

        // commands of kind N in `c[0, cnt)`
        inline uint64_t count_new(const uint8_t* c, uint64_t cnt)
        {
            // byte-wide partial sums keep the compiler's vectors 1 byte per lane
            constexpr uint64_t chunk = 128;
            uint64_t total = 0;
            uint64_t i = 0;
            for (; i + chunk <= cnt; i += chunk)
            {
                uint8_t part = 0;
                for (uint64_t k = 0; k < chunk; ++k)
                    part = uint8_t(part + (c[i + k] >= first_new ? 1 : 0));
                total += part;
            }
            for (; i < cnt; ++i)
                total += c[i] >= first_new ? 1 : 0;
            return total;
        }

        // A flag stream of `bits` flags must have exactly `ones` set, and none past the last.
        inline bool flags_ok(const uint8_t* p, uint64_t bits, uint64_t ones)
        {
            const uint64_t size = (bits + 7) / 8;
            if ((bits & 7) != 0 and (p[size - 1] >> (bits & 7)) != 0)
                return false;
            return popcount(p, size) == ones;
        }

        // Merges the split offset streams back into one array.
        // - `q` = the streams' plain bytes
        // - Writes to `off[0, off_cnt)`: negated distances `-(v + min_dis)`, in command order. Also
        // writes `off[off_cnt]`.
        // - Returns the smallest entry (= the largest distance, negated).
        // - PRECONDITION: `flags_ok` holds for both flag streams, and one byte is readable after
        //   every plane.
        inline int64_t merge_offsets(const uint8_t* const* q,
                                     uint64_t o1_cnt,
                                     uint64_t o2_cnt,
                                     uint64_t o3_cnt,
                                     int32_t* off)
        {
            const uint64_t wide_cnt = o2_cnt + o3_cnt;
            const uint64_t off_cnt = o1_cnt + wide_cnt;
            uint32_t* const all = reinterpret_cast<uint32_t*>(off);
            uint32_t* const wide = all + o1_cnt;
            uint32_t* const three = wide + o2_cnt;
            // read (never selected) once the last entry of a kind is taken
            all[off_cnt] = 0;

            const uint8_t* const o3l = q[s_o3l];
            const uint8_t* const o3m = q[s_o3m];
            const uint8_t* const o3h = q[s_o3h];
            for (uint64_t i = 0; i < o3_cnt; ++i)
                three[i] = uint32_t(o3l[i]) | uint32_t(o3m[i]) << 8 | uint32_t(o3h[i]) << 16;

            // `fn(i, bit)` for every flag, a flag byte at a time: the shifts are constants
            const auto for_each = [](const uint8_t* flags, uint64_t cnt, auto&& fn)
            {
                uint64_t i = 0;
                for (; i + 8 <= cnt; i += 8)
                {
                    const uint32_t f = flags[i / 8];
                    for (uint32_t k = 0; k < 8; ++k)
                        fn(i + k, (f >> k) & 1);
                }
                for (; i < cnt; ++i)
                    fn(i, uint32_t(flags[i / 8] >> (i % 8)) & 1);
            };

            const uint8_t* const o2l = q[s_o2l];
            const uint8_t* const o2h = q[s_o2h];
            uint64_t taken = 0;
            for_each(q[s_f2],
                     wide_cnt,
                     [&](uint64_t i, uint32_t bit)
                     {
                         const uint32_t narrow = uint32_t(o2l[i - taken]) | uint32_t(o2h[i - taken])
                                                                                << 8;
                         wide[i] = narrow ^ ((narrow ^ three[taken]) & (0 - bit));
                         taken += bit;
                     });

            const uint8_t* const o1 = q[s_o1];
            uint32_t largest = 0;
            taken = 0;
            for_each(q[s_f1],
                     off_cnt,
                     [&](uint64_t i, uint32_t bit)
                     {
                         const uint32_t narrow = o1[i - taken];
                         const uint32_t v = narrow ^ ((narrow ^ wide[taken]) & (0 - bit));
                         taken += bit;
                         largest = std::max(largest, v);
                         off[i] = -int32_t(v + uint32_t(min_dis));
                     });
            return -int64_t(largest + min_dis);
        }

        // The core LZ loop, after merge and before DELTA.
        // Runs every command of one block.
        // - Commands at `[cp, cp_end)`, their distances at `off`, literals at `lit`.
        // - Writes to `[out, out_end)`. May write up to `max_cmd_out` bytes past `out_end`.
        // - `r0` = the negated repeat distance carried over from previous blocks or initialised
        // fresh.
        // - `sub` = the block is a SUB block.
        // - `lit`, `out` and `r0` are updated for the caller.
        // - Returns false if a match starts before `dst`, or if a command starts at or past
        // `out_end`.
        // - PRECONDITION: `off` has one entry per command of kind N, plus one more readable
        //   entry. `out_end - out + dec_literal_copy` bytes are readable at `lit`.
        template <class isa_lib, bool sub>
        bool run_commands(const uint8_t* cp,
                          const uint8_t* const cp_end,
                          const int32_t* off,
                          const uint8_t*& lit,
                          uint8_t*& out_ref,
                          uint8_t* const out_end,
                          const uint8_t* const dst,
                          int64_t& r0_ref)
        {
            const uint8_t* lp = lit;
            uint8_t* out = out_ref;
            int64_t r0 = r0_ref;
            const uint8_t* const pack = commands.pack;

            while (cp < cp_end)
            {
                if (out >= out_end) [[unlikely]]
                    return false;
                // Unchecked commands: each moves `out` by `<= max_cmd_out`, so every one of
                // these starts at or before `out_end` and writes below `out_end + max_cmd_out`.
                const uint64_t batch =
                    std::min(uint64_t(cp_end - cp), uint64_t(out_end - out) / max_cmd_out + 1);
                const uint8_t* const batch_end = cp + batch;
                while (cp < batch_end)
                {
                    const uint64_t c = *cp;
                    ++cp;
                    const uint64_t lit_len = pack[2 * c];
                    const uint64_t total = pack[2 * c + 1];

                    if constexpr (sub)
                    {
                        uint8_t a[dec_literal_copy];
                        uint8_t b[dec_literal_copy];
                        memcpy(a, lp, dec_literal_copy);
                        memcpy(b, out + r0, dec_literal_copy);
                        for (uint64_t i = 0; i < dec_literal_copy; ++i)
                            a[i] = uint8_t(a[i] + b[i]);
                        memcpy(out, a, dec_literal_copy);
                    }
                    else
                        memcpy(out, lp, dec_literal_copy);
                    lp += lit_len;

                    // Written with a mask: compilers turn the ternary into a branch that
                    // mispredicts.
                    const int64_t is_new = c >= first_new ? 1 : 0;
                    r0 ^= (r0 ^ int64_t(*off)) & -is_new;
                    off += is_new;

                    uint8_t* const match = out + lit_len;
                    // the match source must not start before `dst`
                    if (match - dst + r0 < 0) [[unlikely]]
                        return false;
                    isa_lib::copy32(match, match + r0);
                    out += total;
                }
            }

            lit = lp, out_ref = out, r0_ref = r0;
            return true;
        }

        // x[i] += x[i - stride] over [b, e), ascending
        inline void delta_restore(uint8_t* b, uint8_t* const e, uint64_t stride)
        {
            // 16 bytes at a time never read what they write
            if (stride >= 16)
            {
                for (; e - b >= 16; b += 16)
                {
                    uint8_t x[16];
                    uint8_t y[16];
                    memcpy(x, b, 16);
                    memcpy(y, b - stride, 16);
                    for (uint64_t i = 0; i < 16; ++i)
                        x[i] = uint8_t(x[i] + y[i]);
                    memcpy(b, x, 16);
                }
            }
            for (; b < e; ++b)
                *b = uint8_t(*b + *(b - stride));
        }

        // Handles one coded block.
        // - To be stored at `out[0, len)`
        // - Block header at `p`, payloads end at or before `pend`,`src_end` is one past the last
        // readable byte
        // - Returns the block's end in the source, or null on a malformed block.
        template <class isa_lib>
        const uint8_t* decode_coded_block(const uint8_t* p,
                                          const uint8_t* const pend,
                                          const uint8_t* const src_end,
                                          uint8_t* const out,
                                          uint64_t len,
                                          uint8_t* const dst,
                                          int64_t& r0,
                                          scratch& S)
        {
            if (uint64_t(pend - p) < block_header_size) [[unlikely]]
                return nullptr;

            // metadata
            const uint8_t mode = p[mode_pos];
            const uint64_t cmd_cnt = load_cnt(p + cmd_cnt_pos);
            const uint64_t lit_cnt = load_cnt(p + lit_cnt_pos);
            const uint64_t o1_cnt = load_cnt(p + o1_cnt_pos);
            const uint64_t o2_cnt = load_cnt(p + o2_cnt_pos);
            const uint64_t o3_cnt = load_cnt(p + o3_cnt_pos);
            const uint32_t coded = loadu2(p + coded_pos);
            p += block_header_size;

            const uint64_t wide_cnt = o2_cnt + o3_cnt;
            const uint64_t off_cnt = o1_cnt + wide_cnt;
            // If these fail, they also bound every scratch buffer by `len`. If they pass, bad
            // block.
            if (cmd_cnt < 1 or cmd_cnt > len or lit_cnt > len or off_cnt > cmd_cnt or
                min_new_tok_match_len * off_cnt > len or mode > mode_sub_delta or
                (coded >> stream_cnt) != 0) [[unlikely]]
                return nullptr;

            // block start and end
            const uint64_t bs = uint64_t(out - dst);
            const uint64_t be = bs + len;

            // location of strides bytes
            const uint8_t* const strides = p;

            // check stride meta-data
            if (mode & mode_delta)
            {
                // number of cells
                const uint64_t cells = (be - 1) / cell_size - bs / cell_size + 1;
                if (uint64_t(pend - strides) < cells) [[unlikely]]
                    return nullptr;
                uint64_t worst = 0;
                for (uint64_t c = 0; c < cells; ++c)
                    worst = std::max<uint64_t>(worst, p[c]);
                // invalid stride
                if (worst > max_stride) [[unlikely]]
                    return nullptr;
                p += cells;
            }

            // stream size after we huffman decode
            const uint64_t plain_size[stream_cnt] = {
                cmd_cnt,
                lit_cnt,
                o1_cnt,
                o2_cnt,
                o2_cnt,
                o3_cnt,
                o3_cnt,
                o3_cnt,
                (off_cnt + 7) / 8,
                (wide_cnt + 7) / 8,
            };

            // stream size before we huffman decode
            uint64_t stored_size[stream_cnt];

            for (uint32_t s = 0; s < stream_cnt; ++s)
            {
                stored_size[s] = plain_size[s];

                // huffman coded
                if ((coded >> s) & 1)
                {
                    // an empty stream is never coded
                    if (plain_size[s] == 0 or uint64_t(pend - p) < coded_size_bytes) [[unlikely]]
                        return nullptr;
                    stored_size[s] = load_cnt(p);
                    p += coded_size_bytes;
                }
            }

            // q[i] = location where we're placing decoded bytes for stream i (possibly scratch)
            const uint8_t* q[stream_cnt];
            uint8_t* plane = S.planes;
            for (uint32_t s = 0; s < stream_cnt; ++s)
            {
                if (uint64_t(pend - p) < stored_size[s]) [[unlikely]]
                    return nullptr;

                // used in place
                q[s] = p;

                // huffman coded
                if ((coded >> s) & 1)
                {
                    // region in scratch space
                    uint8_t* const buf = s == s_cmd ? S.cmd : s == s_lit ? S.lit : plane;

                    // ahead, and pad
                    if (s > s_lit)
                        plane += plain_size[s] + stream_pad;

                    const auto kernel = [](huffman_decoder::kernel_io* io)
                    {
                        if constexpr (isa_lib::has_entropy_kernel)
                            isa_lib::entropy_huffman_kernel(io);
                        else
                            huffman_decoder::kernel_cpp(io);
                    };

                    // huffman failed for some reason
                    if (not huffman_decoder::decode(
                            buf, plain_size[s], p, stored_size[s], *S.table, kernel)) [[unlikely]]
                        return nullptr;

                    q[s] = buf;
                }

                // advance anyway
                p += stored_size[s];
            }

            // The offset byte count streams must correspond exactly to the split offset byte
            // streams.
            if (not flags_ok(q[s_f1], off_cnt, wide_cnt) or not flags_ok(q[s_f2], wide_cnt, o3_cnt))
                [[unlikely]]
                return nullptr;

            // Every command of kind N has one corresponding offset
            if (count_new(q[s_cmd], cmd_cnt) != off_cnt) [[unlikely]]
                return nullptr;

            // The commands read literals 16 bytes at a time, up to `len` bytes in.
            // We move raw literals too close to the end of the source so we can safely over-read.
            if (((coded >> s_lit) & 1) == 0 and
                uint64_t(src_end - q[s_lit]) < len + dec_literal_copy)
            {
                memcpy(S.lit, q[s_lit], lit_cnt);
                q[s_lit] = S.lit;
            }

            // the latter is basically always true, but the kernel needs it at one specific place
            // where it subtracts something from addresses
            const bool use_kernel =
                isa_lib::has_entropy_kernel and reinterpret_cast<uintptr_t>(dst) >= kernel_min_addr;

            // The largest distance to a match for tokens within this block, negated.
            // If we have farthest <= bs, no match of this block can start before `dst`, so we can
            // get rid of certain checks while safe-decoding
            int64_t farthest;
            if constexpr (isa_lib::has_entropy_merge_kernel)
            {
                // prepare io for kernel
                merge_io io;
                for (uint32_t i = 0; i < 6; ++i)
                    io.plane[i] = q[s_o1 + i];
                io.f1 = q[s_f1], io.f2 = q[s_f2];
                io.off_cnt = off_cnt, io.wide_cnt = wide_cnt;
                for (uint32_t i = 0; i < 3; ++i)
                    io.tmp[i] = S.merge_tmp + i * (wide_cnt + 8);
                io.out = S.off;
                io.expand = &expand;
                io.o3_cnt = o3_cnt;
                farthest = isa_lib::entropy_merge_kernel(&io);
            }
            else
                farthest = merge_offsets(q, o1_cnt, o2_cnt, o3_cnt, S.off);

            const uint8_t* lit = q[s_lit];
            uint8_t* o = out;
            uint8_t* const out_end = out + len;
            bool ok;
            if (use_kernel)
            {
                kernel_io io;
                io.cmd = q[s_cmd], io.cmd_end = q[s_cmd] + cmd_cnt;
                io.off = S.off, io.lit = lit;
                io.out = o, io.out_end = out_end, io.dst = dst;
                io.r0 = r0, io.table = commands.pack;
                // No match of this block can start before `dst` if none reaches further back than
                // the block's start. variants: (a) whether block uses SUB or not (b) whether the
                // kernel should test if every match is from >= dst
                const uint64_t variant = ((mode & mode_sub) ? kernel_sub : 0) |
                                         (uint64_t(-farthest) > bs ? kernel_check : 0);
                ok = false;
                if constexpr (isa_lib::has_entropy_kernel)
                    ok = isa_lib::entropy_kernel(&io, variant) != 0;
                lit = io.lit, o = io.out, r0 = io.r0;
            }
            else if (mode & mode_sub)
                ok = run_commands<isa_lib, true>(
                    q[s_cmd], q[s_cmd] + cmd_cnt, S.off, lit, o, out_end, dst, r0);
            else
                ok = run_commands<isa_lib, false>(
                    q[s_cmd], q[s_cmd] + cmd_cnt, S.off, lit, o, out_end, dst, r0);

            // the commands must fill the block fully with the literals
            if (not ok or o != out_end or lit != q[s_lit] + lit_cnt) [[unlikely]]
                return nullptr;

            // reverse delta encoding
            if (mode & mode_delta)
            {
                for (uint64_t c0 = bs / cell_size, c = c0; c * cell_size < be; ++c)
                {
                    const uint64_t stride = strides[c - c0];
                    if (stride == 0)
                        continue;
                    uint8_t* const b = dst + std::max(bs, c * cell_size);
                    uint8_t* const e = dst + std::min(be, (c + 1) * cell_size);
                    if constexpr (isa_lib::has_entropy_kernel)
                        isa_lib::entropy_delta_kernel(b, e, stride);
                    else
                        delta_restore(b, e, stride);
                }
            }
            return p;
        }
    } // namespace entropy_safe_detail

    // Returns number of bytes written to `dst`, and 0 on failure.
    // `src_size` must be the stream's exact size: the raw prefix and suffix are found from its
    // end.
    // `isa_lib` is ISA-dependent.
    template <class isa_lib>
    uint64_t entropy_safe_dimpl(const uint8_t* __restrict src,
                                uint64_t src_size,
                                uint8_t* __restrict dst,
                                uint64_t dst_cap)
    {
        using namespace entropy;
        using namespace entropy_safe_detail;

        // Header fields must exist
        if (src_size < header_size) [[unlikely]]
            return 0;

        const uint64_t header = loadu8(src);
        const uint64_t original_size = header & size_mask;

        // invalid flag
        if (uint8_t(header >> flags_shift) != flags)
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

        if (src_size < header_size + raw_size) [[unlikely]]
            return 0;
        const uint8_t* const src_end = src + src_size;
        // the blocks must end exactly here
        const uint8_t* const pend = src_end - raw_size;

        // Matches, SUB literals and DELTA strides of the first block reach into the prefix.
        memcpy(dst, pend, prefix_size);

        scratch S;
        const uint8_t* p = src + header_size;
        uint8_t* out = dst + prefix_size;
        uint8_t* const body_end = dst + original_size - suffix_size;
        int64_t r0 = -int64_t(initial_dis);

        // process one block per iteration
        while (out < body_end)
        {
            const uint64_t at = uint64_t(out - dst);
            const uint64_t len =
                std::min((at / block_size + 1) * block_size, original_size - suffix_size) - at;

            if (p >= pend) [[unlikely]]
                return 0;

            // stored raw
            if (p[mode_pos] & mode_stored)
            {
                if (p[mode_pos] != mode_stored or uint64_t(pend - p) < 1 + len) [[unlikely]]
                    return 0;
                memcpy(out, p + 1, len);
                p += 1 + len;
            }
            else
            {
                // something went wrong
                if (S.memory == nullptr and
                    not S.allocate(std::min(block_size, original_size - raw_size))) [[unlikely]]
                    return 0;
                p = decode_coded_block<isa_lib>(p, pend, src_end, out, len, dst, r0, S);

                // something went wrong
                if (p == nullptr) [[unlikely]]
                    return 0;
            }
            out += len;
        }

        if (p != pend) [[unlikely]]
            return 0;
        memcpy(body_end, src_end - suffix_size, suffix_size);
        return original_size;
    }
} // namespace misa77

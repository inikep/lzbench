// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#pragma once

#include "format.h"
#include "util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

// Decoder side of the entropy format's Huffman payload:
// [table][`payload_split` bit strings][size of each but the last]
//
// Symbol j (counted from the end of the decoded stream) lives in bit string `j % payload_split`.
// A bit string is read LSB first.
// Since codewords are stored bit-reversed, the low `max_code_len` bits can be used to index the
// table directly.

namespace misa77
{
    namespace huffman_decoder
    {
        using namespace entropy;
        using namespace entropy::huffman;

        inline constexpr uint32_t table_size = uint32_t(1) << max_code_len;
        // Symbols decoded per bit string between refills. A refill leaves >= 56 valid bits.
        inline constexpr uint32_t symbols_per_refill = 5;
        static_assert(symbols_per_refill * max_code_len <= 56);
        inline constexpr uint32_t round_size = symbols_per_refill * payload_split;
        // A refill advances a bit string's pointer by this much at most.
        inline constexpr uint64_t max_refill_advance = 7;
        static_assert(symbols_per_refill * max_code_len + 7 < 8 * (max_refill_advance + 1));
        // Bit 63 of a bit buffer is a marker: leading zeros = bits consumed since the pointer.
        inline constexpr uint64_t marker = uint64_t(1) << 63;

        // bit reversal of `max_code_len`-bit values
        struct reverse_table
        {
            uint16_t r[table_size];

            constexpr reverse_table() : r()
            {
                for (uint32_t i = 0; i < table_size; ++i)
                {
                    uint32_t v = 0;
                    for (uint32_t b = 0; b < max_code_len; ++b)
                        v |= ((i >> b) & 1) << (max_code_len - 1 - b);
                    r[i] = uint16_t(v);
                }
            }
        };
        inline constexpr reverse_table reversed{};

        // Bucket (k, l) of `build`: symbols of segment k (32 symbols each) with code length l.
        inline constexpr uint32_t segment_cnt = 8;
        inline constexpr uint32_t segment_size = symbol_cnt / segment_cnt;
        inline constexpr uint32_t length_cnt = 16; // a nibble

        struct decode_table
        {
            // entry = code length | symbol << 8, indexed by the next `max_code_len` bits.
            // The slack lets `build` copy 16 entries at a time.
            alignas(64) uint16_t t[table_size + 16];

            // `build` scratch. It lives here, at fixed distances from `t`, because the build's
            // speed depends on which of these alias `t` modulo 4 KiB.
            alignas(64) uint16_t bucket[segment_cnt * length_cnt * segment_size];
            uint32_t fill[segment_cnt * length_cnt];
            alignas(64) uint16_t order[symbol_cnt + segment_size];
            uint8_t pad[1600];
            uint32_t end[length_cnt];
            uint8_t tmp[32];
            alignas(64) uint8_t len[symbol_cnt + 16];
        };

        // Layout is hard-coded in the kernels.
        struct kernel_io
        {
            uint8_t* out;                    //   0 one past the next symbol to write (descending)
            uint8_t* out_stop;               //   8 run whole rounds while out > out_stop
            const uint16_t* table;           //  16
            const uint8_t* lim;              //  24 and while the last two pointers are <= lim
            const uint8_t* p[payload_split]; //  32
            uint64_t bits[payload_split];    // 128
        };
        static_assert(offsetof(kernel_io, p) == 32 and offsetof(kernel_io, bits) == 128 and
                      sizeof(kernel_io) == 224 and payload_split == 12);

        // 8 bytes at `p`, zero-extended where they would cross `end`
        inline uint64_t load_checked(const uint8_t* p, const uint8_t* end)
        {
            if (uint64_t(end - p) >= 8)
                return loadu8(p);
            uint64_t v = 0;
            if (p < end)
                memcpy(&v, p, uint64_t(end - p));
            return v;
        }

        // Table description -> 256 code lengths (each 0..15).
        // Returns the description's size, 0 if it doesn't fit in `avail` bytes.
        inline uint64_t read_table(const uint8_t* src, uint64_t avail, uint8_t* len)
        {
            if (avail < 1)
                return 0;
            if (src[0] == table_flat)
            {
                if (avail < flat_table_size)
                    return 0;
                for (uint32_t a = 0; a < symbol_cnt / 2; ++a)
                {
                    const uint8_t b = src[1 + a];
                    len[2 * a] = b & 15;
                    len[2 * a + 1] = b >> 4;
                }
                return flat_table_size;
            }
            if (src[0] != table_sparse or avail < 1 + bitmap_bytes)
                return 0;

            uint32_t present = 0;
            for (uint64_t k = 0; k < bitmap_bytes; k += 8)
                present += uint32_t(__builtin_popcountll(loadu8(src + 1 + k)));
            const uint64_t nibble_bytes = (present + 1) / 2;
            if (avail < 1 + bitmap_bytes + nibble_bytes)
                return 0;

            // one nibble per present symbol, in symbol order
            uint8_t packed[symbol_cnt / 2] = {};
            uint8_t nib[symbol_cnt];
            memcpy(packed, src + 1 + bitmap_bytes, nibble_bytes);
            for (uint32_t a = 0; a < symbol_cnt / 2; ++a)
            {
                nib[2 * a] = packed[a] & 15;
                nib[2 * a + 1] = packed[a] >> 4;
            }
            memset(len, 0, symbol_cnt);
            const uint8_t* next = nib;
            for (uint32_t k = 0; k < bitmap_bytes; k += 8)
                for (uint64_t m = loadu8(src + 1 + k); m != 0; m &= m - 1)
                    len[8 * k + uint32_t(__builtin_ctzll(m))] = *next++;
            return 1 + bitmap_bytes + nibble_bytes;
        }

        // Decode table from 256 code lengths (each 0..15). False unless the lengths form a
        // complete prefix code with lengths <= `max_code_len`, or name exactly one symbol with
        // length 1.
        //
        // 1. One pass drops every symbol's finished table entry into bucket (segment, length):
        //    `segment_cnt` independent counter chains, and no separate counting pass.
        // 2. The buckets are concatenated into (length, symbol) order with fixed-size copies.
        // 3. The table grows by doubling: when it has 2^l entries, every code of length l is
        //    stored once at its bit-reversed value, then the table is copied onto its upper half.
        inline bool build(decode_table& T, const uint8_t* len_in)
        {
            memcpy(T.len, len_in, symbol_cnt);
            const uint8_t* const len = T.len;
            uint16_t* const bucket = T.bucket;
            uint32_t* const fill = T.fill;

            for (uint32_t j = 0; j < segment_cnt * length_cnt; ++j)
                fill[j] = segment_size * j;
            for (uint32_t s = 0; s < segment_size; ++s)
            {
                for (uint32_t k = 0; k < segment_cnt; ++k)
                {
                    const uint64_t a = segment_size * k + s;
                    const uint64_t l = len[a];
                    const uint64_t at = fill[length_cnt * k + l];
                    fill[length_cnt * k + l] = uint32_t(at + 1);
                    bucket[at] = uint16_t(l + (a << 8));
                }
            }

            uint16_t* const order = T.order;
            uint32_t* const end = T.end;
            uint32_t used = 0;
            uint32_t q = 0;
            for (uint32_t l = 1; l <= max_code_len; ++l)
            {
                const uint32_t q0 = q;
                for (uint32_t k = 0; k < segment_cnt; ++k)
                {
                    const uint32_t b = length_cnt * k + l;
                    memcpy(order + q, bucket + b * segment_size, 2 * segment_size);
                    q += fill[b] - b * segment_size;
                }
                end[l] = q;
                used += (q - q0) << (max_code_len - l);
            }
            uint32_t too_long = 0;
            for (uint32_t k = 0; k < segment_cnt; ++k)
                for (uint32_t l = max_code_len + 1; l < length_cnt; ++l)
                    too_long |= fill[length_cnt * k + l] - (length_cnt * k + l) * segment_size;
            if (too_long)
                return false;

            uint16_t* const t = T.t;
            if (used != table_size)
            {
                if (not(q == 1 and end[1] == 1))
                    return false;
                for (uint32_t i = 0; i < table_size; ++i)
                    t[i] = order[0];
                return true;
            }

            uint64_t i = 0;
            uint64_t code = 0;
            uint64_t size = 1;
            for (uint32_t l = 1; l <= max_code_len; ++l)
            {
                if (size < 16)
                {
                    memcpy(T.tmp, t, 32);
                    memcpy(t + size, T.tmp, 32);
                }
                else
                    for (uint64_t k = 0; k < size; k += 16)
                        memcpy(t + size + k, t + k, 32);
                size *= 2;
                code <<= 1;
                const uint32_t shift = max_code_len - l;
                for (const uint64_t e = end[l]; i < e; ++i, ++code)
                    t[uint64_t(reversed.r[code]) >> shift] = order[i];
            }
            return true;
        }

        // Whole rounds while `out > out_stop` and the last two pointers are `<= lim`.
        // The bit strings are walked `group` at a time so that only that many bit buffers are
        // live; consecutive groups are independent, so they still overlap in the pipeline.
        [[gnu::noinline]]
        inline void kernel_cpp(kernel_io* io)
        {
            constexpr uint32_t group = 4;
            static_assert(payload_split % group == 0);
            const uint16_t* const t = io->table;
            uint8_t* out = io->out;
            uint8_t* const out_stop = io->out_stop;
            const uint8_t* const lim = io->lim;

            while (out > out_stop and io->p[payload_split - 2] <= lim and
                   io->p[payload_split - 1] <= lim)
            {
                for (uint32_t g = 0; g < payload_split; g += group)
                {
                    uint64_t b[group];
                    for (uint32_t k = 0; k < group; ++k)
                        b[k] = io->bits[g + k];
                    for (uint32_t r = 0; r < symbols_per_refill; ++r)
                    {
                        for (uint32_t k = 0; k < group; ++k)
                        {
                            const uint16_t e = t[b[k] & (table_size - 1)];
                            b[k] >>= (e & 63);
                            const uint32_t idx = r * payload_split + g + k;
                            // One 2-byte store per symbol: the length byte lands on the next
                            // bit string's slot, which is written later. Not so for the last
                            // bit string: its neighbour belongs to the first group.
                            if (g + k + 1 == payload_split)
                                out[-int64_t(idx) - 1] = uint8_t(e >> 8);
                            else
                                storeu2(out - idx - 2, e);
                        }
                    }
                    for (uint32_t k = 0; k < group; ++k)
                    {
                        const uint64_t lz = uint64_t(__builtin_clzll(b[k]));
                        const uint8_t* const p = io->p[g + k] + (lz >> 3);
                        io->p[g + k] = p;
                        io->bits[g + k] = (loadu8(p) | marker) >> (lz & 7);
                    }
                }
                out -= round_size;
            }
            io->out = out;
        }

        // Bit strings + sizes -> `out[0, n)`, given a built table. `kernel` has `kernel_cpp`'s
        // contract. Never reads outside `src[0, size)` or writes outside `out[0, n)`.
        template <class kernel_fn>
        inline bool decode_streams(uint8_t* out,
                                   uint64_t n,
                                   const uint8_t* src,
                                   uint64_t size,
                                   const decode_table& T,
                                   kernel_fn kernel)
        {
            if (size < size_fields)
                return false;
            const uint8_t* const end = src + size;
            // the size fields end the area of the bit strings
            const uint8_t* const fields = end - size_fields;

            kernel_io io;
            const uint8_t* string_end[payload_split];
            const uint8_t* at = src;
            for (uint32_t k = 0; k < payload_split; ++k)
            {
                io.p[k] = at;
                io.bits[k] = load_checked(at, end) | marker;
                if (k + 1 < payload_split)
                {
                    const uint64_t sz = loadu2(fields + split_size_bytes * k);
                    if (sz > uint64_t(fields - at))
                        return false;
                    at += sz;
                }
                string_end[k] = k + 1 < payload_split ? at : fields;
            }

            io.table = T.t;
            io.out = out + n;
            // p <= lim: the 8-byte load after one more round stays inside the payload
            const uint64_t lim_back = 8 + max_refill_advance;
            io.lim = size >= lim_back ? end - lim_back : src;
            for (;;)
            {
                // A round writes down to `out - round_size - 1`.
                uint64_t rounds = uint64_t(io.out - out - 1) / round_size;
                // The last two pointers are checked by the kernel every round; the others here.
                for (uint32_t k = 0; k + 2 < payload_split; ++k)
                {
                    const uint64_t avail = uint64_t(end - io.p[k]);
                    rounds = std::min(rounds, avail >= 8 ? (avail - 8) / max_refill_advance : 0);
                }
                if (rounds == 0 or io.p[payload_split - 2] > io.lim or
                    io.p[payload_split - 1] > io.lim)
                    break;
                io.out_stop = io.out - rounds * round_size;
                kernel(&io);
            }

            // the rest, one checked symbol at a time
            const uint16_t* const t = T.t;
            for (uint64_t j = uint64_t(out + n - io.out); j < n; ++j)
            {
                const uint32_t k = uint32_t(j % payload_split);
                const uint64_t e = t[io.bits[k] & (table_size - 1)];
                io.bits[k] >>= (e & 63);
                out[n - 1 - j] = uint8_t(e >> 8);
                const uint64_t lz = uint64_t(__builtin_clzll(io.bits[k]));
                io.p[k] += lz >> 3;
                if (io.p[k] > end)
                    return false;
                io.bits[k] = (load_checked(io.p[k], end) | marker) >> (lz & 7);
            }

            // Every bit string must be used up exactly (a partly used last byte counts as used).
            for (uint32_t k = 0; k < payload_split; ++k)
                if (io.p[k] + (__builtin_clzll(io.bits[k]) != 0 ? 1 : 0) != string_end[k])
                    return false;
            return true;
        }

        // The format-specified payload is stored in `src[0, size)`.
        // Decompressed output is written to `out[0, n)`.
        // Returns true iff decode goes through in a valid manner.
        // PRECONDITION: `n > 0`
        template <class kernel_fn>
        inline bool decode(uint8_t* out,
                           uint64_t n,
                           const uint8_t* src,
                           uint64_t size,
                           decode_table& T,
                           kernel_fn kernel)
        {
            uint8_t len[symbol_cnt];
            const uint64_t table_bytes = read_table(src, size, len);
            if (table_bytes == 0 or not build(T, len))
                return false;
            return decode_streams(out, n, src + table_bytes, size - table_bytes, T, kernel);
        }
    } // namespace huffman_decoder
} // namespace misa77

/**
 * zling:
 *  light-weight lossless data compression utility.
 *
 * Copyright (C) 2012-2013 by Zhang Li <zhangli10 at baidu.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the project nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE PROJECT AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE PROJECT OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * @author zhangli10<zhangli10@baidu.com>
 * @brief  manipulate huffman encoding.
 */
#include "libzling_huffman.h"

#ifndef __GNUC__  // __builtin_clz() implement on other compilers
#define __builtin_clz(x) __clz_impl(x)

static inline int __clz_impl(uint32_t x, int c = 32) {
    while (x > 0) {
        c -= 1;
        x /= 2;
    }
    return c;
}
#endif

namespace baidu {
namespace zling {
namespace huffman {

static inline uint32_t RoundDown(uint32_t x) {
    while (x & (-x ^ x)) {
        x &= -x ^ x;
    }
    return x;
}
static inline uint32_t RoundUp(uint32_t x) {
    while (x & (-x ^ x)) {
        x &= -x ^ x;
    }
    return x << 1;
}

void ZlingMakeLengthTable(const uint32_t* freq_table,
                          uint32_t* length_table,
                          int scaling,
                          int max_codes,
                          int max_codelen) {

    static const int kMaxCodes = 1024;  /* max_codes > kMaxCodes not supported */
    int symbols[kMaxCodes];

    // init
    for (int i = 0; i < max_codes; i++) {
        if (freq_table[i] > 0 && freq_table[i] >> scaling == 0) {
            length_table[i] = 1;
        } else {
            length_table[i] = freq_table[i] >> scaling;
        }
    }

    // sort symbols (using CombSort)
    int delta = max_codes;
    int nswap = 1;

    for (int i = 0; i < max_codes; i++) {
        symbols[i] = i;
    }
    while (delta + nswap > 1) {
        nswap = 0;
        delta = int(delta / 1.33);

        for (int i = 0; i + delta < max_codes; i++) {
            if (length_table[symbols[i]] < length_table[symbols[i + delta]]) {
                nswap = 0;
                std::swap(symbols[i], symbols[i + delta]);
            }
        }
    }

    // round frequency
    uint32_t sum = 0;
    uint32_t run = 0;

    for (int i = 0; i < max_codes; i++) {
        sum += length_table[i];
    }
    for (int i = 0; i < max_codes; i++) {
        length_table[i] = RoundDown(length_table[i]);
        run += length_table[i];
    }
    sum = RoundUp(sum);

    while (run < sum) {
        for (int i = 0; i < max_codes; i++) {
            if (run + length_table[symbols[i]] <= sum) {
                run += length_table[symbols[i]];
                length_table[symbols[i]] += length_table[symbols[i]];
            }
        }
    }

    // get code length
    for (int i = 0; i < max_codes; i++) {
        if (length_table[i] > 0) {
            length_table[i] = 31 - (__builtin_clz(sum / length_table[i]));

            // bugfix: 20131229
            // only happens when all symbols except i have zero frequency (sum == length_table[i])
            if (freq_table[i] > 0 && length_table[i] == 0) {
                length_table[i] = 1;
            }

            // code length too long? -- scale and rebuild table
            if (length_table[i] > uint32_t(max_codelen)) {
                ZlingMakeLengthTable(freq_table, length_table, ++scaling, max_codes, max_codelen);
                return;
            }
        }
    }
    return;
}

void ZlingMakeEncodeTable(
    const uint32_t* length_table,
    uint16_t* encode_table,
    int max_codes,
    int max_codelen) {

    int code = 0;
    memset(encode_table, 0, sizeof(encode_table[0]) * max_codes);

    // make code for each symbol
    for (int codelen = 1; codelen <= max_codelen; codelen++) {
        for (int i = 0; i < max_codes; i++) {
            if (length_table[i] == static_cast<uint32_t>(codelen)) {
                encode_table[i] = code;
                code += 1;
            }
        }
        code *= 2;
    }

    // reverse each code
    for (int i = 0; i < max_codes; i++) {
        encode_table[i] = ((encode_table[i] & 0xff00) >> 8 | (encode_table[i] & 0x00ff) << 8);
        encode_table[i] = ((encode_table[i] & 0xf0f0) >> 4 | (encode_table[i] & 0x0f0f) << 4);
        encode_table[i] = ((encode_table[i] & 0xcccc) >> 2 | (encode_table[i] & 0x3333) << 2);
        encode_table[i] = ((encode_table[i] & 0xaaaa) >> 1 | (encode_table[i] & 0x5555) << 1);
        encode_table[i] >>= 16 - length_table[i];
    }
    return;
}

void ZlingMakeDecodeTable(
    const uint32_t* length_table,
    uint16_t* encode_table,
    uint16_t* decode_table,
    int max_codes,
    int max_codelen) {

    memset(decode_table, -1, sizeof(decode_table[0]) * (1 << max_codelen));

    for (int c = 0; c < max_codes; c++) {
        if (length_table[c] > 0 && length_table[c] <= uint16_t(max_codelen)) {
            for (int i = encode_table[c]; i < (1 << max_codelen); i += (1 << length_table[c])) {
                decode_table[i] = c;
            }
        }
    }
    return;
}

}  // namespace huffman
}  // namespace zling
}  // namespace baidu

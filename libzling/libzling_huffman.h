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
#ifndef SRC_LIBZLING_HUFFMAN_H
#define SRC_LIBZLING_HUFFMAN_H

#include "libzling_inc.h"

namespace baidu {
namespace zling {
namespace huffman {

// ZlingMakeDecodeTable: build canonical length table from frequency table,
//  both tables should have kHuffmanSymbols elements.
//
//  arg freq_table   frequency_table
//  arg length_table length_table
//  arg scaling      scaling factor
//  arg max_codes    max codes       -- codes shoude be even
//  arg max_codelen  max code length -- codelen should be < 16
void ZlingMakeLengthTable(const uint32_t* freq_table,
                          uint32_t* length_table,
                          int scaling,
                          int max_codes,
                          int max_codelen);

// ZlingMakeEncodeTable: build encode table from canonical length table.
void ZlingMakeEncodeTable(const uint32_t* length_table,
                          uint16_t* encode_table,
                          int max_codes,
                          int max_codelen);

// ZlingMakeDecodeTable: build decode table from canonical length table.
void ZlingMakeDecodeTable(const uint32_t* length_table,
                          uint16_t* encode_table,
                          uint16_t* decode_table,
                          int max_codes,
                          int max_codelen);

}  // namespace huffman
}  // namespace zling
}  // namespace baidu
#endif  // SRC_LIBZLING_HUFFMAN_H

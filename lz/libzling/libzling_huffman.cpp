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
#include <functional>

namespace baidu {
namespace zling {
namespace huffman {

void ZlingMakeLengthTable(const uint32_t* freq_table, uint32_t* length_table, int max_codes, int max_codelen) {
    std::fill(&length_table[0], &length_table[max_codes], 0);
    auto scaling = 0;

    struct huffman_node {
        int id;
        int weight;
        huffman_node* child1;
        huffman_node* child2;

        inline huffman_node(int id, int weight, huffman_node* child1 = NULL, huffman_node* child2 = NULL) {
            this->id = id;
            this->weight = weight;
            this->child1 = child1;
            this->child2 = child2;
        }

        inline ~huffman_node() {
            delete child1;
            delete child2;
        }

        struct ptr_weight_gt_comparator {
            inline bool operator()(const huffman_node* lhs, const huffman_node* rhs) const {
                return lhs->weight > rhs->weight;
            }
        };
    };

build_huffman:
    // setup heap of leaf nodes
    auto nodes = std::vector<huffman_node*>();

    for (auto i = 0; i < max_codes; i++) {
        if (freq_table[i] > 0) {
            nodes.push_back(new huffman_node(i, (freq_table[i] + ((1 << scaling) - 1)) >> scaling));
        }
    }
    if (nodes.empty()) {
        return;
    }
    auto nodes_heap = std::priority_queue<
        huffman_node*,
        std::vector<huffman_node*>,
        huffman_node::ptr_weight_gt_comparator>(nodes.begin(), nodes.end());

    // construct huffman tree
    while (nodes_heap.size() > 1) {
        auto min1 = nodes_heap.top(); nodes_heap.pop();
        auto min2 = nodes_heap.top(); nodes_heap.pop();
        nodes_heap.push(new huffman_node(-1, min1->weight + min2->weight, min1, min2));
    }

    // extract code length
    std::function<void (huffman_node*, int)> code_length_extractor = [&](auto node, auto code_length) {
        if (node->id >= 0) {
            length_table[node->id] = std::max(code_length, 1);
        } else {
            code_length_extractor(node->child1, code_length + 1);
            code_length_extractor(node->child2, code_length + 1);
        }
    };
    code_length_extractor(nodes_heap.top(), 0);
    delete nodes_heap.top();

    // need rescaling?
    if (*std::max_element(&length_table[0], &length_table[max_codes]) > max_codelen) {
        scaling++;
        goto build_huffman;
    }
    return;
}

void ZlingMakeEncodeTable(const uint32_t* length_table, uint16_t* encode_table, int max_codes, int max_codelen) {
    std::fill(&encode_table[0], &encode_table[max_codes], 0);
    auto code = 0;

    // make code for each symbol
    for (auto codelen = 1; codelen <= max_codelen; codelen++) {
        for (auto i = 0; i < max_codes; i++) {
            if (length_table[i] == static_cast<uint32_t>(codelen)) {
                encode_table[i] = code;
                code += 1;
            }
        }
        code *= 2;
    }

    // reverse each code
    for (auto i = 0; i < max_codes; i++) {
        encode_table[i] = ((encode_table[i] & 0xff00) >> 8 | (encode_table[i] & 0x00ff) << 8);
        encode_table[i] = ((encode_table[i] & 0xf0f0) >> 4 | (encode_table[i] & 0x0f0f) << 4);
        encode_table[i] = ((encode_table[i] & 0xcccc) >> 2 | (encode_table[i] & 0x3333) << 2);
        encode_table[i] = ((encode_table[i] & 0xaaaa) >> 1 | (encode_table[i] & 0x5555) << 1);
        encode_table[i] >>= 16 - length_table[i];
    }
    return;
}

void ZlingMakeDecodeTable(const uint32_t* length_table, uint16_t* encode_table, uint16_t* decode_table,
    int max_codes,
    int max_codelen) {
    std::fill(&decode_table[0], &decode_table[1 << max_codelen], -1);

    for (auto c = 0; c < max_codes; c++) {
        if (length_table[c] > 0 && length_table[c] <= uint16_t(max_codelen)) {
            for (auto i = encode_table[c]; i < (1 << max_codelen); i += (1 << length_table[c])) {
                decode_table[i] = c;
            }
        }
    }
    return;
}

}  // namespace huffman
}  // namespace zling
}  // namespace baidu

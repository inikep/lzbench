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
 * @brief  manipulate ROLZ (reduced offset Lempel-Ziv) compression.
 */
#include "libzling_lz.h"
#include "libzling_debug.h"
#include <iostream>

namespace baidu {
namespace zling {
namespace lz {

static const unsigned char mtfinit[] = {
#   include "tables/table_mtfinit.inc"  /* include auto-generated constant tables */
};
static const unsigned char mtfnext[] = {
#   include "tables/table_mtfnext.inc"  /* include auto-generated constant tables */
};

#ifdef __GNUC__
static inline uint32_t RollingAdd(uint32_t x, uint32_t y) __attribute__((pure));
static inline uint32_t RollingSub(uint32_t x, uint32_t y) __attribute__((pure));
#endif

static inline uint32_t HashContext(unsigned char* ptr) {
    return (*reinterpret_cast<uint32_t*>(ptr) + ptr[2] * 137 + ptr[3] * 13337);
}

static inline uint32_t RollingAdd(uint32_t x, uint32_t y) {
    return (x + y) & (kBucketItemSize - 1);
}
static inline uint32_t RollingSub(uint32_t x, uint32_t y) {
    return (x - y) & (kBucketItemSize - 1);
}

static inline int GetCommonLength(unsigned char* buf1, unsigned char* buf2, int maxlen) {
    unsigned char* p1 = buf1;
    unsigned char* p2 = buf2;

    if (*reinterpret_cast<volatile uint32_t*>(p1) != *reinterpret_cast<volatile uint32_t*>(p2)) {
        return 0;
    }
    while (maxlen >= 4 && *reinterpret_cast<volatile uint32_t*>(p1) == *reinterpret_cast<volatile uint32_t*>(p2)) {
        p1 += 4;
        p2 += 4;
        maxlen -= 4;
    }
    if (maxlen >= 2 && *reinterpret_cast<volatile uint16_t*>(p1) == *reinterpret_cast<volatile uint16_t*>(p2)) {
        p1 += 2;
        p2 += 2;
        maxlen -= 2;
    }
    if (maxlen >= 1 && *reinterpret_cast<volatile uint8_t*>(p1) == *reinterpret_cast<volatile uint8_t*>(p2)) {
        p1 += 1;
        p2 += 1;
        maxlen -= 1;
    }
    return p1 - buf1;
}

static inline void IncrementalCopyFastPath(unsigned char* src, unsigned char* dst, int len) {
    while (dst - src < 4) {
        *reinterpret_cast<volatile uint32_t*>(dst) = *reinterpret_cast<volatile uint32_t*>(src);
        len -= dst - src;
        dst += dst - src;
    }
    while (len > 0) {
        *reinterpret_cast<volatile uint32_t*>(dst) = *reinterpret_cast<volatile uint32_t*>(src);
        len -= 4;
        dst += 4;
        src += 4;
    }
    return;
}

ZlingMTFEncoder::ZlingMTFEncoder() {
    memcpy(m_table, mtfinit, sizeof(m_table));
    for (int i = 0; i < 256; i++) {
        m_index[m_table[i]] = i;
    }
}
unsigned char ZlingMTFEncoder::Encode(unsigned char c) {
    unsigned char i = m_index[c];
    std::swap(m_index[c], m_index[m_table[mtfnext[i]]]);
    std::swap(m_table[i], m_table[mtfnext[i]]);
    return i;
}

ZlingMTFDecoder::ZlingMTFDecoder() {
    memcpy(m_table, mtfinit, sizeof(m_table));
}
unsigned char ZlingMTFDecoder::Decode(unsigned char i) {
    unsigned char c = m_table[i];
    std::swap(m_table[i], m_table[mtfnext[i]]);
    return c;
}


int ZlingRolzEncoder::Encode(unsigned char* ibuf, uint16_t* obuf, int ilen, int olen, int* encpos) {
    int ipos = encpos[0];
    int opos = 0;
    uint16_t word_mru[256][2] = {};

    // first byte
    if (ipos == 0 && opos < olen && ipos < ilen) obuf[opos++] = ibuf[ipos++];
    if (ipos == 1 && opos < olen && ipos < ilen) obuf[opos++] = ibuf[ipos++];

    while (opos + 1 < olen && ipos < ilen) {
        int match_idx;
        int match_len;

        // encode as match
        if (ipos + kMatchMaxLen + 16 < ilen) {  // avoid overflow
            if (MatchAndUpdate(ibuf, ipos, &match_idx, &match_len, m_match_depth)) {
                obuf[opos++] = 258 + match_len - kMatchMinLen;
                obuf[opos++] = match_idx;
                ipos += match_len;
                if (word_mru[ibuf[ipos - 3]][0] != (ibuf[ipos - 2] << 8 | ibuf[ipos - 1])) {
                    word_mru[ibuf[ipos - 3]][1] = word_mru[ibuf[ipos - 3]][0];
                    word_mru[ibuf[ipos - 3]][0] = ibuf[ipos - 2] << 8 | ibuf[ipos - 1];
                }
                continue;
            }
        }

        // encode as word
        if (ipos + 1 < ilen) {
            if (word_mru[ibuf[ipos - 1]][0] == (ibuf[ipos] << 8 | ibuf[ipos + 1])) {
                obuf[opos++] = 256;
                ipos += 2;
                continue;
            }
            if (word_mru[ibuf[ipos - 1]][1] == (ibuf[ipos] << 8 | ibuf[ipos + 1])) {
                obuf[opos++] = 257;
                ipos += 2;
                word_mru[ibuf[ipos - 3]][1] = word_mru[ibuf[ipos - 3]][0];
                word_mru[ibuf[ipos - 3]][0] = ibuf[ipos - 2] << 8 | ibuf[ipos - 1];
                continue;
            }
        }

        // encode as literal
        obuf[opos++] = m_mtf[ibuf[ipos - 1]].Encode(ibuf[ipos]);
        ipos++;
        word_mru[ibuf[ipos - 3]][1] = word_mru[ibuf[ipos - 3]][0];
        word_mru[ibuf[ipos - 3]][0] = ibuf[ipos - 2] << 8 | ibuf[ipos - 1];
    }
    encpos[0] = ipos;
    return opos;
}

void ZlingRolzEncoder::SetLevel(int compression_level) {
    m_match_depth = kPredefinedConfigs[compression_level].m_match_depth;
    m_lazymatch1_depth = kPredefinedConfigs[compression_level].m_lazymatch1_depth;
    m_lazymatch2_depth = kPredefinedConfigs[compression_level].m_lazymatch2_depth;
    return;
}

void ZlingRolzEncoder::Reset() {
    for (int context = 0; context < 256; context++) {
        for (int i = 0; i < kBucketItemSize; i++) {
            m_buckets[context].offset[i] = 0;
            m_buckets[context].suffix[i] = 65535;
        }
        for (int i = 0; i < kBucketItemHash; i++) {
            m_buckets[context].hash[i] = 65535;
        }
        m_buckets[context].head = 0;
    }
    return;
}

int inline ZlingRolzEncoder::MatchAndUpdate(unsigned char* buf, int pos, int* match_idx, int* match_len, int match_depth) {
    int maxlen = kMatchMinLen - 1;
    int maxnode = 0;
    uint32_t hash = HashContext(buf + pos);
    uint8_t  hash_check   = hash / kBucketItemHash % 256;
    uint32_t hash_context = hash % kBucketItemHash;

    ZlingEncodeBucket* bucket = &m_buckets[buf[pos - 1]];
    int node = bucket->hash[hash_context];

    // update befault matching (to make it faster)
    LIBZLING_DEBUG_COUNT("lz:update_bucket_node", 1);
    bucket->head = RollingAdd(bucket->head, 1);
    bucket->suffix[bucket->head] = bucket->hash[hash_context];
    bucket->offset[bucket->head] = pos | hash_check << 24;
    bucket->hash[hash_context] = bucket->head;

    // no match for first position
    // no match for currently updating entry
    if (node == 65535 || node == bucket->head) {
        LIBZLING_DEBUG_COUNT("lz:match_fail", 1);
        return 0;
    }

    // start matching
    for (int i = 0; i < match_depth; i++) {
        LIBZLING_DEBUG_COUNT("lz:access_bucket_node", 1);

        uint32_t offset = bucket->offset[node] & 0xffffff;
        uint8_t  check = bucket->offset[node] >> 24;
        if (check == hash_check) {
            LIBZLING_DEBUG_COUNT("lz:access_original_memory", 1);

            if (buf[pos + maxlen] == buf[offset + maxlen]) {
                LIBZLING_DEBUG_COUNT("lz:find_common_length", 1);
                int len = GetCommonLength(buf + pos, buf + offset, kMatchMaxLen);

                if (len > maxlen) {
                    maxnode = node;
                    maxlen = len;
                    if (maxlen == kMatchMaxLen) {
                        break;
                    }
                }
            }
        }
        node = bucket->suffix[node];

        // end chaining?
        if (node == 65535 || offset <= (bucket->offset[node] & 0xffffff)) {
            break;
        }
    }

    if (maxlen >= kMatchMinLen) {
        if (maxlen < kMatchMinLenEnableLazy) {  // fast and stupid lazy parsing
            if (m_lazymatch1_depth > 0 && MatchLazy(buf, pos + 1, maxlen, m_lazymatch1_depth)) {
                LIBZLING_DEBUG_COUNT("lz:lazy_skip_1", 1);
                LIBZLING_DEBUG_COUNT("lz:match_fail", 1);
                return 0;
            }
            if (m_lazymatch2_depth > 0 && MatchLazy(buf, pos + 2, maxlen, m_lazymatch2_depth)) {
                LIBZLING_DEBUG_COUNT("lz:lazy_skip_2", 1);
                LIBZLING_DEBUG_COUNT("lz:match_fail", 1);
                return 0;
            }
        }
        match_len[0] = maxlen;
        match_idx[0] = RollingSub(bucket->head, maxnode);
        LIBZLING_DEBUG_COUNT("lz:match_succ", 1);
        return 1;
    }
    LIBZLING_DEBUG_COUNT("lz:match_fail", 1);
    return 0;
}

int inline ZlingRolzEncoder::MatchLazy(unsigned char* buf, int pos, int maxlen, int depth) {
    ZlingEncodeBucket* bucket = &m_buckets[buf[pos - 1]];
    uint32_t hash = HashContext(buf + pos);
    uint32_t hash_context = hash % kBucketItemHash;

    int node = bucket->hash[hash_context];
    if (node == 65535) {
        return 0;
    }
    maxlen -= 3;

    for (int i = 0; i < depth; i++) {
        uint32_t offset = bucket->offset[node] & 0xffffff;

        if (*reinterpret_cast<uint32_t*>(buf + pos + maxlen) == *reinterpret_cast<uint32_t*>(buf + offset + maxlen)) {
            return 1;
        }
        node = bucket->suffix[node];

        // end chaining?
        if (node == 65535 || offset <= (bucket->offset[node] & 0xffffff)) {
            break;
        }
    }
    return 0;
}

int ZlingRolzDecoder::Decode(uint16_t* ibuf, unsigned char* obuf, int ilen, int encpos, int* decpos) {
    int opos = decpos[0];
    int ipos = 0;
    int match_idx;
    int match_len;
    int match_offset;
    uint16_t word_mru[256][2] = {};

    // first byte
    if (opos == 0 && ipos < ilen) obuf[opos++] = ibuf[ipos++];
    if (opos == 1 && ipos < ilen) obuf[opos++] = ibuf[ipos++];

    // rest byte
    while (ipos < ilen) {
        if (ibuf[ipos] < 256) {  // process a literal byte
            obuf[opos] = m_mtf[obuf[opos - 1]].Decode(ibuf[ipos]);
            ipos++;
            GetMatchAndUpdate(obuf, opos++, 0);
            word_mru[obuf[opos - 3]][1] = word_mru[obuf[opos - 3]][0];
            word_mru[obuf[opos - 3]][0] = obuf[opos - 2] << 8 | obuf[opos - 1];

        } else if (ibuf[ipos] == 256) {
            uint16_t word = word_mru[obuf[opos - 1]][0];
            ipos++;
            obuf[opos] = (word >> 8) & 0xff; GetMatchAndUpdate(obuf, opos++, 0);
            obuf[opos] = (word >> 0) & 0xff; opos++;

        } else if (ibuf[ipos] == 257) {
            uint16_t word = word_mru[obuf[opos - 1]][1];
            ipos++;
            obuf[opos] = (word >> 8) & 0xff; GetMatchAndUpdate(obuf, opos++, 0);
            obuf[opos] = (word >> 0) & 0xff; opos++;
            word_mru[obuf[opos - 3]][1] = word_mru[obuf[opos - 3]][0];
            word_mru[obuf[opos - 3]][0] = obuf[opos - 2] << 8 | obuf[opos - 1];

        } else {  // process a match
            match_len = ibuf[ipos++] - 258 + kMatchMinLen;
            match_idx = ibuf[ipos++];
            match_offset = GetMatchAndUpdate(obuf, opos, match_idx);

            IncrementalCopyFastPath(&obuf[match_offset], &obuf[opos], match_len);
            opos += match_len;
            if (word_mru[obuf[opos - 3]][0] != (obuf[opos - 2] << 8 | obuf[opos - 1])) {
                word_mru[obuf[opos - 3]][1] = word_mru[obuf[opos - 3]][0];
                word_mru[obuf[opos - 3]][0] = obuf[opos - 2] << 8 | obuf[opos - 1];
            }
        }

        if (opos > encpos) {
            return -1;
        }
    }

    if (opos != encpos) {
        return -1;
    }
    decpos[0] = opos;
    return 0;
}

void ZlingRolzDecoder::Reset() {
    for (int context = 0; context < 256; context++) {
        for (int i = 0; i < kBucketItemSize; i++) {
            m_buckets[context].offset[i] = 0;
        }
        m_buckets[context].head = 0;
    }
    return;
}

int inline ZlingRolzDecoder::GetMatchAndUpdate(unsigned char* buf, int pos, int idx) {
    ZlingDecodeBucket* bucket = &m_buckets[buf[pos - 1]];
    int node;

    // update
    bucket->head = RollingAdd(bucket->head, 1);
    bucket->offset[bucket->head] = pos;

    // get match
    node = RollingSub(bucket->head, idx);
    return bucket->offset[node];
}

}  // namespace lz
}  // namespace zling
}  // namespace baidu

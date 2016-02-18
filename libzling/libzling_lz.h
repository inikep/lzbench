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
#ifndef SRC_LIBZLING_LZ_H
#define SRC_LIBZLING_LZ_H

#include "libzling_inc.h"

namespace baidu {
namespace zling {
namespace lz {

static const int kBucketItemSize = 4096;
static const int kBucketItemHash = 8192;
static const int kMatchMinLenEnableLazy = 128;
static const int kMatchMinLen = 4;
static const int kMatchMaxLen = 259;

static const struct {
    int m_match_depth;
    int m_lazymatch1_depth;
    int m_lazymatch2_depth;

} kPredefinedConfigs[] = {
    {2,  1, 0},
    {4,  1, 0},
    {6,  2, 0},
    {8,  3, 1},
    {16, 4, 2},
};

class ZlingMTFEncoder {
public:
    ZlingMTFEncoder();
    unsigned char Encode(unsigned char c);
private:
    unsigned char m_table[256];
    unsigned char m_index[256];
};

class ZlingMTFDecoder {
public:
    ZlingMTFDecoder();
    unsigned char Decode(unsigned char i);
private:
    unsigned char m_table[256];
};

class ZlingRolzEncoder {
public:
    ZlingRolzEncoder(int compression_level = 0) {
        SetLevel(compression_level);
        Reset();
    }

    /* Encode:
     *  arg ibuf:   input data
     *  arg obuf:   output data (compressed)
     *  arg ilen:   input data length
     *  arg olen:   input data length
     *  arg decpos: start encoding at ibuf[encpos], limited by ilen and olen
     *  ret: out length.
     */
    int  Encode(unsigned char* ibuf, uint16_t* obuf, int ilen, int olen, int* encpos);
    void SetLevel(int compression_level);
    void Reset();

private:
    int MatchAndUpdate(unsigned char* buf, int pos, int* match_idx, int* match_len, int match_depth);
    int MatchLazy(unsigned char* buf, int pos, int maxlen, int depth);

    struct ZlingEncodeBucket {
        uint16_t suffix[kBucketItemSize];
        uint32_t offset[kBucketItemSize];
        uint16_t head;
        uint16_t hash[kBucketItemHash];
    };
    ZlingEncodeBucket m_buckets[256];
    ZlingMTFEncoder m_mtf[256];
    int m_match_depth;
    int m_lazymatch1_depth;
    int m_lazymatch2_depth;

    ZlingRolzEncoder(const ZlingRolzEncoder&);
    ZlingRolzEncoder& operator = (const ZlingRolzEncoder&);
};

class ZlingRolzDecoder {
public:
    ZlingRolzDecoder() {
        Reset();
    }

    /* Decode:
     *  arg ibuf:   input data (compressed)
     *  arg obuf:   output data
     *  arg ilen:   input data length
     *  arg encpos: encpos check
     *  arg decpos: start decoding at obuf[decpos], limited by ilen
     *  ret: -1: failed
     *        0: success
     */
    int  Decode(uint16_t* ibuf, unsigned char* obuf, int ilen, int encpos, int* decpos);
    void Reset();

private:
    int GetMatchAndUpdate(unsigned char* buf, int pos, int idx);

    struct ZlingDecodeBucket {
        uint32_t offset[kBucketItemSize];
        uint16_t head;
    };
    ZlingDecodeBucket m_buckets[256];
    ZlingMTFDecoder m_mtf[256];
    ZlingRolzDecoder(const ZlingRolzDecoder&);
    ZlingRolzDecoder& operator = (const ZlingRolzDecoder&);
};

}  // namespace lz
}  // namespace zling
}  // namespace baidu
#endif  // SRC_LIBZLING_LZ_H

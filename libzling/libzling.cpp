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
 * @brief  libzling.
 */
#include "libzling.h"
#include "libzling_debug.h"
#include "libzling_huffman.h"
#include "libzling_lz.h"

namespace baidu {
namespace zling {

using huffman::ZlingMakeLengthTable;
using huffman::ZlingMakeEncodeTable;
using huffman::ZlingMakeDecodeTable;
using lz::ZlingRolzEncoder;
using lz::ZlingRolzDecoder;

using lz::kMatchMaxLen;
using lz::kMatchMinLen;
using lz::kBucketItemSize;

static const uint32_t matchidx_bitlen[] = {
#   include "tables/table_matchidx_blen.inc"  /* include auto-generated constant tables */
};
static const uint32_t matchidx_code[] = {
#   include "tables/table_matchidx_code.inc"  /* include auto-generated constant tables */
};
static const uint32_t matchidx_base[] = {
#   include "tables/table_matchidx_base.inc"  /* include auto-generated constant tables */
};

static const int kHuffmanCodes1      = 258 + (kMatchMaxLen - kMatchMinLen + 1);
static const int kHuffmanCodes2      = sizeof(matchidx_base) / sizeof(matchidx_base[0]);
static const int kHuffmanMaxLen1     = 15;
static const int kHuffmanMaxLen2     = 8;
static const int kHuffmanMaxLen1Fast = 10;
static const int kSentinelLen        = kMatchMaxLen + 16;

static const int kBlockSizeIn      = 16777216;
static const int kBlockSizeRolz    = 262144;
static const int kBlockSizeHuffman = 393216;

/* codebuf: manipulate code (u64) buffer.
 *  Input();
 *  Output();
 *  Peek();
 *  GetLength();
 */
struct ZlingCodebuf {
    ZlingCodebuf():
        m_buf(0),
        m_len(0) {}

    inline void Input(uint64_t code, int len) {
        m_buf |= code << m_len;
        m_len += len;
        return;
    }
    inline uint64_t Output(int len) {
        uint64_t out = Peek(len);
        m_buf >>= len;
        m_len  -= len;
        return out;
    }
    inline uint64_t Peek(int len) const {
        return m_buf & ~(-1ull << len);
    }
    inline int GetLength() const {
        return m_len;
    }
private:
    uint64_t m_buf;
    int m_len;
};

/* encode/decode allocation resource: auto free */
struct EncodeResource {
    ZlingRolzEncoder* lzencoder;
    unsigned char* ibuf;
    unsigned char* obuf;
    uint16_t* tbuf;
    int level;

    EncodeResource(int level): lzencoder(NULL), ibuf(NULL), obuf(NULL), tbuf(NULL), level(level) {
        try {
            ibuf = new unsigned char[kBlockSizeIn + kSentinelLen];
            obuf = new unsigned char[kBlockSizeHuffman + kSentinelLen];
            tbuf = new uint16_t[kBlockSizeRolz + kSentinelLen];
            lzencoder = new ZlingRolzEncoder(level);

        } catch (const std::bad_alloc& e) {
            delete lzencoder;
            delete [] ibuf;
            delete [] obuf;
            delete [] tbuf;
            throw std::bad_alloc();
        }
    }
    ~EncodeResource() {
        delete lzencoder;
        delete [] ibuf;
        delete [] obuf;
        delete [] tbuf;
    }
};
struct DecodeResource {
    ZlingRolzDecoder* lzdecoder;
    unsigned char* ibuf;
    unsigned char* obuf;
    uint16_t* tbuf;

    DecodeResource(): lzdecoder(NULL), ibuf(NULL), obuf(NULL), tbuf(NULL) {
        try {
            ibuf = new unsigned char[kBlockSizeIn + kSentinelLen];
            obuf = new unsigned char[kBlockSizeHuffman + kSentinelLen];
            tbuf = new uint16_t[kBlockSizeRolz + kSentinelLen];
            lzdecoder = new ZlingRolzDecoder();

        } catch (const std::bad_alloc& e) {
            delete lzdecoder;
            delete [] ibuf;
            delete [] obuf;
            delete [] tbuf;
            throw std::bad_alloc();
        }
    }
    ~DecodeResource() {
        delete lzdecoder;
        delete [] ibuf;
        delete [] obuf;
        delete [] tbuf;
    }
};

#define CHECK_IO_ERROR(io) do { \
    if ((io)->IsErr()) { \
        goto EncodeOrDecodeFinished; \
    } \
} while(0)

static const int kFlagRolzContinue = 1;
static const int kFlagRolzStop     = 0;

int Encode(Inputter* inputter, Outputter* outputter, ActionHandler* action_handler, int level) {
    if (action_handler) {
        action_handler->SetInputterOutputter(inputter, outputter, true);
        action_handler->OnInit();
    }

    EncodeResource res(level);
    int rlen;
    int ilen;
    int olen;
    int encpos;

    while (!inputter->IsEnd() && !inputter->IsErr()) {
        rlen = 0;
        ilen = 0;
        olen = 0;
        encpos = 0;

        while(!inputter->IsEnd() && !inputter->IsErr() && ilen < kBlockSizeIn) {
            ilen += inputter->GetData(res.ibuf + ilen, kBlockSizeIn - ilen);
            CHECK_IO_ERROR(inputter);
        }
        res.lzencoder->Reset();

        while (encpos < ilen) {
            outputter->PutChar(kFlagRolzContinue);
            CHECK_IO_ERROR(outputter);

            // ROLZ encode
            // ============================================================
            int encpos_old = encpos;
            rlen = res.lzencoder->Encode(res.ibuf, res.tbuf, ilen, kBlockSizeRolz, &encpos);

            // HUFFMAN encode
            // ============================================================
            ZlingCodebuf codebuf;
            int opos = 0;
            uint32_t freq_table1[kHuffmanCodes1] = {0};
            uint32_t freq_table2[kHuffmanCodes2] = {0};
            uint32_t length_table1[kHuffmanCodes1 + (kHuffmanCodes1 % 2)] = {0};
            uint32_t length_table2[kHuffmanCodes2 + (kHuffmanCodes2 % 2)] = {0};
            uint16_t encode_table1[kHuffmanCodes1];
            uint16_t encode_table2[kHuffmanCodes2];

            for (int i = 0; i < rlen; i++) {
                freq_table1[res.tbuf[i]] += 1;
                if (res.tbuf[i] >= 258) {
                    freq_table2[matchidx_code[res.tbuf[++i]]] += 1;
                }
            }
            ZlingMakeLengthTable(freq_table1, length_table1, 0, kHuffmanCodes1, kHuffmanMaxLen1);
            ZlingMakeLengthTable(freq_table2, length_table2, 0, kHuffmanCodes2, kHuffmanMaxLen2);

            ZlingMakeEncodeTable(length_table1, encode_table1, kHuffmanCodes1, kHuffmanMaxLen1);
            ZlingMakeEncodeTable(length_table2, encode_table2, kHuffmanCodes2, kHuffmanMaxLen2);

            // write length table
            for (int i = 0; i < kHuffmanCodes1; i += 2) {
                res.obuf[opos++] = length_table1[i] * 16 + length_table1[i + 1];
            }
            for (int i = 0; i < kHuffmanCodes2; i += 2) {
                res.obuf[opos++] = length_table2[i] * 16 + length_table2[i + 1];
            }

            // encode
            for (int i = 0; i < rlen; i++) {
                codebuf.Input(encode_table1[res.tbuf[i]], length_table1[res.tbuf[i]]);
                if (res.tbuf[i] >= 258) {
                    uint32_t code = matchidx_code[res.tbuf[++i]];

                    codebuf.Input(
                        encode_table2[code],
                        length_table2[code]);
                    codebuf.Input(
                        res.tbuf[i] - matchidx_base[code],
                        matchidx_bitlen[code]);
                }
                if (codebuf.GetLength() >= 32) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
                    *reinterpret_cast<uint32_t*>(res.obuf + opos) = codebuf.Output(32);
                    opos += 4;
#else
                    res.obuf[opos++] = codebuf.Output(8);
                    res.obuf[opos++] = codebuf.Output(8);
                    res.obuf[opos++] = codebuf.Output(8);
                    res.obuf[opos++] = codebuf.Output(8);
#endif
                }
            }
            while (codebuf.GetLength() > 0) {
                res.obuf[opos++] = codebuf.Output(8);
            }
            olen = opos;

            // lower level for uncompressible data
            if (1.0 * olen / (encpos - encpos_old + 1) > 0.95) {
                LIBZLING_DEBUG_COUNT("lz:uncompressible", 1);
                res.lzencoder->SetLevel(0);
            } else {
                res.lzencoder->SetLevel(res.level);
            }

            // outputter
            outputter->PutUInt32(encpos); CHECK_IO_ERROR(outputter);
            outputter->PutUInt32(rlen);   CHECK_IO_ERROR(outputter);
            outputter->PutUInt32(olen);   CHECK_IO_ERROR(outputter);

            for (int ooff = 0; !outputter->IsErr() && ooff < olen; ) {
                ooff += outputter->PutData(res.obuf + ooff, olen - ooff);
                CHECK_IO_ERROR(outputter);
            }
        }
        outputter->PutChar(kFlagRolzStop);
        CHECK_IO_ERROR(outputter);

        if (action_handler) {
            action_handler->OnProcess(res.ibuf, ilen);
        }
    }

EncodeOrDecodeFinished:
    if (action_handler) {
        action_handler->OnDone();
    }
    return (inputter->IsErr() || outputter->IsErr()) ? -1 : 0;
}

int Decode(Inputter* inputter, Outputter* outputter, ActionHandler* action_handler) {
    if (action_handler) {
        action_handler->SetInputterOutputter(inputter, outputter, false);
        action_handler->OnInit();
    }

    DecodeResource res;
    int rlen;
    int olen;
    int encflag;
    int encpos;
    int decpos;

    while (!inputter->IsEnd()) {
        olen = 0;
        rlen = 0;
        decpos = 0;
        res.lzdecoder->Reset();

        while (!inputter->IsEnd()) {
            encflag = inputter->GetChar();

            if (encflag != kFlagRolzStop && encflag != kFlagRolzContinue) { /* error: invalid encflag */
                throw std::runtime_error("baidu::zling::Decode(): invalid encflag.");
            }
            if (encflag == kFlagRolzStop) {
                break;
            }

            encpos = inputter->GetUInt32(); CHECK_IO_ERROR(inputter);
            rlen   = inputter->GetUInt32(); CHECK_IO_ERROR(inputter);
            olen   = inputter->GetUInt32(); CHECK_IO_ERROR(inputter);

            if (rlen > kBlockSizeRolz || olen > kBlockSizeHuffman) {
                throw std::runtime_error("baidu::zling::Decode(): invalid block size.");
            }
            for (int ooff = 0; !inputter->IsEnd() && ooff < olen; ) {
                ooff += inputter->GetData(res.obuf + ooff, olen - ooff);
                CHECK_IO_ERROR(inputter);
            }

            // HUFFMAN DECODE
            // ============================================================
            ZlingCodebuf codebuf;
            int opos = 0;
            uint32_t length_table1[kHuffmanCodes1 + (kHuffmanCodes1 % 2)] = {0};
            uint32_t length_table2[kHuffmanCodes2 + (kHuffmanCodes2 % 2)] = {0};
            uint16_t decode_table1[1 << kHuffmanMaxLen1];
            uint16_t decode_table2[1 << kHuffmanMaxLen2];
            uint16_t decode_table1_fast[1 << kHuffmanMaxLen1Fast];
            uint16_t encode_table1[kHuffmanCodes1];
            uint16_t encode_table2[kHuffmanCodes2];

            // read length table
            for (int i = 0; i < kHuffmanCodes1; i += 2) {
                length_table1[i + 0] = res.obuf[opos] / 16;
                length_table1[i + 1] = res.obuf[opos] % 16;
                opos++;
            }
            for (int i = 0; i < kHuffmanCodes2; i += 2) {
                length_table2[i + 0] = res.obuf[opos] / 16;
                length_table2[i + 1] = res.obuf[opos] % 16;
                opos++;
            }
            ZlingMakeEncodeTable(length_table1, encode_table1, kHuffmanCodes1, kHuffmanMaxLen1);
            ZlingMakeEncodeTable(length_table2, encode_table2, kHuffmanCodes2, kHuffmanMaxLen2);

            // decode_table1: 2-level decode table
            ZlingMakeDecodeTable(length_table1,
                                 encode_table1,
                                 decode_table1,
                                 kHuffmanCodes1,
                                 kHuffmanMaxLen1);
            ZlingMakeDecodeTable(length_table1,
                                 encode_table1,
                                 decode_table1_fast,
                                 kHuffmanCodes1,
                                 kHuffmanMaxLen1Fast);

            // decode_table2: 1-level decode table
            ZlingMakeDecodeTable(length_table2,
                                 encode_table2,
                                 decode_table2,
                                 kHuffmanCodes2,
                                 kHuffmanMaxLen2);

            // decode
            for (int i = 0; i < rlen; i++) {
                if (codebuf.GetLength() < 32) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
                    codebuf.Input(*reinterpret_cast<uint32_t*>(res.obuf + opos), 32);
                    opos += 4;
#else
                    codebuf.Input(res.obuf[opos++], 8);
                    codebuf.Input(res.obuf[opos++], 8);
                    codebuf.Input(res.obuf[opos++], 8);
                    codebuf.Input(res.obuf[opos++], 8);
#endif
                }

                res.tbuf[i] = decode_table1_fast[codebuf.Peek(kHuffmanMaxLen1Fast)];
                if (res.tbuf[i] == uint16_t(-1)) {
                    res.tbuf[i] = decode_table1[codebuf.Peek(kHuffmanMaxLen1)];
                }

                if (res.tbuf[i] >= kHuffmanCodes1) { /* error: literal/length >= kHuffmanCodes1 */
                    throw std::runtime_error("baidu::zling::Decode(): invalid huffman stream. (bad code1)");
                }
                codebuf.Output(length_table1[res.tbuf[i]]);

                if (res.tbuf[i] >= 258) {
                    uint32_t code;
                    uint32_t bits;

                    /* error: matchidx.code >= kHuffmanCodes2 */
                    if((code = decode_table2[codebuf.Peek(kHuffmanMaxLen2)]) >= kHuffmanCodes2) {
                        throw std::runtime_error("baidu::zling::Decode(): invalid huffman stream. (bad code2)");
                    }
                    codebuf.Output(length_table2[code]);
                    bits = codebuf.Output(matchidx_bitlen[code]);

                    /* error: matchidx >= kBucketItemSize */
                    if ((res.tbuf[++i] = matchidx_base[code] + bits) >= kBucketItemSize) {
                        throw std::runtime_error("baidu::zling::Decode(): invalid huffman stream. (bad ex-bits)");
                    }
                }
            }

            // ROLZ decode
            // ============================================================
            if (res.lzdecoder->Decode(res.tbuf, res.ibuf, rlen, encpos, &decpos) == -1) { /* error: lz.Decode failed */
                throw std::runtime_error("baidu::zling::Decode(): lzdecode failed.");
            }
        }

        // output
        for (int ioff = 0; !outputter->IsErr() && ioff < decpos; ) {
            ioff += outputter->PutData(res.ibuf + ioff, decpos - ioff);
            CHECK_IO_ERROR(outputter);
        }

        if (action_handler) {
            action_handler->OnProcess(res.ibuf, decpos);
        }
    }

EncodeOrDecodeFinished:
    if (action_handler) {
        action_handler->OnDone();
    }
    return (inputter->IsErr() || outputter->IsErr()) ? -1 : 0;
}

}  // namespace zling
}  // namespace baidu

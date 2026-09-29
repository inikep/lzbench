// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define HUF_STATIC_LINKING_ONLY

#include "openzl/codecs/entropy/deprecated/common_huf_avx2.h"
#include "openzl/codecs/entropy/encode_huffman_kernel.h"
#include "openzl/fse/hist.h"
#include "openzl/fse/huf.h"
#include "openzl/shared/mem.h"
#include "openzl/zl_errors.h"

#define kMaxHufLog 12
#define kMaxHuf16Log 13
#define kNumStates 32

size_t ZS_HufAvx2_encodeBound(size_t srcSize)
{
    return 100 + srcSize;
}

struct HUF_CElt_s {
    uint16_t val;
    uint8_t nbBits;
};

#define ZS_HUF_RET_ERR_IF(cond) \
    do {                        \
        if (cond)               \
            return 0;           \
    } while (0)

size_t ZS_HufAvx2_encode(
        void* dst,
        size_t dstCapacity,
        void const* src,
        size_t srcSize)
{
    uint8_t* ostart = (uint8_t*)dst;
    uint8_t* op     = ostart;
    uint8_t* oend   = ostart + dstCapacity;
    {
        ZS_HUF_RET_ERR_IF((size_t)(oend - op) < 4);
        uint32_t total = (uint32_t)srcSize;
        memcpy(op, &total, sizeof(total));
        op += sizeof(total);
    }
    HUF_CREATE_STATIC_CTABLE(ctable, 255);
    uint32_t msv = 255;
    size_t hlog;
    {
        uint32_t hist[256];
        unsigned cardinality = 0;
        size_t mfs = HIST_count(hist, &msv, &cardinality, src, srcSize);
        ZS_HUF_RET_ERR_IF(HIST_isError(mfs));
        if (mfs == srcSize) {
            ZS_HUF_RET_ERR_IF((oend - op) < 2);
            *op++ = 1;
            *op++ = (uint8_t)msv;
            return (size_t)(op - ostart);
        }
        hlog = HUF_buildCTable(ctable, hist, msv, kMaxHufLog);
        if (HUF_isError(hlog)) {
            ZS_HUF_RET_ERR_IF((size_t)(oend - op) < srcSize + 1);
            *op++ = 0;
            memcpy(op, src, srcSize);
            op += srcSize;
            return (size_t)(op - ostart);
        }
        ZS_HUF_RET_ERR_IF((size_t)(oend - op) < 1);
        *op++ = 2;
    }
    {
        size_t const csize = HUF_writeCTable(
                op, (size_t)(oend - op), ctable, msv, (unsigned)hlog);
        ZS_HUF_RET_ERR_IF(HUF_isError(csize));
        op += csize;
    }

    uint32_t reload[kNumStates];
    uint32_t states[kNumStates];
    uint32_t bits[kNumStates];

    memset(reload, 0xFF, sizeof(reload));
    memset(states, 0, sizeof(states));
    memset(bits, 0, sizeof(bits));

    uint8_t const* istart = (uint8_t const*)src;
    uint8_t const* ip     = istart;
    uint8_t const* iend   = istart + srcSize;

    ZS_HUF_RET_ERR_IF((oend - op) < 4);
    uint8_t* const csize = op;
    op += 4;
    uint8_t* const bstart = op;
    unsigned start        = kNumStates - 1;
    for (ip = iend - 1; ip >= istart; ip -= kNumStates) {
        for (unsigned i = 0; i < kNumStates; ++i) {
            if (ip - i < istart) {
                start = (i - 1) & (kNumStates - 1);
                break;
            }
            uint8_t const symbol  = ip[-(int)i];
            HUF_CElt const elt    = ctable[symbol + 1];
            uint32_t const nbBits = (uint32_t)HUF_getNbBits(elt);
            if (bits[i] + nbBits >= 32) {
                if (reload[i] == 0xFFFFFFFF) {
                    reload[i] = (uint32_t)(op - bstart);
                }
                bits[i] -= 16;
                uint16_t const value = states[i] & 0xFFFF;
                ZS_HUF_RET_ERR_IF((oend - op) < 2);
                ZL_writeLE16(op, value);
                op += 2;
                states[i] >>= 16;
            }
            assert((states[i] & ~((1u << bits[i]) - 1)) == 0);
            // fprintf(stderr, "%d %c: %u %u\n", i, (char)symbol,
            // (unsigned)ctable[symbol].val, (unsigned)ctable[symbol].nbBits);
            assert(nbBits > 0);
            assert(nbBits + bits[i] < 64);
            states[i] |=
                    (uint32_t)((HUF_getValue(elt) >> (64 - nbBits)) << bits[i]);
            bits[i] += nbBits;
            assert(bits[i] <= 32);
        }
    }
    // fprintf(stderr, "start = %u\n", start);
    ZS_HUF_RET_ERR_IF((size_t)(oend - op) < kNumStates * sizeof(uint32_t));
    uint32_t const cs = (uint32_t)((op - csize) - 4);
    memcpy(csize, &cs, sizeof(cs));
    for (unsigned i = 0; i < kNumStates; ++i) {
        unsigned idx = (start - i) & 31;
        // fprintf(stderr, "XXX %u -> %u\n", idx, i);
        // fprintf(stderr, "s[%u] = 0x%x\n", i, states[idx]);
        // assert(bits[idx] >= 16);
        memcpy(op + 4 * i, &states[idx], sizeof(states[idx]));
    }
    op += sizeof(uint32_t) * kNumStates;
    ZS_HUF_RET_ERR_IF((size_t)(oend - op) < kNumStates * sizeof(uint32_t));
    for (unsigned i = 0; i < kNumStates; ++i) {
        unsigned idx = (start - i) & 31;
        memcpy(op + 4 * i, &reload[idx], sizeof(reload[idx]));
    }
    op += sizeof(uint32_t) * kNumStates;
    ZS_HUF_RET_ERR_IF((size_t)(oend - op) < kNumStates);
    for (unsigned i = 0; i < kNumStates; ++i) {
        unsigned idx = (start - i) & 31;
        *op++        = (uint8_t)bits[idx];
    }
    // for (size_t i = 0; i < 32; ++i) {
    //     uint32_t si;
    //     memcpy(&si, initial + 4 * i, sizeof(si));
    //     fprintf(stderr, "s[%zu] = 0x%x (init = %u)\n", i, si,
    //     (unsigned)inited[i]);
    // }
    // fprintf(stderr, "hlog=%zu\n", hlog);
    return (size_t)(op - ostart);
}

size_t ZS_Huf16Avx2_encodeBound(size_t srcSize)
{
    return 100 + 2 * srcSize;
}

size_t ZS_Huf16Avx2_encode(
        void* dst,
        size_t dstCapacity,
        void const* src,
        size_t srcSize)
{
    uint8_t* ostart       = (uint8_t*)dst;
    uint8_t* op           = ostart;
    uint8_t* oend         = ostart + dstCapacity;
    uint16_t const* src16 = (uint16_t const*)src;
    {
        ZS_HUF_RET_ERR_IF((size_t)(oend - op) < 4);
        uint32_t total = (uint32_t)srcSize;
        memcpy(op, &total, sizeof(total));
        op += sizeof(total);
    }
    ZS_Huf16CElt ctable[1u << 12];
    uint16_t msv = (1u << 12) - 1;
    int hlog;
    {
        uint32_t hist[1u << 12];
        memset(hist, 0, sizeof(hist));
        for (size_t i = 0; i < srcSize; ++i) {
            ZS_HUF_RET_ERR_IF(src16[i] >= (1u << 12));
            ++hist[src16[i]];
        }
        while (hist[msv] == 0 && msv > 0) {
            --msv;
        }
        if (hist[msv] == srcSize) {
            ZS_HUF_RET_ERR_IF((size_t)(oend - op) < 2);
            *op++ = 1;
            ZL_writeLE16(op, msv);
            op += 2;
            return (size_t)(op - ostart);
        }
        ZL_Report const report =
                ZS_largeHuffmanBuildCTable(ctable, hist, msv, kMaxHuf16Log);
        ZS_HUF_RET_ERR_IF(ZL_isError(report));
        hlog = (int)ZL_validResult(report);
        ZS_HUF_RET_ERR_IF((size_t)(oend - op) < 1);
        *op++ = 2;
    }
    {
        ZL_WC wc = ZL_WC_wrap(op, (size_t)(oend - op));
        ZL_Report const report =
                ZS_largeHuffmanWriteCTable(&wc, ctable, msv, hlog);
        ZS_HUF_RET_ERR_IF(ZL_isError(report));
        op = ZL_WC_ptr(&wc);
    }

    uint32_t reload[kNumStates];
    uint32_t states[kNumStates];
    uint32_t bits[kNumStates];

    memset(reload, 0xFF, sizeof(reload));
    memset(states, 0, sizeof(states));
    memset(bits, 0, sizeof(bits));

    uint16_t const* istart = src16;
    uint16_t const* ip     = istart;
    uint16_t const* iend   = istart + srcSize;

    ZS_HUF_RET_ERR_IF((size_t)(oend - op) < 4);
    uint8_t* const csize = op;
    op += 4;
    uint8_t* const bstart = op;
    unsigned start        = kNumStates - 1;
    for (ip = iend - 1; ip >= istart; ip -= kNumStates) {
        // for (; ip < iend; ip += 32) {
        for (unsigned i = 0; i < kNumStates; ++i) {
            // for (int i = 0; i < 32; ++i) {
            if (ip - i < istart) {
                start = (i - 1) & (kNumStates - 1);
                break;
            }
            uint16_t const symbol  = ip[-(int)i];
            ZS_Huf16CElt const elt = ctable[symbol];
            ZL_ASSERT_LE(elt.nbBits, kMaxHuf16Log);
            if (bits[i] + elt.nbBits >= 32) {
                if (reload[i] == 0xFFFFFFFF) {
                    reload[i] = (uint32_t)(op - bstart);
                }
                // if (bits[i] > 16) {
                bits[i] -= 16;
                uint16_t const value = states[i] & 0xFFFF;
                // fprintf(stderr, "%zu flush %d (%x): %u -> %u\n", ip - istart,
                // i, (int)value, bits[i] + 16, bits[i]);
                ZS_HUF_RET_ERR_IF((size_t)(oend - op) < 2);
                ZL_writeLE16(op, value);
                op += 2;
                states[i] >>= 16;
            }
            // assert(bits[i] >= 16);
            assert((states[i] & ~((1u << bits[i]) - 1)) == 0);
            // assert(bits[i] <= 16);
            // fprintf(stderr, "%d %c: %u %u\n", i, (char)symbol,
            // (unsigned)ctable[symbol].val, (unsigned)ctable[symbol].nbBits);
            states[i] |= (unsigned)(elt.symbol << bits[i]);
            bits[i] += elt.nbBits;
            assert(bits[i] <= 32);
        }
    }
    // fprintf(stderr, "start = %u\n", start);
    uint32_t const cs = (uint32_t)(op - csize) - 4;
    memcpy(csize, &cs, sizeof(cs));
    ZS_HUF_RET_ERR_IF((size_t)(oend - op) < sizeof(uint32_t) * kNumStates);
    for (unsigned i = 0; i < kNumStates; ++i) {
        unsigned idx = (start - i) & 31;
        // fprintf(stderr, "XXX %u -> %u\n", idx, i);
        // fprintf(stderr, "s[%u] = 0x%x\n", i, states[idx]);
        // assert(bits[idx] >= 16);
        memcpy(op + 4 * i, &states[idx], sizeof(states[idx]));
    }
    op += sizeof(uint32_t) * kNumStates;
    ZS_HUF_RET_ERR_IF((size_t)(oend - op) < sizeof(uint32_t) * kNumStates);
    for (unsigned i = 0; i < kNumStates; ++i) {
        unsigned idx = (start - i) & 31;
        memcpy(op + 4 * i, &reload[idx], sizeof(reload[idx]));
    }
    op += sizeof(uint32_t) * kNumStates;
    ZS_HUF_RET_ERR_IF((size_t)(oend - op) < kNumStates);
    for (unsigned i = 0; i < kNumStates; ++i) {
        unsigned idx = (start - i) & 31;
        *op++        = (uint8_t)bits[idx];
    }
    // for (size_t i = 0; i < 32; ++i) {
    //     uint32_t si;
    //     memcpy(&si, initial + 4 * i, sizeof(si));
    //     fprintf(stderr, "s[%zu] = 0x%x (init = %u)\n", i, si,
    //     (unsigned)inited[i]);
    // }
    // fprintf(stderr, "hlog=%zu\n", hlog);
    return (size_t)(op - ostart);
}

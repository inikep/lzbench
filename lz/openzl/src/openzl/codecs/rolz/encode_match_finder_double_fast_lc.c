// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "openzl/codecs/common/count.h"
#include "openzl/codecs/common/window.h"
#include "openzl/codecs/rolz/encode_match_finder.h"
#include "openzl/codecs/rolz/encode_rolz_sequences.h"
#include "openzl/shared/hash.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/utils.h"

#define kShrink true
#define kMinOffset 16
#define kTokenML 31
#define kTokenLL 15

#define kMaxOffset ((1u << 21) - 1)

#define kSearchStrength 6
#define kLzBackSearch 1

#define kPrefetchAdv 8
#define kPrefetchMask 7

typedef struct {
    ZS_matchFinderCtx base;
    ZS_MatchFinderParameters params;
} ZS_fastLzCtx;

static void ZS_fastLzMatchFinderCtx_releaseInternal(ZS_fastLzCtx* ctx)
{
    if (!ctx)
        return;
    free(ctx);
}

static void ZS_fastLzMatchFinderCtx_release(ZS_matchFinderCtx* ctx)
{
    ZS_fastLzMatchFinderCtx_releaseInternal(
            ZL_CONTAINER_OF(ctx, ZS_fastLzCtx, base));
}

static ZS_matchFinderCtx* ZS_fastLzMatchFinderCtx_create(
        ZS_window const* window,
        ZS_MatchFinderParameters const* params)
{
    ZS_fastLzCtx* const ctx = (ZS_fastLzCtx*)calloc(1, sizeof(ZS_fastLzCtx));
    if (!ctx)
        goto err;
    ctx->params = *params;

    ctx->base.window = window;

    return &ctx->base;
err:
    ZS_fastLzMatchFinderCtx_releaseInternal(ctx);
    return NULL;
}

static void ZS_fastLzMatchFinderCtx_reset(ZS_matchFinderCtx* baseCtx)
{
    (void)baseCtx;
}

static char const* ZS_MatchType_name(ZS_matchType mt)
{
    switch (mt) {
        case ZS_mt_rep0:
            return "REP0";
        case ZS_mt_lz:
            return "LZ";
        case ZS_mt_rep:
        case ZS_mt_rolz:
        case ZS_mt_lits:
        case ZS_mt_lzn:
        default:
            ZL_ASSERT_FAIL("Illegal match type!");
            return "";
    }
}

static void ZS_fastLzMatchFinder_parse(
        ZS_matchFinderCtx* baseCtx,
        ZS_RolzSeqStore* seqs,
        uint8_t const* src,
        size_t size)
{
    uint8_t const* ip           = src;
    uint8_t const* anchor       = ip;
    uint8_t const* const iend   = ip + size;
    uint8_t const* const ilimit = iend - 16;

    ZS_window const* window     = baseCtx->window;
    uint8_t const* const base   = window->base;
    uint32_t const lzMinLength  = 5;
    uint32_t const windowLog    = 21;
    uint32_t const smallHashLog = 16;
    uint32_t const largeHashLog = 17;
    uint32_t const windowSize   = 1u << windowLog;

    uint32_t* const smallHashTable =
            (uint32_t*)malloc(sizeof(uint32_t) << smallHashLog);
    uint32_t* const largeHashTable =
            (uint32_t*)malloc(sizeof(uint32_t) << largeHashLog);
    memset(smallHashTable, 0, sizeof(uint32_t) << smallHashLog);
    memset(largeHashTable, 0, sizeof(uint32_t) << largeHashLog);

    // Skip the first contextDepth positions
    uint32_t rep = kMinOffset;
    ip += rep;

    ZL_ASSERT_LT(lzMinLength, 8);
    // TODO: Repcode search
    while (ip < ilimit) {
        ZS_sequence seq;
        uint32_t const curr     = (uint32_t)(ip - base);
        uint32_t matchLength    = 0;
        uint32_t offset         = 0;
        uint32_t const minIndex = curr > windowSize ? curr - windowSize : 0;
        uint32_t const maxIndex = curr - (kMinOffset - 1);
        ZL_ASSERT_GE(ip, anchor);

        size_t const smallHash = ZL_hashPtr(ip, smallHashLog, lzMinLength);
        size_t const largeHash = ZL_hashPtr(ip, largeHashLog, 8);

        uint32_t smallIndex = smallHashTable[smallHash];
        uint32_t largeIndex = largeHashTable[largeHash];

        uint8_t const* const smallMatch = base + smallIndex;
        uint8_t const* const largeMatch = base + largeIndex;

        largeHashTable[largeHash] = smallHashTable[smallHash] = curr;

        if (ZL_read32(ip + 1) == (ZL_read32(ip + 1 - rep))) {
            offset          = rep;
            matchLength     = (uint32_t)ZS_count(ip + 1, ip + 1 - rep, iend);
            seq.matchType   = ZS_mt_rep0;
            seq.matchCode   = 0;
            seq.matchLength = matchLength;
            ++ip;
            goto _store_sequence;
        }

        if (largeIndex > minIndex && largeIndex < maxIndex
            && ZL_read64(ip) == ZL_read64(largeMatch)) {
            offset          = curr - largeIndex;
            matchLength     = (uint32_t)ZS_count(ip, largeMatch, iend);
            seq.matchType   = ZS_mt_lz;
            seq.matchCode   = offset;
            seq.matchLength = matchLength;
            goto _store_sequence;
        }

        if (smallIndex <= minIndex || smallIndex >= maxIndex
            || ZL_read32(ip) != ZL_read32(smallMatch)) {
            ip += ((ip - anchor) >> kSearchStrength) + 1;
            continue;
        }

        size_t const largeHash1          = ZL_hashPtr(ip + 1, largeHashLog, 8);
        uint32_t const largeIndex1       = largeHashTable[largeHash1];
        uint8_t const* const largeMatch1 = base + largeIndex1;
        largeHashTable[largeHash1]       = curr + 1;

        if (largeIndex1 > minIndex + 1 && largeIndex1 < maxIndex + 1
            && ZL_read64(ip + 1) == ZL_read64(largeMatch1)) {
            offset          = curr + 1 - largeIndex1;
            matchLength     = (uint32_t)ZS_count(ip + 1, largeMatch1, iend);
            seq.matchType   = ZS_mt_lz;
            seq.matchCode   = offset;
            seq.matchLength = matchLength;
            ++ip;
        } else {
            offset          = curr - smallIndex;
            matchLength     = (uint32_t)ZS_count(ip, smallMatch, iend);
            seq.matchType   = ZS_mt_lz;
            seq.matchCode   = offset;
            seq.matchLength = matchLength;
        }

    _store_sequence:
        rep = offset;
        {
            uint8_t const* match               = ip - offset;
            uint8_t const* const lowMatchLimit = base + window->dictLimit;
            while (ip > anchor && match > lowMatchLimit
                   && ip[-1] == match[-1]) {
                --ip;
                --match;
                ++matchLength;
            }
        }
        seq.matchLength   = matchLength;
        seq.literalLength = (uint32_t)(ip - anchor);

        if (kShrink && seq.literalLength > kTokenLL
            && seq.literalLength <= 3 * kTokenLL) {
            while (seq.literalLength > kTokenLL) {
                ZS_sequence litSeq = {
                    .literalLength = kTokenLL,
                    .matchLength   = 0,
                    .matchType     = ZS_mt_rep0,
                    .matchCode     = 0,
                };
                ZS_RolzSeqStore_store(seqs, 0, anchor, iend, &litSeq);
                anchor += kTokenLL;
                seq.literalLength -= kTokenLL;
            }
        }

        bool matchShrunk = false;
        if (kShrink && seq.matchLength > kTokenML
            && seq.matchLength <= 3 * kTokenML) {
            ZL_DLOG(SEQ, "Shrunk from %u to %d", seq.matchLength, kTokenML);
            matchLength     = kTokenML;
            seq.matchLength = kTokenML;
            matchShrunk     = true;
        }

        ZL_DLOG(SEQ,
                "%s mpos=%u code=%u mlen=%u",
                ZS_MatchType_name(seq.matchType),
                (uint32_t)(ip - src),
                seq.matchCode,
                matchLength);
        ZS_RolzSeqStore_store(seqs, 0, anchor, iend, &seq);
        ip += matchLength;
        anchor = ip;

        if (ip <= ilimit) {
            uint32_t const insertIdx                              = curr + 2;
            smallHashTable[ZL_hashPtr(
                    base + insertIdx, smallHashLog, lzMinLength)] = insertIdx;
            largeHashTable[ZL_hashPtr(base + insertIdx, largeHashLog, 8)] =
                    insertIdx;
            smallHashTable[ZL_hashPtr(ip - 1, smallHashLog, lzMinLength)] =
                    (uint32_t)(ip - 1 - base);
            largeHashTable[ZL_hashPtr(ip - 2, largeHashLog, 8)] =
                    (uint32_t)(ip - 2 - base);

            if (matchShrunk && ZL_read32(ip) == ZL_read32(ip - rep)) {
                offset          = rep;
                matchLength     = (uint32_t)ZS_count(ip, ip - rep, iend);
                seq.matchType   = ZS_mt_rep0;
                seq.matchCode   = 0;
                seq.matchLength = matchLength;
                goto _store_sequence;
            }
        }
    }
    ZL_ASSERT_LE(anchor, iend);
    ZS_RolzSeqStore_storeLastLiterals(seqs, anchor, (size_t)(iend - anchor));
    free(smallHashTable);
    free(largeHashTable);
}

const ZS_matchFinder ZS_doubleFastLzMatchFinder = {
    .name        = "doubleFastLz",
    .ctx_create  = ZS_fastLzMatchFinderCtx_create,
    .ctx_reset   = ZS_fastLzMatchFinderCtx_reset,
    .ctx_release = ZS_fastLzMatchFinderCtx_release,
    .parse       = ZS_fastLzMatchFinder_parse,
};

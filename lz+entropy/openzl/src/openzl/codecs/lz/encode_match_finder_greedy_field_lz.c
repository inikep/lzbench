// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "openzl/codecs/common/row_table.h"
#include "openzl/codecs/common/window.h"
#include "openzl/codecs/lz/encode_field_lz_sequences.h"
#include "openzl/codecs/lz/encode_match_finder.h"
#include "openzl/shared/bits.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"
#include "openzl/shared/utils.h"

#define kTableLog 14
#define kRepMinMatch 3
#define kLzMinMatch 7

#define kMaxOffset ((1u << 24) - 1)

#define kSearchStrength 6

#define kPrefetch true

typedef struct {
    ZS_matchFinderCtx base;
    ZS_RowTable table;
    ZS_RowTable table2;
    ZS_MatchFinderParameters params;
} ZS_tokenLzCtx;

static uint32_t minMatchLength(uint32_t fieldSize)
{
    return fieldSize < 4 ? 4 : fieldSize;
}

static ZS_matchFinderCtx* ZS_greedyTokenLzMatchFinderCtx_create(
        ZS_window const* window,
        ZS_MatchFinderParameters const* params)
{
    uint32_t const kMinMatch = minMatchLength(params->fieldSize);
    ZS_tokenLzCtx* const ctx =
            params->alloc.alloc(params->alloc.opaque, sizeof(ZS_tokenLzCtx));
    if (!ctx) {
        return NULL;
    }
    memset(ctx, 0, sizeof(*ctx));
    ctx->params = *params;

    void* tableMem = params->alloc.alloc(
            params->alloc.opaque, ZS_RowTable_tableSize(params->lzTableLog));
    void* table2Mem = params->alloc.alloc(
            params->alloc.opaque, ZS_RowTable_tableSize(params->lzTableLog));
    if (!tableMem || !table2Mem) {
        return NULL;
    }

    ZS_RowTable_init(
            &ctx->table,
            tableMem,
            params->lzTableLog,
            params->fieldSize,
            kMinMatch);
    ZS_RowTable_init(
            &ctx->table2,
            table2Mem,
            params->lzTableLog,
            params->fieldSize,
            kMinMatch * 2);

    ctx->base.window = window;

    return &ctx->base;
}

ZL_FORCE_INLINE bool
ZS_checkMatch(uint8_t const* value, uint8_t const* match, uint32_t kMinMatch)
{
    switch (kMinMatch) {
        case 2:
            return ZL_read16(value) == ZL_read16(match);
        case 3:
            return ZL_read24(value) == ZL_read24(match);
        case 4:
            return ZL_read32(value) == ZL_read32(match);
        default:
        case 8:
            return ZL_read64(value) == ZL_read64(match);
    }
}

ZL_FORCE_INLINE size_t ZS_countFields(
        uint8_t const* ip,
        uint8_t const* match,
        uint8_t const* iend,
        uint32_t const kFieldSize)
{
    uint8_t const* const start  = ip;
    uint8_t const* const ilimit = iend - kFieldSize + 1;
    while (ip < ilimit && ZS_checkMatch(ip, match, kFieldSize)) {
        ip += kFieldSize;
        match += kFieldSize;
    }
    return (size_t)(ip - start);
}

ZL_FORCE_INLINE
bool findMatchGreedy(
        ZS_sequence* seq,
        uint8_t const** matchPtr,
        ZS_RowTable* table,
        ZS_RowTable* table2,
        uint8_t const* base,
        uint8_t const* anchor,
        uint8_t const* ip,
        uint8_t const* iend,
        uint32_t lowLimit,
        uint32_t nbSearches,
        size_t minLength,
        uint32_t kFieldSize)
{
    int const kFieldBits     = ZL_highbit32(kFieldSize);
    uint32_t const kMinMatch = minMatchLength(kFieldSize);

    uint32_t const pos = (uint32_t)(ip - base);

    ZL_ASSERT_LE(ip + 2 * kMinMatch, iend);
    {
        ZS_RowTable_Match const match = ZS_RowTable_getBestMatchAndUpdateT(
                table2,
                base,
                anchor,
                lowLimit,
                pos,
                iend,
                nbSearches,
                minLength,
                kFieldSize,
                2 * kMinMatch);
        if (match.totalLength >= minLength) {
            uint32_t const offset = pos - match.matchIdx;
            ZL_ASSERT_LT(match.matchIdx, pos);
            ZL_ASSERT_EQ(match.totalLength % kFieldSize, 0);
            ZL_ASSERT_EQ(match.backwardLength % kFieldSize, 0);
            ZL_ASSERT_EQ(match.forwardLength % kFieldSize, 0);
            ZL_ASSERT_LE(match.backwardLength, (size_t)(ip - anchor));
            ZL_ASSERT_EQ(offset % kFieldSize, 0);
            *matchPtr        = base + match.matchIdx - match.backwardLength;
            seq->matchType   = ZS_mt_lz;
            seq->matchCode   = offset >> kFieldBits;
            seq->matchLength = (uint32_t)match.totalLength;
            seq->literalLength =
                    (uint32_t)((ip - match.backwardLength) - anchor);
            return true;
        }
    }
    {
        ZS_RowTable_Match const match = ZS_RowTable_getBestMatchAndUpdateT(
                table,
                base,
                anchor,
                lowLimit,
                pos,
                iend,
                nbSearches,
                minLength,
                kFieldSize,
                kMinMatch);
        if (match.totalLength >= minLength) {
            uint32_t const offset = pos - match.matchIdx;
            ZL_ASSERT_LT(match.matchIdx, pos);
            ZL_ASSERT_EQ(match.totalLength % kFieldSize, 0);
            ZL_ASSERT_EQ(match.backwardLength % kFieldSize, 0);
            ZL_ASSERT_EQ(match.forwardLength % kFieldSize, 0);
            ZL_ASSERT_LE(match.backwardLength, (size_t)(ip - anchor));
            ZL_ASSERT_EQ(offset % kFieldSize, 0);
            *matchPtr        = base + match.matchIdx - match.backwardLength;
            seq->matchType   = ZS_mt_lz;
            seq->matchCode   = offset >> kFieldBits;
            seq->matchLength = (uint32_t)match.totalLength;
            seq->literalLength =
                    (uint32_t)((ip - match.backwardLength) - anchor);
            return true;
        }
    }
    return false;
}

ZL_FORCE_INLINE
bool findMatchLazy(
        ZS_sequence* seq,
        uint8_t const** matchPtr,
        ZS_RowTable* table,
        ZS_RowTable* table2,
        uint8_t const* base,
        uint8_t const* anchor,
        uint8_t const* ip,
        uint8_t const* iend,
        uint32_t lowLimit,
        uint32_t nbSearches,
        size_t minLength,
        uint32_t kFieldSize)
{
    (void)table;
    int const kFieldBits     = ZL_highbit32(kFieldSize);
    uint32_t const kMinMatch = minMatchLength(kFieldSize);

    uint32_t const pos = (uint32_t)(ip - base);

    ZL_ASSERT_LE(ip + 2 * kMinMatch, iend);
    ZS_RowTable_fillT(
            table2, base, (uint32_t)(ip - base), kFieldSize, 2 * kMinMatch);
    {
        ZS_RowTable_Match const match = ZS_RowTable_getBestMatchAndUpdateT(
                table2,
                base,
                anchor,
                lowLimit,
                pos,
                iend,
                nbSearches,
                minLength,
                kFieldSize,
                2 * kMinMatch);
        if (match.totalLength >= minLength) {
            uint32_t const offset = pos - match.matchIdx;
            ZL_ASSERT_LT(match.matchIdx, pos);
            ZL_ASSERT_EQ(match.totalLength % kFieldSize, 0);
            ZL_ASSERT_EQ(match.backwardLength % kFieldSize, 0);
            ZL_ASSERT_EQ(match.forwardLength % kFieldSize, 0);
            ZL_ASSERT_LE(match.backwardLength, (size_t)(ip - anchor));
            ZL_ASSERT_EQ(offset % kFieldSize, 0);
            *matchPtr        = base + match.matchIdx - match.backwardLength;
            seq->matchType   = ZS_mt_lz;
            seq->matchCode   = offset >> kFieldBits;
            seq->matchLength = (uint32_t)match.totalLength;
            seq->literalLength =
                    (uint32_t)((ip - match.backwardLength) - anchor);
            return true;
        }
    }
    return false;
}

ZL_FORCE_INLINE void ZS_greedyTokenLzMatchFinder_parseT(
        ZS_matchFinderCtx* baseCtx,
        ZS_seqStore* seqs,
        uint8_t const* src,
        size_t size,
        uint32_t const kFieldSize)
{
    ZS_tokenLzCtx* const ctx = ZL_CONTAINER_OF(baseCtx, ZS_tokenLzCtx, base);

    uint8_t const* ip         = src;
    uint8_t const* anchor     = ip;
    uint8_t const* const iend = ip + size;
    uint8_t const* const ilimit =
            iend - ZL_MAX(2 * kFieldSize, 16); // TODO Hash read size

    uint8_t const* const base      = baseCtx->window->base;
    uint32_t const lowLimit        = baseCtx->window->lowLimit;
    uint32_t const nbSearches      = 8;
    uint8_t const* const windowLow = base + lowLimit;
    ZS_RowTable table              = ctx->table;
    ZS_RowTable table2             = ctx->table2;

    uint32_t const kFieldMask = kFieldSize - 1;
    uint32_t const kMinMatch  = minMatchLength(kFieldSize);

    ZL_ASSERT(ZL_isPow2(kFieldSize));
    int const kFieldBits = ZL_highbit32(kFieldSize);

#define kNumRep 2
    uint32_t rep[kNumRep] = { kFieldSize, 2 * kFieldSize };

    ip += rep[kNumRep - 1];

    table.nextToFill  = (uint32_t)(ip - base);
    table2.nextToFill = (uint32_t)(ip - base);

    // Main search loop
    while (ip < ilimit) {
        ZS_sequence seq;
        ZS_sequence seq2      = { 0 };
        uint8_t const* match  = NULL;
        uint8_t const* match2 = NULL;

        do {
            // first search for repcodes
            uint8_t const* const ip1 = ip + 0;
            for (int r = 0; r < kNumRep; ++r) {
                bool isRepMatch = ZS_checkMatch(ip1, ip1 - rep[r], kMinMatch);
                if (isRepMatch) {
                    match             = ip1 - rep[r];
                    seq.matchType     = ZS_mt_rep;
                    seq.matchCode     = (uint32_t)r;
                    seq.literalLength = (uint32_t)(ip1 - anchor);
                    seq.matchLength   = (uint32_t)ZS_countFields(
                            ip1, ip1 - rep[r], iend, kFieldSize);
                    goto match_found;
                }
            }

            // Search for a `2 * kMinMatch` match - take it if we find it
            // Otherwise, seaarch for a `kMinMatch` match
            bool const matchFound = findMatchGreedy(
                    &seq,
                    &match,
                    &table,
                    &table2,
                    base,
                    anchor,
                    ip,
                    iend,
                    lowLimit,
                    nbSearches,
                    kMinMatch,
                    kFieldSize);
            if (matchFound) {
                // If we find a match, and it is at least `2 * kMinMatch` bytes
                // forward, do another `2 * kMinMatch` search kMinMatch bytes
                // from the end of the match, to try to find a longer match.
                uint8_t const* const matchEnd =
                        anchor + seq.literalLength + seq.matchLength;
                if (matchEnd > ip + kMinMatch && matchEnd < ilimit) {
                    bool const matchFound2 = findMatchLazy(
                            &seq2,
                            &match2,
                            &table,
                            &table2,
                            base,
                            anchor,
                            matchEnd - kMinMatch,
                            iend,
                            lowLimit,
                            nbSearches,
                            2 * kMinMatch,
                            kFieldSize);
                    if (matchFound2) {
                        goto match_found2;
                    }
                }
                ZL_ASSERT_NULL(match2);
                goto match_found;
            }

            // No matches found, countine at the next field
            ip += kFieldSize;
        } while (ip < ilimit);

        continue;
    match_found2: {
        ZL_ASSERT_NN(match2);
        uint8_t const* ip2               = anchor + seq2.literalLength;
        uint8_t const* const windowLimit = windowLow + kFieldSize - 1;
        if (ip2 > anchor && match2 > windowLimit)
            ZL_ASSERT(!(ZS_checkMatch(
                    ip2 - kFieldSize, match2 - kFieldSize, kMinMatch)));
    }
    match_found: {
        ip = anchor + seq.literalLength;

        ZL_ASSERT_GE(seq.matchLength, kFieldSize);
        ZL_ASSERT_GE(seq.matchLength, 4);
        ZL_ASSERT_EQ(seq.literalLength & kFieldMask, 0);
        ZL_ASSERT_EQ(seq.matchLength & kFieldMask, 0);
        ZL_ASSERT_EQ((size_t)(ip - match) & kFieldMask, 0);
        ZL_ASSERT_LT(match, ip);

        ZL_ASSERT(ZS_checkMatch(ip, match, kMinMatch));
        ZL_ASSERT_EQ((size_t)(ip - anchor) & kFieldMask, 0);
        if (ip + seq.matchLength < ilimit) {
            ZL_ASSERT(!ZS_checkMatch(
                    ip + seq.matchLength, match + seq.matchLength, kMinMatch));
        }
        {
            uint8_t const* const windowLimit = windowLow + kFieldSize - 1;
            if (seq.matchType == ZS_mt_lz && ip > anchor && match > windowLimit)
                ZL_ASSERT(!(ZS_checkMatch(
                        ip - kFieldSize, match - kFieldSize, kMinMatch)));

            // Roll-back the match. Only applies to repcodes, lz matches are
            // fully rolled back.
            while (ip > anchor && match > windowLimit
                   && ZS_checkMatch(
                           ip - kFieldSize, match - kFieldSize, kFieldSize)) {
                ip -= kFieldSize;
                match -= kFieldSize;
                seq.matchLength += kFieldSize;
                seq.literalLength -= kFieldSize;
            }
        }

        if (match2 != NULL) {
            // match2 is guaranteed to end after match, so if
            // match2 doesn't have any extra literals, it is
            // also longer, so select it.
            if (seq2.literalLength <= seq.literalLength) {
                ZL_ASSERT_GT(seq2.matchLength, seq.matchLength);
                seq    = seq2;
                match  = match2;
                match2 = NULL;
                ip     = anchor + seq.literalLength;
            } else {
                // No use in using match2 in this case.
                // We'll likely find the same match, or something better,
                // when we search in the next loop after the first match.
                // Using match2 hurts compression ratio, because it might
                // force us into an unprofitible match.
                match2 = NULL;
            }
        }

        ZL_ASSERT_EQ(seq.literalLength & kFieldMask, 0);
        ZL_ASSERT_EQ(seq.matchLength & kFieldMask, 0);
        ZL_ASSERT_EQ((size_t)(ip - match) & kFieldMask, 0);

        // Update the repcodes
        if (seq.matchType == ZS_mt_lz) {
            memmove(&rep[1], &rep[0], (kNumRep - 1) * 4);
            rep[0] = seq.matchCode << kFieldBits;
        } else if (seq.matchType == ZS_mt_rep && seq.matchCode != 0) {
            uint32_t const offset = rep[seq.matchCode];
            memmove(&rep[1], &rep[0], seq.matchCode * 4);
            rep[0] = offset;

            ZL_ASSERT_LE(seq.matchCode, 1);
        }
        // Store the sequence
        ZS_seqStore_store(seqs, anchor, iend, &seq);

        // Fill the tables for every matched position.
        if (ip + seq.matchLength <= ilimit) {
            ZS_RowTable_fillT(
                    &table,
                    base,
                    (uint32_t)(ip + seq.matchLength - base),
                    kFieldSize,
                    kMinMatch);
            ZS_RowTable_fillT(
                    &table2,
                    base,
                    (uint32_t)(ip + seq.matchLength - base),
                    kFieldSize,
                    2 * kMinMatch);
        }

        // Continue the search after the match.
        anchor = ip = ip + seq.matchLength;
    }
    }
    ZL_ASSERT_LE(anchor, iend);
    ZS_seqStore_storeLastLiterals(seqs, anchor, (size_t)(iend - anchor));
}

static void ZS_greedyTokenLzMatchFinder_parseAny(
        ZS_matchFinderCtx* baseCtx,
        ZS_seqStore* seqs,
        uint8_t const* src,
        size_t size,
        uint32_t const fieldSize)
{
    ZS_greedyTokenLzMatchFinder_parseT(baseCtx, seqs, src, size, fieldSize);
}

#define ZS_GREEDY_TOKEN_LZ_MATCH_FINDER_PARSE(kFieldSize)       \
    static void ZS_greedyTokenLzMatchFinder_parse_##kFieldSize( \
            ZS_matchFinderCtx* baseCtx,                         \
            ZS_seqStore* seqs,                                  \
            uint8_t const* src,                                 \
            size_t size)                                        \
    {                                                           \
        ZS_greedyTokenLzMatchFinder_parseT(                     \
                baseCtx, seqs, src, size, kFieldSize);          \
    }

ZS_GREEDY_TOKEN_LZ_MATCH_FINDER_PARSE(1)
ZS_GREEDY_TOKEN_LZ_MATCH_FINDER_PARSE(2)
ZS_GREEDY_TOKEN_LZ_MATCH_FINDER_PARSE(4)
ZS_GREEDY_TOKEN_LZ_MATCH_FINDER_PARSE(8)

static void ZS_greedyTokenLzMatchFinder_parse(
        ZS_matchFinderCtx* baseCtx,
        ZS_seqStore* seqs,
        uint8_t const* src,
        size_t size)
{
    ZS_tokenLzCtx* const ctx = ZL_CONTAINER_OF(baseCtx, ZS_tokenLzCtx, base);
    ZL_ASSERT_NN(ctx);
    uint32_t const fieldSize = ctx->params.fieldSize;
    switch (fieldSize) {
        default:
            ZS_greedyTokenLzMatchFinder_parseAny(
                    baseCtx, seqs, src, size, fieldSize);
            break;
        case 1:
            ZS_greedyTokenLzMatchFinder_parse_1(baseCtx, seqs, src, size);
            break;
        case 2:
            ZS_greedyTokenLzMatchFinder_parse_2(baseCtx, seqs, src, size);
            break;
        case 4:
            ZS_greedyTokenLzMatchFinder_parse_4(baseCtx, seqs, src, size);
            break;
        case 8:
            ZS_greedyTokenLzMatchFinder_parse_8(baseCtx, seqs, src, size);
            break;
    }
}

const ZS_matchFinder ZS_greedyTokenLzMatchFinder = {
    .name       = "greedyTokenLz",
    .ctx_create = ZS_greedyTokenLzMatchFinderCtx_create,
    .parse      = ZS_greedyTokenLzMatchFinder_parse,
};

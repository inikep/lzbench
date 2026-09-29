// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZS_DECOMPRESS_ROLZ_H
#define ZS_DECOMPRESS_ROLZ_H

#include <stdlib.h>
#include <string.h>

#include "openzl/codecs/common/window.h"
#include "openzl/codecs/rolz/common_rolz.h"
#include "openzl/common/debug.h"
#include "openzl/shared/portability.h"
#include "openzl/shared/utils.h"
#include "openzl/zl_errors.h"

ZL_BEGIN_C_DECLS

ZL_Report ZL_rolzDecompress(
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize);

ZL_Report ZS_fastLzDecompress(
        void* dst,
        size_t dstCapacity,
        const void* src,
        size_t srcSize);

typedef struct {
    uint32_t index;
    uint16_t head; /* Only first entry matters... TODO: FIXME */
    uint8_t predictedMatchLength;
} ZS_RolzDEntry2;

typedef struct {
    uint32_t contextDepth;
    uint32_t contextLog;
    uint32_t rowLog;
    uint32_t rowMask;
    uint32_t minLength;
    bool predictMatchLength;
    ZS_RolzDEntry2* table;
} ZS_RolzDTable2;

typedef struct {
    uint32_t index;
    uint32_t length;
} ZS_RolzMatch2;

ZL_Report ZS_RolzDTable2_init(
        ZS_RolzDTable2* table,
        uint32_t contextDepth,
        uint32_t contextLog,
        uint32_t rowLog,
        uint32_t minLength,
        bool predictMatchLength);
void ZS_RolzDTable2_destroy(ZS_RolzDTable2* table);

ZL_FORCE_INLINE ZS_RolzMatch2 ZS_RolzDTable2_get2(
        ZS_RolzDTable2 const* table,
        uint32_t context,
        uint32_t index,
        uint32_t encodedMatchLength,
        uint32_t kRowLog,
        uint32_t kMinLength,
        uint32_t kPredictMatchLength)
{
    ZL_ASSERT_EQ(kRowLog, table->rowLog);
    ZL_ASSERT_EQ(kMinLength, table->minLength);
    ZL_ASSERT_EQ(kPredictMatchLength, (uint32_t)table->predictMatchLength);
    ZL_ASSERT_LT(context, 1u << table->contextLog);
    ZL_ASSERT_LT(index, 1u << table->rowLog);
    uint32_t const kRowMask        = (1u << kRowLog) - 1;
    ZS_RolzDEntry2* const rowStart = table->table + (context << kRowLog);
    uint32_t const pos             = (rowStart->head + index) & kRowMask;
    ZS_RolzMatch2 match;
    match.index = rowStart[pos].index;
    if (kPredictMatchLength)
        match.length = ZS_rolz_decodeMatchLength(
                kMinLength,
                rowStart[pos].predictedMatchLength,
                encodedMatchLength);
    else
        match.length = kMinLength + encodedMatchLength;
    // TODO: Could update the min/predicted match length here
    return match;
}

ZL_INLINE ZS_RolzMatch2 ZS_RolzDTable2_get(
        ZS_RolzDTable2 const* table,
        uint32_t context,
        uint32_t index,
        uint32_t encodedMatchLength)
{
    return ZS_RolzDTable2_get2(
            table,
            context,
            index,
            encodedMatchLength,
            table->rowLog,
            table->minLength,
            table->predictMatchLength);
}

ZL_FORCE_INLINE void ZS_RolzDTable2_put2(
        ZS_RolzDTable2* table,
        uint32_t context,
        uint32_t index,
        uint32_t matchLength,
        uint32_t kRowLog)
{
    ZL_ASSERT_EQ(kRowLog, table->rowLog);
    ZL_DLOG(SEQ,
            "ROLZ PUT %u | ctx=%u | pml=%u",
            index - 1,
            context,
            ZL_MIN(255, matchLength));
    ZL_ASSERT_LT(context, 1u << table->contextLog);
    uint32_t const kRowMask        = (1u << kRowLog) - 1;
    ZS_RolzDEntry2* const rowStart = table->table + (context << kRowLog);
    uint32_t const pos             = (uint32_t)(rowStart->head - 1) & kRowMask;
    rowStart->head                 = (uint16_t)pos;
    rowStart[pos].index            = index;
    rowStart[pos].predictedMatchLength = (uint8_t)ZL_MIN(255, matchLength);
}

ZL_INLINE void ZS_RolzDTable2_put(
        ZS_RolzDTable2* table,
        uint32_t context,
        uint32_t index,
        uint32_t matchLength)
{
    ZS_RolzDTable2_put2(table, context, index, matchLength, table->rowLog);
}

typedef struct {
    uint32_t index;
    uint8_t mlMin;
    uint8_t mlExpect;
} ZS_rolzDEntry;

typedef struct {
    uint32_t ctxBits;
    uint32_t entryMask; //< Each bucket has entryMask+1 entries (power of 2)
    uint32_t entryLog;
    uint32_t anchor;
    uint32_t* head;       //< The current position in the bucket
    ZS_rolzDEntry* table; //< The table of buckets
} ZS_rolzDTable;

ZL_INLINE uint32_t ZS_rolzDEntry_matchLength(ZS_rolzDEntry entry, uint32_t ml)
{
    return ZS_rolz_decodeMatchLength(entry.mlMin, entry.mlExpect, ml);
}

int ZS_rolzDTable_init(
        ZS_rolzDTable* dtable,
        ZS_window const* window,
        uint32_t numEntries,
        uint32_t ctxBits);
void ZS_rolzDTable_destroy(ZS_rolzDTable* dtable);
void ZS_rolzDTable_reset(ZS_rolzDTable* dtable, ZS_window const* window);

ZL_INLINE ZS_rolzDEntry
ZS_rolzDTable_get2(ZS_rolzDTable const* dtable, uint32_t ctx, uint32_t entry)
{
    uint32_t const bucket    = ctx << dtable->entryLog;
    uint32_t const bucketIdx = (dtable->head[ctx] + entry) & dtable->entryMask;
    return dtable->table[bucket + bucketIdx];
}

ZL_INLINE ZS_rolzDEntry ZS_rolzDTable_get(
        ZS_rolzDTable const* dtable,
        uint8_t const* ip,
        uint32_t entry)
{
    uint32_t const ctx = getCtx(ip, dtable->ctxBits);
    return ZS_rolzDTable_get2(dtable, ctx, entry);
}

ZL_INLINE void ZS_rolzDTable_put2(
        ZS_rolzDTable* dtable,
        uint32_t const ctx,
        uint32_t index,
        uint32_t ml)
{
    uint32_t const bucket    = ctx << dtable->entryLog;
    uint32_t const bucketIdx = --dtable->head[ctx] & dtable->entryMask;
    ZL_LOG(POS,
           "ROLZ-DTable[%u,%u] = %u",
           ctx,
           (uint32_t)0 - dtable->head[ctx] - 1,
           index);
    ZL_ASSERT_NE(index, 0);
    dtable->table[bucket + bucketIdx].index = index;
    dtable->table[bucket + bucketIdx].mlMin = MINMATCH;
    dtable->table[bucket + bucketIdx].mlExpect =
            (uint8_t)ZL_MAX(ZL_MIN(ml, 255), MINMATCH);
}

ZL_INLINE void ZS_rolzDTable_putLzMatch2(
        ZS_rolzDTable* dtable,
        uint8_t const* ip,
        uint32_t index,
        uint32_t ml,
        uint32_t const kCtxBits,
        bool putOthers)
{
    if (kRolzUpdate)
        return;
    ZL_ASSERT(kCtxBits == dtable->ctxBits);
    ZL_ASSERT(!N2);
    ZL_ASSERT(!N1);
    if (kRolzInsertEveryPosition) {
        ZL_ASSERT_EQ(dtable->anchor, index);
        do {
            ZS_rolzDTable_put2(dtable, getCtx(ip++, kCtxBits), index++, ml--);
        } while (ml >= kRolzMinMatchLengthInsert);
        dtable->anchor = index;
    } else {
        if (N2 && putOthers)
            ZS_rolzDTable_put2(dtable, getCtx(ip - 2, kCtxBits), index - 2, 0);
        if (N1 && putOthers)
            ZS_rolzDTable_put2(dtable, getCtx(ip - 1, kCtxBits), index - 1, 0);
        if (P0)
            ZS_rolzDTable_put2(dtable, getCtx(ip, kCtxBits), index, ml);
        if (P1 && putOthers && ml > 1)
            ZS_rolzDTable_put2(
                    dtable, getCtx(ip + 1, kCtxBits), index + 1, ml - 1);
    }
}

ZL_INLINE void ZS_rolzDTable_putLzMatch(
        ZS_rolzDTable* dtable,
        uint8_t const* ip,
        uint32_t index,
        uint32_t ml,
        bool putOthers)
{
    ZS_rolzDTable_putLzMatch2(
            dtable, ip, index, ml, dtable->ctxBits, putOthers);
}

ZL_INLINE void ZS_rolzDTable_putRolzMatch(
        ZS_rolzDTable* dtable,
        uint8_t const* ip,
        uint32_t entry,
        uint32_t index,
        uint32_t ml)
{
    if (kRolzUpdate)
        return;
    uint32_t const ctx       = getCtx(ip, dtable->ctxBits);
    uint32_t const bucket    = ctx << dtable->entryLog;
    uint32_t const bucketIdx = (dtable->head[ctx] + entry) & dtable->entryMask;
    ZS_rolzDEntry* dentry    = &dtable->table[bucket + bucketIdx];
    dentry->mlMin    = (uint8_t)ZL_MAX(dentry->mlMin, ZL_MIN(255, ml + 1));
    dentry->mlExpect = (uint8_t)ZL_MAX(dentry->mlExpect, ZL_MIN(255, ml + 1));
    if (kRolzMTF) {
        uint32_t const swpEntry = entry == 0 ? 0 : entry >> 1;
        uint32_t const swpIdx =
                (dtable->head[ctx] + swpEntry) & dtable->entryMask;
        ZS_rolzDEntry* swp = &dtable->table[bucket + swpIdx];
        ZS_rolzDEntry tmp  = *swp;
        *swp               = *dentry;
        *dentry            = tmp;
    }
    if (kRolzInsertEveryPosition) {
        ZL_ASSERT_EQ(dtable->anchor, index);
        do {
            ZS_rolzDTable_put2(
                    dtable, getCtx(ip++, dtable->ctxBits), index++, ml--);
        } while (ml >= kRolzMinMatchLengthInsert);
        dtable->anchor = index;
    } else {
        ZS_rolzDTable_put2(dtable, ctx, index, ml);
        if (R1)
            ZS_rolzDTable_put2(
                    dtable, getCtx(ip + 1, dtable->ctxBits), index + 1, ml - 1);
    }
}

ZL_INLINE void ZS_rolzDTable_put(
        ZS_rolzDTable* dtable,
        uint8_t const* ip,
        uint32_t index,
        uint32_t ml)
{
    uint32_t const ctx = getCtx(ip, dtable->ctxBits);
    ZS_rolzDTable_put2(dtable, ctx, index, ml);
}

ZL_INLINE void ZS_rolzDTable_update2(
        ZS_rolzDTable* dtable,
        ZS_window const* window,
        uint32_t end,
        uint32_t const kCtxBits)
{
    if (!kRolzUpdate && !kRolzInsertEveryPosition)
        return;
    // if (end < (ZL_HASH_READ_SIZE - 1))
    //   return;
    // uint32_t const limit = end - (ZL_HASH_READ_SIZE - 1);
    uint32_t const limit = end;
    if (limit <= dtable->anchor)
        return;
    ZL_LOG(SEQ, "ROLZ-DTable update [%u, %u)", dtable->anchor, limit);
    uint8_t const* ptr = window->base + dtable->anchor;
    uint32_t idx       = dtable->anchor;
    ZL_ASSERT(limit >= window->dictLimit);
    ZL_ASSERT(dtable->anchor - ((kCtxBits + 7) >> 3) >= window->dictLimit);
    for (; idx < limit;) {
        ZS_rolzDTable_put2(
                dtable,
                getCtx(ptr++, kCtxBits),
                idx++,
                kRolzMinMatchLengthInsert);
    }
    dtable->anchor = limit;
}

ZL_INLINE void ZS_rolzDTable_update(
        ZS_rolzDTable* dtable,
        ZS_window const* window,
        uint32_t end)
{
    ZS_rolzDTable_update2(dtable, window, end, dtable->ctxBits);
}

#define HIST_MAX 256
extern uint32_t usesHist[HIST_MAX + 1];
extern uint32_t ageHist[HIST_MAX + 1];
extern uint32_t indexHist[HIST_MAX + 1];
void printUsesHist(void);
void printAgeHist(void);
void printIndexHist(void);

ZL_END_C_DECLS

#endif // ZS_DECOMPRESS_ROLZ_H

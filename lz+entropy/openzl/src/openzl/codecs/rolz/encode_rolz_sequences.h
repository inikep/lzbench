// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZS_COMMON_SEQUENCES_H
#define ZS_COMMON_SEQUENCES_H

#include <string.h>

#include "openzl/codecs/common/copy.h"
#include "openzl/codecs/rolz/common_rolz_sequences.h"
#include "openzl/common/debug.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

// typedef enum {
//   ZS_mt_lz   = 0, //< LZ match
//   ZS_mt_rolz = 1, //< ROLZ match
//   ZS_mt_rep  = 2, //< Repcode
//   ZS_mt_lzn  = 3, //< LZ match w/o insert. Haven't tried this yet.
// } ZS_matchType;

typedef struct {
    uint8_t* start;
    uint8_t* ptr;
    uint8_t* end;
} ZS_byteRange;

typedef struct {
    ZS_sequence* start;
    ZS_sequence* ptr;
    ZS_sequence* end;
} ZS_seqRange;

typedef struct {
    ZS_byteRange litsCtx;
    ZS_byteRange lits;
    ZS_seqRange seqs;
    uint8_t const* fieldBase;
    size_t fieldMask;
} ZS_RolzSeqStore;

int ZS_RolzSeqStore_initExact(
        ZS_RolzSeqStore* seqStore,
        size_t numLiterals,
        size_t numSequences);

int ZS_RolzSeqStore_initBound(
        ZS_RolzSeqStore* seqStore,
        size_t srcSize,
        size_t minMatch);

ZL_INLINE void ZS_RolzSeqStore_setFieldSize(
        ZS_RolzSeqStore* seqStore,
        uint8_t const* fieldBase,
        size_t fieldSize)
{
    seqStore->fieldBase = fieldBase;
    seqStore->fieldMask = fieldSize - 1;
}

void ZS_RolzSeqStore_destroy(ZS_RolzSeqStore* seqStore);

void ZS_RolzSeqStore_reset(ZS_RolzSeqStore* seqStore);

ZL_INLINE void ZS_RolzSeqStore_store(
        ZS_RolzSeqStore* seqStore,
        uint8_t lamCtx,
        uint8_t const* literals,
        uint8_t const* literalsEnd,
        const ZS_sequence* sequence)
{
    (void)lamCtx;
    ZL_LOG(SEQ,
           "Store sequence: mt=%u ll=%u ml=%u mc=%u",
           sequence->matchType,
           sequence->literalLength,
           sequence->matchLength,
           sequence->matchCode);
    ZL_ASSERT_LE(
            seqStore->lits.ptr + sequence->literalLength, seqStore->lits.end);
    ZL_ASSERT_LT(seqStore->seqs.ptr, seqStore->seqs.end);

    // TODO(terrelln): Switch to wildcopy when we have enough space left.
    if (seqStore->fieldMask != 0) {
        size_t const start     = (size_t)(literals - seqStore->fieldBase);
        size_t const fieldMask = seqStore->fieldMask;
        for (size_t i = 0; i < sequence->literalLength; ++i) {
            seqStore->litsCtx.ptr[i] = (uint8_t)((start + i) & fieldMask);
        }
    }
    if (ZL_LIKELY(
                seqStore->litsCtx.ptr > seqStore->litsCtx.start
                && literals + sequence->literalLength
                        < literalsEnd - ZS_WILDCOPY_OVERLENGTH)) {
        if (seqStore->fieldMask == 0) {
            ZS_wildcopy(
                    seqStore->litsCtx.ptr,
                    literals - 1,
                    sequence->literalLength,
                    ZS_wo_no_overlap);
        }
        ZS_wildcopy(
                seqStore->lits.ptr,
                literals,
                sequence->literalLength,
                ZS_wo_no_overlap);
    } else {
        if (seqStore->fieldMask == 0) {
            if (seqStore->litsCtx.ptr == seqStore->litsCtx.start
                && sequence->literalLength > 0) {
                *seqStore->litsCtx.ptr = 0;
                memcpy(seqStore->litsCtx.ptr + 1,
                       literals,
                       sequence->literalLength - 1);
            } else {
                memcpy(seqStore->litsCtx.ptr,
                       literals - 1,
                       sequence->literalLength);
            }
        }
        memcpy(seqStore->lits.ptr, literals, sequence->literalLength);
    }
    seqStore->lits.ptr += sequence->literalLength;
    seqStore->litsCtx.ptr += sequence->literalLength;
    *seqStore->seqs.ptr++ = *sequence;
}

ZL_INLINE void ZS_RolzSeqStore_storeLastLiterals(
        ZS_RolzSeqStore* seqStore,
        uint8_t const* literals,
        size_t size)
{
    ZL_LOG(V9, "Store last literals %u", (uint32_t)size);
    ZL_ASSERT(seqStore->lits.ptr + size <= seqStore->lits.end);
    if (seqStore->fieldMask != 0) {
        size_t const start     = (size_t)(literals - seqStore->fieldBase);
        size_t const fieldMask = seqStore->fieldMask;
        for (size_t i = 0; i < size; ++i) {
            seqStore->litsCtx.ptr[i] = (uint8_t)((start + i) & fieldMask);
        }
    }
    if (seqStore->fieldMask == 0) {
        if (seqStore->litsCtx.ptr == seqStore->litsCtx.start && size > 0) {
            *seqStore->litsCtx.ptr = 0;
            memcpy(seqStore->litsCtx.ptr + 1, literals, size - 1);
        } else {
            memcpy(seqStore->litsCtx.ptr, literals - 1, size);
        }
    }
    memcpy(seqStore->lits.ptr, literals, size);
    seqStore->lits.ptr += size;
    seqStore->litsCtx.ptr += size;
}

ZL_END_C_DECLS

#endif // ZS_COMMON_SEQUENCES_H

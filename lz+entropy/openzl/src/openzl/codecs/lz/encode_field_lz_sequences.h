// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZS_COMMON_SEQUENCES_H
#define ZS_COMMON_SEQUENCES_H

#include <string.h>

#include "openzl/codecs/common/copy.h"
#include "openzl/codecs/lz/common_field_lz.h"
#include "openzl/common/debug.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

typedef enum {
    ZS_mt_lz  = 0, //< LZ match
    ZS_mt_rep = 1, //< rep
} ZS_matchType;

typedef struct {
    uint32_t literalLength;
    uint32_t matchCode;
    uint32_t matchLength;
    uint32_t matchType;
} ZS_sequence;

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
    ZS_byteRange lits;
    ZS_seqRange seqs;
    size_t minMatch;
} ZS_seqStore;

int ZS_seqStore_initExact(
        ZS_seqStore* seqStore,
        size_t numLiterals,
        size_t numSequences,
        size_t minMatch,
        ZL_FieldLz_Allocator alloc);

int ZS_seqStore_initBound(
        ZS_seqStore* seqStore,
        size_t srcSize,
        size_t minMatch,
        ZL_FieldLz_Allocator alloc);

void ZS_seqStore_reset(ZS_seqStore* seqStore);

ZL_INLINE void ZS_seqStore_store(
        ZS_seqStore* seqStore,
        uint8_t const* literals,
        uint8_t const* literalsEnd,
        const ZS_sequence* sequence)
{
    ZL_LOG(SEQ,
           "Store sequence: mt=%u ll=%u ml=%u mc=%u",
           sequence->matchType,
           sequence->literalLength,
           sequence->matchLength,
           sequence->matchCode);
    ZL_ASSERT_GE(sequence->matchLength, seqStore->minMatch);
    ZL_ASSERT_LE(
            seqStore->lits.ptr + sequence->literalLength, seqStore->lits.end);
    ZL_ASSERT_LT(seqStore->seqs.ptr, seqStore->seqs.end);

    // TODO(terrelln): Switch to wildcopy when we have enough space left.
    if (ZL_LIKELY(
                literals + sequence->literalLength
                < literalsEnd - ZS_WILDCOPY_OVERLENGTH)) {
        ZS_wildcopy(
                seqStore->lits.ptr,
                literals,
                sequence->literalLength,
                ZS_wo_no_overlap);
    } else {
        memcpy(seqStore->lits.ptr, literals, sequence->literalLength);
    }
    seqStore->lits.ptr += sequence->literalLength;
    *seqStore->seqs.ptr++ = *sequence;
}

ZL_INLINE void ZS_seqStore_storeLastLiterals(
        ZS_seqStore* seqStore,
        uint8_t const* literals,
        size_t size)
{
    ZL_LOG(V9, "Store last literals %u", (uint32_t)size);
    ZL_ASSERT(seqStore->lits.ptr + size <= seqStore->lits.end);
    memcpy(seqStore->lits.ptr, literals, size);
    seqStore->lits.ptr += size;
}

ZL_END_C_DECLS

#endif // ZS_COMMON_SEQUENCES_H

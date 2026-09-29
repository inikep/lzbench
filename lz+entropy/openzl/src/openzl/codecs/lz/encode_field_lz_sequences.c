// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>

#include "openzl/codecs/lz/encode_field_lz_sequences.h"

int ZS_seqStore_initExact(
        ZS_seqStore* seqStore,
        size_t numLiterals,
        size_t numSequences,
        size_t minMatch,
        ZL_FieldLz_Allocator alloc)
{
    seqStore->minMatch = minMatch;
    seqStore->lits.ptr =
            alloc.alloc(alloc.opaque, numLiterals + ZS_WILDCOPY_OVERLENGTH + 1);
    seqStore->lits.start = seqStore->lits.ptr;
    seqStore->lits.end   = seqStore->lits.start + numLiterals + 1;

    size_t const seqSpace = numSequences * sizeof(ZS_sequence);

    seqStore->seqs.ptr   = (ZS_sequence*)alloc.alloc(alloc.opaque, seqSpace);
    seqStore->seqs.start = seqStore->seqs.ptr;
    seqStore->seqs.end   = seqStore->seqs.start + numSequences;

    if (!seqStore->lits.ptr || !seqStore->seqs.ptr) {
        return 1;
    }

    // Add a 0 byte for the literals context
    *seqStore->lits.start++ = 0;
    seqStore->lits.ptr++;

    return 0;
}

int ZS_seqStore_initBound(
        ZS_seqStore* seqStore,
        size_t srcSize,
        size_t minMatch,
        ZL_FieldLz_Allocator alloc)
{
    size_t const maxNumSeqs = srcSize / minMatch;
    return ZS_seqStore_initExact(
            seqStore, srcSize, maxNumSeqs, minMatch, alloc);
}

void ZS_seqStore_reset(ZS_seqStore* seqStore)
{
    seqStore->lits.ptr = seqStore->lits.start;
    seqStore->seqs.ptr = seqStore->seqs.start;
}

// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openzl/codecs/rolz/encode_rolz_sequences.h"

int ZS_RolzSeqStore_initExact(
        ZS_RolzSeqStore* seqStore,
        size_t numLiterals,
        size_t numSequences)
{
    seqStore->fieldMask = 0;
    seqStore->lits.ptr =
            (uint8_t*)malloc(numLiterals + ZS_WILDCOPY_OVERLENGTH + 1);
    seqStore->lits.start = seqStore->lits.ptr;
    seqStore->lits.end   = seqStore->lits.start + numLiterals + 1;

    seqStore->litsCtx.ptr =
            (uint8_t*)malloc(numLiterals + ZS_WILDCOPY_OVERLENGTH + 1);
    seqStore->litsCtx.start = seqStore->litsCtx.ptr;
    seqStore->litsCtx.end   = seqStore->litsCtx.start + numLiterals + 1;

    size_t const seqSpace = numSequences * sizeof(ZS_sequence);

    seqStore->seqs.ptr   = (ZS_sequence*)malloc(seqSpace);
    seqStore->seqs.start = seqStore->seqs.ptr;
    seqStore->seqs.end   = seqStore->seqs.start + numSequences;

    if (!seqStore->lits.ptr || !seqStore->litsCtx.ptr || !seqStore->seqs.ptr) {
        ZS_RolzSeqStore_destroy(seqStore);
        memset(seqStore, 0, sizeof(*seqStore));
        return 1;
    }

    // Add a 0 byte for the literals context
    *seqStore->lits.start++ = 0;
    seqStore->lits.ptr++;

    return 0;
}

int ZS_RolzSeqStore_initBound(
        ZS_RolzSeqStore* seqStore,
        size_t srcSize,
        size_t minMatch)
{
    size_t const maxNumSeqs = srcSize / minMatch;
    return ZS_RolzSeqStore_initExact(seqStore, srcSize, maxNumSeqs);
}

void ZS_RolzSeqStore_destroy(ZS_RolzSeqStore* seqStore)
{
    if (!seqStore) {
        return;
    }
    ZS_RolzSeqStore_reset(seqStore);
    if (seqStore->lits.start) {
        free(seqStore->lits.start - 1);
    }
    free(seqStore->litsCtx.start);
    free(seqStore->seqs.start);
}

void ZS_RolzSeqStore_reset(ZS_RolzSeqStore* seqStore)
{
    seqStore->lits.ptr    = seqStore->lits.start;
    seqStore->litsCtx.ptr = seqStore->litsCtx.start;
    seqStore->seqs.ptr    = seqStore->seqs.start;
}

// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/tokenize/encode_tokenize4to2_kernel.h"

#include <assert.h> // assert
#include <stdlib.h> // malloc, calloc, free

#include "openzl/common/debug.h"
#include "openzl/shared/bits.h"
#include "openzl/shared/estimate.h"
#include "openzl/shared/xxhash.h" // XXH3_64bits

// TODO :
// currently return a fixed value suitable for SAO speed vectors
// must be updated in favor of a fast method with reasonable accuracy
static int ZS_cardinalityLog(const uint32_t* srcSymbols, size_t nbSymbols)
{
    (void)srcSymbols;
    (void)nbSymbols;
    ZL_CardinalityEstimate c = ZL_estimateCardinality_fixed(
            srcSymbols, nbSymbols, 4, ZL_ESTIMATE_CARDINALITY_16BITS);
    if (c.estimate == 0) {
        return 0;
    }
    return (int)ZL_highbit32((uint32_t)c.estimate) + 1;
}

static int ZS_estimateHashSetLog(int cardLog)
{
    return cardLog + 1;
}

typedef struct {
    uint32_t symbol;
    uint32_t id;
} SymbolDesc;

static size_t ZS_estimateWorkspaceSize(int cardLog)
{
    return sizeof(SymbolDesc) << ZS_estimateHashSetLog(cardLog);
}

typedef struct {
    uint32_t lastID;
    int hashSetLog;
    int nullValue_isPresent;
    uint32_t nullValue_ID;
    SymbolDesc* descArray;
} HashSet;
#define NULL_VALUE 0
static uint16_t insertSymbol(HashSet* hs, int hashSetLog, uint32_t symbol)
{
    if (symbol == NULL_VALUE) {
        if (hs->nullValue_isPresent) {
            return (uint16_t)hs->nullValue_ID;
        }
        hs->nullValue_isPresent = 1;
        hs->nullValue_ID        = hs->lastID;
        return (uint16_t)hs->lastID++;
    }
    SymbolDesc* const da = hs->descArray;
    size_t hashPos =
            (size_t)(XXH3_64bits(&symbol, sizeof(symbol)) >> (64 - hashSetLog));
    while (da[hashPos].symbol != NULL_VALUE) { // assumption : hash set load is
                                               // ~relatively low (<30%)
        if (da[hashPos].symbol == symbol) {    // already present, report id
            assert(da[hashPos].id < 65536);
            return (uint16_t)da[hashPos].id;
        }
        hashPos = (hashPos + 1) & ((1u << hashSetLog) - 1);
    }
    da[hashPos].symbol = symbol;
    da[hashPos].id     = hs->lastID;
    return (uint16_t)hs->lastID++;
}

static void
resetSymbol(HashSet* hs, int hashSetLog, uint32_t symbol, uint16_t index)
{
    if (symbol == NULL_VALUE) {
        if (hs->nullValue_isPresent) {
            hs->nullValue_ID = (uint32_t)index;
        }
    }
    SymbolDesc* const da = hs->descArray;
    size_t hashPos =
            (size_t)(XXH3_64bits(&symbol, sizeof(symbol)) >> (64 - hashSetLog));
    while (da[hashPos].symbol != NULL_VALUE) { // assumption : hash set load is
                                               // ~relatively low (<30%)
        if (da[hashPos].symbol == symbol) {    // already present, report id
            assert(da[hashPos].id < 65536);
            da[hashPos].id = (uint32_t)index;
            return;
        }
        hashPos = (hashPos + 1) & ((1u << hashSetLog) - 1);
    }
}

static void
writeAlphabet(uint32_t* dstAlphabet, size_t alphabetCapacity, HashSet hs)
{
    (void)alphabetCapacity;
    assert(dstAlphabet != NULL);
    assert(hs.hashSetLog < (int)(sizeof(size_t) * 8));
    size_t const hashSetSize = (size_t)1 << hs.hashSetLog;
    for (size_t n = 0; n < hashSetSize; n++) {
        if (hs.descArray[n].symbol != NULL_VALUE) {
            assert(hs.descArray[n].id < alphabetCapacity);
            dstAlphabet[hs.descArray[n].id] = hs.descArray[n].symbol;
        }
    }
    if (hs.nullValue_isPresent) {
        dstAlphabet[hs.nullValue_ID] = NULL_VALUE;
    }
}

static int cmp4(void const* lp, void const* rp)
{
    uint32_t const lhs = *(uint32_t const*)lp;
    uint32_t const rhs = *(uint32_t const*)rp;
    if (lhs < rhs) {
        return -1;
    }
    if (lhs > rhs) {
        return 1;
    }
    return 0;
}

// TODO:
// currently allocates memory on heap
// must be updated to receive pre-allocated workspace from binding layer
size_t ZS_tokenize4to2_encode(
        uint16_t* dstIndex,
        size_t indexCapacity,
        uint32_t* dstAlphabet,
        size_t alphabetCapacity,
        const uint32_t* srcSymbols,
        size_t nbSymbols,
        ZS_TokenizeAlphabetMode alphabetMode)
{
    ZL_LOG(TRANSFORM, "Tokenizing...");
    if (nbSymbols == 0) {
        return 0;
    }

    assert(dstIndex != NULL);
    assert(dstAlphabet != NULL);
    assert(srcSymbols != NULL);
    assert(indexCapacity >= nbSymbols);
    (void)indexCapacity;

    int const cardLog     = ZS_cardinalityLog(srcSymbols, nbSymbols);
    int const hashSetLog  = ZS_estimateHashSetLog(cardLog);
    size_t const wkspSize = ZS_estimateWorkspaceSize(cardLog);
    void* const wksp =
            calloc(1, wkspSize); // Note : preferably, transforms should
                                 // never use malloc/calloc/free directly
    assert(wksp != NULL); // Note : can fail. Another reason to do allocation
                          // in an upper layer.
    HashSet hashSet = { .nullValue_isPresent = 0,
                        .lastID              = 0,
                        .hashSetLog          = hashSetLog,
                        .descArray           = wksp };

    for (size_t n = 0; n < nbSymbols; n++) {
        dstIndex[n] = insertSymbol(&hashSet, hashSetLog, srcSymbols[n]);
    }

    writeAlphabet(dstAlphabet, alphabetCapacity, hashSet);

    if (alphabetMode == ZS_tam_sorted) {
        // TODO: Optimize this...
        qsort(dstAlphabet, hashSet.lastID, 4, cmp4);
        for (size_t id = 0; id < hashSet.lastID; ++id) {
            resetSymbol(&hashSet, hashSetLog, dstAlphabet[id], (uint16_t)id);
        }

        for (size_t n = 0; n < nbSymbols; n++) {
            dstIndex[n] = insertSymbol(&hashSet, hashSetLog, srcSymbols[n]);
        }
    }

    free(wksp);
    ZL_LOG(TRANSFORM, "Finished tokenizaing into %u tokens", hashSet.lastID);
    return hashSet.lastID;
}

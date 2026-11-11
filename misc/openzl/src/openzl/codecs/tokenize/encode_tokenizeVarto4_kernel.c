// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/tokenize/encode_tokenizeVarto4_kernel.h"

#include <assert.h> // assert
#include <string.h> // memcmp, memcpy

#include "openzl/shared/xxhash.h" // XXH3_64bits

static uint32_t ZS_log2(uint32_t v)
{
    assert(v > 0);
    uint32_t p = 0;
    while (v) {
        p++;
        v >>= 1;
    }
    return p - 1;
}

static int ZS_estimateHashLog(uint32_t cardinalityEstimation)
{
    assert(cardinalityEstimation > 0);
    assert(cardinalityEstimation < (UINT_MAX / 3));
    int baseLog = (int)ZS_log2(cardinalityEstimation * 3);
    baseLog += (baseLog < 17) + (baseLog < 15) + (baseLog < 13);
    return baseLog;
}

#define NOT_PRESENT 0 // hashMap's array is memset(0) or calloc
typedef struct {
    uint32_t pos;
    uint32_t len;
    uint32_t id_plus1; // keep the 0 to mean NOT_PRESENT
} SymbolDesc;

static size_t ZS_estimateWorkspaceSize(uint32_t hashLog)
{
    return sizeof(SymbolDesc) << hashLog;
}

size_t ZS_tokenizeVarto4_encode_wkspSize(uint32_t cardinalityEstimation)
{
    uint32_t const hashLog =
            (uint32_t)ZS_estimateHashLog(cardinalityEstimation);
    return ZS_estimateWorkspaceSize(hashLog);
}

typedef struct {
    char* dstBuffer;
    size_t dstPos;
    size_t* symbolSizes;
} DstAlphabet;

typedef struct {
    uint32_t lastID;
    uint32_t hashLog;
    SymbolDesc* descArray;
    const char* srcStart;
    DstAlphabet dstAlphabet; // Alphabet will be written on the go, as hashMap
                             // get populated
} HashMap;

static uint32_t ZS_tv4e_insertToken(HashMap* hs, size_t pos, size_t len)
{
    SymbolDesc* const da     = hs->descArray;
    uint32_t const hashLog   = hs->hashLog;
    size_t const hMask       = ((size_t)1 << hashLog) - 1;
    const char* const srcPtr = hs->srcStart + pos;
    size_t hashPos = (size_t)(XXH3_64bits(srcPtr, len) >> (64 - hashLog));
    // Note: the loop below could become never-ending if the array becomes full,
    //       but the assumption is that the load remains ~relatively low (<30%)
    assert(hs->lastID < hMask);
    while (da[hashPos].id_plus1 != NOT_PRESENT) {
        size_t const curlen = da[hashPos].len;
        if (len == curlen
            && !memcmp(hs->srcStart + da[hashPos].pos, srcPtr, len)) {
            return da[hashPos].id_plus1 - 1;
        }
        hashPos = (hashPos + 1) & hMask;
    }
    // Now this is an available slot
    // Let's create a new token ID and save it into dstAlphabet
    assert(da[hashPos].id_plus1 == NOT_PRESENT);
    da[hashPos].id_plus1 = ++hs->lastID;
    assert(pos < UINT32_MAX);
    da[hashPos].pos = (uint32_t)pos;
    assert(len < UINT32_MAX);
    da[hashPos].len = (uint32_t)len;
    assert(hs->dstAlphabet.dstBuffer != NULL);
    memcpy(hs->dstAlphabet.dstBuffer + hs->dstAlphabet.dstPos,
           srcPtr,
           len); // could be optimized using overwriting (memcpy(,,16) + loop
                 // for large ones)
    hs->dstAlphabet.dstPos += len;
    assert(hs->dstAlphabet.symbolSizes != NULL);
    hs->dstAlphabet.symbolSizes[da[hashPos].id_plus1 - 1] = len;
    // printf("creating token ID %u , of size=%2zu  '%*.*s'\n",
    // da[hashPos].id_plus1 - 1, len, (int)len, (int)len, srcPtr);
    return da[hashPos].id_plus1 - 1;
}

// TODO:
// currently allocates memory on heap
// must be extended, to allow receiving pre-allocate workspace from caller
ZS_TokVar_result ZS_tokenizeVarto4_encode(
        uint32_t* dstIndex,
        size_t indexCapacity,
        void* dstBuffer,
        size_t dstCapacity,
        size_t* symbolSizes,
        size_t ssaCapacity,
        const void* srcBuffer,
        size_t srcBufferSize,
        const size_t* tokenSizes,
        size_t nbTokens,
        uint32_t cardinalityEstimation,
        void* wksp,
        size_t wkspSize)
{
    assert(dstIndex != NULL);
    assert(dstBuffer != NULL);
    assert(symbolSizes != NULL);
    assert(srcBuffer != NULL);
    assert(tokenSizes != NULL);
    assert(indexCapacity >= nbTokens);
    (void)indexCapacity;
    assert(dstCapacity >= srcBufferSize);
    (void)dstCapacity;
    assert(ssaCapacity >= nbTokens);
    (void)ssaCapacity;
    assert(wksp != NULL);
    // consider adding a complex assert, which checks that
    // sum(tokenSizes)==srcBufferSize

    uint32_t const hashLog =
            (uint32_t)ZS_estimateHashLog(cardinalityEstimation);
    size_t const hashMapSize = ZS_estimateWorkspaceSize(hashLog);
    assert(wkspSize >= hashMapSize);
    (void)wkspSize;
    memset(wksp, 0, hashMapSize);
    HashMap hashMap = { .hashLog                 = hashLog,
                        .descArray               = wksp,
                        .srcStart                = srcBuffer,
                        .dstAlphabet.dstBuffer   = dstBuffer,
                        .dstAlphabet.symbolSizes = symbolSizes };

    size_t pos = 0;
    for (size_t n = 0; n < nbTokens; n++) {
        size_t const tokenSize = tokenSizes[n];
        assert(pos + tokenSize <= srcBufferSize);
        (void)srcBufferSize;
        dstIndex[n] = ZS_tv4e_insertToken(&hashMap, pos, tokenSize);
        pos += tokenSize;
    }

    return (ZS_TokVar_result){ .alphabetSize = hashMap.lastID, .dstSize = pos };
}

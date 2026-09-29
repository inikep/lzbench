// Copyright (c) Meta Platforms, Inc. and affiliates.

// note : relative-path include (same directory by default)
#include "decode_dispatch_by_tag_kernel.h"

#include <assert.h>
#include <string.h> // memcpy

static size_t sumArrayST(const size_t array[], size_t arraySize)
{
    size_t total = 0;
    for (size_t n = 0; n < arraySize; n++) {
        total += array[n];
    }
    return total;
}

static inline size_t ZS_DispatchByTag_decode_internal(
        void* restrict dst,
        size_t dstCapacity,
        const void* restrict srcs[],
        const size_t nbElts[],
        size_t nbSrcs,
        size_t eltSize,
        const uint8_t* restrict indexBuffer)
{
    if (dstCapacity) {
        assert(dst != NULL);
    }
    size_t const totalElts         = sumArrayST(nbElts, nbSrcs);
    size_t pos[JOINBY_NB_SRCS_MAX] = { 0 };
    assert(nbSrcs <= JOINBY_NB_SRCS_MAX);
    assert(dstCapacity >= totalElts * eltSize);
    for (size_t n = 0; n < totalElts; n++) {
        int const srcId = indexBuffer[n];
        memcpy(dst, (const char*)srcs[srcId] + (pos[srcId] * eltSize), eltSize);
        dst = (char*)dst + eltSize;
        pos[srcId]++;
    }
    return totalElts;
}

size_t ZS_DispatchByTag_decode(
        void* restrict dst,
        size_t dstCapacity,
        const void* restrict srcs[],
        const size_t nbElts[],
        size_t nbSrcs,
        size_t eltSize,
        const uint8_t* restrict indexBuffer)
{
    switch (eltSize) {
        case 4:
            return ZS_DispatchByTag_decode_internal(
                    dst, dstCapacity, srcs, nbElts, nbSrcs, 4, indexBuffer);
        default:
            return ZS_DispatchByTag_decode_internal(
                    dst,
                    dstCapacity,
                    srcs,
                    nbElts,
                    nbSrcs,
                    eltSize,
                    indexBuffer);
    }
}

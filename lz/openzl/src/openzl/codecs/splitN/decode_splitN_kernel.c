// Copyright (c) Meta Platforms, Inc. and affiliates.

// note : raw transform, minimal include policy
#include "decode_splitN_kernel.h"

#include <assert.h>
#include <string.h> // memcpy

static size_t sumArrayST(const size_t array[], size_t arraySize)
{
    size_t total = 0;
    if (arraySize) {
        assert(array != NULL);
    }
    for (size_t n = 0; n < arraySize; n++) {
        total += array[n];
    }
    return total;
}

size_t ZS_appendN(
        void* restrict dst,
        size_t dstCapacity,
        const void* restrict srcs[],
        const size_t srcSizes[],
        size_t nbSrcs)
{
    if (dstCapacity) {
        assert(dst != NULL);
    }
    size_t const dstSize = sumArrayST(srcSizes, nbSrcs);
    assert(dstSize <= dstCapacity);
    if (nbSrcs) {
        assert(srcSizes != NULL);
        assert(srcs != NULL);
    }
    for (size_t n = 0; n < nbSrcs; n++) {
        size_t const srcSize = srcSizes[n];
        if (srcSize) {
            assert(srcs[n] != NULL);
            memcpy(dst, srcs[n], srcSizes[n]);
            dst = (char*)dst + srcSizes[n];
        }
    }
    return dstSize;
}

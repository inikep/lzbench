// Copyright (c) Meta Platforms, Inc. and affiliates.

// note : this is the raw kernel transform,
//        which implies minimal #include policy
#include "decode_splitByStruct_kernel.h"

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

// Variant dedicated to all @fieldSizes <= 8
// Note : in the future more variant could be created for other sizes,
// typically 4 and 16, maybe 2 and 1, possibly 32, etc.
static size_t rejoin_max8(
        void* restrict dst,
        size_t dstCapacity,
        const void* restrict srcs[],
        const size_t fieldSizes[],
        size_t nbFields,
        const size_t nbElts)
{
    if (dstCapacity) {
        assert(dst != NULL);
    }
    if (!nbElts) {
        return 0;
    }
    assert(srcs != NULL);
    assert(fieldSizes != NULL);
    assert(nbFields);
    size_t minSize = (size_t)(-1);
    for (size_t n = 0; n < nbFields; n++) {
        assert(srcs[n] != NULL);
        assert(fieldSizes[n] <= 8);
        if (fieldSizes[n] < minSize) {
            minSize = fieldSizes[n];
        }
    }
    assert(minSize > 0);
    size_t const nbSafeRounds_precalc[8] = { 8, 4, 3, 2, 2, 2, 2, 1 };
    size_t const nbSafeRounds            = nbSafeRounds_precalc[minSize - 1];

    size_t const structSize = sumArrayST(fieldSizes, nbFields);
    size_t const dstSize    = structSize * nbElts;
    assert(dstSize <= dstCapacity);

    size_t wPos     = 0;
    size_t structNb = 0;
    if (nbElts > nbSafeRounds) {
        for (; structNb < (nbElts - nbSafeRounds); structNb++) {
            for (size_t f = 0; f < nbFields; f++) {
                size_t const fs = fieldSizes[f];
                memcpy((char*)dst + wPos,
                       (const char*)srcs[f] + structNb * fs,
                       8);
                wPos += fs;
            }
        }
    }
    // Finalize, using exact field sizes
    for (; structNb < nbElts; structNb++) {
        for (size_t f = 0; f < nbFields; f++) {
            size_t const fs = fieldSizes[f];
            memcpy((char*)dst + wPos, (const char*)srcs[f] + structNb * fs, fs);
            wPos += fs;
        }
    }
    assert(wPos == dstSize);
    return dstSize;
}

size_t ZS_dispatchArrayFixedSizeStruct_decode(
        void* restrict dst,
        size_t dstCapacity,
        const void* restrict srcs[],
        const size_t fieldSizes[],
        size_t nbFields,
        const size_t nbElts)
{
    if (dstCapacity) {
        assert(dst != NULL);
    }
    if (!nbElts) {
        return 0;
    }
    assert(srcs != NULL);
    assert(fieldSizes != NULL);
    assert(nbFields);
    size_t maxFieldSize = 0;
    size_t minFieldSize = (size_t)(-1);
    for (size_t n = 0; n < nbFields; n++) {
        assert(srcs[n] != NULL);
        if (fieldSizes[n] > maxFieldSize) {
            maxFieldSize = fieldSizes[n];
        }
        if (fieldSizes[n] < minFieldSize) {
            minFieldSize = fieldSizes[n];
        }
    }
    size_t const structSize = sumArrayST(fieldSizes, nbFields);
    size_t const dstSize    = structSize * nbElts;
    assert(dstSize <= dstCapacity);

    // Shortcut to 8-bytes variant
    if (maxFieldSize <= 8 && minFieldSize >= 1) {
        return rejoin_max8(
                dst, dstCapacity, srcs, fieldSizes, nbFields, nbElts);
    }

    // Generic variant (slower)
    size_t pos = 0;
    for (size_t e = 0; e < nbElts; e++) {
        for (size_t f = 0; f < nbFields; f++) {
            size_t const fs = fieldSizes[f];
            // Note : generic variant employs memcpy() directly.
            // Speed can be improved with specialized variants,
            // like `_max8`, using static size copies.
            // Note 2 : In the future,
            // this is an ideal scenario for JIT compilers.
            // The nb and length of each copy is known beforehand,
            // so creating the exact sequence on instructions for each loop
            // looks plausible.
            memcpy((char*)dst + pos, (const char*)srcs[f] + e * fs, fs);
            pos += fs;
        }
    }
    assert(pos == dstSize);
    return dstSize;
}

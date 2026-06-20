// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/shared/histogram.h"
#include "openzl/common/debug.h"
#include "openzl/fse/hist.h"

#include <stdlib.h>
#include <string.h>

ZL_Histogram* ZL_Histogram_create(unsigned maxSymbol)
{
    size_t const histogramSize =
            sizeof(ZL_Histogram) + sizeof(unsigned) * maxSymbol;
    ZL_Histogram* histogram = (ZL_Histogram*)malloc(histogramSize);
    if (histogram != NULL) {
        ZL_Histogram_init(histogram, maxSymbol);
    }
    return histogram;
}

void ZL_Histogram_destroy(ZL_Histogram* histogram)
{
    free(histogram);
}

/// maxSymbol must be a static upper bound of the symbol size,
/// and hist must be large enough.
void ZL_Histogram_init(ZL_Histogram* hist, unsigned maxSymbol)
{
    hist->elementSize  = 0;
    hist->largestCount = 0;
    hist->total        = 0;
    hist->cardinality  = 0;
    hist->maxSymbol    = maxSymbol;
}

static unsigned ZS_Histogram_count16(
        unsigned* count,
        unsigned* maxSymbolValuePtr,
        unsigned* cardinalityPtr,
        uint16_t const* src,
        size_t nbElts)
{
    size_t countSize = (1 + *maxSymbolValuePtr) * sizeof(count[0]);
    memset(count, 0, countSize);
    int max = 0;

    if (nbElts == 0) {
        *maxSymbolValuePtr = 0;
        *cardinalityPtr    = 0;
        return 0;
    }

    for (size_t i = 0; i < nbElts; ++i) {
        count[src[i]] += 1;
        if (src[i] > max) {
            max = src[i];
        }
    }

    // Handle constant case.
    if (count[max] == nbElts) {
        *maxSymbolValuePtr = (unsigned)max;
        *cardinalityPtr    = 1;
        return count[max];
    }

    unsigned largestCount = 0;
    unsigned cardinality  = 0;
    for (int i = 0; i <= max; ++i) {
        if (count[i] > largestCount) {
            largestCount = count[i];
        }
        if (count[i] > 0) {
            cardinality += 1;
        }
    }
    *maxSymbolValuePtr = (unsigned)max;
    *cardinalityPtr    = cardinality;

    return largestCount;
}

unsigned ZL_Histogram_count(
        unsigned* count,
        unsigned* maxSymbolValue,
        unsigned* cardinality,
        void const* src,
        size_t nbElts,
        size_t eltWidth)
{
    if (eltWidth == 1) {
        return (unsigned)HIST_countFast(
                count, maxSymbolValue, cardinality, src, nbElts);
    } else {
        ZL_ASSERT_EQ(eltWidth, 2);
        return ZS_Histogram_count16(
                count, maxSymbolValue, cardinality, src, nbElts);
    }
}

/// all elements in src must be <= maxSymbol.
void ZL_Histogram_build(
        ZL_Histogram* hist,
        void const* src,
        size_t nbElts,
        size_t eltWidth)
{
    hist->elementSize  = (unsigned)eltWidth;
    hist->total        = (unsigned)nbElts;
    hist->largestCount = ZL_Histogram_count(
            hist->count,
            &hist->maxSymbol,
            &hist->cardinality,
            src,
            nbElts,
            eltWidth);
}

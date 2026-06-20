// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZSTRONG_COMMON_HISTOGRAM_H
#define ZSTRONG_COMMON_HISTOGRAM_H

#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/**
 * Generic histogram base class.
 */
typedef struct {
    unsigned total;
    unsigned maxSymbol;
    unsigned largestCount;
    unsigned elementSize;
    unsigned cardinality;
    unsigned count[1];
} ZL_Histogram;

/**
 * Histogram sub-class for 8-bit values.
 */
typedef struct {
    ZL_Histogram base;
    unsigned _count[255];
} ZL_Histogram8;

/**
 * Histogram sub-class for 16-bit values.
 */
typedef struct {
    ZL_Histogram base;
    unsigned _count[(1u << 16) - 1];
} ZL_Histogram16;

ZL_Histogram* ZL_Histogram_create(unsigned maxSymbol);
void ZL_Histogram_destroy(ZL_Histogram*);

/// maxSymbol must be a static upper bound of the symbol size,
/// and hist must be large enough.
void ZL_Histogram_init(ZL_Histogram* hist, unsigned maxSymbol);
/// all elements in src must be <= maxSymbol.
void ZL_Histogram_build(
        ZL_Histogram* hist,
        void const* src,
        size_t nbElts,
        size_t eltWidth);

/**
 * Computes a histogram of the data.
 *
 * @param[out] count The histogram to build. It is zeroed before it is filled.
 * It must be @param[in] maxSymbolValue + 1 elements large.
 * @param[inout] maxSymbolValue Must be a static bound on the maximum symbol
 * value. Set to the maximum symbol value.
 * @param[out] cardinalityPtr The cardinality of the histogram.
 * @param src The data to count.
 * @param nbElts The number of elements in the data.
 * @param eltWidth The width of each element in bytes.
 *
 * @returns The count of the most frequent symbol
 */
unsigned ZL_Histogram_count(
        unsigned* count,
        unsigned* maxSymbolValue,
        unsigned* cardinalityPtr,
        void const* src,
        size_t nbElts,
        size_t eltWidth);

ZL_END_C_DECLS

#endif // ZSTRONG_COMMON_HISTOGRAM_H

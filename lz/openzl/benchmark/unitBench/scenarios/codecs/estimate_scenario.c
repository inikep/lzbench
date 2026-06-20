// Copyright (c) Meta Platforms, Inc. and affiliates.

// This file is named differently from the pattern because the Makefile cannot
// handle multiple files with the same name, even in different directories.

#include "benchmark/unitBench/scenarios/codecs/estimate.h"

#include "openzl/shared/estimate.h"

// --8<-- [start:custom-wrapper]
size_t exact2_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dst;
    (void)dstCapacity;
    uint8_t present[1u << 16];
    memset(present, 0, sizeof(present));
    typedef uint16_t Elt;
    size_t const nbElts = srcSize / sizeof(Elt);
    Elt const* ptr      = (Elt const*)src;

    for (size_t i = 0; i < nbElts; ++i) {
        present[ptr[i]] = 1;
    }
    size_t cardinality = 0;
    for (size_t i = 0; i < sizeof(present); ++i) {
        cardinality += present[i];
    }

    return cardinality;
}
// --8<-- [end:custom-wrapper]

static size_t estimate_impl(
        const void* src,
        size_t srcSize,
        size_t eltSize,
        size_t cardinalityEarlyExit)
{
    size_t const nbElts             = srcSize / eltSize;
    ZL_CardinalityEstimate estimate = ZL_estimateCardinality_fixed(
            src, nbElts, eltSize, cardinalityEarlyExit);
    return estimate.estimate;
}

size_t estimate1_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dst;
    (void)dstCapacity;
    return estimate_impl(src, srcSize, 1, ZL_ESTIMATE_CARDINALITY_ANY);
}
size_t estimate2_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dst;
    (void)dstCapacity;
    return estimate_impl(src, srcSize, 2, ZL_ESTIMATE_CARDINALITY_ANY);
}
size_t estimateLC4_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dst;
    (void)dstCapacity;
    return estimate_impl(src, srcSize, 4, ZL_ESTIMATE_CARDINALITY_16BITS);
}
size_t estimateHLL4_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dst;
    (void)dstCapacity;
    return estimate_impl(src, srcSize, 4, ZL_ESTIMATE_CARDINALITY_ANY);
}
size_t estimateLC8_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dst;
    (void)dstCapacity;
    return estimate_impl(src, srcSize, 8, ZL_ESTIMATE_CARDINALITY_16BITS);
}
size_t estimateHLL8_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dst;
    (void)dstCapacity;
    return estimate_impl(src, srcSize, 8, ZL_ESTIMATE_CARDINALITY_ANY);
}
size_t dimensionality1_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dst;
    (void)dstCapacity;
    size_t const nbElts = srcSize / 1;
    ZL_DimensionalityEstimate estimate =
            ZL_estimateDimensionality(src, nbElts, 1);
    return estimate.dimensionality == ZL_DimensionalityStatus_likely2D
            ? estimate.stride
            : 0;
}
size_t dimensionality2_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dst;
    (void)dstCapacity;
    size_t const nbElts = srcSize / 2;
    ZL_DimensionalityEstimate estimate =
            ZL_estimateDimensionality(src, nbElts, 2);
    return estimate.dimensionality == ZL_DimensionalityStatus_likely2D
            ? estimate.stride
            : 0;
}
size_t dimensionality3_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dst;
    (void)dstCapacity;
    size_t const nbElts = srcSize / 3;
    ZL_DimensionalityEstimate estimate =
            ZL_estimateDimensionality(src, nbElts, 3);
    return estimate.dimensionality == ZL_DimensionalityStatus_likely2D
            ? estimate.stride
            : 0;
}
size_t dimensionality4_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dst;
    (void)dstCapacity;
    size_t const nbElts = srcSize / 4;
    ZL_DimensionalityEstimate estimate =
            ZL_estimateDimensionality(src, nbElts, 4);
    return estimate.dimensionality == ZL_DimensionalityStatus_likely2D
            ? estimate.stride
            : 0;
}
size_t dimensionality8_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dst;
    (void)dstCapacity;
    size_t const nbElts = srcSize / 8;
    ZL_DimensionalityEstimate estimate =
            ZL_estimateDimensionality(src, nbElts, 8);
    return estimate.dimensionality == ZL_DimensionalityStatus_likely2D
            ? estimate.stride
            : 0;
}

// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_SHARED_ESTIMATE_H
#define OPENZL_SHARED_ESTIMATE_H

#include "openzl/shared/portability.h"

#include <stddef.h>
#include <stdint.h>

ZL_BEGIN_C_DECLS

typedef struct {
    uint64_t min;
    uint64_t max;
} ZL_ElementRange;

/**
 * Returns the exact range of a stream of elements.
 */
ZL_ElementRange
ZL_computeUnsignedRange(void const* src, size_t nbElts, size_t eltSize);
ZL_ElementRange ZL_computeUnsignedRange64(uint64_t const* src, size_t srcSize);
ZL_ElementRange ZL_computeUnsignedRange32(uint32_t const* src, size_t srcSize);
ZL_ElementRange ZL_computeUnsignedRange16(uint16_t const* src, size_t srcSize);
ZL_ElementRange ZL_computeUnsignedRange8(uint8_t const* src, size_t srcSize);

/**
 * An estimate of the cardinality of the stream.
 *
 * lowerBound <= estimateLowerBound <= estimate <= estimateUpperBound <=
 * upperBound
 *
 * The lowerBound and upperBound are hard bounds. In the case that they are
 * unknown they are 0 & 2^64-1 respectively. They have no guarantee to be
 * tight.
 *
 * The estimateLowerBound and estimateUpperBound are estimates of the
 * error bars, but note that they are soft bounds, and can still be wrong.
 *
 */
typedef struct {
    uint64_t lowerBound;
    uint64_t estimateLowerBound;
    uint64_t estimate;
    uint64_t estimateUpperBound;
    uint64_t upperBound;
} ZL_CardinalityEstimate;

#define ZL_ESTIMATE_CARDINALITY_ANY 0
#define ZL_ESTIMATE_CARDINALITY_7BITS 128
#define ZL_ESTIMATE_CARDINALITY_8BITS 256
#define ZL_ESTIMATE_CARDINALITY_16BITS 65536
#define ZL_ESTIMATE_CARDINALITY_MAX (1u << 31)

/**
 * Returns an estimate of the cardinality of the stream.
 *
 * @param src Must be nbElts*eltSize bytes.
 * @param cardinalityEarlyExit The maximum interesting cardinality. If the
 * estimator sees that the cardinality is larger than this, it is free to
 * stop early and report any value >= cardinalityEarlyExit. We also use this
 * to select table sizes & implementations to tune the algorithm for speed.
 * It will be automatically adjusted down to respect the bounds that `nbElts`
 * and `eltSize` place on the cardinality.
 *
 * NOTE: The implementation is much faster for cardinalityEarlyExit <= 64K.
 * You should only select a large value if you really care about cardinalities
 * > 64K.
 */
ZL_CardinalityEstimate ZL_estimateCardinality_fixed(
        void const* src,
        size_t nbElts,
        size_t eltSize,
        uint64_t cardinalityEarlyExit);

/**
 * Returns an estimate of the cardinality of the elements in the variable-sized
 * stream. See @ZL_estimateCardinality_fixed.
 *
 * @param srcs The source pointers. `srcs[i]` must have length `eltSizes[i]`.
 * @param eltSizes The sizes.
 * @param nbElts The size of `srcs` and `eltSizes.
 */
ZL_CardinalityEstimate ZL_estimateCardinality_variable(
        void const* const* srcs,
        size_t const* eltSizes,
        size_t nbElts,
        uint64_t cardinalityEarlyExit);

/// A summary of what we estimate the dimensionality is.
typedef enum {
    /// No dimensionality detected.
    ZL_DimensionalityStatus_none,
    /// The data may be 2D, but it isn't strongly dimensional.
    /// Don't blindly assume the data is dimensional with this result.
    /// Use the match information to decide if the 2D structure is strong
    /// enough for your use case.
    ZL_DimensionalityStatus_possibly2D,
    /// The data is very likely 2D, and is strongly dimensional.
    /// We've verified there is a strong dimensionality component, but it may
    /// not be the only correlation that exists.
    ZL_DimensionalityStatus_likely2D,
} ZL_DimensionalityStatus;

/**
 * An estimate of the data's dimensionality.
 */
typedef struct {
    /// What is the dimensionality?
    ZL_DimensionalityStatus dimensionality;
    /// The estimated stride of the dimensionality
    /// NOTE: In number of elements, not bytes!
    size_t stride;
    /// The number of matching elements we've found at an offset that is an
    /// exact multiple of stride. Use this with totalMatches to find the ratio
    /// of matches that are exactly a 2D match.
    size_t strideMatches;
    /// The total number of matching elements at any offset.
    size_t totalMatches;
} ZL_DimensionalityEstimate;

/// Returns an estimate of the dimensionality of src.
ZL_DimensionalityEstimate
ZL_estimateDimensionality(void const* src, size_t nbElts, size_t eltSize);
ZL_DimensionalityEstimate ZL_estimateDimensionality1(
        void const* src,
        size_t nbElts);
ZL_DimensionalityEstimate ZL_estimateDimensionality2(
        void const* src,
        size_t nbElts);
ZL_DimensionalityEstimate ZS_estimateDimensionality3(
        void const* src,
        size_t nbElts);
ZL_DimensionalityEstimate ZL_estimateDimensionality4(
        void const* src,
        size_t nbElts);
ZL_DimensionalityEstimate ZL_estimateDimensionality8(
        void const* src,
        size_t nbElts);

/**
 * @returns The estimated width of the floating point data
 * in the source stream in bytes. If the source stream is not
 * floating point data, this function may return any width.
 */
size_t ZL_guessFloatWidth(void const* src, size_t srcSize);

ZL_END_C_DECLS

#endif // OPENZL_SHARED_ESTIMATE_H

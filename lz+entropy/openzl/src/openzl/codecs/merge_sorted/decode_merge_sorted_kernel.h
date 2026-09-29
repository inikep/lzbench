// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZSTRONG_TRANSFORMS_MERGE_SORTED_DECODE_MERGE_SORTED_KERNEL_H
#define ZSTRONG_TRANSFORMS_MERGE_SORTED_DECODE_MERGE_SORTED_KERNEL_H

#include "openzl/shared/portability.h"
#include "openzl/zl_errors.h"

ZL_BEGIN_C_DECLS

/**
 * Splits @p bitsets and @p merged into 0-8 @p dsts.
 *
 * @param merged The sorted unique values.
 * @param bitsets One bitset for each value in @p merged.
 *                The b'th bit is 1 iff dsts[b] has the value.
 * @returns true iff the split succeeded and all @p dsts are full.
 */
bool ZL_MergeSorted_split8x32(
        uint32_t** dsts,
        uint32_t** dstEnds,
        size_t nbDsts,
        uint8_t const* bitsets,
        uint32_t const* merged,
        size_t nbUniqueValues);

/// Splits bitsets & merged into 0-16 @p dsts.
bool ZL_MergeSorted_split16x32(
        uint32_t** dsts,
        uint32_t** dstEnds,
        size_t nbDsts,
        uint16_t const* bitsets,
        uint32_t const* merged,
        size_t nbUniqueValues);

/// Splits bitsets & merged into 0-32 @p dsts.
bool ZL_MergeSorted_split32x32(
        uint32_t** dsts,
        uint32_t** dstEnds,
        size_t nbDsts,
        uint32_t const* bitsets,
        uint32_t const* merged,
        size_t nbUniqueValues);

/// Splits bitsets & merged into 0-64 @p dsts.
bool ZL_MergeSorted_split64x32(
        uint32_t** dsts,
        uint32_t** dstEnds,
        size_t nbDsts,
        uint64_t const* bitsets,
        uint32_t const* merged,
        size_t nbUniqueValues);

ZL_END_C_DECLS

#endif

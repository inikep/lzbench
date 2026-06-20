// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZSTRONG_TRANSFORMS_MERGE_SORTED_ENCODE_MERGE_SORTED_KERNEL_H
#define ZSTRONG_TRANSFORMS_MERGE_SORTED_ENCODE_MERGE_SORTED_KERNEL_H

#include "openzl/shared/portability.h"
#include "openzl/zl_errors.h"

ZL_BEGIN_C_DECLS

/**
 * Merges 0-8 sources into merged & bitsets.
 * @param[out] merged contains the merged unique values in sorted order.
 * @param[out] bitsets one bitset for each value in merged. Bit b is 1
 *                     iff src[b] contains that value.
 * @returns The number of unique values or an error.
 */
ZL_Report ZL_MergeSorted_merge8x32(
        uint8_t* bitsets,
        uint32_t* merged,
        uint32_t const** srcs,
        uint32_t const** srcEnds,
        size_t nbSrcs);

/// Merges 0-16 sources into merged & bitsets.
ZL_Report ZL_MergeSorted_merge16x32(
        uint16_t* bitsets,
        uint32_t* merged,
        uint32_t const** srcs,
        uint32_t const** srcEnds,
        size_t nbSrcs);

/// Merges 0-32 sources into merged & bitsets.
ZL_Report ZL_MergeSorted_merge32x32(
        uint32_t* bitsets,
        uint32_t* merged,
        uint32_t const** srcs,
        uint32_t const** srcEnds,
        size_t nbSrcs);

/// Merges 0-64 sources into merged & bitsets.
ZL_Report ZL_MergeSorted_merge64x32(
        uint64_t* bitsets,
        uint32_t* merged,
        uint32_t const** srcs,
        uint32_t const** srcEnds,
        size_t nbSrcs);

ZL_END_C_DECLS

#endif

// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_RANGE_PACK_DECODE_RANGE_PACK_KERNEL_H
#define ZSTRONG_TRANSFORMS_RANGE_PACK_DECODE_RANGE_PACK_KERNEL_H

#include "openzl/shared/portability.h"
#include "openzl/zl_errors.h"

ZL_BEGIN_C_DECLS

/**
 * The opposite of ragePackEcnode.
 * Notes:
 * - Only suports widths of 1/2/4/8.
 * - @p dst needs to be at least of size @p nbElts * @p dstWidth
 */
void rangePackDecode(
        void* dst,
        size_t dstWidth,
        const void* src,
        size_t srcWidth,
        size_t nbElts,
        size_t dstMinValue);

ZL_END_C_DECLS

#endif

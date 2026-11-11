// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_RANGE_PACK_ENCODE_RANGE_PACK_KERNEL_H
#define ZSTRONG_TRANSFORMS_RANGE_PACK_ENCODE_RANGE_PACK_KERNEL_H

#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/**
 * rangePackEncode:
 * Packs integers whose range can be contained in a smaller type by storing the
 * diff from the minimal value.
 *
 * Encodes a buffer of @p nbElts unsigned integers of size @p srcWidth bytes
 * from @p src into an buffer @p dst of unsigned integers of size @p dstWidth
 * bytes.
 * @p srcMinValue is substracted from each element in @p src before being stored
 * in @p dst .
 * Notes:
 * - Only suports widths of 1/2/4/8.
 * - @p dst needs to be at least of size @p nbElts * @p dstWidth
 */
void rangePackEncode(
        void* dst,
        size_t dstWidth,
        const void* src,
        size_t srcWidth,
        size_t nbElts,
        size_t srcMinValue);

ZL_END_C_DECLS

#endif

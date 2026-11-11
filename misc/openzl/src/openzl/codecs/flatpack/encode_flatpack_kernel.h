// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZSTRONG_TRANSFORMS_FLATPACK_ENCODE_FLATPACK_KERNEL_H
#define ZSTRONG_TRANSFORMS_FLATPACK_ENCODE_FLATPACK_KERNEL_H

#include "openzl/codecs/flatpack/common_flatpack.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/// @returns A bound on the packed size that guarantees that
/// encoding will succeed.
static inline size_t ZS_flatpackEncodeBound(size_t srcSize)
{
    return srcSize + 1;
}

/**
 * Encodes the @src buffer using the "flatpack" encoding which
 * consists of tokenization & bitpacking. It accept single-byte
 * symbols, so there is only benefit to flatpacking when the
 * cardinality is <= 128.
 *
 * @param alphabet output alphabet buffer of size @alphabetCapacity.
 * It lists the symbols in in sorted order.
 * @param packed output bit-packed indices buffer of size @packedCapacity.
 * @returns The flat packed size which can be checked for errors, and can
 * report the number of bits used to encode, the alphabet size, and the
 * encoded size.
 */
ZS_FlatPackSize ZS_flatpackEncode(
        uint8_t* alphabet,
        size_t alphabetCapacity,
        uint8_t* packed,
        size_t packedCapacity,
        uint8_t const* src,
        size_t srcSize);

ZL_END_C_DECLS

#endif

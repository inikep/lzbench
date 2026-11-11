// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZSTRONG_TRANSFORMS_FLATPACK_DECODE_FLATPACK_KERNEL_H
#define ZSTRONG_TRANSFORMS_FLATPACK_DECODE_FLATPACK_KERNEL_H

#include "openzl/codecs/flatpack/common_flatpack.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/// @returns The exact number of encoded elements given the alphabet size
/// and the packed indices buffer.
static inline size_t ZS_FlatPack_nbElts(
        size_t alphabetSize,
        uint8_t const* packed,
        size_t packedSize)
{
    if (packedSize == 0 || alphabetSize == 0)
        return 0;
    uint32_t const lastByte  = packed[packedSize - 1] | 1;
    size_t const paddingBits = (size_t)ZL_clz32(lastByte << 24) + 1;
    size_t const nbBits = ZS_FlatPack_nbBits((ZS_FlatPackSize){ alphabetSize });
    size_t const packedBits = 8 * packedSize - paddingBits;
    return packedBits / nbBits;
}

/**
 * Decodes the "flatpack" format which is a combination of
 * single-byte tokenization & bitpacking.
 *
 * @param dst The destination buffer with capacity @dstCapacity.
 * Must be >= ZS_FlatPack_nbElts(alphabetSize, packed, packedSize)
 * or the decoding will fail.
 * @param alphabet The alphabet buffer of size @alphabetSize.
 * @param packed The packed indices buffer of sie @packedSize.
 * @returns The flat pack size which can be checked for errors.
 * The number of decoded elements is returned by ZS_FlatPack_nbElts().
 */
ZS_FlatPackSize ZS_flatpackDecode(
        uint8_t* dst,
        size_t dstCapacity,
        uint8_t const* alphabet,
        size_t alphabetSize,
        uint8_t const* packed,
        size_t packedSize);

ZL_END_C_DECLS

#endif

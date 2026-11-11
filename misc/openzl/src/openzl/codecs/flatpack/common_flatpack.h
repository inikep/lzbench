// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZSTRONG_TRANSFORMS_FLATPACK_COMMON_FLATPACK_H
#define ZSTRONG_TRANSFORMS_FLATPACK_COMMON_FLATPACK_H

#include "openzl/shared/bits.h"
#include "openzl/shared/portability.h"
#include "openzl/shared/utils.h"

ZL_BEGIN_C_DECLS

/// Return value for flatpack functions.
typedef struct {
    size_t _size;
} ZS_FlatPackSize;

/// @returns Is the return value reporting an error?
static inline bool ZS_FlatPack_isError(ZS_FlatPackSize size)
{
    return size._size > 256;
}

/// @returns the encoded alphabet size
static inline size_t ZS_FlatPack_alphabetSize(ZS_FlatPackSize size)
{
    ZL_ASSERT(!ZS_FlatPack_isError(size));
    return size._size;
}

/// @returns the number of bits used to encode each index
static inline size_t ZS_FlatPack_nbBits(ZS_FlatPackSize size)
{
    unsigned const alphabetSize = (unsigned)ZS_FlatPack_alphabetSize(size);
    if (alphabetSize <= 1)
        return alphabetSize;
    return 1 + (size_t)ZL_highbit32(alphabetSize - 1);
}

/// @returns the packed size of the indices.
static inline size_t ZS_FlatPack_packedSize(
        ZS_FlatPackSize size,
        size_t srcSize)
{
    if (srcSize == 0)
        return 0;
    return 1 + (ZS_FlatPack_nbBits(size) * srcSize) / 8;
}

ZL_END_C_DECLS

#endif

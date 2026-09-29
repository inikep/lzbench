// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_TOKENIZE_ENCODE_TOKENIZE_KERNEL_H
#define ZSTRONG_TRANSFORMS_TOKENIZE_ENCODE_TOKENIZE_KERNEL_H

#include "openzl/common/map.h"
#include "openzl/shared/portability.h"
#include "openzl/shared/xxhash.h"
#include "openzl/zl_data.h"

ZL_BEGIN_C_DECLS

typedef struct VSFKey {
    const uint8_t* fieldStart;
    uint32_t fieldSize;
} VSFKey;

ZL_FORCE_INLINE size_t MapVSF_hash(VSFKey const* key)
{
    return XXH3_64bits(key->fieldStart, key->fieldSize);
}

ZL_FORCE_INLINE bool MapVSF_eq(VSFKey const* lhs, VSFKey const* rhs)
{
    if (lhs->fieldSize != rhs->fieldSize) {
        return false;
    }

    return memcmp(lhs->fieldStart, rhs->fieldStart, lhs->fieldSize) == 0;
}

ZL_DECLARE_CUSTOM_MAP_TYPE(MapVSF, VSFKey, size_t);

// Constructs an alphabet from a buffer of variable-sized fields
// Note: also modifies @p alphabetFieldSizesSum to be the sum of each
// field size in the alphabet
ZL_Report ZS_buildTokenizeVsfAlphabet(
        MapVSF* tokToIdx,
        size_t* alphabetFieldSizesSum,
        const uint8_t* src,
        const uint32_t* fieldSizes,
        size_t nbElts);

ZL_Report ZS_tokenizeVSFEncode(
        uint8_t* alphabet,
        uint32_t* alphabetFieldSizes,
        size_t alphabetSize,
        uint8_t* indices,
        VSFKey* keysBuffer,
        const uint8_t* src,
        const uint32_t* fieldSizes,
        size_t nbElts,
        MapVSF* tokToIdx,
        size_t idxWidth,
        bool sort);

ZL_END_C_DECLS

#endif // ZSTRONG_TRANSFORMS_TOKENIZE_ENCODE_TOKENIZE_KERNEL_H

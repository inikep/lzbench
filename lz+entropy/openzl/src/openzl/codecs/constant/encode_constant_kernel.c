// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/constant/encode_constant_kernel.h"

#include <string.h> // memcpy

uint8_t ZS_isConstantStream(
        const uint8_t* const src,
        size_t const nbElts,
        size_t const eltWidth)
{
    if (nbElts == 0) {
        return 0;
    }

    return memcmp(src, src + eltWidth, eltWidth * (nbElts - 1)) == 0;
}

void ZS_encodeConstant(
        uint8_t* const dst,
        const uint8_t* const src,
        size_t const eltWidth)
{
    memcpy(dst, src, eltWidth);
}

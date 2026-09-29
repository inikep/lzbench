// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "decode_zigzag_kernel.h"

#include <assert.h>
#include <stdbool.h>

void ZL_zigzagDecode64(int64_t* dst, const uint64_t* src, size_t nbElts)
{
    for (size_t i = 0; i < nbElts; i++) {
        uint64_t const z    = src[i];
        uint64_t const mask = 0ULL - (z & 0x1);
        dst[i]              = (int64_t)((z >> 1) ^ mask);
    }
}

void ZL_zigzagDecode32(int32_t* dst, const uint32_t* src, size_t nbElts)
{
    for (size_t i = 0; i < nbElts; i++) {
        uint32_t const z    = src[i];
        uint32_t const mask = 0U - (z & 0x1);
        dst[i]              = (int32_t)((z >> 1) ^ mask);
    }
}

void ZL_zigzagDecode16(int16_t* dst, const uint16_t* src, size_t nbElts)
{
    for (size_t i = 0; i < nbElts; i++) {
        uint16_t const z    = src[i];
        uint16_t const mask = (uint16_t)0 - (z & 0x1);
        dst[i]              = (int16_t)((z >> 1) ^ mask);
    }
}

void ZL_zigzagDecode8(int8_t* dst, const uint8_t* src, size_t nbElts)
{
    for (size_t i = 0; i < nbElts; i++) {
        uint8_t const z    = src[i];
        uint8_t const mask = (uint8_t)0 - (z & 0x1);
        dst[i]             = (int8_t)((z >> 1) ^ mask);
    }
}

void ZL_zigzagDecode(void* dst, const void* src, size_t nbElts, size_t eltWidth)
{
    switch (eltWidth) {
        case 1:
            ZL_zigzagDecode8((int8_t*)dst, (uint8_t const*)src, nbElts);
            break;
        case 2:
            ZL_zigzagDecode16((int16_t*)dst, (uint16_t const*)src, nbElts);
            break;
        case 4:
            ZL_zigzagDecode32((int32_t*)dst, (uint32_t const*)src, nbElts);
            break;
        case 8:
            ZL_zigzagDecode64((int64_t*)dst, (uint64_t const*)src, nbElts);
            break;
        default:
            assert(false);
            break;
    }
}

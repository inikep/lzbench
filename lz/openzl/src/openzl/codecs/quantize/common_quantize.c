// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/quantize/common_quantize.h"

static uint8_t const offsetToCode[1] = { 0 };
static uint8_t const offsetBits[32]  = { 0,  1,  2,  3,  4,  5,  6,  7,
                                         8,  9,  10, 11, 12, 13, 14, 15,
                                         16, 17, 18, 19, 20, 21, 22, 23,
                                         24, 25, 26, 27, 28, 29, 30, 31 };
static uint32_t const offsetBase[32] = {
    0x1,        0x2,       0x4,       0x8,       0x10,       0x20,
    0x40,       0x80,      0x100,     0x200,     0x400,      0x800,
    0x1000,     0x2000,    0x4000,    0x8000,    0x10000,    0x20000,
    0x40000,    0x80000,   0x100000,  0x200000,  0x400000,   0x800000,
    0x1000000,  0x2000000, 0x4000000, 0x8000000, 0x10000000, 0x20000000,
    0x40000000, 0x80000000
};

ZL_Quantize32Params const ZL_quantizeOffsetsParams = {
    .nbCodes     = 32,
    .valueToCode = offsetToCode,
    .delta       = 0,
    .maxPow2     = 0,
    .bits        = offsetBits,
    .base        = offsetBase,
};

// TODO(terrelln): Tune these parameters better
static uint8_t const lengthToCode[16] = { 0, 1, 2,  3,  4,  5,  6,  7,
                                          8, 9, 10, 11, 12, 13, 14, 15 };
static uint8_t const lengthBits[44]   = { 0,  0,  0,  0,  0,  0,  0,  0,  0,
                                          0,  0,  0,  0,  0,  0,  0,  4,  5,
                                          6,  7,  8,  9,  10, 11, 12, 13, 14,
                                          15, 16, 17, 18, 19, 20, 21, 22, 23,
                                          24, 25, 26, 27, 28, 29, 30, 31 };
static uint32_t const lengthBase[44]  = {
    0,          1,         2,         3,         4,          5,
    6,          7,         8,         9,         10,         11,
    12,         13,        14,        15,        0x10,       0x20,
    0x40,       0x80,      0x100,     0x200,     0x400,      0x800,
    0x1000,     0x2000,    0x4000,    0x8000,    0x10000,    0x20000,
    0x40000,    0x80000,   0x100000,  0x200000,  0x400000,   0x800000,
    0x1000000,  0x2000000, 0x4000000, 0x8000000, 0x10000000, 0x20000000,
    0x40000000, 0x80000000
};

ZL_Quantize32Params const ZL_quantizeLengthsParams = {
    .nbCodes     = 44,
    .valueToCode = lengthToCode,
    .delta       = 16 - 4,
    .maxPow2     = 16,
    .bits        = lengthBits,
    .base        = lengthBase,
};

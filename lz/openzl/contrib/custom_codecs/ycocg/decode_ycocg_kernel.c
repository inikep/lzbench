// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <assert.h>

#include "decode_ycocg_kernel.h"

void YCOCG_decodePixel_RGB24(uint8_t* rgb24, uint8_t y, int16_t co, int16_t cg)
{
    int tmp  = y - (cg / 2);
    rgb24[1] = (uint8_t)(cg + tmp);
    rgb24[2] = (uint8_t)(tmp - (co / 2));
    rgb24[0] = (uint8_t)(rgb24[2] + co);
}

void YCOCG_decodeArray_RGB24(
        uint8_t* rgb24,
        const uint8_t* y,
        const int16_t* co,
        const int16_t* cg,
        size_t nbPixels)
{
    for (size_t n = 0; n < nbPixels; n++) {
        YCOCG_decodePixel_RGB24(rgb24 + 3 * n, y[n], co[n], cg[n]);
    }
}

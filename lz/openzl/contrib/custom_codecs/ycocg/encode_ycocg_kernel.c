// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <assert.h>

#include "encode_ycocg_kernel.h"

void YCOCG_encodePixel_RGB24(
        uint8_t* y,
        int16_t* co,
        int16_t* cg,
        const uint8_t* rgb24)
{
    int r   = rgb24[0];
    int g   = rgb24[1];
    int b   = rgb24[2];
    *co     = (int16_t)(r - b);
    int tmp = b + (*co / 2);
    *cg     = (int16_t)(g - tmp);
    *y      = (uint8_t)(tmp + (*cg / 2));
}

void YCOCG_encodeArray_RGB24(
        uint8_t* y,
        int16_t* co,
        int16_t* cg,
        const uint8_t* rgb24,
        size_t nbPixels)
{
    for (size_t n = 0; n < nbPixels; n++) {
        YCOCG_encodePixel_RGB24(y + n, co + n, cg + n, rgb24 + 3 * n);
    }
}

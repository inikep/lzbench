// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stddef.h> // size_t
#include <stdint.h> // uintX_t

/* Conditions:
 * Output @rgb24 must be already allocated, its capacity must be >=@nbPixel*3
 * Inputs @y, @co and @cg have @nbPixels cells.
 * note that @co and @cg are signed 16-bit types, while @y is 8-bit unsigned.
 */
void YCOCG_decodeArray_RGB24(
        uint8_t* rgb24,
        const uint8_t* y,
        const int16_t* co,
        const int16_t* cg,
        size_t nbPixels);

/* pixel conversion function
 * exposed to test lossless round trip */
void YCOCG_decodePixel_RGB24(uint8_t* rgb24, uint8_t y, int16_t co, int16_t cg);

// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stddef.h> // size_t
#include <stdint.h> // uintX_t

/* Conditions:
 * Input @rgb24 is using RGB 24-bit format, its size is @nbPixel*3
 * Outputs @y, @co and @cg are allocated, and have @nbPixels cells.
 * note that @co and @cg are signed 16-bit types
 */
void YCOCG_encodeArray_RGB24(
        uint8_t* y,
        int16_t* co,
        int16_t* cg,
        const uint8_t* rgb24,
        size_t nbPixels);

/* pixel conversion function
 * exposed to test lossless round trip */
void YCOCG_encodePixel_RGB24(
        uint8_t* y,
        int16_t* co,
        int16_t* cg,
        const uint8_t* rgb24);

// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>  // printf
#include <stdlib.h> // exit
#include "../decode_ycocg_kernel.h"
#include "../encode_ycocg_kernel.h"

void pixel_round_trip(uint8_t rgb24[3])
{
    uint8_t y;
    int16_t co, cg;
    YCOCG_encodePixel_RGB24(&y, &co, &cg, rgb24);
    uint8_t decRgb24[3];
    YCOCG_decodePixel_RGB24(decRgb24, y, co, cg);
    if ((decRgb24[0] == rgb24[0]) && (decRgb24[1] == rgb24[1])
        && (decRgb24[2] == rgb24[2]))
        return;
    printf("Error: decRgc24 [%u,%u,%u] != [%u,%u,%u] srcRgb24 \n",
           decRgb24[0],
           decRgb24[1],
           decRgb24[2],
           rgb24[0],
           rgb24[1],
           rgb24[2]);
    exit(1);
}

int main(void)
{
    printf("checking that all R/G/B values rountrip to/from Y/Co/Cg losslessly\n");
    for (int r = 0; r <= 255; r++) {
        for (int g = 0; g <= 255; g++) {
            for (int b = 0; b <= 255; b++) {
                uint8_t rgb24[3] = { (uint8_t)r, (uint8_t)g, (uint8_t)b };
                pixel_round_trip(rgb24);
            }
        }
    }
    printf("check completed: all colors round-trip losslessly!\n");
    return 0;
}

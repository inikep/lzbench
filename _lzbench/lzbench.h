#ifndef LZBENCH_H
#define LZBENCH_H

#include "compressors.h"

typedef struct
{
    const char* name;
    const char* version;
    int first_level;
    int last_level;
    compress_func compress;
    compress_func decompress;
} compressor_desc_t;

#define LZBENCH_COMPRESSOR_COUNT 9

static const compressor_desc_t comp_desc[LZBENCH_COMPRESSOR_COUNT] =
{
    { "memcpy",  "",            0, 0,  NULL,                    NULL },
    { "brotli",  "2015-10-29",  0, 11, lzbench_brotli_compress, lzbench_brotli_decompress },
    { "crush",   "1.0",         0, 1,  lzbench_crush_compress, lzbench_crush_decompress },
    { "csc",     "3.3",         1, 5,  lzbench_csc_compress, lzbench_csc_decompress },
    { "density", "0.12.5 beta", 1, 3,  lzbench_density_compress, lzbench_density_decompress },
    { "fastlz",  "0.1",         1, 2,  lzbench_fastlz_compress, lzbench_fastlz_decompress },
    { "lz4",     "r131",        0, 0,  lzbench_lz4_compress, lzbench_lz4_decompress },
    { "lz4fast", "r131",        1, 99, lzbench_lz4fast_compress, lzbench_lz4_decompress },
    { "lz4hc",   "r131",        1, 9,  lzbench_lz4hc_compress, lzbench_lz4_decompress }
};

#endif
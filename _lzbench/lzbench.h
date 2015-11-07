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

#define LZBENCH_COMPRESSOR_COUNT 26

static const compressor_desc_t comp_desc[LZBENCH_COMPRESSOR_COUNT] =
{
    { "memcpy",   "",            0, 0,  NULL,                    NULL },
    { "brotli",   "2015-10-29",  0, 11, lzbench_brotli_compress, lzbench_brotli_decompress },
    { "crush",    "1.0",         0, 1,  lzbench_crush_compress, lzbench_crush_decompress },
    { "csc",      "3.3",         1, 5,  lzbench_csc_compress, lzbench_csc_decompress },
    { "density",  "0.12.5 beta", 1, 3,  lzbench_density_compress, lzbench_density_decompress },
    { "fastlz",   "0.1",         1, 2,  lzbench_fastlz_compress, lzbench_fastlz_decompress },
    { "lz4",      "r131",        0, 0,  lzbench_lz4_compress, lzbench_lz4_decompress },
    { "lz4fast",  "r131",        1, 99, lzbench_lz4fast_compress, lzbench_lz4_decompress },
    { "lz4hc",    "r131",        1, 9,  lzbench_lz4hc_compress, lzbench_lz4_decompress },
    { "lz5",      "r131b",       0, 0,  lzbench_lz5_compress, lzbench_lz5_decompress },
    { "lz5hc",    "r131b",       1, 9,  lzbench_lz5hc_compress, lzbench_lz5_decompress },
    { "lzf",      "3.6",         0, 1,  lzbench_lzf_compress, lzbench_lzf_decompress },
    { "lzham",    "1.0 -d26",    0, 1,  lzbench_lzham_compress, lzbench_lzham_decompress },
    { "lzjb",     "2010",        0, 0,  lzbench_lzjb_compress, lzbench_lzjb_decompress },
    { "lzma",     "9.38",        0, 5,  lzbench_lzma_compress, lzbench_lzma_decompress },
    { "lzmat",    "1.01",        0, 0,  lzbench_lzmat_compress, lzbench_lzmat_decompress },
    { "pithy",    "2011-12-24",  0, 0,  lzbench_pithy_compress, lzbench_pithy_decompress },
    { "quicklz",  "1.5.0",       1, 3,  lzbench_quicklz_compress, lzbench_quicklz_decompress },
    { "shrinker", "0.1",         0, 0,  lzbench_shrinker_compress, lzbench_shrinker_decompress },
    { "snappy",   "1.1.3",       0, 0,  lzbench_snappy_compress, lzbench_snappy_decompress },
    { "wflz",     "2015-09-16",  0, 0,  lzbench_wflz_compress, lzbench_wflz_decompress },
    { "yappy",    "2014-03-22",  0, 99, lzbench_yappy_compress, lzbench_yappy_decompress },
    { "zlib",     "1.2.8",       1, 9,  lzbench_zlib_compress, lzbench_zlib_decompress },
    { "zling",    "2015-09-16",  0, 4,  lzbench_zling_compress, lzbench_zling_decompress },
    { "zstd",     "v0.3.4",      0, 0,  lzbench_zstd_compress, lzbench_zstd_decompress },
    { "zstd_HC",  "v0.3.4",      1, 21, lzbench_zstdhc_compress, lzbench_zstd_decompress },
};

#endif
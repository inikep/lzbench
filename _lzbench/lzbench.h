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

#define LZBENCH_COMPRESSOR_COUNT 32

static const compressor_desc_t comp_desc[LZBENCH_COMPRESSOR_COUNT] =
{
    { "memcpy",   "",            0, 0,  NULL,                      NULL },
    { "brieflz",  "1.1.0",       0, 0,  lzbench_brieflz_compress,  lzbench_brieflz_decompress },
    { "brotli",   "2015-10-29",  0, 11, lzbench_brotli_compress,   lzbench_brotli_decompress },
    { "crush",    "1.0",         0, 1,  lzbench_crush_compress,    lzbench_crush_decompress },
    { "csc",      "3.3",         1, 5,  lzbench_csc_compress,      lzbench_csc_decompress },
    { "density",  "0.12.5 beta", 1, 3,  lzbench_density_compress,  lzbench_density_decompress },
    { "fastlz",   "0.1",         1, 2,  lzbench_fastlz_compress,   lzbench_fastlz_decompress },
    { "lz4",      "r131",        0, 0,  lzbench_lz4_compress,      lzbench_lz4_decompress },
    { "lz4fast",  "r131",        1, 99, lzbench_lz4fast_compress,  lzbench_lz4_decompress },
    { "lz4hc",    "r131",        1, 9,  lzbench_lz4hc_compress,    lzbench_lz4_decompress },
    { "lz5",      "r131b",       0, 0,  lzbench_lz5_compress,      lzbench_lz5_decompress },
    { "lz5hc",    "r131b",       1, 9,  lzbench_lz5hc_compress,    lzbench_lz5_decompress },
    { "lzf",      "3.6",         0, 1,  lzbench_lzf_compress,      lzbench_lzf_decompress },
    { "lzham",    "1.0 -d26",    0, 4,  lzbench_lzham_compress,    lzbench_lzham_decompress },
    { "lzjb",     "2010",        0, 0,  lzbench_lzjb_compress,     lzbench_lzjb_decompress },
    { "lzlib",    "1.7",         0, 9,  lzbench_lzlib_compress,    lzbench_lzlib_decompress },
    { "lzma",     "9.38",        0, 9,  lzbench_lzma_compress,     lzbench_lzma_decompress },
    { "lzmat",    "1.01",        0, 0,  lzbench_lzmat_compress,    lzbench_lzmat_decompress },
    { "lzo",      "2.09",        1, 1,  lzbench_lzo_compress,      lzbench_lzo_decompress },
    { "lzrw",     "15-Jul-1991", 1, 4,  lzbench_lzrw_compress,     lzbench_lzrw_decompress },
    { "pithy",    "2011-12-24",  0, 9,  lzbench_pithy_compress,    lzbench_pithy_decompress },
    { "quicklz",  "1.5.0",       1, 3,  lzbench_quicklz_compress,  lzbench_quicklz_decompress },
    { "shrinker", "0.1",         0, 0,  lzbench_shrinker_compress, lzbench_shrinker_decompress },
    { "snappy",   "1.1.3",       0, 0,  lzbench_snappy_compress,   lzbench_snappy_decompress },
    { "tornado",  "0.6a",        1, 16, lzbench_tornado_compress,  lzbench_tornado_decompress },
    { "ucl",      "1.03",       11, 39, lzbench_ucl_compress,      lzbench_ucl_decompress },
    { "wflz",     "2015-09-16",  0, 0,  lzbench_wflz_compress,     lzbench_wflz_decompress },
    { "yappy",    "2014-03-22",  0, 99, lzbench_yappy_compress,    lzbench_yappy_decompress },
    { "zlib",     "1.2.8",       1, 9,  lzbench_zlib_compress,     lzbench_zlib_decompress },
    { "zling",    "2015-09-16",  0, 4,  lzbench_zling_compress,    lzbench_zling_decompress },
    { "zstd",     "v0.3.6",      0, 0,  lzbench_zstd_compress,     lzbench_zstd_decompress },
    { "zstd_HC",  "v0.3.6",      1, 20, lzbench_zstdhc_compress,   lzbench_zstd_decompress },
};

char fast[] = "";
char compr_all[] = "brieflz/brotli,0,2,5,8,11/crush,0,1/csc,1,2,3,4,5/density,1,2,3/fastlz,1,2/lz4/lz4fast,3,17/lz4hc,1,4,9/lz5/lz5hc,1,4,9/" \
              "lzf,0,1/lzham,0,1/lzjb/lzlib,0,1,2,3,4,5,6,7,8,9/lzma,0,1,2,3,4,5/lzmat/lzo,1,9,99,999,1001,1009,1099,1999,2001,2999,3001,3999,4001,4999,5999,6999/" \
              "lzrw,1,2,3,4,5/pithy,0,3,6,9/quicklz,1,2,3/shrinker/snappy/tornado,1,2,3,4,5,6,7,10,13,16/ucl,11,16,19,21,26,29,31,36,39/" \
              "wflz/yappy,1,10,100/zlib,1,6,9/zling,0,1,2,3,4/zstd/zstd_HC,1,5,9,13,17,20";
char compr_fast[] = "density,1,2,3/fastlz,1,2/lz4/lz4fast,3,17/lz5/" \
              "lzf,0,1/lzjb/lzo,1,1001,2001,3001,4001/" \
              "lzrw,1,2,3,4,5/pithy,0,3,6,9/quicklz,1,2/shrinker/snappy/tornado,1,2,3/" \
              "wflz/zstd";
char compr_opt[] = "brotli,6,7,8,9,10,11/csc,1,2,3,4,5/" \
              "lzham,0,1,2,3,4/lzlib,0,1,2,3,4,5,6,7,8,9/lzma,0,1,2,3,4,5,6,7/" \
              "tornado,5,6,7,8,9,10,11,12,13,14,15,16/" \
              "zstd_HC,10,11,12,13,14,15,16,17,18,19,20";
#endif

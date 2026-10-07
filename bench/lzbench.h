/*
 * Copyright (c) Przemyslaw Skibinski <inikep@gmail.com>
 * All rights reserved.
 *
 * This source code is dual-licensed under the GPLv2 and GPLv3 licenses.
 * For additional details, refer to the LICENSE file located in the root
 * directory of this source tree.
 */

#ifndef LZBENCH_H
#define LZBENCH_H

#define _CRT_SECURE_NO_WARNINGS
#define _FILE_OFFSET_BITS 64  // turn off_t into a 64-bit type for ftello() and fseeko()

#include <vector>
#include <string>
#include <string.h> // strcmp
#include "codecs.h"


#define PROGNAME "lzbench"
#define PROGVERSION "2.4.1"
#define PAD_SIZE (1024)
#define MIN_PAGE_SIZE 4096  // smallest page size we expect, if it's wrong the first algorithm might be a bit slower
#define DEFAULT_LOOP_TIME (100*1000000)  // 1/10 of a second
#define GET_COMPRESS_BOUND(insize) (insize + insize/8 + PAD_SIZE) // for brieflz and ucl_nrv2b with "-b64"
#define LZBENCH_PRINT(level, fmt, ...) if (params->verbose >= level) printf(fmt, __VA_ARGS__)
#define LZBENCH_STDERR(level, fmt, ...) if (params->verbose >= level) { fprintf(stderr, fmt, __VA_ARGS__); fflush(stderr); }

#define MAX(a,b) (((a)>(b))?(a):(b))
#ifndef MIN
    #define MIN(a,b) ((a)<(b)?(a):(b))
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(WIN64) || defined(_WIN64)
    #define WINDOWS
#endif

/* **************************************
*  Compiler Options
****************************************/
#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS    /* Disable some Visual warning messages for fopen, strncpy */
#  define _CRT_SECURE_NO_DEPRECATE   /* VS2005 */
#if _MSC_VER <= 1800                 /* (1800 = Visual Studio 2013) */
#define snprintf sprintf_s       /* snprintf unsupported by Visual <= 2013 */
#endif
#endif

#ifdef WINDOWS
    #include <windows.h>
    typedef LARGE_INTEGER bench_rate_t;
    typedef LARGE_INTEGER bench_timer_t;
    #define InitTimer(rate) if (!QueryPerformanceFrequency(&rate)) { printf("QueryPerformance not present"); };
    #define GetTime(now) QueryPerformanceCounter(&now);
    #define GetDiffTime(rate, start_ticks, end_ticks) (1000000000ULL*(end_ticks.QuadPart - start_ticks.QuadPart)/rate.QuadPart)
    #ifndef fseeko
        #ifdef _fseeki64
            #define fseeko _fseeki64
            #define ftello _ftelli64
        #else
            #define fseeko fseek
            #define ftello ftell
        #endif
    #endif
    #define PROGOS "Windows"
#else
    #include <stdarg.h> // va_args
    #include <time.h>
    #include <unistd.h>
    #include <sys/resource.h>
#if defined(__APPLE__) || defined(__MACH__)
    #include <mach/mach_time.h>
    typedef mach_timebase_info_data_t bench_rate_t;
    typedef uint64_t bench_timer_t;
    #define InitTimer(rate) mach_timebase_info(&rate);
    #define GetTime(now) now = mach_absolute_time();
    #define GetDiffTime(rate, start_ticks, end_ticks) ((end_ticks - start_ticks) * (uint64_t)rate.numer) / ((uint64_t)rate.denom)
    #define PROGOS "MacOS"
#else
    typedef struct timespec bench_rate_t;
    typedef struct timespec bench_timer_t;
    #define InitTimer(rate)
    #define GetTime(now) if (clock_gettime(CLOCK_MONOTONIC, &now) == -1 ){ printf("clock_gettime error"); };
    #define GetDiffTime(rate, start_ticks, end_ticks) (1000000000ULL*( end_ticks.tv_sec - start_ticks.tv_sec ) + ( end_ticks.tv_nsec - start_ticks.tv_nsec ))
    #define PROGOS "Linux"
#endif
#endif

typedef unsigned long long uint64;
typedef long long          int64;

typedef struct string_table
{
    std::string col1_algname;
    uint64_t col2_ctime, col3_dtime, col4_comprsize, col5_origsize;
    std::string col6_filename;
    int usedCompThreads, usedDecompThreads, usedCodecThreads;
    string_table(std::string c1, uint64_t c2, uint64_t c3, uint64_t c4, uint64_t c5, std::string filename, int compThreads, int decompThreads, int codecThreads) :
        col1_algname(c1), col2_ctime(c2), col3_dtime(c3), col4_comprsize(c4), col5_origsize(c5), col6_filename(filename), usedCompThreads(compThreads), usedDecompThreads(decompThreads), usedCodecThreads(codecThreads) {}
} string_table_t;

enum textformat_e { MARKDOWN=1, TEXT, TEXT_FULL, CSV, TURBOBENCH, MARKDOWN2 };
enum timetype_e { FASTEST=1, AVERAGE, MEDIAN };

typedef struct
{
    int show_speed, compress_only;
    int threads, codec_threads;
    timetype_e timetype;
    textformat_e textformat;
    size_t chunk_size;
    uint32_t c_iters, d_iters, cspeed, verbose, cmintime, dmintime, cloop_time, dloop_time;
    size_t mem_limit;
    int random_read;
    std::vector<string_table_t> results;
    const char* in_filename;
} lzbench_params_t;

struct less_using_1st_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.col1_algname < struct2.col1_algname); } };
struct less_using_2nd_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.col2_ctime > struct2.col2_ctime); } };
struct less_using_3rd_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.col3_dtime > struct2.col3_dtime); } };
struct less_using_4th_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.col4_comprsize < struct2.col4_comprsize); } };
struct less_using_5th_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.col5_origsize < struct2.col5_origsize); } };

typedef int64_t (*compress_func)(char *in, size_t insize, char *out, size_t outsize, codec_options_t *codec_options);
typedef char* (*init_func)(size_t insize, size_t, size_t);
typedef void (*deinit_func)(char* workmem);

typedef enum {
    NO_THREADING   = 0,                           // Single-threaded only
    INTERNAL_MT    = 1 << 0,                      // Supports internal (built-in) multithreading
    BENCH_POOL_MT  = 1 << 1,                      // Supports lzbench's external thread pool
    FULL_THREADING = INTERNAL_MT | BENCH_POOL_MT  // Supports both modes
} threading_mode_t;

typedef struct
{
    const char* name;
    const char* name_version;
    const char* algorithm;   // modelling stage [+ entropy coder]; some codecs differ by level, see algorithm_by_level
    int first_level;
    int last_level;
    int additional_param;
    threading_mode_t mt_mode;
    compress_func compress;
    compress_func decompress;
    init_func init;
    deinit_func deinit;
    size_t max_input_size;
} compressor_desc_t;


typedef struct
{
    const char* name;
    const char* description;
    const char* params;
} alias_desc_t;

#if defined(_OPENMP)
    #define BSC_THREADING FULL_THREADING
#else
    #define BSC_THREADING BENCH_POOL_MT
#endif

static const compressor_desc_t comp_desc[] =
{
     //                                                                      last_level,       mt_mode,
     // name,       name_version,    algorithm,                     first_level,  additional_param,  compress_func,               decompress_func,               init_func,               deinit_func,             max_input_size
    { "memcpy",     "memcpy",                  "copy",                        0,   0,    0,  BENCH_POOL_MT, lzbench_memcpy,              lzbench_memcpy,                NULL,                    NULL },
    { "aceapex",    "aceapex 2.2.2",           "LZ77 + FSE/Huffman",          1,   3,    0, FULL_THREADING, lzbench_aceapex_compress,    lzbench_aceapex_decompress,    lzbench_aceapex_init,    lzbench_aceapex_deinit },
    { "aceapex_cuda","aceapex_cuda 0.9",       "LZ77 + FSE/Huffman",          1,   2,    0,  NO_THREADING,  lzbench_aceapex_compress,    lzbench_aceapex_cuda_decompress, lzbench_aceapex_cuda_init, lzbench_aceapex_cuda_deinit },
    { "brieflz",    "brieflz 1.3.0",           "LZ77",                        1,   9,    0,  BENCH_POOL_MT, lzbench_brieflz_compress,    lzbench_brieflz_decompress,    lzbench_brieflz_init,    lzbench_brieflz_deinit },
    { "brotli",     "brotli 1.2.0",            "LZ77 + Huffman",              0,  11,    0,  BENCH_POOL_MT, lzbench_brotli_compress,     lzbench_brotli_decompress,     NULL,                    NULL },
    { "brotli22",   "brotli 1.2.0 -d22",       "LZ77 + Huffman",              0,  11,   22,  BENCH_POOL_MT, lzbench_brotli_compress,     lzbench_brotli_decompress,     NULL,                    NULL },
    { "brotli24",   "brotli 1.2.0 -d24",       "LZ77 + Huffman",              0,  11,   24,  BENCH_POOL_MT, lzbench_brotli_compress,     lzbench_brotli_decompress,     NULL,                    NULL },
    { "bsc0",       "bsc 3.3.12 -m0 -e2",      "BWT + QLFC",                  0,   0,    0,  BSC_THREADING, lzbench_bsc_compress,        lzbench_bsc_decompress,        lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bsc1",       "bsc 3.3.12 -m0 -e1",      "BWT + QLFC",                  0,   0,    1,  BSC_THREADING, lzbench_bsc_compress,        lzbench_bsc_decompress,        lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bsc2",       "bsc 3.3.12 -m0 -e0",      "BWT + QLFC",                  0,   0,    2,  BSC_THREADING, lzbench_bsc_compress,        lzbench_bsc_decompress,        lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bsc3",       "bsc 3.3.12 -m3 -e1",      "ST + QLFC",                   0,   0,    3,  BSC_THREADING, lzbench_bsc_compress,        lzbench_bsc_decompress,        lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bsc4",       "bsc 3.3.12 -m4 -e1",      "ST + QLFC",                   0,   0,    4,  BSC_THREADING, lzbench_bsc_compress,        lzbench_bsc_decompress,        lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bsc5",       "bsc 3.3.12 -m5 -e1",      "ST + QLFC",                   0,   0,    5,  BSC_THREADING, lzbench_bsc_compress,        lzbench_bsc_decompress,        lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bsc6",       "bsc 3.3.12 -m6 -e1",      "ST + QLFC",                   0,   0,    6,  BSC_THREADING, lzbench_bsc_compress,        lzbench_bsc_decompress,        lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bsc_cuda0",  "bsc 3.3.12 -G -m0 -e2",   "BWT + QLFC",                  0,   0,    0,  BENCH_POOL_MT, lzbench_bsc_cuda_compress,   lzbench_bsc_cuda_decompress,   lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bsc_cuda1",  "bsc 3.3.12 -G -m0 -e1",   "BWT + QLFC",                  0,   0,    1,  BENCH_POOL_MT, lzbench_bsc_cuda_compress,   lzbench_bsc_cuda_decompress,   lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bsc_cuda2",  "bsc 3.3.12 -G -m0 -e0",   "BWT + QLFC",                  0,   0,    2,  BENCH_POOL_MT, lzbench_bsc_cuda_compress,   lzbench_bsc_cuda_decompress,   lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bsc_cuda3",  "bsc 3.3.12 -G -m3 -e1",   "ST + QLFC",                   0,   0,    3,  BENCH_POOL_MT, lzbench_bsc_cuda_compress,   lzbench_bsc_cuda_decompress,   lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bsc_cuda4",  "bsc 3.3.12 -G -m4 -e1",   "ST + QLFC",                   0,   0,    4,  BENCH_POOL_MT, lzbench_bsc_cuda_compress,   lzbench_bsc_cuda_decompress,   lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bsc_cuda5",  "bsc 3.3.12 -G -m5 -e1",   "ST + QLFC",                   0,   0,    5,  BENCH_POOL_MT, lzbench_bsc_cuda_compress,   lzbench_bsc_cuda_decompress,   lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bsc_cuda6",  "bsc 3.3.12 -G -m6 -e1",   "ST + QLFC",                   0,   0,    6,  BENCH_POOL_MT, lzbench_bsc_cuda_compress,   lzbench_bsc_cuda_decompress,   lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bsc_cuda7",  "bsc 3.3.12 -G -m7 -e0",   "ST + QLFC",                   0,   0,    7,  BENCH_POOL_MT, lzbench_bsc_cuda_compress,   lzbench_bsc_cuda_decompress,   lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bsc_cuda8",  "bsc 3.3.12 -G -m8 -e0",   "ST + QLFC",                   0,   0,    8,  BENCH_POOL_MT, lzbench_bsc_cuda_compress,   lzbench_bsc_cuda_decompress,   lzbench_bsc_init,        NULL,                    LZBENCH_BSC_MAX_INPUT_SIZE },
    { "bzip2",      "bzip2 1.0.8",             "BWT + Huffman",               1,   9,    0,  BENCH_POOL_MT, lzbench_bzip2_compress,      lzbench_bzip2_decompress,      NULL,                    NULL },
    { "bzip3",      "bzip3 1.5.4",             "BWT + CM",                    1,  10,    0,  BENCH_POOL_MT, lzbench_bzip3_compress,      lzbench_bzip3_decompress,      NULL,                    NULL },
    { "crush",      "crush 1.0",               "LZ77",                        0,   2,    0,   NO_THREADING, lzbench_crush_compress,      lzbench_crush_decompress,      NULL,                    NULL },
    { "csc",        "csc 2016-10-13",          "LZ77 + range",                1,   5,    0,  BENCH_POOL_MT, lzbench_csc_compress,        lzbench_csc_decompress,        NULL,                    NULL },
    { "cudaMemcpy", "cudaMemcpy",              "copy",                        0,   0,    0,  BENCH_POOL_MT, lzbench_cuda_memcpy,         lzbench_cuda_memcpy,           lzbench_cuda_init,       lzbench_cuda_deinit },
    { "density",    "density 0.16.6",          "dictionary",                  1,   3,    0,  BENCH_POOL_MT, lzbench_density_compress,    lzbench_density_decompress,    lzbench_density_init,    lzbench_density_deinit },
    { "fastlz",     "fastlz 0.5.0",            "LZ77",                        1,   2,    0,  BENCH_POOL_MT, lzbench_fastlz_compress,     lzbench_fastlz_decompress,     NULL,                    NULL },
    { "fastlzma2",  "fastlzma2 1.0.1",         "LZ77 + range",                1,  10,    0, FULL_THREADING, lzbench_fastlzma2_compress,  lzbench_fastlzma2_decompress,  NULL,                    NULL },
    { "gipfeli",    "gipfeli 2016-07-13",      "LZ77 + prefix code",          0,   0,    0,  BENCH_POOL_MT, lzbench_gipfeli_compress,    lzbench_gipfeli_decompress,    NULL,                    NULL },
    { "glza",       "glza 0.12",               "grammar + range",             0,   0,    0,   NO_THREADING, lzbench_glza_compress,       lzbench_glza_decompress,       NULL,                    NULL },
#if !defined(BENCH_REMOVE_GPUCOMPACT) && defined(BENCH_HAS_CUDA)
    { "gpucompact", "gpucompact 1.1",          "BWT + tANS",                  1,   5,    0,  NO_THREADING,  lzbench_gpucompact_compress, lzbench_gpucompact_decompress, lzbench_gpucompact_init,  lzbench_gpucompact_deinit },
#endif
    { "kanzi",      "kanzi 2.6.0",             "LZ77, ROLZ, BWT or CM",       1,   9,    0, FULL_THREADING, lzbench_kanzi_compress,      lzbench_kanzi_decompress,      NULL,                    NULL },
    { "lbzip2",     "lbzip2 2.6.5",            "BWT + Huffman",               1,   9,    0,  BENCH_POOL_MT, lzbench_lbzip2_compress,     lzbench_lbzip2_decompress,     NULL,                    NULL },
    { "libdeflate", "libdeflate 1.26",         "LZ77 + Huffman",              1,  12,    0,  BENCH_POOL_MT, lzbench_libdeflate_compress, lzbench_libdeflate_decompress, NULL,                    NULL },
    { "lizard",     "lizard 2.1",             "LZ77 (+ Huffman)",            10,  49,    0,  BENCH_POOL_MT, lzbench_lizard_compress,     lzbench_lizard_decompress,     NULL,                    NULL },
    { "lz4",        "lz4 1.10.0",              "LZ77",                        0,   0,    0,  BENCH_POOL_MT, lzbench_lz4_compress,        lzbench_lz4_decompress,        NULL,                    NULL },
    { "lz4fast",    "lz4 1.10.0 --fast",       "LZ77",                        1,  99,    0,  BENCH_POOL_MT, lzbench_lz4fast_compress,    lzbench_lz4_decompress,        NULL,                    NULL },
    { "lz4hc",      "lz4hc 1.10.0",            "LZ77",                        1,  12,    0,  BENCH_POOL_MT, lzbench_lz4hc_compress,      lzbench_lz4_decompress,        NULL,                    NULL },
    { "lzav",       "lzav 5.18",               "LZ77",                        1,   2,    0,  BENCH_POOL_MT, lzbench_lzav_compress,       lzbench_lzav_decompress,       NULL,                    NULL },
    { "lzf",        "lzf 3.6",                 "LZ77",                        0,   1,    0,  BENCH_POOL_MT, lzbench_lzf_compress,        lzbench_lzf_decompress,        NULL,                    NULL },
    { "lzfse",      "lzfse 2017-03-08",        "LZ77 + FSE",                  0,   0,    0,  BENCH_POOL_MT, lzbench_lzfse_compress,      lzbench_lzfse_decompress,      lzbench_lzfse_init,      lzbench_lzfse_deinit },
    { "lzg",        "lzg 1.0.10",              "LZ77",                        1,   9,    0,  BENCH_POOL_MT, lzbench_lzg_compress,        lzbench_lzg_decompress,        NULL,                    NULL },
    { "lzham",      "lzham 1.0 -d26",          "LZ77 + Huffman/arithmetic",   0,   4,    0, FULL_THREADING, lzbench_lzham_compress,      lzbench_lzham_decompress,      NULL,                    NULL },
    { "lzham22",    "lzham 1.0 -d22",          "LZ77 + Huffman/arithmetic",   0,   4,   22, FULL_THREADING, lzbench_lzham_compress,      lzbench_lzham_decompress,      NULL,                    NULL },
    { "lzham24",    "lzham 1.0 -d24",          "LZ77 + Huffman/arithmetic",   0,   4,   24, FULL_THREADING, lzbench_lzham_compress,      lzbench_lzham_decompress,      NULL,                    NULL },
    { "lzjb",       "lzjb 2010",               "LZ77",                        0,   0,    0,  BENCH_POOL_MT, lzbench_lzjb_compress,       lzbench_lzjb_decompress,       NULL,                    NULL },
    { "lzlib",      "lzlib 1.16",              "LZ77 + range",                0,   9,    0,  BENCH_POOL_MT, lzbench_lzlib_compress,      lzbench_lzlib_decompress,      NULL,                    NULL },
    { "lzma",       "lzma 26.03",              "LZ77 + range",                0,   9,    0, FULL_THREADING, lzbench_lzma_compress,       lzbench_lzma_decompress,       NULL,                    NULL },
    { "lzmat",      "lzmat 1.01",              "LZ77",                        0,   0,    0,  BENCH_POOL_MT, lzbench_lzmat_compress,      lzbench_lzmat_decompress,      NULL,                    NULL }, // decompression error (returns 0) and SEGFAULT (?)
    { "lzo1",       "lzo1 2.10",               "LZ77",                        1,  99,    0,  BENCH_POOL_MT, lzbench_lzo1_compress,       lzbench_lzo1_decompress,       lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo1a",      "lzo1a 2.10",              "LZ77",                        1,  99,    0,  BENCH_POOL_MT, lzbench_lzo1a_compress,      lzbench_lzo1a_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo1b",      "lzo1b 2.10",              "LZ77",                        1, 999,    0,  BENCH_POOL_MT, lzbench_lzo1b_compress,      lzbench_lzo1b_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo1c",      "lzo1c 2.10",              "LZ77",                        1, 999,    0,  BENCH_POOL_MT, lzbench_lzo1c_compress,      lzbench_lzo1c_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo1f",      "lzo1f 2.10",              "LZ77",                        1, 999,    0,  BENCH_POOL_MT, lzbench_lzo1f_compress,      lzbench_lzo1f_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo1x",      "lzo1x 2.10",              "LZ77",                        1, 999,    0,  BENCH_POOL_MT, lzbench_lzo1x_compress,      lzbench_lzo1x_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo1y",      "lzo1y 2.10",              "LZ77",                        1, 999,    0,  BENCH_POOL_MT, lzbench_lzo1y_compress,      lzbench_lzo1y_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo1z",      "lzo1z 2.10",            "LZ77",                        999, 999,    0,  BENCH_POOL_MT, lzbench_lzo1z_compress,      lzbench_lzo1z_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo2a",      "lzo2a 2.10",            "LZ77",                        999, 999,    0,  BENCH_POOL_MT, lzbench_lzo2a_compress,      lzbench_lzo2a_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzrw",       "lzrw 15-Jul-1991",        "LZ77",                        1,   5,    0,  BENCH_POOL_MT, lzbench_lzrw_compress,       lzbench_lzrw_decompress,       lzbench_lzrw_init,       lzbench_lzrw_deinit },
    { "lzsse2",     "lzsse2 2019-04-18",       "LZ77",                        0,  17,    0,  BENCH_POOL_MT, lzbench_lzsse2_compress,     lzbench_lzsse2_decompress,     lzbench_lzsse2_init,     lzbench_lzsse2_deinit },
    { "lzsse4",     "lzsse4 2019-04-18",       "LZ77",                        0,  17,    0,  BENCH_POOL_MT, lzbench_lzsse4_compress,     lzbench_lzsse4_decompress,     lzbench_lzsse4_init,     lzbench_lzsse4_deinit },
    { "lzsse4fast", "lzsse4fast 2019-04-18",   "LZ77",                        0,   0,    0,  BENCH_POOL_MT, lzbench_lzsse4fast_compress, lzbench_lzsse4_decompress,     lzbench_lzsse4fast_init, lzbench_lzsse4fast_deinit },
    { "lzsse8",     "lzsse8 2019-04-18",       "LZ77",                        0,  17,    0,  BENCH_POOL_MT, lzbench_lzsse8_compress,     lzbench_lzsse8_decompress,     lzbench_lzsse8_init,     lzbench_lzsse8_deinit },
    { "lzsse8fast", "lzsse8fast 2019-04-18",   "LZ77",                        0,   0,    0,  BENCH_POOL_MT, lzbench_lzsse8fast_compress, lzbench_lzsse8_decompress,     lzbench_lzsse8fast_init, lzbench_lzsse8fast_deinit },
    { "lzvn",       "lzvn 2017-03-08",         "LZ77",                        0,   0,    0,  BENCH_POOL_MT, lzbench_lzvn_compress,       lzbench_lzvn_decompress,       lzbench_lzvn_init,       lzbench_lzvn_deinit },
    { "mbrotli",    "mbrotli 0.5.2",           "LZ77 + Huffman",              0,  11,    0,  BENCH_POOL_MT, lzbench_mbrotli_compress,    lzbench_mbrotli_decompress,    NULL,                    NULL },
    { "mbrotli22",  "mbrotli 0.5.2 -d22",      "LZ77 + Huffman",              0,  11,   22,  BENCH_POOL_MT, lzbench_mbrotli_compress,    lzbench_mbrotli_decompress,    NULL,                    NULL },
    { "mbrotli24",  "mbrotli 0.5.2 -d24",      "LZ77 + Huffman",              0,  11,   24,  BENCH_POOL_MT, lzbench_mbrotli_compress,    lzbench_mbrotli_decompress,    NULL,                    NULL },
    { "memlz",      "memlz 0.5 beta",          "LZP",                         0,   0,    0,  BENCH_POOL_MT, lzbench_memlz_compress,      lzbench_memlz_decompress,      lzbench_memlz_init,      lzbench_memlz_deinit },
    { "misa77",      "misa77 0.6.0",          "LZ77",                        -1,   4,    0,  BENCH_POOL_MT, lzbench_misa77_compress,     lzbench_misa77_decompress,      NULL, NULL },
    { "misa77_safe", "misa77 0.6.0 safe",     "LZ77",                        -1,   3,    0,  BENCH_POOL_MT, lzbench_misa77_compress,     lzbench_misa77_safe_decompress, NULL, NULL },
    { "nvcomp_lz4", "nvcomp_lz4 2.2.0",        "LZ77",                        0,   7,    0,  BENCH_POOL_MT, lzbench_nvcomp_compress,     lzbench_nvcomp_decompress,     lzbench_nvcomp_init,     lzbench_nvcomp_deinit },
    { "openzl_u8",      "openzl 0.2.3 -p u8",      "field LZ + FSE/Huffman",      0,   0,    0,  BENCH_POOL_MT, lzbench_openzl_compress,     lzbench_openzl_decompress,     lzbench_openzl_init_integer(uint8_t),  lzbench_openzl_deinit },
    { "openzl_i8",      "openzl 0.2.3 -p i8",      "field LZ + FSE/Huffman",      0,   0,    0,  BENCH_POOL_MT, lzbench_openzl_compress,     lzbench_openzl_decompress,     lzbench_openzl_init_integer(int8_t),   lzbench_openzl_deinit },
    { "openzl_le_u16",  "openzl 0.2.3 -p le-u16",  "field LZ + FSE/Huffman",      0,   0,    0,  BENCH_POOL_MT, lzbench_openzl_compress,     lzbench_openzl_decompress,     lzbench_openzl_init_integer(uint16_t), lzbench_openzl_deinit },
    { "openzl_le_i16",  "openzl 0.2.3 -p le-i16",  "field LZ + FSE/Huffman",      0,   0,    0,  BENCH_POOL_MT, lzbench_openzl_compress,     lzbench_openzl_decompress,     lzbench_openzl_init_integer(int16_t),  lzbench_openzl_deinit },
    { "openzl_le_u32",  "openzl 0.2.3 -p le-u32",  "field LZ + FSE/Huffman",      0,   0,    0,  BENCH_POOL_MT, lzbench_openzl_compress,     lzbench_openzl_decompress,     lzbench_openzl_init_integer(uint32_t), lzbench_openzl_deinit },
    { "openzl_le_i32",  "openzl 0.2.3 -p le-i32",  "field LZ + FSE/Huffman",      0,   0,    0,  BENCH_POOL_MT, lzbench_openzl_compress,     lzbench_openzl_decompress,     lzbench_openzl_init_integer(int32_t),  lzbench_openzl_deinit },
    { "openzl_le_u64",  "openzl 0.2.3 -p le-u64",  "field LZ + FSE/Huffman",      0,   0,    0,  BENCH_POOL_MT, lzbench_openzl_compress,     lzbench_openzl_decompress,     lzbench_openzl_init_integer(uint64_t), lzbench_openzl_deinit },
    { "openzl_le_i64",  "openzl 0.2.3 -p le-i64",  "field LZ + FSE/Huffman",      0,   0,    0,  BENCH_POOL_MT, lzbench_openzl_compress,     lzbench_openzl_decompress,     lzbench_openzl_init_integer(int64_t),  lzbench_openzl_deinit },
    { "openzl_serial",  "openzl 0.2.3 -p serial",  "LZ77 + FSE/Huffman",          0,   0,    0,  BENCH_POOL_MT, lzbench_openzl_compress,     lzbench_openzl_decompress,     lzbench_openzl_init_serial,            lzbench_openzl_deinit },
    { "openzl_generic", "openzl 0.2.3 'generic'",  "LZ77 + FSE/Huffman",          0,   0,    0,  BENCH_POOL_MT, lzbench_openzl_compress,     lzbench_openzl_decompress,     lzbench_openzl_init_generic,           lzbench_openzl_deinit },
    { "openzl_zstd",    "openzl 0.2.3 'zstd'",   "LZ77 + FSE/Huffman",          -99,  22,    0,  BENCH_POOL_MT, lzbench_openzl_compress,     lzbench_openzl_decompress,     lzbench_openzl_init_zstd,              lzbench_openzl_deinit },
    { "openzl_lz4",     "openzl 0.2.3 'lz4'",    "LZ77",                        -99,  12,    0,  BENCH_POOL_MT, lzbench_openzl_compress,     lzbench_openzl_decompress,     lzbench_openzl_init_lz4,               lzbench_openzl_deinit },
    { "ppmd8",      "ppmd8 26.03",             "PPM + range",                 1,   9,    0,  BENCH_POOL_MT, lzbench_ppmd_compress,       lzbench_ppmd_decompress,       NULL,                    NULL },
    { "pulsar",     "pulsar 2.5.0",            "BWT + ANS",                   0,   0,    0,  BENCH_POOL_MT, lzbench_pulsar_compress,     lzbench_pulsar_decompress,     NULL,                    NULL },
    { "quicklz",    "quicklz 1.5.1 beta 7",    "LZ77",                        1,   3,    0,  BENCH_POOL_MT, lzbench_quicklz_compress,    lzbench_quicklz_decompress,    NULL,                    NULL },
    { "skim",       "skim 0.1.0",              "dictionary",                  0,   0,    0,  BENCH_POOL_MT, lzbench_skim_compress,       lzbench_skim_decompress,       lzbench_skim_init,       lzbench_skim_deinit },
    { "slz_deflate","slz_deflate 1.3.1",       "LZ77 + Huffman",              1,   3,    2,  BENCH_POOL_MT, lzbench_slz_compress,        lzbench_slz_decompress,        NULL,                    NULL },
    { "slz_gzip",   "slz_gzip 1.3.1",          "LZ77 + Huffman",              1,   3,    1,  BENCH_POOL_MT, lzbench_slz_compress,        lzbench_slz_decompress,        NULL,                    NULL },
    { "slz_zlib",   "slz_zlib 1.3.1",          "LZ77 + Huffman",              1,   3,    0,  BENCH_POOL_MT, lzbench_slz_compress,        lzbench_slz_decompress,        NULL,                    NULL },
    { "snappy",     "snappy 1.3.1",            "LZ77",                        0,   0,    0,  BENCH_POOL_MT, lzbench_snappy_compress,     lzbench_snappy_decompress,     NULL,                    NULL },
    { "tamp",       "tamp 2.3.0",              "LZ77",                        8,  15,    0,  BENCH_POOL_MT, lzbench_tamp_compress,       lzbench_tamp_decompress,       lzbench_tamp_init,       lzbench_tamp_deinit },
    { "tornado",    "tornado 0.6a",            "LZ77 (+ Huffman/arithmetic)", 1,  16,    0,   NO_THREADING, lzbench_tornado_compress,    lzbench_tornado_decompress,    NULL,                    NULL },
    { "ucl_nrv2b",  "ucl_nrv2b 1.03",          "LZ77",                        1,   9,    0,  BENCH_POOL_MT, lzbench_ucl_nrv2b_compress,  lzbench_ucl_nrv2b_decompress,  NULL,                    NULL },
    { "ucl_nrv2d",  "ucl_nrv2d 1.03",          "LZ77",                        1,   9,    0,  BENCH_POOL_MT, lzbench_ucl_nrv2d_compress,  lzbench_ucl_nrv2d_decompress,  NULL,                    NULL },
    { "ucl_nrv2e",  "ucl_nrv2e 1.03",          "LZ77",                        1,   9,    0,  BENCH_POOL_MT, lzbench_ucl_nrv2e_compress,  lzbench_ucl_nrv2e_decompress,  NULL,                    NULL },
    { "wflz",       "wflz 2015-09-16",         "LZ77",                        0,   0,    0,  BENCH_POOL_MT, lzbench_wflz_compress,       lzbench_wflz_decompress,       lzbench_wflz_init,       lzbench_wflz_deinit }, // SEGFAULT on decompression with gcc 4.9+ -O3 on Ubuntu
    { "wlz4",       "wlz4 1.0.0",              "LZ77",                        0,   0,    0,  BENCH_POOL_MT, lzbench_wlz4_compress,       lzbench_wlz4_decompress,       lzbench_wlz4_init,       lzbench_wlz4_deinit, 0x7FFFFF00 },
    { "wlz4fast",   "wlz4 1.0.0 --fast",       "LZ77",                        1,  99,    0,  BENCH_POOL_MT, lzbench_wlz4fast_compress,   lzbench_wlz4_decompress,       lzbench_wlz4_init,       lzbench_wlz4_deinit, 0x7FFFFF00 },
    { "wlz4hc",     "wlz4hc 1.0.0",            "LZ77",                        0,  12,    0,  BENCH_POOL_MT, lzbench_wlz4hc_compress,     lzbench_wlz4_decompress,       lzbench_wlz4_init,       lzbench_wlz4_deinit, 0x7FFFFF00 },
    { "wzip",       "wzip 1.0.0",              "LZ77 + Huffman",              0,  13,    0, FULL_THREADING, lzbench_wzip_compress,       lzbench_wzip_decompress,       NULL,                    NULL,                0x7EEEE000 },
    { "xz",         "xz 5.8.4",                "LZ77 + range",                0,   9,    0, FULL_THREADING, lzbench_xz_compress,         lzbench_xz_decompress,         NULL,                    NULL },
    { "yalz77",     "yalz77 2022-07-06",       "LZ77",                        1,  12,    0,  BENCH_POOL_MT, lzbench_yalz77_compress,     lzbench_yalz77_decompress,     NULL,                    NULL },
    { "yappy",      "yappy 2014-03-22",        "LZ77",                        1,  12,    0,   NO_THREADING, lzbench_yappy_compress,      lzbench_yappy_decompress,      lzbench_yappy_init,      NULL },
    { "zlib",       "zlib 1.3.2",              "LZ77 + Huffman",              1,   9,    0,  BENCH_POOL_MT, lzbench_zlib_compress,       lzbench_zlib_decompress,       NULL,                    NULL },
    { "zlib-ng",    "zlib-ng 2.3.3",           "LZ77 + Huffman",              1,   9,    0,  BENCH_POOL_MT, lzbench_zlib_ng_compress,    lzbench_zlib_ng_decompress,    NULL,                    NULL },
    { "zling",      "zling 2018-10-12",        "ROLZ + Huffman",              0,   4,    0,  BENCH_POOL_MT, lzbench_zling_compress,      lzbench_zling_decompress,      NULL,                    NULL },
    { "zpaq",       "zpaq 7.15",               "LZ77, BWT or CM",             1,   5,    0,  BENCH_POOL_MT, lzbench_zpaq_compress,       lzbench_zpaq_decompress,       NULL,                    NULL },
    { "zstd",       "zstd 1.5.7",              "LZ77 + FSE/Huffman",          1,  22,    0, FULL_THREADING, lzbench_zstd_compress,       lzbench_zstd_decompress,       lzbench_zstd_init,       lzbench_zstd_deinit },
    { "zstd22",     "zstd 1.5.7 -d22",        "LZ77 + FSE/Huffman",          16,  22,   22, FULL_THREADING, lzbench_zstd_compress,       lzbench_zstd_decompress,       lzbench_zstd_init,       lzbench_zstd_deinit },
    { "zstd22LDM",  "zstd 1.5.7 --long -d22", "LZ77 + FSE/Huffman",          16,  22,   22, FULL_THREADING, lzbench_zstd_LDM_compress,   lzbench_zstd_decompress,       lzbench_zstd_LDM_init,   lzbench_zstd_deinit },
    { "zstd24",     "zstd 1.5.7 -d24",        "LZ77 + FSE/Huffman",          16,  22,   24, FULL_THREADING, lzbench_zstd_compress,       lzbench_zstd_decompress,       lzbench_zstd_init,       lzbench_zstd_deinit },
    { "zstd24LDM",  "zstd 1.5.7 --long -d24", "LZ77 + FSE/Huffman",          16,  22,   24, FULL_THREADING, lzbench_zstd_LDM_compress,   lzbench_zstd_decompress,       lzbench_zstd_LDM_init,   lzbench_zstd_deinit },
    { "zstdLDM",    "zstd 1.5.7 --long",       "LZ77 + FSE/Huffman",          1,  22,    0, FULL_THREADING, lzbench_zstd_LDM_compress,   lzbench_zstd_decompress,       lzbench_zstd_LDM_init,   lzbench_zstd_deinit },
    { "zstd_fast",  "zstd 1.5.7 --fast",      "LZ77 + FSE/Huffman",          -5,  -1,    0, FULL_THREADING, lzbench_zstd_compress,       lzbench_zstd_decompress,       lzbench_zstd_init,       lzbench_zstd_deinit },
    { "zxc",        "zxc 0.14.1",              "LZ77 + Huffman",              1,   7,    0, BENCH_POOL_MT,  lzbench_zxc_compress,        lzbench_zxc_decompress,        lzbench_zxc_init,        lzbench_zxc_deinit },
};

const long int LZBENCH_COMPRESSOR_COUNT = sizeof(comp_desc)/sizeof(comp_desc[0]);


// Codecs whose algorithm depends on the level; comp_desc[].algorithm is their summary.
// Levels a codec supports but this table does not list fall back to comp_desc[].algorithm.
typedef struct
{
    const char* name;
    int first_level;
    int last_level;
    const char* algorithm;
} algorithm_desc_t;

static const algorithm_desc_t algorithm_by_level[] =
{
    { "kanzi",    1,  1, "LZ77" },                  // LZX, no entropy coder
    { "kanzi",    2,  3, "LZ77 + Huffman" },
    { "kanzi",    4,  4, "ROLZ + ANS" },
    { "kanzi",    5,  5, "BWT + ANS" },
    { "kanzi",    6,  6, "BWT + arithmetic" },      // FPAQ
    { "kanzi",    7,  7, "BWT + CM" },
    { "kanzi",    8,  9, "CM" },                    // TPAQ, TPAQX
    { "lizard",  10, 29, "LZ77" },
    { "lizard",  30, 49, "LZ77 + Huffman" },
    { "tornado",  1,  2, "LZ77" },                  // byte and bit codes
    { "tornado",  3,  4, "LZ77 + Huffman" },
    { "tornado",  5, 16, "LZ77 + arithmetic" },
    { "zpaq",     1,  2, "LZ77" },
    { "zpaq",     3,  4, "LZ77/BWT + CM" },         // chosen per block from the data
    { "zpaq",     5,  5, "CM" },
};

// The algorithm of a codec at a level: from algorithm_by_level, else the codec's own
static inline const char* lzbench_algorithm(const compressor_desc_t* desc, int level)
{
    for (size_t i = 0; i < sizeof(algorithm_by_level)/sizeof(algorithm_by_level[0]); i++)
        if (strcmp(algorithm_by_level[i].name, desc->name) == 0
            && level >= algorithm_by_level[i].first_level && level <= algorithm_by_level[i].last_level)
            return algorithm_by_level[i].algorithm;
    return desc->algorithm;
}


static const alias_desc_t alias_desc[] =
{   // default alias
    // FAST: every level that compressed silesia.tar at 100 MB/s or more on one thread of an AMD EPYC 9555P
    // (3.20 GHz, lzbench 2.4, doc/results); tornado -1 is left out as it fails on incompressible data
    { "FAST", "Refers to compressors capable of achieving compression speeds exceeding 100 MB/s (default alias).",
              "memcpy/brieflz,1,3/brotli,0,2/density/fastlz/kanzi,1,2/libdeflate,1,3/lizard,10,11,12,20,22,30,32,40,42/lz4/" \
              "lz4fast,3,9,17/lz4hc,1/lzav,1/lzf/lzjb/lzo1,1/lzo1a/lzo1b,1,3,6,9,99/lzo1c,1,3,6,9,99/lzo1f,1/lzo1x,1,11,12,15/lzo1y,1/" \
              "lzsse4fast/mbrotli,0,2/memlz/misa77,-1,0/misa77_safe,-1,0/quicklz,1,2/skim/slz_gzip/snappy/yalz77,1/zlib-ng,1/" \
              "zstd,1,2,3,4,5/zstd_fast,-5,-3,-1/zxc,1,3" },
    // CI uses LZ + LZ+ENTROPY + SYMMETRIC for single-threaded testing
    // LZ: LZ codecs, or levels of them, without an entropy coder
    { "LZ",   "Represents LZ-based compressors without an entropy coder.",
              "memcpy/brieflz,1,3,6,8/crush,0,2/fastlz,1,2/kanzi,1/lizard,10,12,15,19,20,22,25,29/lz4fast,17,9,3/lz4/lz4hc,1,4,9,12/lzav/" \
              "lzf,0,1/lzg,1,4,6,8/lzjb/lzo1/lzo1a/lzo1b,1,3,6,9,99,999/lzo1c,1,3,6,9,99,999/lzo1f/lzo1x/lzo1y/lzo1z/lzo2a/" \
              "lzsse2,1,6,12,16/lzsse4fast/lzsse4,1,6,12,16/lzsse8,1,6,12,16/lzvn/memlz/misa77,-1,0,1,2,3,4/misa77_safe,-1,0,1,2,3/quicklz,1,2,3/" \
              "snappy/tamp,8,12,15/tornado,2/ucl_nrv2b,1,6,9/ucl_nrv2d,1,6,9/ucl_nrv2e,1,6,9/wlz4fast,17,9,3/wlz4/wlz4hc,0,2,6,10,12/" \
              "yalz77,1,6,12/zpaq,1" },
    // LZ+ENTROPY: LZ codecs, or levels of them, followed by Huffman, FSE/ANS or range coding
    { "LZ+ENTROPY", "Represents LZ-based compressors with an entropy coder (Huffman, FSE/ANS or range coding).",
              "memcpy/aceapex,1,2/brotli,0,2,5,8,11/fastlzma2,1,3,5,8,10/kanzi,2,3,4/libdeflate,1,3,6,9,12/" \
              "lizard,30,32,35,39,40,42,45,49/lzfse/lzham,0,1/lzlib,0,3,6,9/lzma,0,2,4,6,9/mbrotli,0,2,5,8,11/slz_gzip/" \
              "tornado,6,11,16/wzip,0,1,3,5,9,11,13/xz,1,3,5,7,9/zlib,1,6,9/zlib-ng,1,6,9/zling,0,2,4/zstd_fast,-5,-3,-1/" \
              "zstd,1,2,5,8,11,15,18,22/zxc,1,3,6" },
    { "SYMMETRIC", "Includes compressors with similar compression and decompression speeds.",
              "memcpy/bsc1/bsc4/bsc5/bzip2,1,5,9/bzip3,1,5,9/density,1,2,3/kanzi,5,6,7,8,9/lbzip2,1,5,9/ppmd8,1,4,9/pulsar/skim/zpaq,5" },
    { "ALL",  "Represents all major compressors.",
              "LZ/LZ+ENTROPY/SYMMETRIC" },
    // CI uses FASTEST for multi-threaded testing (except Tornado, which is disabled as it has issues with incompressible data)
    // lzg is left out too: its fast mode is very slow on small files (0.16 MB/s with -T2 -jr on 468 files of lz/lzo and lz/lz4)
    { "FASTEST", "All LZ/LZ+ENTROPY/SYMMETRIC compressors, each at only its fastest level.",
       /* LZ */ "memcpy/brieflz,1/crush,0/fastlz,1/kanzi,1/lizard,10/lz4fast,99/lz4/lz4hc,1/lzav,1/lzf,0/lzjb/" \
              "lzo1,1/lzo1a,1/lzo1b,1/lzo1c,1/lzo1f,1/lzo1x,1/lzo1y,1/lzo1z/lzo2a/lzsse2,1/lzsse4fast/lzsse4,1/lzsse8,1/lzvn/memlz/" \
              "misa77,0/misa77_safe,0/quicklz,1/snappy/tamp,8/ucl_nrv2b,1/ucl_nrv2d,1/ucl_nrv2e,1/wlz4fast,99/wlz4hc,0/yalz77,1/zpaq,1/" \
/* LZ+ENTROPY */ "aceapex,3/brotli,0/fastlzma2,1/libdeflate,1/lzfse/lzham,0/lzlib,0/lzma,0/mbrotli,0/slz_gzip,1/wzip,0/xz,0/" \
              "zlib,1/zlib-ng,1/zling,0/zstd_fast,-5/zstd,1/zxc,1/" \
  /* SYMMETR */ "bsc1/bzip2,1/bzip3,1/density,1/lbzip2,1/ppmd8,1/skim" },
    { "SLOW", "Lists very slow compressors.",
              "memcpy/glza" },
    { "BUGGY", "Lists potentially unstable codecs that may cause segmentation faults.",
              "memcpy/csc/gipfeli/lzmat/lzrw/lzsse8fast/wflz/yappy" }, // these can SEGFAULT
    { "POPULAR", "Includes commonly used compressors.",
              "memcpy/brotli,0,2,5,8,11/bzip2,1,5,9/bzip3,5/kanzi,1,2,3,4,5,6,7,8,9/libdeflate,1,3,6,9,12/" \
              "lz4fast,17,9,3/lz4/lz4hc,1,4,9,12/lzlib,0,3,6,9/lzma,0,2,4,6,9/ppmd8,4/snappy/" \
              "xz,1,3,5,7,9/zlib,1,6,9/zlib-ng,1,6,9/zstd_fast,-5,-3,-1/zstd,1,2,5,8,11,15,18,22" },
    { "MAINSTREAM", "Represents mainstream compressors.",
              "memcpy/lz4fast,17,9,5/lz4/lz4hc,1,3,9/zstd_fast,-5,-3,-1/zstd,1,3,7,12,17,22/zlib,1,6,9/lzma,0,4,9/bzip2,1,9/ppmd8,4" },
    { "INT_MT", "Covers all compressors supporting internal multi-threading with -I option.",
              "memcpy/bsc0/bsc1/bsc4/bsc5/bsc6/fastlzma2,1,5,10/kanzi,1,2,3,4,5,6,7/lzham,1,4/lzma,0,4,9/xz,0,4,9/zstd,1,5,9,14,18,22" },
    // OPT: levels that use an optimal (dynamic programming) parser, from each codec's source: brieflz leparse/btparse,
    // brotli zopfli (quality 10-11), fast-lzma2 FL2_opt/ultra, libdeflate near-optimal, lizard optimalPrice, lz4hc
    // LZ4HC_CLEVEL_OPT_MIN, lzham, lzlib and 7-zip/xz LZMA normal mode, zstd btopt/btultra
    { "OPT", "Includes compressors that use optimal parsing (slow compression, fast decompression).",
              "memcpy/brieflz,5,6,7,8,9/brotli,10,11/fastlzma2,3,4,5,6,7,8,9,10/libdeflate,10,11,12/" \
              "lizard,18,19,26,27,28,29,39,46,47,48,49/lz4hc,10,11,12/" \
              "lzham,0,1,2,3,4/lzlib,1,2,3,4,5,6,7,8,9/lzma,5,6,7,8,9/wlz4hc,8,9,10,11,12/wzip,7,8,9,10,11,12,13/" \
              "xz,4,5,6,7,8,9/zstd,16,17,18,19,20,21,22" },
#if !defined(BENCH_REMOVE_UCL)
    { "UCL",      "Refers to all UCL compressor variants.",
                  "ucl_nrv2b/ucl_nrv2d/ucl_nrv2e" },
#endif
#if !defined(BENCH_REMOVE_LZO)
    { "LZO",      "Represents all LZO compressor variants.",
                  "lzo1/lzo1a/lzo1b/lzo1c/lzo1f/lzo1x/lzo1y/lzo1z/lzo2a" },
#endif
#ifndef BENCH_REMOVE_OPENZL
    { "OPENZL",   "Represents all OpenZL compressor variants.",
                  "openzl_u8/openzl_i8/openzl_le_u16/openzl_le_i16/openzl_le_u32/openzl_le_i32/openzl_le_u64/openzl_le_i64/" \
                  "openzl_serial/openzl_generic/openzl_zstd/openzl_lz4" },
#endif
#if !defined(BENCH_REMOVE_BSC)
    { "BSC",      "Represents all bsc compressor variants.",
                  "bsc0/bsc1/bsc2/bsc3/bsc4/bsc5/bsc6" },
#endif
#if !defined(BENCH_REMOVE_BSC) && defined(BENCH_HAS_CUDA)
    { "BSC_CUDA", "Represents all bsc_cuda compressor variants.",
                  "bsc_cuda0/bsc_cuda1/bsc_cuda2/bsc_cuda3/bsc_cuda4/bsc_cuda5/bsc_cuda6/bsc_cuda7/bsc_cuda8" },
#endif
#ifdef BENCH_HAS_CUDA
    { "CUDA",     "Represents all CUDA-based compressors.",
                  "memcpy/cudaMemcpy/nvcomp_lz4/bsc_cuda/aceapex_cuda/gpucompact" },
#endif
#if !defined(BENCH_REMOVE_LZO)
    { "lzo1",     nullptr, "lzo1,1,99" },
    { "lzo1a",    nullptr, "lzo1a,1,99" },
    { "lzo1b",    nullptr, "lzo1b,1,2,3,4,5,6,7,8,9,99,999" },
    { "lzo1c",    nullptr, "lzo1c,1,2,3,4,5,6,7,8,9,99,999" },
    { "lzo1f",    nullptr, "lzo1f,1,999" },
    { "lzo1x",    nullptr, "lzo1x,1,11,12,15,999" },
    { "lzo1y",    nullptr, "lzo1y,1,999" },
#endif
#ifndef BENCH_REMOVE_OPENZL
    { "openzl_zstd", nullptr, "openzl_zstd,-99,-90,-80,-70,-60,-50,-40,-30,-20,-10,-8,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,8,10,12,14,16,18,20,22" },
    { "openzl_lz4",  nullptr, "openzl_lz4,-99,-90,-80,-70,-60,-50,-40,-30,-20,-10,-8,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12" },
#endif
};

const long int LZBENCH_ALIASES_COUNT = sizeof(alias_desc)/sizeof(alias_desc[0]);

#endif

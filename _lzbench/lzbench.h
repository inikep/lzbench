#ifndef LZBENCH_H
#define LZBENCH_H

#define _CRT_SECURE_NO_WARNINGS
#define _FILE_OFFSET_BITS 64  // turn off_t into a 64-bit type for ftello() and fseeko()
#define __STDC_FORMAT_MACROS // now PRIu64 will work
#include <inttypes.h> // now PRIu64 will work

#include <vector>
#include <string>
#include "compressors.h"
#include "lz5/lz5common.h" // LZ5HC_MAX_CLEVEL

#define PROGNAME "lzbench"
#define PROGVERSION "1.0"
#define PAD_SIZE (16*1024)
#define DEFAULT_LOOP_TIME (100*1000) // 1/10 of a second
#define LZBENCH_DEBUG(level, fmt, args...) if (params->verbose >= level) printf(fmt, ##args)

#define MAX(a,b) ((a)>(b))?(a):(b)
#ifndef MIN
	#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(WIN64) || defined(_WIN64)
	#define WINDOWS
#endif

#ifdef WINDOWS
	#include <windows.h>
	#define InitTimer(x) if (!QueryPerformanceFrequency(&x)) { printf("QueryPerformance not present"); };
	#define GetTime(x) QueryPerformanceCounter(&x); 
	#define GetDiffTime(ticksPerSecond, start_ticks, end_ticks) (1000000ULL*(end_ticks.QuadPart - start_ticks.QuadPart)/ticksPerSecond.QuadPart)
	void uni_sleep(UINT usec) { Sleep(usec); };
	#ifndef __GNUC__
		#define fseeko64 _fseeki64 
		#define ftello64 _ftelli64
	#endif
	#define PROGOS "Windows"
#else
    #include <stdarg.h> // va_args
	#include <time.h>   
	#include <unistd.h>
	#include <sys/resource.h>
	typedef struct timespec LARGE_INTEGER;
	#define InitTimer(x) 
	#define GetTime(x) if(clock_gettime( CLOCK_MONOTONIC , &x) == -1 ){ printf("clock_gettime error"); };
	#define GetDiffTime(ticksPerSecond, start_ticks, end_ticks) (1000000ULL*( end_ticks.tv_sec - start_ticks.tv_sec ) + ( end_ticks.tv_nsec - start_ticks.tv_nsec )/1000)
	void uni_sleep(uint32_t usec) { usleep(usec * 1000); };
	#define PROGOS "Linux"
#endif


typedef struct string_table
{
    std::string column1;
    float column2, column3, column5;
    uint64_t column4;
    string_table(std::string c1, float c2, float c3, uint64_t c4, float c5) : column1(c1), column2(c2), column3(c3), column4(c4), column5(c5) {}
} string_table_t;

enum textformat_e { MARKDOWN=1, TEXT, CSV, TURBOBENCH };
enum timetype_e { FASTEST=1, AVERAGE, MEDIAN };

typedef struct
{
    timetype_e timetype;
    textformat_e textformat;
    size_t chunk_size;
    uint32_t c_iters, d_iters, cspeed, verbose, cmintime, dmintime, cloop_time, dloop_time;
    std::vector<string_table_t> results;
} lzbench_params_t;

struct less_using_1st_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.column1 < struct2.column1); } };
struct less_using_2nd_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.column2 > struct2.column2); } };
struct less_using_3rd_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.column3 > struct2.column3); } };
struct less_using_4th_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.column4 < struct2.column4); } };
struct less_using_5th_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.column5 < struct2.column5); } };

typedef int64_t (*compress_func)(char *in, size_t insize, char *out, size_t outsize, size_t, size_t, char*);
typedef char* (*init_func)(size_t insize);
typedef void (*deinit_func)(char* workmem);

typedef struct
{
    const char* name;
    const char* version;
    int first_level;
    int last_level;
    compress_func compress;
    compress_func decompress;
    init_func init;
    deinit_func deinit;
} compressor_desc_t;


typedef struct
{
    const char* name;
    const char* params;
} alias_desc_t;



#define LZBENCH_COMPRESSOR_COUNT 46

static const compressor_desc_t comp_desc[LZBENCH_COMPRESSOR_COUNT] =
{
    { "memcpy",   "",            0,   0, lzbench_return_0,          lzbench_memcpy,              NULL,                 NULL },
    { "blosclz",  "2015-11-10",  1,   9, lzbench_blosclz_compress,  lzbench_blosclz_decompress,  NULL,                 NULL },
    { "brieflz",  "1.1.0",       0,   0, lzbench_brieflz_compress,  lzbench_brieflz_decompress,  lzbench_brieflz_init, lzbench_brieflz_deinit },
    { "brotli",   "2016-02-04",  0,  11, lzbench_brotli_compress,   lzbench_brotli_decompress,   NULL,                 NULL },
    { "crush",    "1.0",         0,   2, lzbench_crush_compress,    lzbench_crush_decompress,    NULL,                 NULL },
    { "csc",      "3.3",         1,   5, lzbench_csc_compress,      lzbench_csc_decompress,      NULL,                 NULL },
    { "density",  "0.12.5 beta", 1,   3, lzbench_density_compress,  lzbench_density_decompress,  NULL,                 NULL }, // decompression error (shortened output)
    { "fastlz",   "0.1",         1,   2, lzbench_fastlz_compress,   lzbench_fastlz_decompress,   NULL,                 NULL },
    { "gipfeli",  "2015-11-30",  0,   0, lzbench_gipfeli_compress,  lzbench_gipfeli_decompress,  NULL,                 NULL },
    { "lz4",      "r131",        0,   0, lzbench_lz4_compress,      lzbench_lz4_decompress,      NULL,                 NULL },
    { "lz4fast",  "r131",        1,  99, lzbench_lz4fast_compress,  lzbench_lz4_decompress,      NULL,                 NULL },
    { "lz4hc",    "r131",        1,  16, lzbench_lz4hc_compress,    lzbench_lz4_decompress,      NULL,                 NULL },
    { "lz5",      "v1.4.1",      0,   0, lzbench_lz5_compress,      lzbench_lz5_decompress,      NULL,                 NULL },
    { "lz5hc",    "v1.4.1",      1,   LZ5HC_MAX_CLEVEL, lzbench_lz5hc_compress,    lzbench_lz5_decompress,      NULL,                 NULL },
    { "lzf",      "3.6",         0,   1, lzbench_lzf_compress,      lzbench_lzf_decompress,      NULL,                 NULL },
    { "lzg",      "1.0.8",       1,   9, lzbench_lzg_compress,      lzbench_lzg_decompress,      NULL,                 NULL },
    { "lzham",    "1.0 -d26",    0,   4, lzbench_lzham_compress,    lzbench_lzham_decompress,    NULL,                 NULL },
    { "lzjb",     "2010",        0,   0, lzbench_lzjb_compress,     lzbench_lzjb_decompress,     NULL,                 NULL },
    { "lzlib",    "1.7",         0,   9, lzbench_lzlib_compress,    lzbench_lzlib_decompress,    NULL,                 NULL },
    { "lzma",     "9.38",        0,   9, lzbench_lzma_compress,     lzbench_lzma_decompress,     NULL,                 NULL },
    { "lzmat",    "1.01",        0,   0, lzbench_lzmat_compress,    lzbench_lzmat_decompress,    NULL,                 NULL }, // decompression error (returns 0) and SEGFAULT (?)
    { "lzo1",     "2.09",        1,   1, lzbench_lzo1_compress,     lzbench_lzo1_decompress,     lzbench_lzo_init,     lzbench_lzo_deinit },
    { "lzo1a",    "2.09",        1,   1, lzbench_lzo1a_compress,    lzbench_lzo1a_decompress,    lzbench_lzo_init,     lzbench_lzo_deinit },
    { "lzo1b",    "2.09",        1,   1, lzbench_lzo1b_compress,    lzbench_lzo1b_decompress,    lzbench_lzo_init,     lzbench_lzo_deinit },
    { "lzo1c",    "2.09",        1,   1, lzbench_lzo1c_compress,    lzbench_lzo1c_decompress,    lzbench_lzo_init,     lzbench_lzo_deinit },
    { "lzo1f",    "2.09",        1,   1, lzbench_lzo1f_compress,    lzbench_lzo1f_decompress,    lzbench_lzo_init,     lzbench_lzo_deinit },
    { "lzo1x",    "2.09",        1,   1, lzbench_lzo1x_compress,    lzbench_lzo1x_decompress,    lzbench_lzo_init,     lzbench_lzo_deinit },
    { "lzo1y",    "2.09",        1,   1, lzbench_lzo1y_compress,    lzbench_lzo1y_decompress,    lzbench_lzo_init,     lzbench_lzo_deinit },
    { "lzo1z",    "2.09",      999, 999, lzbench_lzo1z_compress,    lzbench_lzo1z_decompress,    lzbench_lzo_init,     lzbench_lzo_deinit },
    { "lzo2a",    "2.09",      999, 999, lzbench_lzo2a_compress,    lzbench_lzo2a_decompress,    lzbench_lzo_init,     lzbench_lzo_deinit },
    { "lzrw",     "15-Jul-1991", 1,   4, lzbench_lzrw_compress,     lzbench_lzrw_decompress,     lzbench_lzrw_init,    lzbench_lzrw_deinit },
    { "pithy",    "2011-12-24",  0,   9, lzbench_pithy_compress,    lzbench_pithy_decompress,    NULL,                 NULL }, // decompression error (returns 0)
    { "quicklz",  "1.5.0",       1,   3, lzbench_quicklz_compress,  lzbench_quicklz_decompress,  NULL,                 NULL },
    { "shrinker", "0.1",         0,   0, lzbench_shrinker_compress, lzbench_shrinker_decompress, NULL,                 NULL },
    { "snappy",   "1.1.3",       0,   0, lzbench_snappy_compress,   lzbench_snappy_decompress,   NULL,                 NULL },
    { "tornado",  "0.6a",        1,  16, lzbench_tornado_compress,  lzbench_tornado_decompress,  NULL,                 NULL },
    { "ucl_nrv2b","1.03",        1,   9, lzbench_ucl_nrv2b_compress,lzbench_ucl_nrv2b_decompress,NULL,                 NULL },
    { "ucl_nrv2d","1.03",        1,   9, lzbench_ucl_nrv2d_compress,lzbench_ucl_nrv2d_decompress,NULL,                 NULL },
    { "ucl_nrv2e","1.03",        1,   9, lzbench_ucl_nrv2e_compress,lzbench_ucl_nrv2e_decompress,NULL,                 NULL },
    { "wflz",     "2015-09-16",  0,   0, lzbench_wflz_compress,     lzbench_wflz_decompress,     lzbench_wflz_init,    lzbench_wflz_deinit }, // SEGFAULT on decompressiom with gcc 4.9+ -O3 on Ubuntu
    { "xz",       "5.2.2",       0,   9, lzbench_xz_compress,       lzbench_xz_decompress,       NULL,                 NULL },
    { "yalz77",   "2015-09-19",  1,  12, lzbench_yalz77_compress,   lzbench_yalz77_decompress,   NULL,                 NULL },
    { "yappy",    "2014-03-22",  0,  99, lzbench_yappy_compress,    lzbench_yappy_decompress,    lzbench_yappy_init,   NULL },
    { "zlib",     "1.2.8",       1,   9, lzbench_zlib_compress,     lzbench_zlib_decompress,     NULL,                 NULL },
    { "zling",    "2016-01-08",  0,   4, lzbench_zling_compress,    lzbench_zling_decompress,    NULL,                 NULL },
    { "zstd",     "v0.5.0",      1,  20, lzbench_zstd_compress,     lzbench_zstd_decompress,     NULL,                 NULL },
};


#define LZBENCH_ALIASES_COUNT 12

static const alias_desc_t alias_desc[LZBENCH_ALIASES_COUNT] =
{
    { "all",  "blosclz,1,3,6,9/brieflz/brotli,0,2,5,8,11/crush,0,1,2/csc,1,2,3,4,5/density,1,2,3/fastlz,1,2/gipfeli/lz4/lz4fast,3,17/lz4hc,1,4,9,12,16/lz5/lz5hc,1,4,9,12,15/" \
              "lzf,0,1/lzg,1,4,6,8/lzham,0,1/lzjb/lzlib,0,3,6,9/lzma,0,2,4,5/lzo/" \
              "lzrw,1,2,3,4,5/pithy,0,3,6,9/quicklz,1,2,3/snappy/tornado,1,2,3,4,5,6,7,10,13,16/ucl_nrv2b,1,6,9/ucl_nrv2d,1,6,9/ucl_nrv2e,1,6,9/" \
              "xz,0,3,6,9/yalz77,1,4,8,12/yappy,1,10,100/zlib,1,6,9/zling,0,1,2,3,4/zstd,1,2,5,9,13,17,20/" \
              "shrinker/wflz/lzmat" // these can SEGFAULT 
    },
    { "fast", "density,1,2,3/fastlz,1,2/lz4/lz4fast,3,17/lz5/" \
              "lzf,0,1/lzjb/lzo1b,1/lzo1c,1/lzo1f,1/lzo1x,1/lzo1y,1/" \
              "lzrw,1,2,3,4,5/pithy,0,3,6,9/quicklz,1,2/shrinker/snappy/tornado,1,2,3/" \
              "zstd" },
    { "opt",  "brotli,6,7,8,9,10,11/csc,1,2,3,4,5/" \
              "lzham,0,1,2,3,4/lzlib,0,1,2,3,4,5,6,7,8,9/lzma,0,1,2,3,4,5,6,7/" \
              "tornado,5,6,7,8,9,10,11,12,13,14,15,16/" \
              "xz,1,2,3,4,5,6,7,8,9/zstd,10,12,14,16,17,18,19,20" },
    { "lzo1",  "lzo1,1,99" },
    { "lzo1a", "lzo1a,1,99" },
    { "lzo1b", "lzo1b,1,2,3,4,5,6,7,8,9,99,999" },
    { "lzo1c", "lzo1c,1,2,3,4,5,6,7,8,9,99,999" },
    { "lzo1f", "lzo1f,1,999" },
    { "lzo1x", "lzo1x,1,11,12,15,999" },
    { "lzo1y", "lzo1y,1,999" },
    { "lzo",   "lzo1/lzo1a/lzo1b/lzo1c/lzo1f/lzo1x/lzo1y/lzo1z/lzo2a" },
    { "ucl",   "ucl_nrv2b/ucl_nrv2d/ucl_nrv2e" },
};

#endif

#ifndef LZBENCH_H
#define LZBENCH_H

#define _CRT_SECURE_NO_WARNINGS
#define _FILE_OFFSET_BITS 64  // turn off_t into a 64-bit type for ftello() and fseeko()

#include <vector>
#include <string>
#include "compressors.h"
#include "lizard/lizard_compress.h"    // LIZARD_MAX_CLEVEL

#define PROGNAME "lzbench"
#define PROGVERSION "1.8"
#define PAD_SIZE (16*1024)
#define MIN_PAGE_SIZE 4096  // smallest page size we expect, if it's wrong the first algorithm might be a bit slower
#define DEFAULT_LOOP_TIME (100*1000000)  // 1/10 of a second
#define GET_COMPRESS_BOUND(insize) (insize + insize/6 + PAD_SIZE)  // for pithy
#define LZBENCH_PRINT(level, fmt, ...) if (params->verbose >= level) printf(fmt, __VA_ARGS__)

#define MAX(a,b) ((a)>(b))?(a):(b)
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
	void uni_sleep(UINT milisec) { Sleep(milisec); };
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
	void uni_sleep(uint32_t milisec) { usleep(milisec * 1000); };
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


typedef struct string_table
{
    std::string col1_algname;
    uint64_t col2_ctime, col3_dtime, col4_comprsize, col5_origsize;
    bool col6_dictionary;
    std::string col7_filename;
    string_table(std::string c1, uint64_t c2, uint64_t c3, uint64_t c4, uint64_t c5, bool dictionary, std::string filename) : col1_algname(c1), col2_ctime(c2), col3_dtime(c3), col4_comprsize(c4), col5_origsize(c5), col6_dictionary(dictionary), col7_filename(filename) {}
} string_table_t;

enum textformat_e { MARKDOWN=1, TEXT, TEXT_FULL, CSV, TURBOBENCH, MARKDOWN2 };
enum timetype_e { FASTEST=1, AVERAGE, MEDIAN };

typedef struct
{
    int show_speed, compress_only;
    timetype_e timetype;
    textformat_e textformat;
    size_t chunk_size;
    uint32_t c_iters, d_iters, cspeed, verbose, cmintime, dmintime, cloop_time, dloop_time;
    size_t mem_limit;
    int random_read;
    std::vector<string_table_t> results;
    const char* in_filename;
    const char* dictionary;
} lzbench_params_t;

struct less_using_1st_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.col1_algname < struct2.col1_algname); } };
struct less_using_2nd_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.col2_ctime > struct2.col2_ctime); } };
struct less_using_3rd_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.col3_dtime > struct2.col3_dtime); } };
struct less_using_4th_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.col4_comprsize < struct2.col4_comprsize); } };
struct less_using_5th_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.col5_origsize < struct2.col5_origsize); } };

typedef int64_t (*compress_func)(char *in, size_t insize, char *out, size_t outsize, size_t, size_t, char*, bool);
typedef char* (*init_func)(size_t insize, size_t, size_t, const std::string&);
typedef void (*deinit_func)(char* workmem);

typedef struct
{
    const char* name;
    const char* version;
    int first_level;
    int last_level;
    int additional_param;
    int max_block_size;
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



#define LZBENCH_COMPRESSOR_COUNT 73

static const compressor_desc_t comp_desc[LZBENCH_COMPRESSOR_COUNT] =
{
    { "memcpy",     "",            0,   0,    0,       0, lzbench_return_0,            lzbench_memcpy,                NULL,                    NULL },
    { "blosclz",    "2.0.0",       1,   9,    0, 64*1024, lzbench_blosclz_compress,    lzbench_blosclz_decompress,    NULL,                    NULL },
    { "brieflz",    "1.3.0",       1,   9,    0,       0, lzbench_brieflz_compress,    lzbench_brieflz_decompress,    lzbench_brieflz_init,    lzbench_brieflz_deinit },
    { "brotli",     "1.0.9",       0,  11,    0,       0, lzbench_brotli_compress,     lzbench_brotli_decompress,     NULL,                    NULL },
    { "brotli22",   "1.0.9",       0,  11,   22,       0, lzbench_brotli_compress,     lzbench_brotli_decompress,     NULL,                    NULL },
    { "brotli24",   "1.0.9",       0,  11,   24,       0, lzbench_brotli_compress,     lzbench_brotli_decompress,     NULL,                    NULL },
    { "bzip2",      "1.0.8",       1,   9,    0,       0, lzbench_bzip2_compress,      lzbench_bzip2_decompress,      NULL,                    NULL },
    { "crush",      "1.0",         0,   2,    0,       0, lzbench_crush_compress,      lzbench_crush_decompress,      NULL,                    NULL },
    { "csc",        "2016-10-13",  1,   5,    0,       0, lzbench_csc_compress,        lzbench_csc_decompress,        NULL,                    NULL },
    { "density",    "0.14.2",      1,   3,    0,       0, lzbench_density_compress,    lzbench_density_decompress,    lzbench_density_init,    lzbench_density_deinit },
    { "fastlz",     "0.5.0",       1,   2,    0,       0, lzbench_fastlz_compress,     lzbench_fastlz_decompress,     NULL,                    NULL },
    { "fastlzma2",   "1.0.1",      1,  10,    0,       0, lzbench_fastlzma2_compress,  lzbench_fastlzma2_decompress,  NULL,                    NULL },
    { "gipfeli",    "2016-07-13",  0,   0,    0,       0, lzbench_gipfeli_compress,    lzbench_gipfeli_decompress,    NULL,                    NULL },
    { "glza",       "0.8",         0,   0,    0,       0, lzbench_glza_compress,       lzbench_glza_decompress,       NULL,                    NULL },
    { "libdeflate", "1.6",         1,  12,    0,       0, lzbench_libdeflate_compress, lzbench_libdeflate_decompress, NULL,                    NULL },
    { "lz4",        "1.9.3",       0,   0,    0,       0, lzbench_lz4_compress,        lzbench_lz4_decompress,        NULL,                    NULL },
    { "lz4fast",    "1.9.3",       1,  99,    0,       0, lzbench_lz4fast_compress,    lzbench_lz4_decompress,        NULL,                    NULL },
    { "lz4hc",      "1.9.3",       1,  12,    0,       0, lzbench_lz4hc_compress,      lzbench_lz4_decompress,        NULL,                    NULL },
    { "lizard",     "1.0",  LIZARD_MIN_CLEVEL, LIZARD_MAX_CLEVEL, 0, 0, lzbench_lizard_compress,      lzbench_lizard_decompress,        NULL,                    NULL },
    { "lzf",        "3.6",         0,   1,    0,       0, lzbench_lzf_compress,        lzbench_lzf_decompress,        NULL,                    NULL },
    { "lzfse",      "2017-03-08",  0,   0,    0,       0, lzbench_lzfse_compress,      lzbench_lzfse_decompress,      lzbench_lzfse_init,      lzbench_lzfse_deinit },
    { "lzg",        "1.0.10",      1,   9,    0,       0, lzbench_lzg_compress,        lzbench_lzg_decompress,        NULL,                    NULL },
    { "lzham",      "1.0 -d26",    0,   4,    0,       0, lzbench_lzham_compress,      lzbench_lzham_decompress,      NULL,                    NULL },
    { "lzham22",    "1.0",         0,   4,   22,       0, lzbench_lzham_compress,      lzbench_lzham_decompress,      NULL,                    NULL },
    { "lzham24",    "1.0",         0,   4,   24,       0, lzbench_lzham_compress,      lzbench_lzham_decompress,      NULL,                    NULL },
    { "lzjb",       "2010",        0,   0,    0,       0, lzbench_lzjb_compress,       lzbench_lzjb_decompress,       NULL,                    NULL },
    { "lzlib",      "1.12-rc2",    0,   9,    0,       0, lzbench_lzlib_compress,      lzbench_lzlib_decompress,      NULL,                    NULL },
    { "lzma",       "19.00",       0,   9,    0,       0, lzbench_lzma_compress,       lzbench_lzma_decompress,       NULL,                    NULL },
    { "lzmat",      "1.01",        0,   0,    0,       0, lzbench_lzmat_compress,      lzbench_lzmat_decompress,      NULL,                    NULL }, // decompression error (returns 0) and SEGFAULT (?)
    { "lzo1",       "2.10",        1,   1,    0,       0, lzbench_lzo1_compress,       lzbench_lzo1_decompress,       lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo1a",      "2.10",        1,   1,    0,       0, lzbench_lzo1a_compress,      lzbench_lzo1a_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo1b",      "2.10",        1,   1,    0,       0, lzbench_lzo1b_compress,      lzbench_lzo1b_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo1c",      "2.10",        1,   1,    0,       0, lzbench_lzo1c_compress,      lzbench_lzo1c_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo1f",      "2.10",        1,   1,    0,       0, lzbench_lzo1f_compress,      lzbench_lzo1f_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo1x",      "2.10",        1,   1,    0,       0, lzbench_lzo1x_compress,      lzbench_lzo1x_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo1y",      "2.10",        1,   1,    0,       0, lzbench_lzo1y_compress,      lzbench_lzo1y_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo1z",      "2.10",      999, 999,    0,       0, lzbench_lzo1z_compress,      lzbench_lzo1z_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzo2a",      "2.10",      999, 999,    0,       0, lzbench_lzo2a_compress,      lzbench_lzo2a_decompress,      lzbench_lzo_init,        lzbench_lzo_deinit },
    { "lzrw",       "15-Jul-1991", 1,   5,    0,       0, lzbench_lzrw_compress,       lzbench_lzrw_decompress,       lzbench_lzrw_init,       lzbench_lzrw_deinit },
    { "lzsse2",     "2019-04-18",  0,  17,    0,       0, lzbench_lzsse2_compress,     lzbench_lzsse2_decompress,     lzbench_lzsse2_init,     lzbench_lzsse2_deinit },
    { "lzsse4",     "2019-04-18",  0,  17,    0,       0, lzbench_lzsse4_compress,     lzbench_lzsse4_decompress,     lzbench_lzsse4_init,     lzbench_lzsse4_deinit },
    { "lzsse4fast", "2019-04-18",  0,   0,    0,       0, lzbench_lzsse4fast_compress, lzbench_lzsse4_decompress,     lzbench_lzsse4fast_init, lzbench_lzsse4fast_deinit },
    { "lzsse8",     "2019-04-18",  0,  17,    0,       0, lzbench_lzsse8_compress,     lzbench_lzsse8_decompress,     lzbench_lzsse8_init,     lzbench_lzsse8_deinit },
    { "lzsse8fast", "2019-04-18",  0,   0,    0,       0, lzbench_lzsse8fast_compress, lzbench_lzsse8_decompress,     lzbench_lzsse8fast_init, lzbench_lzsse8fast_deinit },
    { "lzvn",       "2017-03-08",  0,   0,    0,       0, lzbench_lzvn_compress,       lzbench_lzvn_decompress,       lzbench_lzvn_init,       lzbench_lzvn_deinit },
    { "pithy",      "2011-12-24",  0,   9,    0,       0, lzbench_pithy_compress,      lzbench_pithy_decompress,      NULL,                    NULL }, // decompression error (returns 0)
    { "quicklz",    "1.5.0",       1,   3,    0,       0, lzbench_quicklz_compress,    lzbench_quicklz_decompress,    NULL,                    NULL },
    { "shrinker",   "0.1",         0,   0,    0, 128<<20, lzbench_shrinker_compress,   lzbench_shrinker_decompress,   NULL,                    NULL },
    { "slz_deflate","1.2.0",       1,   3,    2,       0, lzbench_slz_compress,        lzbench_slz_decompress,        NULL,                    NULL },
    { "slz_gzip",   "1.2.0",       1,   3,    1,       0, lzbench_slz_compress,        lzbench_slz_decompress,        NULL,                    NULL },
    { "slz_zlib",   "1.2.0",       1,   3,    0,       0, lzbench_slz_compress,        lzbench_slz_decompress,        NULL,                    NULL },
    { "snappy",     "2020-07-11",  0,   0,    0,       0, lzbench_snappy_compress,     lzbench_snappy_decompress,     NULL,                    NULL },
    { "tornado",    "0.6a",        1,  16,    0,       0, lzbench_tornado_compress,    lzbench_tornado_decompress,    NULL,                    NULL },
    { "ucl_nrv2b",  "1.03",        1,   9,    0,       0, lzbench_ucl_nrv2b_compress,  lzbench_ucl_nrv2b_decompress,  NULL,                    NULL },
    { "ucl_nrv2d",  "1.03",        1,   9,    0,       0, lzbench_ucl_nrv2d_compress,  lzbench_ucl_nrv2d_decompress,  NULL,                    NULL },
    { "ucl_nrv2e",  "1.03",        1,   9,    0,       0, lzbench_ucl_nrv2e_compress,  lzbench_ucl_nrv2e_decompress,  NULL,                    NULL },
    { "wflz",       "2015-09-16",  0,   0,    0,       0, lzbench_wflz_compress,       lzbench_wflz_decompress,       lzbench_wflz_init,       lzbench_wflz_deinit }, // SEGFAULT on decompressiom with gcc 4.9+ -O3 on Ubuntu
    { "xpack",      "2016-06-02",  1,   9,    0,   1<<19, lzbench_xpack_compress,      lzbench_xpack_decompress,      lzbench_xpack_init,      lzbench_xpack_deinit },
    { "xz",         "5.2.5",       0,   9,    0,       0, lzbench_xz_compress,         lzbench_xz_decompress,         NULL,                    NULL },
    { "yalz77",     "2015-09-19",  1,  12,    0,       0, lzbench_yalz77_compress,     lzbench_yalz77_decompress,     NULL,                    NULL },
    { "yappy",      "2014-03-22",  0,  99,    0,       0, lzbench_yappy_compress,      lzbench_yappy_decompress,      lzbench_yappy_init,      NULL },
    { "zlib",       "1.2.11",      1,   9,    0,       0, lzbench_zlib_compress,       lzbench_zlib_decompress,       NULL,                    NULL },
    { "zling",      "2018-10-12",  0,   4,    0,       0, lzbench_zling_compress,      lzbench_zling_decompress,      NULL,                    NULL },
    { "zstd",       "1.5.0",       1,  22,    0,       0, lzbench_zstd_compress,       lzbench_zstd_decompress,       lzbench_zstd_init,       lzbench_zstd_deinit },
    { "zstd_fast",  "1.5.0",       -5, -1,    0,       0, lzbench_zstd_compress,       lzbench_zstd_decompress,       lzbench_zstd_init,       lzbench_zstd_deinit },
    { "zstd22",     "1.5.0",       1,  22,   22,       0, lzbench_zstd_compress,       lzbench_zstd_decompress,       lzbench_zstd_init,       lzbench_zstd_deinit },
    { "zstd24",     "1.5.0",       1,  22,   24,       0, lzbench_zstd_compress,       lzbench_zstd_decompress,       lzbench_zstd_init,       lzbench_zstd_deinit },
    { "zstdLDM",    "1.5.0",       1,  22,    0,       0, lzbench_zstd_LDM_compress,   lzbench_zstd_decompress,       lzbench_zstd_LDM_init,   lzbench_zstd_deinit },
    { "zstd22LDM",  "1.5.0",       1,  22,   22,       0, lzbench_zstd_LDM_compress,   lzbench_zstd_decompress,       lzbench_zstd_LDM_init,   lzbench_zstd_deinit },
    { "zstd24LDM",  "1.5.0",       1,  22,   24,       0, lzbench_zstd_LDM_compress,   lzbench_zstd_decompress,       lzbench_zstd_LDM_init,   lzbench_zstd_deinit },
    { "nakamichi",  "okamigan",    0,   0,    0,       0, lzbench_nakamichi_compress,  lzbench_nakamichi_decompress,  NULL,                    NULL },
    { "cudaMemcpy", "",            0,   0,    0,       0, lzbench_cuda_return_0,       lzbench_cuda_memcpy,           lzbench_cuda_init,       lzbench_cuda_deinit },
    { "nvcomp_lz4", "1.2.2",       0,   5,    0,       0, lzbench_nvcomp_compress,     lzbench_nvcomp_decompress,     lzbench_nvcomp_init,     lzbench_nvcomp_deinit },
};



#define LZBENCH_ALIASES_COUNT 13

static const alias_desc_t alias_desc[LZBENCH_ALIASES_COUNT] =
{
    { "fast", "density/fastlz/lizard,10,11,12,13,14/lz4/lz4fast,3,17/lzf/lzfse/lzjb/lzo1b,1/lzo1c,1/lzo1f,1/lzo1x,1/lzo1y,1/" \
              "lzrw,1,3,4,5/lzsse4fast/lzsse8fast/lzvn/pithy,0,3,6,9/quicklz,1,2/shrinker/snappy/tornado,1,2,3/zstd,1,2,3,4,5" }, // default alias
#if !defined(__arm__) && !defined(__aarch64__)
    { "all",  "blosclz,1,3,6,9/brieflz,1,3,6,8/brotli,0,2,5,8,11/bzip2,1,5,9/" \
              "crush,0,1,2/csc,1,3,5/density,1,2,3/fastlz,1,2/fastlzma2,1,3,5,8,10/gipfeli/libdeflate,1,3,6,9,12/lizard,10,12,15,19,20,22,25,29,30,32,35,39,40,42,45,49/lz4/lz4fast,3,17/lz4hc,1,4,9,12/" \
              "lzf,0,1/lzfse/lzg,1,4,6,8/lzham,0,1/lzjb/lzlib,0,3,6,9/lzma,0,2,4,5,9/lzo1/lzo1a/lzo1b,1,3,6,9,99,999/lzo1c,1,3,6,9,99,999/lzo1f/lzo1x/lzo1y/lzo1z/lzo2a/" \
              "lzrw,1,3,4,5/lzsse2,1,6,12,16/lzsse4,1,6,12,16/lzsse8,1,6,12,16/lzvn/pithy,0,3,6,9/quicklz,1,2,3/slz_gzip/snappy/tornado,1,2,3,4,5,6,7,10,13,16/" \
              "ucl_nrv2b,1,6,9/ucl_nrv2d,1,6,9/ucl_nrv2e,1,6,9/xpack,1,6,9/xz,0,3,6,9/yalz77,1,4,8,12/yappy,1,10,100/zlib,1,6,9/zling,0,1,2,3,4/zstd,1,2,5,8,11,15,18,22/" \
              "shrinker/wflz/lzmat" }, // these can SEGFAULT
#else
    { "all",  "blosclz,1,3,6,9/brieflz,1,3,6,8/brotli,0,2,5,8/bzip2,1,5,9/" \
              "crush,0,1,2/csc,1,3,5/density,1,2,3/fastlz,1,2/gipfeli/libdeflate,1,3,6,9,12/lizard,10,12,15,20,22,25,30,32,35,40,42,45/lz4/lz4fast,3,17/lz4hc,1,4,9/" \
              "lzf,0,1/lzfse/lzg,1,4,6,8/lzham,0,1/lzjb/lzlib,0,3,6,9/lzma,0,2,4,5/lzo1/lzo1a/lzo1b,1,3,6,9,99,999/lzo1c,1,3,6,9,99,999/lzo1f/lzo1x/lzo1y/lzo1z/lzo2a/" \
              "lzrw,1,3,4,5/lzsse2,1,6,12,16/lzsse4,1,6,12,16/lzsse8,1,6,12,16/lzvn/pithy,0,3,6,9/quicklz,1,2,3/slz_gzip/snappy/tornado,1,2,3,4,5,6,7,10,13,16/" \
              "ucl_nrv2b,1,6,9/ucl_nrv2d,1,6,9/ucl_nrv2e,1,6,9/xpack,1,6,9/xz,0,3,6,9/yalz77,1,4,8,12/yappy,1,10,100/zlib,1,6,9/zling,0,1,2,3,4/zstd,1,2,5,8,11,15,18,22/" \
              "shrinker/wflz/lzmat" }, // these can SEGFAULT
#endif
    { "opt",  "brotli,6,7,8,9,10,11/csc,1,2,3,4,5/lzham,0,1,2,3,4/lzlib,0,1,2,3,4,5,6,7,8,9/lzma,0,1,2,3,4,5,6,7,8,9/" \
              "tornado,5,6,7,8,9,10,11,12,13,14,15,16/xz,1,2,3,4,5,6,7,8,9/zstd,18,19,20,21,22" },
    { "lzo1",  "lzo1,1,99" },
    { "lzo1a", "lzo1a,1,99" },
    { "lzo1b", "lzo1b,1,2,3,4,5,6,7,8,9,99,999" },
    { "lzo1c", "lzo1c,1,2,3,4,5,6,7,8,9,99,999" },
    { "lzo1f", "lzo1f,1,999" },
    { "lzo1x", "lzo1x,1,11,12,15,999" },
    { "lzo1y", "lzo1y,1,999" },
    { "lzo",   "lzo1/lzo1a/lzo1b/lzo1c/lzo1f/lzo1x/lzo1y/lzo1z/lzo2a" },
    { "ucl",   "ucl_nrv2b/ucl_nrv2d/ucl_nrv2e" },
    { "cuda",  "cudaMemcpy/nvcomp_lz4,0,1,3,5" },
};

#endif

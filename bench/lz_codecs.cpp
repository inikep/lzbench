/*
 * Copyright (c) Przemyslaw Skibinski <inikep@gmail.com>
 * All rights reserved.
 *
 * This source code is dual-licensed under the GPLv2 and GPLv3 licenses.
 * For additional details, refer to the LICENSE file located in the root
 * directory of this source tree.
 *
 * lz_codecs.cpp: LZ codecs without an entropy coder (lz/), and memcpy
 */

#include "codecs.h"

#include <stdint.h>
#include <stdio.h> // printf
#include <string.h> // memcpy
#include <algorithm> // std::max


int64_t lzbench_memcpy(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    memcpy(outbuf, inbuf, insize);
    return insize;
}



#ifndef BENCH_REMOVE_MEMLZ
#define MEMLZ_IMPLEMENTATION
#include "lz/memlz/memlz.h"

char* lzbench_memlz_init(size_t insize, size_t level, size_t)
{
    return (char*)malloc(sizeof(memlz_state));
}

void lzbench_memlz_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_memlz_compress(char* inbuf, size_t insize, char* outbuf, size_t outsize, codec_options_t* codec_options)
{
    if (!codec_options->work_mem)
        return 0;

    memlz_reset((memlz_state*)codec_options->work_mem);
    return memlz_stream_compress(outbuf, inbuf, insize, (memlz_state*)codec_options->work_mem);
}

int64_t lzbench_memlz_decompress(char* inbuf, size_t insize, char* outbuf, size_t outsize, codec_options_t* codec_options)
{
    memlz_reset((memlz_state*)codec_options->work_mem);
    return (int64_t)memlz_stream_decompress(outbuf, inbuf, (memlz_state*)codec_options->work_mem);
}

#endif // BENCH_REMOVE_MEMLZ



#ifndef BENCH_REMOVE_MISA77
#include "misa77/misa77.h"

int64_t lzbench_misa77_compress(char* inbuf, size_t insize, char* outbuf, size_t outsize, codec_options_t* codec_options)
{
    // Levels run -1..4 and are monotone in ratio and (inversely) in compression speed:
    // -1 and 0 = fast compression, 1 = fastest decompression (the library default),
    // 2 = better ratio, 3 = optimal parse, 4 = "heavy" format (best ratio, slowest compression).
    return (int64_t)misa77::compress((const uint8_t*)inbuf, insize, (uint8_t*)outbuf, outsize, misa77::config((int8_t)codec_options->level));
}

// The decompressor detects the format (light for levels -1..3, heavy for level 4) from the
// stream itself. misa77_safe stops at level 3 because the heavy format has no safe decoder
// yet (misa77::decompress with dconfig(true) rejects heavy streams by returning 0).
int64_t lzbench_misa77_decompress(char* inbuf, size_t insize, char* outbuf, size_t outsize, codec_options_t* codec_options)
{
    return (int64_t)misa77::decompress((const uint8_t*)inbuf, insize, (uint8_t*)outbuf, outsize);
}

int64_t lzbench_misa77_safe_decompress(char* inbuf, size_t insize, char* outbuf, size_t outsize, codec_options_t* codec_options)
{
    return (int64_t)misa77::decompress((const uint8_t*)inbuf, insize, (uint8_t*)outbuf, outsize, misa77::dconfig(true));
}
#endif // BENCH_REMOVE_MISA77



#ifndef BENCH_REMOVE_BRIEFLZ
#include "lz/brieflz/brieflz.h"

char* lzbench_brieflz_init(size_t insize, size_t level, size_t)
{
    return (char*) malloc(blz_workmem_size_level(insize, level));
}

void lzbench_brieflz_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_brieflz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (!codec_options->work_mem)
        return 0;

    int64_t res = blz_pack_level(inbuf, outbuf, insize, (void*)codec_options->work_mem, codec_options->level);

    return res;
}

int64_t lzbench_brieflz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return blz_depack(inbuf, outbuf, outsize);
}

#endif // BENCH_REMOVE_BRIEFLZ



#ifndef BENCH_REMOVE_CRUSH
#include "lz/crush/crush.hpp"

int64_t lzbench_crush_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return crush::compress(codec_options->level, (uint8_t*)inbuf, insize, (uint8_t*)outbuf);
}

int64_t lzbench_crush_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return crush::decompress((uint8_t*)inbuf, (uint8_t*)outbuf, outsize);
}

#endif // BENCH_REMOVE_CRUSH



#ifndef BENCH_REMOVE_FASTLZ
extern "C"
{
    #include "lz/fastlz/fastlz.h"
}

int64_t lzbench_fastlz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return fastlz_compress_level(codec_options->level, inbuf, insize, outbuf);
}

int64_t lzbench_fastlz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return fastlz_decompress(inbuf, insize, outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZ4
#include "lz/lz4/lib/lz4.h"
#include "lz/lz4/lib/lz4hc.h"

int64_t lzbench_lz4_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZ4_compress_default(inbuf, outbuf, insize, outsize);
}

int64_t lzbench_lz4fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZ4_compress_fast(inbuf, outbuf, insize, outsize, codec_options->level);
}

int64_t lzbench_lz4hc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZ4_compress_HC(inbuf, outbuf, insize, outsize, codec_options->level);
}

int64_t lzbench_lz4_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZ4_decompress_safe(inbuf, outbuf, insize, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZAV
#include "lz/lzav/lzav.h"

int64_t lzbench_lzav_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (codec_options->level == 1)
        return lzav_compress_default(inbuf, outbuf, insize, outsize);
    return lzav_compress_hi(inbuf, outbuf, insize, outsize);
}

int64_t lzbench_lzav_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return lzav_decompress(inbuf, outbuf, insize, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZF
extern "C"
{
    #include "lz/lzf/lzf.h"
}

int64_t lzbench_lzf_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (codec_options->level == 0)
        return lzf_compress(inbuf, insize, outbuf, outsize);
    return lzf_compress_very(inbuf, insize, outbuf, outsize);
}

int64_t lzbench_lzf_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return lzf_decompress(inbuf, insize, outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZG
#include "lz/liblzg/lzg.h"

int64_t lzbench_lzg_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzg_encoder_config_t cfg;
    cfg.level = codec_options->level;
    cfg.fast = LZG_TRUE;
    cfg.progressfun = NULL;
    cfg.userdata = NULL;
    return LZG_Encode((const unsigned char*)inbuf, insize, (unsigned char*)outbuf, outsize, &cfg);
}

int64_t lzbench_lzg_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZG_Decode((const unsigned char*)inbuf, insize, (unsigned char*)outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZO
#include "lz/lzo/lzo1.h"
#include "lz/lzo/lzo1a.h"
#include "lz/lzo/lzo1b.h"
#include "lz/lzo/lzo1c.h"
#include "lz/lzo/lzo1f.h"
#include "lz/lzo/lzo1x.h"
#include "lz/lzo/lzo1y.h"
#include "lz/lzo/lzo1z.h"
#include "lz/lzo/lzo2a.h"

char* lzbench_lzo_init(size_t, size_t, size_t)
{
    lzo_init();

    return (char*) malloc(LZO1B_999_MEM_COMPRESS);
}

void lzbench_lzo_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_lzo1_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    if (!codec_options->work_mem)
        return 0;

    if (codec_options->level == 99)
        res = lzo1_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)codec_options->work_mem);
    else
        res = lzo1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)codec_options->work_mem);

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

int64_t lzbench_lzo1a_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;

    if (!codec_options->work_mem)
        return 0;

    if (codec_options->level == 99)
        res = lzo1a_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)codec_options->work_mem);
    else
        res = lzo1a_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)codec_options->work_mem);

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1a_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1a_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

int64_t lzbench_lzo1b_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    switch (codec_options->level)
    {
        default:
        case 1: res = lzo1b_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 2: res = lzo1b_2_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 3: res = lzo1b_3_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 4: res = lzo1b_4_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 5: res = lzo1b_5_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 6: res = lzo1b_6_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 7: res = lzo1b_7_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 8: res = lzo1b_8_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 9: res = lzo1b_9_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 99: res = lzo1b_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 999: res = lzo1b_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
    }

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1b_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1b_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

int64_t lzbench_lzo1c_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    switch (codec_options->level)
    {
        default:
        case 1: res = lzo1c_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 2: res = lzo1c_2_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 3: res = lzo1c_3_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 4: res = lzo1c_4_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 5: res = lzo1c_5_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 6: res = lzo1c_6_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 7: res = lzo1c_7_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 8: res = lzo1c_8_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 9: res = lzo1c_9_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 99: res = lzo1c_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 999: res = lzo1c_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
    }

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1c_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1c_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

int64_t lzbench_lzo1f_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    if (codec_options->level == 999)
        res = lzo1f_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    else
        res = lzo1f_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1f_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1f_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

int64_t lzbench_lzo1x_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    switch (codec_options->level)
    {
        default:
        case 1: res = lzo1x_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 11: res = lzo1x_1_11_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 12: res = lzo1x_1_12_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 15: res = lzo1x_1_15_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 999: res = lzo1x_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
    }

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1x_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1x_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

int64_t lzbench_lzo1y_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    if (codec_options->level == 999)
        res = lzo1y_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    else
        res = lzo1y_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1y_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1y_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

int64_t lzbench_lzo1z_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    res = lzo1z_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1z_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1z_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}


int64_t lzbench_lzo2a_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    res = lzo2a_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo2a_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo2a_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

#endif



#ifndef BENCH_REMOVE_LZSSE
#include "lzsse/lzsse2/lzsse2.h"

char* lzbench_lzsse2_init(size_t insize, size_t, size_t)
{
    return (char*) LZSSE2_MakeOptimalParseState(insize);
}

void lzbench_lzsse2_deinit(char* workmem)
{
    if (!workmem) return;
    LZSSE2_FreeOptimalParseState((LZSSE2_OptimalParseState*) workmem);
}

int64_t lzbench_lzsse2_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (!codec_options->work_mem) return 0;

    return LZSSE2_CompressOptimalParse((LZSSE2_OptimalParseState*) codec_options->work_mem, inbuf, insize, outbuf, outsize, codec_options->level);
}

int64_t lzbench_lzsse2_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZSSE2_Decompress(inbuf, insize, outbuf, outsize);
}


#include "lzsse/lzsse4/lzsse4.h"

char* lzbench_lzsse4_init(size_t insize, size_t, size_t)
{
    return (char*) LZSSE4_MakeOptimalParseState(insize);
}

void lzbench_lzsse4_deinit(char* workmem)
{
    if (!workmem) return;
    LZSSE4_FreeOptimalParseState((LZSSE4_OptimalParseState*) workmem);
}

int64_t lzbench_lzsse4_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (!codec_options->work_mem) return 0;

    return LZSSE4_CompressOptimalParse((LZSSE4_OptimalParseState*) codec_options->work_mem, inbuf, insize, outbuf, outsize, codec_options->level);
}

int64_t lzbench_lzsse4_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZSSE4_Decompress(inbuf, insize, outbuf, outsize);
}

char* lzbench_lzsse4fast_init(size_t, size_t, size_t)
{
    return (char*) LZSSE4_MakeFastParseState();
}

void lzbench_lzsse4fast_deinit(char* workmem)
{
    if (!workmem) return;
    LZSSE4_FreeFastParseState((LZSSE4_FastParseState*) workmem);
}

int64_t lzbench_lzsse4fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (!codec_options->work_mem) return 0;

    return LZSSE4_CompressFast((LZSSE4_FastParseState*) codec_options->work_mem, inbuf, insize, outbuf, outsize);
}


#include "lzsse/lzsse8/lzsse8.h"

char* lzbench_lzsse8_init(size_t insize, size_t, size_t)
{
    return (char*) LZSSE8_MakeOptimalParseState(insize);
}

void lzbench_lzsse8_deinit(char* workmem)
{
    if (!workmem) return;
    LZSSE8_FreeOptimalParseState((LZSSE8_OptimalParseState*) workmem);
}

int64_t lzbench_lzsse8_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (!codec_options->work_mem) return 0;

    return LZSSE8_CompressOptimalParse((LZSSE8_OptimalParseState*) codec_options->work_mem, inbuf, insize, outbuf, outsize, codec_options->level);
}

int64_t lzbench_lzsse8_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZSSE8_Decompress(inbuf, insize, outbuf, outsize);
}

char* lzbench_lzsse8fast_init(size_t, size_t, size_t)
{
    return (char*) LZSSE8_MakeFastParseState();
}

void lzbench_lzsse8fast_deinit(char* workmem)
{
    if (!workmem) return;
    LZSSE8_FreeFastParseState((LZSSE8_FastParseState*) workmem);
}

int64_t lzbench_lzsse8fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (!codec_options->work_mem) return 0;

    return LZSSE8_CompressFast((LZSSE8_FastParseState*) codec_options->work_mem, inbuf, insize, outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_QUICKLZ
#include "quicklz/quicklz151b7.h"

int64_t lzbench_quicklz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int64_t res;
    qlz_state_compress* state = (qlz_state_compress*) calloc(1, std::max(qlz151_get_setting_3(1),std::max(qlz151_get_setting_1(1), qlz151_get_setting_2(1))));
    if (!state)
        return 0;


    switch (codec_options->level)
    {
        default:
        case 1:	res = qlz151_compress_1(inbuf, outbuf, insize, (qlz_state_compress*)state); break;
        case 2:	res = qlz151_compress_2(inbuf, outbuf, insize, (qlz_state_compress*)state); break;
        case 3:	res = qlz151_compress_3(inbuf, outbuf, insize, (qlz_state_compress*)state); break;
    }

    free(state);
    return res;
}

int64_t lzbench_quicklz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int64_t res;
    qlz_state_compress* dstate = (qlz_state_compress*) calloc(1, std::max(qlz151_get_setting_3(2),std::max(qlz151_get_setting_1(2), qlz151_get_setting_2(2))));
    if (!dstate)
        return 0;

    switch (codec_options->level)
    {
        default:
        case 1: res = qlz151_decompress_1(inbuf, outbuf, (qlz_state_decompress*)dstate); break;
        case 2: res = qlz151_decompress_2(inbuf, outbuf, (qlz_state_decompress*)dstate); break;
        case 3: res = qlz151_decompress_3(inbuf, outbuf, (qlz_state_decompress*)dstate); break;
    }

    free(dstate);
    return res;
}

#endif



#ifndef BENCH_REMOVE_SNAPPY
#include "snappy/snappy.h"

int64_t lzbench_snappy_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    snappy::RawCompress(inbuf, insize, outbuf, &outsize);
    return outsize;
}

int64_t lzbench_snappy_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    snappy::RawUncompress(inbuf, insize, outbuf);
    return outsize;
}

#endif



#ifndef BENCH_REMOVE_UCL
#include "ucl/ucl.h"

int64_t lzbench_ucl_nrv2b_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    ucl_uint complen;
    int res = ucl_nrv2b_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen, NULL, codec_options->level, NULL, NULL);

    if (res != UCL_E_OK) return 0;
    return complen;
}

int64_t lzbench_ucl_nrv2b_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    ucl_uint decomplen;
    int res = ucl_nrv2b_decompress_8((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL);

    if (res != UCL_E_OK) return 0;
    return decomplen;
}

int64_t lzbench_ucl_nrv2d_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    ucl_uint complen;
    int res = ucl_nrv2d_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen, NULL, codec_options->level, NULL, NULL);

    if (res != UCL_E_OK) return 0;
    return complen;
}

int64_t lzbench_ucl_nrv2d_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    ucl_uint decomplen;
    int res = ucl_nrv2d_decompress_8((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL);

    if (res != UCL_E_OK) return 0;
    return decomplen;
}

int64_t lzbench_ucl_nrv2e_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    ucl_uint complen;
    int res = ucl_nrv2e_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen, NULL, codec_options->level, NULL, NULL);

    if (res != UCL_E_OK) return 0;
    return complen;
}

int64_t lzbench_ucl_nrv2e_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    ucl_uint decomplen;
    int res = ucl_nrv2e_decompress_8((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL);

    if (res != UCL_E_OK) return 0;
    return decomplen;
}

#endif

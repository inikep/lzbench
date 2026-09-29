/*
 * Copyright (c) Przemyslaw Skibinski <inikep@gmail.com>
 * All rights reserved.
 *
 * This source code is dual-licensed under the GPLv2 and GPLv3 licenses.
 * For additional details, refer to the LICENSE file located in the root
 * directory of this source tree.
 *
 * lz_codecs.cpp: LZ codecs without an entropy coder (lz/, and nvcomp_lz4 from misc/), and memcpy
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



#ifndef BENCH_REMOVE_LZJB
#include "lz/lzjb/lzjb2010.h"

int64_t lzbench_lzjb_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return lzjb_compress2010((uint8_t*)inbuf, (uint8_t*)outbuf, insize, outsize, 0);
}

int64_t lzbench_lzjb_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return lzjb_decompress2010((uint8_t*)inbuf, (uint8_t*)outbuf, insize, outsize, 0);
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



#ifndef BENCH_REMOVE_TAMP
#include "lz/tamp/compressor.h"
#include "lz/tamp/decompressor.h"

char* lzbench_tamp_init(size_t, size_t level, size_t)
{
    return (char*) malloc(1 << level);
}

void lzbench_tamp_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_tamp_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int64_t compressed_size = 0;
    TampConf conf = {
       /* Describes the size of the decompression buffer in bits.
       A 10-bit window represents a 1024-byte buffer.
       Must be in range [8, 15], representing [256, 32678] byte windows. */
       .window = (uint16_t)codec_options->level,
       .literal = 8,
       .use_custom_dictionary = false
    };
    TampCompressor compressor;
    tamp_compressor_init(&compressor, &conf, (unsigned char *)codec_options->work_mem);

    tamp_compressor_compress_and_flush(
            &compressor,
            (unsigned char*) outbuf,
            outsize,
            (size_t *)&compressed_size,
            (unsigned char *)inbuf,
            insize,
            NULL,
            false
    );
    return compressed_size;
}

int64_t lzbench_tamp_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int64_t decompressed_size = 0;
    TampConf conf;
    TampDecompressor decompressor;
    size_t compressed_consumed_size;

    tamp_decompressor_init(&decompressor, NULL, (unsigned char *)codec_options->work_mem, codec_options->level);

    tamp_decompressor_decompress(
        &decompressor,
        (unsigned char *)outbuf,
        outsize,
        (size_t *)&decompressed_size,
        (unsigned char *)inbuf,
        insize,
        NULL
    );

    return decompressed_size;
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




#ifndef BENCH_REMOVE_YALZ77
#include "lz/yalz77/lz77.h"

int64_t lzbench_yalz77_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lz77::compress_t compress(codec_options->level, lz77::DEFAULT_BLOCKSIZE);
    std::string compressed = compress.feed((unsigned char*)inbuf, (unsigned char*)inbuf+insize);
    if (compressed.size() > outsize) return 0;
    memcpy(outbuf, compressed.c_str(), compressed.size());
    return compressed.size();
}

int64_t lzbench_yalz77_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lz77::decompress_t decompress;
    std::string temp;
    decompress.feed((unsigned char*)inbuf, (unsigned char*)inbuf+insize, temp);
    const std::string& decompressed = decompress.result();
    if (decompressed.size() > outsize) return 0;
    memcpy(outbuf, decompressed.c_str(), decompressed.size());
    return decompressed.size();
}

#endif




#if defined(BENCH_HAS_CUDA) && defined(BENCH_HAS_NVCOMP)
#include <cuda_runtime.h>

#define CUDA_CHECK(cond)                                               \
    do {                                                               \
        int err = cond;                                                \
        if (err != nvcompSuccess) {                                    \
            fprintf(stderr, "CUDA failure at %s:%d - Error Code: %d\n",\
                    __FILE__, __LINE__, err);                          \
            return 0;                                                  \
        }                                                              \
    } while (false)

#include "misc/nvcomp/include/nvcomp/lz4.h"

typedef struct {
    cudaStream_t stream;
    size_t max_out_bytes;
    size_t batch_size;

    char* device_input_data;
    void ** device_uncompressed_ptrs;
    size_t* device_uncompressed_bytes;

    char* device_output_data;
    void** device_compressed_ptrs;
    size_t *device_compressed_bytes;

    char* device_temp_ptr;
    size_t device_temp_bytes;

    void ** host_compressed_ptrs;
    size_t* host_compressed_bytes;

    void ** host_uncompressed_ptrs;
    size_t* host_uncompressed_bytes;
    nvcompLZ4FormatOpts opts;
} nvcomp_params_s;

// allocate the host and device memory buffers for the nvcom LZ4 compression and decompression
// the chunk size is configured by the compression level, 0 to 5 inclusive, corresponding to a chunk size from 32 kB to 1 MB
char* lzbench_nvcomp_init(size_t in_bytes, size_t level, size_t)
{
    // allocate the host memory for the algorithm options
    nvcomp_params_s* params = (nvcomp_params_s*) malloc(sizeof(nvcomp_params_s));
    if (!params) return NULL;

    // create a CUDA stream to run the compression/decompression
    int status = 0;
    CUDA_CHECK(cudaStreamCreate(&params->stream));

    // set the chunk size based on the compression level
    params->opts.chunk_size = 1 << (15 + level);
    params->batch_size = (in_bytes + params->opts.chunk_size - 1) / params->opts.chunk_size;

    // allocate device memory for the data to be compressed
    CUDA_CHECK(cudaMalloc(&params->device_input_data, in_bytes));

    // Setup an array of chunk sizes
    CUDA_CHECK(cudaMallocHost((void**)&params->host_uncompressed_bytes, sizeof(size_t)*params->batch_size));
    for (size_t i = 0; i < params->batch_size; ++i) {
        if (i + 1 < params->batch_size) {
            params->host_uncompressed_bytes[i] = params->opts.chunk_size;
        } else {
            // last chunk may be smaller
            params->host_uncompressed_bytes[i] = in_bytes - (params->opts.chunk_size*i);
        }
    }

    // Setup an array of pointers to the start of each chunk
    CUDA_CHECK(cudaMallocHost((void**)&params->host_uncompressed_ptrs, sizeof(size_t)*params->batch_size));
    for (size_t ix_chunk = 0; ix_chunk < params->batch_size; ++ix_chunk) {
        params->host_uncompressed_ptrs[ix_chunk] = params->device_input_data + params->opts.chunk_size*ix_chunk;
    }

    CUDA_CHECK(cudaMalloc((void**)&params->device_uncompressed_bytes, sizeof(size_t) * params->batch_size));
    CUDA_CHECK(cudaMalloc((void**)&params->device_uncompressed_ptrs, sizeof(size_t) * params->batch_size));

    CUDA_CHECK(cudaMemcpyAsync(params->device_uncompressed_bytes, params->host_uncompressed_bytes, sizeof(size_t) * params->batch_size, cudaMemcpyHostToDevice, params->stream));
    CUDA_CHECK(cudaMemcpyAsync(params->device_uncompressed_ptrs, params->host_uncompressed_ptrs, sizeof(size_t) * params->batch_size, cudaMemcpyHostToDevice, params->stream));

    // determine the size of the temporary buffer
    CUDA_CHECK(nvcompBatchedLZ4CompressGetTempSize(params->batch_size, params->opts.chunk_size, nvcompBatchedLZ4DefaultOpts, &params->device_temp_bytes));

    // allocate device memory for the temporary buffer
    CUDA_CHECK(cudaMalloc(&params->device_temp_ptr, params->device_temp_bytes));

    // get the maxmimum output size for each chunk
    CUDA_CHECK(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(params->opts.chunk_size, nvcompBatchedLZ4DefaultOpts, &params->max_out_bytes));

    // allocate device memory for the data to be compressed
    CUDA_CHECK(cudaMalloc(&params->device_output_data, params->batch_size * params->max_out_bytes));

    // Next, allocate output space on the device
    CUDA_CHECK(cudaMallocHost((void**)&params->host_compressed_bytes, sizeof(size_t) * params->batch_size));
    CUDA_CHECK(cudaMallocHost((void**)&params->host_compressed_ptrs, sizeof(size_t) * params->batch_size));
    for(size_t ix_chunk = 0; ix_chunk < params->batch_size; ++ix_chunk) {
        params->host_compressed_ptrs[ix_chunk] = params->device_output_data + params->max_out_bytes*ix_chunk;
    }

    CUDA_CHECK(cudaMalloc((void**)&params->device_compressed_ptrs, sizeof(size_t) * params->batch_size));
    CUDA_CHECK(cudaMemcpyAsync(
        params->device_compressed_ptrs, params->host_compressed_ptrs,
        sizeof(size_t) * params->batch_size, cudaMemcpyHostToDevice, params->stream));

    // allocate space for compressed chunk sizes to be written to
    CUDA_CHECK(cudaMalloc((void**)&params->device_compressed_bytes, sizeof(size_t) * params->batch_size));

    return (char*) params;
}

void lzbench_nvcomp_deinit(char* nvcomp_params)
{
    nvcomp_params_s* params = (nvcomp_params_s*) nvcomp_params;
    if (!params) return;

    // free all the device memory
    cudaFree(params->device_input_data);
    cudaFree(params->device_uncompressed_ptrs);
    cudaFree(params->device_uncompressed_bytes);
    cudaFree(params->device_output_data);
    cudaFree(params->device_compressed_ptrs);
    cudaFree(params->device_compressed_bytes);
    cudaFree(params->device_temp_ptr);
    cudaFreeHost(params->host_compressed_ptrs);
    cudaFreeHost(params->host_compressed_bytes);
    cudaFreeHost(params->host_uncompressed_ptrs);
    cudaFreeHost(params->host_uncompressed_bytes);

    // release the CUDA stream
    cudaStreamDestroy(params->stream);

    // free the host memory for the algorithm options
    free(params);
}

int64_t lzbench_nvcomp_compress(char *inbuf, size_t in_bytes, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    nvcomp_params_s* params = (nvcomp_params_s*) codec_options->work_mem;
    int status = 0;

    // copy the uncompressed data to the device
    CUDA_CHECK(cudaMemcpyAsync(params->device_input_data, inbuf, in_bytes, cudaMemcpyHostToDevice, params->stream));

#if 0
    fprintf(stderr, "COMPRESS device_uncompressed_ptrs=%p device_uncompressed_bytes=%p\n", params->device_uncompressed_ptrs, params->device_uncompressed_bytes);
    fprintf(stderr, "COMPRESS chunk_size=%ld batch_size=%ld\n", params->opts.chunk_size, params->batch_size);
    fprintf(stderr, "COMPRESS device_temp_ptr=%p device_temp_bytes=%ld\n", params->device_temp_ptr, params->device_temp_bytes);
    fprintf(stderr, "COMPRESS device_compressed_ptrs=%p device_compressed_bytes=%p\n", params->device_compressed_ptrs, params->device_compressed_bytes);
#endif

    // call the API to compress the data
    CUDA_CHECK(nvcompBatchedLZ4CompressAsync(
        params->device_uncompressed_ptrs,
        params->device_uncompressed_bytes,
        params->opts.chunk_size, // The maximum chunk size
        params->batch_size,
        params->device_temp_ptr,
        params->device_temp_bytes,
        params->device_compressed_ptrs,
        params->device_compressed_bytes,
        nvcompBatchedLZ4DefaultOpts,
        params->stream));

    // limit the data to be copied back to the size available on the host
    size_t out_bytes = std::min(outsize, params->batch_size * params->max_out_bytes);

    // copy the compressed data back to the host
    CUDA_CHECK(cudaMemcpyAsync(outbuf, params->device_output_data, out_bytes, cudaMemcpyDeviceToHost, params->stream));
    CUDA_CHECK(cudaMemcpyAsync(params->host_compressed_bytes, params->device_compressed_bytes, sizeof(size_t) * params->batch_size, cudaMemcpyDeviceToHost, params->stream));

    // ensure that all operations and copies are complete, and that params->device_compressed_bytes is available
    CUDA_CHECK(cudaStreamSynchronize(params->stream));

    size_t total_out_bytes = 0;
    for (size_t i = 0; i < params->batch_size; ++i) {
        //fprintf(stderr, "COMPRESS host_compressed_bytes[%ld]=%ld\n", i, params->host_compressed_bytes[i]);
        total_out_bytes += params->host_compressed_bytes[i];
    }

    return total_out_bytes;
}

int64_t lzbench_nvcomp_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    nvcomp_params_s* params = (nvcomp_params_s*) codec_options->work_mem;
    int status = 0;
    size_t uncompressed_size = outsize;

    // make sure that original data is cleared from device
    size_t in_bytes = std::min(insize, params->batch_size * params->max_out_bytes);
    CUDA_CHECK(cudaMemsetAsync(params->device_input_data, 0, uncompressed_size));
    CUDA_CHECK(cudaMemsetAsync(params->device_output_data, 0, in_bytes));

    // copy the compressed data to the device
    CUDA_CHECK(cudaMemcpyAsync(params->device_output_data, inbuf, in_bytes, cudaMemcpyHostToDevice, params->stream));

    // decompression the data on the device
    CUDA_CHECK(nvcompBatchedLZ4DecompressAsync(
        params->device_compressed_ptrs,
        params->device_compressed_bytes,
        params->device_uncompressed_bytes,
        nullptr,
        params->batch_size,
        params->device_temp_ptr,
        params->device_temp_bytes,
        params->device_uncompressed_ptrs,
        nullptr,
        params->stream));

    // copy the uncompressed data back to the host
    CUDA_CHECK(cudaMemcpyAsync(outbuf, params->device_input_data, uncompressed_size, cudaMemcpyDeviceToHost, params->stream));

    // ensure that all operations and copies are complete
    CUDA_CHECK(cudaStreamSynchronize(params->stream));

    return uncompressed_size;
}

#endif    // BENCH_HAS_CUDA && BENCH_HAS_NVCOMP

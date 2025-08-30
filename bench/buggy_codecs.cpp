/*
 * Copyright (c) Przemyslaw Skibinski <inikep@gmail.com>
 * All rights reserved.
 *
 * This source code is dual-licensed under the GPLv2 and GPLv3 licenses.
 * For additional details, refer to the LICENSE file located in the root
 * directory of this source tree.
 *
 * buggy_codecs.cpp: potentially unstable codecs that may cause segmentation faults
 */

#include "codecs.h"
#include <algorithm> // std::max


// has to be compiled separated because of collisions with LZMA's 7zTypes.h
#ifndef BENCH_REMOVE_CSC
#include "lz/libcsc/csc_enc.h"
#include "lz/libcsc/csc_dec.h"
#include <string.h> // memcpy

struct MemSeqStream
{
    union {
        ISeqInStream is;
        ISeqOutStream os;
    };
    char *buf;
    size_t len;
};


int stdio_read(void *p, void *buf, size_t *size)
{
    MemSeqStream *sss = (MemSeqStream *)p;
//    *size = fread(buf, 1, *size, sss->f);
    if (*size > sss->len)
        *size = sss->len;
    memcpy(buf, sss->buf, *size);
    sss->buf += *size;
    sss->len -= *size;
    return 0;
}

size_t stdio_write(void *p, const void *buf, size_t size)
{
    MemSeqStream *sss = (MemSeqStream *)p;

    memcpy(sss->buf, buf, size);
    sss->buf += size;
    sss->len += size;
    return size;
}

int64_t lzbench_csc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    MemSeqStream isss, osss;
    CSCProps p;
    int dict_size = codec_options->additional_param;

    if (!dict_size) dict_size = 1<<26;

    if (insize < dict_size)
        dict_size = insize;

    CSCEncProps_Init(&p, dict_size, codec_options->level);
    CSCEnc_WriteProperties(&p, (uint8_t*)outbuf, 0);

    isss.is.Read = stdio_read;
    isss.buf = inbuf;
    isss.len = insize;

    osss.os.Write = stdio_write;
    osss.buf = outbuf + CSC_PROP_SIZE;
    osss.len = CSC_PROP_SIZE;

    CSCEncHandle h = CSCEnc_Create(&p, (ISeqOutStream*)&osss, NULL);
    CSCEnc_Encode(h, (ISeqInStream*)&isss, NULL);
    CSCEnc_Encode_Flush(h);
    CSCEnc_Destroy(h);

    //printf("Estimated memory usage: %llu MB\n", CSCEnc_EstMemUsage(&p) / 1048576ull);
    //printf("insize=%lld osss.len=%lld\n", insize, osss.len);

    return osss.len;
}


int64_t lzbench_csc_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    MemSeqStream isss, osss;
    CSCProps p;

    CSCDec_ReadProperties(&p, (uint8_t*)inbuf);

    isss.is.Read = stdio_read;
    isss.buf = inbuf + CSC_PROP_SIZE;
    isss.len = insize - CSC_PROP_SIZE;

    osss.os.Write = stdio_write;
    osss.buf = outbuf;
    osss.len = 0;

    CSCDecHandle h = CSCDec_Create(&p, (ISeqInStream*)&isss, NULL);
    CSCDec_Decode(h, (ISeqOutStream*)&osss, NULL);
    CSCDec_Destroy(h);

    return osss.len;
}

#endif // BENCH_REMOVE_CSC



#ifndef BENCH_REMOVE_GIPFELI
#include "lz/gipfeli/gipfeli.h"

int64_t lzbench_gipfeli_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int64_t res;
    util::compression::Compressor *gipfeli = util::compression::NewGipfeliCompressor();
    if (gipfeli)
    {
        util::compression::UncheckedByteArraySink sink((char*)outbuf);
        util::compression::ByteArraySource src((const char*)inbuf, insize);
        res = gipfeli->CompressStream(&src, &sink);
        delete gipfeli;
    }
    else res=0;
    return res;
}

int64_t lzbench_gipfeli_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int64_t res = 0;
    util::compression::Compressor *gipfeli = util::compression::NewGipfeliCompressor();
    if (gipfeli)
    {
        util::compression::UncheckedByteArraySink sink((char*)outbuf);
        util::compression::ByteArraySource src((const char*)inbuf, insize);
        if (gipfeli->UncompressStream(&src, &sink))
            res = outsize;
        delete gipfeli;
    }
    return res;
}

#endif



#ifndef BENCH_REMOVE_LZMAT
#include "lz/lzmat/lzmat.h"

int64_t lzbench_lzmat_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    uint32_t complen = outsize;
    if (lzmat_encode((uint8_t*)outbuf, &complen, (uint8_t*)inbuf, insize) != 0)
        return 0;
    return complen;
}

int64_t lzbench_lzmat_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    uint32_t decomplen = outsize;
    if (lzmat_decode((uint8_t*)outbuf, &decomplen, (uint8_t*)inbuf, insize) != 0)
        return 0;
    return decomplen;
}

#endif



#ifndef BENCH_REMOVE_LZRW
extern "C"
{
    #include "lz/lzrw/lzrw.h"
}

char* lzbench_lzrw_init(size_t, size_t, size_t)
{
    return (char*) malloc(lzrw2_req_mem());
}

void lzbench_lzrw_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_lzrw_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    uint32_t complen = 0;
    switch (codec_options->level)
    {
        default:
        case 1: lzrw1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen); break;
        case 2: lzrw1a_compress(COMPRESS_ACTION_COMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen); break;
        case 3: lzrw2_compress(COMPRESS_ACTION_COMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen); break;
        case 4: lzrw3_compress(COMPRESS_ACTION_COMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen); break;
        case 5: lzrw3a_compress(COMPRESS_ACTION_COMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen); break;
    }

    return complen;
}

int64_t lzbench_lzrw_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    uint32_t decomplen = 0;
    switch (codec_options->level)
    {
        default:
        case 1: lzrw1_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen); break;
        case 2: lzrw1a_compress(COMPRESS_ACTION_DECOMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen); break;
        case 3: lzrw2_compress(COMPRESS_ACTION_DECOMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen); break;
        case 4: lzrw3_compress(COMPRESS_ACTION_DECOMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen); break;
        case 5: lzrw3a_compress(COMPRESS_ACTION_DECOMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen); break;
    }

    return decomplen;
}

#endif



#ifndef BENCH_REMOVE_WFLZ
#include "lz/wflz/wfLZ.h"

char* lzbench_wflz_init(size_t, size_t, size_t)
{
    return (char*) malloc(wfLZ_GetWorkMemSize());
}

void lzbench_wflz_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_wflz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    char* workmem = codec_options->work_mem;
    int64_t res;
    if (!workmem)
        return 0;

    if (codec_options->level == 0)
        res = wfLZ_CompressFast((const uint8_t*)inbuf, insize, (uint8_t*)outbuf, (uint8_t*)workmem, 0);
    else
        res = wfLZ_Compress((const uint8_t*)inbuf, insize, (uint8_t*)outbuf, (uint8_t*)workmem, 0);

    return res;
}

int64_t lzbench_wflz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    wfLZ_Decompress((const uint8_t*)inbuf, (uint8_t*)outbuf);
    return outsize;
}

#endif



#ifndef BENCH_REMOVE_YAPPY
#include "lz/yappy/yappy.hpp"

char* lzbench_yappy_init(size_t insize, size_t level, size_t)
{
    YappyFillTables();
    return NULL;
}

int64_t lzbench_yappy_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int yappy_level = (codec_options->level <= 1) ? 0 : 1 << (codec_options->level - 2);
    return YappyCompress((uint8_t*)inbuf, (uint8_t*)outbuf, insize, yappy_level) - (uint8_t*)outbuf;
}

int64_t lzbench_yappy_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return YappyUnCompress((uint8_t*)inbuf, (uint8_t*)inbuf+insize, (uint8_t*)outbuf) - (uint8_t*)outbuf;
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

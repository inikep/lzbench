/*
 * Copyright (c) Przemyslaw Skibinski <inikep@gmail.com>
 * All rights reserved.
 *
 * This source code is dual-licensed under the GPLv2 and GPLv3 licenses.
 * For additional details, refer to the LICENSE file located in the root
 * directory of this source tree.
 *
 * symmetric_codecs.cpp: non-LZ based codecs with similar compression and decompression speeds
 */

#include "codecs.h"


#ifndef BENCH_REMOVE_BSC
#include "bwt/libbsc/libbsc/libbsc.h"

char *lzbench_bsc_init(size_t insize, size_t level, size_t)
{
    int features = LIBBSC_DEFAULT_FEATURES | LIBBSC_FEATURE_CUDA;
    bsc_init(features);
    return 0;
}

int64_t lzbench_bsc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int features = LIBBSC_DEFAULT_FEATURES;
    int lzpHashSize = LIBBSC_DEFAULT_LZPHASHSIZE;
    int lzpMinLen = LIBBSC_DEFAULT_LZPMINLEN;
    int blockSorter = LIBBSC_DEFAULT_BLOCKSORTER;
    int coder = LIBBSC_DEFAULT_CODER;

    int level = codec_options->additional_param;
    blockSorter = level < 3 ? 1 : (int)level;

    coder = level == 0 ? LIBBSC_CODER_QLFC_ADAPTIVE :
            level == 1 ? LIBBSC_CODER_QLFC_STATIC :
            level == 2 ? LIBBSC_CODER_QLFC_FAST :
            level == 7 ? LIBBSC_CODER_QLFC_FAST :
            level == 8 ? LIBBSC_CODER_QLFC_FAST :
            coder;

    int res = bsc_compress((unsigned char *)inbuf, (unsigned char *)outbuf, (int)insize, lzpHashSize, lzpMinLen, blockSorter, coder, features);

    return res;
}

int64_t lzbench_bsc_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int features = LIBBSC_DEFAULT_FEATURES;
    int insize_bsc;
    int outsize_bsc;

    bsc_block_info((unsigned char *)inbuf, LIBBSC_HEADER_SIZE, &insize_bsc, &outsize_bsc, features);
    bsc_decompress((unsigned char *)inbuf, insize_bsc, (unsigned char *)outbuf, outsize_bsc, features);

    return outsize;
}

#ifdef BENCH_HAS_CUDA

int64_t lzbench_bsc_cuda_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int features = LIBBSC_DEFAULT_FEATURES | LIBBSC_FEATURE_CUDA;
    int lzpHashSize = LIBBSC_DEFAULT_LZPHASHSIZE;
    int lzpMinLen = LIBBSC_DEFAULT_LZPMINLEN;
    int blockSorter = LIBBSC_DEFAULT_BLOCKSORTER;
    int coder = LIBBSC_DEFAULT_CODER;

    int level = codec_options->additional_param;
    blockSorter = level < 3 ? 1 : (int)level;

    coder = level == 0 ? LIBBSC_CODER_QLFC_ADAPTIVE :
            level == 1 ? LIBBSC_CODER_QLFC_STATIC :
            level == 2 ? LIBBSC_CODER_QLFC_FAST :
            level == 7 ? LIBBSC_CODER_QLFC_FAST :
            level == 8 ? LIBBSC_CODER_QLFC_FAST :
            coder;

    int res = bsc_compress((unsigned char *)inbuf, (unsigned char *)outbuf, (int)insize, lzpHashSize, lzpMinLen, blockSorter, coder, features);

    return res;
}

int64_t lzbench_bsc_cuda_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int features = LIBBSC_DEFAULT_FEATURES | LIBBSC_FEATURE_CUDA;
    int insize_bsc;
    int outsize_bsc;

    bsc_block_info((unsigned char *)inbuf, LIBBSC_HEADER_SIZE, &insize_bsc, &outsize_bsc, features);
    bsc_decompress((unsigned char *)inbuf, insize_bsc, (unsigned char *)outbuf, outsize_bsc, features);

    return outsize;
}

#endif // BENCH_HAS_CUDA

#endif // BENCH_HAS_BSC



#ifndef BENCH_REMOVE_BZIP2
#include "bwt/bzip2/bzlib.h"

int64_t lzbench_bzip2_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
   unsigned int a_outsize = outsize;
   return BZ2_bzBuffToBuffCompress((char *)outbuf, &a_outsize, (char *)inbuf, (unsigned int)insize, codec_options->level, 0, 0)==BZ_OK ? (int64_t)a_outsize : -1;
}

int64_t lzbench_bzip2_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
   unsigned int a_outsize = outsize;
   return BZ2_bzBuffToBuffDecompress((char *)outbuf, &a_outsize, (char *)inbuf, (unsigned int)insize, 0, 0)==BZ_OK?a_outsize:-1;
}

#endif // BENCH_REMOVE_BZIP2


#ifndef BENCH_REMOVE_BZIP3
#include "bwt/bzip3/include/libbz3.h"

int64_t lzbench_bzip3_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
   size_t real_outsize = outsize;
   uint32_t block_size = 1 << (19 + codec_options->level); // level 1 = 1 MB, level 3 = 4 MB, level 9 = 256 MB, level 10 = 511 MB
   int bzerr = bz3_compress(block_size > (511 << 20) ? (511 << 20) : block_size, (uint8_t*)inbuf, (uint8_t*)outbuf, insize, &real_outsize);
   if (bzerr != BZ3_OK) return bzerr;
   return real_outsize;
}

int64_t lzbench_bzip3_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    size_t real_outsize = outsize;
    int bzerr = bz3_decompress((uint8_t*)inbuf, (uint8_t*)outbuf, insize, &real_outsize);
    if (bzerr != BZ3_OK) return bzerr;
    return real_outsize;
}

#endif // BENCH_REMOVE_BZIP3


#ifndef BENCH_REMOVE_PPMD
#include "misc/7-zip/Ppmd8.h"

int64_t lzbench_ppmd_compress(char* inbuf, size_t insize, char* outbuf, size_t outsize, codec_options_t *codec_options)
{
    struct CharWriter
    {
        IByteOut streamOut;
        char* ptr;

        static void* pmalloc(ISzAllocPtr ip, size_t size)
        {
            (void)ip;
            return malloc(size);
        }

        static void pfree(ISzAllocPtr ip, void* addr)
        {
            (void)ip;
            free(addr);
        }

        static void write(const IByteOut* p, Byte b)
        {
            CharWriter* cw = (CharWriter*)p;
            *cw->ptr++ = (char)b;
        }
    };

    int level = codec_options->level;
    level = (level == 0) ? 1 : ((level < 9) ? level : 9); // valid range for level is [1..9]
    const int modelOrder = 3 + level;
    const int memMb = 1 << (level - 1);
    const int restoreMethod = level < 7 ? PPMD8_RESTORE_METHOD_RESTART : PPMD8_RESTORE_METHOD_CUT_OFF;
    unsigned short wPPMd = (modelOrder - 1) + ((memMb - 1) << 4) + (restoreMethod << 12);

    CharWriter cw;
    cw.streamOut.Write = &CharWriter::write;
    cw.ptr = outbuf;
    CPpmd8 ppmd;
    ppmd.Stream.Out = &cw.streamOut;
    ISzAlloc ialloc = { CharWriter::pmalloc, CharWriter::pfree };

    Ppmd8_Construct(&ppmd);
    Ppmd8_Alloc(&ppmd, memMb << 20, &ialloc);
    Ppmd8_Init_RangeEnc(&ppmd);
    Ppmd8_Init(&ppmd, modelOrder, restoreMethod);

    ppmd.Stream.Out->Write(&cw.streamOut, wPPMd & 0xff);
    ppmd.Stream.Out->Write(&cw.streamOut, wPPMd >> 8);

    for (size_t i = 0; i < insize; ++i)
        Ppmd8_EncodeSymbol(&ppmd, (unsigned char)inbuf[i]);
    Ppmd8_EncodeSymbol(&ppmd, -1); /* EndMark */
    Ppmd8_Flush_RangeEnc(&ppmd);
    Ppmd8_Free(&ppmd, &ialloc);
    return cw.ptr - outbuf;
}

int64_t lzbench_ppmd_decompress(char* inbuf, size_t insize, char* outbuf, size_t outsize, codec_options_t *codec_options)
{
    struct CharReader
    {
        IByteIn streamIn;
        const char* ptr;
        const char* end;

        static void* pmalloc(ISzAllocPtr ip, size_t size)
        {
            (void)ip;
            return malloc(size);
        }

        static void pfree(ISzAllocPtr ip, void* addr)
        {
            (void)ip;
            free(addr);
        }

        static Byte read(const IByteIn* p)
        {
            CharReader* cr = (CharReader*)p;
            if (cr->ptr >= cr->end)
                return 0;
            return *cr->ptr++;
        }
    };

    CharReader cr;
    cr.streamIn.Read = &CharReader::read;
    cr.ptr = inbuf;
    cr.end = inbuf + insize;

    unsigned short wPPMd = CharReader::read(&cr.streamIn) | ((unsigned short)(CharReader::read(&cr.streamIn)) << 8);

    const int modelOrder = (wPPMd & 0xf) + 1;
    const int memMb = ((wPPMd >> 4) & 0xff) + 1;
    const int restoreMethod = wPPMd >> 12;

    CPpmd8 ppmd;
    ppmd.Stream.In = &cr.streamIn;
    ISzAlloc ialloc = { CharReader::pmalloc, CharReader::pfree };

    Ppmd8_Construct(&ppmd);
    Ppmd8_Alloc(&ppmd, memMb << 20, &ialloc);
    Ppmd8_Init_RangeDec(&ppmd);
    Ppmd8_Init(&ppmd, modelOrder, restoreMethod);

    size_t sz = 0;
    for (;;)
    {
        int c = Ppmd8_DecodeSymbol(&ppmd);
        if (cr.ptr > cr.end || c < 0)
            break;
        outbuf[sz++] = (char)(unsigned)c;
    }
    int ret = Ppmd8_RangeDec_IsFinishedOK(&ppmd) && cr.ptr >= cr.end ? 0 : -1;
    Ppmd8_Free(&ppmd, &ialloc);
    return ret == 0 ? (int64_t)sz : (int64_t)0;
}

#endif

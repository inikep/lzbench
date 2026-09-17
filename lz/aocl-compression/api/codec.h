/**
 * Copyright (C) 2022-2025, Advanced Micro Devices. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
 
 /** @file codec.h
 *  
 *  @brief Prototypes of wrapper functions of the supported codec methods.
 *
 *  This file contains the prototypes of wrapper functions of the supported
 *  compression and decompression methods along with a data structure to
 *  hold the function pointers of each of them.
 *
 *  @author S. Biplab Raut
 */
 
#ifndef CODEC_H
#define CODEC_H

typedef AOCL_UINTP (*comp_bound_fp)(AOCL_UINTP inSize);
typedef AOCL_INT64 (*comp_decomp_fp)(AOCL_CHAR *inStream, AOCL_UINTP inSize, AOCL_CHAR *outStream,
                                AOCL_UINTP outSize, AOCL_UINTP, AOCL_UINTP, AOCL_CHAR*);
typedef AOCL_CHAR* (*setup_fp)(AOCL_INTP optOff, AOCL_INTP optLevel, AOCL_UINTP inSize, AOCL_UINTP,
                          AOCL_UINTP);
typedef AOCL_VOID  (*destroy_fp)(AOCL_CHAR *memBuff);

//Method 1
#ifndef AOCL_EXCLUDE_BZIP2
        AOCL_UINTP aocl_bzip2_compressBound(AOCL_UINTP inSize);
        AOCL_INT64 aocl_bzip2_compress(AOCL_CHAR *inBuf, AOCL_UINTP inSize, AOCL_CHAR *outBuf,
                         AOCL_UINTP outSize, AOCL_UINTP level, AOCL_UINTP optVar, AOCL_CHAR *workBuf);
        AOCL_INT64 aocl_bzip2_decompress(AOCL_CHAR *inBuf, AOCL_UINTP inSize, AOCL_CHAR *outBuf,
                         AOCL_UINTP outSize, AOCL_UINTP level, AOCL_UINTP optVar, AOCL_CHAR *workBuf);
        AOCL_CHAR *aocl_bzip2_setup(AOCL_INTP optOff, AOCL_INTP optLevel,
                           AOCL_UINTP insize, AOCL_UINTP level, AOCL_UINTP windowLog);
        AOCL_VOID aocl_bzip2_destroy(AOCL_CHAR* workmem);
#else
    #define aocl_bzip2_compressBound NULL
    #define aocl_bzip2_compress NULL
    #define aocl_bzip2_decompress NULL
    #define aocl_bzip2_setup NULL
    #define aocl_bzip2_destroy NULL
#endif
//Method 2
#ifndef AOCL_EXCLUDE_LZ4
        AOCL_UINTP aocl_lz4_compressBound(AOCL_UINTP inSize);
        AOCL_INT64 aocl_lz4_compress(AOCL_CHAR *inBuf, AOCL_UINTP inSize, AOCL_CHAR *outBuf, 
                         AOCL_UINTP outSize, AOCL_UINTP level, AOCL_UINTP optVar, AOCL_CHAR *workBuf);
        AOCL_INT64 aocl_lz4_decompress(AOCL_CHAR *inBuf, AOCL_UINTP inSize, AOCL_CHAR *outBuf, 
                         AOCL_UINTP outSize, AOCL_UINTP level, AOCL_UINTP optVar, AOCL_CHAR *workBuf);
        AOCL_CHAR *aocl_lz4_setup(AOCL_INTP optOff, AOCL_INTP optLevel,
                         AOCL_UINTP insize, AOCL_UINTP level, AOCL_UINTP windowLog);
        AOCL_VOID aocl_lz4_destroy(AOCL_CHAR* workmem);
#else
    #define aocl_lz4_compressBound NULL
    #define aocl_lz4_compress NULL
    #define aocl_lz4_decompress NULL
    #define aocl_lz4_setup NULL
    #define aocl_lz4_destroy NULL
#endif
//Method 3
#if !defined(AOCL_EXCLUDE_LZ4HC) && !defined(AOCL_EXCLUDE_LZ4)
        AOCL_UINTP aocl_lz4hc_compressBound(AOCL_UINTP inSize);
        AOCL_INT64 aocl_lz4hc_compress(AOCL_CHAR *inBuf, AOCL_UINTP inSize, AOCL_CHAR *outBuf, 
                         AOCL_UINTP outSize, AOCL_UINTP level, AOCL_UINTP optVar, AOCL_CHAR *workBuf);
        AOCL_INT64 aocl_lz4hc_decompress(AOCL_CHAR *inBuf, AOCL_UINTP inSize, AOCL_CHAR *outBuf, 
                         AOCL_UINTP outSize, AOCL_UINTP level, AOCL_UINTP optVar, AOCL_CHAR *workBuf);
        AOCL_CHAR *aocl_lz4hc_setup(AOCL_INTP optOff, AOCL_INTP optLevel,
                         AOCL_UINTP insize, AOCL_UINTP level, AOCL_UINTP windowLog);
        AOCL_VOID aocl_lz4hc_destroy(AOCL_CHAR* workmem);
#else
    #define aocl_lz4hc_compressBound NULL
    #define aocl_lz4hc_compress NULL
    #define aocl_lz4hc_decompress NULL
    #define aocl_lz4hc_setup NULL
    #define aocl_lz4hc_destroy NULL
#endif
//Method 4
#ifndef AOCL_EXCLUDE_LZMA
        AOCL_UINTP aocl_lzma_compressBound(AOCL_UINTP inSize);
        AOCL_INT64 aocl_lzma_compress(AOCL_CHAR *inBuf, AOCL_UINTP inSize, AOCL_CHAR *outBuf, 
                         AOCL_UINTP outSize, AOCL_UINTP level, AOCL_UINTP optVar, AOCL_CHAR *workBuf);
        AOCL_INT64 aocl_lzma_decompress(AOCL_CHAR *inBuf, AOCL_UINTP inSize, AOCL_CHAR *outBuf, 
                         AOCL_UINTP outSize, AOCL_UINTP level, AOCL_UINTP optVar, AOCL_CHAR *workBuf);
        AOCL_CHAR *aocl_lzma_setup(AOCL_INTP optOff, AOCL_INTP optLevel,
                         AOCL_UINTP insize, AOCL_UINTP level, AOCL_UINTP windowLog);
        AOCL_VOID aocl_lzma_destroy(AOCL_CHAR* workmem);
#else
    #define aocl_lzma_compressBound NULL
    #define aocl_lzma_compress NULL
    #define aocl_lzma_decompress NULL
    #define aocl_lzma_setup NULL
    #define aocl_lzma_destroy NULL
#endif
//Method 5
#ifndef AOCL_EXCLUDE_SNAPPY
        AOCL_UINTP aocl_snappy_compressBound(AOCL_UINTP inSize);
        AOCL_INT64 aocl_snappy_compress(AOCL_CHAR *inBuf, AOCL_UINTP inSize, AOCL_CHAR *outBuf, 
                         AOCL_UINTP outSize, AOCL_UINTP level, AOCL_UINTP optVar, AOCL_CHAR *workBuf);
        AOCL_INT64 aocl_snappy_decompress(AOCL_CHAR *inBuf, AOCL_UINTP inSize, AOCL_CHAR *outBuf, 
                         AOCL_UINTP outSize, AOCL_UINTP level, AOCL_UINTP optVar, AOCL_CHAR *workBuf);
        AOCL_CHAR *aocl_snappy_setup(AOCL_INTP optOff, AOCL_INTP optLevel,
                         AOCL_UINTP insize, AOCL_UINTP level, AOCL_UINTP windowLog);
        AOCL_VOID aocl_snappy_destroy(AOCL_CHAR* workmem);
#else
    #define aocl_snappy_compressBound NULL
    #define aocl_snappy_compress NULL
    #define aocl_snappy_decompress NULL
    #define aocl_snappy_setup NULL
    #define aocl_snappy_destroy NULL
#endif
//Method 6
#ifndef AOCL_EXCLUDE_ZLIB
        AOCL_UINTP aocl_zlib_compressBound(AOCL_UINTP inSize);
        AOCL_INT64 aocl_zlib_compress(AOCL_CHAR *inBuf, AOCL_UINTP inSize, AOCL_CHAR *outBuf, 
                         AOCL_UINTP outSize, AOCL_UINTP level, AOCL_UINTP optVar, AOCL_CHAR *workBuf);
        AOCL_INT64 aocl_zlib_decompress(AOCL_CHAR *inBuf, AOCL_UINTP inSize, AOCL_CHAR *outBuf, 
                         AOCL_UINTP outSize, AOCL_UINTP level, AOCL_UINTP optVar, AOCL_CHAR *workBuf);
        AOCL_CHAR *aocl_zlib_setup(AOCL_INTP optOff, AOCL_INTP optLevel,
                         AOCL_UINTP insize, AOCL_UINTP level, AOCL_UINTP windowLog);
        AOCL_VOID aocl_zlib_destroy(AOCL_CHAR* workmem);
#else
    #define aocl_zlib_compressBound NULL
    #define aocl_zlib_compress NULL
    #define aocl_zlib_decompress NULL
    #define aocl_zlib_setup NULL
    #define aocl_zlib_destroy NULL
#endif
//Method 7
#ifndef AOCL_EXCLUDE_ZSTD
        AOCL_UINTP aocl_zstd_compressBound(AOCL_UINTP inSize);
        AOCL_INT64 aocl_zstd_compress(AOCL_CHAR *inBuf, AOCL_UINTP inSize, AOCL_CHAR *outBuf, 
                         AOCL_UINTP outSize, AOCL_UINTP level, AOCL_UINTP optVar, AOCL_CHAR *workBuf);
        AOCL_INT64 aocl_zstd_decompress(AOCL_CHAR *inBuf, AOCL_UINTP inSize, AOCL_CHAR *outBuf, 
                         AOCL_UINTP outSize, AOCL_UINTP level, AOCL_UINTP optVar, AOCL_CHAR *workBuf);
        AOCL_CHAR *aocl_zstd_setup(AOCL_INTP optOff, AOCL_INTP optLevel,
                         AOCL_UINTP insize, AOCL_UINTP level, AOCL_UINTP windowLog);
        AOCL_VOID aocl_zstd_destroy(AOCL_CHAR *workmem);
#else
    #define aocl_zstd_compressBound NULL
	#define aocl_zstd_compress NULL
	#define aocl_zstd_decompress NULL
	#define aocl_zstd_setup NULL
	#define aocl_zstd_destroy NULL
#endif

typedef struct
{
    const AOCL_CHAR* codec_name;
    const AOCL_CHAR* codec_version;
    comp_bound_fp compressBound;
    comp_decomp_fp compress;
    comp_decomp_fp decompress;
    setup_fp setup;
    destroy_fp destroy;
} aocl_codec_t;

static const aocl_codec_t aocl_codec[AOCL_COMPRESSOR_ALGOS_NUM] =
{
    { "aocl_lz4",    "1.10.0",     aocl_lz4_compressBound,    aocl_lz4_compress,    aocl_lz4_decompress,    aocl_lz4_setup,    aocl_lz4_destroy },
    { "aocl_lz4hc",  "1.10.0",     aocl_lz4hc_compressBound,  aocl_lz4hc_compress,  aocl_lz4_decompress,    aocl_lz4hc_setup,  aocl_lz4hc_destroy },
    { "aocl_lzma",   "22.01",      aocl_lzma_compressBound,   aocl_lzma_compress,   aocl_lzma_decompress,   aocl_lzma_setup,   aocl_lzma_destroy },
    { "aocl_bzip2",  "1.0.8",      aocl_bzip2_compressBound,  aocl_bzip2_compress,  aocl_bzip2_decompress,  aocl_bzip2_setup,  aocl_bzip2_destroy },
    { "aocl_snappy", "1.2.1",      aocl_snappy_compressBound, aocl_snappy_compress, aocl_snappy_decompress, aocl_snappy_setup, aocl_snappy_destroy },
    { "aocl_zlib",   "1.3.1",      aocl_zlib_compressBound,   aocl_zlib_compress,   aocl_zlib_decompress,   aocl_zlib_setup,   aocl_zlib_destroy },
    { "aocl_zstd",   "1.5.7",      aocl_zstd_compressBound,   aocl_zstd_compress,   aocl_zstd_decompress,   aocl_zstd_setup,   aocl_zstd_destroy }
};

#endif

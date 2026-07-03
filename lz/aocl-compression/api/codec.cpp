/**
 * Copyright (C) 2022-2024, Advanced Micro Devices. All rights reserved.
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
 
 /** @file codec.cpp
 *  
 *  @brief Wrapper functions of the supported native codec methods.
 *
 *  This file contains the wrapper functions of the supported compression and
 *  decompression methods calling their associated native APIs.
 *
 *  @author S. Biplab Raut
 */
 
#include "types.h"
#include "aocl_compression.h"
#include "codec.h"
#include "utils/utils.h"

//bzip2
#ifndef AOCL_EXCLUDE_BZIP2
#include "algos/bzip2/bzlib.h"
#endif
//lz4
#ifndef AOCL_EXCLUDE_LZ4
#include "algos/lz4/lz4.h"
#endif
//lz4hc
#if !defined(AOCL_EXCLUDE_LZ4HC) && !defined(AOCL_EXCLUDE_LZ4)
#include "algos/lz4/lz4hc.h"
#endif
//lzma
#ifndef AOCL_EXCLUDE_LZMA
#include <string.h>
#include "algos/lzma/Alloc.h"
#include "algos/lzma/LzmaDec.h"
#include "algos/lzma/LzmaEnc.h"
#endif
//snappy
#ifndef AOCL_EXCLUDE_SNAPPY
#include "algos/snappy/snappy.h"
#endif
//zlib
#ifndef AOCL_EXCLUDE_ZLIB
#include "algos/zlib/zlib.h"
#endif
//zstd
#ifndef AOCL_EXCLUDE_ZSTD
#include <cstdlib>
#define ZSTD_STATIC_LINKING_ONLY
#include "algos/zstd/lib/zstd.h"
#endif


#define CODEC_ERROR -1

//bzip2
#ifndef AOCL_EXCLUDE_BZIP2
AOCL_CHAR *aocl_bzip2_setup(AOCL_INTP optOff, AOCL_INTP optLevel,
                       AOCL_UINTP insize, AOCL_UINTP level, AOCL_UINTP windowLog)
{
    return aocl_setup_bzip2(optOff, optLevel, insize, level, windowLog);
}

AOCL_UINTP aocl_bzip2_compressBound(AOCL_UINTP insize)
{
    return BZ2_bzCompressBound(insize);
}

AOCL_INT64 aocl_bzip2_compress(AOCL_CHAR *inbuf, AOCL_UINTP insize, AOCL_CHAR *outbuf, 
						  AOCL_UINTP outsize, AOCL_UINTP level, AOCL_UINTP windowLog, AOCL_CHAR *)
{
    AOCL_UINT32 outSizeL = outsize;
    if (BZ2_bzBuffToBuffCompress((AOCL_CHAR *)outbuf, &outSizeL, (AOCL_CHAR *)inbuf, 
	   (AOCL_UINTP)insize, level, 0, 0)==BZ_OK)
        return outSizeL;

    return CODEC_ERROR;
}

AOCL_INT64 aocl_bzip2_decompress(AOCL_CHAR *inbuf, AOCL_UINTP insize, AOCL_CHAR *outbuf, 
							AOCL_UINTP outsize, AOCL_UINTP level, AOCL_UINTP, AOCL_CHAR *)
{
    AOCL_UINT32 outSizeL = outsize;
    if (BZ2_bzBuffToBuffDecompress((AOCL_CHAR *)outbuf, &outSizeL, (AOCL_CHAR *)inbuf, 
	   (AOCL_UINTP)insize, 0, 0)==BZ_OK)
        return outSizeL;

    return CODEC_ERROR;
}

AOCL_VOID aocl_bzip2_destroy(AOCL_CHAR* workmem) {
    aocl_destroy_bzip2();
}
#endif


//lz4
#ifndef AOCL_EXCLUDE_LZ4
AOCL_CHAR *aocl_lz4_setup(AOCL_INTP optOff, AOCL_INTP optLevel,
                     AOCL_UINTP insize, AOCL_UINTP level, AOCL_UINTP windowLog)
{
    return aocl_setup_lz4(optOff, optLevel, insize, level, windowLog);
}

AOCL_UINTP aocl_lz4_compressBound(AOCL_UINTP insize)
{
    AOCL_INT32 res = LZ4_compressBound(insize);
    if (res > 0)
        return res;
    
    return CODEC_ERROR;
}

#if defined(__GNUC__) && defined(__x86_64__)
/* Changes in code alignment affects performance of LZ4 compress
* functions. Aligning to 16-bytes boundary to fix this instability.*/
__asm__(".p2align 4");
#endif
AOCL_INT64 aocl_lz4_compress(AOCL_CHAR *inbuf, AOCL_UINTP insize, AOCL_CHAR *outbuf,
                        AOCL_UINTP outsize, AOCL_UINTP level, AOCL_UINTP, AOCL_CHAR *)
{
    AOCL_INT32 res = LZ4_compress_default(inbuf, outbuf, insize, outsize);
    if (res > 0)
        return res;
    
    return CODEC_ERROR;
}

#if defined(__GNUC__) && defined(__x86_64__)
/* Changes in code alignment affects performance of LZ4 decompress
* functions. Aligning to fix instability in decompression speed.*/
__asm__(".p2align 4");
__asm__("nop");
__asm__(".p2align 5");
#endif
AOCL_INT64 aocl_lz4_decompress(AOCL_CHAR *inbuf, AOCL_UINTP insize, AOCL_CHAR *outbuf,
                          AOCL_UINTP outsize, AOCL_UINTP level, AOCL_UINTP, AOCL_CHAR *)
{
    AOCL_INT32 res = LZ4_decompress_safe(inbuf, outbuf, insize, outsize);
    if (res >= 0)
        return res;

    return CODEC_ERROR;
}

AOCL_VOID aocl_lz4_destroy(AOCL_CHAR* workmem) {
    aocl_destroy_lz4();
}
#endif


//lz4hc
#if !defined(AOCL_EXCLUDE_LZ4HC) && !defined(AOCL_EXCLUDE_LZ4)
AOCL_CHAR *aocl_lz4hc_setup(AOCL_INTP optOff, AOCL_INTP optLevel,
                       AOCL_UINTP insize, AOCL_UINTP level, AOCL_UINTP windowLog)
{
    return aocl_setup_lz4hc(optOff, optLevel, insize, level, windowLog);
}

AOCL_UINTP aocl_lz4hc_compressBound(AOCL_UINTP insize)
{
    AOCL_INT32 res = LZ4_compressBound(insize);
    if (res > 0)
        return res;
    
    return CODEC_ERROR;
}

AOCL_INT64 aocl_lz4hc_compress(AOCL_CHAR *inbuf, AOCL_UINTP insize, AOCL_CHAR *outbuf,
                          AOCL_UINTP outsize, AOCL_UINTP level, AOCL_UINTP, AOCL_CHAR *)
{
    AOCL_INT32 res = LZ4_compress_HC(inbuf, outbuf, insize, outsize, level);
    if (res > 0)
        return res;

    return CODEC_ERROR;
}

AOCL_INT64 aocl_lz4hc_decompress(AOCL_CHAR *inbuf, AOCL_UINTP insize, AOCL_CHAR *outbuf,
							AOCL_UINTP outsize, AOCL_UINTP, AOCL_UINTP, AOCL_CHAR *)
{
    AOCL_INT32 res = LZ4_decompress_safe(inbuf, outbuf, insize, outsize);
    if (res >= 0)
        return res;

    return CODEC_ERROR;
}

AOCL_VOID aocl_lz4hc_destroy(AOCL_CHAR* workmem) {
    aocl_destroy_lz4hc();
}
#endif


//lzma
#ifndef AOCL_EXCLUDE_LZMA
AOCL_CHAR *aocl_lzma_setup(AOCL_INTP optOff, AOCL_INTP optLevel,
                      AOCL_UINTP insize, AOCL_UINTP level, AOCL_UINTP windowLog)
{
    aocl_setup_lzma_encode(optOff, optLevel, insize, level, windowLog);
    aocl_setup_lzma_decode(optOff, optLevel, insize, level, windowLog);
    return NULL;
}

AOCL_UINTP aocl_lzma_compressBound(AOCL_UINTP insize)
{
    return Lzma_compressBound(insize);
}

AOCL_INT64 aocl_lzma_compress(AOCL_CHAR *inbuf, AOCL_UINTP insize, AOCL_CHAR *outbuf,
                         AOCL_UINTP outsize, AOCL_UINTP level, AOCL_UINTP, AOCL_CHAR *)
{
    CLzmaEncProps encProps;
    AOCL_INTP res;
    AOCL_UINTP headerSize = LZMA_PROPS_SIZE;
    SizeT outLen = outsize - LZMA_PROPS_SIZE;
	
    LzmaEncProps_Init(&encProps);
    encProps.level = level;

    res = LzmaEncode((AOCL_UINT8 *)outbuf+LZMA_PROPS_SIZE, &outLen, (AOCL_UINT8 *)inbuf, 
                     insize, &encProps, (AOCL_UINT8 *)outbuf, &headerSize, 0, NULL, 
                     &g_Alloc, &g_Alloc);
	if (res == SZ_OK)
        return LZMA_PROPS_SIZE + outLen;
    
    return CODEC_ERROR;
}

AOCL_INT64 aocl_lzma_decompress(AOCL_CHAR *inbuf, AOCL_UINTP insize, AOCL_CHAR *outbuf,
						   AOCL_UINTP outsize, AOCL_UINTP, AOCL_UINTP, AOCL_CHAR *)
{
    AOCL_INTP res = SZ_OK;
    SizeT outLen = outsize;
    SizeT srcLen = insize - LZMA_PROPS_SIZE;
    ELzmaStatus status = LZMA_STATUS_NOT_SPECIFIED;
	
    res = LzmaDecode((AOCL_UINT8 *)outbuf, &outLen, (AOCL_UINT8 *)inbuf+LZMA_PROPS_SIZE, 
                     &srcLen, (AOCL_UINT8 *)inbuf, LZMA_PROPS_SIZE, LZMA_FINISH_END,
                     &status, &g_Alloc);
    if (res == SZ_OK ||
        (res == SZ_ERROR_INPUT_EOF && status == LZMA_STATUS_NEEDS_MORE_INPUT 
            && outLen > 0) /* decompression successful, but outsize is > expected output length */)
        return outLen;

    return CODEC_ERROR;
}

AOCL_VOID aocl_lzma_destroy(AOCL_CHAR* workmem) {
    aocl_destroy_lzma_encode();
    aocl_destroy_lzma_decode();
}
#endif


#ifndef AOCL_EXCLUDE_SNAPPY
AOCL_CHAR *aocl_snappy_setup(AOCL_INTP optOff, AOCL_INTP optLevel,
                        AOCL_UINTP insize, AOCL_UINTP level, AOCL_UINTP windowLog)
{
    return snappy::aocl_setup_snappy(optOff, optLevel, insize, level, windowLog);
}

AOCL_UINTP aocl_snappy_compressBound(AOCL_UINTP insize)
{
    AOCL_UINTP res = snappy::MaxCompressedLength(insize);
    if (res > 0)
        return res;
    
    return CODEC_ERROR;
}

AOCL_INT64 aocl_snappy_compress(AOCL_CHAR *inbuf, AOCL_UINTP insize, AOCL_CHAR *outbuf, 
						   AOCL_UINTP outsize, AOCL_UINTP, AOCL_UINTP, AOCL_CHAR *)
{
    AOCL_UINTP max_compressed_length = snappy::MaxCompressedLength(insize);
    if (outsize < max_compressed_length) {
        LOG_FORMATTED(ERR, logCtx, "Insufficient outsize: %zu < MaxCompressedLength: %zu",
            outsize, max_compressed_length);
        return CODEC_ERROR;
    }
    // RawCompress modifies the value of the 4th parameter after successful
    // completion of compression, but returns early on error. Hence, to detect
    // failure we can set the value of the last parameter to something that is
    // not a valid "compressed length" so that after it returns, that value can
    // be checked. If the value remains invalid after RawCompress returns, it
    // should indicate failure.

    // by definition, any value greater than the "max compressed length" is an
    // invalid compressed length, so we choose the value one more than the
    // max_compressed_length here.
    outsize = max_compressed_length + 1;
    snappy::RawCompress(inbuf, insize, outbuf, &outsize);
    if (outsize <= max_compressed_length)
        return outsize;

    return CODEC_ERROR;
}

AOCL_INT64 aocl_snappy_decompress(AOCL_CHAR* inbuf, AOCL_UINTP insize, AOCL_CHAR* outbuf,
    AOCL_UINTP outsize, AOCL_UINTP, AOCL_UINTP, AOCL_CHAR*)
{
    AOCL_UINTP uncompressed_len;
#ifdef AOCL_ENABLE_THREADS
    if (!snappy::GetUncompressedLengthFromMTCompressedBuffer(inbuf, insize, &uncompressed_len) || outsize < uncompressed_len) {
        LOG_UNFORMATTED(ERR, logCtx, "Invalid uncompressed_len");
        return CODEC_ERROR;
    }
#else
    if (!snappy::GetUncompressedLength(inbuf, insize, &uncompressed_len) || outsize < uncompressed_len) {
        LOG_UNFORMATTED(ERR, logCtx, "Invalid uncompressed_len");
        return CODEC_ERROR;
    }
#endif
    bool res = snappy::RawUncompress(inbuf, insize, outbuf);
    if (res)
        return uncompressed_len;

    return CODEC_ERROR;
}

AOCL_VOID aocl_snappy_destroy(AOCL_CHAR* workmem) {
    snappy::aocl_destroy_snappy();
}
#endif


#ifndef AOCL_EXCLUDE_ZLIB
AOCL_CHAR *aocl_zlib_setup(AOCL_INTP optOff, AOCL_INTP optLevel,
                      AOCL_UINTP insize, AOCL_UINTP level, AOCL_UINTP windowLog)
{
    return aocl_setup_zlib (optOff, optLevel, insize, level, windowLog);
}

AOCL_UINTP aocl_zlib_compressBound(AOCL_UINTP insize)
{
    AOCL_UINTP res = compressBound(insize);
    if (res > 0)
        return res;
    
    return CODEC_ERROR;
}

AOCL_INT64 aocl_zlib_compress(AOCL_CHAR *inbuf, AOCL_UINTP insize, AOCL_CHAR *outbuf,
                         AOCL_UINTP outsize, AOCL_UINTP level, AOCL_UINTP, AOCL_CHAR *)
{
    uLongf zencLen = outsize;
    AOCL_INTP res = compress2((AOCL_UINT8 *)outbuf, &zencLen, 
                         (AOCL_UINT8 *)inbuf, insize, level);
    if (res == Z_OK)
        return zencLen;
    
    return CODEC_ERROR;
}

AOCL_INT64 aocl_zlib_decompress(AOCL_CHAR *inbuf, AOCL_UINTP insize, AOCL_CHAR *outbuf, 
						   AOCL_UINTP outsize, AOCL_UINTP, AOCL_UINTP, AOCL_CHAR *)
{
    uLongf zdecLen = outsize;
    AOCL_INTP res = uncompress((AOCL_UINT8*)outbuf, &zdecLen, (AOCL_UINT8 *)inbuf, insize);
    if (res == Z_OK)
        return zdecLen;

    return CODEC_ERROR;
}

AOCL_VOID aocl_zlib_destroy(AOCL_CHAR* workmem) {
    aocl_destroy_zlib();
}
#endif


#ifndef AOCL_EXCLUDE_ZSTD
typedef struct {
    ZSTD_CCtx *cctx;
    ZSTD_DCtx *dctx;
    ZSTD_CDict *cdict;
    ZSTD_parameters zparams;
    ZSTD_customMem cmem;
} zstd_params_t;
AOCL_CHAR *aocl_zstd_setup(AOCL_INTP optOff, AOCL_INTP optLevel,
                      AOCL_UINTP insize, AOCL_UINTP level, AOCL_UINTP windowLog)
{
    zstd_params_t *zstd_params = (zstd_params_t *) 
    malloc(sizeof(zstd_params_t));

    aocl_setup_zstd_encode(optOff, optLevel, insize, level, windowLog);
    aocl_setup_zstd_decode(optOff, optLevel, insize, level, windowLog);
    
    if (!zstd_params) {
        LOG_UNFORMATTED(ERR, logCtx, "zstd_params_t allocation failed");
        return NULL;
    }
    zstd_params->cctx = ZSTD_createCCtx();
    zstd_params->dctx = ZSTD_createDCtx();
    zstd_params->cdict = NULL;

    return (AOCL_CHAR*) zstd_params;
}

AOCL_VOID aocl_zstd_destroy(AOCL_CHAR *workmem)
{
    aocl_destroy_zstd_encode();
    aocl_destroy_zstd_decode();

    zstd_params_t *zstd_params = (zstd_params_t *) workmem;
    if (!zstd_params) {
        LOG_UNFORMATTED(ERR, logCtx, "zstd_params_t deallocation failed");
        return;
    }
    if (zstd_params->cctx)
		ZSTD_freeCCtx(zstd_params->cctx);
    if (zstd_params->dctx)
		ZSTD_freeDCtx(zstd_params->dctx);
    if (zstd_params->cdict)
		ZSTD_freeCDict(zstd_params->cdict);
    free(workmem);
}

AOCL_UINTP aocl_zstd_compressBound(AOCL_UINTP insize)
{
    AOCL_UINTP res = ZSTD_compressBound(insize);
    if (!ZSTD_isError(res))
        return res;

    return CODEC_ERROR;
}

AOCL_INT64 aocl_zstd_compress(AOCL_CHAR *inbuf, AOCL_UINTP insize, AOCL_CHAR *outbuf,
                         AOCL_UINTP outsize, AOCL_UINTP level, AOCL_UINTP windowLog,
                         AOCL_CHAR *workmem)
{
    AOCL_UINTP res;
    zstd_params_t *zstd_params = (zstd_params_t *) workmem;
    
    if (!zstd_params || !zstd_params->cctx) {
        LOG_UNFORMATTED(ERR, logCtx, "zstd_params_t or context invalid");
        return 0;
    }

    zstd_params->zparams = ZSTD_getParams(level, insize, 0);
    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, level);
    zstd_params->zparams.fParams.contentSizeFlag = 1;

    if (windowLog && zstd_params->zparams.cParams.windowLog > windowLog) {
        zstd_params->zparams.cParams.windowLog = windowLog;
        zstd_params->zparams.cParams.chainLog = windowLog + 
			((zstd_params->zparams.cParams.strategy == ZSTD_btlazy2) || 
			(zstd_params->zparams.cParams.strategy == ZSTD_btopt) || 
			(zstd_params->zparams.cParams.strategy == ZSTD_btultra));
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    res = ZSTD_compress_advanced(zstd_params->cctx, outbuf, outsize, inbuf, 
                                insize, NULL, 0, zstd_params->zparams);
#pragma GCC diagnostic pop

    if (!ZSTD_isError(res))
        return res;

    return CODEC_ERROR;
}

AOCL_INT64 aocl_zstd_decompress(AOCL_CHAR *inbuf, AOCL_UINTP insize, AOCL_CHAR *outbuf, 
						   AOCL_UINTP outsize, AOCL_UINTP, AOCL_UINTP, AOCL_CHAR *workmem)
{
    AOCL_UINTP res;
    zstd_params_t *zstd_params = (zstd_params_t *) workmem;
    if (!zstd_params || !zstd_params->dctx) {
        LOG_UNFORMATTED(ERR, logCtx, "zstd_params_t or context invalid");
        return CODEC_ERROR;
    }

    res = ZSTD_decompressDCtx(zstd_params->dctx, outbuf, outsize, 
                              inbuf, insize);
    if (!ZSTD_isError(res))
        return res;

    return CODEC_ERROR;
}
#endif

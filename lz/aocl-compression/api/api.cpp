/**
 * Copyright (C) 2022-2026, Advanced Micro Devices. All rights reserved.
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
 
 /** @file api.c
 *  
 *  @brief Interface APIs and data structures of AOCL Compression library
 *
 *  This file contains the unified interface API set and associated
 *  data structure.
 *
 *  @author S. Biplab Raut
 */
 
#include "types.h"
#include "utils/utils.h"
#include "aocl_compression.h"
#include "codec.h"
#ifdef AOCL_ENABLE_THREADS
#include "threads/threads.h"
#endif

//Unified API function to get compressBound based on the input size
AOCL_INT64 aocl_llc_compressBound(aocl_compression_type codec_type,
                            AOCL_UINTP inSize)
{
    AOCL_INT64 ret;

    LOG_UNFORMATTED(TRACE, logCtx, "Enter");

    LOG_FORMATTED(INFO, logCtx,
       "Calling compressBound for method: %s", aocl_codec[codec_type].codec_name);

    ret = aocl_codec[codec_type].compressBound (inSize);
    
    LOG_UNFORMATTED(TRACE, logCtx, "Exit");  
    if (ret < 0)
        return ERR_COMPRESSION_FAILED;
        
    return ret;
}

//Unified API function to compress the input
AOCL_INT64 aocl_llc_compress(aocl_compression_desc *handle,
                        aocl_compression_type codec_type)
{
    AOCL_INT64 ret;
#ifdef WIN32
    timer clkTick;
#endif
    timeVal startTime, endTime;

    LOG_UNFORMATTED(TRACE, logCtx, "Enter");

    LOG_FORMATTED(INFO, logCtx,
       "Calling compression method: %s", aocl_codec[codec_type].codec_name);
    initTimer(clkTick);
    getTime(startTime);
    
    ret = aocl_codec[codec_type].compress (handle->inBuf,
                                          handle->inSize,
                                          handle->outBuf,
                                          handle->outSize,
                                          handle->level,
                                          handle->optVar,
                                          handle->workBuf);
    
    getTime(endTime);
    if (handle->measureStats == 1)
    {
        handle->cSize = ret;
        handle->cTime = diffTime(clkTick, startTime, endTime);
        handle->cSpeed = (handle->inSize * 1000.0) / handle->cTime;
    }
    
    LOG_UNFORMATTED(TRACE, logCtx, "Exit");

    if (ret < 0)
        return ERR_COMPRESSION_FAILED;
        
    return ret;
}

//Unified API function to decompress the input
AOCL_INT64 aocl_llc_decompress(aocl_compression_desc *handle,
                          aocl_compression_type codec_type)
{
    AOCL_INT64 ret;
#ifdef WIN32
    timer clkTick;
#endif
    timeVal startTime, endTime;
    
    LOG_UNFORMATTED(TRACE, logCtx, "Enter");

    LOG_FORMATTED(INFO, logCtx,
       "Calling decompression method: %s", aocl_codec[codec_type].codec_name);
    initTimer(clkTick);
    getTime(startTime);
    
    ret = aocl_codec[codec_type].decompress (handle->inBuf,
                                            handle->inSize,
                                            handle->outBuf,
                                            handle->outSize,
                                            handle->level,
                                            handle->optVar,
                                            handle->workBuf);
    
    getTime(endTime);
    if (handle->measureStats == 1)
    {
        handle->dSize = ret;
        handle->dTime = diffTime(clkTick, startTime, endTime);
        handle->dSpeed = (handle->dSize * 1000.0) / handle->dTime;
    }

    LOG_UNFORMATTED(TRACE, logCtx, "Exit");

    if (ret < 0)
        return ERR_COMPRESSION_FAILED;
        
    return ret;
}

//API to setup and initialize memory for the compression method
AOCL_INT32 aocl_llc_setup(aocl_compression_desc *handle,
                    aocl_compression_type codec_type)
{

    LOG_UNFORMATTED(TRACE, logCtx, "Enter");

    if ((codec_type < LZ4) || (codec_type >= AOCL_COMPRESSOR_ALGOS_NUM))
    {
        LOG_UNFORMATTED(ERR, logCtx,
            "setup failed !! compression method is not supported.");
        return ERR_UNSUPPORTED_METHOD;
    }

    LOG_FORMATTED(INFO, logCtx,
       "All optimizations are turned %s", (handle->optOff ? "off" : "on"));

    set_cpu_opt_flags((AOCL_VOID *)handle);

    LOG_FORMATTED(INFO, logCtx,
       "Calling setup method for: %s", aocl_codec[codec_type].codec_name);

    if (aocl_codec[codec_type].setup)
    {
        handle->workBuf = aocl_codec[codec_type].setup (handle->optOff,
                                                        handle->optLevel,
                                                        handle->inSize,
                                                        handle->level,
                                                        handle->optVar);
    }
    else
    {
        LOG_UNFORMATTED(ERR, logCtx,
            "setup failed !! compression method is excluded from this library build.");
        LOG_UNFORMATTED(TRACE, logCtx, "Exit");
        return ERR_EXCLUDED_METHOD;
    }

    LOG_UNFORMATTED(TRACE, logCtx, "Exit");
    return 0;
}

//API to destroy memory and deinit the compression method
AOCL_VOID aocl_llc_destroy(aocl_compression_desc *handle,
                      aocl_compression_type codec_type)
{
    LOG_UNFORMATTED(TRACE, logCtx, "Enter");

    LOG_FORMATTED(INFO, logCtx,
       "Calling destroy method for: %s", aocl_codec[codec_type].codec_name);

    if (aocl_codec[codec_type].destroy)
    {
        aocl_codec[codec_type].destroy(handle->workBuf);
    }

    LOG_UNFORMATTED(TRACE, logCtx, "Exit");
}

AOCL_INT32 aocl_llc_skip_rap_frame(AOCL_CHAR* src, AOCL_INT32 src_size)
{
    LOG_UNFORMATTED(TRACE, logCtx, "Enter");
    AOCL_INT32 ret = 0;

#ifdef AOCL_ENABLE_THREADS
    ret = aocl_skip_rap_frame_mt(src, src_size);
#endif

    LOG_UNFORMATTED(TRACE, logCtx, "Exit");
    return ret;

}

AOCL_INT32 aocl_llc_set_max_threads(AOCL_INT32 max_threads)
{
    LOG_UNFORMATTED(TRACE, logCtx, "Enter");

#ifdef AOCL_ENABLE_THREADS
    AOCL_INT32 ret = aocl_set_max_threads_mt(max_threads);
#else
    AOCL_INT32 ret = ERR_UNSUPPORTED_METHOD;
#endif

    LOG_UNFORMATTED(TRACE, logCtx, "Exit");
    return ret;
}

//API to return the compression library version string
const AOCL_CHAR *aocl_llc_version(AOCL_VOID)
{
    return (AOCL_COMPRESSION_LIBRARY_VERSION " " AOCL_BUILD_VERSION);
}

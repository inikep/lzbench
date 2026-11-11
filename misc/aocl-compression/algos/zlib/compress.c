/* compress.c -- compress a memory buffer
 * Copyright (C) 1995-2026 Jean-loup Gailly, Mark Adler
 * Modifications Copyright (C) 2023-2026, Advanced Micro Devices. All rights reserved.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* @(#) $Id$ */

#define ZLIB_INTERNAL
#include "zlib.h"
#include "utils/utils.h"
#include "utils/dispatcher.h"

#ifdef AOCL_ZLIB_OPT
#include "aocl_zlib_utils.h"
#include "aocl_zlib_setup.h"

int zlibOptOff = 0; // default, run reference code
static int setup_ok_zlib = 0; // flag to indicate status of dynamic dispatcher setup
#if !defined(AOCL_ENABLE_THREADS) || defined(AOCL_USE_TBB)
static atomic_flag setup_zlib = ATOMIC_FLAG_INIT;
#endif /* AOCL_ENABLE_THREADS */

/* Dynamic dispatcher setup function for native APIs.
 * All native APIs that call aocl optimized functions within their call stack,
 * must call AOCL_SETUP_NATIVE() at the start of the function. This sets up 
 * appropriate code paths to take based on user defined environment variables,
 * as well as cpu instruction set supported by the runtime machine. */
static void aocl_setup_native(void);
#define AOCL_SETUP_NATIVE() aocl_setup_native()
#else
#define AOCL_SETUP_NATIVE()
#endif /* AOCL_ZLIB_OPT */

/* AOCL-Compression defined setup function that sets up ZLIB with the right
*  AMD optimized zlib routines depending upon the CPU features. */
ZEXTERN char * ZEXPORT aocl_setup_zlib(int optOff, int optLevel, int insize,
    int level, int windowLog)
{
#ifdef AOCL_ZLIB_OPT
    AOCL_ENTER_CRITICAL(setup_zlib)
    if (!setup_ok_zlib) {
        optOff = optOff ? 1 : get_disable_opt_flags(0);
        zlibOptOff = optOff;
        CpuFeatures cpuFeatures = Dispatcher_GetSupportedFeaturesForLevel(Dispatcher_IntToLevel((int)optLevel));
        aocl_setup_deflate(optOff, cpuFeatures);
        aocl_setup_inflate(optOff, cpuFeatures);
        aocl_setup_adler32(optOff, cpuFeatures);
        setup_ok_zlib = 1;
    }
    AOCL_EXIT_CRITICAL(setup_zlib)
#endif /* AOCL_ZLIB_OPT */
    return NULL;
}

#ifdef AOCL_ZLIB_OPT
static void aocl_setup_native(void) {
    AOCL_ENTER_CRITICAL(setup_zlib)
    if (!setup_ok_zlib) {
        int optOff = get_disable_opt_flags(0);
        zlibOptOff = optOff;
        CpuFeatures cpuFeatures = Dispatcher_GetFeaturesFromEnv();
        aocl_setup_deflate(optOff, cpuFeatures);
        aocl_setup_inflate(optOff, cpuFeatures);
        aocl_setup_adler32(optOff, cpuFeatures);
        setup_ok_zlib = 1;
    }
    AOCL_EXIT_CRITICAL(setup_zlib)
}
#endif

ZEXTERN void ZEXPORT aocl_destroy_zlib (void) {
#ifdef AOCL_ZLIB_OPT
    AOCL_ENTER_CRITICAL(setup_zlib)
    setup_ok_zlib = 0;
    AOCL_EXIT_CRITICAL(setup_zlib)
    aocl_destroy_adler32();
    aocl_destroy_deflate();
    aocl_destroy_inflate();
#endif /* AOCL_ZLIB_OPT */
}

#if defined(AOCL_UNIT_TEST)
ZEXTERN void ZEXPORT test_aocl_zlib_set_enable_dquick(int val) {
    if (val)
        set_env_var("AOCL_ZLIB_QUICK_MODE", "1");
    else
        unset_env_var("AOCL_ZLIB_QUICK_MODE");
}

#ifdef AOCL_ZLIB_OPT
/* Getter function for zlibOptOff variable for unit tests */
ZEXTERN int ZEXPORT test_aocl_zlib_get_zlibOptOff(void)
{
    return zlibOptOff;
}
#endif /* AOCL_ZLIB_OPT */

#endif /* AOCL_UNIT_TEST */

#ifdef AOCL_ENABLE_THREADS
#define ZLIB_MT_WINDOW_LEN (32768 << 1)
#include <string.h>
#include "threads/threads.h"
#endif
/* ===========================================================================
     Compresses the source buffer into the destination buffer. The level
   parameter has the same meaning as in deflateInit.  sourceLen is the byte
   length of the source buffer. Upon entry, destLen is the total size of the
   destination buffer, which must be at least 0.1% larger than sourceLen plus
   12 bytes. Upon exit, destLen is the actual size of the compressed buffer.

     compress2 returns Z_OK if success, Z_MEM_ERROR if there was not enough
   memory, Z_BUF_ERROR if there was not enough room in the output buffer,
   Z_STREAM_ERROR if the level parameter is invalid.

     The _z versions of the functions take size_t length arguments.
*/

#ifdef AOCL_ENABLE_THREADS
static inline int compress2_ST_raw(aocl_thread_info_t *cThread, int level, int finalFlush) {
    Bytef *dest = (Bytef*)cThread->dst_trap;
    z_size_t *destLen = (z_size_t*)&(cThread->dst_trap_size);
    Bytef *source = (Bytef*)cThread->partition_src;
    z_size_t sourceLen = cThread->partition_src_size;

    if(destLen == NULL)
    {
        return Z_BUF_ERROR;
    }
    
    z_stream stream;
    int err;
    const uInt max = (uInt)-1;
    z_size_t left;

    left = *destLen;
    *destLen = 0;

    stream.zalloc = (alloc_func)0;
    stream.zfree = (free_func)0;
    stream.opaque = (voidpf)0;

    err = deflateInit2(&stream, level, Z_DEFLATED,
                         -1 * MAX_WBITS, 8, Z_DEFAULT_STRATEGY);

    if (err != Z_OK) return err;

    stream.next_out = dest;
    stream.avail_out = 0;
    stream.next_in = (z_const Bytef *)source;
    stream.avail_in = 0;

    do {
        if (stream.avail_out == 0) {
            stream.avail_out = left > (uLong)max ? max : (uInt)left;
            left -= stream.avail_out;
        }
        if (stream.avail_in == 0) {
            stream.avail_in = sourceLen > (uLong)max ? max : (uInt)sourceLen;
            sourceLen -= stream.avail_in;
        }
        err = deflate(&stream, finalFlush);
        if(finalFlush != Z_FINISH && sourceLen == 0)
            break;
    } while (err == Z_OK);

    *destLen = stream.total_out;
    deflateEnd(&stream);
    return err == Z_STREAM_END ? Z_OK : err;
}

static z_size_t compressBound_ST_raw(z_size_t sourceLen) {
    /* Worst case: each byte -> 9 bits (fixed Huffman deflate). */
    z_size_t fixed_size = FIXED_HUFFFMAN_COMPRESSED_SIZE(sourceLen);

    /* stored_size: size with stored deflate (no compression). Adds 5 bytes/block (worst case as per deflate specification).
       Assumes default memLevel/windowbits. */
    z_size_t stored_size = STORED_ZLIB_COMPRESSED_SIZE(sourceLen);
    if(aocl_zlib_get_enable_dquick()) {
        return (fixed_size > stored_size) ? fixed_size : stored_size;
    }
    return stored_size;
}

static z_size_t compressBound_MT_generic(z_size_t sourceLen, const int wrap) {

    z_size_t sz1 = compressBound_ST_raw(sourceLen);
    z_size_t sz2 = 0;
    uInt wrapper_size = 0;
    COMPRESS_BOUND_MT(sourceLen, compressBound_ST_raw, ZLIB_MT_WINDOW_LEN, WINDOW_FACTOR, sz1, sz2, 0)

    if(wrap == 1)
    {
        wrapper_size = 13; // zlib
    }
#ifdef GZIP
    else if(wrap == 2)
    {
        wrapper_size = 18; // gzip
    }
#endif
    
    return sz2 + wrapper_size;
}

typedef struct {
    aocl_thread_group_t *thread_group_handle;
    int level;
    int wrap;
} compress2_mt_compress_parallel_ctx_t;

static void compress2_mt_compress_parallel_worker(AOCL_UINT32 thread_id, void *context)
{
    compress2_mt_compress_parallel_ctx_t *ctx = (compress2_mt_compress_parallel_ctx_t *)context;
    aocl_thread_group_t *thread_group_handle = ctx->thread_group_handle;
    int level = ctx->level;
    int wrap = ctx->wrap;
    aocl_thread_info_t cur_thread_info;

#ifdef AOCL_THREADS_LOG
    printf("Compress Thread [id: %d] : Inside parallel region\n", (int)thread_id);
#endif
    AOCL_UINT32 cmpr_bound_pad;
    AOCL_UINT32 is_error = Z_OK;
    if (thread_id != (thread_group_handle->num_threads - 1))
        cmpr_bound_pad = compressBound_ST_raw(thread_group_handle->common_part_src_size);
    else
        cmpr_bound_pad = compressBound_ST_raw(thread_group_handle->common_part_src_size +
                                thread_group_handle->leftover_part_src_bytes);
    if (aocl_do_partition_compress_mt(thread_group_handle, &cur_thread_info, cmpr_bound_pad, thread_id) == 0)
    {
        if (thread_id != (thread_group_handle->num_threads - 1))
            is_error = compress2_ST_raw(&cur_thread_info, level, Z_SYNC_FLUSH);
        else
            is_error = compress2_ST_raw(&cur_thread_info, level, Z_FINISH);
        cur_thread_info.last_bytes_len = CALCULATE_CHECKSUM(cur_thread_info.partition_src, cur_thread_info.partition_src_size, wrap);
    }
#ifdef AOCL_THREADS_LOG
    printf("Compress Thread [id: %d] : Return value %d\n", (int)thread_id, is_error);
#endif
    thread_group_handle->threads_info_list[thread_id].partition_src = cur_thread_info.partition_src;
    thread_group_handle->threads_info_list[thread_id].dst_trap = cur_thread_info.dst_trap;
    thread_group_handle->threads_info_list[thread_id].additional_state_info = NULL;
    thread_group_handle->threads_info_list[thread_id].dst_trap_size = cur_thread_info.dst_trap_size;
    thread_group_handle->threads_info_list[thread_id].partition_src_size = cur_thread_info.partition_src_size;
    thread_group_handle->threads_info_list[thread_id].last_bytes_len = cur_thread_info.last_bytes_len;
    thread_group_handle->threads_info_list[thread_id].is_error = is_error;
    thread_group_handle->threads_info_list[thread_id].num_child_threads = 0;
}

typedef struct {
    aocl_thread_group_t *thread_group_handle;
} compress2_mt_memcpy_parallel_ctx_t;

static void compress2_mt_memcpy_parallel_worker(AOCL_UINT32 thread_id, void *context)
{
    compress2_mt_memcpy_parallel_ctx_t *ctx = (compress2_mt_memcpy_parallel_ctx_t *)context;
    aocl_thread_group_t *thread_group_handle = ctx->thread_group_handle;
    aocl_thread_info_t cur_thread_info;

    cur_thread_info = thread_group_handle->threads_info_list[thread_id];
    memcpy(thread_group_handle->dst + cur_thread_info.partition_src_size,
        cur_thread_info.dst_trap, cur_thread_info.dst_trap_size);
}

static inline int compress2_MT_generic(Bytef *dest, z_size_t *destLen, const Bytef *source,
                      z_size_t sourceLen, int level, const int wrap) {

    if ((*destLen) < compressBound_MT_generic(sourceLen, wrap))
        RETURN_DST_SIZE_LESS_THAN_COMPRESSBOUND_ERROR_MT(Z_BUF_ERROR)

    int result = Z_OK;
    aocl_thread_group_t thread_group_handle;
    aocl_thread_info_t cur_thread_info;
    AOCL_INT32 rap_metadata_len = -1;
    AOCL_UINT32 thread_cnt = 0;

    if ((sourceLen > 0 && source == NULL) ||
        destLen == NULL || (*destLen > 0 && dest == NULL))
        return Z_STREAM_ERROR;

    rap_metadata_len = aocl_setup_parallel_compress_mt(&thread_group_handle, (char *)source,
                                                 (char *)dest, sourceLen, *destLen,
                                                 ZLIB_MT_WINDOW_LEN, WINDOW_FACTOR);
    if (rap_metadata_len < 0)
        return Z_MEM_ERROR;

    if (thread_group_handle.num_threads == 1)
    {
        int header_size = insert_Header_generic(dest, level, wrap);
        cur_thread_info.partition_src = (char *)source;
        cur_thread_info.dst_trap = (char *)dest + header_size;
        cur_thread_info.dst_trap_size = (*destLen) - header_size;
        cur_thread_info.partition_src_size = sourceLen;
        result = compress2_ST_raw(&cur_thread_info, level, Z_FINISH);
        AOCL_UINT32 chcksm = CALCULATE_CHECKSUM(source, sourceLen, wrap);
        int trailer_size = insert_Trailer_generic(dest + header_size + cur_thread_info.dst_trap_size,
                        chcksm, sourceLen, wrap);
        *destLen = cur_thread_info.dst_trap_size + header_size + trailer_size;
        return result;
    }
    else
    {
        *destLen = rap_metadata_len;
#ifdef AOCL_THREADS_LOG
#ifndef AOCL_USE_TBB
        printf("Compress Thread [id: %d] : Before parallel region\n", omp_get_thread_num());
#endif
#endif
        {
            compress2_mt_compress_parallel_ctx_t compress_parallel_ctx;
            compress_parallel_ctx.thread_group_handle = &thread_group_handle;
            compress_parallel_ctx.level = level;
            compress_parallel_ctx.wrap = wrap;
            aocl_parallel_for(thread_group_handle.num_threads, compress2_mt_compress_parallel_worker,
                &compress_parallel_ctx);
        }
#ifdef AOCL_THREADS_LOG
#ifndef AOCL_USE_TBB
        printf("Compress Thread [id: %d] : After parallel region\n", omp_get_thread_num());
#endif
#endif
        //Post processing in single-threaded mode: Prepares RAP frame and joins the last sequences of the neighboring threads

        // <-- RAP Header -->
        //Add at the start of the stream : Although it can be at the end or at any other point in the stream, but it is more easier for parsing at the start
        AOCL_CHAR* dst_org = thread_group_handle.dst;
        AOCL_CHAR* dst_ptr = dst_org;
        AOCL_UINT32 decomp_len;
        thread_group_handle.dst += rap_metadata_len;
        dst_ptr += RAP_START_OF_PARTITIONS;
        // <-- RAP Header -->

        int header_size = insert_Header_generic((Bytef*)thread_group_handle.dst, level, wrap);
        thread_group_handle.dst += header_size;
        *destLen += header_size;

        AOCL_UINT32 checksum = (wrap == 1 ? 1 : 0);

        // <-- RAP Metadata payload -->
        for (thread_cnt = 0 ; thread_cnt < thread_group_handle.num_threads; thread_cnt++)
        {
            cur_thread_info = thread_group_handle.threads_info_list[thread_cnt];
            //In case of any thread partitioning or alloc errors, exit the compression process with error
            if (cur_thread_info.is_error || cur_thread_info.dst_trap_size < 0)
            {
                result = cur_thread_info.is_error;
                aocl_destroy_parallel_compress_mt(&thread_group_handle);
    #ifdef AOCL_THREADS_LOG
                printf("Compress Thread [id: %d] : Encountered ERROR\n", thread_cnt);
    #endif
                return result;
            }

            checksum = (AOCL_UINT32) UPDATE_CHECKSUM(checksum, cur_thread_info.last_bytes_len, cur_thread_info.partition_src_size, wrap);

            *(AOCL_UINT32*)dst_ptr = *destLen; //For storing this thread's RAP offset
            dst_ptr += RAP_OFFSET_BYTES;
            *(AOCL_INT32*)dst_ptr = cur_thread_info.dst_trap_size; //For storing this thread's RAP length
            dst_ptr += RAP_LEN_BYTES;
            //For storing this thread's decompressed (src) length
            decomp_len = cur_thread_info.partition_src_size;

            *(AOCL_INT32*)dst_ptr = decomp_len;
            dst_ptr += DECOMP_LEN_BYTES;
            *destLen += (cur_thread_info.dst_trap_size);

            /* compute cumulative dst_trap_size and save in unsued member partition_src_size
            * This is used as offset to indicate starting points of compressed data blocks in dst */
            if(thread_cnt != 0)
                thread_group_handle.threads_info_list[thread_cnt].partition_src_size =
                    thread_group_handle.threads_info_list[thread_cnt - 1].partition_src_size +
                    thread_group_handle.threads_info_list[thread_cnt - 1].dst_trap_size;
            else
                thread_group_handle.threads_info_list[thread_cnt].partition_src_size = 0;

        }

        /* copy compressed data from threads to dst multi-threaded */
        {
            compress2_mt_memcpy_parallel_ctx_t memcpy_parallel_ctx;
            memcpy_parallel_ctx.thread_group_handle = &thread_group_handle;
            /* Copy-out has no per-partition status: a swallowed backend
             * exception must fail, not return a truncated buffer as success. */
            if (aocl_parallel_for(thread_group_handle.num_threads, compress2_mt_memcpy_parallel_worker,
                &memcpy_parallel_ctx) != 0)
            {
                LOG_UNFORMATTED(ERR, logCtx, "Multi-threaded compress copy-out failed");
                aocl_destroy_parallel_compress_mt(&thread_group_handle);
                return Z_MEM_ERROR;
            }
        }
        thread_group_handle.dst += *destLen - rap_metadata_len - header_size;
        int trailer_size = insert_Trailer_generic((Bytef*)thread_group_handle.dst, checksum, sourceLen, wrap);
        *destLen += trailer_size;
        thread_group_handle.dst += trailer_size;

        aocl_destroy_parallel_compress_mt(&thread_group_handle);

        LOG_UNFORMATTED(INFO, logCtx, "Exit");
        return result;
    }
}
#endif /* AOCL_ENABLE_THREADS */

z_size_t ZEXPORT compressBound_gzip(z_size_t sourceLen) {
#ifdef AOCL_ENABLE_THREADS
    LOG_UNFORMATTED(TRACE, logCtx, "Enter");
    AOCL_SETUP_NATIVE();
    return compressBound_MT_generic(sourceLen, 2);
#else
    LOG_UNFORMATTED(ERR, logCtx, "Only supported with multithread library.");
    return Z_VERSION_ERROR;
#endif
}

int ZEXPORT compress2_gzip(Bytef *dest, z_size_t *destLen, const Bytef *source,
                      z_size_t sourceLen, int level) {
#ifdef AOCL_ENABLE_THREADS
    LOG_UNFORMATTED(TRACE, logCtx, "Enter");
    AOCL_SETUP_NATIVE();
    if(destLen == NULL)
    {
        LOG_UNFORMATTED(INFO, logCtx, "Exit");
        return Z_STREAM_ERROR;
    }
    return compress2_MT_generic(dest, destLen, source, sourceLen, level, 2);
#else
    LOG_UNFORMATTED(ERR, logCtx, "Only supported with multithread library.");
    return Z_VERSION_ERROR;
#endif
}

int ZEXPORT compress2_raw(Bytef *dest, z_size_t *destLen, const Bytef *source,
                      z_size_t sourceLen, int level) {
#ifdef AOCL_ENABLE_THREADS
    LOG_UNFORMATTED(TRACE, logCtx, "Enter");
    AOCL_SETUP_NATIVE();
    if(destLen == NULL)
    {
        LOG_UNFORMATTED(INFO, logCtx, "Exit");
        return Z_STREAM_ERROR;
    }
    return compress2_MT_generic(dest, destLen, source, sourceLen, level, 0);
#else
    LOG_UNFORMATTED(ERR, logCtx, "Only supported with multithread library.");
    return Z_VERSION_ERROR;
#endif
}

int ZEXPORT compress2_z(Bytef *dest, z_size_t *destLen, const Bytef *source,
                        z_size_t sourceLen, int level) {
    LOG_UNFORMATTED(TRACE, logCtx, "Enter");
    AOCL_SETUP_NATIVE();
#ifndef AOCL_ENABLE_THREADS //Non threaded
    z_stream stream;
    int err;
    const uInt max = (uInt)-1;
    z_size_t left;

    if ((sourceLen > 0 && source == NULL) ||
        destLen == NULL || (*destLen > 0 && dest == NULL)) {
        LOG_UNFORMATTED(INFO, logCtx, "Exit");
        return Z_STREAM_ERROR;
    }

    left = *destLen;
    *destLen = 0;

    stream.zalloc = (alloc_func)0;
    stream.zfree = (free_func)0;
    stream.opaque = (voidpf)0;

    err = deflateInit(&stream, level);
    if (err != Z_OK)
    {
        LOG_UNFORMATTED(INFO, logCtx, "Exit");
        return err;
    }

    stream.next_out = dest;
    stream.avail_out = 0;
    stream.next_in = (z_const Bytef *)source;
    stream.avail_in = 0;

    do {
        if (stream.avail_out == 0) {
            stream.avail_out = left > (z_size_t)max ? max : (uInt)left;
            left -= stream.avail_out;
        }
        if (stream.avail_in == 0) {
            stream.avail_in = sourceLen > (z_size_t)max ? max :
                                                          (uInt)sourceLen;
            sourceLen -= stream.avail_in;
        }
        err = deflate(&stream, sourceLen ? Z_NO_FLUSH : Z_FINISH);
    } while (err == Z_OK);

    *destLen = (z_size_t)(stream.next_out - dest);
    deflateEnd(&stream);
    LOG_UNFORMATTED(INFO, logCtx, "Exit");
    
    return err == Z_STREAM_END ? Z_OK : err;
#else //Threaded
    return compress2_MT_generic(dest, destLen, source, sourceLen, level, 1);  
#endif /* !AOCL_ENABLE_THREADS */
}
int ZEXPORT compress2(Bytef *dest, uLongf *destLen, const Bytef *source,
                      uLong sourceLen, int level) {
    int ret;
    if(destLen == NULL)
        return Z_STREAM_ERROR;
    z_size_t got = *destLen;
    ret = compress2_z(dest, &got, source, sourceLen, level);
    *destLen = (uLong)got;
    return ret;
}
/* ===========================================================================
 */
int ZEXPORT compress_z(Bytef *dest, z_size_t *destLen, const Bytef *source,
                       z_size_t sourceLen) {
    return compress2_z(dest, destLen, source, sourceLen,
                       Z_DEFAULT_COMPRESSION);
}
int ZEXPORT compress(Bytef *dest, uLongf *destLen, const Bytef *source,
                     uLong sourceLen) {
    return compress2(dest, destLen, source, sourceLen, Z_DEFAULT_COMPRESSION);
}

/* ===========================================================================
     If the default memLevel or windowBits for deflateInit() is changed, then
   this function needs to be updated.
 */

z_size_t ZEXPORT compressBound_z(z_size_t sourceLen) {
    AOCL_SETUP_NATIVE();
#ifdef AOCL_ENABLE_THREADS
    return compressBound_MT_generic(sourceLen, 1);
#elif defined(AOCL_ZLIB_OPT)
    /* Worst case: each byte -> 9 bits (fixed Huffman deflate). 13 bytes for zlib wrapper + safety. */
    uLong fixed_size = FIXED_HUFFFMAN_COMPRESSED_SIZE(sourceLen) + 13;
    /* stored_size: size with stored deflate (no compression). Adds 5 bytes/block (worst case as per deflate specification).
       Assumes default memLevel/windowbits. */
    uLong stored_size = STORED_ZLIB_COMPRESSED_SIZE(sourceLen);

    if(aocl_zlib_get_enable_dquick()) {
        return (fixed_size > stored_size) ? fixed_size : stored_size;
    }
    return stored_size;
#else
    z_size_t bound = sourceLen + (sourceLen >> 12) + (sourceLen >> 14) +
                     (sourceLen >> 25) + 13;
    return bound < sourceLen ? (z_size_t)-1 : bound;
#endif
}
uLong ZEXPORT compressBound(uLong sourceLen) {
    z_size_t bound = compressBound_z(sourceLen);
    return (uLong)bound != bound ? (uLong)-1 : (uLong)bound;
}

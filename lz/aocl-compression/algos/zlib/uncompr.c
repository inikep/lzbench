/* uncompr.c -- decompress a memory buffer
 * Copyright (C) 1995-2003, 2010, 2014, 2016 Jean-loup Gailly, Mark Adler
 * Modifications Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* @(#) $Id$ */

#define ZLIB_INTERNAL
#include "zlib.h"

#include "utils/utils.h"

#ifdef AOCL_ENABLE_THREADS
#include <string.h>
#include "threads/threads.h"
#include "aocl_zlib_utils.h"
#define MAX_WBITS 15
#endif
/* ===========================================================================
     Decompresses the source buffer into the destination buffer.  *sourceLen is
   the byte length of the source buffer. Upon entry, *destLen is the total size
   of the destination buffer, which must be large enough to hold the entire
   uncompressed data. (The size of the uncompressed data must have been saved
   previously by the compressor and transmitted to the decompressor by some
   mechanism outside the scope of this compression library.) Upon exit,
   *destLen is the size of the decompressed data and *sourceLen is the number
   of source bytes consumed. Upon return, source + *sourceLen points to the
   first unused input byte.

     uncompress returns Z_OK if success, Z_MEM_ERROR if there was not enough
   memory, Z_BUF_ERROR if there was not enough room in the output buffer, or
   Z_DATA_ERROR if the input data was corrupted, including if the input data is
   an incomplete zlib stream.
*/
#ifdef AOCL_ENABLE_THREADS

static inline int uncompress2_ST_raw(Bytef *dest, uLongf *destLen, const Bytef *source,
                        uLong *sourceLen, short winBits) {
    z_stream stream;
    int err;
    const uInt max = (uInt)-1;
    uLong len, left;
    Byte buf[1];    /* for detection of incomplete stream when *destLen == 0 */

    len = *sourceLen;
    if (*destLen) {
        left = *destLen;
        *destLen = 0;
    }
    else {
        left = 1;
        dest = buf;
    }

    stream.next_in = (z_const Bytef *)source;
    stream.avail_in = 0;
    stream.zalloc = (alloc_func)0;
    stream.zfree = (free_func)0;
    stream.opaque = (voidpf)0;

    err = inflateInit2(&stream, winBits);
    if (err != Z_OK) return err;

    stream.next_out = dest;
    stream.avail_out = 0;

    do {
        if (stream.avail_out == 0) {
            stream.avail_out = left > (uLong)max ? max : (uInt)left;
            left -= stream.avail_out;
        }
        if (stream.avail_in == 0) {
            stream.avail_in = len > (uLong)max ? max : (uInt)len;
            len -= stream.avail_in;
        }

        err = inflate(&stream, Z_NO_FLUSH);
    } while (err == Z_OK);

    *sourceLen -= len + stream.avail_in;
    if (dest != buf)
        *destLen = stream.total_out;
    else if (stream.total_out && err == Z_BUF_ERROR)
        left = 1;

    inflateEnd(&stream);
    return err == Z_STREAM_END ? Z_OK :
           err == Z_NEED_DICT ? Z_DATA_ERROR  :
           err == Z_BUF_ERROR && left + stream.avail_out ? Z_DATA_ERROR :
           err;
}
static inline int validate_Checksum(AOCL_UINT32 checksum, const Bytef *source,
                        uLong *sourceLen, const int wrap) {
    int isValid = 1; 
    if(wrap == 1)
    {   // zlib
        AOCL_UINT32 adler2 = *((AOCL_UINT32 *)(source + *sourceLen - 4));
        adler2 = ((((adler2) >> 24) & 0xff) + (((adler2) >> 8) & 0xff00) + (((adler2) & 0xff00) << 8) + (((adler2) & 0xff) << 24));
        if(checksum != adler2)
            isValid = 0;
    }
#ifdef GZIP
    else if(wrap == 2) 
    {   // gzip
        AOCL_UINT32 crc2 = *((AOCL_UINT32 *)(source + *sourceLen - 8));
        if(checksum != crc2)
            isValid = 0;
    }
#endif
    return isValid;
}
static inline int uncompress2_MT_generic(Bytef *dest, uLongf *destLen, const Bytef *source,
                        uLong *sourceLen, const int wrap) {
    if(destLen == NULL)
    {
        return Z_BUF_ERROR;
    }
    if(dest == NULL)
    {
        return Z_STREAM_ERROR;
    }
    if(sourceLen == NULL || source == NULL)
    {
        return Z_DATA_ERROR;
    }
    int result = Z_OK;
    aocl_thread_group_t thread_group_handle;
    aocl_thread_info_t cur_thread_info;
    AOCL_INT32 use_ST_decompressor = 0;
    AOCL_UINT32 thread_cnt = 0;
    AOCL_INT32 rap_metadata_len = 0;
    int header_size = 0, trailer_size = 0;

    if(wrap == 1)
    {   // zlib
        if(*sourceLen < 6)
            return Z_DATA_ERROR;
        header_size = 2;
        trailer_size = 4;
    }
#ifdef GZIP
    else if (wrap == 2)
    {   // gzip
        if(*sourceLen < 18)
            return Z_DATA_ERROR;
        header_size = 10;
        trailer_size = 8;
    }
#endif
    uLong org_sourceLen = *sourceLen;
    rap_metadata_len = aocl_setup_parallel_decompress_mt(&thread_group_handle, (char *)source, (char *)dest,
                                                   *sourceLen, *destLen, use_ST_decompressor);

    if(rap_metadata_len < 0)
        return Z_MEM_ERROR;

    if (AOCL_MT_PARTITIONS_NOT_FOUND(thread_group_handle))
    {
        source += (rap_metadata_len + header_size); // skip RAP frame and header
        *sourceLen -= (rap_metadata_len + header_size + trailer_size);
        org_sourceLen -= (rap_metadata_len + header_size);
        result = uncompress2_ST_raw(dest, destLen, source, sourceLen, -1 * MAX_WBITS);
        if(result == Z_OK)
        {
            AOCL_UINT32 chksm = CALCULATE_CHECKSUM(dest, *destLen, wrap);
            if(!validate_Checksum(chksm, source, &org_sourceLen, wrap))
                result = Z_DATA_ERROR;
        }
        return result;

    }
    else
    {
#ifdef AOCL_THREADS_LOG
        printf("Decompress Thread [id: %d] : Before parallel region\n", omp_get_thread_num());
#endif
#pragma omp parallel private(cur_thread_info) shared(thread_group_handle) num_threads(thread_group_handle.num_threads)
        {
#ifdef AOCL_THREADS_LOG
            printf("Decompress Thread [id: %d] : Inside parallel region\n", omp_get_thread_num());
#endif
            AOCL_UINT32 is_error = 1;
            AOCL_UINT32 thread_id = omp_get_thread_num();
            AOCL_INT32 thread_parallel_res = 0;

            AOCL_MT_PROCESS_PARTITION_START(thread_group_handle, ti_cur, thread_id)
            thread_parallel_res = aocl_do_partition_decompress_mt(&thread_group_handle, 
                &cur_thread_info, AOCL_MT_CUR_THREAD_SERIAL_ID(ti_cur));
            ti_cur->additional_state_info = NULL;

            if (thread_parallel_res == AOCL_MT_DECOMP_PARTITION_SUCCESS)
            {
                is_error = uncompress2_ST_raw((Bytef *)cur_thread_info.dst_trap, (uLong *)&(cur_thread_info.dst_trap_size),
                                            (Bytef *)cur_thread_info.partition_src, (uLong *)&(cur_thread_info.partition_src_size), -1 * MAX_WBITS);
                cur_thread_info.last_bytes_len = CALCULATE_CHECKSUM(cur_thread_info.dst_trap, cur_thread_info.dst_trap_size, wrap);
            }//aocl_do_partition_decompress_mt
            else if (thread_parallel_res == AOCL_MT_DECOMP_PARTITION_EMPTY_SRC)
            {
                is_error = 0;
            }
            else // thread_parallel_res == AOCL_MT_DECOMP_PARTITION_ERR_INSUFFICIENT_DST_SPACE
            {
                // uncompress2_ST_raw already returns values from 0 to 2, hence for identifying an error 3 is used.
                is_error = 3;
            }
#ifdef AOCL_THREADS_LOG
            printf("Decompress Thread [id: %d] : Return value %d\n", omp_get_thread_num(), is_error);
#endif
            ti_cur->partition_src = cur_thread_info.partition_src;
            ti_cur->dst_trap = cur_thread_info.dst_trap;
            ti_cur->dst_trap_size = cur_thread_info.dst_trap_size;
            ti_cur->partition_src_size = cur_thread_info.partition_src_size;
            ti_cur->last_bytes_len = cur_thread_info.last_bytes_len; // storing checksum value
            ti_cur->is_error = is_error;
            ti_cur->num_child_threads = 0;
            AOCL_MT_PROCESS_PARTITION_END(ti_cur)

        }//#pragma omp parallel
#ifdef AOCL_THREADS_LOG
        printf("Decompress Thread [id: %d] : After parallel region\n", omp_get_thread_num());
#endif


        /* This block iterates through all threads to check for errors and compute the final checksum. */
        AOCL_UINT32 checksum = (wrap == 1 ? 1 : 0);
        for (thread_cnt = 0; thread_cnt < thread_group_handle.num_threads; thread_cnt++)
        {
            AOCL_MT_PROCESS_PARTITION_START(thread_group_handle, ti_cur, thread_cnt)
            //In case of any thread partitioning or alloc errors, exit the decompression process with error
            if (ti_cur->is_error && ti_cur->is_error != Z_BUF_ERROR)
            {
                result = ti_cur->is_error;
                aocl_destroy_parallel_decompress_mt(&thread_group_handle);
#ifdef AOCL_THREADS_LOG
                printf("Decompress Thread [id: %d] : Encountered ERROR\n", thread_cnt);
#endif
                LOG_FORMATTED(ERR, logCtx, "Decompress Thread [id: %d] : Encountered ERROR", thread_cnt);
                if(ti_cur->is_error == 3)
                    return Z_BUF_ERROR;
                return result;
            }
            result = ti_cur->is_error;
            
            checksum = UPDATE_CHECKSUM(checksum, ti_cur->last_bytes_len, ti_cur->dst_trap_size, wrap);
            AOCL_MT_PROCESS_PARTITION_END(ti_cur)
        }

        // verify uncompressed data integrity
        if(result == Z_OK && !validate_Checksum(checksum, source, &org_sourceLen, wrap))
            result = Z_DATA_ERROR;

        aocl_destroy_parallel_decompress_mt(&thread_group_handle);

        return result;
    }
}
#endif

int ZEXPORT uncompress2_gzip(Bytef *dest, uLongf *destLen, const Bytef *source,
                        uLong *sourceLen) {
#ifdef AOCL_ENABLE_THREADS
    LOG_UNFORMATTED(TRACE, logCtx, "Enter");
    return uncompress2_MT_generic(dest, destLen, source, sourceLen, 2);
#else
    LOG_UNFORMATTED(ERR, logCtx, "Only supported with multithread library.");
    return Z_VERSION_ERROR;
#endif
}

int ZEXPORT uncompress2_raw(Bytef *dest, uLongf *destLen, const Bytef *source,
                        uLong *sourceLen) {
#ifdef AOCL_ENABLE_THREADS
    LOG_UNFORMATTED(TRACE, logCtx, "Enter");
    return uncompress2_MT_generic(dest, destLen, source, sourceLen, 0);
#else
    LOG_UNFORMATTED(ERR, logCtx, "Only supported with multithread library.");
    return Z_VERSION_ERROR;
#endif
}

int ZEXPORT uncompress2(Bytef *dest, uLongf *destLen, const Bytef *source,
                        uLong *sourceLen) {
#ifndef AOCL_ENABLE_THREADS //Non threaded
    if(destLen == NULL)
    {
        return Z_BUF_ERROR;
    }
    else if(sourceLen == NULL)
    {
        return Z_DATA_ERROR;
    }
    
    z_stream stream;
    int err;
    const uInt max = (uInt)-1;
    uLong len, left;
    Byte buf[1];    /* for detection of incomplete stream when *destLen == 0 */

    len = *sourceLen;
    if (*destLen) {
        left = *destLen;
        *destLen = 0;
    }
    else {
        left = 1;
        dest = buf;
    }

    stream.next_in = (z_const Bytef *)source;
    stream.avail_in = 0;
    stream.zalloc = (alloc_func)0;
    stream.zfree = (free_func)0;
    stream.opaque = (voidpf)0;

    err = inflateInit(&stream);
    if (err != Z_OK) return err;

    stream.next_out = dest;
    stream.avail_out = 0;

    do {
        if (stream.avail_out == 0) {
            stream.avail_out = left > (uLong)max ? max : (uInt)left;
            left -= stream.avail_out;
        }
        if (stream.avail_in == 0) {
            stream.avail_in = len > (uLong)max ? max : (uInt)len;
            len -= stream.avail_in;
        }
        err = inflate(&stream, Z_NO_FLUSH);
    } while (err == Z_OK);

    *sourceLen -= len + stream.avail_in;
    if (dest != buf)
        *destLen = stream.total_out;
    else if (stream.total_out && err == Z_BUF_ERROR)
        left = 1;

    inflateEnd(&stream);
    return err == Z_STREAM_END ? Z_OK :
           err == Z_NEED_DICT ? Z_DATA_ERROR  :
           err == Z_BUF_ERROR && left + stream.avail_out ? Z_DATA_ERROR :
           err;
#else //Threaded
    return uncompress2_MT_generic(dest, destLen, source, sourceLen, 1);
#endif /* !AOCL_ENABLE_THREADS */
}

int ZEXPORT uncompress(Bytef *dest, uLongf *destLen, const Bytef *source,
                       uLong sourceLen) {
    LOG_UNFORMATTED(TRACE, logCtx, "Enter");
    int ret = uncompress2(dest, destLen, source, &sourceLen);
    LOG_UNFORMATTED(INFO, logCtx, "Exit");
    return ret;
}

/**
 * Copyright (C) 2023-2026, Advanced Micro Devices. All rights reserved.
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

 /** @file threads.c
 *
 *  @brief Multi-threaded compression and decompression support via SMP threads
 *
 *  This file contains the functions to setup, partition and destroy the compression
 *  decompression methods.
 *
 *  @author S. Biplab Raut
 */

#include <stdlib.h>
#include <assert.h>
#include "api/types.h"
#include "api/aocl_compression.h"
#include "threads.h"
#include "utils/utils.h"

#ifdef AOCL_UNIT_TEST
int(*test_omp_get_max_threads_fp)(void) = omp_get_max_threads;
#endif

static AOCL_INT32 aocl_user_max_threads_mt = 0;
#pragma omp threadprivate(aocl_user_max_threads_mt)

AOCL_INT32 aocl_set_max_threads_mt(AOCL_INT32 max_threads)
{
    if (max_threads <= 0)
    {
        LOG_UNFORMATTED(ERR, logCtx, "Invalid input");
        return ERR_INVALID_INPUT;
    }

    aocl_user_max_threads_mt = max_threads;
    return 0;
}

AOCL_UINT32 aocl_get_max_threads_mt(void)
{
#ifndef AOCL_UNIT_TEST
    AOCL_UINT32 max_threads = omp_get_max_threads();
#else
    AOCL_UINT32 max_threads = test_omp_max_threads_get();
#endif

    if (aocl_user_max_threads_mt > 0 && ((AOCL_UINT32)aocl_user_max_threads_mt < max_threads))
        return (AOCL_UINT32)aocl_user_max_threads_mt;

    return max_threads;
}

AOCL_INT32 aocl_setup_partition_internal(aocl_thread_group_t *thread_grp, 
                                      AOCL_CHAR *src, AOCL_CHAR *dst, AOCL_UINTP in_size,
                                      AOCL_UINTP out_size, AOCL_INT32 window_len,
                                      AOCL_INT32 window_factor)
{
    assert(thread_grp != NULL);
    if (window_len <= 0 || window_factor <= 0) 
    {
        LOG_UNFORMATTED(ERR, logCtx, "Invalid input");
        return ERR_INVALID_INPUT;
    }

    AOCL_UINT32 max_threads = aocl_get_max_threads_mt();
    AOCL_UINTP chunk_size =  (AOCL_UINTP)window_len * window_factor;

    thread_grp->src = src;
    thread_grp->dst = dst;
    thread_grp->src_size = in_size;
    thread_grp->dst_size = out_size;
    thread_grp->search_window_length = window_len;
    thread_grp->threads_info_list = NULL;

    if (thread_grp->src_size < chunk_size)
    {
        LOG_UNFORMATTED(DEBUG, logCtx, "Src size is small. Running on single thread.");
        //Single threaded execution for very small stream (< window_len * window_factor)
        thread_grp->num_threads = 1;
        thread_grp->common_part_src_size = thread_grp->src_size;
        thread_grp->leftover_part_src_bytes = 0;
    }
    else
    {
        //Find number of partitions in the src stream
        AOCL_UINTP num_parallel_partitions = thread_grp->src_size / chunk_size;
        AOCL_UINTP leftover_size = thread_grp->src_size % chunk_size;
        
        //Sufficiently large leftover bytes adds another thread for processing
        if (leftover_size >= chunk_size >> 1)
            num_parallel_partitions++;

        //Find number of threads to process the number of parallel partitions
        if (max_threads >= num_parallel_partitions)
            //Use only upto num_parallel_partitions threads
            thread_grp->num_threads = num_parallel_partitions;
        else
            //Use all the available threads
            thread_grp->num_threads = max_threads;
        LOG_FORMATTED(INFO, logCtx, "Number of threads set to %u", thread_grp->num_threads);

        //Tentative partitioning. Actual partitioning is done at thread level
        thread_grp->common_part_src_size = thread_grp->src_size /
                                                thread_grp->num_threads;
        thread_grp->leftover_part_src_bytes = thread_grp->src_size %
                                                thread_grp->num_threads;
    }
    return 0;
}

AOCL_INT32 aocl_setup_parallel_compress_mt(aocl_thread_group_t *thread_grp, 
                                      AOCL_CHAR *src, AOCL_CHAR *dst, AOCL_UINTP in_size,
                                      AOCL_UINTP out_size, AOCL_INT32 window_len,
                                      AOCL_INT32 window_factor)
{
    if (dst == NULL) 
    {
        LOG_UNFORMATTED(ERR, logCtx, "Invalid input");
        return ERR_INVALID_INPUT;
    }

    AOCL_INT32 res = aocl_setup_partition_internal(thread_grp, src, dst, in_size, out_size, window_len, window_factor);
    if (res != 0)
        return res;

#ifdef AOCL_THREADS_LOG
    printf("Input stream size: [%td], common_part_src_size: [%ld], leftover_part_src_bytes: [%ld]\n",
        thread_grp->src_size, thread_grp->common_part_src_size, thread_grp->leftover_part_src_bytes);
    printf("Number of max threads: [%d], Number of threads set for execution: [%d]\n",
        omp_get_max_threads(), thread_grp->num_threads);
#endif

    AOCL_INT32 rap_frame_len = 0;
    if (thread_grp->num_threads == 1)
        return rap_frame_len;

    //Allocate threads list to hold references to threads_info
    thread_grp->threads_info_list = (aocl_thread_info_t*)malloc(
                    sizeof(aocl_thread_info_t) * thread_grp->num_threads);
                    
    if (thread_grp->threads_info_list == NULL) {
        LOG_UNFORMATTED(ERR, logCtx, "Memory allocation failed");
        return ERR_MEMORY_ALLOC;
    }
    
    memset(thread_grp->threads_info_list, 0, 
                    sizeof(aocl_thread_info_t) * thread_grp->num_threads);
    for (AOCL_UINT32 thread_id = 0; thread_id < thread_grp->num_threads; ++thread_id)
    {
        /* Set to 1 by default.
         * Reset to 0 when thread gets spawned and completes its task successfully.*/
        thread_grp->threads_info_list[thread_id].is_error = 1;
    }

    rap_frame_len = RAP_FRAME_LEN_WITH_DECOMP_LENGTH(thread_grp->num_threads, 0);
    *(AOCL_INT64*)dst = RAP_MAGIC_WORD; //For storing the magic word
    dst += RAP_MAGIC_WORD_BYTES;
    *(AOCL_UINT32*)dst = rap_frame_len; //For storing the total RAP frame length
    dst += RAP_METADATA_LEN_BYTES;
    *(AOCL_UINT32*)dst = thread_grp->num_threads; //For storing the no. of threads

    return rap_frame_len;
}

AOCL_INT32 aocl_do_partition_compress_mt(aocl_thread_group_t *thread_grp,
                                   aocl_thread_info_t *cur_thread_info,
                                   AOCL_UINTP cmpr_bound_pad, AOCL_UINT32 thread_id)
{
    assert(thread_grp != NULL);
    assert(cur_thread_info != NULL);
    cur_thread_info->partition_src = thread_grp->src + 
                                (thread_grp->common_part_src_size * thread_id);
    cur_thread_info->thread_id = thread_id;

    if (thread_id != (thread_grp->num_threads - 1))
        cur_thread_info->partition_src_size = thread_grp->common_part_src_size;
    else
        cur_thread_info->partition_src_size = thread_grp->common_part_src_size +
                                        thread_grp->leftover_part_src_bytes;
    
    cur_thread_info->dst_trap_size = cur_thread_info->partition_src_size +
                                        cmpr_bound_pad;

    cur_thread_info->dst_trap = (AOCL_CHAR *)malloc(cur_thread_info->dst_trap_size);

#ifdef AOCL_THREADS_LOG
    printf("aocl_do_partition_compress_mt(): thread id: [%d] dest_trap_size [%td]\n",
        cur_thread_info->thread_id, cur_thread_info->dst_trap_size);
#endif

    if (cur_thread_info->dst_trap == NULL) 
    {
        LOG_UNFORMATTED(ERR, logCtx, "Memory allocation failed");
        return ERR_MEMORY_ALLOC;
    }

    cur_thread_info->next = NULL;//Unused as of now

    return 0;
}

void aocl_destroy_parallel_compress_mt(aocl_thread_group_t *thread_grp)
{
    assert(thread_grp != NULL);
    if (thread_grp->threads_info_list)
    {
        AOCL_UINT32 thread_cnt;
        for (thread_cnt = 0; thread_cnt < thread_grp->num_threads; thread_cnt++)
        {
            if (thread_grp->threads_info_list[thread_cnt].dst_trap)
            {
                free(thread_grp->threads_info_list[thread_cnt].dst_trap);
                thread_grp->threads_info_list[thread_cnt].dst_trap = NULL;
            }
        }
        free(thread_grp->threads_info_list);
        thread_grp->threads_info_list = NULL;
    }
}

AOCL_INT32 aocl_setup_parallel_decompress_mt(aocl_thread_group_t* thread_grp,
                                        AOCL_CHAR* src, AOCL_CHAR* dst, AOCL_UINTP in_size,
                                        AOCL_UINTP out_size, AOCL_INT32 use_ST_decompressor)
{
    assert(thread_grp != NULL);
    if (src == NULL) 
    {
        LOG_UNFORMATTED(ERR, logCtx, "Invalid input");
        return ERR_INVALID_INPUT;
    }

    AOCL_CHAR* src_base;
    AOCL_UINT32 rap_metadata_len;
    AOCL_UINT32 max_threads = aocl_get_max_threads_mt();

    thread_grp->src = src;
    thread_grp->dst = dst;
    thread_grp->src_size = in_size;
    thread_grp->dst_size = out_size;
    thread_grp->threads_info_list = NULL;

    src_base = thread_grp->src;

    if ((thread_grp->src_size < RAP_MAGIC_WORD_BYTES) ||
        (RAP_MAGIC_WORD != *(AOCL_INT64*)src_base))
    {
        //Stream is very small or not in multi-threaded RAP format
        //Decompress in single threaded mode
        thread_grp->num_threads = 1;
        rap_metadata_len = 0;
    }
    else
    {
        AOCL_CHAR* src_ptr;
        AOCL_UINT32 num_main_threads;
        src_ptr = src_base + RAP_MAGIC_WORD_BYTES;
        rap_metadata_len = *(AOCL_UINT32*)(src_ptr);
        src_ptr += RAP_METADATA_LEN_BYTES;
        num_main_threads = *(AOCL_UINT32*)(src_ptr);

        if (num_main_threads == 0) 
        {
            LOG_UNFORMATTED(ERR, logCtx, "Invalid main thread count value in RAP frame");
            return -1; // invalid main thread count in stream. Must be >= 1.
        }

        if (use_ST_decompressor == 1)
            return rap_metadata_len;
        
        //Find number of threads to process the number of parallel partitions
        AOCL_UINT32 part_per_thread = 1;
        AOCL_UINT32 part_per_thread_rem = 0;
        if (max_threads >= num_main_threads)
            //Use exactly num_main_threads threads
            thread_grp->num_threads = num_main_threads;
        else
        {
            //When max_threads is lesser than num_main_threads, multi partitions are 
            //assigned to each thread as a linked list using next pointer
            thread_grp->num_threads = max_threads;
            part_per_thread = num_main_threads / max_threads;
            part_per_thread_rem = num_main_threads % max_threads;
        }
            
        LOG_FORMATTED(INFO, logCtx, "Number of threads set to %u", thread_grp->num_threads);

        if (thread_grp->num_threads == 1) //process in a single thread
            return rap_metadata_len;

        //Tentative partitioning. Actual partitioning is done at thread level
        assert(part_per_thread >= 1);
        thread_grp->common_part_src_size = thread_grp->dst_size /
            num_main_threads;
        thread_grp->leftover_part_src_bytes = thread_grp->dst_size %
            num_main_threads;

        //Allocate threads list to hold references to threads_info
        thread_grp->threads_info_list = (aocl_thread_info_t*)malloc(
            sizeof(aocl_thread_info_t) * num_main_threads);
        
        if (thread_grp->threads_info_list == NULL) 
            {
                LOG_UNFORMATTED(ERR, logCtx, "Memory allocation failed");
                return ERR_MEMORY_ALLOC;
            }
        
        memset(thread_grp->threads_info_list, 0, //needed to ensure pointer related checks behave as expected
            sizeof(aocl_thread_info_t) * num_main_threads);

        if (part_per_thread == 1 && part_per_thread_rem == 0) 
        { //max_threads >= num_main_threads
            for (AOCL_UINT32 thread_id = 0; thread_id < thread_grp->num_threads; ++thread_id) 
            {
                thread_grp->threads_info_list[thread_id].thread_id = thread_id;
                /* Set to 1 by default.
                * Reset to 0 when thread gets spawned and completes its task successfully.*/
                thread_grp->threads_info_list[thread_id].is_error = 1;
            }
        }
        else
        { //max_threads < num_main_threads 
            /*  
             * num_main_threads     : Threads used during compression
             * max_threads          : Threads available now for decompression
             * part_per_thread      : These many partitions must be evenly distributed among threads
             * part_per_thread_rem  : Assign 1 each to first 'part_per_thread_rem' threads
             * 
             * Example with num_main_threads = 6, max_threads (thread_grp->num_threads) = 2:
             * cpr partitions : P0, P01, P02, P1, P11, P12
             * dpr threads    : T0, T1
             * T0 needs to process : P0, P01, P02
             * T1 needs to process : P1, P11, P12
             * 
             * Data for each partition is stored in thread_grp->threads_info_list in this order:
             * | P0 | P1 | P01 | P02 | P11 | P12 | 
             * 
             * Assignment of partitions to threads (serial_thread_id):
             * dpr threads          : |  T0 |  T1 |  T0 |  T0 |  T1 |  T1 |
             * threads_info_list    : |  P0 |  P1 | P01 | P02 | P11 | P12 |
             * pointer to next node : | P01 | P11 | P02 |   - | P12 |   - |
             * serial_thread_id     : |   0 |   3 |   1 |   2 |   4 |   5 |
             * 
             * As partitions are assigned based on serial_thread_id in aocl_do_partition_decompress_mt(),
             * dpr threads will process consecutive partitions:
             * T0 : 0, 1, 2
             * T1 : 3, 4, 5
            */

            //First thread_grp->num_threads slots hold data for first set of partitions to be processed by each thread 
            aocl_thread_info_t* ti_ptr = thread_grp->threads_info_list + thread_grp->num_threads;
            //Subsequent slots are linked to members of thread_grp->threads_info_list
            AOCL_UINT32 part_rem = part_per_thread_rem;
            AOCL_UINT32 serial_thread_id = 0;
            for (AOCL_UINT32 thread_id = 0; thread_id < thread_grp->num_threads; ++thread_id) 
            {
                AOCL_UINT32 part_cur = part_per_thread - 1; //1 partition for head node, assign rest to linked list
                aocl_thread_info_t* ti_cur = &thread_grp->threads_info_list[thread_id];
                ti_cur->thread_id = serial_thread_id++;
                ti_cur->is_error = 1;
                while (part_cur--) 
                { //link consecutive partitions to linked list 
                    ti_cur->next = ti_ptr;
                    ti_cur = ti_cur->next;
                    ti_cur->thread_id = serial_thread_id++;
                    ti_cur->is_error = 1;
                    ti_ptr++;
                }
                if (part_rem) 
                { //link 1 additional partition
                    ti_cur->next = ti_ptr;
                    ti_cur = ti_cur->next;
                    ti_cur->thread_id = serial_thread_id++;
                    ti_cur->is_error = 1;
                    ti_ptr++;
                    part_rem--;
                }
                ti_cur->next = NULL;
            }
            assert(ti_ptr == (thread_grp->threads_info_list + num_main_threads));
        }

    }
    return rap_metadata_len;
}

// Returns the destination buffer offset for a thread by summing the decompressed sizes of previous threads
static AOCL_UINTP aocl_calculate_dst_offset_internal(AOCL_CHAR const *source, AOCL_INT32 thread_id)
{
   source = source + RAP_START_OF_PARTITIONS;
   AOCL_UINTP len = 0;
   for(AOCL_INT32 i = 0; i < thread_id; i++)
   {
      len += *(AOCL_UINT32 *)(source + RAP_DATA_BYTES);
      source += RAP_DATA_BYTES_WITH_DECOMP_LEN;
   }
   return len;
}

AOCL_INT32 aocl_do_partition_decompress_mt(const aocl_thread_group_t* thread_grp,
                                      aocl_thread_info_t* cur_thread_info, AOCL_UINT32 thread_id)
{
    assert(thread_grp != NULL);
    assert(cur_thread_info != NULL);
    memset(cur_thread_info, 0, sizeof(aocl_thread_info_t)); //needed to ensure pointer related checks behave as expected
        
    AOCL_UINT32 cur_rap_pos = RAP_START_OF_PARTITIONS + 
                            (thread_id * (RAP_DATA_BYTES_WITH_DECOMP_LEN));
    cur_thread_info->partition_src = thread_grp->src +
                            *(AOCL_UINT32*)(thread_grp->src + cur_rap_pos);
    cur_thread_info->partition_src_size = *(AOCL_UINT32*)(thread_grp->src +
                                            cur_rap_pos + RAP_OFFSET_BYTES);
    cur_thread_info->thread_id = thread_id;

    if (cur_thread_info->partition_src_size == 0)
    {
        LOG_FORMATTED(DEBUG, logCtx, "Partition size 0 for thread %u", thread_id);
        cur_thread_info->dst_trap = NULL;
        return AOCL_MT_DECOMP_PARTITION_EMPTY_SRC;
    }

    cur_thread_info->dst_trap_size = (*(AOCL_UINT32*)(thread_grp->src +
                                        cur_rap_pos + RAP_DATA_BYTES));
    AOCL_UINTP dst_offset = aocl_calculate_dst_offset_internal(thread_grp->src, thread_id);

    cur_thread_info->dst_trap = thread_grp->dst + dst_offset;

#ifdef AOCL_THREADS_LOG
    printf("aocl_do_partition_decompress_mt(): thread id: [%d]\n",
        cur_thread_info->thread_id);
#endif

    // Check if sufficent space is there in destination buffer.
    if(dst_offset + cur_thread_info->dst_trap_size > thread_grp->dst_size)
    {
        LOG_FORMATTED(ERR, logCtx, "Thread %u: Decompression failed, destination buffer too small.", thread_id);
        return AOCL_MT_DECOMP_PARTITION_ERR_INSUFFICIENT_DST_SPACE;
    }

    return AOCL_MT_DECOMP_PARTITION_SUCCESS;
}

void aocl_destroy_parallel_decompress_mt(aocl_thread_group_t* thread_grp)
{
    assert(thread_grp != NULL);

    if (thread_grp->threads_info_list)
    {
        free(thread_grp->threads_info_list);
        thread_grp->threads_info_list = NULL;
    }
}

AOCL_INT32 aocl_get_rap_frame_bound_mt(void) {
    AOCL_UINT32 max_threads = aocl_get_max_threads_mt();
    return RAP_FRAME_LEN_WITH_DECOMP_LENGTH(max_threads, 0); // upper bound of rap frame length in bytes based on max threads possible
}

AOCL_INT32 aocl_skip_rap_frame_mt(AOCL_CHAR* src, AOCL_UINTP src_size)
{
    if (src == NULL)
        return ERR_INVALID_INPUT;

    if ((src_size < RAP_MAGIC_WORD_BYTES) ||
        (RAP_MAGIC_WORD != *(AOCL_INT64*)src))
    {
        return 0; //Stream is very small or not in multi-threaded RAP format
    }
    else
    {
        AOCL_CHAR* src_ptr = src + RAP_MAGIC_WORD_BYTES;
        AOCL_UINT32 rap_metadata_len = *(AOCL_UINT32*)(src_ptr);
        return rap_metadata_len;
    }
}

AOCL_INT32 aocl_set_partition_stats_mt(aocl_thread_group_t *thread_grp,
                                    AOCL_UINTP in_size, AOCL_INT32 window_len, AOCL_INT32 window_factor)
{
    return aocl_setup_partition_internal(thread_grp, NULL /* src */, NULL /* dst */, in_size, 0 /* out_size */, window_len, window_factor);
}

#ifdef AOCL_UNIT_TEST
/* Functions to override omp_get_max_threads() for unit testing */
static int test_omp_max_threads = 1;
#ifndef AOCL_ENABLE_THREADS
static atomic_flag setup_test_omp_max_threads = ATOMIC_FLAG_INIT;
#endif
int omp_get_max_threads_manual(void) 
{
    return test_omp_max_threads;
}

int test_omp_max_threads_set(int max_threads) 
{
    int max_limit = omp_get_max_threads();
    if (max_threads > max_limit) 
    {
        return 0;
    }
    else 
    {
        AOCL_ENTER_CRITICAL(setup_test_omp_max_threads)
        test_omp_max_threads = max_threads;
        test_omp_get_max_threads_fp = omp_get_max_threads_manual;
        AOCL_EXIT_CRITICAL(setup_test_omp_max_threads)
        return 1;
    }
}

int test_omp_max_threads_get(void)
{
    int max_threads = 0;
    AOCL_ENTER_CRITICAL(setup_test_omp_max_threads)
    max_threads = test_omp_get_max_threads_fp();
    AOCL_EXIT_CRITICAL(setup_test_omp_max_threads)
    return max_threads;
}

void test_omp_max_threads_reset(void) 
{
    AOCL_ENTER_CRITICAL(setup_test_omp_max_threads)
    test_omp_max_threads = 1;
    test_omp_get_max_threads_fp = omp_get_max_threads;
    AOCL_EXIT_CRITICAL(setup_test_omp_max_threads)
}
#endif

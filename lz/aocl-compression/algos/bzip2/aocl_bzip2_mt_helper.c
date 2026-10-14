/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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
 
 /** @file aocl_bzip2_mt_helper.c
 *  
 *  @brief Helper functions for multithreaded compression/decompression functions.
 *
 *  @author Niranjan Reddy
 */

#include "aocl_bzip2_mt_helper.h"

// Calculates the bit padding needed by matching a fixed trail pattern in the data.
Int32 get_empty_bits(const UChar *compressed_data)
{
    // Reference trail pattern.
    const UChar trail[BZIP2_EOS_MAGIC_NUMBER_BYTES] = {0x17, 0x72, 0x45, 0x38, 0x50, 0x90};
    UChar temp_byte = 0;
    Int32 num_of_matches = 0;
    Int32 bit_padding = 0;
    // Adjust pointer to start before header.
    compressed_data -= BZIP2_EOS_BYTES+1;

    for (Int32 k = 0; k < 2; k++)
    {
        for (Int32 i = 0; i < 8; i++)
        {
            Int32 j = 0;
            while (j < BZIP2_EOS_MAGIC_NUMBER_BYTES)
            {
                // Reconstruct candidate byte with bit shifts.
                temp_byte = ((compressed_data[j] & ((1 << i) - 1)) << (8 - i)) | (compressed_data[j + 1] >> i);
                if (temp_byte != trail[j])
                    break;
                j++;
            }
            if (j == BZIP2_EOS_MAGIC_NUMBER_BYTES)
            {
                // Found a complete match: record padding and break.
                num_of_matches++;
                bit_padding += i;
                break;
            }
        }
        if (num_of_matches)
            break;
        /*
            No match: adjust pointer and add full byte padding.
            This case might occur only from 2nd block onwards in following scenarios:
            Lets say:
                prev block left "f" empty bits
                current block will leave "s" empty bits
            then if (f+s)>8, then that means last byte written is a useless byte. Hence we shift one byte back.
        */
        bit_padding += 8;
        compressed_data--;
    }
    if (num_of_matches != 1)
    {
        // Return error if no unique match is found.
        return -1;
    }
    return bit_padding;
}

// Buffers up to 32 bits and writes full bytes to output as they become available.
void append(Char **output, UInt32 c, Int32 bits, bit_stream *state)
{
    while (state->bits >= 8)
    {
        // Write the most significant byte from the buffer.
        **output = (state->buff >> 24) & 0xff;
        state->buff = (state->buff << 8);
        state->bits -= 8;
        (*output)++;
    }
    // Append new bits into the buffer.
    state->buff = state->buff | ((c & ((1 << bits) - 1)) << (32 - state->bits - bits));
    state->bits += bits;
}

// Flushes any remaining bits in the buffer to the output.
void finish_append(Char **output, bit_stream *state)
{
    while (state->bits > 0)
    {
        **output = (state->buff >> 24) & 0xff;
        state->buff = (state->buff << 8);
        state->bits -= 8;
        (*output)++;
    }
    // Reset the bit stream state.
    state->buff = 0;
    state->bits = 0;
}

void combine_checksum(UInt32 *checksum, UInt32 ans)
{
    *checksum = (*checksum << 1) | (*checksum >> 31);
    *checksum ^= ans;
}

void append_checksum(Char **output, UInt32 checksum, bit_stream *state)
{
    for(Int32 i=0; i < BZIP2_COMBINED_CRC_BYTES; i++)
    {
        append(output, (checksum >> ((3 - i) * 8)) & 0xff, 8, state);
    }
    finish_append(output, state);
}

// Computes a combined checksum from the individual checksums of all blocks handled by the current thread.
UInt32 bz_mt_cur_thread_checksum(mt_checksum_node * current, UInt32 checksum)
{
    while(current)
    {
        combine_checksum(&checksum, current->checksum);
        current = current->next;
    }
    return checksum;
}

void bz_mt_free_checksum_nodes(Int32 num_threads, mt_data_list *mt_head_table)
{
    for (Int32 i = 0; i < num_threads; i++)
    {
        mt_checksum_node *current = mt_head_table[i].head;
        while (current)
        {
            mt_checksum_node *temp = current;
            current = current->next;
            free(temp);
        }
        mt_head_table[i].head = NULL;
    }
}

/*
    Performs post-processing steps after multi-threaded BZIP2 compression.
    It combines the compressed data from each thread, calculates the final checksum, and appends it to the output.
*/
UInt32 aocl_bzip2_mt_post_processing(Char *dest, aocl_thread_group_t *thread_group_handle, mt_data_list* mt_head_table, Int32 rap_frame_length)
{
    Char * output = dest;
    UInt32 checksum = 0;
    bit_stream state = {0, 0};
    UInt32 offset = RAP_START_OF_PARTITIONS + (RAP_DATA_BYTES_WITH_DECOMP_LEN * thread_group_handle->num_threads);
    aocl_thread_info_t * thread_data_table = thread_group_handle->threads_info_list;

    *(AOCL_UINT32*)(&output[RAP_MAGIC_WORD_BYTES]) = rap_frame_length;
    output += RAP_START_OF_PARTITIONS;

    // A table is created, where each element represents, destination ptr, source ptr, and size of copy for all the threads.
    for (Int32 thread_id = 0; thread_id < thread_group_handle->num_threads; thread_id++)
    {
        AOCL_UINTP dst_size = thread_data_table[thread_id].dst_trap_size;

        if(thread_data_table[thread_id].is_error)
            return 0;

        if (thread_id != 0)
            dst_size -= BZIP2_HEADER_BYTES;

        // generate RAP data and write to corresponding location in destination buffer
        *(AOCL_UINT32*)output = offset;
        output += RAP_OFFSET_BYTES;
        *(AOCL_UINT32*)output = dst_size;
        output += RAP_LEN_BYTES;
        *(AOCL_UINT32*)output = thread_data_table[thread_id].partition_src_size;
        output += DECOMP_LEN_BYTES;

        thread_data_table[thread_id].partition_src_size = offset;
        thread_data_table[thread_id].dst_trap_size = dst_size;
        offset += dst_size;

        // Combine the checksums from all threads.
        checksum = bz_mt_cur_thread_checksum(mt_head_table[thread_id].head, checksum);
    }
    
    // Copy the data from all threads to dest.
    #pragma omp parallel shared(thread_group_handle) num_threads(thread_group_handle->num_threads)
    {
        Int32 thread_id = omp_get_thread_num();
        AOCL_CHAR *dst_single = thread_data_table[thread_id].dst_trap;
        AOCL_UINTP dst_size = thread_data_table[thread_id].dst_trap_size;

        if (thread_id != 0)
        {
            dst_single += BZIP2_HEADER_BYTES;
        }

        memcpy(dest+thread_data_table[thread_id].partition_src_size, dst_single, dst_size);
    }
    
    output = dest + offset;

    // Append final checksum.
    append_checksum(&output, checksum, &state);

    return output - dest;
}

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
 
 /** @file aocl_bzip2_mt_helper.h
 *  
 *  @brief Helper functions for multithreaded compression/decompression functions.
 *
 *  @author Niranjan Reddy
 */

#include "threads/threads.h"
#include <string.h>
#include "bzlib_private.h"

/*

BZIP2 frame format:

|<----------------- BZIP2 header --------------->|<---------------"block data" * "number of blocks" ------------->|<---------------- BZIP2 end of sequence ---------------->|
|<------ magic number ----->|<--- block size --->|<--------------------------- content -------------------------->|<-- end of sequence magic number -->|<-- combined CRC -->|
|<-------- 3 bytes  ------->|<----- 1 byte ----->|<---- variable number of "bits", (may not be byte aligned) ---->|<-------------- 6 bytes ----------->|<---- 4 bytes ----->|
*/

#define BZIP2_HEADER_BYTES (3+1)

#define BZIP2_EOS_MAGIC_NUMBER_BYTES 6
#define BZIP2_COMBINED_CRC_BYTES 4

#define BZIP2_EOS_BYTES (BZIP2_EOS_MAGIC_NUMBER_BYTES+BZIP2_COMBINED_CRC_BYTES)

#define INPUT_BLOCK_SIZE 100000

typedef struct bit_stream
{
    UInt32 buff;
    Int32 bits;
} bit_stream;

// Calculates the bit padding needed by matching a fixed trail pattern in the data.
Int32 get_empty_bits(const UChar *compressed_data);

// Buffers up to 32 bits and writes full bytes to output as they become available.
void append(Char **output, UInt32 c, Int32 bits, bit_stream *state);

// Flushes any remaining bits in the buffer to the output.
void finish_append(Char **output, bit_stream *state);

// Computes a combined checksum from the individual checksums of all blocks handled by the current thread.
UInt32 bz_mt_cur_thread_checksum(mt_checksum_node * current, UInt32 checksum);

// Frees the memory allocated for the multithreaded checksum nodes.
void bz_mt_free_checksum_nodes(Int32 num_threads, mt_data_list *mt_head_table);

// Performs post-processing steps after multi-threaded BZIP2 compression.
// It combines the compressed data from each thread, calculates the final checksum, and appends it to the output.
UInt32 aocl_bzip2_mt_post_processing(Char *dest, aocl_thread_group_t *thread_group_handle, mt_data_list* mt_head_table, Int32 rap_frame_length);

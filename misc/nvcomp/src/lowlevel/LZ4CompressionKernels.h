/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "../common.h"
#include "LZ4Types.h"

namespace nvcomp
{
namespace lowlevel
{

extern const int COMP_THREADS_PER_CHUNK;
extern const int DECOMP_THREADS_PER_CHUNK;
extern const int DECOMP_CHUNKS_PER_BLOCK;

/**
 * @brief Compress a batch of memory locations.
 *
 * @param decomp_data The batch items to compress.
 * @param decomp_sizes The size of each batch item to compress.
 * @param batch_size The number of items in the batch.
 * @param temp_data The temporary memory to use.
 * @param temp_bytes The size of the temporary memory.
 * @param comp_data The output location of each batch item.
 * @param comp_sizes
 * @param data_type The type of the input data to compress.
 * @param stream The stream to operate on.
 */
void lz4BatchCompress(
    const uint8_t* const* decomp_data_device,
    const size_t* decomp_sizes_device,
    const size_t max_chunk_size,
    const size_t batch_size,
    void* temp_data,
    size_t temp_bytes,
    uint8_t* const* comp_data_device,
    size_t* const comp_sizes_device,
    nvcompType_t data_type,
    cudaStream_t stream);

void lz4BatchDecompress(
    const uint8_t* const* device_in_ptrs,
    const size_t* device_in_bytes,
    const size_t* device_out_bytes,
    const size_t batch_size,
    void* temp_ptr,
    const size_t temp_bytes,
    uint8_t* const* device_out_ptrs,
    size_t* device_actual_uncompressed_bytes,
    nvcompStatus_t* device_status_ptrs,
    cudaStream_t stream);

/**
 * @brief Calculate the decompressed sizes of each chunk. This is 
 * for when we do not know upfront how much space to allocate before
 * running the decompression kernel. All pointers are GPU accessible.
 *
 * @param device_compressed_ptrs Pointers to compressed LZ4 chunks.
 * @param device_compressed_bytes The size of each compressed LZ4 chunk.
 * @param device_uncompressed_bytes The output calculated decompressed sizes
 * for each chunk.
 * @param batch_size The number of compressed chunks
 * @param stream The cuda stream to run on
 */
void lz4BatchGetDecompressSizes(
    const uint8_t* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream);

size_t lz4ComputeChunksInBatch(
    const size_t* const decomp_data_size,
    const size_t batch_size,
    const size_t chunk_size);

size_t lz4BatchCompressComputeTempSize(
    const size_t max_chunk_size, const size_t batch_size);

size_t lz4DecompressComputeTempSize(
    const size_t max_chunks_in_batch, const size_t chunk_size);

size_t lz4ComputeMaxSize(const size_t chunk_size);

size_t lz4MaxChunkSize();

size_t lz4GetHashTableSize(size_t max_chunk_size);

} // namespace lowlevel

} // namespace nvcomp

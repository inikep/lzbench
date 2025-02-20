/*
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVCOMP_GDEFLATE_H
#define NVCOMP_GDEFLATE_H

#include "nvcomp.h"

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * Batched compression/decompression interface for gdeflate
 *****************************************************************************/

/**
 * GDeflate compression options for the low-level API
 */
typedef struct
{
/**
 * Compression algorithm to use. Permitted values are:
 * 0 : high-throughput, low compression ratio (default)
 * 1 : low-throughput, high compression ratio
 * 2 : highest-throughput, entropy-only compression (use for symmetric compression/decompression performance)
 */
  int algo;
} nvcompBatchedGdeflateOpts_t;

static const nvcompBatchedGdeflateOpts_t nvcompBatchedGdeflateDefaultOpts = {0};

/**
 * @brief Get temporary space required for compression.
 *
 * Chunk size must not exceed
 * 65536 bytes. For best performance, a chunk size of 65536 bytes is
 * recommended.
 *
 * @param batch_size The number of items in the batch.
 * @param max_uncompressed_chunk_bytes The maximum size of a chunk in the
 * batch.
 * @param format_opts The GDeflate compression options to use.
 * @param temp_bytes The size of the required GPU workspace for compression
 * (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedGdeflateCompressGetTempSize(
    size_t batch_size,
    size_t max_uncompressed_chunk_bytes,
    nvcompBatchedGdeflateOpts_t format_opts,
    size_t* temp_bytes);

/**
 * @brief Get the maximum size any chunk could compress to in the batch. That
 * is, the minimum amount of output memory required to be given
 * nvcompBatchedGdeflateCompressAsync() for each batch item.
 *
 * Chunk size must not exceed
 * 65536 bytes. For best performance, a chunk size of 65536 bytes is
 * recommended.
 *
 * @param max_uncompressed_chunk_bytes The maximum size of a chunk in the batch.
 * @param format_opts The GDeflate compression options to use.
 * @param max_compressed_byes The maximum compressed size of the largest chunk
 * (output).
 *
 * @return The nvcompSuccess unless there is an error.
 */
nvcompStatus_t nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
    size_t max_uncompressed_chunk_bytes,
    nvcompBatchedGdeflateOpts_t format_opts,
    size_t* max_compressed_bytes);

/**
 * @brief Perform compression asynchronously. All pointers must point to GPU
 * accessible locations. The individual chunk size must not exceed
 * 65536 bytes. For best performance, a chunk size of 65536 bytes is
 * recommended.
 *
 * @param device_uncompressed_ptrs The pointers on the GPU, to uncompressed batched items.
 * This pointer must be GPU accessible.
 * @param device_uncompressed_bytes The size of each uncompressed batch item on the GPU.
 * @param max_uncompressed_chunk_bytes The maximum size in bytes of the largest
 * chunk in the batch.
 * @param batch_size The number of chunks to compress.
 * @param device_temp_ptr The temporary GPU workspace.
 * @param temp_bytes The size of the temporary GPU workspace.
 * @param device_compressed_ptrs The pointers on the GPU, to the output location for
 * each compressed batch item (output). This pointer must be GPU accessible.
 * @param device_compressed_bytes The compressed size of each chunk on the GPU
 * (output). This pointer must be GPU accessible.
 * @param format_opts The GDeflate compression options to use.
 * @param stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successfully launched, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedGdeflateCompressAsync(
    const void* const* device_uncompressed_ptrs,
    const size_t* device_uncompressed_bytes,
    size_t max_uncompressed_chunk_bytes,
    size_t batch_size,
    void* device_temp_ptr,
    size_t temp_bytes,
    void* const* device_compressed_ptrs,
    size_t* device_compressed_bytes,
    nvcompBatchedGdeflateOpts_t format_opts,
    cudaStream_t stream);


/**
 * @brief Get the amount of temp space required on the GPU for decompression.
 *
 * @param num_chunks The number of items in the batch.
 * @param max_uncompressed_chunk_bytes The size of the largest chunk in bytes
 * when uncompressed.
 * @param temp_bytes The amount of temporary GPU space that will be required to
 * decompress.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedGdeflateDecompressGetTempSize(
    size_t num_chunks,
    size_t max_uncompressed_chunk_bytes,
    size_t* temp_bytes);

/**
 * @brief Perform decompression asynchronously. All pointers must be GPU
 * accessible. In the case where a chunk of compressed data is not a valid GDeflate
 * stream, 0 will be written for the size of the invalid chunk and
 * nvcompStatusCannotDecompress will be flagged for that chunk.
 *
 * @param device_compressed_ptrs The pointers on the GPU, to the compressed
 * chunks.
 * @param device_compressed_bytes The size of each compressed chunk on the GPU.
 * This pointer must be GPU accessible.
 * @param device_uncompressed_bytes The decompressed buffer size. This is needed
 * to prevent OOB accesses. This pointer must be GPU accessible.
 * @param device_actual_uncompressed_bytes The actual calculated decompressed
 * size of each chunk. Can be nullptr if desired, 
 * in which case the actual_uncompressed_bytes is not reported.
 * @param batch_size The number of batch items.
 * @param device_temp_ptr The temporary GPU space.
 * @param temp_bytes The size of the temporary GPU space.
 * @param device_uncompressed_ptrs The pointers on the GPU, to where to
 * uncompress each chunk (output).
 * @param device_statuses The status for each chunk of whether it was
 * decompressed or not. Can be nullptr if desired, 
 * in which case error status is not reported.
 * @param stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedGdeflateDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    nvcompStatus_t* device_statuses,
    cudaStream_t stream);

/**
 * @brief Calculates the decompressed size of each chunk asynchronously. This is
 * needed when we do not know the expected output size. All pointers must be GPU
 * accessible. Note, if the stream is corrupt, the sizes will be garbage.
 *
 * @param device_compress_ptrs The compressed chunks of data. List of pointers
 * must be GPU accessible along with each chunk.
 * @param device_compressed_bytes The size of each compressed chunk. Must be GPU
 * accessible.
 * @param device_uncompressed_bytes The calculated decompressed size of each
 * chunk. Must be GPU accessible.
 * @param batch_size The number of chunks
 * @param stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompBatchedGdeflateGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif // NVCOMP_GDEFLATE_H

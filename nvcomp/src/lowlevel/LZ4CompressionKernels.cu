/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "CudaUtils.h"
#include "LZ4CompressionKernels.h"
#include "LZ4Kernels.cuh"
#include "TempSpaceBroker.h"
#include "common.h"

#include "cuda_runtime.h"
#include "nvcomp_cub.cuh"

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

using double_word_type = uint64_t;
using item_type = uint32_t;

#define OOB_CHECKING 1 // Prevent's crashing of corrupt lz4 sequences

namespace nvcomp {

namespace lowlevel {

template<typename T>
__global__ void lz4CompressBatchKernel(
    const uint8_t* const* device_in_ptr,
    const size_t* const device_in_bytes,
    uint8_t* const* const device_out_ptr,
    size_t* const device_out_bytes,
    offset_type* const temp_space,
    const position_type hash_table_size)
{
  const int bidx = blockIdx.x * blockDim.y + threadIdx.y;

  auto decomp_ptr = device_in_ptr[bidx];
  assert(reinterpret_cast<uintptr_t>(decomp_ptr) % sizeof(T) == 0 && "Input buffer not aligned");
  const size_t decomp_length = device_in_bytes[bidx];

  uint8_t* const comp_ptr = device_out_ptr[bidx];
  size_t* const comp_length = device_out_bytes + bidx;

  offset_type* const hash_table = temp_space + bidx * hash_table_size;

  compressStream(comp_ptr, reinterpret_cast<const T*>(decomp_ptr), hash_table, hash_table_size, decomp_length, comp_length);
}

__global__ void lz4DecompressBatchKernel(
    const uint8_t* const* const device_in_ptrs,
    const size_t* const device_in_bytes,
    const size_t* const device_out_bytes,
    const size_t batch_size,
    uint8_t* const* const device_out_ptrs,
    size_t* device_uncompressed_bytes,
    nvcompStatus_t* device_status_ptrs,
    bool output_decompressed)
{
  const int bid = blockIdx.x * LZ4_DECOMP_CHUNKS_PER_BLOCK + threadIdx.y;

  __shared__ uint8_t buffer[DECOMP_INPUT_BUFFER_SIZE * LZ4_DECOMP_CHUNKS_PER_BLOCK];

  assert(!output_decompressed || device_out_ptrs != nullptr);
  // device_uncompressed_bytes needs to be valid if we are precomputing
  // output size
  assert(output_decompressed || device_uncompressed_bytes != nullptr);

  if (bid < batch_size) {
    uint8_t* const decomp_ptr
        = device_out_ptrs == nullptr ? nullptr : device_out_ptrs[bid];
    const uint8_t* const comp_ptr = device_in_ptrs[bid];
    const position_type chunk_length
        = static_cast<position_type>(device_in_bytes[bid]);
    const position_type output_buf_length
        = output_decompressed
              ? static_cast<position_type>(device_out_bytes[bid])
              : UINT_MAX;

    decompressStream(
        buffer + threadIdx.y * DECOMP_INPUT_BUFFER_SIZE,
        decomp_ptr,
        comp_ptr,
        chunk_length,
        output_buf_length,
        device_uncompressed_bytes ? device_uncompressed_bytes + bid : nullptr,
        device_status_ptrs? device_status_ptrs + bid : nullptr,
        output_decompressed);
  }
}

/******************************************************************************
 * PUBLIC FUNCTIONS ***********************************************************
 *****************************************************************************/

size_t lz4GetHashTableSize(size_t max_chunk_size)
{
  auto roundUpPow2 = [](size_t x) {
    size_t ans = 1;
    while(ans < x)
      ans *= 2;
    return ans;
  };
  // when chunk size is smaller than the max hashtable size round the
  // hashtable size up to the nearest power of 2 of the chunk size.
  // The lower load factor from a significantly larger hashtable size compared
  // to the chunk size doesn't increase performance, however having a smaller
  // hashtable which yields much high cache utilization does.
  return min(roundUpPow2(max_chunk_size), (size_t)MAX_HASH_TABLE_SIZE);
}

void lz4BatchCompress(
    const uint8_t* const* decomp_data_device,
    const size_t* const decomp_sizes_device,
    const size_t max_chunk_size,
    const size_t batch_size,
    void* const temp_data,
    const size_t temp_bytes,
    uint8_t* const* const comp_data_device,
    size_t* const comp_sizes_device,
    nvcompType_t data_type,
    cudaStream_t stream)
{

  position_type HT_size = lz4GetHashTableSize(max_chunk_size);

  const size_t total_required_temp
      = batch_size * HT_size * sizeof(offset_type);
  if (temp_bytes < total_required_temp) {
    throw std::runtime_error(
        "Insufficient temp space: got " + std::to_string(temp_bytes)
        + " bytes, but need " + std::to_string(total_required_temp)
        + " bytes.");
  }

  const dim3 grid(batch_size);
  const dim3 block(LZ4_COMP_THREADS_PER_CHUNK);

  switch (data_type) {
    case NVCOMP_TYPE_BITS:
    case NVCOMP_TYPE_CHAR:
    case NVCOMP_TYPE_UCHAR:
      lz4CompressBatchKernel<uint8_t><<<grid, block, 0, stream>>>(
          decomp_data_device,
          decomp_sizes_device,
          comp_data_device,
          comp_sizes_device,
          static_cast<offset_type*>(temp_data),
          HT_size);
      break;
    case NVCOMP_TYPE_SHORT:
    case NVCOMP_TYPE_USHORT:
      lz4CompressBatchKernel<uint16_t><<<grid, block, 0, stream>>>(
          decomp_data_device,
          decomp_sizes_device,
          comp_data_device,
          comp_sizes_device,
          static_cast<offset_type*>(temp_data),
          HT_size);
      break;
    case NVCOMP_TYPE_INT:
    case NVCOMP_TYPE_UINT:
      lz4CompressBatchKernel<uint32_t><<<grid, block, 0, stream>>>(
          decomp_data_device,
          decomp_sizes_device,
          comp_data_device,
          comp_sizes_device,
          static_cast<offset_type*>(temp_data),
          HT_size);
      break;
    default:
      throw std::invalid_argument("Unsupported input data type");
  }

  CudaUtils::check_last_error();
}

void lz4BatchDecompress(
    const uint8_t* const* const device_in_ptrs,
    const size_t* const device_in_bytes,
    const size_t* const device_out_bytes,
    const size_t batch_size,
    void* const /* temp_ptr */,
    const size_t /* temp_bytes */,
    uint8_t* const* const device_out_ptrs,
    size_t* device_actual_uncompressed_bytes,
    nvcompStatus_t* device_status_ptrs,
    cudaStream_t stream)
{
  const dim3 grid(roundUpDiv(batch_size, LZ4_DECOMP_CHUNKS_PER_BLOCK));
  const dim3 block(LZ4_DECOMP_THREADS_PER_CHUNK, LZ4_DECOMP_CHUNKS_PER_BLOCK);

  lz4DecompressBatchKernel<<<grid, block, 0, stream>>>(
      device_in_ptrs,
      device_in_bytes,
      device_out_bytes,
      batch_size,
      device_out_ptrs,
      device_actual_uncompressed_bytes,
      device_status_ptrs,
      true);
  CudaUtils::check_last_error("lz4DecompressBatchKernel()");
}

void lz4BatchGetDecompressSizes(
    const uint8_t* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream)
{
  const dim3 grid(roundUpDiv(batch_size, LZ4_DECOMP_CHUNKS_PER_BLOCK));
  const dim3 block(LZ4_DECOMP_THREADS_PER_CHUNK, LZ4_DECOMP_CHUNKS_PER_BLOCK);

  lz4DecompressBatchKernel<<<grid, block, 0, stream>>>(
      device_compressed_ptrs,
      device_compressed_bytes,
      nullptr,
      batch_size,
      nullptr,
      device_uncompressed_bytes,
      nullptr,
      false);
  CudaUtils::check_last_error("lz4DecompressBatchKernel()");
}

size_t lz4ComputeChunksInBatch(
    const size_t* const decomp_data_size,
    const size_t batch_size,
    const size_t chunk_size)
{
  size_t num_chunks = 0;

  for (size_t i = 0; i < batch_size; ++i) {
    num_chunks += roundUpDiv(decomp_data_size[i], chunk_size);
  }

  return num_chunks;
}

size_t lz4BatchCompressComputeTempSize(
    const size_t max_chunk_size, const size_t batch_size)
{
  if (max_chunk_size > lz4MaxChunkSize()) {
    throw std::runtime_error(
        "Maximum chunk size for LZ4 is " + std::to_string(lz4MaxChunkSize()));
  }

  return lz4GetHashTableSize(max_chunk_size) * sizeof(offset_type) * batch_size;
}

size_t lz4DecompressComputeTempSize(
    const size_t maxChunksInBatch, const size_t /* chunkSize */)
{
  const size_t header_size = sizeof(chunk_header) * maxChunksInBatch;

  return roundUpTo(header_size, sizeof(size_t));
}

size_t lz4ComputeMaxSize(const size_t size)
{
  if (size > lz4MaxChunkSize()) {
    throw std::runtime_error(
        "Maximum chunk size for LZ4 is " + std::to_string(lz4MaxChunkSize()));
  }
  return maxSizeOfStream(size);
}

size_t lz4MaxChunkSize()
{
  return MAX_CHUNK_SIZE;
}

} // namespace lowlevel
} // namespace nvcomp

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
#include "LZ4HlifKernels.h"
#include "LZ4Kernels.cuh"
#include "TempSpaceBroker.h"
#include "common.h"

#include "nvcomp_common_deps/hlif_shared.cuh"
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

struct LZ4CompressorArgs {
  const position_type hash_table_size;
};

template<typename T>
struct lz4_compress_wrapper : hlif_compress_wrapper {
private: 
  const position_type hash_table_size;
  offset_type* hash_table;
  nvcompStatus_t* status;

public:
  __device__ lz4_compress_wrapper(
      const LZ4CompressorArgs input,
      uint8_t* tmp_buffer,
      uint8_t*, /*share_buffer*/
      nvcompStatus_t* status)
    : hash_table_size(input.hash_table_size),
      status(status)
  {
    hash_table = reinterpret_cast<offset_type*>(tmp_buffer) + blockIdx.x * hash_table_size;
  }
      
  __device__ void compress_chunk(
      uint8_t* tmp_output_buffer,
      const uint8_t* this_decomp_buffer,
      const size_t decomp_size,
      const size_t, // max_comp_chunk_size
      size_t* comp_chunk_size) 
  {
    assert(reinterpret_cast<uintptr_t>(this_decomp_buffer) % sizeof(T) == 0 && "Input buffer not aligned");

    compressStream<T>(
        tmp_output_buffer, 
        reinterpret_cast<const T*>(this_decomp_buffer), 
        hash_table, 
        hash_table_size, 
        decomp_size,
        comp_chunk_size);
  }

  __device__ nvcompStatus_t get_output_status() {
    return *status;
  }

  __device__ FormatType get_format_type() {
    return FormatType::LZ4;
  }

};

struct lz4_decompress_wrapper : hlif_decompress_wrapper {

private:
  uint8_t* this_shared_buffer;
  nvcompStatus_t* status;

public:
  __device__ lz4_decompress_wrapper(uint8_t* shared_buffer, nvcompStatus_t* status)
    : this_shared_buffer(reinterpret_cast<uint8_t*>(shared_buffer) + threadIdx.y * DECOMP_INPUT_BUFFER_SIZE),
      status(status)
  {}
      
  __device__ void decompress_chunk(
      uint8_t* decomp_buffer,
      const uint8_t* comp_buffer,
      const size_t comp_chunk_size,
      const size_t decomp_buffer_size)
  {
    decompressStream(
        this_shared_buffer,
        decomp_buffer,
        comp_buffer,
        comp_chunk_size,
        decomp_buffer_size,
        nullptr, // device_uncompressed_bytes -- unnecessary for HLIF
        status,
        true /* output decompressed */);
  }

  __device__ nvcompStatus_t get_output_status() {
    return *status;
  }
};

void lz4HlifBatchCompress(
    const CompressArgs& compress_args,
    const position_type hash_table_size,
    const uint32_t max_ctas,
    nvcompType_t data_type,
    cudaStream_t stream) 
{
  const dim3 grid(max_ctas);
  const dim3 block(LZ4_COMP_THREADS_PER_CHUNK);

  switch (data_type) {
    case NVCOMP_TYPE_BITS:
    case NVCOMP_TYPE_CHAR:
    case NVCOMP_TYPE_UCHAR:
      HlifCompressBatchKernel<lz4_compress_wrapper<uint8_t>><<<grid, block, 0, stream>>>(
          compress_args,
          LZ4CompressorArgs{hash_table_size});
      break;
    case NVCOMP_TYPE_SHORT:
    case NVCOMP_TYPE_USHORT:
      HlifCompressBatchKernel<lz4_compress_wrapper<uint16_t>><<<grid, block, 0, stream>>>(
          compress_args,
          LZ4CompressorArgs{hash_table_size});
      break;
    case NVCOMP_TYPE_INT:
    case NVCOMP_TYPE_UINT:
      HlifCompressBatchKernel<lz4_compress_wrapper<uint32_t>><<<grid, block, 0, stream>>>(
          compress_args,
          LZ4CompressorArgs{hash_table_size});
      break;
    default:
      throw std::invalid_argument("Unsupported input data type");
  }

  CudaUtils::check_last_error();
}

void lz4HlifBatchDecompress(
    const uint8_t* comp_buffer, 
    uint8_t* decomp_buffer, 
    const size_t raw_chunk_size,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t* comp_chunk_offsets,
    const size_t* comp_chunk_sizes,
    const uint32_t max_ctas,
    cudaStream_t stream,
    nvcompStatus_t* output_status) 
{
  const dim3 grid(max_ctas);
  const dim3 block(LZ4_DECOMP_THREADS_PER_CHUNK, LZ4_DECOMP_CHUNKS_PER_BLOCK);
  constexpr int shmem_size = DECOMP_INPUT_BUFFER_SIZE * LZ4_DECOMP_CHUNKS_PER_BLOCK;
  HlifDecompressBatchKernel<lz4_decompress_wrapper, LZ4_DECOMP_CHUNKS_PER_BLOCK><<<grid, block, shmem_size, stream>>>(
      comp_buffer,
      decomp_buffer,
      raw_chunk_size,
      ix_chunk,
      num_chunks,
      comp_chunk_offsets,
      comp_chunk_sizes,
      output_status);

  CudaUtils::check_last_error();
}

size_t batchedLZ4CompMaxBlockOccupancy(nvcompType_t data_type, const int device_id)
{
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  int num_blocks_per_sm;
  switch (data_type) {
    case NVCOMP_TYPE_BITS:
    case NVCOMP_TYPE_CHAR:
    case NVCOMP_TYPE_UCHAR:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm, 
          HlifCompressBatchKernel<lz4_compress_wrapper<uint8_t>, LZ4CompressorArgs>, 
          LZ4_COMP_THREADS_PER_CHUNK, 
          0);
      break;
    case NVCOMP_TYPE_SHORT:
    case NVCOMP_TYPE_USHORT:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm, 
          HlifCompressBatchKernel<lz4_compress_wrapper<uint16_t>, LZ4CompressorArgs>, 
          LZ4_COMP_THREADS_PER_CHUNK, 
          0);
      break;
    case NVCOMP_TYPE_INT:
    case NVCOMP_TYPE_UINT:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm, 
          HlifCompressBatchKernel<lz4_compress_wrapper<uint32_t>, LZ4CompressorArgs>, 
          LZ4_COMP_THREADS_PER_CHUNK, 
          0);
      break;
    default:
      throw std::invalid_argument("Unsupported input data type");
  }
  
  return device_prop.multiProcessorCount * num_blocks_per_sm;
}

size_t batchedLZ4DecompMaxBlockOccupancy(nvcompType_t data_type, const int device_id)
{
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  int num_blocks_per_sm;
  constexpr int shmem_size = DECOMP_INPUT_BUFFER_SIZE * LZ4_DECOMP_CHUNKS_PER_BLOCK;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_sm, 
      HlifDecompressBatchKernel<lz4_decompress_wrapper, LZ4_DECOMP_CHUNKS_PER_BLOCK>, 
      LZ4_DECOMP_THREADS_PER_CHUNK * LZ4_DECOMP_CHUNKS_PER_BLOCK, 
      shmem_size);
  
  return device_prop.multiProcessorCount * num_blocks_per_sm;
}

} // namespace nvcomp

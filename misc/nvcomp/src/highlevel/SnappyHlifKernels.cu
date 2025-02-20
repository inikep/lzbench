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


#include "highlevel/SnappyHlifKernels.h"
#include "nvcomp_common_deps/hlif_shared.cuh"
#include "SnappyKernels.cuh"
#include "CudaUtils.h"

namespace nvcomp {

struct snappy_compress_wrapper : hlif_compress_wrapper {

private:
  nvcompStatus_t* status;

public:
  __device__ snappy_compress_wrapper(uint8_t* /*tmp_buffer*/, uint8_t* /*share_buffer*/, nvcompStatus_t* status)
   : status(status)
  {}
      
  __device__ void compress_chunk(
      uint8_t* tmp_output_buffer,
      const uint8_t* this_decomp_buffer,
      const size_t decomp_size,
      const size_t max_comp_chunk_size,
      size_t* comp_chunk_size) 
  {
    do_snap(
        this_decomp_buffer,
        decomp_size,
        tmp_output_buffer,
        max_comp_chunk_size,
        nullptr, // snappy status -- could add this later. Need to work through how to do error checking.
        comp_chunk_size);
  }

  __device__ nvcompStatus_t get_output_status() {
    return *status;
  }

  __device__ FormatType get_format_type() {
    return FormatType::Snappy;
  }
};

struct snappy_decompress_wrapper : hlif_decompress_wrapper {

private:
  nvcompStatus_t* status;

public:
  __device__ snappy_decompress_wrapper(uint8_t* /*shared_buffer*/, nvcompStatus_t* status)
    : status(status)
  {}
      
  __device__ void decompress_chunk(
      uint8_t* decomp_buffer,
      const uint8_t* comp_buffer,
      const size_t comp_chunk_size,
      const size_t decomp_buffer_size) 
  {
    do_unsnap(
        comp_buffer,
        comp_chunk_size,
        decomp_buffer,
        decomp_buffer_size,
        status,
        nullptr); // device_uncompressed_bytes -- unnecessary for HLIF
  }

  __device__ nvcompStatus_t get_output_status() {
    return *status;
  }
};

void snappyHlifBatchCompress(
    const CompressArgs& comp_args,
    const uint32_t max_ctas,
    cudaStream_t stream) 
{
  const dim3 grid(max_ctas);
  const dim3 block(COMP_THREADS_PER_BLOCK);

  HlifCompressBatchKernel<snappy_compress_wrapper><<<grid, block, 0, stream>>>(
      comp_args);      
}

void snappyHlifBatchDecompress(
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
  const dim3 block(DECOMP_THREADS_PER_BLOCK);
  HlifDecompressBatchKernel<snappy_decompress_wrapper><<<grid, block, 0, stream>>>(
      comp_buffer,
      decomp_buffer,
      raw_chunk_size,
      ix_chunk,
      num_chunks,
      comp_chunk_offsets,
      comp_chunk_sizes,
      output_status);
}

size_t snappyHlifCompMaxBlockOccupancy(const int device_id) 
{
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  int num_blocks_per_sm;
  constexpr int shmem_size = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_sm, 
      HlifCompressBatchKernel<snappy_compress_wrapper>, 
      COMP_THREADS_PER_BLOCK,
      shmem_size);
  
  return device_prop.multiProcessorCount * num_blocks_per_sm;
}

size_t snappyHlifDecompMaxBlockOccupancy(const int device_id) 
{
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  int num_blocks_per_sm;
  constexpr int shmem_size = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_sm, 
      HlifDecompressBatchKernel<snappy_decompress_wrapper, 1>, 
      DECOMP_THREADS_PER_BLOCK, 
      shmem_size);
  
  return device_prop.multiProcessorCount * num_blocks_per_sm;
}

} // nvcomp namespace

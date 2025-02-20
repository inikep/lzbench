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

#pragma once

#include <cassert>
#include <cooperative_groups.h>
#include <stdio.h>
#include <type_traits>

#include "nvcomp/shared_types.h"
#include "hlif_shared_types.hpp"

namespace cg = cooperative_groups;

// Compress wrapper must meet this requirement
struct hlif_compress_wrapper {
  __device__ void compress_chunk(
      uint8_t* /*scratch_output_buffer*/,
      const uint8_t* /*this_decomp_buffer*/,
      const size_t /*decomp_size*/,
      const size_t /*max_comp_chunk_size*/, 
      size_t*) /*comp_chunk_size*/
  {
    assert(false); // This must be implemented in the derived class
  }
  
  __device__ nvcompStatus_t get_output_status()
  {
    assert(false); // This must be implemented in the derived class
    return nvcompErrorNotSupported;
  } 

  __device__ FormatType get_format_type()
  {
    assert(false); // This must be implemented in the derived class
    return NotSupportedError;
  }

  __device__ ~hlif_compress_wrapper() {};
};

// Decompress wrapper must meet this requirement
struct hlif_decompress_wrapper {
  __device__ void decompress_chunk(
      uint8_t*, /*decomp_buffer*/
      const uint8_t*, /*comp_buffer*/
      const size_t, /*comp_chunk_size*/
      const size_t) /*decomp_buffer_size*/
  {
    assert(false); // This must be implemented in the derived class
  }
      
  __device__ nvcompStatus_t get_output_status()
  {
    assert(false); // This must be implemented in the derived class
    return nvcompErrorNotSupported;
  }
  
  __device__ ~hlif_decompress_wrapper() {}
};

__device__ inline void fill_common_header(
    const CompressArgs& compress_args,
    const FormatType format_type) 
{
  compress_args.common_header->magic_number = 0;
  compress_args.common_header->major_version = 2;
  compress_args.common_header->minor_version = 2;
  compress_args.common_header->format = format_type;
  compress_args.common_header->decomp_data_size = compress_args.decomp_buffer_size;
  compress_args.common_header->num_chunks = compress_args.num_chunks;
  compress_args.common_header->include_chunk_starts = true;
  compress_args.common_header->full_comp_buffer_checksum = 0;
  compress_args.common_header->decomp_buffer_checksum = 0;
  compress_args.common_header->include_per_chunk_comp_buffer_checksums = false;
  compress_args.common_header->include_per_chunk_decomp_buffer_checksums = false;
  compress_args.common_header->uncomp_chunk_size = compress_args.uncomp_chunk_size;
  compress_args.common_header->comp_data_offset = (uintptr_t)compress_args.comp_buffer - (uintptr_t)compress_args.common_header;
}

__device__ inline void copyScratchBuffer(
    size_t* comp_chunk_offsets,
    size_t* comp_chunk_sizes,
    const uint8_t* scratch_output_buffer,
    uint8_t* comp_buffer,
    uint64_t* ix_output,
    uint32_t ix_chunk)
{
  // Do the copy into the final buffer.
  size_t comp_chunk_offset = comp_chunk_offsets[ix_chunk];
  size_t comp_chunk_size = comp_chunk_sizes[ix_chunk];
  const int ix_alignment_input = sizeof(uint32_t) - ((uintptr_t)scratch_output_buffer % sizeof(uint32_t));
  if (ix_alignment_input % 4 == 0) {
    const char4* aligned_input = reinterpret_cast<const char4*>(scratch_output_buffer);
    uint8_t* output = comp_buffer + comp_chunk_offset;
    for (size_t ix = threadIdx.x; ix < comp_chunk_size / 4; ix += blockDim.x) {
      char4 val = aligned_input[ix];
      output[4 * ix] = val.x;
      output[4 * ix + 1] = val.y;
      output[4 * ix + 2] = val.z;
      output[4 * ix + 3] = val.w;
    }
    int rem_bytes = comp_chunk_size % sizeof(uint32_t);
    if (threadIdx.x < rem_bytes) {
      output[comp_chunk_size - rem_bytes + threadIdx.x] = scratch_output_buffer[comp_chunk_size - rem_bytes + threadIdx.x];
    }
  } else {
    for (size_t ix = threadIdx.x; ix < comp_chunk_size; ix += blockDim.x) {
      comp_buffer[comp_chunk_offset + ix] = scratch_output_buffer[ix];
    }
  }
}

template<int chunks_per_block, typename CompressT, typename GroupT>
__device__ inline void HlifCompressBatch(
    const CompressArgs& compression_args,
    CompressT&& compressor,
    GroupT&& cg_group)
{
  if (blockIdx.x == 0 && cg::this_thread_block().thread_rank() == 0) {
    fill_common_header(
        compression_args, 
        compressor.get_format_type());
  }

  __shared__ uint32_t ix_chunks[chunks_per_block];
  volatile uint32_t& this_ix_chunk = ix_chunks[threadIdx.y];

  if (cg_group.thread_rank() == 0) {
    this_ix_chunk = blockIdx.x * chunks_per_block + threadIdx.y;
  }

  cg_group.sync();

  uint8_t* scratch_output_buffer = compression_args.scratch_buffer + this_ix_chunk * compression_args.max_comp_chunk_size;

  int initial_chunks = gridDim.x * chunks_per_block;

  while (this_ix_chunk < compression_args.num_chunks) {
    size_t ix_decomp_start = this_ix_chunk * compression_args.uncomp_chunk_size;
    const uint8_t* this_decomp_buffer = compression_args.decomp_buffer + ix_decomp_start;
    size_t decomp_size = min(compression_args.uncomp_chunk_size, compression_args.decomp_buffer_size - ix_decomp_start);
    compressor.compress_chunk(
        scratch_output_buffer,
        this_decomp_buffer,
        decomp_size,
        compression_args.max_comp_chunk_size,
        &compression_args.comp_chunk_sizes[this_ix_chunk]);

    // Determine the right place to output this buffer.
    if (cg_group.thread_rank() == 0) {
        static_assert(sizeof(uint64_t) == sizeof(unsigned long long int),
          "The cast below requires that the sizes are the same.");
        compression_args.comp_chunk_offsets[this_ix_chunk] = atomicAdd(
          reinterpret_cast<unsigned long long int*>(compression_args.ix_output), 
          compression_args.comp_chunk_sizes[this_ix_chunk]);
    }

    cg_group.sync();

    copyScratchBuffer(
        compression_args.comp_chunk_offsets,
        compression_args.comp_chunk_sizes,
        scratch_output_buffer,
        compression_args.comp_buffer,
        compression_args.ix_output,
        this_ix_chunk);

    // Check for errors. Any error should be reported in the global status value
    if (cg_group.thread_rank() == 0) {
      if (compressor.get_output_status() != nvcompSuccess) {
        *compression_args.output_status = compressor.get_output_status();
      }
    }

    if (cg_group.thread_rank() == 0) {
      this_ix_chunk = initial_chunks + atomicAdd(compression_args.ix_chunk, size_t{1});
    }
    cg_group.sync();
  }
}

template<typename CompressT, 
         typename CompressorArg,
         int chunks_per_block = 1>
__global__ std::enable_if_t<std::is_base_of<hlif_compress_wrapper, CompressT>::value>
HlifCompressBatchKernel(
    CompressArgs compression_args,
    CompressorArg compressor_arg)
{
  extern __shared__ uint8_t share_buffer[];

  uint8_t* free_scratch_buffer = 
      compression_args.scratch_buffer + (compression_args.max_comp_chunk_size * gridDim.x * blockDim.y);
  
  __shared__ nvcompStatus_t output_status[chunks_per_block];
  
  CompressT compressor{compressor_arg, free_scratch_buffer, share_buffer, &output_status[threadIdx.y]};

  auto cta_group = cg::this_thread_block();
  if (chunks_per_block == 1) {
    HlifCompressBatch<chunks_per_block>(compression_args, compressor, cta_group);
  } else {
    HlifCompressBatch<chunks_per_block>(compression_args, compressor, cg::tiled_partition<32>(cta_group));
  }
}

template<typename CompressT, int chunks_per_block = 1>
__global__ std::enable_if_t<std::is_base_of<hlif_compress_wrapper, CompressT>::value>
HlifCompressBatchKernel(CompressArgs compression_args)
{
  extern __shared__ uint8_t share_buffer[];
  
  uint8_t* free_scratch_buffer = 
      compression_args.scratch_buffer + (compression_args.max_comp_chunk_size * gridDim.x * blockDim.y);

  __shared__ nvcompStatus_t output_status[chunks_per_block];

  CompressT compressor{free_scratch_buffer, share_buffer, &output_status[threadIdx.y]};

  auto cta_group = cg::this_thread_block();
  if (chunks_per_block == 1) {
    HlifCompressBatch<chunks_per_block>(compression_args, compressor, cta_group);
  } else {
    HlifCompressBatch<chunks_per_block>(compression_args, compressor, cg::tiled_partition<32>(cta_group));
  }
}

/**
 * @brief Decompresses one or more chunks at a time using a given CTA
 * 
 * Takes in a DecompressT, which executes a device function to decompress a chunk
 * 
 * Can decompress multiple chunks / CTA. In this case, the "X" threads in the block
 * decompress a single chunk. Threadidx.y indicates the chunk index within the CTA.
 * 
 */  

template<typename DecompressT,
         int chunks_per_block,
         typename GroupT>
__device__ inline void HlifDecompressBatch(
    const uint8_t* comp_buffer, 
    uint8_t* decomp_buffer, 
    const size_t uncomp_chunk_size,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t* comp_chunk_offsets,
    const size_t* comp_chunk_sizes,
    uint8_t* share_buffer,
    nvcompStatus_t* kernel_output_status,
    DecompressT& decompressor,
    GroupT&& cg_group)
{
  // If chunks_per_block is 1, any blockDim is allowed
  // Otherwise, the y index is the chunk index
  assert(chunks_per_block == 1 || chunks_per_block == blockDim.y);
  
  __shared__ uint32_t ix_chunks[chunks_per_block];

  int init_chunk_offset = chunks_per_block == 1 ? 0 : threadIdx.y;

  volatile uint32_t& this_ix_chunk = *(ix_chunks + init_chunk_offset);
  if (cg_group.thread_rank() == 0) {
    this_ix_chunk = blockIdx.x * chunks_per_block + init_chunk_offset;
  }

  cg_group.sync();

  int initial_chunks = gridDim.x * chunks_per_block;  
  while (this_ix_chunk < num_chunks) {
    const uint8_t* this_comp_buffer = comp_buffer + comp_chunk_offsets[this_ix_chunk];
    uint8_t* this_decomp_buffer = decomp_buffer + this_ix_chunk * uncomp_chunk_size;

    decompressor.decompress_chunk(
        this_decomp_buffer,
        this_comp_buffer,
        comp_chunk_sizes[this_ix_chunk],
        uncomp_chunk_size);

    // Check for errors. Any error should be reported in the global status value
    if (cg_group.thread_rank() == 0) {
      if (decompressor.get_output_status() != nvcompSuccess) {
        *kernel_output_status = decompressor.get_output_status();
      }
    }

    if (cg_group.thread_rank() == 0) {
      this_ix_chunk = initial_chunks + atomicAdd(ix_chunk, uint32_t{1});
    }

    cg_group.sync();
  }
}    

template<typename DecompressT,
         int chunks_per_block>
__device__ void HlifDecompressBatch(
    const uint8_t* comp_buffer, 
    uint8_t* decomp_buffer, 
    const size_t uncomp_chunk_size,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t* comp_chunk_offsets,
    const size_t* comp_chunk_sizes,
    uint8_t* share_buffer,
    nvcompStatus_t* kernel_output_status,
    DecompressT& decompressor)
{
  // Dispatches to get a cooperative group per-chunk
  auto cta_group = cg::this_thread_block();
  if (chunks_per_block == 1) {
    HlifDecompressBatch<DecompressT, chunks_per_block>(
        comp_buffer, 
        decomp_buffer, 
        uncomp_chunk_size,
        ix_chunk,
        num_chunks,
        comp_chunk_offsets,
        comp_chunk_sizes,
        share_buffer,
        kernel_output_status,
        decompressor,
        cta_group);
  } else {
    HlifDecompressBatch<DecompressT, chunks_per_block>(
        comp_buffer, 
        decomp_buffer, 
        uncomp_chunk_size,
        ix_chunk,
        num_chunks,
        comp_chunk_offsets,
        comp_chunk_sizes,
        share_buffer,
        kernel_output_status,
        decompressor,
        cg::tiled_partition<32>(cta_group));
    assert(blockDim.x == 32);
  }
}

template<typename DecompressT,
         int chunks_per_block = 1,
         typename DecompArg>
__global__ std::enable_if_t<std::is_base_of<hlif_decompress_wrapper, DecompressT>::value> 
HlifDecompressBatchKernel(
    const uint8_t* comp_buffer, 
    uint8_t* decomp_buffer, 
    const size_t uncomp_chunk_size,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t* comp_chunk_offsets,
    const size_t* comp_chunk_sizes,
    nvcompStatus_t* kernel_output_status,
    DecompArg decompress_arg)
{
  extern __shared__ uint8_t share_buffer[];
  __shared__ nvcompStatus_t output_status[chunks_per_block];
  DecompressT decompressor{decompress_arg, share_buffer, &output_status[threadIdx.y]};
  HlifDecompressBatch<DecompressT, chunks_per_block>(
        comp_buffer, 
        decomp_buffer, 
        uncomp_chunk_size,
        ix_chunk,
        num_chunks,
        comp_chunk_offsets,
        comp_chunk_sizes,
        share_buffer,
        kernel_output_status,
        decompressor);
}

template<typename DecompressT,
         int chunks_per_block = 1>
__global__ std::enable_if_t<std::is_base_of<hlif_decompress_wrapper, DecompressT>::value> 
HlifDecompressBatchKernel(
    const uint8_t* comp_buffer, 
    uint8_t* decomp_buffer, 
    const size_t uncomp_chunk_size,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t* comp_chunk_offsets,
    const size_t* comp_chunk_sizes,
    nvcompStatus_t* kernel_output_status)
{
  extern __shared__ uint8_t share_buffer[];
  __shared__ nvcompStatus_t output_status[chunks_per_block];
  DecompressT decompressor{share_buffer, &output_status[threadIdx.y]};

  HlifDecompressBatch<DecompressT, chunks_per_block>(
        comp_buffer, 
        decomp_buffer, 
        uncomp_chunk_size,
        ix_chunk,
        num_chunks,
        comp_chunk_offsets,
        comp_chunk_sizes,
        share_buffer,
        kernel_output_status,
        decompressor);
}

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

#include "lowlevel/SnappyBatchKernels.h"
#include "SnappyKernels.cuh"
#include "CudaUtils.h"

namespace nvcomp {

/**
 * @brief Snappy compression kernel
 * See http://github.com/google/snappy/blob/master/format_description.txt
 *
 * @param[in] inputs Source/Destination buffer information per block
 * @param[out] outputs Compression status per block
 * @param[in] count Number of blocks to compress
 **/
__global__ void __launch_bounds__(COMP_THREADS_PER_BLOCK)
snap_kernel(
  const void* const* __restrict__ device_in_ptr,
  const uint64_t* __restrict__ device_in_bytes,
  void* const* __restrict__ device_out_ptr,
  const uint64_t* __restrict__ device_out_available_bytes,
  gpu_snappy_status_s * __restrict__ outputs,
  uint64_t* device_out_bytes)
{
  const int ix_chunk = blockIdx.x;
  do_snap(reinterpret_cast<const uint8_t*>(device_in_ptr[ix_chunk]),
      device_in_bytes[ix_chunk],
      reinterpret_cast<uint8_t*>(device_out_ptr[ix_chunk]),
      device_out_available_bytes ? device_out_available_bytes[ix_chunk] : 0,
      outputs ? &outputs[ix_chunk] : nullptr,
      &device_out_bytes[ix_chunk]);
}

__global__ void __launch_bounds__(32)
get_uncompressed_sizes_kernel(
  const void* const* __restrict__ device_in_ptr,
  const uint64_t* __restrict__ device_in_bytes,
  uint64_t* __restrict__ device_out_bytes)
{
  int t             = threadIdx.x;
  int strm_id       = blockIdx.x;

  if (t == 0) {
    uint32_t uncompressed_size = 0;
    const uint8_t *cur = reinterpret_cast<const uint8_t *>(device_in_ptr[strm_id]);
    const uint8_t *end = cur + device_in_bytes[strm_id];
    if (cur < end) {
      // Read uncompressed size (varint), limited to 31-bit
      // The size is stored as little-endian varint, from 1 to 5 bytes (as we allow up to 2^31 sizes only)
      // The upper bit of each byte indicates if there is another byte to read to compute the size
      // Please see format details at https://github.com/google/snappy/blob/master/format_description.txt 
      uncompressed_size = *cur++;
      if (uncompressed_size > 0x7f) {
        uint32_t c        = (cur < end) ? *cur++ : 0;
        uncompressed_size = (uncompressed_size & 0x7f) | (c << 7);
        // Check if the most significant bit is set, this indicates we need to read the next byte
        // (maybe even more) to compute the uncompressed size
        // We do it several time stopping if 1) MSB is cleared or 2) we see that the size is >= 2^31
        // which we cannot handle  
        if (uncompressed_size >= (0x80 << 7)) {
          c                 = (cur < end) ? *cur++ : 0;
          uncompressed_size = (uncompressed_size & ((0x7f << 7) | 0x7f)) | (c << 14);
          if (uncompressed_size >= (0x80 << 14)) {
            c = (cur < end) ? *cur++ : 0;
            uncompressed_size =
              (uncompressed_size & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | (c << 21);
            if (uncompressed_size >= (0x80 << 21)) {
              c = (cur < end) ? *cur++ : 0;
              // Snappy format alllows uncompressed sizes larger than 2^31
              // We generate an error in this case
              if (c < 0x8)
                uncompressed_size =
                  (uncompressed_size & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) |
                  (c << 28);
              else
                uncompressed_size = 0;
            }
          }
        }
      }
    }
    device_out_bytes[strm_id] = uncompressed_size;
  }
}

/**
 * @brief Snappy decompression kernel
 * See http://github.com/google/snappy/blob/master/format_description.txt
 *
 * blockDim {DECOMP_THREADS_PER_BLOCK,1,1}
 *
 * @param[in] inputs Source & destination information per block
 * @param[out] outputs Decompression status per block
 **/
__global__ void __launch_bounds__(DECOMP_THREADS_PER_BLOCK) unsnap_kernel(
    const void* const* __restrict__ device_in_ptr,
    const uint64_t* __restrict__ device_in_bytes,
    void* const* __restrict__ device_out_ptr,
    const uint64_t* __restrict__ device_out_available_bytes,
    nvcompStatus_t* const __restrict__ outputs,
    uint64_t* __restrict__ device_out_bytes)
{
  const int ix_chunk = blockIdx.x;
  do_unsnap(reinterpret_cast<const uint8_t*>(device_in_ptr[ix_chunk]),
      device_in_bytes[ix_chunk],
      reinterpret_cast<uint8_t*>(device_out_ptr[ix_chunk]),
      device_out_available_bytes ? device_out_available_bytes[ix_chunk] : 0,
      outputs ? &outputs[ix_chunk] : nullptr,
      device_out_bytes ? &device_out_bytes[ix_chunk] : nullptr);
}

void gpu_snap(
  const void* const* device_in_ptr,
  const size_t* device_in_bytes,
  void* const* device_out_ptr,
  const size_t* device_out_available_bytes,
  gpu_snappy_status_s *outputs,
  size_t* device_out_bytes,
  int count,
  cudaStream_t stream)
{
  dim3 dim_block(COMP_THREADS_PER_BLOCK, 1);  
  dim3 dim_grid(count, 1);
  if (count > 0) { snap_kernel<<<dim_grid, dim_block, 0, stream>>>(
    device_in_ptr, device_in_bytes, device_out_ptr, device_out_available_bytes,
      outputs, device_out_bytes); }
  CudaUtils::check_last_error("Failed to launch Snappy compression CUDA kernel gpu_snap");
}

void gpu_unsnap(
    const void* const* device_in_ptr,
    const size_t* device_in_bytes,
    void* const* device_out_ptr,
    const size_t* device_out_available_bytes,
    nvcompStatus_t* outputs,
    size_t* device_out_bytes,
    int count,
    cudaStream_t stream)
{
  uint32_t count32 = (count > 0) ? count : 0;
  dim3 dim_block(DECOMP_THREADS_PER_BLOCK, 1);     
  dim3 dim_grid(count32, 1);  // TODO: Check max grid dimensions vs max expected count

  unsnap_kernel<<<dim_grid, dim_block, 0, stream>>>(
    device_in_ptr, device_in_bytes, device_out_ptr, device_out_available_bytes,
      outputs, device_out_bytes);
  CudaUtils::check_last_error("Failed to launch Snappy decompression CUDA kernel gpu_unsnap");
}

void gpu_get_uncompressed_sizes(
  const void* const* device_in_ptr,
  const size_t* device_in_bytes,
  size_t* device_out_bytes,
  int count,
  cudaStream_t stream)
{
  dim3 dim_block(32, 1);
  dim3 dim_grid(count, 1);

  get_uncompressed_sizes_kernel<<<dim_grid, dim_block, 0, stream>>>(
    device_in_ptr, device_in_bytes, device_out_bytes);
  CudaUtils::check_last_error("Failed to run Snappy kernel gpu_get_uncompressed_sizes");
}

} // nvcomp namespace

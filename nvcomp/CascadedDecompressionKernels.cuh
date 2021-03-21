/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVCOMP_GPUKERNELS_H
#define NVCOMP_GPUKERNELS_H

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <cub/cub.cuh>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "unpack.h"

namespace nvcomp
{

template <typename V, typename R>
__global__ void vecMultKernel(
    const V* const a, const R* const b, V* const out, const size_t num)
{
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid < num) {
    out[tid] = a[tid] * b[tid];
  }
}

template <typename OUT, typename IN>
__global__ void
convertKernel(const IN* const d_input, OUT* const d_output, const size_t num)
{
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid < num) {
    d_output[tid] = static_cast<OUT>(d_input[tid]);
  }
}

template <typename OUT, typename IN>
__global__ void unpackBytesKernel(
    const void* const d_input,
    OUT* const d_output,
    const unsigned char numBits,
    const IN minValue,
    const size_t num)
{
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid < num) {
    d_output[tid]
        = static_cast<OUT>(unpackBytes(d_input, numBits, minValue, tid));
  }
}

template <typename runT>
__device__ void binarySearch(
    size_t& ind,
    size_t& offset,
    const runT* scan,
    const size_t size,
    size_t val)
{
  size_t low = 0;
  size_t high = size - 1;
  while (high > low) {
    size_t mid = (low + high) / 2;
    if (scan[mid] <= val)
      low = mid + 1;
    else
      high = mid;
  }
  if (low > 0) {
    offset = val - scan[low - 1];
  } else {
    offset = val;
  }
  ind = low;
}

template <typename runT, int threadBlock, int elemsPerThread>
__global__ void searchBlockBoundaries(
    size_t* start_ind,
    size_t* start_off,
    const size_t num_blocks,
    const size_t inputSize,
    const runT* runs_scan)
{
  const int smemLimit = threadBlock * elemsPerThread;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < num_blocks) {
    binarySearch(
        start_ind[tid], start_off[tid], runs_scan, inputSize, tid * smemLimit);
  }
  if (tid == num_blocks - 1) {
    start_ind[num_blocks] = inputSize - 1;
    start_off[num_blocks] = 0;
  }
}

template <
    typename inputT,
    typename outputT,
    typename runT,
    int threadBlock,
    int elemsPerThread,
    bool delta>
__global__ void expandRLEDelta(
    outputT* output,
    const size_t outputSize,
    const inputT* input,
    const runT* runs_scan,
    const inputT* delta_scan,
    const size_t* start_ind,
    const size_t* start_off)
{
  // limit of our expansion determined by available smem
  const int smemLimit = threadBlock * elemsPerThread;

  // all shared memory allocations
  __shared__ inputT in_vals[smemLimit];
  __shared__ inputT out_vals[smemLimit];
  __shared__ int offs[smemLimit];
  __shared__ size_t base_scan;
  __shared__ inputT base_delta;
  __shared__ size_t block_start_ind, block_start_off;
  __shared__ size_t block_end_ind;

// init offs with all zeros
#pragma unroll
  for (int i = 0; i < elemsPerThread; i++)
    offs[threadIdx.x + i * threadBlock] = 0;

  // find input indices and offsets based on output block index
  if (threadIdx.x == 0) {
    block_start_ind = start_ind[blockIdx.x];
    block_start_off = start_off[blockIdx.x];
    block_end_ind = start_ind[blockIdx.x + 1];
  }

  __syncthreads();

  // store values and block offset in shared memory
  for (int i = 0; i < elemsPerThread; i++)
    if (block_start_ind + threadIdx.x + threadBlock * i <= block_end_ind)
      in_vals[threadIdx.x + threadBlock * i]
          = input[block_start_ind + threadIdx.x + threadBlock * i];

  if (threadIdx.x == 0) {
    // starting scan value for this block (-1 because inclusive)
    if (block_start_ind > 0)
      base_scan = runs_scan[block_start_ind - 1];
    else
      base_scan = 0;
    // base delta value for this block
    if (delta) {
      if (block_start_ind > 0)
        base_delta = delta_scan[block_start_ind - 1]
                     + block_start_off * input[block_start_ind];
      else
        base_delta = block_start_off * input[block_start_ind];
    } else
      base_delta = 0;
  }

  __syncthreads();

  // store 1s in scan positions until smem limit
  // 0 0 0 0 1 0 0 1 0 0 0 0
  for (int i = 0; i < elemsPerThread; i++)
    if (block_start_ind + threadIdx.x + threadBlock * i <= block_end_ind) {
      int s = runs_scan[block_start_ind + threadIdx.x + threadBlock * i]
              - base_scan - block_start_off;
      if (s >= 0 && s < smemLimit)
        offs[s] = 1;
    }

  __syncthreads();

  // perform inclusive scan
  // 0 0 0 0 1 1 1 2 2 2 2 2
  typedef cub::BlockScan<int, threadBlock> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  int thread_data[elemsPerThread];
#pragma unroll
  for (int i = 0; i < elemsPerThread; i++)
    thread_data[i] = offs[threadIdx.x * elemsPerThread + i];
  BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);

// write values in their positions in smem
// A A A A B B B C C C C C
#pragma unroll
  for (int i = 0; i < elemsPerThread; i++)
    out_vals[threadIdx.x * elemsPerThread + i] = in_vals[thread_data[i]];

  __syncthreads();

  if (delta) {
    // now perform the delta expand - simply another scan
    // A 2A 3A 3A+B 3A+2B 3A+3B 3A+3B+C 3A+3B+2C ...
    typedef cub::BlockScan<inputT, threadBlock> BlockScanT;
    __shared__ typename BlockScanT::TempStorage temp_storage_t;
    inputT thread_data_t[elemsPerThread];
#pragma unroll
    for (int i = 0; i < elemsPerThread; i++)
      thread_data_t[i] = out_vals[threadIdx.x * elemsPerThread + i];
    BlockScanT(temp_storage_t).InclusiveSum(thread_data_t, thread_data_t);

// store in shared memory
#pragma unroll
    for (int i = 0; i < elemsPerThread; i++)
      out_vals[threadIdx.x * elemsPerThread + i] = thread_data_t[i];

    __syncthreads();
  }

// store final values to global memory
// here we also do type conversion if necessary
#pragma unroll
  for (int i = 0; i < elemsPerThread; i++) {
    size_t outputPos
        = threadIdx.x + threadBlock * i + base_scan + block_start_off;
    if (outputPos < outputSize)
      output[outputPos]
          = (outputT)(out_vals[threadIdx.x + threadBlock * i] + base_delta);
  }
}

} // namespace nvcomp

#endif

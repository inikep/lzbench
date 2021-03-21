/*
 * Copyright (c) Copyright-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "BitPackGPU.h"
#include "DeltaGPU.h"
#include "CascadedSelectorKernels.h"
#include "TempSpaceBroker.h"
#include "common.h"
#include "CascadedCommon.h"
#include "nvcomp.hpp"
#include "type_macros.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nvcomp
{

namespace
{

/**************************************************************************
 * Device Functions
 *************************************************************************/

// Device function to perform RLE on a buffer per block
template <typename VALUE, typename COUNT, int BLOCK_SIZE, int TILE_SIZE>
__device__ void deviceRLEKernel(
    COUNT* const runBuffer,
    VALUE* const valBuffer,
    COUNT* const prefix,
    size_t const num,
    int ITEMS_PER_THREAD)
{
  COUNT sum = 0;
  {
    VALUE val = valBuffer[threadIdx.x * ITEMS_PER_THREAD];
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
      const int tid = threadIdx.x * ITEMS_PER_THREAD + i;
      VALUE nextVal;
      if (tid < num) {
        nextVal = valBuffer[tid + 1];
        sum += nextVal != val;
        val = nextVal;
      }
    }
  }

  __syncthreads();
  // prefixsum bit mask

  {
    typedef cub::BlockScan<COUNT, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    BlockScan(temp_storage).InclusiveSum(sum, sum);

    prefix[threadIdx.x + 1] = sum;
  }

  __syncthreads();

  {
    int outIdx = prefix[threadIdx.x];
    VALUE val = valBuffer[threadIdx.x * ITEMS_PER_THREAD];
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      const int tid = threadIdx.x * ITEMS_PER_THREAD + i;
      const VALUE nextVal = valBuffer[tid + 1];
      if (nextVal != val) {
        runBuffer[outIdx + 1] = tid + 1;
        val = nextVal;
        ++outIdx;
      }
    }
  }
}

// Compute min and max of a buffer
template <typename VALUE, typename COUNT, int BLOCK_SIZE, int TILE_SIZE>
__device__ void deviceFindMinMax(
    const size_t num,
    COUNT* const prefix,
    COUNT* const runBuffer,
    VALUE* const valBuffer,
    VALUE* const valBuffer2,
    bool const prevValFlag,
    COUNT* localRunMin,
    COUNT* localRunMax,
    VALUE* localValMin,
    VALUE* localValMax,
    uint64_t* const maxConnectedRun,
    bool const connectedFlag,
    bool const firstLayer)
{
  COUNT numCompacted = prefix[BLOCK_SIZE];

  if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) {
    COUNT remain = num % TILE_SIZE;
    runBuffer[numCompacted] = (remain == 0) ? (TILE_SIZE) : remain;
  }

  __syncthreads();

  COUNT cur_run = runBuffer[1] - runBuffer[0];
  COUNT cur_run_min = cur_run;
  COUNT cur_run_max = cur_run;

  VALUE cur_val = valBuffer[0];
  VALUE cur_val_min = valBuffer[0];
  VALUE cur_val_max = cur_val_min;

  for (int tid = threadIdx.x; tid < numCompacted; tid += BLOCK_SIZE) {
    cur_run = runBuffer[tid + 1] - runBuffer[tid];

    if (firstLayer && connectedFlag) {
      if (tid == 0) {
        if (prevValFlag)
          atomicAdd(
              (unsigned long long*)(&(maxConnectedRun[blockIdx.x])),
              (unsigned long long)(cur_run));
        else
          atomicAdd((unsigned long long*)(&(maxConnectedRun[blockIdx.x])), 0);
      }
    }

    int validx = runBuffer[tid];
    cur_val = valBuffer[validx];

    cur_run_min = (tid == threadIdx.x) ? cur_run : min(cur_run_min, cur_run);
    cur_run_max = (tid == threadIdx.x) ? cur_run : max(cur_run_max, cur_run);

    if (firstLayer)
      valBuffer2[tid] = cur_val;

    cur_val_min = (tid == threadIdx.x) ? cur_val : min(cur_val_min, cur_val);
    cur_val_max = (tid == threadIdx.x) ? cur_val : max(cur_val_max, cur_val);
  }

  __syncthreads();

  {
    typedef cub::BlockScan<COUNT, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    BlockScan(temp_storage).InclusiveScan(cur_run_min, cur_run_min, cub::Min());
    BlockScan(temp_storage).InclusiveScan(cur_run_max, cur_run_max, cub::Max());
  }

  __syncthreads();
  {
    typedef cub::BlockScan<VALUE, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    BlockScan(temp_storage).InclusiveScan(cur_val_min, cur_val_min, cub::Min());
    BlockScan(temp_storage).InclusiveScan(cur_val_max, cur_val_max, cub::Max());
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    *localRunMin = cur_run_min;
    *localValMin = cur_val_min;
  }
  if (threadIdx.x == (BLOCK_SIZE - 1)) {
    *localRunMax = cur_run_max;
    *localValMax = cur_val_max;
  }

  __syncthreads();
}
/**************************************************************
 * Kernels
 *************************************************************/

// Kernel to perform the fused compression on a sample of the input
template <typename VALUE, typename COUNT, int BLOCK_SIZE, int TILE_SIZE>
__global__ void SampleFusedKernel(
    const VALUE* const in,
    size_t* const sample_offsets,
    const size_t num,
    unsigned long long int* const sizeBuffer

)
{
  __shared__ size_t s_offset;

  if (threadIdx.x == 0) {
    s_offset = sample_offsets[blockIdx.x];
  }
  __syncthreads();

  constexpr const int ITEMS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ COUNT prefix[BLOCK_SIZE + 1];
  __shared__ COUNT runBuffer[TILE_SIZE + 1];
  __shared__ VALUE valBuffer1stRLE[TILE_SIZE + 1];
  __shared__ VALUE valBuffer2ndRLE[TILE_SIZE + 1];

  __shared__ COUNT localRunMinFor1stRLE;
  __shared__ COUNT localRunMaxFor1stRLE;
  __shared__ VALUE localValMinFor1stRLE;
  __shared__ VALUE localValMaxFor1stRLE;

  __shared__ COUNT localRunMinFor2ndRLE;
  __shared__ COUNT localRunMaxFor2ndRLE;
  __shared__ VALUE localValMinFor2ndRLE;
  __shared__ VALUE localValMaxFor2ndRLE;

  // load data
  __syncthreads();
  for (int tid = threadIdx.x; tid < TILE_SIZE; tid += BLOCK_SIZE) {
    int gTid = tid + s_offset;
    if (tid < num) {
      valBuffer1stRLE[tid] = in[gTid];
    } else {
      int maxTid = num + s_offset - 1;
      valBuffer1stRLE[tid] = in[maxTid];
    }
  }

  if (threadIdx.x == 0) {
    prefix[0] = 0;
    runBuffer[0] = 0;
  }

  __syncthreads();

// Have each block run RLE on a tile of sampled input
  size_t tile_size = TILE_SIZE;
  deviceRLEKernel<VALUE, COUNT, BLOCK_SIZE, TILE_SIZE>(
      runBuffer, valBuffer1stRLE, prefix, tile_size, ITEMS_PER_THREAD);

  __syncthreads();

// Find min and max values used to estimate compression ratio
  deviceFindMinMax<VALUE, COUNT, BLOCK_SIZE, TILE_SIZE>(
     num,
      prefix,
      runBuffer,
      valBuffer1stRLE,
      valBuffer2ndRLE,
      false,
      &localRunMinFor1stRLE,
      &localRunMaxFor1stRLE,
      &localValMinFor1stRLE,
      &localValMaxFor1stRLE,
      NULL,
      false,
      true);

  __syncthreads();

  // Initialize input for 2nd RLE stage testing
  size_t numOutFor1stRLE = prefix[BLOCK_SIZE];
  size_t num2ndRLE = prefix[BLOCK_SIZE];
  const int ITEMS_PER_THREAD2 = ceil((double)num2ndRLE / (double)BLOCK_SIZE);

  for (int ph = 0; ph < ITEMS_PER_THREAD2; ph++) {
    int tid = ph * BLOCK_SIZE + threadIdx.x;
    if (tid == 0) {
      valBuffer1stRLE[tid] = in[0];
    } else {
      valBuffer1stRLE[tid] = valBuffer2ndRLE[tid] - valBuffer2ndRLE[tid - 1];
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    prefix[0] = 0;
    runBuffer[0] = 0;
  }

  __syncthreads();

  // Run 2nd RLE on output from first tested stage
  deviceRLEKernel<VALUE, COUNT, BLOCK_SIZE, TILE_SIZE>(
      runBuffer, valBuffer1stRLE, prefix, (size_t)num2ndRLE, ITEMS_PER_THREAD2);

  __syncthreads();

  size_t numOutFor2ndRLE = prefix[BLOCK_SIZE];

  // Compute min and max for 2nd RLE stage
  deviceFindMinMax<VALUE, COUNT, BLOCK_SIZE, TILE_SIZE>(
      num2ndRLE,
      prefix,
      runBuffer,
      valBuffer1stRLE,
      valBuffer2ndRLE,
      false,
      &localRunMinFor2ndRLE,
      &localRunMaxFor2ndRLE,
      &localValMinFor2ndRLE,
      &localValMaxFor2ndRLE,
      NULL,
      false,
      false);

  __syncthreads();

  // Thread 0 computes the statistics of each compression scheme
  // for the data the block processed
  if (threadIdx.x == 0) {
    VALUE valRangeFor1stRLE = localValMaxFor1stRLE - (localValMinFor1stRLE);
    COUNT runRangeFor1stRLE = localRunMaxFor1stRLE - (localRunMinFor1stRLE);
    size_t valBitsFor1stRLE;
    size_t runBitsFor1stRLE;
    size_t valBitsFor2ndRLE;
    size_t runBitsFor2ndRLE;

    // Compute number of bits needed for VALS if using 1 RLE
    if (sizeof(valRangeFor1stRLE) > sizeof(int)) {
      valBitsFor1stRLE
          = sizeof(long long int) * 8
            - __clzll(static_cast<long long int>(valRangeFor1stRLE));
    } else {
      valBitsFor1stRLE
          = sizeof(int) * 8 - __clz(static_cast<int>(valRangeFor1stRLE));
    }

    // Compute number of bits needed for RUNS if using 1 RLE
    if (sizeof(runRangeFor1stRLE) > sizeof(int)) {
      runBitsFor1stRLE
          = sizeof(long long int) * 8
            - __clzll(static_cast<long long int>(runRangeFor1stRLE));
    } else {

      runBitsFor1stRLE
          = sizeof(int) * 8 - __clz(static_cast<int>(runRangeFor1stRLE));
    }

    VALUE valRangeFor2ndRLE = localValMaxFor2ndRLE - (localValMinFor2ndRLE);
    COUNT runRangeFor2ndRLE = localRunMaxFor2ndRLE - (localRunMinFor2ndRLE);

    // Compute number of bits needed for VALS if using 2 RLEs
    if (sizeof(valRangeFor2ndRLE) > sizeof(int)) {
      valBitsFor2ndRLE
          = sizeof(long long int) * 8
            - __clzll(static_cast<long long int>(valRangeFor2ndRLE));
    } else {

      valBitsFor2ndRLE
          = sizeof(int) * 8 - __clz(static_cast<int>(valRangeFor2ndRLE));
    }

    // Compute number of bits needed for RUNS if using 2 RLEs
    if (sizeof(runRangeFor2ndRLE) > sizeof(int)) {
      runBitsFor2ndRLE
          = sizeof(long long int) * 8
            - __clzll(static_cast<long long int>(runRangeFor2ndRLE));
    } else {

      runBitsFor2ndRLE
         = sizeof(int) * 8 - __clz(static_cast<int>(runRangeFor2ndRLE));
    }

    // Ensure tha tthe number of bits will be at least 1
    valBitsFor1stRLE
        = max((unsigned long long)valBitsFor1stRLE, (unsigned long long)1);
    valBitsFor2ndRLE
        = max((unsigned long long)valBitsFor2ndRLE, (unsigned long long)1);
    runBitsFor1stRLE
        = max((unsigned long long)runBitsFor1stRLE, (unsigned long long)1);
    runBitsFor2ndRLE
        = max((unsigned long long)runBitsFor2ndRLE, (unsigned long long)1);

    // Compute sizes of each compressed output of vals and runs for each stage
    size_t run1stRLE = runBitsFor1stRLE * numOutFor1stRLE;
    run1stRLE = roundUpTo(roundUpDiv(run1stRLE, 8ULL), sizeof(COUNT));
    run1stRLE = roundUpTo(run1stRLE, sizeof(size_t));

    size_t val1stRLE = valBitsFor1stRLE * numOutFor1stRLE;
    val1stRLE = roundUpTo(roundUpDiv(val1stRLE, 8ULL), sizeof(VALUE));
    val1stRLE = roundUpTo(val1stRLE, sizeof(size_t));

    size_t val1stDelta = valBitsFor2ndRLE * numOutFor1stRLE;
    val1stDelta = roundUpTo(roundUpDiv(val1stDelta, 8ULL), sizeof(VALUE));
    val1stDelta = roundUpTo(val1stDelta, sizeof(size_t));

    size_t run2ndRLE = runBitsFor2ndRLE * numOutFor2ndRLE;
    run2ndRLE = roundUpTo(roundUpDiv(run2ndRLE, 8ULL), sizeof(COUNT));
    run2ndRLE = roundUpTo(run2ndRLE, sizeof(size_t));

    size_t val2ndRLE = valBitsFor2ndRLE * numOutFor2ndRLE;
    val2ndRLE = roundUpTo(roundUpDiv(val2ndRLE, 8ULL), sizeof(VALUE));
    val2ndRLE = roundUpTo(val2ndRLE, sizeof(size_t));

    // Compute the final compressed size of each compression scheme
    size_t R0D0B1 = valBitsFor1stRLE * num;
    R0D0B1 = roundUpTo(roundUpDiv(R0D0B1, 8ULL), sizeof(VALUE));
    R0D0B1 = roundUpTo(R0D0B1, sizeof(size_t));
    size_t R0D1B1 = valBitsFor2ndRLE * num;
    R0D1B1 = roundUpTo(roundUpDiv(R0D1B1, 8ULL), sizeof(VALUE));
    R0D1B1 = roundUpTo(R0D1B1, sizeof(size_t));

    size_t R1D0B1 = run1stRLE + val1stRLE;
    size_t R1D1B1 = run1stRLE + val1stDelta;
    size_t R2D1B1 = run1stRLE + run2ndRLE + val2ndRLE;

    // Add total compression for block to overall sum of all samples
    atomicAdd(&(sizeBuffer[0]), static_cast<unsigned long long int>(R0D0B1));
    atomicAdd(&(sizeBuffer[1]), static_cast<unsigned long long int>(R0D1B1));
    atomicAdd(&(sizeBuffer[2]), static_cast<unsigned long long int>(R1D0B1));
    atomicAdd(&(sizeBuffer[3]), static_cast<unsigned long long int>(R1D1B1));
    atomicAdd(&(sizeBuffer[4]), static_cast<unsigned long long int>(R2D1B1));
  }
}


/******************************************************************************
 * Internal function calls
******************************************************************************/

/*
 *@brief Selected a cascaded compression scheme by Sampling Fast Selector
 *@param in The input memory location on the GPU
 *@sample_ptrs The input data offsets for each samples
 *@param workspace The workspace memory location on the GPU
 *@param workspaceSize The size of the workspace memory in bytes
 *@param maxNum The number of elements in a sample
 *@param outSizeBuffer The buffer for the size of compreesed data for all
 *schemesin bytes (output)
 *@numSamples The number of samples
 *@NUM_SCHMES The number of cascaded schemes.
 *@param stream The stream to execute the kernel on
 */

template <typename VALUE, typename COUNT>
void SampleFusedInternal(
    const void* const in,
    size_t* const sample_ptrs,
    void* const workspace,
    const size_t workspaceSize,
    const size_t maxNum,
    size_t* const outsizeBuffer,
    size_t const numSamples,
    const int NUM_SCHEMES,
    cudaStream_t stream)

{

  constexpr const int BLOCK_SIZE = 128;
  constexpr const int SAMPLE_TILE_SIZE = 1024;

  if (NUM_SCHEMES != 5) {
    throw std::runtime_error("Number of schemes should be 5\n");
  }

  const size_t grid_size = numSamples;
  const dim3 grid(grid_size);
  const dim3 block(BLOCK_SIZE);

  unsigned long long int* d_sizeBuffer;

  TempSpaceBroker tempSpace(workspace, workspaceSize);
  tempSpace.reserve(&d_sizeBuffer, NUM_SCHEMES);

  cudaMemsetAsync(d_sizeBuffer, 0, sizeof(*d_sizeBuffer) * NUM_SCHEMES, stream);

  const VALUE* const inTyped = static_cast<const VALUE*>(in);

  SampleFusedKernel<VALUE, COUNT, BLOCK_SIZE, SAMPLE_TILE_SIZE>
      <<<grid, block, 0, stream>>>(inTyped, sample_ptrs, maxNum, d_sizeBuffer);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Fail to launch SampleFusedKernel: " + std::to_string(err));
  }

  unsigned long long int size_buffer[NUM_SCHEMES];
  err = cudaMemcpyAsync(
      size_buffer,
      d_sizeBuffer,
      sizeof(unsigned long long int) * NUM_SCHEMES,
      cudaMemcpyDeviceToHost,
      stream);
  err = cudaStreamSynchronize(stream);

  if (err != cudaSuccess) {
    throw std::runtime_error("size buffer cuda memcpy failed\n");
  }

  for (int i = 0; i < NUM_SCHEMES; i++) {
    outsizeBuffer[i] = (size_t)size_buffer[i];
  }
}

/*
 *@brief Selected a cascaded compression scheme by Sampling Fast Selector
 *
 *@param in The input memory location on the GPU
 *@sample_ptrs The input data offsets for each samples
 *@param temp_ptr The workspace memory location on the GPU
 *@param temp_bytes The size of the workspace memory in bytes
 *@param outsize The buffer for the size of compreesed data for all schemesin
 *bytes (output)
 *@param numSamples The number of samples.
 *@param num_schemes The number of cascaded schemes.
 *@param stream The stream to execute the kernel on
 */
template <typename valT, typename runT>
void SampleFusedOption_internal(
    const void* const in,
    size_t* const sample_ptrs,
    const size_t in_bytes,
    void* const temp_ptr,
    const size_t temp_bytes,
    size_t* outsize,
    size_t const numSamples,
    const int num_schemes,
    cudaStream_t stream)
{

  const size_t maxNum = in_bytes / sizeof(valT);

  SampleFusedInternal<valT, runT>(
      in,
      sample_ptrs,
      temp_ptr,
      temp_bytes,
      maxNum,
      outsize,
      numSamples,
      num_schemes,
      stream);
}

} // namespace

void SamplingFastOption(
    const void* const in,
    size_t* const sample_offsets,
    const size_t sample_bytes,
    const size_t num_samples,
    const nvcompType_t in_type,
    void* const workspace,
    const size_t workspaceSize,
    size_t* outsizeBuffer,
    int num_schemes,
    cudaStream_t stream)
{

  const nvcompType_t countType
      = selectRunsType(sample_bytes / sizeOfnvcompType(in_type));

  NVCOMP_TYPE_TWO_SWITCH(
      in_type,
      countType,
      SampleFusedOption_internal,
      in,
      sample_offsets,
      sample_bytes,
      workspace,
      workspaceSize,
      outsizeBuffer,
      num_samples,
      num_schemes,
      stream);
}

} // namespace nvcomp

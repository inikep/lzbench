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

#include "RunLengthEncodeGPU.h"
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

#include <cassert>
#include <stdexcept>
#include <string>

namespace nvcomp
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{
constexpr const size_t ALIGN_OFFSET = 256;
constexpr const int WARP_SIZE = 32;
constexpr const int GLOBAL_TILE_SIZE = 1024;
} // namespace

/******************************************************************************
 * KENRELS ********************************************************************
 *****************************************************************************/

namespace
{

template <typename T, int NUM_THREADS>
__device__ T warpSum(T const initVal)
{
  constexpr const uint32_t mask
      = NUM_THREADS < WARP_SIZE ? (1u << NUM_THREADS) - 1 : 0xffffffff;
  T val = initVal;
  for (int d = NUM_THREADS / 2; d > 0; d /= 2) {
    val += __shfl_down_sync(mask, val, d, NUM_THREADS);
  }

  return val;
}

template <typename T, int BLOCK_SIZE>
__device__ T cooperativeSum(T const initVal, T* const buffer)
{
  // first all warps reduce to single value
  assert(BLOCK_SIZE % WARP_SIZE == 0);
  assert(BLOCK_SIZE <= WARP_SIZE * WARP_SIZE);

  T val = warpSum<T, WARP_SIZE>(initVal);
  if (threadIdx.x % WARP_SIZE == 0) {
    buffer[threadIdx.x / WARP_SIZE] = val;
  }
  __syncthreads();

  if (threadIdx.x < (BLOCK_SIZE / WARP_SIZE)) {
    val = warpSum<T, BLOCK_SIZE / WARP_SIZE>(buffer[threadIdx.x]);
  }

  return val;
}

/**
 * @brief This kernel produces the block sizes for a prefixsum in a subsequent
 * kernel.
 *
 * @tparam VALUE The value type.
 * @tparam RUN The run count type.
 * @param in The input data.
 * @param num The size of the input data.
 * @param blockSize The location to write the block sizes (output).
 */
template <typename VALUE, typename RUN, int BLOCK_SIZE, int TILE_SIZE>
__global__ void rleInitKernel(
    const VALUE* const in,
    const size_t* const numInDevice,
    RUN* const blockSize)
{
  constexpr const int ITEMS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;
  // the algorithm here is to keep reducing "chunks" to a start and end marker

  const int num = static_cast<int>(*numInDevice);

  if (blockIdx.x * TILE_SIZE < num) {
    // we load the preceding value in the first spot
    __shared__ VALUE valBuffer[TILE_SIZE + 1];
    __shared__ RUN buffer[BLOCK_SIZE / WARP_SIZE];

    if (threadIdx.x == 0) {
      valBuffer[0]
          = blockIdx.x > 0 ? in[blockIdx.x * TILE_SIZE - 1] : (in[0] + 1);
    }
    for (int tid = threadIdx.x; tid < TILE_SIZE; tid += BLOCK_SIZE) {
      const int gTid = tid + blockIdx.x * TILE_SIZE;
      // cooperatively populate valBuffer and runBuffer
      if (gTid < num) {
        valBuffer[tid + 1] = in[gTid];
      } else {
        valBuffer[tid + 1] = in[num - 1];
      }
    }

    __syncthreads();

    // build bit mask
    VALUE val = valBuffer[threadIdx.x * ITEMS_PER_THREAD];
    RUN sum = 0;
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
      const int tid = threadIdx.x * ITEMS_PER_THREAD + i;
      const VALUE nextVal = valBuffer[tid + 1];
      sum += nextVal != val;
      val = nextVal;
    }

    sum = cooperativeSum<RUN, BLOCK_SIZE>(sum, buffer);
    if (threadIdx.x == 0) {
      blockSize[blockIdx.x] = sum;
    }
  } else if (threadIdx.x == 0) {
    blockSize[blockIdx.x] = 0;
  }

  if (blockIdx.x == gridDim.x - 1 && threadIdx.x == 0) {
    blockSize[gridDim.x] = 0;
  }
}

template <typename VALUE, typename RUN, int BLOCK_SIZE, int TILE_SIZE>
__global__ void rleReduceKernel(
    const VALUE* const in,
    const size_t* const numInDevice,
    const RUN* const blockPrefix,
    RUN* const blockStart,
    VALUE** const valsPtr,
    RUN** const runsPtr,
    size_t* const numOutDevice)
{
  constexpr const int ITEMS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;
  // the algorithm here is to keep reducing "chunks" to a start and end marker
  const int num = static_cast<int>(*numInDevice);

  if (blockIdx.x * TILE_SIZE < num) {
    VALUE* const vals = *valsPtr;
    RUN* const runs = *runsPtr;

    // we load the preceding value in the first spot
    __shared__ VALUE valBuffer[TILE_SIZE + 1];

    // we store the sum in the last spot
    __shared__ RUN prefix[BLOCK_SIZE + 1];

    if (threadIdx.x == 0) {
      valBuffer[0]
          = blockIdx.x > 0 ? in[blockIdx.x * TILE_SIZE - 1] : (in[0] + 1);
    }
    for (int tid = threadIdx.x; tid < TILE_SIZE; tid += BLOCK_SIZE) {
      const int gTid = tid + blockIdx.x * TILE_SIZE;
      // cooperatively populate valBuffer and runBuffer
      if (gTid < num) {
        valBuffer[tid + 1] = in[gTid];
      } else {
        valBuffer[tid + 1] = in[num - 1];
      }
    }

    __syncthreads();

    // build bit mask
    RUN sum = 0;
    {
      VALUE val = valBuffer[threadIdx.x * ITEMS_PER_THREAD];
      for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        const int tid = threadIdx.x * ITEMS_PER_THREAD + i;
        const VALUE nextVal = valBuffer[tid + 1];
        sum += nextVal != val;
        val = nextVal;
      }
    }

    __syncthreads();

    // prefixsum bit mask
    {
      typedef cub::BlockScan<RUN, BLOCK_SIZE> BlockScan;
      __shared__ typename BlockScan::TempStorage temp_storage;

      BlockScan(temp_storage).ExclusiveSum(sum, sum);

      prefix[threadIdx.x] = sum;
      if (threadIdx.x == 0) {
        prefix[BLOCK_SIZE]
            = blockPrefix[blockIdx.x + 1] - blockPrefix[blockIdx.x];
      }
    }

    __syncthreads();

    __shared__ RUN runBuffer[TILE_SIZE + 1];

    // do local run length encoding with an undifferentiated run count
    {
      int outIdx = prefix[threadIdx.x];
      VALUE val = valBuffer[threadIdx.x * ITEMS_PER_THREAD];
      for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        const int tid = threadIdx.x * ITEMS_PER_THREAD + i;
        const VALUE nextVal = valBuffer[tid + 1];
        if (nextVal != val) {
          runBuffer[outIdx] = tid;

          val = nextVal;
          ++outIdx;
        }
      }
    }

    const RUN numCompacted = prefix[BLOCK_SIZE];
    if (threadIdx.x == 0) {
      runBuffer[numCompacted] = ((blockIdx.x + 1) * TILE_SIZE >= num)
                                    ? ((num - 1) % TILE_SIZE) + 1
                                    : TILE_SIZE;
    }

    __syncthreads();

    // write back to global memory
    const RUN offset = blockPrefix[blockIdx.x];
    for (int tid = threadIdx.x; tid < numCompacted; tid += BLOCK_SIZE) {
      // runs still need to be differentiated -- the last one will need to the
      // number of values
      vals[offset + tid] = valBuffer[runBuffer[tid] + 1];
      assert(runBuffer[tid + 1] >= runBuffer[tid]);
      runs[offset + tid] = runBuffer[tid + 1] - runBuffer[tid];
    }

    if (threadIdx.x == 0) {
      blockStart[blockIdx.x] = runBuffer[0] + blockIdx.x * TILE_SIZE;
    }
  }
  if (blockIdx.x == gridDim.x - 1 && threadIdx.x == BLOCK_SIZE - 1) {
    *numOutDevice = blockPrefix[gridDim.x];
  }
}

/**
 * @brief Fix block join gaps, that is where the run count for a given number
 * fails to account for duplicates in the following block(s). This requires
 * that the first run in each block's output, not be differentiated.
 *
 * @param runs The almost finished runs.
 * @param blockPrefix The previously calculated block prefix.
 * @param num The number of entries.
 */
template <typename RUN, int BLOCK_SIZE, int TILE_SIZE>
__global__ void rleFinalizeKernel(
    RUN** const runsPtr,
    const RUN* const blockStart,
    const RUN* const blockPrefix,
    const size_t* const numInDevice)
{
  const int num = roundUpDiv(static_cast<int>(*numInDevice), TILE_SIZE);

  if (blockIdx.x * BLOCK_SIZE < num) {
    RUN* const runs = *runsPtr;

    // we load the blocks runs plus 1 extra
    __shared__ RUN prefixBuffer[BLOCK_SIZE + 1];

    int tid = threadIdx.x + blockIdx.x * BLOCK_SIZE;

    const RUN bp = tid < num ? blockPrefix[tid] : blockPrefix[num];
    prefixBuffer[threadIdx.x] = bp;

    if (threadIdx.x == 0) {
      prefixBuffer[BLOCK_SIZE] = blockPrefix[(blockIdx.x + 1) * BLOCK_SIZE];
    }

    __syncthreads();

    if (tid < num) {
      if (bp > 0 && (tid + 1 == num || bp < prefixBuffer[threadIdx.x + 1])) {
        // TODO: make this a binary search

        int low = 0;
        int high = tid;
        while (high > low) {
          const int mid = (low + high) / 2;
          if (blockPrefix[mid] == bp) {
            // keep searching down
            high = mid;
          } else {
            // start searching up
            low = mid + 1;
          }
        }

        // we need to fix the count for this block
        runs[bp - 1] += blockStart[tid] - low * TILE_SIZE;
      }
    }
  }
}
} // namespace

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

template <typename T>
size_t downstreamWorkspaceSize(const size_t num)
{
  return sizeof(T) * std::max(1024ULL, 3ULL * roundUpDiv(num, GLOBAL_TILE_SIZE))
         + sizeof(int);
}

template <typename T, typename U>
size_t requiredWorkspaceSizeTyped(const size_t num)
{
  // TODO: this assume large datatype
  T* inPtr = nullptr;
  T* valsPtr = nullptr;
  U* runsPtr = nullptr;
  size_t* numPtr = nullptr;

  size_t workspaceSize = 0;
  cudaError_t err = cub::DeviceRunLengthEncode::Encode(
      nullptr,
      workspaceSize,
      inPtr,
      valsPtr,
      runsPtr,
      numPtr,
      static_cast<int>(num),
      0);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to get workspace size: " + std::to_string(err));
  }

  workspaceSize = std::max(workspaceSize, downstreamWorkspaceSize<U>(num));

  return ALIGN_OFFSET + workspaceSize;
}

template <typename VALUE, typename COUNT>
void compressInternal(
    void* const workspace,
    const size_t workspaceSize,
    void* const outValues,
    void* const outCounts,
    size_t* numOutDevice,
    void const* const in,
    size_t const num,
    cudaStream_t stream)
{
  VALUE* const outValuesTyped = static_cast<VALUE*>(outValues);
  COUNT* const outCountsTyped = static_cast<COUNT*>(outCounts);
  const VALUE* const inTyped = static_cast<const VALUE*>(in);

  const size_t reqWorkspaceSize = RunLengthEncodeGPU::requiredWorkspaceSize(
      num, getnvcompType<VALUE>(), getnvcompType<COUNT>());
  if (workspaceSize < reqWorkspaceSize) {
    throw std::runtime_error(
        "Invalid workspace size: " + std::to_string(workspaceSize)
        + ", need at least " + std::to_string(reqWorkspaceSize));
  }

  void* const alignedWorkspace = align(workspace, ALIGN_OFFSET);
  size_t alignedWorkspaceSize
      = workspaceSize - relativeEndOffset(workspace, alignedWorkspace);

  cudaError_t err = cub::DeviceRunLengthEncode::Encode(
      alignedWorkspace,
      alignedWorkspaceSize,
      inTyped,
      outValuesTyped,
      outCountsTyped,
      numOutDevice,
      static_cast<int>(num),
      stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to schedule cub::DeviceRunLengthEncode::Encode() kernel: "
        + std::to_string(err));
  }
}

template <typename VALUE, typename COUNT>
void compressDownstreamInternal(
    void* const workspace,
    const size_t workspaceSize,
    void** const outValuesPtr,
    void** const outCountsPtr,
    size_t* numOutDevice,
    void const* const in,
    size_t const* numInDevice,
    const size_t maxNum,
    cudaStream_t stream)
{
  VALUE** const outValuesTypedPtr = reinterpret_cast<VALUE**>(outValuesPtr);
  COUNT** const outCountsTypedPtr = reinterpret_cast<COUNT**>(outCountsPtr);
  const VALUE* const inTyped = static_cast<const VALUE*>(in);

  const size_t reqWorkspaceSize = downstreamWorkspaceSize<COUNT>(maxNum);
  if (workspaceSize < reqWorkspaceSize) {
    throw std::runtime_error(
        "Invalid workspace size: " + std::to_string(workspaceSize)
        + ", need at least " + std::to_string(reqWorkspaceSize));
  }

  constexpr const int BLOCK_SIZE = 128;

  const dim3 grid(roundUpDiv(maxNum, GLOBAL_TILE_SIZE));
  const dim3 block(BLOCK_SIZE);

  TempSpaceBroker tempSpace(workspace, workspaceSize);

  COUNT* blockSizes;
  COUNT* blockPrefix;
  COUNT* blockStart;
  tempSpace.reserve(&blockSizes, grid.x);
  tempSpace.reserve(&blockPrefix, grid.x + 1);
  tempSpace.reserve(&blockStart, grid.x);

  void* const scanWorkspace = tempSpace.next();

  // TODO: expand such that the mask calculation is done across the entire
  // array, and the the prefixsum, and then reduction

  // get blocks sizes
  rleInitKernel<VALUE, COUNT, BLOCK_SIZE, GLOBAL_TILE_SIZE>
      <<<grid, block, 0, stream>>>(inTyped, numInDevice, blockSizes);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to launch rleInitKernel: " + std::to_string(err));
  }

  // get output locations
  size_t requiredSpace;
  err = cub::DeviceScan::ExclusiveSum(
      nullptr, requiredSpace, blockSizes, blockPrefix, grid.x + 1, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to query rleScanKernel: " + std::to_string(err));
  }

  size_t scanWorkspaceSize
      = std::max(1024 * sizeof(COUNT), maxNum * sizeof(COUNT));
  if (requiredSpace > scanWorkspaceSize) {
    throw std::runtime_error(
        "Too little workspace: " + std::to_string(scanWorkspaceSize) + ", need "
        + std::to_string(requiredSpace));
  }

  err = cub::DeviceScan::ExclusiveSum(
      scanWorkspace,
      scanWorkspaceSize,
      blockSizes,
      blockPrefix,
      grid.x + 1,
      stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to launch rleScanKernel: " + std::to_string(err)
        + ", with "
          "grid.x = "
        + std::to_string(grid.x + 1) + " items.");
  }

  // do actual compaction
  rleReduceKernel<VALUE, COUNT, BLOCK_SIZE, GLOBAL_TILE_SIZE>
      <<<grid, block, 0, stream>>>(
          inTyped,
          numInDevice,
          blockPrefix,
          blockStart,
          outValuesTypedPtr,
          outCountsTypedPtr,
          numOutDevice);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to launch rleReduceKernel: " + std::to_string(err));
  }

  // fix gaps
  rleFinalizeKernel<COUNT, BLOCK_SIZE, GLOBAL_TILE_SIZE>
      <<<dim3(roundUpDiv(grid.x, block.x)), block, 0, stream>>>(
          outCountsTypedPtr, blockStart, blockPrefix, numInDevice);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to launch rleFinalizeKernel: " + std::to_string(err));
  }
}

} // namespace

/******************************************************************************
 * PUBLIC STATIC FUNCTIONS ****************************************************
 *****************************************************************************/

void RunLengthEncodeGPU::compress(
    void* workspace,
    size_t workspaceSize,
    nvcompType_t valueType,
    void* const outValues,
    nvcompType_t countType,
    void* const outCounts,
    size_t* const numOutDevice,
    const void* const in,
    const size_t num,
    cudaStream_t stream)
{
  NVCOMP_TYPE_TWO_SWITCH(
      valueType,
      countType,
      compressInternal,
      workspace,
      workspaceSize,
      outValues,
      outCounts,
      numOutDevice,
      in,
      num,
      stream);
}

void RunLengthEncodeGPU::compressDownstream(
    void* workspace,
    size_t workspaceSize,
    nvcompType_t valueType,
    void** const outValuesPtr,
    nvcompType_t countType,
    void** const outCountsPtr,
    size_t* const numOutDevice,
    const void* const in,
    const size_t* numInDevice,
    const size_t maxNum,
    cudaStream_t stream)
{
  NVCOMP_TYPE_TWO_SWITCH(
      valueType,
      countType,
      compressDownstreamInternal,
      workspace,
      workspaceSize,
      outValuesPtr,
      outCountsPtr,
      numOutDevice,
      in,
      numInDevice,
      maxNum,
      stream);
}

size_t RunLengthEncodeGPU::requiredWorkspaceSize(
    const size_t num, const nvcompType_t valueType, const nvcompType_t runType)
{
  NVCOMP_TYPE_TWO_SWITCH_RETURN(
      valueType, runType, requiredWorkspaceSizeTyped, num);
}

} // namespace nvcomp

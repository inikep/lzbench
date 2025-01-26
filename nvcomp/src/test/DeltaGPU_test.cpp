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

#define CATCH_CONFIG_MAIN

#include "tests/catch.hpp"
#include "DeltaGPU.h"
#include "common.h"
#include "nvcomp.hpp"

#include "cuda_runtime.h"

#include <cstdlib>

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                     \
  {                                                                            \
    cudaError_t cudaStatus = call;                                             \
    if (cudaSuccess != cudaStatus) {                                           \
      fprintf(                                                                 \
          stderr,                                                              \
          "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s "   \
          "(%d).\n",                                                           \
          #call,                                                               \
          __LINE__,                                                            \
          __FILE__,                                                            \
          cudaGetErrorString(cudaStatus),                                      \
          cudaStatus);                                                         \
      abort();                                                                 \
    }                                                                          \
  }
#endif

using namespace nvcomp;

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

template <typename T>
__global__ void toGPU(
    T* const output,
    T const* const input,
    size_t const num,
    cudaStream_t stream)
{
  CUDA_RT_CALL(cudaMemcpyAsync(
      output, input, num * sizeof(T), cudaMemcpyHostToDevice, stream));
}

template <typename T>
__global__ void fromGPU(
    T* const output,
    T const* const input,
    size_t const num,
    cudaStream_t stream)
{
  CUDA_RT_CALL(cudaMemcpyAsync(
      output, input, num * sizeof(T), cudaMemcpyDeviceToHost, stream));
}

} // namespace

/******************************************************************************
 * UNIT TEST ******************************************************************
 *****************************************************************************/

TEST_CASE("compress_10Thousand_Test", "[small]")
{
  size_t const n = 10000;

  using T = int32_t;

  T *input, *inputHost;
  size_t const numBytes = n * sizeof(*input);

  CUDA_RT_CALL(cudaMalloc((void**)&input, numBytes));

  CUDA_RT_CALL(cudaMallocHost((void**)&inputHost, n * sizeof(*inputHost)));

  float const totalGB = numBytes / (1024.0 * 1024.0 * 1024.0);

  cudaStream_t stream;
  CUDA_RT_CALL(cudaStreamCreate(&stream));

  std::srand(0);

  T last = 0;
  for (size_t i = 0; i < n; ++i) {
    if (std::rand() % 3 == 0) {
      last = std::rand() % 1024;
    }
    inputHost[i] = last;
  }

  toGPU(input, inputHost, n, stream);

  T *output, *outputHost;
  T** outputPtr;

  CUDA_RT_CALL(cudaMalloc((void**)&output, numBytes));
  CUDA_RT_CALL(cudaMallocHost((void**)&outputHost, numBytes));

  CUDA_RT_CALL(cudaMalloc((void**)&outputPtr, sizeof(*outputPtr)));
  CUDA_RT_CALL(cudaMemcpy(
      outputPtr, &output, sizeof(*outputPtr), cudaMemcpyHostToDevice));

  size_t* inputSizePtr;
  CUDA_RT_CALL(cudaMalloc((void**)&inputSizePtr, sizeof(*inputSizePtr)));
  CUDA_RT_CALL(cudaMemcpy(
      inputSizePtr, &n, sizeof(*inputSizePtr), cudaMemcpyHostToDevice));

  void* workspace;
  size_t const workspaceSize = DeltaGPU::requiredWorkspaceSize(n, TypeOf<T>());
  CUDA_RT_CALL(cudaMalloc((void**)&workspace, workspaceSize));

  cudaEvent_t start, stop;

  CUDA_RT_CALL(cudaEventCreate(&start));
  CUDA_RT_CALL(cudaEventCreate(&stop));
  CUDA_RT_CALL(cudaEventRecord(start, stream));

  DeltaGPU::compress(
      workspace,
      workspaceSize,
      TypeOf<T>(),
      (void**)outputPtr,
      input,
      inputSizePtr,
      2 * n,
      stream);
  CUDA_RT_CALL(cudaEventRecord(stop, stream));

  CUDA_RT_CALL(cudaStreamSynchronize(stream));
  float time;
  CUDA_RT_CALL(cudaEventElapsedTime(&time, start, stop));

  fromGPU(outputHost, output, n, stream);
  CUDA_RT_CALL(cudaStreamSynchronize(stream));
  CUDA_RT_CALL(cudaStreamDestroy(stream));

  CUDA_RT_CALL(cudaFree(output));
  CUDA_RT_CALL(cudaFree(outputPtr));
  CUDA_RT_CALL(cudaFree(inputSizePtr));

  // compute Delta on host
  std::vector<T> expected{inputHost[0]};

  for (size_t i = 1; i < n; ++i) {
    expected.emplace_back(inputHost[i] - inputHost[i - 1]);
  }

  // verify output
  for (size_t i = 0; i < n; ++i) {
    CHECK(expected[i] == outputHost[i]);
  }

  CUDA_RT_CALL(cudaFreeHost(outputHost));

  CUDA_RT_CALL(cudaFree(input));
  CUDA_RT_CALL(cudaFreeHost(inputHost));
}

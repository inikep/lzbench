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
#include "RunLengthEncodeGPU.h"
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

template <typename T, typename V>
void compressAsyncTestRandom(const size_t n)
{
  T *input, *inputHost;
  size_t const numBytes = n * sizeof(*input);

  CUDA_RT_CALL(cudaMalloc((void**)&input, numBytes));

  CUDA_RT_CALL(cudaMallocHost((void**)&inputHost, n * sizeof(*inputHost)));

  float const totalGB = numBytes / (1024.0f * 1024.0f * 1024.0f);

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

  T *outputValues, *outputValuesHost;
  V *outputCounts, *outputCountsHost;

  CUDA_RT_CALL(cudaMalloc((void**)&outputValues, sizeof(*outputValues) * n));
  CUDA_RT_CALL(cudaMalloc((void**)&outputCounts, sizeof(*outputCounts) * n));
  CUDA_RT_CALL(
      cudaMallocHost((void**)&outputValuesHost, sizeof(*outputValuesHost) * n));
  CUDA_RT_CALL(
      cudaMallocHost((void**)&outputCountsHost, sizeof(*outputCountsHost) * n));

  void* workspace;
  const size_t maxNum = 2 * n;
  size_t const workspaceSize = RunLengthEncodeGPU::requiredWorkspaceSize(
      maxNum, TypeOf<T>(), TypeOf<V>());
  CUDA_RT_CALL(cudaMalloc((void**)&workspace, workspaceSize));

  // create on device inputs
  size_t* numInDevice;
  size_t* numOutDevice;
  T** outputValuesPtr;
  V** outputCountsPtr;
  CUDA_RT_CALL(cudaMalloc((void**)&numInDevice, sizeof(*numInDevice)));
  CUDA_RT_CALL(cudaMalloc((void**)&numOutDevice, sizeof(*numOutDevice)));
  CUDA_RT_CALL(cudaMalloc((void**)&outputValuesPtr, sizeof(*outputValuesPtr)));
  CUDA_RT_CALL(cudaMalloc((void**)&outputCountsPtr, sizeof(*outputCountsPtr)));

  CUDA_RT_CALL(cudaMemcpy(
      numInDevice, &n, sizeof(*numInDevice), cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(
      outputValuesPtr,
      &outputValues,
      sizeof(outputValues),
      cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(
      outputCountsPtr,
      &outputCounts,
      sizeof(outputCounts),
      cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;

  CUDA_RT_CALL(cudaEventCreate(&start));
  CUDA_RT_CALL(cudaEventCreate(&stop));
  CUDA_RT_CALL(cudaEventRecord(start, stream));

  RunLengthEncodeGPU::compressDownstream(
      workspace,
      workspaceSize,
      TypeOf<T>(),
      reinterpret_cast<void**>(outputValuesPtr),
      TypeOf<V>(),
      reinterpret_cast<void**>(outputCountsPtr),
      numOutDevice,
      input,
      numInDevice,
      maxNum,
      stream);
  CUDA_RT_CALL(cudaEventRecord(stop, stream));

  CUDA_RT_CALL(cudaStreamSynchronize(stream));
  float time;
  CUDA_RT_CALL(cudaEventElapsedTime(&time, start, stop));

  size_t numOut;
  CUDA_RT_CALL(cudaMemcpy(
      &numOut, numOutDevice, sizeof(numOut), cudaMemcpyDeviceToHost));

  fromGPU(outputValuesHost, outputValues, numOut, stream);
  fromGPU(outputCountsHost, outputCounts, numOut, stream);
  CUDA_RT_CALL(cudaStreamSynchronize(stream));
  CUDA_RT_CALL(cudaStreamDestroy(stream));

  CUDA_RT_CALL(cudaFree(outputValues));
  CUDA_RT_CALL(cudaFree(outputCounts));
  CUDA_RT_CALL(cudaFree(outputValuesPtr));
  CUDA_RT_CALL(cudaFree(outputCountsPtr));
  CUDA_RT_CALL(cudaFree(numOutDevice));
  CUDA_RT_CALL(cudaFree(numInDevice));

  // compute RLE on host
  std::vector<T> expectedValues{inputHost[0]};
  std::vector<V> expectedCounts{1};

  for (size_t i = 1; i < n; ++i) {
    if (inputHost[i] == expectedValues.back()) {
      ++expectedCounts.back();
    } else {
      expectedValues.emplace_back(inputHost[i]);
      expectedCounts.emplace_back(1);
    }
  }

  REQUIRE(expectedCounts.size() == numOut);

  // verify output
  for (size_t i = 0; i < expectedCounts.size(); ++i) {
    if (!(expectedValues[i] == outputValuesHost[i]
          && expectedCounts[i] == outputCountsHost[i])) {
      std::cerr << "i = " << i << " exp " << (int64_t)expectedCounts[i] << ":"
                << (int64_t)expectedValues[i] << " act "
                << (int64_t)outputCountsHost[i] << ":"
                << (int64_t)outputValuesHost[i] << std::endl;
    }

    CHECK(expectedValues[i] == outputValuesHost[i]);
    CHECK(expectedCounts[i] == outputCountsHost[i]);
  }

  CUDA_RT_CALL(cudaFreeHost(outputValuesHost));
  CUDA_RT_CALL(cudaFreeHost(outputCountsHost));

  CUDA_RT_CALL(cudaFree(input));
  CUDA_RT_CALL(cudaFreeHost(inputHost));
}

} // namespace

/******************************************************************************
 * UNIT TEST ******************************************************************
 *****************************************************************************/

TEST_CASE("compress_10Million_Test", "[small]")
{
  size_t const n = 10000000;

  using T = int32_t;
  using V = uint32_t;

  T *input, *inputHost;
  size_t const numBytes = n * sizeof(*input);

  CUDA_RT_CALL(cudaMalloc((void**)&input, numBytes));

  CUDA_RT_CALL(cudaMallocHost((void**)&inputHost, n * sizeof(*inputHost)));

  float const totalGB = numBytes / (1024.0f * 1024.0f * 1024.0f);

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

  T *outputValues, *outputValuesHost;
  V *outputCounts, *outputCountsHost;

  CUDA_RT_CALL(cudaMalloc((void**)&outputValues, sizeof(*outputValues) * n));
  CUDA_RT_CALL(cudaMalloc((void**)&outputCounts, sizeof(*outputCounts) * n));
  CUDA_RT_CALL(
      cudaMallocHost((void**)&outputValuesHost, sizeof(*outputValuesHost) * n));
  CUDA_RT_CALL(
      cudaMallocHost((void**)&outputCountsHost, sizeof(*outputCountsHost) * n));

  size_t* numOutDevice;
  CUDA_RT_CALL(cudaMalloc((void**)&numOutDevice, sizeof(*numOutDevice)));

  void* workspace;
  size_t const workspaceSize
      = RunLengthEncodeGPU::requiredWorkspaceSize(n, TypeOf<T>(), TypeOf<V>());
  CUDA_RT_CALL(cudaMalloc((void**)&workspace, workspaceSize));

  cudaEvent_t start, stop;

  CUDA_RT_CALL(cudaEventCreate(&start));
  CUDA_RT_CALL(cudaEventCreate(&stop));
  CUDA_RT_CALL(cudaEventRecord(start, stream));

  size_t numOut = 0;
  RunLengthEncodeGPU::compress(
      workspace,
      workspaceSize,
      TypeOf<T>(),
      outputValues,
      TypeOf<V>(),
      outputCounts,
      numOutDevice,
      input,
      n,
      stream);
  CUDA_RT_CALL(cudaEventRecord(stop, stream));

  CUDA_RT_CALL(cudaStreamSynchronize(stream));
  CUDA_RT_CALL(cudaMemcpy(
      &numOut, numOutDevice, sizeof(numOut), cudaMemcpyDeviceToHost));

  float time;
  CUDA_RT_CALL(cudaEventElapsedTime(&time, start, stop));

  fromGPU(outputValuesHost, outputValues, numOut, stream);
  fromGPU(outputCountsHost, outputCounts, numOut, stream);
  CUDA_RT_CALL(cudaStreamSynchronize(stream));
  CUDA_RT_CALL(cudaStreamDestroy(stream));

  CUDA_RT_CALL(cudaFree(outputValues));
  CUDA_RT_CALL(cudaFree(outputCounts));

  // compute RLE on host
  std::vector<T> expectedValues{inputHost[0]};
  std::vector<V> expectedCounts{1};

  for (size_t i = 1; i < n; ++i) {
    if (inputHost[i] == expectedValues.back()) {
      ++expectedCounts.back();
    } else {
      expectedValues.emplace_back(inputHost[i]);
      expectedCounts.emplace_back(1);
    }
  }

  REQUIRE(expectedCounts.size() == numOut);

  // verify output
  for (size_t i = 0; i < expectedCounts.size(); ++i) {
    CHECK(expectedValues[i] == outputValuesHost[i]);
    CHECK(expectedCounts[i] == outputCountsHost[i]);
  }

  CUDA_RT_CALL(cudaFreeHost(outputValuesHost));
  CUDA_RT_CALL(cudaFreeHost(outputCountsHost));

  CUDA_RT_CALL(cudaFree(input));
  CUDA_RT_CALL(cudaFreeHost(inputHost));
}

TEST_CASE("compressDownstream_10kUniform_Test", "[small]")
{
  using T = int32_t;
  using V = uint32_t;

  size_t const n = 10000;

  T *input, *inputHost;
  size_t const numBytes = n * sizeof(*input);

  CUDA_RT_CALL(cudaMalloc((void**)&input, numBytes));

  CUDA_RT_CALL(cudaMallocHost((void**)&inputHost, n * sizeof(*inputHost)));

  float const totalGB = numBytes / (1024.0f * 1024.0f * 1024.0f);

  cudaStream_t stream;
  CUDA_RT_CALL(cudaStreamCreate(&stream));

  T last = 37;
  for (size_t i = 0; i < n; ++i) {
    inputHost[i] = last;
  }

  toGPU(input, inputHost, n, stream);

  T *outputValues, *outputValuesHost;
  V *outputCounts, *outputCountsHost;

  CUDA_RT_CALL(cudaMalloc((void**)&outputValues, sizeof(*outputValues) * n));
  CUDA_RT_CALL(cudaMalloc((void**)&outputCounts, sizeof(*outputCounts) * n));
  CUDA_RT_CALL(
      cudaMallocHost((void**)&outputValuesHost, sizeof(*outputValuesHost) * n));
  CUDA_RT_CALL(
      cudaMallocHost((void**)&outputCountsHost, sizeof(*outputCountsHost) * n));

  void* workspace;
  const size_t maxNum = 2 * n;
  const size_t workspaceSize = RunLengthEncodeGPU::requiredWorkspaceSize(
      maxNum, TypeOf<T>(), TypeOf<V>());
  CUDA_RT_CALL(cudaMalloc((void**)&workspace, workspaceSize));

  // create on device inputs
  size_t* numInDevice;
  size_t* numOutDevice;
  T** outputValuesPtr;
  V** outputCountsPtr;
  CUDA_RT_CALL(cudaMalloc((void**)&numInDevice, sizeof(*numInDevice)));
  CUDA_RT_CALL(cudaMalloc((void**)&numOutDevice, sizeof(*numOutDevice)));
  CUDA_RT_CALL(cudaMalloc((void**)&outputValuesPtr, sizeof(*outputValuesPtr)));
  CUDA_RT_CALL(cudaMalloc((void**)&outputCountsPtr, sizeof(*outputCountsPtr)));

  CUDA_RT_CALL(cudaMemcpy(
      numInDevice, &n, sizeof(*numInDevice), cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(
      outputValuesPtr,
      &outputValues,
      sizeof(outputValues),
      cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(
      outputCountsPtr,
      &outputCounts,
      sizeof(outputCounts),
      cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;

  CUDA_RT_CALL(cudaEventCreate(&start));
  CUDA_RT_CALL(cudaEventCreate(&stop));
  CUDA_RT_CALL(cudaEventRecord(start, stream));

  RunLengthEncodeGPU::compressDownstream(
      workspace,
      workspaceSize,
      TypeOf<T>(),
      reinterpret_cast<void**>(outputValuesPtr),
      TypeOf<V>(),
      reinterpret_cast<void**>(outputCountsPtr),
      numOutDevice,
      input,
      numInDevice,
      maxNum,
      stream);
  CUDA_RT_CALL(cudaEventRecord(stop, stream));

  CUDA_RT_CALL(cudaStreamSynchronize(stream));
  float time;
  CUDA_RT_CALL(cudaEventElapsedTime(&time, start, stop));

  size_t numOut;
  CUDA_RT_CALL(cudaMemcpy(
      &numOut, numOutDevice, sizeof(numOut), cudaMemcpyDeviceToHost));

  fromGPU(outputValuesHost, outputValues, numOut, stream);
  fromGPU(outputCountsHost, outputCounts, numOut, stream);
  CUDA_RT_CALL(cudaStreamSynchronize(stream));
  CUDA_RT_CALL(cudaStreamDestroy(stream));

  CUDA_RT_CALL(cudaFree(outputValues));
  CUDA_RT_CALL(cudaFree(outputCounts));
  CUDA_RT_CALL(cudaFree(outputValuesPtr));
  CUDA_RT_CALL(cudaFree(outputCountsPtr));
  CUDA_RT_CALL(cudaFree(numOutDevice));
  CUDA_RT_CALL(cudaFree(numInDevice));

  // compute RLE on host
  std::vector<T> expectedValues{inputHost[0]};
  std::vector<V> expectedCounts{1};

  for (size_t i = 1; i < n; ++i) {
    if (inputHost[i] == expectedValues.back()) {
      ++expectedCounts.back();
    } else {
      expectedValues.emplace_back(inputHost[i]);
      expectedCounts.emplace_back(1);
    }
  }

  REQUIRE(expectedCounts.size() == numOut);

  // verify output
  for (size_t i = 0; i < expectedCounts.size(); ++i) {
    if (!(expectedValues[i] == outputValuesHost[i]
          && expectedCounts[i] == outputCountsHost[i])) {
      std::cerr << "i = " << i << " exp " << expectedCounts[i] << ":"
                << expectedValues[i] << " act " << outputCountsHost[i] << ":"
                << outputValuesHost[i] << std::endl;
    }

    CHECK(expectedValues[i] == outputValuesHost[i]);
    CHECK(expectedCounts[i] == outputCountsHost[i]);
  }

  CUDA_RT_CALL(cudaFreeHost(outputValuesHost));
  CUDA_RT_CALL(cudaFreeHost(outputCountsHost));

  CUDA_RT_CALL(cudaFree(input));
  CUDA_RT_CALL(cudaFreeHost(inputHost));
}

TEST_CASE("compressDownstream_10k_16bit_count_Test", "[small]")
{
  const size_t n = 10003;

  compressAsyncTestRandom<uint8_t, uint16_t>(n);
  compressAsyncTestRandom<int8_t, uint16_t>(n);
  compressAsyncTestRandom<uint16_t, uint16_t>(n);
  compressAsyncTestRandom<int16_t, uint16_t>(n);
  compressAsyncTestRandom<int32_t, uint16_t>(n);
  compressAsyncTestRandom<uint32_t, uint16_t>(n);
  compressAsyncTestRandom<int64_t, uint16_t>(n);
  compressAsyncTestRandom<uint64_t, uint16_t>(n);
}

TEST_CASE("compressDownstream_10k_64bit_count_Test", "[small]")
{
  const size_t n = 10003;

  compressAsyncTestRandom<uint8_t, uint64_t>(n);
  compressAsyncTestRandom<int8_t, uint64_t>(n);
  compressAsyncTestRandom<uint16_t, uint64_t>(n);
  compressAsyncTestRandom<int16_t, uint64_t>(n);
  compressAsyncTestRandom<int32_t, uint64_t>(n);
  compressAsyncTestRandom<uint32_t, uint64_t>(n);
  compressAsyncTestRandom<int64_t, uint64_t>(n);
  compressAsyncTestRandom<uint64_t, uint64_t>(n);
}

TEST_CASE("compressDownstream_1024_32bit_count_Test", "[small]")
{
  for (size_t n = 512; n < 2048; ++n) {
    compressAsyncTestRandom<int32_t, uint16_t>(n);
  }
}

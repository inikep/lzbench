/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "cuda_runtime.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// NOTE: this is for testing the C API, and thus must only contain features of
// C, not C++.

#define REQUIRE(a)                                                             \
  do {                                                                         \
    if (!(a)) {                                                                \
      printf("Check " #a " at %d failed.\n", __LINE__);                        \
      return FAIL_TEST;                                                        \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      printf(                                                                  \
          "API call failure \"" #func "\" with %d at " __FILE__ ":%d\n",       \
          (int)rt,                                                             \
          __LINE__);                                                           \
      return FAIL_TEST;                                                        \
    }                                                                          \
  } while (0)

// There's a lot of redundancy in these macros, but if there's a mismatch, it
// will show up at compile time, and things should only need to change if our
// interface changes, which should be very infrequent.
#define GENERATE_TESTS(NAME)                                                   \
  nvcompStatus_t compressGetTempSize(                                          \
      const size_t batch_size,                                                 \
      const size_t max_uncompressed_chunk_bytes,                               \
      size_t* const temp_bytes)                                                \
  {                                                                            \
    return nvcompBatched##NAME##CompressGetTempSize(                           \
        batch_size,                                                            \
        max_uncompressed_chunk_bytes,                                          \
        nvcompBatched##NAME##DefaultOpts,                                      \
        temp_bytes);                                                           \
  }                                                                            \
  nvcompStatus_t compressGetMaxOutputChunkSize(                                \
      const size_t max_uncompressed_chunk_bytes,                               \
      size_t* const max_compressed_bytes)                                      \
  {                                                                            \
    return nvcompBatched##NAME##CompressGetMaxOutputChunkSize(                 \
        max_uncompressed_chunk_bytes,                                          \
        nvcompBatched##NAME##DefaultOpts,                                      \
        max_compressed_bytes);                                                 \
  }                                                                            \
  nvcompStatus_t compressAsync(                                                \
      const void* const* const device_in_ptr,                                  \
      const size_t* const device_in_bytes,                                     \
      const size_t max_uncompressed_chunk_bytes,                               \
      const size_t batch_size,                                                 \
      void* const device_device_temp_ptr,                                      \
      const size_t temp_bytes,                                                 \
      void* const* device_out_ptr,                                             \
      size_t* const device_out_bytes,                                          \
      cudaStream_t stream)                                                     \
  {                                                                            \
    return nvcompBatched##NAME##CompressAsync(                                 \
        device_in_ptr,                                                         \
        device_in_bytes,                                                       \
        max_uncompressed_chunk_bytes,                                          \
        batch_size,                                                            \
        device_device_temp_ptr,                                                \
        temp_bytes,                                                            \
        device_out_ptr,                                                        \
        device_out_bytes,                                                      \
        nvcompBatched##NAME##DefaultOpts,                                      \
        stream);                                                               \
  }                                                                            \
  nvcompStatus_t decompressGetSizeAsync(                                       \
      const void* const* const device_compressed_ptrs,                         \
      const size_t* const device_compressed_bytes,                             \
      size_t* const device_uncompressed_bytes,                                 \
      const size_t batch_size,                                                 \
      cudaStream_t stream)                                                     \
  {                                                                            \
    return nvcompBatched##NAME##GetDecompressSizeAsync(                        \
        device_compressed_ptrs,                                                \
        device_compressed_bytes,                                               \
        device_uncompressed_bytes,                                             \
        batch_size,                                                            \
        stream);                                                               \
  }                                                                            \
  nvcompStatus_t decompressGetTempSize(                                        \
      const size_t num_chunks,                                                 \
      const size_t max_uncompressed_chunk_bytes,                               \
      size_t* const temp_bytes)                                                \
  {                                                                            \
    return nvcompBatched##NAME##DecompressGetTempSize(                         \
        num_chunks, max_uncompressed_chunk_bytes, temp_bytes);                 \
  }                                                                            \
  nvcompStatus_t decompressAsync(                                              \
      const void* const* device_compressed_ptrs,                               \
      const size_t* device_compressed_bytes,                                   \
      const size_t* device_uncompressed_bytes,                                 \
      size_t* device_actual_uncompressed_bytes,                                \
      size_t batch_size,                                                       \
      void* const device_temp_ptr,                                             \
      size_t temp_bytes,                                                       \
      void* const* device_uncompressed_ptrs,                                   \
      nvcompStatus_t* device_status_ptr,                                       \
      cudaStream_t stream)                                                     \
  {                                                                            \
    return nvcompBatched##NAME##DecompressAsync(                               \
        device_compressed_ptrs,                                                \
        device_compressed_bytes,                                               \
        device_uncompressed_bytes,                                             \
        device_actual_uncompressed_bytes,                                      \
        batch_size,                                                            \
        device_temp_ptr,                                                       \
        temp_bytes,                                                            \
        device_uncompressed_ptrs,                                              \
        device_status_ptr,                                                     \
        stream);                                                               \
  }                                                                            \
  typedef int __nvcomp_semicolon_catch

// Declear the test function wrappers
nvcompStatus_t compressGetTempSize(
    const size_t batch_size,
    const size_t max_uncompressed_chunk_bytes,
    size_t* const temp_bytes);

nvcompStatus_t compressGetMaxOutputChunkSize(
    const size_t max_uncompressed_chunk_bytes,
    size_t* const max_compressed_bytes);

nvcompStatus_t compressAsync(
    const void* const* device_in_ptr,
    const size_t* device_in_bytes,
    size_t max_uncompressed_chunk_bytes,
    size_t batch_size,
    void* device_device_temp_ptr,
    size_t temp_bytes,
    void* const* device_out_ptr,
    size_t* device_out_bytes,
    cudaStream_t stream);

nvcompStatus_t decompressGetSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream);

nvcompStatus_t decompressGetTempSize(
    const size_t num_chunks,
    const size_t max_uncompressed_chunk_bytes,
    size_t* const temp_bytes);

nvcompStatus_t decompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    nvcompStatus_t* device_status_ptrs,
    cudaStream_t stream);

static const int PASS_TEST = 1;
static const int FAIL_TEST = 0;

int test_generic_batch_compression_and_decompression(
    const size_t batch_size, const size_t min_size, 
    const size_t max_size, const int support_nullptr)
{
  typedef int T;

  // set a constant seed
  srand(0);

  // prepare input and output on host
  size_t* host_batch_sizes = malloc(batch_size * sizeof(size_t));
  for (size_t i = 0; i < batch_size; ++i) {
    if (max_size > min_size) {
      host_batch_sizes[i] = (rand() % (max_size - min_size)) + min_size;
    } else if (max_size == min_size) {
      host_batch_sizes[i] = max_size;
    } else {
      printf("Invalid max_size (%zu) / min_size (%zu)\n", max_size, min_size);
      return FAIL_TEST;
    }
  }

  size_t* host_batch_bytes = malloc(batch_size * sizeof(size_t));
  size_t max_chunk_size = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    host_batch_bytes[i] = sizeof(T) * host_batch_sizes[i];
    if (host_batch_bytes[i] > max_chunk_size) {
      max_chunk_size = host_batch_bytes[i];
    }
  }

  T** host_input = malloc(sizeof(T*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    host_input[i] = malloc(sizeof(T) * host_batch_sizes[i]);
    for (size_t j = 0; j < host_batch_sizes[i]; ++j) {
      // make sure there should be some repeats to compress
      host_input[i][j] = (rand() % 4) + 300;
    }
  }
  free(host_batch_sizes);

  T** host_output = malloc(sizeof(T*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    host_output[i] = malloc(host_batch_bytes[i]);
  }

  // prepare gpu buffers
  void** host_in_ptrs = malloc(sizeof(void*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&host_in_ptrs[i], host_batch_bytes[i]));
    CUDA_CHECK(cudaMemcpy(
        host_in_ptrs[i],
        host_input[i],
        host_batch_bytes[i],
        cudaMemcpyHostToDevice));
  }
  void** device_in_pointers;
  CUDA_CHECK(cudaMalloc(
      (void**)&device_in_pointers, sizeof(*device_in_pointers) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      device_in_pointers,
      host_in_ptrs,
      sizeof(*device_in_pointers) * batch_size,
      cudaMemcpyHostToDevice));

  size_t* device_batch_bytes;
  CUDA_CHECK(cudaMalloc(
      (void**)&device_batch_bytes, sizeof(*device_batch_bytes) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      device_batch_bytes,
      host_batch_bytes,
      sizeof(*device_batch_bytes) * batch_size,
      cudaMemcpyHostToDevice));

  nvcompStatus_t status;

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = compressGetTempSize(batch_size, max_chunk_size, &comp_temp_bytes);
  if (max_chunk_size > 1<<16) printf("max_chunk_size = %zu\n", max_chunk_size);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t max_comp_out_bytes;
  status = compressGetMaxOutputChunkSize(max_chunk_size, &max_comp_out_bytes);
  REQUIRE(status == nvcompSuccess);

  void** host_comp_out = malloc(sizeof(void*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&host_comp_out[i], max_comp_out_bytes));
  }
  void** device_comp_out;
  CUDA_CHECK(cudaMalloc(
      (void**)&device_comp_out, sizeof(*device_comp_out) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      device_comp_out,
      host_comp_out,
      sizeof(*device_comp_out) * batch_size,
      cudaMemcpyHostToDevice));

  size_t* device_comp_out_bytes;
  CUDA_CHECK(cudaMalloc(
      (void**)&device_comp_out_bytes,
      sizeof(*device_comp_out_bytes) * batch_size));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  status = compressAsync(
      (const void* const*)device_in_pointers,
      device_batch_bytes,
      max_chunk_size,
      batch_size,
      d_comp_temp,
      comp_temp_bytes,
      device_comp_out,
      device_comp_out_bytes,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(d_comp_temp));
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaFree(host_in_ptrs[i]));
  }
  cudaFree(device_in_pointers);
  free(host_in_ptrs);

  size_t temp_bytes;
  status = decompressGetTempSize(batch_size, max_chunk_size, &temp_bytes);

  void* device_temp_ptr;
  CUDA_CHECK(cudaMalloc(&device_temp_ptr, temp_bytes));

  size_t* device_decomp_out_bytes;
  CUDA_CHECK(cudaMalloc(
      (void**)&device_decomp_out_bytes,
      sizeof(*device_decomp_out_bytes) * batch_size));

  status = decompressGetSizeAsync(
      (const void* const*)device_comp_out,
      device_comp_out_bytes,
      device_decomp_out_bytes,
      batch_size,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // copy the output sizes down and check them
  size_t* host_decomp_bytes = malloc(sizeof(size_t) * batch_size);
  CUDA_CHECK(cudaMemcpy(
      host_decomp_bytes,
      device_decomp_out_bytes,
      sizeof(*host_decomp_bytes) * batch_size,
      cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < batch_size; ++i) {
    REQUIRE(host_decomp_bytes[i] == host_batch_bytes[i]);
  }

  void** host_decomp_out = malloc(sizeof(void*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&host_decomp_out[i], host_batch_bytes[i]));
  }
  void** device_decomp_out;
  cudaMalloc(
      (void**)&device_decomp_out, sizeof(*device_decomp_out) * batch_size);
  CUDA_CHECK(cudaMemcpy(
      device_decomp_out,
      host_decomp_out,
      sizeof(*device_decomp_out) * batch_size,
      cudaMemcpyHostToDevice));

  // Test functionality with null device_statuses and device_decomp_out_bytes
  if (support_nullptr)
  {
    status = decompressAsync(
        (const void* const*)device_comp_out,
        device_comp_out_bytes,
        device_batch_bytes,
        NULL,
        batch_size,
        device_temp_ptr,
        temp_bytes,
        (void* const*)device_decomp_out,
        NULL,
        stream);
    REQUIRE(status == nvcompSuccess);
    
    // Verify correctness
    for (size_t i = 0; i < batch_size; i++) {
      CUDA_CHECK(cudaMemcpy(
          host_output[i],
          host_decomp_out[i],
          host_batch_bytes[i],
          cudaMemcpyDeviceToHost));
      for (size_t j = 0; j < host_batch_bytes[i] / sizeof(T); ++j) {
        REQUIRE(host_output[i][j] == host_input[i][j]);
      }
    }
  }

  nvcompStatus_t* device_statuses;
  CUDA_CHECK(cudaMalloc(
      (void**)&device_statuses, sizeof(*device_statuses) * batch_size));
  status = decompressAsync(
      (const void* const*)device_comp_out,
      device_comp_out_bytes,
      device_batch_bytes,
      device_decomp_out_bytes,
      batch_size,
      device_temp_ptr,
      temp_bytes,
      (void* const*)device_decomp_out,
      device_statuses,
      stream);
  REQUIRE(status == nvcompSuccess);

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaStreamDestroy(stream));

  // check statuses
  nvcompStatus_t* host_statuses = malloc(sizeof(*device_statuses) * batch_size);
  CUDA_CHECK(cudaMemcpy(
      host_statuses,
      device_statuses,
      sizeof(*device_statuses) * batch_size,
      cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_statuses));

  for (size_t i = 0; i < batch_size; ++i) {
    REQUIRE(host_statuses[i] == nvcompSuccess);
  }
  free(host_statuses);

  // check output bytes
  CUDA_CHECK(cudaMemcpy(
      host_decomp_bytes,
      device_decomp_out_bytes,
      sizeof(*host_decomp_bytes) * batch_size,
      cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < batch_size; ++i) {
    REQUIRE(host_decomp_bytes[i] == host_batch_bytes[i]);
  }
  free(host_decomp_bytes);
  CUDA_CHECK(cudaFree(device_decomp_out_bytes));

  CUDA_CHECK(cudaFree(device_batch_bytes));
  CUDA_CHECK(cudaFree(device_comp_out_bytes));
  CUDA_CHECK(cudaFree(device_temp_ptr));

  for (size_t i = 0; i < batch_size; i++) {
    CUDA_CHECK(cudaMemcpy(
        host_output[i],
        host_decomp_out[i],
        host_batch_bytes[i],
        cudaMemcpyDeviceToHost));
    // Verify correctness
    for (size_t j = 0; j < host_batch_bytes[i] / sizeof(T); ++j) {
      REQUIRE(host_output[i][j] == host_input[i][j]);
    }
    free(host_input[i]);
  }
  free(host_input);
  free(host_batch_bytes);

  for (size_t i = 0; i < batch_size; i++) {
    CUDA_CHECK(cudaFree(host_comp_out[i]));
    CUDA_CHECK(cudaFree(host_decomp_out[i]));
    free(host_output[i]);
  }
  CUDA_CHECK(cudaFree(device_comp_out));
  free(host_output);
  free(host_comp_out);
  free(host_decomp_out);

  return PASS_TEST;
}

int test_generic_batch_decompression_errors(
    const size_t batch_size, const size_t min_size, const size_t max_size)
{
  typedef int T;

  // in this test, we try to decompress random data
  // -- first we try to get the size of it, which should report 0, or a size
  // larger than the input (see NOTE: below).
  // -- then we try to decompress it, which should report an invalid status

  // set a constant seed
  srand(0);

  // prepare input and output on host
  size_t* host_batch_sizes = malloc(batch_size * sizeof(size_t));
  for (size_t i = 0; i < batch_size; ++i) {
    if (max_size > min_size) {
      host_batch_sizes[i] = (rand() % (max_size - min_size)) + min_size;
    } else if (max_size == min_size) {
      host_batch_sizes[i] = max_size;
    } else {
      printf("Invalid max_size (%zu) / min_size (%zu)\n", max_size, min_size);
      return FAIL_TEST;
    }
  }

  size_t* host_batch_bytes = malloc(batch_size * sizeof(size_t));
  size_t max_chunk_size = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    host_batch_bytes[i] = sizeof(T) * host_batch_sizes[i];
    if (host_batch_bytes[i] > max_chunk_size) {
      max_chunk_size = host_batch_bytes[i];
    }
  }

  T** host_input = malloc(sizeof(T*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    host_input[i] = malloc(sizeof(T) * host_batch_sizes[i]);
    for (size_t j = 0; j < host_batch_sizes[i]; ++j) {
      // make sure there should be some repeats to compress
      host_input[i][j] = (rand() % 4) + 300;
    }
  }
  free(host_batch_sizes);

  T** host_output = malloc(sizeof(T*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    host_output[i] = malloc(host_batch_bytes[i]);
  }

  // prepare gpu buffers
  void** host_in_ptrs = malloc(sizeof(void*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&host_in_ptrs[i], host_batch_bytes[i]));
    CUDA_CHECK(cudaMemcpy(
        host_in_ptrs[i],
        host_input[i],
        host_batch_bytes[i],
        cudaMemcpyHostToDevice));
  }
  void** device_in_pointers;
  CUDA_CHECK(cudaMalloc(
      (void**)&device_in_pointers, sizeof(*device_in_pointers) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      device_in_pointers,
      host_in_ptrs,
      sizeof(*device_in_pointers) * batch_size,
      cudaMemcpyHostToDevice));

  size_t* device_batch_bytes;
  CUDA_CHECK(cudaMalloc(
      (void**)&device_batch_bytes, sizeof(*device_batch_bytes) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      device_batch_bytes,
      host_batch_bytes,
      sizeof(*device_batch_bytes) * batch_size,
      cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  nvcompStatus_t status;

  // attempt to get the size
  size_t* device_decomp_out_bytes;
  CUDA_CHECK(cudaMalloc(
      (void**)&device_decomp_out_bytes,
      sizeof(*device_decomp_out_bytes) * batch_size));
  // initially set all sizes to -1
  CUDA_CHECK(cudaMemset(
      device_decomp_out_bytes,
      -1,
      sizeof(*device_decomp_out_bytes) * batch_size));

  status = decompressGetSizeAsync(
      (const void* const*)device_in_pointers,
      device_batch_bytes,
      device_decomp_out_bytes,
      batch_size,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // copy the output sizes down and check them
  size_t* host_decomp_bytes = malloc(sizeof(size_t) * batch_size);
  CUDA_CHECK(cudaMemcpy(
      host_decomp_bytes,
      device_decomp_out_bytes,
      sizeof(*host_decomp_bytes) * batch_size,
      cudaMemcpyDeviceToHost));

  // We can't gaurantee that decompressor fails to get a size from the data,
  // so here we can only check that the size has been written to.
  for (size_t i = 0; i < batch_size; ++i) {
    REQUIRE(host_decomp_bytes[i] != (size_t)-1);
  }

  // next set the output buffers to be invalid for the returned sizes
  for (size_t i = 0; i < batch_size; ++i) {
    if (host_decomp_bytes[i] == 0
        || host_decomp_bytes[i] > host_batch_bytes[i]) {
      // We either discovered and invalid chunk when getting the size, or the
      // decompress things will it be decompressed to something larger than
      // `host_batch_bytes[i]`, either way specifying `host_batch_bytes[i]`
      // as the output space should cause it to fail.
      host_decomp_bytes[i] = host_batch_bytes[i];
    } else {
      // We're in danger of this noise successfully decompressing, so we
      // intentionally give it a smaller buffer than it requires.
      host_decomp_bytes[i] = host_decomp_bytes[i] - 1;
    }
  }
  CUDA_CHECK(cudaMemcpy(
      device_decomp_out_bytes,
      host_decomp_bytes,
      sizeof(*device_decomp_out_bytes) * batch_size,
      cudaMemcpyHostToDevice));

  // attempt to decompress
  size_t temp_bytes;
  status = decompressGetTempSize(batch_size, max_chunk_size, &temp_bytes);

  void* device_temp_ptr;
  CUDA_CHECK(cudaMalloc(&device_temp_ptr, temp_bytes));

  void** host_decomp_out = malloc(sizeof(void*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&host_decomp_out[i], host_decomp_bytes[i]));
  }
  void** device_decomp_out;
  cudaMalloc(
      (void**)&device_decomp_out, sizeof(*device_decomp_out) * batch_size);
  CUDA_CHECK(cudaMemcpy(
      device_decomp_out,
      host_decomp_out,
      sizeof(*device_decomp_out) * batch_size,
      cudaMemcpyHostToDevice));

  nvcompStatus_t* device_statuses;
  CUDA_CHECK(cudaMalloc(
      (void**)&device_statuses, sizeof(*device_statuses) * batch_size));
  status = decompressAsync(
      (const void* const*)device_in_pointers,
      device_batch_bytes,
      device_decomp_out_bytes,
      device_decomp_out_bytes,
      batch_size,
      device_temp_ptr,
      temp_bytes,
      (void* const*)device_decomp_out,
      device_statuses,
      stream);
  REQUIRE(status == nvcompSuccess);

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaStreamDestroy(stream));

  // clean up inputs
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaFree(host_in_ptrs[i]));
  }
  cudaFree(device_in_pointers);
  free(host_in_ptrs);

  // check statuses
  nvcompStatus_t* host_statuses = malloc(sizeof(*device_statuses) * batch_size);
  CUDA_CHECK(cudaMemcpy(
      host_statuses,
      device_statuses,
      sizeof(*device_statuses) * batch_size,
      cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_statuses));

  for (size_t i = 0; i < batch_size; ++i) {
    if (host_statuses[i] != nvcompErrorCannotDecompress) {
    }
    REQUIRE(host_statuses[i] == nvcompErrorCannotDecompress);
  }
  free(host_statuses);
  CUDA_CHECK(cudaFree(device_decomp_out_bytes));

  CUDA_CHECK(cudaFree(device_batch_bytes));
  CUDA_CHECK(cudaFree(device_temp_ptr));

  for (size_t i = 0; i < batch_size; i++) {
    free(host_input[i]);
  }
  free(host_input);
  free(host_batch_bytes);

  for (size_t i = 0; i < batch_size; i++) {
    CUDA_CHECK(cudaFree(host_decomp_out[i]));
    free(host_output[i]);
  }
  free(host_output);
  free(host_decomp_out);

  return PASS_TEST;
}

#define TEST(bs, min, max, num_tests, rv, crash_safe, support_nullptr)         \
  do {                                                                         \
    ++(num_tests);                                                             \
    if (!test_generic_batch_compression_and_decompression(bs, min, max, support_nullptr)) {     \
      printf(                                                                  \
          "compression and decompression test failed %dx[%d:%d]\n",            \
          (int)(bs),                                                           \
          (int)(min),                                                          \
          (int)(max));                                                         \
      ++(rv);                                                                  \
    }                                                                          \
    if (crash_safe) {                                                          \
      if (!test_generic_batch_decompression_errors(bs, min, max)) {            \
        printf(                                                                \
            "decompression errors test failed %dx[%d:%d]\n",                   \
            (int)(bs),                                                         \
            (int)(min),                                                        \
            (int)(max));                                                       \
        ++(rv);                                                                \
      }                                                                        \
    }                                                                          \
  } while (0)

int main(int argc, char** argv)
{
  if (argc != 1) {
    printf("ERROR: %s accepts no arguments.\n", argv[0]);
    return 1;
  }

  int num_tests = 0;
  int num_failed_tests = 0;

#ifdef CRASH_SAFE
  const int crash_safe = 1;
#else
  const int crash_safe = 0;
#endif

#ifdef SUPPORT_NULLPTR_APIS
  const int support_nullptr = 1;
#else 
  const int support_nullptr = 0;
#endif

  // these macros count the number of failed tests
  TEST(1, 100, 100, num_tests, num_failed_tests, crash_safe, support_nullptr);
  TEST(1, (1<<16) / sizeof(int), (1<<16) / sizeof(int), num_tests, num_failed_tests, crash_safe, support_nullptr);
  TEST(11, 1000, 10000, num_tests, num_failed_tests, crash_safe, support_nullptr);
  TEST(127, 10000, (1<<16) / sizeof(int), num_tests, num_failed_tests, crash_safe, support_nullptr);
  TEST(1025, 100, (1<<16) / sizeof(int), num_tests, num_failed_tests, crash_safe, support_nullptr);
  TEST(10025, 100, 1000, num_tests, num_failed_tests, crash_safe, support_nullptr);

  if (num_failed_tests == 0) {
    printf(
        "SUCCESS: All tests passed: %d/%d\n",
        (num_tests - num_failed_tests),
        num_tests);
  } else {
    printf("FAILURE: %d/%d tests failed\n", num_failed_tests, num_tests);
  }

  // rely on exit code of 0 being a success, and anything else being a failure
  return num_failed_tests;
}

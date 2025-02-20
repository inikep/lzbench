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

#define CATCH_CONFIG_MAIN

#include "../src/common.h"
#include "catch.hpp"
#include "nvcomp.hpp"
#include "nvcomp/cascaded.h"
#include <cuda_runtime.h>

#include <cstdint>
#include <random>
#include <vector>

using run_type = uint16_t;
using nvcomp::roundUpToAlignment;

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = cond;                                                    \
    REQUIRE(err == cudaSuccess);                                               \
  } while (false)

template <typename data_type>
std::vector<data_type> generate_predefined_input_host(
    std::vector<data_type> values, std::vector<size_t> repititions)
{
  std::vector<data_type> generated_input;
  for (size_t val_idx = 0; val_idx < values.size(); val_idx++) {
    // Add `repititions[val_idx]` copies of `values[val_idx]`
    for (size_t repitition_idx = 0; repitition_idx < repititions[val_idx];
         repitition_idx++) {
      generated_input.push_back(values[val_idx]);
    }
  }
  return generated_input;
}

size_t max_compressed_size(size_t uncompressed_size)
{
  return (uncompressed_size + 3) / 4 * 4 + 4;
}

template <typename data_type>
void verify_compression_output(
    const void* compressed_data,
    const size_t compressed_bytes,
    const std::vector<run_type>& runs0,
    const std::vector<run_type>& runs1,
    const std::vector<data_type>& output,
    const data_type delta_value)
{
  // Copy the compressed buffer to the host memory
  std::vector<uint32_t> compressed_data_host(compressed_bytes / 4);
  CUDA_CHECK(cudaMemcpy(
      compressed_data_host.data(),
      compressed_data,
      compressed_bytes,
      cudaMemcpyDeviceToHost));

  // Check the partition header stores 2 RLE layers, 1 Delta layer, no
  // bitpacking, and the input datatype
  REQUIRE(
      compressed_data_host[0]
      == 2 + (1 << 8) + (0 << 16)
             + (static_cast<uint32_t>(nvcomp::TypeOf<data_type>()) << 24));

  // Calculate the location of the first chunk and test array offsets
  uint32_t* chunk_start_ptr = reinterpret_cast<uint32_t*>(
      roundUpToAlignment<data_type>(compressed_data_host.data() + 2));
  REQUIRE(chunk_start_ptr[1] == runs0.size() * sizeof(run_type));
  REQUIRE(chunk_start_ptr[2] == runs1.size() * sizeof(run_type));
  REQUIRE(chunk_start_ptr[3] == runs1.size() * sizeof(data_type));

  // Check the first element of the delta layer in the header
  data_type* delta_metadata_ptr
      = roundUpToAlignment<data_type>(chunk_start_ptr + 4);
  REQUIRE(*delta_metadata_ptr == delta_value);

  // Check run array of the first RLE layer
  run_type* runs_ptr = reinterpret_cast<run_type*>(
      roundUpToAlignment<uint32_t>(delta_metadata_ptr + 1));
  for (auto& run : runs0) {
    REQUIRE(run == *runs_ptr);
    runs_ptr++;
  }

  // Check run array of the second RLE layer
  runs_ptr
      = reinterpret_cast<run_type*>(roundUpToAlignment<uint32_t>(runs_ptr));
  for (auto& run : runs1) {
    REQUIRE(run == *runs_ptr);
    runs_ptr++;
  }

  // Check the data array after the final layer
  data_type* final_array
      = roundUpToAlignment<data_type>(roundUpToAlignment<uint32_t>(runs_ptr));
  for (auto& value : output) {
    REQUIRE(value == *final_array);
    final_array++;
  }
}

/**
 * Verify the number of decompressed bytes match the number of the uncompressed
 * bytes.
 */
void verify_decompressed_sizes(
    size_t batch_size,
    const size_t* decompressed_bytes_device,
    const std::vector<size_t>& uncompressed_bytes_host)
{
  std::vector<size_t> decompressed_bytes_host(batch_size);
  CUDA_CHECK(cudaMemcpy(
      decompressed_bytes_host.data(),
      decompressed_bytes_device,
      sizeof(size_t) * batch_size,
      cudaMemcpyDeviceToHost));

  for (size_t partition_idx = 0; partition_idx < batch_size; partition_idx++) {
    REQUIRE(
        decompressed_bytes_host[partition_idx]
        == uncompressed_bytes_host[partition_idx]);
  }
}

/**
 * Verify decompression outputs match the original uncompressed data.
 */
template <typename data_type>
void verify_decompressed_output(
    size_t batch_size,
    const std::vector<void*>& decompressed_ptrs_host,
    const std::vector<const data_type*>& uncompressed_data_host,
    const std::vector<size_t>& uncompressed_bytes_host)
{

  for (size_t partition_idx = 0; partition_idx < batch_size; partition_idx++) {
    const size_t num_elements
        = uncompressed_bytes_host[partition_idx] / sizeof(data_type);

    std::vector<data_type> decompressed_data_host(num_elements);
    CUDA_CHECK(cudaMemcpy(
        decompressed_data_host.data(),
        decompressed_ptrs_host[partition_idx],
        uncompressed_bytes_host[partition_idx],
        cudaMemcpyDeviceToHost));

    for (size_t element_idx = 0; element_idx < num_elements; element_idx++) {
      REQUIRE(
          decompressed_data_host[element_idx]
          == uncompressed_data_host[partition_idx][element_idx]);
    }
  }
}

/*
 * This test case tests the correctness of batched cascaded compressor and
 * decompressor on predefined data. The test case uses 2 RLE layers, 1 Delta
 * layer, and optionally bitpacking depending on the `use_bp` argument. It first
 * compresses the data and verifies the compressed buffers. Then it decompresses
 * the data and compares against the original values.
 */
template <typename data_type>
void test_predefined_cases(int use_bp)
{
  // Generate input data and copy it to device memory

  std::vector<data_type> input0_host = generate_predefined_input_host(
      std::vector<data_type>{3, 9, 4, 0, 1},
      std::vector<size_t>{1, 20, 13, 25, 6});

  std::vector<data_type> input1_host = generate_predefined_input_host(
      std::vector<data_type>{1, 2, 3, 4, 5, 6},
      std::vector<size_t>{10, 6, 15, 1, 13, 9});

  void* input0_device;
  CUDA_CHECK(
      cudaMalloc(&input0_device, input0_host.size() * sizeof(data_type)));
  CUDA_CHECK(cudaMemcpy(
      input0_device,
      input0_host.data(),
      input0_host.size() * sizeof(data_type),
      cudaMemcpyHostToDevice));

  void* input1_device;
  CUDA_CHECK(
      cudaMalloc(&input1_device, input1_host.size() * sizeof(data_type)));
  CUDA_CHECK(cudaMemcpy(
      input1_device,
      input1_host.data(),
      input1_host.size() * sizeof(data_type),
      cudaMemcpyHostToDevice));

  // Copy uncompressed pointers and sizes to device memory

  std::vector<void*> uncompressed_ptrs_host
      = {input0_device, input1_device, input0_device};
  std::vector<size_t> uncompressed_bytes_host
      = {input0_host.size() * sizeof(data_type),
         input1_host.size() * sizeof(data_type),
         input0_host.size() * sizeof(data_type)};
  const size_t batch_size = uncompressed_ptrs_host.size();

  void** uncompressed_ptrs_device;
  CUDA_CHECK(cudaMalloc(&uncompressed_ptrs_device, sizeof(void*) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      uncompressed_ptrs_device,
      uncompressed_ptrs_host.data(),
      sizeof(void*) * batch_size,
      cudaMemcpyHostToDevice));

  size_t* uncompressed_bytes_device;
  CUDA_CHECK(
      cudaMalloc(&uncompressed_bytes_device, sizeof(size_t) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      uncompressed_bytes_device,
      uncompressed_bytes_host.data(),
      sizeof(size_t) * batch_size,
      cudaMemcpyHostToDevice));

  // Allocate compressed buffers and sizes

  std::vector<void*> compressed_ptrs_host;
  for (size_t partition_idx = 0; partition_idx < batch_size; partition_idx++) {
    void* compressed_ptr;
    CUDA_CHECK(cudaMalloc(
        &compressed_ptr,
        max_compressed_size(uncompressed_bytes_host[partition_idx])));
    compressed_ptrs_host.push_back(compressed_ptr);
  }

  void** compressed_ptrs_device;
  CUDA_CHECK(cudaMalloc(&compressed_ptrs_device, sizeof(void*) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      compressed_ptrs_device,
      compressed_ptrs_host.data(),
      sizeof(void*) * batch_size,
      cudaMemcpyHostToDevice));

  size_t* compressed_bytes_device;
  CUDA_CHECK(cudaMalloc(&compressed_bytes_device, sizeof(size_t) * batch_size));

  // Launch batched compression

  nvcompBatchedCascadedOpts_t comp_opts
      = {batch_size, nvcomp::TypeOf<data_type>(), 2, 1, use_bp};

  auto status = nvcompBatchedCascadedCompressAsync(
      uncompressed_ptrs_device,
      uncompressed_bytes_device,
      0, // not used
      batch_size,
      nullptr, // not used
      0,       // not used
      compressed_ptrs_device,
      compressed_bytes_device,
      comp_opts,
      0);

  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(0));

  // Verify compressed bytes alignment

  std::vector<size_t> compressed_bytes_host(batch_size);
  CUDA_CHECK(cudaMemcpy(
      compressed_bytes_host.data(),
      compressed_bytes_device,
      sizeof(size_t) * batch_size,
      cudaMemcpyDeviceToHost));

  for (auto const& compressed_bytes_partition : compressed_bytes_host) {
    REQUIRE(compressed_bytes_partition % 4 == 0);
    REQUIRE(compressed_bytes_partition % sizeof(data_type) == 0);
  }

  // Check the test case is small enough to fit inside one batch
  constexpr size_t chunk_size = 4096;
  for (auto const& uncompressed_byte : uncompressed_bytes_host) {
    REQUIRE(uncompressed_byte <= chunk_size);
  }

  // Check compression output

  if (!use_bp) {
    // Verify partition0
    {
      std::vector<run_type> runs0 = {1, 20, 13, 25, 6};
      std::vector<run_type> runs1 = {1, 1, 1, 1};
      std::vector<data_type> output = {6, (data_type)-5, (data_type)-4, 1};
      data_type delta_value = 3;
      verify_compression_output(
          compressed_ptrs_host[0],
          compressed_bytes_host[0],
          runs0,
          runs1,
          output,
          delta_value);
    }

    // Verify partition1
    {
      std::vector<run_type> runs0 = {10, 6, 15, 1, 13, 9};
      std::vector<run_type> runs1 = {5};
      std::vector<data_type> output = {1};
      data_type delta_value = 1;
      verify_compression_output(
          compressed_ptrs_host[1],
          compressed_bytes_host[1],
          runs0,
          runs1,
          output,
          delta_value);
    }

    // Verify partition2
    {
      std::vector<run_type> runs0 = {1, 20, 13, 25, 6};
      std::vector<run_type> runs1 = {1, 1, 1, 1};
      std::vector<data_type> output = {6, (data_type)-5, (data_type)-4, 1};
      data_type delta_value = 3;
      verify_compression_output(
          compressed_ptrs_host[2],
          compressed_bytes_host[2],
          runs0,
          runs1,
          output,
          delta_value);
    }
  }

  // Check uncompressed bytes stored in the compressed buffer

  size_t* decompressed_bytes_device;
  CUDA_CHECK(
      cudaMalloc(&decompressed_bytes_device, sizeof(size_t) * batch_size));

  status = nvcompBatchedCascadedGetDecompressSizeAsync(
      compressed_ptrs_device,
      compressed_bytes_device,
      decompressed_bytes_device,
      batch_size,
      0);

  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(0));

  verify_decompressed_sizes(
      batch_size, decompressed_bytes_device, uncompressed_bytes_host);

  // Allocate decompressed buffers

  std::vector<void*> decompressed_ptrs_host;
  for (size_t partition_idx = 0; partition_idx < batch_size; partition_idx++) {
    void* decompressed_ptr;
    CUDA_CHECK(
        cudaMalloc(&decompressed_ptr, uncompressed_bytes_host[partition_idx]));
    decompressed_ptrs_host.push_back(decompressed_ptr);
  }

  void** decompressed_ptrs_device;
  CUDA_CHECK(cudaMalloc(&decompressed_ptrs_device, sizeof(void*) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      decompressed_ptrs_device,
      decompressed_ptrs_host.data(),
      sizeof(void*) * batch_size,
      cudaMemcpyHostToDevice));

  CUDA_CHECK(
      cudaMemset(decompressed_bytes_device, 0, sizeof(size_t) * batch_size));

  // Launch decompression

  nvcompStatus_t* compression_statuses_device;
  CUDA_CHECK(cudaMalloc(
      &compression_statuses_device, sizeof(nvcompStatus_t) * batch_size));

  status = nvcompBatchedCascadedDecompressAsync(
      compressed_ptrs_device,
      compressed_bytes_device,
      uncompressed_bytes_device,
      decompressed_bytes_device,
      batch_size,
      nullptr, // not used
      0,       // not used
      decompressed_ptrs_device,
      compression_statuses_device,
      0);

  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(0));

  std::vector<nvcompStatus_t> compression_statuses_host(batch_size);
  CUDA_CHECK(cudaMemcpy(
      compression_statuses_host.data(),
      compression_statuses_device,
      sizeof(nvcompStatus_t) * batch_size,
      cudaMemcpyDeviceToHost));

  for (auto const& compression_status : compression_statuses_host)
    REQUIRE(compression_status == nvcompSuccess);

  // Verify decompression outputs match the original uncompressed data

  std::vector<const data_type*> uncompressed_data_host
      = {input0_host.data(), input1_host.data(), input0_host.data()};

  verify_decompressed_sizes(
      batch_size, decompressed_bytes_device, uncompressed_bytes_host);

  verify_decompressed_output(
      batch_size,
      decompressed_ptrs_host,
      uncompressed_data_host,
      uncompressed_bytes_host);

  // Cleanup

  CUDA_CHECK(cudaFree(input0_device));
  CUDA_CHECK(cudaFree(input1_device));
  CUDA_CHECK(cudaFree(uncompressed_ptrs_device));
  CUDA_CHECK(cudaFree(uncompressed_bytes_device));
  for (void* const& ptr : compressed_ptrs_host)
    CUDA_CHECK(cudaFree(ptr));
  CUDA_CHECK(cudaFree(compressed_ptrs_device));
  CUDA_CHECK(cudaFree(compressed_bytes_device));
  for (void* const& ptr : decompressed_ptrs_host)
    CUDA_CHECK(cudaFree(ptr));
  CUDA_CHECK(cudaFree(decompressed_bytes_device));
  CUDA_CHECK(cudaFree(decompressed_ptrs_device));
  CUDA_CHECK(cudaFree(compression_statuses_device));
}

/*
 * This test case tests when the compressed size is larger than the uncompressed
 * size, i.e. compression ratio less than 1. In this case, we use the fallback
 * path of directly copying the uncompressed data to the compressed buffers.
 * During the test, we generate random integers as input data. Since
 * random data cannot be effectively compressed by the cascaded compressor, the
 * compression ratio should be less than 1.
 */
template <typename data_type>
void test_fallback_path()
{
  std::vector<int> uncompressed_num_elements = {10, 100, 1000, 10000, 1000};
  const size_t batch_size = uncompressed_num_elements.size();

  // Generate random integers as input data in the host memory

  std::random_device rd;
  std::mt19937 random_generator(rd());

  // int8_t and uint8_t specializations of std::uniform_int_distribution are
  // non-standard, and aren't available on MSVC, so use short instead,
  // but with the range limit of the smaller type, and then cast below.
  using safe_type =
      typename std::conditional<sizeof(data_type) == 1, short, data_type>::type;
  std::uniform_int_distribution<safe_type> dist(
      0, std::numeric_limits<data_type>::max());

  std::vector<std::vector<data_type>> inputs_data(batch_size);
  for (size_t input_idx = 0; input_idx < batch_size; input_idx++) {
    inputs_data[input_idx].resize(uncompressed_num_elements[input_idx]);
    for (int element_idx = 0;
         element_idx < uncompressed_num_elements[input_idx];
         element_idx++) {
      inputs_data[input_idx][element_idx] = static_cast<data_type>(dist(random_generator));
    }
  }

  // Copy the input data and sizes to the device memory

  std::vector<size_t> uncompressed_bytes_host;
  size_t* uncompressed_bytes_device;
  for (size_t input_idx = 0; input_idx < batch_size; input_idx++) {
    uncompressed_bytes_host.push_back(
        uncompressed_num_elements[input_idx] * sizeof(data_type));
  }
  CUDA_CHECK(
      cudaMalloc(&uncompressed_bytes_device, sizeof(size_t) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      uncompressed_bytes_device,
      uncompressed_bytes_host.data(),
      sizeof(size_t) * batch_size,
      cudaMemcpyHostToDevice));

  std::vector<void*> uncompressed_ptrs_host;
  for (size_t input_idx = 0; input_idx < batch_size; input_idx++) {
    void* allocated_buffer;
    CUDA_CHECK(cudaMalloc(
        &allocated_buffer,
        sizeof(data_type) * uncompressed_num_elements[input_idx]));
    CUDA_CHECK(cudaMemcpy(
        allocated_buffer,
        inputs_data[input_idx].data(),
        uncompressed_bytes_host[input_idx],
        cudaMemcpyHostToDevice));
    uncompressed_ptrs_host.push_back(allocated_buffer);
  }

  void** uncompressed_ptrs_device;
  CUDA_CHECK(cudaMalloc(&uncompressed_ptrs_device, sizeof(void*) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      uncompressed_ptrs_device,
      uncompressed_ptrs_host.data(),
      sizeof(void*) * batch_size,
      cudaMemcpyHostToDevice));

  // Allocate compressed buffer

  std::vector<void*> compressed_ptrs_host;
  for (size_t partition_idx = 0; partition_idx < batch_size; partition_idx++) {
    void* allocated_compressed_ptr;
    CUDA_CHECK(cudaMalloc(
        &allocated_compressed_ptr,
        max_compressed_size(uncompressed_bytes_host[partition_idx])));
    compressed_ptrs_host.push_back(allocated_compressed_ptr);
  }

  void** compressed_ptrs_device;
  CUDA_CHECK(cudaMalloc(&compressed_ptrs_device, sizeof(void*) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      compressed_ptrs_device,
      compressed_ptrs_host.data(),
      sizeof(void*) * batch_size,
      cudaMemcpyHostToDevice));

  size_t* compressed_bytes_device;
  CUDA_CHECK(cudaMalloc(&compressed_bytes_device, sizeof(size_t) * batch_size));

  // Launch batched cascaded compression

  nvcompBatchedCascadedOpts_t comp_opts
      = {batch_size, nvcomp::TypeOf<data_type>(), 2, 1, true};

  auto status = nvcompBatchedCascadedCompressAsync(
      uncompressed_ptrs_device,
      uncompressed_bytes_device,
      0, // not used
      batch_size,
      nullptr, // not used
      0,       // not used
      compressed_ptrs_device,
      compressed_bytes_device,
      comp_opts,
      0);

  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(0));

  // Check the metadata in the compressed buffers. It should indicate no
  // compression is used
  for (size_t partition_idx = 0; partition_idx < batch_size; partition_idx++) {
    uint32_t metadata;
    CUDA_CHECK(cudaMemcpy(
        &metadata,
        compressed_ptrs_host[partition_idx],
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost));
    REQUIRE(
        metadata == (static_cast<uint32_t>(nvcomp::TypeOf<data_type>()) << 24));
  }

  // Check uncompressed bytes stored in the compressed buffer

  size_t* decompressed_bytes_device;
  CUDA_CHECK(
      cudaMalloc(&decompressed_bytes_device, sizeof(size_t) * batch_size));

  status = nvcompBatchedCascadedGetDecompressSizeAsync(
      compressed_ptrs_device,
      compressed_bytes_device,
      decompressed_bytes_device,
      batch_size,
      0);

  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(0));

  verify_decompressed_sizes(
      batch_size, decompressed_bytes_device, uncompressed_bytes_host);

  // Allocate decompressed buffers and sizes

  std::vector<void*> decompressed_ptrs_host;
  for (size_t partition_idx = 0; partition_idx < batch_size; partition_idx++) {
    void* allocated_ptr;
    CUDA_CHECK(
        cudaMalloc(&allocated_ptr, uncompressed_bytes_host[partition_idx]));
    decompressed_ptrs_host.push_back(allocated_ptr);
  }

  void** decompressed_ptrs_device;
  CUDA_CHECK(cudaMalloc(&decompressed_ptrs_device, sizeof(void*) * batch_size));
  CUDA_CHECK(cudaMemcpy(
      decompressed_ptrs_device,
      decompressed_ptrs_host.data(),
      sizeof(void*) * batch_size,
      cudaMemcpyHostToDevice));

  CUDA_CHECK(
      cudaMemset(decompressed_bytes_device, 0, sizeof(size_t) * batch_size));

  // Launch decompression

  nvcompStatus_t* compression_statuses_device;
  CUDA_CHECK(cudaMalloc(
      &compression_statuses_device, sizeof(nvcompStatus_t) * batch_size));

  status = nvcompBatchedCascadedDecompressAsync(
      compressed_ptrs_device,
      compressed_bytes_device,
      uncompressed_bytes_device,
      decompressed_bytes_device,
      batch_size,
      nullptr, // not used
      0,       // not used
      decompressed_ptrs_device,
      compression_statuses_device,
      0);

  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(0));

  std::vector<nvcompStatus_t> compression_statuses_host(batch_size);
  CUDA_CHECK(cudaMemcpy(
      compression_statuses_host.data(),
      compression_statuses_device,
      sizeof(nvcompStatus_t) * batch_size,
      cudaMemcpyDeviceToHost));

  for (auto const& compression_status : compression_statuses_host)
    REQUIRE(compression_status == nvcompSuccess);

  // Verify decompression outputs match the original uncompressed data

  std::vector<const data_type*> uncompressed_data_host;
  for (auto const& input : inputs_data) {
    uncompressed_data_host.push_back(input.data());
  }

  verify_decompressed_sizes(
      batch_size, decompressed_bytes_device, uncompressed_bytes_host);

  verify_decompressed_output(
      batch_size,
      decompressed_ptrs_host,
      uncompressed_data_host,
      uncompressed_bytes_host);

  // Cleanup

  CUDA_CHECK(cudaFree(uncompressed_bytes_device));
  for (void* const& ptr : uncompressed_ptrs_host)
    CUDA_CHECK(cudaFree(ptr));
  CUDA_CHECK(cudaFree(uncompressed_ptrs_device));
  for (void* const& ptr : compressed_ptrs_host)
    CUDA_CHECK(cudaFree(ptr));
  CUDA_CHECK(cudaFree(compressed_ptrs_device));
  CUDA_CHECK(cudaFree(compressed_bytes_device));
  for (void* const& ptr : decompressed_ptrs_host)
    CUDA_CHECK(cudaFree(ptr));
  CUDA_CHECK(cudaFree(decompressed_ptrs_device));
  CUDA_CHECK(cudaFree(decompressed_bytes_device));
  CUDA_CHECK(cudaFree(compression_statuses_device));
}

template <typename data_type>
void test_out_of_bound(int use_bp)
{
  std::vector<data_type> input_host = generate_predefined_input_host(
      std::vector<data_type>{1, 2, 3, 4, 5, 6},
      std::vector<size_t>{10, 6, 15, 1, 13, 9});
  const size_t uncompressed_byte = input_host.size() * sizeof(data_type);
  const size_t batch_size = input_host.size();

  void* uncompressed_data;
  CUDA_CHECK(cudaMalloc(&uncompressed_data, uncompressed_byte));
  CUDA_CHECK(cudaMemcpy(
      uncompressed_data,
      input_host.data(),
      uncompressed_byte,
      cudaMemcpyHostToDevice));

  void** uncompressed_ptrs_device;
  CUDA_CHECK(cudaMalloc(&uncompressed_ptrs_device, sizeof(void*)));
  CUDA_CHECK(cudaMemcpy(
      uncompressed_ptrs_device,
      &uncompressed_data,
      sizeof(void*),
      cudaMemcpyHostToDevice));

  size_t* uncompressed_bytes_device;
  CUDA_CHECK(cudaMalloc(&uncompressed_bytes_device, sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(
      uncompressed_bytes_device,
      &uncompressed_byte,
      sizeof(size_t),
      cudaMemcpyHostToDevice));

  void* compressed_data;
  CUDA_CHECK(
      cudaMalloc(&compressed_data, max_compressed_size(uncompressed_byte)));

  void** compressed_ptrs_device;
  CUDA_CHECK(cudaMalloc(&compressed_ptrs_device, sizeof(void*)));
  CUDA_CHECK(cudaMemcpy(
      compressed_ptrs_device,
      &compressed_data,
      sizeof(void*),
      cudaMemcpyHostToDevice));

  size_t* compressed_bytes_device;
  CUDA_CHECK(cudaMalloc(&compressed_bytes_device, sizeof(size_t)));

  nvcompBatchedCascadedOpts_t comp_opts
      = {batch_size, nvcomp::TypeOf<data_type>(), 2, 1, use_bp};

  auto status = nvcompBatchedCascadedCompressAsync(
      uncompressed_ptrs_device,
      uncompressed_bytes_device,
      0, // not used
      1,
      nullptr, // not used
      0,       // not used
      compressed_ptrs_device,
      compressed_bytes_device,
      comp_opts,
      0);

  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(0));

  size_t compressed_byte;
  CUDA_CHECK(cudaMemcpy(
      &compressed_byte,
      compressed_bytes_device,
      sizeof(size_t),
      cudaMemcpyDeviceToHost));

  std::vector<size_t> test_compressed_bytes_host;
  std::vector<size_t> test_decompressed_bytes_host;
  std::vector<nvcompStatus_t> expected_statuses;

  // Case 1: the compressed buffer is truncated

  test_compressed_bytes_host.push_back(compressed_byte / 2);
  test_decompressed_bytes_host.push_back(uncompressed_byte);
  expected_statuses.push_back(nvcompErrorCannotDecompress);

  // Case 2: the decompressed buffer is too small

  test_compressed_bytes_host.push_back(compressed_byte);
  test_decompressed_bytes_host.push_back(uncompressed_byte / 2);
  expected_statuses.push_back(nvcompErrorCannotDecompress);

  test_compressed_bytes_host.push_back(compressed_byte);
  test_decompressed_bytes_host.push_back(uncompressed_byte - 1);
  expected_statuses.push_back(nvcompErrorCannotDecompress);

  // Case 3: correct decompression

  test_compressed_bytes_host.push_back(compressed_byte);
  test_decompressed_bytes_host.push_back(uncompressed_byte);
  expected_statuses.push_back(nvcompSuccess);

  const size_t num_cases = expected_statuses.size();
  std::vector<void*> test_compressed_ptrs_host;
  std::vector<void*> test_decompressed_ptrs_host;

  for (size_t partition_idx = 0; partition_idx < num_cases; partition_idx++) {
    test_compressed_ptrs_host.push_back(compressed_data);

    void* decompressed_ptr;
    CUDA_CHECK(cudaMalloc(
        &decompressed_ptr, test_decompressed_bytes_host[partition_idx]));
    test_decompressed_ptrs_host.push_back(decompressed_ptr);
  }

  void** test_compressed_ptrs_device;
  CUDA_CHECK(
      cudaMalloc(&test_compressed_ptrs_device, sizeof(void*) * num_cases));
  CUDA_CHECK(cudaMemcpy(
      test_compressed_ptrs_device,
      test_compressed_ptrs_host.data(),
      sizeof(void*) * num_cases,
      cudaMemcpyHostToDevice));

  size_t* test_compressed_bytes_device;
  CUDA_CHECK(
      cudaMalloc(&test_compressed_bytes_device, sizeof(size_t) * num_cases));
  CUDA_CHECK(cudaMemcpy(
      test_compressed_bytes_device,
      test_compressed_bytes_host.data(),
      sizeof(size_t) * num_cases,
      cudaMemcpyHostToDevice));

  void** test_decompressed_ptrs_device;
  CUDA_CHECK(
      cudaMalloc(&test_decompressed_ptrs_device, sizeof(void*) * num_cases));
  CUDA_CHECK(cudaMemcpy(
      test_decompressed_ptrs_device,
      test_decompressed_ptrs_host.data(),
      sizeof(void*) * num_cases,
      cudaMemcpyHostToDevice));

  size_t* test_decompressed_bytes_device;
  CUDA_CHECK(
      cudaMalloc(&test_decompressed_bytes_device, sizeof(size_t) * num_cases));
  CUDA_CHECK(cudaMemcpy(
      test_decompressed_bytes_device,
      test_decompressed_bytes_host.data(),
      sizeof(size_t) * num_cases,
      cudaMemcpyHostToDevice));

  size_t* actual_decompressed_bytes;
  CUDA_CHECK(
      cudaMalloc(&actual_decompressed_bytes, sizeof(size_t) * num_cases));

  nvcompStatus_t* decompression_statuses;
  CUDA_CHECK(
      cudaMalloc(&decompression_statuses, sizeof(nvcompStatus_t) * num_cases));

  status = nvcompBatchedCascadedDecompressAsync(
      test_compressed_ptrs_device,
      test_compressed_bytes_device,
      test_decompressed_bytes_device,
      actual_decompressed_bytes,
      num_cases,
      nullptr, // not used
      0,       // not used
      test_decompressed_ptrs_device,
      decompression_statuses,
      0);

  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(0));

  std::vector<nvcompStatus_t> decompression_statuses_host(num_cases);
  CUDA_CHECK(cudaMemcpy(
      decompression_statuses_host.data(),
      decompression_statuses,
      sizeof(nvcompStatus_t) * num_cases,
      cudaMemcpyDeviceToHost));

  for (size_t partition_idx = 0; partition_idx < num_cases; partition_idx++) {
    REQUIRE(
        decompression_statuses_host[partition_idx]
        == expected_statuses[partition_idx]);
  }

  // Cleanup

  CUDA_CHECK(cudaFree(uncompressed_data));
  CUDA_CHECK(cudaFree(uncompressed_ptrs_device));
  CUDA_CHECK(cudaFree(uncompressed_bytes_device));
  CUDA_CHECK(cudaFree(compressed_data));
  CUDA_CHECK(cudaFree(compressed_ptrs_device));
  CUDA_CHECK(cudaFree(compressed_bytes_device));
  for (void* const& ptr : test_decompressed_ptrs_host)
    CUDA_CHECK(cudaFree(ptr));
  CUDA_CHECK(cudaFree(test_compressed_ptrs_device));
  CUDA_CHECK(cudaFree(test_compressed_bytes_device));
  CUDA_CHECK(cudaFree(test_decompressed_ptrs_device));
  CUDA_CHECK(cudaFree(test_decompressed_bytes_device));
  CUDA_CHECK(cudaFree(actual_decompressed_bytes));
  CUDA_CHECK(cudaFree(decompression_statuses));
}

TEST_CASE("BatchedCascadedCompressor predefined-cases", "[nvcomp]")
{
  test_predefined_cases<int8_t>(0);
  test_predefined_cases<int8_t>(1);
  test_predefined_cases<uint8_t>(0);
  test_predefined_cases<uint8_t>(1);
  test_predefined_cases<int16_t>(0);
  test_predefined_cases<int16_t>(1);
  test_predefined_cases<uint16_t>(0);
  test_predefined_cases<uint16_t>(1);
  test_predefined_cases<int32_t>(0);
  test_predefined_cases<int32_t>(1);
  test_predefined_cases<uint32_t>(0);
  test_predefined_cases<uint32_t>(1);
  test_predefined_cases<int64_t>(0);
  test_predefined_cases<int64_t>(1);
  test_predefined_cases<uint64_t>(0);
  test_predefined_cases<uint64_t>(1);
}
TEST_CASE("BatchedCascadedCompressor fallback-path", "[nvcomp]")
{
  test_fallback_path<int8_t>();
  test_fallback_path<uint8_t>();
  test_fallback_path<int16_t>();
  test_fallback_path<uint16_t>();
  test_fallback_path<int32_t>();
  test_fallback_path<uint32_t>();
  test_fallback_path<int64_t>();
  test_fallback_path<uint64_t>();
}

TEST_CASE("BatchedCascadedCompressor invalid-decompressed-size", "[nvcomp]")
{
  void* compressed_buffer;
  size_t* compressed_bytes;
  size_t* uncompressed_bytes;
  // A well-formed compressed buffer should have size at least 8 due to
  // metadata. We delibrately set it to 4 here to see whether the implementation
  // can fail gracefully.
  constexpr size_t compressed_byte_host = 4;
  // Set this field to a number other than 0 here. Later, we will copy the
  // uncompressed byte back to this field, and it should contain 0 due to OOB
  // access.
  size_t uncompressed_byte_host = 1;

  CUDA_CHECK(cudaMalloc(&compressed_buffer, compressed_byte_host));
  CUDA_CHECK(cudaMalloc(&compressed_bytes, sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&uncompressed_bytes, sizeof(size_t)));

  CUDA_CHECK(cudaMemcpy(
      compressed_bytes,
      &compressed_byte_host,
      sizeof(size_t),
      cudaMemcpyHostToDevice));

  auto status = nvcompBatchedCascadedGetDecompressSizeAsync(
      &compressed_buffer, compressed_bytes, uncompressed_bytes, 1, 0);

  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(0));

  CUDA_CHECK(cudaMemcpy(
      &uncompressed_byte_host,
      uncompressed_bytes,
      sizeof(size_t),
      cudaMemcpyDeviceToHost));
  REQUIRE(uncompressed_byte_host == 0);

  CUDA_CHECK(cudaFree(compressed_buffer));
  CUDA_CHECK(cudaFree(compressed_bytes));
  CUDA_CHECK(cudaFree(uncompressed_bytes));
}

TEST_CASE("BatchedCascadedCompressor out-of-bound", "[nvcomp]")
{
  test_out_of_bound<int8_t>(0);
  test_out_of_bound<int8_t>(1);
  test_out_of_bound<uint8_t>(0);
  test_out_of_bound<uint8_t>(1);
  test_out_of_bound<int16_t>(0);
  test_out_of_bound<int16_t>(1);
  test_out_of_bound<uint16_t>(0);
  test_out_of_bound<uint16_t>(1);
  test_out_of_bound<int32_t>(0);
  test_out_of_bound<int32_t>(1);
  test_out_of_bound<uint32_t>(0);
  test_out_of_bound<uint32_t>(1);
  test_out_of_bound<int64_t>(0);
  test_out_of_bound<int64_t>(1);
  test_out_of_bound<uint64_t>(0);
  test_out_of_bound<uint64_t>(1);
}

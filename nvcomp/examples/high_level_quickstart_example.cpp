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

#include <random>
#include <assert.h>
#include <iostream>

#include "nvcomp/lz4.hpp"
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

/* 
  To build, execute
  
  mkdir build
  cd build
  cmake -DBUILD_EXAMPLES=ON ..
  make -j

  To execute, 
  bin/high_level_quickstart_example
*/

using namespace nvcomp;

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = cond;                                                    \
    if (err != cudaSuccess) {                                               \
      std::cerr << "Failure" << std::endl;                                \
      exit(1);                                                              \
    }                                                                         \
  } while (false)

/**
 * In this example, we:
 *  1) compress the input data
 *  2) construct a new manager using the input data for demonstration purposes
 *  3) decompress the input data
 */ 
void decomp_compressed_with_manager_factory_example(uint8_t* device_input_ptrs, const size_t input_buffer_len)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  LZ4Manager nvcomp_manager{chunk_size, data_type, stream};
  CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

  uint8_t* comp_buffer;
  CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));
  
  nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);

  // Construct a new nvcomp manager from the compressed buffer.
  // Note we could use the nvcomp_manager from above, but here we demonstrate how to create a manager 
  // for the use case where a buffer is received and the user doesn't know how it was compressed
  // Also note, creating the manager in this way synchronizes the stream, as the compressed buffer must be read to 
  // construct the manager
  auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);

  DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
  uint8_t* res_decomp_buffer;
  CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

  decomp_nvcomp_manager->decompress(res_decomp_buffer, comp_buffer, decomp_config);

  CUDA_CHECK(cudaFree(comp_buffer));
  CUDA_CHECK(cudaFree(res_decomp_buffer));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaStreamDestroy(stream));
}

/**
 * In this example, we:
 *  1) construct an nvcompManager
 *  2) compress the input data
 *  3) decompress the input data
 */ 
void comp_decomp_with_single_manager(uint8_t* device_input_ptrs, const size_t input_buffer_len)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  LZ4Manager nvcomp_manager{chunk_size, data_type, stream};
  CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

  uint8_t* comp_buffer;
  CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));
  
  nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);

  DecompressionConfig decomp_config = nvcomp_manager.configure_decompression(comp_buffer);
  uint8_t* res_decomp_buffer;
  CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

  nvcomp_manager.decompress(res_decomp_buffer, comp_buffer, decomp_config);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(comp_buffer));
  CUDA_CHECK(cudaFree(res_decomp_buffer));

  CUDA_CHECK(cudaStreamDestroy(stream));
}

/**
 * Additionally, we can use the same manager to execute multiple streamed compressions / decompressions
 * In this example we configure the multiple decompressions by inspecting the compressed buffers
 */  
void multi_comp_decomp_example(const std::vector<uint8_t*>& device_input_ptrs, std::vector<size_t>& input_buffer_lengths)
{
  size_t num_buffers = input_buffer_lengths.size();
  
  using namespace std;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  LZ4Manager nvcomp_manager{chunk_size, data_type, stream};
  
  std::vector<uint8_t*> comp_result_buffers(num_buffers);

  for(size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    uint8_t* input_data = device_input_ptrs[ix_buffer];
    size_t input_length = input_buffer_lengths[ix_buffer];

    auto comp_config = nvcomp_manager.configure_compression(input_length);

    CUDA_CHECK(cudaMalloc(&comp_result_buffers[ix_buffer], comp_config.max_compressed_buffer_size));
    nvcomp_manager.compress(input_data, comp_result_buffers[ix_buffer], comp_config);    
  }

  std::vector<uint8_t*> decomp_result_buffers(num_buffers);
  for(size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    uint8_t* comp_data = comp_result_buffers[ix_buffer];

    auto decomp_config = nvcomp_manager.configure_decompression(comp_data);

    CUDA_CHECK(cudaMalloc(&decomp_result_buffers[ix_buffer], decomp_config.decomp_data_size));

    nvcomp_manager.decompress(decomp_result_buffers[ix_buffer], comp_data, decomp_config);    
  }

  for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    CUDA_CHECK(cudaFree(decomp_result_buffers[ix_buffer]));
    CUDA_CHECK(cudaFree(comp_result_buffers[ix_buffer]));
  }
}

/**
 * Additionally, we can use the same manager to execute multiple streamed compressions / decompressions
 * In this example we configure the multiple decompressions by storing the comp_config's and inspecting those
 */  
void multi_comp_decomp_example_comp_config(const std::vector<uint8_t*>& device_input_ptrs, std::vector<size_t>& input_buffer_lengths)
{
  size_t num_buffers = input_buffer_lengths.size();

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  LZ4Manager nvcomp_manager{chunk_size, data_type, stream};
  
  std::vector<CompressionConfig> comp_configs;
  comp_configs.reserve(num_buffers);

  std::vector<uint8_t*> comp_result_buffers(num_buffers);

  for(size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    uint8_t* input_data = device_input_ptrs[ix_buffer];
    size_t input_length = input_buffer_lengths[ix_buffer];

    comp_configs.push_back(nvcomp_manager.configure_compression(input_length));
    auto& comp_config = comp_configs.back();

    CUDA_CHECK(cudaMalloc(&comp_result_buffers[ix_buffer], comp_config.max_compressed_buffer_size));

    nvcomp_manager.compress(input_data, comp_result_buffers[ix_buffer], comp_config);    
  }

  std::vector<uint8_t*> decomp_result_buffers(num_buffers);
  for(size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    auto decomp_config = nvcomp_manager.configure_decompression(comp_configs[ix_buffer]);

    CUDA_CHECK(cudaMalloc(&decomp_result_buffers[ix_buffer], decomp_config.decomp_data_size));

    nvcomp_manager.decompress(decomp_result_buffers[ix_buffer], comp_result_buffers[ix_buffer], decomp_config);    
  }

  for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    CUDA_CHECK(cudaFree(decomp_result_buffers[ix_buffer]));
    CUDA_CHECK(cudaFree(comp_result_buffers[ix_buffer]));
  }
}

int main()
{
  // Initialize a random array of chars
  const size_t input_buffer_len = 1000000;
  std::vector<uint8_t> uncompressed_data(input_buffer_len);
  
  std::mt19937 random_gen(42);

  // char specialization of std::uniform_int_distribution is
  // non-standard, and isn't available on MSVC, so use short instead,
  // but with the range limited, and then cast below.
  std::uniform_int_distribution<short> uniform_dist(0, 255);
  for (size_t ix = 0; ix < input_buffer_len; ++ix) {
    uncompressed_data[ix] = static_cast<uint8_t>(uniform_dist(random_gen));
  }

  uint8_t* device_input_ptrs;
  CUDA_CHECK(cudaMalloc(&device_input_ptrs, input_buffer_len));
  CUDA_CHECK(cudaMemcpy(device_input_ptrs, uncompressed_data.data(), input_buffer_len, cudaMemcpyDefault));
  
  // Two roundtrip examples
  decomp_compressed_with_manager_factory_example(device_input_ptrs, input_buffer_len);
  comp_decomp_with_single_manager(device_input_ptrs, input_buffer_len);

  CUDA_CHECK(cudaFree(device_input_ptrs));

  // Multi buffer example
  const size_t num_buffers = 10;

  std::vector<uint8_t*> gpu_buffers(num_buffers);
  std::vector<size_t> input_buffer_lengths(num_buffers);

  std::vector<std::vector<uint8_t>> uncompressed_buffers(num_buffers);
  for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    uncompressed_buffers[ix_buffer].resize(input_buffer_len);
    for (size_t ix_byte = 0; ix_byte < input_buffer_len; ++ix_byte) {
      uncompressed_buffers[ix_buffer][ix_byte] = static_cast<uint8_t>(uniform_dist(random_gen));
    }
    CUDA_CHECK(cudaMalloc(&gpu_buffers[ix_buffer], input_buffer_len));
    CUDA_CHECK(cudaMemcpy(gpu_buffers[ix_buffer], uncompressed_buffers[ix_buffer].data(), input_buffer_len, cudaMemcpyDefault));
    input_buffer_lengths[ix_buffer] = input_buffer_len;
  }

  multi_comp_decomp_example(gpu_buffers, input_buffer_lengths);
  multi_comp_decomp_example_comp_config(gpu_buffers, input_buffer_lengths);

  for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    CUDA_CHECK(cudaFree(gpu_buffers[ix_buffer]));
  }
  return 0;
}
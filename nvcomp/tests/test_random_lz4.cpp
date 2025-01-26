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

#define CATCH_CONFIG_MAIN

#ifndef VERBOSE
#define VERBOSE 0
#endif

#include "test_common.h"
#include "nvcomp/lz4.hpp"

// Test method that takes an input data, compresses it (on the CPU),
// decompresses it on the GPU, and verifies it is correct.
// Uses LZ4 Compression
template <typename T>
void test_lz4(const std::vector<T>& data, size_t /*chunk_size*/)
{
  const nvcompType_t type = nvcomp::TypeOf<T>();

  size_t chunk_size = 1 << 16;

#if VERBOSE > 1
  // dump input data
  std::cout << "Input" << std::endl;
  for (size_t i = 0; i < data.size(); i++)
    std::cout << data[i] << " ";
  std::cout << std::endl;
#endif

  // these two items will be the only forms of communication between
  // compression and decompression
  uint8_t* d_comp_out = nullptr;
  size_t comp_out_bytes = 0;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  {
    // this block handles compression, and we scope it to ensure only
    // serialized metadata and compressed data, are the only things passed
    // between compression and decopmression
    std::cout << "----------" << std::endl;
    std::cout << "uncompressed (B): " << data.size() * sizeof(T) << std::endl;

    // create GPU only input buffer
    uint8_t* d_in_data;
    const size_t in_bytes = sizeof(T) * data.size();
    CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
    CUDA_CHECK(
        cudaMemcpy(d_in_data, data.data(), in_bytes, cudaMemcpyHostToDevice));

    LZ4Manager lz4_manager(chunk_size, NVCOMP_TYPE_CHAR, stream);
    
    auto comp_config = lz4_manager.configure_compression(in_bytes);
    CUDA_CHECK(cudaMalloc(&d_comp_out, comp_config.max_compressed_buffer_size));

    size_t* comp_out_bytes_ptr;
    CUDA_CHECK(cudaMallocHost(
        (void**)&comp_out_bytes_ptr, sizeof(*comp_out_bytes_ptr)));

    lz4_manager.compress(
        d_in_data,
        d_comp_out,
        comp_config);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    size_t comp_out_bytes = lz4_manager.get_compressed_output_size(d_comp_out);

    cudaFree(d_in_data);

    std::cout << "comp_size: " << comp_out_bytes
              << ", compressed ratio: " << std::fixed << std::setprecision(2)
              << (double)in_bytes / comp_out_bytes << std::endl;
  }
  {
    // this block handles decompression, and we scope it to ensure only
    // the compressed data and the stream is passed
    // between compression and decompression

    LZ4Manager lz4_manager(chunk_size, NVCOMP_TYPE_CHAR, stream);
    
    auto decomp_config = lz4_manager.configure_decompression(d_comp_out);

    const auto temp_bytes = lz4_manager.get_required_scratch_buffer_size();

    uint8_t* temp_ptr;
    cudaMalloc(&temp_ptr, temp_bytes);
    lz4_manager.set_scratch_buffer(temp_ptr);

    uint8_t* out_ptr = NULL;
    cudaMalloc(&out_ptr, decomp_config.decomp_data_size);

    lz4_manager.decompress(
        out_ptr,
        d_comp_out,
        decomp_config);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(d_comp_out);
    cudaFree(temp_ptr);

    std::vector<T> res(decomp_config.decomp_data_size / sizeof(T));
    cudaMemcpy(&res[0], out_ptr, decomp_config.decomp_data_size, cudaMemcpyDeviceToHost);

#if VERBOSE > 1
    // dump output data
    std::cout << "Output" << std::endl;
    for (size_t i = 0; i < data.size(); i++)
      std::cout << ((T*)out_ptr)[i] << " ";
    std::cout << std::endl;
#endif

    REQUIRE(res == data);
  }
  cudaStreamDestroy(stream);
}

template <typename T>
void test_random_lz4(
    int max_val,
    int max_run,
    size_t chunk_size)
{
  // generate random data
  std::vector<T> data;
  int seed = (max_val ^ max_run ^ static_cast<int>(chunk_size));
  random_runs(data, (T)max_val, (T)max_run, seed);

  test_lz4<T>(data, chunk_size);
}

// int
TEST_CASE("small-LZ4", "[small]")
{
  test_random_lz4<int>(10, 10, 10000);
}
TEST_CASE("medium-LZ4", "[small]")
{
  test_random_lz4<int>(10000, 10, 100000);
}

TEST_CASE("large-LZ4", "[large][bp]")
{
  test_random_lz4<int>(10000, 1000, 10000000);
}



// long long
TEST_CASE("small-LZ4-ll", "[small]")
{
  test_random_lz4<int64_t>(10, 10, 10000);
}
TEST_CASE("large-LZ4-ll", "[large]")
{
  test_random_lz4<int64_t>(10000, 1000, 10000000);
}

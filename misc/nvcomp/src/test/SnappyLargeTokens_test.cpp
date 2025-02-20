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

#include <random>

#include "cuda_runtime.h"

#include "tests/catch.hpp"

#include "CudaUtils.h"
#include "lowlevel/SnappyBatchKernels.h"

#define CUDA_CHECK(func)                                                       \
  {                                                                            \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      printf(                                                                  \
          "API call failure \"" #func "\" with %d at " __FILE__ ":%d\n",       \
          (int)rt,                                                             \
          __LINE__);                                                           \
      REQUIRE(rt == cudaSuccess);                                              \
    }                                                                          \
  }

using namespace std;

const unsigned MAX_SINGLE_BYTE_LITERALS = 60;

void write_num_literals(
    uint32_t num_literals, uint8_t* output, size_t& ix_output)
{
  --num_literals; // recorded as (num - 1)
  // Single byte case
  if (num_literals < MAX_SINGLE_BYTE_LITERALS) {
    output[ix_output++] = (num_literals << 2);
    return;
  }

  // Multi-byte case
  size_t prev_output_ix = ix_output++;
  while (num_literals > 0xff) {
    output[ix_output++] = num_literals & 0xff;
    num_literals = num_literals >> 8;
  }

  const uint8_t num_bytes = (unsigned)(ix_output - prev_output_ix);
  output[ix_output++] = num_literals;

  output[prev_output_ix] = (MAX_SINGLE_BYTE_LITERALS + num_bytes - 1) << 2;
}

void translate_uncompressed_size(
    uint32_t src_len, uint8_t* output, size_t& ix_output)
{
  while (src_len > 0x7f) {
    output[ix_output++] = src_len | 0x80;
    src_len = src_len >> 7;
  }

  output[ix_output++] = src_len;
}

const unsigned SNAPPY_SINGLE_BYTE_MIN_MATCH_LENGTH = 4;
const unsigned SNAPPY_SINGLE_BYTE_MAX_MATCH_LENGTH = 11;
const unsigned SNAPPY_SINGLE_BYTE_MAX_OFFSET = 2047;

void encode_copy(
    uint32_t offset, uint8_t match_length, uint8_t* output, size_t& ix_output)
{
  if (match_length >= SNAPPY_SINGLE_BYTE_MIN_MATCH_LENGTH
      and match_length <= SNAPPY_SINGLE_BYTE_MAX_MATCH_LENGTH
      and offset <= SNAPPY_SINGLE_BYTE_MAX_OFFSET) {
    // 1 byte offset encoding. the  tag byte is: 
    // [5..7: upper 3 bits of offset],[2..4: (match_length - 4)],[0..1: 01, indicates 1 byte offset]
    uint8_t lower_bits = offset & 0xff;
    offset = (offset >> 8) << 5;
    output[ix_output++]
        = ((match_length - SNAPPY_SINGLE_BYTE_MIN_MATCH_LENGTH) << 2) | 0x01
          | offset;
    output[ix_output++] = lower_bits;
  } else {
    // 2 or 4 byte offset encoding
    uint8_t store_len = match_length - 1;
    output[ix_output] = (store_len << 2);
    uint8_t num_offset_bytes;
    if (offset < (1 << 16)) {
      output[ix_output++] |= 0x02;
      num_offset_bytes = 2;
    } else {
      num_offset_bytes = 4;
      output[ix_output++] |= 0x03;
    }

    for (int ix = 0; ix < num_offset_bytes; ++ix) {
      output[ix_output++] = offset & 0xff;
      offset >>= 8;
    }
  }
}

void generate_random_vals(
    std::vector<uint8_t>& res,
    const uint8_t max_val,
    const size_t num_vals,
    int seed)
{
  std::mt19937 eng(seed);
  std::uniform_int_distribution<> distr(0, max_val);

  for (size_t ix = 0; ix < num_vals; ++ix) {
    res.push_back(distr(eng));
  }
}

void compress_single_batch_snappy(
    uint8_t* h_uncomp_data,
    uint8_t* h_comp_data,
    size_t uncomp_data_size,
    size_t avail_comp_size)
{
  // prepare gpu buffers
  void* device_chunk_input_data;
  CUDA_CHECK(cudaMalloc(&device_chunk_input_data, uncomp_data_size));
  CUDA_CHECK(cudaMemcpy(
      device_chunk_input_data,
      h_uncomp_data,
      uncomp_data_size,
      cudaMemcpyHostToDevice));

  void* device_chunk_comp_data;
  CUDA_CHECK(cudaMalloc(&device_chunk_comp_data, avail_comp_size));

  void** d_in_data;
  CUDA_CHECK(cudaMalloc((void**)(&d_in_data), sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(
      d_in_data,
      &device_chunk_input_data,
      sizeof(size_t),
      cudaMemcpyHostToDevice));

  void** d_out_data;
  CUDA_CHECK(cudaMalloc((void**)(&d_out_data), sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(
      d_out_data,
      &device_chunk_comp_data,
      sizeof(size_t),
      cudaMemcpyHostToDevice));

  size_t* d_in_bytes;
  CUDA_CHECK(cudaMalloc(&d_in_bytes, sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(
      d_in_bytes, &uncomp_data_size, sizeof(size_t), cudaMemcpyHostToDevice));

  size_t* d_out_bytes;
  CUDA_CHECK(cudaMalloc(&d_out_bytes, sizeof(size_t)));

  size_t* d_out_avail_bytes;
  CUDA_CHECK(cudaMalloc(&d_out_avail_bytes, sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(
      d_out_avail_bytes,
      &avail_comp_size,
      sizeof(size_t),
      cudaMemcpyHostToDevice));

  nvcomp::gpu_snappy_status_s* d_out_status;
  CUDA_CHECK(cudaMalloc(&d_out_status, sizeof(nvcomp::gpu_snappy_status_s)));

  const int num_chunks = 1;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  nvcomp::gpu_snap(
      d_in_data,
      d_in_bytes,
      d_out_data,
      d_out_avail_bytes,
      d_out_status,
      d_out_bytes,
      num_chunks,
      stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  nvcomp::gpu_snappy_status_s final_status;
  CUDA_CHECK(cudaMemcpy(
      &final_status,
      d_out_status,
      sizeof(nvcomp::gpu_snappy_status_s),
      cudaMemcpyDeviceToHost));
  REQUIRE(final_status.status == 0);

  size_t gpu_compressed_size;
  CUDA_CHECK(cudaMemcpy(
      &gpu_compressed_size,
      d_out_bytes,
      sizeof(size_t),
      cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaMemcpy(
      h_comp_data,
      device_chunk_comp_data,
      gpu_compressed_size,
      cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(device_chunk_comp_data));
  CUDA_CHECK(cudaFree(device_chunk_input_data));
  CUDA_CHECK(cudaFree(d_in_data));
  CUDA_CHECK(cudaFree(d_out_data));
  CUDA_CHECK(cudaFree(d_in_bytes));
  CUDA_CHECK(cudaFree(d_out_bytes));
  CUDA_CHECK(cudaFree(d_out_avail_bytes));
  CUDA_CHECK(cudaFree(d_out_status));
}

void decompress_single_batch_snappy(
    uint8_t* h_comp_data,
    uint8_t* h_decomp_data,
    size_t comp_data_size,
    size_t decomp_data_size)
{
  // prepare gpu buffers
  void* device_chunk_comp_data;
  CUDA_CHECK(cudaMalloc(&device_chunk_comp_data, comp_data_size));
  CUDA_CHECK(cudaMemcpy(
      device_chunk_comp_data,
      h_comp_data,
      comp_data_size,
      cudaMemcpyHostToDevice));

  void* device_chunk_decomp_data;
  CUDA_CHECK(cudaMalloc(&device_chunk_decomp_data, decomp_data_size));

  void** d_in_data;
  CUDA_CHECK(cudaMalloc((void**)(&d_in_data), sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(
      d_in_data,
      &device_chunk_comp_data,
      sizeof(size_t),
      cudaMemcpyHostToDevice));

  void** d_out_data;
  CUDA_CHECK(cudaMalloc((void**)(&d_out_data), sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(
      d_out_data,
      &device_chunk_decomp_data,
      sizeof(size_t),
      cudaMemcpyHostToDevice));

  size_t* d_in_bytes;
  CUDA_CHECK(cudaMalloc(&d_in_bytes, sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(
      d_in_bytes, &comp_data_size, sizeof(size_t), cudaMemcpyHostToDevice));

  size_t* d_out_bytes;
  CUDA_CHECK(cudaMalloc(&d_out_bytes, sizeof(size_t)));

  size_t* d_out_avail_bytes;
  CUDA_CHECK(cudaMalloc(&d_out_avail_bytes, sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(
      d_out_avail_bytes,
      &decomp_data_size,
      sizeof(size_t),
      cudaMemcpyHostToDevice));

  nvcompStatus_t* d_out_status;
  CUDA_CHECK(cudaMalloc(&d_out_status, sizeof(nvcompStatus_t)));

  const int num_chunks = 1;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  nvcomp::gpu_unsnap(
      d_in_data,
      d_in_bytes,
      d_out_data,
      d_out_avail_bytes,
      d_out_status,
      d_out_bytes,
      num_chunks,
      stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  nvcompStatus_t final_status;
  CUDA_CHECK(cudaMemcpy(
      &final_status,
      d_out_status,
      sizeof(nvcompStatus_t),
      cudaMemcpyDeviceToHost));
  REQUIRE(final_status == nvcompSuccess);

  size_t gpu_decompressed_size;
  CUDA_CHECK(cudaMemcpy(
      &gpu_decompressed_size,
      d_out_bytes,
      sizeof(size_t),
      cudaMemcpyDeviceToHost));
  REQUIRE(gpu_decompressed_size == decomp_data_size);

  // Get the number of bytes back
  uint8_t* h_decomp_check_buffer = (uint8_t*)malloc(decomp_data_size);
  CUDA_CHECK(cudaMemcpy(
      h_decomp_check_buffer,
      device_chunk_decomp_data,
      decomp_data_size,
      cudaMemcpyDeviceToHost));

  for (size_t ix = 0; ix < decomp_data_size; ++ix) {
    if (h_decomp_check_buffer[ix] != h_decomp_data[ix]) {
      cerr << "Ix " << ix << " decomp buffer "
           << unsigned(h_decomp_check_buffer[ix]) << " correct val "
           << unsigned(h_decomp_data[ix]) << endl;
    }
    REQUIRE(h_decomp_check_buffer[ix] == h_decomp_data[ix]);
  }

  free(h_decomp_check_buffer);
  CUDA_CHECK(cudaFree(device_chunk_comp_data));
  CUDA_CHECK(cudaFree(device_chunk_decomp_data));
  CUDA_CHECK(cudaFree(d_in_data));
  CUDA_CHECK(cudaFree(d_out_data));
  CUDA_CHECK(cudaFree(d_in_bytes));
  CUDA_CHECK(cudaFree(d_out_bytes));
  CUDA_CHECK(cudaFree(d_out_avail_bytes));
  CUDA_CHECK(cudaFree(d_out_status));
}

// Just testing that the compressed data generator works.
// Just a stream of literals that don't repeat and are <= 256, so can
// check that the literal compression matches.
TEST_CASE("test_mock_literal_compressor", "[small]")
{
  const unsigned LITERAL_SIZE = 256;
  uint8_t true_uncomp_data[LITERAL_SIZE];

  constexpr unsigned COMP_DATA_SIZE = LITERAL_SIZE * 2;
  uint8_t comp_data[COMP_DATA_SIZE];
  size_t ix_output = 0;

  translate_uncompressed_size(LITERAL_SIZE, comp_data, ix_output);
  write_num_literals(LITERAL_SIZE, comp_data, ix_output);

  for (unsigned ix_input = 0; ix_input < LITERAL_SIZE; ++ix_input) {
    uint8_t input_val = ix_input % 256;
    true_uncomp_data[ix_input] = input_val;
    comp_data[ix_output++] = input_val;
  }

  uint8_t gpu_comp_data[COMP_DATA_SIZE];
  compress_single_batch_snappy(
      true_uncomp_data, gpu_comp_data, LITERAL_SIZE, COMP_DATA_SIZE);

  for (size_t ix_comp = 0; ix_comp < ix_output; ++ix_comp) {
    REQUIRE(comp_data[ix_comp] == gpu_comp_data[ix_comp]);
  }
}

// Testing that the mock copy function works. Use small matches that either can
// accomodate. Choose the copies in the same way that the snappy compressor
// does; just sanity checking the formatting here.
TEST_CASE("test_mock_match_compressor", "[small]")
{
  // Make a large literal collection of bytes.
  const unsigned INPUT_SIZE = 256 + 64;
  uint8_t true_uncomp_data[INPUT_SIZE];

  const unsigned COMP_DATA_SIZE = 2048;
  uint8_t comp_data[COMP_DATA_SIZE];
  size_t ix_output = 0;

  translate_uncompressed_size(INPUT_SIZE, comp_data, ix_output);

  // Add 256 literals, in order
  write_num_literals(256, comp_data, ix_output);

  unsigned ix_input = 0;
  for (; ix_input < 256; ++ix_input) {
    uint8_t input_val = ix_input;
    true_uncomp_data[ix_input] = input_val;
    comp_data[ix_output++] = input_val;
  }

  // Then add a match of the first 64 values
  for (int ix = 0; ix < 64; ++ix) {
    true_uncomp_data[ix_input++] = ix;
  }

  encode_copy(256, 64, comp_data, ix_output);

  uint8_t gpu_comp_data[COMP_DATA_SIZE];
  compress_single_batch_snappy(
      true_uncomp_data, gpu_comp_data, INPUT_SIZE, COMP_DATA_SIZE);

  for (size_t ix_comp = 0; ix_comp < ix_output; ++ix_comp) {
    REQUIRE(comp_data[ix_comp] == gpu_comp_data[ix_comp]);
  }
}

// Test decompressing a single collection of > 256 literals. This can't be
// produced by the GPU compressor, so test that the decompressor can accomodate
// this here.
TEST_CASE("decompress_large_literal", "[small]")
{
  // Make a large literal collection of bytes.
  const unsigned LITERAL_SIZE = 512;
  uint8_t true_uncomp_data[LITERAL_SIZE];

  const unsigned COMP_DATA_SIZE = 2048;
  uint8_t comp_data[COMP_DATA_SIZE];
  size_t ix_output = 0;

  translate_uncompressed_size(LITERAL_SIZE, comp_data, ix_output);
  write_num_literals(LITERAL_SIZE, comp_data, ix_output);

  for (unsigned ix_input = 0; ix_input < LITERAL_SIZE; ++ix_input) {
    uint8_t input_val = ix_input % 256;
    true_uncomp_data[ix_input] = input_val;
    comp_data[ix_output++] = input_val;
  }

  decompress_single_batch_snappy(
      comp_data,
      true_uncomp_data,
      ix_output, // comp_data_size
      LITERAL_SIZE);
}

// Performs a series of tests based on the mock compressor writing out a long
// stream of literals, followed by an explicit match that looks back at the
// first 35 bytes of the literal stream.
void test_long_match_case(size_t num_initial_ints)
{
  // Produce a long stream of literals. The test compressor will just encode
  // these as literals without looking for matches
  vector<uint8_t> decomp_vals;

  vector<uint8_t> comp_vals(num_initial_ints << 1);

  const unsigned match_length = 35;

  decomp_vals.reserve(num_initial_ints + match_length);

  generate_random_vals(
      decomp_vals, 255 /* max random val */, num_initial_ints, 42 /*seed*/);

  // Then copy the first match_length values from the randomly generated set
  decomp_vals.insert(
      decomp_vals.end(),
      decomp_vals.begin(),
      decomp_vals.begin() + match_length);

  // Execute the mock compressor
  size_t ix_output = 0;
  translate_uncompressed_size(decomp_vals.size(), comp_vals.data(), ix_output);

  write_num_literals(num_initial_ints, comp_vals.data(), ix_output);

  memcpy(comp_vals.data() + ix_output, decomp_vals.data(), num_initial_ints);
  ix_output += num_initial_ints;

  encode_copy(num_initial_ints, match_length, comp_vals.data(), ix_output);

  decompress_single_batch_snappy(
      comp_vals.data(),
      decomp_vals.data(),
      ix_output, // comp_data_size
      decomp_vals.size());
}

// Test decompressing a match with an offset that exceeds 32 kB. Again, can't be
// produced by the GPU compressor. Also note that this won't be produced by
// Google's snappy implementation, but they indicate that the decompressor
// should not rely on this, so test this here.
TEST_CASE("decompress_long_2B_match_case", "[small]")
{
  const size_t num_ints = (1 << 15) + 100;
  test_long_match_case(num_ints);
}

// Test decompressing a match with an offset that exceeds 64 kB. This leads to a
// 4-byte offset. Again, can't be produced by the GPU compressor. Also note that
// this won't be produced by Google's snappy implementation, but they indicate
// that the decompressor should not rely on this, so test this here.
TEST_CASE("decompress_long_4B_match_case", "[small]")
{
  const size_t num_ints = (1 << 16) + 100;
  test_long_match_case(num_ints);
}

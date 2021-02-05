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

#include "CascadedSelector.h"
#include "CascadedSelectorKernels.h"
#include "TempSpaceBroker.h"
#include "common.h"
#include "nvcomp.hpp"
#include "type_macros.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <stdlib.h>
#include <vector>
#include <cstddef>
#include <stdexcept>

using namespace std;
using namespace nvcomp;

namespace
{
// Number of compression schemes to try to determine best configuration
constexpr int const NUM_SCHEMES = 5; 
// Maximum value allowed for sample_size
constexpr int const MAX_SAMPLE_SIZE = 1024;

template <typename T>
void get_workspace_size_internal(const size_t num_samples, size_t* temp_size)
{
  size_t required_size = sizeof(size_t) * num_samples;
  required_size = roundUpTo(required_size, 8);
  required_size += sizeof(unsigned long long int) * NUM_SCHEMES;
  *temp_size = required_size;
}

template <typename T>
nvcompCascadedFormatOpts internal_select(
    const void* input_data,
    const size_t in_bytes,
    const size_t sample_ele,
    const size_t num_samples,
    double* comp_ratio,
    void* d_temp_comp,
    const size_t workspace_size,
    const size_t max_size,
    unsigned seed,
    cudaStream_t stream)
{

  if (workspace_size < max_size) {
    throw std::runtime_error(
        "Insufficient workspace for perform selection: "
        + std::to_string(workspace_size) + " / " + std::to_string(max_size));
  }

  if (sample_ele > MAX_SAMPLE_SIZE) {
    throw std::runtime_error("sample size is too large, the maximum number of "
                             "elements per sample is 1024\n");
  }

  const size_t sample_bytes = sample_ele * sizeof(T);

  size_t num_chunks = in_bytes / sample_bytes;
  size_t bracket_size = num_chunks / num_samples;
  std::vector<size_t> sample_offsets(num_samples);

  // TODO: The last chunk of the input data is discarded. Change it so that the
  // last chunk can be also incuded during the sampling process.

  std::minstd_rand0 g1(seed);

  for (size_t i = 0; i < num_samples; i++) {
    int idx = g1() % bracket_size;
    sample_offsets[i] = idx + bracket_size * i;
  }

  size_t out_sizes[NUM_SCHEMES];
  std::vector<size_t> sample_ptrs(num_samples);

  for (size_t i = 0; i < num_samples; i++) {

    size_t sample_off = sample_offsets[i] * sample_ele;
    sample_ptrs[i] = sample_off;
  }

  size_t* d_sample_ptrs;
  TempSpaceBroker tempSpace(d_temp_comp, workspace_size);
  tempSpace.reserve(&d_sample_ptrs, num_samples);

  cudaMemcpyAsync(
      d_sample_ptrs,
      sample_ptrs.data(),
      sizeof(size_t) * num_samples,
      cudaMemcpyHostToDevice,
      stream);

  SamplingFastOption(
      input_data,
      d_sample_ptrs,
      sample_bytes,
      num_samples,
      getnvcompType<T>(),
      tempSpace.next(),
      tempSpace.spaceLeft(),
      out_sizes,
      NUM_SCHEMES,
      stream);

  cudaError_t err = cudaStreamSynchronize(stream);

  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Fail to launch SampleFusedOption: " + std::to_string(err));
  }

  std::vector<size_t> outsizeVector;
  for (int i = 0; i < NUM_SCHEMES; i++) {
    outsizeVector.push_back(out_sizes[i]);
  }

  std::vector<size_t>::iterator result;
  result = std::min_element(outsizeVector.begin(), outsizeVector.end());

  int idx = std::distance(outsizeVector.begin(), result);
  int RLEs = idx / 2;
  int Deltas = (RLEs == 2) ? 1 : (idx % 2);

  *comp_ratio
      = ((double)(sample_bytes * num_samples) / (double)(outsizeVector[idx]));
  nvcompCascadedFormatOpts opts = {RLEs, Deltas, 1};

  return opts;
}
} // namespace

namespace nvcomp
{

namespace internal
{

// Define types that are acceptable for Cascaded Compression
template class CascadedSelector<int8_t>;
template class CascadedSelector<uint8_t>;
template class CascadedSelector<int16_t>;
template class CascadedSelector<uint16_t>;
template class CascadedSelector<int32_t>;
template class CascadedSelector<uint32_t>;
template class CascadedSelector<int64_t>;
template class CascadedSelector<uint64_t>;

template<typename T>
inline CascadedSelector<T>::CascadedSelector(
    const void* input,
    const size_t byte_len,
    nvcompCascadedSelectorOpts selector_opts):
    input_data(input),
    input_byte_len(byte_len),
    max_temp_size(0),
    opts(selector_opts)
{
  size_t temp;

  NVCOMP_TYPE_ONE_SWITCH(
      getnvcompType<T>(), get_workspace_size_internal, opts.num_samples, &temp);

  temp = roundUpTo(temp, 8);

  this->max_temp_size = temp;
}

template<typename T>
inline size_t CascadedSelector<T>::get_temp_size() const

{
  return max_temp_size;
}



template<typename T>
inline nvcompCascadedFormatOpts CascadedSelector<T>::select_config(
    void* d_workspace, size_t workspace_size, double* comp_ratio, cudaStream_t stream)
{

  NVCOMP_TYPE_ONE_SWITCH_RETURN(
      getnvcompType<T>(),
      internal_select,
      input_data,
      input_byte_len,
      opts.sample_size,
      opts.num_samples,
      comp_ratio,
      d_workspace,
      workspace_size,
      max_temp_size,
      opts.seed,
      stream);
}


template<typename T>
inline nvcompCascadedFormatOpts CascadedSelector<T>::select_config(
    void* d_workspace, size_t workspace_size, cudaStream_t stream)
{

  double comp_ratio;

  NVCOMP_TYPE_ONE_SWITCH_RETURN(
      getnvcompType<T>(),
      internal_select,
      input_data,
      input_byte_len,
      opts.sample_size,
      opts.num_samples,
      &comp_ratio,
      d_workspace,
      workspace_size,
      max_temp_size,
      opts.seed,
      stream);
}

} // namespace nvcomp
} // namespace nvcomp

nvcompError_t nvcompCascadedSelectorGetTempSize(
    size_t in_bytes,
    nvcompType_t in_type,
    nvcompCascadedSelectorOpts selector_opts,
    size_t* temp_bytes)
{
  // check that input is big enough to get all the samples
  if(in_bytes < (selector_opts.sample_size * selector_opts.num_samples)) {
    return nvcompErrorInvalidValue;
  }

  NVCOMP_TYPE_ONE_SWITCH(
      in_type, get_workspace_size_internal, selector_opts.num_samples, temp_bytes);

  *temp_bytes = roundUpTo(*temp_bytes, 8);
  return nvcompSuccess;
}


nvcompCascadedFormatOpts callSelectorSelectConfig(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    nvcompCascadedSelectorOpts opts,
    void* temp_ptr,
    size_t temp_bytes,
    double* est_ratio,
    cudaStream_t stream)
{

  size_t required_bytes;
  nvcompCascadedSelectorGetTempSize(in_bytes, in_type, opts, &required_bytes);

  NVCOMP_TYPE_ONE_SWITCH_RETURN(
      in_type,
      internal_select,
      in_ptr,
      in_bytes,
      opts.sample_size,
      opts.num_samples,
      est_ratio,
      temp_ptr,
      temp_bytes,
      required_bytes,
      opts.seed,
      stream);
}
  

nvcompError_t nvcompCascadedSelectorSelectConfig(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    nvcompCascadedSelectorOpts selector_opts,
    void* temp_ptr,
    size_t temp_bytes,
    nvcompCascadedFormatOpts* format_opts,
    double* est_ratio,
    cudaStream_t stream)
{

  *format_opts = callSelectorSelectConfig(in_ptr, in_bytes, in_type, selector_opts, temp_ptr, temp_bytes, est_ratio, stream);
      
  return nvcompSuccess;
}

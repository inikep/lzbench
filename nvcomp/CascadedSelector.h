/*
 * Copyright (c) Copyright 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "cascaded.hpp"

namespace nvcomp
{
namespace internal
{

/**
 *@brief Primary class for the Cascaded Selector used to determine the
 * best configuration to run cascaded compression on a given input.
 *
 *@param T the datatype of the input
 */
template <typename T>
class CascadedSelector
{
private:
  const void* input_data;
  size_t input_byte_len;
  size_t max_temp_size; // Internal variable used to store the temp buffer size
  nvcompCascadedSelectorOpts opts; // Sampling options

public:
  /**
   *@brief Create a new CascadedSelector for the given input data
   *
   *@param input The input data device pointer to select a cheme for
   *@param byte_len The number of bytes of input data
   *@param num_sample_ele The number of elements in a sample
   *@param num_sample The number of samples
   *@param type The type of input data
   */
  CascadedSelector(
      const void* input,
      size_t byte_len,
      nvcompCascadedSelectorOpts opts);

  // disable copying
  CascadedSelector(const CascadedSelector&) = delete;
  CascadedSelector& operator=(const CascadedSelector&) = delete;

  /*
   *@brief return the required size of workspace buffer in bytes
   */
  size_t get_temp_size() const;

  /*
   *@brief Select a CascadedSelector compression scheme that can provide the
   *best compression ratio and reports estimated compression ratio.
   *
   *@param d_worksapce The device potiner for the workspace
   *@param workspace_len The size of workspace buffer in bytes
   *@param comp_ratio The estimated compssion ratio using the bbest scheme (output)
   *@param stream The input stream to run the select function
   *@return Selected Cascaded options (RLE, Delta encoding, bit packing)
   */
  nvcompCascadedFormatOpts select_config(
      void* d_workspace,
      size_t workspace_len,
      double* comp_ratio,
      cudaStream_t stream);

  /*
   *@brief Select a CascadedSelector compression scheme that can provide the
   *best compression ratio - does NOT return estimated compression ratio.
   *
   *@param d_worksapce The device potiner for the workspace
   *@param workspace_len The size of workspace buffer in bytes
   *@param stream The input stream to run the select function
   *@return Selected Cascaded options (RLE, Delta encoding, bit packing)
   */
  nvcompCascadedFormatOpts select_config(
      void* d_workspace,
      size_t workspace_len,
      cudaStream_t stream);

};

}
}

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

#pragma once

#include "cascaded.h"
#include "common.h"
#include "CascadedCommon.h"

namespace nvcomp
{

class nvcompCascadedCompressionGPU
{
public:
  /**
   * @brief Compute the required temporary workspace on the GPU
   * required for the given compression configuration.
   *
   * @param in_ptr The pointer to the input data on the GPU.
   * @param in_bytes The length in bytes of the input data.
   * @param in_type The input data type.
   * @param opts The compression configuration to use.
   * @param temp_bytes The required minimum size of the temporary workspace
   * (output).
   */
  static void computeWorkspaceSize(
      const void* in_ptr,
      size_t in_bytes,
      nvcompType_t in_type,
      const nvcompCascadedFormatOpts* opts,
      size_t* temp_bytes);

  /**
   * @brief Generate the compression metadata information.
   *
   * @param in_ptr The pointer to the input data on the GPU.
   * @param in_bytes The length in bytes of the input data.
   * @param in_type The input data type.
   * @param opts The compression configuration to use.
   * @param temp_ptr The allocated temporary workspace on the GPU.
   * @param temp_bytes The size of the allocated temporary workspace on the
   * GPU.
   * @param out_bytes The required output space on the GPU in bytes to store
   * the compressed data (output).
   */
  static void generateOutputUpperBound(
      const void* in_ptr,
      size_t in_bytes,
      nvcompType_t in_type,
      const nvcompCascadedFormatOpts* opts,
      void* temp_ptr,
      size_t temp_bytes,
      size_t* out_bytes);

  /**
   * @brief Start the compression on the GPU. This will be asynchronous, if
   * `out_bytes` is pinned memory, otherwise this method block until all work is
   * done.
   *
   * @param in_ptr The pointer to the input data on the GPU.
   * @param in_bytes The length in bytes of the input data.
   * @param in_type The input data type.
   * @param opts The compression configuration to use.
   * @param temp_ptr The allocated temporary workspace on the GPU.
   * @param temp_bytes The size of the allocated temporary workspace on the
   * GPU.
   * @param metadata The metadata to use for compression and update.
   * @param out_ptr The location to write the output to.
   * @param out_bytes The size of the compressed data (output).
   * @param stream The stream to queue the compression on.
   */
  static void compressAsync(
      const void* in_ptr,
      size_t in_bytes,
      nvcompType_t in_type,
      const nvcompCascadedFormatOpts* opts,
      void* temp_ptr,
      const size_t temp_bytes,
      void* out_ptr,
      size_t* out_bytes,
      cudaStream_t stream);
};

} // namespace nvcomp

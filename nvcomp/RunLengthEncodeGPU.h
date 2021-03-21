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

#ifndef NVCOMP_RUNLENGTHENCODEGPU_H
#define NVCOMP_RUNLENGTHENCODEGPU_H

#include "cascaded.h"
#include "cuda_runtime.h"

#include <cstddef>

namespace nvcomp
{

class RunLengthEncodeGPU
{
public:
  /**
   * @brief Encode a set of data using run length encoding.
   *
   * @param workspace Temporary workspace to be used by this operation.
   * @param workspaceSize The size of the workspace in bytes.
   * @param valueType The type of outValues and in.
   * @param outValues The location to save the compressed values to.
   * @param countType The type to use for storing counts.
   * @param outCounts The location to save the counts to.
   * @param numOutDevice The size of outValue and outCounts stored on the
   * device (output).
   * @param in The location to read the input from.
   * @param num The number of values to encode.
   * @param stream The stream to execute the kernel on.
   */
  static void compress(
      void* workspace,
      size_t workspaceSize,
      nvcompType_t valueType,
      void* const outValues,
      nvcompType_t countType,
      void* const outCounts,
      size_t* numOutDevice,
      const void* in,
      const size_t num,
      cudaStream_t stream);

  /**
   * @brief Encode a set of data using run length encoding.
   *
   * @param workspace Temporary workspace to be used by this operation.
   * @param workspaceSize The size of the workspace in bytes.
   * @param valueType The type of outValues and in.
   * @param outValuesPtr The location to save the compressed values to on the
   * device.
   * @param countType The type to use for storing counts.
   * @param outCountsPtr The location to save the counts to on the device.
   * @param numOut The size of outValue and outCounts.
   * @param in The location to read the input from.
   * @param num The number of values to encode, to be .
   * @param maxNum The maximum number of values to encode.
   * @param stream The stream to execute the kernel on.
   */
  static void compressDownstream(
      void* workspace,
      size_t workspaceSize,
      nvcompType_t valueType,
      void** const outValuesPtr,
      nvcompType_t countType,
      void** const outCountsPtr,
      size_t* numOutDevice,
      const void* in,
      const size_t* numDevice,
      const size_t maxNum,
      cudaStream_t stream);

  /**
   * @brief Get the required size of the workspace in bytes.
   *
   * @param num The number of elements to compress.
   * @param valueType The type of elements.
   * @param countType The type to accumulate counts in.
   *
   * @return The size in bytes of the required workspace.
   */
  static size_t requiredWorkspaceSize(
      size_t num, nvcompType_t valueType, nvcompType_t countType);

private:
};

} // namespace nvcomp

#endif

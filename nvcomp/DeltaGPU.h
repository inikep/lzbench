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

#ifndef NVCOMP_DELTAGPU_H
#define NVCOMP_DELTAGPU_H

#include "cascaded.h"
#include "cuda_runtime.h"

#include <cstddef>

namespace nvcomp
{

class DeltaGPU
{
public:
  /**
   * @brief Encode a series of values using delta encoding. That is for the
   * series [a, b, c, ..., z], store [a, b-a, b-c, ..., z-y].
   *
   * @param workspace The temporary workspace to use.
   * @param workspaceSize The size of the temporary workspace to use.
   * @param valueType The type of value to delta encode.
   * @param outValuesPtr The location to write output stored on the device.
   * @param inValues The input location.
   * @param numDevice The actual number of values on the device.
   * @param maxNum The maximum number of values.
   * @param stream The stream to operate on.
   */
  static void compress(
      void* workspace,
      size_t workspaceSize,
      nvcompType_t valueType,
      void** const outValuesPtr,
      const void* inValues,
      const size_t* numDevice,
      const size_t maxNum,
      cudaStream_t stream);

  /**
   * @brief Get the required size of the workspace in bytes.
   *
   * @param num The number of elements to compress.
   * @param type The type of elements.
   *
   * @return The size in bytes of the required workspace.
   */
  static size_t requiredWorkspaceSize(size_t num, nvcompType_t type);
};

} // namespace nvcomp

#endif

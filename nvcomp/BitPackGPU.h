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
#include "cuda_runtime.h"

#include <cstddef>

namespace nvcomp
{

class BitPackGPU
{
public:
  /**
   * @brief Pack a series of values into uniform width bits. That is, reduce
   * each value to `numBits` bits, and compress together so that only
   * `numBits`*`num` total bits are used for output.
   *
   * @param workspace The workspace used internally by kernels.
   * @param workspaceSize The size of the workspace in bytes. If it is too
   * small to use, an exception will be thrown.
   * @param inType The type of the input elements.
   * @param out The output memory location on the GPU. It should equal to the
   * length of the input vector.
   * @param in The input memory location on the GPU.
   * @param numDevice The number of values to pack on the device.
   * @param maxNum The maximum number of values to pack.
   * @param minValueDevicePtr The minimum value found while packing (OUTPUT).
   * When unpacking, it must be added to the packed bits in order retrieve
   * the original number. The value must reside in device memory.
   * @param numBitsDevicePtr The number of bits used to represent the numbers
   * (OUTPUT). The value must reside in device memory
   * @param stream The stream to execute the kernel on.
   */
  static void compress(
      void* workspace,
      size_t workspaceSize,
      nvcompType_t inType,
      void* const* outPtr,
      const void* in,
      const size_t* numDevice,
      size_t maxNum,
      void* const* const minValueDevicePtr,
      unsigned char* const* const numBitsDevicePtr,
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

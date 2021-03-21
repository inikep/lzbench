/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NVCOMP_CUDAUTILS_H
#define NVCOMP_CUDAUTILS_H

#include "cuda_runtime.h"

#include <stdexcept>
#include <string>

namespace nvcomp
{

enum CopyDirection {
  HOST_TO_DEVICE = cudaMemcpyHostToDevice,
  DEVICE_TO_HOST = cudaMemcpyDeviceToHost,
  DEVICE_TO_DEVICE = cudaMemcpyDeviceToDevice
};

class CudaUtils
{
public:
  /**
   * @brief Convert cuda errors into exceptions. Will throw an exception
   * unless `err == cudaSuccess`.
   *
   * @param err The error.
   * @param msg The message to attach to the exception.
   */
  static void check(const cudaError_t err, const std::string& msg)
  {
    if (err != cudaSuccess) {
      std::string errorStr(
          "Encountered Cuda Error: " + std::to_string(err) + ": '"
          + std::string(cudaGetErrorString(err)) + "'");
      if (!msg.empty()) {
        errorStr += ": " + msg;
      }
      errorStr += ".";

      throw std::runtime_error(errorStr);
    }
  }

  static void sync(cudaStream_t stream)
  {
    check(cudaStreamSynchronize(stream), "Failed to sync with stream");
  }

  static void check_last_error(const std::string& msg = "")
  {
    check(cudaGetLastError(), msg);
  }

  /**
   * @brief Perform checked asynchronous memcpy.
   *
   * @tparam T The data type.
   * @param dst The destination address.
   * @param src The source address.
   * @param count The number of elements to copy.
   * @param kind The direction of the copy.
   * @param stream THe stream to operate on.
   */
  template <typename T>
  static void copy_async(
      T* const dst,
      const T* const src,
      const size_t count,
      const CopyDirection kind,
      cudaStream_t stream)
  {
    check(
        cudaMemcpyAsync(dst, src, sizeof(T) * count,
          static_cast<cudaMemcpyKind>(kind), stream),
        "CudaUtils::copy_async(dst, src, count, kind, stream)");
  }

  /**
   * @brief Perform a synchronous memcpy.
   *
   * @tparam T The data type.
   * @param dst The destination address.
   * @param src The source address.
   * @param count The number of elements to copy.
   * @param kind The direction of the copy.
   */
  template <typename T>
  static void copy(
      T* const dst,
      const T* const src,
      const size_t count,
      const CopyDirection kind)
  {
    check(
        cudaMemcpy(dst, src, sizeof(T) * count, static_cast<cudaMemcpyKind>(kind)),
        "CudaUtils::copy(dst, src, count, kind)");
  }


};

} // namespace nvcomp

#endif

/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "CudaUtils.h"

#include <sstream>
#include <stdexcept>

namespace nvcomp
{

namespace
{
std::string to_string(const void* const ptr)
{
  std::ostringstream oss;
  oss << ptr;
  return oss.str();
}
} // namespace

void CudaUtils::check(const cudaError_t err, const std::string& msg)
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

void CudaUtils::sync(cudaStream_t stream)
{
  check(cudaStreamSynchronize(stream), "Failed to sync with stream");
}

void CudaUtils::check_last_error(const std::string& msg)
{
  check(cudaGetLastError(), msg);
}

const void* CudaUtils::void_device_pointer(const void* const ptr)
{
  cudaPointerAttributes attr;
  check(
      cudaPointerGetAttributes(&attr, ptr),
      "Failed to get pointer "
      "attributes for pointer: "
          + to_string(ptr));

  if (!attr.devicePointer) {
    throw std::runtime_error(
        "Memory location is not accessible by the "
        "current GPU: "
        + to_string(ptr));
  }

  return attr.devicePointer;
}

bool CudaUtils::is_device_pointer(const void* const ptr)
{
  cudaPointerAttributes attr;

  cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

  if (err == cudaErrorInvalidValue) {
    int cuda_version;
    check(
        cudaRuntimeGetVersion(&cuda_version),
        "Failed to get runtime "
        "verison.");

    if (cuda_version < 11000) {
      // error is normal for non-device memory -- clear the error and return
      // false
      (void)cudaGetLastError();
      return false;
    }
  }

  // if we continue, make sure we successfully got pointer information
  check(
      err,
      "Failed to get pointer "
      "attributes for pointer: "
          + to_string(ptr));

  return attr.type == cudaMemoryTypeDevice;
}

void* CudaUtils::void_device_pointer(void* const ptr)
{
  cudaPointerAttributes attr;
  // we don't need to worry about the difference between cuda 10 and cuda 11
  // here, as if it's not a device pointer, we want throw an exception either
  // way.
  check(
      cudaPointerGetAttributes(&attr, ptr),
      "Failed to get pointer "
      "attributes for pointer: "
          + to_string(ptr));

  if (!attr.devicePointer) {
    throw std::runtime_error(
        "Memory location is not accessible by the "
        "current GPU: "
        + to_string(ptr));
  }

  return attr.devicePointer;
}

} // namespace nvcomp

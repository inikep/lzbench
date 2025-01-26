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

#include "tests/catch.hpp"
#include "CudaUtils.h"

#include "cuda_runtime.h"

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                     \
  {                                                                            \
    cudaError_t cudaStatus = call;                                             \
    if (cudaSuccess != cudaStatus) {                                           \
      fprintf(                                                                 \
          stderr,                                                              \
          "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s "   \
          "(%d).\n",                                                           \
          #call,                                                               \
          __LINE__,                                                            \
          __FILE__,                                                            \
          cudaGetErrorString(cudaStatus),                                      \
          cudaStatus);                                                         \
      abort();                                                                 \
    }                                                                          \
  }
#endif

using namespace nvcomp;

/******************************************************************************
 * UNIT TEST ******************************************************************
 *****************************************************************************/

TEST_CASE("IsDevicePointerTest", "[small]")
{
  // check a device pointer - true
  size_t* dev_ptr;
  CUDA_RT_CALL(cudaMalloc((void**)&dev_ptr, sizeof(*dev_ptr)));
  REQUIRE(CudaUtils::is_device_pointer(dev_ptr));
  CUDA_RT_CALL(cudaFree(dev_ptr));

  // check a uvm pointer - false
  size_t* managed_ptr;
  CUDA_RT_CALL(cudaMallocManaged((void**)&managed_ptr, sizeof(*managed_ptr)));
  REQUIRE(!CudaUtils::is_device_pointer(managed_ptr));
  CUDA_RT_CALL(cudaFree(managed_ptr));

  // check a pinned pointer - false
  size_t* pinned_ptr;
  CUDA_RT_CALL(cudaMallocHost((void**)&pinned_ptr, sizeof(*pinned_ptr)));
  REQUIRE(!CudaUtils::is_device_pointer(pinned_ptr));
  CUDA_RT_CALL(cudaFreeHost(pinned_ptr));

  // check an unregistered pointer - false
  size_t unregistered;
  REQUIRE(!CudaUtils::is_device_pointer(&unregistered));

  // check a null pointer - should be false
  REQUIRE(!CudaUtils::is_device_pointer(nullptr));
}

TEST_CASE("DevicePointerTest", "[small]")
{
  // check a device pointer - should be equal
  size_t* dev_ptr;
  CUDA_RT_CALL(cudaMalloc((void**)&dev_ptr, sizeof(*dev_ptr)));
  REQUIRE(CudaUtils::device_pointer(dev_ptr) == dev_ptr);
  CUDA_RT_CALL(cudaFree(dev_ptr));

  // check a uvm pointer - should succeed and return a device pointer
  size_t* managed_ptr;
  CUDA_RT_CALL(cudaMallocManaged((void**)&managed_ptr, sizeof(*managed_ptr)));
  size_t* managed_dev_ptr = CudaUtils::device_pointer(managed_ptr);
  CUDA_RT_CALL(cudaMemset(managed_dev_ptr, 0, sizeof(*managed_dev_ptr)));
  CUDA_RT_CALL(cudaFree(managed_ptr));

  // check a pinned pointer - should succeed and return a device pointer
  size_t* pinned_ptr;
  CUDA_RT_CALL(cudaMallocHost((void**)&pinned_ptr, sizeof(*pinned_ptr)));
  size_t* pinned_dev_ptr = CudaUtils::device_pointer(pinned_ptr);
  CUDA_RT_CALL(cudaMemset(pinned_dev_ptr, 0, sizeof(*pinned_dev_ptr)));
  CUDA_RT_CALL(cudaFreeHost(pinned_ptr));

  // check an unregistered pointer - should throw an exception
  try {
    size_t unregistered;
    CudaUtils::device_pointer(&unregistered);
    REQUIRE(false); // uncreachable
  } catch (const std::exception&) {
    // pass
  }

  // check a null pointer - should throw an exception
  try {
    CudaUtils::device_pointer(static_cast<void*>(nullptr));
  } catch (const std::exception&) {
    // pass
  }
}

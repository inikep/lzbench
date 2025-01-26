/*
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "nvcomp/ans.h"

#include "Check.h"
#include "CudaUtils.h"
#include "common.h"
#include "nvcomp.h"
#include "nvcomp.hpp"
#include "type_macros.h"

#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

#ifdef ENABLE_ANS
#include "ans.h"
#endif

using namespace nvcomp;

#define MAYBE_UNUSED(x) (void)(x)

nvcompStatus_t nvcompBatchedANSDecompressGetTempSize(
    const size_t num_chunks,
    const size_t max_uncompressed_chunk_size,
    size_t* const temp_bytes)
{
#ifdef ENABLE_ANS
  CHECK_NOT_NULL(temp_bytes);
  ans::decompressGetTempSize(num_chunks, max_uncompressed_chunk_size, temp_bytes);
  return nvcompSuccess;
#else
  (void)num_chunks;
  (void)max_uncompressed_chunk_size;
  (void)temp_bytes;
  std::cerr << "ERROR: nvcomp configured without GPU ANS support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return nvcompErrorNotSupported;
#endif
}

nvcompStatus_t nvcompBatchedANSDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    const size_t temp_bytes,
    void* const* device_uncompressed_ptr,
    nvcompStatus_t* device_statuses,
    cudaStream_t stream)
{
#ifdef ENABLE_ANS
  try {
    ans::decompressAsync(
      CudaUtils::device_pointer(device_compressed_ptrs),
      CudaUtils::device_pointer(device_compressed_bytes),
      CudaUtils::device_pointer(device_uncompressed_bytes),
      device_actual_uncompressed_bytes ? CudaUtils::device_pointer(device_actual_uncompressed_bytes) : nullptr,
      0, batch_size, device_temp_ptr, temp_bytes,
      CudaUtils::device_pointer(device_uncompressed_ptr),
      device_statuses ? CudaUtils::device_pointer(device_statuses) : nullptr,
      stream);
  } catch (const std::exception& e) {
     return Check::exception_to_error(e, "nvcompBatchedANSDecompressAsync()");
  }
  return nvcompSuccess;
#else
  (void)device_compressed_ptrs;
  (void)device_compressed_bytes;
  (void)device_uncompressed_bytes;
  (void)device_actual_uncompressed_bytes;
  (void)batch_size;
  (void)device_temp_ptr;
  (void)temp_bytes;
  (void)device_uncompressed_ptr;
  (void)device_statuses;
  (void)stream;
  std::cerr << "ERROR: nvcomp configured without GPU ANS support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return nvcompErrorNotSupported;
#endif
}

nvcompStatus_t nvcompBatchedANSCompressGetTempSize(
    size_t batch_size,
    size_t max_chunk_size,
    nvcompBatchedANSOpts_t /* format_opts */,
    size_t* temp_bytes)
{
#ifdef ENABLE_ANS
  CHECK_NOT_NULL(temp_bytes);
  ans::compressGetTempSize(batch_size, max_chunk_size, temp_bytes);
  return nvcompSuccess;
#else
  (void)batch_size;
  (void)max_chunk_size;
  (void)temp_bytes;
  std::cerr << "ERROR: nvcomp configured without GPU ANS support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return nvcompErrorNotSupported;
#endif
}

nvcompStatus_t nvcompBatchedANSCompressGetMaxOutputChunkSize(
    size_t max_chunk_size,
    nvcompBatchedANSOpts_t /* format_opts */,
    size_t* max_compressed_size)
{
#ifdef ENABLE_ANS
  CHECK_NOT_NULL(max_compressed_size);
  ans::compressGetMaxOutputChunkSize(max_chunk_size, max_compressed_size);
  return nvcompSuccess;
#else
  (void)max_chunk_size;
  (void)max_compressed_size;
  std::cerr << "ERROR: nvcomp configured without GPU ANS support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return nvcompErrorNotSupported;
#endif
}

nvcompStatus_t nvcompBatchedANSCompressAsync(
    const void* const* device_uncompressed_ptr,
    const size_t* device_uncompressed_bytes,
    size_t max_uncompressed_chunk_bytes,
    size_t batch_size,
    void* device_temp_ptr,
    size_t temp_bytes,
    void* const* device_compressed_ptr,
    size_t* device_compressed_bytes,
    nvcompBatchedANSOpts_t format_opts,
    cudaStream_t stream)
{
#ifdef ENABLE_ANS
  assert(format_opts.type == nvcompANSType_t::nvcomp_rANS);
  MAYBE_UNUSED(format_opts);
  ans::ansType_t ans_type = ans::ansType_t::rANS;

  try {
    ans::compressAsync(
        ans_type,
        CudaUtils::device_pointer(device_uncompressed_ptr),
        CudaUtils::device_pointer(device_uncompressed_bytes),
        max_uncompressed_chunk_bytes,
        batch_size,
        device_temp_ptr,
        temp_bytes,
        CudaUtils::device_pointer(device_compressed_ptr),
        CudaUtils::device_pointer(device_compressed_bytes),
        stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompBatchedANSCompressAsync()");
  }
  return nvcompSuccess;
#else
  (void)device_uncompressed_ptr;
  (void)device_uncompressed_bytes;
  (void)max_uncompressed_chunk_bytes;
  (void)batch_size;
  (void)device_temp_ptr;
  (void)temp_bytes;
  (void)device_compressed_ptr;
  (void)device_compressed_bytes;
  (void)format_opts;
  (void)stream;
  std::cerr << "ERROR: nvcomp configured without GPU ANS support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return nvcompErrorNotSupported;
#endif
}

nvcompStatus_t nvcompBatchedANSGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* /* device_compressed_bytes */,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream) {
#ifdef ENABLE_ANS
  ans::getDecompressSizeAsync(
      device_compressed_ptrs,
      device_uncompressed_bytes,
      batch_size,
      stream);
  return nvcompSuccess;
#else
  (void)device_compressed_ptrs;
  (void)device_uncompressed_bytes;
  (void)batch_size;
  (void)stream;
  std::cerr << "ERROR: nvcomp configured without GPU ANS support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return nvcompErrorNotSupported;
#endif
}

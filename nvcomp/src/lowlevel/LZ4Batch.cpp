/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "nvcomp/lz4.h"

#include "Check.h"
#include "CudaUtils.h"
#include "LZ4CompressionKernels.h"
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

using namespace nvcomp;
using namespace nvcomp::lowlevel;

nvcompStatus_t nvcompBatchedLZ4DecompressGetTempSize(
    const size_t num_chunks,
    const size_t max_uncompressed_chunk_size,
    size_t* const temp_bytes)
{
  CHECK_NOT_NULL(temp_bytes);

  try {
    *temp_bytes
        = lz4DecompressComputeTempSize(num_chunks, max_uncompressed_chunk_size);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedLZ4DecompressGetTempSize()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedLZ4DecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    nvcompStatus_t* device_statuses,
    cudaStream_t stream)
{
  // NOTE: if we start using `max_uncompressed_chunk_bytes`, we need to check
  // to make sure it is not zero, as we have notified users to supply zero if
  // they are not finding the maximum size.

  try {
    lz4BatchDecompress(
        CudaUtils::device_pointer(
            reinterpret_cast<const uint8_t* const*>(device_compressed_ptrs)),
        CudaUtils::device_pointer(device_compressed_bytes),
        CudaUtils::device_pointer(device_uncompressed_bytes),
        batch_size,
        CudaUtils::device_pointer(device_temp_ptr),
        temp_bytes,
        CudaUtils::device_pointer(
            reinterpret_cast<uint8_t* const*>(device_uncompressed_ptrs)),
        device_actual_uncompressed_bytes ? CudaUtils::device_pointer(device_actual_uncompressed_bytes) : nullptr,
        device_statuses ? CudaUtils::device_pointer(device_statuses) : nullptr,
        stream);

  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompBatchedLZ4DecompressAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedLZ4GetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream)
{
  CHECK_NOT_NULL(device_compressed_ptrs);
  CHECK_NOT_NULL(device_compressed_bytes);
  CHECK_NOT_NULL(device_uncompressed_bytes);

  try {
    lz4BatchGetDecompressSizes(
        CudaUtils::device_pointer(
            reinterpret_cast<const uint8_t* const*>(device_compressed_ptrs)),
        CudaUtils::device_pointer(device_compressed_bytes),
        CudaUtils::device_pointer(device_uncompressed_bytes),
        batch_size,
        stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedLZ4GetDecompressSizeAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedLZ4CompressGetTempSize(
    const size_t batch_size,
    const size_t max_chunk_size,
    const nvcompBatchedLZ4Opts_t /* format_opts */,
    size_t* const temp_bytes)
{
  CHECK_NOT_NULL(temp_bytes);

  try {
    *temp_bytes = lz4BatchCompressComputeTempSize(max_chunk_size, batch_size);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedLZ4CompressGetTempSize()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
    const size_t max_chunk_size,
    const nvcompBatchedLZ4Opts_t /* format_opts */,
    size_t* const max_compressed_size)
{
  CHECK_NOT_NULL(max_compressed_size);

  try {
    *max_compressed_size = lz4ComputeMaxSize(max_chunk_size);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedLZ4CompressGetOutputSize()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedLZ4CompressAsync(
    const void* const* const device_uncompressed_ptrs,
    const size_t* const device_uncompressed_bytes,
    const size_t max_uncompressed_chunk_size,
    const size_t batch_size,
    void* const device_temp_ptr,
    const size_t temp_bytes,
    void* const* const device_compressed_ptrs,
    size_t* const device_compressed_bytes,
    const nvcompBatchedLZ4Opts_t format_opts,
    cudaStream_t stream)
{
  // NOTE: if we start using `max_uncompressed_chunk_bytes`, we need to check
  // to make sure it is not zero, as we have notified users to supply zero if
  // they are not finding the maximum size.

  try {
    lz4BatchCompress(
        CudaUtils::device_pointer(
            reinterpret_cast<const uint8_t* const*>(device_uncompressed_ptrs)),
        CudaUtils::device_pointer(device_uncompressed_bytes),
        max_uncompressed_chunk_size,
        batch_size,
        device_temp_ptr,
        temp_bytes,
        CudaUtils::device_pointer(
            reinterpret_cast<uint8_t* const*>(device_compressed_ptrs)),
        CudaUtils::device_pointer(device_compressed_bytes),
        format_opts.data_type,
        stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompBatchedLZ4CompressAsync()");
  }

  return nvcompSuccess;
}

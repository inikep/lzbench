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

#include "lz4.h"

#include "Check.h"
#include "CudaUtils.h"
#include "LZ4BatchCompressor.h"
#include "LZ4CompressionKernels.h"
#include "LZ4Metadata.h"
#include "LZ4MetadataOnGPU.h"
#include "MutableLZ4MetadataOnGPU.h"
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

int LZ4IsData(const void* const in_ptr, size_t in_bytes, cudaStream_t stream)
{
  // Need at least 2 size_t variables to be valid.
  if(in_ptr == NULL || in_bytes < sizeof(size_t)) {
    return false;
  }
  size_t header_val;
  CudaUtils::copy_async(
      &header_val,
      static_cast<const size_t*>(in_ptr),
      1,
      DEVICE_TO_HOST,
      stream);
  CudaUtils::sync(stream);
  return (header_val == LZ4_FLAG); 
}

nvcompError_t nvcompLZ4DecompressGetMetadata(
    const void* const in_ptr,
    const size_t in_bytes,
    void** const metadata_ptr,
    cudaStream_t stream)
{
  return API_WRAPPER(
      nvcompBatchedLZ4DecompressGetMetadata(
          (const void**)&in_ptr, &in_bytes, 1, metadata_ptr, stream),
      "nvcompLZ4DecompressGetMetadata()");
}

void nvcompLZ4DecompressDestroyMetadata(void* const metadata_ptr)
{
  nvcompBatchedLZ4DecompressDestroyMetadata(metadata_ptr);
}

nvcompError_t
nvcompLZ4DecompressGetTempSize(const void* metadata_ptr, size_t* temp_bytes)
{
  return API_WRAPPER(
      nvcompBatchedLZ4DecompressGetTempSize(metadata_ptr, temp_bytes),
      "nvcompLZ4DecompressGetTempSize()");
}

nvcompError_t
nvcompLZ4DecompressGetOutputSize(const void* metadata_ptr, size_t* output_bytes)
{
  return API_WRAPPER(
      nvcompBatchedLZ4DecompressGetOutputSize(metadata_ptr, 1, output_bytes),
      "nvcompLZ4DecompressGetOutputSize()");
}

nvcompError_t nvcompLZ4DecompressAsync(
    const void* const in_ptr,
    const size_t in_bytes,
    void* const temp_ptr,
    const size_t temp_bytes,
    const void* const metadata_ptr,
    void* const out_ptr,
    const size_t out_bytes,
    cudaStream_t stream)
{
  return API_WRAPPER(
      nvcompBatchedLZ4DecompressAsync(
          &in_ptr,
          &in_bytes,
          1,
          temp_ptr,
          temp_bytes,
          (const void* const*)metadata_ptr,
          &out_ptr,
          &out_bytes,
          stream),
      "nvcompLZ4DecompressAsync()");
}

nvcompError_t nvcompLZ4CompressGetTempSize(
    const void* in_ptr,
    const size_t in_bytes,
    nvcompType_t /*in_type*/,
    const nvcompLZ4FormatOpts* const format_opts,
    size_t* const temp_bytes)
{
  return API_WRAPPER(
      nvcompBatchedLZ4CompressGetTempSize(
          &in_ptr, &in_bytes, 1, format_opts, temp_bytes),
      "nvcompLZ4CompressGetTempSize()");
}

nvcompError_t nvcompLZ4CompressGetOutputSize(
    const void* const in_ptr,
    const size_t in_bytes,
    const nvcompType_t /*in_type*/,
    const nvcompLZ4FormatOpts* format_opts,
    void* const temp_ptr,
    const size_t temp_bytes,
    size_t* const out_bytes,
    const int exact_out_bytes)
{
  if (exact_out_bytes) {
    std::cerr
        << "LZ4CompressGetOutputSize(): Exact output bytes is unimplemented at "
           "this time."
        << std::endl;
    return nvcompErrorInvalidValue;
  }

  return API_WRAPPER(
      nvcompBatchedLZ4CompressGetOutputSize(
          &in_ptr, &in_bytes, 1, format_opts, temp_ptr, temp_bytes, out_bytes),
      "LZ4CompressGetOutputSize()");
}

nvcompError_t nvcompLZ4CompressAsync(
    const void* in_ptr,
    const size_t in_bytes,
    const nvcompType_t /* in_type */,
    const nvcompLZ4FormatOpts* format_opts,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  return API_WRAPPER(
      nvcompBatchedLZ4CompressAsync(
          &in_ptr,
          &in_bytes,
          1,
          format_opts,
          temp_ptr,
          temp_bytes,
          &out_ptr,
          out_bytes,
          stream),
      "nvcompLZ4CompressAsync");
}

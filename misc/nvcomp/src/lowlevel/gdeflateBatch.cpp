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

#include "nvcomp/gdeflate.h"

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

#ifdef ENABLE_GDEFLATE
#include "gdeflate.h"
#include "gdeflateKernels.h"
#endif

using namespace nvcomp;

#ifdef ENABLE_GDEFLATE
gdeflate::gdeflate_compression_algo getGdeflateEnumFromFormatOpts(nvcompBatchedGdeflateOpts_t format_opts) {
  gdeflate::gdeflate_compression_algo algo;
  switch(format_opts.algo) {
    case (0) :
      algo = gdeflate::HIGH_THROUGHPUT;
      break;
    case(1) :
      algo = gdeflate::HIGH_COMPRESSION;
      break;
    case(2) :
      algo = gdeflate::ENTROPY_ONLY;
      break;
    default :
      throw std::invalid_argument("Invalid format_opts.algo value (not 0, 1 or 2)");
  }
  return algo;
}
#endif

nvcompStatus_t nvcompBatchedGdeflateDecompressGetTempSize(
    const size_t num_chunks,
    const size_t max_uncompressed_chunk_size,
    size_t* const temp_bytes)
{
#ifdef ENABLE_GDEFLATE
  CHECK_NOT_NULL(temp_bytes);

  try {
    gdeflate::decompressGetTempSize(num_chunks, max_uncompressed_chunk_size, temp_bytes);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedGdeflateDecompressGetTempSize()");
  }

  return nvcompSuccess;
#else
  (void)num_chunks;
  (void)max_uncompressed_chunk_size;
  (void)temp_bytes;
  std::cerr << "ERROR: nvcomp configured without gdeflate support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return nvcompErrorNotSupported;
#endif
}

nvcompStatus_t nvcompBatchedGdeflateDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    nvcompStatus_t* device_status_ptrs,
    cudaStream_t stream)
{
#ifdef ENABLE_GDEFLATE
  // NOTE: if we start using `max_uncompressed_chunk_bytes`, we need to check
  // to make sure it is not zero, as we have notified users to supply zero if
  // they are not finding the maximum size.

  try {
    // Use device_status_ptrs as temp space to store gdeflate statuses
    static_assert(sizeof(nvcompStatus_t) == sizeof(gdeflate::gdeflateStatus_t),
        "Mismatched sizes of nvcompStatus_t and gdeflateStatus_t");
    auto device_statuses = reinterpret_cast<gdeflate::gdeflateStatus_t*>(device_status_ptrs);

    // Run the decompression kernel
    gdeflate::decompressAsync(device_compressed_ptrs, device_compressed_bytes,
        device_uncompressed_bytes, device_actual_uncompressed_bytes,
        0, batch_size, device_temp_ptr, temp_bytes,
        device_uncompressed_ptrs, device_statuses, stream);

    // Launch a kernel to convert the output statuses
    if(device_status_ptrs) convertGdeflateOutputStatuses(device_status_ptrs, batch_size, stream);

  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompBatchedGdeflateDecompressAsync()");
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
  (void)device_uncompressed_ptrs;
  (void)device_status_ptrs;
  (void)stream;
  std::cerr << "ERROR: nvcomp configured without gdeflate support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return nvcompErrorNotSupported;
#endif
}

nvcompStatus_t nvcompBatchedGdeflateGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream) {
#ifdef ENABLE_GDEFLATE
  try {
    gdeflate::getDecompressSizeAsync(device_compressed_ptrs, device_compressed_bytes,
        device_uncompressed_bytes, batch_size, stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompBatchedGdeflateDecompressAsync()");
  }

  return nvcompSuccess;
#else
  (void)device_compressed_ptrs;
  (void)device_compressed_bytes;
  (void)device_uncompressed_bytes;
  (void)batch_size;
  (void)stream;
  std::cerr << "ERROR: nvcomp configured without gdeflate support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return nvcompErrorNotSupported;
#endif
}

nvcompStatus_t nvcompBatchedGdeflateCompressGetTempSize(
    const size_t batch_size,
    const size_t max_chunk_size,
    nvcompBatchedGdeflateOpts_t format_opts,
    size_t* const temp_bytes)
{
#ifdef ENABLE_GDEFLATE
  CHECK_NOT_NULL(temp_bytes);


  try {
    gdeflate::gdeflate_compression_algo algo = getGdeflateEnumFromFormatOpts(format_opts);
    gdeflate::compressGetTempSize(batch_size, max_chunk_size, temp_bytes, algo);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedGdeflateCompressGetTempSize()");
  }

  return nvcompSuccess;
#else
  (void)batch_size;
  (void)max_chunk_size;
  (void)format_opts;
  (void)temp_bytes;
  std::cerr << "ERROR: nvcomp configured without gdeflate support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return nvcompErrorNotSupported;
#endif
}

nvcompStatus_t nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
    size_t max_chunk_size,
    nvcompBatchedGdeflateOpts_t /* format_opts */,
    size_t* max_compressed_size)
{
#ifdef ENABLE_GDEFLATE
  CHECK_NOT_NULL(max_compressed_size);

  try {
    gdeflate::compressGetMaxOutputChunkSize(max_chunk_size, max_compressed_size);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedGdeflateCompressGetOutputSize()");
  }

  return nvcompSuccess;
#else
  (void)max_chunk_size;
  (void)max_compressed_size;
  std::cerr << "ERROR: nvcomp configured without gdeflate support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return nvcompErrorNotSupported;
#endif
}

nvcompStatus_t nvcompBatchedGdeflateCompressAsync(
    const void* const* const device_in_ptrs,
    const size_t* const device_in_bytes,
    const size_t max_uncompressed_chunk_size,
    const size_t batch_size,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const* const device_out_ptrs,
    size_t* const device_out_bytes,
    nvcompBatchedGdeflateOpts_t format_opts,
    cudaStream_t stream)
{
#ifdef ENABLE_GDEFLATE
  try {
    gdeflate::gdeflate_compression_algo algo = getGdeflateEnumFromFormatOpts(format_opts);
    gdeflate::compressAsync(device_in_ptrs, device_in_bytes, max_uncompressed_chunk_size,
        batch_size, temp_ptr, temp_bytes, device_out_ptrs, device_out_bytes, algo, stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompBatchedGdeflateCompressAsync()");
  }

  return nvcompSuccess;
#else
  (void)device_in_ptrs;
  (void)device_in_bytes;
  (void)max_uncompressed_chunk_size;
  (void)batch_size;
  (void)temp_ptr;
  (void)temp_bytes;
  (void)device_out_ptrs;
  (void)device_out_bytes;
  (void)format_opts;
  (void)stream;
  std::cerr << "ERROR: nvcomp configured without gdeflate support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return nvcompErrorNotSupported;
#endif
}

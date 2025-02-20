#pragma once

/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <assert.h>

#include "nvcomp/bitcomp.hpp"
#include "CudaUtils.h"
#include "nvcomp_common_deps/hlif_shared_types.hpp"
#include "highlevel/ManagerBase.hpp"

namespace nvcomp {

struct BitcompSingleStreamManager : ManagerBase<BitcompFormatSpecHeader> {
private:
  BitcompFormatSpecHeader* format_spec;

public:
  BitcompSingleStreamManager(nvcompType_t data_type, int bitcomp_algo = 0, cudaStream_t user_stream = 0, const int device_id = 0)
    : ManagerBase(user_stream, device_id),      
      format_spec()
  {
    CudaUtils::check(cudaHostAlloc(&format_spec, sizeof(BitcompFormatSpecHeader), cudaHostAllocDefault));
    format_spec->data_type = data_type;
    format_spec->algo = bitcomp_algo;
    int  major;
    CudaUtils::check(cudaDeviceGetAttribute (&major, cudaDevAttrComputeCapabilityMajor, device_id));
    if (major < 7)
      throw NVCompException(nvcompErrorNotSupported, "Bitcomp requires GPU architectures >= 70");

    finish_init();
  }

  virtual ~BitcompSingleStreamManager() 
  {
    CudaUtils::check(cudaFreeHost(format_spec));
  }

  BitcompSingleStreamManager(const BitcompSingleStreamManager&) = delete;
  BitcompSingleStreamManager& operator=(const BitcompSingleStreamManager&) = delete;

  /**
   * @brief Required helper that actually does the compression 
   * 
   * @param common_header header filled in by this routine (GPU accessible)
   * @param decomp_buffer The uncompressed input data (GPU accessible)
   * @param decomp_buffer_size The length of the uncompressed input data
   * @param comp_buffer The location to output the compressed data to (GPU accessible).
   * @param comp_config Resulted from configure_compression given this decomp_buffer_size.
   * 
   */
  void do_compress(
      CommonHeader* common_header,
      const uint8_t* decomp_buffer, 
      uint8_t* comp_buffer,
      const CompressionConfig& comp_config) final override;


  /**
   * @brief Required helper that actually does the decompression 
   *
   * @param decomp_buffer The location to output the decompressed data to (GPU accessible).
   * @param comp_buffer The compressed input data (GPU accessible).
   * @param decomp_config Resulted from configure_decompression given this decomp_buffer_size.
   */
  void do_decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& config) final override;

  /**
   * @brief Optionally does additional decompression configuration 
   */
  void do_configure_decompression(DecompressionConfig&, const CommonHeader*) final override
  {
  }

  /**
   * @brief Optionally does additional decompression configuration
   */
  void do_configure_decompression(DecompressionConfig&, const CompressionConfig&) final override
  {
  }

  /**
   * @brief Computes the required scratch buffer size 
   */
  size_t compute_scratch_buffer_size() final override
  {
    return 0;
  }

  /**
   * @brief Computes the maximum compressed output size for a given
   * uncompressed buffer.
   */
  size_t calculate_max_compressed_output_size(CompressionConfig& comp_config) final override;

  /**
   * @brief Retrieves a CPU-accessible pointer to the FormatSpecHeader
   */
  BitcompFormatSpecHeader* get_format_header() final override
  {
    return format_spec;
  }
};

// BitcompManager implementation

BitcompManager::BitcompManager(
    nvcompType_t data_type,
    int bitcomp_algo,
    cudaStream_t user_stream,
    const int device_id)
{
#ifdef ENABLE_BITCOMP
  impl = std::make_unique<BitcompSingleStreamManager>(
      data_type, bitcomp_algo, user_stream, device_id);
#else
  throw NVCompException(nvcompErrorNotSupported, "Bitcomp support not available in this build.");
#endif
}

BitcompManager::~BitcompManager() 
{}

} // namespace nvcomp

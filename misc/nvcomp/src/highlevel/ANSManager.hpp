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

#pragma once

#include "nvcomp/ans.h"
#include "nvcomp/ans.hpp"
#include "BatchManager.hpp"
#ifdef ENABLE_ANS
#include "ans_hlif.h"
#endif

namespace nvcomp {

#ifdef ENABLE_ANS

struct ANSBatchManager : BatchManager<ANSFormatSpecHeader>
{
private:
  ANSFormatSpecHeader* format_spec = nullptr;

public:
  ANSBatchManager(
      size_t uncomp_chunk_size,
      cudaStream_t user_stream = 0,
      const int device_id = 0)
   : BatchManager(uncomp_chunk_size, user_stream, device_id)
  {
    CudaUtils::check(cudaHostAlloc(
        &format_spec, sizeof(ANSFormatSpecHeader), cudaHostAllocDefault));
    finish_init();
  }

  virtual ~ANSBatchManager()
  {
    CudaUtils::check(cudaFreeHost(format_spec));
  }

  ANSBatchManager(const ANSBatchManager&) = delete;
  ANSBatchManager& operator=(const ANSBatchManager&) = delete;

  size_t compute_max_compressed_chunk_size() final override
  {
    size_t max_comp_chunk_size;
    nvcompBatchedANSCompressGetMaxOutputChunkSize(
        get_uncomp_chunk_size(),
        nvcompBatchedANSDefaultOpts,
        &max_comp_chunk_size);
    return max_comp_chunk_size;
  }

  uint32_t compute_compression_max_block_occupancy() final override
  {
    return ans::hlif::getBatchedCompMaxBlockOccupancy(device_id);
  }

  uint32_t compute_decompression_max_block_occupancy() final override
  {
    return ans::hlif::getBatchedDecompMaxBlockOccupancy(device_id);
  }

  ANSFormatSpecHeader* get_format_header() final override
  {
    return format_spec;
  }

  void do_batch_compress(const CompressArgs& compress_args) final override
  {
    ans::hlif::batchCompress(compress_args, get_max_comp_ctas(), user_stream);
  }

  void do_batch_decompress(
      const uint8_t* comp_data_buffer,
      uint8_t* decomp_buffer,
      const uint32_t num_chunks,
      const size_t* comp_chunk_offsets,
      const size_t* comp_chunk_sizes,
      nvcompStatus_t* output_status) final override
  {
    ans::hlif::batchDecompress(
        comp_data_buffer,
        decomp_buffer,
        get_uncomp_chunk_size(),
        ix_chunk,
        num_chunks,
        comp_chunk_offsets,
        comp_chunk_sizes,
        get_max_decomp_ctas(),
        user_stream,
        output_status);
  }

private:
  size_t compute_scratch_buffer_size() final override
  {
    auto chunks_per_cta = ans::hlif::getBatchedCompChunksPerCTA();
    auto chunk_scratch_size = get_max_comp_chunk_size() + ans::hlif::getChunkTmpSize();
    return get_max_comp_ctas() * chunks_per_cta * chunk_scratch_size;
  }
};
#endif

// ANSManager implementation

ANSManager::ANSManager(
    size_t uncomp_chunk_size,
    cudaStream_t user_stream,
    const int device_id)
{
#ifdef ENABLE_ANS
  impl = std::make_unique<ANSBatchManager>(uncomp_chunk_size,
                                           user_stream,
                                           device_id);
#else
  (void)uncomp_chunk_size;
  (void)user_stream;
  (void)device_id;
  throw std::runtime_error("nvcomp configured without ANS support. Please check the README for configuration instructions");
#endif
}

ANSManager::~ANSManager()
{}

} // namespace nvcomp

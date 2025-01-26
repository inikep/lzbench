#pragma once

/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <memory>

#include "nvcomp/cascaded.hpp"
#include "Check.h"
#include "CudaUtils.h"
#include "common.h"
#include "nvcomp/cascaded.h"
#include "nvcomp_common_deps/hlif_shared_types.hpp"
#include "highlevel/CascadedHlifKernels.h"
#include "highlevel/BatchManager.hpp"

namespace nvcomp {

struct CascadedBatchManager : BatchManager<CascadedFormatSpecHeader> {
private:
  CascadedFormatSpecHeader* format_spec;

public:
  CascadedBatchManager(
      const nvcompBatchedCascadedOpts_t& options = nvcompBatchedCascadedDefaultOpts,
      cudaStream_t user_stream = 0,
      int device_id = 0) :
      BatchManager(options.chunk_size, user_stream, device_id),
      format_spec(nullptr)
  {
    CudaUtils::check(cudaHostAlloc(
        &format_spec, sizeof(CascadedFormatSpecHeader), cudaHostAllocDefault));
    format_spec->options = options;

    finish_init();
  }

  virtual ~CascadedBatchManager()
  {
    CudaUtils::check(cudaFreeHost(format_spec));
  }

  CascadedBatchManager(const CascadedBatchManager&) = delete;
  CascadedBatchManager& operator=(const CascadedBatchManager&) = delete;

  size_t compute_max_compressed_chunk_size() final override
  {
    size_t max_comp_chunk_size;
    nvcompBatchedCascadedCompressGetMaxOutputChunkSize(
        get_uncomp_chunk_size(),
        nvcompBatchedCascadedDefaultOpts,
        &max_comp_chunk_size);
    return max_comp_chunk_size;
  }

  uint32_t compute_compression_max_block_occupancy() final override
  {
    return cascadedHlifCompMaxBlockOccupancy(
        device_id, format_spec->options.type);
  }

  uint32_t compute_decompression_max_block_occupancy() final override
  {
    return cascadedHlifDecompMaxBlockOccupancy(
        device_id, format_spec->options.type);
  }

  CascadedFormatSpecHeader* get_format_header() final override
  {
    return format_spec;
  }

  void do_batch_compress(const CompressArgs& compress_args) final override
  {
    cascadedHlifBatchCompress(
        compress_args,
        get_max_comp_ctas(),
        user_stream,
        &(format_spec->options));
  }

  void do_batch_decompress(
      const uint8_t* comp_data_buffer,
      uint8_t* decomp_buffer,
      const uint32_t num_chunks,
      const size_t* comp_chunk_offsets,
      const size_t* comp_chunk_sizes,
      nvcompStatus_t* output_status) final override
  {
    cascadedHlifBatchDecompress(
        comp_data_buffer,
        decomp_buffer,
        get_uncomp_chunk_size(),
        ix_chunk,
        num_chunks,
        comp_chunk_offsets,
        comp_chunk_sizes,
        get_max_decomp_ctas(),
        user_stream,
        output_status,
        &(format_spec->options));
  }
};


// CascadedManager implementation

CascadedManager::CascadedManager(
    const nvcompBatchedCascadedOpts_t& options,
    cudaStream_t user_stream,
    int device_id)
{
  impl = std::make_unique<CascadedBatchManager>(
      options, user_stream, device_id);
}

CascadedManager::~CascadedManager()
{
}

} // namespace nvcomp

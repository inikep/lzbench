/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVCOMP_MUTABLEBATCHEDLZ4METADATAONGPU_H
#define NVCOMP_MUTABLEBATCHEDLZ4METADATAONGPU_H

#include "LZ4Metadata.h"

#include "cuda_runtime.h"

#include <cstddef>
#include <vector>

namespace nvcomp
{

class MutableBatchedLZ4MetadataOnGPU
{
public:
  MutableBatchedLZ4MetadataOnGPU(
      void* const* out_ptrs,
      const size_t* const max_out_sizes,
      size_t batch_size);

  MutableBatchedLZ4MetadataOnGPU(const MutableBatchedLZ4MetadataOnGPU& other)
      = delete;
  MutableBatchedLZ4MetadataOnGPU&
  operator=(const MutableBatchedLZ4MetadataOnGPU& other)
      = delete;

  void copyToGPU(
      const std::vector<LZ4Metadata>& metadata,
      void* temp_space,
      size_t temp_size,
      size_t* serialized_sizes,
      cudaStream_t stream);

  size_t* compressed_prefix_ptr(const size_t index);

private:
  std::vector<uint8_t*> m_buffer;
  size_t m_batch_size;
  void* const* m_out_ptrs;
  const size_t* m_max_out_sizes;
};

} // namespace nvcomp

#endif

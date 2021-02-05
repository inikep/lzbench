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

#ifndef MUTABLELZ4LZ4METADATAONGPU_H
#define MUTABLELZ4LZ4METADATAONGPU_H

#include "LZ4Metadata.h"
#include "LZ4MetadataOnGPU.h"

namespace nvcomp
{

class MutableLZ4MetadataOnGPU : public LZ4MetadataOnGPU
{
public:
  /**
   * @brief Create a new serialized metadata object. This is used either to
   * copy metadata to the GPU, or copy it to the CPU from the GPU.
   *
   * @param ptr The memory location on the GPU that will be used for the
   * serialized metadata.
   * @param maxSize The maximum size the metadata can occupy (the size of the
   * allocation usually).
   */
  MutableLZ4MetadataOnGPU(void* ptr, size_t maxSize);

  MutableLZ4MetadataOnGPU(const MutableLZ4MetadataOnGPU& other);
  MutableLZ4MetadataOnGPU& operator=(const MutableLZ4MetadataOnGPU& other);

  /**
   * @brief Copy and serialize the given metadata object to to the GPU
   * asynchronously.
   *
   * @param metadata The metadata object to serialize.
   * @param stream The stream to asynchronously execute on.
   */
  void copyToGPU(const LZ4Metadata& metadata, cudaStream_t stream);

  size_t* compressed_prefix_ptr();

private:
  void* m_mutable_ptr;
};

} // namespace nvcomp

#endif

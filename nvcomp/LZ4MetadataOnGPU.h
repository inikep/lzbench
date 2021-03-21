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

#ifndef LZ4METADATAONGPU_H
#define LZ4METADATAONGPU_H

#include "LZ4Metadata.h"

namespace nvcomp
{

class LZ4MetadataOnGPU
{
public:
  /**
   * @brief Get the serialized size a metadata object would occupy on the GPU.
   *
   * @param metadata The metadata.
   *
   * @return The size in bytes that would be occupied.
   */
  static size_t getSerializedSizeOf(const LZ4Metadata& metadata);

  /**
   * @brief Get the offset from the start of the metadata, to where the first
   * chunk of compressed data is stored.
   *
   * @param metadata The metadata.
   *
   * @return The offset in bytes.
   */
  static size_t getCompressedDataOffset(const LZ4Metadata& metadata);

  /**
   * @brief Create a new serialized metadata object. This is used either to
   * copy metadata to the GPU, or copy it to the CPU from the GPU.
   *
   * @param ptr The memory location on the GPU that will be used for the
   * serialized metadata.
   * @param maxSize The maximum size the metadata can occupy (the size of the
   * allocation usually).
   */
  LZ4MetadataOnGPU(const void* ptr, size_t maxSize);

  LZ4MetadataOnGPU(const LZ4MetadataOnGPU& other);
  LZ4MetadataOnGPU& operator=(const LZ4MetadataOnGPU& other);

  virtual ~LZ4MetadataOnGPU() = default;

  /**
   * @brief Get the size of this metadata on the GPU.
   *
   * @return The size in bytes.
   */
  size_t getSerializedSize() const;

  const size_t* compressed_prefix_ptr() const;

  /**
   * @brief Get a copy of the metadata on the CPU. This syncs with this stream.
   *
   * @param stream The stream to copy on.
   *
   * @return The metadata on the CPU.
   */
  LZ4Metadata copyToHost(cudaStream_t stream);

protected:
  size_t max_size() const;

  void set_serialized_size(const size_t size);

private:
  const void* m_ptr;
  size_t m_maxSize;
  size_t m_numChunks;
  size_t m_serializedSize;
};

} // namespace nvcomp

#endif

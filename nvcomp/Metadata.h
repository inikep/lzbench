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

#ifndef NVCOMP_METADATA_H
#define NVCOMP_METADATA_H

#include "nvcomp.h"

namespace nvcomp
{

class Metadata
{
public:
  /**
   * @brief Create a new metadta object.
   *
   * @param type The type of data element to compress.
   * @param uncompressedBytes The size of the data while uncompressed.
   * @param compressedBytes The size of the data and metadata compressed.
   * @param compressionType The type of compressed metadata this is.
   */
  Metadata(
      nvcompType_t type,
      size_t uncompressedBytes,
      size_t compressedBytes,
      int compressionType);

  virtual ~Metadata() = default;

  /**
   * @brief Get the type of value that is compressed.
   *
   * @return The value type.
   */
  nvcompType_t getValueType() const;

  /**
   * @brief Get the size of the uncompressed data in bytes.
   *
   * @return The size of the uncompressed data.
   */
  size_t getUncompressedSize() const;

  /**
   * @brief Get the size of the compressed data and metadata in bytes.
   *
   * @return The size in bytes.
   */
  size_t getCompressedSize() const;

  /**
   * @brief Get the number of uncompressed elements.
   *
   * @return The number of elements.
   */
  size_t getNumUncompressedElements() const;

  /**
   * @brief Get the type of compression used.
   *
   * @return The type of compression.
   */
  int getCompressionType() const;

protected:
  void setUncompressedSize(size_t bytes);

  void setCompressedSize(size_t bytes);

private:
  /**
   * @brief The datatype of decompressed elements.
   */
  nvcompType_t m_type;

  /**
   * @brief The size in bytes of the uncompressed data.
   */
  size_t m_uncompressedBytes;

  /**
   * @brief The size in bytes of the compressed data (including serialized
   * metadata).
   */
  size_t m_compressedBytes;

  int m_compressionType;
};

} // namespace nvcomp

#endif

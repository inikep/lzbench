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

#ifndef NVCOMP_LZ4METADATA_H
#define NVCOMP_LZ4METADATA_H

#include "Metadata.h"
#include "lz4.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#define LZ4_FLAG 4

namespace nvcomp
{

class LZ4Metadata : public Metadata
{
public:
  constexpr static int COMPRESSION_ID = 0x4000;

  /**
   * @brief Order of header values stored at the beginning of the metadata. Each
   * offset is 8 bytes.
   */
  enum LZ4MetadataField
  {
    Header = 0,
    MetadataBytes = 1,
    UncompressedSize = 2,
    ChunkSize = 3,
    OffsetAddr = 4
  };

  /**
   * @brief Create a new metadta object.
   *
   * @param opts The cascaded compression options.
   * @param type The type of data element to compress.
   * @param uncompressedBytes The size of the data while uncompressed.
   * @param compressedBytes The size of the data and metadata compressed.
   */
  LZ4Metadata(
      const nvcompType_t type,
      const size_t uncompChunkBytes,
      const size_t uncompressedBytes,
      const size_t compressedBytes);

  /**
   * @brief Create metadata object from unformatted metadata memory copied onto
   * the CPU.
   *
   * @param memPtr The memory containing the metadata on the CPU.  This is in
   * the raw format that is contained at the beginning of any memory compressed
   * by the nvcomp LZ4 compressor.
   * @param compressedBytes The total size of the data in memPtr
   */
  LZ4Metadata(const void* const memPtr, size_t compressedBytes);

  /**
   * @brief Get number of bytes per chunk
   *
   * @return number of bytes in each chunk (except for the last one
   */
  size_t getUncompChunkSize() const;

  /**
   * @brief Get number of chunks in the entire dataset
   *
   * @return number of chunks in the dataset
   */
  size_t getNumChunks() const;

  /**
   * @brief Get offset of a particular chunk
   *
   * @return the byte offset of chunk idx in the compressed dataset
   */
  size_t getChunkOffset(size_t idx);

  size_t getMetadataSize() const;

  size_t* getOffsetArray();

private:
  size_t m_uncompChunkBytes;
  size_t m_metadataBytes;
  size_t m_version;

  /**
   * @brief The offsets of each chunk in the compressed dataset
   */
  std::vector<size_t> m_chunkOffsets;
};

} // namespace nvcomp

#endif

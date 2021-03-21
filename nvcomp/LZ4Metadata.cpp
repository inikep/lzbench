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

#include "LZ4Metadata.h"
#include "common.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <iostream>

namespace nvcomp
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr const size_t NULL_OFFSET = static_cast<size_t>(-1);
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

LZ4Metadata::LZ4Metadata(
    const nvcompType_t type,
    const size_t uncompChunkBytes,
    const size_t uncompressedBytes,
    const size_t compressedBytes) :
    Metadata(type, uncompressedBytes, compressedBytes, COMPRESSION_ID),
    m_uncompChunkBytes(uncompChunkBytes),
    m_metadataBytes(),
    m_version(0),
    m_chunkOffsets()
{
  // TODO - error checking for byte sizes
}

LZ4Metadata::LZ4Metadata(const void* const memPtr, size_t compressedBytes) :
    LZ4Metadata(
        NVCOMP_TYPE_UCHAR,
        ((const size_t*)memPtr)[ChunkSize],
        ((const size_t*)memPtr)[UncompressedSize],
        compressedBytes)
{
  m_chunkOffsets.resize(getNumChunks() + 1);
  std::copy(
      static_cast<const size_t*>(memPtr) + OffsetAddr,
      static_cast<const size_t*>(memPtr) + OffsetAddr + getNumChunks(),
      m_chunkOffsets.begin());
  m_chunkOffsets.back() = compressedBytes;
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

size_t LZ4Metadata::getUncompChunkSize() const
{
  return m_uncompChunkBytes;
}

size_t LZ4Metadata::getNumChunks() const
{
  return roundUpDiv(getUncompressedSize(), m_uncompChunkBytes);
}

size_t LZ4Metadata::getChunkOffset(size_t idx)
{
  if (idx > roundUpDiv(getUncompressedSize(), m_uncompChunkBytes)) {
    throw std::runtime_error(
        "Invalid chunk index: " + std::to_string(idx) + ", total chunks is "
        + std::to_string(
              roundUpDiv(getUncompressedSize(), m_uncompChunkBytes)));
    return 0;
  } else {
    return m_chunkOffsets[idx];
  }
}

size_t LZ4Metadata::getMetadataSize() const
{
  return m_metadataBytes;
}

size_t* LZ4Metadata::getOffsetArray()
{
  return m_chunkOffsets.data();
}

} // namespace nvcomp

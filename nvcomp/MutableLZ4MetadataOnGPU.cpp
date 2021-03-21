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

#include "MutableLZ4MetadataOnGPU.h"
#include "CudaUtils.h"

#include <cassert>
#include <stdexcept>

namespace nvcomp
{

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

MutableLZ4MetadataOnGPU::MutableLZ4MetadataOnGPU(
    void* const ptr, const size_t maxSize) :
    LZ4MetadataOnGPU(ptr, maxSize),
    m_mutable_ptr(ptr)
{
  if (ptr == nullptr) {
    throw std::runtime_error("Cannot have nullptr for metadata location.");
  }
}

MutableLZ4MetadataOnGPU::MutableLZ4MetadataOnGPU(
    const MutableLZ4MetadataOnGPU& other) :
    MutableLZ4MetadataOnGPU(other.m_mutable_ptr, other.max_size())
{
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

MutableLZ4MetadataOnGPU& MutableLZ4MetadataOnGPU::
operator=(const MutableLZ4MetadataOnGPU& other)
{
  LZ4MetadataOnGPU::operator=(other);
  m_mutable_ptr = other.m_mutable_ptr;

  return *this;
}

void MutableLZ4MetadataOnGPU::copyToGPU(
    const LZ4Metadata& metadata, cudaStream_t stream)
{
  const size_t required_size = getSerializedSizeOf(metadata);
  if (required_size > max_size()) {
    throw std::runtime_error(
        "Insufficient space for metadata: " + std::to_string(required_size)
        + " / " + std::to_string(max_size()));
  }

  const size_t num_metadata_values = LZ4Metadata::OffsetAddr + 1;

  std::vector<size_t> buffer(num_metadata_values);
  buffer[LZ4Metadata::Header] = LZ4_FLAG;
  buffer[LZ4Metadata::MetadataBytes] = required_size;
  buffer[LZ4Metadata::UncompressedSize] = metadata.getUncompressedSize();
  buffer[LZ4Metadata::ChunkSize] = metadata.getUncompChunkSize();
  buffer[LZ4Metadata::OffsetAddr] = buffer[LZ4Metadata::MetadataBytes];

  CudaUtils::copy_async(
      static_cast<size_t*>(m_mutable_ptr),
      buffer.data(),
      buffer.size(),
      HOST_TO_DEVICE,
      stream);

  set_serialized_size(required_size);
}

size_t* MutableLZ4MetadataOnGPU::compressed_prefix_ptr()
{
  size_t* const ptr
      = static_cast<size_t*>(m_mutable_ptr) + LZ4Metadata::OffsetAddr;

  // make sure we're in sync with the const version
  assert(LZ4MetadataOnGPU::compressed_prefix_ptr() == ptr);

  return ptr;
}

} // namespace nvcomp

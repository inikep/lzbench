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

#include "MutableBatchedLZ4MetadataOnGPU.h"

#include "CudaUtils.h"
#include "LZ4Metadata.h"
#include "LZ4MetadataOnGPU.h"
#include "TempSpaceBroker.h"
#include "common.h"

namespace nvcomp
{

/******************************************************************************
 * TYPES **********************************************************************
 *****************************************************************************/

namespace
{

struct temp_metadata_t
{
  size_t header;
  size_t metadata_bytes;
  size_t uncompressed_size;
  size_t chunk_size;
  size_t offset_addr;
  // used to determine where the metadata goes
  void* metadata_dest;
};

} // namespace

/******************************************************************************
 * KERNELS ********************************************************************
 *****************************************************************************/

namespace
{

__global__ void distributeMetadataKernel(
    const temp_metadata_t* const metadata_device, const size_t batch_size)
{
  const int bidx = threadIdx.x + blockDim.x * blockIdx.x;

  if (bidx < batch_size) {
    size_t* buffer
        = reinterpret_cast<size_t*>(metadata_device[bidx].metadata_dest);
    buffer[LZ4Metadata::Header] = metadata_device[bidx].header;
    buffer[LZ4Metadata::MetadataBytes] = metadata_device[bidx].metadata_bytes;
    buffer[LZ4Metadata::UncompressedSize]
        = metadata_device[bidx].uncompressed_size;
    buffer[LZ4Metadata::ChunkSize] = metadata_device[bidx].chunk_size;
    buffer[LZ4Metadata::OffsetAddr] = metadata_device[bidx].offset_addr;
  }
}

} // namespace

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

MutableBatchedLZ4MetadataOnGPU::MutableBatchedLZ4MetadataOnGPU(
    void* const* const out_ptrs,
    const size_t* const max_out_sizes,
    const size_t batch_size) :
    m_buffer(),
    m_batch_size(batch_size),
    m_out_ptrs(out_ptrs),
    m_max_out_sizes(max_out_sizes)
{
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void MutableBatchedLZ4MetadataOnGPU::copyToGPU(
    const std::vector<LZ4Metadata>& metadata,
    void* temp_space,
    size_t temp_size,
    size_t* serialized_sizes,
    cudaStream_t stream)
{
  if (metadata.size() != m_batch_size) {
    throw std::runtime_error(
        "Mismatch batch size (" + std::to_string(m_batch_size)
        + ") and number of metadatas (" + std::to_string(metadata.size())
        + ").");
  }

  // setup temp space
  TempSpaceBroker broker(temp_space, temp_size);

  temp_metadata_t* metadata_device;
  broker.reserve(&metadata_device, m_batch_size);

  m_buffer.resize(sizeof(temp_metadata_t) * m_batch_size);
  temp_metadata_t* const metadata_host
      = reinterpret_cast<temp_metadata_t*>(m_buffer.data());

  // copy metadata to buffer
  for (size_t i = 0; i < m_batch_size; ++i) {
    serialized_sizes[i] = LZ4MetadataOnGPU::getSerializedSizeOf(metadata[i]);
    if (serialized_sizes[i] > m_max_out_sizes[i]) {
      throw std::runtime_error(
          "Insufficient space for metadata for item " + std::to_string(i)
          + " : " + std::to_string(m_max_out_sizes[i]) + " / "
          + std::to_string(serialized_sizes[i]));
    }

    metadata_host[i].header = LZ4_FLAG;
    metadata_host[i].metadata_bytes = serialized_sizes[i];
    metadata_host[i].uncompressed_size = metadata[i].getUncompressedSize();
    metadata_host[i].chunk_size = metadata[i].getUncompChunkSize();
    metadata_host[i].offset_addr = metadata_host[i].metadata_bytes;
    metadata_host[i].metadata_dest = m_out_ptrs[i];
  }

  // copy buffer to gpu
  CudaUtils::copy_async(
      metadata_device, metadata_host, m_batch_size, HOST_TO_DEVICE, stream);

  // distribute metadata
  const dim3 block(128);
  const dim3 grid(roundUpDiv(m_batch_size, block.x));

  distributeMetadataKernel<<<grid, block, 0, stream>>>(
      metadata_device, m_batch_size);
  CudaUtils::check_last_error();
}

size_t*
MutableBatchedLZ4MetadataOnGPU::compressed_prefix_ptr(const size_t index)
{
  size_t* const ptr
      = static_cast<size_t*>(m_out_ptrs[index]) + LZ4Metadata::OffsetAddr;

  return ptr;
}

} // namespace nvcomp

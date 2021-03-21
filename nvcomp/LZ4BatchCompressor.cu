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

#include "CudaUtils.h"
#include "LZ4BatchCompressor.h"
#include "LZ4CompressionKernels.h"
#include "TempSpaceBroker.h"
#include "common.h"

namespace nvcomp
{

/******************************************************************************
 * KERNELS ********************************************************************
 *****************************************************************************/

namespace
{
template <int BLOCK_SIZE>
__global__ void collectItemSizes(
    size_t* const item_sizes,
    const size_t* const* const item_prefixes,
    const size_t* const uncomp_sizes,
    const size_t chunk_size,
    const size_t batch_size)
{
  const size_t item = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  if (item < batch_size) {
    const size_t num_chunks = roundUpDiv(uncomp_sizes[item], chunk_size);

    item_sizes[item] = item_prefixes[item][num_chunks];
  }
}

} // namespace

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

size_t compute_staging_bytes(
    const size_t* const decomp_data_size,
    const size_t batch_size,
    const size_t chunk_size)
{
  const size_t num_chunks
      = lz4ComputeChunksInBatch(decomp_data_size, batch_size, chunk_size);

  const size_t staging_bytes = roundUpTo(
      lz4CompressComputeTempSize(num_chunks, chunk_size), sizeof(size_t));

  return staging_bytes;
}

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

size_t LZ4BatchCompressor::calculate_workspace_size(
    const size_t* const decomp_data_size,
    const size_t batch_size,
    const size_t chunk_size)
{
  const size_t staging_bytes
      = compute_staging_bytes(decomp_data_size, batch_size, chunk_size);
  const size_t pointer_bytes = 2 * batch_size * sizeof(uint8_t*);
  const size_t size_bytes
      = (batch_size * sizeof(size_t*)) + (batch_size * sizeof(size_t));
  const size_t offset_bytes = (batch_size + 1) * sizeof(size_t);

  return staging_bytes + pointer_bytes + size_bytes + offset_bytes;
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

LZ4BatchCompressor::LZ4BatchCompressor(
    const uint8_t* const* decomp_data,
    const size_t* decomp_data_size,
    const size_t batch_size,
    const size_t chunk_size) :
    m_batch_size(batch_size),
    m_chunk_size(chunk_size),
    m_buffer(),
    m_input_ptrs(nullptr),
    m_input_sizes(nullptr),
    m_output_ptrs(nullptr),
    m_output_sizes(nullptr),
    m_output_offsets(nullptr),
    m_workspace(nullptr),
    m_workspace_size(0),
    m_host_item_sizes(nullptr),
    m_output_configured(false)
{
  const size_t input_ptrs = batch_size * sizeof(*decomp_data);
  const size_t input_sizes = batch_size * sizeof(*decomp_data_size);
  const size_t output_ptrs = batch_size * sizeof(uint8_t*);
  const size_t output_sizes = batch_size * sizeof(size_t*);
  const size_t output_offsets = batch_size * sizeof(size_t);

  const size_t buffer_size
      = input_ptrs + input_sizes + output_sizes + output_ptrs + output_offsets;

  m_buffer.resize(buffer_size);

  m_input_ptrs = reinterpret_cast<const uint8_t**>(m_buffer.data());
  m_input_sizes = reinterpret_cast<size_t*>(m_buffer.data() + input_ptrs);
  m_output_ptrs
      = reinterpret_cast<uint8_t**>(m_buffer.data() + input_ptrs + input_sizes);
  m_output_sizes = reinterpret_cast<size_t**>(
      m_buffer.data() + input_ptrs + input_sizes + output_ptrs);
  m_output_offsets = reinterpret_cast<size_t*>(
      m_buffer.data() + input_sizes + input_ptrs + output_sizes
      + output_offsets);

  std::copy(decomp_data_size, decomp_data_size + batch_size, m_input_sizes);
  std::copy(decomp_data, decomp_data + batch_size, m_input_ptrs);
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

size_t LZ4BatchCompressor::get_workspace_size() const
{
  return calculate_workspace_size(m_input_sizes, m_batch_size, m_chunk_size);
}

void LZ4BatchCompressor::configure_workspace(
    void* const workspace, const size_t size)
{
  const size_t required_Size = get_workspace_size();
  if (size < required_Size) {
    throw std::runtime_error(
        "Insufficient workspace size: " + std::to_string(size) + " / "
        + std::to_string(required_Size));
  } else if (workspace == nullptr) {
    throw std::runtime_error("Workspace cannot be null.");
  }

  m_workspace = workspace;
  m_workspace_size = size;
}

void LZ4BatchCompressor::configure_output(
    uint8_t* const* const device_locations,
    size_t* const* const device_sizes,
    const size_t* const device_offsets,
    size_t* const host_item_sizes)
{
  std::copy(device_sizes, device_sizes + m_batch_size, m_output_sizes);
  std::copy(device_locations, device_locations + m_batch_size, m_output_ptrs);
  std::copy(device_offsets, device_offsets + m_batch_size, m_output_offsets);
  m_host_item_sizes = host_item_sizes;
  m_output_configured = true;
}

void LZ4BatchCompressor::compress_async(cudaStream_t stream)
{
  if (!is_workspace_configured()) {
    throw std::runtime_error(
        "Workspace must be configured before compressing.");
  } else if (!is_output_configured()) {
    throw std::runtime_error("Output must be configured before compressing.");
  }

  TempSpaceBroker temp(m_workspace, m_workspace_size);

  uint8_t* workspace;
  const size_t workspace_size
      = compute_staging_bytes(m_input_sizes, m_batch_size, m_chunk_size);
  temp.reserve(&workspace, workspace_size);

  // these have all the same size, and generally should on all platforms as
  // the definition of size_t should make it the same size
  static_assert(
      alignof(size_t) == alignof(uint8_t*),
      "Pointers must have the same alignment as size_t");

  const uint8_t** in_ptrs_device;
  temp.reserve(&in_ptrs_device, m_batch_size);

  size_t* in_sizes_device;
  temp.reserve(&in_sizes_device, m_batch_size);

  uint8_t** out_ptrs_device;
  temp.reserve(&out_ptrs_device, m_batch_size);

  size_t** out_sizes_device;
  temp.reserve(&out_sizes_device, m_batch_size);

  size_t* out_prefix_offsets_device;
  temp.reserve(&out_prefix_offsets_device, m_batch_size);

  const size_t buffer_size = reinterpret_cast<const uint8_t*>(
                                 out_prefix_offsets_device + m_batch_size)
                             - reinterpret_cast<const uint8_t*>(in_ptrs_device);

  if (buffer_size > m_buffer.size()) {
    throw std::runtime_error(
        "Internal error: mismatched buffer sizes: "
        + std::to_string(buffer_size) + " / "
        + std::to_string(m_buffer.size()));
  }

  // just cop from the first pointer to the end of the last
  CudaUtils::copy_async<uint8_t>(
      reinterpret_cast<uint8_t*>(in_ptrs_device),
      m_buffer.data(),
      buffer_size,
      HOST_TO_DEVICE,
      stream);

  // TODO: implement step_size
  lz4CompressBatch(
      in_ptrs_device,
      in_sizes_device,
      m_input_sizes,
      m_batch_size,
      m_chunk_size,
      workspace,
      workspace_size,
      out_ptrs_device,
      out_sizes_device,
      out_prefix_offsets_device,
      stream);

  // repurpose in item sizes for collecting batch totals
  if (m_host_item_sizes) {
    size_t* item_sizes_device = in_sizes_device;

    // We're using 64 threads here to maximize the number of active thread
    // blocks we can have for a given number of items, as we would expect to
    // struggle to make use of much of the GPU with a number of items in the
    // hundreds or thousands.
    constexpr const int BLOCK_SIZE = 64;

    const dim3 grid(roundUpDiv(m_batch_size, BLOCK_SIZE));
    const dim3 block(BLOCK_SIZE);

    collectItemSizes<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        item_sizes_device,
        out_sizes_device,
        in_sizes_device,
        m_chunk_size,
        m_batch_size);

    CudaUtils::copy_async(
        m_host_item_sizes, item_sizes_device, m_batch_size,
        DEVICE_TO_HOST, stream);
  }
}

/******************************************************************************
 * PRIVATE METHODS ************************************************************
 *****************************************************************************/

bool LZ4BatchCompressor::is_workspace_configured() const
{
  return m_workspace != nullptr;
}

bool LZ4BatchCompressor::is_output_configured() const
{
  return m_output_configured;
}

} // namespace nvcomp

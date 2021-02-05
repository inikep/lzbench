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

#include "cuda_runtime.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nvcomp
{

class LZ4BatchCompressor
{
public:
  static size_t calculate_workspace_size(
      const size_t* decomp_data_size, size_t batch_size, size_t chunk_size);

  /**
   * @brief Create a new LZ4BatchCompressor.
   *
   * @param batch_size The number of items in the batch.
   * @param chunk_size The size of each chunk to compress.
   */
  LZ4BatchCompressor(
      const uint8_t* const* decomp_data,
      const size_t* decomp_data_size,
      const size_t batch_size,
      const size_t chunk_size);

  LZ4BatchCompressor(const LZ4BatchCompressor& other) = delete;
  LZ4BatchCompressor& operator=(const LZ4BatchCompressor& other) = delete;

  /**
   * @brief Get the size of the workspace required.
   *
   * @return The size of the workspace in bytes.
   */
  size_t get_workspace_size() const;

  /**
   * @brief Set the allocated workspace.
   *
   * @param workspace The workspace.
   * @param size The size of the workspace in bytes.
   */
  void configure_workspace(void* workspace, size_t size);

  void configure_output(
      uint8_t* const* device_locations,
      size_t* const* device_sizes,
      const size_t* device_offsets,
      size_t* const host_item_sizes);

  void compress_async(cudaStream_t stream);

private:
  size_t m_batch_size;
  size_t m_chunk_size;
  std::vector<uint8_t> m_buffer;
  const uint8_t** m_input_ptrs;
  size_t* m_input_sizes;
  uint8_t** m_output_ptrs;
  size_t** m_output_sizes;
  size_t* m_output_offsets;
  void* m_workspace;
  size_t m_workspace_size;
  size_t* m_host_item_sizes;
  bool m_output_configured;

  bool is_workspace_configured() const;

  bool is_output_configured() const;
};

} // namespace nvcomp

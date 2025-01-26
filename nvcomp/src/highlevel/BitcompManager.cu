/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <memory>

#include "Check.h"
#include "CudaUtils.h"
#include "common.h"
#include "nvcomp_common_deps/hlif_shared_types.hpp"
#include "nvcomp/nvcompManager.hpp"
#include "highlevel/ManagerBase.hpp"
#include "nvcomp/bitcomp.hpp"
#include "BitcompManager.hpp"

#ifdef ENABLE_BITCOMP

#include <bitcomp.h>

namespace nvcomp {

  // Convert the NVCOMP type to a BITCOMP type
  bitcompDataType_t bitcomp_data_type (nvcompType_t data_type)
  {
    switch (data_type) {
    case NVCOMP_TYPE_CHAR:
      return BITCOMP_SIGNED_8BIT;
      break;
    case NVCOMP_TYPE_USHORT:
      return BITCOMP_UNSIGNED_16BIT;
      break;
    case NVCOMP_TYPE_SHORT:
      return BITCOMP_SIGNED_16BIT;
      break;
    case NVCOMP_TYPE_UINT:
      return BITCOMP_UNSIGNED_32BIT;
      break;
    case NVCOMP_TYPE_INT:
      return BITCOMP_SIGNED_32BIT;
      break;
    case NVCOMP_TYPE_ULONGLONG:
      return BITCOMP_UNSIGNED_64BIT;
      break;
    case NVCOMP_TYPE_LONGLONG:
      return BITCOMP_SIGNED_64BIT;
      break;
    default:
      return BITCOMP_UNSIGNED_8BIT;
    }
  }

  /**
   * @brief Single-threaded kernel to update the common header 
   * 
   * @param common_header header filled in by this routine (GPU accessible)
   * @param comp_buffer The location to output the compressed data to (GPU accessible).
   * @param decomp_buffer_size The length of the uncompressed input data
   * 
   */
  __global__ void bitcomp_header_k (CommonHeader *common_header, uint8_t* comp_buffer, uint64_t decomp_buffer_size)
  {
    common_header->magic_number = 0;
    common_header->major_version = NVCOMP_MAJOR_VERSION;
    common_header->minor_version = NVCOMP_MINOR_VERSION;
    common_header->format = FormatType::Bitcomp;
    common_header->decomp_data_size = decomp_buffer_size;
    common_header->num_chunks = 0;
    common_header->include_chunk_starts = false;
    common_header->full_comp_buffer_checksum = 0;
    common_header->decomp_buffer_checksum = 0;
    common_header->include_per_chunk_comp_buffer_checksums = false;
    common_header->include_per_chunk_decomp_buffer_checksums = false;
    common_header->uncomp_chunk_size = 0;
    common_header->comp_data_offset = (uintptr_t)comp_buffer - (uintptr_t)common_header;
  }

  /**
   * @brief Bitcomp compression helper 
   * 
   * @param common_header header filled in by this routine (GPU accessible)
   * @param decomp_buffer The uncompressed input data (GPU accessible)
   * @param decomp_buffer_size The length of the uncompressed input data
   * @param comp_buffer The location to output the compressed data to (GPU accessible).
   * @param comp_config Resulted from configure_compression given this decomp_buffer_size.
   * 
   */
  void BitcompSingleStreamManager::do_compress(
      CommonHeader* common_header,
      const uint8_t* decomp_buffer, 
      uint8_t* comp_buffer,
      const CompressionConfig& comp_config)
  {
    bitcompHandle_t handle;
    CHECK_EQ(
        bitcompCreatePlan(
            &handle,
            comp_config.uncompressed_buffer_size,
            bitcomp_data_type(format_spec->data_type),
            BITCOMP_LOSSLESS,
            static_cast<bitcompAlgorithm_t>(format_spec->algo)),
        BITCOMP_SUCCESS);

    CHECK_EQ(bitcompSetStream(handle, user_stream), BITCOMP_SUCCESS);

    CHECK_EQ(
        bitcompCompressLossless(handle, decomp_buffer, comp_buffer),
        BITCOMP_SUCCESS);

    bitcomp_header_k<<<1, 1, 0, user_stream>>>(
        common_header, comp_buffer, comp_config.uncompressed_buffer_size);

    CHECK_EQ(
        bitcompGetCompressedSizeAsync(
            comp_buffer, &common_header->comp_data_size, user_stream),
        BITCOMP_SUCCESS);

    CHECK_EQ(bitcompDestroyPlan(handle), BITCOMP_SUCCESS);
  }

  /**
   * @brief Bitcomp decompression helper 
   *
   * @param decomp_buffer The location to output the decompressed data to (GPU accessible).
   * @param comp_buffer The compressed input data (GPU accessible).
   * @param decomp_config Resulted from configure_decompression given this decomp_buffer_size.
   */
  void BitcompSingleStreamManager::do_decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& config)
  {
    bitcompHandle_t handle;
    CHECK_EQ(
        bitcompCreatePlan(
            &handle,
            config.decomp_data_size,
            bitcomp_data_type(format_spec->data_type),
            BITCOMP_LOSSLESS,
            static_cast<bitcompAlgorithm_t>(format_spec->algo)),
        BITCOMP_SUCCESS);

    CHECK_EQ(bitcompSetStream(handle, user_stream), BITCOMP_SUCCESS);

    CHECK_EQ(bitcompUncompress(handle, comp_buffer, decomp_buffer), BITCOMP_SUCCESS);

    CHECK_EQ(bitcompDestroyPlan(handle), BITCOMP_SUCCESS);
  }

  /**
   * @brief Computes the maximum compressed output size for a given
   * uncompressed buffer.
   */
  size_t BitcompSingleStreamManager::calculate_max_compressed_output_size(CompressionConfig& comp_config)
  {
    return bitcompMaxBuflen (comp_config.uncompressed_buffer_size);
  }

} // namespace nvcomp

#else // ENABLE_BITCOMP

namespace nvcomp {
void BitcompSingleStreamManager::do_compress(CommonHeader*, const uint8_t*, uint8_t*, const CompressionConfig&)
{
  throw NVCompException(nvcompErrorNotSupported, "Bitcomp support not available in this build.");
}
void BitcompSingleStreamManager::do_decompress(uint8_t*, const uint8_t*, const DecompressionConfig&)
{
  throw NVCompException(nvcompErrorNotSupported, "Bitcomp support not available in this build.");
}
size_t BitcompSingleStreamManager::calculate_max_compressed_output_size(CompressionConfig&)
{
  throw NVCompException(nvcompErrorNotSupported, "Bitcomp support not available in this build.");
}
}

#endif // ENABLE_BITCOMP
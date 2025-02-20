/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include "ManagerBase.hpp"
#include "common.h"

namespace nvcomp {

/**
 * Base class for compression formats that are able to use the 
 * nvcomp shared HLIF logic code. 
 * 
 * This class does compression by splitting the uncompressed buffer into chunks. 
 * It compresses each chunk independently and outputs the chunks into a gapless 
 * result buffer in arbitrary chunk ordering. The header then includes the information
 * needed to decompress the chunks back into the original ordering.
 * 
 * Generally, the code in hlif_shared.cuh can be used to implement 
 * do_batch_(compress/decompress). In this case, device code for compression / decompression 
 * can be shared between the low level batch API and the BatchManager extension.
 * 
 */
template<typename FormatSpecHeader>
struct BatchManager : ManagerBase<FormatSpecHeader> {

protected: // members
  uint32_t* ix_chunk;
  using ManagerBase<FormatSpecHeader>::user_stream;

private: // members
  uint32_t max_comp_ctas;
  uint32_t max_decomp_ctas;
  size_t max_comp_chunk_size;
  size_t uncomp_chunk_size;

public: // API
  BatchManager(size_t uncomp_chunk_size, cudaStream_t user_stream = 0, int device_id = 0)
    : ManagerBase<FormatSpecHeader>(user_stream, device_id),
      ix_chunk(0),
      max_comp_ctas(0),
      max_decomp_ctas(0),
      max_comp_chunk_size(0),
      uncomp_chunk_size(uncomp_chunk_size)
  {
    CudaUtils::check(cudaMalloc(&ix_chunk, sizeof(uint32_t)));
  }

  virtual ~BatchManager() {
    CudaUtils::check(cudaFree(ix_chunk));
  }

  BatchManager& operator=(const BatchManager&) = delete;     
  BatchManager(const BatchManager&) = delete;

  void do_decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& config) final override
  {
    const size_t* comp_chunk_offsets = roundUpToAlignment<const size_t>(comp_buffer);
    const size_t* comp_chunk_sizes = comp_chunk_offsets + config.num_chunks;
    const uint32_t* comp_chunk_checksums = reinterpret_cast<const uint32_t*>(comp_chunk_sizes + config.num_chunks);
    const uint32_t* decomp_chunk_checksums = comp_chunk_checksums + config.num_chunks;
    const uint8_t* comp_data_buffer = reinterpret_cast<const uint8_t*>(decomp_chunk_checksums + config.num_chunks);

    CudaUtils::check(cudaMemsetAsync(ix_chunk, 0, sizeof(uint32_t), user_stream));
    do_batch_decompress(
        comp_data_buffer,
        decomp_buffer,
        config.num_chunks,
        comp_chunk_offsets,
        comp_chunk_sizes,
        config.get_status());
  }
  
  /**
   * @brief Configures the decompression
   * 
   * Synchronizes the user_stream
   */ 
  virtual void do_configure_decompression(
      DecompressionConfig& decomp_config,
      const CommonHeader* common_header) final override 
  {
    CudaUtils::check(cudaMemcpyAsync(&decomp_config.num_chunks, 
        &common_header->num_chunks, 
        sizeof(size_t),
        cudaMemcpyDefault,
        user_stream));
  }

  /**
   * @brief Optionally does additional decompression configuration without syncing the stream
   */
  virtual void do_configure_decompression(
      DecompressionConfig& decomp_config,
      const CompressionConfig& comp_config) 
  {
    decomp_config.num_chunks = comp_config.num_chunks;
  } 

private: // pure virtual functions
  /**
   * @brief Computes the maximum compressed chunk size given the member variable
   * uncomp_chunk_size
   */ 
  virtual size_t compute_max_compressed_chunk_size() = 0;

  /**
   * @brief Computes the maximum CTA occupancy for compression
   */ 
  virtual uint32_t compute_compression_max_block_occupancy() = 0;

  /**
   * @brief Computes the maximum CTA occupancy for decompression
   */ 
  virtual uint32_t compute_decompression_max_block_occupancy() = 0;

  /**
   * @brief Does the batch level compression
   */ 
  virtual void do_batch_compress(const CompressArgs& compress_args) = 0;

  /**
   * @brief Does the batch level decompression
   */ 
  virtual void do_batch_decompress(
      const uint8_t* comp_data_buffer,
      uint8_t* decomp_buffer,
      const uint32_t num_chunks,
      const size_t* comp_chunk_offsets,
      const size_t* comp_chunk_sizes,
      nvcompStatus_t* output_status) = 0;

protected: // derived helpers
  void finish_init() {
    max_comp_chunk_size = compute_max_compressed_chunk_size();    
    
    format_specific_init();
    max_comp_ctas = compute_compression_max_block_occupancy();
    max_decomp_ctas = compute_decompression_max_block_occupancy();
    
    ManagerBase<FormatSpecHeader>::finish_init();    
  }

protected: // accessors
  uint32_t get_max_comp_ctas() {
    return max_comp_ctas;
  }

  uint32_t get_max_decomp_ctas() {
    return max_decomp_ctas;
  }

  size_t get_max_comp_chunk_size() {
    return max_comp_chunk_size;
  }

  size_t get_uncomp_chunk_size() {
    return uncomp_chunk_size;
  }


private: // helper API overrides
  size_t calculate_max_compressed_output_size(CompressionConfig& comp_config) final override
  {
    const size_t comp_buffer_size = max_comp_chunk_size * comp_config.num_chunks;

    const size_t chunk_offsets_size = sizeof(ChunkStartOffset_t) * comp_config.num_chunks;
    const size_t chunk_sizes_size = sizeof(uint32_t) * comp_config.num_chunks;
    // *2 for decomp and comp checksums
    const size_t checksum_size = sizeof(Checksum_t) * comp_config.num_chunks * 2;

    return sizeof(CommonHeader) + sizeof(FormatSpecHeader) + 
        chunk_offsets_size + chunk_sizes_size + checksum_size + comp_buffer_size;
  }

  void do_compress(
      CommonHeader* common_header,
      const uint8_t* decomp_buffer, 
      uint8_t* comp_buffer,
      const CompressionConfig& comp_config) final override
  {    
    CompressArgs compress_args;
    compress_args.common_header = common_header;
    compress_args.decomp_buffer = decomp_buffer;
    compress_args.decomp_buffer_size = comp_config.uncompressed_buffer_size;
    compress_args.scratch_buffer = ManagerBase<FormatSpecHeader>::scratch_buffer;
    compress_args.uncomp_chunk_size = uncomp_chunk_size;
    compress_args.ix_output = &common_header->comp_data_size;
    compress_args.ix_chunk = ix_chunk;
    
    compress_args.num_chunks = comp_config.num_chunks;
    compress_args.max_comp_chunk_size = max_comp_chunk_size;

    // Pad so that the comp chunk offsets are properly aligned
    compress_args.comp_chunk_offsets = roundUpToAlignment<size_t>(comp_buffer);
    compress_args.comp_chunk_sizes = compress_args.comp_chunk_offsets + comp_config.num_chunks;    

    uint32_t* comp_chunk_checksums = reinterpret_cast<uint32_t*>(compress_args.comp_chunk_sizes + comp_config.num_chunks);
    uint32_t* decomp_chunk_checksums = comp_chunk_checksums + comp_config.num_chunks;
    
    compress_args.comp_buffer = reinterpret_cast<uint8_t*>(decomp_chunk_checksums + comp_config.num_chunks);
    compress_args.output_status = comp_config.get_status();

    CudaUtils::check(cudaMemsetAsync(ix_chunk, 0, sizeof(uint32_t), user_stream));    
    
    do_batch_compress(compress_args);
  }

  virtual void do_configure_compression(CompressionConfig& config) final override
  {
    config.num_chunks = roundUpDiv(config.uncompressed_buffer_size, uncomp_chunk_size);
  }

  /**
   * @brief Computes the required scratch space size
   * 
   * Note: This can be overridden if the format needs additional scratch space 
   * beyond that used for compressing blocks. See LZ4BatchManager for an example.
   */
  virtual size_t compute_scratch_buffer_size() override
  {
    return max_comp_ctas * max_comp_chunk_size;
  }

  /**
   * @brief Optional helper that is called in the finish_init sequence
   */  
  virtual void format_specific_init() 
  {}

};

} // namespace nvcomp

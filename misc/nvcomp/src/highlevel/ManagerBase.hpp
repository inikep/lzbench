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

#include <memory>
#include <vector>

#include "nvcomp/nvcompManager.hpp"

#include "Check.h"
#include "CudaUtils.h"
#include "PinnedPtrs.hpp"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

namespace nvcomp {

/**
 * @brief ManagerBase contains shared functionality amongst the different nvcompManager types
 * 
 * - Intended that all Managers will inherit from this class directly or indirectly.
 *
 * - Contains a CPU/GPU-accessible memory pool for result statuses to avoid repeated 
 *   allocations when tasked with multiple compressions / decompressions.
 * 
 * - Templated on the particular format's FormatSpecHeader so that some operations can be shared here. 
 *   This is likely to be inherited by template classes. In this case, 
 *   some usage trickery is suggested to get around dependent name lookup issues.
 *   https://en.cppreference.com/w/cpp/language/dependent_name
 *   
 */
#include <assert.h>

template <typename FormatSpecHeader>
struct ManagerBase : nvcompManagerBase {

protected: // members
  CommonHeader* common_header_cpu;
  cudaStream_t user_stream;
  uint8_t* scratch_buffer;
  size_t scratch_buffer_size;
  int device_id;
  PinnedPtrPool<nvcompStatus_t> status_pool;
  bool manager_filled_scratch_buffer;

private: // members
  bool scratch_buffer_filled;

protected: // members
  bool finished_init;

public: // API
  /**
   * @brief Construct a ManagerBase
   * 
   * @param user_stream The stream to use for all operations. Optional, defaults to the default stream
   * @param device_id The default device ID to use for all operations. Optional, defaults to the default device
   */
  ManagerBase(cudaStream_t user_stream = 0, int device_id = 0) 
    : common_header_cpu(),
      user_stream(user_stream),
      scratch_buffer(nullptr),
      scratch_buffer_size(0),
      device_id(device_id),
      status_pool(),
      manager_filled_scratch_buffer(false),
      scratch_buffer_filled(false),
      finished_init(false)
  {
    CudaUtils::check(cudaHostAlloc(&common_header_cpu, sizeof(CommonHeader), cudaHostAllocDefault));
  }

  size_t get_required_scratch_buffer_size() final override {
    return scratch_buffer_size;
  }

  // Disable copying
  ManagerBase(const ManagerBase&) = delete;
  ManagerBase& operator=(const ManagerBase&) = delete;
  ManagerBase() = delete;     

  size_t get_compressed_output_size(uint8_t* comp_buffer) final override {
    CommonHeader* common_header = reinterpret_cast<CommonHeader*>(comp_buffer);
    
    CudaUtils::check(cudaMemcpy(common_header_cpu, 
        common_header, 
        sizeof(CommonHeader),
        cudaMemcpyDefault));

    return common_header_cpu->comp_data_size + common_header_cpu->comp_data_offset;
  };
  
  virtual ~ManagerBase() {
    CudaUtils::check(cudaFreeHost(common_header_cpu));
    if (scratch_buffer_filled) {
      if (manager_filled_scratch_buffer) {
        CudaUtils::check(cudaFree(scratch_buffer));
      }
    }
  }

  CompressionConfig configure_compression(const size_t decomp_buffer_size) final override
  {    
    CompressionConfig comp_config{status_pool, decomp_buffer_size};

    do_configure_compression(comp_config);

    comp_config.max_compressed_buffer_size = calculate_max_compressed_output_size(comp_config);

    return comp_config;
  }

  virtual DecompressionConfig configure_decompression(const uint8_t* comp_buffer) final override
  {
    const CommonHeader* common_header = reinterpret_cast<const CommonHeader*>(comp_buffer);
    DecompressionConfig decomp_config{status_pool};
    
    CudaUtils::check(cudaMemcpyAsync(&decomp_config.decomp_data_size, 
        &common_header->decomp_data_size, 
        sizeof(size_t),
        cudaMemcpyDefault,
        user_stream));
    
    do_configure_decompression(decomp_config, common_header);

    return decomp_config;
  }

  virtual DecompressionConfig configure_decompression(const CompressionConfig& comp_config) final override
  {
    DecompressionConfig decomp_config{status_pool};
    
    decomp_config.decomp_data_size = comp_config.uncompressed_buffer_size;    
    
    do_configure_decompression(decomp_config, comp_config);

    return decomp_config;
  }

  void set_scratch_buffer(uint8_t* new_scratch_buffer) final override
  {
    if (scratch_buffer_filled) {
      if (manager_filled_scratch_buffer) {
        #if CUDART_VERSION >= 11020
          CudaUtils::check(cudaFreeAsync(scratch_buffer, user_stream));
        #else
          CudaUtils::check(cudaFree(scratch_buffer));
        #endif
        manager_filled_scratch_buffer = false;
      }
    } else {
      scratch_buffer_filled = true;
    }
    scratch_buffer = new_scratch_buffer;
  }

  virtual void compress(
      const uint8_t* decomp_buffer, 
      uint8_t* comp_buffer,
      const CompressionConfig& comp_config) 
  {
    assert(finished_init);

    if (!scratch_buffer_filled) {
      #if CUDART_VERSION >= 11020
        CudaUtils::check(cudaMallocAsync(&scratch_buffer, scratch_buffer_size, user_stream));
      #else
        CudaUtils::check(cudaMalloc(&scratch_buffer, scratch_buffer_size));
      #endif
      scratch_buffer_filled = true;
      manager_filled_scratch_buffer = true;
    }    

    CommonHeader* common_header = reinterpret_cast<CommonHeader*>(comp_buffer);
    FormatSpecHeader* comp_format_header = reinterpret_cast<FormatSpecHeader*>(common_header + 1);
    CudaUtils::check(cudaMemcpyAsync(comp_format_header, get_format_header(), sizeof(FormatSpecHeader), cudaMemcpyDefault, user_stream));

    CudaUtils::check(cudaMemsetAsync(&common_header->comp_data_size, 0, sizeof(uint64_t), user_stream));

    uint8_t* new_comp_buffer = comp_buffer + sizeof(CommonHeader) + sizeof(FormatSpecHeader);
    do_compress(common_header, decomp_buffer, new_comp_buffer, comp_config);
  }

  virtual void decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& config)
  {
    assert(finished_init);

    if (!scratch_buffer_filled) {
      #if CUDART_VERSION >= 11020
        CudaUtils::check(cudaMallocAsync(&scratch_buffer, scratch_buffer_size, user_stream));
      #else
        CudaUtils::check(cudaMalloc(&scratch_buffer, scratch_buffer_size));
      #endif
      scratch_buffer_filled = true;
      manager_filled_scratch_buffer = true;
    }    

    const uint8_t* new_comp_buffer = comp_buffer + sizeof(CommonHeader) + sizeof(FormatSpecHeader);

    do_decompress(decomp_buffer, new_comp_buffer, config);
  }
  
protected: // helpers 
  virtual void finish_init() {
    scratch_buffer_size = compute_scratch_buffer_size();
    finished_init = true;
  }

private: // helpers

  /**
   * @brief Required helper that actually does the compression 
   * 
   * @param common_header header filled in by this routine (GPU accessible)
   * @param decomp_buffer The uncompressed input data (GPU accessible)
   * @param decomp_buffer_size The length of the uncompressed input data
   * @param comp_buffer The location to output the compressed data to (GPU accessible).
   * @param comp_config Resulted from configure_compression given this decomp_buffer_size.
   * 
   */
  virtual void do_compress(
      CommonHeader* common_header,
      const uint8_t* decomp_buffer, 
      uint8_t* comp_buffer,
      const CompressionConfig& comp_config) = 0;

  /**
   * @brief Required helper that actually does the decompression 
   *
   * @param decomp_buffer The location to output the decompressed data to (GPU accessible).
   * @param comp_buffer The compressed input data (GPU accessible).
   * @param decomp_config Resulted from configure_decompression given this decomp_buffer_size.
   */
  virtual void do_decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& config) = 0;

  /**
   * @brief Optionally does additional decompression configuration 
   */
  virtual void do_configure_decompression(
      DecompressionConfig& decomp_config,
      const CommonHeader* common_header) = 0; 

  /**
   * @brief Optionally does additional decompression configuration 
   */
  virtual void do_configure_decompression(
      DecompressionConfig& decomp_config,
      const CompressionConfig& comp_config) = 0; 

  /**
   * @brief Optionally does additional compression configuration 
   */
  virtual void do_configure_compression(CompressionConfig&) {}

  /**
   * @brief Computes the required scratch buffer size 
   */
  virtual size_t compute_scratch_buffer_size() = 0;

  /**
   * @brief Computes the maximum compressed output size for a given
   * uncompressed buffer.
   */
  virtual size_t calculate_max_compressed_output_size(CompressionConfig& comp_config) = 0;

  /**
   * @brief Retrieves a CPU-accessible pointer to the FormatSpecHeader
   */
  virtual FormatSpecHeader* get_format_header() = 0;

};

} // namespace nvcomp

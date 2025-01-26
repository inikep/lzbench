#pragma once

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

#include <memory>
#include <vector>

#include "CudaUtils.h"
#include "nvcomp_common_deps/hlif_shared_types.hpp"
#include "highlevel/PinnedPtrs.hpp"
#include "nvcomp/nvcompManager.hpp"

namespace nvcomp {

/******************************************************************************
 * CLASSES ********************************************************************
 *****************************************************************************/

/**
 * @brief Config used to aggregate information about the compression of a particular buffer.
 * 
 * Contains a "PinnedPtrHandle" to an nvcompStatus. After the compression is complete,
 * the user can check the result status which resides in pinned host memory.
 */
struct CompressionConfig::CompressionConfigImpl {
private: 
  std::unique_ptr<PinnedPtrPool<nvcompStatus_t>::PinnedPtrHandle> status;

public:
  /**
   * @brief Construct the config given an nvcompStatus_t memory pool
   */
  CompressionConfigImpl(PinnedPtrPool<nvcompStatus_t>& pool);

  /**
   * @brief Get the raw nvcompStatus_t*
   */
  nvcompStatus_t* get_status() const;
};

/**
 * @brief Config used to aggregate information about a particular decompression.
 * 
 * Contains a "PinnedPtrHandle" to an nvcompStatus. After the decompression is complete,
 * the user can check the result status which resides in pinned host memory.
 */
struct DecompressionConfig::DecompressionConfigImpl {
private: 
  std::unique_ptr<PinnedPtrPool<nvcompStatus_t>::PinnedPtrHandle> status;

public:
  size_t decomp_data_size;
  uint32_t num_chunks;

  /**
   * @brief Construct the config given an nvcompStatus_t memory pool
   */
  DecompressionConfigImpl(PinnedPtrPool<nvcompStatus_t>& pool);

  /**
   * @brief Get the raw nvcompStatus_t*
   */
  nvcompStatus_t* get_status() const;
};

} // namespace nvcomp
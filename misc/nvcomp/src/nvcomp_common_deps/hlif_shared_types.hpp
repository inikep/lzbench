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

#include <stdint.h>
#include "nvcomp/shared_types.h"

typedef uint64_t ChunkStartOffset_t;
typedef uint32_t Checksum_t;

enum FormatType : uint8_t {
  LZ4 = 0,
  Snappy = 1,
  ANS = 2,
  GDeflate = 3,
  Cascaded = 4,
  Bitcomp = 5,
  NotSupportedError = 6  
};

struct CommonHeader {
  uint32_t magic_number; // 
  uint8_t major_version;
  uint8_t minor_version;
  FormatType format;
  uint64_t comp_data_size;
  uint64_t decomp_data_size;
  size_t num_chunks;
  bool include_chunk_starts;
  Checksum_t full_comp_buffer_checksum;
  Checksum_t decomp_buffer_checksum;
  bool include_per_chunk_comp_buffer_checksums;
  bool include_per_chunk_decomp_buffer_checksums;
  size_t uncomp_chunk_size;
  uint32_t comp_data_offset;
};

struct CompressArgs {
  CommonHeader* common_header;
  const uint8_t* decomp_buffer;
  size_t decomp_buffer_size; 
  uint8_t* comp_buffer; 
  uint8_t* scratch_buffer;
  size_t uncomp_chunk_size;
  size_t* ix_output;
  uint32_t* ix_chunk;
  size_t num_chunks;
  size_t max_comp_chunk_size;
  size_t* comp_chunk_offsets;
  size_t* comp_chunk_sizes;
  nvcompStatus_t* output_status;
};


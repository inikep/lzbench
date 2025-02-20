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

#include "benchmark_template_chunked.cuh"
#include "nvcomp/cascaded.h"

#include <iostream>
#include <vector>

// Template benchmark uses a fixed format Opts, so just defaulting to INT for
// now.
// TODO: Update benchmark to accept type as a command-line parameter
static const nvcompBatchedCascadedOpts_t nvcompBatchedCascadedTestOpts
    = {4096, NVCOMP_TYPE_UINT, 2, 1, 1};

static bool isCascadedInputValid(const std::vector<std::vector<char>>& data)
{
  size_t typeSize = 1;
  auto type = nvcompBatchedCascadedTestOpts.type;
  switch (type) {
  case NVCOMP_TYPE_CHAR:
  case NVCOMP_TYPE_UCHAR:
    static_assert(
        sizeof(uint8_t) == 1 && sizeof(int8_t) == 1,
        "Compile-time check for clarity");
    return true;
  case NVCOMP_TYPE_SHORT:
  case NVCOMP_TYPE_USHORT:
    static_assert(
        sizeof(uint16_t) == 2 && sizeof(int16_t) == 2,
        "Compile-time check for clarity");
    typeSize = sizeof(uint16_t);
    break;
  case NVCOMP_TYPE_INT:
  case NVCOMP_TYPE_UINT:
    static_assert(
        sizeof(uint32_t) == 4 && sizeof(int32_t) == 4,
        "Compile-time check for clarity");
    typeSize = sizeof(uint32_t);
    break;
  case NVCOMP_TYPE_LONGLONG:
  case NVCOMP_TYPE_ULONGLONG:
    static_assert(
        sizeof(uint64_t) == 8 && sizeof(int64_t) == 8,
        "Compile-time check for clarity");
    typeSize = sizeof(uint64_t);
    break;
  default:
    std::cerr << "ERROR: Cascaded data type must be 0-7 (CHAR, UCHAR, SHORT, "
                 "USHORT, INT, UINT, LONGLONG, or ULONGLONG), "
                 "but it is "
              << int(type) << std::endl;
    return false;
  }

  for (const auto& chunk : data) {
    if ((chunk.size() % typeSize) != 0) {
      std::cerr << "ERROR: Input data must have a length and chunk size that "
                   "are a multiple of "
                << typeSize << ", the size of the specified data type."
                << std::endl;
      return false;
    }
  }
  return true;
}

GENERATE_CHUNKED_BENCHMARK(
    nvcompBatchedCascadedCompressGetTempSize,
    nvcompBatchedCascadedCompressGetMaxOutputChunkSize,
    nvcompBatchedCascadedCompressAsync,
    nvcompBatchedCascadedDecompressGetTempSize,
    nvcompBatchedCascadedDecompressAsync,
    isCascadedInputValid,
    nvcompBatchedCascadedTestOpts);

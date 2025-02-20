/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "nvcomp.h"
#include "nvcomp.hpp"
#include "nvcomp/cascaded.h"

#include "../src/common.h"
#include "catch.hpp"

#include <vector>
#include <cuda_profiler_api.h>
#include <iomanip>
#include <random>

using namespace nvcomp;

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      throw;                                                                   \
    }                                                                          \
  } while (0);


template <typename valT, typename runT>
void random_runs(
    std::vector<valT>& res, const valT max_val, const runT max_run, int seed)
{
  std::mt19937 eng(seed);
  std::uniform_int_distribution<runT> distr(0, max_run);

  for (valT val = 0; val < max_val; val++) {
    runT run = distr(eng);
    res.insert(res.end(), run, val);
  }
}

template <typename T>
void dump(const std::string desc, std::vector<T>& data, size_t size)
{
#if VERBOSE > 0
  std::cout << desc << ": ";
  for (size_t i = 0; i < size; i++)
    std::cout << data[i] << " ";
  std::cout << std::endl;
#endif
}



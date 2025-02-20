/*
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "nvcomp/gdeflate.h"

#include "Check.h"
#include "CudaUtils.h"
#include "common.h"
#include "nvcomp.h"
#include "nvcomp.hpp"
#include "type_macros.h"

#include <cassert>

#ifdef ENABLE_GDEFLATE
#include "gdeflate.h"
#include "gdeflateKernels.h"

namespace nvcomp
{

// The Bitcomp batch decompression outputs bitcompResult_t statuses.
// Need to convert them to nvcompStatus_t.
__global__ void convertGdeflateOutputStatusesKernel(nvcompStatus_t *statuses, size_t batch_size) {
  static_assert(
      sizeof(nvcompStatus_t) == sizeof(gdeflate::gdeflateStatus_t),
      "gdeflate and nvcomp statuses must be the same size");

  size_t index = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (index >= batch_size)
    return;

  auto ier = reinterpret_cast<gdeflate::gdeflateStatus_t *>(statuses)[index];
  nvcompStatus_t nvcomp_err = nvcompSuccess;
  if (ier != gdeflate::gdeflateSuccess)
    nvcomp_err = nvcompErrorCannotDecompress;
  statuses[index] = nvcomp_err;
}

void convertGdeflateOutputStatuses(
    nvcompStatus_t *statuses,
    size_t batch_size,
    cudaStream_t stream) {
    const int threads = 512;
    int blocks = (batch_size - 1) / threads + 1;
    convertGdeflateOutputStatusesKernel<<<blocks,threads,0,stream>>>(statuses, batch_size);
}

} // namespace nvcomp

#endif

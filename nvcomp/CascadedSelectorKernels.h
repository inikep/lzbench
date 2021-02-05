/*
 * Copyright (c) Copyright-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef CASCADEDSELECTORKERNEL_HPP
#define CASCADEDSELECTORKERNEL_HPP

#include "cascaded.hpp"
#include "nvcomp.hpp"

namespace nvcomp
{
  /*
   *@brief Computes the sum of all sample output buffer with all different
   *schemes.
   *@param in The input memory location on the GPU
   *@param sample_offsets, the offsets to compute the starting point for the
   *samples
   *@param sample_bytes The size of the sample memory location on the GPU in
   *bytes
   *@param num_samples The number of samples
   *@param in_type The type of the input elements
   *@param workspace The workspace used internally by kernels
   *@param workspaceSize The size of the workspace size in bytes
   *@param outsize The computed output buffere size (result)
   *@param NUM_SCHEMES The number of schemes.
   *@param stream The stream to execute the kernel on
   **/

  void SamplingFastOption(
      const void* const in,
      size_t* const sample_offsets,
      const size_t sample_bytes,
      const size_t num_samples,
      const nvcompType_t in_type,
      void* const workspace,
      const size_t workspaceSize,
      size_t* outsize,
      int NUM_SCHEMES,
      cudaStream_t stream);

} // namespace nvcomp

#endif

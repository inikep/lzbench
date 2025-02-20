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

#ifndef NVCOMP_UNPACK_H
#define NVCOMP_UNPACK_H

#include <limits>
#include <cstdint>
#include <cstdio>
#include <cassert>

#ifdef __CUDACC__
#define NVCOMP_HOST_DEVICE __device__ __host__
#else
#define NVCOMP_HOST_DEVICE
#endif

namespace nvcomp
{
  
template <typename T>
NVCOMP_HOST_DEVICE T unpackBytes(
    const void* data, const uint8_t numBits, const T minValue, const size_t i)
{
  using U = typename std::make_unsigned<T>::type;

  if (numBits == 0) {
    return minValue;
  } else {
    // enough space to hold 64 bits with up to 7 bit offset
    uint8_t scratch[9];

    // shifting by width of the type is UB
    const U mask = numBits < sizeof(T)*8U ? static_cast<U>((1ULL << numBits) -
        1) : static_cast<U>(-1);
    const uint8_t* byte_data = reinterpret_cast<decltype(byte_data)>(data);

    // Specialized
    // Need to copy into scratch because
    // GPU can only address n byte datatype on multiple of n address
    // TODO: add an optimized version in case numBits aligns to word size
    //       boundaries (1,2,4,8,16,32 and 64 bits)
    size_t start_byte = (i * numBits) / 8;
    // end_byte needed so we don't attempt to read from illegal memory
    size_t end_byte = ((i + 1) * numBits - 1) / 8;
    assert(end_byte - start_byte <= sizeof(scratch));

    for (size_t j = start_byte, k = 0; j <= end_byte; ++j, ++k) {
      scratch[k] = byte_data[j];
    }

    const int bitOffset = (i * numBits) % 8;
    U baseValue = 0;
    for (size_t k = 0; k <= end_byte - start_byte; ++k) {
      U shifted;
      if (k > 0) {
        shifted = static_cast<U>(scratch[k]) << ((k * 8) - bitOffset);
      } else {
        shifted = static_cast<U>(scratch[k]) >> bitOffset;
      }
      baseValue |= mask & shifted;
    }

    const T value = baseValue + minValue;
    return value;
  }
}

}

#undef NVCOMP_HOST_DEVICE

#endif

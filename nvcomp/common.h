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

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace nvcomp
{

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

// returns nano-seconds
inline uint64_t get_time(timespec start, timespec end)
{
  constexpr const uint64_t BILLION = 1000000000ULL;
  const uint64_t elapsed_time
      = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  return elapsed_time;
}

// size in bytes, returns GB/s
inline double gibs(struct timespec start, struct timespec end, size_t s)
{
  uint64_t t = get_time(start, end);
  return (double)s / t * 1e9 / 1024 / 1024 / 1024;
}

// size in bytes, returns GB/s
inline double gbs(struct timespec start, struct timespec end, size_t s)
{
  uint64_t t = get_time(start, end);
  return (double)s / t;
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace
{
template <typename T>
T* align(T* const ptr, const size_t alignment)
{
  const size_t bits = reinterpret_cast<size_t>(ptr);
  const size_t mask = alignment - 1;

  return reinterpret_cast<T*>(((bits - 1) | mask) + 1);
}

template <typename T>
size_t
relativeEndOffset(const void* start, const T* subsection, const size_t length)
{
  std::ptrdiff_t diff = reinterpret_cast<const char*>(subsection)
                        - static_cast<const char*>(start);
  return static_cast<size_t>(diff) + length * sizeof(T);
}

template <typename T = size_t>
T relativeEndOffset(const void* start, const void* subsection)
{
  std::ptrdiff_t diff = reinterpret_cast<const char*>(subsection)
                        - static_cast<const char*>(start);
  return static_cast<T>(diff);
}

template <typename U, typename T>
constexpr __host__ __device__ U roundUpDiv(U const num, T const chunk)
{
  return (num / chunk) + (num % chunk > 0);
}

template <typename U, typename T>
constexpr __host__ __device__ U roundDownTo(U const num, T const chunk)
{
  return (num / chunk) * chunk;
}

template <typename U, typename T>
constexpr __host__ __device__ U roundUpTo(U const num, T const chunk)
{
  return roundUpDiv(num, chunk) * chunk;
}

} // namespace

__inline__ size_t sizeOfnvcompType(nvcompType_t type)
{
  switch (type) {
  case NVCOMP_TYPE_BITS:
    return 1;
  case NVCOMP_TYPE_CHAR:
    return sizeof(int8_t);
  case NVCOMP_TYPE_UCHAR:
    return sizeof(uint8_t);
  case NVCOMP_TYPE_SHORT:
    return sizeof(int16_t);
  case NVCOMP_TYPE_USHORT:
    return sizeof(uint16_t);
  case NVCOMP_TYPE_INT:
    return sizeof(int32_t);
  case NVCOMP_TYPE_UINT:
    return sizeof(uint32_t);
  case NVCOMP_TYPE_LONGLONG:
    return sizeof(int64_t);
  case NVCOMP_TYPE_ULONGLONG:
    return sizeof(uint64_t);
  default:
    throw std::runtime_error("Unsupported type " + std::to_string(type));
  }
}

} // namespace nvcomp

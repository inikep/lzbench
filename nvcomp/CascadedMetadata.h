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

#ifndef CASCACDEDMETADATA_H
#define CASCACDEDMETADATA_H

#include "Metadata.h"
#include "cascaded.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef __CUDACC__
#define NVCOMP_HOST_DEVICE __device__ __host__
#else
#define NVCOMP_HOST_DEVICE
#endif

namespace nvcomp
{

class CascadedMetadata : public Metadata
{
public:
  constexpr static int COMPRESSION_ID = 0x1000;

  constexpr static const size_t MAX_NUM_RLES = 8;

  union MinValue
  {
    MinValue() = default;
    MinValue(const int8_t num) : i8(num)
    {
    }
    MinValue(const uint8_t num) : u8(num)
    {
    }
    MinValue(const int16_t num) : i16(num)
    {
    }
    MinValue(const uint16_t num) : u16(num)
    {
    }
    MinValue(const int32_t num) : i32(num)
    {
    }
    MinValue(const uint32_t num) : u32(num)
    {
    }
    MinValue(const int64_t num) : i64(num)
    {
    }
    MinValue(const uint64_t num) : u64(num)
    {
    }

    int8_t i8;
    uint8_t u8;
    int16_t i16;
    uint16_t u16;
    int32_t i32;
    uint32_t u32;
    int64_t i64;
    uint64_t u64;
  };

  static_assert(std::is_pod<MinValue>::value, "MinValue must be a POD type.");

  struct Header
  {
    uint64_t length;
    MinValue minValue;
    uint8_t numBits;
  };

  /**
   * @brief Create a new metadta object.
   *
   * @param opts The cascaded compression options.
   * @param type The type of data element to compress.
   * @param uncompressedBytes The size of the data while uncompressed.
   * @param compressedBytes The size of the data and metadata compressed.
   */
  CascadedMetadata(
      nvcompCascadedFormatOpts opts,
      nvcompType_t type,
      size_t uncompressedBytes,
      size_t compressedBytes);

  /**
   * @brief Get the number of Run Length Encodings in the scheme.
   *
   * @return The number of Run Length Encodings.
   */
  int getNumRLEs() const;

  /**
   * @brief Get the number of delta encodings in the scheme.
   *
   * @return The number of delta encodings.
   */
  int getNumDeltas() const;

  /**
   * @brief Get the number inputs/layers in the scheme.
   *
   * @return The number of inputs/layers.
   */
  unsigned int getNumInputs() const;

  /**
   * @brief Get whether to use bitpacking or not.
   *
   * @return True if bit packing is used.
   */
  bool useBitPacking() const;

  /**
   * @brief Check if any data offsets have been set.
   *
   * @return True if any of the data offsets have been set.
   */
  bool haveAnyOffsetsBeenSet() const;

  /**
   * @brief Check if all data offsets have been set.
   *
   * @return True if all of the data offsets have been set.
   */
  bool haveAllOffsetsBeenSet() const;

  /**
   * @brief Set the header for the given input/layer.
   *
   * @param index The index of the input/layer.
   * @param header The header to set.
   *
   * @throw An exception if the index is invalid.
   */
  void setHeader(size_t index, Header header);

  /**
   * @brief Set the data offset for the given input/layer. The offset should be
   * from the start of the serialized metadata.
   *
   * @param index The input/layer index.
   * @param offset The offset in bytes.
   *
   * @throw An exception if the index is invalid.
   */
  void setDataOffset(size_t index, size_t offset);

  /**
   * @brief Get the header for the given input/layer.
   *
   * @param index The index of the input/layer.
   *
   * @return The header.
   *
   * @throw An exception if the index is invalid.
   */
  Header getHeader(size_t index) const;

  /**
   * @brief Get the length in elements of a given input/layer.
   *
   * @param index The index of the input/layer.
   *
   * @return The length in elements.
   */
  size_t getNumElementsOf(size_t index) const;

  /**
   * @brief Get the data offset for the given input/layer.
   *
   * @param index The index of the input/layer.
   *
   * @return The offset in bytes.
   *
   * @throw An exception if the given index is invalid, or the offset for that
   * index has not yet been set.
   */
  size_t getDataOffset(size_t index) const;

  /**
   * @brief Check if a given layer is a saved layer (compressed).
   *
   * @param index The index of the input/layer.
   *
   * @return True if the layer's data is saved.
   */
  bool isSaved(size_t index) const;

  /**
   * @brief Get the type associated with the input/layer.
   *
   * @param index The index of the input/layer.
   *
   * @return The type.
   */
  nvcompType_t getDataType(size_t index) const;

  template <typename T>
  static inline NVCOMP_HOST_DEVICE T*
  getMinValueLocation(CascadedMetadata::Header* const deviceHeader);

  /**
   * @brief Get the minimum value associated with a given layer/input. If the
   * given layer input does not contain a bit packing or the template type does
   * not batch the packed type, the return value is unspecified.
   *
   * @tparam T The type of value that is packed.
   * @param index T The index of the layer/input.
   *
   * @return The minimum value of the bitpacking.
   */
  template <typename T>
  T getMinValueOf(const size_t index) const
  {
    Header header = getHeader(index);
    const T minValue = *CascadedMetadata::getMinValueLocation<T>(&header);
    return minValue;
  }

private:
  /**
   * @brief The configuration of cascaded compression to be used/used.
   */
  nvcompCascadedFormatOpts m_formatOpts;

  /**
   * @brief The header for each layer, output or not (that is, this will be the
   * length of `getNumInputs()`). This is stored in the
   * metadata such that the exact size of intermediate layers can be known
   * before decompression starts, allowing it to be done asyncronously, and
   * allowing memory allocations only of the exact sizes needed.
   */
  std::vector<Header> m_headers;

  /**
   * @brief The offsets of data for each layer, output or not (that is,
   * this will be the length of `getNumInputs()`). However, the offset for a
   * non-output layer is unspeficied.
   */
  std::vector<size_t> m_dataOffsets;

  std::vector<nvcompType_t> m_dataType;

  std::vector<bool> m_isSaved;

  /**
   * @brief Setup the offset arrays, after m_formatOpts has been set or
   * changed. This will wipe any previously stored data.
   */
  void initialize();
};

template <>
inline NVCOMP_HOST_DEVICE int8_t* CascadedMetadata::getMinValueLocation<int8_t>(
    CascadedMetadata::Header* const deviceHeader)
{
  return &(deviceHeader->minValue.i8);
}

template <>
inline NVCOMP_HOST_DEVICE uint8_t*
CascadedMetadata::getMinValueLocation<uint8_t>(
    CascadedMetadata::Header* const deviceHeader)
{
  return &(deviceHeader->minValue.u8);
}

template <>
inline NVCOMP_HOST_DEVICE int16_t*
CascadedMetadata::getMinValueLocation<int16_t>(
    CascadedMetadata::Header* const deviceHeader)
{
  return &(deviceHeader->minValue.i16);
}

template <>
inline NVCOMP_HOST_DEVICE uint16_t*
CascadedMetadata::getMinValueLocation<uint16_t>(
    CascadedMetadata::Header* const deviceHeader)
{
  return &(deviceHeader->minValue.u16);
}

template <>
inline NVCOMP_HOST_DEVICE int32_t*
CascadedMetadata::getMinValueLocation<int32_t>(
    CascadedMetadata::Header* const deviceHeader)
{
  return &(deviceHeader->minValue.i32);
}

template <>
inline NVCOMP_HOST_DEVICE uint32_t*
CascadedMetadata::getMinValueLocation<uint32_t>(
    CascadedMetadata::Header* const deviceHeader)
{
  return &(deviceHeader->minValue.u32);
}

template <>
inline NVCOMP_HOST_DEVICE int64_t*
CascadedMetadata::getMinValueLocation<int64_t>(
    CascadedMetadata::Header* const deviceHeader)
{
  return &(deviceHeader->minValue.i64);
}

template <>
inline NVCOMP_HOST_DEVICE uint64_t*
CascadedMetadata::getMinValueLocation<uint64_t>(
    CascadedMetadata::Header* const deviceHeader)
{
  return &(deviceHeader->minValue.u64);
}

} // namespace nvcomp

#undef NVCOMP_HOST_DEVICE

#endif

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

#include "CascadedCommon.h"
#include "CascadedMetadataOnGPU.h"
#include "CudaUtils.h"
#include "common.h"

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace nvcomp
{

/******************************************************************************
 * TYPES **********************************************************************
 *****************************************************************************/

namespace
{
using VERSION_TYPE = uint16_t;
using NUM_RLES_TYPE = uint8_t;
using NUM_DELTAS_TYPE = uint8_t;
using USE_BITPACKING_TYPE = uint8_t;
using COMP_BYTES_TYPE = uint64_t;
using DECOMP_BYTES_TYPE = uint64_t;
using IN_TYPE_TYPE = int32_t;
using NUM_INPUTS_TYPE = int32_t;
using OFFSET_TYPE = uint64_t;
using HEADER_TYPE = CascadedMetadata::Header;
} // namespace

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr const size_t NULL_NUM_INPUTS = static_cast<size_t>(-1);

constexpr const size_t MAX_NUM_RLES = CascadedMetadata::MAX_NUM_RLES;

enum OffsetType : unsigned int
{
  OFFSET_VERSION = 0,
  OFFSET_NUM_RLES
  = roundUpTo(OFFSET_VERSION + sizeof(VERSION_TYPE), sizeof(NUM_RLES_TYPE)),
  OFFSET_NUM_DELTAS
  = roundUpTo(OFFSET_NUM_RLES + sizeof(NUM_RLES_TYPE), sizeof(NUM_DELTAS_TYPE)),
  OFFSET_USE_BITPACKING = roundUpTo(
      OFFSET_NUM_DELTAS + sizeof(NUM_DELTAS_TYPE), sizeof(USE_BITPACKING_TYPE)),
  OFFSET_COMP_BYTES = roundUpTo(
      OFFSET_USE_BITPACKING + sizeof(USE_BITPACKING_TYPE),
      sizeof(COMP_BYTES_TYPE)),
  OFFSET_DECOMP_BYTES = roundUpTo(
      OFFSET_COMP_BYTES + sizeof(COMP_BYTES_TYPE), sizeof(DECOMP_BYTES_TYPE)),
  OFFSET_IN_TYPE = roundUpTo(
      OFFSET_DECOMP_BYTES + sizeof(DECOMP_BYTES_TYPE), sizeof(IN_TYPE_TYPE)),
  OFFSET_NUM_INPUTS
  = roundUpTo(OFFSET_IN_TYPE + sizeof(IN_TYPE_TYPE), sizeof(NUM_INPUTS_TYPE)),
  OFFSET_HEADERS
  = roundUpTo(OFFSET_NUM_INPUTS + sizeof(NUM_INPUTS_TYPE), sizeof(OFFSET_TYPE))
};
} // namespace

/******************************************************************************
 * DEVICE FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

inline constexpr __device__ __host__ unsigned int
serializedMetadataSize(const int numInputs)
{
  return OFFSET_HEADERS + sizeof(OFFSET_TYPE) * numInputs
         + sizeof(HEADER_TYPE) * numInputs;
}

template <typename T, OffsetType offset>
__device__ __host__ void setField(uint8_t* const data, T const v)
{
  *reinterpret_cast<T*>(data + offset) = v;
}

template <typename T, OffsetType offset>
__device__ __host__ void
setField(uint8_t* const data, T const v, const size_t dynamicOffset)
{
  reinterpret_cast<T*>(data + offset)[dynamicOffset] = v;
}

template <typename T, OffsetType offset>
__device__ __host__ T getField(const uint8_t* const data)
{
  return *reinterpret_cast<const T*>(data + offset);
}

template <typename T, OffsetType offset>
__device__ __host__ T
getField(const uint8_t* const data, const size_t dynamicOffset)
{
  return reinterpret_cast<const T*>(data + offset)[dynamicOffset];
}

__device__ __host__ size_t getOffsetsOffset(size_t numInputs)
{
  return roundUpTo(
      OFFSET_HEADERS + sizeof(HEADER_TYPE) * numInputs, sizeof(OFFSET_TYPE));
}

} // namespace

/******************************************************************************
 * KERNELS ********************************************************************
 *****************************************************************************/

namespace
{

__global__ void serializeV1(
    void* const dest,
    const size_t destSize,
    const int numRLEs,
    const int numDeltas,
    const bool useBitPacking,
    const size_t comp_bytes,
    const size_t decomp_bytes,
    const nvcompType_t in_type,
    const int numInputs,
    size_t* const serializedSizeDevice)
{
  using Chunk = uint32_t;

  __shared__ uint8_t localBuffer[serializedMetadataSize(MAX_NUM_RLES + 1)];

  assert(blockIdx.x == 0);

  // master thread assigns local buffer
  if (threadIdx.x == 0) {
    setField<VERSION_TYPE, OFFSET_VERSION>(localBuffer, 1);
    setField<NUM_RLES_TYPE, OFFSET_NUM_RLES>(localBuffer, numRLEs);
    setField<NUM_DELTAS_TYPE, OFFSET_NUM_DELTAS>(localBuffer, numDeltas);
    setField<USE_BITPACKING_TYPE, OFFSET_USE_BITPACKING>(
        localBuffer, useBitPacking);
    setField<COMP_BYTES_TYPE, OFFSET_COMP_BYTES>(localBuffer, comp_bytes);
    setField<DECOMP_BYTES_TYPE, OFFSET_DECOMP_BYTES>(localBuffer, decomp_bytes);
    setField<IN_TYPE_TYPE, OFFSET_IN_TYPE>(localBuffer, in_type);
    setField<NUM_INPUTS_TYPE, OFFSET_NUM_INPUTS>(localBuffer, numInputs);

    if (serializedSizeDevice) {
      *serializedSizeDevice = serializedMetadataSize(numInputs);
    }
  }

  __syncthreads();

  // all threads copy to global memory
  for (int idx = threadIdx.x; idx * sizeof(Chunk) < OFFSET_HEADERS;
       idx += blockDim.x) {
    static_cast<Chunk*>(dest)[idx] = reinterpret_cast<Chunk*>(localBuffer)[idx];
  }
}

__global__ void setOffset(
    void* const serializedMetadata,
    const size_t index,
    const size_t* const offsetDevice)
{
  assert(blockIdx.x == 0);
  assert(threadIdx.x == 0);

  uint8_t* const serializedBytes = static_cast<uint8_t*>(serializedMetadata);

  const int numInputs
      = getField<NUM_INPUTS_TYPE, OFFSET_NUM_INPUTS>(serializedBytes);

  // dataOffsets
  setField<OFFSET_TYPE, static_cast<OffsetType>(0)>(
      serializedBytes + getOffsetsOffset(numInputs),
      static_cast<OFFSET_TYPE>(*offsetDevice),
      index);
}

} // namespace

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

template <typename T>
constexpr bool isFixedWidth()
{
  return std::is_same<T, char>::value || std::is_same<T, int8_t>::value
         || std::is_same<T, uint8_t>::value || std::is_same<T, int16_t>::value
         || std::is_same<T, uint16_t>::value || std::is_same<T, int32_t>::value
         || std::is_same<T, uint32_t>::value || std::is_same<T, int64_t>::value
         || std::is_same<T, uint64_t>::value;
}

template <typename T>
size_t readFixedWidthData(
    T* const val,
    const void* const ptr,
    const size_t offset,
    const size_t maxSize)
{
  assert(isFixedWidth<T>());

  size_t newOffset = offset + sizeof(*val);

  if (newOffset > maxSize) {
    throw std::runtime_error(
        "Not enough room to read member, need at least "
        + std::to_string(newOffset) + " bytes, but given only "
        + std::to_string(maxSize));
  }

  *val = *reinterpret_cast<const T*>(static_cast<const char*>(ptr) + offset);

  return newOffset;
}

CascadedMetadata deserializeMetadataFromGPUVersion1(
    const void* const devicePtr, const size_t size, cudaStream_t stream)
{
  NUM_INPUTS_TYPE numInputs;
  CudaUtils::copy_async(
      &numInputs,
      (const NUM_INPUTS_TYPE*)(static_cast<const uint8_t*>(devicePtr) + OFFSET_NUM_INPUTS),
      1,
      DEVICE_TO_HOST,
      stream);
  CudaUtils::sync(stream);

  std::vector<uint8_t> localBuffer(serializedMetadataSize(numInputs));
  if (size < localBuffer.size()) {
    throw std::runtime_error(
        "Insufficient space to deserialize metadata "
        "from: "
        + std::to_string(size) + " but require "
        + std::to_string(localBuffer.size()));
  }

  CudaUtils::copy_async(
      (uint8_t*)localBuffer.data(),
      (const uint8_t*)devicePtr,
      localBuffer.size(),
      DEVICE_TO_HOST,
      stream);
  CudaUtils::sync(stream);

  // here we convert to types of fixed width by the C++ standard rather than
  // just doing a memcpy of the struct, to ensure portability.

  nvcompCascadedFormatOpts format_opts;
  format_opts.num_RLEs
      = getField<NUM_RLES_TYPE, OFFSET_NUM_RLES>(localBuffer.data());
  format_opts.num_deltas
      = getField<NUM_DELTAS_TYPE, OFFSET_NUM_DELTAS>(localBuffer.data());
  format_opts.use_bp = getField<USE_BITPACKING_TYPE, OFFSET_USE_BITPACKING>(
      localBuffer.data());
  const COMP_BYTES_TYPE comp_bytes
      = getField<COMP_BYTES_TYPE, OFFSET_COMP_BYTES>(localBuffer.data());
  const DECOMP_BYTES_TYPE decomp_bytes
      = getField<DECOMP_BYTES_TYPE, OFFSET_DECOMP_BYTES>(localBuffer.data());
  const IN_TYPE_TYPE in_type
      = getField<IN_TYPE_TYPE, OFFSET_IN_TYPE>(localBuffer.data());

  CascadedMetadata metadata(
      format_opts,
      static_cast<nvcompType_t>(in_type),
      decomp_bytes,
      comp_bytes);

  if (numInputs != static_cast<int>(metadata.getNumInputs())) {
    throw std::runtime_error(
        "Mismatch in numInputs while deserializing "
        "metadata: "
        + std::to_string(numInputs) + " vs. "
        + std::to_string(metadata.getNumInputs()));
  }

  std::vector<OFFSET_TYPE> hdrOffsets(metadata.getNumInputs());
  std::vector<OFFSET_TYPE> dataOffsets(metadata.getNumInputs());

  for (size_t i = 0; i < metadata.getNumInputs(); ++i) {
    const HEADER_TYPE header
        = getField<HEADER_TYPE, OFFSET_HEADERS>(localBuffer.data(), i);
    metadata.setHeader(i, header);
  }
  for (size_t i = 0; i < metadata.getNumInputs(); ++i) {
    const OFFSET_TYPE dataOffset
        = getField<OFFSET_TYPE, static_cast<OffsetType>(0)>(
            localBuffer.data() + getOffsetsOffset(metadata.getNumInputs()), i);
    metadata.setDataOffset(i, dataOffset);
  }

  return metadata;
}

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

size_t
CascadedMetadataOnGPU::getSerializedSizeOf(const CascadedMetadata& metadata)
{
  return serializedMetadataSize(metadata.getNumInputs());
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

CascadedMetadataOnGPU::CascadedMetadataOnGPU(void* const ptr, size_t maxSize) :
    m_ptr(ptr),
    m_maxSize(maxSize),
    m_numInputs(NULL_NUM_INPUTS)
{
  if (ptr == nullptr) {
    throw std::runtime_error("Location given to CascadedMetadataOnGPU() must "
                             "be a valid pointer.");
  } else if (m_maxSize < OFFSET_HEADERS) {
    throw std::runtime_error(
        "Maximum size given to CascadedMetdataOnGPU() "
        "must be greater than "
        + std::to_string(OFFSET_HEADERS) + " bytes.");
  }
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void CascadedMetadataOnGPU::copyToGPU(
    const CascadedMetadata& metadata,
    size_t* const serializedSizeDevice,
    cudaStream_t stream)
{
  const size_t requiredSize = serializedMetadataSize(metadata.getNumInputs());
  if (m_maxSize < requiredSize) {
    throw std::runtime_error(
        "Invalid space for serialized metadata on GPU: "
        + std::to_string(m_maxSize) + " when " + std::to_string(requiredSize)
        + " is needed.");
  }

  serializeV1<<<1, 64, 0, stream>>>(
      m_ptr,
      m_maxSize,
      metadata.getNumRLEs(),
      metadata.getNumDeltas(),
      metadata.useBitPacking(),
      metadata.getCompressedSize(),
      metadata.getUncompressedSize(),
      metadata.getValueType(),
      metadata.getNumInputs(),
      serializedSizeDevice);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to launch metadata serialization kernel: "
        + std::to_string(err));
  }

  if (metadata.haveAnyOffsetsBeenSet()) {
    if (!metadata.haveAllOffsetsBeenSet()) {
      throw std::runtime_error(
          "Only some offsets have been set before calling "
          "CascadedMetadataOnGPU::copyToGPU(). This is not "
          "a valid state for copying CascadedMetadata to the GPU.");
    }

    for (size_t i = 0; i < metadata.getNumInputs(); ++i) {
      const HEADER_TYPE header = metadata.getHeader(i);
      CudaUtils::copy_async(
          (HEADER_TYPE*)(static_cast<uint8_t*>(m_ptr) + OFFSET_HEADERS + i * sizeof(header)),
          &header,
          1,
          HOST_TO_DEVICE,
          stream);
    }

    for (size_t i = 0; i < metadata.getNumInputs(); ++i) {
      const OFFSET_TYPE offset = metadata.getDataOffset(i);
      CudaUtils::copy_async(
          (OFFSET_TYPE*)(static_cast<uint8_t*>(m_ptr)
              + getOffsetsOffset(metadata.getNumInputs())
              + sizeof(OFFSET_TYPE) * i),
          &offset,
          1,
          HOST_TO_DEVICE,
          stream);
    }
  }

  // if successful, set the number of inputs
  m_numInputs = metadata.getNumInputs();
}

void CascadedMetadataOnGPU::copyToGPU(
    const CascadedMetadata& metadata, cudaStream_t stream)
{
  copyToGPU(metadata, nullptr, stream);
}

size_t CascadedMetadataOnGPU::getSerializedSize() const
{
  verifyInitialized();

  return serializedMetadataSize(m_numInputs);
}

void CascadedMetadataOnGPU::saveOffset(
    size_t index, const size_t* offsetDevice, cudaStream_t stream)
{
  verifyInitialized();

  if (index >= m_numInputs) {
    throw std::runtime_error(
        "Invalid input index " + std::to_string(index) + " / "
        + std::to_string(m_numInputs)
        + " given to "
          "CascadedMetadataOnGPU::saveOffsets().");
  }

  setOffset<<<1, 1, 0, stream>>>(m_ptr, index, offsetDevice);
}

void CascadedMetadataOnGPU::setCompressedSizeFromGPU(
    const size_t* sizeOnDevice, cudaStream_t stream)
{
  // TODO: re-write so that we don't depend on 64-bit architecture
  static_assert(
      sizeof(size_t) == sizeof(COMP_BYTES_TYPE),
      "Requires size_t be 64 bits wide.");

  COMP_BYTES_TYPE* const compBytesDevice = reinterpret_cast<COMP_BYTES_TYPE*>(
      static_cast<char*>(m_ptr) + OFFSET_COMP_BYTES);

  cudaError_t err = cudaMemcpyAsync(
      compBytesDevice,
      sizeOnDevice,
      sizeof(COMP_BYTES_TYPE),
      cudaMemcpyDeviceToDevice,
      stream);

  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Async memcpy in "
        "CascadedMetadataOnGPU::setCompressedSizeFromGPU() failed with: "
        + std::to_string(err));
  }
}

CascadedMetadata CascadedMetadataOnGPU::copyToHost(cudaStream_t stream)
{
  // read the version of the serialized metadata.
  VERSION_TYPE version;

  CudaUtils::copy_async(
      &version,
      static_cast<const VERSION_TYPE*>(m_ptr),
      1,
      DEVICE_TO_HOST,
      stream);
  CudaUtils::sync(stream);

  if (version == 1) {
    CascadedMetadata metadata
        = deserializeMetadataFromGPUVersion1(m_ptr, m_maxSize, stream);
    m_numInputs = metadata.getNumInputs();

    return metadata;
  } else {
    throw std::runtime_error(
        "Unsupported Metadata version: " + std::to_string(version));
  }
}

CascadedMetadata::Header*
CascadedMetadataOnGPU::getHeaderLocation(const size_t index)
{
  verifyIndex(index);

  return reinterpret_cast<HEADER_TYPE*>(
             static_cast<uint8_t*>(m_ptr) + OFFSET_HEADERS)
         + index;
}

const CascadedMetadata::Header*
CascadedMetadataOnGPU::getHeaderLocation(const size_t index) const
{
  verifyIndex(index);

  return reinterpret_cast<const HEADER_TYPE*>(
             static_cast<const uint8_t*>(m_ptr) + OFFSET_HEADERS)
         + index;
}

/******************************************************************************
 * PRIVATE METHODS ************************************************************
 *****************************************************************************/

void CascadedMetadataOnGPU::verifyInitialized() const
{
  if (m_numInputs == NULL_NUM_INPUTS) {
    throw std::runtime_error("CascadedMetadataOnGPU::copyToGPU() or "
                             "CascadedMetdataOnGPU::copyToHost() must be "
                             "called before other methods.");
  }
}

void CascadedMetadataOnGPU::verifyIndex(const size_t index) const
{
  verifyInitialized();

  if (index >= m_numInputs) {
    throw std::runtime_error(
        "Invalid index to set header for: " + std::to_string(index) + " / "
        + std::to_string(m_numInputs));
  }
}

} // namespace nvcomp

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

#include "CascadedCompressionGPU.h"

#include "BitPackGPU.h"
#include "CascadedMetadata.h"
#include "CascadedMetadataOnGPU.h"
#include "Check.h"
#include "CudaUtils.h"
#include "DeltaGPU.h"
#include "RunLengthEncodeGPU.h"
#include "TempSpaceBroker.h"
#include "nvcomp.h"
#include "nvcomp.hpp"
#include "type_macros.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nvcomp
{

/******************************************************************************
 * KERNELS ********************************************************************
 *****************************************************************************/

namespace
{

template <typename T>
__global__ void dereferenceDevice(T* const outValue, T* const* const ref)
{
  assert(threadIdx.x == 0);
  assert(blockIdx.x == 0);

  *outValue = **ref;
}

template <typename T>
__global__ void configureBitPackHeader(
    CascadedMetadata::Header* const header,
    T** const minValueDevicePtr,
    unsigned char** const numBitsDevicePtr)
{
  // setup the header and pointers into it
  assert(blockIdx.x == 0);
  assert(threadIdx.x == 0);

  *minValueDevicePtr = CascadedMetadata::getMinValueLocation<T>(header);
  *numBitsDevicePtr = &header->numBits;
}

/**
 * @brief Asynchronously perform a device to device copy, where the destination
 * address and number of elements to copy are stored on the device.
 *
 * @tparam T The type of element to copy.
 * @tparam BLOCK_SIZE The size of each thread block.
 * @param destDPtr The pointer to the destination address to copy elements to,
 * stored on the device.
 * @param src The source address to copy elements from.
 * @param numElementsDPtr The number of elements to copy, stored on the device.
 */
template <typename T, int BLOCK_SIZE>
__global__ void deferredCopy(
    T** const destDPtr, const T* const src, const size_t* const numElementsDPtr)
{
  assert(blockDim.x == BLOCK_SIZE);

  T* const dest = *destDPtr;

  const size_t num = *numElementsDPtr;

  for (int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < num;
       idx += gridDim.x * BLOCK_SIZE) {
    dest[idx] = src[idx];
  }
}

/**
 * @brief Asynchronously perform a device to device copy, where the number of
 * elements to copy is stored on the device.
 *
 * @tparam T The type of element to copy.
 * @tparam BLOCK_SIZE The size of each thread block to use.
 * @param dest The destination address to copy to.
 * @param src The source address to copy from.
 * @param numElementsDPtr The number of elements to copy, stored on the device.
 */
template <typename T, int BLOCK_SIZE>
__global__ void deferredCopy(
    T* const dest, const T* const src, const size_t* const numElementsDPtr)
{
  assert(blockDim.x == BLOCK_SIZE);

  const size_t num = *numElementsDPtr;

  for (int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < num;
       idx += gridDim.x * BLOCK_SIZE) {
    dest[idx] = src[idx];
  }
}

template <typename T>
__global__ void
offsetPointerAsync(T* const src, T** const dst, const size_t* const offset)
{
  assert(threadIdx.x == 0);
  assert(blockIdx.x == 0);

  *dst = src + *offset;
}

__global__ void offsetAndAlignPointerAsync(
    void* const src, void** const dst, size_t* const offset)
{
  assert(threadIdx.x == 0);
  assert(blockIdx.x == 0);

  // update the offset if we need to
  const size_t unalignedOffset = *offset;
  const size_t alignedOffset = roundUpTo(unalignedOffset, sizeof(size_t));
  if (alignedOffset != unalignedOffset) {
    *offset = alignedOffset;
  }

  *dst = static_cast<char*>(src) + alignedOffset;
}

template <typename VALUE, typename RUN>
__global__ void configTempSpacePointers(
    VALUE* const vals,
    VALUE** const valsPtr,
    RUN* const runs,
    RUN** const runsPtr,
    VALUE* const delta,
    VALUE** const deltaPtr)
{
  assert(threadIdx.x == 0);
  assert(blockIdx.x == 0);

  *valsPtr = vals;
  *runsPtr = runs;
  *deltaPtr = delta;
}

template <typename T>
__global__ void increaseOffsetByBitPacking(
    size_t* const offsetDevice, const CascadedMetadata::Header* const header)
{
  assert(threadIdx.x == 0);
  assert(blockIdx.x == 0);

  const size_t temp_size = roundUpTo(
      roundUpDiv(header->length * header->numBits, 8ULL), sizeof(T));

  *offsetDevice += temp_size;
}

template <typename T>
__global__ void increaseOffsetByRaw(
    size_t* const offsetDevice, const CascadedMetadata::Header* const header)
{
  assert(threadIdx.x == 0);
  assert(blockIdx.x == 0);

  const size_t temp_size = header->length * sizeof(T);

  *offsetDevice += temp_size;
}

/**
 * @brief This kernel allows copying to the device from a stack variable
 * asynchronously.
 *
 * @tparam T The type of variable to copy.
 * @param hostValue The value to copy.
 * @param deviceValue The location to copy to.
 */
template <typename T>
__global__ void asyncPODCopyKernel(const T hostValue, T* const deviceValue)
{
  static_assert(std::is_pod<T>::value, "Must be a POD to do async copy.");

  assert(threadIdx.x == 0);
  assert(blockIdx.x == 0);

  *deviceValue = hostValue;
}

} // namespace

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

void checkAlignmentOf(void* const ptr, const size_t alignment)
{
  void* aligned_ptr = ptr;
  size_t space = alignment;
  if (std::align(alignment, alignment, aligned_ptr, space) == nullptr
      || ptr != aligned_ptr) {
    std::ostringstream oss;
    oss << ptr;
    throw std::runtime_error(
        "Incorrectly aligned buffer: " + oss.str() + ", should be aligned to "
        + std::to_string(alignment));
  }
}

/**
 * @brief This copies the input to the device from a stack variable
 * asynchronously. While this is inefficient, it is better than synchronizing or
 * pinning the variable.
 *
 * @tparam T The type of variable to copy.
 * @param hostValue The value to copy.
 * @param deviceValue The location to copy to.
 */
template <typename T>
void asyncPODCopy(const T& value, T* const destination, cudaStream_t stream)
{
  asyncPODCopyKernel<<<dim3(1), dim3(1), 0, stream>>>(value, destination);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to launch asyncPODCopyKernel "
        "kernel: "
        + std::to_string(err));
  }
}

/**
 * @brief Bit pack or copy the elements to an output address.
 *
 * @tparam T The type of element to pack/copy.
 * @param headerDPtr The header, stored on the device.
 * @param temp_ptr The temporary workspace allocated (on the device).
 * @param temp_bytes The size of the temporary workspace.
 * @param outputDPtr The pointer to the location to output the elements to (on
 * the device), stored on the device.
 * @param input The input elements (on the device).
 * @param numElementsDPtr The pointer to the number of elements, stored on the
 * device.
 * @param maxNum The maximum number of elements.
 * @param offsetDPtr The current offset output, to be increased by
 * the number of bytes written by this function.
 * @param bitPacking Whether or not to perform bitpacking on this data.
 * @param stream The stream to asynchronously perform work on.
 */
template <typename T>
void packToOutput(
    CascadedMetadata::Header* const headerDPtr,
    void* const temp_ptr,
    const size_t temp_bytes,
    void** const outputDPtr,
    const T* const input,
    const size_t* const numElementsDPtr,
    const size_t maxNum,
    size_t* const offsetDPtr,
    const bool bitPacking,
    cudaStream_t stream)
{
  CudaUtils::copy_async(
      &(headerDPtr->length), numElementsDPtr, 1, DEVICE_TO_DEVICE, stream);

  if (bitPacking) {
    TempSpaceBroker tempSpace(temp_ptr, temp_bytes);

    void** bitPackOutputPtr;
    void** minValueDevicePtr;
    unsigned char** numBitsDevicePtr;
    tempSpace.reserve(&bitPackOutputPtr, 1);
    tempSpace.reserve(&minValueDevicePtr, 1);
    tempSpace.reserve(&numBitsDevicePtr, 1);

    configureBitPackHeader<<<1, 1, 0, stream>>>(
        headerDPtr, reinterpret_cast<T**>(minValueDevicePtr), numBitsDevicePtr);

    void* const packTemp = reinterpret_cast<void*>(numBitsDevicePtr + 1);
    const size_t packTempSize
        = temp_bytes
          - (static_cast<char*>(packTemp) - static_cast<char*>(temp_ptr));

    BitPackGPU::compress(
        packTemp,
        packTempSize,
        getnvcompType<T>(),
        outputDPtr,
        input,
        numElementsDPtr,
        maxNum,
        minValueDevicePtr,
        numBitsDevicePtr,
        stream);

    increaseOffsetByBitPacking<T><<<1, 1, 0, stream>>>(offsetDPtr, headerDPtr);
  } else {
    constexpr const int BLOCK_SIZE = 512;

    const dim3 grid(std::min(1024, roundUpDiv<int, int>(maxNum, BLOCK_SIZE)));
    const dim3 block(BLOCK_SIZE);

    deferredCopy<T, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<T**>(outputDPtr), input, numElementsDPtr);

    increaseOffsetByRaw<T><<<1, 1, 0, stream>>>(offsetDPtr, headerDPtr);
  }
}

template <typename valT, typename runT>
void generateTypedOutputUpperBound(
    const void* const /*in_ptr*/,
    const size_t in_bytes,
    const nvcompCascadedFormatOpts* const opts,
    void* const temp_ptr,
    const size_t temp_bytes,
    size_t* const out_bytes)
{
  if (temp_bytes > 0) {
    CHECK_NOT_NULL(temp_ptr);

    // only check if its non-null
    checkAlignmentOf(temp_ptr, sizeof(size_t));
  }

  CascadedMetadata metadata(*opts, getnvcompType<valT>(), in_bytes, 0);

  const int numRLEs = metadata.getNumRLEs();
  const int numDeltas = metadata.getNumDeltas();
  const bool bitPacking = metadata.useBitPacking();

  // assume single chunk for now
  // TODO: implement a multi-chunk version
  const size_t outputSize = in_bytes / sizeof(valT);
  assert(outputSize * sizeof(valT) == in_bytes);
  int vals_id = 0;

  // initialize config
  nvcompType_t type = getnvcompType<valT>();
  nvcompIntConfig_t* config = createConfig(&metadata);

  // First past - set layers assume nothing actual compresses.
  // TODO: This will be a
  // gross over estimation of the output size, but the better option would
  // be to probably just assume 1:1 output/input, and error out during
  // compression if we fail to achieve that (maybe just set RLE, Delta, and BP
  // to 0, and do a memcpy, so that user's wont have to handle the error case
  // in their code).

  // A step can be RLE+Delta, RLE, or Delta, with final outputs conditionally
  // having bit packing applied
  const int numSteps = std::max(numRLEs, numDeltas);
  for (int r = numSteps - 1; r >= 0; r--) {
    const int inputId = vals_id;
    if (numSteps - r - 1 < numRLEs) {
      const int runId = ++vals_id;
      const int valId = ++vals_id;

      nvcompConfigAddRLE_BP(
          config,
          inputId,
          outputSize,
          valId,
          type,
          bitPacking,
          runId,
          type,
          bitPacking);

      // store vals (apply delta if necessary)
      if (numRLEs - 1 - r < numDeltas) {
        const int deltaId = ++vals_id;
        if (r == 0) {
          nvcompConfigAddDelta_BP(
              config, valId, outputSize, deltaId, type, bitPacking);
        } else {
          nvcompConfigAddDelta_BP(
              config,
              deltaId,
              outputSize,
              valId,
              type,
              0); // no bitpacking when delta is used as an intermediate step
        }
      }
    } else {
      // RLE-less step
      const int deltaId = ++vals_id;

      if (r == 0) {
        nvcompConfigAddDelta_BP(
            config, inputId, outputSize, deltaId, type, bitPacking);
      } else {
        nvcompConfigAddDelta_BP(
            config,
            deltaId,
            outputSize,
            inputId,
            type,
            0); // no bitpacking when delta is used as an intermediate step
      }
    }
  }

  destroyConfig(config);

  // we will abort compression if we can't fit into out_bytes.
  const size_t serializedMetadataSize
      = CascadedMetadataOnGPU::getSerializedSizeOf(metadata);

  // This may be overkill, as most datatypes we use are aligned to size_t,
  // which on x86_64 is 8 bytes, where as this will be 16 bytes. In theory a
  // smart compiler could potentially generate instructions for some of our
  // structure that at 16-byte aligned.
  const size_t wordSize = alignof(std::max_align_t);

  // space for metadata, each set of 'runs', one set of 'vals'.
  *out_bytes = roundUpTo(serializedMetadataSize, wordSize)
               + roundUpTo(sizeof(runT) * outputSize, wordSize) * numRLEs
               + roundUpTo(sizeof(valT) * outputSize, wordSize);
}

template <typename valT, typename runT>
void compressTypedAsync(
    const void* const in_ptr,
    const size_t in_bytes,
    const nvcompCascadedFormatOpts* const format_opts,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  const nvcompType_t type = getnvcompType<valT>();

  CascadedMetadata metadata(*format_opts, type, in_bytes, 0);

  const int numRLEs = metadata.getNumRLEs();
  const int numDeltas = metadata.getNumDeltas();
  const bool bitPacking = metadata.useBitPacking();

  // assume single chunk for now
  // TODO: implement a multi-chunk version
  const size_t maxNum = in_bytes / sizeof(valT);
  int vals_id = 0;

  TempSpaceBroker tempSpace(temp_ptr, temp_bytes);

  size_t* offsetDevice;
  tempSpace.reserve(&offsetDevice, 1);

  CascadedMetadataOnGPU metadataOnGPU(out_ptr, *out_bytes);

  metadataOnGPU.copyToGPU(metadata, offsetDevice, stream);

  valT* vals_delta = nullptr;
  valT* vals_output = nullptr;
  runT* runs_output = nullptr;

  if (numRLEs > 0 || numDeltas > 0) {
    tempSpace.reserve(&vals_output, maxNum);
    if (numRLEs > 0) {
      tempSpace.reserve(&runs_output, maxNum);
    }
    tempSpace.reserve(&vals_delta, maxNum);
  }

  size_t* numRunsDevice;
  size_t* outputSizePtr;
  tempSpace.reserve(&numRunsDevice, 1);
  tempSpace.reserve(&outputSizePtr, 1);

  runT** runs_output_ptr;
  valT** vals_output_ptr;
  valT** vals_delta_ptr;
  tempSpace.reserve(&runs_output_ptr, 1);
  tempSpace.reserve(&vals_output_ptr, 1);
  tempSpace.reserve(&vals_delta_ptr, 1);

  void** bit_out_ptr;
  tempSpace.reserve(&bit_out_ptr, 1);

  cudaError_t* statusDevice;
  tempSpace.reserve(&statusDevice, 1);

  configTempSpacePointers<<<1, 1, 0, stream>>>(
      vals_output,
      vals_output_ptr,
      runs_output,
      runs_output_ptr,
      vals_delta,
      vals_delta_ptr);

  // Set first offset to end of metadata
  metadataOnGPU.saveOffset(vals_id, offsetDevice, stream);

  // Second pass - perform compression and store in the memory allocated above.

  // A step can be RLE+Delta, RLE, or Delta, with final outputs conditionally
  // having bit packing applied
  const int numSteps = std::max(numRLEs, numDeltas);
  for (int r = numSteps - 1; r >= 0; r--) {
    int nextValId;
    const bool firstLayer = r == std::max(numRLEs - 1, numDeltas - 1);
    const valT* const vals_input
        = firstLayer ? static_cast<const valT*>(in_ptr) : vals_delta;

    if (numSteps - r - 1 < numRLEs) {
      const int runId = ++vals_id;
      const int valId = ++vals_id;

      // rle always first
      if (firstLayer) {
        RunLengthEncodeGPU::compress(
            tempSpace.next(),
            tempSpace.spaceLeft(),
            getnvcompType<valT>(),
            vals_output,
            getnvcompType<runT>(),
            runs_output,
            numRunsDevice,
            vals_input,
            maxNum,
            stream);
      } else {
        RunLengthEncodeGPU::compressDownstream(
            tempSpace.next(),
            tempSpace.spaceLeft(),
            getnvcompType<valT>(),
            (void**)vals_output_ptr,
            getnvcompType<runT>(),
            (void**)runs_output_ptr,
            numRunsDevice,
            vals_input,
            outputSizePtr,
            maxNum,
            stream);
      }

      // save initial offset
      CascadedMetadata::Header* const valHdr
          = metadataOnGPU.getHeaderLocation(valId);
      CudaUtils::copy_async(
          &(valHdr->length), numRunsDevice, 1, DEVICE_TO_DEVICE, stream);

      metadataOnGPU.saveOffset(valId, offsetDevice, stream);

      CascadedMetadata::Header* const runHdr
          = metadataOnGPU.getHeaderLocation(runId);
      CudaUtils::copy_async(
          &(runHdr->length), numRunsDevice, 1, DEVICE_TO_DEVICE, stream);

      // store vals (apply delta if necessary)
      if (numRLEs - 1 - r < numDeltas) {
        DeltaGPU::compress(
            tempSpace.next(),
            tempSpace.spaceLeft(),
            getnvcompType<valT>(),
            (void**)vals_delta_ptr,
            vals_output,
            numRunsDevice,
            maxNum,
            stream);

        const int id = ++vals_id;
        nextValId = id;

        CascadedMetadata::Header* const hdr
            = metadataOnGPU.getHeaderLocation(id);
        CudaUtils::copy_async(
            &(hdr->length), numRunsDevice, 1, DEVICE_TO_DEVICE, stream);

        metadataOnGPU.saveOffset(id, offsetDevice, stream);
      } else {
        constexpr const int COPY_BLOCK_SIZE = 512;
        const dim3 grid(std::min(
            4096, static_cast<int>(roundUpDiv(maxNum, COPY_BLOCK_SIZE))));
        const dim3 block(COPY_BLOCK_SIZE);

        deferredCopy<valT, COPY_BLOCK_SIZE><<<grid, block, 0, stream>>>(
            vals_delta, vals_output, numRunsDevice);

        nextValId = valId;
      }

      offsetAndAlignPointerAsync<<<1, 1, 0, stream>>>(
          out_ptr, bit_out_ptr, offsetDevice);

      metadataOnGPU.saveOffset(runId, offsetDevice, stream);

      // pack runs into bytes
      packToOutput(
          metadataOnGPU.getHeaderLocation(runId),
          tempSpace.next(),
          tempSpace.spaceLeft(),
          bit_out_ptr,
          runs_output,
          numRunsDevice,
          maxNum,
          offsetDevice,
          bitPacking,
          stream);
    } else {
      if (!firstLayer) {
        CudaUtils::copy_async(
            numRunsDevice, outputSizePtr, 1, DEVICE_TO_DEVICE, stream);
      } else {
        CudaUtils::copy_async(
            numRunsDevice, &maxNum, 1, HOST_TO_DEVICE, stream);
      }

      // No RLE
      DeltaGPU::compress(
          tempSpace.next(),
          tempSpace.spaceLeft(),
          getnvcompType<valT>(),
          (void**)vals_output_ptr,
          vals_input,
          numRunsDevice,
          maxNum,
          stream);

      // we need to copy the delta to final delta buffer
      {
        constexpr const int COPY_BLOCK_SIZE = 512;
        const dim3 grid(std::min(
            4096, static_cast<int>(roundUpDiv(maxNum, COPY_BLOCK_SIZE))));
        const dim3 block(COPY_BLOCK_SIZE);

        deferredCopy<valT, COPY_BLOCK_SIZE><<<grid, block, 0, stream>>>(
            vals_delta, vals_output, numRunsDevice);
      }

      const int id = ++vals_id;
      nextValId = id;

      CascadedMetadata::Header* const hdr = metadataOnGPU.getHeaderLocation(id);
      CudaUtils::copy_async(
          &(hdr->length), numRunsDevice, 1, DEVICE_TO_DEVICE, stream);
      metadataOnGPU.saveOffset(id, offsetDevice, stream);
    }
    if (r == 0) {
      offsetAndAlignPointerAsync<<<1, 1, 0, stream>>>(
          out_ptr, bit_out_ptr, offsetDevice);

      metadataOnGPU.saveOffset(nextValId, offsetDevice, stream);

      // pack runs into bytes
      packToOutput(
          metadataOnGPU.getHeaderLocation(nextValId),
          tempSpace.next(),
          tempSpace.spaceLeft(),
          bit_out_ptr,
          vals_delta,
          numRunsDevice,
          maxNum,
          offsetDevice,
          bitPacking,
          stream);
    } else {
      // update current RLE size
      CudaUtils::copy_async(
          outputSizePtr, numRunsDevice, 1, DEVICE_TO_DEVICE, stream);
    }
  }

  // If there are no RLEs or Deltas, we will do a single BP step.
  if (numRLEs == 0 && numDeltas == 0) {
    const int nextValId = ++vals_id;
    const valT* const vals_input = static_cast<const valT*>(in_ptr);

    CudaUtils::copy_async(numRunsDevice, &maxNum, 1, HOST_TO_DEVICE, stream);

    offsetAndAlignPointerAsync<<<1, 1, 0, stream>>>(
        out_ptr, bit_out_ptr, offsetDevice);

    metadataOnGPU.saveOffset(nextValId, offsetDevice, stream);

    // pack runs into bytes
    packToOutput(
        metadataOnGPU.getHeaderLocation(nextValId),
        tempSpace.next(),
        tempSpace.spaceLeft(),
        bit_out_ptr,
        vals_input,
        numRunsDevice,
        maxNum,
        offsetDevice,
        bitPacking,
        stream);
  }

  // async copy output
  metadataOnGPU.setCompressedSizeFromGPU(offsetDevice, stream);
  CudaUtils::copy_async(out_bytes, offsetDevice, 1, DEVICE_TO_HOST, stream);
}

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

void nvcompCascadedCompressionGPU::computeWorkspaceSize(
    const void* /*in_ptr*/,
    const size_t in_bytes,
    const nvcompType_t in_type,
    const nvcompCascadedFormatOpts* const opts,
    size_t* const temp_bytes)
{
  size_t kernelBytes = 0;

  // get at least enough for intermediate gpu values
  size_t ioBytes = 1024;

  const size_t numIn = in_bytes / sizeOfnvcompType(in_type);
  const nvcompType_t runType = selectRunsType(numIn);

  if (opts->use_bp) {
    // max of runs and values
    kernelBytes = std::max(
        kernelBytes, BitPackGPU::requiredWorkspaceSize(numIn, in_type));
    kernelBytes = std::max(
        kernelBytes, BitPackGPU::requiredWorkspaceSize(numIn, runType));
  }

  if (opts->num_deltas > 0) {
    kernelBytes = std::max(
        kernelBytes, DeltaGPU::requiredWorkspaceSize(numIn, in_type));
  }

  if (opts->num_RLEs > 0) {
    kernelBytes = std::max(
        kernelBytes,
        RunLengthEncodeGPU::requiredWorkspaceSize(numIn, in_type, runType));

    ioBytes += (2 * in_bytes) + numIn * sizeOfnvcompType(runType);
  } else if (opts->num_deltas > 0) {
    ioBytes += 2 * in_bytes;
  }

  *temp_bytes = kernelBytes + ioBytes;
}

void nvcompCascadedCompressionGPU::generateOutputUpperBound(
    const void* const in_ptr,
    const size_t in_bytes,
    const nvcompType_t in_type,
    const nvcompCascadedFormatOpts* const opts,
    void* const temp_ptr,
    const size_t temp_bytes,
    size_t* const out_bytes)
{
  CHECK_NOT_NULL(in_ptr);
  CHECK_NOT_NULL(opts);
  if (temp_bytes > 0) {
    CHECK_NOT_NULL(temp_ptr);
  }
  CHECK_NOT_NULL(out_bytes);

  const nvcompType_t countType
      = selectRunsType(in_bytes / sizeOfnvcompType(in_type));

  NVCOMP_TYPE_TWO_SWITCH(
      in_type,
      countType,
      generateTypedOutputUpperBound,
      in_ptr,
      in_bytes,
      opts,
      temp_ptr,
      temp_bytes,
      out_bytes);
}

void nvcompCascadedCompressionGPU::compressAsync(
    const void* const in_ptr,
    const size_t in_bytes,
    const nvcompType_t in_type,
    const nvcompCascadedFormatOpts* const cascadedOpts,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  CHECK_NOT_NULL(in_ptr);
  CHECK_NOT_NULL(cascadedOpts);
  CHECK_NOT_NULL(temp_ptr);
  CHECK_NOT_NULL(out_ptr);
  CHECK_NOT_NULL(out_bytes);

  checkAlignmentOf(out_ptr, sizeof(size_t));
  checkAlignmentOf(temp_ptr, sizeof(size_t));

  const nvcompType_t countType
      = selectRunsType(in_bytes / sizeOfnvcompType(in_type));

  NVCOMP_TYPE_TWO_SWITCH(
      in_type,
      countType,
      compressTypedAsync,
      in_ptr,
      in_bytes,
      cascadedOpts,
      temp_ptr,
      temp_bytes,
      out_ptr,
      out_bytes,
      stream);
}

} // namespace nvcomp

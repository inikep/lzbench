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

#ifndef CASCADEDMETADATAONGPU_H
#define CASCADEDMETADATAONGPU_H

#include "CascadedMetadata.h"

#include "cuda_runtime.h"

namespace nvcomp
{

class CascadedMetadataOnGPU
{
public:
  /**
   * @brief Get the serialized size a metadata object would occupy on the GPU.
   *
   * @param metadata The metadata.
   *
   * @return The size in bytes that would be occupied.
   */
  static size_t getSerializedSizeOf(const CascadedMetadata& metadata);

  /**
   * @brief Create a new serialized metadata object. This is used either to
   * copy metadata to the GPU, or copy it to the CPU from the GPU.
   *
   * @param ptr The memory location on the GPU that will be used for the
   * serialized metadata.
   * @param maxSize The maximum size the metadata can occupy (the size of the
   * allocation usually).
   */
  CascadedMetadataOnGPU(void* ptr, size_t maxSize);

  // disable copying
  CascadedMetadataOnGPU(const CascadedMetadataOnGPU&) = delete;
  CascadedMetadataOnGPU& operator=(const CascadedMetadataOnGPU&) = delete;

  /**
   * @brief Copy and serialize the given metadata object to to the GPU
   * asynchronously.
   *
   * @param metadata The metadata object to serialize.
   * @param serializedSizeDPtr The pointer to the location on the device to
   * save the serialized size to (may be NULL).
   * @param stream The stream to asynchronously execute on.
   */
  void copyToGPU(
      const CascadedMetadata& metadata,
      size_t* serializedSizeDPtr,
      cudaStream_t stream);

  /**
   * @brief Copy and serialize the given metadata object to to the GPU
   * asynchronously.
   *
   * @param metadata The metadata object to serialize.
   * @param stream The stream to asynchronously execute on.
   */
  void copyToGPU(const CascadedMetadata& metadata, cudaStream_t stream);

  /**
   * @brief Get the size of this metadata on the GPU.
   *
   * @return The size in bytes.
   */
  size_t getSerializedSize() const;

  /**
   * @brief Get a copy of the metadata on the CPU. This will synchronize with
   * the stream.
   *
   * @stream The stream to transfer on.
   *
   * @return The metadata on the CPU.
   */
  CascadedMetadata copyToHost(cudaStream_t stream);

  /**
   * @brief Save the offset scalar stored on the device to the serialized
   * metadata.
   *
   * @param index The layer/input to set the data offset for.
   * @param offsetDPtr The pointer to the offset on the device.
   * @param stream The stream to asynchronously operate on.
   */
  void saveOffset(size_t index, const size_t* offsetDPtr, cudaStream_t stream);

  /**
   * @brief Set the compressed size in the serialized metadata.
   *
   * @param sizeDPtr The pointer to the compressed size on the device.
   * @param stream The stream to asynchronously operator on.
   */
  void setCompressedSizeFromGPU(const size_t* sizeDptr, cudaStream_t stream);

  /**
   * @brief Get a pointer to the given header on the device.
   *
   * @param index The input/layer to get the header of.
   *
   * @return The pointer to the header on the device.
   */
  CascadedMetadata::Header* getHeaderLocation(size_t index);

  /**
   * @brief Get a pointer to the given header on the device.
   *
   * @param index The input/layer to get the header of.
   *
   * @return The pointer to the header on the device.
   */
  const CascadedMetadata::Header* getHeaderLocation(size_t index) const;

private:
  void* m_ptr;
  size_t m_maxSize;
  size_t m_numInputs;

  /**
   * @brief Ensures that copyToGPU() has already been called (even if the
   * asynchronous work associated with it has not finished) on this object and
   * necessary information has been set, to allow quering.
   *
   * @throw An exception if copyToGPU() has not been successfully called.
   */
  void verifyInitialized() const;

  /**
   * @brief Verify that the given index is a valid one to operate on. If it
   * is greater than the number of inputs, or the number of inputs has not
   * been initialized, an exception will be thrown.
   *
   * @param index The index to check.
   *
   * @throw An exception if the index is not valid.
   */
  void verifyIndex(size_t index) const;
};

} // namespace nvcomp

#endif

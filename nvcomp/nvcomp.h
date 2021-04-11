/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVCOMP_H
#define NVCOMP_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

#define NVCOMP_MAJOR_VERSION 1
#define NVCOMP_MINOR_VERSION 2
#define NVCOMP_PATCH_VERSION 3

typedef enum nvcompError_t
{
  nvcompSuccess = 0,
  nvcompErrorInvalidValue = 10,
  nvcompErrorNotSupported = 11,
  nvcompErrorCudaError = 1000,
  nvcompErrorInternal = 10000
} nvcompError_t;

/* Supported datatypes */
typedef enum nvcompType_t
{
  NVCOMP_TYPE_CHAR = 0,      // 1B
  NVCOMP_TYPE_UCHAR = 1,     // 1B
  NVCOMP_TYPE_SHORT = 2,     // 2B
  NVCOMP_TYPE_USHORT = 3,    // 2B
  NVCOMP_TYPE_INT = 4,       // 4B
  NVCOMP_TYPE_UINT = 5,      // 4B
  NVCOMP_TYPE_LONGLONG = 6,  // 8B
  NVCOMP_TYPE_ULONGLONG = 7, // 8B
  NVCOMP_TYPE_BITS = 0xff    // 1b
} nvcompType_t;

/******************************************************************************
 * FUNCTION PROTOTYPES ********************************************************
 *****************************************************************************/

/**
 * @brief Extracts the metadata from the input in_ptr on the device and copies
 *it to the host.
 *
 * @param in_ptr The compressed memory on the device.
 * @param in_bytes The size of the compressed memory on the device.
 * @param metadata_ptr The metadata on the host to create from the compresesd
 * data.
 * @param stream The stream to use for reading memory from the device.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompDecompressGetMetadata(
    const void* in_ptr,
    size_t in_bytes,
    void** metadata_ptr,
    cudaStream_t stream);

/**
 * @brief Destroys the metadata object and frees the associated memory.
 *
 * @param metadata_ptr The pointer to destroy.
 */
void nvcompDecompressDestroyMetadata(void* metadata_ptr);

/**
 * @brief Computes the required temporary workspace required to perform
 * decompression.
 *
 * @para metadata_ptr The metadata.
 * @param temp_bytes The size of the required temporary workspace in bytes
 * (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t
nvcompDecompressGetTempSize(const void* metadata_ptr, size_t* temp_bytes);

/**
 * @brief Computes the size of the uncompressed data in bytes.
 *
 * @para metadata_ptr The metadata.
 * @param output_bytes The size of the uncompressed data (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t
nvcompDecompressGetOutputSize(const void* metadata_ptr, size_t* output_bytes);

/**
 * @brief Get the type of the compressed data.
 *
 * @param metadata_ptr The metadata.
 * @param type The data type (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t
nvcompDecompressGetType(const void* metadata_ptr, nvcompType_t* type);

/**
 * @brief Check if the compressed data must be decompressed to the same data
 * type from which was compressed.
 *
 * @param metadata_ptr The metadata.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
int nvcompDecompressIsTypeSensitive(const void* metadata_ptr);

/**
 * @brief Perform the asynchronous decompression.
 *
 * @param in_ptr The compressed data on the device to decompress.
 * @param in_bytes The size of the compressed data.
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace.
 * @param metadata_ptr The metadata.
 * @param out_ptr The output location on the device.
 * @param out_bytes The size of the output location.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompDecompressAsync(
    const void* in_ptr,
    size_t in_bytes,
    void* temp_ptr,
    size_t temp_bytes,
    const void* metadata_ptr,
    void* out_ptr,
    size_t out_bytes,
    cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif

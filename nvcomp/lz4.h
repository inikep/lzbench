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

#ifndef NVCOMP_LZ4_H
#define NVCOMP_LZ4_H

#include "nvcomp.h"

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure for configuring LZ4 compression.
 */
typedef struct
{
  /**
   * @brief The size of each chunk of data to decompress indepentently with
   * LZ4. Must be within the range of [32768, 16777216]. Larger sizes will
   * result in higher compression, but with decreased parallelism. The
   * recommended size is 65536.
   */
  size_t chunk_size;
} nvcompLZ4FormatOpts;

/**
 * @brief Check if a given chunk of compressed data on the GPU is LZ4.
 *
 * @param in_ptr The compressed data.
 * @param in_bytes The size of the compressed data.
 * @param stream The stream to fetch data from the GPU on.
 *
 * @return 1 If the data is compressed via LZ4.
 */
int LZ4IsData(const void* const in_ptr, size_t in_bytes, cudaStream_t stream);

/**
 * @brief Check if the given CPU-accessible metadata is for LZ4.
 *
 * @param metadata_ptr The metadata pointer.
 *
 * @return 1 if the data is for LZ4.
 */
int LZ4IsMetadata(const void* const metadata_ptr);

/**
 * @brief Extracts the metadata from the input in_ptr on the device and copies
 * it to the host. This function synchronizes on the stream.
 *
 * @param in_ptr The compressed memory on the device.
 * @param in_bytes The size of the compressed memory on the device.
 * @param metadata_ptr The metadata on the host to create from the compresesd
 * data.
 * @param stream The stream to use for reading memory from the device.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompLZ4DecompressGetMetadata(
    const void* in_ptr,
    size_t in_bytes,
    void** metadata_ptr,
    cudaStream_t stream);

/**
 * @brief Destroys the metadata object and frees the associated memory.
 *
 * @para metadata_ptr The metadata to destroy.
 */
void nvcompLZ4DecompressDestroyMetadata(void* metadata_ptr);

/**
 * @brief Computes the temporary storage size needed to decompress.
 *
 * @param metadata_ptr The metadata.
 * @param temp_bytes The size of temporary workspace required to perform
 * decomrpession, in bytes (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t
nvcompLZ4DecompressGetTempSize(const void* metadata_ptr, size_t* temp_bytes);

/**
 * @brief Computes the decompressed size of the data.
 *
 * @param metadata_ptr The metadata.
 * @param output_bytes The size of the decompressed data in bytes (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompLZ4DecompressGetOutputSize(
    const void* metadata_ptr, size_t* output_bytes);

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
nvcompError_t nvcompLZ4DecompressAsync(
    const void* const in_ptr,
    const size_t in_bytes,
    void* const temp_ptr,
    const size_t temp_bytes,
    const void* const metadata_ptr,
    void* const out_ptr,
    const size_t out_bytes,
    cudaStream_t stream);

/**
 * @brief Get the temporary workspace size required to perform compression.
 *
 * @param in_ptr The uncompressed data on the device.
 * @param in_bytes The size of the uncompressed data in bytes.
 * @param in_type The type of the uncompressed data.
 * @param format_opts The lz4 format options.
 * @param temp_bytes The size of the required temporary workspace in bytes
 * (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompLZ4CompressGetTempSize(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    const nvcompLZ4FormatOpts* format_opts,
    size_t* temp_bytes);

/**
 * @brief Get the required output size to perform compression.
 *
 * @param in_ptr The uncompressed data on the device.
 * @param in_bytes The size of the uncompressed data in bytes.
 * @param in_type The type of the uncompressed data.
 * @param format_opts The lz4 format options.
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace in bytes.
 * @param out_bytes The required size of the output location in bytes (output).
 * @param exact_out_bytes Whether or not to compute the exact number of bytes
 * needed, or quickly compute a loose upper bound.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompLZ4CompressGetOutputSize(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    const nvcompLZ4FormatOpts* format_opts,
    void* temp_ptr,
    size_t temp_bytes,
    size_t* out_bytes,
    int exact_out_bytes);

/**
 * @brief Perform asynchronous compression. The pointer `out_bytes` must be to
 * pinned memory for this to be asynchronous.
 *
 * @param in_ptr The uncompressed data on the device.
 * @param in_bytes The size of the uncompressed data in bytes.
 * @param in_type The data type of the uncompressed data.
 * @param format_opts The lz4 format options.
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace in bytes.
 * @param out_ptr The location to write compresesd data to on the device.
 * @param out_bytes The size of the output location on input, and the size of
 * the compressed data on output. If pinned memory, the stream must be
 * synchronized with, before reading.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompLZ4CompressAsync(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    const nvcompLZ4FormatOpts* format_opts,
    void* temp_ptr,
    size_t temp_bytes,
    void* out_ptr,
    size_t* out_bytes,
    cudaStream_t stream);


/**************************************************************************
 *  - Experimental - Subject to change -
 * Batched compression/decompression interface for LZ4
 * ************************************************************************/

/**
 * @brief Extracts the metadata from all the input baches in_ptr on the device and copies
 * them to the host. This function synchronizes on the stream.
 *
 * @param in_ptr Array of compressed chunks on the device.
 * @param in_bytes Array of sizes of the compressed chunks on the device.
 * @param batch_size Number of chunks in the batch (cardinality of in_bytes and in_ptr)
 * @param metadata_ptr The batch of metadata on the host to create from all the compresesed
 * data chunks in the batch.
 * @param stream The stream to use for reading memory from the device.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBatchedLZ4DecompressGetMetadata(
    const void** in_ptr,
    const size_t* in_bytes,
    size_t batch_size,
    void** metadata_ptr,
    cudaStream_t stream);

/**
 * @brief Destroys metadata and frees the associated memory.
 *
 * @para metadata_ptr List of metadata to destroy.
 */
void nvcompBatchedLZ4DecompressDestroyMetadata(void* metadata_ptr);

/**
 * @brief Computes the temporary storage size needed to decompress the batch of data.
 *
 * @param metadata_ptr The metadata for all compressed chunks in the batch.
 * @param temp_bytes The size of temporary workspace required to perform
 * decomrpession of all chunks in the batch, in bytes (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t
nvcompBatchedLZ4DecompressGetTempSize(const void* metadata_ptr, size_t* temp_bytes);

/**
 * @brief Computes the decompressed size of each chunk of in the batch.
 *
 * @param metadata_ptr The metadata for all compressed chunks.
 * @param batch_size The number of chunks in the batch (cardinality of output_bytes).
 * @param output_bytes Array of sizes of the decompressed data in bytes (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBatchedLZ4DecompressGetOutputSize(
    const void* metadata_ptr, size_t batch_size, size_t* output_bytes);

/**
 * @brief Perform the asynchronous decompression on batch of compressed chunks of data.
 *
 * @param in_ptr Array of compressed data chunks on the device to decompress.
 * @param in_bytes The sizes of each chunk of compressed data.
 * @param batch_size The number of chunks in the batch (cardinality of other inputs).
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace.
 * @param metadata_ptr The metadata of all chunks in the batch.
 * @param out_ptr The output location on the device.
 * @param out_bytes The sizes of each decompressed chunk.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBatchedLZ4DecompressAsync(
    const void* const* in_ptr,
    const size_t* in_bytes,
    size_t batch_size,
    void* const temp_ptr,
    const size_t temp_bytes,
    const void* metadata_ptr,
    void* const* out_ptr,
    const size_t* out_bytes,
    cudaStream_t stream);

/**
 * @brief Get the temporary workspace size required to perform compression of entire batch.
 *
 * @param in_ptr Array of uncompressed data chunks on the device.
 * @param in_bytes The sizes of each uncompressed data chunk in bytes.
 * @param batch_size The number of chunks in the batch.
 * @param format_opts The lz4 format options.
 * @param temp_bytes The size of the required temporary workspace in bytes to
 * compress the entire batch (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBatchedLZ4CompressGetTempSize(
    const void* const* in_ptr,
    const size_t* in_bytes,
    size_t batch_size,
    const nvcompLZ4FormatOpts* format_opts,
    size_t* temp_bytes);

/**
 * @brief Get the required output sizes of each chunk to perform compression.
 *
 * @param in_ptr Array of uncompressed chunks on the device.
 * @param in_bytes The sizes of each uncompressed chunk data in bytes.
 * @param batch_size The number of chunks in the batch (cardinality of in_ptr and in_bytes).
 * @param format_opts The lz4 format options.
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace in bytes.
 * @param out_bytes The required sizes of the output location for each chunk in bytes (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBatchedLZ4CompressGetOutputSize(
    const void* const* in_ptr,
    const size_t* in_bytes,
    size_t batch_size, 
    const nvcompLZ4FormatOpts* format_opts,
    void* temp_ptr,
    size_t temp_bytes,
    size_t* out_bytes);

/**
 * @brief Perform asynchronous compression. The pointer `out_bytes` must be to
 * pinned memory for this to be asynchronous.
 *
 * @param in_ptr Array of uncompressed chunks that make up the batch, on the device.
 * @param in_bytes The sizes of each chunk of uncompressed data in bytes.
 * @param batch_size The number of chunks in the entire batch.
 * @param format_opts The lz4 format options.
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace in bytes.
 * @param out_ptr The starting location to write all the compresesd chunks to on the device.
 * @param out_bytes The sizes of each compressed chunk of data, summed up totals the size of
 * the compressed data on output. If pinned memory, the stream must be
 * synchronized with, before reading.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBatchedLZ4CompressAsync(
    const void* const* in_ptr,
    const size_t* in_bytes,
    size_t batch_size,
    const nvcompLZ4FormatOpts* format_opts,
    void* temp_ptr,
    size_t temp_bytes,
    void* const* out_ptr,
    size_t* out_bytes,
    cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif

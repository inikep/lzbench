/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "common.h"
#include "nvcomp.h"
#include "nvcomp/cascaded.h"
#include "type_macros.h"
#include "CascadedKernels.cuh"
#include "Check.h"
#include "CudaUtils.h"

#include <cstdint>

using nvcomp::larger_t;
using nvcomp::roundUpDiv;
using nvcomp::roundUpTo;
using nvcomp::roundUpToAlignment;
using nvcomp::default_chunk_size;
using nvcomp::cascaded_compress_threadblock_size;
using nvcomp::cascaded_decompress_threadblock_size;
using nvcomp::partition_metadata_size;
using nvcomp::compute_smem_size;
using nvcomp::Check;
using nvcomp::CudaUtils;

namespace
{

/**
 * @brief Batched cascaded compression kernel.
 *
 * All cascaded compression layers are fused together in this single kernel.
 *
 * @tparam data_type Data type of each element.
 * @tparam threadblock_size Number of threads in a threadblock. This argument
 * must match the configuration specified when launching this kernel.
 * @tparam chunk_size Input size that is loaded into shared memory at a time.
 * This argument must be a multiple of the size of `data_type`.
 *
 * @param[in] batch_size Number of partitions to compress.
 * @param[in] uncompressed_data Array with size \p batch_size of pointers to
 * input uncompressed partitions.
 * @param[in] uncompressed_bytes Sizes of input uncompressed partitions in
 * bytes.
 * @param[out] compressed_data Array with size \p batch_size of output locations
 * of the compressed buffers. Each compressed buffer must start at a location
 * aligned with both 4B and the data type.
 * @param[out] compressed_bytes Number of bytes decompressed of all partitions.
 * @param[in] comp_opts Compression format used.
 */
template <
    typename data_type,
    typename size_type,
    int threadblock_size,
    int chunk_size = default_chunk_size>
__global__ void cascaded_compression_kernel(
    int batch_size,
    const data_type* const* uncompressed_data,
    const size_type* uncompressed_bytes,
    void* const* compressed_data,
    size_type* compressed_bytes,
    nvcompBatchedCascadedOpts_t comp_opts)
{
  nvcomp::do_cascaded_compression_kernel<
      data_type,
      size_type,
      threadblock_size,
      chunk_size>(
      batch_size,
      blockIdx.x,
      gridDim.x,
      uncompressed_data,
      uncompressed_bytes,
      compressed_data,
      compressed_bytes,
      comp_opts);
}
/**
 * @brief Kernel to perform batched cascaded decompression. Extracts the
 * datatype from the metadata of the compressed buffer, then checks of the
 * templated call type matches.  If it matches, it allocates the correct amount
 * of shared memory and runs decompression.  Otherwise, it just exits.
 *
 * @tparam bitwidth_test Data type to use for underlying decompression.  If
 * datatype found in metadata matches, perform compression, else exit.
 * @tparam size_type Data type used for size measures, typically size_t is used.
 * @tparam threadblock_size Number of threads in a threadblock. This argument
 * must match the configuration specified when launching this kernel.
 * @tparam chunk_size Number of bytes for each uncompressed chunk to fit inside
 * shared memory. This argument must match the chunk size specified during
 * compression.
 *
 * @param[in] batch_size Number of partitions to decompress.
 * @param[in] compressed_data Array of size \p batch_size where each element is
 * a pointer to the compressed data of a partition.
 * @param[in] compressed_bytes Sizes of the compressed buffers corresponding to
 * \p compressed_data.
 * @param[out] decompressed_data Pointers to the output decompressed buffers.
 * @param[in] decompressed_buffer_bytes Sizes of the decompressed buffers in
 * bytes.
 * @param[out] actual_decompressed_bytes Actual number of bytes decompressed for
 * all partitions.
 */
template <
    int bitwidth_test,
    typename size_type,
    int threadblock_size,
    int chunk_size = default_chunk_size>
__global__ void cascaded_decompression_kernel_type_check(
    int batch_size,
    const void* const* compressed_data,
    const size_type* compressed_bytes,
    void* const* decompressed_data,
    const size_type* decompressed_buffer_bytes,
    size_type* actual_decompressed_bytes,
    nvcompStatus_t* statuses)
{
  // Extract datatype from compressed data
  const auto partition_metadata_ptr
      = reinterpret_cast<const uint8_t*>(compressed_data[0]);
  const auto type = static_cast<nvcompType_t>(partition_metadata_ptr[3]);

  switch (bitwidth_test) {
  case 1:
    if (type == NVCOMP_TYPE_CHAR || type == NVCOMP_TYPE_UCHAR) {
      // allocate shmem and run fcn for 1-byte type
      const int shmem_size = compute_smem_size<chunk_size, 1, 4>();
      __shared__ uint8_t shmem[shmem_size];

      nvcomp::template cascaded_decompression_fcn<
          uint8_t,
          size_type,
          threadblock_size>(
          batch_size,
          blockIdx.x,
          gridDim.x,
          compressed_data,
          compressed_bytes,
          decompressed_data,
          decompressed_buffer_bytes,
          actual_decompressed_bytes,
          (void*)shmem,
          statuses);
    }
    break;
  case 2:
    if (type == NVCOMP_TYPE_SHORT || type == NVCOMP_TYPE_USHORT) {
      // allocate shmem and run fcn for 2-byte type
      const int shmem_size = compute_smem_size<chunk_size, 2, 4>();
      __shared__ uint8_t shmem[shmem_size];

      nvcomp::template cascaded_decompression_fcn<
          uint16_t,
          size_type,
          threadblock_size>(
          batch_size,
          blockIdx.x,
          gridDim.x,
          compressed_data,
          compressed_bytes,
          decompressed_data,
          decompressed_buffer_bytes,
          actual_decompressed_bytes,
          (void*)shmem,
          statuses);
    }
    break;
  case 4:
    if (type == NVCOMP_TYPE_INT || type == NVCOMP_TYPE_UINT) {
      // allocate shmem and run fcn for 4-byte type
      const int shmem_size = compute_smem_size<chunk_size, 4, 4>();
      __shared__ uint8_t shmem[shmem_size];

      nvcomp::template cascaded_decompression_fcn<
          uint32_t,
          size_type,
          threadblock_size>(
          batch_size,
          blockIdx.x,
          gridDim.x,
          compressed_data,
          compressed_bytes,
          decompressed_data,
          decompressed_buffer_bytes,
          actual_decompressed_bytes,
          (void*)shmem,
          statuses);
    }
    break;
  case 8:
    if (type == NVCOMP_TYPE_LONGLONG || type == NVCOMP_TYPE_ULONGLONG) {
      // allocate shmem and run fcn for 8-byte type
      const int shmem_size = compute_smem_size<chunk_size, 8, 8>();
      __shared__ uint8_t shmem[shmem_size];

      nvcomp::template cascaded_decompression_fcn<
          uint64_t,
          size_type,
          threadblock_size>(
          batch_size,
          blockIdx.x,
          gridDim.x,
          compressed_data,
          compressed_bytes,
          decompressed_data,
          decompressed_buffer_bytes,
          actual_decompressed_bytes,
          (void*)shmem,
          statuses);
    }
    break;
  }
}

__global__ void get_decompress_size_kernel(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size)
{
  for (size_t partition_idx = blockIdx.x * blockDim.x + threadIdx.x;
       partition_idx < batch_size;
       partition_idx += gridDim.x * blockDim.x) {
    if (device_compressed_bytes[partition_idx] < partition_metadata_size) {
      // The compressed buffer should always have enough space for metadata. If
      // not, we report error.
      device_uncompressed_bytes[partition_idx] = 0;
    } else {
      auto compressed_data
          = static_cast<const uint32_t*>(device_compressed_ptrs[partition_idx]);
      device_uncompressed_bytes[partition_idx] = compressed_data[1];
    }
  }
}

template <typename data_type>
void cascaded_batched_compression_typed(
    const nvcompBatchedCascadedOpts_t format_opts,
    const void* const* device_uncompressed_ptrs,
    const size_t* device_uncompressed_bytes,
    size_t batch_size,
    void* const* device_compressed_ptrs,
    size_t* device_compressed_bytes,
    cudaStream_t stream)
{
  constexpr int threadblock_size = cascaded_compress_threadblock_size;
  cascaded_compression_kernel<data_type, size_t, threadblock_size>
      <<<batch_size, threadblock_size, 0, stream>>>(
          batch_size,
          reinterpret_cast<const data_type* const*>(device_uncompressed_ptrs),
          device_uncompressed_bytes,
          device_compressed_ptrs,
          device_compressed_bytes,
          format_opts);
}

} // namespace

nvcompStatus_t nvcompBatchedCascadedCompressGetTempSize(
    size_t batch_size,
    size_t max_uncompressed_chunk_bytes,
    nvcompBatchedCascadedOpts_t format_opts,
    size_t* temp_bytes)
{

  *temp_bytes = 0;

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedCascadedCompressGetMaxOutputChunkSize(
    size_t max_uncompressed_chunk_bytes,
    nvcompBatchedCascadedOpts_t format_opts,
    size_t* max_compressed_bytes)
{

  *max_compressed_bytes = roundUpTo(max_uncompressed_chunk_bytes, 4) + 8;

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedCascadedCompressAsync(
    const void* const* device_uncompressed_ptrs,
    const size_t* device_uncompressed_bytes,
    size_t max_uncompressed_chunk_bytes, // not used
    size_t batch_size,
    void* device_temp_ptr, // not used
    size_t temp_bytes,     // not used
    void* const* device_compressed_ptrs,
    size_t* device_compressed_bytes,
    const nvcompBatchedCascadedOpts_t format_opts,
    cudaStream_t stream)
{
  try {
    NVCOMP_TYPE_ONE_SWITCH(
        format_opts.type,
        cascaded_batched_compression_typed,
        format_opts,
        device_uncompressed_ptrs,
        device_uncompressed_bytes,
        batch_size,
        device_compressed_ptrs,
        device_compressed_bytes,
        stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompBatchedCascadedCompressAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedCascadedDecompressGetTempSize(
    size_t num_chunks, size_t max_uncompressed_chunk_bytes, size_t* temp_bytes)
{
  *temp_bytes = 0;
  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedCascadedDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr, // can be nullptr
    size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    nvcompStatus_t* device_statuses,
    cudaStream_t stream)
{
  try {
    // Just call kernel to perform compression. Macro for datatype happens
    // within kernel
    constexpr int threadblock_size = cascaded_decompress_threadblock_size;

    // call for all 4 possible sizes, all except the correct one will
    // immediately exit.

    // CHAR or UCHAR
    cascaded_decompression_kernel_type_check<1, size_t, threadblock_size>
        <<<batch_size, threadblock_size, 0, stream>>>(
            batch_size,
            device_compressed_ptrs,
            device_compressed_bytes,
            device_uncompressed_ptrs,
            device_uncompressed_bytes,
            device_actual_uncompressed_bytes,
            device_statuses);
    CudaUtils::check_last_error();
    // SHORT or USHORT
    cascaded_decompression_kernel_type_check<2, size_t, threadblock_size>
        <<<batch_size, threadblock_size, 0, stream>>>(
            batch_size,
            device_compressed_ptrs,
            device_compressed_bytes,
            device_uncompressed_ptrs,
            device_uncompressed_bytes,
            device_actual_uncompressed_bytes,
            device_statuses);
    CudaUtils::check_last_error();
    // INT or UINT
    cascaded_decompression_kernel_type_check<4, size_t, threadblock_size>
        <<<batch_size, threadblock_size, 0, stream>>>(
            batch_size,
            device_compressed_ptrs,
            device_compressed_bytes,
            device_uncompressed_ptrs,
            device_uncompressed_bytes,
            device_actual_uncompressed_bytes,
            device_statuses);
    CudaUtils::check_last_error();
    // LONGLONG or ULONGLONG
    cascaded_decompression_kernel_type_check<8, size_t, threadblock_size>
        <<<batch_size, threadblock_size, 0, stream>>>(
            batch_size,
            device_compressed_ptrs,
            device_compressed_bytes,
            device_uncompressed_ptrs,
            device_uncompressed_bytes,
            device_actual_uncompressed_bytes,
            device_statuses);
    CudaUtils::check_last_error();
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedCascadedDecompressAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedCascadedGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream)
{
  try {
    get_decompress_size_kernel<<<
        roundUpDiv(batch_size, cascaded_decompress_threadblock_size),
        cascaded_decompress_threadblock_size,
        0,
        stream>>>(
        device_compressed_ptrs,
        device_compressed_bytes,
        device_uncompressed_bytes,
        batch_size);
    CudaUtils::check_last_error();
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedCascadedGetDecompressSizeAsync()");
  }

  return nvcompSuccess;
}

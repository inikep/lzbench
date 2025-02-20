/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "CascadedKernels.cuh"
#include "highlevel/CascadedHlifKernels.h"
#include "nvcomp_common_deps/hlif_shared.cuh"
#include "nvcomp/cascaded.h"
#include "CudaUtils.h"

namespace nvcomp {

template <
    typename data_type,
    typename size_type,
    int threadblock_size,
    int chunk_size = default_chunk_size>
struct cascaded_compress_wrapper : hlif_compress_wrapper
{
private:
  nvcompStatus_t* status;
  const nvcompBatchedCascadedOpts_t options;

public:
  __device__ cascaded_compress_wrapper(
      const nvcompBatchedCascadedOpts_t options,
      uint8_t* /*tmp_buffer*/,
      uint8_t* /*share_buffer*/,
      nvcompStatus_t* status) :
      status(status), options(options)
  {}
      
  __device__ void compress_chunk(
      uint8_t* tmp_output_buffer,
      const uint8_t* this_decomp_buffer,
      const size_t decomp_size,
      const size_t max_comp_chunk_size,
      size_t* comp_chunk_size)
  {
    do_cascaded_compression_kernel<
        data_type,
        size_type,
        threadblock_size,
        chunk_size>(
        1,
        0,
        1,
        reinterpret_cast<const data_type* const*>(&this_decomp_buffer),
        &decomp_size,
        reinterpret_cast<void* const*>(&tmp_output_buffer),
        comp_chunk_size,
        options);
  }

  __device__ nvcompStatus_t& get_output_status() {
    return *status;
  }

  __device__ FormatType get_format_type() {
    return FormatType::Cascaded;
  }
};

template <
    typename data_type,
    typename size_type,
    int threadblock_size,
    int chunk_size = default_chunk_size>
struct cascaded_decompress_wrapper : hlif_decompress_wrapper
{

private:
  nvcompStatus_t* status;
  const nvcompBatchedCascadedOpts_t options;

public:
  __device__ cascaded_decompress_wrapper(
      const nvcompBatchedCascadedOpts_t options,
      uint8_t* /*shared_buffer*/,
      nvcompStatus_t* status) :
      status(status), options(options)
  {}
      
  __device__ void decompress_chunk(
      uint8_t* decomp_buffer,
      const uint8_t* comp_buffer,
      const size_t comp_chunk_size,
      const size_t decomp_buffer_size)
  {
    size_t actual_decompressed_bytes;
    nvcompStatus_t status;

    // allocate shmem and run fcn for data_type
    constexpr int shmem_size = compute_smem_size<
        chunk_size,
        sizeof(data_type),
        ((sizeof(data_type) <= 4) ? 4 : 8)>();
    __shared__ uint8_t shmem[shmem_size];

    cascaded_decompression_fcn<
        data_type,
        size_type,
        threadblock_size,
        chunk_size>(
        1,
        0,
        1,
        reinterpret_cast<const void* const*>(&comp_buffer),
        &comp_chunk_size,
        reinterpret_cast<void* const*>(&decomp_buffer),
        &decomp_buffer_size,
        &actual_decompressed_bytes,
        shmem,
        &status);
  }

  __device__ nvcompStatus_t& get_output_status() {
    return *status;
  }
};

void cascadedHlifBatchCompress(
    const CompressArgs& compress_args,
    const uint32_t max_ctas,
    cudaStream_t stream,
    const nvcompBatchedCascadedOpts_t* options)
{
  const dim3 batch_size(max_ctas);
  constexpr int threadblock_size = cascaded_compress_threadblock_size;

  const nvcompType_t type = options->type;
  if (type == NVCOMP_TYPE_CHAR || type == NVCOMP_TYPE_UCHAR) {
    HlifCompressBatchKernel<
        cascaded_compress_wrapper<uint8_t, size_t, threadblock_size>,
        const nvcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(compress_args, *options);
  } else if (type == NVCOMP_TYPE_SHORT || type == NVCOMP_TYPE_USHORT) {
    HlifCompressBatchKernel<
        cascaded_compress_wrapper<uint16_t, size_t, threadblock_size>,
        const nvcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(compress_args, *options);
  } else if (type == NVCOMP_TYPE_INT || type == NVCOMP_TYPE_UINT) {
    HlifCompressBatchKernel<
        cascaded_compress_wrapper<uint32_t, size_t, threadblock_size>,
        const nvcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(compress_args, *options);
  } else if (type == NVCOMP_TYPE_LONGLONG || type == NVCOMP_TYPE_ULONGLONG) {
    HlifCompressBatchKernel<
        cascaded_compress_wrapper<uint64_t, size_t, threadblock_size>,
        const nvcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(compress_args, *options);
  }
}

void cascadedHlifBatchDecompress(
    const uint8_t* comp_buffer, 
    uint8_t* decomp_buffer, 
    const size_t raw_chunk_size,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t* comp_chunk_offsets,
    const size_t* comp_chunk_sizes,
    const uint32_t max_ctas,
    cudaStream_t stream,
    nvcompStatus_t* output_status,
    const nvcompBatchedCascadedOpts_t* options)
{
  const dim3 batch_size(max_ctas);
  constexpr int threadblock_size = cascaded_decompress_threadblock_size;

  const nvcompType_t type = options->type;
  if (type == NVCOMP_TYPE_CHAR || type == NVCOMP_TYPE_UCHAR) {
    HlifDecompressBatchKernel<
        cascaded_decompress_wrapper<uint8_t, size_t, threadblock_size>,
        1,
        const nvcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(
            comp_buffer,
            decomp_buffer,
            raw_chunk_size,
            ix_chunk,
            num_chunks,
            comp_chunk_offsets,
            comp_chunk_sizes,
            output_status,
            *options);
  } else if (type == NVCOMP_TYPE_SHORT || type == NVCOMP_TYPE_USHORT) {
    HlifDecompressBatchKernel<
        cascaded_decompress_wrapper<uint16_t, size_t, threadblock_size>,
        1,
        const nvcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(
            comp_buffer,
            decomp_buffer,
            raw_chunk_size,
            ix_chunk,
            num_chunks,
            comp_chunk_offsets,
            comp_chunk_sizes,
            output_status,
            *options);
  } else if (type == NVCOMP_TYPE_INT || type == NVCOMP_TYPE_UINT) {
    HlifDecompressBatchKernel<
        cascaded_decompress_wrapper<uint32_t, size_t, threadblock_size>,
        1,
        const nvcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(
            comp_buffer,
            decomp_buffer,
            raw_chunk_size,
            ix_chunk,
            num_chunks,
            comp_chunk_offsets,
            comp_chunk_sizes,
            output_status,
            *options);
  } else if (type == NVCOMP_TYPE_LONGLONG || type == NVCOMP_TYPE_ULONGLONG) {
    HlifDecompressBatchKernel<
        cascaded_decompress_wrapper<uint64_t, size_t, threadblock_size>,
        1,
        const nvcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(
            comp_buffer,
            decomp_buffer,
            raw_chunk_size,
            ix_chunk,
            num_chunks,
            comp_chunk_offsets,
            comp_chunk_sizes,
            output_status,
            *options);
  }

}

size_t cascadedHlifCompMaxBlockOccupancy(const int device_id, nvcompType_t type)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  int numBlocksPerSM = 1;
  // This kernel only uses fixed-size shared memory, not shared memory
  // determined at kernel invocation time.
  constexpr int runtime_shmem_size = 0;
  constexpr int threadblock_size = cascaded_compress_threadblock_size;
  // The values will almost certainly be identical for all data types,
  // but just in case, handle types separately.
  if (type == NVCOMP_TYPE_CHAR || type == NVCOMP_TYPE_UCHAR) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifCompressBatchKernel<
            cascaded_compress_wrapper<uint8_t, size_t, threadblock_size>,
            const nvcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  } else if (type == NVCOMP_TYPE_SHORT || type == NVCOMP_TYPE_USHORT) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifCompressBatchKernel<
            cascaded_compress_wrapper<uint16_t, size_t, threadblock_size>,
            const nvcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  } else if (type == NVCOMP_TYPE_INT || type == NVCOMP_TYPE_UINT) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifCompressBatchKernel<
            cascaded_compress_wrapper<uint32_t, size_t, threadblock_size>,
            const nvcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  } else if (type == NVCOMP_TYPE_LONGLONG || type == NVCOMP_TYPE_ULONGLONG) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifCompressBatchKernel<
            cascaded_compress_wrapper<uint64_t, size_t, threadblock_size>,
            const nvcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  }

  return deviceProp.multiProcessorCount * numBlocksPerSM;
}

size_t cascadedHlifDecompMaxBlockOccupancy(
    const int device_id, nvcompType_t type)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  int numBlocksPerSM = 1;
  // This kernel only uses fixed-size shared memory, not shared memory
  // determined at kernel invocation time.
  constexpr int runtime_shmem_size = 0;
  constexpr int threadblock_size = cascaded_decompress_threadblock_size;
  // The values will almost certainly be identical for all data types,
  // but just in case, handle types separately.
  if (type == NVCOMP_TYPE_CHAR || type == NVCOMP_TYPE_UCHAR) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifDecompressBatchKernel<
            cascaded_decompress_wrapper<uint8_t, size_t, threadblock_size>,
            1,
            const nvcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  } else if (type == NVCOMP_TYPE_SHORT || type == NVCOMP_TYPE_USHORT) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifDecompressBatchKernel<
            cascaded_decompress_wrapper<uint16_t, size_t, threadblock_size>,
            1,
            const nvcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  } else if (type == NVCOMP_TYPE_INT || type == NVCOMP_TYPE_UINT) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifDecompressBatchKernel<
            cascaded_decompress_wrapper<uint32_t, size_t, threadblock_size>,
            1,
            const nvcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  } else if (type == NVCOMP_TYPE_LONGLONG || type == NVCOMP_TYPE_ULONGLONG) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifDecompressBatchKernel<
            cascaded_decompress_wrapper<uint64_t, size_t, threadblock_size>,
            1,
            const nvcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  }

  return deviceProp.multiProcessorCount * numBlocksPerSM;
}

} // nvcomp namespace

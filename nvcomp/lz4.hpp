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

#ifndef NVCOMP_LZ4_HPP
#define NVCOMP_LZ4_HPP

#include "lz4.h"
#include "nvcomp.hpp"

namespace nvcomp
{
/**
 * @brief Alternate compressor offered by nvcomp: LZ4
 * Compression and decompression run asynchronously, but compress() requires
 * that the compressed size (*out_bytes) is known and buffers are allocated. Can
 * define synchronous wrapper that includes size estimation kernel + allocation.
 * 
 * @param T the type of element to compress.
 */
template <typename T>
class LZ4Compressor : public Compressor<T>
{
public:
  /**
   * @brief Default constructor that sets the format opts for the compression
   * and saves the pointer to the input data to compress
   *
   * @param in_ptr The input array to be compressed
   * @param num_elements size of the input array
   * @param chunk_size size of chunks that are eached compressed separately.
   * Controls the temporary storage needed.
   */
  LZ4Compressor(const T* in_ptr, size_t num_elements, size_t chunk_size);

  // disable copying
  LZ4Compressor(const LZ4Compressor&) = delete;
  LZ4Compressor& operator=(const LZ4Compressor&) = delete;

  /**
   * @brief Get the size of the temporary buffer required for decompression.
   *
   * @return The size in bytes.
   */
  size_t get_temp_size() override;

  /**
   * @brief Get the exact size the data will compress to. This can be used in
   * place of `get_max_output_size()` to get the minimum size of the
   * allocation that should be passed to `compress()`. This however, may take
   * similar amount of time to compression itself, and may execute synchronously
   * on the device.
   *
   * For LZ4Compression, this is not yet implemented, and will always
   * throw an exception.
   *
   * @param comp_temp The temporary workspace.
   * @param comp_temp_bytes THe size of the temporary workspace.
   *
   * @return The exact size in bytes.
   *
   * @throw NVCompressionException Will always be thrown.
   */
  size_t
  get_exact_output_size(void* comp_temp, size_t comp_temp_bytes) override;

  /**
   * @brief Get the maximum size the data could compressed to. This is the
   * upper bound of the minimum size of the allocation that should be
   * passed to `compress()`.
   *
   * @param comp_temp The temporary workspace.
   * @param comp_temp_bytes THe size of the temporary workspace.
   *
   * @return The maximum size in bytes.
   */
  size_t get_max_output_size(void* comp_temp, size_t comp_temp_bytes) override;

private:
  /**
   * @brief Perform compression asynchronously.
   *
   * @param temp_ptr The temporary workspace on the device.
   * @param temp_bytes The size of the temporary workspace.
   * @param out_ptr The output location the the device (for compressed data).
   * @param out_bytes The size of the output location on the device on input,
   * and the size of the compressed data on output.
   * @param stream The stream to operate on.
   *
   * @throw NVCompException If compression fails to launch on the stream.
   */
  void do_compress(
      void* temp_ptr,
      size_t temp_bytes,
      void* out_ptr,
      size_t* out_bytes,
      cudaStream_t stream) override;

  nvcompLZ4FormatOpts m_opts;
};

/******************************************************************************
 * METHOD IMPLEMENTATIONS *****************************************************
 *****************************************************************************/

template <typename T>
inline LZ4Compressor<T>::LZ4Compressor(
    const T* in_ptr, const size_t num_elements, const size_t chunk_size) :
    Compressor<T>(in_ptr, num_elements),
    m_opts{chunk_size}
{
  // do nothing
}

template <typename T>
inline size_t LZ4Compressor<T>::get_temp_size()
{
  size_t comp_temp_bytes;
  nvcompError_t status = nvcompLZ4CompressGetTempSize(
      this->get_uncompressed_data(),
      this->get_uncompressed_size(),
      this->get_type(),
      &m_opts,
      &comp_temp_bytes);
  throwExceptionIfError(status, "GetTempSize failed");

  return comp_temp_bytes;
}

template <typename T>
inline size_t LZ4Compressor<T>::get_exact_output_size(
    void* const comp_temp, const size_t comp_temp_bytes)
{
  size_t comp_out_bytes;
  nvcompError_t status = nvcompLZ4CompressGetOutputSize(
      this->get_uncompressed_data(),
      this->get_uncompressed_size(),
      this->get_type(),
      &m_opts,
      comp_temp,
      comp_temp_bytes,
      &comp_out_bytes,
      true);
  throwExceptionIfError(status, "LZ4CompressGetOutputSize() for exact failed");

  return comp_out_bytes;
}

template <typename T>
inline size_t LZ4Compressor<T>::get_max_output_size(
    void* const comp_temp, const size_t comp_temp_bytes)
{
  size_t comp_out_bytes;
  nvcompError_t status = nvcompLZ4CompressGetOutputSize(
      this->get_uncompressed_data(),
      this->get_uncompressed_size(),
      this->get_type(),
      &m_opts,
      comp_temp,
      comp_temp_bytes,
      &comp_out_bytes,
      false);
  throwExceptionIfError(
      status, "LZ4CompressGetOutputSize() for in exact failed");

  return comp_out_bytes;
}

template <typename T>
inline void LZ4Compressor<T>::do_compress(
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  nvcompError_t status = nvcompLZ4CompressAsync(
      this->get_uncompressed_data(),
      this->get_uncompressed_size(),
      this->get_type(),
      &m_opts,
      temp_ptr,
      temp_bytes,
      out_ptr,
      out_bytes,
      stream);
  throwExceptionIfError(status, "LZ4CompressAsync() failed");
}

} // namespace nvcomp
#endif

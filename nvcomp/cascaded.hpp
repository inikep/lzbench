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

#ifndef NVCOMP_CASCADED_HPP
#define NVCOMP_CASCADED_HPP

#include "cascaded.h"
#include "nvcomp.hpp"
#include <chrono>

namespace nvcomp
{

/**
 * @brief Primary compressor offered by nvcomp: RLE-Delta w/ bit-packing
 * Compression and decompression run asynchronously, but compress() requires
 * that the compressed size (*out_btyes) is known and buffers allocated. Can
 * define synchronous wrapper that includes size estimation kernel + allocation.
 *
 * @tparam T The type to compress.
 */
template <typename T>
class CascadedCompressor : public Compressor<T>
{

public:
  /**
   * @brief Create a new CascadedCompressor.
   *
   * NOTE: Currently, cascaded compression is limited to 2^31-1 bytes. To
   * compress larger data, break it up into chunks.
   *
   * @param in_ptr The input data on the GPU to compress.
   * @param num_elements The number of elements to compress.
   * @param num_RLEs The number of Run Length encodings to perform.
   * @param num_deltas The number of Deltas to perform.
   * @param use_bp Whether or not to bitpack the end result.
   */
  CascadedCompressor(
      const T* in_ptr,
      const size_t num_elements,
      int num_RLEs,
      int num_deltas,
      bool use_bp);

  /**
   * @brief Create a new CascadedCompressor without defining the configuration (RLE, delta, bp).
   * Runs the cascaded selector before compression to determine the configuration.
   *
   * NOTE: This the do_compress call synchronous on the CUDA stream.  For Asynchronous execution,
   * you must select the configuration manually.
   *
   * @param in_ptr The input data on the GPU to compress.
   * @param num_elements The number of elements to compress.
   */
  CascadedCompressor(
      const T* in_ptr,
      const size_t num_elements);

  // disable copying
  CascadedCompressor(const CascadedCompressor&) = delete;
  CascadedCompressor& operator=(const CascadedCompressor&) = delete;

  /**
   * @brief Get size of the temporary worksace in bytes, required to perform
   * compression.
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
   * For Cascaded compression, this is not yet implemented, and will always
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

  nvcompCascadedFormatOpts m_opts;
};

/******************************************************************************
 * Cascaded Selector **********************************************************
 *****************************************************************************/
/**
 *@brief Primary class for the Cascaded Selector used to determine the
 * best configuration to run cascaded compression on a given input.
 *
 *@param T the datatype of the input
 */
template <typename T>
class CascadedSelector
{
private:
  const void* input_data;
  size_t input_byte_len;
  size_t max_temp_size; // Internal variable used to store the temp buffer size
  nvcompCascadedSelectorOpts opts; // Sampling options

public:
  /**
   *@brief Create a new CascadedSelector for the given input data
   *
   *@param input The input data device pointer to select a cheme for
   *@param byte_len The number of bytes of input data
   *@param num_sample_ele The number of elements in a sample
   *@param num_sample The number of samples
   *@param type The type of input data
   */
  CascadedSelector(
      const void* input, size_t byte_len, nvcompCascadedSelectorOpts opts);

  // disable copying
  CascadedSelector(const CascadedSelector&) = delete;
  CascadedSelector& operator=(const CascadedSelector&) = delete;

  /*
   *@brief return the required size of workspace buffer in bytes
   */
  size_t get_temp_size() const;

  /*
   *@brief Select a CascadedSelector compression scheme that can provide the
   *best compression ratio and reports estimated compression ratio.
   *
   *@param d_worksapce The device potiner for the workspace
   *@param workspace_len The size of workspace buffer in bytes
   *@param comp_ratio The estimated compssion ratio using the bbest scheme
   *(output)
   *@param stream The input stream to run the select function
   *@return Selected Cascaded options (RLE, Delta encoding, bit packing)
   */
  nvcompCascadedFormatOpts select_config(
      void* d_workspace,
      size_t workspace_len,
      double* comp_ratio,
      cudaStream_t stream);

  /*
   *@brief Select a CascadedSelector compression scheme that can provide the
   *best compression ratio - does NOT return estimated compression ratio.
   *
   *@param d_worksapce The device potiner for the workspace
   *@param workspace_len The size of workspace buffer in bytes
   *@param stream The input stream to run the select function
   *@return Selected Cascaded options (RLE, Delta encoding, bit packing)
   */
  nvcompCascadedFormatOpts
  select_config(void* d_workspace, size_t workspace_len, cudaStream_t stream);
};

/******************************************************************************
 * METHOD IMPLEMENTATIONS *****************************************************
 *****************************************************************************/

template <typename T>
inline CascadedCompressor<T>::CascadedCompressor(
    const T* const in_ptr,
    const size_t num_elements,
    const int num_RLEs,
    const int num_deltas,
    const bool use_bp) :
    Compressor<T>(in_ptr, num_elements),
    m_opts{num_RLEs, num_deltas, use_bp}
{
  // do nothing
}

template <typename T>
inline CascadedCompressor<T>::CascadedCompressor(
    const T* const in_ptr,
    const size_t num_elements) :
    Compressor<T>(in_ptr, num_elements),
    m_opts{-1,-1,-1} // Use -1 as a special maker to run selector
{
  // do nothing
}

template <typename T>
inline size_t CascadedCompressor<T>::get_temp_size()
{
  size_t comp_temp_bytes;
  if(m_opts.num_RLEs == -1) { // If we need to run the selector
    nvcompError_t status = nvcompCascadedCompressAutoGetTempSize(
        this->get_uncompressed_data(),
        this->get_uncompressed_size(),
        this->get_type(),
        &comp_temp_bytes);
    throwExceptionIfError(status, "GetTempSize failed");
  }
  else {
    nvcompError_t status = nvcompCascadedCompressGetTempSize(
        this->get_uncompressed_data(),
        this->get_uncompressed_size(),
        this->get_type(),
        &m_opts,
        &comp_temp_bytes);
    throwExceptionIfError(status, "GetTempSize failed");
  }

  return comp_temp_bytes;
}

template <typename T>
inline size_t CascadedCompressor<T>::get_exact_output_size(
    void* const comp_temp, const size_t comp_temp_bytes)
{
  size_t comp_out_bytes;
  if(m_opts.num_RLEs == -1) { // If we need to run the selector
    throw "Exact output size not supported when using Auto Selector";
    return 0;
  }
  nvcompError_t status = nvcompCascadedCompressGetOutputSize(
      this->get_uncompressed_data(),
      this->get_uncompressed_size(),
      this->get_type(),
      &m_opts,
      comp_temp,
      comp_temp_bytes,
      &comp_out_bytes,
      true);
  throwExceptionIfError(
      status, "nvcompCascadedCompressGetOutputSize() for exact failed");

  return comp_out_bytes;
}

template <typename T>
inline size_t CascadedCompressor<T>::get_max_output_size(
    void* comp_temp, size_t comp_temp_bytes)
{
  size_t comp_out_bytes;
  if(m_opts.num_RLEs == -1) { // If we need to run the selector
    nvcompError_t status = nvcompCascadedCompressAutoGetOutputSize(
        this->get_uncompressed_data(),
        this->get_uncompressed_size(),
        this->get_type(),
        comp_temp,
        comp_temp_bytes,
        &comp_out_bytes);
    throwExceptionIfError(
        status, "nvcompCascadedCompressGetOutputSize() for in exact failed");
  }
  else {
    nvcompError_t status = nvcompCascadedCompressGetOutputSize(
        this->get_uncompressed_data(),
        this->get_uncompressed_size(),
        this->get_type(),
        &m_opts,
        comp_temp,
        comp_temp_bytes,
        &comp_out_bytes,
        false);
    throwExceptionIfError(
        status, "nvcompCascadedCompressGetOutputSize() for in exact failed");
  }

  return comp_out_bytes;
}

template <typename T>
inline void CascadedCompressor<T>::do_compress(
    void* temp_ptr,
    size_t temp_bytes,
    void* out_ptr,
    size_t* out_bytes,
    cudaStream_t stream)
{

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

  if(m_opts.num_RLEs == -1) { // If we need to run the selector
    nvcompError_t status = nvcompCascadedCompressAuto(
        this->get_uncompressed_data(),
        this->get_uncompressed_size(),
        this->get_type(),
        temp_ptr,
        temp_bytes,
        out_ptr,
        out_bytes,
        seed,
        stream);
    throwExceptionIfError(status, "nvcompCascadedCompressAsync() failed");
  }
  else {
    nvcompError_t status = nvcompCascadedCompressAsync(
        this->get_uncompressed_data(),
        this->get_uncompressed_size(),
        this->get_type(),
        &m_opts,
        temp_ptr,
        temp_bytes,
        out_ptr,
        out_bytes,
        stream);
    throwExceptionIfError(status, "nvcompCascadedCompressAsync() failed");
  }
}

template <typename T>
inline CascadedSelector<T>::CascadedSelector(
    const void* input,
    const size_t byte_len,
    nvcompCascadedSelectorOpts selector_opts) :
    input_data(input),
    input_byte_len(byte_len),
    max_temp_size(0),
    opts(selector_opts)
{
  size_t temp;
  nvcompError_t status = nvcompCascadedSelectorGetTempSize(
      input_byte_len, getnvcompType<T>(), opts, &temp);
  throwExceptionIfError(status, "SelectorGetTempSize failed");

  this->max_temp_size = temp;
}

template <typename T>
inline size_t CascadedSelector<T>::get_temp_size() const

{
  return max_temp_size;
}

template <typename T>
inline nvcompCascadedFormatOpts CascadedSelector<T>::select_config(
    void* d_workspace,
    size_t workspace_size,
    double* comp_ratio,
    cudaStream_t stream)
{
  nvcompCascadedFormatOpts cascadedOpts;
  nvcompError_t status = nvcompCascadedSelectorSelectConfig(
      input_data,
      input_byte_len,
      getnvcompType<T>(),
      opts,
      d_workspace,
      workspace_size,
      &cascadedOpts,
      comp_ratio,
      stream);
  throwExceptionIfError(status, "SelectorSelectConfig failed");

  return cascadedOpts;
}

template <typename T>
inline nvcompCascadedFormatOpts CascadedSelector<T>::select_config(
    void* d_workspace, size_t workspace_size, cudaStream_t stream)
{
  double comp_ratio;
  return select_config(d_workspace, workspace_size, &comp_ratio, stream);
}

} // namespace nvcomp
#endif

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

#ifndef NVCOMP_API_HPP
#define NVCOMP_API_HPP

#include "nvcomp.h"
#include "lz4.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace nvcomp
{

/******************************************************************************
 * CLASSES ********************************************************************
 *****************************************************************************/

/**
 * @brief The top-level exception throw by nvcomp C++ methods.
 */
class NVCompException : public std::runtime_error
{
public:
  /**
   * @brief Create a new NVCompException.
   *
   * @param err The error associated with the exception.
   * @param msg The error message.
   */
  NVCompException(nvcompError_t err, const std::string& msg) :
      std::runtime_error(msg + " : code=" + std::to_string(err) + "."),
      m_err(err)
  {
    // do nothing
  }

  nvcompError_t get_error() const
  {
    return m_err;
  }

private:
  nvcompError_t m_err;
};

/**
 * @brief Top-level compressor class. This class takes in data on the device,
 * and compresses it to another location on the device.
 *
 * @tparam T The type of element to compress.
 */
template <typename T>
class Compressor
{
public:
  /**
   * @brief Create a new compressor with the given input elements. The input
   * elements must remain available at the given memory location until
   * compression is finished executing.
   *
   * @param in_elements The input elements on the device.
   * @param num_in_elements The number of input elements.
   *
   * @throw NVCompressionExcpetion If `in_elements` is null.
   */
  Compressor(const T* in_elements, const size_t num_in_elements);

  // disable copying
  Compressor(const Compressor& other) = delete;
  Compressor& operator=(const Compressor& other) = delete;

  /**
   * @brief Virtual destructor.
   */
  virtual ~Compressor() = default;

  /**
   * @brief Launch asynchronous compression. If the `out_bytes` is pageable
   * memory, this method will block.
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
  void compress_async(
      void* temp_ptr,
      size_t temp_bytes,
      void* out_ptr,
      size_t* out_bytes,
      cudaStream_t stream);

  /**
   * @brief Get the data type being compressed by this compressor.
   *
   * @return The data type.
   */
  nvcompType_t get_type() const;

  /**
   * @brief Get the size of the uncompressed data in bytes.
   *
   * @return The size in bytes.
   */
  size_t get_uncompressed_size() const;

  /**
   * @brief Get the memory location of the uncompressed data.
   *
   * @return The uncompressed data.
   */
  const T* get_uncompressed_data() const;

  /**
   * @brief Get size of the temporary worksace in bytes, required to perform
   * compression.
   *
   * @return The size in bytes.
   */
  virtual size_t get_temp_size() = 0;

  /**
   * @brief Get the maximum size the data could compressed to. This is the
   * minimum size of the allocation that should be passed to `compress()`.
   *
   * @param comp_temp The temporary workspace.
   * @param comp_temp_bytes THe size of the temporary workspace.
   *
   * @return The maximum size in bytes.
   */
  virtual size_t get_max_output_size(void* comp_temp, size_t comp_temp_bytes)
      = 0;

  /**
   * @brief Get the exact size the data will compress to. This can be used in
   * place of `get_max_output_size()` to get the minimum size of the
   * allocation that should be passed to `compress()`. This however, may take
   * similar amount of time to compression itself, and may execute synchronously
   * on the device.
   *
   * NOTE: Some compression implementation may choose not to implement this, and
   * instead this method will throw an exception.
   *
   * @param comp_temp The temporary workspace.
   * @param comp_temp_bytes THe size of the temporary workspace.
   *
   * @return The exact size in bytes.
   *
   * @throw NVCompressionException If the exact output size cannot be
   * determined.
   */
  virtual size_t get_exact_output_size(void* comp_temp, size_t comp_temp_bytes);

private:
  /**
   * @brief Child classes implementing compression, should implement this
   * method, rather than the `compress()` method.
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
  virtual void do_compress(
      void* temp_ptr,
      size_t temp_bytes,
      void* out_ptr,
      size_t* out_bytes,
      cudaStream_t stream)
      = 0;

  // Member variables
  const T* m_in_elements;
  size_t m_num_in_elements;
};

/**
 * @brief Top-level decompress class. The compression type is read from the
 * metadata at the start of the compressed data.
 *
 * @tparam T The type to decompress to.
 */
template <typename T>
class Decompressor
{

public:
  /**
   * @brief Create a new Decompressor object. This method synchronizes with the
   * passed in stream.
   *
   * @param compressed_data The compressed data on the device to decompress.
   * @param compressed_data_size The size of the compressed data.
   * @param stream The stream to use to retrieve metadata from the device.
   *
   * @throw NVCompressionExcpetion If the metadata cannot be read from the
   * device.
   */
  Decompressor(
      const void* const compressed_data,
      const size_t compressed_data_size,
      cudaStream_t stream);

  // disable copying
  Decompressor(const Decompressor& other) = delete;
  Decompressor& operator=(const Decompressor& other) = delete;

  /**
   * @brief Decompress the given data asynchronously.
   *
   * @param temp_ptr The temporary workspace on the device to use.
   * @param temp_bytes The size of the temporary workspace.
   * @param out_ptr The location to write the uncompressed data to on the
   * device.
   * @param out_num_elements The size of the output location in number of
   * elements.
   * @param stream The stream to operate on.
   *
   * @throw NVCompException If decompression fails to launch on the stream.
   */
  void decompress_async(
      void* const temp_ptr,
      const size_t temp_bytes,
      T* const out_ptr,
      const size_t out_num_elements,
      cudaStream_t stream);

  /**
   * @brief Get the size of the temporary buffer required for decompression.
   *
   * @return The size in bytes.
   */
  size_t get_temp_size();

  /**
   * @brief Get the size of the output buffer in bytes.
   *
   * @return The size in bytes.
   */
  size_t get_output_size();

  /**
   * @brief Get the number of elements that will be decompressed.
   *
   * @return The number of elements.
   */
  size_t get_num_elements();

private:
  std::unique_ptr<void, void (*)(void*)> m_metadata;
  const void* m_compressed_data;
  size_t m_compressed_data_size;
};

/******************************************************************************
 * INLINE DEFINITIONS AND HELPER FUNCTIONS ************************************
 *****************************************************************************/

template <typename T>
inline nvcompType_t getnvcompType()
{
  if (std::is_same<T, int8_t>::value) {
    return NVCOMP_TYPE_CHAR;
  } else if (std::is_same<T, uint8_t>::value) {
    return NVCOMP_TYPE_UCHAR;
  } else if (std::is_same<T, int16_t>::value) {
    return NVCOMP_TYPE_SHORT;
  } else if (std::is_same<T, uint16_t>::value) {
    return NVCOMP_TYPE_USHORT;
  } else if (std::is_same<T, int32_t>::value) {
    return NVCOMP_TYPE_INT;
  } else if (std::is_same<T, uint32_t>::value) {
    return NVCOMP_TYPE_UINT;
  } else if (std::is_same<T, int64_t>::value) {
    return NVCOMP_TYPE_LONGLONG;
  } else if (std::is_same<T, uint64_t>::value) {
    return NVCOMP_TYPE_ULONGLONG;
  } else {
    throw NVCompException(
        nvcompErrorNotSupported, "nvcomp does not support the given type.");
  }
}

inline void throwExceptionIfError(nvcompError_t error, const std::string& msg)
{
  if (error != nvcompSuccess) {
    throw NVCompException(error, msg);
  }
}

template <typename T>
inline Compressor<T>::Compressor(
    const T* const in_elements, const size_t num_in_elements) :
    m_in_elements(in_elements),
    m_num_in_elements(num_in_elements)
{
  // do nothing
}

template <typename T>
inline void Compressor<T>::compress_async(
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  // Splitting into compress/do_compress allows putting extra logic in the
  // base class that all derived types should have, e.g., logging.
  return do_compress(temp_ptr, temp_bytes, out_ptr, out_bytes, stream);
}

template <typename T>
inline nvcompType_t Compressor<T>::get_type() const
{
  return getnvcompType<T>();
}

template <typename T>
inline size_t Compressor<T>::get_uncompressed_size() const
{
  return sizeof(*m_in_elements) * m_num_in_elements;
}

template <typename T>
inline const T* Compressor<T>::get_uncompressed_data() const
{
  return m_in_elements;
}

template <typename T>
inline size_t Compressor<T>::get_exact_output_size(
    void* const /* comp_temp */, const size_t /* comp_temp_bytes */)
{
  throw NVCompException(
      nvcompErrorNotSupported,
      "Getting the exact output "
      "buffer size is not supported by this compressor.");
}

template <typename T>
inline Decompressor<T>::Decompressor(
    const void* const compressed_data,
    const size_t compressed_data_size,
    cudaStream_t stream) :
    m_metadata(nullptr, nvcompDecompressDestroyMetadata),
    m_compressed_data(compressed_data),
    m_compressed_data_size(compressed_data_size)
{
  void* ptr;
  nvcompError_t err = nvcompDecompressGetMetadata(
      m_compressed_data, m_compressed_data_size, &ptr, stream);
  throwExceptionIfError(err, "Failed to get metadata");

  m_metadata.reset(ptr);

  // verify template type compatibility
  if (!LZ4IsMetadata(m_metadata.get())) {
    // lz4 is type independent
    nvcompType_t type;
    err = nvcompDecompressGetType(m_metadata.get(), &type);
    throwExceptionIfError(err, "Failed to get metadata");

    if (type != getnvcompType<T>()) {
      throw NVCompException(
          nvcompErrorInvalidValue,
          "Template type is "
          "not compatible with compressed type.");
    }
  }
}

template <typename T>
inline void Decompressor<T>::decompress_async(
    void* const temp_ptr,
    const size_t temp_bytes,
    T* const out_ptr,
    const size_t out_num_elements,
    cudaStream_t stream)
{
  const size_t num_bytes = out_num_elements * sizeof(*out_ptr);

  nvcompError_t err = nvcompDecompressAsync(
      m_compressed_data,
      m_compressed_data_size,
      temp_ptr,
      temp_bytes,
      m_metadata.get(),
      out_ptr,
      num_bytes,
      stream);
  throwExceptionIfError(err, "Failed to launch async decompression");
}

template <typename T>
inline size_t Decompressor<T>::get_temp_size()
{
  size_t bytes;
  nvcompError_t err = nvcompDecompressGetTempSize(m_metadata.get(), &bytes);
  throwExceptionIfError(
      err,
      "Failed to get temporary workspace size needed "
      "to decompress data: ");

  return bytes;
}

template <typename T>
inline size_t Decompressor<T>::get_output_size()
{
  size_t bytes;
  nvcompError_t err = nvcompDecompressGetOutputSize(m_metadata.get(), &bytes);
  throwExceptionIfError(err, "Failed to get decompressed size of data");

  return bytes;
}

template <typename T>
inline size_t Decompressor<T>::get_num_elements()
{
  return get_output_size() / sizeof(T);
}

} // namespace nvcomp

#endif

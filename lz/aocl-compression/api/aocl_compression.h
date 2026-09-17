/**
 * Copyright (C) 2022-2026, Advanced Micro Devices. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
 
 /** @file aocl_compression.h
 *  
 *  @brief Interface APIs and data structures of AOCL-Compression library
 *
 *  This file contains the unified interface API set and associated
 *  data structure.
 *
 *  @author S. Biplab Raut
 */
 
#ifndef AOCL_COMPRESSION_H
#define AOCL_COMPRESSION_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {

#endif
/**
 * \defgroup API        Standardized/Unified API
 * \defgroup LZ4_API    LZ4 API
 * \defgroup SNAPPY_API SNAPPY API
 * \defgroup ZLIB_API   ZLIB API
 * \defgroup BZIP2_API  BZIP2 API
 * \defgroup LZMA_API   LZMA API
 * \defgroup LZ4HC_API  LZ4HC API
 * \defgroup ZSTD_API   ZSTD API
 */

/**
 * \addtogroup API
 *  @brief Interface APIs and data structures of AOCL Compression library are described in this section.
 *
 *  This file contains the unified interface API set and associated data structure.
 * @{
 */

/// @cond DOXYGEN_SHOULD_SKIP_THIS
#ifndef EXPORT_SYM_DYN
#ifdef _WINDOWS
/**
 * You can export data, functions, classes, or class member functions from a DLL
 * using the __declspec(dllexport) keyword. __declspec(dllexport) adds the export
 * directive to the object file so you do not need to use a .def file.
 * 
 */
#define EXPORT_SYM_DYN __declspec(dllexport)
#else
/**
 * For Linux EXPORT_SYM_DYN is NULL, by default the symbols are publicly exposed.
 */
#define EXPORT_SYM_DYN
#endif
#endif

#define AOCL_COMPRESSION_LIBRARY_VERSION "AOCL-Compression 5.3.0"
#define INTERNAL_LIBRARY_VERSION "AOCL LOSSLESS DATA COMPRESSION 4.0"
/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

 /**
  * @brief Error codes supported by unified APIs of AOCL-Compression library.
  *
  * 
  */
typedef enum
{
    ERR_MEMORY_ALLOC = -6,         ///<Memory allocation failure
    ERR_INVALID_INPUT,             ///<Invalid input parameter provided
    ERR_UNSUPPORTED_METHOD,        ///<Compression method not supported by the library
    ERR_EXCLUDED_METHOD,           ///<Compression method excluded from this library build
    ERR_COMPRESSION_FAILED,        ///<Failure during compression/decompression
    ERR_COMPRESSION_INVALID_OUTPUT ///<Invalid compression/decompression output
} aocl_error_type;

/**
 * @brief Types of compression methods supported.
 * 
 * Optimizations are included for all the supported methods
 */
typedef enum
{
    LZ4 = 0,
    LZ4HC,
    LZMA,
    BZIP2,
    SNAPPY,
    ZLIB,
    ZSTD,
    AOCL_COMPRESSOR_ALGOS_NUM
} aocl_compression_type;

/**
 * @brief This acts as a handle for the compression and decompression of AOCL Compression library.
 * 
 */
typedef struct
{
    char *inBuf;         /**<  Pointer to input buffer data                           */
    char *outBuf;        /**<  Pointer to output buffer data                          */
    char *workBuf;       /**<  Pointer to temporary work buffer                       */
    size_t inSize;       /**<  Input data length                                      */                      
    size_t outSize;      /**<  Output data length                                     */ 
    size_t level;        /**<  Requested compression level                            */
    size_t optVar;       /**<  Additional variables or parameters                     */
    int numThreads;      /**<  Number of threads available for multi-threading        */
    int numMPIranks;     /**<  Number of available multi-core MPI ranks               */
    size_t memLimit;     /**<  Maximum memory limit for compression/decompression     */
    int measureStats;    /**<  Measure speed and size of compression/decompression    */
    uint64_t cSize;      /**<  Size of compressed output                              */
    uint64_t dSize;      /**<  Size of decompressed output                            */
    uint64_t cTime;      /**<  Time to compress input                                 */
    uint64_t dTime;      /**<  Time to decompress input                               */
    float cSpeed;        /**<  Speed of compression                                   */
    float dSpeed;        /**<  Speed of decompression                                 */
    int optOff;          /**<  Turn off all optimizations                             */
    int optLevel;        /**<  Optimization level:  \n
                               0 - non-SIMD algorithmic optimizations, \n
                               1 - SSE2 optimizations, \n
                               2 - AVX optimizations, \n
                               3 - AVX2 optimizations, \n
                               4 - AVX512 optimizations                               */
    //size_t chunk_size; //Unused variable
} aocl_compression_desc;

/**
 * @brief Interface API to provide the maximum size that compression may output in a "worst case" scenario (input data not compressible).
 * 
 * This function is primarily useful for memory allocation purposes (destination buffer size).
 * 
 * | Parameters | Direction   | Description |
 * |:-----------|:-----------:|:------------|
 * | \b codec_type | in      | Select the algorithm to be used for compression, choose from aocl_compression_type. |
 * | \b inSize     | in      | The size of input data to be compressed in bytes. |
 * 
 * @note inSize cannot exceed maximum supported value for respective codec_type.
 * 
 * @return 
 * | Result     | Description |
 * |:-----------|:------------|
 * | Success    |Returns an upper bound on the compressed size.      |
 * | Fail       |`ERR_COMPRESSION_FAILED`                            |
 */
EXPORT_SYM_DYN int64_t aocl_llc_compressBound(aocl_compression_type codec_type,
                                            size_t inSize);

/**
 * @brief Interface API to compress data.
 * 
 * | Parameters | Direction   | Description |
 * |:-----------|:-----------:|:------------|
 * | \b handle     | in,out  | This acts as a handle for compression and decompression. For more information, refer to aocl_compression_desc. |
 * | \b codec_type | in      | Select the algorithm to be used for compression, choose from aocl_compression_type. |
 * 
 * 
 * @return 
 * | Result     | Description |
 * |:-----------|:------------|
 * | Success    |Number of bytes decompressed      |
 * | Fail       |`ERR_COMPRESSION_FAILED`          |
 * | ^          |`ERR_COMPRESSION_INVALID_OUTPUT`  |
 */
EXPORT_SYM_DYN int64_t aocl_llc_compress(aocl_compression_desc *handle,
                            aocl_compression_type codec_type);

/**
 * @brief Interface API to decompress data.
 * 
 * | Parameters | Direction   | Description |
 * |:-----------|:-----------:|:------------|
 * | \b handle     | in,out  | This acts as a handle for compression and decompression. For more information, refer to aocl_compression_desc. |
 * | \b codec_type | in      | Select the algorithm to be used for compression, choose from aocl_compression_type. |
 * 
 * @return 
 * | Result     | Description |
 * |:-----------|:------------|
 * | Success    |Numbers of bytes decompressed     |
 * | Fail       |`ERR_COMPRESSION_FAILED`          |
 * | ^          |`ERR_COMPRESSION_INVALID_OUTPUT`  |
 * 
 */

EXPORT_SYM_DYN int64_t aocl_llc_decompress(aocl_compression_desc *handle,
                              aocl_compression_type codec_type);

/**
 * @brief Interface API to setup the compression method.
 * 
 * | Parameters | Direction   | Description |
 * |:-----------|:-----------:|:------------|
 * | \b handle     | in,out  | This acts as a handle for compression and decompression. For more information, refer to aocl_compression_desc. |
 * | \b codec_type | in      | Select the algorithm to be used for compression, choose from aocl_compression_type. |
 * 
 * @return 
 * | Result     | Description |
 * |:-----------|:------------|
 * | Success    | \b 0                           |
 * | Fail       | `ERR_UNSUPPORTED_METHOD`       |
 * | ^          | `ERR_EXCLUDED_METHOD`          |
 * 
 */

EXPORT_SYM_DYN int32_t aocl_llc_setup(aocl_compression_desc *handle,
                      aocl_compression_type codec_type);

/**
 * @brief Interface API to destroy the compression method.
 * 
 * | Parameters | Direction   | Description |
 * |:-----------|:-----------:|:------------|
 * | \b handle     | in,out  | This acts as a handle for compression and decompression. For more information, refer to aocl_compression_desc. |
 * | \b codec_type | in      | Select the algorithm to be used for compression, choose from aocl_compression_type. |
 * 
 * @return void 
 */
EXPORT_SYM_DYN void aocl_llc_destroy(aocl_compression_desc *handle,
                        aocl_compression_type codec_type);
/**
 * @brief Interface API to get the compression library version string.
 * 
 * @return AOCL library version 
 */

EXPORT_SYM_DYN const char *aocl_llc_version(void);

/**
 * @brief Interface API to get the length of the RAP frame in the compressed stream.
 *
 * Legacy single threaded decompressors can call this API to know how many bytes of the compressed
 * stream to skip to get the format compliant compressed stream that they can decompress.
 * 
 * @note  Presence of RAP frame is determined by checking for the magic word: 0x434C4C5F4C434F41 
 *        (ASCII encoding of AOCL_LLC) at the start of the stream.
 *
 * | Parameters      | Direction   | Description |
 * |:----------------|:-----------:|:------------|
 * | \b src          | in          | Input stream buffer pointer. |
 * | \b src_size     | in          | Input stream buffer size. |
 * 
 * @return
 * | Result     | Description |
 * |:-----------|:------------|
 * | RAP_frame_length | Length of RAP frame bytes in src |
 * | 0                | If RAP frame does not exist      |
 * | Fail             | `ERR_INVALID_INPUT`              |
 *
 */
EXPORT_SYM_DYN int32_t aocl_llc_skip_rap_frame(char* src, int32_t src_size);

/**
 * @brief Interface API to set maximum number of OpenMP threads used by AOCL
 * multithreaded compression/decompression flow for the calling application
 * thread.
 *
 * @note  This API does not validate or prevent thread oversubscription.
 *        Application-level thread orchestration remains the caller's
 *        responsibility.
 * 
 * If this API is not called, thread selection defaults to `omp_get_max_threads()`.
 *
 * | Parameters      | Direction   | Description |
 * |:----------------|:-----------:|:------------|
 * |    max_threads  | in          | Maximum threads to use. Must be > 0. |
 *
 * @return
 * | Result     | Description |
 * |:-----------|:------------|
 * | Success    | 0 |
 * | Fail       | `ERR_INVALID_INPUT` |
 * | ^          | `ERR_UNSUPPORTED_METHOD` when AOCL is built without thread support |
 */
EXPORT_SYM_DYN int32_t aocl_llc_set_max_threads(int32_t max_threads);

/**
 * @}
 */
#ifdef __cplusplus
}
#endif

#endif

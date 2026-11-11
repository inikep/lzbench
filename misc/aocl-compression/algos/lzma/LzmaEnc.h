/*  LzmaEnc.h -- LZMA Encoder
2019-10-30 : Igor Pavlov : Public domain */

/*
* Modifications Copyright (C) 2022-2026, Advanced Micro Devices. All rights reserved.
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

#ifndef __LZMA_ENC_H
#define __LZMA_ENC_H

#include "7zTypes.h"

EXTERN_C_BEGIN

/*!
 * \addtogroup LZMA_API
 * @{
*/

/// @cond DOXYGEN_SHOULD_SKIP_THIS
#define LZMA_PROPS_SIZE 5
/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

/**
 * @brief Structure to hold configurable parameters that can be used by LZMA encoder.
 * 
 */
typedef struct _CLzmaEncProps
{
  int level;        /**< Control degree of compression. Lower level gives less compression at higher speed. \n 0 <= level <= 9 */
  UInt32 dictSize;  /**< size of dictionary / search buffer \n (1 << 12) <= dictSize <= (1 << 27) for 32-bit version \n
                      (1 << 12) <= dictSize <= (3 << 29) for 64-bit version \n 
                      \b default = (1 << 24) */
  int lc;           /**< Number of high bits of the previous byte to use as a context for literal encoding. \n 
                        0 <= lc <= 8, \n \b default = 3 */
  int lp;           /**< Number of low bits of the dictionary position to include in literal posState. \n 
                         0 <= lp <= 4, \n \b default = 0 */
  int pb;           /**< Number of low bits of processedPos to include in posState. \n 
                         0 <= pb <= 4, \n \b default = 2 */
  int algo;         /**< Dictionary search algo to use: 0 - fast (hash chain), 1 - normal (binary search tree). \n 
                         0 - fast, 1 - normal, \n \b default = 1 */
  int fb;           /**< Number of fast bytes. \n 
                         5 <= fb <= 273, \n \b default = 32 */
  int btMode;       /**< 0 - hashChain Mode, 1 - binTree mode - normal, \n \b default = 1 */
  int numHashBytes; /**< Number of bytes used to compute hash. \n 
                         2, 3 or 4, \n \b default = 4 */
  UInt32 mc;        /**< Cut value, limit on number of nodes to search in dictionary. \n 
                         1 <= mc <= (1 << 30), \n \b default = 32 */
  unsigned writeEndMark;  /**< 0 - do not write EOPM, 1 - write EOPM, \n \b default = 0 */
  int numThreads;   /**< Threads used for processing. \n 
                         1 or 2, \n \b default = 1 */

  UInt64 reduceSize; /**< estimated size of data that will be compressed. default = (UInt64)(Int64)-1. \n 
                        Encoder uses this value to reduce dictionary size */

  UInt64 affinity; /**< thread affinity mask for multi-threading */

#ifdef AOCL_LZMA_OPT
  size_t srcLen;
  int cacheEfficientStrategy; /**< 0: disabled, 1 : enabled, -1: optimal defaults */
#endif
} CLzmaEncProps;

/*!
 * @name Encode Functions
 * @{
 */

/*! @brief Init properties. Properties are set to auto select mode.
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b p       | out         | Lzma encode properties object |
*
* @return void
* 
*/
LZMALIB_API void LzmaEncProps_Init(CLzmaEncProps *p);

/*! @brief Set default values for properties in auto select mode.
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b p       | out         | Lzma encode properties object |
*
* @return void
* 
*/
LZMALIB_API void LzmaEncProps_Normalize(CLzmaEncProps *p);

/*! @brief Normalize props2 and return dictionary size.
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b props2  | out         | Lzma encode properties object |
*
* @return dictionary size
*/
LZMALIB_API UInt32 LzmaEncProps_GetDictSize(const CLzmaEncProps *props2);

/**
 * @}
*/

/* ---------- CLzmaEncHandle Interface ---------- */

/* LzmaEnc* functions can return the following exit codes:
SRes:
  SZ_OK           - OK
  SZ_ERROR_MEM    - Memory allocation error
  SZ_ERROR_PARAM  - Incorrect parameter in props
  SZ_ERROR_WRITE  - ISeqOutStream write callback error
  SZ_ERROR_OUTPUT_EOF - output buffer overflow - version with (Byte *) output
  SZ_ERROR_PROGRESS - some break from progress callback
  SZ_ERROR_THREAD - error in multithreading functions (only for Mt version)
*/

/** @brief Pointer to context object that maintains state of LZMA encoder. */
typedef void * CLzmaEncHandle;

/*!
 * @name Encode Functions
 * @{
 */

/*! @brief Construct Lzma encoder
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b alloc   | in          | Allocator object |
*
* @return Lzma encoder handle
* 
*/
LZMALIB_API CLzmaEncHandle LzmaEnc_Create(ISzAllocPtr alloc);

/*! @brief Free Lzma encoder
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b p          | out         | Lzma encoder handle |
* | \b alloc      | in          | Allocator object |
* | \b allocBig   | in          | Allocator object for large blocks |
*
* @return void
*
* @note p is invalid after this call and must not be reused or destroyed again.
*       Passing NULL is a no-op.
*/
LZMALIB_API void LzmaEnc_Destroy(CLzmaEncHandle p, ISzAllocPtr alloc, ISzAllocPtr allocBig);

/*! @brief Update properties in p with values in props
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b p       | out         | Lzma encoder handle |
* | \b props   | in          | Lzma encoder properties |
*
* @return 
* | Result     | Description |
* |:-----------|:------------|
* | Success    |SZ_OK                      |
* | Fail       |SZ_ERROR_PARAM      - Incorrect parameter in props  |
* 
*/
LZMALIB_API SRes LzmaEnc_SetProps(CLzmaEncHandle p, const CLzmaEncProps *props);

/*! @brief Set expected data size in p to expectedDataSiize
*
* | Parameters           | Direction   | Description |
* |:---------------------|:-----------:|:------------|
* | \b p                 | out         | Lzma encoder handle |
* | \b expectedDataSiize | in          | Expected size of data |
*
* @return void
* 
*/
LZMALIB_API void LzmaEnc_SetDataSize(CLzmaEncHandle p, UInt64 expectedDataSiize);

/*! @brief Build header bytes and save in properties
*
* | Parameters    | Direction   | Description |
* |:--------------|:-----------:|:------------|
* | \b p          | in          | Lzma encoder handle |
* | \b properties | out         | Buffer to write header bytes into |
* | \b size       | in,out      | Number of bytes saved in properties |
*
* @return 
* | Result     | Description |
* |:-----------|:------------|
* | Success    |SZ_OK        |
* | Fail       |SZ_ERROR_PARAM      - Incorrect parameter in props |
* 
*/
LZMALIB_API SRes LzmaEnc_WriteProperties(CLzmaEncHandle p, Byte *properties, SizeT *size);

/*! @brief Get write end mark saved within p
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b p       | in          | Lzma encoder handle |
*
* @return writeEndMark
* 
*/
LZMALIB_API unsigned LzmaEnc_IsWriteEndMark(CLzmaEncHandle p);

/*! @brief Incremental compression using streaming interfaces
*
* | Parameters      | Direction   | Description |
* |:----------------|:-----------:|:------------|
* | \b p            | in,out      | Lzma encoder handle |
* | \b outStream    | in, out     | Properly initialized interface to save compressed data |
* | \b inStream     | in          | Properly initialized interface to read uncompressed data |
* | \b alloc        | in          | Allocator object |
* | \b allocBig     | in          | Allocator object for large blocks |
*
* @return 
* | Result     | Description |
* |:-----------|:------------|
* | Success    |SZ_OK                      |
* | Fail       |SZ_ERROR_MEM        - Memory allocation error |
* | ^          |SZ_ERROR_PARAM      - Incorrect parameters    |
* | ^          |SZ_ERROR_WRITE      - ISeqOutStream write callback error |
* | ^          |SZ_ERROR_READ       - ISeqOutStream read callback error  |
* | ^          |SZ_ERROR_PROGRESS   - some break from progress callback  |
* 
* @note Passed allocator objects should be valid.
*/
LZMALIB_API SRes LzmaEnc_Encode(CLzmaEncHandle p, ISeqOutStream *outStream, ISeqInStream *inStream,
    ICompressProgress *progress, ISzAllocPtr alloc, ISzAllocPtr allocBig);


/*! @brief Encode src in-memory and save compressed data to dest
*
* | Parameters      | Direction   | Description |
* |:----------------|:-----------:|:------------|
* | \b p            | in,out      | Lzma encoder handle |
* | \b dest         | out         | Destination buffer to hold compressed data |
* | \b destLen      | out         | Size of compressed data written to dest |
* | \b src          | in          | Source buffer with uncompressed data |
* | \b srcLen       | in          | Size of uncompressed data in src |
* | \b writeEndMark | in          | If non-0, finish stream with end mark |
* | \b progress     | in          | Compression progress indicator |
* | \b alloc        | in          | Allocator object |
* | \b allocBig     | in          | Allocator object for large blocks |
*
* @return 
* | Result     | Description |
* |:-----------|:------------|
* | Success    |SZ_OK                      |
* | Fail       |SZ_ERROR_MEM        - Memory allocation error |
* | ^          |SZ_ERROR_PARAM      - Incorrect parameter in props  |
* | ^          |SZ_ERROR_WRITE      - ISeqOutStream write callback error  |
* | ^          |SZ_ERROR_OUTPUT_EOF - output buffer overflow - version with (Byte *) output  |
* | ^          |SZ_ERROR_PROGRESS   - some break from progress callback  |
* 
* @note Passed allocator objects should be valid.
*/
LZMALIB_API SRes LzmaEnc_MemEncode(CLzmaEncHandle p, Byte *dest, SizeT *destLen, const Byte *src, SizeT srcLen,
    int writeEndMark, ICompressProgress *progress, ISzAllocPtr alloc, ISzAllocPtr allocBig);

/**
 * @}
 */

/// @cond DOXYGEN_SHOULD_SKIP_THIS
/*
* Provides the maximum size that LZMA compression may output in a "worst case" scenario (input data not compressible)
* where `insize` is the size of source buffer to be compressed.
*/
size_t Lzma_compressBound(size_t insize);
/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

/* ---------- One Call Interface ---------- */
/*!
 * @name Encode One Call Interface
 * @{
 */

/*!
*
*  @rst 
*  .. _LzmaEncode:
*  @endrst
*
* @brief Encode data in src and save compressed data to dest
*
* | Parameters      | Direction   | Description |
* |:----------------|:-----------:|:------------|
* | \b dest         | out         | Destination buffer to hold compressed data |
* | \b destLen      | out         | Size of compressed data written to dest |
* | \b src          | in          | Source buffer with uncompressed data |
* | \b srcLen       | in          | Size of uncompressed data in src |
* | \b props        | in          | Properties to control compression method |
* | \b propsEncoded | out         | Buffer to save header bytes |
* | \b propsSize    | out         | Size of header bytes |
* | \b writeEndMark | in          | If non-0, finish stream with end mark |
* | \b progress     | in          | Compression progress indicator |
* | \b alloc        | in          | Allocator object |
* | \b allocBig     | in          | Allocator object for large blocks |
*
* @return
* | Result     | Description |
* |:-----------|:------------|
* | Success    |SZ_OK                      |
* | Fail       |SZ_ERROR_MEM        - Memory allocation error |
* | ^          |SZ_ERROR_PARAM      - Incorrect parameter in props  |
* | ^          |SZ_ERROR_WRITE      - ISeqOutStream write callback error  |
* | ^          |SZ_ERROR_OUTPUT_EOF - output buffer overflow - version with (Byte *) output  |
* | ^          |SZ_ERROR_PROGRESS   - some break from progress callback  |
*
*/
LZMALIB_API SRes LzmaEncode(Byte *dest, SizeT *destLen, const Byte *src, SizeT srcLen,
    const CLzmaEncProps *props, Byte *propsEncoded, SizeT *propsSize, int writeEndMark,
    ICompressProgress *progress, ISzAllocPtr alloc, ISzAllocPtr allocBig);

/**
 * @}
*/

/// @cond DOXYGEN_SHOULD_SKIP_THIS

/**
 * @name AOCL Functions
 * @brief These functions are not part of open source code, these are introduced by AOCL-Compression
 * library to control AOCL introduced optimization levels dynamically.
 * 
 * @note These functions are for internal purposes only, not recommended for external use.
 * 
 * @{
 */

/*!
 * @brief AOCL-Compression defined setup function that configures code path dynamically with the right
 * AMD optimized lzma routines depending upon the detected CPU features if `optOff=0`.
 * 
 * Except for the initial call, it's necessary to execute aocl_destroy_lzma_encode() before any subsequent calls
 * to this function. Failure to call the destroy function prior to invoking this function will result
 * in lzma following the code path of set at first setup call or  the most recent setup call that was
 * preceded by the destroy function.
 * 
 * @param optOff Turn on/off all AOCL-Compression optimizations.
 * @param optLevel Optimization level: 0 - C optimization, 1 - SSE2, 2 - AVX, 3 - AVX2, 4 - AVX512 .
 * @param insize Input data length.
 * @param level Requested compression level.
 * @param windowLog Largest match distance : larger == more compression, more memory needed during decompression.
 * 
 * @return \b NULL .
 */
LZMALIB_API void aocl_setup_lzma_encode(int optOff, int optLevel, size_t insize,
  size_t level, size_t windowLog);

/*!
 * @brief It is necessary to execute this destroy function after the initial invocation of the
 * aocl_setup_lzma_encode() function, prior to initiating the setup function again.
 */
LZMALIB_API void aocl_destroy_lzma_encode(void);

/**
 * @}
 */

/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

/*!
 * @name Encode Functions
 * @{
 */
#ifdef AOCL_LZMA_OPT
/*! @brief AOCL optimized LzmaEncProps_Normalize function. \n 
*          Set default values for properties in auto select mode.
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b p       | out         | Lzma encode properties object |
*
* @return void
* 
*/
LZMALIB_API void AOCL_LzmaEncProps_Normalize(CLzmaEncProps* p);

/*! @brief AOCL optimized LzmaEnc_SetProps function. \n
*          Update properties in pp with values in props2
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b pp       | out         | Lzma encoder handle |
* | \b props2   | in          | Lzma encoder properties |
*
* @return 
* | Result     | Description |
* |:-----------|:------------|
* | Success    |SZ_OK                      |
* | Fail       |SZ_ERROR_PARAM      - Incorrect parameter in props2  |
* 
*/
LZMALIB_API SRes AOCL_LzmaEnc_SetProps(CLzmaEncHandle pp, const CLzmaEncProps* props2);
#endif

/**
 * @}
*/

#ifdef AOCL_UNIT_TEST
typedef struct { // struct to expose CLzmaEnc properties for unit testing
    unsigned numFastBytes;
    unsigned lc, lp, pb;
    BoolInt fastMode;
    BoolInt writeEndMark;
    UInt32 dictSize;

    //MFB params
    Byte btMode;
    UInt32 cutValue;
    UInt16 level;
    UInt16 cacheEfficientSearch;
    UInt32 numHashBytes;
} TestCLzmaEnc;

LZMALIB_API void Test_LzmaEncProps_Normalize_Dyn(CLzmaEncProps* p);
LZMALIB_API SRes Test_SetProps_Dyn(CLzmaEncHandle pp, const CLzmaEncProps* props);
LZMALIB_API UInt64 Test_SetDataSize(CLzmaEncHandle pp, UInt64 expectedDataSize);
LZMALIB_API SRes Test_WriteProperties(CLzmaEncHandle pp, Byte* props, SizeT* size, UInt32 dictSize);
LZMALIB_API unsigned Test_IsWriteEndMark(CLzmaEncHandle pp, unsigned wem);
LZMALIB_API TestCLzmaEnc Get_CLzmaEnc_Params(CLzmaEncHandle pp);
#endif

/**
 * @}
 */

EXTERN_C_END

#endif

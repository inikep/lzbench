/*
   LZ4 HC - High Compression Mode of LZ4
   Header File
   Copyright (C) 2011-2020, Yann Collet.
   Modifications Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.

   BSD 2-Clause License (http://www.opensource.org/licenses/bsd-license.php)

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

       * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following disclaimer
   in the documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   You can contact the author at :
   - LZ4 source repository : https://github.com/lz4/lz4
   - LZ4 public forum : https://groups.google.com/forum/#!forum/lz4c
*/


#ifndef LZ4_HC_H_19834876238432
#define LZ4_HC_H_19834876238432

#if defined (__cplusplus)
extern "C" {
#endif

/*!
 * \addtogroup LZ4HC_API
 *
 * @brief  LZ4HC is a high compression variant of LZ4. This HC variant makes a full search, in contrast with the fast scan of 
 * “regular LZ4”, resulting in improved compression ratio.  But the speed is quite impacted: the HC version is between 6x and 10x 
 * slower than the Fast one.
 * 
 * LZ4HC compression uses lesser memory as it loads individual Assets from an Asset Bundle quickly. \n
 * Similar to LZ4, the LZ4HC also provides in-memory compression and decompression.
 * LZ4HC generates the LZ4-compressed block like LZ4 and uses the same decompression technique. 
 * 
 * @{
*/

/* --- Dependency --- */
/* note : lz4hc requires lz4.h/lz4.c for compilation */
#include "lz4.h"   /* stddef, LZ4LIB_API, LZ4_DEPRECATED */
#include "aoclAlgoOpt.h" /* AOCL Optimization flags */

/// @cond DOXYGEN_SHOULD_SKIP_THIS 

/* --- Useful constants --- */
#define LZ4HC_CLEVEL_MIN         2
#define LZ4HC_CLEVEL_DEFAULT     9
#define LZ4HC_CLEVEL_OPT_MIN    10
#define LZ4HC_CLEVEL_MAX        12

/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */


/*-************************************
 *  Block Compression
 **************************************/
/**
 * @name Block Compression Functions
 * @{
*/
/*!
 *  @rst 
 *  .. _LZ4_compress_HC:
 *  @endrst
 * 
 *  @brief Compress data from `src` into `dst`, using the powerful but slower "HC" algorithm.
 *
 * | Parameters | Direction   | Description |
 * |:-----------|:-----------:|:------------|
 * | \b src              | in  | Source buffer, the data which you want to compress is copied/or pointed here. |
 * | \b dst              | out | Destination buffer, compressed data is kept here, memory should be allocated already. |
 * | \b srcSize          | in  | Size of buffer `src`. Maximum supported value is LZ4_MAX_INPUT_SIZE. |
 * | \b dstCapacity      | in  | Size of buffer `dst` (which must be already allocated). |
 * | \b compressionLevel | in  | It is used to set the correct context level for compression. |
 * 
 *  | Result | Description |
 *  |:-------|:------------|
 *  | success| The number of bytes written into `dst` |
 *  | Fail   |  0                                     |
 * 
 *  @note Compression is guaranteed to succeed if `dstCapacity >= LZ4_compressBound(srcSize)`.
 *  @note `compressionLevel` : value between 1 and LZ4HC_CLEVEL_MAX (inclusive) will work. \n
 *         Values > `LZ4HC_CLEVEL_MAX` behave the same as `LZ4HC_CLEVEL_MAX`.
 * 
 */
LZ4LIB_API int LZ4_compress_HC (const char* src, char* dst, int srcSize, int dstCapacity, int compressionLevel);

/* Note :
 *   Decompression functions are provided within "lz4.h" (BSD license)
 */
 

/*! 
 *  @brief  This function is used to provide the size of `state`.
 *  @return Returns the amount of memory which must be allocated for its state.
 */
LZ4LIB_API int LZ4_sizeofStateHC(void);
#ifdef AOCL_LZ4HC_OPT
/*!
 * @brief  This function is the AOCL variant of LZ4_sizeofStateHC() and used to provide the size of `state` of type AOCL_LZ4_streamHC_t.
 * @return Returns the amount of memory which must be allocated to its state.
 */
LZ4LIB_API int AOCL_LZ4_sizeofStateHC(void);
#endif

/*!
 * @brief Same as LZ4_compress_HC(), but using an externally allocated memory segment for `state`.
 * 
 * | Parameters | Direction   | Description |
 * |:-----------|:-----------:|:------------|
 * | \b stateHC          | in,out | It acts as a handle for compression. |
 * | \b src              | in     | Source buffer, the data which you want to compress is copied/or pointed here. |
 * | \b dst              | out    | Destination buffer, compressed data is kept here, memory should be allocated already. |
 * | \b srcSize          | in     | Size of buffer `src`. Maximum supported value is LZ4_MAX_INPUT_SIZE. |
 * | \b maxDstSize       | in     | Size of buffer `dst` (which must be already allocated). |
 * | \b compressionLevel | in     | It is used to set the correct context level for compression. |
 * 
 *  | Result | Description |
 *  |:-------|:------------|
 *  | success| The number of bytes written into `dst` |
 *  | Fail   |  0                                     |
 * 
 * @note `stateHC` When environment variable AOCL_DISABLE_OPT is ON, size of stateHC must be provided by LZ4_sizeofStateHC(). 
 *               When AOCL_DISABLE_OPT is OFF, size of stateHC must be provided by AOCL_LZ4_sizeofStateHC() for compression level 6, 7, 8, 9 and
 *               LZ4_sizeofStateHC() for other levels respectively.
 * @note Memory segment must be aligned on 8-bytes boundaries (which a normal `malloc()` should do properly).
*/
LZ4LIB_API int LZ4_compress_HC_extStateHC(void* stateHC, const char* src, char* dst, int srcSize, int maxDstSize, int compressionLevel);

/*! 
 *  @brief Will compress as much data as possible from `src` to fit into `targetDstSize` budget.
 * 
 *  |Parameters|Direction|Description|
 *  |:---------|:-------:|:----------|
 *  | \b stateHC          | in,out | It acts as a handle for compression.|
 *  | \b src              |  in    | Source buffer, the data which you want to compress is copied/or pointed here.|
 *  | \b dst              |  out   | Destination buffer, compressed data is kept here, memory should be allocated already.|
 *  | \b srcSizePtr       | in,out | Pointer to size of buffer `src`. Maximum supported value is LZ4_MAX_INPUT_SIZE.|
 *  | \b targetDstSize    |  in    | Size of buffer `dst` (which must be already allocated).|
 *  | \b compressionLevel |  in    | It is used to set the correct context level for compression. |
 * 
 *  @return 
 *  | Result | Description |
 *  |:-------|:------------|
 *  | success| The number of bytes written into `dst` (necessarily <= targetDstSize) |
 *  | ^      | `*srcSizePtr` is updated to indicate how much bytes were read from `src` |
 *  | Fail   |  0                                     |
 * 
 * @note `stateHC` When environment variable AOCL_DISABLE_OPT is ON, size of stateHC must be provided by LZ4_sizeofStateHC(). 
 *               When AOCL_DISABLE_OPT is OFF, size of stateHC must be provided by AOCL_LZ4_sizeofStateHC() for compression level 6, 7, 8, 9 and
 *               LZ4_sizeofStateHC() for other levels respectively.
 * @note Memory segment must be aligned on 8-bytes boundaries (which a normal `malloc()` should do properly).
 * @warning Requires v1.9.0+
 */
LZ4LIB_API int LZ4_compress_HC_destSize(void* stateHC,
                                        const char* src, char* dst,
                                        int* srcSizePtr, int targetDstSize,
                                        int compressionLevel);
/**
 * @}
*/

/// @cond DOXYGEN_SHOULD_SKIP_THIS

/**
 * @name AOCL Functions
 * @brief These functions are not part of open source code, these are introduced by AOCL-Compression
 * library to control AOCL introduced optimization dynamically.
 * 
 * @note These functions are for internal purposes only, not recommended for external use.
 * 
 * @{
 */

/**
 * @brief AOCL-Compression defined setup function that configures code path dynamically with the right
 * AMD optimized lz4hc routines depending upon the detected CPU features if `optOff=0`.
 * 
 * Except for the initial call, it's necessary to execute aocl_destroy_lz4hc() before any subsequent calls
 * to this function. Failure to call the destroy function prior to invoking this function will result
 * in lz4hc following the code path of set at first setup call or  the most recent setup call that was
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
LZ4LIB_API char* aocl_setup_lz4hc(int optOff, int optLevel, size_t insize,
    size_t level, size_t windowLog);

/**
 * @brief It is necessary to execute this destroy function after the initial invocation of the
 * aocl_setup_lz4hc() function, prior to initiating the setup function again.
 */
LZ4LIB_API void aocl_destroy_lz4hc(void);

/**
 * @}
 */

/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

/*-************************************
 *  Streaming Compression
 *  Bufferless synchronous API
 **************************************/
/**
 * @name Streaming Compression Functions
 * @{
*/

/// @cond DOXYGEN_SHOULD_SKIP_THIS
 typedef union LZ4_streamHC_u LZ4_streamHC_t;   /* incomplete type (defined later) */
 typedef union AOCL_LZ4_streamHC_u AOCL_LZ4_streamHC_t;   /* incomplete type (defined later) */
 /// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

/* LZ4_createStreamHC() and LZ4_freeStreamHC() :
 *  These functions create and release memory for LZ4 HC streaming state.
 *  Newly created states are automatically initialized.
 *  A same state can be used multiple times consecutively,
 *  starting with LZ4_resetStreamHC_fast() to start a new stream of blocks.
 */

/*!
 * @brief Creates the memory for LZ4 HC streaming state.
 * Newly created states are automatically initialized.
 * 
 *  @return 
 *  | Result  | Description |
 *  |:--------|:------------|
 *  | Success | Pointer to LZ4HC streaming state |
 *  | Fail    | \b NULL  |
*/
LZ4LIB_API LZ4_streamHC_t* LZ4_createStreamHC(void);

#ifdef AOCL_LZ4HC_OPT
 /*  AOCL_LZ4_createStreamHC() and AOCL_LZ4_freeStreamHC() :
  *  These functions create and release memory for LZ4 HC streaming state.
  *  Newly created states are automatically initialized.
  *  A same state can be used multiple times consecutively,
  *  starting with AOCL_LZ4_resetStreamHC_fast() to start a new stream of blocks.
  */

  /*!
   * @brief Creates the memory for LZ4 HC streaming state of type AOCL_LZ4_StreamHC_t.
   * Newly created states are automatically initialized.
   *
   *  @return
   *  | Result  | Description |
   *  |:--------|:------------|
   *  | Success | Pointer to LZ4HC streaming state |
   *  | Fail    | \b NULL  |
  */
LZ4LIB_API AOCL_LZ4_streamHC_t* AOCL_LZ4_createStreamHC(void);
#endif  /* AOCL_LZ4HC_OPT*/

/*!
 * @brief Releases memory for LZ4 HC streaming state.
 * 
 *  |Parameters|Direction|Description|
 *  |:---------|:-------:|:----------|
 *  | \b streamHCPtr | in,out | Streaming compression state that is automatically initialized. The same state can be re-used multiple times.|
 * 
 * @return \b 0
*/
LZ4LIB_API int             LZ4_freeStreamHC (LZ4_streamHC_t* streamHCPtr);

#ifdef AOCL_LZ4HC_OPT
 /*!
  * @brief Releases memory for LZ4 HC streaming state of type AOCL_LZ4_StreamHC_t.
  *
  *  |Parameters|Direction|Description|
  *  |:---------|:-------:|:----------|
  *  | \b streamHCPtr | in,out | Streaming compression state that is automatically initialized. The same state can be re-used multiple times.|
  *
  * @return \b 0
 */
LZ4LIB_API int             AOCL_LZ4_freeStreamHC(AOCL_LZ4_streamHC_t* streamHCPtr);
#endif  /* AOCL_LZ4HC_OPT */


/*
  These functions compress data in successive blocks of any size,
  using previous blocks as dictionary, to improve compression ratio.
  One key assumption is that previous blocks (up to 64 KB) remain read-accessible while compressing next blocks.
  There is an exception for ring buffers, which can be smaller than 64 KB.
  Ring-buffer scenario is automatically detected and handled within LZ4_compress_HC_continue().

  Before starting compression, state must be allocated and properly initialized.
  LZ4_createStreamHC() does both, though compression level is set to LZ4HC_CLEVEL_DEFAULT.

  Selecting the compression level can be done with LZ4_resetStreamHC_fast() (starts a new stream)
  or LZ4_setCompressionLevel() (anytime, between blocks in the same stream) (experimental).
  LZ4_resetStreamHC_fast() only works on states which have been properly initialized at least once,
  which is automatically the case when state is created using LZ4_createStreamHC().

  After reset, a first "fictional block" can be designated as initial dictionary,
  using LZ4_loadDictHC() (Optional).
  Note: In order for LZ4_loadDictHC() to create the correct data structure,
  it is essential to set the compression level _before_ loading the dictionary.

  Invoke LZ4_compress_HC_continue() to compress each successive block.
  The number of blocks is unlimited.
  Previous input blocks, including initial dictionary when present,
  must remain accessible and unmodified during compression.

  It's allowed to update compression level anytime between blocks,
  using LZ4_setCompressionLevel() (experimental).

  'dst' buffer should be sized to handle worst case scenarios
  (see LZ4_compressBound(), it ensures compression success).
  In case of failure, the API does not guarantee recovery,
  so the state _must_ be reset.
  To ensure compression success
  whenever `dst` buffer size cannot be made >= LZ4_compressBound(),
  consider using LZ4_compress_HC_continue_destSize().

  Whenever previous input blocks can't be preserved unmodified in-place during compression of next blocks,
  it's possible to copy the last blocks into a more stable memory space, using LZ4_saveDictHC().
  Return value of LZ4_saveDictHC() is the size of dictionary effectively saved into 'safeBuffer' (<= 64 KB)

  After completing a streaming compression,
  it's possible to start a new stream of blocks, using the same LZ4_streamHC_t state,
  just by resetting it, using LZ4_resetStreamHC_fast().
*/

 /*!
 * @brief It only works on states which have been properly initialized at least once,
 *  which is automatically the case when state is created using LZ4_createStreamHC(). 
 * 
 * Selecting the compression level can be done using this function.
 * 
 *  | Parameters | Direction | Description |
 *  |:-----------|:---------:|:------------|
 *  | \b streamHCPtr      | in,out | Streaming compression state that is automatically initialized. The same state can be re-used multiple times. |
 *  | \b compressionLevel | in     | It is used to set the correct context level for compression. |
 * 
 * @return \b void 
 * 
 * @note   After completing a streaming compression,
 *         it's possible to start a new stream of blocks, using the same LZ4_streamHC_t state,
 *         just by resetting it, using LZ4_resetStreamHC_fast().
 * @warning Requires v1.9.0+.
 * 
 */
LZ4LIB_API void LZ4_resetStreamHC_fast(LZ4_streamHC_t* streamHCPtr, int compressionLevel);   /* v1.9.0+ */

#ifdef AOCL_LZ4HC_OPT
 /*!
  * @brief AOCL variant of LZ4_resetStreamHC_fast(). 
  * It only works on states which have been properly initialized at least once,
  * which is automatically the case when state is created using AOCL_LZ4_createStreamHC().
  *
  * Selecting the compression level can be done using this function.
  *
  *  | Parameters | Direction | Description |
  *  |:-----------|:---------:|:------------|
  *  | \b streamHCPtr      | in,out | Streaming compression state that is automatically initialized. The same state can be re-used multiple times. |
  *  | \b compressionLevel | in     | It is used to set the correct context level for compression. |
  *
  * @return \b void
  *
  * @note   After completing a streaming compression,
  *         it's possible to start a new stream of blocks, using the same AOCL_LZ4_streamHC_t state,
  *         just by resetting it, using AOCL_LZ4_resetStreamHC_fast().
  * @warning Requires v1.9.0+.
  *
  */
LZ4LIB_API void AOCL_LZ4_resetStreamHC_fast(AOCL_LZ4_streamHC_t* streamHCPtr, int compressionLevel);   /* v1.9.0+ */
#endif /* AOCL_LZ4HC_OPT */

/*!
 * @brief A first "fictional block" can be designated as initial dictionary, using this function.
 * 
 *  It does full initialization of `streamHCPtr`, as there are bad side-effects of using resetFast().
 *  The same dictionary will have to be loaded on decompression side for successful decoding.
 *  Dictionary are useful for better compression of small data (KB range).
 * 
 *  |Parameters|Direction|Description|
 *  |:---------|:-------:|:----------|
 *  | \b streamHCPtr | in,out | Streaming compression state that is automatically initialized. The same state can be re-used multiple times. |
 *  | \b dictionary  | in,out | Dictionary buffer. |
 *  | \b dictSize    | in     | Size of dictionary. |
 * 
 * @return
 *  | Result    | Description  |
 *  |:----------|:-------------|
 *  | Success   | Loaded dictionary size, in bytes (necessarily <= 64 KB). |
 *  | Fail      | 0  if streamHCPtr is NULL or dictionary is NULL. |
 * 
*/
LZ4LIB_API int  LZ4_loadDictHC (LZ4_streamHC_t* streamHCPtr, const char* dictionary, int dictSize);

/*!
 * @brief Invoked to compress each successive block. The number of blocks is unlimited. \n
 *        Previous input blocks, including initial dictionary when present,must remain accessible 
 *        and unmodified during compression.
 * 
 *  |Parameters|Direction|Description|
 *  |:---------|:-------:|:----------|
 *  | \b streamHCPtr | in,out | Streaming compression state that is automatically initialized. The same state can be re-used multiple times.|
 *  | \b src         |  in    | Source buffer, the data which you want to compress is copied/or pointed here.|
 *  | \b dst         |  out   | Destination buffer, compressed data is kept here, memory should be allocated already.|
 *  | \b srcSize     |  in    | Maximum supported value is LZ4_MAX_INPUT_SIZE.|
 *  | \b maxDstSize  |  in    | Maximum size of buffer `dst` (which must be already allocated).|
 * 
 * @note 
 * -# It's allowed to update compression level anytime between blocks, using LZ4_setCompressionLevel()
 * 
 * -# `dst` buffer should be sized to handle worst case scenarios (see LZ4_compressBound(), it ensures compression success).
 * -# In case of failure, the API does not guarantee recovery, so the state must be reset.
 * -# To ensure compression success whenever `dst` buffer size cannot be made >= LZ4_compressBound(), consider using LZ4_compress_HC_continue_destSize().
 * 
 * -# There is an exception for ring buffers, which can be smaller than 64 KB.
 *  Ring-buffer scenario is automatically detected and handled within LZ4_compress_HC_continue().
 *  
 * @return
 *  | Result    | Description  |
 *  |:----------|:-------------|
 *  | Success   | Size of compressed block. |
 *  | Fail      | 0  (typically, when cannot fit into `dst`). |
*/
LZ4LIB_API int LZ4_compress_HC_continue (LZ4_streamHC_t* streamHCPtr,
                                         const char* src, char* dst,
                                         int srcSize, int maxDstSize);

/*! 
 *  @brief Similar to LZ4_compress_HC_continue(), but will read as much data as possible from `src` to fit into `targetDstSize` budget.
 * 
 *  |Parameters|Direction|Description|
 *  |:---------|:-------:|:----------|
 *  | \b LZ4_streamHCPtr | in,out | Streaming compression state that is automatically initialized. The same state can be re-used multiple times.|
 *  | \b src             |  in    | Source buffer, the data which you want to compress is copied/or pointed here.|
 *  | \b dst             |  out   | Destination buffer, compressed data is kept here, memory should be allocated already.|
 *  | \b srcSizePtr      | in,out | Pointer to size of buffer `src`. Maximum supported value is LZ4_MAX_INPUT_SIZE.|
 *  | \b targetDstSize   |  in    | Size of buffer `dst` (which must be already allocated).|
 *  
 *  @return 
 *  | Result | Description |
 *  |:-------|:------------|
 *  | success| The number of bytes written into `dst` (necessarily <= targetDstSize) |
 *  | ^      | `*srcSizePtr` is updated to indicate how much bytes were read from `src` |
 *  | Fail   |  0                                     |
 *  
 *  @note This function may not consume the entire input.
 *  @warning Requires v1.9.0+
 */
LZ4LIB_API int LZ4_compress_HC_continue_destSize(LZ4_streamHC_t* LZ4_streamHCPtr,
                                                 const char* src, char* dst,
                                                 int* srcSizePtr, int targetDstSize);

/*!
 * @brief  It saves history content into a user-provided buffer which is then used to continue compression.
 * 
 *  | Parameters | Direction | Description |
 *  |:-----------|:---------:|:------------|
 *  | \b streamHCPtr | in,out | Streaming compression state that is automatically initialized. The same state can be re-used multiple times.|
 *  | \b safeBuffer  |  in    | Buffer to store dictionary.|
 *  | \b maxDictSize |  in    | Size of safeBuffer, memory should be allocated such that dictionary would fit inside safeBuffer.This is schematically equivalent to a memcpy() followed by LZ4_loadDictHC(),but is much faster, because LZ4_saveDictHC() doesn't need to rebuild tables.|
 * 
 * @return
 * | Result  | Description |  
 * |:--------|:------------|
 * | Success | Saved dictionary size in bytes (necessarily <= maxDictSize) |
 * | Fail    | 0 |
 * 
 * @note Whenever previous input blocks can't be preserved unmodified in-place during compression of next blocks,
  it's possible to copy the last blocks into a more stable memory space, using LZ4_saveDictHC().
*/
LZ4LIB_API int LZ4_saveDictHC (LZ4_streamHC_t* streamHCPtr, char* safeBuffer, int maxDictSize);


/*! LZ4_attach_HC_dictionary() : 
 *  @brief This API allows for the efficient re-use of a static dictionary many times.
 *         Stable since v1.10.0
 *
 *  | Parameters | Direction | Description |
 *  |:-----------|:---------:|:------------|
 *  | \b working_stream    | in,out | A pointer to the LZ4_streamHC_t structure representing the working compression stream. |
 *  | \b dictionary_stream |   in   | A pointer to the LZ4_streamHC_t structure representing the dictionary stream. | 
 * 
 * @return \b NULL
 * 
 * @note
 * -# Rather than re-loading the dictionary buffer into a working context before
 *    each compression, or copying a pre-loaded dictionary's LZ4_streamHC_t into a
 *    working LZ4_streamHC_t, this function introduces a no-copy setup mechanism,
 *    in which the working stream references the dictionary stream in-place.
 *
 * -# Several assumptions are made about the state of the dictionary stream.
 *    Currently, only streams which have been prepared by LZ4_loadDictHC() should
 *    be expected to work.
 *
 * -# Alternatively, the provided dictionary stream pointer may be NULL, in which
 *    case any existing dictionary stream is unset.
 *
 * -# A dictionary should only be attached to a stream without any history (i.e.,
 *    a stream that has just been reset).
 *
 * -# The dictionary will remain attached to the working stream only for the
 *    current stream session. Calls to LZ4_resetStreamHC(_fast) will remove the
 *    dictionary context association from the working stream. The dictionary
 *    stream (and source buffer) must remain in-place / accessible / unchanged
 *    through the lifetime of the stream session.
 */
LZ4LIB_API void
LZ4_attach_HC_dictionary(LZ4_streamHC_t* working_stream,
                   const LZ4_streamHC_t* dictionary_stream);


/*^**********************************************
 * !!!!!!   STATIC LINKING ONLY   !!!!!!
 ***********************************************/

/*-******************************************************************
 * PRIVATE DEFINITIONS :
 * Do not use these definitions directly.
 * They are merely exposed to allow static allocation of `LZ4_streamHC_t`.
 * Declare an `LZ4_streamHC_t` directly, rather than any type below.
 * Even then, only do so in the context of static linking, as definitions may change between versions.
 ********************************************************************/

/// @cond DOXYGEN_SHOULD_SKIP_THIS
#define LZ4HC_DICTIONARY_LOGSIZE 16
#define LZ4HC_MAXD (1<<LZ4HC_DICTIONARY_LOGSIZE)
#define LZ4HC_MAXD_MASK (LZ4HC_MAXD - 1)

#define LZ4HC_HASH_LOG 15
#define LZ4HC_HASHTABLESIZE (1 << LZ4HC_HASH_LOG)
#define LZ4HC_HASH_MASK (LZ4HC_HASHTABLESIZE - 1)

/* Never ever use these definitions directly !
 * Declare or allocate an LZ4_streamHC_t instead.
**/
typedef struct LZ4HC_CCtx_internal LZ4HC_CCtx_internal;
struct LZ4HC_CCtx_internal
{
    LZ4_u32 hashTable[LZ4HC_HASHTABLESIZE];
    LZ4_u16 chainTable[LZ4HC_MAXD];
    const LZ4_byte* end;         /**< Next block here to continue on current prefix */
    const LZ4_byte* prefixStart; /**< Indexes relative to this position */
    const LZ4_byte* dictStart;   /**< Alternate reference for extDict */
    LZ4_u32 dictLimit;           /**< Below that point, need extDict */
    LZ4_u32 lowLimit;            /**< Below that point, no more history */
    LZ4_u32 nextToUpdate;        /**< Index from which to continue dictionary update */
    short   compressionLevel;    /**< Is a measure of the compression quality */
    LZ4_i8  favorDecSpeed;       /**< Favor decompression speed if this flag set,
                                      otherwise, favor compression ratio */
    LZ4_i8  dirty;               /**< Stream has to be fully reset if this flag is set */
    const LZ4HC_CCtx_internal* dictCtx; /**< Current context of dictionary */
};

/* Macros to be used in cache efficient hashchain implementation */
#define CHAIN_TYPE LZ4_u32

#ifdef AOCL_LZ4HC_OPT
#define HASH_CHAIN_ALLOC 128 /* 1 for Head and rest as Hash chain max */
#define CF_HC_CHAIN_TABLE_SZ (LZ4HC_HASHTABLESIZE * HASH_CHAIN_ALLOC)
#define CF_HC_HASH_TABLE_SZ 0
#define AOCL_LZ4HC_HASHTABLESIZE CF_HC_HASH_TABLE_SZ  /* fixing hashtable size */
#define AOCL_LZ4HC_MAXD CF_HC_CHAIN_TABLE_SZ  /* fixinng chainTable size */
#endif  /* AOCL_LZ4HC_OPT*/
/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */


#ifdef AOCL_LZ4HC_OPT

/// @cond DOXYGEN_SHOULD_SKIP_THIS
typedef struct AOCL_LZ4HC_CCtx_internal AOCL_LZ4HC_CCtx_internal;
/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

struct AOCL_LZ4HC_CCtx_internal
{ 
    /* DONOT REQUIRE HASHTABLE */  
    CHAIN_TYPE   chainTable[AOCL_LZ4HC_MAXD];
    const LZ4_byte* end;       /**< Next block here to continue on current prefix */
    const LZ4_byte* prefixStart;    /**< All index relative to this position */
    const LZ4_byte* dictStart;  /**< Alternate base for extDict */
    LZ4_u32   dictLimit;       /**< Below that point, need extDict */
    LZ4_u32   lowLimit;        /**< Below that point, no more dict */
    LZ4_u32   nextToUpdate;    /**< Index from which to continue dictionary update */
    short     compressionLevel; /**< Is a measure of the compression quality */
    LZ4_i8    favorDecSpeed;   /**< Favor decompression speed if this flag set,
                                  otherwise, favor compression ratio */
    LZ4_i8    dirty;           /**< Stream has to be fully reset if this flag is set */
    const AOCL_LZ4HC_CCtx_internal* dictCtx; /**< Current context of dictionary */
};
#endif /* AOCL_LZ4HC_OPT */

/* Do not use these definitions directly !
 * Declare or allocate an LZ4_streamHC_t instead.
 */
/// @cond DOXYGEN_SHOULD_SKIP_THIS

#define LZ4_STREAMHC_MINSIZE       262200  /* static size, for inter-version compatibility */
#ifdef AOCL_LZ4HC_OPT
#define AOCL_LZ4_STREAMHC_MINSIZE       (262200 - (LZ4HC_MAXD*2) + (CF_HC_CHAIN_TABLE_SZ*sizeof(CHAIN_TYPE)) + (CF_HC_HASH_TABLE_SZ*sizeof(LZ4_u32)) ) /* static size, for inter-version compatibility */
#endif

union LZ4_streamHC_u {
    char minStateSize[LZ4_STREAMHC_MINSIZE];
    LZ4HC_CCtx_internal internal_donotuse;
}; /* previously typedef'd to LZ4_streamHC_t */

#ifdef AOCL_LZ4HC_OPT
union AOCL_LZ4_streamHC_u {
    char minStateSize[AOCL_LZ4_STREAMHC_MINSIZE];
    AOCL_LZ4HC_CCtx_internal internal_donotuse;
}; /* previously typedef'd to AOCL_LZ4_streamHC_t */
#endif

/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

/* LZ4_streamHC_t :
 * This structure allows static allocation of LZ4 HC streaming state.
 * This can be used to allocate statically on stack, or as part of a larger structure.
 *
 * Such state **must** be initialized using LZ4_initStreamHC() before first use.
 *
 * Note that invoking LZ4_initStreamHC() is not required when
 * the state was created using LZ4_createStreamHC() (which is recommended).
 * Using the normal builder, a newly created state is automatically initialized.
 *
 * Static allocation shall only be used in combination with static linking.
 */

/* LZ4_initStreamHC() : v1.9.0+
 * Required before first use of a statically allocated LZ4_streamHC_t.
 * Before v1.9.0 : use LZ4_resetStreamHC() instead
 */

/*!
 * @brief Used to properly initialize a newly declared LZ4_streamHC_t.
 *  It can also initialize any arbitrary buffer of sufficient size,
 *  and will return a pointer of proper type upon initialization.
 * 
 * |Parameters|Direction|Description|
 * |:---------|:-------:|:----------|
 * | \b buffer  | in,out| buffer to store the LZ4 HC streaming state. |
 * | \b size    | in    |size of the buffer. |
 * 
 * 
 *  @return
 *  | Result   | Description  |
 *  |:---------|:-------------|
 *  | Success  | A pointer to LZ4HC streaming state. |
 *  | Fail     | \b NULL if `buffer == NULL` or `size < sizeof(LZ4_streamHC_t)` or if alignment conditions are not respected. |
 * 
 *  @note Invoking LZ4_initStreamHC() is not required when
 *  the state was created using LZ4_createStreamHC() (which is recommended).
 *  Using the normal builder, a newly created state is automatically initialized.
 * 
 * @note Static allocation shall only be used in combination with static linking.
 * 
 * @warning Requires v1.9.0+. 
 */
LZ4LIB_API LZ4_streamHC_t* LZ4_initStreamHC(void* buffer, size_t size);

#ifdef AOCL_LZ4HC_OPT
/*!
 * @brief Used to properly initialize a newly declared AOCL_LZ4_streamHC_t.
 *  It can also initialize any arbitrary buffer of sufficient size,
 *  and will return a pointer of proper type upon initialization.
 *
 * |Parameters|Direction|Description|
 * |:---------|:-------:|:----------|
 * | \b buffer  | in,out| buffer to store the LZ4 HC streaming state. |
 * | \b size    | in    |size of the buffer. |
 *
 *
 *  @return
 *  | Result   | Description  |
 *  |:---------|:-------------|
 *  | Success  | A pointer to LZ4HC streaming state. |
 *  | Fail     | \b NULL if `buffer == NULL` or `size < sizeof(AOCL_LZ4_streamHC_t)` or if alignment conditions are not respected. |
 *
 *  @note Invoking AOCL_LZ4_initStreamHC() is not required when
 *  the state was created using AOCL_LZ4_createStreamHC() (which is recommended).
 *  Using the normal builder, a newly created state is automatically initialized.
 *
 * @note Static allocation shall only be used in combination with static linking.
 *
 * @warning Requires v1.9.0+.
 */
LZ4LIB_API AOCL_LZ4_streamHC_t* AOCL_LZ4_initStreamHC(void* buffer, size_t size);
#endif /* AOCL_LZ4HC_OPT */

/// @cond DOXYGEN_SHOULD_SKIP_THIS
/*===   Enums   ===*/
typedef enum { noDictCtx, usingDictCtxHc } dictCtx_directive;
typedef enum { favorCompressionRatio = 0, favorDecompressionSpeed } HCfavor_e;
/*===   Struct   ===*/
typedef struct {
    int off;
    int len;
    int back;  /* negative value */
} LZ4HC_match_t;
/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

#ifdef AOCL_UNIT_TEST
#ifdef AOCL_LZ4HC_OPT
/* Test wrapper function AOCL_LZ4HC_init_internal for unit testing */
LZ4LIB_API void Test_AOCL_LZ4HC_init_internal(AOCL_LZ4HC_CCtx_internal* hc4, const LZ4_byte* start);

/* Test wrapper function of AOCL_LZ4HC_Insert for unit testing */
LZ4LIB_API void Test_AOCL_LZ4HC_Insert(AOCL_LZ4HC_CCtx_internal* hc4, const LZ4_byte* ip, const int Hash_Chain_Max, const int Hash_Chain_Slot_Sz);

/* Test wrapper function of AOCL_LZ4HC_InsertAndGetWiderMatch for unit testing */
LZ4LIB_API LZ4HC_match_t Test_AOCL_LZ4HC_InsertAndGetWiderMatch(
    AOCL_LZ4HC_CCtx_internal* const hc4,
    const LZ4_byte* const ip,
    const LZ4_byte* const iLowLimit, const LZ4_byte* const iHighLimit,
    int longest,
    const int maxNbAttempts,
    const int patternAnalysis, const int chainSwap,
    const dictCtx_directive dict,
    const HCfavor_e favorDecSpeed,
    int Hash_Chain_Max,
    int Hash_Chain_Slot_Sz);
#endif

/* Test wrapper function of LZ4_compress_HC_extStateHC_fastReset for unit testing */
LZ4LIB_API int Test_LZ4_compress_HC_extStateHC_fastReset(
    void* state,
    const char* src, char* dst,
    int srcSize, int dstCapacity,
    int compressionLevel);
#endif /* AOCL_UNIT_TEST */

/**
 * @}
*/

#ifndef AOCL_EXCLUDE_DEPRECATED_APIS
/*-************************************
*  Deprecated Functions
**************************************/
/* see lz4.h LZ4_DISABLE_DEPRECATE_WARNINGS to turn off deprecation warnings */

/*! 
 @name Deprecated Compression Functions
 @{
*/

/* deprecated compression functions */
/*! @brief LZ4_compressHC() is deprecated, use LZ4_compress_HC() instead */
LZ4_DEPRECATED("use LZ4_compress_HC() instead") LZ4LIB_API int LZ4_compressHC               (const char* source, char* dest, int inputSize);

/*! @brief LZ4_compressHC_limitedOutput() is deprecated, use LZ4_compress_HC() instead */
LZ4_DEPRECATED("use LZ4_compress_HC() instead") LZ4LIB_API int LZ4_compressHC_limitedOutput (const char* source, char* dest, int inputSize, int maxOutputSize);

/*! @brief LZ4_compressHC2() is deprecated, use LZ4_compress_HC() instead */
LZ4_DEPRECATED("use LZ4_compress_HC() instead") LZ4LIB_API int LZ4_compressHC2              (const char* source, char* dest, int inputSize, int compressionLevel);

/*! @brief LZ4_compressHC2_limitedOutput() is deprecated, use LZ4_compress_HC() instead */
LZ4_DEPRECATED("use LZ4_compress_HC() instead") LZ4LIB_API int LZ4_compressHC2_limitedOutput(const char* source, char* dest, int inputSize, int maxOutputSize, int compressionLevel);

/*! @brief LZ4_compressHC_withStateHC() is deprecated, use LZ4_compress_HC_extStateHC() instead */
LZ4_DEPRECATED("use LZ4_compress_HC_extStateHC() instead") LZ4LIB_API int LZ4_compressHC_withStateHC               (void* state, const char* source, char* dest, int inputSize);

/*! @brief LZ4_compressHC_limitedOutput_withStateHC() is deprecated, use LZ4_compress_HC_extStateHC() instead */
LZ4_DEPRECATED("use LZ4_compress_HC_extStateHC() instead") LZ4LIB_API int LZ4_compressHC_limitedOutput_withStateHC (void* state, const char* source, char* dest, int inputSize, int maxOutputSize);

/*! @brief LZ4_compressHC2_withStateHC() is deprecated, use LZ4_compress_HC_extStateHC() instead */
LZ4_DEPRECATED("use LZ4_compress_HC_extStateHC() instead") LZ4LIB_API int LZ4_compressHC2_withStateHC              (void* state, const char* source, char* dest, int inputSize, int compressionLevel);

/*! @brief LZ4_compressHC2_limitedOutput_withStateHC() is deprecated, use LZ4_compress_HC_extStateHC() instead */
LZ4_DEPRECATED("use LZ4_compress_HC_extStateHC() instead") LZ4LIB_API int LZ4_compressHC2_limitedOutput_withStateHC(void* state, const char* source, char* dest, int inputSize, int maxOutputSize, int compressionLevel);

/*! @brief LZ4_compressHC_continue() is deprecated, use LZ4_compress_HC_continue() instead */
LZ4_DEPRECATED("use LZ4_compress_HC_continue() instead") LZ4LIB_API int LZ4_compressHC_continue               (LZ4_streamHC_t* LZ4_streamHCPtr, const char* source, char* dest, int inputSize);

/*! @brief LZ4_compressHC_limitedOutput_continue() is deprecated, use LZ4_compress_HC_continue() instead */
LZ4_DEPRECATED("use LZ4_compress_HC_continue() instead") LZ4LIB_API int LZ4_compressHC_limitedOutput_continue (LZ4_streamHC_t* LZ4_streamHCPtr, const char* source, char* dest, int inputSize, int maxOutputSize);

/**
 * @}
*/

/* Obsolete streaming functions; degraded functionality; do not use!
 *
 * In order to perform streaming compression, these functions depended on data
 * that is no longer tracked in the state. They have been preserved as well as
 * possible: using them will still produce a correct output. However, use of
 * LZ4_slideInputBufferHC() will truncate the history of the stream, rather
 * than preserve a window-sized chunk of history.
 */

/// @cond DOXYGEN_SHOULD_SKIP_THIS
/**
 * @name Obsolete Streaming Functions
 * @{
*/
#if !defined(LZ4_STATIC_LINKING_ONLY_DISABLE_MEMORY_ALLOCATION)
/*! @brief Use LZ4_createStreamHC() instead. */
LZ4_DEPRECATED("use LZ4_createStreamHC() instead") LZ4LIB_API void* LZ4_createHC (const char* inputBuffer);

/*! @brief Use LZ4_freeStreamHC() instead. */
LZ4_DEPRECATED("use LZ4_freeStreamHC() instead") LZ4LIB_API   int   LZ4_freeHC (void* LZ4HC_Data);
#endif

/*! @brief Use LZ4_saveDictHC() instead. */
LZ4_DEPRECATED("use LZ4_saveDictHC() instead") LZ4LIB_API     char* LZ4_slideInputBufferHC (void* LZ4HC_Data);

/*! @brief Use LZ4_compress_HC_continue() instead. */
LZ4_DEPRECATED("use LZ4_compress_HC_continue() instead") LZ4LIB_API int LZ4_compressHC2_continue               (void* LZ4HC_Data, const char* source, char* dest, int inputSize, int compressionLevel);

/*! @brief Use LZ4_compress_HC_continue() instead. */
LZ4_DEPRECATED("use LZ4_compress_HC_continue() instead") LZ4LIB_API int LZ4_compressHC2_limitedOutput_continue (void* LZ4HC_Data, const char* source, char* dest, int inputSize, int maxOutputSize, int compressionLevel);

/*! @brief Use LZ4_createStreamHC() instead. */
LZ4_DEPRECATED("use LZ4_createStreamHC() instead") LZ4LIB_API int   LZ4_sizeofStreamStateHC(void);

/*! @brief Use LZ4_initStreamHC() instead. */
LZ4_DEPRECATED("use LZ4_initStreamHC() instead") LZ4LIB_API  int   LZ4_resetStreamStateHC(void* state, char* inputBuffer);

#endif /* AOCL_EXCLUDE_DEPRECATED_APIS */

/*! 
 * @brief It is now replaced by LZ4_initStreamHC().
 *
 * @note 
 * -# The intention is to emphasize the difference with LZ4_resetStreamHC_fast(),
 *    which is now the recommended function to start a new stream of blocks,
 *    but cannot be used to initialize a memory segment containing arbitrary garbage data.
 *
 * -# It is recommended to switch to LZ4_initStreamHC().
 * 
 * -# LZ4_resetStreamHC() will generate deprecation warnings in a future version.
 */
LZ4LIB_API void LZ4_resetStreamHC (LZ4_streamHC_t* streamHCPtr, int compressionLevel);

/**
 * @}
*/
/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

#if defined (__cplusplus)
}
#endif

#endif /* LZ4_HC_H_19834876238432 */


/*-**************************************************
 * !!!!!     STATIC LINKING ONLY     !!!!!
 * Following definitions are considered experimental.
 * They should not be linked from DLL,
 * as there is no guarantee of API stability yet.
 * Prototypes will be promoted to "stable" status
 * after successful usage in real-life scenarios.
 ***************************************************/
#ifdef LZ4_HC_STATIC_LINKING_ONLY   /* protection macro */
#ifndef LZ4_HC_SLO_098092834
#define LZ4_HC_SLO_098092834

#define LZ4_STATIC_LINKING_ONLY   /* LZ4LIB_STATIC_API */
#include "lz4.h"

#if defined (__cplusplus)
extern "C" {
#endif

/* LZ4_setCompressionLevel() : v1.8.0+ (experimental)
 *  It's possible to change compression level
 *  between successive invocations of LZ4_compress_HC_continue*()
 *  for dynamic adaptation.
 */
LZ4LIB_STATIC_API void LZ4_setCompressionLevel(
    LZ4_streamHC_t* LZ4_streamHCPtr, int compressionLevel);

#ifdef AOCL_LZ4HC_OPT
/* This is AOCL variant of LZ4_setCompressionLevel() used to set compression level in the streamPtr of type AOCL_LZ4_StreamHC_t. */
LZ4LIB_STATIC_API void AOCL_LZ4_setCompressionLevel(
    AOCL_LZ4_streamHC_t* AOCL_LZ4_streamHCPtr, int compressionLevel);
#endif /* AOCL_LZ4HC_OPT */

/* LZ4_favorDecompressionSpeed() : v1.8.2+ (experimental)
 *  Opt. Parser will favor decompression speed over compression ratio.
 *  Only applicable to levels >= LZ4HC_CLEVEL_OPT_MIN.
 */
LZ4LIB_STATIC_API void LZ4_favorDecompressionSpeed(
    LZ4_streamHC_t* LZ4_streamHCPtr, int favor);

/* LZ4_resetStreamHC_fast() : v1.9.0+
 *  When an LZ4_streamHC_t is known to be in a internally coherent state,
 *  it can often be prepared for a new compression with almost no work, only
 *  sometimes falling back to the full, expensive reset that is always required
 *  when the stream is in an indeterminate state (i.e., the reset performed by
 *  LZ4_resetStreamHC()).
 *
 *  LZ4_streamHCs are guaranteed to be in a valid state when:
 *  - returned from LZ4_createStreamHC()
 *  - reset by LZ4_resetStreamHC()
 *  - memset(stream, 0, sizeof(LZ4_streamHC_t))
 *  - the stream was in a valid state and was reset by LZ4_resetStreamHC_fast()
 *  - the stream was in a valid state and was then used in any compression call
 *    that returned success
 *  - the stream was in an indeterminate state and was used in a compression
 *    call that fully reset the state (LZ4_compress_HC_extStateHC()) and that
 *    returned success
 *
 *  Note:
 *  A stream that was last used in a compression call that returned an error
 *  may be passed to this function. However, it will be fully reset, which will
 *  clear any existing history and settings from the context.
 */
LZ4LIB_STATIC_API void LZ4_resetStreamHC_fast(
    LZ4_streamHC_t* LZ4_streamHCPtr, int compressionLevel);

/* LZ4_compress_HC_extStateHC_fastReset() :
 *  A variant of LZ4_compress_HC_extStateHC().
 *
 *  Using this variant avoids an expensive initialization step. It is only safe
 *  to call if the state buffer is known to be correctly initialized already
 *  (see above comment on LZ4_resetStreamHC_fast() for a definition of
 *  "correctly initialized"). From a high level, the difference is that this
 *  function initializes the provided state with a call to
 *  LZ4_resetStreamHC_fast() while LZ4_compress_HC_extStateHC() starts with a
 *  call to LZ4_resetStreamHC().
 * 
 * @note `state` When environment variable AOCL_DISABLE_OPT is ON, `state` size must be provided by LZ4_sizeofStateHC(). 
 *               When AOCL_DISABLE_OPT is OFF, `state` size must be provided by AOCL_LZ4_sizeofStateHC() for compression level 6, 7, 8, 9 and
 *               LZ4_sizeofStateHC() for other levels respectively.
 *               Also, state must be intialised by LZ4_resetStreamHC_fast() in case when `state` size is provided by LZ4_sizeofStateHC() or
 *               it must be intialised by AOCL_LZ4_resetStreamHC_fast() in case when `state` size is provided by AOCL_LZ4_sizeofStateHC().                   
 */
LZ4LIB_STATIC_API int LZ4_compress_HC_extStateHC_fastReset (
    void* state,
    const char* src, char* dst,
    int srcSize, int dstCapacity,
    int compressionLevel);

#if defined (__cplusplus)
}
#endif

#endif   /* LZ4_HC_SLO_098092834 */
#endif   /* LZ4_HC_STATIC_LINKING_ONLY */

/**
 * @}
*/

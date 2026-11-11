/* LzmaDec.h -- LZMA Decoder
2020-03-19 : Igor Pavlov : Public domain */

/*
* Modifications Copyright (C) 2022-2025, Advanced Micro Devices. All rights reserved.
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

#ifndef __LZMA_DEC_H
#define __LZMA_DEC_H

#include "7zTypes.h"

EXTERN_C_BEGIN

/*!
 * \addtogroup LZMA_API
 * @brief
 * LZMA is a lossless compression algorithm that provides a high degree of compression.
 * Its compression ratios are lower than other LZ77 based methods for most inputs 
 * (in the range of 25-30 for Silesia dataset).
 * The lower compression ratio comes at the expense of lower compression speed. 
 * However, it provides good decompression speed (better than BZIP2, which can give
 * compression ratios close to LZMA).
 * 
 * The LZMA compression library provides in-memory compression and decompression functions.
 * Typical usage is as follows :
 * 1. Call LzmaEncProps_Init() to initialize CLzmaEncProps object.
 * 2. Update _CLzmaEncProps, if any specific user settings are desired, such as compression level.
 * 3. To compress a file, load file to a source buffer and pass this and a destination buffer to LzmaEncode().
 * LzmaEncode() performs in-memory compression and writes the compressed data to the destination buffer.
 * 4. To decompress, call LzmaDecode() by passing compressed data as source and a destination buffer
 * to hold uncompressed bytes.
 *
 * @{
*/

/// @cond DOXYGEN_SHOULD_SKIP_THIS
/* #define _LZMA_PROB32 */
/** _LZMA_PROB32 can increase the speed on some CPUs,
   but memory usage for CLzmaDec::probs will be doubled in that case. */
    typedef
#ifdef _LZMA_PROB32
    UInt32
#else
    UInt16
#endif
    CLzmaProb;
/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */


/* ---------- LZMA Properties ---------- */

/// @cond DOXYGEN_SHOULD_SKIP_THIS
#define LZMA_PROPS_SIZE 5
/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

/**
 * @brief Structure to hold header parameters that are stored in LZMA compressed streams.
 * 
 */
typedef struct _CLzmaProps
{
    Byte lc; /**< number of high bits of the previous byte to use as a context for literal encoding (default 3). */
    Byte lp; /**< number of low bits of the dictionary position to include in literal posState (default 0). */
    Byte pb; /**< number of low bits of processedPos to include in posState (default 2). */
    Byte _pad_;
    UInt32 dicSize; /**< size of dictionary / search buffer to use to find matches */
} CLzmaProps;


/*!
 * @name Decode Functions
 * @{
 */

/* LzmaProps_Decode - decodes properties
Returns:
  SZ_OK
  SZ_ERROR_UNSUPPORTED - Unsupported properties
*/

/*! @brief decodes header bytes in `data` and sets properties in `p`.
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b p       | out         | Properties object to be set |
* | \b data    | in          | Header bytes from compressed stream |
* | \b size    | in          | Size of header in bytes |
*
* @return 
* | Result     | Description |
* |:-----------|:------------|
* | Success    |SZ_OK                                         |
* | Fail       |SZ_ERROR_UNSUPPORTED - Unsupported properties |
* | Fail       |SZ_ERROR_PARAM - Invalid input                |
* 
*/
LZMALIB_API SRes LzmaProps_Decode(CLzmaProps* p, const Byte* data, unsigned size);

/**
 * @}
*/

/* ---------- LZMA Decoder state ---------- */

/// @cond DOXYGEN_SHOULD_SKIP_THIS
#define LZMA_REQUIRED_INPUT_MAX 20  /**< LZMA_REQUIRED_INPUT_MAX = number of required input bytes for worst case. \n
                                         Num bits = log2((2^11 / 31) ^ 22) + 26 < 134 + 26 = 160; */
/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

typedef struct
{
    /* Don't change this structure. ASM code can use it. */
    CLzmaProps prop;       /**< properties read from header bytes in compressed data */
    CLzmaProb* probs;      /**< all context model probabilities */
    CLzmaProb* probs_1664; /**< all context model probabilities */
    Byte* dic;             /**< circular buffer of decompressed bytes. Used as reference to copy from for future matches */
    SizeT dicBufSize;      /**< dictionary size */
    SizeT dicPos;          /**< current position in dictionary */
    const Byte* buf;       /**< input stream of compressed bytes */
    UInt32 range;          /**< range coder: range size */
    UInt32 code;           /**< range coder: encoded point within range */
    UInt32 processedPos;   /**< indicator of bytes decompressed until now. Incremented by 1 or incremented by len based on type of decompress operation performed */
    UInt32 checkDicSize;   /**< indicator for situation where bytes to be processed is more than bytes that can fit in dest buffer */
    UInt32 reps[4];        /**< offsets for repeated matches rep0-3 */
    UInt32 state;          /**< current state in state machine */
    UInt32 remainLen;      /**< shows status of LZMA decoder: \n 
                                < kMatchSpecLenStart  : the number of bytes to be copied with (rep0) offset \n 
                                = kMatchSpecLenStart  : the LZMA stream was finished with end mark \n 
                                = kMatchSpecLenStart + 1  : need init range coder \n 
                                = kMatchSpecLenStart + 2  : need init range coder and state \n 
                                = kMatchSpecLen_Error_Fail                : Internal Code Failure \n 
                                = kMatchSpecLen_Error_Data + [0 ... 273]  : LZMA Data Error */

    UInt32 numProbs;       /**< number of items in probs table */
    unsigned tempBufSize;  /**< size of temporary buffer used */
    Byte tempBuf[LZMA_REQUIRED_INPUT_MAX]; /**< temporary workspace buffer */
} CLzmaDec;

/*! @brief First operation to call before setting up CLzmaDec
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b p       | out         | Decoder object to be initialized |
*/
#define LzmaDec_Construct(p) \
{ (p)->dic = NULL; (p)->probs = NULL; }

/*!
 * @name Decode Functions
 * @{
 */

/*! @brief Initialize LZMA decoder
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b p       | out         | Decoder object to be initialized |
*
* @return void
* 
*/
LZMALIB_API void LzmaDec_Init(CLzmaDec* p);

/**
 * @}
*/

/*! @brief 
   There are two types of LZMA streams:
     - Stream with end mark. That end mark adds about 6 bytes to compressed size.
     - Stream without end mark. You must know exact uncompressed size to decompress such stream.

   ELzmaFinishMode has meaning only if the decoding reaches output limit.

   @note 1. You must use `LZMA_FINISH_END`, when you know that current output buffer
   covers last bytes of block. In other cases you must use `LZMA_FINISH_ANY`. \n

   @note 2.  If LZMA decoder sees end marker before reaching output limit, it returns `SZ_OK`,
   and output value of destLen will be less than output buffer size limit. 
   You can check status result also. \n

   @note 3.  You can use multiple checks to test data integrity after full decompression: \n
      -  Check Result and "status" variable. \n
      -  Check that output(destLen) = uncompressedSize, if you know real uncompressedSize. \n
      -  Check that output(srcLen) = compressedSize, if you know real compressedSize. \n
        You must use correct finish mode in that case. */
typedef enum
{
    LZMA_FINISH_ANY,   /**< finish at any point */
    LZMA_FINISH_END    /**< block must be finished at the end */
} ELzmaFinishMode;

/*! @brief ELzmaStatus is used as output status of decode function call. */
typedef enum
{
    LZMA_STATUS_NOT_SPECIFIED,               /**< use main error code instead */
    LZMA_STATUS_FINISHED_WITH_MARK,          /**< stream was finished with end mark. */
    LZMA_STATUS_NOT_FINISHED,                /**< stream was not finished */
    LZMA_STATUS_NEEDS_MORE_INPUT,            /**< you must provide more input bytes */
    LZMA_STATUS_MAYBE_FINISHED_WITHOUT_MARK  /**< there is probability that stream was finished without end mark */
} ELzmaStatus;


/* ---------- Interfaces ---------- */

/* There are 3 levels of interfaces:
     1) Dictionary Interface
     2) Buffer Interface
     3) One Call Interface
   You can select any of these interfaces, but don't mix functions from different
   groups for same object. */


   /* There are two variants to allocate state for Dictionary Interface:
        1) LzmaDec_Allocate / LzmaDec_Free
        2) LzmaDec_AllocateProbs / LzmaDec_FreeProbs
      You can use variant 2, if you set dictionary buffer manually.
      For Buffer Interface you must always use variant 1.

   LzmaDec_Allocate* can return:
     SZ_OK
     SZ_ERROR_MEM         - Memory allocation error
     SZ_ERROR_UNSUPPORTED - Unsupported properties
   */

/*!
 * @name Decode Functions
 * @{
 */

/*! @brief Allocate probability tables in decoder object.
*          Sets properties in p by calling LzmaProps_Decode().
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b p         | out         | Decoder object that holds prob tables |
* | \b props     | in          | Header bytes from compressed stream |
* | \b propsSize | in          | Size of header in bytes |
* | \b alloc     | in          | Allocator object |
* 
* @return 
* | Result     | Description |
* |:-----------|:------------|
* | Success    |SZ_OK                                          |
* | Fail       |SZ_ERROR_MEM         - Memory allocation error |
* | ^          |SZ_ERROR_PARAM       - Incorrect parameter     |
* | ^          |SZ_ERROR_UNSUPPORTED - Unsupported properties  |
* 
*/
LZMALIB_API SRes LzmaDec_AllocateProbs(CLzmaDec* p, const Byte* props, unsigned propsSize, ISzAllocPtr alloc);

/*! @brief Free probability tables in decoder object.
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b p        | out         | Decoder object that holds prob tables |
* | \b alloc    | in          | Allocator object |
*
* @return void
* 
*/
LZMALIB_API void LzmaDec_FreeProbs(CLzmaDec* p, ISzAllocPtr alloc);

/*! @brief Allocate probability tables in decoder object.
*          Sets properties in p by calling LzmaProps_Decode().
*          Allocate dictionary buffer.
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b p         | out         | Decoder object that holds prob tables and dictionary |
* | \b props     | in          | Header bytes from compressed stream |
* | \b propsSize | in          | Size of header in bytes |
* | \b alloc     | in          | Allocator object |
*
* @return 
* | Result     | Description |
* |:-----------|:------------|
* | Success    |SZ_OK                                          |
* | Fail       |SZ_ERROR_MEM         - Memory allocation error |
* | ^          |SZ_ERROR_PARAM       - Incorrect parameter     |
* | ^          |SZ_ERROR_UNSUPPORTED - Unsupported properties  |
* 
*/
LZMALIB_API SRes LzmaDec_Allocate(CLzmaDec* p, const Byte* props, unsigned propsSize, ISzAllocPtr alloc);

/*! @brief Free probability tables and dictionary in decoder object.
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b p        | out         | Decoder object that holds prob tables and dictionary |
* | \b alloc    | in          | Allocator object |
*
* @return void
* 
*/
LZMALIB_API void LzmaDec_Free(CLzmaDec* p, ISzAllocPtr alloc);

/* ---------- Dictionary Interface ---------- */

/* You can use it, if you want to eliminate the overhead for data copying from
   dictionary to some other external buffer.
   You must work with CLzmaDec variables directly in this interface.

   STEPS:
     LzmaDec_Construct()
     LzmaDec_Allocate()
     for (each new stream)
     {
       LzmaDec_Init()
       while (it needs more decompression)
       {
         LzmaDec_DecodeToDic()
         use data from CLzmaDec::dic and update CLzmaDec::dicPos
       }
     }
     LzmaDec_Free()
*/

/* LzmaDec_DecodeToDic

   The decoding to internal dictionary buffer (CLzmaDec::dic).
   You must manually update CLzmaDec::dicPos, if it reaches CLzmaDec::dicBufSize !!!

finishMode:
  It has meaning only if the decoding reaches output limit (dicLimit).
  LZMA_FINISH_ANY - Decode just dicLimit bytes.
  LZMA_FINISH_END - Stream must be finished after dicLimit.

Returns:
  SZ_OK
    status:
      LZMA_STATUS_FINISHED_WITH_MARK
      LZMA_STATUS_NOT_FINISHED
      LZMA_STATUS_NEEDS_MORE_INPUT
      LZMA_STATUS_MAYBE_FINISHED_WITHOUT_MARK
  SZ_ERROR_DATA - Data error
  SZ_ERROR_FAIL - Some unexpected error: internal error of code, memory corruption or hardware failure
*/


/*! @brief Decode src and write decompressed data into internal dictionary buffer.
* You can use it, if you want to eliminate the overhead for data copying from
* dictionary to some other external buffer.
* You must work with CLzmaDec variables directly in this interface.
* 
* STEPS:
* ~~~~~~~~~~~~
     LzmaDec_Construct()
     LzmaDec_Allocate()
     for (each new stream)
     {
       LzmaDec_Init()
       while (it needs more decompression)
       {
         LzmaDec_DecodeToDic()
         use data from CLzmaDec::dic and update CLzmaDec::dicPos
       }
     }
     LzmaDec_Free()
* ~~~~~~~~~~~~
* 
*  When decoding to internal dictionary buffer (CLzmaDec::dic),
*  you must manually update CLzmaDec::dicPos, if it reaches CLzmaDec::dicBufSize !!!
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b p          | in,out      | Decoder object that contains properties and dictionary buffer |
* | \b dicLimit   | in          | Max number of bytes that can be decompressed and saved in dictionary |
* | \b src        | in          | Source buffer containing compressed data |
* | \b srcLen     | in,out      | Length of source buffer |
* | \b finishMode |             | It has meaning only if the decoding reaches output limit (dicLimit). |
* | ^             | ^           | - LZMA_FINISH_ANY - Decode just dicLimit bytes.                      |
* | ^             | ^           | - LZMA_FINISH_END - Stream must be finished after dicLimit.          |
* | \b status     | out         | Decompression status at the end of current operation |
*
* @return 
* | Result     | Description |
* |:-----------|:------------|
* | Success    |SZ_OK                      |
* | Fail       |SZ_ERROR_DATA - Data error |
* | ^          |SZ_ERROR_PARAM       - Incorrect parameter |
* | ^          |SZ_ERROR_FAIL - Some unexpected error: internal error of code, memory corruption or hardware failure  |
* 
*/

LZMALIB_API SRes LzmaDec_DecodeToDic(CLzmaDec* p, SizeT dicLimit,
    const Byte* src, SizeT* srcLen, ELzmaFinishMode finishMode, ELzmaStatus* status);




/* ---------- Buffer Interface ---------- */

/* It's zlib-like interface.
   See LzmaDec_DecodeToDic description for information about STEPS and return results,
   but you must use LzmaDec_DecodeToBuf instead of LzmaDec_DecodeToDic and you don't need
   to work with CLzmaDec variables manually.

finishMode:
  It has meaning only if the decoding reaches output limit (*destLen).
  LZMA_FINISH_ANY - Decode just destLen bytes.
  LZMA_FINISH_END - Stream must be finished after (*destLen).
*/

/// @cond DOXYGEN_SHOULD_SKIP_THIS
LZMALIB_API SRes LzmaDec_DecodeToBuf(CLzmaDec* p, Byte* dest, SizeT* destLen,
    const Byte* src, SizeT* srcLen, ELzmaFinishMode finishMode, ELzmaStatus* status);
/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

/**
 * @}
*/

/* ---------- One Call Interface ---------- */

/* LzmaDecode

finishMode:
  It has meaning only if the decoding reaches output limit (*destLen).
  LZMA_FINISH_ANY - Decode just destLen bytes.
  LZMA_FINISH_END - Stream must be finished after (*destLen).

Returns:
  SZ_OK
    status:
      LZMA_STATUS_FINISHED_WITH_MARK
      LZMA_STATUS_NOT_FINISHED
      LZMA_STATUS_MAYBE_FINISHED_WITHOUT_MARK
  SZ_ERROR_DATA - Data error
  SZ_ERROR_MEM  - Memory allocation error
  SZ_ERROR_UNSUPPORTED - Unsupported properties
  SZ_ERROR_INPUT_EOF - It needs more bytes in input buffer (src).
  SZ_ERROR_FAIL - Some unexpected error: internal error of code, memory corruption or hardware failure
*/

/*!
 * @name Decode One Call Interface
 * @{
 */

/*! @brief Decode compressed data in `src` and save result to `dest`
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b dest       | out         | Destination buffer to save decompressed data |
* | \b destLen    | out         | Size of decompressed data|
* | \b src        | in          | Source buffer containing compressed data |
* | \b srcLen     | in,out      | Length of source buffer |
* | \b propData   | in          | Header bytes in compressed source data |
* | \b propSize   | in          | Size of header |
* | \b finishMode |             | It has meaning only if the decoding reaches output limit (*destLen). |
* | ^             | ^           | - LZMA_FINISH_ANY - Decode just destLen bytes.                       |
* | ^             | ^           | - LZMA_FINISH_END - Stream must be finished after (*destLen).        |
* | \b status     | out         | Decompression status at the end of current operation |
* | \b alloc      | in          | Memory allocator object |
*
* @return 
* | Result     | Description |
* |:-----------|:------------|
* | Success    |SZ_OK                      |
* | Fail       |SZ_ERROR_DATA - Data error |
* | ^          |SZ_ERROR_MEM  - Memory allocation error  |
* | ^          |SZ_ERROR_PARAM       - Incorrect parameter |
* | ^          |SZ_ERROR_UNSUPPORTED - Unsupported properties |
* | ^          |SZ_ERROR_INPUT_EOF - It needs more bytes in input buffer (src) |
* | ^          |SZ_ERROR_FAIL - Some unexpected error: internal error of code, memory corruption or hardware failure  |
* 
*/
LZMALIB_API SRes LzmaDecode(Byte* dest, SizeT* destLen, const Byte* src, SizeT* srcLen,
    const Byte* propData, unsigned propSize, ELzmaFinishMode finishMode,
    ELzmaStatus* status, ISzAllocPtr alloc);
    
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
 * Except for the initial call, it's necessary to execute aocl_destroy_lzma_decode() before any subsequent calls
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
LZMALIB_API void aocl_setup_lzma_decode(int optOff, int optLevel, size_t insize,
    size_t level, size_t windowLog);

/**
 * @brief It is necessary to execute this destroy function after the initial invocation of the
 * aocl_setup_lzma_decode() function, prior to initiating the setup function again.
 */
LZMALIB_API void aocl_destroy_lzma_decode(void);

/**
 * @}
 */

/// @endcond DOXYGEN_SHOULD_SKIP_THIS

/**
 * @}
 */

EXTERN_C_END

#ifdef AOCL_UNIT_TEST
/* Move these APIs within the scope of gtest once the framework is ready */
EXTERN_C_BEGIN
LZMALIB_API void Test_Rc_Get_Bit_2_Dec_Ref(const Byte* buf, UInt32* range, UInt32* code, CLzmaProb* prob, unsigned symbol);
LZMALIB_API void Test_Rc_Get_Bit_2_Dec_Opt(const Byte* buf, UInt32* range, UInt32* code, CLzmaProb* prob, unsigned symbol);
LZMALIB_API void Test_Rc_Rev_Bit_Dec_Ref(const Byte* buf, UInt32* range, UInt32* code, CLzmaProb* prob, unsigned symbol);
LZMALIB_API void Test_Rc_Rev_Bit_Dec_Opt(const Byte* buf, UInt32* range, UInt32* code, CLzmaProb* prob, unsigned symbol);
EXTERN_C_END
#endif /* AOCL_UNIT_TEST */
#endif

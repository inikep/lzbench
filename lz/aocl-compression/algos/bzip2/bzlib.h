
/*-------------------------------------------------------------*/
/*--- Public header file for the library.                   ---*/
/*---                                               bzlib.h ---*/
/*-------------------------------------------------------------*/

/* ------------------------------------------------------------------
   This file is part of bzip2/libbzip2, a program and library for
   lossless, block-sorting data compression.

   bzip2/libbzip2 version 1.0.8 of 13 July 2019
   Copyright (C) 1996-2019 Julian Seward <jseward@acm.org>
   Modifications Copyright (C) 2024, Advanced Micro Devices. All rights reserved.

   Please read the WARNING, DISCLAIMER and PATENTS sections in the 
   README file.

   This program is released under the terms of the license contained
   in the file LICENSE.
   ------------------------------------------------------------------ */

/*!
 * \addtogroup BZIP2_API
 * @brief bzip2 compresses files using the Burrows-Wheeler block-sorting text compression algorithm, and Huffman coding.
 *  Compression is considerably better than that achieved by more conventional LZ77/LZ78-based compressors, 
 *  and approaches the performance of the PPM family of statistical compressors.
 *
 * @{
*/
#ifndef _BZLIB_H
#define _BZLIB_H

#ifdef __cplusplus
extern "C" {
#endif

/// @cond DOXYGEN_SHOULD_SKIP_THIS

#define BZ_RUN               0
#define BZ_FLUSH             1
#define BZ_FINISH            2

#define BZ_OK                0
#define BZ_RUN_OK            1
#define BZ_FLUSH_OK          2
#define BZ_FINISH_OK         3
#define BZ_STREAM_END        4
#define BZ_SEQUENCE_ERROR    (-1)
#define BZ_PARAM_ERROR       (-2)
#define BZ_MEM_ERROR         (-3)
#define BZ_DATA_ERROR        (-4)
#define BZ_DATA_ERROR_MAGIC  (-5)
#define BZ_IO_ERROR          (-6)
#define BZ_UNEXPECTED_EOF    (-7)
#define BZ_OUTBUFF_FULL      (-8)
#define BZ_CONFIG_ERROR      (-9)

/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

/*!
 * @note The fields of `bz_stream` comprise the entirety of the user-visible data. 
 */
typedef 
   struct {
      char *next_in;                /**< Points to the data to be compressed when called during compression, \n
                                         whereas when called during decompression, it points to compressed data.*/
      unsigned int avail_in;        /**< Indicates the number of bytes the library may read. */
      unsigned int total_in_lo32;   /**< Used to store lower 32 bits of the total number of input bytes processed.*/
      unsigned int total_in_hi32;   /**< Used to store upper 32 bits of the total number of input bytes processed.*/

      char *next_out;               /**< Points to a buffer in which the compressed data is to be placed when called during compression, \n 
                                         whereas when called during decompression, it points to a buffer in which the 
                                         uncompressed output is to be placed  */
      unsigned int avail_out;       /**< Indicates the available output space. */
      unsigned int total_out_lo32;  /**< Used to store lower 32 bits of the total number of output bytes processed.*/
      unsigned int total_out_hi32;  /**< Used to store upper 32 bits of the total number of output bytes processed.*/

      void *state;                  /**< A pointer, pointing to the private data structures. */

      void *(*bzalloc)(void *,int,int); /**< Custom memory allocator. The call `bzalloc ( opaque, n, m )` is expected to 
                                             return a pointer p to n * m bytes of memory */
      void (*bzfree)(void *,void *);    /**< `bzfree ( opaque, p )` is expected to free the memory allocated using `bzalloc`.*/
      void *opaque;                     /**< The value `opaque` is passed to as the first argument to all calls to `bzalloc` 
                                             and `bzfree`, but is otherwise ignored by the library.*/
   } 
   bz_stream;


/// @cond DOXYGEN_SHOULD_SKIP_THIS 
#ifndef BZ_IMPORT
#define BZ_EXPORT
#endif

#ifndef BZ_NO_STDIO
/* Need a definitition for FILE */
#include <stdio.h>
#endif

#ifdef _WIN32
#   include <windows.h>
#   ifdef small
      /* windows.h define small to char */
#      undef small
#   endif
#   ifdef BZ_EXPORT
#   define BZ_API(func) WINAPI func
#   ifdef BZIP2_DLL_EXPORT
#   define BZ_EXTERN extern __declspec(dllexport)
#   else
#   define BZ_EXTERN extern
#   endif /* BZIP2_DLL_EXPORT */
#   else
   /* import windows dll dynamically */
#   define BZ_API(func) (WINAPI * func)
#   define BZ_EXTERN
#   endif
#else
#   define BZ_API(func) func
#   define BZ_EXTERN extern
#endif

/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

/*-- Core (low-level) library functions --*/
/**
 * @name Core (low-level) library functions
 * @{
*/

/*!
* @brief Prepares for compression.
* 
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b strm             | in,out   | Holds all data pertaining to the compression activity. It should be allocated and initialised prior to the call.  |
* | \b blockSize100k    | in       | Specifies the block size to be used for compression. It should be a value between 1 and 9 inclusive, and the actual block size used is 100000 x this figure. |
* | \b verbosity        | in       | A number between 0 and 4 inclusive. 0 is silent, and greater numbers give increasingly verbose monitoring/debugging output. |
* | \b workFactor       | in       | Controls how the compression phase behaves when presented with worst case, highly repetitive, input data. |
*
* @return
* | Result     | Description |
* |:-----------|:------------|
* | Success    | `BZ_OK`       |
* | Fail       | `BZ_CONFIG_ERROR` - If the library has been mis-compiled |
* | ^          | `BZ_PARAM_ERROR` \n If strm is NULL \n or blockSize < 1 or blockSize > 9 \n or verbosity < 0 or verbosity > 4 \n or workFactor < 0 or workFactor > 250 |
* | ^          | `BZ_MEM_ERROR` - If available memory is insufficient. |
*
* @note 
* -# `blockSize100k` = 9 gives the best compression but takes most memory. 
* -# If the library has been compiled with  `-DBZ_NO_STDIO`, no monitoring/debugging output will appear for any verbosity setting.
* -# If compression runs into difficulties caused by repetitive data, the library switches from the <b> standard sorting algorithm </b> to a <b> fallback algorithm </b>.
* The fallback is slower than the standard algorithm by perhaps a factor of three, but always behaves reasonably, no matter how bad the input.
* 
* -# Lower values of `workFactor` reduce the amount of effort the standard algorithm will expend before resorting to the fallback. Set this parameter carefully; 
*  - too low, and many inputs will be handled by the fallback algorithm and so compress rather slowly, 
*  - too high, and your average-to-worst case compression times can become very large. 
*  - The default value of \b 30 gives reasonable behaviour over a wide range of circumstances. \n
* Allowable values range from \b 0 to \b 250 inclusive. 0 is a special case, equivalent to using the default value of 30.
* 
* -# Upon return, the internal state will have been allocated and initialised, and `total_in_lo32`, `total_in_hi32`, `total_out_lo32` and `total_out_hi32` will have been set to zero. 
* These four fields are used by the library to inform the caller of the total amount of data passed into and out of the library, respectively. You should not try to change them. \n 
* As of version 1.0, 64-bit counts are maintained, even on 32-bit platforms, using the `_hi32` fields to store the upper 32 bits of the count. So, for example, the total amount of data in is (total_in_hi32 << 32) + total_in_lo32.
*/
BZ_EXTERN int BZ_API(BZ2_bzCompressInit) ( 
      bz_stream* strm, 
      int        blockSize100k, 
      int        verbosity, 
      int        workFactor 
   );

/*!
* @brief Provides more input and/or output buffer space for the library. 
* The caller maintains input and output buffers, and calls BZ2_bzCompress to transfer data between them. \n
* BZ2_bzCompress updates `next_in`, `avail_in` and `total_in` to reflect the number of bytes it has read. It also 
* updates `next_out`, `avail_out` and `total_out` to reflect the number of bytes output. \n
* A second purpose of BZ2_bzCompress is to request a change of mode of the compressed stream.
*
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b strm      | in,out    | Holds all data pertaining to the compression activity. |
* | \b action    | in        | An integer value indicating the action to be taken.|
*
* @return
*
* Here's a table which shows which actions are allowable in each state, what action will be taken, what the next state is, 
* and what the non-error return values are. Note that you can't explicitly ask what state the stream is in, 
* but nor do you need to -- it can be inferred from the values returned by BZ2_bzCompress.
*
* | State/ Action | Action Taken | Next State | Return Value |
* |:--------------|:-------------|:-----------|:-------------|
* | IDLE/any         | Illegal. \n IDLE state only exists after BZ2_bzCompressEnd or before BZ2_bzCompressInit. | - | BZ_SEQUENCE_ERROR |
* | RUNNING/BZ_RUN   | Compress from `next_in` to `next_out` as much as possible.| RUNNING | BZ_RUN_OK |
* | RUNNING/BZ_FLUSH | Remember current value of `next_in`. \n Compress from `next_in` to `next_out` as much as possible, but do not accept any more input. | FLUSHING|BZ_FLUSH_OK |
* | RUNNING/BZ_FINISH| Remember current value of `next_in`. \n Compress from `next_in` to `next_out` as much as possible, but do not accept any more input.| FINISHING| BZ_FINISH_OK|
* | FLUSHING/BZ_FLUSH| Compress from `next_in` to `next_out` as much as possible, but do not accept any more input.| | |
* | ^                |  If all the existing input has been used up and all compressed output has been removed | RUNNING | BZ_RUN_OK|
* | ^                |  else  | FLUSHING | BZ_FLUSH_OK|
* | FLUSHING/other   | Illegal. | - |BZ_SEQUENCE_ERROR |
* | FINISHING/BZ_FINISH | Compress from `next_in` to `next_out` as much as possible, but to not accept any more input. | | |
* | ^                   | If all the existing input has been used up and all compressed output has been removed|IDLE | BZ_STREAM_END|
* | ^                   | else |FINISHING |BZ_FINISH_OK |
* | FINISHING/other     | Illegal.| - | BZ_SEQUENCE_ERROR|
*
* Trivial other possible return value:
*  - `BZ_PARAM_ERROR` - If strm is NULL, or strm->s is NULL
*
* @note
* -# Before each call to `BZ2_bzCompress`, `next_in` should point at the data to be compressed, and `avail_in `should 
* indicate the number of bytes the library may read. Similarly, `next_out` should point to a buffer in which the compressed 
* data is to be placed, with `avail_out` indicating the available output space.
*
* -# You may provide and remove as little or as much data as you like on each call of `BZ2_bzCompress`. In the limit, 
* it is acceptable to supply and remove data one byte at a time, although this would be terribly inefficient. 
* You should always ensure that at least one byte of output space is available at each call.
* 
*/
BZ_EXTERN int BZ_API(BZ2_bzCompress) ( 
      bz_stream* strm, 
      int action 
   );

/*!
* @brief Releases all memory associated with a compression stream.
* 
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b strm    | in,out  | Holds all data pertaining to the compression activity. |
*
* @return
* | Result     | Description |
* |:-----------|:------------|
* | Success    | `BZ_OK`       |
* | Fail       | `BZ_PARAM_ERROR` - If strm is NULL or strm->s is NULL|
*/
BZ_EXTERN int BZ_API(BZ2_bzCompressEnd) ( 
      bz_stream* strm 
   );

/*!
* @brief  Prepares for decompression. 
* 
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b strm      | in,out | Holds all data pertaining to the decompression activity. `strm` should be allocated and initialised before the call.|
* | \b verbosity | in     | A number between 0 and 4 inclusive. 0 is silent, and greater numbers give increasingly verbose monitoring/debugging output.|
* | \b small     | in     | If `small` is \b nonzero , the library will use an alternative decompression algorithm which uses less memory but at the cost of decompressing more slowly. |
*
* @return
* | Result     | Description |
* |:-----------|:------------|
* | Success    | `BZ_OK`     |
* | Fail       | `BZ_CONFIG_ERROR` - If the library has been mis-compiled |
* | ^          | `BZ_PARAM_ERROR` - If ( small != 0 && small != 1 ) or (verbosity < 0 \|\| verbosity > 4) |
* | ^          | `BZ_MEM_ERROR` - If available memory is insufficient. |
*
* @note 
* -# Fields `bzalloc`, `bzfree` and `opaque` should be set if a custom memory allocator is required, 
* or made NULL for the normal malloc / free routines. \n 
* Upon return, the internal state will have been initialised, and `total_in` and `total_out` will be zero.
* -# If the library has been compiled with  `-DBZ_NO_STDIO`, no monitoring/debugging output will appear for any verbosity setting.
* -# The amount of memory needed to decompress a stream cannot be determined until the stream's header has been read, 
* so even if `BZ2_bzDecompressInit` succeeds, a subsequent `BZ2_bzDecompress` could fail with `BZ_MEM_ERROR`.
*/
BZ_EXTERN int BZ_API(BZ2_bzDecompressInit) ( 
      bz_stream *strm, 
      int       verbosity, 
      int       small
   );

/*!
* @brief Provides more input and/or output buffer space for the library. 
* The caller maintains input and output buffers, and uses BZ2_bzDecompress to transfer data between them. \n 
* BZ2_bzDecompress updates `next_in`, `avail_in` and `total_in` to reflect the number of bytes it has read. It also 
* updates `next_out`, `avail_out` and `total_out` to reflect the number of bytes output.
* 
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b strm    | in,out      | Holds all data pertaining to the decompression activity.
*
* @return
* | Result     | Description |
* |:-----------|:------------|
* | Success    | `BZ_OK`     |
* | Fail       | `BZ_PARAM_ERROR` - If strm is NULL or strm->s is NULL or strm->avail_out < 1 |
* | ^          | `BZ_SEQUENCE_ERROR` - If s->state is BZ_X_IDLE |
* | ^          | `BZ_DATA_ERROR` - If a data integrity error is detected in the compressed stream |
* | ^          | `BZ_DATA_ERROR_MAGIC` - If the compressed stream doesn't begin with the right magic bytes |
* | ^          | `BZ_MEM_ERROR` - If available memory is insufficient. |
* | ^          | `BZ_STREAM_END` - If the logical end of the data stream was detected and all output in has been consumed, eg s->avail_out > 0 |
*
* @note 
* -# Before each call to BZ2_bzDecompress, `next_in` should point at the compressed data, and `avail_in` should 
* indicate the number of bytes the library may read.
* Similarly, `next_out` should point to a buffer in which the uncompressed output is to be placed,
* with `avail_out` indicating available output space. 
*
* -# You may provide and remove as little or as much data as you like on each call of BZ2_bzDecompress. 
* In the limit, it is acceptable to supply and remove data one byte at a time, although this would be terribly inefficient. 
* You should always ensure that at least one byte of output space is available at each call.

* -# You should provide input and remove output as described above, and repeatedly call BZ2_bzDecompress until 
* `BZ_STREAM_END` is returned. Appearance of `BZ_STREAM_END` denotes that BZ2_bzDecompress has detected the logical end of 
* the compressed stream. BZ2_bzDecompress will not produce `BZ_STREAM_END` until all output data has been placed into the
* output buffer, so once `BZ_STREAM_END` appears, you are guaranteed to have available all the decompressed output, 
* and BZ2_bzDecompressEnd can safely be called.

* -# If case of an error return value, you should call `BZ2_bzDecompressEnd` to clean up and release memory.
*/
BZ_EXTERN int BZ_API(BZ2_bzDecompress) ( 
      bz_stream* strm 
   );

/*!
* @brief Releases all memory associated with a decompression stream.
* 
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b strm    | in,out      | Holds all data pertaining to the decompression activity. |
*
* @return
* | Result     | Description |
* |:-----------|:------------|
* | Success    | `BZ_OK`       |
* | Fail       | `BZ_PARAM_ERROR` - If strm is NULL or strm->s is NULL|
* 
*/
BZ_EXTERN int BZ_API(BZ2_bzDecompressEnd) ( 
      bz_stream *strm 
   );

/**
 * @}
*/

/*-- High(er) level library functions --*/

/// @cond DOXYGEN_SHOULD_SKIP_THIS 
#ifndef BZ_NO_STDIO
#define BZ_MAX_UNUSED 5000

typedef void BZFILE;
/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

/**
 * @name High(er) level library functions
 * @brief This interface provides functions for reading and writing bzip2 format files. First, some general points.
 *
 * -# All of the functions take an `int*` first argument, `bzerror`. After each call, `bzerror` should be consulted first 
 * to determine the outcome of the call. If `bzerror` is `BZ_OK`, the call completed successfully, and only then should the 
 * return value of the function (if any) be consulted. If `bzerror` is `BZ_IO_ERROR`, there was an error reading/writing the 
 * underlying compressed file, and you should then consult `errno` / `perror` to determine the cause of the difficulty. 
 * `bzerror` may also be set to various other values; precise details are given on a per-function basis below.
 *
 * -# If `bzerror` indicates an error (ie, anything except `BZ_OK` and `BZ_STREAM_END`), you should immediately 
 * call `BZ2_bzReadClose` (or `BZ2_bzWriteClose`, depending on whether you are attempting to read or to write) to 
 * free up all resources associated with the stream. Once an error has been indicated, behaviour of all calls except 
 * `BZ2_bzReadClose` / `BZ2_bzWriteClose` is undefined. The implication is that 
 *  - `bzerror` should be checked after each call, and 
 *  - if bzerror indicates an error, `BZ2_bzReadClose` / `BZ2_bzWriteClose` should then be called to clean up.
 * 
 * -# The `FILE*` arguments passed to `BZ2_bzReadOpen` / `BZ2_bzWriteOpen` should be set to binary mode. 
 * Most Unix systems will do this by default, but other platforms, including Windows and Mac, will not. 
 * If you omit this, you may encounter problems when moving code to new platforms.
 *
 * -# Memory allocation requests are handled by `malloc` / `free`. At present there is no facility for 
 * user-defined memory allocators in the file I/O functions (could easily be added, though).
 * @{
*/

/*!
* @brief Prepare to read compressed data from file handle f.
* 
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b bzerror    | out | It should be consulted first to determine the outcome of the call. If bzerror is `BZ_OK`, the call completed successfully, and only then should the return value of the function (if any) be consulted. |
* | \b f          | in  | Refers to a file which has been opened for reading, and for which the error indicator (ferror(f)) is not set.|
* | \b verbosity  | in  | A number between 0 and 4 inclusive. 0 is silent, and greater numbers give increasingly verbose monitoring/debugging output. |
* | \b small      | in  | If `small` is 1, the library will try to decompress using less memory, at the expense of speed. |
* | \b unused     | in,out  | Points to data that will be decompressed by `BZ2_bzRead` before starting to read from the file f. |
* | \b nUnused    | in,out  | Number of bytes to be decompressed by `BZ2_bzRead` starting at `unused`.  |
*
* @return
* | Result     | Description |
* |:-----------|:------------|
* | Success    | Pointer to an abstract BZFILE, if bzerror is `BZ_OK` |
* | Fail       | NULL |
*
* @note
* -# Possible assignments to bzerror:
*  - `BZ_CONFIG_ERROR` - If the library has been mis-compiled
*  - `BZ_PARAM_ERROR` - If
*     - f is NULL
*     - or small is neither 0 nor 1
*     - or ( unused == NULL && nUnused != 0 )
*     - or ( unused != NULL && !(0 <= nUnused <= BZ_MAX_UNUSED) )
*  - `BZ_IO_ERROR` - If ferror(f) is nonzero
*  - `BZ_MEM_ERROR` - If available memory is insufficient.
*  - `BZ_OK` - otherwise.
* 
* -# `BZ2_bzRead` will decompress the nUnused bytes starting at unused, before starting to read from the file f. 
*     At most `BZ_MAX_UNUSED` bytes may be supplied like this. If this facility is not required, you should pass 
*     \b NULL and \b 0 for unused and nUnused respectively.
* 
* -# The amount of memory needed to decompress a file cannot be determined until the file's header has been read. 
*    So it is possible that `BZ2_bzReadOpen` returns `BZ_OK` but a subsequent call of `BZ2_bzRead` will return `BZ_MEM_ERROR`.
* 
*/
BZ_EXTERN BZFILE* BZ_API(BZ2_bzReadOpen) ( 
      int*  bzerror,   
      FILE* f, 
      int   verbosity, 
      int   small,
      void* unused,    
      int   nUnused 
   );

/*!
* @brief Releases all memory pertaining to the compressed file b.
* 
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b bzerror    | out    | It should be consulted first to determine the outcome of the call. If bzerror is `BZ_OK`, the call completed successfully, and only then should the return value of the function (if any) be consulted. |
* | \b b          | in,out | Refers to the compressed file. |
*
* @return void 

* @note
* -# Possible assignments to bzerror:
*     - `BZ_SEQUENCE_ERROR` - If b was opened with BZ2_bzOpenWrite
*     - `BZ_OK` - otherwise
* -# `BZ2_bzReadClose` does not call fclose on the underlying file handle, so you should do that yourself if appropriate. 
* -# `BZ2_bzReadClose` should be called to clean up after all error situations.
* 
*/
BZ_EXTERN void BZ_API(BZ2_bzReadClose) ( 
      int*    bzerror, 
      BZFILE* b 
   );

/*!
* @brief Returns data which was read from the compressed file but was not needed to get to the logical end-of-stream.
* 
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b bzerror    | out  | It should be consulted first to determine the outcome of the call. If bzerror is `BZ_OK`, the call completed successfully, and only then should the return value of the function (if any) be consulted. |
* | \b b          | in   | Refers to the compressed file. |
* | \b unused     | in,out  | *unused is set to the address of the data. |
* | \b nUnused    | in,out  | *nUnused is set to the number of bytes. It is a value between \b 0 and \b BZ_MAX_UNUSED inclusive.  |
*
* @return void
*
* @note
* -# Possible assignments to bzerror:
*  - `BZ_PARAM_ERROR` - If
*        - b is NULL or
*        - unused is NULL or nUnused is NULL
*  - `BZ_SEQUENCE_ERROR` - If 
*        - BZ_STREAM_END has not been signalled or
*        - b was opened with BZ2_bzWriteOpen
*  - `BZ_OK` - otherwise
* 
* -# This function may only be called once `BZ2_bzRead` has signalled BZ_STREAM_END but before `BZ2_bzReadClose`.
* 
*/
BZ_EXTERN void BZ_API(BZ2_bzReadGetUnused) ( 
      int*    bzerror, 
      BZFILE* b, 
      void**  unused,  
      int*    nUnused 
   );

/*!
* @brief Reads up to len (uncompressed) bytes from the compressed file b into the buffer buf.
* 
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b bzerror | out    | It should be consulted first to determine the outcome of the call. If bzerror is `BZ_OK`, the call completed successfully, and only then should the return value of the function (if any) be consulted. |
* | \b b       | in,out | Refers to the compressed file. |
* | \b buf     | in,out | Buffer into which len (uncompressed) bytes is read into. |
* | \b len     | in     | Integer variable indicating length. |

*
* @return
* | Result     | Description |
* |:-----------|:------------|
* | Success    | number of bytes read , if bzerror is BZ_OK or BZ_STREAM_END |
* | Fail       | undefined |
*
* @note
* -# Possible assignments to bzerror:
*  - `BZ_PARAM_ERROR` - If b is NULL or buf is NULL or len < 0
*  - `BZ_SEQUENCE_ERROR` - If b was opened with BZ2_bzWriteOpen
*  - `BZ_IO_ERROR` - If there is an error reading from the compressed file
*  - `BZ_UNEXPECTED_EOF` - If the compressed file ended before the logical end-of-stream was detected
*  - `BZ_DATA_ERROR` - If a data integrity error was detected in the compressed stream
*  - `BZ_DATA_ERROR_MAGIC` - If the stream does not begin with the requisite header bytes 
*                            (ie, is not a bzip2 data file).  This is really a special case of BZ_DATA_ERROR.
*  - `BZ_MEM_ERROR` - If available memory is insufficient.
*  - `BZ_STREAM_END` - If the logical end of stream was detected.
*  - `BZ_OK` - otherwise.
* 
* -# If the read was successful, bzerror is set to `BZ_OK` and the number of bytes read is returned. If the logical 
*    end-of-stream was detected, bzerror will be set to `BZ_STREAM_END`, and the number of bytes read is returned. 
*    All other bzerror values denote an error.
*
* -# BZ2_bzRead will supply len bytes, unless the logical stream end is detected or an error occurs. Because of this, 
*    it is possible to detect the stream end by observing when the number of bytes returned is less than the number requested. 
*    Nevertheless, this is regarded as inadvisable; you should instead check bzerror after every call and watch out for BZ_STREAM_END.
*
* -# Internally, BZ2_bzRead copies data from the compressed file in chunks of size `BZ_MAX_UNUSED` bytes before decompressing it. 
*    If the file contains more bytes than strictly needed to reach the logical end-of-stream, BZ2_bzRead will almost certainly 
*    read some of the trailing data before signalling `BZ_SEQUENCE_END`. To collect the read but unused data once`BZ_SEQUENCE_END` 
*    has appeared, call BZ2_bzReadGetUnused immediately before BZ2_bzReadClose.
* 
*/
BZ_EXTERN int BZ_API(BZ2_bzRead) ( 
      int*    bzerror, 
      BZFILE* b, 
      void*   buf, 
      int     len 
   );

/*!
* @brief Prepare to write compressed data to file handle f.
* 
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b bzerror       | out | It should be consulted first to determine the outcome of the call. If bzerror is `BZ_OK`, the call completed successfully, and only then should the return value of the function (if any) be consulted. |
* | \b f             | in  | Refers to a file which has been opened for writing, and for which the error indicator (ferror(f)) is not set.|
* | \b blockSize100k | in  | Specifies the block size to be used for compression. It should be a value between 1 and 9 inclusive, and the actual block size used is 100000 x this figure. |
* | \b verbosity     | in  | A number between 0 and 4 inclusive. 0 is silent, and greater numbers give increasingly verbose monitoring/debugging output. |
* | \b workFactor    | in  | Controls how the compression phase behaves when presented with worst case, highly repetitive, input data. |
*
* @return
* | Result     | Description |
* |:-----------|:------------|
* | Success    | Pointer to an abstract BZFILE, if bzerror is `BZ_OK` |
* | Fail       | NULL |
*
* @note
* -# Possible assignments to bzerror:
*  - `BZ_CONFIG_ERROR` - If the library has been mis-compiled
*  - `BZ_PARAM_ERROR` - If 
*        - f is NULL or
*        - blockSize100k < 1 or blockSize100k > 9
*  - `BZ_IO_ERROR` - If ferror(f) is nonzero
*  - `BZ_MEM_ERROR` - If available memory is insufficient
*  - `BZ_OK` - otherwise
* 
* -# All required memory is allocated at this stage, so if the call completes successfully, `BZ_MEM_ERROR` cannot be 
*    signalled by a subsequent call to `BZ2_bzWrite`.
* 
*/
BZ_EXTERN BZFILE* BZ_API(BZ2_bzWriteOpen) ( 
      int*  bzerror,      
      FILE* f, 
      int   blockSize100k, 
      int   verbosity, 
      int   workFactor 
   );

/*!
* @brief Absorbs len bytes from the buffer buf, eventually to be compressed and written to the file.
* 
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b bzerror | out    | It should be consulted first to determine the outcome of the call. If bzerror is `BZ_OK`, the call completed successfully, and only then should the return value of the function (if any) be consulted. |
* | \b b       | in,out | Refers to the file into which compressed data will be written. |
* | \b buf     | in     | Buffer that consists the data to be compressed. |
* | \b len     | in     | Integer variable indicating length. |
*
* @return void
*
* @note
* -# Possible assignments to bzerror:
*  - `BZ_PARAM_ERROR` - If b is NULL or buf is NULL or len < 0
*  - `BZ_SEQUENCE_ERROR` - If b was opened with BZ2_bzReadOpen
*  - `BZ_IO_ERROR` - If there is an error writing the compressed file.
*  - `BZ_OK` - otherwise
* 
*/
BZ_EXTERN void BZ_API(BZ2_bzWrite) ( 
      int*    bzerror, 
      BZFILE* b, 
      void*   buf, 
      int     len 
   );

/*!
* @brief Compresses and flushes to the compressed file all data so far supplied by `BZ2_bzWrite`. 
* 
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b bzerror     | out    | It should be consulted first to determine the outcome of the call. If bzerror is `BZ_OK`, the call completed successfully, and only then should the return value of the function (if any) be consulted. |
* | \b b           | in,out | Refers to the file into which compressed data will be written. |
* | \b abandon     | in     | In case of no error, in order to attempt to complete the compression operation, or to `fflush` the compressed file, pass a nonzero value to abandon. |
* | \b nbytes_in   | in,out | If `nbytes_in` is non-null, `*nbytes_in` will be set to be the total volume of uncompressed data handled.  |
* | \b nbytes_out  | in,out | If `nbytes_out` is non-null, `*nbytes_out` will be set to the total volume of compressed data written.  |
*
* @return void
*
* @note
* -# Possible assignments to bzerror:
*     - `BZ_SEQUENCE_ERROR` - If b was opened with BZ2_bzReadOpen
*     - `BZ_IO_ERROR` - If there is an error writing the compressed file
*     - `BZ_OK` - otherwise
*
* -#  The logical end-of-stream markers are also written, so subsequent calls to `BZ2_bzWrite` are illegal. 
*      All memory associated with the compressed file `b` is released. `fflush` is called on the compressed file, 
*      but it is not `fclose`'d. 
*
* -# If BZ2_bzWriteClose is called to clean up after an error, the only action is to release the memory. 
*    The library records the error codes issued by previous calls, so this situation will be detected automatically. \n 
*    There is no attempt to complete the compression operation, nor to `fflush` the compressed file. 
*    You can force this behaviour to happen even in the case of no error, by passing a nonzero value to abandon.
*
*/
BZ_EXTERN void BZ_API(BZ2_bzWriteClose) ( 
      int*          bzerror, 
      BZFILE*       b, 
      int           abandon, 
      unsigned int* nbytes_in, 
      unsigned int* nbytes_out 
   );


/*!
* @brief Compresses and flushes to the compressed file all data so far supplied by `BZ2_bzWrite`. 
* 
* @note For compatibility with older versions of the library, BZ2_bzWriteClose() only yields the lower 32 bits of these counts. 
*       Use BZ2_bzWriteClose64() if you want the full 64 bit counts. These two functions are otherwise absolutely identical.
*/
BZ_EXTERN void BZ_API(BZ2_bzWriteClose64) ( 
      int*          bzerror, 
      BZFILE*       b, 
      int           abandon, 
      unsigned int* nbytes_in_lo32, 
      unsigned int* nbytes_in_hi32, 
      unsigned int* nbytes_out_lo32, 
      unsigned int* nbytes_out_hi32
   );
#endif
/**
 * @}
*/

/*-- Utility functions --*/
/**
 * @name Utility functions
 * @{
*/

/// @cond DOXYGEN_SHOULD_SKIP_THIS
/*
* Provides the maximum size that BZIP2 compression may output in a "worst case" scenario (input data not compressible)
* where `insize` is the size of source buffer to be compressed.
*/
unsigned int BZ2_bzCompressBound(unsigned int insize);
/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

/*!
*
*  @rst 
*  .. _BZ2_bzBuffToBuffCompress:
*  @endrst
*
* @brief Attempts to compress the data in `source` into the destination buffer, `dest`. 
* 
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b dest          | out    | Destination buffer, compressed data is kept here, memory should be allocated already. |
* | \b destLen       | in,out | Size of buffer `dest` (which must be already allocated).|
* | \b source        | in     | Source buffer, the data to be compressed is copied/or pointed here. |
* | \b sourceLen     | in     | Size of buffer `source`. |
* | \b blockSize100k | in     | Specifies the block size to be used for compression. It should be a value between 1 and 9 inclusive, and the actual block size used is 100000 x this figure.    |
* | \b verbosity     | in     | A number between 0 and 4 inclusive. 0 is silent, and greater numbers give increasingly verbose monitoring/debugging output.  |
* | \b workFactor    | in     | Controls how the compression phase behaves when presented with worst case, highly repetitive, input data. |
*
* @return
* | Result     | Description |
* |:-----------|:------------|
* | Success    | `BZ_OK`       |
* | Fail       | `BZ_CONFIG_ERROR` - If the library has been mis-compiled |
* | ^          | `BZ_PARAM_ERROR` \n If dest is NULL or destLen is NULL or \n  blockSize100k < 1 or blockSize100k > 9 or \n verbosity < 0 or verbosity > 4 or \n workFactor < 0 or workFactor > 250 |
* | ^          | `BZ_MEM_ERROR` - If available memory is insufficient |
* | ^          | `BZ_OUTBUFF_FULL` - If the size of the compressed data exceeds *destLen |
* 
* @note
* -# If the destination buffer is big enough, *destLen is set to the size of the compressed data, and `BZ_OK` is returned. 
* If the compressed data won't fit, *destLen is unchanged, and `BZ_OUTBUFF_FULL` is returned.
* 
* -# To guarantee that the compressed data will fit in its buffer, allocate an output buffer of size 
* <b> 1% larger than the uncompressed data, plus six hundred </b> extra bytes.
*
* -# `blockSize100k` = 9 gives the best compression but takes most memory. 
*
* -# If compression runs into difficulties caused by repetitive data, the library switches from the <b> standard sorting algorithm </b> to a <b> fallback algorithm </b>.
* The fallback is slower than the standard algorithm by perhaps a factor of three, but always behaves reasonably, no matter how bad the input.

* -# Lower values of `workFactor` reduce the amount of effort the standard algorithm will expend before resorting to the fallback. Set this parameter carefully; 
*  - too low, and many inputs will be handled by the fallback algorithm and so compress rather slowly, 
*  - too high, and your average-to-worst case compression times can become very large. 
*  - The default value of \b 30 gives reasonable behaviour over a wide range of circumstances. \n
* Allowable values range from \b 0 to \b 250 inclusive. 0 is a special case, equivalent to using the default value of 30.
* 
* -# Compression is a one-shot event, done with a single call to this function. 
* The resulting compressed data is a complete bzip2 format data stream. There is no mechanism for making additional 
* calls to provide extra input data. If you want that kind of mechanism, use the low-level interface.
* 
*/
BZ_EXTERN int BZ_API(BZ2_bzBuffToBuffCompress) ( 
      char*         dest, 
      unsigned int* destLen,
      char*         source, 
      unsigned int  sourceLen,
      int           blockSize100k, 
      int           verbosity, 
      int           workFactor 
   );

/*!
*
*  @rst 
*  .. _BZ2_bzBuffToBuffDecompress:
*  @endrst
*
* @brief Attempts to decompress the data in `source` into the destination buffer, `dest`. 
* 
* | Parameters | Direction   | Description |
* |:-----------|:-----------:|:------------|
* | \b dest          | out    | This is the destination buffer, the data is decompressed to this buffer. |
* | \b destLen       | in,out | Size of buffer `dest` (which must be already allocated).|
* | \b source        | in     | Source buffer, the data to be decompressed is pointed here. |
* | \b sourceLen     | in     | Size of buffer `source`. |
* | \b small         | in     | If `small` is \b nonzero , the library will use an alternative decompression algorithm which uses less memory but at the cost of decompressing more slowly. |
* | \b verbosity     | in     | A number between 0 and 4 inclusive. 0 is silent, and greater numbers give increasingly verbose monitoring/debugging output.  |
*
* @return
* | Result     | Description |
* |:-----------|:------------|
* | Success    | `BZ_OK`       |
* | Fail       | `BZ_CONFIG_ERROR` - If the library has been mis-compiled |
* | ^          | `BZ_PARAM_ERROR` \n if dest is NULL or destLen is NULL or \n  small != 0 && small != 1 or \n verbosity < 0 or verbosity > 4 |
* | ^          | `BZ_MEM_ERROR` - If available memory is insufficient. |
* | ^          | `BZ_OUTBUFF_FULL` - If the size of the uncompressed data exceeds *destLen |
* | ^          | `BZ_DATA_ERROR` - If a data integrity error was detected in the compressed data |
* | ^          | `BZ_DATA_ERROR_MAGIC` - If the compressed data doesn't begin with the right magic bytes |
* | ^          | `BZ_UNEXPECTED_EOF` - If the compressed data ends unexpectedly |
* 
* @note
* -# If the destination buffer is big enough, *destLen is set to the size of the uncompressed data, and `BZ_OK` is returned. 
* If the uncompressed data won't fit, *destLen is unchanged, and `BZ_OUTBUFF_FULL` is returned.
* 
* -# `source` is assumed to hold a complete bzip2 format data stream. 
* BZ2_bzBuffToBuffDecompress tries to decompress the entirety of the stream into the output buffer.
*
* -# BZ2_bzBuffToBuffDecompress will not write data at or beyond dest[*destLen], even in case of buffer overflow.
* 
*/
BZ_EXTERN int BZ_API(BZ2_bzBuffToBuffDecompress) ( 
      char*         dest, 
      unsigned int* destLen,
      char*         source, 
      unsigned int  sourceLen,
      int           small, 
      int           verbosity 
   );

/**
 * @}
*/


/// @cond DOXYGEN_SHOULD_SKIP_THIS
/*--
   Code contributed by Yoshioka Tsuneo (tsuneo@rr.iij4u.or.jp)
   to support better zlib compatibility.
   This code is not _officially_ part of libbzip2 (yet);
   I haven't tested it, documented it, or considered the
   threading-safeness of it.
   If this code breaks, please contact both Yoshioka and me.
--*/

BZ_EXTERN const char * BZ_API(BZ2_bzlibVersion) (
      void
   );

#ifndef BZ_NO_STDIO
BZ_EXTERN BZFILE * BZ_API(BZ2_bzopen) (
      const char *path,
      const char *mode
   );

BZ_EXTERN BZFILE * BZ_API(BZ2_bzdopen) (
      int        fd,
      const char *mode
   );
         
BZ_EXTERN int BZ_API(BZ2_bzread) (
      BZFILE* b, 
      void* buf, 
      int len 
   );

BZ_EXTERN int BZ_API(BZ2_bzwrite) (
      BZFILE* b, 
      void*   buf, 
      int     len 
   );

BZ_EXTERN int BZ_API(BZ2_bzflush) (
      BZFILE* b
   );

BZ_EXTERN void BZ_API(BZ2_bzclose) (
      BZFILE* b
   );

BZ_EXTERN const char * BZ_API(BZ2_bzerror) (
      BZFILE *b, 
      int    *errnum
   );
#endif

/**
 * @name AOCL Functions
 * @brief These functions are not part of open source code, these are introduced by AOCL-Compression
 * library to control AOCL introduced optimization levels dynamically.
 * 
 * @note These functions are for internal purposes only, not recommended for external use.
 * 
 * @{
 */

/**
 * @brief AOCL-Compression defined setup function that configures code path dynamically with the right
 * AMD optimized bzip2 routines depending upon the detected CPU features if `optOff=0`.
 * 
 * Except for the initial call, it's necessary to execute aocl_destroy_bzip2() before any subsequent calls
 * to this function. Failure to call the destroy function prior to invoking this function will result
 * in bzip2 following the code path of set at first setup call or  the most recent setup call that was
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
BZ_EXTERN char * BZ_API(aocl_setup_bzip2) (
      int optOff,
      int optLevel,
      size_t insize,
      size_t level,
      size_t windowLog
   );

/**
 * @brief It is necessary to execute this destroy function after the initial invocation of the
 * aocl_setup_bzip2() function, prior to initiating the setup function again.
 */
BZ_EXTERN void BZ_API(aocl_destroy_bzip2) (void);

/**
 * @}
 */

/// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

#ifdef AOCL_UNIT_TEST
/* Wrapper function for libsais function for unit testing. */
BZ_EXTERN int BZ_API(Test_libsais(const unsigned char * T, int * SA, int n, int fs, int * freq));
#endif /* AOCL_UNIT_TEST */

#ifdef __cplusplus
}
#endif

#endif
/**
 * @}
*/

/*-------------------------------------------------------------*/
/*--- end                                           bzlib.h ---*/
/*-------------------------------------------------------------*/

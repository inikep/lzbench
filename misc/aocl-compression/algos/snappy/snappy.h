// Copyright 2005 and onwards Google Inc.
// Modifications Copyright (C) 2022-2025, Advanced Micro Devices. All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// A light-weight compression algorithm.  It is designed for speed of
// compression and decompression, rather than for the utmost in space
// savings.
//
// For getting better compression ratios when you are compressing data
// with long repeated sequences or compressing data that is similar to
// other data, while still compressing fast, you might look at first
// using BMDiff and then compressing the output of BMDiff with
// Snappy.

#ifndef THIRD_PARTY_SNAPPY_SNAPPY_H__
#define THIRD_PARTY_SNAPPY_SNAPPY_H__

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "snappy-stubs-public.h"

// AOCL definitions start
#ifndef SNAPPYLIB_VISIBILITY
#  if defined(__GNUC__) && (__GNUC__ >= 4)
#    define SNAPPYLIB_VISIBILITY __attribute__ ((visibility ("default")))
#  else
#    define SNAPPYLIB_VISIBILITY
#  endif
#endif
#if defined(SNAPPY_DLL_EXPORT) && (SNAPPY_DLL_EXPORT==1)
#  define SNAPPYLIB_API __declspec(dllexport) SNAPPYLIB_VISIBILITY
#elif defined(SNAPPY_DLL_IMPORT) && (SNAPPY_DLL_IMPORT==1)
#  define SNAPPYLIB_API __declspec(dllimport) SNAPPYLIB_VISIBILITY /* It isn't required but allows to generate better code, saving a function pointer load from the IAT and an indirect jump.*/
#else
#  define SNAPPYLIB_API SNAPPYLIB_VISIBILITY
#endif
#ifdef AOCL_SNAPPY_MATCH_SKIP_OPT
#define AOCL_SNAPPY_MATCH_SKIPPING_THRESHOLD 8 // longer skips in next iter if bytes_between_hash_lookups grows above this
#endif
// AOCL definitions end

namespace snappy {
/*!
 * \addtogroup SNAPPY_API
 * @brief
 * A light-weight compression algorithm.  It is designed for speed of
 * compression and decompression, rather than for the utmost in space
 * savings.
 *
 * For getting better compression ratios when you are compressing data
 * with long repeated sequences or compressing data that is similar to
 * other data, while still compressing fast, you might look at first
 * using BMDiff and then compressing the output of BMDiff with
 * Snappy.
 * @{
 */
  class Source;
  class Sink;

/**
 * @brief Options for configuring compression.
 */
  struct CompressionOptions {
    /**
     * @brief Compression level.
     * 
     * Level 1 is the fastest.
     * Level 2 is a little slower but provides better compression. Level 2 is
     * **EXPERIMENTAL** for the time being. It might happen that we decide to
     * fall back to level 1 in the future.
     * Levels 3+ are currently not supported. We plan to support levels up to
     * 9 in the future.
     * If you played with other compression algorithms, level 1 is equivalent to
     * fast mode (level 1) of LZ4, level 2 is equivalent to LZ4's level 2 mode
     * and compresses somewhere around zstd:-3 and zstd:-2 but generally with
     * faster decompression speeds than snappy:1 and zstd:-3.
     */
    // Compression level.
    // Level 1 is the fastest
    // Level 2 is a little slower but provides better compression. Level 2 is
    // **EXPERIMENTAL** for the time being. It might happen that we decide to
    // fall back to level 1 in the future.
    // Levels 3+ are currently not supported. We plan to support levels up to
    // 9 in the future.
    // If you played with other compression algorithms, level 1 is equivalent to
    // fast mode (level 1) of LZ4, level 2 is equivalent to LZ4's level 2 mode
    // and compresses somewhere around zstd:-3 and zstd:-2 but generally with
    // faster decompression speeds than snappy:1 and zstd:-3.
    int level = DefaultCompressionLevel();

    /**
     * @brief Default constructor.
     */
    constexpr CompressionOptions() = default;

    /**
     * @brief Constructor with specified compression level.
     * 
     * @param compression_level The desired compression level.
     */
    constexpr CompressionOptions(int compression_level)
        : level(compression_level) {}

    /**
     * @brief Get the minimum compression level.
     */
    static constexpr int MinCompressionLevel() { return 1; }

    /**
     * @brief Get the maximum compression level.
     */
    static constexpr int MaxCompressionLevel() { return 2; }

    /**
     * @brief Get the default compression level.
     */
    static constexpr int DefaultCompressionLevel() { return 1; }
  };

/**
 * @name Generic compression/decompression routines.
 */

  // ------------------------------------------------------------------------
  // Generic compression/decompression routines.
  // ------------------------------------------------------------------------
  
  /**
   * @brief
   * Compress the bytes read from "*source" and append to "*sink". Return the
   * number of bytes written.
   *
   *  |Parameters |Direction|Description                                                                                                                                                                                                      |
   *  |:----------|:-------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   *  | \b reader | in,out  | A Source is an interface that yields a sequence of bytes, you can initialize it by calling snappy::ByteArraySource(inBuf,inBufLen);where inBuf is the pointer to original data and inBufLen is the size of inBuf|
   *  | \b writer | in,out  | A Sink is an interface that consumes a sequence of bytes, you can initialize it by calling snappy::UncheckedByteArraySink(dest); where dest is the pointer to the destination buffer.                           |
   *
   *  @return
   *  |Result | Description                                         |
   *  |:------|:----------------------------------------------------|
   *  |Success| Return the number of bytes written.                 |
   *  |Failure| Return 0 upon failure or NULL parameters are passed |
   */
  // Compress the bytes read from "*reader" and append to "*writer". Return the
  // number of bytes written.
  // First version is to preserve ABI.
  SNAPPYLIB_API size_t Compress(Source* reader, Sink* writer);

  /**
 * @brief
 * 
 * Same as `Compress` above but takes additional CompressionOptions.
 *
 *  |Parameters |Direction|Description                                                                                                                                                                                                      |
 *  |:----------|:-------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 *  | \b reader | in,out  | A Source is an interface that yields a sequence of bytes, you can initialize it by calling snappy::ByteArraySource(inBuf,inBufLen);where inBuf is the pointer to original data and inBufLen is the size of inBuf|
 *  | \b writer | in,out  | A Sink is an interface that consumes a sequence of bytes, you can initialize it by calling snappy::UncheckedByteArraySink(dest); where dest is the pointer to the destination buffer.                           |
 *  | \b options| in      | Compression options.                                                                                                                                                                                            |
 *
 *  @return
 *  |Result | Description                                         |
 *  |:------|:----------------------------------------------------|
 *  |Success| Return the number of bytes written.                 |
 *  |Failure| Return 0 upon failure or NULL parameters are passed |
 */
  SNAPPYLIB_API size_t Compress(Source* reader, Sink* writer,
                  CompressionOptions options);

  /**
   * @brief
   * Find the uncompressed length of the given stream, as given by the header.
   * Note that the true length could deviate from this; the stream could e.g.
   * be truncated.
   *
   *  |Parameters   |Direction|Description                                                                                                                                                                                                       |
   *  |:------------|:-------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   *  | \b source   | in,out  | A Source is an interface that yields a sequence of bytes, you can initialize it by calling snappy::ByteArraySource(inBuf,inBufLen); where inBuf is the pointer to original data and inBufLen is the size of inBuf|
   *  | \b result   | out     | Uncompressed length of the given stream is stored here.                                                                                                                                                          |
   *
   * @note Also note that this leaves "*source" in a state that is unsuitable for
   * further operations, such as RawUncompress(). You will need to rewind
   * or recreate the source yourself before attempting any further calls.
   *
   *  @return
   *  |Result | Description                                                          |
   *  |:------|:---------------------------------------------------------------------|
   *  |Success| If the data inside the source is uncorrupted it will return \b true. |
   *  |Failure| It will return \b false if the data inside the source is corrupted.  |
   */
  // Find the uncompressed length of the given stream, as given by the header.
  // Note that the true length could deviate from this; the stream could e.g.
  // be truncated.
  //
  // Also note that this leaves "*source" in a state that is unsuitable for
  // further operations, such as RawUncompress(). You will need to rewind
  // or recreate the source yourself before attempting any further calls.
  SNAPPYLIB_API bool GetUncompressedLength(Source* source, uint32_t* result);

/**
 * @name Higher-level string based routines.
 * @brief Higher-level string based routines (should be sufficient for most users)
 * @{
 */
  // ------------------------------------------------------------------------
  // Higher-level string based routines (should be sufficient for most users)
  // ------------------------------------------------------------------------

  /**
   * @brief
   * Sets "*compressed" to the compressed version of "input[0,input_length-1]".
   * Original contents of *compressed are lost.
   *
   *  |Parameters       |Direction|Description                                                          |
   *  |:----------------|:-------:|:------------------------------------------------------------------- |
   *  | \b input        | in      | This is the buffer where the data we want to compress is accessible.|
   *  | \b input_length | in      | Length of the input buffer.                                         |
   *  | \b compressed   | in,out  | This is a buffer in which compressed data is stored.                |
   *
   * @attention REQUIRES: "input[]" is not an alias of "*compressed".
   *
   *  @return
   *  |Result | Description                                         |
   *  |:------|:----------------------------------------------------|
   *  |Success| Return the number of bytes written.                 |
   *  |Failure| Return 0 upon failure or NULL parameters are passed |
   */
  // Sets "*compressed" to the compressed version of "input[0..input_length-1]".
  // Original contents of *compressed are lost.
  //
  // REQUIRES: "input[]" is not an alias of "*compressed".
  // First version is to preserve ABI.
  SNAPPYLIB_API size_t Compress(const char* input, size_t input_length,
                  std::string* compressed);

  /**
   * @brief
   * 
   * Same as `Compress` above but takes additional CompressionOptions.
   *
   *  |Parameters       |Direction|Description                                                          |
   *  |:----------------|:-------:|:------------------------------------------------------------------- |
   *  | \b input        | in      | This is the buffer where the data we want to compress is accessible.|
   *  | \b input_length | in      | Length of the input buffer.                                         |
   *  | \b compressed   | in,out  | This is a buffer in which compressed data is stored.                |
   *  | \b options      | in      | Compression options.                                                |
   *
   * @attention REQUIRES: "input[]" is not an alias of "*compressed".
   *
   *  @return
   *  |Result | Description                                         |
   *  |:------|:----------------------------------------------------|
   *  |Success| Return the number of bytes written.                 |
   *  |Failure| Return 0 upon failure or NULL parameters are passed |
   */
  SNAPPYLIB_API size_t Compress(const char* input, size_t input_length,
                  std::string* compressed, CompressionOptions options);

  /**
   * @brief
   * Same as `Compress` above but taking an `iovec` array as input. Note that
   * this function preprocesses the inputs to compute the sum of
   * `iov[0..iov_cnt-1].iov_len` before reading. To avoid this, use
   * `RawCompressFromIOVec` below.
   * First version is to preserve ABI.
   *
   *  |Parameters       |Direction|Description                                                          |
   *  |:----------------|:-------:|:------------------------------------------------------------------- |
   *  | \b iov          | in      | Input iovec array.                                                  |
   *  | \b iov_cnt      | in      | Length of iovec array.                                              |
   *  | \b compressed   | in,out  | This is a buffer in which compressed data is stored.                |
   *
   *  @return
   *  |Result | Description                                         |
   *  |:------|:----------------------------------------------------|
   *  |Success| Return the number of bytes written.                 |
   *  |Failure| Return 0 upon failure or NULL parameters are passed |
   */
  // Same as `Compress` above but taking an `iovec` array as input. Note that
  // this function preprocesses the inputs to compute the sum of
  // `iov[0..iov_cnt-1].iov_len` before reading. To avoid this, use
  // `RawCompressFromIOVec` below.
  // First version is to preserve ABI.
  SNAPPYLIB_API size_t CompressFromIOVec(const struct iovec* iov, size_t iov_cnt,
                           std::string* compressed);

  /**
   * @brief
   * 
   * Same as `CompressFromIOVec` above but takes additional CompressionOptions.
   *
   *  |Parameters       |Direction|Description                                                          |
   *  |:----------------|:-------:|:------------------------------------------------------------------- |
   *  | \b iov          | in      | Input iovec array.                                                  |
   *  | \b iov_cnt      | in      | Length of iovec array.                                              |
   *  | \b compressed   | in,out  | This is a buffer in which compressed data is stored.                |
   *  | \b options      | in      | Compression options.                                                |
   *
   *  @return
   *  |Result | Description                                         |
   *  |:------|:----------------------------------------------------|
   *  |Success| Return the number of bytes written.                 |
   *  |Failure| Return 0 upon failure or NULL parameters are passed |
   */
  SNAPPYLIB_API size_t CompressFromIOVec(const struct iovec* iov, size_t iov_cnt,
                           std::string* compressed,
                           CompressionOptions options);


  /**
   * @brief
   *  Decompresses "compressed[0,compressed_length-1]" to "*uncompressed".
   *  Original contents of "*uncompressed" are lost.
   *
   *  |Parameters            |Direction|Description|
   *  |:---------------------|:-------:|:----------|
   *  | \b compressed        | in      | This is a buffer which contains compressed data.|
   *  | \b compressed_length | in      | This is the length of the compressed buffer.|
   *  | \b uncompressed      | out     | Uncompressed data is stored in this buffer.|
   *
   * @attention REQUIRES: "compressed[]" is not an alias of "*uncompressed".
   *
   *  @return
   *  |Result | Description                                                                                                        |
   *  |:------|:-------------------------------------------------------------------------------------------------------------------|
   *  |Success| If the data inside the compressed is successfully decompressed it will return \b true. |
   *  |Failure| It will return \b false if the message is corrupted and could not be decompressed.                                                |
   */
  // Decompresses "compressed[0..compressed_length-1]" to "*uncompressed".
  // Original contents of "*uncompressed" are lost.
  //
  // REQUIRES: "compressed[]" is not an alias of "*uncompressed".
  //
  // returns false if the message is corrupted and could not be decompressed
  SNAPPYLIB_API bool Uncompress(const char* compressed, size_t compressed_length,
                  std::string* uncompressed);

/**
 * @}
 */

/**
 * @name Generic compression/decompression routines.
 * @{
 */

  /**
   * @brief Decompresses "compressed" to "*uncompressed".
   * 
   *  |Parameters       |Direction|Description                                                                                                                                                                                                         |
   *  |:----------------|:-------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   *  | \b compressed   |  in,out | A Source is an interface that yields a sequence of bytes, you can initialize it by calling snappy::ByteArraySource(inBuf,inBufLen); where inBuf is the pointer to original data and inBufLen is the size of inBuf .|
   *  | \b uncompressed |  in,out | A Sink is an interface that consumes a sequence of bytes, you can initialize it by calling snappy::UncheckedByteArraySink(dest); where dest is the pointer to the destination buffer.                              |
   * 
   *  @return
   *  |Result | Description                                                                 |
   *  |:------|:----------------------------------------------------------------------------|
   *  |Success| Returns \b true if successful.                                              |
   *  |Failure| Returns \b false  if the message is corrupted and could not be decompressed.|
   */
  // Decompresses "compressed" to "*uncompressed".
  //
  // returns false if the message is corrupted and could not be decompressed
  SNAPPYLIB_API bool Uncompress(Source* compressed, Sink* uncompressed);

  /**
   * @brief 
   * This routine decompresses as much of the "compressed" as possible
   * into sink.  It returns the number of valid bytes added to sink
   * (extra invalid bytes may have been added due to errors; the caller
   * should ignore those). The emitted data typically has length
   * GetUncompressedLength(), but may be shorter if an error is
   * encountered.
   *
   *  |Parameters       |Direction|Description                                                                                                                                                                                                         |
   *  |:----------------|:-------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   *  | \b compressed   |  in,out | A Source is an interface that yields a sequence of bytes, you can initialize it by calling snappy::ByteArraySource(inBuf,inBufLen); where inBuf is the pointer to original data and inBufLen is the size of inBuf .|
   *  | \b uncompressed |  in,out | A Sink is an interface that consumes a sequence of bytes, you can initialize it by calling snappy::UncheckedByteArraySink(dest); where dest is the pointer to the destination buffer.                              |
   *
   *  @return
   *  |Result | Description                                                                                                                                |
   *  |:------|:-------------------------------------------------------------------------------------------------------------------------------------------|
   *  |Success| It returns the number of valid bytes added to sink (extra invalid bytes may have been added due to errors; the caller should ignore those) |
   *  |Failure| Returns 0 if the message is corrupted and could not be decompressed or NULL parameters are passed.                                         |
   */
  // This routine uncompresses as much of the "compressed" as possible
  // into sink.  It returns the number of valid bytes added to sink
  // (extra invalid bytes may have been added due to errors; the caller
  // should ignore those). The emitted data typically has length
  // GetUncompressedLength(), but may be shorter if an error is
  // encountered.
  SNAPPYLIB_API size_t UncompressAsMuchAsPossible(Source* compressed, Sink* uncompressed);

/**
 * @}
 */

/**
 * @name Lower-level character array based routines.
 * @brief These May be useful for efficiency reasons in certain circumstances.
 * @{
 */
  // ------------------------------------------------------------------------
  // Lower-level character array based routines.  May be useful for
  // efficiency reasons in certain circumstances.
  // ------------------------------------------------------------------------

  /**
   * 
   * @rst 
   * .. _RawCompress:
   * @endrst
   * 
   * @brief 
   *
   * Takes the data stored in "input[0..input_length]" and stores
   * it in the array pointed to by "compressed".
   *
   * "*compressed_length" is set to the length of the compressed output.
   *
   *  |Parameters            |Direction|Description                                                          |
   *  |:---------------------|:-------:|:--------------------------------------------------------------------|
   *  | \b input             |  in     | This is the buffer where the data we want to compress is accessible.|
   *  | \b input_length      |  in     | Length of the input buffer.                                         |
   *  | \b compressed        |  out    | This is a buffer in which compressed data is stored.                |
   *  | \b compressed_length |  out    | The length of the data after compression is stored in this.         |
   * 
   * @attention REQUIRES: "compressed" must point to an area of memory that is at
   * least "MaxCompressedLength(input_length)" bytes in length.
   * 
   * @note \b Example:
   *  \code{.cpp}
   *            char  output = new char[snappy::MaxCompressedLength(input_length)];
   *            size_t output_length;
   *            RawCompress(input, input_length, output, &output_length);
   *            ... Process(output, output_length) ...
   *            delete [] output;
   * \endcode
   * @return \b  void
   */
  // REQUIRES: "compressed" must point to an area of memory that is at
  // least "MaxCompressedLength(input_length)" bytes in length.
  //
  // Takes the data stored in "input[0..input_length]" and stores
  // it in the array pointed to by "compressed".
  //
  // "*compressed_length" is set to the length of the compressed output.
  //
  // Example:
  //    char* output = new char[snappy::MaxCompressedLength(input_length)];
  //    size_t output_length;
  //    RawCompress(input, input_length, output, &output_length);
  //    ... Process(output, output_length) ...
  //    delete [] output;
  SNAPPYLIB_API void RawCompress(const char* input, size_t input_length, char* compressed,
                   size_t* compressed_length);

  /**
   * 
   *  @rst 
   *  .. _RawCompress_with_options:
   *  @endrst
   *
   * Same as `RawCompress` above but takes additional CompressionOptions
   *
   *  |Parameters            |Direction|Description                                                          |
   *  |:---------------------|:-------:|:--------------------------------------------------------------------|
   *  | \b input             |  in     | This is the buffer where the data we want to compress is accessible.|
   *  | \b input_length      |  in     | Length of the input buffer.                                         |
   *  | \b compressed        |  out    | This is a buffer in which compressed data is stored.                |
   *  | \b compressed_length |  out    | The length of the data after compression is stored in this.         |
   */
  SNAPPYLIB_API void RawCompress(const char* input, size_t input_length, char* compressed,
                   size_t* compressed_length, CompressionOptions options);

  /**
   * @brief 
   *
   * Same as `RawCompress` above but taking an `iovec` array as input. Note that
   * `uncompressed_length` is the total number of bytes to be read from the
   * elements of `iov` (_not_ the number of elements in `iov`).
   *
   *  |Parameters              |Direction|Description                                                          |
   *  |:-----------------------|:-------:|:--------------------------------------------------------------------|
   *  | \b iov                 |  in     | Input iovec array.                                                  |
   *  | \b uncompressed_length |  in     | Total number of bytes to be read from the elements of `iov`.        |
   *  | \b compressed          |  out    | This is a buffer in which compressed data is stored.                |
   *  | \b compressed_length   |  out    | The length of the data after compression is stored in this.         |
   * 
   */
  // Same as `RawCompress` above but taking an `iovec` array as input. Note that
  // `uncompressed_length` is the total number of bytes to be read from the
  // elements of `iov` (_not_ the number of elements in `iov`).
  SNAPPYLIB_API void RawCompressFromIOVec(const struct iovec* iov, size_t uncompressed_length,
                            char* compressed, size_t* compressed_length);
                            
  /**
   * @brief 
   *
   * Same as `RawCompressFromIOVec` above but takes additional CompressionOptions.
   *
   *  |Parameters              |Direction|Description                                                          |
   *  |:-----------------------|:-------:|:--------------------------------------------------------------------|
   *  | \b iov                 |  in     | Input iovec array.                                                  |
   *  | \b uncompressed_length |  in     | Total number of bytes to be read from the elements of `iov`.        |
   *  | \b compressed          |  out    | This is a buffer in which compressed data is stored.                |
   *  | \b compressed_length   |  out    | The length of the data after compression is stored in this.         |
   *  | \b options             | in      | Compression options.                                                |
   */
  SNAPPYLIB_API void RawCompressFromIOVec(const struct iovec* iov, size_t uncompressed_length,
                            char* compressed, size_t* compressed_length,
                            CompressionOptions options);

  /**
   * @rst 
   * .. _RawUncompress:
   * @endrst
   *
   * @brief 
   * Given data in "compressed[0..compressed_length-1]" generated by
   * calling the Snappy::Compress routine, this routine
   * stores the uncompressed data to
   *    uncompressed[0..GetUncompressedLength(compressed)-1] .
   * 
   *  @note When decompressing data that was compressed using multi-threaded APIs, you should call 
   *  `GetUncompressedLengthFromMTCompressedBuffer` instead of `GetUncompressedLength` to correctly 
   *  allocate memory for the `uncompressed` buffer.
   *
   *  |Parameters            |Direction| Description                                      |
   *  |:---------------------|:-------:|:-------------------------------------------------|
   *  | \b compressed        |  in     | This is a buffer which contains compressed data. |
   *  | \b compressed_length |  in     | This is the length of the compressed buffer.     |
   *  | \b uncompressed      |  out    | Uncompressed data is stored in this buffer.      |
   * 
   *  @return
   *  |Result | Description                                                            |
   *  |:------|:-----------------------------------------------------------------------|
   *  |Success|Returns \b true if successful.                                         |
   *  |Failure|Returns \b false if the message is corrupted and could not be decrypted.|
   */
  // Given data in "compressed[0..compressed_length-1]" generated by
  // calling the Snappy::Compress routine, this routine
  // stores the uncompressed data to
  //    uncompressed[0..GetUncompressedLength(compressed)-1]
  // returns false if the message is corrupted and could not be decrypted
  SNAPPYLIB_API bool RawUncompress(const char* compressed, size_t compressed_length,
                     char* uncompressed);

  /**
   * @brief
   * Given data from the byte source 'compressed' generated by calling
   * the Snappy::Compress routine, this routine stores the uncompressed
   * data to
   *    uncompressed[0..GetUncompressedLength(compressed,compressed_length)-1] .
   *
   *  |Parameters       |Direction|Description                                                                                                                                                                                                         |
   *  |:----------------|:-------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   *  | \b compressed   | in,out  | A Source is an interface that yields a sequence of bytes, you can initialize it by calling snappy::ByteArraySource(inBuf,inBufLen); where inBuf is the pointer to original data and inBufLen is the size of inBuf .|
   *  | \b uncompressed | out     | Uncompressed data is stored in this buffer.                                                                                                                                                                        |
   *
   *  @return
   *  |Result | Description                                                             |
   *  |:------|:------------------------------------------------------------------------|
   *  |Success| Returns \b true if successful.                                          |
   *  |Failure| Returns \b false if the message is corrupted and could not be decrypted.|
   */
  // Given data from the byte source 'compressed' generated by calling
  // the Snappy::Compress routine, this routine stores the uncompressed
  // data to
  //    uncompressed[0..GetUncompressedLength(compressed,compressed_length)-1]
  // returns false if the message is corrupted and could not be decrypted
  SNAPPYLIB_API bool RawUncompress(Source* compressed, char* uncompressed);

  /**
   * @brief 
   * Given data in "compressed[0..compressed_length-1]" generated by
   * calling the Snappy::Compress routine, this routine
   * stores the uncompressed data to the iovec "iov". The number of physical
   * buffers in "iov" is given by iov_cnt and their cumulative size
   * must be at least GetUncompressedLength(compressed). The individual buffers
   * in "iov" must not overlap with each other.
   *
   *  |Parameters            |Direction|Description                                                                                                     |
   *  |:---------------------|:-------:|:---------------------------------------------------------------------------------------------------------------|
   *  | \b compressed        | in      | This is a buffer which contains compressed data.                                                               |
   *  | \b compressed_length | in      | This is the length of the compressed buffer.                                                                   |
   *  | \b iov               | in,out  | The struct iovec defines one vector element. Normally, this structure is used as an array of multiple elements. |
   *  | \b iov_cnt           | in,out  | This is the number of \a iovec structures in the array of \a iov .                                             |
   * 
   *  @return
   *  |Result | Description                                                             |
   *  |:------|:------------------------------------------------------------------------|
   *  |Success| Returns \b true if successful.                                          |
   *  |Failure| Returns \b false if the message is corrupted and could not be decrypted.|
   */
  // Given data in "compressed[0..compressed_length-1]" generated by
  // calling the Snappy::Compress routine, this routine
  // stores the uncompressed data to the iovec "iov". The number of physical
  // buffers in "iov" is given by iov_cnt and their cumulative size
  // must be at least GetUncompressedLength(compressed). The individual buffers
  // in "iov" must not overlap with each other.
  //
  // returns false if the message is corrupted and could not be decrypted
  SNAPPYLIB_API bool RawUncompressToIOVec(const char* compressed, size_t compressed_length,
                            const struct iovec* iov, size_t iov_cnt);

  /**
   * @brief 
   * Given data from the byte source 'compressed' generated by calling
   * the Snappy::Compress routine, this routine stores the uncompressed
   * data to the iovec "iov". The number of physical
   * buffers in "iov" is given by iov_cnt and their cumulative size
   * must be at least GetUncompressedLength(compressed). The individual buffers
   * in "iov" must not overlap with each other.
   *
   *  |Parameters     |Direction|Description                                                                                                                                                                                                          |
   *  |:--------------|:-------:|:------------------------------------------------------------------                                                                                                                                                  |
   *  | \b compressed | in,out  | A Source is an interface that yields a sequence of bytes, you can initialize it by calling snappy::ByteArraySource(inBuf,inBufLen); where inBuf is the pointer to original data and inBufLen is the size of inBuf . |
   *  | \b iov        | in,out  | The struct iovec defines one vector element. Normally, this structure is used as an array of multiple elements.                                                                                                     |
   *  | \b iov_cnt    | out     | This is the number of \a iovec structures in the array of \a iov .                                                                                                                                                  |
   * 
   *  @return
   *  |Result | Description                                                             |
   *  |:------|:------------------------------------------------------------------------|
   *  |Success| Returns \b true if successful.                                          |
   *  |Failure| Returns \b false if the message is corrupted and could not be decrypted.|
   */
  // Given data from the byte source 'compressed' generated by calling
  // the Snappy::Compress routine, this routine stores the uncompressed
  // data to the iovec "iov". The number of physical
  // buffers in "iov" is given by iov_cnt and their cumulative size
  // must be at least GetUncompressedLength(compressed). The individual buffers
  // in "iov" must not overlap with each other.
  //
  // returns false if the message is corrupted and could not be decrypted
  SNAPPYLIB_API bool RawUncompressToIOVec(Source* compressed, const struct iovec* iov,
                            size_t iov_cnt);

/**
 * @}
 */

/**
 * @name Helper Functions.
 * @{
 */
  /**
   * @brief This function determines the maximal size of the compressed representation of
   * input data that is "source_bytes" bytes in length.
   * 
   *  |Parameters       |Direction| Description                 |
   *  |:----------------|:-------:|:----------------------------|
   *  | \b source_bytes |   in    | The size of source in bytes.|
   *
   *  @return
   *  |Result | Description                                                                                                   |
   *  |:------|:--------------------------------------------------------------------------------------------------------------|
   *  |Success|Returns the maximal size of the compressed representation of input data that is "source_bytes" bytes in length.|
   */
  // Returns the maximal size of the compressed representation of
  // input data that is "source_bytes" bytes in length;
  SNAPPYLIB_API size_t MaxCompressedLength(size_t source_bytes);

  /**
   * @brief Get the Uncompressed Length object.
   * 
   * This operation takes O(1) time.
   * 
   * @attention REQUIRES: "compressed[]" was produced by RawCompress() or Compress().
   *
   *  |Parameters            |Direction| Description                                                                 |
   *  |:---------------------|:-------:|:----------------------------------------------------------------------------|
   *  | \b compressed        |  in     | This is a buffer which contains compressed data.                            |
   *  | \b compressed_length |  in     | This is the length of the compressed buffer.                                |
   *  | \b result            |  out    |  This is the pointer to type size_t where the uncompressed length is stored.|
   *
   *  @return
   *  |Result | Description                            |
   *  |:------|:---------------------------------------|
   *  |Success| Returns \b true on successful parsing. |
   *  |Failure| Returns \b false on parsing error.     |
   */
  // REQUIRES: "compressed[]" was produced by RawCompress() or Compress()
  // Returns true and stores the length of the uncompressed data in
  // *result normally.  Returns false on parsing error.
  // This operation takes O(1) time.
  SNAPPYLIB_API bool GetUncompressedLength(const char* compressed, size_t compressed_length,
                             size_t* result);

  /**
  * @brief Get the Uncompressed Length object from the AOCL multithreaded compressor's compressed buffer.
  * 
  * This operation takes O(1) time.
  * 
  * @attention REQUIRES: "compressed[]" was produced by RawCompress() or Compress() IN AOCL's MULTITHREADED MODE.
  *
  *  |Parameters            |Direction| Description                                                                 |
  *  |:---------------------|:-------:|:----------------------------------------------------------------------------|
  *  | \b compressed        |  in     | This is a buffer which contains compressed data. (along with the RAP frame) |
  *  | \b compressed_length |  in     | This is the length of the compressed buffer (including the RAP frame).      |
  *  | \b result            |  out    | This is the pointer to type size_t where the uncompressed length is stored. |
  *
  *  @return
  *  |Result | Description                            |
  *  |:------|:---------------------------------------|
  *  |Success| Returns \b true on successful parsing. |
  *  |Failure| Returns \b false on parsing error.     |
  */
  SNAPPYLIB_API bool GetUncompressedLengthFromMTCompressedBuffer(const char* compressed, size_t compressed_length,
                             size_t* result);

  /**
   * @brief 
   * Returns \b true iff the contents of "compressed[]" can be uncompressed
   * successfully.  Does not return the uncompressed data.  Takes
   * time proportional to compressed_length, but is usually at least
   * a factor of four faster than actual decompression.
   *
   *  |Parameters            |Direction|Description                                      |
   *  |:---------------------|:-------:|:------------------------------------------------|
   *  | \b compressed        |  in     | This is a buffer which contains compressed data.|
   *  | \b compressed_length |  in     | This is the length of the compressed buffer.    |
   *
   *  @return
   *  |Result | Description                                                                          |
   *  |:------|:-------------------------------------------------------------------------------------|
   *  |Success| Returns \b true iff the contents of "compressed[]" can be uncompressed successfully. |
   *  |Failure| Returns \b false if error.                                                           |
   */
  // Returns true iff the contents of "compressed[]" can be uncompressed
  // successfully.  Does not return the uncompressed data.  Takes
  // time proportional to compressed_length, but is usually at least
  // a factor of four faster than actual decompression.
  SNAPPYLIB_API bool IsValidCompressedBuffer(const char* compressed,
                               size_t compressed_length);

  /**
   * @brief
   * Returns \b true iff the contents of "compressed" can be uncompressed
   * successfully.  Does not return the uncompressed data.  Takes
   * time proportional to *compressed length, but is usually at least
   * a factor of four faster than actual decompression.
   * On success, consumes all of *compressed.  On failure, consumes an
   * unspecified prefix of *compressed.
   *
   *  |Parameters     |Direction|Description                                                                                                                                                                                                         |
   *  |:--------------|:-------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   *  | \b compressed | in,out  | A Source is an interface that yields a sequence of bytes, you can initialize it by calling snappy::ByteArraySource(inBuf,inBufLen); where inBuf is the pointer to original data and inBufLen is the size of inBuf .|
   *
   *  @return
   *  |Result | Description                                                                        |
   *  |:------|:-----------------------------------------------------------------------------------|
   *  |Success| Returns \b true iff the contents of "compressed" can be uncompressed successfully. |
   *  |Failure| Returns \b false if error.                                                         |
   */
  // Returns true iff the contents of "compressed" can be uncompressed
  // successfully.  Does not return the uncompressed data.  Takes
  // time proportional to *compressed length, but is usually at least
  // a factor of four faster than actual decompression.
  // On success, consumes all of *compressed.  On failure, consumes an
  // unspecified prefix of *compressed.
  SNAPPYLIB_API bool IsValidCompressed(Source* compressed);

  /**
   * @}
   */

  /// @cond DOXYGEN_SHOULD_SKIP_THIS

  // The size of a compression block. Note that many parts of the compression
  // code assumes that kBlockSize <= 65536; in particular, the hash table
  // can only store 16-bit offsets, and EmitCopy() also assumes the offset
  // is 65535 bytes or less. Note also that if you change this, it will
  // affect the framing format (see framing_format.txt).
  //
  // Note that there might be older data around that is compressed with larger
  // block sizes, so the decompression code should not rely on the
  // non-existence of long backreferences.
  static constexpr int kBlockLog = 16;
  static constexpr size_t kBlockSize = 1 << kBlockLog;

  static constexpr int kMinHashTableBits = 8;
  static constexpr size_t kMinHashTableSize = 1 << kMinHashTableBits;

  static constexpr int kMaxHashTableBits = 15;
  static constexpr size_t kMaxHashTableSize = 1 << kMaxHashTableBits;
#if defined(AOCL_SNAPPY_OPT) && !defined(AOCL_SNAPPY_HIGH_COMPRESSION)
  static constexpr int AOCL_kMaxHashTableBits = 14;
  static constexpr size_t AOCL_kMaxHashTableSize = 1 << AOCL_kMaxHashTableBits;
#endif

    /* AOCL-Compression defined setup function that configures with the right
*  AMD optimized snappy routines depending upon the detected CPU features. */

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
 * AMD optimized snappy routines depending upon the detected CPU features if `optOff=0`.
 * 
 * Except for the initial call, it's necessary to execute aocl_destroy_snappy() before any subsequent calls
 * to this function. Failure to call the destroy function prior to invoking this function will result
 * in snappy following the code path of set at first setup call or  the most recent setup call that was
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
 SNAPPYLIB_API char * aocl_setup_snappy(int optOff, int optLevel, size_t insize,
                           size_t level, size_t windowLog);

/**
 * @brief It is necessary to execute this destroy function after the initial invocation of the
 * aocl_setup_snappy() function, prior to initiating the setup function again.
 */
 SNAPPYLIB_API void aocl_destroy_snappy(void);

/**
 * @}
 */

 /**
 * This class is created to expose internal functions which are not available external to this method.
 *
 * The test cases written for API level testing needed these internal functions, but can't access them directly
 * so a separate class was needed for calling those internal functions.
 */
 class SNAPPYLIB_API SNAPPY_Gtest_Util
 {
 public:
     static Source* ByteArraySource_ext(const char* p, size_t n);
     static Sink* UncheckedByteArraySink_ext(char* dest);
     static void Append32(std::string* s, uint32_t value);
 };

 /// @endcond /* DOXYGEN_SHOULD_SKIP_THIS */

 /*! @} end doxygen SNAPPY_API*/
}  // end namespace snappy

#endif  // THIRD_PARTY_SNAPPY_SNAPPY_H__

/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Interface to compression/decompression functions          */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

   Copyright (c) 2009-2025 Ilya Grebnov <ilya.grebnov@gmail.com>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Please see the file LICENSE for full copyright information and file AUTHORS
for full list of contributors.

See also the bsc and libbsc web site:
  http://libbsc.com/ for more information.

--*/

#ifndef _LIBBSC_LIBBSC_H
#define _LIBBSC_LIBBSC_H

#define LIBBSC_VERSION_MAJOR           3
#define LIBBSC_VERSION_MINOR           3
#define LIBBSC_VERSION_PATCH           5
#define LIBBSC_VERSION_STRING          "3.3.5"

#define LIBBSC_NO_ERROR                0
#define LIBBSC_BAD_PARAMETER          -1
#define LIBBSC_NOT_ENOUGH_MEMORY      -2
#define LIBBSC_NOT_COMPRESSIBLE       -3
#define LIBBSC_NOT_SUPPORTED          -4
#define LIBBSC_UNEXPECTED_EOB         -5
#define LIBBSC_DATA_CORRUPT           -6

#define LIBBSC_GPU_ERROR              -7
#define LIBBSC_GPU_NOT_SUPPORTED      -8
#define LIBBSC_GPU_NOT_ENOUGH_MEMORY  -9

#define LIBBSC_BLOCKSORTER_NONE        0
#define LIBBSC_BLOCKSORTER_BWT         1

#ifdef LIBBSC_SORT_TRANSFORM_SUPPORT

  #define LIBBSC_BLOCKSORTER_ST3       3
  #define LIBBSC_BLOCKSORTER_ST4       4
  #define LIBBSC_BLOCKSORTER_ST5       5
  #define LIBBSC_BLOCKSORTER_ST6       6
  #define LIBBSC_BLOCKSORTER_ST7       7
  #define LIBBSC_BLOCKSORTER_ST8       8

#endif

#define LIBBSC_CODER_NONE              0
#define LIBBSC_CODER_QLFC_STATIC       1
#define LIBBSC_CODER_QLFC_ADAPTIVE     2
#define LIBBSC_CODER_QLFC_FAST         3

#define LIBBSC_FEATURE_NONE            0
#define LIBBSC_FEATURE_FASTMODE        1
#define LIBBSC_FEATURE_MULTITHREADING  2
#define LIBBSC_FEATURE_LARGEPAGES      4
#define LIBBSC_FEATURE_CUDA            8

#define LIBBSC_DEFAULT_LZPHASHSIZE     15
#define LIBBSC_DEFAULT_LZPMINLEN       128
#define LIBBSC_DEFAULT_BLOCKSORTER     LIBBSC_BLOCKSORTER_BWT
#define LIBBSC_DEFAULT_CODER           LIBBSC_CODER_QLFC_STATIC
#define LIBBSC_DEFAULT_FEATURES        LIBBSC_FEATURE_FASTMODE | LIBBSC_FEATURE_MULTITHREADING

#define LIBBSC_HEADER_SIZE             28

#ifdef __cplusplus
extern "C" {
#endif

    /**
    * You should call this function (or @ref bsc_init_full) before you call any of the other functions in libbsc.
    * @param features - the set of additional features.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_init(int features);

    /**
    * You should call this function (or @ref bsc_init) before you call any of the other functions in libbsc.
    * @param features - the set of additional features.
    * @param malloc - function to use to allocate buffers
    * @param zero_malloc - function to use to allocate zero-filled buffers
    * @param free - function used to free buffers
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_init_full(int features, void* (* malloc)(size_t size), void* (* zero_malloc)(size_t size), void (* free)(void* address));

    /**
    * Compress a memory block.
    * @param input                              - the input memory block of n bytes.
    * @param output                             - the output memory block of n + LIBBSC_HEADER_SIZE bytes.
    * @param n                                  - the length of the input memory block.
    * @param lzpHashSize                        - the hash table size if LZP enabled, 0 otherwise. Must be in range [0, 10..28].
    * @param lzpMinLen                          - the minimum match length if LZP enabled, 0 otherwise. Must be in range [0, 4..255].
    * @param blockSorter                        - the block sorting algorithm. Must be in range [ST3..ST8, BWT].
    * @param coder                              - the entropy coding algorithm. Must be in range [MTF or QLFC].
    * @param features                           - the set of additional features.
    * @return the length of compressed memory block if no error occurred, error code otherwise.
    */
    int bsc_compress(const unsigned char * input, unsigned char * output, int n, int lzpHashSize, int lzpMinLen, int blockSorter, int coder, int features);

    /**
    * Store a memory block.
    * @param input                              - the input memory block of n bytes.
    * @param output                             - the output memory block of n + LIBBSC_HEADER_SIZE bytes.
    * @param n                                  - the length of the input memory block.
    * @param features                           - the set of additional features.
    * @return the length of stored memory block if no error occurred, error code otherwise.
    */
    int bsc_store(const unsigned char * input, unsigned char * output, int n, int features);

    /**
    * Determinate the sizes of input and output memory blocks for bsc_decompress function.
    * @param blockHeader                        - the header of input(compressed) memory block of headerSize bytes.
    * @param headerSize                         - the length of header, should be at least LIBBSC_HEADER_SIZE bytes.
    * @param pBlockSize                         - the length of the input memory block for bsc_decompress function.
    * @param pDataSize                          - the length of the output memory block for bsc_decompress function.
    * @param features                           - the set of additional features.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_block_info(const unsigned char * blockHeader, int headerSize, int * pBlockSize, int * pDataSize, int features);

    /**
    * Decompress a memory block.
    * Note : You should call bsc_block_info function to determinate the sizes of input and output memory blocks.
    * @param input                              - the input memory block of inputSize bytes.
    * @param inputSize                          - the length of the input memory block.
    * @param output                             - the output memory block of outputSize bytes.
    * @param outputSize                         - the length of the output memory block.
    * @param features                           - the set of additional features.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_decompress(const unsigned char * input, int inputSize, unsigned char * output, int outputSize, int features);

#ifdef __cplusplus
}
#endif

#endif

/*-------------------------------------------------*/
/* End                                    libbsc.h */
/*-------------------------------------------------*/

/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Interface to data preprocessing filters                   */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

   Copyright (c) 2009-2021 Ilya Grebnov <ilya.grebnov@gmail.com>

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

#ifndef _LIBBSC_FILTERS_H
#define _LIBBSC_FILTERS_H

#define LIBBSC_CONTEXTS_FOLLOWING    1
#define LIBBSC_CONTEXTS_PRECEDING    2

#ifdef __cplusplus
extern "C" {
#endif

    /**
    * Autodetects segments for better compression of heterogeneous files.
    * @param input      - the input memory block of n bytes.
    * @param n          - the length of the input memory block.
    * @param segments   - the output array of segments of k elements size.
    * @param k          - the size of the output segments array.
    * @param features   - the set of additional features.
    * @return The number of segments if no error occurred, error code otherwise.
    */
    int bsc_detect_segments(const unsigned char * input, int n, int * segments, int k, int features);

    /**
    * Autodetects order of contexts for better compression of binary files.
    * @param input      - the input memory block of n bytes.
    * @param n          - the length of the input memory block.
    * @param features   - the set of additional features.
    * @return The detected contexts order if no error occurred, error code otherwise.
    */
    int bsc_detect_contextsorder(const unsigned char * input, int n, int features);

    /**
    * Reverses memory block to change order of contexts.
    * @param T          - the input/output memory block of n bytes.
    * @param n          - the length of the memory block.
    * @param features   - the set of additional features.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_reverse_block(unsigned char * T, int n, int features);

    /**
    * Autodetects record size for better compression of multimedia files.
    * @param input      - the input memory block of n bytes.
    * @param n          - the length of the input memory block.
    * @param features   - the set of additional features.
    * @return The size of record if no error occurred, error code otherwise.
    */
    int bsc_detect_recordsize(const unsigned char * input, int n, int features);

    /**
    * Reorders memory block for specific size of record (Forward transform).
    * @param T          - the input/output memory block of n bytes.
    * @param n          - the length of the memory block.
    * @param recordSize - the size of record.
    * @param features   - the set of additional features.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_reorder_forward(unsigned char * T, int n, int recordSize, int features);

    /**
    * Reorders memory block for specific size of record (Reverse transform).
    * @param T          - the input/output memory block of n bytes.
    * @param n          - the length of the memory block.
    * @param recordSize - the size of record.
    * @param features   - the set of additional features.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_reorder_reverse(unsigned char * T, int n, int recordSize, int features);

#ifdef __cplusplus
}
#endif

#endif

/*-------------------------------------------------*/
/* End                                   filters.h */
/*-------------------------------------------------*/

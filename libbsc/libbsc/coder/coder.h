/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Interface to second stage encoding functions              */
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

#ifndef _LIBBSC_CODER_H
#define _LIBBSC_CODER_H

#ifdef __cplusplus
extern "C" {
#endif

    /**
    * You should call this function before you call any of the other functions in coder.
    * @param features   - the set of additional features.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_coder_init(int features);

    /**
    * Compress a memory block using Quantized Local Frequency Coding.
    * @param input      - the input memory block of n bytes.
    * @param output     - the output memory block of n bytes.
    * @param n          - the length of the input memory block.
    * @param coder      - the entropy coding algorithm.
    * @param features   - the set of additional features.
    * @return the length of compressed memory block if no error occurred, error code otherwise.
    */
    int bsc_coder_compress(const unsigned char * input, unsigned char * output, int n, int coder, int features);

    /**
    * Decompress a memory block using Quantized Local Frequency Coding.
    * @param input      - the input memory block.
    * @param output     - the output memory block.
    * @param coder      - the entropy coding algorithm.
    * @param features   - the set of additional features.
    * @return the length of decompressed memory block if no error occurred, error code otherwise.
    */
    int bsc_coder_decompress(const unsigned char * input, unsigned char * output, int coder, int features);

#ifdef __cplusplus
}
#endif

#endif

/*-----------------------------------------------------------*/
/* End                                               coder.h */
/*-----------------------------------------------------------*/

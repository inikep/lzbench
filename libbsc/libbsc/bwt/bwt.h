/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Interface to Burrows Wheeler Transform                    */
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

#ifndef _LIBBSC_BWT_H
#define _LIBBSC_BWT_H

#ifdef __cplusplus
extern "C" {
#endif

    /**
    * Constructs the burrows wheeler transformed string of a given string.
    * @param T              - the input/output string of n chars.
    * @param n              - the length of the given string.
    * @param num_indexes    - the length of secondary indexes array, can be NULL.
    * @param indexes        - the secondary indexes array, can be NULL.
    * @param features       - the set of additional features.
    * @return the primary index if no error occurred, error code otherwise.
    */
    int bsc_bwt_encode(unsigned char * T, int n, unsigned char * num_indexes, int * indexes, int features);

    /**
    * Reconstructs the original string from burrows wheeler transformed string.
    * @param T              - the input/output string of n chars.
    * @param n              - the length of the given string.
    * @param index          - the primary index.
    * @param num_indexes    - the length of secondary indexes array, can be 0.
    * @param indexes        - the secondary indexes array, can be NULL.
    * @param features       - the set of additional features.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_bwt_decode(unsigned char * T, int n, int index, unsigned char num_indexes, int * indexes, int features);

#ifdef __cplusplus
}
#endif

#endif

/*-----------------------------------------------------------*/
/* End                                                 bwt.h */
/*-----------------------------------------------------------*/

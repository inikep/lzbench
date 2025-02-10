/*--

This file is a part of libcubwt, a library for CUDA accelerated
burrows wheeler transform construction and inversion.

   Copyright (c) 2022-2025 Ilya Grebnov <ilya.grebnov@gmail.com>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Please see the file LICENSE for full copyright and license details.

--*/

#ifndef LIBCUBWT_CUH
#define LIBCUBWT_CUH 1

#define LIBCUBWT_VERSION_MAJOR          1
#define LIBCUBWT_VERSION_MINOR          6
#define LIBCUBWT_VERSION_PATCH          1
#define LIBCUBWT_VERSION_STRING	        "1.6.1"

#define LIBCUBWT_NO_ERROR               0
#define LIBCUBWT_BAD_PARAMETER          -1
#define LIBCUBWT_NOT_ENOUGH_MEMORY      -2

#define LIBCUBWT_GPU_ERROR              -7
#define LIBCUBWT_GPU_NOT_SUPPORTED      -8
#define LIBCUBWT_GPU_NOT_ENOUGH_MEMORY  -9

#ifdef __cplusplus
extern "C" {
#endif

    #include <stdint.h>

    /**
    * Allocates storage on the CUDA device that allows reusing allocated memory with each libcubwt operation.
    * @param device_storage A reference to the memory pointer where the allocated device storage will be saved.
    * @param max_length This parameter controls the amount of allocated memory, ensuring that libcubwt operations
    *        can accommodate strings of lengths up to this value for both forward and reverse Burrows-Wheeler Transforms.
    *        The method currently allocates approximately 20.5 times the string length, which is the necessary amount of memory
    *        for the forward Burrows-Wheeler Transform of a string at maximum length. This allocation is also sufficient and
    *        optimal for the reverse Burrows-Wheeler Transform. However, if performance is less critical, or if device memory is limited,
    *        'max_length' can be lowered. Allocating memory at approximately 6.8 times the string length should still yield about 90%
    *        of the optimal performance for the reverse Burrows-Wheeler Transform. This effectively means that reverse
    *        Burrows-Wheeler Transform operations can be performed with storage allocated using a 'max_length' parameter
    *        at a third of the input string's maximum length.
    * @return LIBCUBWT_NO_ERROR if no error occurred, libcubwt error code otherwise.
    */
    int64_t libcubwt_allocate_device_storage(void ** device_storage, int64_t max_length);

    /**
    * Destroys the previously allocated storage on the CUDA device.
    * @param device_storage The previously allocated storage on the CUDA device.
    * @return LIBCUBWT_NO_ERROR if no error occurred, libcubwt error code otherwise.
    */
    int64_t libcubwt_free_device_storage(void * device_storage);

    /**
    * Constructs the Burrows-Wheeler Transform (BWT) of a given string.
    * @param device_storage The previously allocated storage on the CUDA device.
    * @param T [0..n-1] The input string.
    * @param L [0..n-1] The output string (can be T).
    * @param n The length of the input string.
    * @return The primary index if no error occurred, libcubwt error code otherwise.
    */
    int64_t libcubwt_bwt(void * device_storage, const uint8_t * T, uint8_t * L, int64_t n);

    /**
    * Constructs the Burrows-Wheeler Transform (BWT) of a given string with auxiliary indexes.
    * @param device_storage The previously allocated storage on the CUDA device.
    * @param T [0..n-1] The input string.
    * @param L [0..n-1] The output string (can be T).
    * @param n The length of the input string.
    * @param r The sampling rate for auxiliary indexes (must be power of 2).
    * @param I [0..(n-1)/r] The output auxiliary indexes.
    * @return LIBCUBWT_NO_ERROR if no error occurred, libcubwt error code otherwise.
    */
    int64_t libcubwt_bwt_aux(void * device_storage, const uint8_t * T, uint8_t * L, int64_t n, int64_t r, uint32_t * I);

    /**
    * Reconstructs the original string from a given burrows-wheeler transformed string (BWT) with primary index.
    * @param device_storage The previously allocated storage on the CUDA device.
    * @param T [0..n-1] The input string.
    * @param U [0..n-1] The output string (can be T).
    * @param n The length of the given string.
    * @param freq [0..255] The input symbol frequency table (can be NULL).
    * @param i The primary index.
    * @return LIBCUBWT_NO_ERROR if no error occurred, libcubwt error code otherwise.
    */
    int64_t libcubwt_unbwt(void * device_storage, const uint8_t * T, uint8_t * U, int64_t n, const int32_t * freq, int32_t i);

#ifdef __cplusplus
}
#endif

#endif

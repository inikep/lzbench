/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Interface to platform specific functions and constants    */
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

#ifndef _LIBBSC_PLATFORM_H
#define _LIBBSC_PLATFORM_H

#define ALPHABET_SIZE (256)

#define LIBBSC_CPU_FEATURE_NONE      0
#define LIBBSC_CPU_FEATURE_A64       1
#define LIBBSC_CPU_FEATURE_SSE2      2
#define LIBBSC_CPU_FEATURE_SSE3      3
#define LIBBSC_CPU_FEATURE_SSSE3     4
#define LIBBSC_CPU_FEATURE_SSE41     5
#define LIBBSC_CPU_FEATURE_SSE42     6
#define LIBBSC_CPU_FEATURE_AVX       7
#define LIBBSC_CPU_FEATURE_AVX2      8
#define LIBBSC_CPU_FEATURE_AVX512F   9
#define LIBBSC_CPU_FEATURE_AVX512BW  10

#if (defined(_M_AMD64) || defined(_M_X64) || defined(__amd64)) && !defined(__x86_64__)
    #define __x86_64__ 1
#endif

#if defined(_M_ARM64) && !defined(__aarch64__)
    #define __aarch64__ 1
#endif

#ifndef LIBBSC_CPU_FEATURE
    #if defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
        #define LIBBSC_CPU_FEATURE LIBBSC_CPU_FEATURE_AVX512BW
    #elif defined(__AVX512F__) || defined(__AVX512__)
        #define LIBBSC_CPU_FEATURE LIBBSC_CPU_FEATURE_AVX512F
    #elif defined(__AVX2__)
        #define LIBBSC_CPU_FEATURE LIBBSC_CPU_FEATURE_AVX2
    #elif defined(__AVX__)
        #define LIBBSC_CPU_FEATURE LIBBSC_CPU_FEATURE_AVX
    #elif defined(__SSE4_2__)
        #define LIBBSC_CPU_FEATURE LIBBSC_CPU_FEATURE_SSE42
    #elif defined(__SSE4_1__)
        #define LIBBSC_CPU_FEATURE LIBBSC_CPU_FEATURE_SSE41
    #elif defined(__SSSE3__)
        #define LIBBSC_CPU_FEATURE LIBBSC_CPU_FEATURE_SSSE3
    #elif defined(__SSE3__)
        #define LIBBSC_CPU_FEATURE LIBBSC_CPU_FEATURE_SSE3
    #elif defined(__SSE2__) || defined(__x86_64__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
        #define LIBBSC_CPU_FEATURE LIBBSC_CPU_FEATURE_SSE2
    #elif defined(__aarch64__)
        #define LIBBSC_CPU_FEATURE LIBBSC_CPU_FEATURE_A64
    #else
        #define LIBBSC_CPU_FEATURE LIBBSC_CPU_FEATURE_NONE
    #endif
#endif

#if defined(_OPENMP) && defined(LIBBSC_OPENMP_SUPPORT)
    #include <omp.h>
    #define LIBBSC_OPENMP
#endif

#if LIBBSC_CPU_FEATURE >= LIBBSC_CPU_FEATURE_SSE2
    #if defined(_MSC_VER)
        #include <intrin.h>
    #elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
        #include <x86intrin.h>
    #endif
#elif LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_A64
    #include <arm_neon.h>
#endif

#if defined(__GNUC__)
    #define INLINE __inline__
#elif defined(_MSC_VER)
    #define INLINE __forceinline
#elif defined(__IBMC__)
    #define INLINE _Inline
#elif defined(__cplusplus)
    #define INLINE inline
#else
    #define INLINE /* */
#endif

#if defined(_MSC_VER)
    #define NOINLINE __declspec(noinline)
#elif defined(__GNUC__)
    #define NOINLINE __attribute__ ((noinline))
#else
    #define NOINLINE /* */
#endif

#if defined(_MSC_VER)
    #define ALIGNED(x) __declspec(align(x))
#elif defined(__GNUC__)
    #define ALIGNED(x) __attribute__ ((aligned(x)))
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__)
    #define RESTRICT __restrict__
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
    #define RESTRICT __restrict
#else
    #define RESTRICT /* */
#endif

#if defined(__GNUC__) || defined(__clang__)
    #define bsc_byteswap_uint64(x)    (__builtin_bswap64(x))
    #define bsc_bit_scan_reverse(x)   (__builtin_clz(x) ^ 31)
    #define bsc_bit_scan_reverse64(x) (__builtin_clzll(x) ^ 63)
    #define bsc_bit_scan_forward(x)   (__builtin_ctz(x))
    #define bsc_bit_scan_forward64(x) (__builtin_ctzll(x))
#elif defined(_MSC_VER)
    #define bsc_byteswap_uint64(x)  (_byteswap_uint64(x))

    #pragma intrinsic(_BitScanReverse)
    #pragma intrinsic(_BitScanForward)

    static inline __forceinline unsigned long bsc_bit_scan_reverse(unsigned long x) 
    {
       unsigned long index;
       _BitScanReverse(&index, x);
       return index;
    }

    static inline __forceinline unsigned long bsc_bit_scan_forward(unsigned long x) 
    {
       unsigned long index;
       _BitScanForward(&index, x);
       return index;
    }

    #if defined(__x86_64__) || defined(__aarch64__)
    static inline __forceinline unsigned long bsc_bit_scan_reverse64(unsigned long long x) 
    {
       unsigned long index;
        _BitScanReverse64(&index, x);
       return index;
    }
    #endif

    #if defined(__x86_64__) || defined(__aarch64__)
    static inline __forceinline unsigned long bsc_bit_scan_forward64(unsigned long long x) 
    {
       unsigned long index;
        _BitScanForward64(&index, x);
       return index;
    }
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
    * You should call this function before you call any of the other platform specific functions.
    * @param malloc - function to use to allocate buffers
    * @param zero_malloc - function to use to allocate zero-filled buffers
    * @param free - function used to free buffers
    * @param features   - the set of additional features.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_platform_init(int features, void* (* malloc)(size_t size), void* (* zero_malloc)(size_t size), void (* free)(void* address));

    /**
    * Allocates memory blocks.
    * @param size       - bytes to allocate.
    * @return a pointer to allocated space or NULL if there is insufficient memory available.
    */
    void * bsc_malloc(size_t size);

    /**
    * Allocates memory blocks and initializes all its bits to zero.
    * @param size       - bytes to allocate.
    * @return a pointer to allocated space or NULL if there is insufficient memory available.
    */
    void * bsc_zero_malloc(size_t size);

    /**
    * Deallocates or frees a memory block.
    * @param address    - previously allocated memory block to be freed.
    */
    void bsc_free(void * address);

    /**
    * Detects supported CPU features (Streaming SIMD Extensions).
    * @return highest supported CPU feature.
    */
    int bsc_get_cpu_features(void);

#ifdef __cplusplus
}
#endif

#endif

/*-----------------------------------------------------------*/
/* End                                            platform.h */
/*-----------------------------------------------------------*/

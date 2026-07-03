/*--

This file is a part of libsais, a library for linear time suffix array,
longest common prefix array and burrows wheeler transform construction.

   Copyright (c) 2021-2024 Ilya Grebnov <ilya.grebnov@gmail.com>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Please see the file LICENSE for full copyright information.

--*/

/**
 * Modifications Copyright (C) 2024-2026, Advanced Micro Devices. All rights reserved.
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

#include "libsais.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#if defined(LIBSAIS_OPENMP)
    #include <omp.h>
#else
    #define UNUSED(_x)                  (void)(_x)
#endif

typedef int32_t                         sa_sint_t;
typedef uint32_t                        sa_uint_t;
typedef ptrdiff_t                       fast_sint_t;
typedef size_t                          fast_uint_t;

#define SAINT_BIT                       (32)
#define SAINT_MAX                       INT32_MAX
#define SAINT_MIN                       INT32_MIN

#define ALPHABET_SIZE                   (1 << CHAR_BIT)
#define UNBWT_FASTBITS                  (17)

#define SUFFIX_GROUP_BIT                (SAINT_BIT - 1)
#define SUFFIX_GROUP_MARKER             (((sa_sint_t)1) << (SUFFIX_GROUP_BIT - 1))

#define BUCKETS_INDEX2(_c, _s)          (((_c) << 1) + (_s))

#define LIBSAIS_LOCAL_BUFFER_SIZE       (1024)
#define LIBSAIS_PER_THREAD_CACHE_SIZE   (24576)

#define AOCL_BWT
#ifdef AOCL_BWT
#define MOD(i, n) ((i+n)%n)
#endif

typedef struct LIBSAIS_THREAD_CACHE
{
        sa_sint_t                       symbol;
        sa_sint_t                       index;
} LIBSAIS_THREAD_CACHE;

typedef union LIBSAIS_THREAD_STATE
{
    struct
    {
        fast_sint_t                     position;
        fast_sint_t                     count;

        fast_sint_t                     m;
        fast_sint_t                     last_lms_suffix;

        sa_sint_t *                     buckets;
        LIBSAIS_THREAD_CACHE *          cache;
    } state;

    uint8_t padding[64];
} LIBSAIS_THREAD_STATE;

typedef struct LIBSAIS_CONTEXT
{
    sa_sint_t *                         buckets;
    LIBSAIS_THREAD_STATE *              thread_state;
    fast_sint_t                         threads;
} LIBSAIS_CONTEXT;

typedef struct LIBSAIS_UNBWT_CONTEXT
{
    sa_uint_t *                         bucket2;
    uint16_t *                          fastbits;
    sa_uint_t *                         buckets;
    fast_sint_t                         threads;
} LIBSAIS_UNBWT_CONTEXT;

#if defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
    #define RESTRICT __restrict
#else
    #error Your compiler, configuration or platform is not supported.
#endif

#if defined(__has_builtin)
    #if __has_builtin(__builtin_prefetch)
        #define HAS_BUILTIN_PREFETCH
    #endif
#elif defined(__GNUC__) && (((__GNUC__ == 3) && (__GNUC_MINOR__ >= 2)) || (__GNUC__ >= 4))
    #define HAS_BUILTIN_PREFETCH
#endif

#if defined(__has_builtin)
    #if __has_builtin(__builtin_bswap16)
        #define HAS_BUILTIN_BSWAP16
    #endif
#elif defined(__GNUC__) && (((__GNUC__ == 4) && (__GNUC_MINOR__ >= 8)) || (__GNUC__ >= 5))
    #define HAS_BUILTIN_BSWAP16
#endif

#if defined(HAS_BUILTIN_PREFETCH)
    #define libsais_prefetchr(address) __builtin_prefetch((const void *)(address), 0, 3)
    #define libsais_prefetchw(address) __builtin_prefetch((const void *)(address), 1, 3)
#elif defined (_M_IX86) || defined (_M_AMD64)
    #include <intrin.h>
    #define libsais_prefetchr(address) _mm_prefetch((const void *)(address), _MM_HINT_T0)
    #define libsais_prefetchw(address) _m_prefetchw((const void *)(address))
#elif defined (_M_ARM)
    #include <intrin.h>
    #define libsais_prefetchr(address) __prefetch((const void *)(address))
    #define libsais_prefetchw(address) __prefetchw((const void *)(address))
#elif defined (_M_ARM64)
    #include <intrin.h>
    #define libsais_prefetchr(address) __prefetch2((const void *)(address), 0)
    #define libsais_prefetchw(address) __prefetch2((const void *)(address), 16)
#else
    #error Your compiler, configuration or platform is not supported.
#endif

#if !defined(__LITTLE_ENDIAN__) && !defined(__BIG_ENDIAN__)
    #if defined(_LITTLE_ENDIAN) \
            || (defined(BYTE_ORDER) && defined(LITTLE_ENDIAN) && BYTE_ORDER == LITTLE_ENDIAN) \
            || (defined(_BYTE_ORDER) && defined(_LITTLE_ENDIAN) && _BYTE_ORDER == _LITTLE_ENDIAN) \
            || (defined(__BYTE_ORDER) && defined(__LITTLE_ENDIAN) && __BYTE_ORDER == __LITTLE_ENDIAN) \
            || (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
        #define __LITTLE_ENDIAN__
    #elif defined(_BIG_ENDIAN) \
            || (defined(BYTE_ORDER) && defined(BIG_ENDIAN) && BYTE_ORDER == BIG_ENDIAN) \
            || (defined(_BYTE_ORDER) && defined(_BIG_ENDIAN) && _BYTE_ORDER == _BIG_ENDIAN) \
            || (defined(__BYTE_ORDER) && defined(__BIG_ENDIAN) && __BYTE_ORDER == __BIG_ENDIAN) \
            || (defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
        #define __BIG_ENDIAN__
    #elif defined(_WIN32)
        #define __LITTLE_ENDIAN__
    #endif
#endif

#if defined(__LITTLE_ENDIAN__) && !defined(__BIG_ENDIAN__)
    #if defined(HAS_BUILTIN_BSWAP16)
        #define libsais_bswap16(x) (__builtin_bswap16(x))
    #elif defined(_MSC_VER) && !defined(__INTEL_COMPILER)
        #define libsais_bswap16(x) (_byteswap_ushort(x))
    #else
        #define libsais_bswap16(x) ((uint16_t)(x >> 8) | (uint16_t)(x << 8))
    #endif
#elif !defined(__LITTLE_ENDIAN__) && defined(__BIG_ENDIAN__)
    #define libsais_bswap16(x) (x)
#else
    #error Your compiler, configuration or platform is not supported.
#endif

static void * libsais_align_up(const void * address, size_t alignment)
{
    return (void *)((((ptrdiff_t)address) + ((ptrdiff_t)alignment) - 1) & (-((ptrdiff_t)alignment)));
}

static void * libsais_alloc_aligned(size_t size, size_t alignment)
{
    void * address = malloc(size + sizeof(short) + alignment - 1);
    if (address != NULL)
    {
        void * aligned_address = libsais_align_up((void *)((ptrdiff_t)address + (ptrdiff_t)(sizeof(short))), alignment);
        ((short *)aligned_address)[-1] = (short)((ptrdiff_t)aligned_address - (ptrdiff_t)address);

        return aligned_address;
    }

    return NULL;
}

static void libsais_free_aligned(void * aligned_address)
{
    if (aligned_address != NULL)
    {
        free((void *)((ptrdiff_t)aligned_address - ((short *)aligned_address)[-1]));
    }
}

static LIBSAIS_THREAD_STATE * libsais_alloc_thread_state(sa_sint_t threads)
{
    LIBSAIS_THREAD_STATE *  RESTRICT thread_state    = (LIBSAIS_THREAD_STATE *)libsais_alloc_aligned((size_t)threads * sizeof(LIBSAIS_THREAD_STATE), 4096);
    sa_sint_t *             RESTRICT thread_buckets  = (sa_sint_t *)libsais_alloc_aligned((size_t)threads * 4 * ALPHABET_SIZE * sizeof(sa_sint_t), 4096);
    LIBSAIS_THREAD_CACHE *  RESTRICT thread_cache    = (LIBSAIS_THREAD_CACHE *)libsais_alloc_aligned((size_t)threads * LIBSAIS_PER_THREAD_CACHE_SIZE * sizeof(LIBSAIS_THREAD_CACHE), 4096);

    if (thread_state != NULL && thread_buckets != NULL && thread_cache != NULL)
    {
        fast_sint_t t;
        for (t = 0; t < threads; ++t)
        { 
            thread_state[t].state.buckets   = thread_buckets;   thread_buckets  += 4 * ALPHABET_SIZE;
            thread_state[t].state.cache     = thread_cache;     thread_cache    += LIBSAIS_PER_THREAD_CACHE_SIZE;
        }

        return thread_state;
    }

    libsais_free_aligned(thread_cache);
    libsais_free_aligned(thread_buckets);
    libsais_free_aligned(thread_state);
    return NULL;
}

static void libsais_free_thread_state(LIBSAIS_THREAD_STATE * thread_state)
{
    if (thread_state != NULL)
    {
        libsais_free_aligned(thread_state[0].state.cache);
        libsais_free_aligned(thread_state[0].state.buckets);
        libsais_free_aligned(thread_state);
    }
}

static LIBSAIS_CONTEXT * libsais_create_ctx_main(sa_sint_t threads)
{
    LIBSAIS_CONTEXT *       RESTRICT ctx            = (LIBSAIS_CONTEXT *)libsais_alloc_aligned(sizeof(LIBSAIS_CONTEXT), 64);
    sa_sint_t *             RESTRICT buckets        = (sa_sint_t *)libsais_alloc_aligned((size_t)8 * ALPHABET_SIZE * sizeof(sa_sint_t), 4096);
    LIBSAIS_THREAD_STATE *  RESTRICT thread_state   = threads > 1 ? libsais_alloc_thread_state(threads) : NULL;

    if (ctx != NULL && buckets != NULL && (thread_state != NULL || threads == 1))
    {
        ctx->buckets = buckets;
        ctx->threads = threads;
        ctx->thread_state = thread_state;

        return ctx;
    }

    libsais_free_thread_state(thread_state);
    libsais_free_aligned(buckets);
    libsais_free_aligned(ctx);
    return NULL;
}

static void libsais_free_ctx_main(LIBSAIS_CONTEXT * ctx)
{
    if (ctx != NULL)
    {
        libsais_free_thread_state(ctx->thread_state);
        libsais_free_aligned(ctx->buckets);
        libsais_free_aligned(ctx);
    }
}

#if defined(LIBSAIS_OPENMP)

static sa_sint_t libsais_count_negative_marked_suffixes(sa_sint_t * RESTRICT SA, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    sa_sint_t count = 0;

    fast_sint_t i; for (i = omp_block_start; i < omp_block_start + omp_block_size; ++i) { count += (SA[i] < 0); }

    return count;
}

static sa_sint_t libsais_count_zero_marked_suffixes(sa_sint_t * RESTRICT SA, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    sa_sint_t count = 0;

    fast_sint_t i; for (i = omp_block_start; i < omp_block_start + omp_block_size; ++i) { count += (SA[i] == 0); }

    return count;
}

static void libsais_place_cached_suffixes(sa_sint_t * RESTRICT SA, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 3; i < j; i += 4)
    {
        libsais_prefetchr(&cache[i + 2 * prefetch_distance]);

        libsais_prefetchw(&SA[cache[i + prefetch_distance + 0].symbol]);
        libsais_prefetchw(&SA[cache[i + prefetch_distance + 1].symbol]);
        libsais_prefetchw(&SA[cache[i + prefetch_distance + 2].symbol]);
        libsais_prefetchw(&SA[cache[i + prefetch_distance + 3].symbol]);

        SA[cache[i + 0].symbol] = cache[i + 0].index;
        SA[cache[i + 1].symbol] = cache[i + 1].index;
        SA[cache[i + 2].symbol] = cache[i + 2].index;
        SA[cache[i + 3].symbol] = cache[i + 3].index;
    }

    for (j += prefetch_distance + 3; i < j; i += 1)
    {
        SA[cache[i].symbol] = cache[i].index;
    }
}

static void libsais_compact_and_place_cached_suffixes(sa_sint_t * RESTRICT SA, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j, l;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - 3, l = omp_block_start; i < j; i += 4)
    {
        libsais_prefetchw(&cache[i + prefetch_distance]);

        cache[l] = cache[i + 0]; l += cache[l].symbol >= 0;
        cache[l] = cache[i + 1]; l += cache[l].symbol >= 0;
        cache[l] = cache[i + 2]; l += cache[l].symbol >= 0;
        cache[l] = cache[i + 3]; l += cache[l].symbol >= 0;
    }

    for (j += 3; i < j; i += 1)
    {
        cache[l] = cache[i]; l += cache[l].symbol >= 0;
    }

    libsais_place_cached_suffixes(SA, cache, omp_block_start, l - omp_block_start);
}

static void libsais_accumulate_counts_s32_2(sa_sint_t * RESTRICT bucket00, fast_sint_t bucket_size, fast_sint_t bucket_stride)
{
    sa_sint_t * RESTRICT bucket01 = bucket00 - bucket_stride;
    fast_sint_t s; for (s = 0; s < bucket_size; s += 1) { bucket00[s] = bucket00[s] + bucket01[s]; }
}

static void libsais_accumulate_counts_s32_3(sa_sint_t * RESTRICT bucket00, fast_sint_t bucket_size, fast_sint_t bucket_stride)
{
    sa_sint_t * RESTRICT bucket01 = bucket00 - bucket_stride;
    sa_sint_t * RESTRICT bucket02 = bucket01 - bucket_stride;
    fast_sint_t s; for (s = 0; s < bucket_size; s += 1) { bucket00[s] = bucket00[s] + bucket01[s] + bucket02[s]; }
}

static void libsais_accumulate_counts_s32_4(sa_sint_t * RESTRICT bucket00, fast_sint_t bucket_size, fast_sint_t bucket_stride)
{
    sa_sint_t * RESTRICT bucket01 = bucket00 - bucket_stride;
    sa_sint_t * RESTRICT bucket02 = bucket01 - bucket_stride;
    sa_sint_t * RESTRICT bucket03 = bucket02 - bucket_stride;
    fast_sint_t s; for (s = 0; s < bucket_size; s += 1) { bucket00[s] = bucket00[s] + bucket01[s] + bucket02[s] + bucket03[s]; }
}

static void libsais_accumulate_counts_s32_5(sa_sint_t * RESTRICT bucket00, fast_sint_t bucket_size, fast_sint_t bucket_stride)
{
    sa_sint_t * RESTRICT bucket01 = bucket00 - bucket_stride;
    sa_sint_t * RESTRICT bucket02 = bucket01 - bucket_stride;
    sa_sint_t * RESTRICT bucket03 = bucket02 - bucket_stride;
    sa_sint_t * RESTRICT bucket04 = bucket03 - bucket_stride;
    fast_sint_t s; for (s = 0; s < bucket_size; s += 1) { bucket00[s] = bucket00[s] + bucket01[s] + bucket02[s] + bucket03[s] + bucket04[s]; }
}

static void libsais_accumulate_counts_s32_6(sa_sint_t * RESTRICT bucket00, fast_sint_t bucket_size, fast_sint_t bucket_stride)
{
    sa_sint_t * RESTRICT bucket01 = bucket00 - bucket_stride;
    sa_sint_t * RESTRICT bucket02 = bucket01 - bucket_stride;
    sa_sint_t * RESTRICT bucket03 = bucket02 - bucket_stride;
    sa_sint_t * RESTRICT bucket04 = bucket03 - bucket_stride;
    sa_sint_t * RESTRICT bucket05 = bucket04 - bucket_stride;
    fast_sint_t s; for (s = 0; s < bucket_size; s += 1) { bucket00[s] = bucket00[s] + bucket01[s] + bucket02[s] + bucket03[s] + bucket04[s] + bucket05[s]; }
}

static void libsais_accumulate_counts_s32_7(sa_sint_t * RESTRICT bucket00, fast_sint_t bucket_size, fast_sint_t bucket_stride)
{
    sa_sint_t * RESTRICT bucket01 = bucket00 - bucket_stride;
    sa_sint_t * RESTRICT bucket02 = bucket01 - bucket_stride;
    sa_sint_t * RESTRICT bucket03 = bucket02 - bucket_stride;
    sa_sint_t * RESTRICT bucket04 = bucket03 - bucket_stride;
    sa_sint_t * RESTRICT bucket05 = bucket04 - bucket_stride;
    sa_sint_t * RESTRICT bucket06 = bucket05 - bucket_stride;
    fast_sint_t s; for (s = 0; s < bucket_size; s += 1) { bucket00[s] = bucket00[s] + bucket01[s] + bucket02[s] + bucket03[s] + bucket04[s] + bucket05[s] + bucket06[s]; }
}

static void libsais_accumulate_counts_s32_8(sa_sint_t * RESTRICT bucket00, fast_sint_t bucket_size, fast_sint_t bucket_stride)
{
    sa_sint_t * RESTRICT bucket01 = bucket00 - bucket_stride;
    sa_sint_t * RESTRICT bucket02 = bucket01 - bucket_stride;
    sa_sint_t * RESTRICT bucket03 = bucket02 - bucket_stride;
    sa_sint_t * RESTRICT bucket04 = bucket03 - bucket_stride;
    sa_sint_t * RESTRICT bucket05 = bucket04 - bucket_stride;
    sa_sint_t * RESTRICT bucket06 = bucket05 - bucket_stride;
    sa_sint_t * RESTRICT bucket07 = bucket06 - bucket_stride;
    fast_sint_t s; for (s = 0; s < bucket_size; s += 1) { bucket00[s] = bucket00[s] + bucket01[s] + bucket02[s] + bucket03[s] + bucket04[s] + bucket05[s] + bucket06[s] + bucket07[s]; }
}

static void libsais_accumulate_counts_s32_9(sa_sint_t * RESTRICT bucket00, fast_sint_t bucket_size, fast_sint_t bucket_stride)
{
    sa_sint_t * RESTRICT bucket01 = bucket00 - bucket_stride;
    sa_sint_t * RESTRICT bucket02 = bucket01 - bucket_stride;
    sa_sint_t * RESTRICT bucket03 = bucket02 - bucket_stride;
    sa_sint_t * RESTRICT bucket04 = bucket03 - bucket_stride;
    sa_sint_t * RESTRICT bucket05 = bucket04 - bucket_stride;
    sa_sint_t * RESTRICT bucket06 = bucket05 - bucket_stride;
    sa_sint_t * RESTRICT bucket07 = bucket06 - bucket_stride;
    sa_sint_t * RESTRICT bucket08 = bucket07 - bucket_stride;
    fast_sint_t s; for (s = 0; s < bucket_size; s += 1) { bucket00[s] = bucket00[s] + bucket01[s] + bucket02[s] + bucket03[s] + bucket04[s] + bucket05[s] + bucket06[s] + bucket07[s] + bucket08[s]; }
}

static void libsais_accumulate_counts_s32(sa_sint_t * RESTRICT buckets, fast_sint_t bucket_size, fast_sint_t bucket_stride, fast_sint_t num_buckets)
{
    while (num_buckets >= 9)
    {
        libsais_accumulate_counts_s32_9(buckets - (num_buckets - 9) * bucket_stride, bucket_size, bucket_stride); num_buckets -= 8;
    }

    switch (num_buckets)
    {
        case 1: break;
        case 2: libsais_accumulate_counts_s32_2(buckets, bucket_size, bucket_stride); break;
        case 3: libsais_accumulate_counts_s32_3(buckets, bucket_size, bucket_stride); break;
        case 4: libsais_accumulate_counts_s32_4(buckets, bucket_size, bucket_stride); break;
        case 5: libsais_accumulate_counts_s32_5(buckets, bucket_size, bucket_stride); break;
        case 6: libsais_accumulate_counts_s32_6(buckets, bucket_size, bucket_stride); break;
        case 7: libsais_accumulate_counts_s32_7(buckets, bucket_size, bucket_stride); break;
        case 8: libsais_accumulate_counts_s32_8(buckets, bucket_size, bucket_stride); break;
    }
}

#endif

#ifndef AOCL_BWT
static void libsais_gather_lms_suffixes_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, fast_sint_t m, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    if (omp_block_size > 0)
    {
        const fast_sint_t prefetch_distance = 128;

        fast_sint_t i, j = omp_block_start + omp_block_size, c0 = T[omp_block_start + omp_block_size - 1], c1 = -1;
#ifdef AOCL_BWT
        c1 = T[0];
        if(c1 == c0) j = 0;
#endif /* AOCL_BWT */
        while (j < n && (c1 = T[j]) == c0) { ++j; }

        fast_uint_t s = c0 >= c1;

        for (i = omp_block_start + omp_block_size - 2, j = omp_block_start + 3; i >= j; i -= 4)
        {
            libsais_prefetchr(&T[i - prefetch_distance]);

            c1 = T[i - 0]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i + 1); m -= ((s & 3) == 1);
            c0 = T[i - 1]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i - 0); m -= ((s & 3) == 1);
            c1 = T[i - 2]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i - 1); m -= ((s & 3) == 1);
            c0 = T[i - 3]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i - 2); m -= ((s & 3) == 1);
        }
#ifdef AOCL_BWT
        for (j -= 3; i >= j && m > (omp_block_size-1)/2; i -= 1)
#else
        for (j -= 3; i >= j; i -= 1)
#endif /* AOCL_BWT */
        {
            c1 = c0; c0 = T[i]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i + 1); m -= ((s & 3) == 1);
        }
#ifdef AOCL_BWT
        if(m <= (omp_block_size-1)/2) return;
        // if the last element m changes i.e, s&3 == 1 then 0 is also an LMS character, i = -1
        c1 = c0; c0 = T[omp_block_size+i]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i + 1); m -= ((s & 3) == 1);
        if(m <= (omp_block_size-1)/2) return;
#endif /* AOCL_BWT */

        SA[m] = (sa_sint_t)(i + 1);
    }
}

static void libsais_gather_lms_suffixes_8u_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 65536 && omp_get_dynamic() == 0)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (n / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : n - omp_block_start;

        if (omp_num_threads == 1)
        {
            libsais_gather_lms_suffixes_8u(T, SA, n, (fast_sint_t)n - 1, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            fast_sint_t t, m = 0; for (t = omp_num_threads - 1; t > omp_thread_num; --t) { m += thread_state[t].state.m; }

            libsais_gather_lms_suffixes_8u(T, SA, n, (fast_sint_t)n - 1 - m, omp_block_start, omp_block_size);

            #pragma omp barrier

            if (thread_state[omp_thread_num].state.m > 0)
            {
                SA[(fast_sint_t)n - 1 - m] = (sa_sint_t)thread_state[omp_thread_num].state.last_lms_suffix;
            }
        }
#endif
    }
}
#endif /* AOCL_BWT */

static sa_sint_t libsais_gather_lms_suffixes_32s(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t             i   = n - 2;
    sa_sint_t             m   = n - 1;
    fast_uint_t           s   = 1;
    fast_sint_t           c0  = T[n - 1];
    fast_sint_t           c1  = 0;

    for (; i >= 3; i -= 4)
    {
        libsais_prefetchr(&T[i - prefetch_distance]);

        c1 = T[i - 0]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = i + 1; m -= ((s & 3) == 1);
        c0 = T[i - 1]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = i - 0; m -= ((s & 3) == 1);
        c1 = T[i - 2]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = i - 1; m -= ((s & 3) == 1);
        c0 = T[i - 3]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = i - 2; m -= ((s & 3) == 1);
    }

    for (; i >= 0; i -= 1)
    {
        c1 = c0; c0 = T[i]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = i + 1; m -= ((s & 3) == 1);
    }

    return n - 1 - m;
}

static sa_sint_t libsais_gather_compacted_lms_suffixes_32s(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t             i   = n - 2;
    sa_sint_t             m   = n - 1;
    fast_uint_t           s   = 1;
    fast_sint_t           c0  = T[n - 1];
    fast_sint_t           c1  = 0;

    for (; i >= 3; i -= 4)
    {
        libsais_prefetchr(&T[i - prefetch_distance]);

        c1 = T[i - 0]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = i + 1; m -= ((fast_sint_t)(s & 3) == (c0 >= 0));
        c0 = T[i - 1]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = i - 0; m -= ((fast_sint_t)(s & 3) == (c1 >= 0));
        c1 = T[i - 2]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = i - 1; m -= ((fast_sint_t)(s & 3) == (c0 >= 0));
        c0 = T[i - 3]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = i - 2; m -= ((fast_sint_t)(s & 3) == (c1 >= 0));
    }

    for (; i >= 0; i -= 1)
    {
        c1 = c0; c0 = T[i]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = i + 1; m -= ((fast_sint_t)(s & 3) == (c1 >= 0));
    }

    return n - 1 - m;
}

#if defined(LIBSAIS_OPENMP)

static void libsais_count_lms_suffixes_32s_4k(const sa_sint_t * RESTRICT T, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets)
{
    const fast_sint_t prefetch_distance = 32;

    memset(buckets, 0, 4 * (size_t)k * sizeof(sa_sint_t));

    sa_sint_t             i   = n - 2;
    fast_uint_t           s   = 1;
    fast_sint_t           c0  = T[n - 1];
    fast_sint_t           c1  = 0;

    for (; i >= prefetch_distance + 3; i -= 4)
    {
        libsais_prefetchr(&T[i - 2 * prefetch_distance]);

        libsais_prefetchw(&buckets[BUCKETS_INDEX4(T[i - prefetch_distance - 0], 0)]);
        libsais_prefetchw(&buckets[BUCKETS_INDEX4(T[i - prefetch_distance - 1], 0)]);
        libsais_prefetchw(&buckets[BUCKETS_INDEX4(T[i - prefetch_distance - 2], 0)]);
        libsais_prefetchw(&buckets[BUCKETS_INDEX4(T[i - prefetch_distance - 3], 0)]);

        c1 = T[i - 0]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1)));
        buckets[BUCKETS_INDEX4((fast_uint_t)c0, s & 3)]++;

        c0 = T[i - 1]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1)));
        buckets[BUCKETS_INDEX4((fast_uint_t)c1, s & 3)]++;

        c1 = T[i - 2]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1)));
        buckets[BUCKETS_INDEX4((fast_uint_t)c0, s & 3)]++;

        c0 = T[i - 3]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1)));
        buckets[BUCKETS_INDEX4((fast_uint_t)c1, s & 3)]++;
    }

    for (; i >= 0; i -= 1)
    {
        c1 = c0; c0 = T[i]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1)));
        buckets[BUCKETS_INDEX4((fast_uint_t)c1, s & 3)]++;
    }

    buckets[BUCKETS_INDEX4((fast_uint_t)c0, (s << 1) & 3)]++;
}

#endif

static void libsais_count_lms_suffixes_32s_2k(const sa_sint_t * RESTRICT T, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets)
{
    const fast_sint_t prefetch_distance = 32;

    memset(buckets, 0, 2 * (size_t)k * sizeof(sa_sint_t));

    sa_sint_t             i   = n - 2;
    fast_uint_t           s   = 1;
    fast_sint_t           c0  = T[n - 1];
    fast_sint_t           c1  = 0;

#ifdef AOCL_BWT
    c1 = T[0];
    sa_sint_t j = 0;
    while (j < n && (c1 = T[j]) == c0) { ++j; }
    s = c0 > c1;
#endif /* AOCL_BWT */

    for (; i >= prefetch_distance + 3; i -= 4)
    {
        libsais_prefetchr(&T[i - 2 * prefetch_distance]);

        libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 0], 0)]);
        libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 1], 0)]);
        libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 2], 0)]);
        libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 3], 0)]);

        c1 = T[i - 0]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1)));
        buckets[BUCKETS_INDEX2((fast_uint_t)c0, (s & 3) == 1)]++;

        c0 = T[i - 1]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1)));
        buckets[BUCKETS_INDEX2((fast_uint_t)c1, (s & 3) == 1)]++;

        c1 = T[i - 2]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1)));
        buckets[BUCKETS_INDEX2((fast_uint_t)c0, (s & 3) == 1)]++;

        c0 = T[i - 3]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1)));
        buckets[BUCKETS_INDEX2((fast_uint_t)c1, (s & 3) == 1)]++;
    }

    for (; i >= 0; i -= 1)
    {
        c1 = c0; c0 = T[i]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1)));
        buckets[BUCKETS_INDEX2((fast_uint_t)c1, (s & 3) == 1)]++;
    }
#ifdef AOCL_BWT
    c1 = c0; c0 = T[n-1]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1)));
    buckets[BUCKETS_INDEX2((fast_uint_t)c1, (s & 3) == 1)]++;
#else
    buckets[BUCKETS_INDEX2((fast_uint_t)c0, 0)]++;
#endif /* AOCL_BWT */
}

#if defined(LIBSAIS_OPENMP)

static void libsais_count_compacted_lms_suffixes_32s_2k(const sa_sint_t * RESTRICT T, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets)
{
    const fast_sint_t prefetch_distance = 32;

    memset(buckets, 0, 2 * (size_t)k * sizeof(sa_sint_t));

    sa_sint_t             i   = n - 2;
    fast_uint_t           s   = 1;
    fast_sint_t           c0  = T[n - 1];
    fast_sint_t           c1  = 0;

    for (; i >= prefetch_distance + 3; i -= 4)
    {
        libsais_prefetchr(&T[i - 2 * prefetch_distance]);

        libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 0] & SAINT_MAX, 0)]);
        libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 1] & SAINT_MAX, 0)]);
        libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 2] & SAINT_MAX, 0)]);
        libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 3] & SAINT_MAX, 0)]);

        c1 = T[i - 0]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1)));
        c0 &= SAINT_MAX; buckets[BUCKETS_INDEX2((fast_uint_t)c0, (s & 3) == 1)]++;

        c0 = T[i - 1]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1)));
        c1 &= SAINT_MAX; buckets[BUCKETS_INDEX2((fast_uint_t)c1, (s & 3) == 1)]++;

        c1 = T[i - 2]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1)));
        c0 &= SAINT_MAX; buckets[BUCKETS_INDEX2((fast_uint_t)c0, (s & 3) == 1)]++;

        c0 = T[i - 3]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1)));
        c1 &= SAINT_MAX; buckets[BUCKETS_INDEX2((fast_uint_t)c1, (s & 3) == 1)]++;
    }

    for (; i >= 0; i -= 1)
    {
        c1 = c0; c0 = T[i]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1)));
        c1 &= SAINT_MAX; buckets[BUCKETS_INDEX2((fast_uint_t)c1, (s & 3) == 1)]++;
    }

    c0 &= SAINT_MAX; buckets[BUCKETS_INDEX2((fast_uint_t)c0, 0)]++;
}

#endif

#if !defined(AOCL_BWT) || defined(AOCL_UNIT_TEST)
static sa_sint_t libsais_count_and_gather_lms_suffixes_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t * RESTRICT buckets, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    memset(buckets, 0, (size_t)4 * ALPHABET_SIZE * sizeof(sa_sint_t));

    fast_sint_t m = omp_block_start + omp_block_size - 1;

    if (omp_block_size > 0)
    {
        const fast_sint_t prefetch_distance = 128;

        fast_sint_t i, j = m + 1, c0 = T[m], c1 = -1;

#ifdef AOCL_BWT
        // First character is considered for comparing with last character
        // instead of senital
        c1 = T[0];
        if(c1 == c0) j = 0;
#endif /* AOCL_BWT */

        while (j < n && (c1 = T[j]) == c0) { ++j; }

        fast_uint_t s = c0 >= c1;

        /*
            4 slots are allocated per character for a total of 256*4=1024 different indexes within buckets.
            Each slot indicates both the character and its type which is one of : LL,LS,SL,SS.
            The index is computed as follows:
            1. char : "T[i]"
            2. comparision of present character to next character : "T[i]>T[i+1]"
            3. comparision of previous character to present char  : "T[i-1]>T[i]"
            
            so all these are encoded as:
                (T[i] << 2)    (T[i]>T[i+1])   (T[i-1]>T[i])
                    ^                ^               ^
                    |                |               |
            2nd bit to 9th bit    1st bit          0th bit
                1. char              2.              3.

            For storing this 1024 different indexes are required.
        */
        for (i = m - 1, j = omp_block_start + 3; i >= j; i -= 4)
        {
            libsais_prefetchr(&T[i - prefetch_distance]);

            c1 = T[i - 0]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i + 1); m -= ((s & 3) == 1);
            buckets[BUCKETS_INDEX4((fast_uint_t)c0, s & 3)]++;

            c0 = T[i - 1]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i - 0); m -= ((s & 3) == 1);
            buckets[BUCKETS_INDEX4((fast_uint_t)c1, s & 3)]++;

            c1 = T[i - 2]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i - 1); m -= ((s & 3) == 1);
            buckets[BUCKETS_INDEX4((fast_uint_t)c0, s & 3)]++;

            c0 = T[i - 3]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i - 2); m -= ((s & 3) == 1);
            buckets[BUCKETS_INDEX4((fast_uint_t)c1, s & 3)]++;
        }

        for (j -= 3; i >= j; i -= 1)
        {
            c1 = c0; c0 = T[i]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i + 1); m -= ((s & 3) == 1);
            buckets[BUCKETS_INDEX4((fast_uint_t)c1, s & 3)]++;
        }
#ifdef AOCL_BWT
        // To calculate 3rd part, i.e, (T[i-1]>T[i]) prev element is required
        // since current index is 0, so previous element should be `n-1`th element,
        c1 = (i >= 0) ? T[i] : T[omp_block_start + omp_block_size - 1];
#else
        c1 = (i >= 0) ? T[i] : -1;
#endif /* AOCL_BWT */
        s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i + 1); m -= ((s & 3) == 1);
        buckets[BUCKETS_INDEX4((fast_uint_t)c0, s & 3)]++;
    }

    return (sa_sint_t)(omp_block_start + omp_block_size - 1 - m);
}

static sa_sint_t libsais_count_and_gather_lms_suffixes_8u_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    sa_sint_t m = 0;

#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 65536 && omp_get_dynamic() == 0)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (n / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : n - omp_block_start;

        if (omp_num_threads == 1)
        {
            // for single thread `omp_block_start = 0`, `omp_block_size = n` 
            m = libsais_count_and_gather_lms_suffixes_8u(T, SA, n, buckets, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                thread_state[omp_thread_num].state.position = omp_block_start + omp_block_size;
                thread_state[omp_thread_num].state.m = libsais_count_and_gather_lms_suffixes_8u(T, SA, n, thread_state[omp_thread_num].state.buckets, omp_block_start, omp_block_size);

                if (thread_state[omp_thread_num].state.m > 0)
                {
                    thread_state[omp_thread_num].state.last_lms_suffix = SA[thread_state[omp_thread_num].state.position - 1];
                }
            }

            #pragma omp barrier

            #pragma omp master
            {
                memset(buckets, 0, (size_t)4 * ALPHABET_SIZE * sizeof(sa_sint_t));

                fast_sint_t t;
                for (t = omp_num_threads - 1; t >= 0; --t)
                {
                    m += (sa_sint_t)thread_state[t].state.m;

                    if (t != omp_num_threads - 1 && thread_state[t].state.m > 0)
                    {
                        memcpy(&SA[n - m], &SA[thread_state[t].state.position - thread_state[t].state.m], (size_t)thread_state[t].state.m * sizeof(sa_sint_t));
                    }

                    {
                        sa_sint_t * RESTRICT temp_bucket = thread_state[t].state.buckets;
                        fast_sint_t s; for (s = 0; s < 4 * ALPHABET_SIZE; s += 1) { sa_sint_t A = buckets[s], B = temp_bucket[s]; buckets[s] = A + B; temp_bucket[s] = A; }
                    }
                }
            }
        }
#endif
    }

    return m;
}

int32_t Test_count_and_gather_lms_suffixes(const uint8_t * T, int32_t * SA, int32_t n, int32_t * buckets)
{
    return libsais_count_and_gather_lms_suffixes_8u_omp(T, SA, n, buckets, 1, NULL);
}
#endif /* !defined(AOCL_BWT) || defined(AOCL_UNIT_TEST) */

static sa_sint_t libsais_count_and_gather_lms_suffixes_32s_4k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    memset(buckets, 0, 4 * (size_t)k * sizeof(sa_sint_t));

    fast_sint_t m = omp_block_start + omp_block_size - 1;

    if (omp_block_size > 0)
    {
        const fast_sint_t prefetch_distance = 32;

#ifdef AOCL_BWT
        fast_sint_t i, j = m + 1, c0 = T[m], c1 = T[0];
        if(c0 == c1) j = 0;
#else
        fast_sint_t i, j = m + 1, c0 = T[m], c1 = -1;
#endif /* AOCL_BWT */

        while (j < n && (c1 = T[j]) == c0) { ++j; }

        fast_uint_t s = c0 >= c1;

        for (i = m - 1, j = omp_block_start + prefetch_distance + 3; i >= j; i -= 4)
        {
            libsais_prefetchr(&T[i - 2 * prefetch_distance]);

            libsais_prefetchw(&buckets[BUCKETS_INDEX4(T[i - prefetch_distance - 0], 0)]);
            libsais_prefetchw(&buckets[BUCKETS_INDEX4(T[i - prefetch_distance - 1], 0)]);
            libsais_prefetchw(&buckets[BUCKETS_INDEX4(T[i - prefetch_distance - 2], 0)]);
            libsais_prefetchw(&buckets[BUCKETS_INDEX4(T[i - prefetch_distance - 3], 0)]);

            c1 = T[i - 0]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i + 1); m -= ((s & 3) == 1);
            buckets[BUCKETS_INDEX4((fast_uint_t)c0, s & 3)]++;

            c0 = T[i - 1]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i - 0); m -= ((s & 3) == 1);
            buckets[BUCKETS_INDEX4((fast_uint_t)c1, s & 3)]++;

            c1 = T[i - 2]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i - 1); m -= ((s & 3) == 1);
            buckets[BUCKETS_INDEX4((fast_uint_t)c0, s & 3)]++;

            c0 = T[i - 3]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i - 2); m -= ((s & 3) == 1);
            buckets[BUCKETS_INDEX4((fast_uint_t)c1, s & 3)]++;
        }

        for (j -= prefetch_distance + 3; i >= j; i -= 1)
        {
            c1 = c0; c0 = T[i]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i + 1); m -= ((s & 3) == 1);
            buckets[BUCKETS_INDEX4((fast_uint_t)c1, s & 3)]++;
        }

#ifdef AOCL_BWT
        c1 = (i >= 0) ? T[i] : T[omp_block_size - 1];
#else
        c1 = (i >= 0) ? T[i] : -1;
#endif /* AOCL_BWT */
        s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i + 1); m -= ((s & 3) == 1);
        buckets[BUCKETS_INDEX4((fast_uint_t)c0, s & 3)]++;
    }

    return (sa_sint_t)(omp_block_start + omp_block_size - 1 - m);
}

static sa_sint_t libsais_count_and_gather_lms_suffixes_32s_2k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    memset(buckets, 0, 2 * (size_t)k * sizeof(sa_sint_t));

    fast_sint_t m = omp_block_start + omp_block_size - 1;

    if (omp_block_size > 0)
    {
        const fast_sint_t prefetch_distance = 32;

#ifdef AOCL_BWT
        fast_sint_t i, j = m + 1, c0 = T[m], c1 = T[0];
        if(c0 == c1) j = 0;
#else
        fast_sint_t i, j = m + 1, c0 = T[m], c1 = -1;
#endif /* AOCL_BWT */

        while (j < n && (c1 = T[j]) == c0) { ++j; }

        fast_uint_t s = c0 >= c1;

        for (i = m - 1, j = omp_block_start + prefetch_distance + 3; i >= j; i -= 4)
        {
            libsais_prefetchr(&T[i - 2 * prefetch_distance]);

            libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 0], 0)]);
            libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 1], 0)]);
            libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 2], 0)]);
            libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 3], 0)]);

            c1 = T[i - 0]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i + 1); m -= ((s & 3) == 1);
            buckets[BUCKETS_INDEX2((fast_uint_t)c0, (s & 3) == 1)]++;

            c0 = T[i - 1]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i - 0); m -= ((s & 3) == 1);
            buckets[BUCKETS_INDEX2((fast_uint_t)c1, (s & 3) == 1)]++;

            c1 = T[i - 2]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i - 1); m -= ((s & 3) == 1);
            buckets[BUCKETS_INDEX2((fast_uint_t)c0, (s & 3) == 1)]++;

            c0 = T[i - 3]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i - 2); m -= ((s & 3) == 1);
            buckets[BUCKETS_INDEX2((fast_uint_t)c1, (s & 3) == 1)]++;
        }

        for (j -= prefetch_distance + 3; i >= j; i -= 1)
        {
            c1 = c0; c0 = T[i]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i + 1); m -= ((s & 3) == 1);
            buckets[BUCKETS_INDEX2((fast_uint_t)c1, (s & 3) == 1)]++;
        }
#ifdef AOCL_BWT
        c1 = (i >= 0) ? T[i] : T[omp_block_size - 1];
        s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1)));
        buckets[BUCKETS_INDEX2((fast_uint_t)c0, (s & 3) == 1)]++;
        if(m > (omp_block_size-1)/2)
        {
            SA[m] = (sa_sint_t)(i + 1);
            m -= ((s & 3) == 1);
        }
#else
        c1 = (i >= 0) ? T[i] : -1; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i + 1); m -= ((s & 3) == 1);
        buckets[BUCKETS_INDEX2((fast_uint_t)c0, (s & 3) == 1)]++;
#endif /* AOCL_BWT */
    }

    return (sa_sint_t)(omp_block_start + omp_block_size - 1 - m);
}

static sa_sint_t libsais_count_and_gather_compacted_lms_suffixes_32s_2k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    memset(buckets, 0, 2 * (size_t)k * sizeof(sa_sint_t));

    fast_sint_t m = omp_block_start + omp_block_size - 1;

    if (omp_block_size > 0)
    {
        const fast_sint_t prefetch_distance = 32;

        fast_sint_t i, j = m + 1, c0 = T[m], c1 = -1;
#ifdef AOCL_BWT
        c1 = T[0];
        if(c1 == c0) j = 0;
#endif /* AOCL_BWT */

        while (j < n && (c1 = T[j]) == c0) { ++j; }

        fast_uint_t s = c0 >= c1;
#ifdef AOCL_BWT
        i = m - 1;
        j = omp_block_start + prefetch_distance + 3;
#else /* AOCL_BWT */
        for (i = m - 1, j = omp_block_start + prefetch_distance + 3; i >= j; i -= 4)
        {
            libsais_prefetchr(&T[i - 2 * prefetch_distance]);

            libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 0] & SAINT_MAX, 0)]);
            libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 1] & SAINT_MAX, 0)]);
            libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 2] & SAINT_MAX, 0)]);
            libsais_prefetchw(&buckets[BUCKETS_INDEX2(T[i - prefetch_distance - 3] & SAINT_MAX, 0)]);

            c1 = T[i - 0]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i + 1); m -= ((fast_sint_t)(s & 3) == (c0 >= 0));
            c0 &= SAINT_MAX; buckets[BUCKETS_INDEX2((fast_uint_t)c0, (s & 3) == 1)]++;

            c0 = T[i - 1]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i - 0); m -= ((fast_sint_t)(s & 3) == (c1 >= 0));
            c1 &= SAINT_MAX; buckets[BUCKETS_INDEX2((fast_uint_t)c1, (s & 3) == 1)]++;

            c1 = T[i - 2]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i - 1); m -= ((fast_sint_t)(s & 3) == (c0 >= 0));
            c0 &= SAINT_MAX; buckets[BUCKETS_INDEX2((fast_uint_t)c0, (s & 3) == 1)]++;

            c0 = T[i - 3]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i - 2); m -= ((fast_sint_t)(s & 3) == (c1 >= 0));
            c1 &= SAINT_MAX; buckets[BUCKETS_INDEX2((fast_uint_t)c1, (s & 3) == 1)]++;
        }
#endif /* AOCL_BWT */

        for (j -= prefetch_distance + 3; i >= j; i -= 1)
        {
            // No two LMS indexes would be consecutive. Distinct LMS indexes are indicated by setting MSB bit, so all distinct LMS indexes would be negative.
            // Hence no two distinct elements could be positive, so no two adjacent elements can be negative.
            // Hence if T[i] is distinct i.e, negative, then T[i-1] would be positive, so `s=01` at T[i-1]
            // So `m -= ((fast_sint_t)(s & 3) == (c1 >= 0));` where `c1 = T[i]`, so the before expression evaluates to m-=0
            // This way all distinct elements are skipped. 
            c1 = c0; c0 = T[i]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i + 1); m -= ((fast_sint_t)(s & 3) == (c1 >= 0));
            c1 &= SAINT_MAX; buckets[BUCKETS_INDEX2((fast_uint_t)c1, (s & 3) == 1)]++;
        }
#ifdef AOCL_BWT
        c1 = (i>=0) ? T[i] : T[omp_block_start + omp_block_size - 1];
#else
        c1 = (i >= 0) ? T[i] : -1;
#endif /* AOCL_BWT */
        s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); SA[m] = (sa_sint_t)(i + 1); m -= ((fast_sint_t)(s & 3) == (c0 >= 0));
        c0 &= SAINT_MAX; buckets[BUCKETS_INDEX2((fast_uint_t)c0, (s & 3) == 1)]++;
    }

    return (sa_sint_t)(omp_block_start + omp_block_size - 1 - m);
}

#if defined(LIBSAIS_OPENMP)

static fast_sint_t libsais_get_bucket_stride(fast_sint_t free_space, fast_sint_t bucket_size, fast_sint_t num_buckets)
{
    fast_sint_t bucket_size_1024 = (bucket_size + 1023) & (-1024); if (free_space / (num_buckets - 1) >= bucket_size_1024) { return bucket_size_1024; }
    fast_sint_t bucket_size_16 = (bucket_size + 15) & (-16); if (free_space / (num_buckets - 1) >= bucket_size_16) { return bucket_size_16; }

    return bucket_size;
}

static sa_sint_t libsais_count_and_gather_lms_suffixes_32s_4k_fs_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    sa_sint_t m = 0;

#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (n / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : n - omp_block_start;

        if (omp_num_threads == 1)
        {
            m = libsais_count_and_gather_lms_suffixes_32s_4k(T, SA, n, k, buckets, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            fast_sint_t bucket_size       = 4 * (fast_sint_t)k;
            fast_sint_t bucket_stride     = libsais_get_bucket_stride(buckets - &SA[n], bucket_size, omp_num_threads);

            {
                thread_state[omp_thread_num].state.position = omp_block_start + omp_block_size;
                thread_state[omp_thread_num].state.count = libsais_count_and_gather_lms_suffixes_32s_4k(T, SA, n, k, buckets - (omp_thread_num * bucket_stride), omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            if (omp_thread_num == omp_num_threads - 1)
            {
                fast_sint_t t;
                for (t = omp_num_threads - 1; t >= 0; --t)
                {
                    m += (sa_sint_t)thread_state[t].state.count;

                    if (t != omp_num_threads - 1 && thread_state[t].state.count > 0)
                    {
                        memcpy(&SA[n - m], &SA[thread_state[t].state.position - thread_state[t].state.count], (size_t)thread_state[t].state.count * sizeof(sa_sint_t));
                    }
                }
            }
            else
            {
                omp_num_threads     = omp_num_threads - 1;
                omp_block_stride    = (bucket_size / omp_num_threads) & (-16);
                omp_block_start     = omp_thread_num * omp_block_stride;
                omp_block_size      = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : bucket_size - omp_block_start;

                libsais_accumulate_counts_s32(buckets + omp_block_start, omp_block_size, bucket_stride, omp_num_threads + 1);
            }
        }
#endif
    }

    return m;
}

static sa_sint_t libsais_count_and_gather_lms_suffixes_32s_2k_fs_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    sa_sint_t m = 0;

#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (n / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : n - omp_block_start;

        if (omp_num_threads == 1)
        {
            m = libsais_count_and_gather_lms_suffixes_32s_2k(T, SA, n, k, buckets, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            fast_sint_t bucket_size       = 2 * (fast_sint_t)k;
            fast_sint_t bucket_stride     = libsais_get_bucket_stride(buckets - &SA[n], bucket_size, omp_num_threads);

            {
                thread_state[omp_thread_num].state.position = omp_block_start + omp_block_size;
                thread_state[omp_thread_num].state.count = libsais_count_and_gather_lms_suffixes_32s_2k(T, SA, n, k, buckets - (omp_thread_num * bucket_stride), omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            if (omp_thread_num == omp_num_threads - 1)
            {
                fast_sint_t t;
                for (t = omp_num_threads - 1; t >= 0; --t)
                {
                    m += (sa_sint_t)thread_state[t].state.count;

                    if (t != omp_num_threads - 1 && thread_state[t].state.count > 0)
                    {
                        memcpy(&SA[n - m], &SA[thread_state[t].state.position - thread_state[t].state.count], (size_t)thread_state[t].state.count * sizeof(sa_sint_t));
                    }
                }
            }
            else
            {
                omp_num_threads     = omp_num_threads - 1;
                omp_block_stride    = (bucket_size / omp_num_threads) & (-16);
                omp_block_start     = omp_thread_num * omp_block_stride;
                omp_block_size      = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : bucket_size - omp_block_start;

                libsais_accumulate_counts_s32(buckets + omp_block_start, omp_block_size, bucket_stride, omp_num_threads + 1);
            }
        }
#endif
    }

    return m;
}

static void libsais_count_and_gather_compacted_lms_suffixes_32s_2k_fs_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (n / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : n - omp_block_start;

        if (omp_num_threads == 1)
        {
            libsais_count_and_gather_compacted_lms_suffixes_32s_2k(T, SA, n, k, buckets, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            fast_sint_t bucket_size       = 2 * (fast_sint_t)k;
            fast_sint_t bucket_stride     = libsais_get_bucket_stride(buckets - &SA[(fast_sint_t)n + (fast_sint_t)n], bucket_size, omp_num_threads);

            {
                thread_state[omp_thread_num].state.position = omp_block_start + omp_block_size;
                thread_state[omp_thread_num].state.count = libsais_count_and_gather_compacted_lms_suffixes_32s_2k(T, SA + n, n, k, buckets - (omp_thread_num * bucket_stride), omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            {
                fast_sint_t t, m = 0; for (t = omp_num_threads - 1; t >= omp_thread_num; --t) { m += (sa_sint_t)thread_state[t].state.count; }

                if (thread_state[omp_thread_num].state.count > 0)
                {
                    memcpy(&SA[n - m], &SA[n + thread_state[omp_thread_num].state.position - thread_state[omp_thread_num].state.count], (size_t)thread_state[omp_thread_num].state.count * sizeof(sa_sint_t));
                }
            }

            {
                omp_block_stride    = (bucket_size / omp_num_threads) & (-16);
                omp_block_start     = omp_thread_num * omp_block_stride;
                omp_block_size      = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : bucket_size - omp_block_start;

                libsais_accumulate_counts_s32(buckets + omp_block_start, omp_block_size, bucket_stride, omp_num_threads);
            }
        }
#endif
    }
}

#endif

static sa_sint_t libsais_count_and_gather_lms_suffixes_32s_4k_nofs_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads)
{
    sa_sint_t m = 0;

#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(2) if(threads > 1 && n >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads);

        fast_sint_t omp_num_threads   = 1;
#endif
        if (omp_num_threads == 1)
        {
            m = libsais_count_and_gather_lms_suffixes_32s_4k(T, SA, n, k, buckets, 0, n);
        }
#if defined(LIBSAIS_OPENMP)
        else if (omp_thread_num == 0)
        {
            libsais_count_lms_suffixes_32s_4k(T, n, k, buckets);
        }
        else
        {
            m = libsais_gather_lms_suffixes_32s(T, SA, n);
        }
#endif
    }

    return m;
}

static sa_sint_t libsais_count_and_gather_lms_suffixes_32s_2k_nofs_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads)
{
    sa_sint_t m = 0;

#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(2) if(threads > 1 && n >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads);

        fast_sint_t omp_num_threads   = 1;
#endif
        if (omp_num_threads == 1)
        {
            m = libsais_count_and_gather_lms_suffixes_32s_2k(T, SA, n, k, buckets, 0, n);
        }
#if defined(LIBSAIS_OPENMP)
        else if (omp_thread_num == 0)
        {
            libsais_count_lms_suffixes_32s_2k(T, n, k, buckets);
        }
        else
        {
            m = libsais_gather_lms_suffixes_32s(T, SA, n);
        }
#endif
    }

    return m;
}

static sa_sint_t libsais_count_and_gather_compacted_lms_suffixes_32s_2k_nofs_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads)
{
    sa_sint_t m = 0;

#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(2) if(threads > 1 && n >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads);

        fast_sint_t omp_num_threads   = 1;
#endif
        if (omp_num_threads == 1)
        {
            m = libsais_count_and_gather_compacted_lms_suffixes_32s_2k(T, SA, n, k, buckets, 0, n);
        }
#if defined(LIBSAIS_OPENMP)
        else if (omp_thread_num == 0)
        {
            libsais_count_compacted_lms_suffixes_32s_2k(T, n, k, buckets);
        }
        else
        {
            m = libsais_gather_compacted_lms_suffixes_32s(T, SA, n);
        }
#endif
    }

    return m;
}

static sa_sint_t libsais_count_and_gather_lms_suffixes_32s_4k_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    sa_sint_t m;

#if defined(LIBSAIS_OPENMP)
    sa_sint_t max_threads = (sa_sint_t)((buckets - &SA[n]) / ((4 * (fast_sint_t)k + 15) & (-16))); if (max_threads > threads) { max_threads = threads; }
    if (max_threads > 1 && n >= 65536 && n / k >= 2)
    {
        if (max_threads > n / 16 / k) { max_threads = n / 16 / k; }
        m = libsais_count_and_gather_lms_suffixes_32s_4k_fs_omp(T, SA, n, k, buckets, max_threads > 2 ? max_threads : 2, thread_state);
    }
    else
#else
    UNUSED(thread_state);
#endif
    {
        m = libsais_count_and_gather_lms_suffixes_32s_4k_nofs_omp(T, SA, n, k, buckets, threads);
    }

    return m;
}

static sa_sint_t libsais_count_and_gather_lms_suffixes_32s_2k_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    sa_sint_t m;

#if defined(LIBSAIS_OPENMP)
    sa_sint_t max_threads = (sa_sint_t)((buckets - &SA[n]) / ((2 * (fast_sint_t)k + 15) & (-16))); if (max_threads > threads) { max_threads = threads; }
    if (max_threads > 1 && n >= 65536 && n / k >= 2)
    {
        if (max_threads > n / 8 / k) { max_threads = n / 8 / k; }
        m = libsais_count_and_gather_lms_suffixes_32s_2k_fs_omp(T, SA, n, k, buckets, max_threads > 2 ? max_threads : 2, thread_state);
    }
    else
#else
    UNUSED(thread_state);
#endif
    {
        m = libsais_count_and_gather_lms_suffixes_32s_2k_nofs_omp(T, SA, n, k, buckets, threads);
    }

    return m;
}

static void libsais_count_and_gather_compacted_lms_suffixes_32s_2k_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    sa_sint_t max_threads = (sa_sint_t)((buckets - &SA[(fast_sint_t)n + (fast_sint_t)n]) / ((2 * (fast_sint_t)k + 15) & (-16))); if (max_threads > threads) { max_threads = threads; }
    if (max_threads > 1 && n >= 65536 && n / k >= 2)
    {
        if (max_threads > n / 8 / k) { max_threads = n / 8 / k; }
        libsais_count_and_gather_compacted_lms_suffixes_32s_2k_fs_omp(T, SA, n, k, buckets, max_threads > 2 ? max_threads : 2, thread_state);
    }
    else
#else
    UNUSED(thread_state);
#endif
    {
        libsais_count_and_gather_compacted_lms_suffixes_32s_2k_nofs_omp(T, SA, n, k, buckets, threads);
    }
}

static void libsais_count_suffixes_32s(const sa_sint_t * RESTRICT T, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets)
{
    const fast_sint_t prefetch_distance = 32;

    memset(buckets, 0, (size_t)k * sizeof(sa_sint_t));

    fast_sint_t i, j;
    for (i = 0, j = (fast_sint_t)n - 7; i < j; i += 8)
    {
        libsais_prefetchr(&T[i + prefetch_distance]);

        buckets[T[i + 0]]++;
        buckets[T[i + 1]]++;
        buckets[T[i + 2]]++;
        buckets[T[i + 3]]++;
        buckets[T[i + 4]]++;
        buckets[T[i + 5]]++;
        buckets[T[i + 6]]++;
        buckets[T[i + 7]]++;
    }

    for (j += 7; i < j; i += 1)
    {
        buckets[T[i]]++;
    }
}

static sa_sint_t libsais_initialize_buckets_start_and_end_8u(sa_sint_t * RESTRICT buckets, sa_sint_t * RESTRICT freq)
{
    sa_sint_t * RESTRICT bucket_start = &buckets[6 * ALPHABET_SIZE];
    sa_sint_t * RESTRICT bucket_end   = &buckets[7 * ALPHABET_SIZE];

    fast_sint_t k = -1;

    if (freq != NULL)
    {
        fast_sint_t i, j; sa_sint_t sum = 0;
        for (i = BUCKETS_INDEX4(0, 0), j = 0; i <= BUCKETS_INDEX4(ALPHABET_SIZE - 1, 0); i += BUCKETS_INDEX4(1, 0), j += 1)
        {
            sa_sint_t total = buckets[i + BUCKETS_INDEX4(0, 0)] + buckets[i + BUCKETS_INDEX4(0, 1)] + buckets[i + BUCKETS_INDEX4(0, 2)] + buckets[i + BUCKETS_INDEX4(0, 3)];

            bucket_start[j] = sum; sum += total; bucket_end[j] = sum; k = total > 0 ? j : k; freq[j] = total;
        }
    }
    else
    {
        fast_sint_t i, j; sa_sint_t sum = 0;
        for (i = BUCKETS_INDEX4(0, 0), j = 0; i <= BUCKETS_INDEX4(ALPHABET_SIZE - 1, 0); i += BUCKETS_INDEX4(1, 0), j += 1)
        {
            sa_sint_t total = buckets[i + BUCKETS_INDEX4(0, 0)] + buckets[i + BUCKETS_INDEX4(0, 1)] + buckets[i + BUCKETS_INDEX4(0, 2)] + buckets[i + BUCKETS_INDEX4(0, 3)];

            bucket_start[j] = sum; sum += total; bucket_end[j] = sum; k = total > 0 ? j : k;
        }
    }

    return (sa_sint_t)(k + 1);
}

static void libsais_initialize_buckets_start_and_end_32s_6k(sa_sint_t k, sa_sint_t * RESTRICT buckets)
{
    sa_sint_t * RESTRICT bucket_start = &buckets[4 * (fast_sint_t)k];
    sa_sint_t * RESTRICT bucket_end   = &buckets[5 * (fast_sint_t)k];

    fast_sint_t i, j; sa_sint_t sum = 0;
    for (i = BUCKETS_INDEX4(0, 0), j = 0; i <= BUCKETS_INDEX4((fast_sint_t)k - 1, 0); i += BUCKETS_INDEX4(1, 0), j += 1)
    {
        bucket_start[j] = sum;
        sum += buckets[i + BUCKETS_INDEX4(0, 0)] + buckets[i + BUCKETS_INDEX4(0, 1)] + buckets[i + BUCKETS_INDEX4(0, 2)] + buckets[i + BUCKETS_INDEX4(0, 3)];
        bucket_end[j] = sum;
    }
}

static void libsais_initialize_buckets_start_and_end_32s_4k(sa_sint_t k, sa_sint_t * RESTRICT buckets)
{
    sa_sint_t * RESTRICT bucket_start = &buckets[2 * (fast_sint_t)k];
    sa_sint_t * RESTRICT bucket_end   = &buckets[3 * (fast_sint_t)k];

    fast_sint_t i, j; sa_sint_t sum = 0;
    for (i = BUCKETS_INDEX2(0, 0), j = 0; i <= BUCKETS_INDEX2((fast_sint_t)k - 1, 0); i += BUCKETS_INDEX2(1, 0), j += 1)
    { 
        bucket_start[j] = sum;
        sum += buckets[i + BUCKETS_INDEX2(0, 0)] + buckets[i + BUCKETS_INDEX2(0, 1)];
        bucket_end[j] = sum;
    }
}

static void libsais_initialize_buckets_end_32s_2k(sa_sint_t k, sa_sint_t * RESTRICT buckets)
{
    fast_sint_t i; sa_sint_t sum0 = 0;
    for (i = BUCKETS_INDEX2(0, 0); i <= BUCKETS_INDEX2((fast_sint_t)k - 1, 0); i += BUCKETS_INDEX2(1, 0))
    { 
        sum0 += buckets[i + BUCKETS_INDEX2(0, 0)] + buckets[i + BUCKETS_INDEX2(0, 1)]; buckets[i + BUCKETS_INDEX2(0, 0)] = sum0;
    }
}

static void libsais_initialize_buckets_start_and_end_32s_2k(sa_sint_t k, sa_sint_t * RESTRICT buckets)
{
    fast_sint_t i, j;
    for (i = BUCKETS_INDEX2(0, 0), j = 0; i <= BUCKETS_INDEX2((fast_sint_t)k - 1, 0); i += BUCKETS_INDEX2(1, 0), j += 1)
    {
        buckets[j] = buckets[i];
    }

    buckets[k] = 0; memcpy(&buckets[k + 1], buckets, ((size_t)k - 1) * sizeof(sa_sint_t));
}

static void libsais_initialize_buckets_start_32s_1k(sa_sint_t k, sa_sint_t * RESTRICT buckets)
{
    fast_sint_t i; sa_sint_t sum = 0;
    for (i = 0; i <= (fast_sint_t)k - 1; i += 1) { sa_sint_t tmp = buckets[i]; buckets[i] = sum; sum += tmp; }
}

static void libsais_initialize_buckets_end_32s_1k(sa_sint_t k, sa_sint_t * RESTRICT buckets)
{
    fast_sint_t i; sa_sint_t sum = 0;
    for (i = 0; i <= (fast_sint_t)k - 1; i += 1) { sum += buckets[i]; buckets[i] = sum; }
}

static sa_sint_t libsais_initialize_buckets_for_lms_suffixes_radix_sort_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT buckets, sa_sint_t first_lms_suffix)
{
#ifndef AOCL_BWT
    {
        fast_uint_t     s = 0;
        fast_sint_t     c0 = T[first_lms_suffix];
        fast_sint_t     c1 = 0;

        for (; --first_lms_suffix >= 0; )
        {
            c1 = c0; c0 = T[first_lms_suffix]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1)));
            buckets[BUCKETS_INDEX4((fast_uint_t)c1, s & 3)]--;
        }

        buckets[BUCKETS_INDEX4((fast_uint_t)c0, (s << 1) & 3)]--;
    }
#endif /* AOCL_BWT */
    {
        sa_sint_t * RESTRICT temp_bucket = &buckets[4 * ALPHABET_SIZE];

        fast_sint_t i, j; sa_sint_t sum = 0;
        for (i = BUCKETS_INDEX4(0, 0), j = BUCKETS_INDEX2(0, 0); i <= BUCKETS_INDEX4(ALPHABET_SIZE - 1, 0); i += BUCKETS_INDEX4(1, 0), j += BUCKETS_INDEX2(1, 0))
        { 
            temp_bucket[j + BUCKETS_INDEX2(0, 1)] = sum; sum += buckets[i + BUCKETS_INDEX4(0, 1)] + buckets[i + BUCKETS_INDEX4(0, 3)]; temp_bucket[j] = sum;
        }

        return sum;
    }
}

static void libsais_initialize_buckets_for_lms_suffixes_radix_sort_32s_2k(const sa_sint_t * RESTRICT T, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t first_lms_suffix)
{
#ifndef AOCL_BWT
    buckets[BUCKETS_INDEX2(T[first_lms_suffix], 0)]++;
    buckets[BUCKETS_INDEX2(T[first_lms_suffix], 1)]--;
#endif /* AOCL_BWT */
    fast_sint_t i; sa_sint_t sum0 = 0, sum1 = 0;
    for (i = BUCKETS_INDEX2(0, 0); i <= BUCKETS_INDEX2((fast_sint_t)k - 1, 0); i += BUCKETS_INDEX2(1, 0))
    { 
        sum0 += buckets[i + BUCKETS_INDEX2(0, 0)] + buckets[i + BUCKETS_INDEX2(0, 1)];
        sum1 += buckets[i + BUCKETS_INDEX2(0, 1)];
        
        buckets[i + BUCKETS_INDEX2(0, 0)] = sum0;
        buckets[i + BUCKETS_INDEX2(0, 1)] = sum1;
    }
}

static sa_sint_t libsais_initialize_buckets_for_lms_suffixes_radix_sort_32s_6k(const sa_sint_t * RESTRICT T, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t first_lms_suffix)
{
#ifndef AOCL_BWT
    {
        fast_uint_t     s = 0;
        fast_sint_t     c0 = T[first_lms_suffix];
        fast_sint_t     c1 = 0;

        for (; --first_lms_suffix >= 0; )
        {
            c1 = c0; c0 = T[first_lms_suffix]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1)));
            buckets[BUCKETS_INDEX4((fast_uint_t)c1, s & 3)]--;
        }

        buckets[BUCKETS_INDEX4((fast_uint_t)c0, (s << 1) & 3)]--;
    }
#endif /* AOCL_BWT */

    {
        sa_sint_t * RESTRICT temp_bucket = &buckets[4 * (fast_sint_t)k];

        fast_sint_t i, j; sa_sint_t sum = 0;
        for (i = BUCKETS_INDEX4(0, 0), j = 0; i <= BUCKETS_INDEX4((fast_sint_t)k - 1, 0); i += BUCKETS_INDEX4(1, 0), j += 1)
        { 
            sum += buckets[i + BUCKETS_INDEX4(0, 1)] + buckets[i + BUCKETS_INDEX4(0, 3)]; temp_bucket[j] = sum;
        }

        return sum;
    }
}

static void libsais_initialize_buckets_for_radix_and_partial_sorting_32s_4k(const sa_sint_t * RESTRICT T, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t first_lms_suffix)
{
    sa_sint_t * RESTRICT bucket_start = &buckets[2 * (fast_sint_t)k];
    sa_sint_t * RESTRICT bucket_end   = &buckets[3 * (fast_sint_t)k];

    buckets[BUCKETS_INDEX2(T[first_lms_suffix], 0)]++;
    buckets[BUCKETS_INDEX2(T[first_lms_suffix], 1)]--;

    fast_sint_t i, j; sa_sint_t sum0 = 0, sum1 = 0;
    for (i = BUCKETS_INDEX2(0, 0), j = 0; i <= BUCKETS_INDEX2((fast_sint_t)k - 1, 0); i += BUCKETS_INDEX2(1, 0), j += 1)
    { 
        bucket_start[j] = sum1;

        sum0 += buckets[i + BUCKETS_INDEX2(0, 1)];
        sum1 += buckets[i + BUCKETS_INDEX2(0, 0)] + buckets[i + BUCKETS_INDEX2(0, 1)];
        buckets[i + BUCKETS_INDEX2(0, 1)] = sum0;

        bucket_end[j] = sum1;
    }
}

static void libsais_radix_sort_lms_suffixes_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 3; i >= j; i -= 4)
    {
        libsais_prefetchr(&SA[i - 2 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i - prefetch_distance - 0]]);
        libsais_prefetchr(&T[SA[i - prefetch_distance - 1]]);
        libsais_prefetchr(&T[SA[i - prefetch_distance - 2]]);
        libsais_prefetchr(&T[SA[i - prefetch_distance - 3]]);

        sa_sint_t p0 = SA[i - 0]; SA[--induction_bucket[BUCKETS_INDEX2(T[p0], 0)]] = p0;
        sa_sint_t p1 = SA[i - 1]; SA[--induction_bucket[BUCKETS_INDEX2(T[p1], 0)]] = p1;
        sa_sint_t p2 = SA[i - 2]; SA[--induction_bucket[BUCKETS_INDEX2(T[p2], 0)]] = p2;
        sa_sint_t p3 = SA[i - 3]; SA[--induction_bucket[BUCKETS_INDEX2(T[p3], 0)]] = p3;
    }

    for (j -= prefetch_distance + 3; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i]; SA[--induction_bucket[BUCKETS_INDEX2(T[p], 0)]] = p;
    }
}

static void libsais_radix_sort_lms_suffixes_8u_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 65536 && m >= 65536 && omp_get_dynamic() == 0)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_num_threads   = 1;
#endif
        if (omp_num_threads == 1)
        {
#ifdef AOCL_BWT
            libsais_radix_sort_lms_suffixes_8u(T, SA, &buckets[4 * ALPHABET_SIZE], (fast_sint_t)n - (fast_sint_t)m, (fast_sint_t)m);
#else
            libsais_radix_sort_lms_suffixes_8u(T, SA, &buckets[4 * ALPHABET_SIZE], (fast_sint_t)n - (fast_sint_t)m + 1, (fast_sint_t)m - 1);
#endif /* AOCL_BWT */
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                sa_sint_t * RESTRICT src_bucket = &buckets[4 * ALPHABET_SIZE];
                sa_sint_t * RESTRICT dst_bucket = thread_state[omp_thread_num].state.buckets;

                fast_sint_t i, j;
                for (i = BUCKETS_INDEX2(0, 0), j = BUCKETS_INDEX4(0, 1); i <= BUCKETS_INDEX2(ALPHABET_SIZE - 1, 0); i += BUCKETS_INDEX2(1, 0), j += BUCKETS_INDEX4(1, 0))
                {
                    dst_bucket[i] = src_bucket[i] - dst_bucket[j];
                }
            }

            {
                fast_sint_t t, omp_block_start = 0, omp_block_size = thread_state[omp_thread_num].state.m;
                for (t = omp_num_threads - 1; t >= omp_thread_num; --t) omp_block_start += thread_state[t].state.m;

                if (omp_block_start == (fast_sint_t)m && omp_block_size > 0)
                {
                    omp_block_start -= 1; omp_block_size -= 1;
                }

                libsais_radix_sort_lms_suffixes_8u(T, SA, thread_state[omp_thread_num].state.buckets, (fast_sint_t)n - omp_block_start, omp_block_size);
            }
        }
#endif
    }
}

static void libsais_radix_sort_lms_suffixes_32s_6k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + 2 * prefetch_distance + 3; i >= j; i -= 4)
    {
        libsais_prefetchr(&SA[i - 3 * prefetch_distance]);
        
        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 0]]);
        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 1]]);
        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 2]]);
        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 3]]);

        libsais_prefetchw(&induction_bucket[T[SA[i - prefetch_distance - 0]]]);
        libsais_prefetchw(&induction_bucket[T[SA[i - prefetch_distance - 1]]]);
        libsais_prefetchw(&induction_bucket[T[SA[i - prefetch_distance - 2]]]);
        libsais_prefetchw(&induction_bucket[T[SA[i - prefetch_distance - 3]]]);

        sa_sint_t p0 = SA[i - 0]; SA[--induction_bucket[T[p0]]] = p0;
        sa_sint_t p1 = SA[i - 1]; SA[--induction_bucket[T[p1]]] = p1;
        sa_sint_t p2 = SA[i - 2]; SA[--induction_bucket[T[p2]]] = p2;
        sa_sint_t p3 = SA[i - 3]; SA[--induction_bucket[T[p3]]] = p3;
    }

    for (j -= 2 * prefetch_distance + 3; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i]; SA[--induction_bucket[T[p]]] = p;
    }
}

static void libsais_radix_sort_lms_suffixes_32s_2k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + 2 * prefetch_distance + 3; i >= j; i -= 4)
    {
        libsais_prefetchr(&SA[i - 3 * prefetch_distance]);
        
        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 0]]);
        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 1]]);
        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 2]]);
        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 3]]);

        libsais_prefetchw(&induction_bucket[BUCKETS_INDEX2(T[SA[i - prefetch_distance - 0]], 0)]);
        libsais_prefetchw(&induction_bucket[BUCKETS_INDEX2(T[SA[i - prefetch_distance - 1]], 0)]);
        libsais_prefetchw(&induction_bucket[BUCKETS_INDEX2(T[SA[i - prefetch_distance - 2]], 0)]);
        libsais_prefetchw(&induction_bucket[BUCKETS_INDEX2(T[SA[i - prefetch_distance - 3]], 0)]);

        sa_sint_t p0 = SA[i - 0]; SA[--induction_bucket[BUCKETS_INDEX2(T[p0], 0)]] = p0;
        sa_sint_t p1 = SA[i - 1]; SA[--induction_bucket[BUCKETS_INDEX2(T[p1], 0)]] = p1;
        sa_sint_t p2 = SA[i - 2]; SA[--induction_bucket[BUCKETS_INDEX2(T[p2], 0)]] = p2;
        sa_sint_t p3 = SA[i - 3]; SA[--induction_bucket[BUCKETS_INDEX2(T[p3], 0)]] = p3;
    }

    for (j -= 2 * prefetch_distance + 3; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i]; SA[--induction_bucket[BUCKETS_INDEX2(T[p], 0)]] = p;
    }
}

#if defined(LIBSAIS_OPENMP)

static void libsais_radix_sort_lms_suffixes_32s_block_gather(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 3; i < j; i += 4)
    {
        libsais_prefetchr(&SA[i + 2 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i + prefetch_distance + 0]]);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 1]]);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 2]]);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 3]]);

        libsais_prefetchw(&cache[i + prefetch_distance]);

        cache[i + 0].symbol = T[cache[i + 0].index = SA[i + 0]];
        cache[i + 1].symbol = T[cache[i + 1].index = SA[i + 1]];
        cache[i + 2].symbol = T[cache[i + 2].index = SA[i + 2]];
        cache[i + 3].symbol = T[cache[i + 3].index = SA[i + 3]];
    }

    for (j += prefetch_distance + 3; i < j; i += 1)
    {
        cache[i].symbol = T[cache[i].index = SA[i]];
    }
}

static void libsais_radix_sort_lms_suffixes_32s_6k_block_sort(sa_sint_t * RESTRICT induction_bucket, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 3; i >= j; i -= 4)
    {
        libsais_prefetchw(&cache[i - 2 * prefetch_distance]);

        libsais_prefetchw(&induction_bucket[cache[i - prefetch_distance - 0].symbol]);
        libsais_prefetchw(&induction_bucket[cache[i - prefetch_distance - 1].symbol]);
        libsais_prefetchw(&induction_bucket[cache[i - prefetch_distance - 2].symbol]);
        libsais_prefetchw(&induction_bucket[cache[i - prefetch_distance - 3].symbol]);

        cache[i - 0].symbol = --induction_bucket[cache[i - 0].symbol];
        cache[i - 1].symbol = --induction_bucket[cache[i - 1].symbol];
        cache[i - 2].symbol = --induction_bucket[cache[i - 2].symbol];
        cache[i - 3].symbol = --induction_bucket[cache[i - 3].symbol];
    }

    for (j -= prefetch_distance + 3; i >= j; i -= 1)
    {
        cache[i].symbol = --induction_bucket[cache[i].symbol];
    }
}

static void libsais_radix_sort_lms_suffixes_32s_2k_block_sort(sa_sint_t * RESTRICT induction_bucket, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 3; i >= j; i -= 4)
    {
        libsais_prefetchw(&cache[i - 2 * prefetch_distance]);

        libsais_prefetchw(&induction_bucket[BUCKETS_INDEX2(cache[i - prefetch_distance - 0].symbol, 0)]);
        libsais_prefetchw(&induction_bucket[BUCKETS_INDEX2(cache[i - prefetch_distance - 1].symbol, 0)]);
        libsais_prefetchw(&induction_bucket[BUCKETS_INDEX2(cache[i - prefetch_distance - 2].symbol, 0)]);
        libsais_prefetchw(&induction_bucket[BUCKETS_INDEX2(cache[i - prefetch_distance - 3].symbol, 0)]);

        cache[i - 0].symbol = --induction_bucket[BUCKETS_INDEX2(cache[i - 0].symbol, 0)];
        cache[i - 1].symbol = --induction_bucket[BUCKETS_INDEX2(cache[i - 1].symbol, 0)];
        cache[i - 2].symbol = --induction_bucket[BUCKETS_INDEX2(cache[i - 2].symbol, 0)];
        cache[i - 3].symbol = --induction_bucket[BUCKETS_INDEX2(cache[i - 3].symbol, 0)];
    }

    for (j -= prefetch_distance + 3; i >= j; i -= 1)
    {
        cache[i].symbol = --induction_bucket[BUCKETS_INDEX2(cache[i].symbol, 0)];
    }
}

static void libsais_radix_sort_lms_suffixes_32s_6k_block_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 16384)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(cache);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            libsais_radix_sort_lms_suffixes_32s_6k(T, SA, induction_bucket, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                libsais_radix_sort_lms_suffixes_32s_block_gather(T, SA, cache - block_start, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                libsais_radix_sort_lms_suffixes_32s_6k_block_sort(induction_bucket, cache - block_start, block_start, block_size);
            }

            #pragma omp barrier

            {
                libsais_place_cached_suffixes(SA, cache - block_start, omp_block_start, omp_block_size);
            }
        }
#endif
    }
}

static void libsais_radix_sort_lms_suffixes_32s_2k_block_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 16384)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(cache);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            libsais_radix_sort_lms_suffixes_32s_2k(T, SA, induction_bucket, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                libsais_radix_sort_lms_suffixes_32s_block_gather(T, SA, cache - block_start, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                libsais_radix_sort_lms_suffixes_32s_2k_block_sort(induction_bucket, cache - block_start, block_start, block_size);
            }

            #pragma omp barrier

            {
                libsais_place_cached_suffixes(SA, cache - block_start, omp_block_start, omp_block_size);
            }
        }
#endif
    }
}

#endif

static void libsais_radix_sort_lms_suffixes_32s_6k_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t * RESTRICT induction_bucket, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    if (threads == 1 || m < 65536)
    {
#ifdef AOCL_BWT
        // "n-m" represents starting index, "m" represents number of elements that should be considered from starting index
        libsais_radix_sort_lms_suffixes_32s_6k(T, SA, induction_bucket, (fast_sint_t)n - (fast_sint_t)m, (fast_sint_t)m);
#else
        libsais_radix_sort_lms_suffixes_32s_6k(T, SA, induction_bucket, (fast_sint_t)n - (fast_sint_t)m + 1, (fast_sint_t)m - 1);
#endif /* AOCL_BWT */
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start, block_end;
        for (block_start = 0; block_start < (fast_sint_t)m - 1; block_start = block_end)
        {
            block_end = block_start + (fast_sint_t)threads * LIBSAIS_PER_THREAD_CACHE_SIZE; if (block_end >= m) { block_end = (fast_sint_t)m - 1; }

            libsais_radix_sort_lms_suffixes_32s_6k_block_omp(T, SA, induction_bucket, thread_state[0].state.cache, (fast_sint_t)n - block_end, block_end - block_start, threads);
        }
    }
#else
    UNUSED(thread_state);
#endif
}

static void libsais_radix_sort_lms_suffixes_32s_2k_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t * RESTRICT induction_bucket, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    if (threads == 1 || m < 65536)
    {
        libsais_radix_sort_lms_suffixes_32s_2k(T, SA, induction_bucket, (fast_sint_t)n - (fast_sint_t)m + 1, (fast_sint_t)m - 1);
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start, block_end;
        for (block_start = 0; block_start < (fast_sint_t)m - 1; block_start = block_end)
        {
            block_end = block_start + (fast_sint_t)threads * LIBSAIS_PER_THREAD_CACHE_SIZE; if (block_end >= m) { block_end = (fast_sint_t)m - 1; }

            libsais_radix_sort_lms_suffixes_32s_2k_block_omp(T, SA, induction_bucket, thread_state[0].state.cache, (fast_sint_t)n - block_end, block_end - block_start, threads);
        }
    }
#else
    UNUSED(thread_state);
#endif
}

static sa_sint_t libsais_radix_sort_lms_suffixes_32s_1k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t * RESTRICT buckets)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t             i = n - 2;
    sa_sint_t             m = 0;
    fast_uint_t           s = 1;
    fast_sint_t           c0 = T[n - 1];
    fast_sint_t           c1 = 0;
    fast_sint_t           c2 = 0;

    for (; i >= prefetch_distance + 3; i -= 4)
    {
        libsais_prefetchr(&T[i - 2 * prefetch_distance]);

        libsais_prefetchw(&buckets[T[i - prefetch_distance - 0]]);
        libsais_prefetchw(&buckets[T[i - prefetch_distance - 1]]);
        libsais_prefetchw(&buckets[T[i - prefetch_distance - 2]]);
        libsais_prefetchw(&buckets[T[i - prefetch_distance - 3]]);

        c1 = T[i - 0]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); 
        if ((s & 3) == 1) { SA[--buckets[c2 = c0]] = i + 1; m++; }
        
        c0 = T[i - 1]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); 
        if ((s & 3) == 1) { SA[--buckets[c2 = c1]] = i - 0; m++; }

        c1 = T[i - 2]; s = (s << 1) + (fast_uint_t)(c1 > (c0 - (fast_sint_t)(s & 1))); 
        if ((s & 3) == 1) { SA[--buckets[c2 = c0]] = i - 1; m++; }

        c0 = T[i - 3]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); 
        if ((s & 3) == 1) { SA[--buckets[c2 = c1]] = i - 2; m++; }
    }

    for (; i >= 0; i -= 1)
    {
        c1 = c0; c0 = T[i]; s = (s << 1) + (fast_uint_t)(c0 > (c1 - (fast_sint_t)(s & 1))); 
        if ((s & 3) == 1) { SA[--buckets[c2 = c1]] = i + 1; m++; }
    }

    if (m > 1)
    {
        SA[buckets[c2]] = 0;
    }

    return m;
}

static void libsais_radix_sort_set_markers_32s_6k(sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 3; i < j; i += 4)
    {
        libsais_prefetchr(&induction_bucket[i + 2 * prefetch_distance]);

        libsais_prefetchw(&SA[induction_bucket[i + prefetch_distance + 0]]);
        libsais_prefetchw(&SA[induction_bucket[i + prefetch_distance + 1]]);
        libsais_prefetchw(&SA[induction_bucket[i + prefetch_distance + 2]]);
        libsais_prefetchw(&SA[induction_bucket[i + prefetch_distance + 3]]);

        SA[induction_bucket[i + 0]] |= SAINT_MIN;
        SA[induction_bucket[i + 1]] |= SAINT_MIN;
        SA[induction_bucket[i + 2]] |= SAINT_MIN;
        SA[induction_bucket[i + 3]] |= SAINT_MIN;
    }

    for (j += prefetch_distance + 3; i < j; i += 1)
    {
        SA[induction_bucket[i]] |= SAINT_MIN;
    }
}

static void libsais_radix_sort_set_markers_32s_4k(sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 3; i < j; i += 4)
    {
        libsais_prefetchr(&induction_bucket[BUCKETS_INDEX2(i + 2 * prefetch_distance, 0)]);

        libsais_prefetchw(&SA[induction_bucket[BUCKETS_INDEX2(i + prefetch_distance + 0, 0)]]);
        libsais_prefetchw(&SA[induction_bucket[BUCKETS_INDEX2(i + prefetch_distance + 1, 0)]]);
        libsais_prefetchw(&SA[induction_bucket[BUCKETS_INDEX2(i + prefetch_distance + 2, 0)]]);
        libsais_prefetchw(&SA[induction_bucket[BUCKETS_INDEX2(i + prefetch_distance + 3, 0)]]);

        SA[induction_bucket[BUCKETS_INDEX2(i + 0, 0)]] |= SUFFIX_GROUP_MARKER;
        SA[induction_bucket[BUCKETS_INDEX2(i + 1, 0)]] |= SUFFIX_GROUP_MARKER;
        SA[induction_bucket[BUCKETS_INDEX2(i + 2, 0)]] |= SUFFIX_GROUP_MARKER;
        SA[induction_bucket[BUCKETS_INDEX2(i + 3, 0)]] |= SUFFIX_GROUP_MARKER;
    }

    for (j += prefetch_distance + 3; i < j; i += 1)
    {
        SA[induction_bucket[BUCKETS_INDEX2(i, 0)]] |= SUFFIX_GROUP_MARKER;
    }
}

static void libsais_radix_sort_set_markers_32s_6k_omp(sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT induction_bucket, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && k >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
        fast_sint_t omp_block_stride  = (((fast_sint_t)k - 1) / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : (fast_sint_t)k - 1 - omp_block_start;
#else
        UNUSED(threads);

        fast_sint_t omp_block_start   = 0;
        fast_sint_t omp_block_size    = (fast_sint_t)k - 1;
#endif

        libsais_radix_sort_set_markers_32s_6k(SA, induction_bucket, omp_block_start, omp_block_size);
    }
}

static void libsais_radix_sort_set_markers_32s_4k_omp(sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT induction_bucket, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && k >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
        fast_sint_t omp_block_stride  = (((fast_sint_t)k - 1) / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : (fast_sint_t)k - 1 - omp_block_start;
#else
        UNUSED(threads);

        fast_sint_t omp_block_start   = 0;
        fast_sint_t omp_block_size    = (fast_sint_t)k - 1;
#endif

        libsais_radix_sort_set_markers_32s_4k(SA, induction_bucket, omp_block_start, omp_block_size);
    }
}

static void libsais_initialize_buckets_for_partial_sorting_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT buckets, sa_sint_t first_lms_suffix, sa_sint_t left_suffixes_count)
{
    sa_sint_t * RESTRICT temp_bucket = &buckets[4 * ALPHABET_SIZE];

#ifndef AOCL_BWT
    buckets[BUCKETS_INDEX4((fast_uint_t)T[first_lms_suffix], 1)]++;
#endif /* AOCL_BWT */

    fast_sint_t i, j; sa_sint_t sum0 = left_suffixes_count + 1, sum1 = 0;
#ifdef AOCL_BWT
    sum0 = left_suffixes_count;
#endif /* AOCL_BWT */
    for (i = BUCKETS_INDEX4(0, 0), j = BUCKETS_INDEX2(0, 0); i <= BUCKETS_INDEX4(ALPHABET_SIZE - 1, 0); i += BUCKETS_INDEX4(1, 0), j += BUCKETS_INDEX2(1, 0))
    { 
        temp_bucket[j + BUCKETS_INDEX2(0, 0)] = sum0;

        sum0 += buckets[i + BUCKETS_INDEX4(0, 0)] + buckets[i + BUCKETS_INDEX4(0, 2)];
        sum1 += buckets[i + BUCKETS_INDEX4(0, 1)];

        buckets[j + BUCKETS_INDEX2(0, 0)] = sum0;
        buckets[j + BUCKETS_INDEX2(0, 1)] = sum1;
    }
}

static void libsais_initialize_buckets_for_partial_sorting_32s_6k(const sa_sint_t * RESTRICT T, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t first_lms_suffix, sa_sint_t left_suffixes_count)
{
    // even numbers contain tails of S types, odd contains LMS sums
    sa_sint_t * RESTRICT temp_bucket = &buckets[4 * (fast_sint_t)k];

    fast_sint_t i, j; sa_sint_t sum0 = left_suffixes_count + 1, sum1 = 0, sum2 = 0;
#ifndef AOCL_BWT
    for (first_lms_suffix = T[first_lms_suffix], i = BUCKETS_INDEX4(0, 0), j = BUCKETS_INDEX2(0, 0); i <= BUCKETS_INDEX4((fast_sint_t)first_lms_suffix - 1, 0); i += BUCKETS_INDEX4(1, 0), j += BUCKETS_INDEX2(1, 0))
    {
        sa_sint_t SS = buckets[i + BUCKETS_INDEX4(0, 0)];
        sa_sint_t LS = buckets[i + BUCKETS_INDEX4(0, 1)];
        sa_sint_t SL = buckets[i + BUCKETS_INDEX4(0, 2)];
        sa_sint_t LL = buckets[i + BUCKETS_INDEX4(0, 3)];

        buckets[i + BUCKETS_INDEX4(0, 0)] = sum0;
        buckets[i + BUCKETS_INDEX4(0, 1)] = sum2;
        buckets[i + BUCKETS_INDEX4(0, 2)] = 0;
        buckets[i + BUCKETS_INDEX4(0, 3)] = 0;

        sum0 += SS + SL; sum1 += LS; sum2 += LS + LL;

        temp_bucket[j + BUCKETS_INDEX2(0, 0)] = sum0;
        temp_bucket[j + BUCKETS_INDEX2(0, 1)] = sum1;
    }
    // buckets[] even indexes contain S types head and odds indexes contain heads of L types
    for (sum1 += 1; i <= BUCKETS_INDEX4((fast_sint_t)k - 1, 0); i += BUCKETS_INDEX4(1, 0), j += BUCKETS_INDEX2(1, 0))
    { 
        sa_sint_t SS = buckets[i + BUCKETS_INDEX4(0, 0)];
        sa_sint_t LS = buckets[i + BUCKETS_INDEX4(0, 1)];
        sa_sint_t SL = buckets[i + BUCKETS_INDEX4(0, 2)];
        sa_sint_t LL = buckets[i + BUCKETS_INDEX4(0, 3)];

        buckets[i + BUCKETS_INDEX4(0, 0)] = sum0;
        buckets[i + BUCKETS_INDEX4(0, 1)] = sum2;
        buckets[i + BUCKETS_INDEX4(0, 2)] = 0;
        buckets[i + BUCKETS_INDEX4(0, 3)] = 0;

        sum0 += SS + SL; sum1 += LS; sum2 += LS + LL;

        temp_bucket[j + BUCKETS_INDEX2(0, 0)] = sum0;
        temp_bucket[j + BUCKETS_INDEX2(0, 1)] = sum1;
    }
#else
    sum0 = left_suffixes_count;
    for (i = BUCKETS_INDEX4(0, 0), j = BUCKETS_INDEX2(0, 0); i <= BUCKETS_INDEX4((fast_sint_t)k - 1, 0); i += BUCKETS_INDEX4(1, 0), j += BUCKETS_INDEX2(1, 0))
    { 
        sa_sint_t SS = buckets[i + BUCKETS_INDEX4(0, 0)];
        sa_sint_t LS = buckets[i + BUCKETS_INDEX4(0, 1)];
        sa_sint_t SL = buckets[i + BUCKETS_INDEX4(0, 2)];
        sa_sint_t LL = buckets[i + BUCKETS_INDEX4(0, 3)];

        buckets[i + BUCKETS_INDEX4(0, 0)] = sum0;
        buckets[i + BUCKETS_INDEX4(0, 1)] = sum2;
        buckets[i + BUCKETS_INDEX4(0, 2)] = 0;
        buckets[i + BUCKETS_INDEX4(0, 3)] = 0;

        sum0 += SS + SL; sum1 += LS; sum2 += LS + LL;

        temp_bucket[j + BUCKETS_INDEX2(0, 0)] = sum0;
        temp_bucket[j + BUCKETS_INDEX2(0, 1)] = sum1;
    }
#endif /* AOCL_BWT */
}

#ifndef AOCL_BWT
static sa_sint_t libsais_partial_sorting_scan_left_to_right_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, sa_sint_t d, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT induction_bucket = &buckets[4 * ALPHABET_SIZE];
    sa_sint_t * RESTRICT distinct_names   = &buckets[2 * ALPHABET_SIZE];

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchr(&SA[i + 2 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i + prefetch_distance + 0] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 0] & SAINT_MAX] - 2);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 1] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 1] & SAINT_MAX] - 2);

        sa_sint_t p0 = SA[i + 0]; d += (p0 < 0); p0 &= SAINT_MAX; sa_sint_t v0 = BUCKETS_INDEX2(T[p0 - 1], T[p0 - 2] >= T[p0 - 1]);
        SA[induction_bucket[v0]++] = (p0 - 1) | ((sa_sint_t)(distinct_names[v0] != d) << (SAINT_BIT - 1)); distinct_names[v0] = d;

        sa_sint_t p1 = SA[i + 1]; d += (p1 < 0); p1 &= SAINT_MAX; sa_sint_t v1 = BUCKETS_INDEX2(T[p1 - 1], T[p1 - 2] >= T[p1 - 1]);
        SA[induction_bucket[v1]++] = (p1 - 1) | ((sa_sint_t)(distinct_names[v1] != d) << (SAINT_BIT - 1)); distinct_names[v1] = d;
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t p = SA[i]; d += (p < 0); p &= SAINT_MAX; sa_sint_t v = BUCKETS_INDEX2(T[p - 1], T[p - 2] >= T[p - 1]);
        SA[induction_bucket[v]++] = (p - 1) | ((sa_sint_t)(distinct_names[v] != d) << (SAINT_BIT - 1)); distinct_names[v] = d;
    }

    return d;
}
#else
static sa_sint_t AOCL_libsais_partial_sorting_scan_left_to_right_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, sa_sint_t d, fast_sint_t omp_block_start, fast_sint_t omp_block_size, fast_sint_t n)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT induction_bucket = &buckets[4 * ALPHABET_SIZE];
    sa_sint_t * RESTRICT distinct_names   = &buckets[2 * ALPHABET_SIZE];
    d = 1;
    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchr(&SA[i + 2 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i + prefetch_distance + 0] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 0] & SAINT_MAX] - 2);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 1] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 1] & SAINT_MAX] - 2);

        sa_sint_t p0 = SA[i + 0];
        d += (p0 < 0);
        p0 &= SAINT_MAX;

        sa_sint_t v0 = BUCKETS_INDEX2(T[p0 - 1], T[p0 - 2] >= T[p0 - 1]);
        SA[induction_bucket[v0]++] = MOD(p0 - 1, n) | ((sa_sint_t)(distinct_names[v0] != d) << (SAINT_BIT - 1));

        distinct_names[v0] = d;

        sa_sint_t p1 = SA[i + 1];
        d += (p1 < 0);
        p1 &= SAINT_MAX;

        sa_sint_t v1 = BUCKETS_INDEX2(T[p1 - 1], T[p1 - 2] >= T[p1 - 1]);
        SA[induction_bucket[v1]++] = MOD(p1 - 1, n) | ((sa_sint_t)(distinct_names[v1] != d) << (SAINT_BIT - 1));

        distinct_names[v1] = d;
    }
    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t p = SA[i];
        d += (p < 0);
        p &= SAINT_MAX;
        sa_sint_t v = BUCKETS_INDEX2(T[p - 1], T[p - 2] >= T[p - 1]);
        SA[induction_bucket[v]++] = MOD(p - 1, n) | ((sa_sint_t)(distinct_names[v] != d) << (SAINT_BIT - 1));
        distinct_names[v] = d;
    }

    return d;
}
#endif /* AOCL_BWT */

#if defined(LIBSAIS_OPENMP)

static void libsais_partial_sorting_scan_left_to_right_8u_block_prepare(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size, LIBSAIS_THREAD_STATE * RESTRICT state)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT induction_bucket = &buckets[0 * ALPHABET_SIZE];
    sa_sint_t * RESTRICT distinct_names   = &buckets[2 * ALPHABET_SIZE];

    memset(induction_bucket, 0, (size_t)2 * k * sizeof(sa_sint_t));
    memset(distinct_names  , 0, (size_t)2 * k * sizeof(sa_sint_t));

    fast_sint_t i, j, count = 0; sa_sint_t d = 1;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchr(&SA[i + 2 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i + prefetch_distance + 0] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 0] & SAINT_MAX] - 2);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 1] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 1] & SAINT_MAX] - 2);

        sa_sint_t p0 = cache[count].index = SA[i + 0]; d += (p0 < 0); p0 &= SAINT_MAX; sa_sint_t v0 = cache[count++].symbol = BUCKETS_INDEX2(T[p0 - 1], T[p0 - 2] >= T[p0 - 1]); induction_bucket[v0]++; distinct_names[v0] = d;
        sa_sint_t p1 = cache[count].index = SA[i + 1]; d += (p1 < 0); p1 &= SAINT_MAX; sa_sint_t v1 = cache[count++].symbol = BUCKETS_INDEX2(T[p1 - 1], T[p1 - 2] >= T[p1 - 1]); induction_bucket[v1]++; distinct_names[v1] = d;
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t p = cache[count].index = SA[i]; d += (p < 0); p &= SAINT_MAX; sa_sint_t v = cache[count++].symbol = BUCKETS_INDEX2(T[p - 1], T[p - 2] >= T[p - 1]); induction_bucket[v]++; distinct_names[v] = d;
    }

    state[0].state.position   = (fast_sint_t)d - 1;
    state[0].state.count      = count;
}

static void libsais_partial_sorting_scan_left_to_right_8u_block_place(sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t count, sa_sint_t d)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT induction_bucket = &buckets[0 * ALPHABET_SIZE];
    sa_sint_t * RESTRICT distinct_names   = &buckets[2 * ALPHABET_SIZE];

    fast_sint_t i, j;
    for (i = 0, j = count - 1; i < j; i += 2)
    {
        libsais_prefetchr(&cache[i + prefetch_distance]);

        sa_sint_t p0 = cache[i + 0].index; d += (p0 < 0); sa_sint_t v0 = cache[i + 0].symbol;
        SA[induction_bucket[v0]++] = (p0 - 1) | ((sa_sint_t)(distinct_names[v0] != d) << (SAINT_BIT - 1)); distinct_names[v0] = d;

        sa_sint_t p1 = cache[i + 1].index; d += (p1 < 0); sa_sint_t v1 = cache[i + 1].symbol;
        SA[induction_bucket[v1]++] = (p1 - 1) | ((sa_sint_t)(distinct_names[v1] != d) << (SAINT_BIT - 1)); distinct_names[v1] = d;
    }

    for (j += 1; i < j; i += 1)
    {
        sa_sint_t p = cache[i].index; d += (p < 0); sa_sint_t v = cache[i].symbol;
        SA[induction_bucket[v]++] = (p - 1) | ((sa_sint_t)(distinct_names[v] != d) << (SAINT_BIT - 1)); distinct_names[v] = d;
    }
}

static sa_sint_t libsais_partial_sorting_scan_left_to_right_8u_block_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t d, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 64 * max(k, 256) && omp_get_dynamic() == 0)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(k); UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            d = libsais_partial_sorting_scan_left_to_right_8u(T, SA, buckets, d, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                libsais_partial_sorting_scan_left_to_right_8u_block_prepare(T, SA, k, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, omp_block_start, omp_block_size, &thread_state[omp_thread_num]);
            }

            #pragma omp barrier

            #pragma omp master
            {
                sa_sint_t * RESTRICT induction_bucket = &buckets[4 * ALPHABET_SIZE];
                sa_sint_t * RESTRICT distinct_names   = &buckets[2 * ALPHABET_SIZE];

                fast_sint_t t;
                for (t = 0; t < omp_num_threads; ++t)
                {
                    sa_sint_t * RESTRICT temp_induction_bucket    = &thread_state[t].state.buckets[0 * ALPHABET_SIZE];
                    sa_sint_t * RESTRICT temp_distinct_names      = &thread_state[t].state.buckets[2 * ALPHABET_SIZE];

                    fast_sint_t c; 
                    for (c = 0; c < 2 * k; c += 1) { sa_sint_t A = induction_bucket[c], B = temp_induction_bucket[c]; induction_bucket[c] = A + B; temp_induction_bucket[c] = A; }

                    for (d -= 1, c = 0; c < 2 * k; c += 1) { sa_sint_t A = distinct_names[c], B = temp_distinct_names[c], D = B + d; distinct_names[c] = B > 0 ? D : A; temp_distinct_names[c] = A; }
                    d += 1 + (sa_sint_t)thread_state[t].state.position; thread_state[t].state.position = (fast_sint_t)d - thread_state[t].state.position;
                }
            }

            #pragma omp barrier

            {
                libsais_partial_sorting_scan_left_to_right_8u_block_place(SA, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, thread_state[omp_thread_num].state.count, (sa_sint_t)thread_state[omp_thread_num].state.position);
            }
        }
#endif
    }

    return d;
}

#endif

static sa_sint_t libsais_partial_sorting_scan_left_to_right_8u_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t left_suffixes_count, sa_sint_t d, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#ifndef AOCL_BWT
    sa_sint_t * RESTRICT induction_bucket = &buckets[4 * ALPHABET_SIZE];
    sa_sint_t * RESTRICT distinct_names   = &buckets[2 * ALPHABET_SIZE];

    // In left to right scan, left most elements are lexicographically the smallest, rightmost are lexicographically largest,
    // in below statement senital character which is the lowest of all characters is considered, so for BWT we want to ignore below two statements.
    SA[induction_bucket[BUCKETS_INDEX2(T[n - 1], T[n - 2] >= T[n - 1])]++] = (n - 1) | SAINT_MIN;
    distinct_names[BUCKETS_INDEX2(T[n - 1], T[n - 2] >= T[n - 1])] = ++d;
#endif /* AOCL_BWT */

    if (threads == 1 || left_suffixes_count < 65536)
    {
#ifdef AOCL_BWT
        d = AOCL_libsais_partial_sorting_scan_left_to_right_8u(T, SA, buckets, d, 0, left_suffixes_count, n);
#else
        d = libsais_partial_sorting_scan_left_to_right_8u(T, SA, buckets, d, 0, left_suffixes_count);
#endif /* AOCL_BWT */
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start;
        for (block_start = 0; block_start < left_suffixes_count; )
        {
            if (SA[block_start] == 0)
            {
                block_start++;
            }
            else
            {
                fast_sint_t block_max_end = block_start + ((fast_sint_t)threads) * (LIBSAIS_PER_THREAD_CACHE_SIZE - 16 * (fast_sint_t)threads); if (block_max_end > left_suffixes_count) { block_max_end = left_suffixes_count;}
                fast_sint_t block_end     = block_start + 1; while (block_end < block_max_end && SA[block_end] != 0) { block_end++; }
                fast_sint_t block_size    = block_end - block_start;

                if (block_size < 32)
                {
                    for (; block_start < block_end; block_start += 1)
                    {
                        sa_sint_t p = SA[block_start]; d += (p < 0); p &= SAINT_MAX; sa_sint_t v = BUCKETS_INDEX2(T[p - 1], T[p - 2] >= T[p - 1]);
                        SA[induction_bucket[v]++] = (p - 1) | ((sa_sint_t)(distinct_names[v] != d) << (SAINT_BIT - 1)); distinct_names[v] = d;
                    }
                }
                else
                {
                    d = libsais_partial_sorting_scan_left_to_right_8u_block_omp(T, SA, k, buckets, d, block_start, block_size, threads, thread_state);
                    block_start = block_end;
                }
            }
        }
    }
#else
    UNUSED(k); UNUSED(thread_state);
#endif

    return d;
}


#ifndef AOCL_BWT
static sa_sint_t libsais_partial_sorting_scan_left_to_right_32s_6k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, sa_sint_t d, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - 2 * prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchr(&SA[i + 3 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i + 2 * prefetch_distance + 0] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + 2 * prefetch_distance + 0] & SAINT_MAX] - 2);
        libsais_prefetchr(&T[SA[i + 2 * prefetch_distance + 1] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + 2 * prefetch_distance + 1] & SAINT_MAX] - 2);

        sa_sint_t p0 = SA[i + prefetch_distance + 0] & SAINT_MAX; sa_sint_t v0 = BUCKETS_INDEX4(T[p0 - (p0 > 0)], 0); libsais_prefetchw(&buckets[v0]);
        sa_sint_t p1 = SA[i + prefetch_distance + 1] & SAINT_MAX; sa_sint_t v1 = BUCKETS_INDEX4(T[p1 - (p1 > 0)], 0); libsais_prefetchw(&buckets[v1]);

        sa_sint_t p2 = SA[i + 0]; d += (p2 < 0); p2 &= SAINT_MAX; sa_sint_t v2 = BUCKETS_INDEX4(T[p2 - 1], T[p2 - 2] >= T[p2 - 1]);
        SA[buckets[v2]++] = (p2 - 1) | ((sa_sint_t)(buckets[2 + v2] != d) << (SAINT_BIT - 1)); buckets[2 + v2] = d;

        sa_sint_t p3 = SA[i + 1]; d += (p3 < 0); p3 &= SAINT_MAX; sa_sint_t v3 = BUCKETS_INDEX4(T[p3 - 1], T[p3 - 2] >= T[p3 - 1]);
        SA[buckets[v3]++] = (p3 - 1) | ((sa_sint_t)(buckets[2 + v3] != d) << (SAINT_BIT - 1)); buckets[2 + v3] = d;
    }

    for (j += 2 * prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t p = SA[i]; d += (p < 0); p &= SAINT_MAX; sa_sint_t v = BUCKETS_INDEX4(T[p - 1], T[p - 2] >= T[p - 1]);
        SA[buckets[v]++] = (p - 1) | ((sa_sint_t)(buckets[2 + v] != d) << (SAINT_BIT - 1)); buckets[2 + v] = d;
    }

    return d;
}
#else
static sa_sint_t AOCL_libsais_partial_sorting_scan_left_to_right_32s_6k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, sa_sint_t d, fast_sint_t omp_block_start, fast_sint_t omp_block_size, fast_sint_t n)
{
    const fast_sint_t prefetch_distance = 32;
    d = 1;
    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - 2 * prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchr(&SA[i + 3 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i + 2 * prefetch_distance + 0] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + 2 * prefetch_distance + 0] & SAINT_MAX] - 2);
        libsais_prefetchr(&T[SA[i + 2 * prefetch_distance + 1] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + 2 * prefetch_distance + 1] & SAINT_MAX] - 2);

        sa_sint_t p0 = SA[i + prefetch_distance + 0] & SAINT_MAX; sa_sint_t v0 = BUCKETS_INDEX4(T[p0 - 1], 0); libsais_prefetchw(&buckets[v0]);
        sa_sint_t p1 = SA[i + prefetch_distance + 1] & SAINT_MAX; sa_sint_t v1 = BUCKETS_INDEX4(T[p1 - 1], 0); libsais_prefetchw(&buckets[v1]);

        sa_sint_t p2 = SA[i + 0]; d += (p2 < 0); p2 &= SAINT_MAX; sa_sint_t v2 = BUCKETS_INDEX4(T[p2 - 1], T[p2 - 2] >= T[p2 - 1]);
        SA[buckets[v2]++] = MOD(p2 - 1, n) | ((sa_sint_t)(buckets[2 + v2] != d) << (SAINT_BIT - 1)); buckets[2 + v2] = d;

        sa_sint_t p3 = SA[i + 1]; d += (p3 < 0); p3 &= SAINT_MAX; sa_sint_t v3 = BUCKETS_INDEX4(T[p3 - 1], T[p3 - 2] >= T[p3 - 1]);
        SA[buckets[v3]++] = MOD(p3 - 1, n) | ((sa_sint_t)(buckets[2 + v3] != d) << (SAINT_BIT - 1)); buckets[2 + v3] = d;
    }

    for (j += 2 * prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t p = SA[i]; d += (p < 0); p &= SAINT_MAX; sa_sint_t v = BUCKETS_INDEX4(T[p - 1], T[p - 2] >= T[p - 1]);
        SA[buckets[v]++] = MOD(p - 1, n) | ((sa_sint_t)(buckets[2 + v] != d) << (SAINT_BIT - 1)); buckets[2 + v] = d;
    }

    return d;
}
#endif /* AOCL_BWT */

static sa_sint_t libsais_partial_sorting_scan_left_to_right_32s_4k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t d, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT induction_bucket = &buckets[2 * (fast_sint_t)k];
    sa_sint_t * RESTRICT distinct_names   = &buckets[0 * (fast_sint_t)k];

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - 2 * prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&SA[i + 3 * prefetch_distance]);

        sa_sint_t s0 = SA[i + 2 * prefetch_distance + 0]; const sa_sint_t * Ts0 = &T[s0 & ~SUFFIX_GROUP_MARKER] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i + 2 * prefetch_distance + 1]; const sa_sint_t * Ts1 = &T[s1 & ~SUFFIX_GROUP_MARKER] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);
        sa_sint_t s2 = SA[i + 1 * prefetch_distance + 0]; if (s2 > 0) { const fast_sint_t Ts2 = T[(s2 & ~SUFFIX_GROUP_MARKER) - 1]; libsais_prefetchw(&induction_bucket[Ts2]); libsais_prefetchw(&distinct_names[BUCKETS_INDEX2(Ts2, 0)]); }
        sa_sint_t s3 = SA[i + 1 * prefetch_distance + 1]; if (s3 > 0) { const fast_sint_t Ts3 = T[(s3 & ~SUFFIX_GROUP_MARKER) - 1]; libsais_prefetchw(&induction_bucket[Ts3]); libsais_prefetchw(&distinct_names[BUCKETS_INDEX2(Ts3, 0)]); }

        sa_sint_t p0 = SA[i + 0]; SA[i + 0] = p0 & SAINT_MAX;
        if (p0 > 0)
        {
            SA[i + 0] = 0; d += (p0 >> (SUFFIX_GROUP_BIT - 1)); p0 &= ~SUFFIX_GROUP_MARKER; sa_sint_t v0 = BUCKETS_INDEX2(T[p0 - 1], T[p0 - 2] < T[p0 - 1]);
            SA[induction_bucket[T[p0 - 1]]++] = (p0 - 1) | ((sa_sint_t)(T[p0 - 2] < T[p0 - 1]) << (SAINT_BIT - 1)) | ((sa_sint_t)(distinct_names[v0] != d) << (SUFFIX_GROUP_BIT - 1)); distinct_names[v0] = d;
        }

        sa_sint_t p1 = SA[i + 1]; SA[i + 1] = p1 & SAINT_MAX;
        if (p1 > 0)
        {
            SA[i + 1] = 0; d += (p1 >> (SUFFIX_GROUP_BIT - 1)); p1 &= ~SUFFIX_GROUP_MARKER; sa_sint_t v1 = BUCKETS_INDEX2(T[p1 - 1], T[p1 - 2] < T[p1 - 1]);
            SA[induction_bucket[T[p1 - 1]]++] = (p1 - 1) | ((sa_sint_t)(T[p1 - 2] < T[p1 - 1]) << (SAINT_BIT - 1)) | ((sa_sint_t)(distinct_names[v1] != d) << (SUFFIX_GROUP_BIT - 1)); distinct_names[v1] = d;
        }
    }

    for (j += 2 * prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t p = SA[i]; SA[i] = p & SAINT_MAX;
        if (p > 0)
        {
            SA[i] = 0; d += (p >> (SUFFIX_GROUP_BIT - 1)); p &= ~SUFFIX_GROUP_MARKER; sa_sint_t v = BUCKETS_INDEX2(T[p - 1], T[p - 2] < T[p - 1]);
            SA[induction_bucket[T[p - 1]]++] = (p - 1) | ((sa_sint_t)(T[p - 2] < T[p - 1]) << (SAINT_BIT - 1)) | ((sa_sint_t)(distinct_names[v] != d) << (SUFFIX_GROUP_BIT - 1)); distinct_names[v] = d;
        }
    }

    return d;
}

static void libsais_partial_sorting_scan_left_to_right_32s_1k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - 2 * prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&SA[i + 3 * prefetch_distance]);

        sa_sint_t s0 = SA[i + 2 * prefetch_distance + 0]; const sa_sint_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i + 2 * prefetch_distance + 1]; const sa_sint_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);
        sa_sint_t s2 = SA[i + 1 * prefetch_distance + 0]; if (s2 > 0) { libsais_prefetchw(&induction_bucket[T[s2 - 1]]); libsais_prefetchr(&T[s2] - 2); }
        sa_sint_t s3 = SA[i + 1 * prefetch_distance + 1]; if (s3 > 0) { libsais_prefetchw(&induction_bucket[T[s3 - 1]]); libsais_prefetchr(&T[s3] - 2); }

        sa_sint_t p0 = SA[i + 0]; SA[i + 0] = p0 & SAINT_MAX; if (p0 > 0) { SA[i + 0] = 0; SA[induction_bucket[T[p0 - 1]]++] = (p0 - 1) | ((sa_sint_t)(T[p0 - 2] < T[p0 - 1]) << (SAINT_BIT - 1)); }
        sa_sint_t p1 = SA[i + 1]; SA[i + 1] = p1 & SAINT_MAX; if (p1 > 0) { SA[i + 1] = 0; SA[induction_bucket[T[p1 - 1]]++] = (p1 - 1) | ((sa_sint_t)(T[p1 - 2] < T[p1 - 1]) << (SAINT_BIT - 1)); }
    }

    for (j += 2 * prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t p = SA[i]; SA[i] = p & SAINT_MAX; if (p > 0) { SA[i] = 0; SA[induction_bucket[T[p - 1]]++] = (p - 1) | ((sa_sint_t)(T[p - 2] < T[p - 1]) << (SAINT_BIT - 1)); }
    }
}

#if defined(LIBSAIS_OPENMP)

static void libsais_partial_sorting_scan_left_to_right_32s_6k_block_gather(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchr(&SA[i + 2 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i + prefetch_distance + 0] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 0] & SAINT_MAX] - 2);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 1] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 1] & SAINT_MAX] - 2);

        libsais_prefetchw(&cache[i + prefetch_distance]);

        sa_sint_t p0 = cache[i + 0].index = SA[i + 0]; sa_sint_t symbol0 = 0; p0 &= SAINT_MAX; if (p0 != 0) { symbol0 = BUCKETS_INDEX4(T[p0 - 1], T[p0 - 2] >= T[p0 - 1]); } cache[i + 0].symbol = symbol0;
        sa_sint_t p1 = cache[i + 1].index = SA[i + 1]; sa_sint_t symbol1 = 0; p1 &= SAINT_MAX; if (p1 != 0) { symbol1 = BUCKETS_INDEX4(T[p1 - 1], T[p1 - 2] >= T[p1 - 1]); } cache[i + 1].symbol = symbol1;
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t p = cache[i].index = SA[i]; sa_sint_t symbol = 0; p &= SAINT_MAX; if (p != 0) { symbol = BUCKETS_INDEX4(T[p - 1], T[p - 2] >= T[p - 1]); } cache[i].symbol = symbol;
    }
}

static void libsais_partial_sorting_scan_left_to_right_32s_4k_block_gather(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&SA[i + 2 * prefetch_distance]);

        sa_sint_t s0 = SA[i + prefetch_distance + 0]; const sa_sint_t * Ts0 = &T[s0 & ~SUFFIX_GROUP_MARKER] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i + prefetch_distance + 1]; const sa_sint_t * Ts1 = &T[s1 & ~SUFFIX_GROUP_MARKER] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

        libsais_prefetchw(&cache[i + prefetch_distance]);

        sa_sint_t symbol0 = SAINT_MIN, p0 = SA[i + 0]; if (p0 > 0) { cache[i + 0].index = p0; p0 &= ~SUFFIX_GROUP_MARKER; symbol0 = BUCKETS_INDEX2(T[p0 - 1], T[p0 - 2] < T[p0 - 1]); p0 = 0; } cache[i + 0].symbol = symbol0; SA[i + 0] = p0 & SAINT_MAX;
        sa_sint_t symbol1 = SAINT_MIN, p1 = SA[i + 1]; if (p1 > 0) { cache[i + 1].index = p1; p1 &= ~SUFFIX_GROUP_MARKER; symbol1 = BUCKETS_INDEX2(T[p1 - 1], T[p1 - 2] < T[p1 - 1]); p1 = 0; } cache[i + 1].symbol = symbol1; SA[i + 1] = p1 & SAINT_MAX;
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t symbol = SAINT_MIN, p = SA[i]; if (p > 0) { cache[i].index = p; p &= ~SUFFIX_GROUP_MARKER; symbol = BUCKETS_INDEX2(T[p - 1], T[p - 2] < T[p - 1]); p = 0; } cache[i].symbol = symbol; SA[i] = p & SAINT_MAX;
    }
}

static void libsais_partial_sorting_scan_left_to_right_32s_1k_block_gather(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&SA[i + 2 * prefetch_distance]);

        sa_sint_t s0 = SA[i + prefetch_distance + 0]; const sa_sint_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i + prefetch_distance + 1]; const sa_sint_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

        libsais_prefetchw(&cache[i + prefetch_distance]);

        sa_sint_t symbol0 = SAINT_MIN, p0 = SA[i + 0]; if (p0 > 0) { cache[i + 0].index = (p0 - 1) | ((sa_sint_t)(T[p0 - 2] < T[p0 - 1]) << (SAINT_BIT - 1)); symbol0 = T[p0 - 1]; p0 = 0; } cache[i + 0].symbol = symbol0; SA[i + 0] = p0 & SAINT_MAX;
        sa_sint_t symbol1 = SAINT_MIN, p1 = SA[i + 1]; if (p1 > 0) { cache[i + 1].index = (p1 - 1) | ((sa_sint_t)(T[p1 - 2] < T[p1 - 1]) << (SAINT_BIT - 1)); symbol1 = T[p1 - 1]; p1 = 0; } cache[i + 1].symbol = symbol1; SA[i + 1] = p1 & SAINT_MAX;
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t symbol = SAINT_MIN, p = SA[i]; if (p > 0) { cache[i].index = (p - 1) | ((sa_sint_t)(T[p - 2] < T[p - 1]) << (SAINT_BIT - 1)); symbol = T[p - 1]; p = 0; } cache[i].symbol = symbol; SA[i] = p & SAINT_MAX;
    }
}

static sa_sint_t libsais_partial_sorting_scan_left_to_right_32s_6k_block_sort(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT buckets, sa_sint_t d, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j, omp_block_end = omp_block_start + omp_block_size;
    for (i = omp_block_start, j = omp_block_end - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&cache[i + 2 * prefetch_distance]);

        libsais_prefetchw(&buckets[cache[i + prefetch_distance + 0].symbol]);
        libsais_prefetchw(&buckets[cache[i + prefetch_distance + 1].symbol]);

        sa_sint_t v0 = cache[i + 0].symbol, p0 = cache[i + 0].index; d += (p0 < 0); cache[i + 0].symbol = buckets[v0]++; cache[i + 0].index = (p0 - 1) | ((sa_sint_t)(buckets[2 + v0] != d) << (SAINT_BIT - 1)); buckets[2 + v0] = d;
        if (cache[i + 0].symbol < omp_block_end) { sa_sint_t s = cache[i + 0].symbol, q = (cache[s].index = cache[i + 0].index) & SAINT_MAX; cache[s].symbol = BUCKETS_INDEX4(T[q - 1], T[q - 2] >= T[q - 1]); }

        sa_sint_t v1 = cache[i + 1].symbol, p1 = cache[i + 1].index; d += (p1 < 0); cache[i + 1].symbol = buckets[v1]++; cache[i + 1].index = (p1 - 1) | ((sa_sint_t)(buckets[2 + v1] != d) << (SAINT_BIT - 1)); buckets[2 + v1] = d;
        if (cache[i + 1].symbol < omp_block_end) { sa_sint_t s = cache[i + 1].symbol, q = (cache[s].index = cache[i + 1].index) & SAINT_MAX; cache[s].symbol = BUCKETS_INDEX4(T[q - 1], T[q - 2] >= T[q - 1]); }
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t v = cache[i].symbol, p = cache[i].index; d += (p < 0); cache[i].symbol = buckets[v]++; cache[i].index = (p - 1) | ((sa_sint_t)(buckets[2 + v] != d) << (SAINT_BIT - 1)); buckets[2 + v] = d;
        if (cache[i].symbol < omp_block_end) { sa_sint_t s = cache[i].symbol, q = (cache[s].index = cache[i].index) & SAINT_MAX; cache[s].symbol = BUCKETS_INDEX4(T[q - 1], T[q - 2] >= T[q - 1]); }
    }

    return d;
}

static sa_sint_t libsais_partial_sorting_scan_left_to_right_32s_4k_block_sort(const sa_sint_t * RESTRICT T, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t d, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT induction_bucket = &buckets[2 * (fast_sint_t)k];
    sa_sint_t * RESTRICT distinct_names   = &buckets[0 * (fast_sint_t)k];

    fast_sint_t i, j, omp_block_end = omp_block_start + omp_block_size;
    for (i = omp_block_start, j = omp_block_end - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&cache[i + 2 * prefetch_distance]);

        sa_sint_t s0 = cache[i + prefetch_distance + 0].symbol; const sa_sint_t * Is0 = &induction_bucket[s0 >> 1]; libsais_prefetchw(s0 >= 0 ? Is0 : NULL); const sa_sint_t * Ds0 = &distinct_names[s0]; libsais_prefetchw(s0 >= 0 ? Ds0 : NULL); 
        sa_sint_t s1 = cache[i + prefetch_distance + 1].symbol; const sa_sint_t * Is1 = &induction_bucket[s1 >> 1]; libsais_prefetchw(s1 >= 0 ? Is1 : NULL); const sa_sint_t * Ds1 = &distinct_names[s1]; libsais_prefetchw(s1 >= 0 ? Ds1 : NULL);
        
        sa_sint_t v0 = cache[i + 0].symbol;
        if (v0 >= 0)
        {
            sa_sint_t p0 = cache[i + 0].index; d += (p0 >> (SUFFIX_GROUP_BIT - 1)); cache[i + 0].symbol = induction_bucket[v0 >> 1]++; cache[i + 0].index = (p0 - 1) | (v0 << (SAINT_BIT - 1)) | ((sa_sint_t)(distinct_names[v0] != d) << (SUFFIX_GROUP_BIT - 1)); distinct_names[v0] = d;
            if (cache[i + 0].symbol < omp_block_end) { sa_sint_t ni = cache[i + 0].symbol, np = cache[i + 0].index; if (np > 0) { cache[ni].index = np; np &= ~SUFFIX_GROUP_MARKER; cache[ni].symbol = BUCKETS_INDEX2(T[np - 1], T[np - 2] < T[np - 1]); np = 0; } cache[i + 0].index = np & SAINT_MAX; }
        }

        sa_sint_t v1 = cache[i + 1].symbol;
        if (v1 >= 0)
        {
            sa_sint_t p1 = cache[i + 1].index; d += (p1 >> (SUFFIX_GROUP_BIT - 1)); cache[i + 1].symbol = induction_bucket[v1 >> 1]++; cache[i + 1].index = (p1 - 1) | (v1 << (SAINT_BIT - 1)) | ((sa_sint_t)(distinct_names[v1] != d) << (SUFFIX_GROUP_BIT - 1)); distinct_names[v1] = d;
            if (cache[i + 1].symbol < omp_block_end) { sa_sint_t ni = cache[i + 1].symbol, np = cache[i + 1].index; if (np > 0) { cache[ni].index = np; np &= ~SUFFIX_GROUP_MARKER; cache[ni].symbol = BUCKETS_INDEX2(T[np - 1], T[np - 2] < T[np - 1]); np = 0; } cache[i + 1].index = np & SAINT_MAX; }
        }
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t v = cache[i].symbol;
        if (v >= 0)
        {
            sa_sint_t p = cache[i].index; d += (p >> (SUFFIX_GROUP_BIT - 1)); cache[i].symbol = induction_bucket[v >> 1]++; cache[i].index = (p - 1) | (v << (SAINT_BIT - 1)) | ((sa_sint_t)(distinct_names[v] != d) << (SUFFIX_GROUP_BIT - 1)); distinct_names[v] = d;
            if (cache[i].symbol < omp_block_end) { sa_sint_t ni = cache[i].symbol, np = cache[i].index; if (np > 0) { cache[ni].index = np; np &= ~SUFFIX_GROUP_MARKER; cache[ni].symbol = BUCKETS_INDEX2(T[np - 1], T[np - 2] < T[np - 1]); np = 0; } cache[i].index = np & SAINT_MAX; }
        }
    }

    return d;
}

static void libsais_partial_sorting_scan_left_to_right_32s_1k_block_sort(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT induction_bucket, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j, omp_block_end = omp_block_start + omp_block_size;
    for (i = omp_block_start, j = omp_block_end - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&cache[i + 2 * prefetch_distance]);

        sa_sint_t s0 = cache[i + prefetch_distance + 0].symbol; const sa_sint_t * Is0 = &induction_bucket[s0]; libsais_prefetchw(s0 >= 0 ? Is0 : NULL);
        sa_sint_t s1 = cache[i + prefetch_distance + 1].symbol; const sa_sint_t * Is1 = &induction_bucket[s1]; libsais_prefetchw(s1 >= 0 ? Is1 : NULL);
        
        sa_sint_t v0 = cache[i + 0].symbol;
        if (v0 >= 0)
        {
            cache[i + 0].symbol = induction_bucket[v0]++;
            if (cache[i + 0].symbol < omp_block_end) { sa_sint_t ni = cache[i + 0].symbol, np = cache[i + 0].index; if (np > 0) { cache[ni].index = (np - 1) | ((sa_sint_t)(T[np - 2] < T[np - 1]) << (SAINT_BIT - 1)); cache[ni].symbol = T[np - 1]; np = 0; } cache[i + 0].index = np & SAINT_MAX; }
        }

        sa_sint_t v1 = cache[i + 1].symbol;
        if (v1 >= 0)
        {
            cache[i + 1].symbol = induction_bucket[v1]++;
            if (cache[i + 1].symbol < omp_block_end) { sa_sint_t ni = cache[i + 1].symbol, np = cache[i + 1].index; if (np > 0) { cache[ni].index = (np - 1) | ((sa_sint_t)(T[np - 2] < T[np - 1]) << (SAINT_BIT - 1)); cache[ni].symbol = T[np - 1]; np = 0; } cache[i + 1].index = np & SAINT_MAX; }
        }
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t v = cache[i].symbol;
        if (v >= 0)
        {
            cache[i].symbol = induction_bucket[v]++;
            if (cache[i].symbol < omp_block_end) { sa_sint_t ni = cache[i].symbol, np = cache[i].index; if (np > 0) { cache[ni].index = (np - 1) | ((sa_sint_t)(T[np - 2] < T[np - 1]) << (SAINT_BIT - 1)); cache[ni].symbol = T[np - 1]; np = 0; } cache[i].index = np & SAINT_MAX; }
        }
    }
}

static sa_sint_t libsais_partial_sorting_scan_left_to_right_32s_6k_block_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, sa_sint_t d, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 16384)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(cache);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            d = libsais_partial_sorting_scan_left_to_right_32s_6k(T, SA, buckets, d, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                libsais_partial_sorting_scan_left_to_right_32s_6k_block_gather(T, SA, cache - block_start, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                d = libsais_partial_sorting_scan_left_to_right_32s_6k_block_sort(T, buckets, d, cache - block_start, block_start, block_size);
            }

            #pragma omp barrier

            {
                libsais_place_cached_suffixes(SA, cache - block_start, omp_block_start, omp_block_size);
            }
        }
#endif
    }

    return d;
}

static sa_sint_t libsais_partial_sorting_scan_left_to_right_32s_4k_block_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t d, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 16384)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(cache);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            d = libsais_partial_sorting_scan_left_to_right_32s_4k(T, SA, k, buckets, d, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                libsais_partial_sorting_scan_left_to_right_32s_4k_block_gather(T, SA, cache - block_start, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                d = libsais_partial_sorting_scan_left_to_right_32s_4k_block_sort(T, k, buckets, d, cache - block_start, block_start, block_size);
            }

            #pragma omp barrier

            {
                libsais_compact_and_place_cached_suffixes(SA, cache - block_start, omp_block_start, omp_block_size);
            }
        }
#endif
    }

    return d;
}

static void libsais_partial_sorting_scan_left_to_right_32s_1k_block_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 16384)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(cache);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            libsais_partial_sorting_scan_left_to_right_32s_1k(T, SA, buckets, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                libsais_partial_sorting_scan_left_to_right_32s_1k_block_gather(T, SA, cache - block_start, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                libsais_partial_sorting_scan_left_to_right_32s_1k_block_sort(T, buckets, cache - block_start, block_start, block_size);
            }

            #pragma omp barrier

            {
                libsais_compact_and_place_cached_suffixes(SA, cache - block_start, omp_block_start, omp_block_size);
            }
        }
#endif
    }
}

#endif

static sa_sint_t libsais_partial_sorting_scan_left_to_right_32s_6k_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t * RESTRICT buckets, sa_sint_t left_suffixes_count, sa_sint_t d, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#ifndef AOCL_BWT
    SA[buckets[BUCKETS_INDEX4(T[n - 1], T[n - 2] >= T[n - 1])]++] = (n - 1) | SAINT_MIN;
    buckets[2 + BUCKETS_INDEX4(T[n - 1], T[n - 2] >= T[n - 1])] = ++d;
#endif /* AOCL_BWT */
    if (threads == 1 || left_suffixes_count < 65536)
    {
#ifdef AOCL_BWT
        d = AOCL_libsais_partial_sorting_scan_left_to_right_32s_6k(T, SA, buckets, d, 0, left_suffixes_count, n);
#else
        d = libsais_partial_sorting_scan_left_to_right_32s_6k(T, SA, buckets, d, 0, left_suffixes_count);
#endif /* AOCL_BWT */
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start, block_end;
        for (block_start = 0; block_start < left_suffixes_count; block_start = block_end)
        {
            block_end = block_start + (fast_sint_t)threads * LIBSAIS_PER_THREAD_CACHE_SIZE; if (block_end > left_suffixes_count) { block_end = left_suffixes_count; }

            d = libsais_partial_sorting_scan_left_to_right_32s_6k_block_omp(T, SA, buckets, d, thread_state[0].state.cache, block_start, block_end - block_start, threads);
        }
    }
#else
    UNUSED(thread_state);
#endif

    return d;
}

static sa_sint_t libsais_partial_sorting_scan_left_to_right_32s_4k_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t d, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    sa_sint_t * RESTRICT induction_bucket = &buckets[2 * (fast_sint_t)k];
    sa_sint_t * RESTRICT distinct_names   = &buckets[0 * (fast_sint_t)k];

    SA[induction_bucket[T[n - 1]]++] = (n - 1) | ((sa_sint_t)(T[n - 2] < T[n - 1]) << (SAINT_BIT - 1)) | SUFFIX_GROUP_MARKER;
    distinct_names[BUCKETS_INDEX2(T[n - 1], T[n - 2] < T[n - 1])] = ++d;

    if (threads == 1 || n < 65536)
    {
        d = libsais_partial_sorting_scan_left_to_right_32s_4k(T, SA, k, buckets, d, 0, n);
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start, block_end;
        for (block_start = 0; block_start < n; block_start = block_end)
        {
            block_end = block_start + (fast_sint_t)threads * LIBSAIS_PER_THREAD_CACHE_SIZE; if (block_end > n) { block_end = n; }

            d = libsais_partial_sorting_scan_left_to_right_32s_4k_block_omp(T, SA, k, buckets, d, thread_state[0].state.cache, block_start, block_end - block_start, threads);
        }
    }
#else
    UNUSED(thread_state);
#endif

    return d;
}

static void libsais_partial_sorting_scan_left_to_right_32s_1k_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    SA[buckets[T[n - 1]]++] = (n - 1) | ((sa_sint_t)(T[n - 2] < T[n - 1]) << (SAINT_BIT - 1));

    if (threads == 1 || n < 65536)
    {
       libsais_partial_sorting_scan_left_to_right_32s_1k(T, SA, buckets, 0, n);
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start, block_end;
        for (block_start = 0; block_start < n; block_start = block_end)
        {
            block_end = block_start + (fast_sint_t)threads * LIBSAIS_PER_THREAD_CACHE_SIZE; if (block_end > n) { block_end = n; }

            libsais_partial_sorting_scan_left_to_right_32s_1k_block_omp(T, SA, buckets, thread_state[0].state.cache, block_start, block_end - block_start, threads);
        }
    }
#else
    UNUSED(thread_state);
#endif
}

static void libsais_partial_sorting_shift_markers_8u_omp(sa_sint_t * RESTRICT SA, sa_sint_t n, const sa_sint_t * RESTRICT buckets, sa_sint_t threads)
{
    const fast_sint_t prefetch_distance = 32;

    const sa_sint_t * RESTRICT temp_bucket = &buckets[4 * ALPHABET_SIZE];

    fast_sint_t c;

#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel for schedule(static, 1) num_threads(threads) if(threads > 1 && n >= 65536)
#else
    UNUSED(threads); UNUSED(n);
#endif
    for (c = BUCKETS_INDEX2(ALPHABET_SIZE - 1, 0); c >= BUCKETS_INDEX2(1, 0); c -= BUCKETS_INDEX2(1, 0))
    {
        fast_sint_t i, j; sa_sint_t s = SAINT_MIN;
        for (i = (fast_sint_t)temp_bucket[c] - 1, j = (fast_sint_t)buckets[c - BUCKETS_INDEX2(1, 0)] + 3; i >= j; i -= 4)
        {
            libsais_prefetchw(&SA[i - prefetch_distance]);

            sa_sint_t p0 = SA[i - 0], q0 = (p0 & SAINT_MIN) ^ s; s = s ^ q0; SA[i - 0] = p0 ^ q0;
            sa_sint_t p1 = SA[i - 1], q1 = (p1 & SAINT_MIN) ^ s; s = s ^ q1; SA[i - 1] = p1 ^ q1;
            sa_sint_t p2 = SA[i - 2], q2 = (p2 & SAINT_MIN) ^ s; s = s ^ q2; SA[i - 2] = p2 ^ q2;
            sa_sint_t p3 = SA[i - 3], q3 = (p3 & SAINT_MIN) ^ s; s = s ^ q3; SA[i - 3] = p3 ^ q3;
        }

        for (j -= 3; i >= j; i -= 1)
        {
            sa_sint_t p = SA[i], q = (p & SAINT_MIN) ^ s; s = s ^ q; SA[i] = p ^ q;
        }
    }
}

static void libsais_partial_sorting_shift_markers_32s_6k_omp(sa_sint_t * RESTRICT SA, sa_sint_t k, const sa_sint_t * RESTRICT buckets, sa_sint_t threads)
{
    const fast_sint_t prefetch_distance = 32;

    const sa_sint_t * RESTRICT temp_bucket = &buckets[4 * (fast_sint_t)k];
    
    fast_sint_t c;

#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel for schedule(static, 1) num_threads(threads) if(threads > 1 && k >= 65536)
#else
    UNUSED(threads);
#endif
    for (c = (fast_sint_t)k - 1; c >= 1; c -= 1)
    {
        fast_sint_t i, j; sa_sint_t s = SAINT_MIN;
        for (i = (fast_sint_t)buckets[BUCKETS_INDEX4(c, 0)] - 1, j = (fast_sint_t)temp_bucket[BUCKETS_INDEX2(c - 1, 0)] + 3; i >= j; i -= 4)
        {
            libsais_prefetchw(&SA[i - prefetch_distance]);

            sa_sint_t p0 = SA[i - 0], q0 = (p0 & SAINT_MIN) ^ s; s = s ^ q0; SA[i - 0] = p0 ^ q0;
            sa_sint_t p1 = SA[i - 1], q1 = (p1 & SAINT_MIN) ^ s; s = s ^ q1; SA[i - 1] = p1 ^ q1;
            sa_sint_t p2 = SA[i - 2], q2 = (p2 & SAINT_MIN) ^ s; s = s ^ q2; SA[i - 2] = p2 ^ q2;
            sa_sint_t p3 = SA[i - 3], q3 = (p3 & SAINT_MIN) ^ s; s = s ^ q3; SA[i - 3] = p3 ^ q3;
        }

        for (j -= 3; i >= j; i -= 1)
        {
            sa_sint_t p = SA[i], q = (p & SAINT_MIN) ^ s; s = s ^ q; SA[i] = p ^ q;
        }
    }
}

static void libsais_partial_sorting_shift_markers_32s_4k(sa_sint_t * RESTRICT SA, sa_sint_t n)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i; sa_sint_t s = SUFFIX_GROUP_MARKER;
    for (i = (fast_sint_t)n - 1; i >= 3; i -= 4)
    {
        libsais_prefetchw(&SA[i - prefetch_distance]);

        sa_sint_t p0 = SA[i - 0], q0 = ((p0 & SUFFIX_GROUP_MARKER) ^ s) & ((sa_sint_t)(p0 > 0) << ((SUFFIX_GROUP_BIT - 1))); s = s ^ q0; SA[i - 0] = p0 ^ q0;
        sa_sint_t p1 = SA[i - 1], q1 = ((p1 & SUFFIX_GROUP_MARKER) ^ s) & ((sa_sint_t)(p1 > 0) << ((SUFFIX_GROUP_BIT - 1))); s = s ^ q1; SA[i - 1] = p1 ^ q1;
        sa_sint_t p2 = SA[i - 2], q2 = ((p2 & SUFFIX_GROUP_MARKER) ^ s) & ((sa_sint_t)(p2 > 0) << ((SUFFIX_GROUP_BIT - 1))); s = s ^ q2; SA[i - 2] = p2 ^ q2;
        sa_sint_t p3 = SA[i - 3], q3 = ((p3 & SUFFIX_GROUP_MARKER) ^ s) & ((sa_sint_t)(p3 > 0) << ((SUFFIX_GROUP_BIT - 1))); s = s ^ q3; SA[i - 3] = p3 ^ q3;
    }

    for (; i >= 0; i -= 1)
    {
        sa_sint_t p = SA[i], q = ((p & SUFFIX_GROUP_MARKER) ^ s) & ((sa_sint_t)(p > 0) << ((SUFFIX_GROUP_BIT - 1))); s = s ^ q; SA[i] = p ^ q;
    }
}

static void libsais_partial_sorting_shift_buckets_32s_6k(sa_sint_t k, sa_sint_t * RESTRICT buckets)
{
    sa_sint_t * RESTRICT temp_bucket = &buckets[4 * (fast_sint_t)k];

    fast_sint_t i;
    for (i = BUCKETS_INDEX2(0, 0); i <= BUCKETS_INDEX2((fast_sint_t)k - 1, 0); i += BUCKETS_INDEX2(1, 0))
    {
        buckets[2 * i + BUCKETS_INDEX4(0, 0)] = temp_bucket[i + BUCKETS_INDEX2(0, 0)];
        buckets[2 * i + BUCKETS_INDEX4(0, 1)] = temp_bucket[i + BUCKETS_INDEX2(0, 1)];
    }
}

#ifndef AOCL_BWT
static sa_sint_t libsais_partial_sorting_scan_right_to_left_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, sa_sint_t d, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT induction_bucket = &buckets[0 * ALPHABET_SIZE];
    sa_sint_t * RESTRICT distinct_names   = &buckets[2 * ALPHABET_SIZE];

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 1; i >= j; i -= 2)
    {
        libsais_prefetchr(&SA[i - 2 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i - prefetch_distance - 0] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i - prefetch_distance - 0] & SAINT_MAX] - 2);
        libsais_prefetchr(&T[SA[i - prefetch_distance - 1] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i - prefetch_distance - 1] & SAINT_MAX] - 2);

        sa_sint_t p0 = SA[i - 0]; d += (p0 < 0); p0 &= SAINT_MAX; sa_sint_t v0 = BUCKETS_INDEX2(T[p0 - 1], T[p0 - 2] > T[p0 - 1]);
        SA[--induction_bucket[v0]] = (p0 - 1) | ((sa_sint_t)(distinct_names[v0] != d) << (SAINT_BIT - 1)); distinct_names[v0] = d;

        sa_sint_t p1 = SA[i - 1]; d += (p1 < 0); p1 &= SAINT_MAX; sa_sint_t v1 = BUCKETS_INDEX2(T[p1 - 1], T[p1 - 2] > T[p1 - 1]);
        SA[--induction_bucket[v1]] = (p1 - 1) | ((sa_sint_t)(distinct_names[v1] != d) << (SAINT_BIT - 1)); distinct_names[v1] = d;
    }

    for (j -= prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i]; d += (p < 0); p &= SAINT_MAX; sa_sint_t v = BUCKETS_INDEX2(T[p - 1], T[p - 2] > T[p - 1]);
        SA[--induction_bucket[v]] = (p - 1) | ((sa_sint_t)(distinct_names[v] != d) << (SAINT_BIT - 1)); distinct_names[v] = d;
    }

    return d;
}
#else
static sa_sint_t AOCL_libsais_partial_sorting_scan_right_to_left_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, sa_sint_t d, fast_sint_t omp_block_start, fast_sint_t omp_block_size, fast_sint_t n)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT induction_bucket = &buckets[0 * ALPHABET_SIZE];
    sa_sint_t * RESTRICT distinct_names   = &buckets[2 * ALPHABET_SIZE];

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 1; i >= j; i -= 2)
    {
        libsais_prefetchr(&SA[i - 2 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i - prefetch_distance - 0] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i - prefetch_distance - 0] & SAINT_MAX] - 2);
        libsais_prefetchr(&T[SA[i - prefetch_distance - 1] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i - prefetch_distance - 1] & SAINT_MAX] - 2);

        sa_sint_t p0 = SA[i - 0];
        d += (p0 < 0);
        p0 &= SAINT_MAX;
        sa_sint_t v0 = BUCKETS_INDEX2(T[p0 - 1], T[p0 - 2] > T[p0 - 1]);
        SA[--induction_bucket[v0]] = MOD(p0 - 1, n) | ((sa_sint_t)(distinct_names[v0] != d) << (SAINT_BIT - 1));
        distinct_names[v0] = d;

        sa_sint_t p1 = SA[i - 1];
        d += (p1 < 0);
        p1 &= SAINT_MAX;
        sa_sint_t v1 = BUCKETS_INDEX2(T[p1 - 1], T[p1 - 2] > T[p1 - 1]);
        SA[--induction_bucket[v1]] = MOD(p1 - 1, n) | ((sa_sint_t)(distinct_names[v1] != d) << (SAINT_BIT - 1));
        distinct_names[v1] = d;
    }

    for (j -= prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i];
        d += (p < 0);
        p &= SAINT_MAX;
        sa_sint_t v = BUCKETS_INDEX2(T[p - 1], T[p - 2] > T[p - 1]);
        SA[--induction_bucket[v]] = MOD(p - 1, n) | ((sa_sint_t)(distinct_names[v] != d) << (SAINT_BIT - 1));
        distinct_names[v] = d;
    }

    return d;
}
#endif /* AOCL_BWT */
#if defined(LIBSAIS_OPENMP)

static void libsais_partial_sorting_scan_right_to_left_8u_block_prepare(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size, LIBSAIS_THREAD_STATE * RESTRICT state)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT induction_bucket = &buckets[0 * ALPHABET_SIZE];
    sa_sint_t * RESTRICT distinct_names   = &buckets[2 * ALPHABET_SIZE];

    memset(induction_bucket, 0, (size_t)2 * k * sizeof(sa_sint_t));
    memset(distinct_names  , 0, (size_t)2 * k * sizeof(sa_sint_t));

    fast_sint_t i, j, count = 0; sa_sint_t d = 1;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 1; i >= j; i -= 2)
    {
        libsais_prefetchr(&SA[i - 2 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i - prefetch_distance - 0] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i - prefetch_distance - 0] & SAINT_MAX] - 2);
        libsais_prefetchr(&T[SA[i - prefetch_distance - 1] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i - prefetch_distance - 1] & SAINT_MAX] - 2);

        sa_sint_t p0 = cache[count].index = SA[i - 0]; d += (p0 < 0); p0 &= SAINT_MAX; sa_sint_t v0 = cache[count++].symbol = BUCKETS_INDEX2(T[p0 - 1], T[p0 - 2] > T[p0 - 1]); induction_bucket[v0]++; distinct_names[v0] = d;
        sa_sint_t p1 = cache[count].index = SA[i - 1]; d += (p1 < 0); p1 &= SAINT_MAX; sa_sint_t v1 = cache[count++].symbol = BUCKETS_INDEX2(T[p1 - 1], T[p1 - 2] > T[p1 - 1]); induction_bucket[v1]++; distinct_names[v1] = d;
    }

    for (j -= prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t p = cache[count].index = SA[i]; d += (p < 0); p &= SAINT_MAX; sa_sint_t v = cache[count++].symbol = BUCKETS_INDEX2(T[p - 1], T[p - 2] > T[p - 1]); induction_bucket[v]++; distinct_names[v] = d;
    }

    state[0].state.position   = (fast_sint_t)d - 1;
    state[0].state.count      = count;
}

static void libsais_partial_sorting_scan_right_to_left_8u_block_place(sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t count, sa_sint_t d)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT induction_bucket = &buckets[0 * ALPHABET_SIZE];
    sa_sint_t * RESTRICT distinct_names   = &buckets[2 * ALPHABET_SIZE];

    fast_sint_t i, j;
    for (i = 0, j = count - 1; i < j; i += 2)
    {
        libsais_prefetchr(&cache[i + prefetch_distance]);

        sa_sint_t p0 = cache[i + 0].index; d += (p0 < 0); sa_sint_t v0 = cache[i + 0].symbol;
        SA[--induction_bucket[v0]] = (p0 - 1) | ((sa_sint_t)(distinct_names[v0] != d) << (SAINT_BIT - 1)); distinct_names[v0] = d;

        sa_sint_t p1 = cache[i + 1].index; d += (p1 < 0); sa_sint_t v1 = cache[i + 1].symbol;
        SA[--induction_bucket[v1]] = (p1 - 1) | ((sa_sint_t)(distinct_names[v1] != d) << (SAINT_BIT - 1)); distinct_names[v1] = d;
    }

    for (j += 1; i < j; i += 1)
    {
        sa_sint_t p = cache[i].index; d += (p < 0); sa_sint_t v = cache[i].symbol;
        SA[--induction_bucket[v]] = (p - 1) | ((sa_sint_t)(distinct_names[v] != d) << (SAINT_BIT - 1)); distinct_names[v] = d;
    }
}

static sa_sint_t libsais_partial_sorting_scan_right_to_left_8u_block_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t d, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 64 * max(k, 256) && omp_get_dynamic() == 0)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(k); UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            d = libsais_partial_sorting_scan_right_to_left_8u(T, SA, buckets, d, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                libsais_partial_sorting_scan_right_to_left_8u_block_prepare(T, SA, k, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, omp_block_start, omp_block_size, &thread_state[omp_thread_num]);
            }

            #pragma omp barrier

            #pragma omp master
            {
                sa_sint_t * RESTRICT induction_bucket = &buckets[0 * ALPHABET_SIZE];
                sa_sint_t * RESTRICT distinct_names   = &buckets[2 * ALPHABET_SIZE];

                fast_sint_t t;
                for (t = omp_num_threads - 1; t >= 0; --t)
                {
                    sa_sint_t * RESTRICT temp_induction_bucket    = &thread_state[t].state.buckets[0 * ALPHABET_SIZE];
                    sa_sint_t * RESTRICT temp_distinct_names      = &thread_state[t].state.buckets[2 * ALPHABET_SIZE];

                    fast_sint_t c; 
                    for (c = 0; c < 2 * k; c += 1) { sa_sint_t A = induction_bucket[c], B = temp_induction_bucket[c]; induction_bucket[c] = A - B; temp_induction_bucket[c] = A; }

                    for (d -= 1, c = 0; c < 2 * k; c += 1) { sa_sint_t A = distinct_names[c], B = temp_distinct_names[c], D = B + d; distinct_names[c] = B > 0 ? D : A; temp_distinct_names[c] = A; }
                    d += 1 + (sa_sint_t)thread_state[t].state.position; thread_state[t].state.position = (fast_sint_t)d - thread_state[t].state.position;
                }
            }

            #pragma omp barrier

            {
                libsais_partial_sorting_scan_right_to_left_8u_block_place(SA, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, thread_state[omp_thread_num].state.count, (sa_sint_t)thread_state[omp_thread_num].state.position);
            }
        }
#endif
    }

    return d;
}

#endif

static void libsais_partial_sorting_scan_right_to_left_8u_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t first_lms_suffix, sa_sint_t left_suffixes_count, sa_sint_t d, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#ifdef AOCL_BWT
    fast_sint_t scan_start    = (fast_sint_t)left_suffixes_count;
    fast_sint_t scan_end      = (fast_sint_t)n;
#else
    fast_sint_t scan_start    = (fast_sint_t)left_suffixes_count + 1;
    fast_sint_t scan_end      = (fast_sint_t)n - (fast_sint_t)first_lms_suffix;
#endif /* AOCL_BWT */
    if (threads == 1 || (scan_end - scan_start) < 65536)
    {
#ifdef AOCL_BWT
        AOCL_libsais_partial_sorting_scan_right_to_left_8u(T, SA, buckets, d, scan_start, scan_end - scan_start, n);
#else
        libsais_partial_sorting_scan_right_to_left_8u(T, SA, buckets, d, scan_start, scan_end - scan_start);
#endif /* AOCL_BWT */
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        sa_sint_t * RESTRICT induction_bucket = &buckets[0 * ALPHABET_SIZE];
        sa_sint_t * RESTRICT distinct_names   = &buckets[2 * ALPHABET_SIZE];

        fast_sint_t block_start;
        for (block_start = scan_end - 1; block_start >= scan_start; )
        {
            if (SA[block_start] == 0)
            {
                block_start--;
            }
            else
            {
                fast_sint_t block_max_end = block_start - ((fast_sint_t)threads) * (LIBSAIS_PER_THREAD_CACHE_SIZE - 16 * (fast_sint_t)threads); if (block_max_end < scan_start) { block_max_end = scan_start - 1; }
                fast_sint_t block_end     = block_start - 1; while (block_end > block_max_end && SA[block_end] != 0) { block_end--; }
                fast_sint_t block_size    = block_start - block_end;

                if (block_size < 32)
                {
                    for (; block_start > block_end; block_start -= 1)
                    {
                        sa_sint_t p = SA[block_start]; d += (p < 0); p &= SAINT_MAX; sa_sint_t v = BUCKETS_INDEX2(T[p - 1], T[p - 2] > T[p - 1]);
                        SA[--induction_bucket[v]] = (p - 1) | ((sa_sint_t)(distinct_names[v] != d) << (SAINT_BIT - 1)); distinct_names[v] = d;
                    }
                }
                else
                {
                    d = libsais_partial_sorting_scan_right_to_left_8u_block_omp(T, SA, k, buckets, d, block_end + 1, block_size, threads, thread_state);
                    block_start = block_end;
                }
            }
        }
    }
#else
    UNUSED(k); UNUSED(thread_state);
#endif
}

#ifndef AOCL_BWT
static sa_sint_t libsais_partial_sorting_scan_right_to_left_32s_6k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, sa_sint_t d, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + 2 * prefetch_distance + 1; i >= j; i -= 2)
    {
        libsais_prefetchr(&SA[i - 3 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 0] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 0] & SAINT_MAX] - 2);
        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 1] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 1] & SAINT_MAX] - 2);

        sa_sint_t p0 = SA[i - prefetch_distance - 0] & SAINT_MAX; sa_sint_t v0 = BUCKETS_INDEX4(T[p0 - (p0 > 0)], 0); libsais_prefetchw(&buckets[v0]);
        sa_sint_t p1 = SA[i - prefetch_distance - 1] & SAINT_MAX; sa_sint_t v1 = BUCKETS_INDEX4(T[p1 - (p1 > 0)], 0); libsais_prefetchw(&buckets[v1]);

        sa_sint_t p2 = SA[i - 0]; d += (p2 < 0); p2 &= SAINT_MAX; sa_sint_t v2 = BUCKETS_INDEX4(T[p2 - 1], T[p2 - 2] > T[p2 - 1]);
        SA[--buckets[v2]] = (p2 - 1) | ((sa_sint_t)(buckets[2 + v2] != d) << (SAINT_BIT - 1)); buckets[2 + v2] = d;

        sa_sint_t p3 = SA[i - 1]; d += (p3 < 0); p3 &= SAINT_MAX; sa_sint_t v3 = BUCKETS_INDEX4(T[p3 - 1], T[p3 - 2] > T[p3 - 1]);
        SA[--buckets[v3]] = (p3 - 1) | ((sa_sint_t)(buckets[2 + v3] != d) << (SAINT_BIT - 1)); buckets[2 + v3] = d;
    }

    for (j -= 2 * prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i]; d += (p < 0); p &= SAINT_MAX; sa_sint_t v = BUCKETS_INDEX4(T[p - 1], T[p - 2] > T[p - 1]);
        SA[--buckets[v]] = (p - 1) | ((sa_sint_t)(buckets[2 + v] != d) << (SAINT_BIT - 1)); buckets[2 + v] = d;
    }

    return d;
}
#else
static sa_sint_t AOCL_libsais_partial_sorting_scan_right_to_left_32s_6k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, sa_sint_t d, fast_sint_t omp_block_start, fast_sint_t omp_block_size, fast_sint_t n)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + 2 * prefetch_distance + 1; i >= j; i -= 2)
    {
        libsais_prefetchr(&SA[i - 3 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 0] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 0] & SAINT_MAX] - 2);
        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 1] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i - 2 * prefetch_distance - 1] & SAINT_MAX] - 2);

        sa_sint_t p0 = SA[i - prefetch_distance - 0] & SAINT_MAX; sa_sint_t v0 = BUCKETS_INDEX4(T[p0 - 1], 0); libsais_prefetchw(&buckets[v0]);
        sa_sint_t p1 = SA[i - prefetch_distance - 1] & SAINT_MAX; sa_sint_t v1 = BUCKETS_INDEX4(T[p1 - 1], 0); libsais_prefetchw(&buckets[v1]);

        sa_sint_t p2 = SA[i - 0]; d += (p2 < 0); p2 &= SAINT_MAX; sa_sint_t v2 = BUCKETS_INDEX4(T[p2 - 1], T[p2 - 2] > T[p2 - 1]);
        SA[--buckets[v2]] = MOD(p2 - 1, n) | ((sa_sint_t)(buckets[2 + v2] != d) << (SAINT_BIT - 1)); buckets[2 + v2] = d;

        sa_sint_t p3 = SA[i - 1]; d += (p3 < 0); p3 &= SAINT_MAX; sa_sint_t v3 = BUCKETS_INDEX4(T[p3 - 1], T[p3 - 2] > T[p3 - 1]);
        SA[--buckets[v3]] = MOD(p3 - 1, n) | ((sa_sint_t)(buckets[2 + v3] != d) << (SAINT_BIT - 1)); buckets[2 + v3] = d;
    }

    for (j -= 2 * prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i]; d += (p < 0); p &= SAINT_MAX; sa_sint_t v = BUCKETS_INDEX4(T[p - 1], T[p - 2] > T[p - 1]);
        SA[--buckets[v]] = MOD(p - 1, n) | ((sa_sint_t)(buckets[2 + v] != d) << (SAINT_BIT - 1)); buckets[2 + v] = d;
    }

    return d;
}
#endif /* AOCL_BWT */
static sa_sint_t libsais_partial_sorting_scan_right_to_left_32s_4k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t d, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT induction_bucket = &buckets[3 * (fast_sint_t)k];
    sa_sint_t * RESTRICT distinct_names   = &buckets[0 * (fast_sint_t)k];

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + 2 * prefetch_distance + 1; i >= j; i -= 2)
    {
        libsais_prefetchw(&SA[i - 3 * prefetch_distance]);

        sa_sint_t s0 = SA[i - 2 * prefetch_distance - 0]; const sa_sint_t * Ts0 = &T[s0 & ~SUFFIX_GROUP_MARKER] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i - 2 * prefetch_distance - 1]; const sa_sint_t * Ts1 = &T[s1 & ~SUFFIX_GROUP_MARKER] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);
        sa_sint_t s2 = SA[i - 1 * prefetch_distance - 0]; if (s2 > 0) { const fast_sint_t Ts2 = T[(s2 & ~SUFFIX_GROUP_MARKER) - 1]; libsais_prefetchw(&induction_bucket[Ts2]); libsais_prefetchw(&distinct_names[BUCKETS_INDEX2(Ts2, 0)]); }
        sa_sint_t s3 = SA[i - 1 * prefetch_distance - 1]; if (s3 > 0) { const fast_sint_t Ts3 = T[(s3 & ~SUFFIX_GROUP_MARKER) - 1]; libsais_prefetchw(&induction_bucket[Ts3]); libsais_prefetchw(&distinct_names[BUCKETS_INDEX2(Ts3, 0)]); }

        sa_sint_t p0 = SA[i - 0];
        if (p0 > 0)
        {
            SA[i - 0] = 0; d += (p0 >> (SUFFIX_GROUP_BIT - 1)); p0 &= ~SUFFIX_GROUP_MARKER; sa_sint_t v0 = BUCKETS_INDEX2(T[p0 - 1], T[p0 - 2] > T[p0 - 1]);
            SA[--induction_bucket[T[p0 - 1]]] = (p0 - 1) | ((sa_sint_t)(T[p0 - 2] > T[p0 - 1]) << (SAINT_BIT - 1)) | ((sa_sint_t)(distinct_names[v0] != d) << (SUFFIX_GROUP_BIT - 1)); distinct_names[v0] = d;
        }

        sa_sint_t p1 = SA[i - 1];
        if (p1 > 0)
        {
            SA[i - 1] = 0; d += (p1 >> (SUFFIX_GROUP_BIT - 1)); p1 &= ~SUFFIX_GROUP_MARKER; sa_sint_t v1 = BUCKETS_INDEX2(T[p1 - 1], T[p1 - 2] > T[p1 - 1]);
            SA[--induction_bucket[T[p1 - 1]]] = (p1 - 1) | ((sa_sint_t)(T[p1 - 2] > T[p1 - 1]) << (SAINT_BIT - 1)) | ((sa_sint_t)(distinct_names[v1] != d) << (SUFFIX_GROUP_BIT - 1)); distinct_names[v1] = d;
        }
    }

    for (j -= 2 * prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i];
        if (p > 0)
        {
            SA[i] = 0; d += (p >> (SUFFIX_GROUP_BIT - 1)); p &= ~SUFFIX_GROUP_MARKER; sa_sint_t v = BUCKETS_INDEX2(T[p - 1], T[p - 2] > T[p - 1]);
            SA[--induction_bucket[T[p - 1]]] = (p - 1) | ((sa_sint_t)(T[p - 2] > T[p - 1]) << (SAINT_BIT - 1)) | ((sa_sint_t)(distinct_names[v] != d) << (SUFFIX_GROUP_BIT - 1)); distinct_names[v] = d;
        }
    }

    return d;
}

static void libsais_partial_sorting_scan_right_to_left_32s_1k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + 2 * prefetch_distance + 1; i >= j; i -= 2)
    {
        libsais_prefetchw(&SA[i - 3 * prefetch_distance]);

        sa_sint_t s0 = SA[i - 2 * prefetch_distance - 0]; const sa_sint_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i - 2 * prefetch_distance - 1]; const sa_sint_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);
        sa_sint_t s2 = SA[i - 1 * prefetch_distance - 0]; if (s2 > 0) { libsais_prefetchw(&induction_bucket[T[s2 - 1]]); libsais_prefetchr(&T[s2] - 2); }
        sa_sint_t s3 = SA[i - 1 * prefetch_distance - 1]; if (s3 > 0) { libsais_prefetchw(&induction_bucket[T[s3 - 1]]); libsais_prefetchr(&T[s3] - 2); }

        sa_sint_t p0 = SA[i - 0]; if (p0 > 0) { SA[i - 0] = 0; SA[--induction_bucket[T[p0 - 1]]] = (p0 - 1) | ((sa_sint_t)(T[p0 - 2] > T[p0 - 1]) << (SAINT_BIT - 1)); }
        sa_sint_t p1 = SA[i - 1]; if (p1 > 0) { SA[i - 1] = 0; SA[--induction_bucket[T[p1 - 1]]] = (p1 - 1) | ((sa_sint_t)(T[p1 - 2] > T[p1 - 1]) << (SAINT_BIT - 1)); }
    }

    for (j -= 2 * prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i]; if (p > 0) { SA[i] = 0; SA[--induction_bucket[T[p - 1]]] = (p - 1) | ((sa_sint_t)(T[p - 2] > T[p - 1]) << (SAINT_BIT - 1)); }
    }
}

#if defined(LIBSAIS_OPENMP)

static void libsais_partial_sorting_scan_right_to_left_32s_6k_block_gather(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchr(&SA[i + 2 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i + prefetch_distance + 0] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 0] & SAINT_MAX] - 2);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 1] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + prefetch_distance + 1] & SAINT_MAX] - 2);

        libsais_prefetchw(&cache[i + prefetch_distance]);

        sa_sint_t p0 = cache[i + 0].index = SA[i + 0]; sa_sint_t symbol0 = 0; p0 &= SAINT_MAX; if (p0 != 0) { symbol0 = BUCKETS_INDEX4(T[p0 - 1], T[p0 - 2] > T[p0 - 1]); } cache[i + 0].symbol = symbol0;
        sa_sint_t p1 = cache[i + 1].index = SA[i + 1]; sa_sint_t symbol1 = 0; p1 &= SAINT_MAX; if (p1 != 0) { symbol1 = BUCKETS_INDEX4(T[p1 - 1], T[p1 - 2] > T[p1 - 1]); } cache[i + 1].symbol = symbol1;
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t p = cache[i].index = SA[i]; sa_sint_t symbol = 0; p &= SAINT_MAX; if (p != 0) { symbol = BUCKETS_INDEX4(T[p - 1], T[p - 2] > T[p - 1]); } cache[i].symbol = symbol;
    }
}

static void libsais_partial_sorting_scan_right_to_left_32s_4k_block_gather(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&SA[i + 2 * prefetch_distance]);

        sa_sint_t s0 = SA[i + prefetch_distance + 0]; const sa_sint_t * Ts0 = &T[s0 & ~SUFFIX_GROUP_MARKER] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i + prefetch_distance + 1]; const sa_sint_t * Ts1 = &T[s1 & ~SUFFIX_GROUP_MARKER] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

        libsais_prefetchw(&cache[i + prefetch_distance]);

        sa_sint_t symbol0 = SAINT_MIN, p0 = SA[i + 0]; if (p0 > 0) { SA[i + 0] = 0; cache[i + 0].index = p0; p0 &= ~SUFFIX_GROUP_MARKER; symbol0 = BUCKETS_INDEX2(T[p0 - 1], T[p0 - 2] > T[p0 - 1]); } cache[i + 0].symbol = symbol0;
        sa_sint_t symbol1 = SAINT_MIN, p1 = SA[i + 1]; if (p1 > 0) { SA[i + 1] = 0; cache[i + 1].index = p1; p1 &= ~SUFFIX_GROUP_MARKER; symbol1 = BUCKETS_INDEX2(T[p1 - 1], T[p1 - 2] > T[p1 - 1]); } cache[i + 1].symbol = symbol1;
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t symbol = SAINT_MIN, p = SA[i]; if (p > 0) { SA[i] = 0; cache[i].index = p; p &= ~SUFFIX_GROUP_MARKER; symbol = BUCKETS_INDEX2(T[p - 1], T[p - 2] > T[p - 1]); } cache[i].symbol = symbol;
    }
}

static void libsais_partial_sorting_scan_right_to_left_32s_1k_block_gather(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&SA[i + 2 * prefetch_distance]);

        sa_sint_t s0 = SA[i + prefetch_distance + 0]; const sa_sint_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i + prefetch_distance + 1]; const sa_sint_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

        libsais_prefetchw(&cache[i + prefetch_distance]);

        sa_sint_t symbol0 = SAINT_MIN, p0 = SA[i + 0]; if (p0 > 0) { SA[i + 0] = 0; cache[i + 0].index = (p0 - 1) | ((sa_sint_t)(T[p0 - 2] > T[p0 - 1]) << (SAINT_BIT - 1)); symbol0 = T[p0 - 1]; } cache[i + 0].symbol = symbol0;
        sa_sint_t symbol1 = SAINT_MIN, p1 = SA[i + 1]; if (p1 > 0) { SA[i + 1] = 0; cache[i + 1].index = (p1 - 1) | ((sa_sint_t)(T[p1 - 2] > T[p1 - 1]) << (SAINT_BIT - 1)); symbol1 = T[p1 - 1]; } cache[i + 1].symbol = symbol1;
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t symbol = SAINT_MIN, p = SA[i]; if (p > 0) { SA[i] = 0; cache[i].index = (p - 1) | ((sa_sint_t)(T[p - 2] > T[p - 1]) << (SAINT_BIT - 1)); symbol = T[p - 1]; } cache[i].symbol = symbol;
    }
}

static sa_sint_t libsais_partial_sorting_scan_right_to_left_32s_6k_block_sort(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT buckets, sa_sint_t d, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 1; i >= j; i -= 2)
    {
        libsais_prefetchw(&cache[i - 2 * prefetch_distance]);

        libsais_prefetchw(&buckets[cache[i - prefetch_distance - 0].symbol]);
        libsais_prefetchw(&buckets[cache[i - prefetch_distance - 1].symbol]);

        sa_sint_t v0 = cache[i - 0].symbol, p0 = cache[i - 0].index; d += (p0 < 0); cache[i - 0].symbol = --buckets[v0]; cache[i - 0].index = (p0 - 1) | ((sa_sint_t)(buckets[2 + v0] != d) << (SAINT_BIT - 1)); buckets[2 + v0] = d;
        if (cache[i - 0].symbol >= omp_block_start) { sa_sint_t s = cache[i - 0].symbol, q = (cache[s].index = cache[i - 0].index) & SAINT_MAX; cache[s].symbol = BUCKETS_INDEX4(T[q - 1], T[q - 2] > T[q - 1]); }

        sa_sint_t v1 = cache[i - 1].symbol, p1 = cache[i - 1].index; d += (p1 < 0); cache[i - 1].symbol = --buckets[v1]; cache[i - 1].index = (p1 - 1) | ((sa_sint_t)(buckets[2 + v1] != d) << (SAINT_BIT - 1)); buckets[2 + v1] = d;
        if (cache[i - 1].symbol >= omp_block_start) { sa_sint_t s = cache[i - 1].symbol, q = (cache[s].index = cache[i - 1].index) & SAINT_MAX; cache[s].symbol = BUCKETS_INDEX4(T[q - 1], T[q - 2] > T[q - 1]); }
    }

    for (j -= prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t v = cache[i].symbol, p = cache[i].index; d += (p < 0); cache[i].symbol = --buckets[v]; cache[i].index = (p - 1) | ((sa_sint_t)(buckets[2 + v] != d) << (SAINT_BIT - 1)); buckets[2 + v] = d;
        if (cache[i].symbol >= omp_block_start) { sa_sint_t s = cache[i].symbol, q = (cache[s].index = cache[i].index) & SAINT_MAX; cache[s].symbol = BUCKETS_INDEX4(T[q - 1], T[q - 2] > T[q - 1]); }
    }

    return d;
}

static sa_sint_t libsais_partial_sorting_scan_right_to_left_32s_4k_block_sort(const sa_sint_t * RESTRICT T, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t d, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT induction_bucket = &buckets[3 * (fast_sint_t)k];
    sa_sint_t * RESTRICT distinct_names   = &buckets[0 * (fast_sint_t)k];

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 1; i >= j; i -= 2)
    {
        libsais_prefetchw(&cache[i - 2 * prefetch_distance]);

        sa_sint_t s0 = cache[i - prefetch_distance - 0].symbol; const sa_sint_t * Is0 = &induction_bucket[s0 >> 1]; libsais_prefetchw(s0 >= 0 ? Is0 : NULL); const sa_sint_t * Ds0 = &distinct_names[s0]; libsais_prefetchw(s0 >= 0 ? Ds0 : NULL); 
        sa_sint_t s1 = cache[i - prefetch_distance - 1].symbol; const sa_sint_t * Is1 = &induction_bucket[s1 >> 1]; libsais_prefetchw(s1 >= 0 ? Is1 : NULL); const sa_sint_t * Ds1 = &distinct_names[s1]; libsais_prefetchw(s1 >= 0 ? Ds1 : NULL);

        sa_sint_t v0 = cache[i - 0].symbol;
        if (v0 >= 0)
        {
            sa_sint_t p0 = cache[i - 0].index; d += (p0 >> (SUFFIX_GROUP_BIT - 1)); cache[i - 0].symbol = --induction_bucket[v0 >> 1]; cache[i - 0].index = (p0 - 1) | (v0 << (SAINT_BIT - 1)) | ((sa_sint_t)(distinct_names[v0] != d) << (SUFFIX_GROUP_BIT - 1)); distinct_names[v0] = d;
            if (cache[i - 0].symbol >= omp_block_start) { sa_sint_t ni = cache[i - 0].symbol, np = cache[i - 0].index; if (np > 0) { cache[i - 0].index = 0; cache[ni].index = np; np &= ~SUFFIX_GROUP_MARKER; cache[ni].symbol = BUCKETS_INDEX2(T[np - 1], T[np - 2] > T[np - 1]); } }
        }

        sa_sint_t v1 = cache[i - 1].symbol;
        if (v1 >= 0)
        {
            sa_sint_t p1 = cache[i - 1].index; d += (p1 >> (SUFFIX_GROUP_BIT - 1)); cache[i - 1].symbol = --induction_bucket[v1 >> 1]; cache[i - 1].index = (p1 - 1) | (v1 << (SAINT_BIT - 1)) | ((sa_sint_t)(distinct_names[v1] != d) << (SUFFIX_GROUP_BIT - 1)); distinct_names[v1] = d;
            if (cache[i - 1].symbol >= omp_block_start) { sa_sint_t ni = cache[i - 1].symbol, np = cache[i - 1].index; if (np > 0) { cache[i - 1].index = 0; cache[ni].index = np; np &= ~SUFFIX_GROUP_MARKER; cache[ni].symbol = BUCKETS_INDEX2(T[np - 1], T[np - 2] > T[np - 1]); } }
        }
    }

    for (j -= prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t v = cache[i].symbol;
        if (v >= 0)
        {
            sa_sint_t p = cache[i].index; d += (p >> (SUFFIX_GROUP_BIT - 1)); cache[i].symbol = --induction_bucket[v >> 1]; cache[i].index = (p - 1) | (v << (SAINT_BIT - 1)) | ((sa_sint_t)(distinct_names[v] != d) << (SUFFIX_GROUP_BIT - 1)); distinct_names[v] = d;
            if (cache[i].symbol >= omp_block_start) { sa_sint_t ni = cache[i].symbol, np = cache[i].index; if (np > 0) { cache[i].index = 0; cache[ni].index = np; np &= ~SUFFIX_GROUP_MARKER; cache[ni].symbol = BUCKETS_INDEX2(T[np - 1], T[np - 2] > T[np - 1]); } }
        }
    }

    return d;
}

static void libsais_partial_sorting_scan_right_to_left_32s_1k_block_sort(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT induction_bucket, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 1; i >= j; i -= 2)
    {
        libsais_prefetchw(&cache[i - 2 * prefetch_distance]);

        sa_sint_t s0 = cache[i - prefetch_distance - 0].symbol; const sa_sint_t * Is0 = &induction_bucket[s0]; libsais_prefetchw(s0 >= 0 ? Is0 : NULL);
        sa_sint_t s1 = cache[i - prefetch_distance - 1].symbol; const sa_sint_t * Is1 = &induction_bucket[s1]; libsais_prefetchw(s1 >= 0 ? Is1 : NULL);

        sa_sint_t v0 = cache[i - 0].symbol;
        if (v0 >= 0)
        {
            cache[i - 0].symbol = --induction_bucket[v0];
            if (cache[i - 0].symbol >= omp_block_start) { sa_sint_t ni = cache[i - 0].symbol, np = cache[i - 0].index; if (np > 0) { cache[i - 0].index = 0; cache[ni].index = (np - 1) | ((sa_sint_t)(T[np - 2] > T[np - 1]) << (SAINT_BIT - 1)); cache[ni].symbol = T[np - 1]; } }
        }

        sa_sint_t v1 = cache[i - 1].symbol;
        if (v1 >= 0)
        {
            cache[i - 1].symbol = --induction_bucket[v1];
            if (cache[i - 1].symbol >= omp_block_start) { sa_sint_t ni = cache[i - 1].symbol, np = cache[i - 1].index; if (np > 0) { cache[i - 1].index = 0; cache[ni].index = (np - 1) | ((sa_sint_t)(T[np - 2] > T[np - 1]) << (SAINT_BIT - 1)); cache[ni].symbol = T[np - 1]; }}
        }
    }

    for (j -= prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t v = cache[i].symbol;
        if (v >= 0)
        {
            cache[i].symbol = --induction_bucket[v];
            if (cache[i].symbol >= omp_block_start) { sa_sint_t ni = cache[i].symbol, np = cache[i].index; if (np > 0) { cache[i].index = 0; cache[ni].index = (np - 1) | ((sa_sint_t)(T[np - 2] > T[np - 1]) << (SAINT_BIT - 1)); cache[ni].symbol = T[np - 1]; } }
        }
    }
}

static sa_sint_t libsais_partial_sorting_scan_right_to_left_32s_6k_block_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, sa_sint_t d, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 16384)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(cache);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            d = libsais_partial_sorting_scan_right_to_left_32s_6k(T, SA, buckets, d, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                libsais_partial_sorting_scan_right_to_left_32s_6k_block_gather(T, SA, cache - block_start, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                d = libsais_partial_sorting_scan_right_to_left_32s_6k_block_sort(T, buckets, d, cache - block_start, block_start, block_size);
            }

            #pragma omp barrier

            {
                libsais_place_cached_suffixes(SA, cache - block_start, omp_block_start, omp_block_size);
            }
        }
#endif
    }

    return d;
}

static sa_sint_t libsais_partial_sorting_scan_right_to_left_32s_4k_block_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t d, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 16384)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(cache);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            d = libsais_partial_sorting_scan_right_to_left_32s_4k(T, SA, k, buckets, d, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                libsais_partial_sorting_scan_right_to_left_32s_4k_block_gather(T, SA, cache - block_start, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                d = libsais_partial_sorting_scan_right_to_left_32s_4k_block_sort(T, k, buckets, d, cache - block_start, block_start, block_size);
            }

            #pragma omp barrier

            {
                libsais_compact_and_place_cached_suffixes(SA, cache - block_start, omp_block_start, omp_block_size);
            }
        }
#endif
    }

    return d;
}

static void libsais_partial_sorting_scan_right_to_left_32s_1k_block_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 16384)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(cache);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            libsais_partial_sorting_scan_right_to_left_32s_1k(T, SA, buckets, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                libsais_partial_sorting_scan_right_to_left_32s_1k_block_gather(T, SA, cache - block_start, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                libsais_partial_sorting_scan_right_to_left_32s_1k_block_sort(T, buckets, cache - block_start, block_start, block_size);
            }

            #pragma omp barrier

            {
                libsais_compact_and_place_cached_suffixes(SA, cache - block_start, omp_block_start, omp_block_size);
            }
        }
#endif
    }
}

#endif

static sa_sint_t libsais_partial_sorting_scan_right_to_left_32s_6k_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t * RESTRICT buckets, sa_sint_t first_lms_suffix, sa_sint_t left_suffixes_count, sa_sint_t d, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#ifdef AOCL_BWT
    fast_sint_t scan_start = (fast_sint_t)left_suffixes_count;
    fast_sint_t scan_end = (fast_sint_t)n;
#else
    fast_sint_t scan_start    = (fast_sint_t)left_suffixes_count + 1;
    fast_sint_t scan_end      = (fast_sint_t)n - (fast_sint_t)first_lms_suffix;
#endif /* AOCL_BWT */

    if (threads == 1 || (scan_end - scan_start) < 65536)
    {
#ifdef AOCL_BWT
        d = AOCL_libsais_partial_sorting_scan_right_to_left_32s_6k(T, SA, buckets, d, scan_start, n - scan_start, n);
#else
        d = libsais_partial_sorting_scan_right_to_left_32s_6k(T, SA, buckets, d, scan_start, scan_end - scan_start);
#endif /* AOCL_BWT */
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start, block_end;
        for (block_start = scan_end - 1; block_start >= scan_start; block_start = block_end)
        {
            block_end = block_start - (fast_sint_t)threads * LIBSAIS_PER_THREAD_CACHE_SIZE; if (block_end < scan_start) { block_end = scan_start - 1; }

            d = libsais_partial_sorting_scan_right_to_left_32s_6k_block_omp(T, SA, buckets, d, thread_state[0].state.cache, block_end + 1, block_start - block_end, threads);
        }
    }
#else
    UNUSED(thread_state);
#endif

    return d;
}

static sa_sint_t libsais_partial_sorting_scan_right_to_left_32s_4k_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t d, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    if (threads == 1 || n < 65536)
    {
        d = libsais_partial_sorting_scan_right_to_left_32s_4k(T, SA, k, buckets, d, 0, n);
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start, block_end;
        for (block_start = (fast_sint_t)n - 1; block_start >= 0; block_start = block_end)
        {
            block_end = block_start - (fast_sint_t)threads * LIBSAIS_PER_THREAD_CACHE_SIZE; if (block_end < 0) { block_end = -1; }

            d = libsais_partial_sorting_scan_right_to_left_32s_4k_block_omp(T, SA, k, buckets, d, thread_state[0].state.cache, block_end + 1, block_start - block_end, threads);
        }
    }
#else
    UNUSED(thread_state);
#endif

    return d;
}

static void libsais_partial_sorting_scan_right_to_left_32s_1k_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    if (threads == 1 || n < 65536)
    {
        libsais_partial_sorting_scan_right_to_left_32s_1k(T, SA, buckets, 0, n);
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start, block_end;
        for (block_start = (fast_sint_t)n - 1; block_start >= 0; block_start = block_end)
        {
            block_end = block_start - (fast_sint_t)threads * LIBSAIS_PER_THREAD_CACHE_SIZE; if (block_end < 0) { block_end = -1; }

            libsais_partial_sorting_scan_right_to_left_32s_1k_block_omp(T, SA, buckets, thread_state[0].state.cache, block_end + 1, block_start - block_end, threads);
        }
    }
#else
    UNUSED(thread_state);
#endif
}

static fast_sint_t libsais_partial_sorting_gather_lms_suffixes_32s_4k(sa_sint_t * RESTRICT SA, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j, l;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - 3, l = omp_block_start; i < j; i += 4)
    {
        libsais_prefetchr(&SA[i + prefetch_distance]);

        sa_sint_t s0 = SA[i + 0]; SA[l] = (s0 - SUFFIX_GROUP_MARKER) & (~SUFFIX_GROUP_MARKER); l += (s0 < 0);
        sa_sint_t s1 = SA[i + 1]; SA[l] = (s1 - SUFFIX_GROUP_MARKER) & (~SUFFIX_GROUP_MARKER); l += (s1 < 0);
        sa_sint_t s2 = SA[i + 2]; SA[l] = (s2 - SUFFIX_GROUP_MARKER) & (~SUFFIX_GROUP_MARKER); l += (s2 < 0);
        sa_sint_t s3 = SA[i + 3]; SA[l] = (s3 - SUFFIX_GROUP_MARKER) & (~SUFFIX_GROUP_MARKER); l += (s3 < 0);
    }

    for (j += 3; i < j; i += 1)
    {
        sa_sint_t s = SA[i]; SA[l] = (s - SUFFIX_GROUP_MARKER) & (~SUFFIX_GROUP_MARKER); l += (s < 0);
    }

    return l;
}

static fast_sint_t libsais_partial_sorting_gather_lms_suffixes_32s_1k(sa_sint_t * RESTRICT SA, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j, l;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - 3, l = omp_block_start; i < j; i += 4)
    {
        libsais_prefetchr(&SA[i + prefetch_distance]);

        sa_sint_t s0 = SA[i + 0]; SA[l] = s0 & SAINT_MAX; l += (s0 < 0);
        sa_sint_t s1 = SA[i + 1]; SA[l] = s1 & SAINT_MAX; l += (s1 < 0);
        sa_sint_t s2 = SA[i + 2]; SA[l] = s2 & SAINT_MAX; l += (s2 < 0);
        sa_sint_t s3 = SA[i + 3]; SA[l] = s3 & SAINT_MAX; l += (s3 < 0);
    }

    for (j += 3; i < j; i += 1)
    {
        sa_sint_t s = SA[i]; SA[l] = s & SAINT_MAX; l += (s < 0);
    }

    return l;
}

static void libsais_partial_sorting_gather_lms_suffixes_32s_4k_omp(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (n / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : n - omp_block_start;

        if (omp_num_threads == 1)
        {
            libsais_partial_sorting_gather_lms_suffixes_32s_4k(SA, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                thread_state[omp_thread_num].state.position = omp_block_start;
                thread_state[omp_thread_num].state.count = libsais_partial_sorting_gather_lms_suffixes_32s_4k(SA, omp_block_start, omp_block_size) - omp_block_start;
            }

            #pragma omp barrier

            #pragma omp master
            {
                fast_sint_t t, position = 0;
                for (t = 0; t < omp_num_threads; ++t)
                { 
                    if (t > 0 && thread_state[t].state.count > 0)
                    {
                        memmove(&SA[position], &SA[thread_state[t].state.position], (size_t)thread_state[t].state.count * sizeof(sa_sint_t));
                    }

                    position += thread_state[t].state.count;
                }
            }
        }
#endif
    }
}

static void libsais_partial_sorting_gather_lms_suffixes_32s_1k_omp(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (n / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : n - omp_block_start;

        if (omp_num_threads == 1)
        {
            libsais_partial_sorting_gather_lms_suffixes_32s_1k(SA, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                thread_state[omp_thread_num].state.position = omp_block_start;
                thread_state[omp_thread_num].state.count = libsais_partial_sorting_gather_lms_suffixes_32s_1k(SA, omp_block_start, omp_block_size) - omp_block_start;
            }

            #pragma omp barrier

            #pragma omp master
            {
                fast_sint_t t, position = 0;
                for (t = 0; t < omp_num_threads; ++t)
                { 
                    if (t > 0 && thread_state[t].state.count > 0)
                    {
                        memmove(&SA[position], &SA[thread_state[t].state.position], (size_t)thread_state[t].state.count * sizeof(sa_sint_t));
                    }

                    position += thread_state[t].state.count;
                }
            }
        }
#endif
    }
}

static void libsais_induce_partial_order_8u_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t first_lms_suffix, sa_sint_t left_suffixes_count, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    memset(&buckets[2 * ALPHABET_SIZE], 0, (size_t)2 * ALPHABET_SIZE * sizeof(sa_sint_t));
    /*
        SA[]:
        |<-------------------------------- n integers -------------------------------->|

        |<------------------------------- used --------------------------------------->|
        |<------ left_suffixes_count ------>|<---- (n-left_suffixes_count)------------>|
        |          prev is L type           |            prev is S type                |
    */
    sa_sint_t d = libsais_partial_sorting_scan_left_to_right_8u_omp(T, SA, n, k, buckets, left_suffixes_count, 0, threads, thread_state);
    libsais_partial_sorting_shift_markers_8u_omp(SA, n, buckets, threads);
    libsais_partial_sorting_scan_right_to_left_8u_omp(T, SA, n, k, buckets, first_lms_suffix, left_suffixes_count, d, threads, thread_state);
}

static void libsais_induce_partial_order_32s_6k_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t first_lms_suffix, sa_sint_t left_suffixes_count, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    // Similar to `distinct_names` array in `AOCL_libsais_partial_sorting_scan_left_to_right_8u` the array `buckets[4*index+"var"]` here var can be 2 or 3,
    // BWT Modifications:   New function with prefix AOCL_ is created. When the current element is 0, n-1 is considered as prev element
    sa_sint_t d = libsais_partial_sorting_scan_left_to_right_32s_6k_omp(T, SA, n, buckets, left_suffixes_count, 0, threads, thread_state);
    // Changes: None
    libsais_partial_sorting_shift_markers_32s_6k_omp(SA, k, buckets, threads);
    // &buckets[0] is reset to tail pointers with help of buckets[4*k], earlier it used to contian head pointers.
    libsais_partial_sorting_shift_buckets_32s_6k(k, buckets);
    // similar to left_to_right scan, all LMS are partially sorted after this
    // BWT Modifications:   New function with prefix AOCL_ is created. When the current element is 0, n-1 is considered as prev element
    libsais_partial_sorting_scan_right_to_left_32s_6k_omp(T, SA, n, buckets, first_lms_suffix, left_suffixes_count, d, threads, thread_state);
}

static void libsais_induce_partial_order_32s_4k_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    memset(buckets, 0, 2 * (size_t)k * sizeof(sa_sint_t));

    sa_sint_t d = libsais_partial_sorting_scan_left_to_right_32s_4k_omp(T, SA, n, k, buckets, 0, threads, thread_state);
    libsais_partial_sorting_shift_markers_32s_4k(SA, n);
    libsais_partial_sorting_scan_right_to_left_32s_4k_omp(T, SA, n, k, buckets, d, threads, thread_state);
    libsais_partial_sorting_gather_lms_suffixes_32s_4k_omp(SA, n, threads, thread_state);
}

static void libsais_induce_partial_order_32s_2k_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    libsais_partial_sorting_scan_left_to_right_32s_1k_omp(T, SA, n, &buckets[1 * (fast_sint_t)k], threads, thread_state);
    libsais_partial_sorting_scan_right_to_left_32s_1k_omp(T, SA, n, &buckets[0 * (fast_sint_t)k], threads, thread_state);
    libsais_partial_sorting_gather_lms_suffixes_32s_1k_omp(SA, n, threads, thread_state);
}

static void libsais_induce_partial_order_32s_1k_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    libsais_count_suffixes_32s(T, n, k, buckets);
    libsais_initialize_buckets_start_32s_1k(k, buckets);
    libsais_partial_sorting_scan_left_to_right_32s_1k_omp(T, SA, n, buckets, threads, thread_state);

    libsais_count_suffixes_32s(T, n, k, buckets);
    libsais_initialize_buckets_end_32s_1k(k, buckets);
    libsais_partial_sorting_scan_right_to_left_32s_1k_omp(T, SA, n, buckets, threads, thread_state);

    libsais_partial_sorting_gather_lms_suffixes_32s_1k_omp(SA, n, threads, thread_state);
}

static sa_sint_t libsais_renumber_lms_suffixes_8u(sa_sint_t * RESTRICT SA, sa_sint_t m, sa_sint_t name, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT SAm = &SA[m];

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 3; i < j; i += 4)
    {
        libsais_prefetchr(&SA[i + 2 * prefetch_distance]);

        libsais_prefetchw(&SAm[(SA[i + prefetch_distance + 0] & SAINT_MAX) >> 1]);
        libsais_prefetchw(&SAm[(SA[i + prefetch_distance + 1] & SAINT_MAX) >> 1]);
        libsais_prefetchw(&SAm[(SA[i + prefetch_distance + 2] & SAINT_MAX) >> 1]);
        libsais_prefetchw(&SAm[(SA[i + prefetch_distance + 3] & SAINT_MAX) >> 1]);

        sa_sint_t p0 = SA[i + 0]; SAm[(p0 & SAINT_MAX) >> 1] = name | SAINT_MIN; name += p0 < 0;
        sa_sint_t p1 = SA[i + 1]; SAm[(p1 & SAINT_MAX) >> 1] = name | SAINT_MIN; name += p1 < 0;
        sa_sint_t p2 = SA[i + 2]; SAm[(p2 & SAINT_MAX) >> 1] = name | SAINT_MIN; name += p2 < 0;
        sa_sint_t p3 = SA[i + 3]; SAm[(p3 & SAINT_MAX) >> 1] = name | SAINT_MIN; name += p3 < 0;
    }

    for (j += prefetch_distance + 3; i < j; i += 1)
    {
        sa_sint_t p = SA[i]; SAm[(p & SAINT_MAX) >> 1] = name | SAINT_MIN; name += p < 0;
    }

    return name;
}

static fast_sint_t libsais_gather_marked_lms_suffixes(sa_sint_t * RESTRICT SA, sa_sint_t m, fast_sint_t l, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    l -= 1;

    fast_sint_t i, j;
    for (i = (fast_sint_t)m + omp_block_start + omp_block_size - 1, j = (fast_sint_t)m + omp_block_start + 3; i >= j; i -= 4)
    {
        libsais_prefetchr(&SA[i - prefetch_distance]);

        sa_sint_t s0 = SA[i - 0]; SA[l] = s0 & SAINT_MAX; l -= s0 < 0;
        sa_sint_t s1 = SA[i - 1]; SA[l] = s1 & SAINT_MAX; l -= s1 < 0;
        sa_sint_t s2 = SA[i - 2]; SA[l] = s2 & SAINT_MAX; l -= s2 < 0;
        sa_sint_t s3 = SA[i - 3]; SA[l] = s3 & SAINT_MAX; l -= s3 < 0;
    }

    for (j -= 3; i >= j; i -= 1)
    {
        sa_sint_t s = SA[i]; SA[l] = s & SAINT_MAX; l -= s < 0;
    }

    l += 1;

    return l;
}

static sa_sint_t libsais_renumber_lms_suffixes_8u_omp(sa_sint_t * RESTRICT SA, sa_sint_t m, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    sa_sint_t name = 0;

#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && m >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (m / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : m - omp_block_start;

        if (omp_num_threads == 1)
        {
            name = libsais_renumber_lms_suffixes_8u(SA, m, 0, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                thread_state[omp_thread_num].state.count = libsais_count_negative_marked_suffixes(SA, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            {
                fast_sint_t t, count = 0; for (t = 0; t < omp_thread_num; ++t) { count += thread_state[t].state.count; }

                if (omp_thread_num == omp_num_threads - 1)
                {
                    name = (sa_sint_t)(count + thread_state[omp_thread_num].state.count);
                }

                libsais_renumber_lms_suffixes_8u(SA, m, (sa_sint_t)count, omp_block_start, omp_block_size);
            }
        }
#endif
    }

    return name;
}

static void libsais_gather_marked_lms_suffixes_omp(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t fs, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 131072)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (((fast_sint_t)n >> 1) / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
#ifdef AOCL_BWT
        /*
            There are cases where the last element could be LMS character, simple n/2 won't handle that case
            so (n+1)/2 will handle even if the last index is LMS character
        */
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : ((fast_sint_t)(n+1) >> 1) - omp_block_start;
#else
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : ((fast_sint_t)n >> 1) - omp_block_start;
#endif /* AOCL_BWT */
        if (omp_num_threads == 1)
        {
            libsais_gather_marked_lms_suffixes(SA, m, (fast_sint_t)n + (fast_sint_t)fs, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                if (omp_thread_num < omp_num_threads - 1)
                {
                    thread_state[omp_thread_num].state.position = libsais_gather_marked_lms_suffixes(SA, m, (fast_sint_t)m + omp_block_start + omp_block_size, omp_block_start, omp_block_size);
                    thread_state[omp_thread_num].state.count = (fast_sint_t)m + omp_block_start + omp_block_size - thread_state[omp_thread_num].state.position;
                }
                else
                {
                    thread_state[omp_thread_num].state.position = libsais_gather_marked_lms_suffixes(SA, m, (fast_sint_t)n + (fast_sint_t)fs, omp_block_start, omp_block_size);
                    thread_state[omp_thread_num].state.count = (fast_sint_t)n + (fast_sint_t)fs - thread_state[omp_thread_num].state.position;
                }
            }

            #pragma omp barrier

            #pragma omp master
            {
                fast_sint_t t, position = (fast_sint_t)n + (fast_sint_t)fs;
                    
                for (t = omp_num_threads - 1; t >= 0; --t)
                { 
                    position -= thread_state[t].state.count;
                    if (t != omp_num_threads - 1 && thread_state[t].state.count > 0)
                    {
                        memmove(&SA[position], &SA[thread_state[t].state.position], (size_t)thread_state[t].state.count * sizeof(sa_sint_t));
                    }
                }
            }
        }
#endif
    }
}

static sa_sint_t libsais_renumber_and_gather_lms_suffixes_omp(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t fs, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#ifdef AOCL_BWT
    /*
        There are cases where the last element in original array could be LMS character,
        so making all the n/2 elements 0 won't handle that case,
        hence (n+1)/2 elements are initialized to zero to consider the last character in original array.
    */
    memset(&SA[m], 0, ((size_t)(n+1) >> 1) * sizeof(sa_sint_t));
#else
    memset(&SA[m], 0, ((size_t)n >> 1) * sizeof(sa_sint_t));
#endif /* AOCL_BWT */

    sa_sint_t name = libsais_renumber_lms_suffixes_8u_omp(SA, m, threads, thread_state);
    if (name < m)
    {
        libsais_gather_marked_lms_suffixes_omp(SA, n, m, fs, threads, thread_state);
    }
    else
    {
        fast_sint_t i; for (i = 0; i < m; i += 1) { SA[i] &= SAINT_MAX; }
    }

    return name;
}

static sa_sint_t libsais_renumber_distinct_lms_suffixes_32s_4k(sa_sint_t * RESTRICT SA, sa_sint_t m, sa_sint_t name, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT SAm = &SA[m];

    fast_sint_t i, j; sa_sint_t p0, p1, p2, p3 = 0;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 3; i < j; i += 4)
    {
        libsais_prefetchw(&SA[i + 2 * prefetch_distance]);

        libsais_prefetchw(&SAm[(SA[i + prefetch_distance + 0] & SAINT_MAX) >> 1]);
        libsais_prefetchw(&SAm[(SA[i + prefetch_distance + 1] & SAINT_MAX) >> 1]);
        libsais_prefetchw(&SAm[(SA[i + prefetch_distance + 2] & SAINT_MAX) >> 1]);
        libsais_prefetchw(&SAm[(SA[i + prefetch_distance + 3] & SAINT_MAX) >> 1]);

        p0 = SA[i + 0]; SAm[(SA[i + 0] = p0 & SAINT_MAX) >> 1] = name | (p0 & p3 & SAINT_MIN); name += p0 < 0;
        p1 = SA[i + 1]; SAm[(SA[i + 1] = p1 & SAINT_MAX) >> 1] = name | (p1 & p0 & SAINT_MIN); name += p1 < 0;
        p2 = SA[i + 2]; SAm[(SA[i + 2] = p2 & SAINT_MAX) >> 1] = name | (p2 & p1 & SAINT_MIN); name += p2 < 0;
        p3 = SA[i + 3]; SAm[(SA[i + 3] = p3 & SAINT_MAX) >> 1] = name | (p3 & p2 & SAINT_MIN); name += p3 < 0;
    }

    for (j += prefetch_distance + 3; i < j; i += 1)
    {
        p2 = p3; p3 = SA[i]; SAm[(SA[i] = p3 & SAINT_MAX) >> 1] = name | (p3 & p2 & SAINT_MIN); name += p3 < 0;
    }

    return name;
}

static void libsais_mark_distinct_lms_suffixes_32s(sa_sint_t * RESTRICT SA, sa_sint_t m, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j; sa_sint_t p0, p1, p2, p3 = 0;
    for (i = (fast_sint_t)m + omp_block_start, j = (fast_sint_t)m + omp_block_start + omp_block_size - 3; i < j; i += 4)
    {
        libsais_prefetchw(&SA[i + prefetch_distance]);

        p0 = SA[i + 0]; SA[i + 0] = p0 & (p3 | SAINT_MAX); p0 = (p0 == 0) ? p3 : p0;
        p1 = SA[i + 1]; SA[i + 1] = p1 & (p0 | SAINT_MAX); p1 = (p1 == 0) ? p0 : p1;
        p2 = SA[i + 2]; SA[i + 2] = p2 & (p1 | SAINT_MAX); p2 = (p2 == 0) ? p1 : p2;
        p3 = SA[i + 3]; SA[i + 3] = p3 & (p2 | SAINT_MAX); p3 = (p3 == 0) ? p2 : p3;
    }

    for (j += 3; i < j; i += 1)
    {
        p2 = p3; p3 = SA[i]; SA[i] = p3 & (p2 | SAINT_MAX); p3 = (p3 == 0) ? p2 : p3;
    }
}

static void libsais_clamp_lms_suffixes_length_32s(sa_sint_t * RESTRICT SA, sa_sint_t m, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT SAm = &SA[m];

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - 3; i < j; i += 4)
    {
        libsais_prefetchw(&SAm[i + prefetch_distance]);

        SAm[i + 0] = (SAm[i + 0] < 0 ? SAm[i + 0] : 0) & SAINT_MAX;
        SAm[i + 1] = (SAm[i + 1] < 0 ? SAm[i + 1] : 0) & SAINT_MAX;
        SAm[i + 2] = (SAm[i + 2] < 0 ? SAm[i + 2] : 0) & SAINT_MAX;
        SAm[i + 3] = (SAm[i + 3] < 0 ? SAm[i + 3] : 0) & SAINT_MAX;
    }

    for (j += 3; i < j; i += 1)
    {
        SAm[i] = (SAm[i] < 0 ? SAm[i] : 0) & SAINT_MAX;
    }
}

static sa_sint_t libsais_renumber_distinct_lms_suffixes_32s_4k_omp(sa_sint_t * RESTRICT SA, sa_sint_t m, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    sa_sint_t name = 0;

#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && m >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (m / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : m - omp_block_start;

        if (omp_num_threads == 1)
        {
            name = libsais_renumber_distinct_lms_suffixes_32s_4k(SA, m, 1, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                thread_state[omp_thread_num].state.count = libsais_count_negative_marked_suffixes(SA, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            {
                fast_sint_t t, count = 1; for (t = 0; t < omp_thread_num; ++t) { count += thread_state[t].state.count; }

                if (omp_thread_num == omp_num_threads - 1)
                {
                    name = (sa_sint_t)(count + thread_state[omp_thread_num].state.count);
                }

                libsais_renumber_distinct_lms_suffixes_32s_4k(SA, m, (sa_sint_t)count, omp_block_start, omp_block_size);
            }
        }
#endif
    }

    return name - 1;
}

static void libsais_mark_distinct_lms_suffixes_32s_omp(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 131072)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
        fast_sint_t omp_block_stride  = (((fast_sint_t)n >> 1) / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : ((fast_sint_t)n >> 1) - omp_block_start;
#else
        UNUSED(threads);

        fast_sint_t omp_block_start   = 0;
    #ifdef AOCL_BWT
        fast_sint_t omp_block_size    = (fast_sint_t)(n+1) >> 1;
    #else
        fast_sint_t omp_block_size    = (fast_sint_t)n >> 1;
    #endif /* AOCL_BWT */
#endif
        libsais_mark_distinct_lms_suffixes_32s(SA, m, omp_block_start, omp_block_size);
    }
}

static void libsais_clamp_lms_suffixes_length_32s_omp(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 131072)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
        fast_sint_t omp_block_stride  = (((fast_sint_t)n >> 1) / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : ((fast_sint_t)n >> 1) - omp_block_start;
#else
        UNUSED(threads);

        fast_sint_t omp_block_start   = 0;
        fast_sint_t omp_block_size    = (fast_sint_t)n >> 1;
#endif
        libsais_clamp_lms_suffixes_length_32s(SA, m, omp_block_start, omp_block_size);
    }
}

static sa_sint_t libsais_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#ifdef AOCL_BWT
    memset(&SA[m], 0, ((size_t)(n+1) >> 1) * sizeof(sa_sint_t));
#else
    memset(&SA[m], 0, ((size_t)n >> 1) * sizeof(sa_sint_t));
#endif /* AOCL_BWT */

    sa_sint_t name = libsais_renumber_distinct_lms_suffixes_32s_4k_omp(SA, m, threads, thread_state);
    if (name < m)
    {
        libsais_mark_distinct_lms_suffixes_32s_omp(SA, n, m, threads);
    }

    return name;
}

static sa_sint_t libsais_renumber_and_mark_distinct_lms_suffixes_32s_1k_omp(sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t threads)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT SAm = &SA[m];

    {
        libsais_gather_lms_suffixes_32s(T, SA, n);

        memset(&SA[m], 0, ((size_t)n - (size_t)m - (size_t)m) * sizeof(sa_sint_t));

        fast_sint_t i, j;
        for (i = (fast_sint_t)n - (fast_sint_t)m, j = (fast_sint_t)n - 1 - prefetch_distance - 3; i < j; i += 4)
        {
            libsais_prefetchr(&SA[i + 2 * prefetch_distance]);

            libsais_prefetchw(&SAm[((sa_uint_t)SA[i + prefetch_distance + 0]) >> 1]);
            libsais_prefetchw(&SAm[((sa_uint_t)SA[i + prefetch_distance + 1]) >> 1]);
            libsais_prefetchw(&SAm[((sa_uint_t)SA[i + prefetch_distance + 2]) >> 1]);
            libsais_prefetchw(&SAm[((sa_uint_t)SA[i + prefetch_distance + 3]) >> 1]);

            SAm[((sa_uint_t)SA[i + 0]) >> 1] = SA[i + 1] - SA[i + 0] + 1 + SAINT_MIN;
            SAm[((sa_uint_t)SA[i + 1]) >> 1] = SA[i + 2] - SA[i + 1] + 1 + SAINT_MIN;
            SAm[((sa_uint_t)SA[i + 2]) >> 1] = SA[i + 3] - SA[i + 2] + 1 + SAINT_MIN;
            SAm[((sa_uint_t)SA[i + 3]) >> 1] = SA[i + 4] - SA[i + 3] + 1 + SAINT_MIN;
        }

        for (j += prefetch_distance + 3; i < j; i += 1)
        {
            SAm[((sa_uint_t)SA[i]) >> 1] = SA[i + 1] - SA[i] + 1 + SAINT_MIN;
        }

        SAm[((sa_uint_t)SA[n - 1]) >> 1] = 1 + SAINT_MIN;
    }

    {
        libsais_clamp_lms_suffixes_length_32s_omp(SA, n, m, threads);
    }

    sa_sint_t name = 1;

    {
        fast_sint_t i, j, p = SA[0], plen = SAm[p >> 1]; sa_sint_t pdiff = SAINT_MIN;
        for (i = 1, j = m - prefetch_distance - 1; i < j; i += 2)
        {
            libsais_prefetchr(&SA[i + 2 * prefetch_distance]);
            
            libsais_prefetchw(&SAm[((sa_uint_t)SA[i + prefetch_distance + 0]) >> 1]); libsais_prefetchr(&T[((sa_uint_t)SA[i + prefetch_distance + 0])]);
            libsais_prefetchw(&SAm[((sa_uint_t)SA[i + prefetch_distance + 1]) >> 1]); libsais_prefetchr(&T[((sa_uint_t)SA[i + prefetch_distance + 1])]);

            fast_sint_t q = SA[i + 0], qlen = SAm[q >> 1]; sa_sint_t qdiff = SAINT_MIN;
            if (plen == qlen) { fast_sint_t l = 0; do { if (T[p + l] != T[q + l]) { break; } } while (++l < qlen); qdiff = (sa_sint_t)(l - qlen) & SAINT_MIN; }
            SAm[p >> 1] = name | (pdiff & qdiff); name += (qdiff < 0);

            p = SA[i + 1]; plen = SAm[p >> 1]; pdiff = SAINT_MIN;
            if (qlen == plen) { fast_sint_t l = 0; do { if (T[q + l] != T[p + l]) { break; } } while (++l < plen); pdiff = (sa_sint_t)(l - plen) & SAINT_MIN; }
            SAm[q >> 1] = name | (qdiff & pdiff); name += (pdiff < 0);
        }

        for (j += prefetch_distance + 1; i < j; i += 1)
        {
            fast_sint_t q = SA[i], qlen = SAm[q >> 1]; sa_sint_t qdiff = SAINT_MIN;
            if (plen == qlen) { fast_sint_t l = 0; do { if (T[p + l] != T[q + l]) { break; } } while (++l < plen); qdiff = (sa_sint_t)(l - plen) & SAINT_MIN; }
            SAm[p >> 1] = name | (pdiff & qdiff); name += (qdiff < 0);

            p = q; plen = qlen; pdiff = qdiff;
        }

        SAm[p >> 1] = name | pdiff; name++;
    }

    if (name <= m)
    {
        libsais_mark_distinct_lms_suffixes_32s_omp(SA, n, m, threads);
    }

    return name - 1;
}

static void libsais_reconstruct_lms_suffixes(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    const sa_sint_t * RESTRICT SAnm = &SA[n - m];

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 3; i < j; i += 4)
    {
        libsais_prefetchw(&SA[i + 2 * prefetch_distance]);

        libsais_prefetchr(&SAnm[SA[i + prefetch_distance + 0]]);
        libsais_prefetchr(&SAnm[SA[i + prefetch_distance + 1]]);
        libsais_prefetchr(&SAnm[SA[i + prefetch_distance + 2]]);
        libsais_prefetchr(&SAnm[SA[i + prefetch_distance + 3]]);

        SA[i + 0] = SAnm[SA[i + 0]];
        SA[i + 1] = SAnm[SA[i + 1]];
        SA[i + 2] = SAnm[SA[i + 2]];
        SA[i + 3] = SAnm[SA[i + 3]];
    }

    for (j += prefetch_distance + 3; i < j; i += 1)
    {
        SA[i] = SAnm[SA[i]];
    }
}

static void libsais_reconstruct_lms_suffixes_omp(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && m >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
        fast_sint_t omp_block_stride  = (m / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : m - omp_block_start;
#else
        UNUSED(threads);

        fast_sint_t omp_block_start   = 0;
        fast_sint_t omp_block_size    = m;
#endif

        libsais_reconstruct_lms_suffixes(SA, n, m, omp_block_start, omp_block_size);
    }
}

static void libsais_place_lms_suffixes_interval_8u(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, const sa_sint_t * RESTRICT buckets)
{
    const sa_sint_t * RESTRICT bucket_end = &buckets[7 * ALPHABET_SIZE];

    fast_sint_t c, j = n;
    for (c = ALPHABET_SIZE - 2; c >= 0; --c)
    {
        fast_sint_t l = (fast_sint_t)buckets[BUCKETS_INDEX2(c, 1) + BUCKETS_INDEX2(1, 0)] - (fast_sint_t)buckets[BUCKETS_INDEX2(c, 1)];
        if (l > 0)
        {
            fast_sint_t i = bucket_end[c];
            if (j - i > 0)
            {
                #ifdef AOCL_BWT
                    memset(&SA[i], 255, (size_t)(j - i) * sizeof(sa_sint_t));
                #else
                    memset(&SA[i], 0, (size_t)(j - i) * sizeof(sa_sint_t));
                #endif /* AOCL_BWT */
            }

            memmove(&SA[j = (i - l)], &SA[m -= (sa_sint_t)l], (size_t)l * sizeof(sa_sint_t));
        }
    }
#ifdef AOCL_BWT
    memset(&SA[0], 255, (size_t)j * sizeof(sa_sint_t));
#else
    memset(&SA[0], 0, (size_t)j * sizeof(sa_sint_t));
#endif /* AOCL_BWT */
}

static void libsais_place_lms_suffixes_interval_32s_4k(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t m, const sa_sint_t * RESTRICT buckets)
{
    const sa_sint_t * RESTRICT bucket_end = &buckets[3 * (fast_sint_t)k];

    fast_sint_t c, j = n;
    for (c = (fast_sint_t)k - 2; c >= 0; --c)
    {
        fast_sint_t l = (fast_sint_t)buckets[BUCKETS_INDEX2(c, 1) + BUCKETS_INDEX2(1, 0)] - (fast_sint_t)buckets[BUCKETS_INDEX2(c, 1)];
        if (l > 0)
        {
            fast_sint_t i = bucket_end[c];
            if (j - i > 0)
            {
                memset(&SA[i], 0, (size_t)(j - i) * sizeof(sa_sint_t));
            }

            memmove(&SA[j = (i - l)], &SA[m -= (sa_sint_t)l], (size_t)l * sizeof(sa_sint_t));
        }
    }

    memset(&SA[0], 0, (size_t)j * sizeof(sa_sint_t));
}

static void libsais_place_lms_suffixes_interval_32s_2k(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t m, const sa_sint_t * RESTRICT buckets)
{
    fast_sint_t j = n;

    if (k > 1)
    {
        fast_sint_t c;
        for (c = BUCKETS_INDEX2((fast_sint_t)k - 2, 0); c >= BUCKETS_INDEX2(0, 0); c -= BUCKETS_INDEX2(1, 0))
        {
            fast_sint_t l = (fast_sint_t)buckets[c + BUCKETS_INDEX2(1, 1)] - (fast_sint_t)buckets[c + BUCKETS_INDEX2(0, 1)];
            if (l > 0)
            {
                fast_sint_t i = buckets[c];
                if (j - i > 0)
                {
                    memset(&SA[i], 0, (size_t)(j - i) * sizeof(sa_sint_t));
                }

                memmove(&SA[j = (i - l)], &SA[m -= (sa_sint_t)l], (size_t)l * sizeof(sa_sint_t));
            }
        }
    }

    memset(&SA[0], 0, (size_t)j * sizeof(sa_sint_t));
}

static void libsais_place_lms_suffixes_interval_32s_1k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t m, sa_sint_t * RESTRICT buckets)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t c = k - 1; fast_sint_t i, l = buckets[c];
    for (i = (fast_sint_t)m - 1; i >= prefetch_distance + 3; i -= 4)
    {
        libsais_prefetchr(&SA[i - 2 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i - prefetch_distance - 0]]);
        libsais_prefetchr(&T[SA[i - prefetch_distance - 1]]);
        libsais_prefetchr(&T[SA[i - prefetch_distance - 2]]);
        libsais_prefetchr(&T[SA[i - prefetch_distance - 3]]);

        sa_sint_t p0 = SA[i - 0]; if (T[p0] != c) { c = T[p0]; memset(&SA[buckets[c]], 0, (size_t)(l - buckets[c]) * sizeof(sa_sint_t)); l = buckets[c]; } SA[--l] = p0;
        sa_sint_t p1 = SA[i - 1]; if (T[p1] != c) { c = T[p1]; memset(&SA[buckets[c]], 0, (size_t)(l - buckets[c]) * sizeof(sa_sint_t)); l = buckets[c]; } SA[--l] = p1;
        sa_sint_t p2 = SA[i - 2]; if (T[p2] != c) { c = T[p2]; memset(&SA[buckets[c]], 0, (size_t)(l - buckets[c]) * sizeof(sa_sint_t)); l = buckets[c]; } SA[--l] = p2;
        sa_sint_t p3 = SA[i - 3]; if (T[p3] != c) { c = T[p3]; memset(&SA[buckets[c]], 0, (size_t)(l - buckets[c]) * sizeof(sa_sint_t)); l = buckets[c]; } SA[--l] = p3;
    }

    for (; i >= 0; i -= 1)
    {
        sa_sint_t p = SA[i]; if (T[p] != c) { c = T[p]; memset(&SA[buckets[c]], 0, (size_t)(l - buckets[c]) * sizeof(sa_sint_t)); l = buckets[c]; } SA[--l] = p;
    }

    memset(&SA[0], 0, (size_t)l * sizeof(sa_sint_t));
}

static void libsais_place_lms_suffixes_histogram_32s_6k(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t m, const sa_sint_t * RESTRICT buckets)
{
    const sa_sint_t * RESTRICT bucket_end = &buckets[5 * (fast_sint_t)k];

    fast_sint_t c, j = n;
    for (c = (fast_sint_t)k - 2; c >= 0; --c)
    {
        fast_sint_t l = (fast_sint_t)buckets[BUCKETS_INDEX4(c, 1)];
        if (l > 0)
        {
            fast_sint_t i = bucket_end[c];
            if (j - i > 0)
            {
                #ifdef AOCL_BWT
                    memset(&SA[i], 255, (size_t)(j - i) * sizeof(sa_sint_t));
                #else
                    memset(&SA[i], 0, (size_t)(j - i) * sizeof(sa_sint_t));
                #endif /* AOCL_BWT */
            }

            memmove(&SA[j = (i - l)], &SA[m -= (sa_sint_t)l], (size_t)l * sizeof(sa_sint_t));
        }
    }

#ifdef AOCL_BWT
    memset(&SA[0], 255, (size_t)j * sizeof(sa_sint_t));
#else
    memset(&SA[0], 0, (size_t)j * sizeof(sa_sint_t));
#endif /* AOCL_BWT */
}

static void libsais_place_lms_suffixes_histogram_32s_4k(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t m, const sa_sint_t * RESTRICT buckets)
{
    const sa_sint_t * RESTRICT bucket_end = &buckets[3 * (fast_sint_t)k];

    fast_sint_t c, j = n;
    for (c = (fast_sint_t)k - 2; c >= 0; --c)
    {
        fast_sint_t l = (fast_sint_t)buckets[BUCKETS_INDEX2(c, 1)];
        if (l > 0)
        {
            fast_sint_t i = bucket_end[c];
            if (j - i > 0)
            {
                #ifdef AOCL_BWT
                    memset(&SA[i], 255, (size_t)(j - i) * sizeof(sa_sint_t));
                #else
                    memset(&SA[i], 0, (size_t)(j - i) * sizeof(sa_sint_t));
                #endif /* AOCL_BWT */
            }

            memmove(&SA[j = (i - l)], &SA[m -= (sa_sint_t)l], (size_t)l * sizeof(sa_sint_t));
        }
    }
#ifdef AOCL_BWT
    memset(&SA[0], 255, (size_t)j * sizeof(sa_sint_t));
#else
    memset(&SA[0], 0, (size_t)j * sizeof(sa_sint_t));
#endif /* AOCL_BWT */
}

static void libsais_place_lms_suffixes_histogram_32s_2k(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t m, const sa_sint_t * RESTRICT buckets)
{
    fast_sint_t j = n;

    if (k > 1)
    {
        fast_sint_t c;
        for (c = BUCKETS_INDEX2((fast_sint_t)k - 2, 0); c >= BUCKETS_INDEX2(0, 0); c -= BUCKETS_INDEX2(1, 0))
        {
            fast_sint_t l = (fast_sint_t)buckets[c + BUCKETS_INDEX2(0, 1)];
            if (l > 0)
            {
                fast_sint_t i = buckets[c];
                if (j - i > 0)
                {
                    memset(&SA[i], 0, (size_t)(j - i) * sizeof(sa_sint_t));
                }

                memmove(&SA[j = (i - l)], &SA[m -= (sa_sint_t)l], (size_t)l * sizeof(sa_sint_t));
            }
        }
    }

    memset(&SA[0], 0, (size_t)j * sizeof(sa_sint_t));
}

static void libsais_final_bwt_scan_left_to_right_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&SA[i + 2 * prefetch_distance]);

        sa_sint_t s0 = SA[i + prefetch_distance + 0]; const uint8_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i + prefetch_distance + 1]; const uint8_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

        sa_sint_t p0 = SA[i + 0]; SA[i + 0] = p0 & SAINT_MAX; if (p0 > 0) { p0--; SA[i + 0] = T[p0] | SAINT_MIN; SA[induction_bucket[T[p0]]++] = p0 | ((sa_sint_t)(T[p0 - (p0 > 0)] < T[p0]) << (SAINT_BIT - 1)); }
        sa_sint_t p1 = SA[i + 1]; SA[i + 1] = p1 & SAINT_MAX; if (p1 > 0) { p1--; SA[i + 1] = T[p1] | SAINT_MIN; SA[induction_bucket[T[p1]]++] = p1 | ((sa_sint_t)(T[p1 - (p1 > 0)] < T[p1]) << (SAINT_BIT - 1)); }
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t p = SA[i]; SA[i] = p & SAINT_MAX; if (p > 0) { p--; SA[i] = T[p] | SAINT_MIN; SA[induction_bucket[T[p]]++] = p | ((sa_sint_t)(T[p - (p > 0)] < T[p]) << (SAINT_BIT - 1)); }
    }
}

static void libsais_final_bwt_aux_scan_left_to_right_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t rm, sa_sint_t * RESTRICT I, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&SA[i + 2 * prefetch_distance]);

        sa_sint_t s0 = SA[i + prefetch_distance + 0]; const uint8_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i + prefetch_distance + 1]; const uint8_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

        sa_sint_t p0 = SA[i + 0]; SA[i + 0] = p0 & SAINT_MAX; if (p0 > 0) { p0--; SA[i + 0] = T[p0] | SAINT_MIN; SA[induction_bucket[T[p0]]++] = p0 | ((sa_sint_t)(T[p0 - (p0 > 0)] < T[p0]) << (SAINT_BIT - 1)); if ((p0 & rm) == 0) { I[p0 / (rm + 1)] = induction_bucket[T[p0]]; }}
        sa_sint_t p1 = SA[i + 1]; SA[i + 1] = p1 & SAINT_MAX; if (p1 > 0) { p1--; SA[i + 1] = T[p1] | SAINT_MIN; SA[induction_bucket[T[p1]]++] = p1 | ((sa_sint_t)(T[p1 - (p1 > 0)] < T[p1]) << (SAINT_BIT - 1)); if ((p1 & rm) == 0) { I[p1 / (rm + 1)] = induction_bucket[T[p1]]; }}
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t p = SA[i]; SA[i] = p & SAINT_MAX; if (p > 0) { p--; SA[i] = T[p] | SAINT_MIN; SA[induction_bucket[T[p]]++] = p | ((sa_sint_t)(T[p - (p > 0)] < T[p]) << (SAINT_BIT - 1)); if ((p & rm) == 0) { I[p / (rm + 1)] = induction_bucket[T[p]]; } }
    }
}

#ifndef AOCL_BWT
static void libsais_final_sorting_scan_left_to_right_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&SA[i + 2 * prefetch_distance]);

        sa_sint_t s0 = SA[i + prefetch_distance + 0]; const uint8_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i + prefetch_distance + 1]; const uint8_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

        sa_sint_t p0 = SA[i + 0]; SA[i + 0] = p0 ^ SAINT_MIN; if (p0 > 0) { p0--; SA[induction_bucket[T[p0]]++] = p0 | ((sa_sint_t)(T[p0 - (p0 > 0)] < T[p0]) << (SAINT_BIT - 1)); }
        sa_sint_t p1 = SA[i + 1]; SA[i + 1] = p1 ^ SAINT_MIN; if (p1 > 0) { p1--; SA[induction_bucket[T[p1]]++] = p1 | ((sa_sint_t)(T[p1 - (p1 > 0)] < T[p1]) << (SAINT_BIT - 1)); }
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t p = SA[i]; SA[i] = p ^ SAINT_MIN; if (p > 0) { p--; SA[induction_bucket[T[p]]++] = p | ((sa_sint_t)(T[p - (p > 0)] < T[p]) << (SAINT_BIT - 1)); }
    }
}
#else
static sa_sint_t AOCL_libsais_final_sorting_scan_left_to_right_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    sa_sint_t zero_index = -1;
    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size; i < j; i += 1)
    {
        sa_sint_t p = SA[i];
        SA[i] = p ^ SAINT_MIN;
        if (p > 1)
        {
            p--;
            // Whenever the integer is made negative, those will be skipped to consider for sorting
            // SA[i] = p ^ SAINT_MIN; makes the integers positive if they are negative already
            // Below, negative sign will be formed if prev char is S
            SA[induction_bucket[T[p]]++] = p | ((sa_sint_t)(T[p - 1] < T[p]) << (SAINT_BIT - 1));
        }
        else if(p == 1)
        {
            p--;
            zero_index = induction_bucket[T[p]];
            SA[induction_bucket[T[p]]++] = p | ((sa_sint_t)(T[(omp_block_size + p - 1)%omp_block_size] < T[p]) << (SAINT_BIT - 1));
        }
        else if(p == 0)
        {
            p = omp_block_size-1;
            SA[induction_bucket[T[p]]++] = p | ((sa_sint_t)(T[(omp_block_size + p - 1)%omp_block_size] < T[p]) << (SAINT_BIT - 1));
        }
    }
    return zero_index;
}
#endif /* AOCL_BWT */

static void libsais_final_sorting_scan_left_to_right_32s(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
#ifndef AOCL_BWT
    for (i = omp_block_start, j = omp_block_start + omp_block_size - 2 * prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&SA[i + 3 * prefetch_distance]);

        sa_sint_t s0 = SA[i + 2 * prefetch_distance + 0]; const sa_sint_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i + 2 * prefetch_distance + 1]; const sa_sint_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);
        sa_sint_t s2 = SA[i + 1 * prefetch_distance + 0]; if (s2 > 0) { libsais_prefetchw(&induction_bucket[T[s2 - 1]]); libsais_prefetchr(&T[s2] - 2); }
        sa_sint_t s3 = SA[i + 1 * prefetch_distance + 1]; if (s3 > 0) { libsais_prefetchw(&induction_bucket[T[s3 - 1]]); libsais_prefetchr(&T[s3] - 2); }

        sa_sint_t p0 = SA[i + 0]; SA[i + 0] = p0 ^ SAINT_MIN; if (p0 > 0) { p0--; SA[induction_bucket[T[p0]]++] = p0 | ((sa_sint_t)(T[p0 - (p0 > 0)] < T[p0]) << (SAINT_BIT - 1)); }
        sa_sint_t p1 = SA[i + 1]; SA[i + 1] = p1 ^ SAINT_MIN; if (p1 > 0) { p1--; SA[induction_bucket[T[p1]]++] = p1 | ((sa_sint_t)(T[p1 - (p1 > 0)] < T[p1]) << (SAINT_BIT - 1)); }
    }
    
    for (j += 2 * prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t p = SA[i]; SA[i] = p ^ SAINT_MIN; if (p > 0) { p--; SA[induction_bucket[T[p]]++] = p | ((sa_sint_t)(T[p - (p > 0)] < T[p]) << (SAINT_BIT - 1)); }
    }
#else
    i = omp_block_start, j = omp_block_start + omp_block_size - 2 * prefetch_distance - 1;

    for (j += 2 * prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t p = SA[i];
        SA[i] = p ^ SAINT_MIN;
        if (p > 1)
        {
            p--;
            // Whenever the integer is made negative, those will be skipped to consider for sorting
            // SA[i] = p ^ SAINT_MIN; makes the integers positive if they are negative already
            // Below, negative sign will be formed if prev char is S
            SA[induction_bucket[T[p]]++] = p | ((sa_sint_t)(T[p - 1] < T[p]) << (SAINT_BIT - 1));
        }
        else if(p == 1)
        {
            p--;
            SA[induction_bucket[T[p]]++] = p | ((sa_sint_t)(T[(omp_block_size + p - 1)%omp_block_size] < T[p]) << (SAINT_BIT - 1));
        }
        else if(p==0)
        {
            p = omp_block_size-1;
            SA[induction_bucket[T[p]]++] = p | ((sa_sint_t)(T[(omp_block_size + p - 1)%omp_block_size] < T[p]) << (SAINT_BIT - 1));
        }
    }
#endif /* AOCL_BWT */
}

#if defined(LIBSAIS_OPENMP)

static fast_sint_t libsais_final_bwt_scan_left_to_right_8u_block_prepare(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
   const fast_sint_t prefetch_distance = 32;

   memset(buckets, 0, k * sizeof(sa_sint_t));

   fast_sint_t i, j, count = 0;
   for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
   {
       libsais_prefetchw(&SA[i + 2 * prefetch_distance]);

       sa_sint_t s0 = SA[i + prefetch_distance + 0]; const uint8_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
       sa_sint_t s1 = SA[i + prefetch_distance + 1]; const uint8_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

       sa_sint_t p0 = SA[i + 0]; SA[i + 0] = p0 & SAINT_MAX; if (p0 > 0) { p0--; SA[i + 0] = T[p0] | SAINT_MIN; buckets[cache[count].symbol = T[p0]]++; cache[count++].index = p0 | ((sa_sint_t)(T[p0 - (p0 > 0)] < T[p0]) << (SAINT_BIT - 1)); }
       sa_sint_t p1 = SA[i + 1]; SA[i + 1] = p1 & SAINT_MAX; if (p1 > 0) { p1--; SA[i + 1] = T[p1] | SAINT_MIN; buckets[cache[count].symbol = T[p1]]++; cache[count++].index = p1 | ((sa_sint_t)(T[p1 - (p1 > 0)] < T[p1]) << (SAINT_BIT - 1)); }
   }

   for (j += prefetch_distance + 1; i < j; i += 1)
   {
       sa_sint_t p = SA[i]; SA[i] = p & SAINT_MAX; if (p > 0) { p--; SA[i] = T[p] | SAINT_MIN; buckets[cache[count].symbol = T[p]]++; cache[count++].index = p | ((sa_sint_t)(T[p - (p > 0)] < T[p]) << (SAINT_BIT - 1)); }
   }

   return count;
}

static fast_sint_t libsais_final_sorting_scan_left_to_right_8u_block_prepare(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
   const fast_sint_t prefetch_distance = 32;

   memset(buckets, 0, k * sizeof(sa_sint_t));

   fast_sint_t i, j, count = 0;
   for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
   {
       libsais_prefetchw(&SA[i + 2 * prefetch_distance]);

       sa_sint_t s0 = SA[i + prefetch_distance + 0]; const uint8_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
       sa_sint_t s1 = SA[i + prefetch_distance + 1]; const uint8_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

       sa_sint_t p0 = SA[i + 0]; SA[i + 0] = p0 ^ SAINT_MIN; if (p0 > 0) { p0--; buckets[cache[count].symbol = T[p0]]++; cache[count++].index = p0 | ((sa_sint_t)(T[p0 - (p0 > 0)] < T[p0]) << (SAINT_BIT - 1)); }
       sa_sint_t p1 = SA[i + 1]; SA[i + 1] = p1 ^ SAINT_MIN; if (p1 > 0) { p1--; buckets[cache[count].symbol = T[p1]]++; cache[count++].index = p1 | ((sa_sint_t)(T[p1 - (p1 > 0)] < T[p1]) << (SAINT_BIT - 1)); }
   }

   for (j += prefetch_distance + 1; i < j; i += 1)
   {
       sa_sint_t p = SA[i]; SA[i] = p ^ SAINT_MIN; if (p > 0) { p--; buckets[cache[count].symbol = T[p]]++; cache[count++].index = p | ((sa_sint_t)(T[p - (p > 0)] < T[p]) << (SAINT_BIT - 1)); }
   }

   return count;
}

static void libsais_final_order_scan_left_to_right_8u_block_place(sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t count)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = 0, j = count - 3; i < j; i += 4)
    {
        libsais_prefetchr(&cache[i + prefetch_distance]);

        SA[buckets[cache[i + 0].symbol]++] = cache[i + 0].index;
        SA[buckets[cache[i + 1].symbol]++] = cache[i + 1].index;
        SA[buckets[cache[i + 2].symbol]++] = cache[i + 2].index;
        SA[buckets[cache[i + 3].symbol]++] = cache[i + 3].index;
    }

    for (j += 3; i < j; i += 1)
    {
        SA[buckets[cache[i].symbol]++] = cache[i].index;
    }
}

static void libsais_final_bwt_aux_scan_left_to_right_8u_block_place(sa_sint_t * RESTRICT SA, sa_sint_t rm, sa_sint_t * RESTRICT I, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t count)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = 0, j = count - 3; i < j; i += 4)
    {
        libsais_prefetchr(&cache[i + prefetch_distance]);

        SA[buckets[cache[i + 0].symbol]++] = cache[i + 0].index; if ((cache[i + 0].index & rm) == 0) { I[(cache[i + 0].index & SAINT_MAX) / (rm + 1)] = buckets[cache[i + 0].symbol]; }
        SA[buckets[cache[i + 1].symbol]++] = cache[i + 1].index; if ((cache[i + 1].index & rm) == 0) { I[(cache[i + 1].index & SAINT_MAX) / (rm + 1)] = buckets[cache[i + 1].symbol]; }
        SA[buckets[cache[i + 2].symbol]++] = cache[i + 2].index; if ((cache[i + 2].index & rm) == 0) { I[(cache[i + 2].index & SAINT_MAX) / (rm + 1)] = buckets[cache[i + 2].symbol]; }
        SA[buckets[cache[i + 3].symbol]++] = cache[i + 3].index; if ((cache[i + 3].index & rm) == 0) { I[(cache[i + 3].index & SAINT_MAX) / (rm + 1)] = buckets[cache[i + 3].symbol]; }
    }

    for (j += 3; i < j; i += 1)
    {
        SA[buckets[cache[i].symbol]++] = cache[i].index; if ((cache[i].index & rm) == 0) { I[(cache[i].index & SAINT_MAX) / (rm + 1)] = buckets[cache[i].symbol]; }
    }
}

static void libsais_final_sorting_scan_left_to_right_32s_block_gather(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&SA[i + 2 * prefetch_distance]);

        sa_sint_t s0 = SA[i + prefetch_distance + 0]; const sa_sint_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i + prefetch_distance + 1]; const sa_sint_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

        libsais_prefetchw(&cache[i + prefetch_distance]);

        sa_sint_t symbol0 = SAINT_MIN, p0 = SA[i + 0]; SA[i + 0] = p0 ^ SAINT_MIN; if (p0 > 0) { p0--; cache[i + 0].index = p0 | ((sa_sint_t)(T[p0 - (p0 > 0)] < T[p0]) << (SAINT_BIT - 1)); symbol0 = T[p0]; } cache[i + 0].symbol = symbol0;
        sa_sint_t symbol1 = SAINT_MIN, p1 = SA[i + 1]; SA[i + 1] = p1 ^ SAINT_MIN; if (p1 > 0) { p1--; cache[i + 1].index = p1 | ((sa_sint_t)(T[p1 - (p1 > 0)] < T[p1]) << (SAINT_BIT - 1)); symbol1 = T[p1]; } cache[i + 1].symbol = symbol1;
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t symbol = SAINT_MIN, p = SA[i]; SA[i] = p ^ SAINT_MIN; if (p > 0) { p--; cache[i].index = p | ((sa_sint_t)(T[p - (p > 0)] < T[p]) << (SAINT_BIT - 1)); symbol = T[p]; } cache[i].symbol = symbol;
    }
}

static void libsais_final_sorting_scan_left_to_right_32s_block_sort(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT induction_bucket, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j, omp_block_end = omp_block_start + omp_block_size;
    for (i = omp_block_start, j = omp_block_end - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&cache[i + 2 * prefetch_distance]);

        sa_sint_t s0 = cache[i + prefetch_distance + 0].symbol; const sa_sint_t * Is0 = &induction_bucket[s0]; libsais_prefetchw(s0 >= 0 ? Is0 : NULL);
        sa_sint_t s1 = cache[i + prefetch_distance + 1].symbol; const sa_sint_t * Is1 = &induction_bucket[s1]; libsais_prefetchw(s1 >= 0 ? Is1 : NULL);
        
        sa_sint_t v0 = cache[i + 0].symbol;
        if (v0 >= 0)
        {
            cache[i + 0].symbol = induction_bucket[v0]++;
            if (cache[i + 0].symbol < omp_block_end) { sa_sint_t ni = cache[i + 0].symbol, np = cache[i + 0].index; cache[i + 0].index = np ^ SAINT_MIN; if (np > 0) { np--; cache[ni].index = np | ((sa_sint_t)(T[np - (np > 0)] < T[np]) << (SAINT_BIT - 1)); cache[ni].symbol = T[np]; } }
        }

        sa_sint_t v1 = cache[i + 1].symbol;
        if (v1 >= 0)
        {
            cache[i + 1].symbol = induction_bucket[v1]++;
            if (cache[i + 1].symbol < omp_block_end) { sa_sint_t ni = cache[i + 1].symbol, np = cache[i + 1].index; cache[i + 1].index = np ^ SAINT_MIN; if (np > 0) { np--; cache[ni].index = np | ((sa_sint_t)(T[np - (np > 0)] < T[np]) << (SAINT_BIT - 1)); cache[ni].symbol = T[np]; } }
        }
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t v = cache[i].symbol;
        if (v >= 0)
        {
            cache[i].symbol = induction_bucket[v]++;
            if (cache[i].symbol < omp_block_end) { sa_sint_t ni = cache[i].symbol, np = cache[i].index; cache[i].index = np ^ SAINT_MIN; if (np > 0) { np--; cache[ni].index = np | ((sa_sint_t)(T[np - (np > 0)] < T[np]) << (SAINT_BIT - 1)); cache[ni].symbol = T[np]; } }
        }
    }
}

static void libsais_final_bwt_scan_left_to_right_8u_block_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT induction_bucket, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 64 * max(k, 256) && omp_get_dynamic() == 0)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(k); UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            libsais_final_bwt_scan_left_to_right_8u(T, SA, induction_bucket, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                thread_state[omp_thread_num].state.count = libsais_final_bwt_scan_left_to_right_8u_block_prepare(T, SA, k, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                fast_sint_t t;
                for (t = 0; t < omp_num_threads; ++t)
                {
                    sa_sint_t * RESTRICT temp_bucket = thread_state[t].state.buckets;
                    fast_sint_t c; for (c = 0; c < k; c += 1) { sa_sint_t A = induction_bucket[c], B = temp_bucket[c]; induction_bucket[c] = A + B; temp_bucket[c] = A; }
                }
            }

            #pragma omp barrier

            {
                libsais_final_order_scan_left_to_right_8u_block_place(SA, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, thread_state[omp_thread_num].state.count);
            }
        }
#endif
    }
}

static void libsais_final_bwt_aux_scan_left_to_right_8u_block_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t rm, sa_sint_t * RESTRICT I, sa_sint_t * RESTRICT induction_bucket, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 64 * max(k, 256) && omp_get_dynamic() == 0)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(k); UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            libsais_final_bwt_aux_scan_left_to_right_8u(T, SA, rm, I, induction_bucket, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                thread_state[omp_thread_num].state.count = libsais_final_bwt_scan_left_to_right_8u_block_prepare(T, SA, k, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                fast_sint_t t;
                for (t = 0; t < omp_num_threads; ++t)
                {
                    sa_sint_t * RESTRICT temp_bucket = thread_state[t].state.buckets;
                    fast_sint_t c; for (c = 0; c < k; c += 1) { sa_sint_t A = induction_bucket[c], B = temp_bucket[c]; induction_bucket[c] = A + B; temp_bucket[c] = A; }
                }
            }

            #pragma omp barrier

            {
                libsais_final_bwt_aux_scan_left_to_right_8u_block_place(SA, rm, I, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, thread_state[omp_thread_num].state.count);
            }
        }
#endif
    }
}

static void libsais_final_sorting_scan_left_to_right_8u_block_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT induction_bucket, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 64 * max(k, 256) && omp_get_dynamic() == 0)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(k); UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            libsais_final_sorting_scan_left_to_right_8u(T, SA, induction_bucket, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                thread_state[omp_thread_num].state.count = libsais_final_sorting_scan_left_to_right_8u_block_prepare(T, SA, k, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                fast_sint_t t;
                for (t = 0; t < omp_num_threads; ++t)
                {
                    sa_sint_t * RESTRICT temp_bucket = thread_state[t].state.buckets;
                    fast_sint_t c; for (c = 0; c < k; c += 1) { sa_sint_t A = induction_bucket[c], B = temp_bucket[c]; induction_bucket[c] = A + B; temp_bucket[c] = A; }
                }
            }

            #pragma omp barrier

            {
                libsais_final_order_scan_left_to_right_8u_block_place(SA, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, thread_state[omp_thread_num].state.count);
            }
        }
#endif
    }
}

static void libsais_final_sorting_scan_left_to_right_32s_block_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 16384)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(cache);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            libsais_final_sorting_scan_left_to_right_32s(T, SA, buckets, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                libsais_final_sorting_scan_left_to_right_32s_block_gather(T, SA, cache - block_start, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                libsais_final_sorting_scan_left_to_right_32s_block_sort(T, buckets, cache - block_start, block_start, block_size);
            }

            #pragma omp barrier

            {
                libsais_compact_and_place_cached_suffixes(SA, cache - block_start, omp_block_start, omp_block_size);
            }
        }
#endif
    }
}

#endif

static void libsais_final_bwt_scan_left_to_right_8u_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, fast_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT induction_bucket, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    SA[induction_bucket[T[(sa_sint_t)n - 1]]++] = ((sa_sint_t)n - 1) | ((sa_sint_t)(T[(sa_sint_t)n - 2] < T[(sa_sint_t)n - 1]) << (SAINT_BIT - 1));

    if (threads == 1 || n < 65536)
    {
        libsais_final_bwt_scan_left_to_right_8u(T, SA, induction_bucket, 0, n);
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start;
        for (block_start = 0; block_start < n; )
        {
            if (SA[block_start] == 0)
            {
                block_start++;
            }
            else
            {
                fast_sint_t block_max_end = block_start + ((fast_sint_t)threads) * (LIBSAIS_PER_THREAD_CACHE_SIZE - 16 * (fast_sint_t)threads); if (block_max_end > n) { block_max_end = n;}
                fast_sint_t block_end     = block_start + 1; while (block_end < block_max_end && SA[block_end] != 0) { block_end++; }
                fast_sint_t block_size    = block_end - block_start;

                if (block_size < 32)
                {
                    for (; block_start < block_end; block_start += 1)
                    {
                        sa_sint_t p = SA[block_start]; SA[block_start] = p & SAINT_MAX; if (p > 0) { p--; SA[block_start] = T[p] | SAINT_MIN; SA[induction_bucket[T[p]]++] = p | ((sa_sint_t)(T[p - (p > 0)] < T[p]) << (SAINT_BIT - 1)); }
                    }
                }
                else
                {
                    libsais_final_bwt_scan_left_to_right_8u_block_omp(T, SA, k, induction_bucket, block_start, block_size, threads, thread_state);
                    block_start = block_end;
                }
            }
        }
    }
#else
    UNUSED(k); UNUSED(thread_state);
#endif
}

static void libsais_final_bwt_aux_scan_left_to_right_8u_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, fast_sint_t n, sa_sint_t k, sa_sint_t rm, sa_sint_t * RESTRICT I, sa_sint_t * RESTRICT induction_bucket, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    SA[induction_bucket[T[(sa_sint_t)n - 1]]++] = ((sa_sint_t)n - 1) | ((sa_sint_t)(T[(sa_sint_t)n - 2] < T[(sa_sint_t)n - 1]) << (SAINT_BIT - 1));

    if ((((sa_sint_t)n - 1) & rm) == 0) { I[((sa_sint_t)n - 1) / (rm + 1)] = induction_bucket[T[(sa_sint_t)n - 1]]; }

    if (threads == 1 || n < 65536)
    {
        libsais_final_bwt_aux_scan_left_to_right_8u(T, SA, rm, I, induction_bucket, 0, n);
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start;
        for (block_start = 0; block_start < n; )
        {
            if (SA[block_start] == 0)
            {
                block_start++;
            }
            else
            {
                fast_sint_t block_max_end = block_start + ((fast_sint_t)threads) * (LIBSAIS_PER_THREAD_CACHE_SIZE - 16 * (fast_sint_t)threads); if (block_max_end > n) { block_max_end = n;}
                fast_sint_t block_end     = block_start + 1; while (block_end < block_max_end && SA[block_end] != 0) { block_end++; }
                fast_sint_t block_size    = block_end - block_start;

                if (block_size < 32)
                {
                    for (; block_start < block_end; block_start += 1)
                    {
                        sa_sint_t p = SA[block_start]; SA[block_start] = p & SAINT_MAX; if (p > 0) { p--; SA[block_start] = T[p] | SAINT_MIN; SA[induction_bucket[T[p]]++] = p | ((sa_sint_t)(T[p - (p > 0)] < T[p]) << (SAINT_BIT - 1)); if ((p & rm) == 0) { I[p / (rm + 1)] = induction_bucket[T[p]]; } }
                    }
                }
                else
                {
                    libsais_final_bwt_aux_scan_left_to_right_8u_block_omp(T, SA, k, rm, I, induction_bucket, block_start, block_size, threads, thread_state);
                    block_start = block_end;
                }
            }
        }
    }
#else
    UNUSED(k); UNUSED(thread_state);
#endif
}

#ifndef AOCL_BWT
static void libsais_final_sorting_scan_left_to_right_8u_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, fast_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT induction_bucket, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    SA[induction_bucket[T[(sa_sint_t)n - 1]]++] = ((sa_sint_t)n - 1) | ((sa_sint_t)(T[(sa_sint_t)n - 2] < T[(sa_sint_t)n - 1]) << (SAINT_BIT - 1));

    if (threads == 1 || n < 65536)
    {
        libsais_final_sorting_scan_left_to_right_8u(T, SA, induction_bucket, 0, n);
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start;
        for (block_start = 0; block_start < n; )
        {
            if (SA[block_start] == 0)
            {
                block_start++;
            }
            else
            {
                fast_sint_t block_max_end = block_start + ((fast_sint_t)threads) * (LIBSAIS_PER_THREAD_CACHE_SIZE - 16 * (fast_sint_t)threads); if (block_max_end > n) { block_max_end = n;}
                fast_sint_t block_end     = block_start + 1; while (block_end < block_max_end && SA[block_end] != 0) { block_end++; }
                fast_sint_t block_size    = block_end - block_start;

                if (block_size < 32)
                {
                    for (; block_start < block_end; block_start += 1)
                    {
                        sa_sint_t p = SA[block_start]; SA[block_start] = p ^ SAINT_MIN; if (p > 0) { p--; SA[induction_bucket[T[p]]++] = p | ((sa_sint_t)(T[p - (p > 0)] < T[p]) << (SAINT_BIT - 1)); }
                    }
                }
                else
                {
                    libsais_final_sorting_scan_left_to_right_8u_block_omp(T, SA, k, induction_bucket, block_start, block_size, threads, thread_state);
                    block_start = block_end;
                }
            }
        }
    }
#else
    UNUSED(k); UNUSED(thread_state);
#endif
}
#endif /* AOCL_BWT */
static void libsais_final_sorting_scan_left_to_right_32s_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t * RESTRICT induction_bucket, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#ifndef AOCL_BWT
    SA[induction_bucket[T[n - 1]]++] = (n - 1) | ((sa_sint_t)(T[n - 2] < T[n - 1]) << (SAINT_BIT - 1));
#endif /* AOCL_BWT */
    if (threads == 1 || n < 65536)
    {
        libsais_final_sorting_scan_left_to_right_32s(T, SA, induction_bucket, 0, n);
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start, block_end;
        for (block_start = 0; block_start < n; block_start = block_end)
        {
            block_end = block_start + (fast_sint_t)threads * LIBSAIS_PER_THREAD_CACHE_SIZE; if (block_end > n) { block_end = n; }

            libsais_final_sorting_scan_left_to_right_32s_block_omp(T, SA, induction_bucket, thread_state[0].state.cache, block_start, block_end - block_start, threads);
        }
    }
#else
    UNUSED(thread_state);
#endif
}

static sa_sint_t libsais_final_bwt_scan_right_to_left_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j; sa_sint_t index = -1;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 1; i >= j; i -= 2)
    {
        libsais_prefetchw(&SA[i - 2 * prefetch_distance]);

        sa_sint_t s0 = SA[i - prefetch_distance - 0]; const uint8_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i - prefetch_distance - 1]; const uint8_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

        sa_sint_t p0 = SA[i - 0]; index = (p0 == 0) ? (sa_sint_t)(i - 0) : index;
        SA[i - 0] = p0 & SAINT_MAX; if (p0 > 0) { p0--; uint8_t c0 = T[p0 - (p0 > 0)], c1 = T[p0]; SA[i - 0] = c1; sa_sint_t t = c0 | SAINT_MIN; SA[--induction_bucket[c1]] = (c0 <= c1) ? p0 : t; }

        sa_sint_t p1 = SA[i - 1]; index = (p1 == 0) ? (sa_sint_t)(i - 1) : index;
        SA[i - 1] = p1 & SAINT_MAX; if (p1 > 0) { p1--; uint8_t c0 = T[p1 - (p1 > 0)], c1 = T[p1]; SA[i - 1] = c1; sa_sint_t t = c0 | SAINT_MIN; SA[--induction_bucket[c1]] = (c0 <= c1) ? p1 : t; }
    }

    for (j -= prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i]; index = (p == 0) ? (sa_sint_t)i : index;
        SA[i] = p & SAINT_MAX; if (p > 0) { p--; uint8_t c0 = T[p - (p > 0)], c1 = T[p]; SA[i] = c1; sa_sint_t t = c0 | SAINT_MIN; SA[--induction_bucket[c1]] = (c0 <= c1) ? p : t; }
    }

    return index;
}

static void libsais_final_bwt_aux_scan_right_to_left_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t rm, sa_sint_t * RESTRICT I, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 1; i >= j; i -= 2)
    {
        libsais_prefetchw(&SA[i - 2 * prefetch_distance]);

        sa_sint_t s0 = SA[i - prefetch_distance - 0]; const uint8_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i - prefetch_distance - 1]; const uint8_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

        sa_sint_t p0 = SA[i - 0];
        SA[i - 0] = p0 & SAINT_MAX; if (p0 > 0) { p0--; uint8_t c0 = T[p0 - (p0 > 0)], c1 = T[p0]; SA[i - 0] = c1; sa_sint_t t = c0 | SAINT_MIN; SA[--induction_bucket[c1]] = (c0 <= c1) ? p0 : t; if ((p0 & rm) == 0) { I[p0 / (rm + 1)] = induction_bucket[T[p0]] + 1; } }

        sa_sint_t p1 = SA[i - 1];
        SA[i - 1] = p1 & SAINT_MAX; if (p1 > 0) { p1--; uint8_t c0 = T[p1 - (p1 > 0)], c1 = T[p1]; SA[i - 1] = c1; sa_sint_t t = c0 | SAINT_MIN; SA[--induction_bucket[c1]] = (c0 <= c1) ? p1 : t; if ((p1 & rm) == 0) { I[p1 / (rm + 1)] = induction_bucket[T[p1]] + 1; } }
    }

    for (j -= prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i];
        SA[i] = p & SAINT_MAX; if (p > 0) { p--; uint8_t c0 = T[p - (p > 0)], c1 = T[p]; SA[i] = c1; sa_sint_t t = c0 | SAINT_MIN; SA[--induction_bucket[c1]] = (c0 <= c1) ? p : t; if ((p & rm) == 0) { I[p / (rm + 1)] = induction_bucket[T[p]] + 1; } }
    }
}

#ifndef AOCL_BWT
static void libsais_final_sorting_scan_right_to_left_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 1; i >= j; i -= 2)
    {
        libsais_prefetchw(&SA[i - 2 * prefetch_distance]);

        sa_sint_t s0 = SA[i - prefetch_distance - 0]; const uint8_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i - prefetch_distance - 1]; const uint8_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

        sa_sint_t p0 = SA[i - 0]; SA[i - 0] = p0 & SAINT_MAX; if (p0 > 0) { p0--; SA[--induction_bucket[T[p0]]] = p0 | ((sa_sint_t)(T[p0 - (p0 > 0)] > T[p0]) << (SAINT_BIT - 1)); }
        sa_sint_t p1 = SA[i - 1]; SA[i - 1] = p1 & SAINT_MAX; if (p1 > 0) { p1--; SA[--induction_bucket[T[p1]]] = p1 | ((sa_sint_t)(T[p1 - (p1 > 0)] > T[p1]) << (SAINT_BIT - 1)); }
    }

    for (j -= prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i]; SA[i] = p & SAINT_MAX; if (p > 0) { p--; SA[--induction_bucket[T[p]]] = p | ((sa_sint_t)(T[p - (p > 0)] > T[p]) << (SAINT_BIT - 1)); }
    }
}
#else
static sa_sint_t AOCL_libsais_final_sorting_scan_right_to_left_8u(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    sa_sint_t zero_index = -1;
    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i];
        sa_sint_t temp = p & SAINT_MAX;
        uint8_t c = T[(temp > 0)?(temp-1):(omp_block_size-1)];
        SA[i] = c;
        if (p > 1)
        {
            p--;
            // Whenever the integer is made negative, those will be skipped to consider for sorting
            // SA[i] = p ^ SAINT_MIN; makes the integers positive if they are negative already
            SA[--induction_bucket[c]] = p | ((sa_sint_t)(T[p - (p > 0)] > c) << (SAINT_BIT - 1));
        }
        else if(p == 1)
        {
            p--;
            SA[--induction_bucket[c]] = p | ((sa_sint_t)(T[(omp_block_size + p - 1)%omp_block_size] > c) << (SAINT_BIT - 1));
            zero_index = induction_bucket[c];
        }
        else if(p == 0)
        {
            p = omp_block_size-1;
            SA[--induction_bucket[c]] = p | ((sa_sint_t)(T[(omp_block_size + p - 1)%omp_block_size] > c) << (SAINT_BIT - 1));
        }
    }
    return zero_index;
}
#endif /* AOCL_BWT */

static void libsais_final_sorting_scan_right_to_left_32s(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT induction_bucket, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
#ifndef AOCL_BWT
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + 2 * prefetch_distance + 1; i >= j; i -= 2)
    {
        libsais_prefetchw(&SA[i - 3 * prefetch_distance]);

        sa_sint_t s0 = SA[i - 2 * prefetch_distance - 0]; const sa_sint_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i - 2 * prefetch_distance - 1]; const sa_sint_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);
        sa_sint_t s2 = SA[i - 1 * prefetch_distance - 0]; if (s2 > 0) { libsais_prefetchw(&induction_bucket[T[s2 - 1]]); libsais_prefetchr(&T[s2] - 2); }
        sa_sint_t s3 = SA[i - 1 * prefetch_distance - 1]; if (s3 > 0) { libsais_prefetchw(&induction_bucket[T[s3 - 1]]); libsais_prefetchr(&T[s3] - 2); }

        sa_sint_t p0 = SA[i - 0]; SA[i - 0] = p0 & SAINT_MAX; if (p0 > 0) { p0--; SA[--induction_bucket[T[p0]]] = p0 | ((sa_sint_t)(T[p0 - (p0 > 0)] > T[p0]) << (SAINT_BIT - 1)); }
        sa_sint_t p1 = SA[i - 1]; SA[i - 1] = p1 & SAINT_MAX; if (p1 > 0) { p1--; SA[--induction_bucket[T[p1]]] = p1 | ((sa_sint_t)(T[p1 - (p1 > 0)] > T[p1]) << (SAINT_BIT - 1)); }
    }

    for (j -= 2 * prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i]; SA[i] = p & SAINT_MAX; if (p > 0) { p--; SA[--induction_bucket[T[p]]] = p | ((sa_sint_t)(T[p - (p > 0)] > T[p]) << (SAINT_BIT - 1)); }
    }
#else
    i = omp_block_start + omp_block_size - 1, j = omp_block_start + 2 * prefetch_distance + 1;
    for (j -= 2 * prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i];
        SA[i] = p & SAINT_MAX;
        if (p > 1)
        {
            p--;
            // Whenever the integer is made negative, those will be skipped to consider for sorting
            // SA[i] = p ^ SAINT_MIN; makes the integers positive if they are negative already
            SA[--induction_bucket[T[p]]] = p | ((sa_sint_t)(T[p - (p > 0)] > T[p]) << (SAINT_BIT - 1));
        }
        else if(p == 1)
        {
            p--;
            SA[--induction_bucket[T[p]]] = p | ((sa_sint_t)(T[(omp_block_size + p - 1)%omp_block_size] > T[p]) << (SAINT_BIT - 1));
        }
        else if(p == 0)
        {
            p = omp_block_size-1;
            SA[--induction_bucket[T[p]]] = p | ((sa_sint_t)(T[(omp_block_size + p - 1)%omp_block_size] > T[p]) << (SAINT_BIT - 1));
        }
    }
#endif /* AOCL_BWT */
}

#if defined(LIBSAIS_OPENMP)

static fast_sint_t libsais_final_bwt_scan_right_to_left_8u_block_prepare(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
   const fast_sint_t prefetch_distance = 32;

   memset(buckets, 0, k * sizeof(sa_sint_t));

   fast_sint_t i, j, count = 0;
   for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 1; i >= j; i -= 2)
   {
       libsais_prefetchw(&SA[i - 2 * prefetch_distance]);

       sa_sint_t s0 = SA[i - prefetch_distance - 0]; const uint8_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
       sa_sint_t s1 = SA[i - prefetch_distance - 1]; const uint8_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

       sa_sint_t p0 = SA[i - 0]; SA[i - 0] = p0 & SAINT_MAX; if (p0 > 0) { p0--; uint8_t c0 = T[p0 - (p0 > 0)], c1 = T[p0]; SA[i - 0] = c1; sa_sint_t t = c0 | SAINT_MIN; buckets[cache[count].symbol = c1]++; cache[count++].index = (c0 <= c1) ? p0 : t; }
       sa_sint_t p1 = SA[i - 1]; SA[i - 1] = p1 & SAINT_MAX; if (p1 > 0) { p1--; uint8_t c0 = T[p1 - (p1 > 0)], c1 = T[p1]; SA[i - 1] = c1; sa_sint_t t = c0 | SAINT_MIN; buckets[cache[count].symbol = c1]++; cache[count++].index = (c0 <= c1) ? p1 : t; }
   }

   for (j -= prefetch_distance + 1; i >= j; i -= 1)
   {
       sa_sint_t p = SA[i]; SA[i] = p & SAINT_MAX; if (p > 0) { p--; uint8_t c0 = T[p - (p > 0)], c1 = T[p]; SA[i] = c1; sa_sint_t t = c0 | SAINT_MIN; buckets[cache[count].symbol = c1]++; cache[count++].index = (c0 <= c1) ? p : t; }
   }

   return count;
}

static fast_sint_t libsais_final_bwt_aux_scan_right_to_left_8u_block_prepare(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
   const fast_sint_t prefetch_distance = 32;

   memset(buckets, 0, k * sizeof(sa_sint_t));

   fast_sint_t i, j, count = 0;
   for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 1; i >= j; i -= 2)
   {
       libsais_prefetchw(&SA[i - 2 * prefetch_distance]);

       sa_sint_t s0 = SA[i - prefetch_distance - 0]; const uint8_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
       sa_sint_t s1 = SA[i - prefetch_distance - 1]; const uint8_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

       sa_sint_t p0 = SA[i - 0]; SA[i - 0] = p0 & SAINT_MAX; if (p0 > 0) { p0--; uint8_t c0 = T[p0 - (p0 > 0)], c1 = T[p0]; SA[i - 0] = c1; sa_sint_t t = c0 | SAINT_MIN; buckets[cache[count].symbol = c1]++; cache[count].index = (c0 <= c1) ? p0 : t; cache[count + 1].index = p0; count += 2; }
       sa_sint_t p1 = SA[i - 1]; SA[i - 1] = p1 & SAINT_MAX; if (p1 > 0) { p1--; uint8_t c0 = T[p1 - (p1 > 0)], c1 = T[p1]; SA[i - 1] = c1; sa_sint_t t = c0 | SAINT_MIN; buckets[cache[count].symbol = c1]++; cache[count].index = (c0 <= c1) ? p1 : t; cache[count + 1].index = p1; count += 2; }
   }

   for (j -= prefetch_distance + 1; i >= j; i -= 1)
   {
       sa_sint_t p = SA[i]; SA[i] = p & SAINT_MAX; if (p > 0) { p--; uint8_t c0 = T[p - (p > 0)], c1 = T[p]; SA[i] = c1; sa_sint_t t = c0 | SAINT_MIN; buckets[cache[count].symbol = c1]++; cache[count].index = (c0 <= c1) ? p : t; cache[count + 1].index = p; count += 2; }
   }

   return count;
}

static fast_sint_t libsais_final_sorting_scan_right_to_left_8u_block_prepare(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
   const fast_sint_t prefetch_distance = 32;

   memset(buckets, 0, k * sizeof(sa_sint_t));

   fast_sint_t i, j, count = 0;
   for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 1; i >= j; i -= 2)
   {
       libsais_prefetchw(&SA[i - 2 * prefetch_distance]);

       sa_sint_t s0 = SA[i - prefetch_distance - 0]; const uint8_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
       sa_sint_t s1 = SA[i - prefetch_distance - 1]; const uint8_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

       sa_sint_t p0 = SA[i - 0]; SA[i - 0] = p0 & SAINT_MAX; if (p0 > 0) { p0--; buckets[cache[count].symbol = T[p0]]++; cache[count++].index = p0 | ((sa_sint_t)(T[p0 - (p0 > 0)] > T[p0]) << (SAINT_BIT - 1)); }
       sa_sint_t p1 = SA[i - 1]; SA[i - 1] = p1 & SAINT_MAX; if (p1 > 0) { p1--; buckets[cache[count].symbol = T[p1]]++; cache[count++].index = p1 | ((sa_sint_t)(T[p1 - (p1 > 0)] > T[p1]) << (SAINT_BIT - 1)); }
   }

   for (j -= prefetch_distance + 1; i >= j; i -= 1)
   {
       sa_sint_t p = SA[i]; SA[i] = p & SAINT_MAX; if (p > 0) { p--; buckets[cache[count].symbol = T[p]]++; cache[count++].index = p | ((sa_sint_t)(T[p - (p > 0)] > T[p]) << (SAINT_BIT - 1)); }
   }

   return count;
}

static void libsais_final_order_scan_right_to_left_8u_block_place(sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t count)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = 0, j = count - 3; i < j; i += 4)
    {
        libsais_prefetchr(&cache[i + prefetch_distance]);

        SA[--buckets[cache[i + 0].symbol]] = cache[i + 0].index;
        SA[--buckets[cache[i + 1].symbol]] = cache[i + 1].index;
        SA[--buckets[cache[i + 2].symbol]] = cache[i + 2].index;
        SA[--buckets[cache[i + 3].symbol]] = cache[i + 3].index;
    }

    for (j += 3; i < j; i += 1)
    {
        SA[--buckets[cache[i].symbol]] = cache[i].index;
    }
}

static void libsais_final_bwt_aux_scan_right_to_left_8u_block_place(sa_sint_t * RESTRICT SA, sa_sint_t rm, sa_sint_t * RESTRICT I, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t count)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = 0, j = count - 6; i < j; i += 8)
    {
        libsais_prefetchr(&cache[i + prefetch_distance]);

        SA[--buckets[cache[i + 0].symbol]] = cache[i + 0].index; if ((cache[i + 1].index & rm) == 0) { I[cache[i + 1].index / (rm + 1)] = buckets[cache[i + 0].symbol] + 1; }
        SA[--buckets[cache[i + 2].symbol]] = cache[i + 2].index; if ((cache[i + 3].index & rm) == 0) { I[cache[i + 3].index / (rm + 1)] = buckets[cache[i + 2].symbol] + 1; }
        SA[--buckets[cache[i + 4].symbol]] = cache[i + 4].index; if ((cache[i + 5].index & rm) == 0) { I[cache[i + 5].index / (rm + 1)] = buckets[cache[i + 4].symbol] + 1; }
        SA[--buckets[cache[i + 6].symbol]] = cache[i + 6].index; if ((cache[i + 7].index & rm) == 0) { I[cache[i + 7].index / (rm + 1)] = buckets[cache[i + 6].symbol] + 1; }
    }

    for (j += 6; i < j; i += 2)
    {
        SA[--buckets[cache[i].symbol]] = cache[i].index; if ((cache[i + 1].index & rm) == 0) { I[(cache[i + 1].index & SAINT_MAX) / (rm + 1)] = buckets[cache[i].symbol] + 1; }
    }
}

static void libsais_final_sorting_scan_right_to_left_32s_block_gather(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchw(&SA[i + 2 * prefetch_distance]);

        sa_sint_t s0 = SA[i + prefetch_distance + 0]; const sa_sint_t * Ts0 = &T[s0] - 1; libsais_prefetchr(s0 > 0 ? Ts0 : NULL); Ts0--; libsais_prefetchr(s0 > 0 ? Ts0 : NULL);
        sa_sint_t s1 = SA[i + prefetch_distance + 1]; const sa_sint_t * Ts1 = &T[s1] - 1; libsais_prefetchr(s1 > 0 ? Ts1 : NULL); Ts1--; libsais_prefetchr(s1 > 0 ? Ts1 : NULL);

        libsais_prefetchw(&cache[i + prefetch_distance]);

        sa_sint_t symbol0 = SAINT_MIN, p0 = SA[i + 0]; SA[i + 0] = p0 & SAINT_MAX; if (p0 > 0) { p0--; cache[i + 0].index = p0 | ((sa_sint_t)(T[p0 - (p0 > 0)] > T[p0]) << (SAINT_BIT - 1)); symbol0 = T[p0]; } cache[i + 0].symbol = symbol0;
        sa_sint_t symbol1 = SAINT_MIN, p1 = SA[i + 1]; SA[i + 1] = p1 & SAINT_MAX; if (p1 > 0) { p1--; cache[i + 1].index = p1 | ((sa_sint_t)(T[p1 - (p1 > 0)] > T[p1]) << (SAINT_BIT - 1)); symbol1 = T[p1]; } cache[i + 1].symbol = symbol1;
    }

    for (j += prefetch_distance + 1; i < j; i += 1)
    {
        sa_sint_t symbol = SAINT_MIN, p = SA[i]; SA[i] = p & SAINT_MAX; if (p > 0) { p--; cache[i].index = p | ((sa_sint_t)(T[p - (p > 0)] > T[p]) << (SAINT_BIT - 1)); symbol = T[p]; } cache[i].symbol = symbol;
    }
}

static void libsais_final_sorting_scan_right_to_left_32s_block_sort(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT induction_bucket, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start + omp_block_size - 1, j = omp_block_start + prefetch_distance + 1; i >= j; i -= 2)
    {
        libsais_prefetchw(&cache[i - 2 * prefetch_distance]);

        sa_sint_t s0 = cache[i - prefetch_distance - 0].symbol; const sa_sint_t * Is0 = &induction_bucket[s0]; libsais_prefetchw(s0 >= 0 ? Is0 : NULL);
        sa_sint_t s1 = cache[i - prefetch_distance - 1].symbol; const sa_sint_t * Is1 = &induction_bucket[s1]; libsais_prefetchw(s1 >= 0 ? Is1 : NULL);

        sa_sint_t v0 = cache[i - 0].symbol;
        if (v0 >= 0)
        {
            cache[i - 0].symbol = --induction_bucket[v0];
            if (cache[i - 0].symbol >= omp_block_start) { sa_sint_t ni = cache[i - 0].symbol, np = cache[i - 0].index; cache[i - 0].index = np & SAINT_MAX; if (np > 0) { np--; cache[ni].index = np | ((sa_sint_t)(T[np - (np > 0)] > T[np]) << (SAINT_BIT - 1)); cache[ni].symbol = T[np]; } }
        }

        sa_sint_t v1 = cache[i - 1].symbol;
        if (v1 >= 0)
        {
            cache[i - 1].symbol = --induction_bucket[v1];
            if (cache[i - 1].symbol >= omp_block_start) { sa_sint_t ni = cache[i - 1].symbol, np = cache[i - 1].index; cache[i - 1].index = np & SAINT_MAX; if (np > 0) { np--; cache[ni].index = np | ((sa_sint_t)(T[np - (np > 0)] > T[np]) << (SAINT_BIT - 1)); cache[ni].symbol = T[np]; } }
        }
    }

    for (j -= prefetch_distance + 1; i >= j; i -= 1)
    {
        sa_sint_t v = cache[i].symbol;
        if (v >= 0)
        {
            cache[i].symbol = --induction_bucket[v];
            if (cache[i].symbol >= omp_block_start) { sa_sint_t ni = cache[i].symbol, np = cache[i].index; cache[i].index = np & SAINT_MAX; if (np > 0) { np--; cache[ni].index = np | ((sa_sint_t)(T[np - (np > 0)] > T[np]) << (SAINT_BIT - 1)); cache[ni].symbol = T[np]; } }
        }
    }
}

static void libsais_final_bwt_scan_right_to_left_8u_block_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT induction_bucket, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 64 * max(k, 256) && omp_get_dynamic() == 0)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(k); UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            libsais_final_bwt_scan_right_to_left_8u(T, SA, induction_bucket, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                thread_state[omp_thread_num].state.count = libsais_final_bwt_scan_right_to_left_8u_block_prepare(T, SA, k, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                fast_sint_t t;
                for (t = omp_num_threads - 1; t >= 0; --t)
                {
                    sa_sint_t * RESTRICT temp_bucket = thread_state[t].state.buckets;
                    fast_sint_t c; for (c = 0; c < k; c += 1) { sa_sint_t A = induction_bucket[c], B = temp_bucket[c]; induction_bucket[c] = A - B; temp_bucket[c] = A; }
                }
            }

            #pragma omp barrier

            {
                libsais_final_order_scan_right_to_left_8u_block_place(SA, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, thread_state[omp_thread_num].state.count);
            }
        }
#endif
    }
}

static void libsais_final_bwt_aux_scan_right_to_left_8u_block_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t rm, sa_sint_t * RESTRICT I, sa_sint_t * RESTRICT induction_bucket, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 64 * max(k, 256) && omp_get_dynamic() == 0)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(k); UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            libsais_final_bwt_aux_scan_right_to_left_8u(T, SA, rm, I, induction_bucket, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                thread_state[omp_thread_num].state.count = libsais_final_bwt_aux_scan_right_to_left_8u_block_prepare(T, SA, k, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                fast_sint_t t;
                for (t = omp_num_threads - 1; t >= 0; --t)
                {
                    sa_sint_t * RESTRICT temp_bucket = thread_state[t].state.buckets;
                    fast_sint_t c; for (c = 0; c < k; c += 1) { sa_sint_t A = induction_bucket[c], B = temp_bucket[c]; induction_bucket[c] = A - B; temp_bucket[c] = A; }
                }
            }

            #pragma omp barrier

            {
                libsais_final_bwt_aux_scan_right_to_left_8u_block_place(SA, rm, I, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, thread_state[omp_thread_num].state.count);
            }
        }
#endif
    }
}

static void libsais_final_sorting_scan_right_to_left_8u_block_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t k, sa_sint_t * RESTRICT induction_bucket, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 64 * max(k, 256) && omp_get_dynamic() == 0)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(k); UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            libsais_final_sorting_scan_right_to_left_8u(T, SA, induction_bucket, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                thread_state[omp_thread_num].state.count = libsais_final_sorting_scan_right_to_left_8u_block_prepare(T, SA, k, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                fast_sint_t t;
                for (t = omp_num_threads - 1; t >= 0; --t)
                {
                    sa_sint_t * RESTRICT temp_bucket = thread_state[t].state.buckets;
                    fast_sint_t c; for (c = 0; c < k; c += 1) { sa_sint_t A = induction_bucket[c], B = temp_bucket[c]; induction_bucket[c] = A - B; temp_bucket[c] = A; }
                }
            }

            #pragma omp barrier

            {
                libsais_final_order_scan_right_to_left_8u_block_place(SA, thread_state[omp_thread_num].state.buckets, thread_state[omp_thread_num].state.cache, thread_state[omp_thread_num].state.count);
            }
        }
#endif
    }
}

static void libsais_final_sorting_scan_right_to_left_32s_block_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT buckets, LIBSAIS_THREAD_CACHE * RESTRICT cache, fast_sint_t block_start, fast_sint_t block_size, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && block_size >= 16384)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(cache);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (block_size / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : block_size - omp_block_start;

        omp_block_start += block_start;

        if (omp_num_threads == 1)
        {
            libsais_final_sorting_scan_right_to_left_32s(T, SA, buckets, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                libsais_final_sorting_scan_right_to_left_32s_block_gather(T, SA, cache - block_start, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                libsais_final_sorting_scan_right_to_left_32s_block_sort(T, buckets, cache - block_start, block_start, block_size);
            }

            #pragma omp barrier

            {
                libsais_compact_and_place_cached_suffixes(SA, cache - block_start, omp_block_start, omp_block_size);
            }
        }
#endif
    }
}

#endif

static sa_sint_t libsais_final_bwt_scan_right_to_left_8u_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT induction_bucket, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    sa_sint_t index = -1;

    if (threads == 1 || n < 65536)
    {
        index = libsais_final_bwt_scan_right_to_left_8u(T, SA, induction_bucket, 0, n);
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start;
        for (block_start = (fast_sint_t)n - 1; block_start >= 0; )
        {
            if (SA[block_start] == 0)
            {
                index = (sa_sint_t)block_start--;
            }
            else
            {
                fast_sint_t block_max_end = block_start - ((fast_sint_t)threads) * (LIBSAIS_PER_THREAD_CACHE_SIZE - 16 * (fast_sint_t)threads); if (block_max_end < 0) { block_max_end = -1; }
                fast_sint_t block_end     = block_start - 1; while (block_end > block_max_end && SA[block_end] != 0) { block_end--; }
                fast_sint_t block_size    = block_start - block_end;

                if (block_size < 32)
                {
                    for (; block_start > block_end; block_start -= 1)
                    {
                        sa_sint_t p = SA[block_start]; SA[block_start] = p & SAINT_MAX; if (p > 0) { p--; uint8_t c0 = T[p - (p > 0)], c1 = T[p]; SA[block_start] = c1; sa_sint_t t = c0 | SAINT_MIN; SA[--induction_bucket[c1]] = (c0 <= c1) ? p : t; }
                    }
                }
                else
                {
                    libsais_final_bwt_scan_right_to_left_8u_block_omp(T, SA, k, induction_bucket, block_end + 1, block_size, threads, thread_state);
                    block_start = block_end;
                }
            }
        }
    }
#else
    UNUSED(k); UNUSED(thread_state);
#endif

    return index;
}

static void libsais_final_bwt_aux_scan_right_to_left_8u_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t rm, sa_sint_t * RESTRICT I, sa_sint_t * RESTRICT induction_bucket, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    if (threads == 1 || n < 65536)
    {
        libsais_final_bwt_aux_scan_right_to_left_8u(T, SA, rm, I, induction_bucket, 0, n);
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start;
        for (block_start = (fast_sint_t)n - 1; block_start >= 0; )
        {
            if (SA[block_start] == 0)
            {
                block_start--;
            }
            else
            {
                fast_sint_t block_max_end = block_start - ((fast_sint_t)threads) * ((LIBSAIS_PER_THREAD_CACHE_SIZE - 16 * (fast_sint_t)threads) / 2); if (block_max_end < 0) { block_max_end = -1; }
                fast_sint_t block_end     = block_start - 1; while (block_end > block_max_end && SA[block_end] != 0) { block_end--; }
                fast_sint_t block_size    = block_start - block_end;

                if (block_size < 32)
                {
                    for (; block_start > block_end; block_start -= 1)
                    {
                        sa_sint_t p = SA[block_start]; SA[block_start] = p & SAINT_MAX; if (p > 0) { p--; uint8_t c0 = T[p - (p > 0)], c1 = T[p]; SA[block_start] = c1; sa_sint_t t = c0 | SAINT_MIN; SA[--induction_bucket[c1]] = (c0 <= c1) ? p : t; if ((p & rm) == 0) { I[p / (rm + 1)] = induction_bucket[T[p]] + 1; } }
                    }
                }
                else
                {
                    libsais_final_bwt_aux_scan_right_to_left_8u_block_omp(T, SA, k, rm, I, induction_bucket, block_end + 1, block_size, threads, thread_state);
                    block_start = block_end;
                }
            }
        }
    }
#else
    UNUSED(k); UNUSED(thread_state);
#endif
}
#ifndef AOCL_BWT
static void libsais_final_sorting_scan_right_to_left_8u_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT induction_bucket, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    if (threads == 1 || n < 65536)
    {
        libsais_final_sorting_scan_right_to_left_8u(T, SA, induction_bucket, 0, n);
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start;
        for (block_start = (fast_sint_t)n - 1; block_start >= 0; )
        {
            if (SA[block_start] == 0)
            {
                block_start--;
            }
            else
            {
                fast_sint_t block_max_end = block_start - ((fast_sint_t)threads) * (LIBSAIS_PER_THREAD_CACHE_SIZE - 16 * (fast_sint_t)threads); if (block_max_end < -1) { block_max_end = -1; }
                fast_sint_t block_end     = block_start - 1; while (block_end > block_max_end && SA[block_end] != 0) { block_end--; }
                fast_sint_t block_size    = block_start - block_end;

                if (block_size < 32)
                {
                    for (; block_start > block_end; block_start -= 1)
                    {
                        sa_sint_t p = SA[block_start]; SA[block_start] = p & SAINT_MAX; if (p > 0) { p--; SA[--induction_bucket[T[p]]] = p | ((sa_sint_t)(T[p - (p > 0)] > T[p]) << (SAINT_BIT - 1)); }
                    }
                }
                else
                {
                    libsais_final_sorting_scan_right_to_left_8u_block_omp(T, SA, k, induction_bucket, block_end + 1, block_size, threads, thread_state);
                    block_start = block_end;
                }
            }
        }
    }
#else
    UNUSED(k); UNUSED(thread_state);
#endif
}
#endif /* AOCL_BWT */
static void libsais_final_sorting_scan_right_to_left_32s_omp(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t * RESTRICT induction_bucket, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    if (threads == 1 || n < 65536)
    {
        libsais_final_sorting_scan_right_to_left_32s(T, SA, induction_bucket, 0, n);
    }
#if defined(LIBSAIS_OPENMP)
    else
    {
        fast_sint_t block_start, block_end;
        for (block_start = (fast_sint_t)n - 1; block_start >= 0; block_start = block_end)
        {
            block_end = block_start - (fast_sint_t)threads * LIBSAIS_PER_THREAD_CACHE_SIZE; if (block_end < 0) { block_end = -1; }

            libsais_final_sorting_scan_right_to_left_32s_block_omp(T, SA, induction_bucket, thread_state[0].state.cache, block_end + 1, block_start - block_end, threads);
        }
    }
#else
    UNUSED(thread_state);
#endif
}

static void libsais_clear_lms_suffixes_omp(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT bucket_start, sa_sint_t * RESTRICT bucket_end, sa_sint_t threads)
{
    fast_sint_t c;

#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel for schedule(static, 1) num_threads(threads) if(threads > 1 && n >= 65536)
#else
    UNUSED(threads); UNUSED(n);
#endif
    for (c = 0; c < k; ++c)
    {
        if (bucket_end[c] > bucket_start[c])
        {
            memset(&SA[bucket_start[c]], 0, ((size_t)bucket_end[c] - (size_t)bucket_start[c]) * sizeof(sa_sint_t));
        }
    }
}

static sa_sint_t libsais_induce_final_order_8u_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t bwt, sa_sint_t r, sa_sint_t * RESTRICT I, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    if (!bwt)
    {
        #ifdef AOCL_BWT
            sa_sint_t zero_index1, zero_index2;

            zero_index1 = AOCL_libsais_final_sorting_scan_left_to_right_8u(T, SA, &buckets[6 * ALPHABET_SIZE], 0, n);
            if (threads > 1 && n >= 65536) { libsais_clear_lms_suffixes_omp(SA, n, ALPHABET_SIZE, &buckets[6 * ALPHABET_SIZE], &buckets[7 * ALPHABET_SIZE], threads); }
            zero_index2 = AOCL_libsais_final_sorting_scan_right_to_left_8u(T, SA, &buckets[7 * ALPHABET_SIZE], 0, n);    
            
            if(zero_index1 == -1)
                return zero_index2;
            return zero_index1;
        #else
            libsais_final_sorting_scan_left_to_right_8u_omp(T, SA, n, k, &buckets[6 * ALPHABET_SIZE], threads, thread_state);
            if (threads > 1 && n >= 65536) { libsais_clear_lms_suffixes_omp(SA, n, ALPHABET_SIZE, &buckets[6 * ALPHABET_SIZE], &buckets[7 * ALPHABET_SIZE], threads); }
            libsais_final_sorting_scan_right_to_left_8u_omp(T, SA, n, k, &buckets[7 * ALPHABET_SIZE], threads, thread_state);
            return 0;
        #endif /* AOCL_BWT */
    }
    else if (I != NULL)
    {
        libsais_final_bwt_aux_scan_left_to_right_8u_omp(T, SA, n, k, r - 1, I, &buckets[6 * ALPHABET_SIZE], threads, thread_state);
        if (threads > 1 && n >= 65536) { libsais_clear_lms_suffixes_omp(SA, n, ALPHABET_SIZE, &buckets[6 * ALPHABET_SIZE], &buckets[7 * ALPHABET_SIZE], threads); }
        libsais_final_bwt_aux_scan_right_to_left_8u_omp(T, SA, n, k, r - 1, I, &buckets[7 * ALPHABET_SIZE], threads, thread_state);
        return 0;
    }
    else
    {
        libsais_final_bwt_scan_left_to_right_8u_omp(T, SA, n, k, &buckets[6 * ALPHABET_SIZE], threads, thread_state);
        if (threads > 1 && n >= 65536) { libsais_clear_lms_suffixes_omp(SA, n, ALPHABET_SIZE, &buckets[6 * ALPHABET_SIZE], &buckets[7 * ALPHABET_SIZE], threads); }
        return libsais_final_bwt_scan_right_to_left_8u_omp(T, SA, n, k, &buckets[7 * ALPHABET_SIZE], threads, thread_state);
    }
}

static void libsais_induce_final_order_32s_6k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    libsais_final_sorting_scan_left_to_right_32s_omp(T, SA, n, &buckets[4 * (fast_sint_t)k], threads, thread_state);
    libsais_final_sorting_scan_right_to_left_32s_omp(T, SA, n, &buckets[5 * (fast_sint_t)k], threads, thread_state);
}

static void libsais_induce_final_order_32s_4k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    libsais_final_sorting_scan_left_to_right_32s_omp(T, SA, n, &buckets[2 * (fast_sint_t)k], threads, thread_state);
    libsais_final_sorting_scan_right_to_left_32s_omp(T, SA, n, &buckets[3 * (fast_sint_t)k], threads, thread_state);
}

static void libsais_induce_final_order_32s_2k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    libsais_final_sorting_scan_left_to_right_32s_omp(T, SA, n, &buckets[1 * (fast_sint_t)k], threads, thread_state);
    libsais_final_sorting_scan_right_to_left_32s_omp(T, SA, n, &buckets[0 * (fast_sint_t)k], threads, thread_state);
}

static void libsais_induce_final_order_32s_1k(const sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    libsais_count_suffixes_32s(T, n, k, buckets);
    libsais_initialize_buckets_start_32s_1k(k, buckets);
    libsais_final_sorting_scan_left_to_right_32s_omp(T, SA, n, buckets, threads, thread_state);

    libsais_count_suffixes_32s(T, n, k, buckets);
    libsais_initialize_buckets_end_32s_1k(k, buckets);
    libsais_final_sorting_scan_right_to_left_32s_omp(T, SA, n, buckets, threads, thread_state);
}

static sa_sint_t libsais_renumber_unique_and_nonunique_lms_suffixes_32s(sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t m, sa_sint_t f, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT SAm = &SA[m];

    sa_sint_t i, j;
    for (i = (sa_sint_t)omp_block_start, j = (sa_sint_t)omp_block_start + (sa_sint_t)omp_block_size - 2 * (sa_sint_t)prefetch_distance - 3; i < j; i += 4)
    {
        libsais_prefetchr(&SA[i + 3 * prefetch_distance]);

        libsais_prefetchw(&SAm[((sa_uint_t)SA[i + 2 * prefetch_distance + 0]) >> 1]);
        libsais_prefetchw(&SAm[((sa_uint_t)SA[i + 2 * prefetch_distance + 1]) >> 1]);
        libsais_prefetchw(&SAm[((sa_uint_t)SA[i + 2 * prefetch_distance + 2]) >> 1]);
        libsais_prefetchw(&SAm[((sa_uint_t)SA[i + 2 * prefetch_distance + 3]) >> 1]);

        sa_uint_t q0 = (sa_uint_t)SA[i + prefetch_distance + 0]; const sa_sint_t * Tq0 = &T[q0]; libsais_prefetchw(SAm[q0 >> 1] < 0 ? Tq0 : NULL);
        sa_uint_t q1 = (sa_uint_t)SA[i + prefetch_distance + 1]; const sa_sint_t * Tq1 = &T[q1]; libsais_prefetchw(SAm[q1 >> 1] < 0 ? Tq1 : NULL);
        sa_uint_t q2 = (sa_uint_t)SA[i + prefetch_distance + 2]; const sa_sint_t * Tq2 = &T[q2]; libsais_prefetchw(SAm[q2 >> 1] < 0 ? Tq2 : NULL);
        sa_uint_t q3 = (sa_uint_t)SA[i + prefetch_distance + 3]; const sa_sint_t * Tq3 = &T[q3]; libsais_prefetchw(SAm[q3 >> 1] < 0 ? Tq3 : NULL);

        sa_uint_t p0 = (sa_uint_t)SA[i + 0]; sa_sint_t s0 = SAm[p0 >> 1]; if (s0 < 0) { T[p0] |= SAINT_MIN; f++; s0 = i + 0 + SAINT_MIN + f; } SAm[p0 >> 1] = s0 - f;
        sa_uint_t p1 = (sa_uint_t)SA[i + 1]; sa_sint_t s1 = SAm[p1 >> 1]; if (s1 < 0) { T[p1] |= SAINT_MIN; f++; s1 = i + 1 + SAINT_MIN + f; } SAm[p1 >> 1] = s1 - f;
        sa_uint_t p2 = (sa_uint_t)SA[i + 2]; sa_sint_t s2 = SAm[p2 >> 1]; if (s2 < 0) { T[p2] |= SAINT_MIN; f++; s2 = i + 2 + SAINT_MIN + f; } SAm[p2 >> 1] = s2 - f;
        sa_uint_t p3 = (sa_uint_t)SA[i + 3]; sa_sint_t s3 = SAm[p3 >> 1]; if (s3 < 0) { T[p3] |= SAINT_MIN; f++; s3 = i + 3 + SAINT_MIN + f; } SAm[p3 >> 1] = s3 - f;
    }

    for (j += 2 * (sa_sint_t)prefetch_distance + 3; i < j; i += 1)
    {
        sa_uint_t p = (sa_uint_t)SA[i]; sa_sint_t s = SAm[p >> 1]; if (s < 0) { T[p] |= SAINT_MIN; f++; s = i + SAINT_MIN + f; } SAm[p >> 1] = s - f;
    }

    return f;
}

static void libsais_compact_unique_and_nonunique_lms_suffixes_32s(sa_sint_t * RESTRICT SA, sa_sint_t m, fast_sint_t * pl, fast_sint_t * pr, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT SAl = &SA[0];
    sa_sint_t * RESTRICT SAr = &SA[0];

    fast_sint_t i, j, l = *pl - 1, r = *pr - 1;
    for (i = (fast_sint_t)m + omp_block_start + omp_block_size - 1, j = (fast_sint_t)m + omp_block_start + 3; i >= j; i -= 4)
    {
        libsais_prefetchr(&SA[i - prefetch_distance]);

        sa_sint_t p0 = SA[i - 0]; SAl[l] = p0 & SAINT_MAX; l -= p0 < 0; SAr[r] = p0 - 1; r -= p0 > 0;
        sa_sint_t p1 = SA[i - 1]; SAl[l] = p1 & SAINT_MAX; l -= p1 < 0; SAr[r] = p1 - 1; r -= p1 > 0;
        sa_sint_t p2 = SA[i - 2]; SAl[l] = p2 & SAINT_MAX; l -= p2 < 0; SAr[r] = p2 - 1; r -= p2 > 0;
        sa_sint_t p3 = SA[i - 3]; SAl[l] = p3 & SAINT_MAX; l -= p3 < 0; SAr[r] = p3 - 1; r -= p3 > 0;
    }

    for (j -= 3; i >= j; i -= 1)
    {
        sa_sint_t p = SA[i]; SAl[l] = p & SAINT_MAX; l -= p < 0; SAr[r] = p - 1; r -= p > 0;
    }
    
    *pl = l + 1; *pr = r + 1;
}


#if defined(LIBSAIS_OPENMP)

static sa_sint_t libsais_count_unique_suffixes(sa_sint_t * RESTRICT SA, sa_sint_t m, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    sa_sint_t * RESTRICT SAm = &SA[m];

    fast_sint_t i, j; sa_sint_t f0 = 0, f1 = 0, f2 = 0, f3 = 0;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 3; i < j; i += 4)
    {
        libsais_prefetchr(&SA[i + 2 * prefetch_distance]);

        libsais_prefetchr(&SAm[((sa_uint_t)SA[i + prefetch_distance + 0]) >> 1]);
        libsais_prefetchr(&SAm[((sa_uint_t)SA[i + prefetch_distance + 1]) >> 1]);
        libsais_prefetchr(&SAm[((sa_uint_t)SA[i + prefetch_distance + 2]) >> 1]);
        libsais_prefetchr(&SAm[((sa_uint_t)SA[i + prefetch_distance + 3]) >> 1]);

        f0 += SAm[((sa_uint_t)SA[i + 0]) >> 1] < 0;
        f1 += SAm[((sa_uint_t)SA[i + 1]) >> 1] < 0;
        f2 += SAm[((sa_uint_t)SA[i + 2]) >> 1] < 0;
        f3 += SAm[((sa_uint_t)SA[i + 3]) >> 1] < 0;
    }

    for (j += prefetch_distance + 3; i < j; i += 1)
    {
        f0 += SAm[((sa_uint_t)SA[i]) >> 1] < 0;
    }

    return f0 + f1 + f2 + f3;
}

#endif

static sa_sint_t libsais_renumber_unique_and_nonunique_lms_suffixes_32s_omp(sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t m, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    sa_sint_t f = 0;

#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && m >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (m / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : m - omp_block_start;

        if (omp_num_threads == 1)
        {
            f = libsais_renumber_unique_and_nonunique_lms_suffixes_32s(T, SA, m, 0, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                thread_state[omp_thread_num].state.count = libsais_count_unique_suffixes(SA, m, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            {
                fast_sint_t t, count = 0; for (t = 0; t < omp_thread_num; ++t) { count += thread_state[t].state.count; }

                if (omp_thread_num == omp_num_threads - 1)
                {
                    f = (sa_sint_t)(count + thread_state[omp_thread_num].state.count);
                }

                libsais_renumber_unique_and_nonunique_lms_suffixes_32s(T, SA, m, (sa_sint_t)count, omp_block_start, omp_block_size);
            }
        }
#endif
    }

    return f;
}

static void libsais_compact_unique_and_nonunique_lms_suffixes_32s_omp(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t fs, sa_sint_t f, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 131072 && m < fs)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
#ifdef AOCL_BWT
        fast_sint_t omp_block_stride  = (((fast_sint_t)(n+1) >> 1) / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : ((fast_sint_t)(n+1) >> 1) - omp_block_start;
#else
        fast_sint_t omp_block_stride  = (((fast_sint_t)n >> 1) / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : ((fast_sint_t)n >> 1) - omp_block_start;
#endif /* AOCL_BWT */

        if (omp_num_threads == 1)
        {
            fast_sint_t l = m, r = (fast_sint_t)n + (fast_sint_t)fs;
            libsais_compact_unique_and_nonunique_lms_suffixes_32s(SA, m, &l, &r, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                thread_state[omp_thread_num].state.position   = (fast_sint_t)m + ((fast_sint_t)n >> 1) + omp_block_start + omp_block_size;
                thread_state[omp_thread_num].state.count      = (fast_sint_t)m + omp_block_start + omp_block_size;

                libsais_compact_unique_and_nonunique_lms_suffixes_32s(SA, m, &thread_state[omp_thread_num].state.position, &thread_state[omp_thread_num].state.count, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                fast_sint_t t, position;

                for (position = m, t = omp_num_threads - 1; t >= 0; --t)
                { 
                    fast_sint_t omp_block_end     = t < omp_num_threads - 1 ? omp_block_stride * (t + 1) : ((fast_sint_t)n >> 1);
                    fast_sint_t count             = ((fast_sint_t)m + ((fast_sint_t)n >> 1) + omp_block_end - thread_state[t].state.position);

                    if (count > 0)
                    {
                        position -= count; memcpy(&SA[position], &SA[thread_state[t].state.position], (size_t)count * sizeof(sa_sint_t));
                    }
                }

                for (position = (fast_sint_t)n + (fast_sint_t)fs, t = omp_num_threads - 1; t >= 0; --t)
                {
                    fast_sint_t omp_block_end     = t < omp_num_threads - 1 ? omp_block_stride * (t + 1) : ((fast_sint_t)n >> 1);
                    fast_sint_t count             = ((fast_sint_t)m + omp_block_end - thread_state[t].state.count);

                    if (count > 0)
                    {
                        position -= count; memcpy(&SA[position], &SA[thread_state[t].state.count], (size_t)count * sizeof(sa_sint_t));
                    }
                }
            }
        }
#endif
    }

    memcpy(&SA[(fast_sint_t)n + (fast_sint_t)fs - (fast_sint_t)m], &SA[(fast_sint_t)m - (fast_sint_t)f], (size_t)f * sizeof(sa_sint_t));
}

static sa_sint_t libsais_compact_lms_suffixes_32s_omp(sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t fs, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    sa_sint_t f = libsais_renumber_unique_and_nonunique_lms_suffixes_32s_omp(T, SA, m, threads, thread_state);
    libsais_compact_unique_and_nonunique_lms_suffixes_32s_omp(SA, n, m, fs, f, threads, thread_state);

    return f;
}

#ifndef AOCL_BWT
static void libsais_merge_unique_lms_suffixes_32s(sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, fast_sint_t l, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    const sa_sint_t * RESTRICT SAnm = &SA[(fast_sint_t)n - (fast_sint_t)m - 1 + l];

    sa_sint_t i, j; fast_sint_t tmp = *SAnm++;
    for (i = (sa_sint_t)omp_block_start, j = (sa_sint_t)omp_block_start + (sa_sint_t)omp_block_size - 6; i < j; i += 4)
    {
        libsais_prefetchr(&T[i + prefetch_distance]);

        sa_sint_t c0 = T[i + 0]; if (c0 < 0) { T[i + 0] = c0 & SAINT_MAX; SA[tmp] = i + 0; i++; tmp = *SAnm++; }
        sa_sint_t c1 = T[i + 1]; if (c1 < 0) { T[i + 1] = c1 & SAINT_MAX; SA[tmp] = i + 1; i++; tmp = *SAnm++; }
        sa_sint_t c2 = T[i + 2]; if (c2 < 0) { T[i + 2] = c2 & SAINT_MAX; SA[tmp] = i + 2; i++; tmp = *SAnm++; }
        sa_sint_t c3 = T[i + 3]; if (c3 < 0) { T[i + 3] = c3 & SAINT_MAX; SA[tmp] = i + 3; i++; tmp = *SAnm++; }
    }

    for (j += 6; i < j; i += 1)
    {
        sa_sint_t c = T[i]; if (c < 0) { T[i] = c & SAINT_MAX; SA[tmp] = i; i++; tmp = *SAnm++; }
    }
}
#else
static void AOCL_libsais_merge_unique_lms_suffixes_32s(sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, fast_sint_t l, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    const sa_sint_t * RESTRICT SAnm = &SA[n-m];

    sa_sint_t i, j; fast_sint_t tmp = *SAnm++;
    for (i = (sa_sint_t)omp_block_start, j = (sa_sint_t)omp_block_start + (sa_sint_t)omp_block_size - 6; i < j; i += 4)
    {
        libsais_prefetchr(&T[i + prefetch_distance]);

        sa_sint_t c0 = T[i + 0]; if (c0 < 0) { T[i + 0] = c0 & SAINT_MAX; SA[tmp] = i + 0; i++; tmp = *SAnm++; }
        sa_sint_t c1 = T[i + 1]; if (c1 < 0) { T[i + 1] = c1 & SAINT_MAX; SA[tmp] = i + 1; i++; tmp = *SAnm++; }
        sa_sint_t c2 = T[i + 2]; if (c2 < 0) { T[i + 2] = c2 & SAINT_MAX; SA[tmp] = i + 2; i++; tmp = *SAnm++; }
        sa_sint_t c3 = T[i + 3]; if (c3 < 0) { T[i + 3] = c3 & SAINT_MAX; SA[tmp] = i + 3; i++; tmp = *SAnm++; }
    }

    for (j += 6; i < j; i += 1)
    {
        sa_sint_t c = T[i]; if (c < 0) { T[i] = c & SAINT_MAX; SA[tmp] = i; i++; tmp = *SAnm++; }
    }
}
#endif /* AOCL_BWT */
static void libsais_merge_nonunique_lms_suffixes_32s(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, fast_sint_t l, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

#ifdef AOCL_BWT
    const sa_sint_t * RESTRICT SAnm = &SA[(fast_sint_t)n - (fast_sint_t)m + l];
#else
    const sa_sint_t * RESTRICT SAnm = &SA[(fast_sint_t)n - (fast_sint_t)m - 1 + l];
#endif /* AOCL_BWT */

    fast_sint_t i, j; sa_sint_t tmp = *SAnm++;
#ifdef AOCL_BWT
    for (i = omp_block_start, j = omp_block_start + omp_block_size - 3; i < j; i += 4)
    {
        libsais_prefetchr(&SA[i + prefetch_distance]);

        if (SA[i + 0] < 0) { SA[i + 0] = tmp; tmp = *SAnm++; }
        if (SA[i + 1] < 0) { SA[i + 1] = tmp; tmp = *SAnm++; }
        if (SA[i + 2] < 0) { SA[i + 2] = tmp; tmp = *SAnm++; }
        if (SA[i + 3] < 0) { SA[i + 3] = tmp; tmp = *SAnm++; }
    }

    for (j += 3; i < j; i += 1)
    {
        if (SA[i] < 0) { SA[i] = tmp; tmp = *SAnm++; }
    }
#else
    for (i = omp_block_start, j = omp_block_start + omp_block_size - 3; i < j; i += 4)
    {
        libsais_prefetchr(&SA[i + prefetch_distance]);

        if (SA[i + 0] == 0) { SA[i + 0] = tmp; tmp = *SAnm++; }
        if (SA[i + 1] == 0) { SA[i + 1] = tmp; tmp = *SAnm++; }
        if (SA[i + 2] == 0) { SA[i + 2] = tmp; tmp = *SAnm++; }
        if (SA[i + 3] == 0) { SA[i + 3] = tmp; tmp = *SAnm++; }
    }

    for (j += 3; i < j; i += 1)
    {
        if (SA[i] == 0) { SA[i] = tmp; tmp = *SAnm++; }
    }
#endif /* AOCL_BWT */
}

#ifndef AOCL_BWT
static void libsais_merge_unique_lms_suffixes_32s_omp(sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (n / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : n - omp_block_start;

        if (omp_num_threads == 1)
        {
            libsais_merge_unique_lms_suffixes_32s(T, SA, n, m, 0, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                thread_state[omp_thread_num].state.count = libsais_count_negative_marked_suffixes(T, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            {
                fast_sint_t t, count = 0; for (t = 0; t < omp_thread_num; ++t) { count += thread_state[t].state.count; }

                libsais_merge_unique_lms_suffixes_32s(T, SA, n, m, count, omp_block_start, omp_block_size);
            }
        }
#endif
    }
}
#endif /* AOCL_BWT */

static void libsais_merge_nonunique_lms_suffixes_32s_omp(sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t f, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && m >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads); UNUSED(thread_state);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (m / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : m - omp_block_start;

        if (omp_num_threads == 1)
        {
            libsais_merge_nonunique_lms_suffixes_32s(SA, n, m, f, omp_block_start, omp_block_size);
        }
#if defined(LIBSAIS_OPENMP)
        else
        {
            {
                thread_state[omp_thread_num].state.count = libsais_count_zero_marked_suffixes(SA, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            {
                fast_sint_t t, count = f; for (t = 0; t < omp_thread_num; ++t) { count += thread_state[t].state.count; }

                libsais_merge_nonunique_lms_suffixes_32s(SA, n, m, count, omp_block_start, omp_block_size);
            }
        }
#endif
    }
}

static void libsais_merge_compacted_lms_suffixes_32s_omp(sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t f, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
#ifdef AOCL_BWT
    AOCL_libsais_merge_unique_lms_suffixes_32s(T, SA, n, m, f, 0, n);
#else
    libsais_merge_unique_lms_suffixes_32s_omp(T, SA, n, m, threads, thread_state);
#endif /* AOCL_BWT */
    libsais_merge_nonunique_lms_suffixes_32s_omp(SA, n, m, f, threads, thread_state);
}

static void libsais_reconstruct_compacted_lms_suffixes_32s_2k_omp(sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t m, sa_sint_t fs, sa_sint_t f, sa_sint_t * RESTRICT buckets, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    if (f > 0)
    {
#ifdef AOCL_BWT
        // All distinct elements are copied
        memmove(&SA[n - m], &SA[n + fs - m], (size_t)f * sizeof(sa_sint_t));
        sa_sint_t temp = SA[n-m+f-1];
#else
        memmove(&SA[n - m - 1], &SA[n + fs - m], (size_t)f * sizeof(sa_sint_t));
#endif /* AOCL_BWT */
        // Gathers all LMSs from SA[n-1] to SA[n+f-m] (excludes MSB set elements in T)
        libsais_count_and_gather_compacted_lms_suffixes_32s_2k_omp(T, SA, n, k, buckets, threads, thread_state);
#ifdef AOCL_BWT
        SA[n-m+f-1] = temp;
#endif /* AOCL_BWT */
        // From SA[0] to SA[m-f], all ranks of renamed indexes are replaced with actual indexes that occur in `T`.
        libsais_reconstruct_lms_suffixes_omp(SA, n, m - f, threads);

#ifdef AOCL_BWT
        // The result from above function is stored starting from &SA[n-m+f], these are just m-f elements, f elements would be at &SA[n-m].
        memcpy(&SA[n - m + f], &SA[0], ((size_t)m - (size_t)f) * sizeof(sa_sint_t));
        // &SA[0] buffer is initialized to -1 instead of 0 for marking empty spaces, if `0` is used to mark empty space, then `0` LMS index would be indistinguishable from empty space.
        memset(&SA[0], -1, (size_t)m * sizeof(sa_sint_t));
#else
        memcpy(&SA[n - m - 1 + f], &SA[0], ((size_t)m - (size_t)f) * sizeof(sa_sint_t));
        memset(&SA[0], 0, (size_t)m * sizeof(sa_sint_t));
#endif /* AOCL_BWT */
        /*

            BWT Modifications:  Max LMS indexes that could be generated are considered to be n/2 instead of (n-1)/2
                                Empty spaces are marked by -1, so checks are modified to consider -1 as empty space instead of `0`
        */
        libsais_merge_compacted_lms_suffixes_32s_omp(T, SA, n, m, f, threads, thread_state);
    }
    else
    {
        // Similar to `libsais_count_and_gather_lms_suffixes_8u_omp` from `libsais_main_8u`
        // BWT Modifications: Overriding rank of last element when m = n/2 is prevented.
        libsais_count_and_gather_lms_suffixes_32s_2k(T, SA, n, k, buckets, 0, n);

        // This same function is called in the above part of if condition, the difference is that this function considers all m elements to reconstruct LMS indexes in order,
        // but in the other case, only non-distinct LMS indexes are considered for reconstruction.
        libsais_reconstruct_lms_suffixes_omp(SA, n, m, threads);
    }
}

static void libsais_reconstruct_compacted_lms_suffixes_32s_1k_omp(sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t m, sa_sint_t fs, sa_sint_t f, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    if (f > 0)
    {
        memmove(&SA[n - m - 1], &SA[n + fs - m], (size_t)f * sizeof(sa_sint_t));

        libsais_gather_compacted_lms_suffixes_32s(T, SA, n);
        libsais_reconstruct_lms_suffixes_omp(SA, n, m - f, threads);

        memcpy(&SA[n - m - 1 + f], &SA[0], ((size_t)m - (size_t)f) * sizeof(sa_sint_t));
        memset(&SA[0], 0, (size_t)m * sizeof(sa_sint_t));

        libsais_merge_compacted_lms_suffixes_32s_omp(T, SA, n, m, f, threads, thread_state);
    }
    else
    {
        libsais_gather_lms_suffixes_32s(T, SA, n);
        libsais_reconstruct_lms_suffixes_omp(SA, n, m, threads);
    }
}

static sa_sint_t libsais_main_32s_recursion(sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t fs, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state, sa_sint_t * RESTRICT local_buffer)
{
    fs = fs < (SAINT_MAX - n) ? fs : (SAINT_MAX - n);
    /*
        n:  represents the total number of LMS characters generated in previous recursion step, for current step this is the total input size considered for BWT sorting.
        T:  points to LMS names generated after renaming LMS indexes of parent function.
        SA: points to original SA, its actual length is equal to length of original input array, but for current recursive step it is only considered of length n.
        k:  number of unique LMS names being used, i.e [0, k-1], the max value of any integer in T would not cross k-1.

        Relationship between data in libsais_main_8u and current function call:
        ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
        ║   Lets use the following naming convention:                                                                       ║
        ║    _____________________________________________________________________________________________________________  ║
        ║   |variable description| libsais_main_8u | libsais_main_32s_recursion current | libsais_main_32s_recursion next | ║
        ║   |--------------------+-----------------+------------------------------------+---------------------------------+ ║
        ║   | Input array        | T_prev          | T                                  | T_next                          | ║
        ║   | Suffix array       | SA_prev         | SA                                 | SA_next                         | ║
        ║   | input length       | n_prev          | n                                  | n_next                          | ║
        ║   | free space         | fs_prev         | fs                                 | fs_next                         | ║
        ║   | buckets            | buckets_prev    | buckets                            | buckets_next                    | ║
        ║   | number of LMS      | m_prev          | m                                  | m_next                          | ║
        ║   | number of uniques  | k_prev          | k                                  | k_next                          | ║
        ║   --------------------------------------------------------------------------------------------------------------- ║
        ║                                                                                                                   ║
        ║   SA_prev[]:                                                                                                      ║
        ║   |<-------------------------------------------------- n_prev --------------------------------------------->|     ║
        ║   |<-------- m_prev -------->|<---------------------- fs ----------------------->|<-------- m_prev -------->|     ║
        ║                                                                                                                   ║
        ║   |<--------- SA ----------->|                /<------------ buckets ---------->/|<---------- T ----------->|     ║
        ║   |<---------- n ----------->|                /<-------------- 6*k ------------>/|<---------- n ----------->|     ║
        ║                                               | If SA_prev has enough space left |                                ║
        ║                                               | buckets use data from it; else   |                                ║
        ║                                               | data is dynamically allocated    |                                ║
        ║                                                                                                                   ║
        ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

        buckets[]: can be primarily divided into 6 parts, each part has `k` integers:
        |----------------|----------------|----------------|----------------|----------------|----------------|
        |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|

        Path calculation according to data size
        ```````````````````````````````````````
        fs = n-2*m: represents the remaining data that is available for use (here n is previous recursion step's input size, m is previous number of LMS indexes, which is equal to current n)
        From previous SA array, first m elements are taken by current SA, last m elements are taken by current T.

        "fs/k >= _number":
        since k represents max value that any integer from T can take, so _number represents number of different buckets that can be utilized

        There are 4 different cases:
        fs / k >= 6 (6 buckets)
        fs / k >= 4 (4 buckets)
        fs / k >= 2 (2 buckets)
        fs / k <  2
    */
#ifdef AOCL_BWT
    if (k > 0)
#else
    if (k > 0 && ((fs / k >= 6) || (LIBSAIS_LOCAL_BUFFER_SIZE / k >= 6)))
#endif /* AOCL_BWT */
    {
        sa_sint_t alignment = (fs - 1024) / k >= 6 ? (sa_sint_t)1024 : (sa_sint_t)16;
        sa_sint_t * buckets = (fs - alignment) / k >= 6 ? (sa_sint_t *)libsais_align_up(&SA[n + fs - 6 * (fast_sint_t)k - alignment], (size_t)alignment * sizeof(sa_sint_t)) : &SA[n + fs - 6 * (fast_sint_t)k];
        buckets = (LIBSAIS_LOCAL_BUFFER_SIZE / k >= 6) ? local_buffer : buckets;
#ifdef AOCL_BWT
        sa_sint_t * dynamic_memory = NULL;
        sa_sint_t dynamic_memory_alloc = 0;
        /*
            If SA array doesn't contain enough space for storing 6 buckets each of size k,
            or
            If buckets[] array overlap with current T[-2]
            in those cases memory is allocated dynamically.
        */
        if ((!(k > 0 && ((fs / k >= 6) || (LIBSAIS_LOCAL_BUFFER_SIZE / k >= 6)))) || 
            ((SA < buckets) && (buckets < T) && (&buckets[6 * k - 1] >= &T[-2])))
        {
            dynamic_memory_alloc = 1;
        }

        if(dynamic_memory_alloc)
        {
            const sa_sint_t align = 1024;
            // Allocating buffer to accomodate 6 bucket containers of size `k+1`. Extra `align` bytes will be useful for alignment in `libsais_align_up`.
            dynamic_memory = (sa_sint_t *)calloc(6*(k+1) + align, sizeof(sa_sint_t));
            buckets = (sa_sint_t *)libsais_align_up(dynamic_memory, align * sizeof(sa_sint_t));
        }
#endif /* AOCL_BWT */
        /*
            Refer to `libsais_count_and_gather_lms_suffixes_8u_omp` from `libsais_main_8u`.

            SA[]:
            |<----------------------------- n ---------------------------------->|

                               |<--------------- initialization ---------------->|
                               |<--last m elements are updated to LMS indexes -->|


            buckets[]:
            |----------------|----------------|----------------|----------------|----------------|----------------|
            |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|

            |<-------------------------- initialization ----------------------->|
        */
        sa_sint_t m = libsais_count_and_gather_lms_suffixes_32s_4k_omp(T, SA, n, k, buckets, threads, thread_state);
        if (m > 1)
        {
            // all indexes in SA except LMS indexes are made 0
            memset(SA, 0, ((size_t)n - (size_t)m) * sizeof(sa_sint_t));

            sa_sint_t first_lms_suffix    = SA[n - m];
            /*
                Refer to `libsais_initialize_buckets_for_lms_suffixes_radix_sort_8u` comments from `libsais_main_8u`,
                current function is very similar to the aforementioned function but the difference is only single bucket is changed
                `&buckets[4*k]` and previous type is L type is updated for finding tail limits of all unique input integers.

                buckets[]:
                |----------------|----------------|----------------|----------------|----------------|----------------|
                |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|

                                                                                    |<-------------->|
                                                                                      initialization
            */
            sa_sint_t left_suffixes_count = libsais_initialize_buckets_for_lms_suffixes_radix_sort_32s_6k(T, k, buckets, first_lms_suffix);
            /*
                Refer to `libsais_radix_sort_lms_suffixes_8u_omp` comments from `libsais_main_8u`.

                SA[]:
                |<-------------------------------- n -------------------------------->|

                |<----------- initialized ----------->|<--- used but not modified --->|
                                                      |<------ LMS indexes (m) ------>|

                buckets[]:
                |----------------|----------------|----------------|----------------|----------------|----------------|
                |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|

                                                                                    |<---- used ---->|
            */
            libsais_radix_sort_lms_suffixes_32s_6k_omp(T, SA, n, m, &buckets[4 * (fast_sint_t)k], threads, thread_state);
            /*
                MSB bit is set for all the buckets's starting index.
                
                SA[]:
                |<--------------------------- n ints -------------------------------->|

                |<----------- modified ----------->|
                |<------------ (n-m) ------------->|

                buckets[]:
                |----------------|----------------|----------------|----------------|----------------|----------------|
                |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|

                                                                                    |< read through >|

                BWT Modifications: None
            */
            if ((n / 8192) < k) { libsais_radix_sort_set_markers_32s_6k_omp(SA, k, &buckets[4 * (fast_sint_t)k], threads); }
            if (threads > 1 && n >= 65536) { memset(&SA[(fast_sint_t)n - (fast_sint_t)m], 0, (size_t)m * sizeof(sa_sint_t)); }
            /*
                &buckets[4*k]:  Even contains previous is S type, starting index of S type is left_suffixes_count
                                odd contains LMS character, here even is 2*i & odd is 2*i+1.
                                All buckets points to tails.
                &buckets[0]:    even contains previous is S type same as above, odd contains prev is L type
                                for &buckets[0] -> even is 4*i & odd is 4*i+1
                                Here all buckets points to heads.

                buckets[]:
                |----------------|----------------|----------------|----------------|----------------|----------------|
                |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|

                |<---------------------------------------------- modified ------------------------------------------->|
                |<------ heads of "prev is S type" & heads of "prev is L type" ---->|<- alternating tail pointers  -->|
                                                                                           of "prev is S type" and 
                                                                                               "LMS occurances"
                |<--------------------------- 1st part ---------------------------->|<----------- 2nd part ---------->|
                
                Here 1st part is concatenation of following sequence repeated "k" times:
                |<-- i -->|<-- ii -->|<-- iii -->|<-- iv -->|
                i: heads of prev is S type.
               ii: heads of prev is L type.
              iii: Initialized to zero.
               iv: Initialized to zero.

                BWT Modifications:  In current implementation S type suffixes start from index left_suffixes_count, unlike earlier where it used to start from left_suffixes_count+1.
                                    Earlier left_suffixes_count ignored first LMS index in `libsais_initialize_buckets_for_lms_suffixes_radix_sort_32s_6k` now that is not the case.
            */
            libsais_initialize_buckets_for_partial_sorting_32s_6k(T, k, buckets, first_lms_suffix, left_suffixes_count);
            /*
                LMS indexes are partilly sorted, similar to `libsais_induce_partial_order_8u_omp` refer to those comments from `libsais_main_8u`.

                SA[]:
                |<------------------------------------ n ------------------------------------->|

                |<------------------------------ used & altered ------------------------------>|
                |<----------- initialization ----------->|
                |<-- partially sorted LMS indexes (m) -->|
                

                buckets[]:
                |----------------|----------------|----------------|----------------|----------------|----------------|
                |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|

                |<----------------------------- modified -------------------------->|<------ used not modified ------>|
                |<------ heads of "prev is S type" & heads of "prev is L type" ---->|
                |<--------------------------- 1st part ---------------------------->|<----------- 2nd part ---------->|
                
                Here 1st part is concatenation of following sequence repeated "k" times:
                |<-- i -->|<-- ii -->|<-- iii -->|<-- iv -->|
               i: 
                    a. heads of prev is S type
                    b. tail pointers of "prev is S type"
              ii:  
                    a. heads of prev is L type
                    b. tail pointers of "LMS occurances"
                 a -> left to right scan, b -> right to left scan
             iii: used as a temporary buffer to store ranks from which "previous rank" is fetched
              iv: used as a temporary buffer to store ranks from which "previous rank" is fetched
            */
            libsais_induce_partial_order_32s_6k_omp(T, SA, n, k, buckets, first_lms_suffix, left_suffixes_count, threads, thread_state);
            /*
                if (n/8192) < k:
                    Some LMS substrings are unique, meaning they have distinct ranks. These substrings are inherently sorted, and their positions relative to others are fixed.
                    However, unique LMS substrings might still be needed for sorting non-unique LMS substrings during recursion, except when a unique LMS substring is immediately followed by another unique one.
                    In such cases, the rank of the following unique substring is redundant in the compacted string and can be omitted.
                    To reduce the size of the compacted string, libsais identifies unique LMS substrings during the ranking and scattering phase by scanning LMS positions in the suffix array.
                    It detects consecutive unique ranks using the MSB and marks redundant ranks for removal. The library then re-scans LMS positions in suffix order to recompute ranks for the remaining LMS substrings.
                    The MSB of the first symbol of an LMS substring marks it as removed from the compacted string.
                    Finally, libsais builds the compacted string using the recalculated ranks of the remaining LMS substrings and saves the final positions of the removed ones before proceeding with recursion.
                    When scattering LMS substring ranks to form the compacted string, the MSB of the rank indicates uniqueness. 
                if not:
                    Refer to `libsais_renumber_and_gather_lms_suffixes_omp` comments from `libsais_main_8u`.

                SA[]:
                |<---------------------------------- n --------------------------------------->|

                |<------ (m) LMS indexes ------->|<----------- (n-m) used & altered ---------->|
                |<--- used but not modified ---->|
                                                              |<--renumbered LMS indexes (m)-->|

                BWT Modifications:   Earlier last element couldn't be LMS type, now we are considering last element could be LMS and doing things accordingly.
            */
            sa_sint_t names = (n / 8192) < k
                ? libsais_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(SA, n, m, threads, thread_state)
                : libsais_renumber_and_gather_lms_suffixes_omp(SA, n, m, fs, threads, thread_state);

            if (names < m)
            {
                /*
                    if (n/8192)<k:
                        So currently `fs` elements after SA+n are free, all these distinct & renamed elements are placed at the last part of `fs` elements.
                        First f elements are distinct elements and next m-f elements are renamed elements.
                    
                    SA_prev[]:
                    |<------------------------------------------------------ n_prev ------------------------------------------------->|
                    |<------------ m_prev -------------->|<----------------- fs ---------------->|<------------ m_prev -------------->|

                    |<------------ updated ------------->|            |<------ initialized ----->|<------------ updated ------------->|
                    |<------------- SA ----------------->|            |<----------- m ---------->|<--------------- T ---------------->|
                    |<-------------- n ----------------->|            |<-- f -->|<---- (m-f) --->|<--------------- n ---------------->|
                                                        unique LMS indexes ↲             ↳ non-unique LMS indexes
                    
                    BWT Modifications:  None
                */
                sa_sint_t f = (n / 8192) < k
                    ? libsais_compact_lms_suffixes_32s_omp(T, SA, n, m, fs, threads, thread_state)
                    : 0;
                /*
                    This is Recursive step.
                    Here we ignore f distinct elements and only try to apply BWT sort to m-f elements.
                    `SA + n + fs - m + f` represents pointer to starting of next T, which is just (m-f) elements behind current `T` pointer.

                    fwd:  The current data is forwarded to next function call & might change.
                    nfwd: The current data is not forwarded to next function call & hence unchanged.

                    SA_prev[]:
                    |<------------------------------------------------------ n_prev ------------------------------------------------->|
                    |<------------ m_prev -------------->|<----------------- fs ---------------->|<------------ m_prev -------------->|

                    |<------------- SA ----------------->|           |<----------- m ----------->|
                    |<-------------- n ----------------->|           |<-- f -->|<---- (m-f) ---->| ─────────────────────┐
                                                                          └─── unique LMS indexes              non-unique LMS indexes

                    |<-------------------- fwd --------------------->|<--nfwd->|<fwd & unchanged>|<------------ nfwd ---------------->|
                    |<---- (m-f) ---->|<------ fs + n - 2*m +f ----->|         |<---- (m-f) ---->|                         
                    |<--- SA_next --->|                                        |<---- T_next --->|
                    |<--- n_next ---->|<--------- fs_next ---------->|         |<---- n_next --->|

                */
                sa_sint_t * T_next = SA + n + fs - m + f;
                sa_sint_t n_next = m - f;
                sa_sint_t t1 = T_next[-1];
                sa_sint_t t2 = T_next[-2];
                T_next[-1] = T_next[n_next-1];
                T_next[-2] = T_next[n_next-2];
                if (libsais_main_32s_recursion(T_next, SA, n_next, names - f, fs + n - 2 * m + f - AOCL_LIBSAIS_MOD_ELEMENTS, threads, thread_state, local_buffer) != 0)
                {
                    return -2;
                }
                T_next[-1] = t1;
                T_next[-2] = t2;
                /*
                    First f elements are copied from where distinct & renamed elements are stored after `libsais_compact_lms_suffixes_32s_omp`,
                    & then renamed indexes (m-f elements) are copied from the same place to the last part of SA array, so totally `f + 'm-f'` elements which is m elements are copied
                    to last part of SA array. From the above function, all m-f renamed elements are BWT sorted whose results are at the beginning of SA array.
                    Since distinct elements position is already known from first f elements and MSB markings from `T` buffer & currently we have the BWT sorted rankings of m-f elements,
                    with help of these two we construct the sorted LMS indexes of current `T` buffer.

                    SA_prev[]:
                    |<------------------------------------------------------ n_prev ------------------------------------------------->|
                    |<------------ m_prev -------------->|<----------------- fs ---------------->|<------------ m_prev -------------->|

                    |<--------- used & altered --------->|            |<----- read through ----->|<------------ updated ------------->|
                    |<------------- SA ----------------->|            |<----------- m ---------->|<--------------- T ---------------->|
                    |<-------------- n ----------------->|            |<-- f -->|<---- (m-f) --->|<--------------- n ---------------->|
                                                        unique LMS indexes ↲             ↳ non-unique LMS indexes
                    |<------- m ------->| → result
                    |   fully sorted    |
                    |    LMS indexes    |

                    buckets[]:
                    |----------------|----------------|----------------|----------------|----------------|----------------|
                    |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|

                    |<-------- initialization ------->|
                    |   alternating "LMS occurances"  |
                    |  and "leftover type occurances" |

                    BWT Modifications:  Earlier utmost LMS index that could be generated was assumed to be (n-1)/2, now limits are increased to consider n/2.
                */
                libsais_reconstruct_compacted_lms_suffixes_32s_2k_omp(T, SA, n, k, m, fs, f, buckets, threads, thread_state);
            }
            else
            {
                /*
                    This function is used to fill &buckets[0] buffer, odd indexes contains sum of LMS indexes & even contains leftover indexes of each character

                    buckets[]:
                    |----------------|----------------|----------------|----------------|----------------|----------------|
                    |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|

                    |<-------- initialization ------->|
                    |   alternating "LMS occurances"  |
                    |  and "leftover type occurances" |
                    
                    BWT Modifications:  Last character is compared with first character instead of senital character,
                                        First character is compared with last character to get its previous type.  
                */
                libsais_count_lms_suffixes_32s_2k(T, n, k, buckets);
            }
            // Following three functions are similar to 
            // `libsais_initialize_buckets_for_lms_suffixes_radix_sort_8u`
            // `libsais_place_lms_suffixes_interval_8u`
            // `libsais_induce_final_order_8u_omp`
            // from "libsais_main_8u" function.
            libsais_initialize_buckets_start_and_end_32s_4k(k, buckets);
            libsais_place_lms_suffixes_histogram_32s_4k(SA, n, k, m, buckets);
            libsais_induce_final_order_32s_4k(T, SA, n, k, buckets, threads, thread_state);
        }
        else
        if(m == 1)
        {
            SA[0] = SA[n - 1];
            // Following three functions are similar to 
            // `libsais_initialize_buckets_for_lms_suffixes_radix_sort_8u`
            // `libsais_place_lms_suffixes_interval_8u`
            // `libsais_induce_final_order_8u_omp`
            // from "libsais_main_8u" function.
            libsais_initialize_buckets_start_and_end_32s_6k(k, buckets);
            libsais_place_lms_suffixes_histogram_32s_6k(SA, n, k, m, buckets);
            libsais_induce_final_order_32s_6k(T, SA, n, k, buckets, threads, thread_state);
        }
        else
        {
            /*
                If `m` is zero, which means there is no LMS indexes, this case occurs only when all elements are same,
                in such cases the order of indexes doesn't matter, so initiating each index with certain order according to the order of occurrence.
            */
            for (sa_sint_t i = 0; i < n; i++)
            {
                SA[i] = i;
            }
        }
#ifdef AOCL_BWT
        if(dynamic_memory_alloc) free(dynamic_memory);
#endif /* AOCL_BWT */
        return 0;
    }
    else if (k > 0 && ((fs / k >= 4) || (LIBSAIS_LOCAL_BUFFER_SIZE / k >= 4)))
    {
        sa_sint_t alignment = (fs - 1024) / k >= 4 ? (sa_sint_t)1024 : (sa_sint_t)16;
        sa_sint_t * RESTRICT buckets = (fs - alignment) / k >= 4 ? (sa_sint_t *)libsais_align_up(&SA[n + fs - 4 * (fast_sint_t)k - alignment], (size_t)alignment * sizeof(sa_sint_t)) : &SA[n + fs - 4 * (fast_sint_t)k];
        buckets = (LIBSAIS_LOCAL_BUFFER_SIZE / k >= 4) ? local_buffer : buckets;

        sa_sint_t m = libsais_count_and_gather_lms_suffixes_32s_2k_omp(T, SA, n, k, buckets, threads, thread_state);
        if (m > 1)
        {
            libsais_initialize_buckets_for_radix_and_partial_sorting_32s_4k(T, k, buckets, SA[n - m]);

            libsais_radix_sort_lms_suffixes_32s_2k_omp(T, SA, n, m, &buckets[1], threads, thread_state);
            libsais_radix_sort_set_markers_32s_4k_omp(SA, k, &buckets[1], threads);
            
            libsais_place_lms_suffixes_interval_32s_4k(SA, n, k, m - 1, buckets);
            libsais_induce_partial_order_32s_4k_omp(T, SA, n, k, buckets, threads, thread_state);

            sa_sint_t names = libsais_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(SA, n, m, threads, thread_state);
            if (names < m)
            {
                sa_sint_t f = libsais_compact_lms_suffixes_32s_omp(T, SA, n, m, fs, threads, thread_state);

                if (libsais_main_32s_recursion(SA + n + fs - m + f, SA, m - f, names - f, fs + n - 2 * m + f, threads, thread_state, local_buffer) != 0)
                {
                    return -2;
                }

                libsais_reconstruct_compacted_lms_suffixes_32s_2k_omp(T, SA, n, k, m, fs, f, buckets, threads, thread_state);
            }
            else
            {
                libsais_count_lms_suffixes_32s_2k(T, n, k, buckets);
            }
        }
        else
        {
            SA[0] = SA[n - 1];
        }

        libsais_initialize_buckets_start_and_end_32s_4k(k, buckets);
        libsais_place_lms_suffixes_histogram_32s_4k(SA, n, k, m, buckets);
        libsais_induce_final_order_32s_4k(T, SA, n, k, buckets, threads, thread_state);

        return 0;
    }
    else if (k > 0 && ((fs / k >= 2) || (LIBSAIS_LOCAL_BUFFER_SIZE / k >= 2)))
    {
        sa_sint_t alignment = (fs - 1024) / k >= 2 ? (sa_sint_t)1024 : (sa_sint_t)16;
        sa_sint_t * RESTRICT buckets = (fs - alignment) / k >= 2 ? (sa_sint_t *)libsais_align_up(&SA[n + fs - 2 * (fast_sint_t)k - alignment], (size_t)alignment * sizeof(sa_sint_t)) : &SA[n + fs - 2 * (fast_sint_t)k];
        buckets = (LIBSAIS_LOCAL_BUFFER_SIZE / k >= 2) ? local_buffer : buckets;

        sa_sint_t m = libsais_count_and_gather_lms_suffixes_32s_2k_omp(T, SA, n, k, buckets, threads, thread_state);
        if (m > 1)
        {
            libsais_initialize_buckets_for_lms_suffixes_radix_sort_32s_2k(T, k, buckets, SA[n - m]);

            libsais_radix_sort_lms_suffixes_32s_2k_omp(T, SA, n, m, &buckets[1], threads, thread_state);
            libsais_place_lms_suffixes_interval_32s_2k(SA, n, k, m - 1, buckets);

            libsais_initialize_buckets_start_and_end_32s_2k(k, buckets);
            libsais_induce_partial_order_32s_2k_omp(T, SA, n, k, buckets, threads, thread_state);

            sa_sint_t names = libsais_renumber_and_mark_distinct_lms_suffixes_32s_1k_omp(T, SA, n, m, threads);
            if (names < m)
            {
                sa_sint_t f = libsais_compact_lms_suffixes_32s_omp(T, SA, n, m, fs, threads, thread_state);

                if (libsais_main_32s_recursion(SA + n + fs - m + f, SA, m - f, names - f, fs + n - 2 * m + f, threads, thread_state, local_buffer) != 0)
                {
                    return -2;
                }

                libsais_reconstruct_compacted_lms_suffixes_32s_2k_omp(T, SA, n, k, m, fs, f, buckets, threads, thread_state);
            }
            else
            {
                libsais_count_lms_suffixes_32s_2k(T, n, k, buckets);
            }
        }
        else
        {
            SA[0] = SA[n - 1];
        }

        libsais_initialize_buckets_end_32s_2k(k, buckets);
        libsais_place_lms_suffixes_histogram_32s_2k(SA, n, k, m, buckets);

        libsais_initialize_buckets_start_and_end_32s_2k(k, buckets);
        libsais_induce_final_order_32s_2k(T, SA, n, k, buckets, threads, thread_state);

        return 0;
    }
    else
    {
        sa_sint_t * buffer = fs < k ? (sa_sint_t *)libsais_alloc_aligned((size_t)k * sizeof(sa_sint_t), 4096) : (sa_sint_t *)NULL;

        sa_sint_t alignment = fs - 1024 >= k ? (sa_sint_t)1024 : (sa_sint_t)16;
        sa_sint_t * RESTRICT buckets = fs - alignment >= k ? (sa_sint_t *)libsais_align_up(&SA[n + fs - k - alignment], (size_t)alignment * sizeof(sa_sint_t)) : fs >= k ? &SA[n + fs - k] : buffer;

        if (buckets == NULL) { return -2; }

        memset(SA, 0, (size_t)n * sizeof(sa_sint_t));

        libsais_count_suffixes_32s(T, n, k, buckets); 
        libsais_initialize_buckets_end_32s_1k(k, buckets);

        sa_sint_t m = libsais_radix_sort_lms_suffixes_32s_1k(T, SA, n, buckets);
        if (m > 1)
        {
            libsais_induce_partial_order_32s_1k_omp(T, SA, n, k, buckets, threads, thread_state);

            sa_sint_t names = libsais_renumber_and_mark_distinct_lms_suffixes_32s_1k_omp(T, SA, n, m, threads);
            if (names < m)
            {
                if (buffer != NULL) { libsais_free_aligned(buffer); buckets = NULL; }

                sa_sint_t f = libsais_compact_lms_suffixes_32s_omp(T, SA, n, m, fs, threads, thread_state);

                if (libsais_main_32s_recursion(SA + n + fs - m + f, SA, m - f, names - f, fs + n - 2 * m + f, threads, thread_state, local_buffer) != 0)
                {
                    return -2;
                }

                libsais_reconstruct_compacted_lms_suffixes_32s_1k_omp(T, SA, n, m, fs, f, threads, thread_state);

                if (buckets == NULL) { buckets = buffer = (sa_sint_t *)libsais_alloc_aligned((size_t)k * sizeof(sa_sint_t), 4096); }
                if (buckets == NULL) { return -2; }
            }
            
            libsais_count_suffixes_32s(T, n, k, buckets);
            libsais_initialize_buckets_end_32s_1k(k, buckets);
            libsais_place_lms_suffixes_interval_32s_1k(T, SA, k, m, buckets);
        }

        libsais_induce_final_order_32s_1k(T, SA, n, k, buckets, threads, thread_state);
        libsais_free_aligned(buffer);

        return 0;
    }
}

static sa_sint_t libsais_main_32s_entry(sa_sint_t * RESTRICT T, sa_sint_t * RESTRICT SA, sa_sint_t n, sa_sint_t k, sa_sint_t fs, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    sa_sint_t local_buffer[LIBSAIS_LOCAL_BUFFER_SIZE];

    return libsais_main_32s_recursion(T, SA, n, k, fs, threads, thread_state, local_buffer);
}

static sa_sint_t libsais_main_8u(const uint8_t * T, sa_sint_t * SA, sa_sint_t n, sa_sint_t * RESTRICT buckets, sa_sint_t bwt, sa_sint_t r, sa_sint_t * RESTRICT I, sa_sint_t fs, sa_sint_t * freq, sa_sint_t threads, LIBSAIS_THREAD_STATE * RESTRICT thread_state)
{
    fs = fs < (SAINT_MAX - n) ? fs : (SAINT_MAX - n);

    /*
        T[]:
        |<----------------------- n unsigned chars ------------------------->|

        SA[]:
        |<-------------------------- n integers ---------------------------->|

        buckets[]: can be primarily divided into 8 parts, each part has ALPHABET_SIZE integers:
        |----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
        |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|<------ 6 ----->|<------ 7 ----->|

    */

#ifndef AOCL_BWT
    /*
        All LMS indexes are stored in SA array, [from n-m to n-1].
        The array buckets[] is modified to store the number of occurrences of a character along with its current type and previous type which is L/S, so there are 4 combinations: LL,LS,SL,SS.
        Integer "m" is returned, which represents number of LMS indexes.

        SA[]:
        |<-------------------------- n ints -------------------------------->|

                           |<--------------- initialization ---------------->|
                           |<--last m elements are updated to LMS indexes -->|


        buckets[]:
        |----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
        |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|<------ 6 ----->|<------ 7 ----->|

        |<-------------------------- initialization ----------------------->|

        BWT Modifications:  Last character is compared with first character instead of senital character,
                            First character is compared with last character to get its previous type.
    */
    sa_sint_t m = libsais_count_and_gather_lms_suffixes_8u_omp(T, SA, n, buckets, threads, thread_state);
#else
    /*
    Before:
        if (SA[1] == -1) -> no LMS character index at SA[1], or SA[1] also contains an LMS character index.
        Note: &SA[1] = &(s->ptr[1])

        |<------------------------------- SA --------------------------------------------->|
        |<- SA[0] ->|<-------------------------- SA[1....] ------------------------------->|
              ^     |<---------------------- SA ------------------------------------------>|
              |     |<- SA[1] ->|<------- LMS characters -------->|<------ buckets ------->|
        length of         ^         from SA[1] to SA[sa_index-1]      SA[sa_index] to SA[sa_index + 4*1024 - 1]
        LMS array is      |
        stored          SA[1] == -1,
                            -> means no LMS chrctr
                        SA[1] != -1
                            -> LMS character is prsnt

    After:
        All LMS indexes are stored in SA array, [from n-m to n-1].
        The array buckets[] is modified to store the number of occurrences of a character along with its current type and previous type which is L/S, so there are 4 combinations: LL,LS,SL,SS.
        Integer "m" is returned, which represents number of LMS indexes.

        SA[]:
        |<-------------------------- n ints -------------------------------->|

                           |<--------------- initialization ---------------->|
                           |<--last m elements are updated to LMS indexes -->|


        buckets[]:
        |----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
        |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|<------ 6 ----->|<------ 7 ----->|

        |<-------------------------- initialization ----------------------->|

        BWT Modifications:  Last character is compared with first character instead of senital character,
                            First character is compared with last character to get its previous type.
    */
    sa_sint_t m = SA[0];
    memcpy(buckets, &SA[m+1], sizeof(sa_sint_t)*4*ALPHABET_SIZE);
    if(SA[1] == -1)
    {
        m--;
        memmove(&SA[n-m], &SA[2],  sizeof(sa_sint_t)*m);
    }
    else
    {
        memmove(&SA[n-m], &SA[1], sizeof(sa_sint_t)*m);
    }
#endif /* AOCL_BWT */
    /*
        bucket_start = &buckets[6 * ALPHABET_SIZE];
        bucket_end   = &buckets[7 * ALPHABET_SIZE];
        ALPHABET_SIZE is 256,
        bucket_start buffer represents head positions of each char in final SA,
        bucket_end buffer represents tail positions of each char in final SA

        buckets[]:
        |----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
        |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|<------ 6 ----->|<------ 7 ----->|
        
                                                                                                              |<------- initialization--------->|
                                                                                                              |<head positions>|<tail positions>|


        BWT Modifications:  None
    */
    sa_sint_t k = libsais_initialize_buckets_start_and_end_8u(buckets, freq);

    if (m > 0)
    {
        #ifdef AOCL_BWT
            sa_sint_t * temp_SA = (sa_sint_t *)libsais_alloc_aligned(sizeof(sa_sint_t)*m, 4096);
            memcpy(temp_SA, &SA[n-m], sizeof(sa_sint_t)*m);
        #endif /* AOCL_BWT */
        /* Represents first lms suffix index */
        sa_sint_t first_lms_suffix    = SA[n - m];

        /*
            &buckets[4*ALPHABET_SIZE] buffer is modified to contain: buckets[4*char + 1]+buckets[4*char + 3]
            which means (previous character is L type)
            decimal->1:  bin->01: type->SL
            decimal->3:  bin->11: type->LL
            
            all are stored from index 4*ALPHABET_SIZE to 6*ALPHABET_SIZE
            `4*ALPHABET_SIZE + i + 1` contains head limits (odd indexes)
            `4*ALPHABET_SIZE + i` contains tail limits of each character (even indexes)

            return: number of character occurrences where previous type is L is returned.

            buckets[]:
            |----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
            |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|<------ 6 ----->|<------ 7 ----->|
            
                                                                                |<------- initialization -------->|
                                                                                |<---- alternating tail and ----> |
                                                                                |  head positions of prev L type  |
            
            BWT Modifications:   Generally all buckets[i] are erased starting from index `first_lms_suffix` to `0`
                                but for BWT none of the indexes are erased.
        */
        sa_sint_t left_suffixes_count = libsais_initialize_buckets_for_lms_suffixes_radix_sort_8u(T, buckets, first_lms_suffix);

        if (threads > 1 && n >= 65536) { memset(SA, 0, ((size_t)n - (size_t)m) * sizeof(sa_sint_t)); }

        /*
            last m elements of SA are traversed which contains LMS indexes, and are sorted in SA itself by radix sort,
            &buckets[4*ALPHABET_SIZE] which contains tail indexes, is also modified in the process

            SA[]:
            |<----------------------- (n-m) ----------------->|<---(m) LMS indexes ------->|

            |<--------------- initialization ---------------->|<------ read through ------>|
            |<----- LMS indexes are scattered here ---------->|
            |       depending on the buckets they belong      |


            buckets[]:
            |----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
            |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|<------ 6 ----->|<------ 7 ----->|
            
                                                                                |<------- used & altered -------->|

            BWT Modifications:   If unchanged i.e, original execution the above step removes first LMS index from buckets
                                and also preceding characters's buckets, hence the number of LMS index becomes m-1.
                                But for BWT compatibility first index is considered so number of LMS indexes is still m,
                                hence in the following function instead of passing m-1 as number of elements we pass m.
        */
        libsais_radix_sort_lms_suffixes_8u_omp(T, SA, n, m, buckets, threads, thread_state);
        if (threads > 1 && n >= 65536) { memset(&SA[(fast_sint_t)n - (fast_sint_t)m], 0, (size_t)m * sizeof(sa_sint_t)); }

        /*
            &buckets[4*ALPHABET_SIZE]:   It contains head indexes of buckets, (4*char + 0, 4* char + 2 (prev char is S type)), in even indexes.
                                    SA can be seen as two halves, first half from index `0` to `left_suffixes_count-1` contains prev type is L.
                                    from `left_suffixes_count` index till end, previous type is S begins.
            &buckets[0*ALPHABET_SIZE]:   are modified to store tail pointers, all odd indexes contains sum of LMS occurrences
                                    even indexes contain 4*char+0, 4*char+2 (previous character is S type)

            buckets[]:
            |----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
            |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|<------ 6 ----->|<------ 7 ----->|
            
            |<------- initialization -------->|                                 |<------- initialization -------->|
            |<-alternating "prev is S type" ->|                                 |<- even indexes are updated to ->|
            |   and "sum of LMS occurances"   |                                 |    contain prev char is S type  |

            BWT Modifications:   In original implementation, `left_suffixes_count` didn't contain first LMS count, so second half needs to start with `left_suffixes_count+1`,
                                after modification first LMS is considered, so 2nd half (prev type is S type) starts with just `left_suffixes_count`.
        */
        libsais_initialize_buckets_for_partial_sorting_8u(T, buckets, first_lms_suffix, left_suffixes_count);

        /*
            In this step partial sorting of LMS indexes is done in the buffer SA & also MSB is used to mark if the adjacent LMS substrings are identical or not.
            The library uses MSB by marking the most significant bit (MSB) of positions in the suffix array that start new ranking groups.
            Each time a position is processed during induced sorting, the library checks the MSB and increments the current rank if the beginning of a new ranking group is encountered.
            Additionally, for each pointer in an induction bucket, the rank of the previous induced position is maintained. Whenever another position is induced,
            this "previous rank" is used to determine whether to mark the newly induced position as the beginning of a new rank group.

            &buckets[4*ALPHABET_SIZE]:   used for scanning left to right, stores head pointers (odd "prev char is L type in original array", even "prev char is S type in original array")
                                    prev is S type starts from a minimum index of `left_suffixes_count`, prev is L type starts from a minimum of `0` index.
            &buckets[0*ALPHABET_SIZE]:   same as above, but contains tail pointers.
            &buckets[2*ALPHABET_SIZE]:   used as a temporary buffer to store ranks from which "previous rank" is fetched.

            SA[]:
            |<------------------------------------ n ------------------------------------->|

            |<----------------------------- used & altered ------------------------------->|

            |<--------------- initialization ---------------->|
            |<-- partially sorted LMS indexes (m integers) -->|


            buckets[]:
            |----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
            |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|<------ 6 ----->|<------ 7 ----->|
            
            |<------- used & altered -------->|<----- used as temp buffer ----->|<-------- used & altered ------->|

            BWT Modifications:   New functions are created which will be called in call stack of this function to consider previous index of `0` to be `n-1`
                                and next element of `n-1` is `0`, for this purpose "MOD" macro is used.
        */
        libsais_induce_partial_order_8u_omp(T, SA, n, k, buckets, first_lms_suffix, left_suffixes_count, threads, thread_state);

        /*
            Two tasks are performed in this function. 1st is renaming LMS indexes 2nd is placing the renamed indexes in original order.
            One observation about number of LMS indexes is, if n is the length of original array, utmost n/2 LMS indexes could be generated.

            SA[]:
            |<---------------------------------- n --------------------------------------->|

                                        |<-------------- (n-m) used & altered ------------>|
                                                         |<-- renumbered LMS indexes (m)-->|
                                               

            BWT Modifications:   Earlier last element couldn't be LMS type, now we are considering last element could be LMS.
        */
        sa_sint_t names = libsais_renumber_and_gather_lms_suffixes_omp(SA, n, m, fs, threads, thread_state);

        // The below condition if satisfied implies all names are non-unique, in order to clarify the order of repeated names, SA-IS operation is performed again.
        if (names < m)
        {
            /*
                Since duplicate names are encountered which implies order of some indexes are ambiguous, SAIS is applied on the renamed buffer in recursion until we encounter unique names.
                The result will be: first `m` elements of `SA` will contain the rank of the LMS indexes.

                SA[]:
                |<----------------------------------- n -------------------------------------->|
                                                             
                |<--------------- used & altered --------------->|<------- read trough ------->|
                |<-------- updated ------->|                     |<- renumbered LMS indexes -->|
                |<--- LMS indexes ranks -->|
                |<---------- m ----------->|<-- fs + n - 2*m --->|<----------- m ------------->|
                                                (here fs = 0)
                

                BWT Modifications:  There are multiple changes, primarily: if enough space is not available, we dynamically allocate 6 * (current number of LMS indexes generated).
                                    Earlier different implementations were taken depending on available free space in SA.
            */
            sa_sint_t * T_next = SA + n + fs - m;
            T_next[-1] = T_next[m-1];
            T_next[-2] = T_next[m-2];
            if (libsais_main_32s_entry(T_next, SA, m, names, fs + n - 2 * m - AOCL_LIBSAIS_MOD_ELEMENTS, threads, thread_state) != 0)
            {
                #ifdef AOCL_BWT
                    libsais_free_aligned(temp_SA);
                #endif /* AOCL_BWT */
                return -2;
            }
            /*
                Same as the first step, refer to `libsais_count_and_gather_lms_suffixes_8u_omp` comments

                SA[]:
                |<---------------------------------- n --------------------------------------->|
                                                             
                                                                  |<----- initialization ----->|
                                                                  |<----- (m) LMS indexes ---->|


                BWT Modifications:  Suffixes would be gathered in last `m` elements of SA array, and their rank would be available in first `m` elements.
                                    If incase `m` is half of the input array, the last LMS character gathered i.e, first element of 2nd half, overwrites the last rank i.e,
                                    last element of 1st half which would be in the middle of SA array.
                                    To prevent this, the function exits once the pointer moves beyond first half.
            */
            #ifdef AOCL_BWT
                memcpy(&SA[n-m], temp_SA,  sizeof(sa_sint_t)*m);
            #else
                libsais_gather_lms_suffixes_8u_omp(T, SA, n, threads, thread_state);
            #endif /* AOCL_BWT */
            /*
                From the above step, all LMS indexes are gathered from SA[n-m] to SA[n-1]
                The rank of these indexes are stored from SA[0] to SA[m-1], in the recursion step.
                From the above to data this function will reorder LMS indexes according to their ranks in SA[0] to SA[m-1].

                SA[]:
                |<---------------------------------- n --------------------------------------->|
                                                             
                |<------- read through ------>|                      |<------- updated ------->|
                |<---- LMS indexes ranks ---->|                      |<-- sorted LMS indexes ->|
                                                                     |<--------- m ----------->|


                BWT Modifications:  None
            */
            libsais_reconstruct_lms_suffixes_omp(SA, n, m, threads);
        }
        #ifdef AOCL_BWT
            libsais_free_aligned(temp_SA);
        #endif /* AOCL_BWT */
        /*
            Distributes all LMS indexes in accordance to their order in last part of each character's respective bucket.

            SA[]:
            |<---------------------------------- n  -------------------------------------->|
                                                            
            |<-------- m LMS indexes are updated whose positions are spread across n ----->|

            buckets[]:
            |----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
            |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|<------ 6 ----->|<------ 7 ----->|
            
                                                                                                                  |<---- used but not modified ---->|
                                                                                                                  |<head positions>|<tail positions>|

            BWT Modifications:  Except LMS indexes all other elements are initialized to -1 in SA, earlier they were initialized to 0.
                                If all elements are initialized to 0 we can't differentiate if it is a non-LMS index or LMS index, if index `0` is LMS index, to avoid that ambiguity
                                non-LMS indexes are initialized to -1.
        */
        libsais_place_lms_suffixes_interval_8u(SA, n, m, buckets);
    }
    else
    {
        /*
            If `m` is zero, which means there is no LMS indexes, this case occurs only when all elements are same.
            Hence initializing all elements to same character.
        */
        for (sa_sint_t i = 0; i < n; i++)
        {
            SA[i] = T[0];
        }
        return 0;
    }

    /*
        Final left_to_right & right_to_left scan.

        SA[]:
        |<---------------------------------- n  -------------------------------------->|
                                                        
        |<------------------- all indexes are updated and sorted --------------------->|

        buckets[]:
        |----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
        |<------ 0 ----->|<------ 1 ----->|<------ 2 ----->|<------ 3 ----->|<------ 4 ----->|<------ 5 ----->|<------ 6 ----->|<------ 7 ----->|
        
                                                                                                              |<------- used & altered -------->|
                                                                                                              |<head positions>|<tail positions>|

        BWT Modifications:  Loop unrolling is commented out. Changes are made to consider index `0` as LMS character.
                            Index of zero index of original array in SA array is returned which is needed for decoding BWT string to original string.
    */
    return libsais_induce_final_order_8u_omp(T, SA, n, k, bwt, r, I, buckets, threads, thread_state);
}

static sa_sint_t libsais_main(const uint8_t * T, sa_sint_t * SA, sa_sint_t n, sa_sint_t bwt, sa_sint_t r, sa_sint_t * I, sa_sint_t fs, sa_sint_t * freq, sa_sint_t threads)
{
    LIBSAIS_THREAD_STATE *  RESTRICT thread_state   = threads > 1 ? libsais_alloc_thread_state(threads) : NULL;
    sa_sint_t *             RESTRICT buckets        = (sa_sint_t *)libsais_alloc_aligned((size_t)8 * ALPHABET_SIZE * sizeof(sa_sint_t), 4096);

    sa_sint_t index = buckets != NULL && (thread_state != NULL || threads == 1)
        ? libsais_main_8u(T, SA, n, buckets, bwt, r, I, fs, freq, threads, thread_state)
        : -2;

    libsais_free_aligned(buckets);
    libsais_free_thread_state(thread_state);

    return index;
}

static sa_sint_t libsais_main_int(sa_sint_t * T, sa_sint_t * SA, sa_sint_t n, sa_sint_t k, sa_sint_t fs, sa_sint_t threads)
{
    LIBSAIS_THREAD_STATE * RESTRICT thread_state = threads > 1 ? libsais_alloc_thread_state(threads) : NULL;

    sa_sint_t index = thread_state != NULL || threads == 1
        ? libsais_main_32s_entry(T, SA, n, k, fs, threads, thread_state)
        : -2;

    libsais_free_thread_state(thread_state);

    return index;
}

static sa_sint_t libsais_main_ctx(const LIBSAIS_CONTEXT * ctx, const uint8_t * T, sa_sint_t * SA, sa_sint_t n, sa_sint_t bwt, sa_sint_t r, sa_sint_t * I, sa_sint_t fs, sa_sint_t * freq)
{
    return ctx != NULL && (ctx->buckets != NULL && (ctx->thread_state != NULL || ctx->threads == 1))
        ? libsais_main_8u(T, SA, n, ctx->buckets, bwt, r, I, fs, freq, (sa_sint_t)ctx->threads, ctx->thread_state)
        : -2;
}

static void libsais_bwt_copy_8u(uint8_t * RESTRICT U, sa_sint_t * RESTRICT A, sa_sint_t n)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = 0, j = (fast_sint_t)n - 7; i < j; i += 8)
    {
        libsais_prefetchr(&A[i + prefetch_distance]);

        U[i + 0] = (uint8_t)A[i + 0];
        U[i + 1] = (uint8_t)A[i + 1];
        U[i + 2] = (uint8_t)A[i + 2];
        U[i + 3] = (uint8_t)A[i + 3];
        U[i + 4] = (uint8_t)A[i + 4];
        U[i + 5] = (uint8_t)A[i + 5];
        U[i + 6] = (uint8_t)A[i + 6];
        U[i + 7] = (uint8_t)A[i + 7];
    }

    for (j += 7; i < j; i += 1)
    {
        U[i] = (uint8_t)A[i];
    }
}

#if defined(LIBSAIS_OPENMP)

static void libsais_bwt_copy_8u_omp(uint8_t * RESTRICT U, sa_sint_t * RESTRICT A, sa_sint_t n, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
        fast_sint_t omp_block_stride  = ((fast_sint_t)n / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : (fast_sint_t)n - omp_block_start;
#else
        UNUSED(threads);

        fast_sint_t omp_block_start   = 0;
        fast_sint_t omp_block_size    = (fast_sint_t)n;
#endif

        libsais_bwt_copy_8u(U + omp_block_start, A + omp_block_start, (sa_sint_t)omp_block_size);
    }
}

#endif

void * libsais_create_ctx(void)
{
    return (void *)libsais_create_ctx_main(1);
}

void libsais_free_ctx(void * ctx)
{
    libsais_free_ctx_main((LIBSAIS_CONTEXT *)ctx);
}

int32_t libsais(const uint8_t * T, int32_t * SA, int32_t n, int32_t fs, int32_t * freq)
{
    if ((T == NULL) || (SA == NULL) || (n < 0) || (fs < 0))
    {
        return -1;
    }
    else if (n < 2)
    {
        if (freq != NULL) { memset(freq, 0, ALPHABET_SIZE * sizeof(int32_t)); }
#ifndef AOCL_BWT
        if (n == 1) { SA[0] = 0; if (freq != NULL) { freq[T[0]]++; } }
#else
        if (n == 1) { SA[0] = T[0]; if (freq != NULL) { freq[T[0]]++; } }
#endif /* AOCL_BWT */
        return 0;
    }

    return libsais_main(T, SA, n, 0, 0, NULL, fs, freq, 1);
}

int32_t libsais_int(int32_t * T, int32_t * SA, int32_t n, int32_t k, int32_t fs)
{
    if ((T == NULL) || (SA == NULL) || (n < 0) || (fs < 0))
    {
        return -1;
    }
    else if (n < 2)
    {
        if (n == 1) { SA[0] = 0; }
        return 0;
    }

    return libsais_main_int(T, SA, n, k, fs, 1);
}

int32_t libsais_ctx(const void * ctx, const uint8_t * T, int32_t * SA, int32_t n, int32_t fs, int32_t * freq)
{
    if ((ctx == NULL) || (T == NULL) || (SA == NULL) || (n < 0) || (fs < 0))
    {
        return -1;
    }
    else if (n < 2)
    {
        if (freq != NULL) { memset(freq, 0, ALPHABET_SIZE * sizeof(int32_t)); }
        if (n == 1) { SA[0] = 0; if (freq != NULL) { freq[T[0]]++; } }
        return 0;
    }

    return libsais_main_ctx((const LIBSAIS_CONTEXT *)ctx, T, SA, n, 0, 0, NULL, fs, freq);
}

int32_t libsais_bwt(const uint8_t * T, uint8_t * U, int32_t * A, int32_t n, int32_t fs, int32_t * freq)
{
    if ((T == NULL) || (U == NULL) || (A == NULL) || (n < 0) || (fs < 0))
    { 
        return -1; 
    }
    else if (n <= 1) 
    { 
        if (freq != NULL) { memset(freq, 0, ALPHABET_SIZE * sizeof(int32_t)); }
        if (n == 1) { U[0] = T[0]; if (freq != NULL) { freq[T[0]]++; } }
        return n;
    }

    sa_sint_t index = libsais_main(T, A, n, 1, 0, NULL, fs, freq, 1);
    if (index >= 0) 
    { 
        index++;

        U[0] = T[n - 1];
        libsais_bwt_copy_8u(U + 1, A, index - 1);
        libsais_bwt_copy_8u(U + index, A + index, n - index);
    }

    return index;
}

int32_t libsais_bwt_aux(const uint8_t * T, uint8_t * U, int32_t * A, int32_t n, int32_t fs, int32_t * freq, int32_t r, int32_t * I)
{
    if ((T == NULL) || (U == NULL) || (A == NULL) || (n < 0) || (fs < 0) || (r < 2) || ((r & (r - 1)) != 0) || (I == NULL))
    { 
        return -1; 
    }
    else if (n <= 1) 
    {
        if (freq != NULL) { memset(freq, 0, ALPHABET_SIZE * sizeof(int32_t)); }
        if (n == 1) { U[0] = T[0]; if (freq != NULL) { freq[T[0]]++; } }
        I[0] = n;
        return 0;
    }

    if (libsais_main(T, A, n, 1, r, I, fs, freq, 1) != 0)
    {
        return -2;
    }

    U[0] = T[n - 1];
    libsais_bwt_copy_8u(U + 1, A, I[0] - 1);
    libsais_bwt_copy_8u(U + I[0], A + I[0], n - I[0]);

    return 0;
}

int32_t libsais_bwt_ctx(const void * ctx, const uint8_t * T, uint8_t * U, int32_t * A, int32_t n, int32_t fs, int32_t * freq)
{
    if ((ctx == NULL) || (T == NULL) || (U == NULL) || (A == NULL) || (n < 0) || (fs < 0))
    { 
        return -1; 
    }
    else if (n <= 1) 
    {
        if (freq != NULL) { memset(freq, 0, ALPHABET_SIZE * sizeof(int32_t)); }
        if (n == 1) { U[0] = T[0]; if (freq != NULL) { freq[T[0]]++; } }
        return n;
    }

    sa_sint_t index = libsais_main_ctx((const LIBSAIS_CONTEXT *)ctx, T, A, n, 1, 0, NULL, fs, freq);
    if (index >= 0) 
    { 
        index++;

        U[0] = T[n - 1];

#if defined(LIBSAIS_OPENMP)
        libsais_bwt_copy_8u_omp(U + 1, A, index - 1, (sa_sint_t)((const LIBSAIS_CONTEXT *)ctx)->threads);
        libsais_bwt_copy_8u_omp(U + index, A + index, n - index, (sa_sint_t)((const LIBSAIS_CONTEXT *)ctx)->threads);
#else
        libsais_bwt_copy_8u(U + 1, A, index - 1);
        libsais_bwt_copy_8u(U + index, A + index, n - index);
#endif
    }

    return index;
}

int32_t libsais_bwt_aux_ctx(const void * ctx, const uint8_t * T, uint8_t * U, int32_t * A, int32_t n, int32_t fs, int32_t * freq, int32_t r, int32_t * I)
{
    if ((ctx == NULL) || (T == NULL) || (U == NULL) || (A == NULL) || (n < 0) || (fs < 0) || (r < 2) || ((r & (r - 1)) != 0) || (I == NULL))
    { 
        return -1; 
    }
    else if (n <= 1) 
    {
        if (freq != NULL) { memset(freq, 0, ALPHABET_SIZE * sizeof(int32_t)); }
        if (n == 1) { U[0] = T[0]; if (freq != NULL) { freq[T[0]]++; } }
        I[0] = n;
        return 0;
    }

    if (libsais_main_ctx((const LIBSAIS_CONTEXT *)ctx, T, A, n, 1, r, I, fs, freq) != 0)
    {
        return -2;
    }

    U[0] = T[n - 1];

#if defined(LIBSAIS_OPENMP)
    libsais_bwt_copy_8u_omp(U + 1, A, I[0] - 1, (sa_sint_t)((const LIBSAIS_CONTEXT *)ctx)->threads);
    libsais_bwt_copy_8u_omp(U + I[0], A + I[0], n - I[0], (sa_sint_t)((const LIBSAIS_CONTEXT *)ctx)->threads);
#else
    libsais_bwt_copy_8u(U + 1, A, I[0] - 1);
    libsais_bwt_copy_8u(U + I[0], A + I[0], n - I[0]);
#endif

    return 0;
}

#if defined(LIBSAIS_OPENMP)

void * libsais_create_ctx_omp(int32_t threads)
{
    if (threads < 0) { return NULL; }

    threads = threads > 0 ? threads : omp_get_max_threads();
    return (void *)libsais_create_ctx_main(threads);
}

int32_t libsais_omp(const uint8_t * T, int32_t * SA, int32_t n, int32_t fs, int32_t * freq, int32_t threads)
{
    if ((T == NULL) || (SA == NULL) || (n < 0) || (fs < 0) || (threads < 0))
    {
        return -1;
    }
    else if (n < 2)
    {
        if (freq != NULL) { memset(freq, 0, ALPHABET_SIZE * sizeof(int32_t)); }
        if (n == 1) { SA[0] = 0; if (freq != NULL) { freq[T[0]]++; } }
        return 0;
    }

    threads = threads > 0 ? threads : omp_get_max_threads();

    return libsais_main(T, SA, n, 0, 0, NULL, fs, freq, threads);
}

int32_t libsais_int_omp(int32_t * T, int32_t * SA, int32_t n, int32_t k, int32_t fs, int32_t threads)
{
    if ((T == NULL) || (SA == NULL) || (n < 0) || (fs < 0) || (threads < 0))
    {
        return -1;
    }
    else if (n < 2)
    {
        if (n == 1) { SA[0] = 0; }
        return 0;
    }

    threads = threads > 0 ? threads : omp_get_max_threads();

    return libsais_main_int(T, SA, n, k, fs, threads);
}

int32_t libsais_bwt_omp(const uint8_t * T, uint8_t * U, int32_t * A, int32_t n, int32_t fs, int32_t * freq, int32_t threads)
{
    if ((T == NULL) || (U == NULL) || (A == NULL) || (n < 0) || (fs < 0) || (threads < 0))
    {
        return -1;
    }
    else if (n <= 1)
    {
        if (freq != NULL) { memset(freq, 0, ALPHABET_SIZE * sizeof(int32_t)); }
        if (n == 1) { U[0] = T[0]; if (freq != NULL) { freq[T[0]]++; } }
        return n;
    }

    threads = threads > 0 ? threads : omp_get_max_threads();

    sa_sint_t index = libsais_main(T, A, n, 1, 0, NULL, fs, freq, threads);
    if (index >= 0)
    {
        index++;

        U[0] = T[n - 1];
        libsais_bwt_copy_8u_omp(U + 1, A, index - 1, threads);
        libsais_bwt_copy_8u_omp(U + index, A + index, n - index, threads);
    }

    return index;
}

int32_t libsais_bwt_aux_omp(const uint8_t * T, uint8_t * U, int32_t * A, int32_t n, int32_t fs, int32_t * freq, int32_t r, int32_t * I, int32_t threads)
{
    if ((T == NULL) || (U == NULL) || (A == NULL) || (n < 0) || (fs < 0) || (r < 2) || ((r & (r - 1)) != 0) || (I == NULL) || (threads < 0))
    {
        return -1;
    }
    else if (n <= 1)
    {
        if (freq != NULL) { memset(freq, 0, ALPHABET_SIZE * sizeof(int32_t)); }
        if (n == 1) { U[0] = T[0]; if (freq != NULL) { freq[T[0]]++; } }
        I[0] = n;
        return 0;
    }

    threads = threads > 0 ? threads : omp_get_max_threads();

    if (libsais_main(T, A, n, 1, r, I, fs, freq, threads) != 0)
    {
        return -2;
    }

    U[0] = T[n - 1];
    libsais_bwt_copy_8u_omp(U + 1, A, I[0] - 1, threads);
    libsais_bwt_copy_8u_omp(U + I[0], A + I[0], n - I[0], threads);

    return 0;
}

#endif

static LIBSAIS_UNBWT_CONTEXT * libsais_unbwt_create_ctx_main(sa_sint_t threads)
{
    LIBSAIS_UNBWT_CONTEXT *     RESTRICT ctx            = (LIBSAIS_UNBWT_CONTEXT *)libsais_alloc_aligned(sizeof(LIBSAIS_UNBWT_CONTEXT), 64);
    sa_uint_t *                 RESTRICT bucket2        = (sa_uint_t *)libsais_alloc_aligned(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(sa_uint_t), 4096);
    uint16_t *                  RESTRICT fastbits       = (uint16_t *)libsais_alloc_aligned((1 + (1 << UNBWT_FASTBITS)) * sizeof(uint16_t), 4096);
    sa_uint_t *                 RESTRICT buckets        = threads > 1 ? (sa_uint_t *)libsais_alloc_aligned((size_t)threads * (ALPHABET_SIZE + (ALPHABET_SIZE * ALPHABET_SIZE)) * sizeof(sa_uint_t), 4096) : NULL;

    if (ctx != NULL && bucket2 != NULL && fastbits != NULL && (buckets != NULL || threads == 1))
    {
        ctx->bucket2    = bucket2;
        ctx->fastbits   = fastbits;
        ctx->buckets    = buckets;
        ctx->threads    = threads;

        return ctx;
    }

    libsais_free_aligned(buckets);
    libsais_free_aligned(fastbits);
    libsais_free_aligned(bucket2);
    libsais_free_aligned(ctx);

    return NULL;
}

static void libsais_unbwt_free_ctx_main(LIBSAIS_UNBWT_CONTEXT * ctx)
{
    if (ctx != NULL)
    {
        libsais_free_aligned(ctx->buckets);
        libsais_free_aligned(ctx->fastbits);
        libsais_free_aligned(ctx->bucket2);
        libsais_free_aligned(ctx);
    }
}

static void libsais_unbwt_compute_histogram(const uint8_t * RESTRICT T, fast_sint_t n, sa_uint_t * RESTRICT count)
{
    const fast_sint_t prefetch_distance = 256;

    const uint8_t * RESTRICT T_p = T;

    if (n >= 1024)
    {
        sa_uint_t copy[4 * (ALPHABET_SIZE + 16)];

        memset(copy, 0, (size_t)4 * (ALPHABET_SIZE + 16) * sizeof(sa_uint_t));

        sa_uint_t * RESTRICT copy0 = copy + 0 * (ALPHABET_SIZE + 16);
        sa_uint_t * RESTRICT copy1 = copy + 1 * (ALPHABET_SIZE + 16);
        sa_uint_t * RESTRICT copy2 = copy + 2 * (ALPHABET_SIZE + 16);
        sa_uint_t * RESTRICT copy3 = copy + 3 * (ALPHABET_SIZE + 16);

        for (; T_p < (uint8_t * )((ptrdiff_t)(T + 63) & (-64)); T_p += 1) { copy0[T_p[0]]++; }

        fast_uint_t x = ((const uint32_t *)(const void *)T_p)[0], y = ((const uint32_t *)(const void *)T_p)[1];

        for (; T_p < (uint8_t * )((ptrdiff_t)(T + n - 8) & (-64)); T_p += 64)
        { 
            libsais_prefetchr(&T_p[prefetch_distance]);

            fast_uint_t z = ((const uint32_t *)(const void *)T_p)[2], w = ((const uint32_t *)(const void *)T_p)[3];
            copy0[(uint8_t)x]++; x >>= 8; copy1[(uint8_t)x]++; x >>= 8; copy2[(uint8_t)x]++; x >>= 8; copy3[x]++;
            copy0[(uint8_t)y]++; y >>= 8; copy1[(uint8_t)y]++; y >>= 8; copy2[(uint8_t)y]++; y >>= 8; copy3[y]++;

            x = ((const uint32_t *)(const void *)T_p)[4]; y = ((const uint32_t *)(const void *)T_p)[5];
            copy0[(uint8_t)z]++; z >>= 8; copy1[(uint8_t)z]++; z >>= 8; copy2[(uint8_t)z]++; z >>= 8; copy3[z]++;
            copy0[(uint8_t)w]++; w >>= 8; copy1[(uint8_t)w]++; w >>= 8; copy2[(uint8_t)w]++; w >>= 8; copy3[w]++;

            z = ((const uint32_t *)(const void *)T_p)[6]; w = ((const uint32_t *)(const void *)T_p)[7];
            copy0[(uint8_t)x]++; x >>= 8; copy1[(uint8_t)x]++; x >>= 8; copy2[(uint8_t)x]++; x >>= 8; copy3[x]++;
            copy0[(uint8_t)y]++; y >>= 8; copy1[(uint8_t)y]++; y >>= 8; copy2[(uint8_t)y]++; y >>= 8; copy3[y]++;

            x = ((const uint32_t *)(const void *)T_p)[8]; y = ((const uint32_t *)(const void *)T_p)[9];
            copy0[(uint8_t)z]++; z >>= 8; copy1[(uint8_t)z]++; z >>= 8; copy2[(uint8_t)z]++; z >>= 8; copy3[z]++;
            copy0[(uint8_t)w]++; w >>= 8; copy1[(uint8_t)w]++; w >>= 8; copy2[(uint8_t)w]++; w >>= 8; copy3[w]++;

            z = ((const uint32_t *)(const void *)T_p)[10]; w = ((const uint32_t *)(const void *)T_p)[11];
            copy0[(uint8_t)x]++; x >>= 8; copy1[(uint8_t)x]++; x >>= 8; copy2[(uint8_t)x]++; x >>= 8; copy3[x]++;
            copy0[(uint8_t)y]++; y >>= 8; copy1[(uint8_t)y]++; y >>= 8; copy2[(uint8_t)y]++; y >>= 8; copy3[y]++;

            x = ((const uint32_t *)(const void *)T_p)[12]; y = ((const uint32_t *)(const void *)T_p)[13];
            copy0[(uint8_t)z]++; z >>= 8; copy1[(uint8_t)z]++; z >>= 8; copy2[(uint8_t)z]++; z >>= 8; copy3[z]++;
            copy0[(uint8_t)w]++; w >>= 8; copy1[(uint8_t)w]++; w >>= 8; copy2[(uint8_t)w]++; w >>= 8; copy3[w]++;

            z = ((const uint32_t *)(const void *)T_p)[14]; w = ((const uint32_t *)(const void *)T_p)[15];
            copy0[(uint8_t)x]++; x >>= 8; copy1[(uint8_t)x]++; x >>= 8; copy2[(uint8_t)x]++; x >>= 8; copy3[x]++;
            copy0[(uint8_t)y]++; y >>= 8; copy1[(uint8_t)y]++; y >>= 8; copy2[(uint8_t)y]++; y >>= 8; copy3[y]++;

            x = ((const uint32_t *)(const void *)T_p)[16]; y = ((const uint32_t *)(const void *)T_p)[17];
            copy0[(uint8_t)z]++; z >>= 8; copy1[(uint8_t)z]++; z >>= 8; copy2[(uint8_t)z]++; z >>= 8; copy3[z]++;
            copy0[(uint8_t)w]++; w >>= 8; copy1[(uint8_t)w]++; w >>= 8; copy2[(uint8_t)w]++; w >>= 8; copy3[w]++;
        }

        copy0[(uint8_t)x]++; x >>= 8; copy1[(uint8_t)x]++; x >>= 8; copy2[(uint8_t)x]++; x >>= 8; copy3[x]++;
        copy0[(uint8_t)y]++; y >>= 8; copy1[(uint8_t)y]++; y >>= 8; copy2[(uint8_t)y]++; y >>= 8; copy3[y]++;

        T_p += 8;

        fast_uint_t i; for (i = 0; i < ALPHABET_SIZE; i++) { count[i] += copy0[i] + copy1[i] + copy2[i] + copy3[i]; }
    }

    for (; T_p < T + n; T_p += 1) { count[T_p[0]]++; }
}

static void libsais_unbwt_transpose_bucket2(sa_uint_t * RESTRICT bucket2)
{
    fast_uint_t x, y, c, d;
    for (x = 0; x != ALPHABET_SIZE; x += 16)
    {
        for (c = x; c != x + 16; ++c)
        {
            for (d = c + 1; d != x + 16; ++d)
            {
                sa_uint_t tmp = bucket2[(d << 8) + c]; bucket2[(d << 8) + c] = bucket2[(c << 8) + d]; bucket2[(c << 8) + d] = tmp;
            }
        }

        for (y = x + 16; y != ALPHABET_SIZE; y += 16)
        {
            for (c = x; c != x + 16; ++c)
            {
                sa_uint_t * bucket2_yc = &bucket2[(y << 8) + c];
                sa_uint_t * bucket2_cy = &bucket2[(c << 8) + y];

                sa_uint_t tmp00 = bucket2_yc[ 0 * 256]; bucket2_yc[ 0 * 256] = bucket2_cy[ 0]; bucket2_cy[ 0] = tmp00;
                sa_uint_t tmp01 = bucket2_yc[ 1 * 256]; bucket2_yc[ 1 * 256] = bucket2_cy[ 1]; bucket2_cy[ 1] = tmp01;
                sa_uint_t tmp02 = bucket2_yc[ 2 * 256]; bucket2_yc[ 2 * 256] = bucket2_cy[ 2]; bucket2_cy[ 2] = tmp02;
                sa_uint_t tmp03 = bucket2_yc[ 3 * 256]; bucket2_yc[ 3 * 256] = bucket2_cy[ 3]; bucket2_cy[ 3] = tmp03;
                sa_uint_t tmp04 = bucket2_yc[ 4 * 256]; bucket2_yc[ 4 * 256] = bucket2_cy[ 4]; bucket2_cy[ 4] = tmp04;
                sa_uint_t tmp05 = bucket2_yc[ 5 * 256]; bucket2_yc[ 5 * 256] = bucket2_cy[ 5]; bucket2_cy[ 5] = tmp05;
                sa_uint_t tmp06 = bucket2_yc[ 6 * 256]; bucket2_yc[ 6 * 256] = bucket2_cy[ 6]; bucket2_cy[ 6] = tmp06;
                sa_uint_t tmp07 = bucket2_yc[ 7 * 256]; bucket2_yc[ 7 * 256] = bucket2_cy[ 7]; bucket2_cy[ 7] = tmp07;
                sa_uint_t tmp08 = bucket2_yc[ 8 * 256]; bucket2_yc[ 8 * 256] = bucket2_cy[ 8]; bucket2_cy[ 8] = tmp08;
                sa_uint_t tmp09 = bucket2_yc[ 9 * 256]; bucket2_yc[ 9 * 256] = bucket2_cy[ 9]; bucket2_cy[ 9] = tmp09;
                sa_uint_t tmp10 = bucket2_yc[10 * 256]; bucket2_yc[10 * 256] = bucket2_cy[10]; bucket2_cy[10] = tmp10;
                sa_uint_t tmp11 = bucket2_yc[11 * 256]; bucket2_yc[11 * 256] = bucket2_cy[11]; bucket2_cy[11] = tmp11;
                sa_uint_t tmp12 = bucket2_yc[12 * 256]; bucket2_yc[12 * 256] = bucket2_cy[12]; bucket2_cy[12] = tmp12;
                sa_uint_t tmp13 = bucket2_yc[13 * 256]; bucket2_yc[13 * 256] = bucket2_cy[13]; bucket2_cy[13] = tmp13;
                sa_uint_t tmp14 = bucket2_yc[14 * 256]; bucket2_yc[14 * 256] = bucket2_cy[14]; bucket2_cy[14] = tmp14;
                sa_uint_t tmp15 = bucket2_yc[15 * 256]; bucket2_yc[15 * 256] = bucket2_cy[15]; bucket2_cy[15] = tmp15;
            }
        }
    }
}

static void libsais_unbwt_compute_bigram_histogram_single(const uint8_t * RESTRICT T, sa_uint_t * RESTRICT bucket1, sa_uint_t * RESTRICT bucket2, fast_uint_t index)
{
    fast_uint_t sum, c;
    for (sum = 1, c = 0; c < ALPHABET_SIZE; ++c)
    {
        fast_uint_t prev = sum; sum += bucket1[c]; bucket1[c] = (sa_uint_t)prev;
        if (prev != sum)
        {
            sa_uint_t * RESTRICT bucket2_p = &bucket2[c << 8];

            {
                fast_uint_t hi = index; if (sum < hi) { hi = sum; }
                libsais_unbwt_compute_histogram(&T[prev], (fast_sint_t)(hi - prev), bucket2_p);
            }

            {
                fast_uint_t lo = index + 1; if (prev > lo) { lo = prev; }
                libsais_unbwt_compute_histogram(&T[lo - 1], (fast_sint_t)(sum - lo), bucket2_p);
            }
        }
    }

    libsais_unbwt_transpose_bucket2(bucket2);
}

static void libsais_unbwt_calculate_fastbits(sa_uint_t * RESTRICT bucket2, uint16_t * RESTRICT fastbits, fast_uint_t lastc, fast_uint_t shift)
{
    fast_uint_t v, w, sum, c, d;
    for (v = 0, w = 0, sum = 1, c = 0; c < ALPHABET_SIZE; ++c)
    {
        if (c == lastc) { sum += 1; }

        for (d = 0; d < ALPHABET_SIZE; ++d, ++w)
        {
            fast_uint_t prev = sum; sum += bucket2[w]; bucket2[w] = (sa_uint_t)prev;
            if (prev != sum)
            {
                for (; v <= ((sum - 1) >> shift); ++v) { fastbits[v] = (uint16_t)w; }
            }
        }
    }
}

static void libsais_unbwt_calculate_biPSI(const uint8_t * RESTRICT T, sa_uint_t * RESTRICT P, sa_uint_t * RESTRICT bucket1, sa_uint_t * RESTRICT bucket2, fast_uint_t index, fast_sint_t omp_block_start, fast_sint_t omp_block_end)
{
    {
        fast_sint_t i = omp_block_start, j = (fast_sint_t)index; if (omp_block_end < j) { j = omp_block_end; }
        for (; i < j; ++i)
        {
            fast_uint_t c = T[i];
            fast_uint_t p = bucket1[c]++;
            fast_sint_t t = (fast_sint_t)(index - p);

            if (t != 0)
            {
                fast_uint_t w = (((fast_uint_t)T[p + (fast_uint_t)(t >> ((sizeof(fast_sint_t) * 8) - 1))]) << 8) + c;
                P[bucket2[w]++] = (sa_uint_t)i;
            }
        }
    }

    {
        fast_sint_t i = (fast_sint_t)index, j = omp_block_end; if (omp_block_start > i) { i = omp_block_start; }
        for (i += 1; i <= j; ++i)
        {
            fast_uint_t c = T[i - 1];
            fast_uint_t p = bucket1[c]++;
            fast_sint_t t = (fast_sint_t)(index - p);

            if (t != 0)
            {
                fast_uint_t w = (((fast_uint_t)T[p + (fast_uint_t)(t >> ((sizeof(fast_sint_t) * 8) - 1))]) << 8) + c;
                P[bucket2[w]++] = (sa_uint_t)i;
            }
        }
    }
}

static void libsais_unbwt_init_single(const uint8_t * RESTRICT T, sa_uint_t * RESTRICT P, sa_sint_t n, const sa_sint_t * freq, const sa_uint_t * RESTRICT I, sa_uint_t * RESTRICT bucket2, uint16_t * RESTRICT fastbits)
{
    sa_uint_t bucket1[ALPHABET_SIZE];

    fast_uint_t index = I[0];
    fast_uint_t lastc = T[0];
    fast_uint_t shift = 0; while ((n >> shift) > ((sa_sint_t)1 << UNBWT_FASTBITS)) { shift++; }

    if (freq != NULL)
    {
        memcpy(bucket1, freq, ALPHABET_SIZE * sizeof(sa_uint_t));
    }
    else
    {
        memset(bucket1, 0, ALPHABET_SIZE * sizeof(sa_uint_t));
        libsais_unbwt_compute_histogram(T, n, bucket1);
    }

    memset(bucket2, 0, ALPHABET_SIZE * ALPHABET_SIZE * sizeof(sa_uint_t));
    libsais_unbwt_compute_bigram_histogram_single(T, bucket1, bucket2, index);

    libsais_unbwt_calculate_fastbits(bucket2, fastbits, lastc, shift);
    libsais_unbwt_calculate_biPSI(T, P, bucket1, bucket2, index, 0, n);
}

#if defined(LIBSAIS_OPENMP)

static void libsais_unbwt_compute_bigram_histogram_parallel(const uint8_t * RESTRICT T, fast_uint_t index, sa_uint_t * RESTRICT bucket1, sa_uint_t * RESTRICT bucket2, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    fast_sint_t i;
    for (i = omp_block_start; i < omp_block_start + omp_block_size; ++i)
    {
        fast_uint_t c = T[i];
        fast_uint_t p = bucket1[c]++;
        fast_sint_t t = (fast_sint_t)(index - p);

        if (t != 0)
        {
            fast_uint_t w = (((fast_uint_t)T[p + (fast_uint_t)(t >> ((sizeof(fast_sint_t) * 8) - 1))]) << 8) + c;
            bucket2[w]++;
        }
    }
}

static void libsais_unbwt_init_parallel(const uint8_t * RESTRICT T, sa_uint_t * RESTRICT P, sa_sint_t n, const sa_sint_t * freq, const sa_uint_t * RESTRICT I, sa_uint_t * RESTRICT bucket2, uint16_t * RESTRICT fastbits, sa_uint_t * RESTRICT buckets, sa_sint_t threads)
{
    sa_uint_t bucket1[ALPHABET_SIZE];

    fast_uint_t index = I[0];
    fast_uint_t lastc = T[0];
    fast_uint_t shift = 0; while ((n >> shift) > ((sa_sint_t)1 << UNBWT_FASTBITS)) { shift++; }

    memset(bucket1, 0, ALPHABET_SIZE * sizeof(sa_uint_t));
    memset(bucket2, 0, ALPHABET_SIZE * ALPHABET_SIZE * sizeof(sa_uint_t));

    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 65536)
    {
        fast_sint_t omp_thread_num  = omp_get_thread_num();
        fast_sint_t omp_num_threads = omp_get_num_threads();

        if (omp_num_threads == 1)
        {
            libsais_unbwt_init_single(T, P, n, freq, I, bucket2, fastbits);
        }
        else
        {
            sa_uint_t * RESTRICT bucket1_local  = buckets + omp_thread_num * (ALPHABET_SIZE + (ALPHABET_SIZE * ALPHABET_SIZE));
            sa_uint_t * RESTRICT bucket2_local  = bucket1_local + ALPHABET_SIZE;

            fast_sint_t omp_block_stride        = (n / omp_num_threads) & (-16);
            fast_sint_t omp_block_start         = omp_thread_num * omp_block_stride;
            fast_sint_t omp_block_size          = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : n - omp_block_start;

            {
                memset(bucket1_local, 0, ALPHABET_SIZE * sizeof(sa_uint_t));
                libsais_unbwt_compute_histogram(T + omp_block_start, omp_block_size, bucket1_local);
            }

            #pragma omp barrier

            #pragma omp master
            {
                {
                    sa_uint_t * RESTRICT bucket1_temp = buckets;

                    fast_sint_t t;
                    for (t = 0; t < omp_num_threads; ++t, bucket1_temp += ALPHABET_SIZE + (ALPHABET_SIZE * ALPHABET_SIZE))
                    {
                        fast_sint_t c; for (c = 0; c < ALPHABET_SIZE; c += 1) { sa_uint_t A = bucket1[c], B = bucket1_temp[c]; bucket1[c] = A + B; bucket1_temp[c] = A; }
                    }
                }

                {
                    fast_uint_t sum, c;
                    for (sum = 1, c = 0; c < ALPHABET_SIZE; ++c) { fast_uint_t prev = sum; sum += bucket1[c]; bucket1[c] = (sa_uint_t)prev; }
                }
            }

            #pragma omp barrier

            {
                fast_sint_t c; for (c = 0; c < ALPHABET_SIZE; c += 1) { sa_uint_t A = bucket1[c], B = bucket1_local[c]; bucket1_local[c] = A + B; }

                memset(bucket2_local, 0, ALPHABET_SIZE * ALPHABET_SIZE * sizeof(sa_uint_t));
                libsais_unbwt_compute_bigram_histogram_parallel(T, index, bucket1_local, bucket2_local, omp_block_start, omp_block_size);
            }

            #pragma omp barrier

            {
                fast_sint_t omp_bucket2_stride  = ((ALPHABET_SIZE * ALPHABET_SIZE) / omp_num_threads) & (-16);
                fast_sint_t omp_bucket2_start   = omp_thread_num * omp_bucket2_stride;
                fast_sint_t omp_bucket2_size    = omp_thread_num < omp_num_threads - 1 ? omp_bucket2_stride : (ALPHABET_SIZE * ALPHABET_SIZE) - omp_bucket2_start;

                sa_uint_t * RESTRICT bucket2_temp = buckets + ALPHABET_SIZE;

                fast_sint_t t;
                for (t = 0; t < omp_num_threads; ++t, bucket2_temp += ALPHABET_SIZE + (ALPHABET_SIZE * ALPHABET_SIZE))
                {
                    fast_sint_t c; for (c = omp_bucket2_start; c < omp_bucket2_start + omp_bucket2_size; c += 1) { sa_uint_t A = bucket2[c], B = bucket2_temp[c]; bucket2[c] = A + B; bucket2_temp[c] = A; }
                }
            }

            #pragma omp barrier

            #pragma omp master
            {

                libsais_unbwt_calculate_fastbits(bucket2, fastbits, lastc, shift);

                {
                    fast_sint_t t;
                    for (t = omp_num_threads - 1; t >= 1; --t) 
                    { 
                        sa_uint_t * RESTRICT dst_bucket1 = buckets + t * (ALPHABET_SIZE + (ALPHABET_SIZE * ALPHABET_SIZE));
                        sa_uint_t * RESTRICT src_bucket1 = dst_bucket1 - (ALPHABET_SIZE + (ALPHABET_SIZE * ALPHABET_SIZE));

                        memcpy(dst_bucket1, src_bucket1, ALPHABET_SIZE * sizeof(sa_uint_t));
                    }

                    memcpy(buckets, bucket1, ALPHABET_SIZE * sizeof(sa_uint_t));
                }
            }

            #pragma omp barrier

            {
                fast_sint_t c; for (c = 0; c < ALPHABET_SIZE * ALPHABET_SIZE; c += 1) { sa_uint_t A = bucket2[c], B = bucket2_local[c]; bucket2_local[c] = A + B; }

                libsais_unbwt_calculate_biPSI(T, P, bucket1_local, bucket2_local, index, omp_block_start, omp_block_start + omp_block_size);
            }

            #pragma omp barrier

            #pragma omp master
            {
                memcpy(bucket2, buckets + ALPHABET_SIZE + (omp_num_threads - 1) * (ALPHABET_SIZE + (ALPHABET_SIZE * ALPHABET_SIZE)), ALPHABET_SIZE * ALPHABET_SIZE * sizeof(sa_uint_t));
            }
        }
    }
}

#endif

static void libsais_unbwt_decode_1(uint8_t * RESTRICT U, sa_uint_t * RESTRICT P, sa_uint_t * RESTRICT bucket2, uint16_t * RESTRICT fastbits, fast_uint_t shift, fast_uint_t * i0, fast_uint_t k)
{
    uint16_t * RESTRICT U0 = (uint16_t *)(void *)U;

    fast_uint_t i, p0 = *i0;

    for (i = 0; i != k; ++i)
    {
        uint16_t c0 = fastbits[p0 >> shift]; if (bucket2[c0] <= p0) { do { c0++; } while (bucket2[c0] <= p0); } p0 = P[p0]; U0[i] = libsais_bswap16(c0);
    }

    *i0 = p0;
}

static void libsais_unbwt_decode_2(uint8_t * RESTRICT U, sa_uint_t * RESTRICT P, sa_uint_t * RESTRICT bucket2, uint16_t * RESTRICT fastbits, fast_uint_t shift, fast_uint_t r, fast_uint_t * i0, fast_uint_t * i1, fast_uint_t k)
{
    uint16_t * RESTRICT U0 = (uint16_t *)(void *)U;
    uint16_t * RESTRICT U1 = (uint16_t *)(void *)(((uint8_t *)U0) + r);

    fast_uint_t i, p0 = *i0, p1 = *i1;

    for (i = 0; i != k; ++i)
    {
        uint16_t c0 = fastbits[p0 >> shift]; if (bucket2[c0] <= p0) { do { c0++; } while (bucket2[c0] <= p0); } p0 = P[p0]; U0[i] = libsais_bswap16(c0);
        uint16_t c1 = fastbits[p1 >> shift]; if (bucket2[c1] <= p1) { do { c1++; } while (bucket2[c1] <= p1); } p1 = P[p1]; U1[i] = libsais_bswap16(c1);
    }

    *i0 = p0; *i1 = p1;
}

static void libsais_unbwt_decode_3(uint8_t * RESTRICT U, sa_uint_t * RESTRICT P, sa_uint_t * RESTRICT bucket2, uint16_t * RESTRICT fastbits, fast_uint_t shift, fast_uint_t r, fast_uint_t * i0, fast_uint_t * i1, fast_uint_t * i2, fast_uint_t k)
{
    uint16_t * RESTRICT U0 = (uint16_t *)(void *)U;
    uint16_t * RESTRICT U1 = (uint16_t *)(void *)(((uint8_t *)U0) + r);
    uint16_t * RESTRICT U2 = (uint16_t *)(void *)(((uint8_t *)U1) + r);

    fast_uint_t i, p0 = *i0, p1 = *i1, p2 = *i2;

    for (i = 0; i != k; ++i)
    {
        uint16_t c0 = fastbits[p0 >> shift]; if (bucket2[c0] <= p0) { do { c0++; } while (bucket2[c0] <= p0); } p0 = P[p0]; U0[i] = libsais_bswap16(c0);
        uint16_t c1 = fastbits[p1 >> shift]; if (bucket2[c1] <= p1) { do { c1++; } while (bucket2[c1] <= p1); } p1 = P[p1]; U1[i] = libsais_bswap16(c1);
        uint16_t c2 = fastbits[p2 >> shift]; if (bucket2[c2] <= p2) { do { c2++; } while (bucket2[c2] <= p2); } p2 = P[p2]; U2[i] = libsais_bswap16(c2);
    }

    *i0 = p0; *i1 = p1; *i2 = p2;
}

static void libsais_unbwt_decode_4(uint8_t * RESTRICT U, sa_uint_t * RESTRICT P, sa_uint_t * RESTRICT bucket2, uint16_t * RESTRICT fastbits, fast_uint_t shift, fast_uint_t r, fast_uint_t * i0, fast_uint_t * i1, fast_uint_t * i2, fast_uint_t * i3, fast_uint_t k)
{
    uint16_t * RESTRICT U0 = (uint16_t *)(void *)U;
    uint16_t * RESTRICT U1 = (uint16_t *)(void *)(((uint8_t *)U0) + r);
    uint16_t * RESTRICT U2 = (uint16_t *)(void *)(((uint8_t *)U1) + r);
    uint16_t * RESTRICT U3 = (uint16_t *)(void *)(((uint8_t *)U2) + r);

    fast_uint_t i, p0 = *i0, p1 = *i1, p2 = *i2, p3 = *i3;

    for (i = 0; i != k; ++i)
    {
        uint16_t c0 = fastbits[p0 >> shift]; if (bucket2[c0] <= p0) { do { c0++; } while (bucket2[c0] <= p0); } p0 = P[p0]; U0[i] = libsais_bswap16(c0);
        uint16_t c1 = fastbits[p1 >> shift]; if (bucket2[c1] <= p1) { do { c1++; } while (bucket2[c1] <= p1); } p1 = P[p1]; U1[i] = libsais_bswap16(c1);
        uint16_t c2 = fastbits[p2 >> shift]; if (bucket2[c2] <= p2) { do { c2++; } while (bucket2[c2] <= p2); } p2 = P[p2]; U2[i] = libsais_bswap16(c2);
        uint16_t c3 = fastbits[p3 >> shift]; if (bucket2[c3] <= p3) { do { c3++; } while (bucket2[c3] <= p3); } p3 = P[p3]; U3[i] = libsais_bswap16(c3);
    }

    *i0 = p0; *i1 = p1; *i2 = p2; *i3 = p3;
}

static void libsais_unbwt_decode_5(uint8_t * RESTRICT U, sa_uint_t * RESTRICT P, sa_uint_t * RESTRICT bucket2, uint16_t * RESTRICT fastbits, fast_uint_t shift, fast_uint_t r, fast_uint_t * i0, fast_uint_t * i1, fast_uint_t * i2, fast_uint_t * i3, fast_uint_t * i4, fast_uint_t k)
{
    uint16_t * RESTRICT U0 = (uint16_t *)(void *)U;
    uint16_t * RESTRICT U1 = (uint16_t *)(void *)(((uint8_t *)U0) + r);
    uint16_t * RESTRICT U2 = (uint16_t *)(void *)(((uint8_t *)U1) + r);
    uint16_t * RESTRICT U3 = (uint16_t *)(void *)(((uint8_t *)U2) + r);
    uint16_t * RESTRICT U4 = (uint16_t *)(void *)(((uint8_t *)U3) + r);

    fast_uint_t i, p0 = *i0, p1 = *i1, p2 = *i2, p3 = *i3, p4 = *i4;

    for (i = 0; i != k; ++i)
    {
        uint16_t c0 = fastbits[p0 >> shift]; if (bucket2[c0] <= p0) { do { c0++; } while (bucket2[c0] <= p0); } p0 = P[p0]; U0[i] = libsais_bswap16(c0);
        uint16_t c1 = fastbits[p1 >> shift]; if (bucket2[c1] <= p1) { do { c1++; } while (bucket2[c1] <= p1); } p1 = P[p1]; U1[i] = libsais_bswap16(c1);
        uint16_t c2 = fastbits[p2 >> shift]; if (bucket2[c2] <= p2) { do { c2++; } while (bucket2[c2] <= p2); } p2 = P[p2]; U2[i] = libsais_bswap16(c2);
        uint16_t c3 = fastbits[p3 >> shift]; if (bucket2[c3] <= p3) { do { c3++; } while (bucket2[c3] <= p3); } p3 = P[p3]; U3[i] = libsais_bswap16(c3);
        uint16_t c4 = fastbits[p4 >> shift]; if (bucket2[c4] <= p4) { do { c4++; } while (bucket2[c4] <= p4); } p4 = P[p4]; U4[i] = libsais_bswap16(c4);
    }

    *i0 = p0; *i1 = p1; *i2 = p2; *i3 = p3; *i4 = p4;
}

static void libsais_unbwt_decode_6(uint8_t * RESTRICT U, sa_uint_t * RESTRICT P, sa_uint_t * RESTRICT bucket2, uint16_t * RESTRICT fastbits, fast_uint_t shift, fast_uint_t r, fast_uint_t * i0, fast_uint_t * i1, fast_uint_t * i2, fast_uint_t * i3, fast_uint_t * i4, fast_uint_t * i5, fast_uint_t k)
{
    uint16_t * RESTRICT U0 = (uint16_t *)(void *)U;
    uint16_t * RESTRICT U1 = (uint16_t *)(void *)(((uint8_t *)U0) + r);
    uint16_t * RESTRICT U2 = (uint16_t *)(void *)(((uint8_t *)U1) + r);
    uint16_t * RESTRICT U3 = (uint16_t *)(void *)(((uint8_t *)U2) + r);
    uint16_t * RESTRICT U4 = (uint16_t *)(void *)(((uint8_t *)U3) + r);
    uint16_t * RESTRICT U5 = (uint16_t *)(void *)(((uint8_t *)U4) + r);

    fast_uint_t i, p0 = *i0, p1 = *i1, p2 = *i2, p3 = *i3, p4 = *i4, p5 = *i5;

    for (i = 0; i != k; ++i)
    {
        uint16_t c0 = fastbits[p0 >> shift]; if (bucket2[c0] <= p0) { do { c0++; } while (bucket2[c0] <= p0); } p0 = P[p0]; U0[i] = libsais_bswap16(c0);
        uint16_t c1 = fastbits[p1 >> shift]; if (bucket2[c1] <= p1) { do { c1++; } while (bucket2[c1] <= p1); } p1 = P[p1]; U1[i] = libsais_bswap16(c1);
        uint16_t c2 = fastbits[p2 >> shift]; if (bucket2[c2] <= p2) { do { c2++; } while (bucket2[c2] <= p2); } p2 = P[p2]; U2[i] = libsais_bswap16(c2);
        uint16_t c3 = fastbits[p3 >> shift]; if (bucket2[c3] <= p3) { do { c3++; } while (bucket2[c3] <= p3); } p3 = P[p3]; U3[i] = libsais_bswap16(c3);
        uint16_t c4 = fastbits[p4 >> shift]; if (bucket2[c4] <= p4) { do { c4++; } while (bucket2[c4] <= p4); } p4 = P[p4]; U4[i] = libsais_bswap16(c4);
        uint16_t c5 = fastbits[p5 >> shift]; if (bucket2[c5] <= p5) { do { c5++; } while (bucket2[c5] <= p5); } p5 = P[p5]; U5[i] = libsais_bswap16(c5);
    }

    *i0 = p0; *i1 = p1; *i2 = p2; *i3 = p3; *i4 = p4; *i5 = p5;
}

static void libsais_unbwt_decode_7(uint8_t * RESTRICT U, sa_uint_t * RESTRICT P, sa_uint_t * RESTRICT bucket2, uint16_t * RESTRICT fastbits, fast_uint_t shift, fast_uint_t r, fast_uint_t * i0, fast_uint_t * i1, fast_uint_t * i2, fast_uint_t * i3, fast_uint_t * i4, fast_uint_t * i5, fast_uint_t * i6, fast_uint_t k)
{
    uint16_t * RESTRICT U0 = (uint16_t *)(void *)U;
    uint16_t * RESTRICT U1 = (uint16_t *)(void *)(((uint8_t *)U0) + r);
    uint16_t * RESTRICT U2 = (uint16_t *)(void *)(((uint8_t *)U1) + r);
    uint16_t * RESTRICT U3 = (uint16_t *)(void *)(((uint8_t *)U2) + r);
    uint16_t * RESTRICT U4 = (uint16_t *)(void *)(((uint8_t *)U3) + r);
    uint16_t * RESTRICT U5 = (uint16_t *)(void *)(((uint8_t *)U4) + r);
    uint16_t * RESTRICT U6 = (uint16_t *)(void *)(((uint8_t *)U5) + r);

    fast_uint_t i, p0 = *i0, p1 = *i1, p2 = *i2, p3 = *i3, p4 = *i4, p5 = *i5, p6 = *i6;

    for (i = 0; i != k; ++i)
    {
        uint16_t c0 = fastbits[p0 >> shift]; if (bucket2[c0] <= p0) { do { c0++; } while (bucket2[c0] <= p0); } p0 = P[p0]; U0[i] = libsais_bswap16(c0);
        uint16_t c1 = fastbits[p1 >> shift]; if (bucket2[c1] <= p1) { do { c1++; } while (bucket2[c1] <= p1); } p1 = P[p1]; U1[i] = libsais_bswap16(c1);
        uint16_t c2 = fastbits[p2 >> shift]; if (bucket2[c2] <= p2) { do { c2++; } while (bucket2[c2] <= p2); } p2 = P[p2]; U2[i] = libsais_bswap16(c2);
        uint16_t c3 = fastbits[p3 >> shift]; if (bucket2[c3] <= p3) { do { c3++; } while (bucket2[c3] <= p3); } p3 = P[p3]; U3[i] = libsais_bswap16(c3);
        uint16_t c4 = fastbits[p4 >> shift]; if (bucket2[c4] <= p4) { do { c4++; } while (bucket2[c4] <= p4); } p4 = P[p4]; U4[i] = libsais_bswap16(c4);
        uint16_t c5 = fastbits[p5 >> shift]; if (bucket2[c5] <= p5) { do { c5++; } while (bucket2[c5] <= p5); } p5 = P[p5]; U5[i] = libsais_bswap16(c5);
        uint16_t c6 = fastbits[p6 >> shift]; if (bucket2[c6] <= p6) { do { c6++; } while (bucket2[c6] <= p6); } p6 = P[p6]; U6[i] = libsais_bswap16(c6);
    }

    *i0 = p0; *i1 = p1; *i2 = p2; *i3 = p3; *i4 = p4; *i5 = p5; *i6 = p6;
}

static void libsais_unbwt_decode_8(uint8_t * RESTRICT U, sa_uint_t * RESTRICT P, sa_uint_t * RESTRICT bucket2, uint16_t * RESTRICT fastbits, fast_uint_t shift, fast_uint_t r, fast_uint_t * i0, fast_uint_t * i1, fast_uint_t * i2, fast_uint_t * i3, fast_uint_t * i4, fast_uint_t * i5, fast_uint_t * i6, fast_uint_t * i7, fast_uint_t k)
{
    uint16_t * RESTRICT U0 = (uint16_t *)(void *)U;
    uint16_t * RESTRICT U1 = (uint16_t *)(void *)(((uint8_t *)U0) + r);
    uint16_t * RESTRICT U2 = (uint16_t *)(void *)(((uint8_t *)U1) + r);
    uint16_t * RESTRICT U3 = (uint16_t *)(void *)(((uint8_t *)U2) + r);
    uint16_t * RESTRICT U4 = (uint16_t *)(void *)(((uint8_t *)U3) + r);
    uint16_t * RESTRICT U5 = (uint16_t *)(void *)(((uint8_t *)U4) + r);
    uint16_t * RESTRICT U6 = (uint16_t *)(void *)(((uint8_t *)U5) + r);
    uint16_t * RESTRICT U7 = (uint16_t *)(void *)(((uint8_t *)U6) + r);

    fast_uint_t i, p0 = *i0, p1 = *i1, p2 = *i2, p3 = *i3, p4 = *i4, p5 = *i5, p6 = *i6, p7 = *i7;

    for (i = 0; i != k; ++i)
    {
        uint16_t c0 = fastbits[p0 >> shift]; if (bucket2[c0] <= p0) { do { c0++; } while (bucket2[c0] <= p0); } p0 = P[p0]; U0[i] = libsais_bswap16(c0);
        uint16_t c1 = fastbits[p1 >> shift]; if (bucket2[c1] <= p1) { do { c1++; } while (bucket2[c1] <= p1); } p1 = P[p1]; U1[i] = libsais_bswap16(c1);
        uint16_t c2 = fastbits[p2 >> shift]; if (bucket2[c2] <= p2) { do { c2++; } while (bucket2[c2] <= p2); } p2 = P[p2]; U2[i] = libsais_bswap16(c2);
        uint16_t c3 = fastbits[p3 >> shift]; if (bucket2[c3] <= p3) { do { c3++; } while (bucket2[c3] <= p3); } p3 = P[p3]; U3[i] = libsais_bswap16(c3);
        uint16_t c4 = fastbits[p4 >> shift]; if (bucket2[c4] <= p4) { do { c4++; } while (bucket2[c4] <= p4); } p4 = P[p4]; U4[i] = libsais_bswap16(c4);
        uint16_t c5 = fastbits[p5 >> shift]; if (bucket2[c5] <= p5) { do { c5++; } while (bucket2[c5] <= p5); } p5 = P[p5]; U5[i] = libsais_bswap16(c5);
        uint16_t c6 = fastbits[p6 >> shift]; if (bucket2[c6] <= p6) { do { c6++; } while (bucket2[c6] <= p6); } p6 = P[p6]; U6[i] = libsais_bswap16(c6);
        uint16_t c7 = fastbits[p7 >> shift]; if (bucket2[c7] <= p7) { do { c7++; } while (bucket2[c7] <= p7); } p7 = P[p7]; U7[i] = libsais_bswap16(c7);
    }

    *i0 = p0; *i1 = p1; *i2 = p2; *i3 = p3; *i4 = p4; *i5 = p5; *i6 = p6; *i7 = p7;
}

static void libsais_unbwt_decode(uint8_t * RESTRICT U, sa_uint_t * RESTRICT P, sa_sint_t n, sa_sint_t r, const sa_uint_t * RESTRICT I, sa_uint_t * RESTRICT bucket2, uint16_t * RESTRICT fastbits, fast_sint_t blocks, fast_uint_t remainder)
{
    fast_uint_t shift       = 0; while ((n >> shift) > ((sa_sint_t)1 << UNBWT_FASTBITS)) { shift++; }
    fast_uint_t offset      = 0;

    while (blocks > 8)
    {
        fast_uint_t i0 = I[0], i1 = I[1], i2 = I[2], i3 = I[3], i4 = I[4], i5 = I[5], i6 = I[6], i7 = I[7];
        libsais_unbwt_decode_8(U + offset, P, bucket2, fastbits, shift, (fast_uint_t)r, &i0, &i1, &i2, &i3, &i4, &i5, &i6, &i7, (fast_uint_t)r >> 1);
        I += 8; blocks -= 8; offset += 8 * (fast_uint_t)r;
    }

    if (blocks == 1)
    {
        fast_uint_t i0 = I[0];
        libsais_unbwt_decode_1(U + offset, P, bucket2, fastbits, shift, &i0, remainder >> 1);
    }
    else if (blocks == 2)
    {
        fast_uint_t i0 = I[0], i1 = I[1];
        libsais_unbwt_decode_2(U + offset, P, bucket2, fastbits, shift, (fast_uint_t)r, &i0, &i1, remainder >> 1);
        libsais_unbwt_decode_1(U + offset + 2 * (remainder >> 1), P, bucket2, fastbits, shift, &i0, ((fast_uint_t)r >> 1) - (remainder >> 1));
    }
    else if (blocks == 3)
    {
        fast_uint_t i0 = I[0], i1 = I[1], i2 = I[2];
        libsais_unbwt_decode_3(U + offset, P, bucket2, fastbits, shift, (fast_uint_t)r, &i0, &i1, &i2, remainder >> 1);
        libsais_unbwt_decode_2(U + offset + 2 * (remainder >> 1), P, bucket2, fastbits, shift, (fast_uint_t)r, &i0, &i1, ((fast_uint_t)r >> 1) - (remainder >> 1));
    }
    else if (blocks == 4)
    {
        fast_uint_t i0 = I[0], i1 = I[1], i2 = I[2], i3 = I[3];
        libsais_unbwt_decode_4(U + offset, P, bucket2, fastbits, shift, (fast_uint_t)r, &i0, &i1, &i2, &i3, remainder >> 1);
        libsais_unbwt_decode_3(U + offset + 2 * (remainder >> 1), P, bucket2, fastbits, shift, (fast_uint_t)r, &i0, &i1, &i2, ((fast_uint_t)r >> 1) - (remainder >> 1));
    }
    else if (blocks == 5)
    {
        fast_uint_t i0 = I[0], i1 = I[1], i2 = I[2], i3 = I[3], i4 = I[4];
        libsais_unbwt_decode_5(U + offset, P, bucket2, fastbits, shift, (fast_uint_t)r, &i0, &i1, &i2, &i3, &i4, remainder >> 1);
        libsais_unbwt_decode_4(U + offset + 2 * (remainder >> 1), P, bucket2, fastbits, shift, (fast_uint_t)r, &i0, &i1, &i2, &i3, ((fast_uint_t)r >> 1) - (remainder >> 1));
    }
    else if (blocks == 6)
    {
        fast_uint_t i0 = I[0], i1 = I[1], i2 = I[2], i3 = I[3], i4 = I[4], i5 = I[5];
        libsais_unbwt_decode_6(U + offset, P, bucket2, fastbits, shift, (fast_uint_t)r, &i0, &i1, &i2, &i3, &i4, &i5, remainder >> 1);
        libsais_unbwt_decode_5(U + offset + 2 * (remainder >> 1), P, bucket2, fastbits, shift, (fast_uint_t)r, &i0, &i1, &i2, &i3, &i4, ((fast_uint_t)r >> 1) - (remainder >> 1));
    }
    else if (blocks == 7)
    {
        fast_uint_t i0 = I[0], i1 = I[1], i2 = I[2], i3 = I[3], i4 = I[4], i5 = I[5], i6 = I[6];
        libsais_unbwt_decode_7(U + offset, P, bucket2, fastbits, shift, (fast_uint_t)r, &i0, &i1, &i2, &i3, &i4, &i5, &i6, remainder >> 1);
        libsais_unbwt_decode_6(U + offset + 2 * (remainder >> 1), P, bucket2, fastbits, shift, (fast_uint_t)r, &i0, &i1, &i2, &i3, &i4, &i5, ((fast_uint_t)r >> 1) - (remainder >> 1));
    }
    else
    {
        fast_uint_t i0 = I[0], i1 = I[1], i2 = I[2], i3 = I[3], i4 = I[4], i5 = I[5], i6 = I[6], i7 = I[7];
        libsais_unbwt_decode_8(U + offset, P, bucket2, fastbits, shift, (fast_uint_t)r, &i0, &i1, &i2, &i3, &i4, &i5, &i6, &i7, remainder >> 1);
        libsais_unbwt_decode_7(U + offset + 2 * (remainder >> 1), P, bucket2, fastbits, shift, (fast_uint_t)r, &i0, &i1, &i2, &i3, &i4, &i5, &i6, ((fast_uint_t)r >> 1) - (remainder >> 1));
    }
}

static void libsais_unbwt_decode_omp(const uint8_t * RESTRICT T, uint8_t * RESTRICT U, sa_uint_t * RESTRICT P, sa_sint_t n, sa_sint_t r, const sa_uint_t * RESTRICT I, sa_uint_t * RESTRICT bucket2, uint16_t * RESTRICT fastbits, sa_sint_t threads)
{
    fast_uint_t lastc       = T[0];
    fast_sint_t blocks      = 1 + (((fast_sint_t)n - 1) / (fast_sint_t)r);
    fast_uint_t remainder   = (fast_uint_t)n - ((fast_uint_t)r * ((fast_uint_t)blocks - 1));

#if defined(LIBSAIS_OPENMP)
    fast_sint_t max_threads = blocks < threads ? blocks : threads;
    #pragma omp parallel num_threads(max_threads) if(max_threads > 1 && n >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num      = omp_get_thread_num();
        fast_sint_t omp_num_threads     = omp_get_num_threads();
#else
        UNUSED(threads);

        fast_sint_t omp_thread_num      = 0;
        fast_sint_t omp_num_threads     = 1;
#endif

        fast_sint_t omp_block_stride    = blocks / omp_num_threads;
        fast_sint_t omp_block_remainder = blocks % omp_num_threads;
        fast_sint_t omp_block_size      = omp_block_stride + (omp_thread_num < omp_block_remainder);
        fast_sint_t omp_block_start     = omp_block_stride * omp_thread_num + (omp_thread_num < omp_block_remainder ? omp_thread_num : omp_block_remainder);

        libsais_unbwt_decode(U + r * omp_block_start, P, n, r, I + omp_block_start, bucket2, fastbits, omp_block_size, omp_thread_num < omp_num_threads - 1 ? (fast_uint_t)r : remainder);
    }

    U[n - 1] = (uint8_t)lastc;
}

static sa_sint_t libsais_unbwt_core(const uint8_t * RESTRICT T, uint8_t * RESTRICT U, sa_uint_t * RESTRICT P, sa_sint_t n, const sa_sint_t * freq, sa_sint_t r, const sa_uint_t * RESTRICT I, sa_uint_t * RESTRICT bucket2, uint16_t * RESTRICT fastbits, sa_uint_t * RESTRICT buckets, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    if (threads > 1 && n >= 262144)
    {
        libsais_unbwt_init_parallel(T, P, n, freq, I, bucket2, fastbits, buckets, threads);
    }
    else
#else
    UNUSED(buckets);
#endif
    {
        libsais_unbwt_init_single(T, P, n, freq, I, bucket2, fastbits);
    }

    libsais_unbwt_decode_omp(T, U, P, n, r, I, bucket2, fastbits, threads);
    return 0;
}

static sa_sint_t libsais_unbwt_main(const uint8_t * T, uint8_t * U, sa_uint_t * P, sa_sint_t n, const sa_sint_t * freq, sa_sint_t r, const sa_uint_t * I, sa_sint_t threads)
{
    fast_uint_t shift = 0; while ((n >> shift) > ((sa_sint_t)1 << UNBWT_FASTBITS)) { shift++; }

    sa_uint_t *     RESTRICT bucket2        = (sa_uint_t *)libsais_alloc_aligned(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(sa_uint_t), 4096);
    uint16_t *      RESTRICT fastbits       = (uint16_t *)libsais_alloc_aligned(((size_t)1 + (size_t)(n >> shift)) * sizeof(uint16_t), 4096);
    sa_uint_t *     RESTRICT buckets        = threads > 1 && n >= 262144 ? (sa_uint_t *)libsais_alloc_aligned((size_t)threads * (ALPHABET_SIZE + (ALPHABET_SIZE * ALPHABET_SIZE)) * sizeof(sa_uint_t), 4096) : NULL;

    sa_sint_t index = bucket2 != NULL && fastbits != NULL && (buckets != NULL || threads == 1 || n < 262144)
        ? libsais_unbwt_core(T, U, P, n, freq, r, I, bucket2, fastbits, buckets, threads)
        : -2;

    libsais_free_aligned(buckets);
    libsais_free_aligned(fastbits);
    libsais_free_aligned(bucket2);

    return index;
}

static sa_sint_t libsais_unbwt_main_ctx(const LIBSAIS_UNBWT_CONTEXT * ctx, const uint8_t * T, uint8_t * U, sa_uint_t * P, sa_sint_t n, const sa_sint_t * freq, sa_sint_t r, const sa_uint_t * I)
{
    return ctx != NULL && ctx->bucket2 != NULL && ctx->fastbits != NULL && (ctx->buckets != NULL || ctx->threads == 1)
        ? libsais_unbwt_core(T, U, P, n, freq, r, I, ctx->bucket2, ctx->fastbits, ctx->buckets, (sa_sint_t)ctx->threads)
        : -2;
}

void * libsais_unbwt_create_ctx(void)
{
    return (void *)libsais_unbwt_create_ctx_main(1);
}

void libsais_unbwt_free_ctx(void * ctx)
{
    libsais_unbwt_free_ctx_main((LIBSAIS_UNBWT_CONTEXT *)ctx);
}

int32_t libsais_unbwt(const uint8_t * T, uint8_t * U, int32_t * A, int32_t n, const int32_t * freq, int32_t i)
{
    return libsais_unbwt_aux(T, U, A, n, freq, n, &i);
}

int32_t libsais_unbwt_ctx(const void * ctx, const uint8_t * T, uint8_t * U, int32_t * A, int32_t n, const int32_t * freq, int32_t i)
{
    return libsais_unbwt_aux_ctx(ctx, T, U, A, n, freq, n, &i);
}

int32_t libsais_unbwt_aux(const uint8_t * T, uint8_t * U, int32_t * A, int32_t n, const int32_t * freq, int32_t r, const int32_t * I)
{
    if ((T == NULL) || (U == NULL) || (A == NULL) || (n < 0) || ((r != n) && ((r < 2) || ((r & (r - 1)) != 0))) || (I == NULL))
    {
        return -1;
    }
    else if (n <= 1)
    {
        if (I[0] != n) { return -1; }
        if (n == 1) { U[0] = T[0]; }
        return 0;
    }

    fast_sint_t t; for (t = 0; t <= (n - 1) / r; ++t) { if (I[t] <= 0 || I[t] > n) { return -1; } }

    return libsais_unbwt_main(T, U, (sa_uint_t *)A, n, freq, r, (const sa_uint_t *)I, 1);
}

int32_t libsais_unbwt_aux_ctx(const void * ctx, const uint8_t * T, uint8_t * U, int32_t * A, int32_t n, const int32_t * freq, int32_t r, const int32_t * I)
{
    if ((T == NULL) || (U == NULL) || (A == NULL) || (n < 0) || ((r != n) && ((r < 2) || ((r & (r - 1)) != 0))) || (I == NULL))
    {
        return -1;
    }
    else if (n <= 1)
    {
        if (I[0] != n) { return -1; }
        if (n == 1) { U[0] = T[0]; }
        return 0;
    }

    fast_sint_t t; for (t = 0; t <= (n - 1) / r; ++t) { if (I[t] <= 0 || I[t] > n) { return -1; } }

    return libsais_unbwt_main_ctx((const LIBSAIS_UNBWT_CONTEXT *)ctx, T, U, (sa_uint_t *)A, n, freq, r, (const sa_uint_t *)I);
}

#if defined(LIBSAIS_OPENMP)

void * libsais_unbwt_create_ctx_omp(int32_t threads)
{
    if (threads < 0) { return NULL; }

    threads = threads > 0 ? threads : omp_get_max_threads();
    return (void *)libsais_unbwt_create_ctx_main(threads);
}

int32_t libsais_unbwt_omp(const uint8_t * T, uint8_t * U, int32_t * A, int32_t n, const int32_t * freq, int32_t i, int32_t threads)
{
    return libsais_unbwt_aux_omp(T, U, A, n, freq, n, &i, threads);
}

int32_t libsais_unbwt_aux_omp(const uint8_t * T, uint8_t * U, int32_t * A, int32_t n, const int32_t * freq, int32_t r, const int32_t * I, int32_t threads)
{
    if ((T == NULL) || (U == NULL) || (A == NULL) || (n < 0) || ((r != n) && ((r < 2) || ((r & (r - 1)) != 0))) || (I == NULL) || (threads < 0))
    {
        return -1;
    }
    else if (n <= 1)
    {
        if (I[0] != n) { return -1; }
        if (n == 1) { U[0] = T[0]; }
        return 0;
    }

    fast_sint_t t; for (t = 0; t <= (n - 1) / r; ++t) { if (I[t] <= 0 || I[t] > n) { return -1; } }

    threads = threads > 0 ? threads : omp_get_max_threads();
    return libsais_unbwt_main(T, U, (sa_uint_t *)A, n, freq, r, (const sa_uint_t *)I, threads);
}

#endif

static void libsais_compute_phi(const sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT PLCP, sa_sint_t n, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j; sa_sint_t k = omp_block_start > 0 ? SA[omp_block_start - 1] : n;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 3; i < j; i += 4)
    {
        libsais_prefetchr(&SA[i + 2 * prefetch_distance]);

        libsais_prefetchw(&PLCP[SA[i + prefetch_distance + 0]]);
        libsais_prefetchw(&PLCP[SA[i + prefetch_distance + 1]]);

        PLCP[SA[i + 0]] = k; k = SA[i + 0];
        PLCP[SA[i + 1]] = k; k = SA[i + 1];

        libsais_prefetchw(&PLCP[SA[i + prefetch_distance + 2]]);
        libsais_prefetchw(&PLCP[SA[i + prefetch_distance + 3]]);

        PLCP[SA[i + 2]] = k; k = SA[i + 2];
        PLCP[SA[i + 3]] = k; k = SA[i + 3];
    }

    for (j += prefetch_distance + 3; i < j; i += 1)
    {
        PLCP[SA[i]] = k; k = SA[i];
    }
}

static void libsais_compute_phi_omp(const sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT PLCP, sa_sint_t n, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (n / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : n - omp_block_start;

        libsais_compute_phi(SA, PLCP, n, omp_block_start, omp_block_size);
    }
}

static void libsais_compute_plcp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT PLCP, fast_sint_t n, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j, l = 0;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance; i < j; i += 1)
    {
        libsais_prefetchw(&PLCP[i + 2 * prefetch_distance]);
        libsais_prefetchr(&T[PLCP[i + prefetch_distance] + l]);

        fast_sint_t k = PLCP[i], m = n - (i > k ? i : k);
        while (l < m && T[i + l] == T[k + l]) { l++; }

        PLCP[i] = (sa_sint_t)l; l -= (l != 0);
    }

    for (j += prefetch_distance; i < j; i += 1)
    {
        fast_sint_t k = PLCP[i], m = n - (i > k ? i : k);
        while (l < m && T[i + l] == T[k + l]) { l++; }

        PLCP[i] = (sa_sint_t)l; l -= (l != 0);
    }
}

static void libsais_compute_plcp_omp(const uint8_t * RESTRICT T, sa_sint_t * RESTRICT PLCP, sa_sint_t n, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (n / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : n - omp_block_start;

        libsais_compute_plcp(T, PLCP, n, omp_block_start, omp_block_size);
    }
}

static void libsais_compute_plcp_int(const int32_t * RESTRICT T, sa_sint_t * RESTRICT PLCP, fast_sint_t n, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j, l = 0;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance; i < j; i += 1)
    {
        libsais_prefetchw(&PLCP[i + 2 * prefetch_distance]);
        libsais_prefetchr(&T[PLCP[i + prefetch_distance] + l]);

        fast_sint_t k = PLCP[i], m = n - (i > k ? i : k);
        while (l < m && T[i + l] == T[k + l]) { l++; }

        PLCP[i] = (sa_sint_t)l; l -= (l != 0);
    }

    for (j += prefetch_distance; i < j; i += 1)
    {
        fast_sint_t k = PLCP[i], m = n - (i > k ? i : k);
        while (l < m && T[i + l] == T[k + l]) { l++; }

        PLCP[i] = (sa_sint_t)l; l -= (l != 0);
    }
}

static void libsais_compute_plcp_int_omp(const int32_t * RESTRICT T, sa_sint_t * RESTRICT PLCP, sa_sint_t n, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (n / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : n - omp_block_start;

        libsais_compute_plcp_int(T, PLCP, n, omp_block_start, omp_block_size);
    }
}

static void libsais_compute_lcp(const sa_sint_t * RESTRICT PLCP, const sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT LCP, fast_sint_t omp_block_start, fast_sint_t omp_block_size)
{
    const fast_sint_t prefetch_distance = 32;

    fast_sint_t i, j;
    for (i = omp_block_start, j = omp_block_start + omp_block_size - prefetch_distance - 3; i < j; i += 4)
    {
        libsais_prefetchr(&SA[i + 2 * prefetch_distance]);
        libsais_prefetchw(&LCP[i + prefetch_distance]);

        libsais_prefetchr(&PLCP[SA[i + prefetch_distance + 0]]);
        libsais_prefetchr(&PLCP[SA[i + prefetch_distance + 1]]);

        LCP[i + 0] = PLCP[SA[i + 0]];
        LCP[i + 1] = PLCP[SA[i + 1]];

        libsais_prefetchr(&PLCP[SA[i + prefetch_distance + 2]]);
        libsais_prefetchr(&PLCP[SA[i + prefetch_distance + 3]]);

        LCP[i + 2] = PLCP[SA[i + 2]];
        LCP[i + 3] = PLCP[SA[i + 3]];
    }

    for (j += prefetch_distance + 3; i < j; i += 1)
    {
        LCP[i] = PLCP[SA[i]];
    }
}

static void libsais_compute_lcp_omp(const sa_sint_t * RESTRICT PLCP, const sa_sint_t * RESTRICT SA, sa_sint_t * RESTRICT LCP, sa_sint_t n, sa_sint_t threads)
{
#if defined(LIBSAIS_OPENMP)
    #pragma omp parallel num_threads(threads) if(threads > 1 && n >= 65536)
#endif
    {
#if defined(LIBSAIS_OPENMP)
        fast_sint_t omp_thread_num    = omp_get_thread_num();
        fast_sint_t omp_num_threads   = omp_get_num_threads();
#else
        UNUSED(threads);

        fast_sint_t omp_thread_num    = 0;
        fast_sint_t omp_num_threads   = 1;
#endif
        fast_sint_t omp_block_stride  = (n / omp_num_threads) & (-16);
        fast_sint_t omp_block_start   = omp_thread_num * omp_block_stride;
        fast_sint_t omp_block_size    = omp_thread_num < omp_num_threads - 1 ? omp_block_stride : n - omp_block_start;

        libsais_compute_lcp(PLCP, SA, LCP, omp_block_start, omp_block_size);
    }
}

int32_t libsais_plcp(const uint8_t * T, const int32_t * SA, int32_t * PLCP, int32_t n)
{
    if ((T == NULL) || (SA == NULL) || (PLCP == NULL) || (n < 0))
    {
        return -1;
    }
    else if (n <= 1)
    {
        if (n == 1) { PLCP[0] = 0; }
        return 0;
    }

    libsais_compute_phi_omp(SA, PLCP, n, 1);
    libsais_compute_plcp_omp(T, PLCP, n, 1);

    return 0;
}

int32_t libsais_plcp_int(const int32_t * T, const int32_t * SA, int32_t * PLCP, int32_t n)
{
    if ((T == NULL) || (SA == NULL) || (PLCP == NULL) || (n < 0))
    {
        return -1;
    }
    else if (n <= 1)
    {
        if (n == 1) { PLCP[0] = 0; }
        return 0;
    }

    libsais_compute_phi_omp(SA, PLCP, n, 1);
    libsais_compute_plcp_int_omp(T, PLCP, n, 1);

    return 0;
}

int32_t libsais_lcp(const int32_t * PLCP, const int32_t * SA, int32_t * LCP, int32_t n)
{
    if ((PLCP == NULL) || (SA == NULL) || (LCP == NULL) || (n < 0))
    {
        return -1;
    }
    else if (n <= 1)
    {
        if (n == 1) { LCP[0] = PLCP[SA[0]]; }
        return 0;
    }

    libsais_compute_lcp_omp(PLCP, SA, LCP, n, 1);

    return 0;
}

#if defined(LIBSAIS_OPENMP)

int32_t libsais_plcp_omp(const uint8_t * T, const int32_t * SA, int32_t * PLCP, int32_t n, int32_t threads)
{
    if ((T == NULL) || (SA == NULL) || (PLCP == NULL) || (n < 0) || (threads < 0))
    {
        return -1;
    }
    else if (n <= 1)
    {
        if (n == 1) { PLCP[0] = 0; }
        return 0;
    }
    
    threads = threads > 0 ? threads : omp_get_max_threads();

    libsais_compute_phi_omp(SA, PLCP, n, threads);
    libsais_compute_plcp_omp(T, PLCP, n, threads);

    return 0;
}

int32_t libsais_plcp_int_omp(const int32_t * T, const int32_t * SA, int32_t * PLCP, int32_t n, int32_t threads)
{
    if ((T == NULL) || (SA == NULL) || (PLCP == NULL) || (n < 0) || (threads < 0))
    {
        return -1;
    }
    else if (n <= 1)
    {
        if (n == 1) { PLCP[0] = 0; }
        return 0;
    }
    
    threads = threads > 0 ? threads : omp_get_max_threads();

    libsais_compute_phi_omp(SA, PLCP, n, threads);
    libsais_compute_plcp_int_omp(T, PLCP, n, threads);

    return 0;
}

int32_t libsais_lcp_omp(const int32_t * PLCP, const int32_t * SA, int32_t * LCP, int32_t n, int32_t threads)
{
    if ((PLCP == NULL) || (SA == NULL) || (LCP == NULL) || (n < 0) || (threads < 0))
    {
        return -1;
    }
    else if (n <= 1)
    {
        if (n == 1) { LCP[0] = PLCP[SA[0]]; }
        return 0;
    }

    threads = threads > 0 ? threads : omp_get_max_threads();

    libsais_compute_lcp_omp(PLCP, SA, LCP, n, threads);

    return 0;
}

#endif

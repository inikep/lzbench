/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Quantized Local Frequency Coding functions                */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

   Copyright (c) 2009-2024 Ilya Grebnov <ilya.grebnov@gmail.com>

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

#include <stddef.h>
#include <stdlib.h>
#include <memory.h>

#include "qlfc.h"

#include "../../libbsc.h"
#include "../../platform/platform.h"

#include "../common/rangecoder.h"
#include "../common/tables.h"
#include "../common/predictor.h"

#include "qlfc_model.h"

#if defined(LIBBSC_DYNAMIC_CPU_DISPATCH)
    unsigned char * bsc_qlfc_transform(const unsigned char * RESTRICT input, unsigned char * RESTRICT buffer, int n, unsigned char * RESTRICT MTFTable);
    unsigned char * bsc_qlfc_transform_avx2(const unsigned char * RESTRICT input, unsigned char * RESTRICT buffer, int n, unsigned char * RESTRICT MTFTable);
    unsigned char * bsc_qlfc_transform_avx(const unsigned char * RESTRICT input, unsigned char * RESTRICT buffer, int n, unsigned char * RESTRICT MTFTable);
    unsigned char * bsc_qlfc_transform_sse2(const unsigned char * RESTRICT input, unsigned char * RESTRICT buffer, int n, unsigned char * RESTRICT MTFTable);

    int bsc_qlfc_adaptive_encode(const unsigned char * input, unsigned char * output, unsigned char * buffer, int inputSize, int outputSize, QlfcStatisticalModel1 * model);
    int bsc_qlfc_adaptive_encode_avx2(const unsigned char * input, unsigned char * output, unsigned char * buffer, int inputSize, int outputSize, QlfcStatisticalModel1 * model);
    int bsc_qlfc_adaptive_encode_sse2(const unsigned char * input, unsigned char * output, unsigned char * buffer, int inputSize, int outputSize, QlfcStatisticalModel1 * model);

    int bsc_qlfc_static_encode(const unsigned char * input, unsigned char * output, unsigned char * buffer, int inputSize, int outputSize, QlfcStatisticalModel1 * model);
    int bsc_qlfc_static_encode_avx2(const unsigned char * input, unsigned char * output, unsigned char * buffer, int inputSize, int outputSize, QlfcStatisticalModel1 * model);
    int bsc_qlfc_static_encode_sse2(const unsigned char * input, unsigned char * output, unsigned char * buffer, int inputSize, int outputSize, QlfcStatisticalModel1 * model);

    int bsc_qlfc_fast_encode(const unsigned char * input, unsigned char * output, unsigned char * buffer, int inputSize, int outputSize, QlfcStatisticalModel2 * model);
    int bsc_qlfc_fast_encode_avx2(const unsigned char * input, unsigned char * output, unsigned char * buffer, int inputSize, int outputSize, QlfcStatisticalModel2 * model);
    int bsc_qlfc_fast_encode_sse2(const unsigned char * input, unsigned char * output, unsigned char * buffer, int inputSize, int outputSize, QlfcStatisticalModel2 * model);

    int bsc_qlfc_adaptive_decode(const unsigned char * input, unsigned char * output, QlfcStatisticalModel1 * model);
    int bsc_qlfc_adaptive_decode_avx(const unsigned char * input, unsigned char * output, QlfcStatisticalModel1 * model);
    int bsc_qlfc_adaptive_decode_sse41(const unsigned char * input, unsigned char * output, QlfcStatisticalModel1 * model);
    int bsc_qlfc_adaptive_decode_sse2(const unsigned char * input, unsigned char * output, QlfcStatisticalModel1 * model);

    int bsc_qlfc_static_decode(const unsigned char * input, unsigned char * output, QlfcStatisticalModel1 * model);
    int bsc_qlfc_static_decode_avx(const unsigned char * input, unsigned char * output, QlfcStatisticalModel1 * model);
    int bsc_qlfc_static_decode_sse41(const unsigned char * input, unsigned char * output, QlfcStatisticalModel1 * model);
    int bsc_qlfc_static_decode_sse2(const unsigned char * input, unsigned char * output, QlfcStatisticalModel1 * model);

    int bsc_qlfc_fast_decode(const unsigned char * input, unsigned char * output, QlfcStatisticalModel2 * model);
    int bsc_qlfc_fast_decode_avx(const unsigned char * input, unsigned char * output, QlfcStatisticalModel2 * model);
    int bsc_qlfc_fast_decode_sse41(const unsigned char * input, unsigned char * output, QlfcStatisticalModel2 * model);
    int bsc_qlfc_fast_decode_sse2(const unsigned char * input, unsigned char * output, QlfcStatisticalModel2 * model);

    #if LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_SSE2
        int bsc_qlfc_adaptive_encode(const unsigned char * input, unsigned char * output, unsigned char * buffer, int inputSize, int outputSize, QlfcStatisticalModel1 * model)
        {
            if (bsc_get_cpu_features() >= LIBBSC_CPU_FEATURE_AVX2) { return bsc_qlfc_adaptive_encode_avx2(input, output, buffer, inputSize, outputSize, model); }

            return bsc_qlfc_adaptive_encode_sse2(input, output, buffer, inputSize, outputSize, model);
        }

        int bsc_qlfc_static_encode(const unsigned char * input, unsigned char * output, unsigned char * buffer, int inputSize, int outputSize, QlfcStatisticalModel1 * model)
        {
            if (bsc_get_cpu_features() >= LIBBSC_CPU_FEATURE_AVX2) { return bsc_qlfc_static_encode_avx2(input, output, buffer, inputSize, outputSize, model); }

            return bsc_qlfc_static_encode_sse2(input, output, buffer, inputSize, outputSize, model);
        }

        int bsc_qlfc_fast_encode(const unsigned char * input, unsigned char * output, unsigned char * buffer, int inputSize, int outputSize, QlfcStatisticalModel2 * model)
        {
            if (bsc_get_cpu_features() >= LIBBSC_CPU_FEATURE_AVX2) { return bsc_qlfc_fast_encode_avx2(input, output, buffer, inputSize, outputSize, model); }

            return bsc_qlfc_fast_encode_sse2(input, output, buffer, inputSize, outputSize, model);
        }

        unsigned char * bsc_qlfc_transform(const unsigned char * input, unsigned char * buffer, int n, unsigned char * MTFTable)
        {
            if (bsc_get_cpu_features() >= LIBBSC_CPU_FEATURE_AVX2) { return bsc_qlfc_transform_avx2(input, buffer, n, MTFTable); }
            if (bsc_get_cpu_features() >= LIBBSC_CPU_FEATURE_AVX)  { return bsc_qlfc_transform_avx (input, buffer, n, MTFTable); }

            return bsc_qlfc_transform_sse2(input, buffer, n, MTFTable);
        }

        int bsc_qlfc_adaptive_decode(const unsigned char * input, unsigned char * output, QlfcStatisticalModel1 * model)
        {
            if (bsc_get_cpu_features() >= LIBBSC_CPU_FEATURE_AVX)   { return bsc_qlfc_adaptive_decode_avx  (input, output, model); }
            if (bsc_get_cpu_features() >= LIBBSC_CPU_FEATURE_SSE41) { return bsc_qlfc_adaptive_decode_sse41(input, output, model); }

            return bsc_qlfc_adaptive_decode_sse2(input, output, model);
        }

        int bsc_qlfc_static_decode(const unsigned char * input, unsigned char * output, QlfcStatisticalModel1 * model)
        {
            if (bsc_get_cpu_features() >= LIBBSC_CPU_FEATURE_AVX)   { return bsc_qlfc_static_decode_avx  (input, output, model); }
            if (bsc_get_cpu_features() >= LIBBSC_CPU_FEATURE_SSE41) { return bsc_qlfc_static_decode_sse41(input, output, model); }

            return bsc_qlfc_static_decode_sse2(input, output, model);
        }

        int bsc_qlfc_fast_decode(const unsigned char * input, unsigned char * output, QlfcStatisticalModel2 * model)
        {
            if (bsc_get_cpu_features() >= LIBBSC_CPU_FEATURE_AVX)   { return bsc_qlfc_fast_decode_avx  (input, output, model); }
            if (bsc_get_cpu_features() >= LIBBSC_CPU_FEATURE_SSE41) { return bsc_qlfc_fast_decode_sse41(input, output, model); }

            return bsc_qlfc_fast_decode_sse2(input, output, model);
        }
    #endif

    #if LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_AVX2
        #define QLFC_TRANSFORM_FUNCTION_NAME       bsc_qlfc_transform_avx2
        #define QLFC_TRANSFORM_SCAN_FUNCTION_NAME  bsc_qlfc_transform_scan_avx2
        #define QLFC_ADAPTIVE_ENCODE_FUNCTION_NAME bsc_qlfc_adaptive_encode_avx2
        #define QLFC_STATIC_ENCODE_FUNCTION_NAME   bsc_qlfc_static_encode_avx2
        #define QLFC_FAST_ENCODE_FUNCTION_NAME     bsc_qlfc_fast_encode_avx2
    #elif LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_AVX
        #define QLFC_TRANSFORM_FUNCTION_NAME       bsc_qlfc_transform_avx
        #define QLFC_TRANSFORM_SCAN_FUNCTION_NAME  bsc_qlfc_transform_scan_avx
        #define QLFC_ADAPTIVE_DECODE_FUNCTION_NAME bsc_qlfc_adaptive_decode_avx
        #define QLFC_STATIC_DECODE_FUNCTION_NAME   bsc_qlfc_static_decode_avx
        #define QLFC_FAST_DECODE_FUNCTION_NAME     bsc_qlfc_fast_decode_avx
    #elif LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_SSE41
        #define QLFC_ADAPTIVE_DECODE_FUNCTION_NAME bsc_qlfc_adaptive_decode_sse41
        #define QLFC_STATIC_DECODE_FUNCTION_NAME   bsc_qlfc_static_decode_sse41
        #define QLFC_FAST_DECODE_FUNCTION_NAME     bsc_qlfc_fast_decode_sse41
    #elif LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_SSE2
        #define QLFC_TRANSFORM_FUNCTION_NAME       bsc_qlfc_transform_sse2
        #define QLFC_TRANSFORM_SCAN_FUNCTION_NAME  bsc_qlfc_transform_scan_sse2
        #define QLFC_ADAPTIVE_ENCODE_FUNCTION_NAME bsc_qlfc_adaptive_encode_sse2
        #define QLFC_STATIC_ENCODE_FUNCTION_NAME   bsc_qlfc_static_encode_sse2
        #define QLFC_FAST_ENCODE_FUNCTION_NAME     bsc_qlfc_fast_encode_sse2
        #define QLFC_ADAPTIVE_DECODE_FUNCTION_NAME bsc_qlfc_adaptive_decode_sse2
        #define QLFC_STATIC_DECODE_FUNCTION_NAME   bsc_qlfc_static_decode_sse2
        #define QLFC_FAST_DECODE_FUNCTION_NAME     bsc_qlfc_fast_decode_sse2
    #endif
#else
    #define QLFC_TRANSFORM_FUNCTION_NAME       bsc_qlfc_transform
    #define QLFC_TRANSFORM_SCAN_FUNCTION_NAME  bsc_qlfc_transform_scan
    #define QLFC_ADAPTIVE_ENCODE_FUNCTION_NAME bsc_qlfc_adaptive_encode
    #define QLFC_STATIC_ENCODE_FUNCTION_NAME   bsc_qlfc_static_encode
    #define QLFC_FAST_ENCODE_FUNCTION_NAME     bsc_qlfc_fast_encode
    #define QLFC_ADAPTIVE_DECODE_FUNCTION_NAME bsc_qlfc_adaptive_decode
    #define QLFC_STATIC_DECODE_FUNCTION_NAME   bsc_qlfc_static_decode
    #define QLFC_FAST_DECODE_FUNCTION_NAME     bsc_qlfc_fast_decode
#endif

#if defined(QLFC_TRANSFORM_FUNCTION_NAME)

#if LIBBSC_CPU_FEATURE >= LIBBSC_CPU_FEATURE_SSE2

INLINE ptrdiff_t QLFC_TRANSFORM_SCAN_FUNCTION_NAME (const unsigned char * RESTRICT input, ptrdiff_t i, unsigned char currentChar)
{
#if LIBBSC_CPU_FEATURE >= LIBBSC_CPU_FEATURE_AVX2
    __m256i v = _mm256_set1_epi8(currentChar);

    while (i >= 32)
    {
        i -= 32; int m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)(input + i)), v));
        if (m != (int)0xffffffff) { return i + bsc_bit_scan_reverse(((unsigned int)(~m))); }
    }
#elif LIBBSC_CPU_FEATURE >= LIBBSC_CPU_FEATURE_SSE2
    __m128i v = _mm_set1_epi8(currentChar);

    while (i >= 16)
    {
        i -= 16; int m = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(input + i)), v));
        if (m != 0xffff) { return i + bsc_bit_scan_reverse((unsigned int)(m ^ 0xffff)); }
    }
#endif

    do {} while ((--i >= 0) && (input[i] == currentChar)); return i;
}

unsigned char * QLFC_TRANSFORM_FUNCTION_NAME (const unsigned char * RESTRICT input, unsigned char * RESTRICT buffer, int n, unsigned char * RESTRICT MTFTable)
{
    signed char ALIGNED(64) ranks[ALPHABET_SIZE];
    signed char ALIGNED(64) flags[ALPHABET_SIZE];

    for (ptrdiff_t i = 0; i < ALPHABET_SIZE; ++i) { ranks[i] = (signed char)(i - 128); }
    for (ptrdiff_t i = 0; i < ALPHABET_SIZE; ++i) { flags[i] = 0; }

    ptrdiff_t i = (ptrdiff_t)n - 1, j = n; signed char nSymbols = 0;

    for (; i >= 0; )
    {
        unsigned char currentChar1 = input[i]; i = QLFC_TRANSFORM_SCAN_FUNCTION_NAME(input, i, currentChar1); if (i < 0) { i = 0; break; }
        unsigned char currentChar2 = input[i]; i = QLFC_TRANSFORM_SCAN_FUNCTION_NAME(input, i, currentChar2);

        signed char rank1 = ranks[currentChar1], rank2 = ranks[currentChar2]; rank2 += rank1 > rank2;

        buffer[--j] = rank1 + 128; if (flags[currentChar1] == 0) { flags[currentChar1] = 1; buffer[j] = nSymbols++; }
        buffer[--j] = rank2 + 128; if (flags[currentChar2] == 0) { flags[currentChar2] = 1; buffer[j] = nSymbols++; }

        for (int t = 0 * 32; t < 1 * 32; ++t) { ranks[t] -= (rank1 > ranks[t] ? (signed char)-1 : (signed char)0) + (rank2 > ranks[t] ? (signed char)-1 : (signed char)0); }
        for (int t = 1 * 32; t < 2 * 32; ++t) { ranks[t] -= (rank1 > ranks[t] ? (signed char)-1 : (signed char)0) + (rank2 > ranks[t] ? (signed char)-1 : (signed char)0); }
        for (int t = 2 * 32; t < 3 * 32; ++t) { ranks[t] -= (rank1 > ranks[t] ? (signed char)-1 : (signed char)0) + (rank2 > ranks[t] ? (signed char)-1 : (signed char)0); }
        for (int t = 3 * 32; t < 4 * 32; ++t) { ranks[t] -= (rank1 > ranks[t] ? (signed char)-1 : (signed char)0) + (rank2 > ranks[t] ? (signed char)-1 : (signed char)0); }
        for (int t = 4 * 32; t < 5 * 32; ++t) { ranks[t] -= (rank1 > ranks[t] ? (signed char)-1 : (signed char)0) + (rank2 > ranks[t] ? (signed char)-1 : (signed char)0); }
        for (int t = 5 * 32; t < 6 * 32; ++t) { ranks[t] -= (rank1 > ranks[t] ? (signed char)-1 : (signed char)0) + (rank2 > ranks[t] ? (signed char)-1 : (signed char)0); }
        for (int t = 6 * 32; t < 7 * 32; ++t) { ranks[t] -= (rank1 > ranks[t] ? (signed char)-1 : (signed char)0) + (rank2 > ranks[t] ? (signed char)-1 : (signed char)0); }
        for (int t = 7 * 32; t < 8 * 32; ++t) { ranks[t] -= (rank1 > ranks[t] ? (signed char)-1 : (signed char)0) + (rank2 > ranks[t] ? (signed char)-1 : (signed char)0); }
        
        ranks[currentChar1] = -127; ranks[currentChar2] = -128;
    }

    if (i >= 0)
    {
        unsigned char currentChar = input[0]; signed char rank = ranks[currentChar];

        buffer[--j] = rank + 128; if (flags[currentChar] == 0) { flags[currentChar] = 1; buffer[j] = nSymbols++; }

        for (int t = 0 * 32; t < 1 * 32; ++t) { ranks[t] -= (ranks[t] < rank ? -1 : 0); }
        for (int t = 1 * 32; t < 2 * 32; ++t) { ranks[t] -= (ranks[t] < rank ? -1 : 0); }
        for (int t = 2 * 32; t < 3 * 32; ++t) { ranks[t] -= (ranks[t] < rank ? -1 : 0); }
        for (int t = 3 * 32; t < 4 * 32; ++t) { ranks[t] -= (ranks[t] < rank ? -1 : 0); }
        for (int t = 4 * 32; t < 5 * 32; ++t) { ranks[t] -= (ranks[t] < rank ? -1 : 0); }
        for (int t = 5 * 32; t < 6 * 32; ++t) { ranks[t] -= (ranks[t] < rank ? -1 : 0); }
        for (int t = 6 * 32; t < 7 * 32; ++t) { ranks[t] -= (ranks[t] < rank ? -1 : 0); }
        for (int t = 7 * 32; t < 8 * 32; ++t) { ranks[t] -= (ranks[t] < rank ? -1 : 0); }
        ranks[currentChar] = -128;
    }

    buffer[n - 1] = 1;

    for (ptrdiff_t i = 0; i < ALPHABET_SIZE; ++i) { MTFTable[ranks[i] + 128] = (unsigned char)i; }
    for (ptrdiff_t i = 1; i < ALPHABET_SIZE; ++i) { if (flags[MTFTable[i]] == 0) { MTFTable[i] = MTFTable[i - 1]; break; } }

    return buffer + j;
}

#elif LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_A64

INLINE ptrdiff_t QLFC_TRANSFORM_SCAN_FUNCTION_NAME (const unsigned char * RESTRICT input, ptrdiff_t i, unsigned long long currentChar)
{
    unsigned long long v = currentChar; v |= (v << 8); v |= (v << 16); v |= (v << 32);

    while (i >= 8)
    {
        i -= 8; unsigned long long m = (*(unsigned long long const *)(input + i)) ^ v;
        if (m != 0) { return i + (bsc_bit_scan_reverse64(m) / 8); }
    }

    do {} while ((--i >= 0) && (input[i] == currentChar)); return i;
}

unsigned char * QLFC_TRANSFORM_FUNCTION_NAME (const unsigned char * RESTRICT input, unsigned char * RESTRICT buffer, int n, unsigned char * RESTRICT MTFTable)
{
    signed char ALIGNED(64) ranks[ALPHABET_SIZE];
    signed char ALIGNED(64) flags[ALPHABET_SIZE];

    for (ptrdiff_t i = 0; i < ALPHABET_SIZE; ++i) { ranks[i] = (signed char)(i - 128); }
    for (ptrdiff_t i = 0; i < ALPHABET_SIZE; ++i) { flags[i] = 0; }

    ptrdiff_t i = (ptrdiff_t)n - 1, j = n; signed char nSymbols = 0;

    for (; i >= 0;)
    {
        unsigned char currentChar1 = input[i]; i = QLFC_TRANSFORM_SCAN_FUNCTION_NAME(input, i, currentChar1); if (i < 0) { i = 0; break; }
        unsigned char currentChar2 = input[i]; i = QLFC_TRANSFORM_SCAN_FUNCTION_NAME(input, i, currentChar2);

        signed char rank1 = ranks[currentChar1], rank2 = ranks[currentChar2]; rank2 += rank1 > rank2;

        buffer[--j] = rank1 + 128; if (flags[currentChar1] == 0) { flags[currentChar1] = 1; buffer[j] = nSymbols++; }
        buffer[--j] = rank2 + 128; if (flags[currentChar2] == 0) { flags[currentChar2] = 1; buffer[j] = nSymbols++; }

        int8x16_t r1 = vdupq_n_s8(rank1), r2 = vdupq_n_s8(rank2), x, y;

        x = vld1q_s8((int8_t const *)(ranks + 16 * 0)); y = vld1q_s8((int8_t const *)(ranks + 16 * 1));
        x = vsubq_s8(vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r1, x))), vreinterpretq_s8_u8(vcgtq_s8(r2, x)));
        y = vsubq_s8(vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r1, y))), vreinterpretq_s8_u8(vcgtq_s8(r2, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 0), x); vst1q_s8((int8_t *)(ranks + 16 * 1), y);

        x = vld1q_s8((int8_t const *)(ranks + 16 * 2)); y = vld1q_s8((int8_t const *)(ranks + 16 * 3));
        x = vsubq_s8(vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r1, x))), vreinterpretq_s8_u8(vcgtq_s8(r2, x)));
        y = vsubq_s8(vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r1, y))), vreinterpretq_s8_u8(vcgtq_s8(r2, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 2), x); vst1q_s8((int8_t *)(ranks + 16 * 3), y);

        x = vld1q_s8((int8_t const *)(ranks + 16 * 4)); y = vld1q_s8((int8_t const *)(ranks + 16 * 5));
        x = vsubq_s8(vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r1, x))), vreinterpretq_s8_u8(vcgtq_s8(r2, x)));
        y = vsubq_s8(vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r1, y))), vreinterpretq_s8_u8(vcgtq_s8(r2, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 4), x); vst1q_s8((int8_t *)(ranks + 16 * 5), y);

        x = vld1q_s8((int8_t const *)(ranks + 16 * 6)); y = vld1q_s8((int8_t const *)(ranks + 16 * 7));
        x = vsubq_s8(vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r1, x))), vreinterpretq_s8_u8(vcgtq_s8(r2, x)));
        y = vsubq_s8(vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r1, y))), vreinterpretq_s8_u8(vcgtq_s8(r2, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 6), x); vst1q_s8((int8_t *)(ranks + 16 * 7), y);      
       
        x = vld1q_s8((int8_t const *)(ranks + 16 * 8)); y = vld1q_s8((int8_t const *)(ranks + 16 * 9));
        x = vsubq_s8(vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r1, x))), vreinterpretq_s8_u8(vcgtq_s8(r2, x)));
        y = vsubq_s8(vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r1, y))), vreinterpretq_s8_u8(vcgtq_s8(r2, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 8), x); vst1q_s8((int8_t *)(ranks + 16 * 9), y);

        x = vld1q_s8((int8_t const *)(ranks + 16 * 10)); y = vld1q_s8((int8_t const *)(ranks + 16 * 11));
        x = vsubq_s8(vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r1, x))), vreinterpretq_s8_u8(vcgtq_s8(r2, x)));
        y = vsubq_s8(vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r1, y))), vreinterpretq_s8_u8(vcgtq_s8(r2, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 10), x); vst1q_s8((int8_t *)(ranks + 16 * 11), y);

        x = vld1q_s8((int8_t const *)(ranks + 16 * 12)); y = vld1q_s8((int8_t const *)(ranks + 16 * 13));
        x = vsubq_s8(vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r1, x))), vreinterpretq_s8_u8(vcgtq_s8(r2, x)));
        y = vsubq_s8(vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r1, y))), vreinterpretq_s8_u8(vcgtq_s8(r2, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 12), x); vst1q_s8((int8_t *)(ranks + 16 * 13), y);

        x = vld1q_s8((int8_t const *)(ranks + 16 * 14)); y = vld1q_s8((int8_t const *)(ranks + 16 * 15));
        x = vsubq_s8(vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r1, x))), vreinterpretq_s8_u8(vcgtq_s8(r2, x)));
        y = vsubq_s8(vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r1, y))), vreinterpretq_s8_u8(vcgtq_s8(r2, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 14), x); vst1q_s8((int8_t *)(ranks + 16 * 15), y);

        ranks[currentChar1] = -127; ranks[currentChar2] = -128;
    }

    if (i >= 0)
    {
        unsigned char currentChar = input[0]; signed char rank = ranks[currentChar];

        buffer[--j] = rank + 128; if (flags[currentChar] == 0) { flags[currentChar] = 1; buffer[j] = nSymbols++; }

        int8x16_t r = vdupq_n_s8(rank), x, y;

        x = vld1q_s8((int8_t const *)(ranks + 16 * 0)); y = vld1q_s8((int8_t const *)(ranks + 16 * 1));
        x = vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r, x)));
        y = vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 0), x); vst1q_s8((int8_t *)(ranks + 16 * 1), y);

        x = vld1q_s8((int8_t const *)(ranks + 16 * 2)); y = vld1q_s8((int8_t const *)(ranks + 16 * 3));
        x = vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r, x)));
        y = vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 2), x); vst1q_s8((int8_t *)(ranks + 16 * 3), y);

        x = vld1q_s8((int8_t const *)(ranks + 16 * 4)); y = vld1q_s8((int8_t const *)(ranks + 16 * 5));
        x = vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r, x)));
        y = vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 4), x); vst1q_s8((int8_t *)(ranks + 16 * 5), y);

        x = vld1q_s8((int8_t const *)(ranks + 16 * 6)); y = vld1q_s8((int8_t const *)(ranks + 16 * 7));
        x = vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r, x)));
        y = vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 6), x); vst1q_s8((int8_t *)(ranks + 16 * 7), y);
       
        x = vld1q_s8((int8_t const *)(ranks + 16 * 8)); y = vld1q_s8((int8_t const *)(ranks + 16 * 9));
        x = vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r, x)));
        y = vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 8), x); vst1q_s8((int8_t *)(ranks + 16 * 9), y);

        x = vld1q_s8((int8_t const *)(ranks + 16 * 10)); y = vld1q_s8((int8_t const *)(ranks + 16 * 11));
        x = vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r, x)));
        y = vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 10), x); vst1q_s8((int8_t *)(ranks + 16 * 11), y);

        x = vld1q_s8((int8_t const *)(ranks + 16 * 12)); y = vld1q_s8((int8_t const *)(ranks + 16 * 13));
        x = vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r, x)));
        y = vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 12), x); vst1q_s8((int8_t *)(ranks + 16 * 13), y);

        x = vld1q_s8((int8_t const *)(ranks + 16 * 14)); y = vld1q_s8((int8_t const *)(ranks + 16 * 15));
        x = vsubq_s8(x, vreinterpretq_s8_u8(vcgtq_s8(r, x)));
        y = vsubq_s8(y, vreinterpretq_s8_u8(vcgtq_s8(r, y)));
        vst1q_s8((int8_t *)(ranks + 16 * 14), x); vst1q_s8((int8_t *)(ranks + 16 * 15), y);

        ranks[currentChar] = -128;
    }

    buffer[n - 1] = 1;

    for (ptrdiff_t i = 0; i < ALPHABET_SIZE; ++i) { MTFTable[ranks[i] + 128] = (unsigned char)i; }
    for (ptrdiff_t i = 1; i < ALPHABET_SIZE; ++i) { if (flags[MTFTable[i]] == 0) { MTFTable[i] = MTFTable[i - 1]; break; } }

    return buffer + j;
}

#else

unsigned char * QLFC_TRANSFORM_FUNCTION_NAME (const unsigned char * RESTRICT input, unsigned char * RESTRICT buffer, int n, unsigned char * RESTRICT MTFTable)
{
    unsigned char Flag[ALPHABET_SIZE];

    for (int i = 0; i < ALPHABET_SIZE; ++i) Flag[i] = 0;
    for (int i = 0; i < ALPHABET_SIZE; ++i) MTFTable[i] = i;

    if (input[n - 1] == 0)
    {
        MTFTable[0] = 1; MTFTable[1] = 0;
    }

    int index = n, nSymbols = 0;
    for (int i = n - 1; i >= 0;)
    {
        unsigned char currentChar = input[i--];
        for (; (i >= 0) && (input[i] == currentChar); --i) ;

        unsigned char previousChar = MTFTable[0], rank = 1; MTFTable[0] = currentChar;
        while (true)
        {
            unsigned char temporaryChar0 = MTFTable[rank + 0]; MTFTable[rank + 0] = previousChar;
            if (temporaryChar0 == currentChar) { rank += 0; break; }

            unsigned char temporaryChar1 = MTFTable[rank + 1]; MTFTable[rank + 1] = temporaryChar0;
            if (temporaryChar1 == currentChar) { rank += 1; break; }

            unsigned char temporaryChar2 = MTFTable[rank + 2]; MTFTable[rank + 2] = temporaryChar1;
            if (temporaryChar2 == currentChar) { rank += 2; break; }

            unsigned char temporaryChar3 = MTFTable[rank + 3]; MTFTable[rank + 3] = temporaryChar2;
            if (temporaryChar3 == currentChar) { rank += 3; break; }

            rank += 4; previousChar = temporaryChar3;
        }

        if (Flag[currentChar] == 0)
        {
            Flag[currentChar] = 1;
            rank = nSymbols++;
        }

        buffer[--index] = rank;
    }

    buffer[n - 1] = 1;

    for (int rank = 1; rank < ALPHABET_SIZE; ++rank)
    {
        if (Flag[MTFTable[rank]] == 0)
        {
            MTFTable[rank] = MTFTable[rank - 1];
            break;
        }
    }

    return buffer + index;
}

#endif

#endif

#if defined(QLFC_ADAPTIVE_ENCODE_FUNCTION_NAME)

int QLFC_ADAPTIVE_ENCODE_FUNCTION_NAME (const unsigned char * input, unsigned char * output, unsigned char * buffer, int inputSize, int outputSize, QlfcStatisticalModel1 * model)
{
    unsigned char MTFTable[ALPHABET_SIZE];

    bsc_qlfc_init_model(model);

    int contextRank0 = 0;
    int contextRank4 = 0;
    int contextRun   = 0;
    int maxRank      = 7;
    int avgRank      = 0;

    unsigned char rankHistory[ALPHABET_SIZE], runHistory[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i)
    {
        rankHistory[i] = runHistory[i] = 0;
    }

    unsigned char * rankArray = bsc_qlfc_transform(input, buffer, inputSize, MTFTable);

    RangeCoder coder;

    coder.InitEncoder(output, outputSize);
    coder.EncodeWord((unsigned int)inputSize);

    unsigned char usedChar[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i) usedChar[i] = 0;

    int prevChar = -1;
    for (int rank = 0; rank < ALPHABET_SIZE; ++rank)
    {
        int currentChar = MTFTable[rank];

        for (int bit = 7; bit >= 0; --bit)
        {
            bool bit0 = false, bit1 = false;

            for (int c = 0; c < ALPHABET_SIZE; ++c)
            {
                if (c == prevChar || usedChar[c] == 0)
                {
                    if ((currentChar >> (bit + 1)) == (c >> (bit + 1)))
                    {
                        if (c & (1 << bit)) bit1 = true; else bit0 = true;
                        if (bit0 && bit1) break;
                    }
                }
            }

            if (bit0 && bit1)
            {
                coder.EncodeBit(currentChar & (1 << bit));
            }
        }

        if (currentChar == prevChar)
        {
            maxRank = bsc_bit_scan_reverse(rank - 1);
            break;
        }

        prevChar = currentChar; usedChar[currentChar] = 1;
    }

    const unsigned char * inputEnd      = input  + inputSize;
    const unsigned char * rankArrayEnd  = buffer + inputSize;

    for (; rankArray < rankArrayEnd; )
    {
        if (coder.CheckEOB())
        {
            return LIBBSC_NOT_COMPRESSIBLE;
        }

        int currentChar = *input, runSize;
        {
            const unsigned char * inputStart = input++;

            if (rankArray >= rankArrayEnd - 16)
            {
                while ((input < inputEnd) && (*input == currentChar)) { input++; }
            }
            else
            {
#if LIBBSC_CPU_FEATURE >= LIBBSC_CPU_FEATURE_SSE2
                __m128i v = _mm_set1_epi8(currentChar);

                while (true)
                {
                   int m = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)input), v));
                   if (m != 0xffff)
                   {
                      input += bsc_bit_scan_forward((unsigned int)(~m));
                      break;
                   }

                   input += 16;
                }
#elif LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_A64
                unsigned long long v = currentChar; v |= (v << 8); v |= (v << 16); v |= (v << 32);

                while (true)
                {
                    unsigned long long m = (*(unsigned long long const *)input) ^ v;
                    if (m != 0)
                    {
                        input += bsc_bit_scan_forward64(m) / 8;
                        break;
                    }

                    input += 8;
                }
#else
                while (*input == currentChar) { input++; }
#endif
            }

            runSize = (int)(input - inputStart);
        }

        int                 rank            =   *rankArray++;
        int                 history         =   rankHistory[currentChar];
        int                 state           =   model_rank_state(contextRank4, contextRun, history);

        short *            RESTRICT statePredictor  = & model->Rank.StateModel[state];
        short *            RESTRICT charPredictor   = & model->Rank.CharModel[currentChar];
        short *            RESTRICT staticPredictor = & model->Rank.StaticModel;
        ProbabilityMixer * RESTRICT mixer           = & model->mixerOfRank[currentChar];

        if (avgRank < 32)
        {
            if (rank == 1)
            {
                rankHistory[currentChar] = 0;

                int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

                ProbabilityCounter::UpdateBit0(*statePredictor,  M_RANK_TS_TH0, M_RANK_TS_AR0);
                ProbabilityCounter::UpdateBit0(*charPredictor,   M_RANK_TC_TH0, M_RANK_TC_AR0);
                ProbabilityCounter::UpdateBit0(*staticPredictor, M_RANK_TP_TH0, M_RANK_TP_AR0);

                coder.EncodeBit0(mixer->MixupAndUpdateBit0(probability0, probability1, probability2, M_RANK_TM_LR0, M_RANK_TM_LR1, M_RANK_TM_LR2, M_RANK_TM_TH0, M_RANK_TM_AR0));
            }
            else
            {
                {
                    int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

                    ProbabilityCounter::UpdateBit1(*statePredictor,  M_RANK_TS_TH1, M_RANK_TS_AR1);
                    ProbabilityCounter::UpdateBit1(*charPredictor,   M_RANK_TC_TH1, M_RANK_TC_AR1);
                    ProbabilityCounter::UpdateBit1(*staticPredictor, M_RANK_TP_TH1, M_RANK_TP_AR1);

                    coder.EncodeBit1(mixer->MixupAndUpdateBit1(probability0, probability1, probability2, M_RANK_TM_LR0, M_RANK_TM_LR1, M_RANK_TM_LR2, M_RANK_TM_TH1, M_RANK_TM_AR1));
                }

                int bitRankSize = bsc_bit_scan_reverse(rank); rankHistory[currentChar] = bitRankSize;

                statePredictor  = & model->Rank.Exponent.StateModel[state][0];
                charPredictor   = & model->Rank.Exponent.CharModel[currentChar][0];
                staticPredictor = & model->Rank.Exponent.StaticModel[0];
                mixer           = & model->mixerOfRankExponent[history < 1 ? 1 : history][1];

                for (int bit = 1; bit < bitRankSize; ++bit, ++statePredictor, ++charPredictor, ++staticPredictor)
                {
                    int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

                    ProbabilityCounter::UpdateBit1(*statePredictor,  M_RANK_ES_TH1, M_RANK_ES_AR1);
                    ProbabilityCounter::UpdateBit1(*charPredictor,   M_RANK_EC_TH1, M_RANK_EC_AR1);
                    ProbabilityCounter::UpdateBit1(*staticPredictor, M_RANK_EP_TH1, M_RANK_EP_AR1);

                    coder.EncodeBit1(mixer->MixupAndUpdateBit1(probability0, probability1, probability2, M_RANK_EM_LR0, M_RANK_EM_LR1, M_RANK_EM_LR2, M_RANK_EM_TH1, M_RANK_EM_AR1));

                    mixer = & model->mixerOfRankExponent[history <= bit ? bit + 1 : history][bit + 1];
                }
                if (bitRankSize < maxRank)
                {
                    int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

                    ProbabilityCounter::UpdateBit0(*statePredictor,  M_RANK_ES_TH0, M_RANK_ES_AR0);
                    ProbabilityCounter::UpdateBit0(*charPredictor,   M_RANK_EC_TH0, M_RANK_EC_AR0);
                    ProbabilityCounter::UpdateBit0(*staticPredictor, M_RANK_EP_TH0, M_RANK_EP_AR0);

                    coder.EncodeBit0(mixer->MixupAndUpdateBit0(probability0, probability1, probability2, M_RANK_EM_LR0, M_RANK_EM_LR1, M_RANK_EM_LR2, M_RANK_EM_TH0, M_RANK_EM_AR0));
                }

                statePredictor  = & model->Rank.Mantissa[bitRankSize].StateModel[state][0];
                charPredictor   = & model->Rank.Mantissa[bitRankSize].CharModel[currentChar][0];
                staticPredictor = & model->Rank.Mantissa[bitRankSize].StaticModel[0];
                mixer           = & model->mixerOfRankMantissa[bitRankSize];

                for (int context = 1, bit = bitRankSize - 1; bit >= 0; --bit)
                {
                    if (rank & (1 << bit))
                    {
                        int probability0 = charPredictor[context], probability1 = statePredictor[context], probability2 = staticPredictor[context];

                        ProbabilityCounter::UpdateBit1(statePredictor[context],  M_RANK_MS_TH1, M_RANK_MS_AR1);
                        ProbabilityCounter::UpdateBit1(charPredictor[context],   M_RANK_MC_TH1, M_RANK_MC_AR1);
                        ProbabilityCounter::UpdateBit1(staticPredictor[context], M_RANK_MP_TH1, M_RANK_MP_AR1);

                        coder.EncodeBit1(mixer->MixupAndUpdateBit1(probability0, probability1, probability2, M_RANK_MM_LR0, M_RANK_MM_LR1, M_RANK_MM_LR2, M_RANK_MM_TH1, M_RANK_MM_AR1));

                        context += context + 1;
                    }
                    else
                    {
                        int probability0 = charPredictor[context], probability1 = statePredictor[context], probability2 = staticPredictor[context];

                        ProbabilityCounter::UpdateBit0(statePredictor[context],  M_RANK_MS_TH0, M_RANK_MS_AR0);
                        ProbabilityCounter::UpdateBit0(charPredictor[context],   M_RANK_MC_TH0, M_RANK_MC_AR0);
                        ProbabilityCounter::UpdateBit0(staticPredictor[context], M_RANK_MP_TH0, M_RANK_MP_AR0);

                        coder.EncodeBit0(mixer->MixupAndUpdateBit0(probability0, probability1, probability2, M_RANK_MM_LR0, M_RANK_MM_LR1, M_RANK_MM_LR2, M_RANK_MM_TH0, M_RANK_MM_AR0));

                        context += context;
                    }
                }
            }
        }
        else
        {
            rankHistory[currentChar] = (unsigned char)bsc_bit_scan_reverse(rank);

            statePredictor  = & model->Rank.Escape.StateModel[state][0];
            charPredictor   = & model->Rank.Escape.CharModel[currentChar][0];
            staticPredictor = & model->Rank.Escape.StaticModel[0];

            for (int context = 1, bit = maxRank; bit >= 0; --bit)
            {
                mixer = & model->mixerOfRankEscape[context];

                if (rank & (1 << bit))
                {
                    int probability0 = charPredictor[context], probability1 = statePredictor[context], probability2 = staticPredictor[context];

                    ProbabilityCounter::UpdateBit1(statePredictor[context],  M_RANK_PS_TH1, M_RANK_PS_AR1);
                    ProbabilityCounter::UpdateBit1(charPredictor[context],   M_RANK_PC_TH1, M_RANK_PC_AR1);
                    ProbabilityCounter::UpdateBit1(staticPredictor[context], M_RANK_PP_TH1, M_RANK_PP_AR1);

                    coder.EncodeBit1(mixer->MixupAndUpdateBit1(probability0, probability1, probability2, M_RANK_PM_LR0, M_RANK_PM_LR1, M_RANK_PM_LR2, M_RANK_PM_TH1, M_RANK_PM_AR1));

                    context += context + 1;
                }
                else
                {
                    int probability0 = charPredictor[context], probability1 = statePredictor[context], probability2 = staticPredictor[context];

                    ProbabilityCounter::UpdateBit0(statePredictor[context],  M_RANK_PS_TH0, M_RANK_PS_AR0);
                    ProbabilityCounter::UpdateBit0(charPredictor[context],   M_RANK_PC_TH0, M_RANK_PC_AR0);
                    ProbabilityCounter::UpdateBit0(staticPredictor[context], M_RANK_PP_TH0, M_RANK_PP_AR0);

                    coder.EncodeBit0(mixer->MixupAndUpdateBit0(probability0, probability1, probability2, M_RANK_PM_LR0, M_RANK_PM_LR1, M_RANK_PM_LR2, M_RANK_PM_TH0, M_RANK_PM_AR0));

                    context += context;
                }
            }
        }

        avgRank         =   (avgRank * 124 + rank * 4) >> 7;
        rank            =   rank - 1;
        history         =   runHistory[currentChar];
        state           =   model_run_state(contextRank0, contextRun, rank, history);
        statePredictor  = & model->Run.StateModel[state];
        charPredictor   = & model->Run.CharModel[currentChar];
        staticPredictor = & model->Run.StaticModel;
        mixer           = & model->mixerOfRun[currentChar];

        if (runSize == 1)
        {
            runHistory[currentChar] = (runHistory[currentChar] + 2) >> 2;

            int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

            ProbabilityCounter::UpdateBit0(*statePredictor,  M_RUN_TS_TH0, M_RUN_TS_AR0);
            ProbabilityCounter::UpdateBit0(*charPredictor,   M_RUN_TC_TH0, M_RUN_TC_AR0);
            ProbabilityCounter::UpdateBit0(*staticPredictor, M_RUN_TP_TH0, M_RUN_TP_AR0);

            coder.EncodeBit0(mixer->MixupAndUpdateBit0(probability0, probability1, probability2, M_RUN_TM_LR0, M_RUN_TM_LR1, M_RUN_TM_LR2, M_RUN_TM_TH0, M_RUN_TM_AR0));
        }
        else
        {
            {
                int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

                ProbabilityCounter::UpdateBit1(*statePredictor,  M_RUN_TS_TH1, M_RUN_TS_AR1);
                ProbabilityCounter::UpdateBit1(*charPredictor,   M_RUN_TC_TH1, M_RUN_TC_AR1);
                ProbabilityCounter::UpdateBit1(*staticPredictor, M_RUN_TP_TH1, M_RUN_TP_AR1);

                coder.EncodeBit1(mixer->MixupAndUpdateBit1(probability0, probability1, probability2, M_RUN_TM_LR0, M_RUN_TM_LR1, M_RUN_TM_LR2, M_RUN_TM_TH1, M_RUN_TM_AR1));
            }

            int bitRunSize = bsc_bit_scan_reverse(runSize); runHistory[currentChar] = (runHistory[currentChar] + 3 * bitRunSize + 3) >> 2;

            statePredictor  = & model->Run.Exponent.StateModel[state][0];
            charPredictor   = & model->Run.Exponent.CharModel[currentChar][0];
            staticPredictor = & model->Run.Exponent.StaticModel[0];
            mixer           = & model->mixerOfRunExponent[history < 1 ? 1 : history][1];

            for (int bit = 1; bit < bitRunSize; ++bit, ++statePredictor, ++charPredictor, ++staticPredictor)
            {
                int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

                ProbabilityCounter::UpdateBit1(*statePredictor,  M_RUN_ES_TH1, M_RUN_ES_AR1);
                ProbabilityCounter::UpdateBit1(*charPredictor,   M_RUN_EC_TH1, M_RUN_EC_AR1);
                ProbabilityCounter::UpdateBit1(*staticPredictor, M_RUN_EP_TH1, M_RUN_EP_AR1);

                coder.EncodeBit1(mixer->MixupAndUpdateBit1(probability0, probability1, probability2, M_RUN_EM_LR0, M_RUN_EM_LR1, M_RUN_EM_LR2, M_RUN_EM_TH1, M_RUN_EM_AR1));

                mixer = & model->mixerOfRunExponent[history <= bit ? bit + 1 : history][bit + 1];
            }
            {
                int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

                ProbabilityCounter::UpdateBit0(*statePredictor,  M_RUN_ES_TH0, M_RUN_ES_AR0);
                ProbabilityCounter::UpdateBit0(*charPredictor,   M_RUN_EC_TH0, M_RUN_EC_AR0);
                ProbabilityCounter::UpdateBit0(*staticPredictor, M_RUN_EP_TH0, M_RUN_EP_AR0);

                coder.EncodeBit0(mixer->MixupAndUpdateBit0(probability0, probability1, probability2, M_RUN_EM_LR0, M_RUN_EM_LR1, M_RUN_EM_LR2, M_RUN_EM_TH0, M_RUN_EM_AR0));
            }

            statePredictor  = & model->Run.Mantissa[bitRunSize].StateModel[state][0];
            charPredictor   = & model->Run.Mantissa[bitRunSize].CharModel[currentChar][0];
            staticPredictor = & model->Run.Mantissa[bitRunSize].StaticModel[0];
            mixer           = & model->mixerOfRunMantissa[bitRunSize];

            for (int context = 1, bit = bitRunSize - 1; bit >= 0; --bit)
            {
                if (runSize & (1 << bit))
                {
                    int probability0 = charPredictor[context], probability1 = statePredictor[context], probability2 = staticPredictor[context];

                    ProbabilityCounter::UpdateBit1(statePredictor[context],  M_RUN_MS_TH1, M_RUN_MS_AR1);
                    ProbabilityCounter::UpdateBit1(charPredictor[context],   M_RUN_MC_TH1, M_RUN_MC_AR1);
                    ProbabilityCounter::UpdateBit1(staticPredictor[context], M_RUN_MP_TH1, M_RUN_MP_AR1);

                    coder.EncodeBit1(mixer->MixupAndUpdateBit1(probability0, probability1, probability2, M_RUN_MM_LR0, M_RUN_MM_LR1, M_RUN_MM_LR2, M_RUN_MM_TH1, M_RUN_MM_AR1));

                    if (bitRunSize <= 5) context += context + 1; else context++;
                }
                else
                {
                    int probability0 = charPredictor[context], probability1 = statePredictor[context], probability2 = staticPredictor[context];

                    ProbabilityCounter::UpdateBit0(statePredictor[context],  M_RUN_MS_TH0, M_RUN_MS_AR0);
                    ProbabilityCounter::UpdateBit0(charPredictor[context],   M_RUN_MC_TH0, M_RUN_MC_AR0);
                    ProbabilityCounter::UpdateBit0(staticPredictor[context], M_RUN_MP_TH0, M_RUN_MP_AR0);

                    coder.EncodeBit0(mixer->MixupAndUpdateBit0(probability0, probability1, probability2, M_RUN_MM_LR0, M_RUN_MM_LR1, M_RUN_MM_LR2, M_RUN_MM_TH0, M_RUN_MM_AR0));

                    if (bitRunSize <= 5) context += context + 0; else context++;
                }
            }
        }

        contextRank0 = ((contextRank0 << 1) | (rank == 0   ? 1    : 0)) & 0x7;
        contextRank4 = ((contextRank4 << 2) | (rank < 3    ? rank : 3)) & 0xff;
        contextRun   = ((contextRun   << 1) | (runSize < 3 ? 1    : 0)) & 0xf;
    }

    return coder.FinishEncoder();
}

#endif

#if defined(QLFC_STATIC_ENCODE_FUNCTION_NAME)

int QLFC_STATIC_ENCODE_FUNCTION_NAME (const unsigned char * input, unsigned char * output, unsigned char * buffer, int inputSize, int outputSize, QlfcStatisticalModel1 * model)
{
    unsigned char MTFTable[ALPHABET_SIZE];

    bsc_qlfc_init_model(model);

    int contextRank0 = 0;
    int contextRank4 = 0;
    int contextRun   = 0;
    int maxRank      = 7;
    int avgRank      = 0;

    unsigned char rankHistory[ALPHABET_SIZE], runHistory[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i)
    {
        rankHistory[i] = runHistory[i] = 0;
    }

    unsigned char * rankArray = bsc_qlfc_transform(input, buffer, inputSize, MTFTable);

    RangeCoder coder;

    coder.InitEncoder(output, outputSize);
    coder.EncodeWord((unsigned int)inputSize);

    unsigned char usedChar[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i) usedChar[i] = 0;

    int prevChar = -1;
    for (int rank = 0; rank < ALPHABET_SIZE; ++rank)
    {
        int currentChar = MTFTable[rank];

        for (int bit = 7; bit >= 0; --bit)
        {
            bool bit0 = false, bit1 = false;

            for (int c = 0; c < ALPHABET_SIZE; ++c)
            {
                if (c == prevChar || usedChar[c] == 0)
                {
                    if ((currentChar >> (bit + 1)) == (c >> (bit + 1)))
                    {
                        if (c & (1 << bit)) bit1 = true; else bit0 = true;
                        if (bit0 && bit1) break;
                    }
                }
            }

            if (bit0 && bit1)
            {
                coder.EncodeBit(currentChar & (1 << bit));
            }
        }

        if (currentChar == prevChar)
        {
            maxRank = bsc_bit_scan_reverse(rank - 1);
            break;
        }

        prevChar = currentChar; usedChar[currentChar] = 1;
    }

    const unsigned char * inputEnd      = input  + inputSize;
    const unsigned char * rankArrayEnd  = buffer + inputSize;

    for (; rankArray < rankArrayEnd; )
    {
        if (coder.CheckEOB())
        {
            return LIBBSC_NOT_COMPRESSIBLE;
        }

        int currentChar = *input, runSize;
        {
            const unsigned char * inputStart = input++;

            if (rankArray >= rankArrayEnd - 16)
            {
                while ((input < inputEnd) && (*input == currentChar)) { input++; }
            }
            else
            {
#if LIBBSC_CPU_FEATURE >= LIBBSC_CPU_FEATURE_SSE2
                __m128i v = _mm_set1_epi8(currentChar);

                while (true)
                {
                   int m = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)input), v));
                   if (m != 0xffff)
                   {
                      input += bsc_bit_scan_forward((unsigned int)(~m));
                      break;
                   }

                   input += 16;
                }
#elif LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_A64
                unsigned long long v = currentChar; v |= (v << 8); v |= (v << 16); v |= (v << 32);

                while (true)
                {
                    unsigned long long m = (*(unsigned long long const *)input) ^ v;
                    if (m != 0)
                    {
                        input += bsc_bit_scan_forward64(m) / 8;
                        break;
                    }

                    input += 8;
                }
#else
                while (*input == currentChar) { input++; }
#endif
            }

            runSize = (int)(input - inputStart);
        }

        int                 rank            =   *rankArray++;
        int                 history         =   rankHistory[currentChar];
        int                 state           =   model_rank_state(contextRank4, contextRun, history);

        short * RESTRICT    statePredictor  = & model->Rank.StateModel[state];
        short * RESTRICT    charPredictor   = & model->Rank.CharModel[currentChar];
        short * RESTRICT    staticPredictor = & model->Rank.StaticModel;

        if (avgRank < 32)
        {
            if (rank == 1)
            {
                rankHistory[currentChar] = 0;

                int probability = ((*charPredictor) * F_RANK_TM_LR0 + (*statePredictor) * F_RANK_TM_LR1 + (*staticPredictor) * F_RANK_TM_LR2) >> 5;

                ProbabilityCounter::UpdateBit0(*statePredictor,  F_RANK_TS_TH0, F_RANK_TS_AR0);
                ProbabilityCounter::UpdateBit0(*charPredictor,   F_RANK_TC_TH0, F_RANK_TC_AR0);
                ProbabilityCounter::UpdateBit0(*staticPredictor, F_RANK_TP_TH0, F_RANK_TP_AR0);

                coder.EncodeBit0(probability);
            }
            else
            {
                {
                    int probability = ((*charPredictor) * F_RANK_TM_LR0 + (*statePredictor) * F_RANK_TM_LR1 + (*staticPredictor) * F_RANK_TM_LR2) >> 5;

                    ProbabilityCounter::UpdateBit1(*statePredictor,  F_RANK_TS_TH1, F_RANK_TS_AR1);
                    ProbabilityCounter::UpdateBit1(*charPredictor,   F_RANK_TC_TH1, F_RANK_TC_AR1);
                    ProbabilityCounter::UpdateBit1(*staticPredictor, F_RANK_TP_TH1, F_RANK_TP_AR1);

                    coder.EncodeBit1(probability);
                }

                int bitRankSize = bsc_bit_scan_reverse(rank); rankHistory[currentChar] = bitRankSize;

                statePredictor  = & model->Rank.Exponent.StateModel[state][0];
                charPredictor   = & model->Rank.Exponent.CharModel[currentChar][0];
                staticPredictor = & model->Rank.Exponent.StaticModel[0];

                for (int bit = 1; bit < bitRankSize; ++bit, ++statePredictor, ++charPredictor, ++staticPredictor)
                {
                    int probability = ((*charPredictor) * F_RANK_EM_LR0 + (*statePredictor) * F_RANK_EM_LR1 + (*staticPredictor) * F_RANK_EM_LR2) >> 5;

                    ProbabilityCounter::UpdateBit1(*statePredictor,  F_RANK_ES_TH1, F_RANK_ES_AR1);
                    ProbabilityCounter::UpdateBit1(*charPredictor,   F_RANK_EC_TH1, F_RANK_EC_AR1);
                    ProbabilityCounter::UpdateBit1(*staticPredictor, F_RANK_EP_TH1, F_RANK_EP_AR1);

                    coder.EncodeBit1(probability);
                }
                if (bitRankSize < maxRank)
                {
                    int probability = ((*charPredictor) * F_RANK_EM_LR0 + (*statePredictor) * F_RANK_EM_LR1 + (*staticPredictor) * F_RANK_EM_LR2) >> 5;

                    ProbabilityCounter::UpdateBit0(*statePredictor,  F_RANK_ES_TH0, F_RANK_ES_AR0);
                    ProbabilityCounter::UpdateBit0(*charPredictor,   F_RANK_EC_TH0, F_RANK_EC_AR0);
                    ProbabilityCounter::UpdateBit0(*staticPredictor, F_RANK_EP_TH0, F_RANK_EP_AR0);

                    coder.EncodeBit0(probability);
                }

                statePredictor  = & model->Rank.Mantissa[bitRankSize].StateModel[state][0];
                charPredictor   = & model->Rank.Mantissa[bitRankSize].CharModel[currentChar][0];
                staticPredictor = & model->Rank.Mantissa[bitRankSize].StaticModel[0];

                for (int context = 1, bit = bitRankSize - 1; bit >= 0; --bit)
                {
                    int probability = (charPredictor[context] * F_RANK_MM_LR0 + statePredictor[context] * F_RANK_MM_LR1 + staticPredictor[context] * F_RANK_MM_LR2) >> 5;

                    unsigned int b = (rank >> bit) & 1;
                    ProbabilityCounter::UpdateBit(b, statePredictor[context],  F_RANK_MS_TH0, F_RANK_MS_AR0, F_RANK_MS_TH1, F_RANK_MS_AR1);
                    ProbabilityCounter::UpdateBit(b, charPredictor[context],   F_RANK_MC_TH0, F_RANK_MC_AR0, F_RANK_MC_TH1, F_RANK_MC_AR1);
                    ProbabilityCounter::UpdateBit(b, staticPredictor[context], F_RANK_MP_TH0, F_RANK_MP_AR0, F_RANK_MP_TH1, F_RANK_MP_AR1);

                    context += context + b; coder.EncodeBit(b, probability);
                }
            }
        }
        else
        {
            rankHistory[currentChar] = (unsigned char)bsc_bit_scan_reverse(rank);

            statePredictor  = & model->Rank.Escape.StateModel[state][0];
            charPredictor   = & model->Rank.Escape.CharModel[currentChar][0];
            staticPredictor = & model->Rank.Escape.StaticModel[0];

            for (int context = 1, bit = maxRank; bit >= 0; --bit)
            {
                int probability = (charPredictor[context] * F_RANK_PM_LR0 + statePredictor[context] * F_RANK_PM_LR1 + staticPredictor[context] * F_RANK_PM_LR2) >> 5;

                unsigned int b = (rank >> bit) & 1;
                ProbabilityCounter::UpdateBit(b, statePredictor[context],  F_RANK_PS_TH0, F_RANK_PS_AR0, F_RANK_PS_TH1, F_RANK_PS_AR1);
                ProbabilityCounter::UpdateBit(b, charPredictor[context],   F_RANK_PC_TH0, F_RANK_PC_AR0, F_RANK_PC_TH1, F_RANK_PC_AR1);
                ProbabilityCounter::UpdateBit(b, staticPredictor[context], F_RANK_PP_TH0, F_RANK_PP_AR0, F_RANK_PP_TH1, F_RANK_PP_AR1);

                context += context + b; coder.EncodeBit(b, probability);
            }
        }

        avgRank         =   (avgRank * 124 + rank * 4) >> 7;
        rank            =   rank - 1;
        history         =   runHistory[currentChar];
        state           =   model_run_state(contextRank0, contextRun, rank, history);
        statePredictor  = & model->Run.StateModel[state];
        charPredictor   = & model->Run.CharModel[currentChar];
        staticPredictor = & model->Run.StaticModel;

        if (runSize == 1)
        {
            runHistory[currentChar] = (runHistory[currentChar] + 2) >> 2;

            int probability = ((*charPredictor) * F_RUN_TM_LR0 + (*statePredictor) * F_RUN_TM_LR1 + (*staticPredictor) * F_RUN_TM_LR2) >> 5;

            ProbabilityCounter::UpdateBit0(*statePredictor,  F_RUN_TS_TH0, F_RUN_TS_AR0);
            ProbabilityCounter::UpdateBit0(*charPredictor,   F_RUN_TC_TH0, F_RUN_TC_AR0);
            ProbabilityCounter::UpdateBit0(*staticPredictor, F_RUN_TP_TH0, F_RUN_TP_AR0);

            coder.EncodeBit0(probability);
        }
        else
        {
            {
                int probability = ((*charPredictor) * F_RUN_TM_LR0 + (*statePredictor) * F_RUN_TM_LR1 + (*staticPredictor) * F_RUN_TM_LR2) >> 5;

                ProbabilityCounter::UpdateBit1(*statePredictor,  F_RUN_TS_TH1, F_RUN_TS_AR1);
                ProbabilityCounter::UpdateBit1(*charPredictor,   F_RUN_TC_TH1, F_RUN_TC_AR1);
                ProbabilityCounter::UpdateBit1(*staticPredictor, F_RUN_TP_TH1, F_RUN_TP_AR1);

                coder.EncodeBit1(probability);
            }

            int bitRunSize = bsc_bit_scan_reverse(runSize); runHistory[currentChar] = (runHistory[currentChar] + 3 * bitRunSize + 3) >> 2;

            statePredictor  = & model->Run.Exponent.StateModel[state][0];
            charPredictor   = & model->Run.Exponent.CharModel[currentChar][0];
            staticPredictor = & model->Run.Exponent.StaticModel[0];

            for (int bit = 1; bit < bitRunSize; ++bit, ++statePredictor, ++charPredictor, ++staticPredictor)
            {
                int probability = ((*charPredictor) * F_RUN_EM_LR0 + (*statePredictor) * F_RUN_EM_LR1 + (*staticPredictor) * F_RUN_EM_LR2) >> 5;

                ProbabilityCounter::UpdateBit1(*statePredictor,  F_RUN_ES_TH1, F_RUN_ES_AR1);
                ProbabilityCounter::UpdateBit1(*charPredictor,   F_RUN_EC_TH1, F_RUN_EC_AR1);
                ProbabilityCounter::UpdateBit1(*staticPredictor, F_RUN_EP_TH1, F_RUN_EP_AR1);

                coder.EncodeBit1(probability);
            }
            {
                int probability = ((*charPredictor) * F_RUN_EM_LR0 + (*statePredictor) * F_RUN_EM_LR1 + (*staticPredictor) * F_RUN_EM_LR2) >> 5;

                ProbabilityCounter::UpdateBit0(*statePredictor,  F_RUN_ES_TH0, F_RUN_ES_AR0);
                ProbabilityCounter::UpdateBit0(*charPredictor,   F_RUN_EC_TH0, F_RUN_EC_AR0);
                ProbabilityCounter::UpdateBit0(*staticPredictor, F_RUN_EP_TH0, F_RUN_EP_AR0);

                coder.EncodeBit0(probability);
            }

            statePredictor  = & model->Run.Mantissa[bitRunSize].StateModel[state][0];
            charPredictor   = & model->Run.Mantissa[bitRunSize].CharModel[currentChar][0];
            staticPredictor = & model->Run.Mantissa[bitRunSize].StaticModel[0];

            for (int context = 1, bit = bitRunSize - 1; bit >= 0; --bit)
            {
                int probability = (charPredictor[context] * F_RUN_MM_LR0 + statePredictor[context] * F_RUN_MM_LR1 + staticPredictor[context] * F_RUN_MM_LR2) >> 5;

                unsigned int b = (runSize >> bit) & 1;
                ProbabilityCounter::UpdateBit(b, statePredictor[context],  F_RUN_MS_TH0, F_RUN_MS_AR0, F_RUN_MS_TH1, F_RUN_MS_AR1);
                ProbabilityCounter::UpdateBit(b, charPredictor[context],   F_RUN_MC_TH0, F_RUN_MC_AR0, F_RUN_MC_TH1, F_RUN_MC_AR1);
                ProbabilityCounter::UpdateBit(b, staticPredictor[context], F_RUN_MP_TH0, F_RUN_MP_AR0, F_RUN_MP_TH1, F_RUN_MP_AR1);

                int ctx = context + context + b; context++; if (bitRunSize <= 5) { context = ctx; } coder.EncodeBit(b, probability);
            }
        }

        contextRank0 = ((contextRank0 << 1) | (rank == 0   ? 1    : 0)) & 0x7;
        contextRank4 = ((contextRank4 << 2) | (rank < 3    ? rank : 3)) & 0xff;
        contextRun   = ((contextRun   << 1) | (runSize < 3 ? 1    : 0)) & 0xf;
    }

    return coder.FinishEncoder();
}

#endif

#if defined(QLFC_FAST_ENCODE_FUNCTION_NAME)

int QLFC_FAST_ENCODE_FUNCTION_NAME (const unsigned char * RESTRICT input, unsigned char * RESTRICT output, unsigned char * RESTRICT buffer, int inputSize, int outputSize, QlfcStatisticalModel2 * model)
{
    unsigned char MTFTable[ALPHABET_SIZE];

    bsc_qlfc_init_model(model);

    unsigned char * RESTRICT ranks = bsc_qlfc_transform(input, buffer, inputSize, MTFTable);

    RangeCoder coder;

    coder.InitEncoder(output, outputSize);
    coder.EncodeWord((unsigned int)inputSize);

    unsigned char usedChar[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i) usedChar[i] = 0;

    int prevChar = -1;
    for (int rank = 0; rank < ALPHABET_SIZE; ++rank)
    {
        int currentChar = MTFTable[rank];

        for (int bit = 7; bit >= 0; --bit)
        {
            bool bit0 = false, bit1 = false;

            for (int c = 0; c < ALPHABET_SIZE; ++c)
            {
                if (c == prevChar || usedChar[c] == 0)
                {
                    if ((currentChar >> (bit + 1)) == (c >> (bit + 1)))
                    {
                        if (c & (1 << bit)) bit1 = true; else bit0 = true;
                        if (bit0 && bit1) break;
                    }
                }
            }

            if (bit0 && bit1)
            {
                coder.EncodeBit<1>(currentChar & (1 << bit), 1);
            }
        }

        if (currentChar == prevChar)
        {
            break;
        }

        prevChar = currentChar; usedChar[currentChar] = 1;
    }

    const unsigned char * inputEnd      = input  + inputSize;
    const unsigned char * ranksEnd      = buffer + inputSize;

    for (; ranks < ranksEnd; )
    {
        if (coder.CheckEOB())
        {
            return LIBBSC_NOT_COMPRESSIBLE;
        }

        unsigned int currentRank = *ranks++;
        unsigned int currentChar = *input;
        unsigned int currentRun;

        {
            const unsigned char * runStart = input++;

            if (ranks < ranksEnd - 16)
            {
#if LIBBSC_CPU_FEATURE >= LIBBSC_CPU_FEATURE_SSE2
                __m128i v = _mm_set1_epi8(currentChar);

                while (true)
                {
                   int m = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)input), v));
                   if (m != 0xffff)
                   {
                      input += bsc_bit_scan_forward((unsigned int)(~m));
                      break;
                   }

                   input += 16;
                }
#elif LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_A64
                unsigned long long v = currentChar; v |= (v << 8); v |= (v << 16); v |= (v << 32);

                while (true)
                {
                    unsigned long long m = (*(unsigned long long const *)input) ^ v;
                    if (m != 0)
                    {
                        input += bsc_bit_scan_forward64(m) / 8;
                        break;
                    }

                    input += 8;
                }
#else
                while (*input == currentChar) { input++; }
#endif
            }
            else
            {
                while ((input < inputEnd) && (*input == currentChar)) { input++; }
            }

            currentRun = (unsigned int)(input - runStart);
        }

        {
            short * RESTRICT predictor = &model->Rank.Exponent[currentChar][0];

            if (currentRank == 1)
            {
                int p = predictor[0]; ProbabilityCounter::UpdateBit<4>(predictor[0], 8016); coder.EncodeBit0<13>(p);
            }
            else
            {
                {
                    int p = predictor[0]; ProbabilityCounter::UpdateBit<4>(predictor[0], 83); coder.EncodeBit1<13>(p);
                }

                int bitRankSize = bsc_bit_scan_reverse(currentRank);

                for (int bit = 1; bit < bitRankSize; ++bit)
                {
                    int p = predictor[bit]; ProbabilityCounter::UpdateBit<4>(predictor[bit], 122); coder.EncodeBit1<13>(p);
                }

                if (bitRankSize < 7)
                {
                    int p = predictor[bitRankSize]; ProbabilityCounter::UpdateBit<4>(predictor[bitRankSize], 8114); coder.EncodeBit0<13>(p);
                }

                predictor = &model->Rank.Mantissa[currentChar][bitRankSize][0];

                for (int context = 1, bit = bitRankSize - 1; bit >= 0; --bit)
                {
                    unsigned int b = (currentRank >> bit) & 1;

                    int p = predictor[context]; ProbabilityCounter::UpdateBit<7>(b, predictor[context], 7999, 235); coder.EncodeBit<13>(b, p);

                    context += context + b; 
                }
            }
        }

        {
            short * RESTRICT predictor = &model->Run.Exponent[currentChar][0];

            if (currentRun == 1)
            {
                int p = predictor[0]; ProbabilityCounter::UpdateBit<5>(predictor[0], 2025); coder.EncodeBit0<11>(p);
            }
            else
            {
                {
                    int p = predictor[0]; ProbabilityCounter::UpdateBit<5>(predictor[0], 42); coder.EncodeBit1<11>(p);
                }

                int bitRunSize = bsc_bit_scan_reverse(currentRun);

                for (int bit = 1; bit < bitRunSize; ++bit)
                {
                    int p = predictor[bit]; ProbabilityCounter::UpdateBit<4>(predictor[bit], 142); coder.EncodeBit1<11>(p);
                }

                {
                    int p = predictor[bitRunSize]; ProbabilityCounter::UpdateBit<4>(predictor[bitRunSize], 1962); coder.EncodeBit0<11>(p);
                }

                predictor = &model->Run.Mantissa[currentChar][bitRunSize][0];

                if (bitRunSize <= 5)
                {
                    for (int context = 1, bit = bitRunSize - 1; bit >= 0; --bit)
                    {
                        unsigned int b = (currentRun >> bit) & 1;

                        int p = predictor[context]; ProbabilityCounter::UpdateBit<6>(b, predictor[context], 1951, 147); coder.EncodeBit<11>(b, p);

                        context += context + b;
                    }
                }
                else
                {
                    for (int context = 1, bit = bitRunSize - 1; bit >= 0; --bit)
                    {
                        unsigned int b = (currentRun >> bit) & 1;

                        int p = predictor[context]; ProbabilityCounter::UpdateBit<5>(b, predictor[context], 1987, 46); coder.EncodeBit<11>(b, p);

                        context += 1;
                    }
                }
            }
        }
    }
    
    return coder.FinishEncoder();
}

#endif

#if (defined(QLFC_ADAPTIVE_DECODE_FUNCTION_NAME) || defined(QLFC_STATIC_DECODE_FUNCTION_NAME) || defined(QLFC_FAST_DECODE_FUNCTION_NAME)) && (LIBBSC_CPU_FEATURE >= LIBBSC_CPU_FEATURE_SSE41 || LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_A64)

static const unsigned char ALIGNED(64) rank16_shuffle[16][16] =
{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 12, 13, 14, 15},
    {1, 2, 3, 4, 5, 6, 7, 8, 0, 9, 10, 11, 12, 13, 14, 15},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10, 11, 12, 13, 14, 15},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 15},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 12, 13, 14, 15},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 13, 14, 15},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 14, 15},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 15},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0},
};

#endif

#if defined(QLFC_ADAPTIVE_DECODE_FUNCTION_NAME)

int QLFC_ADAPTIVE_DECODE_FUNCTION_NAME (const unsigned char * input, unsigned char * output, QlfcStatisticalModel1 * model)
{
    RangeCoder coder;

    unsigned char ALIGNED(64) MTFTable[ALPHABET_SIZE];

    bsc_qlfc_init_model(model);

    int contextRank0 = 0;
    int contextRank4 = 0;
    int contextRun   = 0;
    int maxRank      = 7;
    int avgRank      = 0;

    unsigned char rankHistory[ALPHABET_SIZE], runHistory[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i)
    {
        rankHistory[i] = runHistory[i] = 0;
    }

    coder.InitDecoder(input);
    int n = (int)coder.DecodeWord();

    unsigned char usedChar[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i) usedChar[i] = 0;

    int prevChar = -1;
    for (int rank = 0; rank < ALPHABET_SIZE; ++rank)
    {
        int currentChar = 0;

        for (int bit = 7; bit >= 0; --bit)
        {
            bool bit0 = false, bit1 = false;

            for (int c = 0; c < ALPHABET_SIZE; ++c)
            {
                if (c == prevChar || usedChar[c] == 0)
                {
                    if (currentChar == (c >> (bit + 1)))
                    {
                        if (c & (1 << bit)) bit1 = true; else bit0 = true;
                        if (bit0 && bit1) break;
                    }
                }
            }

            if (bit0 && bit1)
            {
                currentChar += currentChar + coder.DecodeBit();
            }
            else
            {
                if (bit0) currentChar += currentChar + 0;
                if (bit1) currentChar += currentChar + 1;
            }
        }

        MTFTable[rank] =  currentChar;

        if (currentChar == prevChar)
        {
            maxRank = bsc_bit_scan_reverse(rank - 1);
            break;
        }

        prevChar = currentChar; usedChar[currentChar] = 1;
    }

    for (int i = 0; i < n;)
    {
        int                 currentChar     =   MTFTable[0];
        int                 history         =   rankHistory[currentChar];
        int                 state           =   model_rank_state(contextRank4, contextRun, history);

        short *            RESTRICT statePredictor  = & model->Rank.StateModel[state];
        short *            RESTRICT charPredictor   = & model->Rank.CharModel[currentChar];
        short *            RESTRICT staticPredictor = & model->Rank.StaticModel;
        ProbabilityMixer * RESTRICT mixer           = & model->mixerOfRank[currentChar];

        int rank = 1;
        if (avgRank < 32)
        {
            if (coder.DecodeBit(mixer->Mixup(*charPredictor, *statePredictor, *staticPredictor)))
            {
                ProbabilityCounter::UpdateBit1(*statePredictor,  M_RANK_TS_TH1, M_RANK_TS_AR1);
                ProbabilityCounter::UpdateBit1(*charPredictor,   M_RANK_TC_TH1, M_RANK_TC_AR1);
                ProbabilityCounter::UpdateBit1(*staticPredictor, M_RANK_TP_TH1, M_RANK_TP_AR1);
                mixer->UpdateBit1(M_RANK_TM_LR0, M_RANK_TM_LR1, M_RANK_TM_LR2, M_RANK_TM_TH1, M_RANK_TM_AR1);

                statePredictor  = & model->Rank.Exponent.StateModel[state][0];
                charPredictor   = & model->Rank.Exponent.CharModel[currentChar][0];
                staticPredictor = & model->Rank.Exponent.StaticModel[0];
                mixer           = & model->mixerOfRankExponent[history < 1 ? 1 : history][1];

                int bitRankSize = 1;
                while (true)
                {
                    if (bitRankSize == maxRank) break;
                    if (coder.DecodeBit(mixer->Mixup(*charPredictor, *statePredictor, *staticPredictor)))
                    {
                        ProbabilityCounter::UpdateBit1(*statePredictor,  M_RANK_ES_TH1, M_RANK_ES_AR1); statePredictor++;
                        ProbabilityCounter::UpdateBit1(*charPredictor,   M_RANK_EC_TH1, M_RANK_EC_AR1); charPredictor++;
                        ProbabilityCounter::UpdateBit1(*staticPredictor, M_RANK_EP_TH1, M_RANK_EP_AR1); staticPredictor++;
                        mixer->UpdateBit1(M_RANK_EM_LR0, M_RANK_EM_LR1, M_RANK_EM_LR2, M_RANK_EM_TH1, M_RANK_EM_AR1);
                        bitRankSize++;
                        mixer = & model->mixerOfRankExponent[history < bitRankSize ? bitRankSize : history][bitRankSize];
                    }
                    else
                    {
                        ProbabilityCounter::UpdateBit0(*statePredictor,  M_RANK_ES_TH0, M_RANK_ES_AR0);
                        ProbabilityCounter::UpdateBit0(*charPredictor,   M_RANK_EC_TH0, M_RANK_EC_AR0);
                        ProbabilityCounter::UpdateBit0(*staticPredictor, M_RANK_EP_TH0, M_RANK_EP_AR0);
                        mixer->UpdateBit0(M_RANK_EM_LR0, M_RANK_EM_LR1, M_RANK_EM_LR2, M_RANK_EM_TH0, M_RANK_EM_AR0);
                        break;
                    }
                }

                rankHistory[currentChar] = bitRankSize;

                statePredictor  = & model->Rank.Mantissa[bitRankSize].StateModel[state][0];
                charPredictor   = & model->Rank.Mantissa[bitRankSize].CharModel[currentChar][0];
                staticPredictor = & model->Rank.Mantissa[bitRankSize].StaticModel[0];
                mixer           = & model->mixerOfRankMantissa[bitRankSize];

                for (int bit = bitRankSize - 1; bit >= 0; --bit)
                {
                    if (coder.DecodeBit(mixer->Mixup(charPredictor[rank], statePredictor[rank], staticPredictor[rank])))
                    {
                        ProbabilityCounter::UpdateBit1(statePredictor[rank],  M_RANK_MS_TH1, M_RANK_MS_AR1);
                        ProbabilityCounter::UpdateBit1(charPredictor[rank],   M_RANK_MC_TH1, M_RANK_MC_AR1);
                        ProbabilityCounter::UpdateBit1(staticPredictor[rank], M_RANK_MP_TH1, M_RANK_MP_AR1);
                        mixer->UpdateBit1(M_RANK_MM_LR0, M_RANK_MM_LR1, M_RANK_MM_LR2, M_RANK_MM_TH1, M_RANK_MM_AR1);
                        rank += rank + 1;
                    }
                    else
                    {
                        ProbabilityCounter::UpdateBit0(statePredictor[rank],  M_RANK_MS_TH0, M_RANK_MS_AR0);
                        ProbabilityCounter::UpdateBit0(charPredictor[rank],   M_RANK_MC_TH0, M_RANK_MC_AR0);
                        ProbabilityCounter::UpdateBit0(staticPredictor[rank], M_RANK_MP_TH0, M_RANK_MP_AR0);
                        mixer->UpdateBit0(M_RANK_MM_LR0, M_RANK_MM_LR1, M_RANK_MM_LR2, M_RANK_MM_TH0, M_RANK_MM_AR0);
                        rank += rank;
                    }
                }
            }
            else
            {
                rankHistory[currentChar] = 0;
                ProbabilityCounter::UpdateBit0(*statePredictor, M_RANK_TS_TH0,  M_RANK_TS_AR0);
                ProbabilityCounter::UpdateBit0(*charPredictor, M_RANK_TC_TH0,   M_RANK_TC_AR0);
                ProbabilityCounter::UpdateBit0(*staticPredictor, M_RANK_TP_TH0, M_RANK_TP_AR0);
                mixer->UpdateBit0(M_RANK_TM_LR0, M_RANK_TM_LR1, M_RANK_TM_LR2, M_RANK_TM_TH0, M_RANK_TM_AR0);
            }
        }
        else
        {
            statePredictor  = & model->Rank.Escape.StateModel[state][0];
            charPredictor   = & model->Rank.Escape.CharModel[currentChar][0];
            staticPredictor = & model->Rank.Escape.StaticModel[0];

            rank = 0;
            for (int context = 1, bit = maxRank; bit >= 0; --bit)
            {
                mixer = & model->mixerOfRankEscape[context];

                if (coder.DecodeBit(mixer->Mixup(charPredictor[context], statePredictor[context], staticPredictor[context])))
                {
                    ProbabilityCounter::UpdateBit1(statePredictor[context],  M_RANK_PS_TH1, M_RANK_PS_AR1);
                    ProbabilityCounter::UpdateBit1(charPredictor[context],   M_RANK_PC_TH1, M_RANK_PC_AR1);
                    ProbabilityCounter::UpdateBit1(staticPredictor[context], M_RANK_PP_TH1, M_RANK_PP_AR1);
                    mixer->UpdateBit1(M_RANK_PM_LR0, M_RANK_PM_LR1, M_RANK_PM_LR2, M_RANK_PM_TH1, M_RANK_PM_AR1);
                    context += context + 1; rank += rank + 1;
                }
                else
                {
                    ProbabilityCounter::UpdateBit0(statePredictor[context],  M_RANK_PS_TH0, M_RANK_PS_AR0);
                    ProbabilityCounter::UpdateBit0(charPredictor[context],   M_RANK_PC_TH0, M_RANK_PC_AR0);
                    ProbabilityCounter::UpdateBit0(staticPredictor[context], M_RANK_PP_TH0, M_RANK_PP_AR0);
                    mixer->UpdateBit0(M_RANK_PM_LR0, M_RANK_PM_LR1, M_RANK_PM_LR2, M_RANK_PM_TH0, M_RANK_PM_AR0);
                    context += context; rank += rank;
                }
            }

            rankHistory[currentChar] = (unsigned char)bsc_bit_scan_reverse(rank);
        }

        {
#if LIBBSC_CPU_FEATURE >= LIBBSC_CPU_FEATURE_SSE41
            __m128i * MTFTable_p = (__m128i *)&MTFTable[rank & (-16)];
            __m128i r = _mm_load_si128(MTFTable_p); _mm_store_si128(MTFTable_p, _mm_shuffle_epi8(_mm_insert_epi8(r, currentChar, 0), _mm_load_si128((const __m128i *)&rank16_shuffle[rank & 15][0])));

            while ((--MTFTable_p) >= (__m128i *)MTFTable)
            {
                __m128i t = _mm_load_si128(MTFTable_p); _mm_store_si128(MTFTable_p, _mm_alignr_epi8(r, t, 1)); r = t;
            }
#elif LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_A64
            uint8x16_t * MTFTable_p = (uint8x16_t *)&MTFTable[rank & (-16)];
            uint8x16_t r = vld1q_u8((const unsigned char *)MTFTable_p); vst1q_u8((unsigned char *)MTFTable_p, vqtbl1q_u8(vsetq_lane_u8((unsigned char)currentChar, r, 0), vld1q_u8((const unsigned char *)&rank16_shuffle[rank & 15][0])));
                    
            while ((--MTFTable_p) >= (uint8x16_t *)MTFTable)
            {
                uint8x16_t t = vld1q_u8((const unsigned char *)MTFTable_p); vst1q_u8((unsigned char *)MTFTable_p, vextq_u8(t, r, 1)); r = t;
            }
#else
            for (int r = 0; r < rank; ++r)
            {
                MTFTable[r] = MTFTable[r + 1];
            }
            MTFTable[rank] = currentChar;
#endif
        }

        avgRank         =   (avgRank * 124 + rank * 4) >> 7;
        rank            =   rank - 1;
        history         =   runHistory[currentChar];
        state           =   model_run_state(contextRank0, contextRun, rank, history);
        statePredictor  = & model->Run.StateModel[state];
        charPredictor   = & model->Run.CharModel[currentChar];
        staticPredictor = & model->Run.StaticModel;
        mixer           = & model->mixerOfRun[currentChar];

        int runSize = 1;
        if (coder.DecodeBit(mixer->Mixup(*charPredictor, *statePredictor, *staticPredictor)))
        {
            ProbabilityCounter::UpdateBit1(*statePredictor,  M_RUN_TS_TH1, M_RUN_TS_AR1);
            ProbabilityCounter::UpdateBit1(*charPredictor,   M_RUN_TC_TH1, M_RUN_TC_AR1);
            ProbabilityCounter::UpdateBit1(*staticPredictor, M_RUN_TP_TH1, M_RUN_TP_AR1);
            mixer->UpdateBit1(M_RUN_TM_LR0, M_RUN_TM_LR1, M_RUN_TM_LR2, M_RUN_TM_TH1, M_RUN_TM_AR1);

            statePredictor  = & model->Run.Exponent.StateModel[state][0];
            charPredictor   = & model->Run.Exponent.CharModel[currentChar][0];
            staticPredictor = & model->Run.Exponent.StaticModel[0];
            mixer           = & model->mixerOfRunExponent[history < 1 ? 1 : history][1];

            int bitRunSize = 1;
            while (true)
            {
                if (coder.DecodeBit(mixer->Mixup(*charPredictor, *statePredictor, *staticPredictor)))
                {
                    ProbabilityCounter::UpdateBit1(*statePredictor,  M_RUN_ES_TH1, M_RUN_ES_AR1); statePredictor++;
                    ProbabilityCounter::UpdateBit1(*charPredictor,   M_RUN_EC_TH1, M_RUN_EC_AR1); charPredictor++;
                    ProbabilityCounter::UpdateBit1(*staticPredictor, M_RUN_EP_TH1, M_RUN_EP_AR1); staticPredictor++;
                    mixer->UpdateBit1(M_RUN_EM_LR0, M_RUN_EM_LR1, M_RUN_EM_LR2, M_RUN_EM_TH1, M_RUN_EM_AR1);
                    bitRunSize++; mixer = & model->mixerOfRunExponent[history < bitRunSize ? bitRunSize : history][bitRunSize];
                }
                else
                {
                    ProbabilityCounter::UpdateBit0(*statePredictor,  M_RUN_ES_TH0, M_RUN_ES_AR0);
                    ProbabilityCounter::UpdateBit0(*charPredictor,   M_RUN_EC_TH0, M_RUN_EC_AR0);
                    ProbabilityCounter::UpdateBit0(*staticPredictor, M_RUN_EP_TH0, M_RUN_EP_AR0);
                    mixer->UpdateBit0(M_RUN_EM_LR0, M_RUN_EM_LR1, M_RUN_EM_LR2, M_RUN_EM_TH0, M_RUN_EM_AR0);
                    break;
                }
            }

            runHistory[currentChar] = (runHistory[currentChar] + 3 * bitRunSize + 3) >> 2;

            statePredictor  = & model->Run.Mantissa[bitRunSize].StateModel[state][0];
            charPredictor   = & model->Run.Mantissa[bitRunSize].CharModel[currentChar][0];
            staticPredictor = & model->Run.Mantissa[bitRunSize].StaticModel[0];
            mixer           = & model->mixerOfRunMantissa[bitRunSize];

            for (int context = 1, bit = bitRunSize - 1; bit >= 0; --bit)
            {
                if (coder.DecodeBit(mixer->Mixup(charPredictor[context], statePredictor[context], staticPredictor[context])))
                {
                    ProbabilityCounter::UpdateBit1(statePredictor[context],  M_RUN_MS_TH1, M_RUN_MS_AR1);
                    ProbabilityCounter::UpdateBit1(charPredictor[context],   M_RUN_MC_TH1, M_RUN_MC_AR1);
                    ProbabilityCounter::UpdateBit1(staticPredictor[context], M_RUN_MP_TH1, M_RUN_MP_AR1);
                    mixer->UpdateBit1(M_RUN_MM_LR0, M_RUN_MM_LR1, M_RUN_MM_LR2, M_RUN_MM_TH1, M_RUN_MM_AR1);
                    runSize += runSize + 1; if (bitRunSize <= 5) context += context + 1; else context++;
                }
                else
                {
                    ProbabilityCounter::UpdateBit0(statePredictor[context],  M_RUN_MS_TH0, M_RUN_MS_AR0);
                    ProbabilityCounter::UpdateBit0(charPredictor[context],   M_RUN_MC_TH0, M_RUN_MC_AR0);
                    ProbabilityCounter::UpdateBit0(staticPredictor[context], M_RUN_MP_TH0, M_RUN_MP_AR0);
                    mixer->UpdateBit0(M_RUN_MM_LR0, M_RUN_MM_LR1, M_RUN_MM_LR2, M_RUN_MM_TH0, M_RUN_MM_AR0);
                    runSize += runSize; if (bitRunSize <= 5) context += context; else context++;
                }
            }

        }
        else
        {
            runHistory[currentChar] = (runHistory[currentChar] + 2) >> 2;
            ProbabilityCounter::UpdateBit0(*statePredictor,  M_RUN_TS_TH0, M_RUN_TS_AR0);
            ProbabilityCounter::UpdateBit0(*charPredictor,   M_RUN_TC_TH0, M_RUN_TC_AR0);
            ProbabilityCounter::UpdateBit0(*staticPredictor, M_RUN_TP_TH0, M_RUN_TP_AR0);
            mixer->UpdateBit0(M_RUN_TM_LR0, M_RUN_TM_LR1, M_RUN_TM_LR2, M_RUN_TM_TH0, M_RUN_TM_AR0);
        }

        contextRank0 = ((contextRank0 << 1) | (rank == 0   ? 1    : 0)) & 0x7;
        contextRank4 = ((contextRank4 << 2) | (rank < 3    ? rank : 3)) & 0xff;
        contextRun   = ((contextRun   << 1) | (runSize < 3 ? 1    : 0)) & 0xf;

        for (; runSize > 0; --runSize) output[i++] = currentChar;
    }

    return n;
}

#endif

#if defined(QLFC_STATIC_DECODE_FUNCTION_NAME)

int QLFC_STATIC_DECODE_FUNCTION_NAME (const unsigned char * input, unsigned char * output, QlfcStatisticalModel1 * model)
{
    RangeCoder coder;

    unsigned char ALIGNED(64) MTFTable[ALPHABET_SIZE];

    bsc_qlfc_init_model(model);

    int contextRank0 = 0;
    int contextRank4 = 0;
    int contextRun   = 0;
    int maxRank      = 7;
    int avgRank      = 0;

    unsigned char rankHistory[ALPHABET_SIZE], runHistory[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i)
    {
        rankHistory[i] = runHistory[i] = 0;
    }

    coder.InitDecoder(input);
    int n = (int)coder.DecodeWord();

    unsigned char usedChar[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i) usedChar[i] = 0;

    int prevChar = -1;
    for (int rank = 0; rank < ALPHABET_SIZE; ++rank)
    {
        int currentChar = 0;

        for (int bit = 7; bit >= 0; --bit)
        {
            bool bit0 = false, bit1 = false;

            for (int c = 0; c < ALPHABET_SIZE; ++c)
            {
                if (c == prevChar || usedChar[c] == 0)
                {
                    if (currentChar == (c >> (bit + 1)))
                    {
                        if (c & (1 << bit)) bit1 = true; else bit0 = true;
                        if (bit0 && bit1) break;
                    }
                }
            }

            if (bit0 && bit1)
            {
                currentChar += currentChar + coder.DecodeBit();
            }
            else
            {
                if (bit0) currentChar += currentChar + 0;
                if (bit1) currentChar += currentChar + 1;
            }
        }

        MTFTable[rank] =  currentChar;

        if (currentChar == prevChar)
        {
            maxRank = bsc_bit_scan_reverse(rank - 1);
            break;
        }

        prevChar = currentChar; usedChar[currentChar] = 1;
    }

    for (int i = 0; i < n;)
    {
        int                 currentChar     =   MTFTable[0];
        int                 history         =   rankHistory[currentChar];
        int                 state           =   model_rank_state(contextRank4, contextRun, history);

        short * RESTRICT    statePredictor  = & model->Rank.StateModel[state];
        short * RESTRICT    charPredictor   = & model->Rank.CharModel[currentChar];
        short * RESTRICT    staticPredictor = & model->Rank.StaticModel;

        int rank = 1;
        if (avgRank < 32)
        {
            if (coder.DecodeBit((*charPredictor * F_RANK_TM_LR0 + *statePredictor * F_RANK_TM_LR1 + *staticPredictor * F_RANK_TM_LR2) >> 5))
            {
                ProbabilityCounter::UpdateBit1(*statePredictor,  F_RANK_TS_TH1, F_RANK_TS_AR1);
                ProbabilityCounter::UpdateBit1(*charPredictor,   F_RANK_TC_TH1, F_RANK_TC_AR1);
                ProbabilityCounter::UpdateBit1(*staticPredictor, F_RANK_TP_TH1, F_RANK_TP_AR1);

                statePredictor  = & model->Rank.Exponent.StateModel[state][0];
                charPredictor   = & model->Rank.Exponent.CharModel[currentChar][0];
                staticPredictor = & model->Rank.Exponent.StaticModel[0];

                int bitRankSize = 1;
                while (true)
                {
                    if (bitRankSize == maxRank) break;
                    if (coder.DecodeBit((*charPredictor * F_RANK_EM_LR0 + *statePredictor * F_RANK_EM_LR1 + *staticPredictor * F_RANK_EM_LR2) >> 5))
                    {
                        ProbabilityCounter::UpdateBit1(*statePredictor,  F_RANK_ES_TH1, F_RANK_ES_AR1); statePredictor++;
                        ProbabilityCounter::UpdateBit1(*charPredictor,   F_RANK_EC_TH1, F_RANK_EC_AR1); charPredictor++;
                        ProbabilityCounter::UpdateBit1(*staticPredictor, F_RANK_EP_TH1, F_RANK_EP_AR1); staticPredictor++;
                        bitRankSize++;
                    }
                    else
                    {
                        ProbabilityCounter::UpdateBit0(*statePredictor,  F_RANK_ES_TH0, F_RANK_ES_AR0);
                        ProbabilityCounter::UpdateBit0(*charPredictor,   F_RANK_EC_TH0, F_RANK_EC_AR0);
                        ProbabilityCounter::UpdateBit0(*staticPredictor, F_RANK_EP_TH0, F_RANK_EP_AR0);
                        break;
                    }
                }

                rankHistory[currentChar] = bitRankSize;

                statePredictor  = & model->Rank.Mantissa[bitRankSize].StateModel[state][0];
                charPredictor   = & model->Rank.Mantissa[bitRankSize].CharModel[currentChar][0];
                staticPredictor = & model->Rank.Mantissa[bitRankSize].StaticModel[0];

                for (int bit = bitRankSize - 1; bit >= 0; --bit)
                {
                    unsigned int b = (unsigned int)coder.DecodeBit((charPredictor[rank] * F_RANK_MM_LR0 + statePredictor[rank] * F_RANK_MM_LR1 + staticPredictor[rank] * F_RANK_MM_LR2) >> 5);

                    ProbabilityCounter::UpdateBit(b, statePredictor[rank],  F_RANK_MS_TH0, F_RANK_MS_AR0, F_RANK_MS_TH1, F_RANK_MS_AR1);
                    ProbabilityCounter::UpdateBit(b, charPredictor[rank],   F_RANK_MC_TH0, F_RANK_MC_AR0, F_RANK_MC_TH1, F_RANK_MC_AR1);
                    ProbabilityCounter::UpdateBit(b, staticPredictor[rank], F_RANK_MP_TH0, F_RANK_MP_AR0, F_RANK_MP_TH1, F_RANK_MP_AR1);

                    rank += rank + b;
                }
            }
            else
            {
                rankHistory[currentChar] = 0;
                ProbabilityCounter::UpdateBit0(*statePredictor,  F_RANK_TS_TH0, F_RANK_TS_AR0);
                ProbabilityCounter::UpdateBit0(*charPredictor,   F_RANK_TC_TH0, F_RANK_TC_AR0);
                ProbabilityCounter::UpdateBit0(*staticPredictor, F_RANK_TP_TH0, F_RANK_TP_AR0);
            }
        }
        else
        {
            statePredictor  = & model->Rank.Escape.StateModel[state][0];
            charPredictor   = & model->Rank.Escape.CharModel[currentChar][0];
            staticPredictor = & model->Rank.Escape.StaticModel[0];

            rank = 0;
            for (int context = 1, bit = maxRank; bit >= 0; --bit)
            {
                unsigned int b = (unsigned int)coder.DecodeBit((charPredictor[context] * F_RANK_PM_LR0 + statePredictor[context] * F_RANK_PM_LR1 + staticPredictor[context] * F_RANK_PM_LR2) >> 5);

                ProbabilityCounter::UpdateBit(b, statePredictor[context],  F_RANK_PS_TH0, F_RANK_PS_AR0, F_RANK_PS_TH1, F_RANK_PS_AR1);
                ProbabilityCounter::UpdateBit(b, charPredictor[context],   F_RANK_PC_TH0, F_RANK_PC_AR0, F_RANK_PC_TH1, F_RANK_PC_AR1);
                ProbabilityCounter::UpdateBit(b, staticPredictor[context], F_RANK_PP_TH0, F_RANK_PP_AR0, F_RANK_PP_TH1, F_RANK_PP_AR1);
                
                context += context + b; rank += rank + b;
            }

            rankHistory[currentChar] = (unsigned char)bsc_bit_scan_reverse(rank);
        }

        {
#if LIBBSC_CPU_FEATURE >= LIBBSC_CPU_FEATURE_SSE41
            __m128i * MTFTable_p = (__m128i *)&MTFTable[rank & (-16)];
            __m128i r = _mm_load_si128(MTFTable_p); _mm_store_si128(MTFTable_p, _mm_shuffle_epi8(_mm_insert_epi8(r, currentChar, 0), _mm_load_si128((const __m128i *)&rank16_shuffle[rank & 15][0])));

            while ((--MTFTable_p) >= (__m128i *)MTFTable)
            {
                __m128i t = _mm_load_si128(MTFTable_p); _mm_store_si128(MTFTable_p, _mm_alignr_epi8(r, t, 1)); r = t;
            }
#elif LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_A64
            uint8x16_t* MTFTable_p = (uint8x16_t*)&MTFTable[rank & (-16)];
            uint8x16_t r = vld1q_u8((const unsigned char*)MTFTable_p); vst1q_u8((unsigned char*)MTFTable_p, vqtbl1q_u8(vsetq_lane_u8((unsigned char)currentChar, r, 0), vld1q_u8((const unsigned char *)&rank16_shuffle[rank & 15][0])));

            while ((--MTFTable_p) >= (uint8x16_t*)MTFTable)
            {
                uint8x16_t t = vld1q_u8((const unsigned char*)MTFTable_p); vst1q_u8((unsigned char*)MTFTable_p, vextq_u8(t, r, 1)); r = t;
            }
#else
            for (int r = 0; r < rank; ++r)
            {
                MTFTable[r] = MTFTable[r + 1];
            }
            MTFTable[rank] = currentChar;
#endif
        }

        avgRank         =   (avgRank * 124 + rank * 4) >> 7;
        rank            =   rank - 1;
        history         =   runHistory[currentChar];
        state           =   model_run_state(contextRank0, contextRun, rank, history);
        statePredictor  = & model->Run.StateModel[state];
        charPredictor   = & model->Run.CharModel[currentChar];
        staticPredictor = & model->Run.StaticModel;

        int runSize = 1;
        if (coder.DecodeBit((*charPredictor * F_RUN_TM_LR0 + *statePredictor * F_RUN_TM_LR1 + *staticPredictor * F_RUN_TM_LR2) >> 5))
        {
            ProbabilityCounter::UpdateBit1(*statePredictor,  F_RUN_TS_TH1, F_RUN_TS_AR1);
            ProbabilityCounter::UpdateBit1(*charPredictor,   F_RUN_TC_TH1, F_RUN_TC_AR1);
            ProbabilityCounter::UpdateBit1(*staticPredictor, F_RUN_TP_TH1, F_RUN_TP_AR1);

            statePredictor  = & model->Run.Exponent.StateModel[state][0];
            charPredictor   = & model->Run.Exponent.CharModel[currentChar][0];
            staticPredictor = & model->Run.Exponent.StaticModel[0];

            int bitRunSize = 1;
            while (true)
            {
                if (coder.DecodeBit((*charPredictor * F_RUN_EM_LR0 + *statePredictor * F_RUN_EM_LR1 + *staticPredictor * F_RUN_EM_LR2) >> 5))
                {
                    ProbabilityCounter::UpdateBit1(*statePredictor,  F_RUN_ES_TH1, F_RUN_ES_AR1); statePredictor++;
                    ProbabilityCounter::UpdateBit1(*charPredictor,   F_RUN_EC_TH1, F_RUN_EC_AR1); charPredictor++;
                    ProbabilityCounter::UpdateBit1(*staticPredictor, F_RUN_EP_TH1, F_RUN_EP_AR1); staticPredictor++;
                    bitRunSize++;
                }
                else
                {
                    ProbabilityCounter::UpdateBit0(*statePredictor,  F_RUN_ES_TH0, F_RUN_ES_AR0);
                    ProbabilityCounter::UpdateBit0(*charPredictor,   F_RUN_EC_TH0, F_RUN_EC_AR0);
                    ProbabilityCounter::UpdateBit0(*staticPredictor, F_RUN_EP_TH0, F_RUN_EP_AR0);
                    break;
                }
            }

            runHistory[currentChar] = (runHistory[currentChar] + 3 * bitRunSize + 3) >> 2;

            statePredictor  = & model->Run.Mantissa[bitRunSize].StateModel[state][0];
            charPredictor   = & model->Run.Mantissa[bitRunSize].CharModel[currentChar][0];
            staticPredictor = & model->Run.Mantissa[bitRunSize].StaticModel[0];

            for (int context = 1, bit = bitRunSize - 1; bit >= 0; --bit)
            {
                unsigned int b = (unsigned int)coder.DecodeBit((charPredictor[context] * F_RUN_MM_LR0 + statePredictor[context] * F_RUN_MM_LR1 + staticPredictor[context] * F_RUN_MM_LR2) >> 5);

                ProbabilityCounter::UpdateBit(b, statePredictor[context],  F_RUN_MS_TH0, F_RUN_MS_AR0, F_RUN_MS_TH1, F_RUN_MS_AR1);
                ProbabilityCounter::UpdateBit(b, charPredictor[context],   F_RUN_MC_TH0, F_RUN_MC_AR0, F_RUN_MC_TH1, F_RUN_MC_AR1);
                ProbabilityCounter::UpdateBit(b, staticPredictor[context], F_RUN_MP_TH0, F_RUN_MP_AR0, F_RUN_MP_TH1, F_RUN_MP_AR1);
                
                runSize += runSize + b; int ctx = context + context + b; context++; if (bitRunSize <= 5) { context = ctx; }
            }
        }
        else
        {
            runHistory[currentChar] = (runHistory[currentChar] + 2) >> 2;
            ProbabilityCounter::UpdateBit0(*statePredictor,  F_RUN_TS_TH0, F_RUN_TS_AR0);
            ProbabilityCounter::UpdateBit0(*charPredictor,   F_RUN_TC_TH0, F_RUN_TC_AR0);
            ProbabilityCounter::UpdateBit0(*staticPredictor, F_RUN_TP_TH0, F_RUN_TP_AR0);
        }

        contextRank0 = ((contextRank0 << 1) | (rank == 0   ? 1    : 0)) & 0x7;
        contextRank4 = ((contextRank4 << 2) | (rank < 3    ? rank : 3)) & 0xff;
        contextRun   = ((contextRun   << 1) | (runSize < 3 ? 1    : 0)) & 0xf;
        
        for (; runSize > 0; --runSize) output[i++] = currentChar;
    }

    return n;
}

#endif

#if defined(QLFC_STATIC_DECODE_FUNCTION_NAME)

int QLFC_FAST_DECODE_FUNCTION_NAME (const unsigned char * input, unsigned char * output, QlfcStatisticalModel2 * model)
{
    unsigned char ALIGNED(64) MTFTable[ALPHABET_SIZE];

    bsc_qlfc_init_model(model);

    RangeCoder coder;
    coder.InitDecoder(input);

    int n = (int)coder.DecodeWord();

    unsigned char usedChar[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i) usedChar[i] = 0;

    int prevChar = -1;
    for (int rank = 0; rank < ALPHABET_SIZE; ++rank)
    {
        int currentChar = 0;

        for (int bit = 7; bit >= 0; --bit)
        {
            bool bit0 = false, bit1 = false;

            for (int c = 0; c < ALPHABET_SIZE; ++c)
            {
                if (c == prevChar || usedChar[c] == 0)
                {
                    if (currentChar == (c >> (bit + 1)))
                    {
                        if (c & (1 << bit)) bit1 = true; else bit0 = true;
                        if (bit0 && bit1) break;
                    }
                }
            }

            if (bit0 && bit1)
            {
                currentChar += currentChar + coder.DecodeBit<1>(1);
            }
            else
            {
                if (bit0) currentChar += currentChar + 0;
                if (bit1) currentChar += currentChar + 1;
            }
        }

        MTFTable[rank] =  currentChar;

        if (currentChar == prevChar)
        {
            break;
        }

        prevChar = currentChar; usedChar[currentChar] = 1;
    }

    const unsigned char * outputEnd = output + n;

    for (; output < outputEnd; )
    {
        unsigned int currentChar = MTFTable[0];

        {
            short * RESTRICT predictor = &model->Rank.Exponent[currentChar][0];

            int p = predictor[0];
            if (coder.PeakBit<13>(p))
            {
                ProbabilityCounter::UpdateBit<4>(predictor[0], 83);
                coder.DecodeBit1<13>(p);

                int bitRankSize = 1;
                while (bitRankSize < 7)
                {
                    p = predictor[bitRankSize];
                    if (coder.PeakBit<13>(p))
                    {
                        ProbabilityCounter::UpdateBit<4>(predictor[bitRankSize], 122);
                        bitRankSize++;
                        coder.DecodeBit1<13>(p);
                    }
                    else
                    {
                        ProbabilityCounter::UpdateBit<4>(predictor[bitRankSize], 8114);
                        coder.DecodeBit0<13>(p);
                        break;
                    }
                }

                predictor = & model->Rank.Mantissa[currentChar][bitRankSize][0];

                unsigned int rank = 1;
                while (--bitRankSize >= 0)
                {
                    unsigned int b = coder.DecodeBit<13>(predictor[rank]);
                    ProbabilityCounter::UpdateBit<7>(b, predictor[rank], 7999, 235);
                    rank += rank + b;
                }

                {
#if LIBBSC_CPU_FEATURE >= LIBBSC_CPU_FEATURE_SSE41
                    __m128i * MTFTable_p = (__m128i *)&MTFTable[rank & (-16)];
                    __m128i r = _mm_load_si128(MTFTable_p); _mm_store_si128(MTFTable_p, _mm_shuffle_epi8(_mm_insert_epi8(r, currentChar, 0), _mm_load_si128((const __m128i *)&rank16_shuffle[rank & 15][0])));

                    while ((--MTFTable_p) >= (__m128i *)MTFTable)
                    {
                        __m128i t = _mm_load_si128(MTFTable_p); _mm_store_si128(MTFTable_p, _mm_alignr_epi8(r, t, 1)); r = t;
                    }
#elif LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_A64
                    uint8x16_t * MTFTable_p = (uint8x16_t *)&MTFTable[rank & (-16)];
                    uint8x16_t r = vld1q_u8((const unsigned char *)MTFTable_p); vst1q_u8((unsigned char *)MTFTable_p, vqtbl1q_u8(vsetq_lane_u8((unsigned char)currentChar, r, 0), vld1q_u8((const unsigned char *)&rank16_shuffle[rank & 15][0])));
                    
                    while ((--MTFTable_p) >= (uint8x16_t *)MTFTable)
                    {
                        uint8x16_t t = vld1q_u8((const unsigned char *)MTFTable_p); vst1q_u8((unsigned char *)MTFTable_p, vextq_u8(t, r, 1)); r = t;
                    }
#else
                    for (unsigned int r = 0; r < rank; ++r)
                    {
                        MTFTable[r] = MTFTable[r + 1];
                    }

                    MTFTable[rank] = currentChar;
#endif
                }
            }
            else
            {
                MTFTable[0] = MTFTable[1]; MTFTable[1] = currentChar; ProbabilityCounter::UpdateBit<4>(predictor[0], 8016); coder.DecodeBit0<13>(p);
            }
        }

        {
            short * RESTRICT predictor = &model->Run.Exponent[currentChar][0];

            int p = predictor[0];
            if (coder.PeakBit<11>(p))
            {
                ProbabilityCounter::UpdateBit<5>(predictor[0], 42);
                coder.DecodeBit1<11>(p);

                int bitRunSize = 1;
                while (true)
                {
                    p = predictor[bitRunSize];
                    if (coder.PeakBit<11>(p))
                    {
                        ProbabilityCounter::UpdateBit<4>(predictor[bitRunSize], 142);
                        bitRunSize++;
                        coder.DecodeBit1<11>(p);
                    }
                    else
                    {
                        ProbabilityCounter::UpdateBit<4>(predictor[bitRunSize], 1962);
                        coder.DecodeBit0<11>(p);
                        break;
                    }
                }

                predictor = &model->Run.Mantissa[currentChar][bitRunSize][0];

                if (bitRunSize <= 5)
                {
                    unsigned int runSize = 1;
                    while (--bitRunSize >= 0)
                    {
                        unsigned int b = coder.DecodeBit<11>(predictor[runSize]);
                        ProbabilityCounter::UpdateBit<6>(b, predictor[runSize], 1951, 147);
                        runSize += runSize + b;
                    }

                    for (; runSize > 0; --runSize) { *output++ = currentChar; }
                }
                else
                {
                    unsigned int runSize = 1;
                    for (int context = 1; context <= bitRunSize; ++context)
                    {
                        unsigned int b = coder.DecodeBit<11>(predictor[context]);
                        ProbabilityCounter::UpdateBit<5>(b, predictor[context], 1987, 46);
                        runSize += runSize + b;
                    }

                    for (; runSize > 0; --runSize) { *output++ = currentChar; }
                }
            }
            else
            {
                *output++ = currentChar; ProbabilityCounter::UpdateBit<5>(predictor[0], 2025); coder.DecodeBit0<11>(p);
            }
        }
    }

    return n;
}

#endif

#if !defined(LIBBSC_DYNAMIC_CPU_DISPATCH) || LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_SSE2

int bsc_qlfc_init(int features)
{
    return bsc_qlfc_init_static_model();
}

int bsc_qlfc_static_encode_block(const unsigned char * input, unsigned char * output, int inputSize, int outputSize)
{
    if (QlfcStatisticalModel1 * model = (QlfcStatisticalModel1 *)bsc_malloc(sizeof(QlfcStatisticalModel1)))
    {
        if (unsigned char * buffer = (unsigned char *)bsc_malloc(inputSize * sizeof(unsigned char)))
        {
            int result = bsc_qlfc_static_encode(input, output, buffer, inputSize, outputSize, model);

            bsc_free(buffer); bsc_free(model);

            return result;
        };
        bsc_free(model);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_qlfc_adaptive_encode_block(const unsigned char * input, unsigned char * output, int inputSize, int outputSize)
{
    if (QlfcStatisticalModel1 * model = (QlfcStatisticalModel1 *)bsc_malloc(sizeof(QlfcStatisticalModel1)))
    {
        if (unsigned char * buffer = (unsigned char *)bsc_malloc(inputSize * sizeof(unsigned char)))
        {
            int result = bsc_qlfc_adaptive_encode(input, output, buffer, inputSize, outputSize, model);

            bsc_free(buffer); bsc_free(model);

            return result;
        };
        bsc_free(model);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_qlfc_fast_encode_block(const unsigned char * input, unsigned char * output, int inputSize, int outputSize)
{
    if (QlfcStatisticalModel2 * model = (QlfcStatisticalModel2 *)bsc_malloc(sizeof(QlfcStatisticalModel2)))
    {
        if (unsigned char * buffer = (unsigned char *)bsc_malloc(inputSize * sizeof(unsigned char)))
        {
            int result = bsc_qlfc_fast_encode(input, output, buffer, inputSize, outputSize, model);

            bsc_free(buffer); bsc_free(model);

            return result;
        };
        bsc_free(model);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_qlfc_static_decode_block(const unsigned char * input, unsigned char * output)
{
    if (QlfcStatisticalModel1 * model = (QlfcStatisticalModel1 *)bsc_malloc(sizeof(QlfcStatisticalModel1)))
    {
        int result = bsc_qlfc_static_decode(input, output, model);

        bsc_free(model);

        return result;
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_qlfc_adaptive_decode_block(const unsigned char * input, unsigned char * output)
{
    if (QlfcStatisticalModel1 * model = (QlfcStatisticalModel1 *)bsc_malloc(sizeof(QlfcStatisticalModel1)))
    {
        int result = bsc_qlfc_adaptive_decode(input, output, model);

        bsc_free(model);

        return result;
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_qlfc_fast_decode_block(const unsigned char * input, unsigned char * output)
{
    if (QlfcStatisticalModel2 * model = (QlfcStatisticalModel2 *)bsc_malloc(sizeof(QlfcStatisticalModel2)))
    {
        int result = bsc_qlfc_fast_decode(input, output, model);

        bsc_free(model);

        return result;
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

#endif

/*-----------------------------------------------------------*/
/* End                                              qlfc.cpp */
/*-----------------------------------------------------------*/

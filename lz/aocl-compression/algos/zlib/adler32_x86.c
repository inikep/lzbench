/**
 * Copyright (C) 2022-2026, Advanced Micro Devices. All rights reserved.
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

#include "utils/utils.h"
#include <immintrin.h>
#include <stdint.h>
#include "zutil.h"

#ifdef AOCL_ZLIB_OPT
#include "aocl_zlib_setup.h"
#include "aocl_zlib_fmv_utils.h"
#include "aocl_zlib_dispatch_variants.h"
/* Shared FMV selection helper and variant entry layouts are centralized in
 * aocl_zlib_fmv_utils.h and aocl_zlib_dispatch_variants.h. */
/* Dynamic dispatcher setup function for native APIs.
 * All native APIs that call aocl optimized functions within their call stack,
 * must call AOCL_SETUP_NATIVE() at the start of the function. This sets up 
 * appropriate code paths to take based on user defined environment variables,
 * as well as cpu instruction set supported by the runtime machine. */
static void aocl_setup_native(void);
#define AOCL_SETUP_NATIVE() aocl_setup_native()

#if defined(__GNUC__) && (__GNUC__ < 11)
#define MM256_EXTRACT_FIRST_INT32(x) _mm256_extract_epi32(x, 0)
#else
#define MM256_EXTRACT_FIRST_INT32(x) _mm256_cvtsi256_si32(x)
#endif

static int setup_ok_zlib_adler = 0; // flag to indicate status of dynamic dispatcher setup
#ifndef AOCL_ENABLE_THREADS
static atomic_flag setup_zlib_adler = ATOMIC_FLAG_INIT;
#endif /* AOCL_ENABLE_THREADS */

/* Largest prime smaller than 65536 */
#define BASE 65521U
/* NMAX is the largest n such that 255n(n+1)/2 + (n+1)(BASE-1) <= 2^32-1 */
#define NMAX 5552

#define ITER_SZ 64

#define DO1(buf,i)  {sum_A += (buf)[i]; sum_B += sum_A;}
#define DO2(buf,i)  DO1(buf,i); DO1(buf,i+1);
#define DO4(buf,i)  DO2(buf,i); DO2(buf,i+2);
#define DO8(buf)  DO4(buf,0); DO4(buf,4);

static inline uint32_t adler32_with_copy(uint32_t adler, Bytef *dst, const Bytef *buf, z_size_t len, const short copy)
{
    if(copy)
    {
        zmemcpy(dst, buf, len);
    }
    return adler32(adler, buf, len);
}

/* Function pointer holding the optimized variant as per the detected CPU
 * features */
static uint32_t (*adler32_x86_with_copy_fp)(uint32_t adler, Bytef* dst, const Bytef* buf, z_size_t len, const short copy) =
adler32_with_copy;

// This function separation prevents compiler from generating VZEROUPPER instruction
// because of transition from VEX to Non-VEX code resulting in performance drop
static inline uint32_t adler32_rem_len_with_copy(uint32_t adler, Bytef *dst, const Bytef *buf, z_size_t len, const short copy)
{
    uint32_t sum_A = adler & 0xffff;
    uint32_t sum_B = adler >> 16;
    if (len) {
        while (len >= 8)
        {
            len -= 8;
            DO8(buf);
            if(copy)
            {
                zmemcpy(dst, buf, 8);
                dst += 8;
            }
            buf += 8;
        }

        while (len--)
        {
            if(copy)
            {
                *dst++ = *buf;
            }
            sum_B += (sum_A += *buf++);
        }

        if (sum_A >= BASE)
            sum_A -= BASE;
        sum_B %= BASE;
    }

    return sum_A | (sum_B << 16);
}

#ifdef AOCL_ZLIB_AVX_OPT
__attribute__((__target__("avx"))) // uses SSSE3 intrinsics
static inline uint32_t adler32_x86_avx_with_copy(uint32_t adler, Bytef *dst, const Bytef *buf, z_size_t len, const short copy)
{
    AOCL_SIMD_UNIT_TEST(DEBUG, logCtx, "Enter");
    uint32_t sum_A = adler & 0xffff;
    uint32_t sum_B = adler >> 16;

    z_size_t  itr_cnt = len / ITER_SZ;
    len -= itr_cnt * ITER_SZ;

    // coeff1[16]: {64, 63, 62, ..., 49}
    const __m128i coeff1 = _mm_setr_epi8(64,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49);
    // coeff2[16]: {48, 47, 46, ..., 33}
    const __m128i coeff2 = _mm_setr_epi8(48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33);
    // coeff3[16]: {32, 31, 30, ..., 17}
    const __m128i coeff3 = _mm_setr_epi8(32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17);
    // coeff4[16]: {16, 15, 14, ..., 1}
    const __m128i coeff4 = _mm_setr_epi8(16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
    const __m128i zero = _mm_setzero_si128();
    // octa_ones[8]: {1, 1, ..., 1}
    const __m128i octa_ones = _mm_set1_epi16(1);
    
    while (itr_cnt)
    {
        __m128i vos, vcs, vbs, batch1, batch2, mad0, mad1;
        z_size_t n = NMAX / ITER_SZ; 
        if (n > itr_cnt)
            n = itr_cnt;
        itr_cnt -= n;

        // vos[4]: {sum_A * n, 0, 0, 0}
        vos = _mm_set_epi32(0, 0, 0, sum_A * n);
        // vcs[4]: {sum_B, 0, 0, 0}
        vcs = _mm_set_epi32(0, 0, 0, sum_B);
        vbs = zero;

        while(n--)
        {
        /*
            This loop works on 64 byte data in single iteration and stores partial results that helps in computing 
            two 16-bit checksums after exit

            sum_A = sum_A + B1 + B2 + .. + B64
            sum_B = 64*sum_A + 64*B1 + 63*B2 + .. +1*B64

            vbs : stores sum_A's partial computation of consecutive bytes in adjacent four 32-bit numbers
            vcs : stores sum_B's partial computation of consecutive bytes in adjacent four 32-bit numbers
            vos : accumulating vbs results per iteration for future calculation of sum_B
        */
            // batch1[16]: {B1, B2, ..., B16}
            batch1 = _mm_lddqu_si128((__m128i*)(buf));
            // batch2[16]: {B17, B18, ..., B32}
            batch2 = _mm_lddqu_si128((__m128i*)(buf + 16));

            if(copy)
            {
                _mm_storeu_si128((__m128i*)dst, batch1);
                _mm_storeu_si128((__m128i*)(dst + 16), batch2);
                dst += 32;
            }

            // vos[4]: {vos[0] + vbs[0], ..., vos[3]}
            vos = _mm_add_epi32(vos, vbs);
            // vbs[4]: {vbs[0] + S[B1, ..., B8], 0, vbs[2] + S[B9, ..., B16], 0}
            vbs = _mm_add_epi32(vbs, _mm_sad_epu8(batch1, zero));
            // mad0[8]: {64*B1 + 63*B2, ..., 50*B15 + 49*B16}
            mad0 = _mm_maddubs_epi16(batch1, coeff1);
            // vcs[4]: {vcs[0] + (64*B1 + 63*B2) + (62*B3 + 61*B4), ..., vcs[3] + (52*B13 + 51*B14) + (50*B15 + 49*B16)}
            vcs = _mm_add_epi32(vcs, _mm_madd_epi16(mad0, octa_ones));

            // vbs[4]: {vbs[0] + S[B17, ..., B24], 0, vbs[2] + S[B25, ..., B32], 0}
            vbs = _mm_add_epi32(vbs, _mm_sad_epu8(batch2, zero));
            // mad1[8]: {48*B17 + 47*B18, ..., 34*B31 + 33*B32}
            mad1 = _mm_maddubs_epi16(batch2, coeff2);
            // vcs[4]: {vcs[0] + (48*B17 + 47*B18) + (46*B19 + 45*B20), ..., vcs[3] + (36*B29 + 35*B30) + (34*B31 + 33*B32)}
            vcs = _mm_add_epi32(vcs, _mm_madd_epi16(mad1, octa_ones));

            // batch1: {B33, B34, ..., B48}
            batch1 = _mm_lddqu_si128((__m128i*)(buf + 32));
            // batch2: {B49, B50, ..., B64}
            batch2 = _mm_lddqu_si128((__m128i*)(buf + 48));

            if(copy)
            {
                _mm_storeu_si128((__m128i*)dst, batch1);
                _mm_storeu_si128((__m128i*)(dst + 16), batch2);
                dst += 32;
            }

            // vbs[4]: {vbs[0] + S[B33, ..., B40], 0, vbs[2] + S[B41, ..., B48], 0}
            vbs = _mm_add_epi32(vbs, _mm_sad_epu8(batch1, zero));
            // mad0[8]: {32*B33 + 31*B34, ..., 18*B47 + 17*B48}
            mad0 = _mm_maddubs_epi16(batch1, coeff3);
            // vcs[4]: {vcs[0] + (32*B33 + 31*B34) + (30*B35 + 29*B36), ..., vcs[3] + (18*B45 + 17*B46) + (16*B47 + 15*B48)}
            vcs = _mm_add_epi32(vcs, _mm_madd_epi16(mad0, octa_ones));

            // vbs[4]: {vbs[0] + S[B49, ..., B56], 0, vbs[2] + S[B57, ..., B64], 0}
            vbs = _mm_add_epi32(vbs, _mm_sad_epu8(batch2, zero));
            // mad1[8]: {16*B49 + 15*B50, ..., 2*B63 + 1*B64}
            mad1 = _mm_maddubs_epi16(batch2, coeff4);
            // vcs[4]: {vcs[0] + (16*B49 + 15*B50) + (14*B51 + 13*B52), ..., vcs[3] + (4*B61 + 3*B62) + (2*B63 + 1*B64)}
            vcs = _mm_add_epi32(vcs, _mm_madd_epi16(mad1, octa_ones));

            buf += ITER_SZ;
        }

        // Shuffling and adding vbs data to compute 64*n byte sum in lower 32-bit number
        vbs = _mm_add_epi32(vbs, _mm_shuffle_epi32(vbs, _MM_SHUFFLE(2,3,0,1)));
        vbs = _mm_add_epi32(vbs, _mm_shuffle_epi32(vbs, _MM_SHUFFLE(1,0,3,2)));

        sum_A += _mm_cvtsi128_si32(vbs);

        vcs = _mm_add_epi32(vcs, _mm_slli_epi32(vos, 6));
        // Shuffling and adding vcs data to accumulate sum_B in lower 32-bit number
        vcs = _mm_add_epi32(vcs, _mm_shuffle_epi32(vcs, _MM_SHUFFLE(2,3,0,1)));
        vcs = _mm_add_epi32(vcs, _mm_shuffle_epi32(vcs, _MM_SHUFFLE(1,0,3,2)));

        sum_B = _mm_cvtsi128_si32(vcs);

        sum_A %= BASE;
        sum_B %= BASE;
    }

    return adler32_rem_len_with_copy(sum_A | (sum_B << 16), dst, buf, len, copy);
}
#endif /* AOCL_ZLIB_AVX_OPT */

#ifdef AOCL_ZLIB_AVX2_OPT
__attribute__((__target__("avx2")))
static inline uint32_t adler32_x86_avx2_with_copy(uint32_t adler, Bytef *dst, const Bytef *buf, z_size_t len, const short copy)
{
    AOCL_SIMD_UNIT_TEST(DEBUG, logCtx, "Enter");
    uint32_t sum_A = adler & 0xffff;
    uint32_t sum_B = adler >> 16;

    z_size_t  itr_cnt = len / ITER_SZ;
    len -= itr_cnt * ITER_SZ;
    
    // coeff1[32]: {64, 63, 62, ..., 33}
    const __m256i coeff1 = _mm256_setr_epi8(64,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33);
    // coeff1[32]: {32, 31, 30, ..., 1}
    const __m256i coeff2 = _mm256_setr_epi8(32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
    const __m256i zero = _mm256_setzero_si256();
    // sixteen_ones[16]: {1, 1, ..., 1, 1}
    const __m256i sixteen_ones = _mm256_set1_epi16(1);

    while (itr_cnt)
    {
        __m256i vos, vcs, vbs, batch1, batch2, mad0, mad1, vbs_i;
        z_size_t n = NMAX / ITER_SZ; 
        if (n > itr_cnt)
            n = itr_cnt;
        itr_cnt -= n;

        // vos[8]: {sum_A * n, 0, 0, ..., 0}
        vos = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, sum_A * n);
        // vcs[8]: {sum_B, 0, 0, ..., 0}
        vcs = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, sum_B);
        vbs = zero;

        while(n--)
        {
            // batch1[32]: {B1, B2, ..., B32}
            batch1 = _mm256_lddqu_si256((__m256i*)(buf));
            // batch1[32]: {B33, B34, ..., B64}
            batch2 = _mm256_lddqu_si256((__m256i*)(buf + 32));
            if(copy)
            {
                _mm256_storeu_si256((__m256i*)dst, batch1);
                _mm256_storeu_si256((__m256i*)(dst + 32), batch2);
                dst += ITER_SZ;
            }
            // mad0[8]: {(64*B1 + 63*B2) + (62*B3 + 61*B4), ..., (36*B29 + 35*B30) + (34*B31 + 33*B32)}
            mad0 = _mm256_madd_epi16(_mm256_maddubs_epi16(batch1, coeff1), sixteen_ones);
            // mad1[8]: {(32*B33 + 31*B34) + (30*B35 + 29*B36), ..., (4*B61 + 3*B62) + (2*B63 + 1*B64)}
            mad1 = _mm256_madd_epi16(_mm256_maddubs_epi16(batch2, coeff2), sixteen_ones);

            // vcs[8]: {vcs[0] + mad0[0], ..., vcs[7] + mad0[7]}
            vcs = _mm256_add_epi32(vcs, mad0);
            // vcs[8]: {vcs[0] + mad1[0], ..., vcs[7] + mad1[7]}
            vcs = _mm256_add_epi32(vcs, mad1);

            // vos[8]: {vos[0] + vbs[0], ..., vos[7] + vbs[7]}
            vos = _mm256_add_epi32(vos, vbs);

            // vbs_i[8]: {S[B1, ...,B8] + S[B33, ...,B40], 0, ..., S[B25, ...,B32] + S[B57, ...,B64], 0}
            vbs_i = _mm256_add_epi32(_mm256_sad_epu8(batch1, zero),  _mm256_sad_epu8(batch2, zero));
            // vbs[8]: {vbs[0] + vbs_i[0], 0, ..., vbs[6] + vbs_i[6], 0}
            vbs = _mm256_add_epi32(vbs, vbs_i);

            buf += ITER_SZ;
        }
        // vbs[8]: A | 0 | B | 0 | C | 0 | D | 0 => A+B | 0+0 | B+A | 0+0 | C+D | 0+0 | D+C | 0+0
        vbs = _mm256_add_epi32(vbs, _mm256_shuffle_epi32(vbs, 206));
        // sum_A = A+B+C+D
        sum_A += MM256_EXTRACT_FIRST_INT32(vbs) + _mm_cvtsi128_si32(_mm256_extracti128_si256(vbs, 1));

        vcs = _mm256_add_epi32(vcs, _mm256_slli_epi32(vos, 6));
        // vcs[8]: A | B | C | D | E | F | G | H => A+C | B+D | C+A | D+B | E+G | F+H | G+E | H+F
        vcs = _mm256_add_epi32(vcs, _mm256_shuffle_epi32(vcs, 78));
        // vcs[8]: A+C | B+D | C+A | D+B | E+G | F+H | G+E | H+F => A+C+B+D | B+D+A+C | C+A+D+B | D+B+C+A | E+G+F+H | F+H+E+G | G+E+H+F | H+F+G+E
        vcs = _mm256_add_epi32(vcs, _mm256_shuffle_epi32(vcs, 177));
        // sum_B = A+C+B+D+E+G+F+H
        sum_B = MM256_EXTRACT_FIRST_INT32(vcs) + _mm_cvtsi128_si32(_mm256_extracti128_si256(vcs, 1));

        sum_A %= BASE;
        sum_B %= BASE;
    }
    return adler32_rem_len_with_copy(sum_A | (sum_B << 16), dst, buf, len, copy);
}
#endif /* AOCL_ZLIB_AVX2_OPT */

uint32_t ZLIB_INTERNAL adler32_x86_internal_with_copy(uint32_t sum_A, Bytef *dst, const Bytef* buf, z_size_t len, const short copy)
{
    if(buf == NULL)
        return 1;

    if (LIKELY(buf && len >= 32))
    {
        return adler32_x86_with_copy_fp(sum_A, dst, buf, len, copy);
    }
    return adler32_with_copy(sum_A, dst, buf, len, copy);
}

static inline void aocl_setup_adler32_fmv(int optOff, CpuFeatures cpuFeatures)
{
    if (UNLIKELY(optOff == 1))
    {
        adler32_x86_with_copy_fp = adler32_with_copy;
    }
    else
    {
        /* FMV variant table (highest priority first). */
        static const AoclZlibAdler32Variant adler32_variants[] = {
#ifdef AOCL_ZLIB_AVX2_OPT
            { FEATURE_AVX2, adler32_x86_avx2_with_copy },
#endif
#ifdef AOCL_ZLIB_AVX_OPT
            { FEATURE_AVX,  adler32_x86_avx_with_copy },
#endif
            { 0,            adler32_with_copy }
        };

        /* Select first compatible FMV variant for detected CPU features. */
        size_t variant_index = aocl_zlib_select_fmv_variant(
            adler32_variants,
            AOCL_ZLIB_ARRAY_SIZE(adler32_variants),
            sizeof(adler32_variants[0]),
            offsetof(AoclZlibAdler32Variant, required_features),
            cpuFeatures);

        if (variant_index < AOCL_ZLIB_ARRAY_SIZE(adler32_variants)) {
            adler32_x86_with_copy_fp = adler32_variants[variant_index].impl;
            return;
        }
    }
}


void ZLIB_INTERNAL aocl_setup_adler32(int optOff, CpuFeatures cpuFeatures){
    AOCL_ENTER_CRITICAL(setup_zlib_adler)
    if (!setup_ok_zlib_adler) {
        optOff = optOff ? 1 : get_disable_opt_flags(0);
        aocl_setup_adler32_fmv(optOff, cpuFeatures);
        setup_ok_zlib_adler = 1;
    }
    AOCL_EXIT_CRITICAL(setup_zlib_adler)
}

static void aocl_setup_native(void) {
    AOCL_ENTER_CRITICAL(setup_zlib_adler)
    if (!setup_ok_zlib_adler) {
        CpuFeatures cpuFeatures = Dispatcher_GetFeaturesFromEnv();
        int optOff = get_disable_opt_flags(0);
        aocl_setup_adler32_fmv(optOff, cpuFeatures);
        setup_ok_zlib_adler = 1;
    }
    AOCL_EXIT_CRITICAL(setup_zlib_adler)
}

void ZLIB_INTERNAL aocl_destroy_adler32(void) {
    AOCL_ENTER_CRITICAL(setup_zlib_adler)
    setup_ok_zlib_adler = 0;
    AOCL_EXIT_CRITICAL(setup_zlib_adler)
}

#endif /* AOCL_ZLIB_OPT */

/* This function intercepts non optimized code path and orchestrate
 * optimized code flow path */
uInt ZEXPORT adler32_x86(uInt sum_A, const Bytef* buf, z_size_t len)
{
#ifdef AOCL_ZLIB_OPT
    AOCL_SETUP_NATIVE();
    return adler32_x86_internal_with_copy(sum_A, Z_NULL, buf, len, 0);
#else
    return adler32(sum_A, buf, len);
#endif /* AOCL_ZLIB_OPT */
}

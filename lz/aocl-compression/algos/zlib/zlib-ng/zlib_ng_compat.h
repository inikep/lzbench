/**
 * Copyright (C) 2025-2026, Advanced Micro Devices. All rights reserved.
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

#ifndef ZLIB_NG_COMPAT_H
#define ZLIB_NG_COMPAT_H
/* DO NOT include any zlib headers here */
#include "aoclAlgoOpt.h"

#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/******************* aocl mapping *******************/
/* Conditional compilation flags mapping */
#ifdef AOCL_ZLIB_SSE2_OPT
#define X86_SSE2 1
#endif

#ifdef AOCL_ZLIB_AVX512_OPT
    #define X86_AVX512 1
    #define X86_AVX512VNNI 1
#endif

/* Visibility macros mapping */
#ifdef HAVE_HIDDEN
    #define Z_INTERNAL __attribute__((visibility ("hidden")))
#else
    #define Z_INTERNAL
#endif

/********************** zutil.h **********************/
#define MAX_BITS 15
/* all codes must not exceed MAX_BITS bits */
#define MAX_DIST_EXTRA_BITS 13
/* maximum number of extra distance bits */

/********************* zbuild.h **********************/
/* Minimum of a and b. */
#define MIN(a, b) ((a) > (b) ? (b) : (a))
/* Maximum of a and b. */
#define MAX(a, b) ((a) < (b) ? (b) : (a))

/* Only enable likely/unlikely if the compiler is known to support it */
#if (defined(__GNUC__) && (__GNUC__ >= 3)) || defined(__INTEL_COMPILER) || defined(__clang__)
#  define ZNG_LIKELY(x)             __builtin_expect(!!(x), 1)
#  define ZNG_UNLIKELY(x)           __builtin_expect(!!(x), 0)
#else
#  define ZNG_LIKELY(x)             x
#  define ZNG_UNLIKELY(x)           x
#endif /* (un)likely */

#if defined(HAVE_ATTRIBUTE_ALIGNED)
#  define ALIGNED_(x) __attribute__ ((aligned(x)))
#elif defined(_MSC_VER)
#  define ALIGNED_(x) __declspec(align(x))
#else
/* TODO: Define ALIGNED_ for your compiler */
/*#  define ALIGNED_(x) */
/* Force build error, else segfaults can occur if buffers are created 
 * with aligned assumption and accessed with aligned load/store */
#  error "ZlibNg ALIGNED_ macro is not defined for this compiler"
#endif

/* Diagnostic functions */
#ifdef ZLIB_DEBUG
extern void z_error(char *m);
#  include <stdio.h>
#  define Assert(cond, msg) {int _cond = (cond); if (!_cond) z_error(msg);}
#else
#  define Assert(cond, msg)
#endif

#  if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_I86)) && !defined(_M_ARM64EC)  /* _mm_prefetch() is not defined outside of x86/x64 */
#    include <mmintrin.h>
#    define PREFETCH_L1(ptr)  _mm_prefetch((const char*)(ptr), _MM_HINT_T0)
#    define PREFETCH_L2(ptr)  _mm_prefetch((const char*)(ptr), _MM_HINT_T1)
#  elif defined(__GNUC__) && ( (__GNUC__ >= 4) || ( (__GNUC__ == 3) && (__GNUC_MINOR__ >= 1) ) )
#    define PREFETCH_L1(ptr)  __builtin_prefetch((ptr), 0 /* rw==read */, 3 /* locality */)
#    define PREFETCH_L2(ptr)  __builtin_prefetch((ptr), 0 /* rw==read */, 2 /* locality */)
#  else
#    define PREFETCH_L1(ptr) do { (void)(ptr); } while (0)  /* disabled */
#    define PREFETCH_L2(ptr) do { (void)(ptr); } while (0)  /* disabled */
#  endif

#endif /* ZLIB_NG_COMPAT_H */ 

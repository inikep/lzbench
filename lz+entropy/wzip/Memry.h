/*
 * Memry.h - memory access helpers (unaligned loads and stores, wild copies)
 * Copyright (c) 2018-present, Yingquan (Cody) Wu.
 * SPDX-License-Identifier: BSD-2-Clause (see LICENSE)
 * Portions derived from Zstandard (lib/common/mem.h), Copyright (c) Meta Platforms, Inc. and affiliates
 * (BSD 3-Clause), and from LZ4, Copyright (c) 2011-present Yann Collet (BSD 2-Clause); see NOTICE.
 */

#ifndef __MEMOP_H_MODULE__
#define __MEMOP_H_MODULE__

#if defined (__cplusplus)
extern "C" {
#endif

/*-****************************************
*  Compiler specifics
******************************************/
/* force inlining */

#if defined (__GNUC__) || defined(__cplusplus) || defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L   /* C99 */
#  define INLINE_KEYWORD inline
#else
#  define INLINE_KEYWORD
#endif

#if defined(__GNUC__)
#  define FORCE_INLINE_ATTR __attribute__((always_inline))
#elif defined(_MSC_VER)
#  define FORCE_INLINE_ATTR __forceinline
#else
#  define FORCE_INLINE_ATTR
#endif

/**
 * ForceInlineTemplate is used to define C "templates", which take constant
 * parameters. They must be inlined for the compiler to eliminate the constant
 * branches.
 */
#define ForceInlineTemplate static INLINE_KEYWORD FORCE_INLINE_ATTR
 /**
  * HINT_INLINE is used to help the compiler generate better code. It is *not*
  * used for "templates", so it can be tweaked based on the compilers
  * performance.
  *
  * gcc-4.8 and gcc-4.9 have been shown to benefit from leaving off the
  * always_inline attribute.
  *
  * clang up to 5.0.0 (trunk) benefit tremendously from the always_inline
  * attribute.
  */
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 4 && __GNUC_MINOR__ >= 8 && __GNUC__ < 5
#  define HintInline static INLINE_KEYWORD
#else
#  define HintInline static INLINE_KEYWORD FORCE_INLINE_ATTR
#endif


  /*-****************************************
  *  Dependencies
  ******************************************/
#include <stddef.h>     /* size_t, ptrdiff_t */
#include <string.h>     /* memcpy */
#include <assert.h>

#if defined(_MSC_VER)   /* Visual Studio */
#   include <stdlib.h>  /* _byteswap_ulong */
#   include <intrin.h>  /* _byteswap_* */
#endif
#if defined(__aarch64__)
#   include <arm_neon.h>  /* vld1q_u8, vst1q_u8: MemCopy8, MemCopy16 */
#endif
#if defined(__GNUC__)
#  define MemStatic static __inline __attribute__((unused))
#elif defined (__cplusplus) || (defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) /* C99 */)
#  define MemStatic static inline
#elif defined(_MSC_VER)
#  define MemStatic static __inline
#else
#  define MemStatic static  /* this version may generate warnings for unused static functions; disable the relevant warning */
#endif

#ifndef __has_builtin
#  define __has_builtin(x) 0  /* compat. with non-clang compilers */
#endif

/* code only tested on 32 and 64 bits systems */
#define MemStatic_Assert(c)   { enum { MEM_static_assert = 1/(int)(!!(c)) }; }
MemStatic void MEM_Check(void) { MemStatic_Assert((sizeof(size_t)==4) || (sizeof(size_t)==8)); }



/* FORCE_O2_GCC_PPC64LE and FORCE_O2_INLINE_GCC_PPC64LE
 * gcc on ppc64le generates an unrolled SIMDized loop for LZ4_wildCopy8,
 * together with a simple 8-byte copy loop as a fall-back path.
 * However, this optimization hurts the decompression speed by >30%,
 * because the execution does not go to the optimized loop
 * for typical compressible data, and all of the preamble checks
 * before going to the fall-back path become useless overhead.
 * This optimization happens only with the -O3 flag, and -O2 generates
 * a simple 8-byte copy loop.
 * With gcc on ppc64le, all of the LZ_decompress_* and LZ_wildCopy8
 * functions are annotated with __attribute__((optimize("O2"))),
 * and also LZ_wildCopy8 is forcibly inlined, so that the O2 attribute
 * of LZ_wildCopy8 does not affect the compression speed.
 */
#if defined(__PPC64__) && defined(__LITTLE_ENDIAN__) && defined(__GNUC__) && !defined(__clang__)
#  define FORCE_O2_GCC_PPC64LE __attribute__((optimize("O2")))
#  define FORCE_O2_INLINE_GCC_PPC64LE __attribute__((optimize("O2"))) ForceInlineTemplate
#else
#  define FORCE_O2_GCC_PPC64LE
#  define FORCE_O2_INLINE_GCC_PPC64LE MemStatic
#endif
#if (defined(__GNUC__) && (__GNUC__ >= 3)) || (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 800)) || defined(__clang__)
#  define expect(expr,value)    (__builtin_expect ((expr),(value)) )
#else
#  define expect(expr,value)    (expr)
#endif

#ifndef likely
#define likely(expr)     expect((expr) != 0, 1)
#endif
#ifndef unlikely
#define unlikely(expr)   expect((expr) != 0, 0)
#endif


/*-**************************************************************
*  Basic Types
*****************************************************************/
#if  !defined (__VMS) && (defined (__cplusplus) || (defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) /* C99 */) )
# include <stdint.h>
  typedef   uint8_t Uint8;
  typedef  uint16_t Uint16;
  typedef   int16_t Sint16;
  typedef  uint32_t Uint32;
  typedef   int32_t Sint32;
  typedef  uint64_t Uint64;
  typedef   int64_t Sint64;
#else
# include <limits.h>
#if CHAR_BIT != 8
#  error "this implementation requires char to be exactly 8-bit type"
#endif
  typedef unsigned char      Uint8;
#if USHRT_MAX != 65535
#  error "this implementation requires short to be exactly 16-bit type"
#endif
  typedef unsigned short      Uint16;
  typedef   signed short      Sint16;
#if UINT_MAX != 4294967295
#  error "this implementation requires int to be exactly 32-bit type"
#endif
  typedef unsigned int        Uint32;
  typedef   signed int        Sint32;
/* note : there are no limits defined for long long type in C90.
 * limits exist in C99, however, in such case, <stdint.h> is preferred */
  typedef unsigned long long  Uint64;
  typedef   signed long long  Sint64;
#endif

#if defined(__x86_64__)
  typedef Uint64    reg_t;   /* 64-bits in x32 mode */
#else
  typedef size_t    reg_t;   /* 32-bits in x32 mode */
#endif

#define REG_SIZE sizeof(reg_t)

/*-**************************************************************
*  Memory I/O
*****************************************************************/
/* MEM_FORCE_MEMORY_ACCESS :
 * By default, access to unaligned memory is controlled by `memcpy()`, which is safe and portable.
 * Unfortunately, on some target/compiler combinations, the generated assembly is sub-optimal.
 * The below switch allow to select different access method for improved performance.
 * Method 0 (default) : use `memcpy()`. Safe and portable.
 * Method 1 : `__packed` statement. It depends on compiler extension (i.e., not portable).
 *            This method is safe if your compiler supports it, and *generally* as fast or faster than `memcpy`.
 * Method 2 : direct access. This method is portable but violate C standard.
 *            It can generate buggy code on targets depending on alignment.
 *            In some circumstances, it's the only known way to get the most performance (i.e. GCC + ARMv6)
 * See http://fastcompression.blogspot.fr/2015/08/accessing-unaligned-memory.html for details.
 * Prefer these methods in priority order (0 > 1 > 2)
 */
#ifndef MEM_FORCE_MEMORY_ACCESS   /* can be defined externally, on command line for example */
#  if defined(__GNUC__) && ( defined(__ARM_ARCH_6__) || defined(__ARM_ARCH_6J__) || defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6ZK__) || defined(__ARM_ARCH_6T2__) )
#    define MEM_FORCE_MEMORY_ACCESS 2
#  elif defined(__INTEL_COMPILER) || defined(__GNUC__)
#    define MEM_FORCE_MEMORY_ACCESS 1
#  endif
#endif

MemStatic unsigned MEM_In32bits(void) { return sizeof(reg_t)==4; }
MemStatic unsigned MEM_In64bits(void) { return sizeof(reg_t)==8; }

MemStatic unsigned MEM_IsLittleEndian(void)
{
    const union { Uint32 u; Uint8 c[4]; } one = { 1 };   /* don't use static : performance detrimental  */
    return one.c[0];
}

#if defined(MEM_FORCE_MEMORY_ACCESS) && (MEM_FORCE_MEMORY_ACCESS==2)

/* violates C standard, by lying on structure alignment.
Only use if no other choice to achieve best performance on target platform */
MemStatic Uint16 MemRead2(const void* memPtr) { return *(const Uint16*) memPtr; }
MemStatic Uint32 MemRead4(const void* memPtr) { return *(const Uint32*) memPtr; }
MemStatic Uint64 MemRead8(const void* memPtr) { return *(const Uint64*) memPtr; }
MemStatic size_t MemReadARCH(const void* memPtr) { return *(const reg_t*) memPtr; }

MemStatic void MemWrite2(void* memPtr, Uint16 value) { *(Uint16*)memPtr = value; }
MemStatic void MemWrite4(void* memPtr, Uint32 value) { *(Uint32*)memPtr = value; }
MemStatic void MemWrite8(void* memPtr, Uint64 value) { *(Uint64*)memPtr = value; }

#elif defined(MEM_FORCE_MEMORY_ACCESS) && (MEM_FORCE_MEMORY_ACCESS==1)

/* __pack instructions are safer, but compiler specific, hence potentially problematic for some compilers */
/* currently only defined for gcc and icc */
#if defined(_MSC_VER) || (defined(__INTEL_COMPILER) && defined(WIN32))
    __pragma( pack(push, 1) )
    typedef struct { Uint16 v; } unalign16;
    typedef struct { Uint32 v; } unalign32;
    typedef struct { Uint64 v; } unalign64;
    typedef struct { size_t v; } unalignArch;
    __pragma( pack(pop) )
#else
    typedef struct { Uint16 v; } __attribute__((packed)) unalign16;
    typedef struct { Uint32 v; } __attribute__((packed)) unalign32;
    typedef struct { Uint64 v; } __attribute__((packed)) unalign64;
    typedef struct { size_t v; } __attribute__((packed)) unalignArch;
#endif

MemStatic Uint16 MemRead2(const void* ptr) { return ((const unalign16*)ptr)->v; }
MemStatic Uint32 MemRead4(const void* ptr) { return ((const unalign32*)ptr)->v; }
MemStatic Uint64 MemRead8(const void* ptr) { return ((const unalign64*)ptr)->v; }
MemStatic size_t MemReadARCH(const void* ptr) { return ((const unalignArch*)ptr)->v; }

MemStatic void MemWrite2(void* memPtr, Uint16 value) { ((unalign16*)memPtr)->v = value; }
MemStatic void MemWrite4(void* memPtr, Uint32 value) { ((unalign32*)memPtr)->v = value; }
MemStatic void MemWrite8(void* memPtr, Uint64 value) { ((unalign64*)memPtr)->v = value; }

#else

/* default method, safe and standard.
   can sometimes prove slower */

MemStatic Uint16 MemRead2(const void* memPtr)
{
    Uint16 val; memcpy(&val, memPtr, sizeof(val)); return val;
}

MemStatic Uint32 MemRead4(const void* memPtr)
{
    Uint32 val; memcpy(&val, memPtr, sizeof(val)); return val;
}

MemStatic Uint64 MemRead8(const void* memPtr)
{
    Uint64 val; memcpy(&val, memPtr, sizeof(val)); return val;
}

MemStatic reg_t MemReadARCH(const void* memPtr)
{
    reg_t val; memcpy(&val, memPtr, sizeof(val)); return val;
}

MemStatic void MemWrite2(void* memPtr, Uint16 value)
{
    memcpy(memPtr, &value, sizeof(value));
}

MemStatic void MemWrite4(void* memPtr, Uint32 value)
{
    memcpy(memPtr, &value, sizeof(value));
}

MemStatic void MemWrite8(void* memPtr, Uint64 value)
{
    memcpy(memPtr, &value, sizeof(value));
}

#endif /* MEM_FORCE_MEMORY_ACCESS */

MemStatic Uint32 MemSwap4(Uint32 in)
{
#if defined(_MSC_VER)     /* Visual Studio */
    return _byteswap_ulong(in);
#elif (defined (__GNUC__) && (__GNUC__ * 100 + __GNUC_MINOR__ >= 403)) \
  || (defined(__clang__) && __has_builtin(__builtin_bswap32))
    return __builtin_bswap32(in);
#else
    return  ((in << 24) & 0xff000000 ) |
            ((in <<  8) & 0x00ff0000 ) |
            ((in >>  8) & 0x0000ff00 ) |
            ((in >> 24) & 0x000000ff );
#endif
}

MemStatic Uint64 MemSwap8(Uint64 in)
{
#if defined(_MSC_VER)     /* Visual Studio */
    return _byteswap_uint64(in);
#elif (defined (__GNUC__) && (__GNUC__ * 100 + __GNUC_MINOR__ >= 403)) \
  || (defined(__clang__) && __has_builtin(__builtin_bswap64))
    return __builtin_bswap64(in);
#else
    return  ((in << 56) & 0xff00000000000000ULL) |
            ((in << 40) & 0x00ff000000000000ULL) |
            ((in << 24) & 0x0000ff0000000000ULL) |
            ((in << 8)  & 0x000000ff00000000ULL) |
            ((in >> 8)  & 0x00000000ff000000ULL) |
            ((in >> 24) & 0x0000000000ff0000ULL) |
            ((in >> 40) & 0x000000000000ff00ULL) |
            ((in >> 56) & 0x00000000000000ffULL);
#endif
}

MemStatic size_t MEM_SwapST(size_t in)
{
    if (MEM_In32bits())
        return (size_t)MemSwap4((Uint32)in);
    else
        return (size_t)MemSwap8((Uint64)in);
}

/*=== Little endian r/w ===*/

MemStatic Uint16 MemReadLE2(const void* memPtr)
{
    if (MEM_IsLittleEndian())
        return MemRead2(memPtr);
    else {
        const Uint8* p = (const Uint8*)memPtr;
        return (Uint16)(p[0] + (p[1]<<8));
    }
}

MemStatic void MemWriteLE2(void* memPtr, Uint16 val)
{
    if (MEM_IsLittleEndian()) {
        MemWrite2(memPtr, val);
    } else {
        Uint8* p = (Uint8*)memPtr;
        p[0] = (Uint8)val;
        p[1] = (Uint8)(val>>8);
    }
}

MemStatic Uint32 MemReadLE3(const void* memPtr)
{
    return MemReadLE2(memPtr) + (((const Uint8*)memPtr)[2] << 16);
}

MemStatic void MemWriteLE3(void* memPtr, Uint32 val)
{
    MemWriteLE2(memPtr, (Uint16)val);
    ((Uint8*)memPtr)[2] = (Uint8)(val>>16);
}

MemStatic Uint32 MemReadLE4(const void* memPtr)
{
    if (MEM_IsLittleEndian())
        return MemRead4(memPtr);
    else
        return MemSwap4(MemRead4(memPtr));
}

MemStatic void MemWriteLE4(void* memPtr, Uint32 val32)
{
    if (MEM_IsLittleEndian())
        MemWrite4(memPtr, val32);
    else
        MemWrite4(memPtr, MemSwap4(val32));
}

MemStatic Uint64 MemReadLE8(const void* memPtr)
{
    if (MEM_IsLittleEndian())
        return MemRead8(memPtr);
    else
        return MemSwap8(MemRead8(memPtr));
}

MemStatic void MemWriteLE8(void* memPtr, Uint64 val64)
{
    if (MEM_IsLittleEndian())
        MemWrite8(memPtr, val64);
    else
        MemWrite8(memPtr, MemSwap8(val64));
}

MemStatic size_t MemReadLEST(const void* memPtr)
{
    if (MEM_In32bits())
        return (size_t)MemReadLE4(memPtr);
    else
        return (size_t)MemReadLE8(memPtr);
}

MemStatic void MemWriteLEST(void* memPtr, size_t val)
{
    if (MEM_In32bits())
        MemWriteLE4(memPtr, (Uint32)val);
    else
        MemWriteLE8(memPtr, (Uint64)val);
}

/*=== Big endian r/w ===*/

MemStatic Uint32 MemReadBE4(const void* memPtr)
{
    if (MEM_IsLittleEndian())
        return MemSwap4(MemRead4(memPtr));
    else
        return MemRead4(memPtr);
}

MemStatic void MemWriteBE4(void* memPtr, Uint32 val32)
{
    if (MEM_IsLittleEndian())
        MemWrite4(memPtr, MemSwap4(val32));
    else
        MemWrite4(memPtr, val32);
}

MemStatic Uint64 MemReadBE8(const void* memPtr)
{
    if (MEM_IsLittleEndian())
        return MemSwap8(MemRead8(memPtr));
    else
        return MemRead8(memPtr);
}

MemStatic void MemWriteBE8(void* memPtr, Uint64 val64)
{
    if (MEM_IsLittleEndian())
        MemWrite8(memPtr, MemSwap8(val64));
    else
        MemWrite8(memPtr, val64);
}

MemStatic size_t MemReadBEST(const void* memPtr)
{
    if (MEM_In32bits())
        return (size_t)MemReadBE4(memPtr);
    else
        return (size_t)MemReadBE8(memPtr);
}

MemStatic void MemWriteBEST(void* memPtr, size_t val)
{
    if (MEM_In32bits())
        MemWriteBE4(memPtr, (Uint32)val);
    else
        MemWriteBE8(memPtr, (Uint64)val);
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Custom designed memcpy ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

MemStatic void MemCopy8(void* dst, const void* src) 
{
#ifdef __aarch64__
    vst1_u8((Uint8*)dst, vld1_u8((const Uint8*)src));
#else
    memcpy(dst, src, 8);
#endif
}

MemStatic void MemCopy16(void* dst, const void* src) {
#ifdef __aarch64__
    vst1q_u8((Uint8*)dst, vld1q_u8((const Uint8*)src));
#else
    memcpy(dst, src, 16);
#endif
}

/* Custom version of memcpy() in 16-byte steps: writes up to 15 bytes past destEnd (16 if length==0), on every
   target; the decoders' margins count on it (an arm64 variant in 32-byte steps wrote up to 31) */
ForceInlineTemplate void MemWildCopy(void* dest, const void* src, void * const _destEnd)
{
    Uint8* const destEnd = (Uint8*)_destEnd;
    Uint8* destPtr = (Uint8*)dest;
    Uint8* srcPtr = (Uint8*)src;

    do {
        MemCopy16(destPtr, srcPtr);
        destPtr += 16;
        srcPtr += 16;
    } while ( destPtr < destEnd );
}

/* It requires that dest and src must be at least 8 bytes apart */
/* should be faster for decoding, but strangely, not verified on all platform */
FORCE_O2_INLINE_GCC_PPC64LE
void MemWildCopy_Overlap(void* dst, const void* src, void * const dstEnd)   
{
	const Uint8* ip = (const Uint8*)src;
	Uint8* op = (Uint8*)dst;
	Uint8* const oend = (Uint8*)dstEnd;
    do {
		MemCopy8(op, ip);
		op += 8;
		ip += 8;
    } while (op < oend);
}

/* FORCE_SW_BITCOUNT
* Define this parameter if your target system or compiler does not support hardware bit count
*/
#if defined(_MSC_VER) && defined(_WIN32_WCE)   /* Visual Studio for WinCE doesn't support Hardware bit count */
#define FORCE_SW_BITCOUNT
#endif

MemStatic unsigned High_Bit32(Uint32 val)
{
    assert(val != 0);
    {
#   if defined(_MSC_VER)   /* Visual */
        unsigned long r = 0;
        _BitScanReverse(&r, val);
        return (unsigned)r;
#   elif defined(__GNUC__) && (__GNUC__ >= 3)   /* Use GCC Intrinsic */
        return 31 - __builtin_clz(val);
#   else   /* Software version */
        static const unsigned DeBruijnClz[32] = { 0,  9,  1, 10, 13, 21,  2, 29,
                                                 11, 14, 16, 18, 22, 25,  3, 30,
                                                  8, 12, 20, 28, 15, 17, 24,  7,
                                                 19, 27, 23,  6, 26,  5,  4, 31 };
        U32 v = val;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return DeBruijnClz[(U32)(v * 0x07C4ACDDU) >> 27];
#   endif
    }
}

#define N_Bits(y)   ( High_Bit32(y)+1 )

/* this function assumes val is unsized non-zero */
MemStatic unsigned N_ZeroBytes(reg_t val)
{
    assert(val != 0);
    if (MEM_IsLittleEndian()) {
        if (MEM_In64bits()) {
#       if defined(_MSC_VER) && defined(_WIN64) && !defined(FORCE_SW_BITCOUNT)
            unsigned long r = 0;
            _BitScanForward64(&r, (Uint64)val);
            return (int)(r >> 3);
#       elif (defined(__clang__) || (defined(__GNUC__) && (__GNUC__>=3))) && !defined(FORCE_SW_BITCOUNT)
            return (__builtin_ctzll((Uint64)val) >> 3);
#       else
            static const int DeBruijnBytePos[64] = { 0, 0, 0, 0, 0, 1, 1, 2,
                                                     0, 3, 1, 3, 1, 4, 2, 7,
                                                     0, 2, 3, 6, 1, 5, 3, 5,
                                                     1, 3, 4, 4, 2, 5, 6, 7,
                                                     7, 0, 1, 2, 3, 3, 4, 6,
                                                     2, 6, 5, 5, 3, 4, 5, 6,
                                                     7, 1, 2, 4, 6, 4, 4, 5,
                                                     7, 2, 6, 5, 7, 6, 7, 7 };
            return DeBruijnBytePos[((Uint64)((val & -(long long)val) * 0x0218A392CDABBD3FULL)) >> 58];
#       endif
        }
        else /* 32 bits */ {
#       if defined(_MSC_VER) && !defined(FORCE_SW_BITCOUNT)
            unsigned long r;
            _BitScanForward(&r, (Uint32)val);
            return (int)(r >> 3);
#       elif (defined(__clang__) || (defined(__GNUC__) && (__GNUC__>=3))) && !defined(FORCE_SW_BITCOUNT)
            return (__builtin_ctz((Uint32)val) >> 3);
#       else
            static const int DeBruijnBytePos[32] = { 0, 0, 3, 0, 3, 1, 3, 0,
                                                     3, 2, 2, 1, 3, 2, 0, 1,
                                                     3, 3, 1, 2, 2, 2, 2, 0,
                                                     3, 1, 2, 0, 1, 0, 1, 1 };
            return DeBruijnBytePos[((Uint32)((val & -(INT4)val) * 0x077CB531U)) >> 27];
#       endif
        }
    }
    else   /* Big Endian CPU */ {
        if (MEM_In64bits()) {   /* 64-bits */
#       if defined(_MSC_VER) && defined(_WIN64) && !defined(FORCE_SW_BITCOUNT)
            unsigned long r = 0;
            _BitScanReverse64(&r, val);
            return (unsigned)(r >> 3);
#       elif (defined(__clang__) || (defined(__GNUC__) && (__GNUC__>=3))) && !defined(FORCE_SW_BITCOUNT)
            return (__builtin_clzll((Uint64)val) >> 3);
#       else
            static const Uint32 by32 = sizeof(val) * 4;  /* 32 on 64 bits (goal), 16 on 32 bits.
                Just to avoid some static analyzer complaining about shift by 32 on 32-bits target.
                Note that this code path is never triggered in 32-bits mode. */
            unsigned r;
            if (!(val >> by32)) { r = 4; }
            else { r = 0; val >>= by32; }
            if (!(val >> 16)) { r += 2; val >>= 8; }
            else { val >>= 24; }
            r += (!val);
            return r;
#       endif
        }
        else /* 32 bits */ {
#       if defined(_MSC_VER) && !defined(FORCE_SW_BITCOUNT)
            unsigned long r = 0;
            _BitScanReverse(&r, (unsigned long)val);
            return (unsigned)(r >> 3);
#       elif (defined(__clang__) || (defined(__GNUC__) && (__GNUC__>=3))) && !defined(FORCE_SW_BITCOUNT)
            return (__builtin_clz((Uint32)val) >> 3);
#       else
            unsigned r;
            if (!(val >> 16)) { r = 2; val >>= 8; }
            else { r = 0; val >>= 24; }
            r += (!val);
            return r;
#       endif
        }
    }
}
static const Uint64 ByteMask[8] = { 0,   0xFF,  0xFFFF,  0xFFFFFF,  0xFFFFFFFF,     0xFFFFFFFFFF,  0xFFFFFFFFFFFF,  0xFFFFFFFFFFFFFF };
#define HashPrime4 2654435761U
#define HashPrime8 11400714785074694791ULL

ForceInlineTemplate Uint32 Hash_16B(const Uint8* const stream)
{
    Uint64 hashV8;
    Uint32 hashV4;

    if (MEM_In64bits()) {
        Uint64 seq8 = MemReadARCH(stream);
        hashV8 = (seq8 * HashPrime8);
        seq8 = MemReadARCH(stream + 8);
        hashV8 = (hashV8 + seq8) * HashPrime8;
        return (Uint32)( (hashV8 >> 42) ^ (hashV8 >> 21) ^ hashV8);
    }
    else {
        reg_t seqA4 = MemReadARCH(stream);
        reg_t seqB4 = MemReadARCH(&stream[4]);

        hashV4 = (Uint32)((seqA4 * HashPrime4 + seqB4) * HashPrime4);
        seqA4 = MemReadARCH(stream+8);
        seqB4 = MemReadARCH(stream+12);

        hashV4 = (Uint32)((seqA4 * HashPrime4 + seqB4) * HashPrime4*HashPrime4 + hashV4);
        return ((hashV4 >> 13) ^ (hashV4 >> 3) ^ hashV4);
    }
}

ForceInlineTemplate Uint32 Hash_12B(const Uint8* const stream)
{
    Uint64 hashV8;
    Uint32 hashV4;

    if (MEM_In64bits()) {
        Uint64 seq8 = MemReadARCH(stream);
        hashV8 = (seq8 * HashPrime8);
        hashV8 = hashV8 + MemRead4(stream + 8) * HashPrime4;
        return (Uint32)((hashV8 >> 42) ^ (hashV8 >> 21) ^ hashV8);
    }
    else {
        reg_t seqA4 = MemReadARCH(stream);
        reg_t seqB4 = MemReadARCH(&stream[4]);

        hashV4 = (Uint32)((seqA4 * HashPrime4 + seqB4) * HashPrime4);
        seqA4 = MemReadARCH(stream + 8);

        hashV4 = (Uint32)( HashPrime4 * HashPrime4 * seqA4 + hashV4);
        return ((hashV4 >> 13) ^ (hashV4 >> 3) ^ hashV4);
    }
}

ForceInlineTemplate Uint32 Hash_10B(const Uint8* const stream)
{
    Uint64 hashV8;
    Uint32 hashV4;

    if (MEM_In64bits()) {
        Uint64 seq8 = MemReadARCH(stream);
        hashV8 = (seq8 * HashPrime4);
        hashV8 = hashV8 + MemRead2(stream + 8) * HashPrime4;
        return (Uint32)((hashV8 >> 42) ^ (hashV8 >> 21) ^ hashV8);
    }
    else {
        reg_t seqA4 = MemReadARCH(stream);
        reg_t seqB4 = MemReadARCH(&stream[4]);

        hashV4 = (Uint32)(seqA4 * HashPrime4 + seqB4);
        seqA4 = MemRead2(stream + 8);

        hashV4 = (Uint32)(HashPrime4 * HashPrime4 * seqA4 + hashV4);
        return ((hashV4 >> 13) ^ (hashV4 >> 3) ^ hashV4);
    }
}
ForceInlineTemplate Uint32 Hash_8B(const Uint8* const stream)
{
    Uint64 hashV8;
    Uint32 hashV4;

    if (MEM_In64bits()) {
        Uint64 seq8 = MemReadARCH(stream);
        hashV8 = (seq8 * HashPrime8);
        return (Uint32)((hashV8 >> 21) ^ (hashV8 >> 42) ^ hashV8);
    }
    else {
        reg_t seqA4 = MemReadARCH(stream);
        reg_t seqB4 = MemReadARCH(&stream[4]);

        hashV4 = (Uint32)((seqA4 * HashPrime4 + seqB4) * HashPrime4);
        return ((hashV4 >> 13) ^ (hashV4 >> 3) ^ hashV4);
    }

}

ForceInlineTemplate Uint32 Hash_7B(const Uint8* const stream)
{
    Uint64 hashV8;
    Uint32 hashV4;

    if (MEM_In64bits()) {
        Uint64 seq8 = MemReadARCH(stream);
        if (MEM_IsLittleEndian()) 
            hashV8 = ((seq8 & ByteMask[7]) * HashPrime8);
        else 
            hashV8 = ((seq8 >> 8) * HashPrime8);
        return (Uint32)((hashV8 >> 21) ^ (hashV8 >> 42) ^ hashV8);
    }
    else {
        reg_t seqA4 = MemReadARCH(stream);
        reg_t seqB4 = MemReadARCH(&stream[4]);
        if (MEM_IsLittleEndian()) {
            hashV4 = (Uint32)((seqA4 * HashPrime4 + (seqB4 & ByteMask[3])) * HashPrime4);           
        }
        else {
            hashV4 = (Uint32)((seqA4 * HashPrime4 + (seqB4 >> 8)) * HashPrime4);
        }
        return ((hashV4 >> 13) ^ (hashV4 >> 3) ^ hashV4);
    }

}

ForceInlineTemplate Uint32 Hash_6B(const Uint8* const stream)
{
    Uint64 hashV8;
    Uint32 hashV4;

    if (MEM_In64bits()) {
        Uint64 seq8 = MemReadARCH(stream);
        if (MEM_IsLittleEndian()) {
            hashV8 = ((seq8 & ByteMask[6]) * HashPrime8);
        } else {
            hashV8 = ((seq8 >> 16) * HashPrime8);
        }
        return (Uint32)((hashV8 >> 21) ^ (hashV8 >> 42) ^ hashV8);
    }
    else {
        reg_t seqA4 = MemReadARCH(stream);
        reg_t seqB4 = MemReadARCH(&stream[4]);
        if (MEM_IsLittleEndian()) {
            hashV4 = (Uint32)((seqA4 * HashPrime4 + (seqB4 << 16)) * HashPrime4);
        } else {
            hashV4 = (Uint32)(seqA4 * HashPrime4 + (seqB4 >> 16)) * HashPrime4;
        }
        return ((hashV4 >> 13) ^ (hashV4 >> 3) ^ hashV4);
    }

}

ForceInlineTemplate Uint32 Hash_5B(const Uint8* const stream)
{
    Uint64 hashV8;
    Uint32 hashV4;
    reg_t seq = MemReadARCH(stream);

    if (MEM_In64bits()) {
        if (MEM_IsLittleEndian()) {
            hashV8 = (seq & ByteMask[5]) * HashPrime8;
        } else {
            hashV8 = (seq >> 24) * HashPrime8;
        }
        return (Uint32)((hashV8 >> 21) ^ (hashV8 >> 42) ^ hashV8);
    }
    else {
        reg_t seqB = MemReadARCH(&stream[4]);
        if (MEM_IsLittleEndian()) {
            hashV4 = (Uint32)(seq * HashPrime4 + (seqB << 24)) * HashPrime4;
        }  else {
            hashV4 = (Uint32)(seq * HashPrime4 + (seqB >> 24)) * HashPrime4;
        }
        return ((hashV4 >> 13) ^ (hashV4 >> 3) ^ hashV4);
    }
}

ForceInlineTemplate Uint32 Hash_4B(const Uint8* const stream)
{
    Uint32 seq4 = MemRead4(stream);

    Uint32 hashV4 = seq4 * HashPrime4;
    return ((hashV4 >> 13) ^ hashV4 ^ (hashV4 >> 3));
}

ForceInlineTemplate Uint32 Hash_3B(const Uint8* const stream)
{
    Uint32 seq4 = MemReadLE4(stream);

    Uint32 hashV4 = ((seq4 & ByteMask[3]) * HashPrime4);
    return (Uint32)(hashV4 ^ (hashV4 >> 13) ^ (hashV4 >> 3));
}

#if defined (__cplusplus)
}
#endif

#endif /* MEM_H_MODULE */


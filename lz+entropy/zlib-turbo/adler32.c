/* adler32.c -- compute the Adler-32 checksum of a data stream
 * Copyright (C) 1995-2011, 2016 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* @(#) $Id$ */

#include "zutil.h"

#define BASE 65521U     /* largest prime smaller than 65536 */
#define NMAX 5552
/* NMAX is the largest n such that 255n(n+1)/2 + (n+1)(BASE-1) <= 2^32-1 */

#define DO1(buf,i)  {adler += (buf)[i]; sum2 += adler;}
#define DO2(buf,i)  DO1(buf,i); DO1(buf,i+1);
#define DO4(buf,i)  DO2(buf,i); DO2(buf,i+2);
#define DO8(buf,i)  DO4(buf,i); DO4(buf,i+4);
#define DO16(buf)   DO8(buf,0); DO8(buf,8);

/* use NO_DIVIDE if your processor does not do division in hardware --
   try it both ways to see which is faster */
#ifdef NO_DIVIDE
/* note that this assumes BASE is 65521, where 65536 % 65521 == 15
   (thank you to John Reiser for pointing this out) */
#  define CHOP(a) \
    do { \
        unsigned long tmp = a >> 16; \
        a &= 0xffffUL; \
        a += (tmp << 4) - tmp; \
    } while (0)
#  define MOD28(a) \
    do { \
        CHOP(a); \
        if (a >= BASE) a -= BASE; \
    } while (0)
#  define MOD(a) \
    do { \
        CHOP(a); \
        MOD28(a); \
    } while (0)
#  define MOD63(a) \
    do { /* this assumes a is not negative */ \
        z_off64_t tmp = a >> 32; \
        a &= 0xffffffffL; \
        a += (tmp << 8) - (tmp << 5) + tmp; \
        tmp = a >> 16; \
        a &= 0xffffL; \
        a += (tmp << 4) - tmp; \
        tmp = a >> 16; \
        a &= 0xffffL; \
        a += (tmp << 4) - tmp; \
        if (a >= BASE) a -= BASE; \
    } while (0)
#else
#  define MOD(a) a %= BASE
#  define MOD28(a) a %= BASE
#  define MOD63(a) a %= BASE
#endif

/* Eight bytes at a time, on 64-bit targets with fast 64-bit multiplication
   (those for which crc32.c uses eight-byte words, and 64-bit MSVC). Define
   Z_ADLER_SWAR as 0 to turn it off, or as 1 to force it on. */
#ifndef Z_ADLER_SWAR
#  if defined(__x86_64__) || defined(__aarch64__) || defined(_M_X64) || \
      defined(_M_ARM64)
#    define Z_ADLER_SWAR 1
#  else
#    define Z_ADLER_SWAR 0
#  endif
#endif
#if Z_ADLER_SWAR && !defined(Z_U8)
#  undef Z_ADLER_SWAR
#  define Z_ADLER_SWAR 0
#endif

#if Z_ADLER_SWAR
/* The eight bytes at p as a little-endian number. */
local Z_U8 load64le(const Bytef *p) {
    Z_U8 v = 1;

    if (*(unsigned char *)&v) {
        zmemcpy(&v, p, 8);
        return v;
    }
    return (Z_U8)p[0] | ((Z_U8)p[1] << 8) | ((Z_U8)p[2] << 16) |
           ((Z_U8)p[3] << 24) | ((Z_U8)p[4] << 32) | ((Z_U8)p[5] << 40) |
           ((Z_U8)p[6] << 48) | ((Z_U8)p[7] << 56);
}
#endif

/* zlib-turbo: with -DZ_TURBO_SIMD and a compiler target with AVX2 (-mavx2,
   -march=haswell or x86-64-v3 or later, or /arch:AVX2 with MSVC), thirty-two
   bytes at a time. There is no check at run time. */
#if defined(Z_TURBO_SIMD) && defined(__AVX2__)
#  define Z_TURBO_ADLER_AVX2
#  include <immintrin.h>

/* Update the sums *a and *s, each less than BASE, with the len bytes at buf,
   len a multiple of 32, and leave them less than BASE. For each block of 32
   bytes b[0..31], a grows by the sum of the bytes and s by 32 times a before
   the block plus the sum of (32 - i) b[i]. The vector sums are reduced every
   5536 bytes, at most NMAX, so no 32-bit lane overflows. */
local void adler32_avx2(unsigned long *a, unsigned long *s, const Bytef *buf,
                        z_size_t len) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i taps = _mm256_setr_epi8(32, 31, 30, 29, 28, 27, 26, 25,
                                          24, 23, 22, 21, 20, 19, 18, 17,
                                          16, 15, 14, 13, 12, 11, 10, 9,
                                          8, 7, 6, 5, 4, 3, 2, 1);
    unsigned long adler = *a, sum2 = *s;

    while (len) {
        z_size_t k = len < (NMAX / 32) * 32 ? len : (NMAX / 32) * 32;
        __m256i va = zero, vp = zero, vs = zero, v;
        __m128i t;

        len -= k;
        k /= 32;
        sum2 = (sum2 + 32 * adler * (unsigned long)k) % BASE;
        /* below, s gains at most 3.93e9 (all bytes 255): with sum2 reduced
           first, that fits 32 bits, as unsigned long is on 64-bit Windows */
        do {
            v = _mm256_loadu_si256((const __m256i *)buf);
            vp = _mm256_add_epi32(vp, va);      /* sum of a before each block */
            va = _mm256_add_epi32(va, _mm256_sad_epu8(v, zero));
            vs = _mm256_add_epi32(vs, _mm256_madd_epi16(
                     _mm256_maddubs_epi16(v, taps), ones));
            buf += 32;
        } while (--k);
        vs = _mm256_add_epi32(vs, _mm256_slli_epi32(vp, 5));

        /* add up the eight lanes of each */
        t = _mm_add_epi32(_mm256_castsi256_si128(va),
                          _mm256_extracti128_si256(va, 1));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0x4e));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0xb1));
        adler += (unsigned long)(unsigned)_mm_cvtsi128_si32(t);
        t = _mm_add_epi32(_mm256_castsi256_si128(vs),
                          _mm256_extracti128_si256(vs, 1));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0x4e));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0xb1));
        sum2 += (unsigned long)(unsigned)_mm_cvtsi128_si32(t);
        adler %= BASE;
        sum2 %= BASE;
    }
    *a = adler;
    *s = sum2;
}
#endif

/* ========================================================================= */
uLong ZEXPORT adler32_z(uLong adler, const Bytef *buf, z_size_t len) {
    unsigned long sum2;
#if !Z_ADLER_SWAR
    unsigned n;
#endif

    /* split Adler-32 into component sums */
    sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;

    /* in case user likes doing a byte at a time, keep it fast */
    if (len == 1) {
        adler += buf[0];
        if (adler >= BASE)
            adler -= BASE;
        sum2 += adler;
        if (sum2 >= BASE)
            sum2 -= BASE;
        return adler | (sum2 << 16);
    }

    /* initial Adler-32 value (deferred check for len == 1 speed) */
    if (buf == Z_NULL)
        return 1L;

#ifdef Z_TURBO_ADLER_AVX2
    /* all but the last 0..31 bytes with AVX2, the rest below */
    if (len >= 64) {
        z_size_t n = len & ~(z_size_t)31;

        adler32_avx2(&adler, &sum2, buf, n);
        buf += n;
        len -= n;
    }
#endif

    /* in case short lengths are provided, keep it somewhat fast */
    if (len < 16) {
        while (len--) {
            adler += *buf++;
            sum2 += adler;
        }
        if (adler >= BASE)
            adler -= BASE;
        MOD28(sum2);            /* only added so many BASE's */
        return adler | (sum2 << 16);
    }

#if Z_ADLER_SWAR
    /* Eight bytes at a time. With the even and the odd bytes of a word in
       16-bit lanes, one multiplication adds them up and two weight them: the
       top 16 bits of v * (c0 + c1 2^16 + c2 2^32 + c3 2^48) are
       v0 c3 + v1 c2 + v2 c1 + v3 c0, as no partial sum reaches 2^16 (at most
       255 * 20). A word adds 8 * adler + sum((8 - i) * byte[i]) to sum2. In 64
       bits the sums need a modulo only every 2^20 bytes. */
    {
        Z_U8 a = adler, s = sum2, x, e, o;
        Z_U8 m = ((Z_U8)0xff00ff << 32) | 0xff00ff;
        Z_U8 ones = ((Z_U8)0x10001 << 32) | 0x10001;
        Z_U8 we = ((Z_U8)0x80006 << 32) | 0x40002;  /* bytes 0, 2, 4, 6 */
        Z_U8 wo = ((Z_U8)0x70005 << 32) | 0x30001;  /* bytes 1, 3, 5, 7 */
        z_size_t k;

        do {
            k = len < 0x100000 ? len >> 3 : 0x20000;
            len -= k << 3;
            do {
                x = load64le(buf);
                e = x & m;
                o = (x >> 8) & m;
                s += (a << 3) + ((e * we) >> 48) + ((o * wo) >> 48);
                a += ((e + o) * ones) >> 48;
                buf += 8;
            } while (--k);
            a %= BASE;
            s %= BASE;
        } while (len >= 8);
        adler = (unsigned long)a;
        sum2 = (unsigned long)s;
        while (len--) {
            adler += *buf++;
            sum2 += adler;
        }
        if (adler >= BASE)
            adler -= BASE;
        MOD28(sum2);
        return adler | (sum2 << 16);
    }
#else
    /* do length NMAX blocks -- requires just one modulo operation */
    while (len >= NMAX) {
        len -= NMAX;
        n = NMAX / 16;          /* NMAX is divisible by 16 */
        do {
            DO16(buf);          /* 16 sums unrolled */
            buf += 16;
        } while (--n);
        MOD(adler);
        MOD(sum2);
    }

    /* do remaining bytes (less than NMAX, still just one modulo) */
    if (len) {                  /* avoid modulos if none remaining */
        while (len >= 16) {
            len -= 16;
            DO16(buf);
            buf += 16;
        }
        while (len--) {
            adler += *buf++;
            sum2 += adler;
        }
        MOD(adler);
        MOD(sum2);
    }

    /* return recombined sums */
    return adler | (sum2 << 16);
#endif
}

/* ========================================================================= */
uLong ZEXPORT adler32(uLong adler, const Bytef *buf, uInt len) {
    return adler32_z(adler, buf, len);
}

/* ========================================================================= */
local uLong adler32_combine_(uLong adler1, uLong adler2, z_off64_t len2) {
    unsigned long sum1;
    unsigned long sum2;
    unsigned rem;

    /* for negative len, return invalid adler32 as a clue for debugging */
    if (len2 < 0)
        return 0xffffffffUL;

    /* the derivation of this formula is left as an exercise for the reader */
    MOD63(len2);                /* assumes len2 >= 0 */
    rem = (unsigned)len2;
    sum1 = adler1 & 0xffff;
    sum2 = rem * sum1;
    MOD(sum2);
    sum1 += (adler2 & 0xffff) + BASE - 1;
    sum2 += ((adler1 >> 16) & 0xffff) + ((adler2 >> 16) & 0xffff) + BASE - rem;
    if (sum1 >= BASE) sum1 -= BASE;
    if (sum1 >= BASE) sum1 -= BASE;
    if (sum2 >= ((unsigned long)BASE << 1)) sum2 -= ((unsigned long)BASE << 1);
    if (sum2 >= BASE) sum2 -= BASE;
    return sum1 | (sum2 << 16);
}

/* ========================================================================= */
uLong ZEXPORT adler32_combine(uLong adler1, uLong adler2, z_off_t len2) {
    return adler32_combine_(adler1, adler2, len2);
}

uLong ZEXPORT adler32_combine64(uLong adler1, uLong adler2, z_off64_t len2) {
    return adler32_combine_(adler1, adler2, len2);
}

// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_DIVIDE_COMMON_GCD_H
#define ZSTRONG_TRANSFORMS_DIVIDE_COMMON_GCD_H

#include <limits.h>             // UCHAR_MAX
#include "openzl/shared/bits.h" // ZL_ctz32, ZL_ctz64
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

ZL_INLINE uint8_t ZL_getMultiplicativeInverse8(int* shift, uint8_t divisor)
{
    assert(divisor != 0);
    // For X = 2^8 and Y = divisor find AX + BY = 1 => B = inv(Y)
    *shift  = ZL_ctz32(divisor);
    divisor = (uint8_t)(divisor >> *shift);
    if (divisor == 1) {
        return 1;
    }
    // If 2^8 - 1 % divisor + 1 = 2 ^ 8 % divisor unless divisor is a power
    // of 2. This is not the case because we bit shift to remove powers of 2.
    uint8_t q  = (uint8_t)(UCHAR_MAX / divisor);
    uint8_t x  = (uint8_t)(UCHAR_MAX - divisor * q + 1);
    uint8_t y  = divisor;
    uint8_t sb = 1;
    uint8_t ts = 1;
    uint8_t tl = q;
    while (x != 0) {
        q = y / x;
        y = y - (uint8_t)(x * q);
        // Swap x and y as well as tl and ts
        y ^= x;
        x ^= y;
        y ^= x;
        ts += (uint8_t)(tl * q);
        tl ^= ts;
        ts ^= tl;
        tl ^= ts;
        sb ^= 1;
    }
    if (sb == 0) { // If sb is 0, then ts is negative since tl is positive
        return (uint8_t)((uint8_t)((uint8_t)0 - ts + CHAR_MAX + 1)
                         + (uint8_t)CHAR_MAX + 1);
    } else {
        return (uint8_t)ts;
    }
}

ZL_INLINE uint32_t ZL_getMultiplicativeInverse32(int* shift, uint32_t divisor)
{
    assert(divisor != 0);
    // For X = 2^32 and Y = divisor find AX + BY = 1 => B = inv(Y)
    *shift = ZL_ctz32(divisor);
    divisor >>= *shift;
    if (divisor == 1) {
        return 1;
    }
    // If 2^32 - 1 % divisor + 1 = 2 ^ 32 % divisor unless divisor is a power
    // of 2. This is not the case because we bit shift to remove powers of 2.
    uint32_t q  = UINT_MAX / divisor;
    uint32_t x  = UINT_MAX - divisor * q + 1;
    uint32_t y  = divisor;
    uint32_t sb = 1;
    uint32_t ts = 1;
    uint32_t tl = q;
    while (x != 0) {
        q = y / x;
        y = y - x * q;
        // Swap x and y as well as tl and ts
        y ^= x;
        x ^= y;
        y ^= x;
        ts += tl * q;
        tl ^= ts;
        ts ^= tl;
        tl ^= ts;
        sb ^= 1;
    }
    if (sb == 0) { // If sb is 0, then ts is negative since tl is positive
        return (uint32_t)((uint32_t)0 - ts + INT_MAX + 1) + (uint32_t)INT_MAX
                + 1;
    } else {
        return (uint32_t)ts;
    }
}

ZL_INLINE uint64_t ZL_getMultiplicativeInverse64(int* shift, uint64_t divisor)
{
    assert(divisor != 0);
    // For X = 2^64 and Y = divisor find AX + BY = 1 => B = inv(Y)
    *shift = ZL_ctz64(divisor);
    divisor >>= *shift;
    if (divisor == 1) {
        return 1;
    }
    // If 2^64 - 1 % divisor + 1 = 2 ^ 64 % divisor unless divisor is a power
    // of 2. This is not the case because we bit shift to remove powers of 2.
    uint64_t q  = ULLONG_MAX / divisor;
    uint64_t x  = ULLONG_MAX - divisor * q + 1;
    uint64_t y  = divisor;
    uint64_t sb = 1;
    uint64_t ts = 1;
    uint64_t tl = q;
    while (x != 0) {
        q = y / x;
        y = y - x * q;
        // Swap x and y as well as tl and ts
        y ^= x;
        x ^= y;
        y ^= x;
        ts += tl * q;
        tl ^= ts;
        ts ^= tl;
        tl ^= ts;
        sb ^= 1;
    }
    if (sb == 0) { // If sb is 0, then ts is negative since tl is positive
        return (uint64_t)((uint64_t)0 - ts + LLONG_MAX + 1)
                + (uint64_t)LLONG_MAX + 1;
    } else {
        return (uint64_t)ts;
    }
}

ZL_INLINE uint64_t ZL_gcdImpl(uint64_t a, uint64_t b)
{
    if (a < b) {
        uint64_t t = b;
        b          = a;
        a          = t;
    }
    if (b == 0) {
        return a;
    }
    a %= b;
    // Early return for b=GCD(a,b) which is expected to be common
    if (a == 0) {
        return b;
    }
    uint64_t t;
    int az    = ZL_ctz64(a);
    int bz    = ZL_ctz64(b);
    int shift = az < bz ? az : bz;
    a >>= az;
    b >>= bz;
    while (a != b) {
        if (a < b) {
            t = b - a;
        } else {
            t = a - b;
            a = b;
        }
        b = t >> ZL_ctz64(t);
    }
    return a << shift;
}

ZL_INLINE size_t ZL_firstIndexNotDivisibleBy8(
        uint8_t const* src,
        size_t srcSize,
        uint64_t divisor)
{
    int shift        = 0;
    uint8_t overflow = (uint8_t)(UCHAR_MAX / divisor);
    uint8_t inverse  = ZL_getMultiplicativeInverse8(&shift, (uint8_t)divisor);
    for (size_t i = 0; i < srcSize; ++i) {
        uint8_t val = ((const uint8_t*)src)[i];
        uint8_t div = (uint8_t)(val * inverse >> shift);
        uint8_t dup = div * (uint8_t)divisor;
        if (dup != val || div > overflow) {
            return i;
        }
    }
    return srcSize;
}

ZL_INLINE size_t ZL_firstIndexNotDivisibleBy16(
        uint16_t const* src,
        size_t srcSize,
        uint64_t divisor)
{
    int shift         = 0;
    uint16_t overflow = (uint16_t)(USHRT_MAX / divisor);
    uint32_t inverse = ZL_getMultiplicativeInverse32(&shift, (uint32_t)divisor);
    for (size_t i = 0; i < srcSize; ++i) {
        uint16_t val = ((const uint16_t*)src)[i];
        uint32_t div = (uint16_t)(val * inverse >> shift);
        uint16_t dup = (uint16_t)(div * (uint32_t)divisor);
        if (dup != val || div > overflow) {
            return i;
        }
    }
    return srcSize;
}

ZL_INLINE size_t ZL_firstIndexNotDivisibleBy32(
        uint32_t const* src,
        size_t srcSize,
        uint64_t divisor)
{
    int shift         = 0;
    uint32_t overflow = (uint32_t)(UINT_MAX / divisor);
    uint32_t inverse = ZL_getMultiplicativeInverse32(&shift, (uint32_t)divisor);
    for (size_t i = 0; i < srcSize; ++i) {
        uint32_t val = ((const uint32_t*)src)[i];
        uint32_t div = val * inverse >> shift;
        uint32_t dup = div * (uint32_t)divisor;
        if (dup != val || div > overflow) {
            return i;
        }
    }
    return srcSize;
}

ZL_INLINE size_t ZL_firstIndexNotDivisibleBy64(
        uint64_t const* src,
        size_t srcSize,
        uint64_t divisor)
{
    int shift         = 0;
    uint64_t overflow = ULLONG_MAX / divisor;
    uint64_t inverse = ZL_getMultiplicativeInverse64(&shift, (uint64_t)divisor);
    for (size_t i = 0; i < srcSize; ++i) {
        uint64_t val = ((const uint64_t*)src)[i];
        uint64_t div = val * inverse >> shift;
        uint64_t dup = div * (uint64_t)divisor;
        if (dup != val || div > overflow) {
            return i;
        }
    }
    return srcSize;
}

/**
 * Returns the GCD of all values in the array @p in with length @p inputLength
 *and width @p intWidth. Returns 0 if the array is empty or all 0s.
 **/
ZL_INLINE uint64_t
ZL_gcdVec(const void* in, size_t inputLength, size_t intWidth)
{
    if (inputLength <= 1) {
        return 1;
    }
    size_t i;
    uint64_t gcd = 0;
    switch (intWidth) {
        case 1: {
            const uint8_t* in8 = (const uint8_t*)in;
            for (i = 0; i < inputLength && in8[i] == 0; ++i) {
            }
            if (i == inputLength) {
                // All values are 0
                return 1;
            }
            gcd = in8[i++];
            for (;;) {
                i += ZL_firstIndexNotDivisibleBy8(
                        in8 + i, inputLength - i, gcd);
                if (i == inputLength)
                    break;
                gcd = ZL_gcdImpl(in8[i], gcd);
                i++;
            }
            break;
        }
        case 2: {
            const uint16_t* in16 = (const uint16_t*)in;
            for (i = 0; i < inputLength && in16[i] == 0; ++i) {
            }
            if (i == inputLength) {
                // All values are 0
                return 1;
            }
            gcd = in16[i++];
            for (;;) {
                i += ZL_firstIndexNotDivisibleBy16(
                        in16 + i, inputLength - i, gcd);
                if (i == inputLength)
                    break;
                gcd = ZL_gcdImpl(in16[i], gcd);
                i++;
            }
            break;
        }
        case 4: {
            const uint32_t* in32 = (const uint32_t*)in;
            for (i = 0; i < inputLength && in32[i] == 0; ++i) {
            }
            if (i == inputLength) {
                // All values are 0
                return 1;
            }
            gcd = in32[i++];
            for (;;) {
                i += ZL_firstIndexNotDivisibleBy32(
                        in32 + i, inputLength - i, gcd);
                if (i == inputLength)
                    break;
                gcd = ZL_gcdImpl(in32[i], gcd);
                i++;
            }
            break;
        }
        case 8: {
            const uint64_t* in64 = (const uint64_t*)in;
            for (i = 0; i < inputLength && in64[i] == 0; ++i) {
            }
            if (i == inputLength) {
                // All values are 0
                return 1;
            }
            gcd = in64[i++];
            for (;;) {
                i += ZL_firstIndexNotDivisibleBy64(
                        in64 + i, inputLength - i, gcd);
                if (i == inputLength)
                    break;
                gcd = ZL_gcdImpl(in64[i], gcd);
                i++;
            }
            break;
        }
    }

    return gcd;
}

ZL_END_C_DECLS

#endif // ZSTRONG_TRANSFORMS_DIVIDE_COMMON_GCD_H

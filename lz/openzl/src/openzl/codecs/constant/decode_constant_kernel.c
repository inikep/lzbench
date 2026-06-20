// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/constant/decode_constant_kernel.h"

#include <string.h>

#include "openzl/common/assertion.h"

// Local noinline attribute for decode_constant_kernel
#ifdef _MSC_VER
#    define ZL_CONSTANT_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#    define ZL_CONSTANT_NOINLINE __attribute__((noinline))
#else
#    define ZL_CONSTANT_NOINLINE
#endif

ZL_FORCE_INLINE void ZS_decodeConstant_impl_fallback(
        uint8_t* const dst,
        size_t const dstNbElts,
        const uint8_t* const src,
        size_t const eltWidth,
        void* kEltBuffer,
        size_t const kEltWidth)
{
    ZL_ASSERT_GE(2 * eltWidth, kEltWidth);
    uint8_t* nextElt = dst;
    memcpy(kEltBuffer, src, eltWidth);
    for (size_t i = 0; i < dstNbElts - 1; ++i) {
        memcpy(nextElt, kEltBuffer, kEltWidth);
        nextElt += eltWidth;
    }
    memcpy(nextElt, kEltBuffer, eltWidth);
}

#if ZL_HAS_AVX2
#    include <immintrin.h>

ZL_FORCE_INLINE void ZS_decodeConstant_impl_avx2(
        uint8_t* const dst,
        size_t const dstNbElts,
        const uint8_t* const src,
        size_t const eltWidth,
        void* kEltBuffer,
        size_t const kEltWidth)
{
    // Load vector with max copies of src
    __m256i vec;
    size_t const eltsPerStore     = sizeof(__m256i) / eltWidth;
    size_t const eltBytesPerStore = eltsPerStore * eltWidth;
    for (size_t i = 0; i < eltBytesPerStore; i += eltWidth) {
        memcpy((uint8_t*)&vec + i, src, eltWidth);
    }

    // Repeatedly store into dst buffer
    size_t const totalDstBytes    = dstNbElts * eltWidth;
    uint8_t* const storeLoopLimit = dst + totalDstBytes - sizeof(__m256i) - 1;
    size_t const nbStores =
            (totalDstBytes - sizeof(__m256i) + (eltBytesPerStore - 1))
            / eltBytesPerStore;
    uint8_t* nextElt = dst;
    while (nextElt <= storeLoopLimit) {
        _mm256_storeu_si256((__m256i_u*)nextElt, vec);
        nextElt += eltBytesPerStore;
    }
    size_t const eltsStored   = eltsPerStore * nbStores;
    size_t const remDstNbElts = dstNbElts - eltsStored;

    ZS_decodeConstant_impl_fallback(
            nextElt, remDstNbElts, src, eltWidth, kEltBuffer, kEltWidth);
}

#endif // ZL_HAS_AVX2

ZL_FORCE_INLINE void ZS_decodeConstant_impl(
        uint8_t* const dst,
        size_t const dstNbElts,
        const uint8_t* const src,
        size_t const eltWidth,
        void* kEltBuffer,
        size_t const kEltWidth)
{
    if (eltWidth == kEltWidth) {
        uint8_t* nextElt = dst;
        for (size_t i = 0; i < dstNbElts; ++i) {
            memcpy(nextElt, src, kEltWidth);
            nextElt += kEltWidth;
        }
    } else {
#if ZL_HAS_AVX2
        if (eltWidth <= 16) {
            ZS_decodeConstant_impl_avx2(
                    dst, dstNbElts, src, eltWidth, kEltBuffer, kEltWidth);
        } else {
            ZS_decodeConstant_impl_fallback(
                    dst, dstNbElts, src, eltWidth, kEltBuffer, kEltWidth);
        }
#else
        ZS_decodeConstant_impl_fallback(
                dst, dstNbElts, src, eltWidth, kEltBuffer, kEltWidth);
#endif
    }
}

#define ZS_GEN_DECODE_CONSTANT(kEltWidth)                             \
    static ZL_CONSTANT_NOINLINE void ZS_decodeConstant_##kEltWidth(   \
            uint8_t* const dst,                                       \
            size_t const dstNbElts,                                   \
            const uint8_t* const src,                                 \
            size_t const eltWidth,                                    \
            void* eltBuffer)                                          \
    {                                                                 \
        ZS_decodeConstant_impl(                                       \
                dst, dstNbElts, src, eltWidth, eltBuffer, kEltWidth); \
    }

ZS_GEN_DECODE_CONSTANT(1)
ZS_GEN_DECODE_CONSTANT(2)
ZS_GEN_DECODE_CONSTANT(4)
ZS_GEN_DECODE_CONSTANT(8)
ZS_GEN_DECODE_CONSTANT(16)
ZS_GEN_DECODE_CONSTANT(32)

static ZL_CONSTANT_NOINLINE void ZS_decodeConstant_generic(
        uint8_t* const dst,
        size_t const dstNbElts,
        const uint8_t* const src,
        size_t const eltWidth,
        void* buffer)
{
    ZS_decodeConstant_impl(dst, dstNbElts, src, eltWidth, buffer, eltWidth);
}

void ZS_decodeConstant(
        uint8_t* const dst,
        size_t const dstNbElts,
        const uint8_t* const src,
        size_t const eltWidth,
        void* buffer)
{
    if (eltWidth == 1) {
        ZS_decodeConstant_1(dst, dstNbElts, src, eltWidth, buffer);
    } else if (eltWidth == 2) {
        ZS_decodeConstant_2(dst, dstNbElts, src, eltWidth, buffer);
    } else if (eltWidth <= 4) {
        ZS_decodeConstant_4(dst, dstNbElts, src, eltWidth, buffer);
    } else if (eltWidth <= 8) {
        ZS_decodeConstant_8(dst, dstNbElts, src, eltWidth, buffer);
    } else if (eltWidth <= 16) {
        ZS_decodeConstant_16(dst, dstNbElts, src, eltWidth, buffer);
    } else if (eltWidth <= 32) {
        ZS_decodeConstant_32(dst, dstNbElts, src, eltWidth, buffer);
    } else {
        ZS_decodeConstant_generic(dst, dstNbElts, src, eltWidth, buffer);
    }
}

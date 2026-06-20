// Copyright (c) Meta Platforms, Inc. and affiliates.

/// MinGW: Use the ANSI stdio functions (e.g. to get correct printf for 64-bits)
#undef __USE_MINGW_ANSI_STDIO
#define __USE_MINGW_ANSI_STDIO 1

/* ===   Dependencies   === */
#include "benchmark/unitBench/scenarios/codecs/entropy.h"

#include "openzl/codecs/entropy/deprecated/common_entropy.h"

size_t entropyEncode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;

    ZS_Entropy_EncodeParameters params = {
        .allowedTypes = ZS_Entropy_TypeMask_huf | ZS_Entropy_TypeMask_multi,
        .encodeSpeed  = ZL_EncodeSpeed_fromBaseline(ZL_EncodeSpeedBaseline_any),
        .decodeSpeed  = ZL_DecodeSpeed_fromBaseline(ZL_DecodeSpeedBaseline_any),
        .precomputedHistogram = NULL,
        .cardinalityEstimate  = 0,
        .maxValueUpperBound   = 0,
        .allowAvx2Huffman     = true,
        .blockSplits          = NULL,
        .tableManager         = NULL,
    };

    uint8_t* const dst8 = (uint8_t*)dst;
    ZL_WC wc            = ZL_WC_wrap(dst8, dstCapacity);
    ZL_REQUIRE_SUCCESS(ZS_Entropy_encode(&wc, src, srcSize, 1, &params));
    size_t const csize = (size_t)(ZL_WC_ptr(&wc) - dst8);
    return csize;
}

size_t
entropyDecode_preparation(void* src, size_t srcSize, const BenchPayload* bp)
{
    (void)bp;
    ZS_Entropy_EncodeParameters params = {
        .allowedTypes = ZS_Entropy_TypeMask_huf | ZS_Entropy_TypeMask_multi,
        .encodeSpeed  = ZL_EncodeSpeed_fromBaseline(ZL_EncodeSpeedBaseline_any),
        .decodeSpeed  = ZL_DecodeSpeed_fromBaseline(ZL_DecodeSpeedBaseline_any),
        .precomputedHistogram = NULL,
        .cardinalityEstimate  = 0,
        .maxValueUpperBound   = 0,
        .allowAvx2Huffman     = true,
        .tableManager         = NULL,
    };

    size_t const dstCapacity = ZS_Entropy_encodedSizeBound(srcSize, 1);

    uint8_t* const dst = (uint8_t*)malloc(dstCapacity);
    ZL_REQUIRE_NN(dst);
    ZL_WC wc = ZL_WC_wrap(dst, dstCapacity);
    ZL_REQUIRE_SUCCESS(ZS_Entropy_encode(&wc, src, srcSize, 1, &params));
    size_t const csize = (size_t)(ZL_WC_ptr(&wc) - dst);
    ZL_REQUIRE_LE(csize, srcSize);
    memcpy(src, dst, csize);
    free(dst);
    ZL_LOG(V, "prepared %zu -> %zu", srcSize, csize);
    return csize;
}

size_t fseEncode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;

    ZS_Entropy_EncodeParameters const params =
            ZS_Entropy_EncodeParameters_fromAllowedTypes(
                    ZS_Entropy_TypeMask_fse);

    uint8_t* const dst8 = (uint8_t*)dst;
    ZL_WC wc            = ZL_WC_wrap(dst8, dstCapacity);
    ZL_REQUIRE_SUCCESS(ZS_Entropy_encode(&wc, src, srcSize, 1, &params));
    size_t const csize = (size_t)(ZL_WC_ptr(&wc) - dst8);
    return csize;
}

size_t fseDecode_preparation(void* src, size_t srcSize, const BenchPayload* bp)
{
    (void)bp;
    ZS_Entropy_EncodeParameters const params =
            ZS_Entropy_EncodeParameters_fromAllowedTypes(
                    ZS_Entropy_TypeMask_fse);

    size_t const dstCapacity = ZS_Entropy_encodedSizeBound(srcSize, 1);

    uint8_t* const dst = (uint8_t*)malloc(dstCapacity);
    ZL_REQUIRE_NN(dst);
    ZL_WC wc = ZL_WC_wrap(dst, dstCapacity);
    ZL_REQUIRE_SUCCESS(ZS_Entropy_encode(&wc, src, srcSize, 1, &params));
    size_t const csize = (size_t)(ZL_WC_ptr(&wc) - dst);
    ZL_REQUIRE_LE(csize, srcSize);
    memcpy(src, dst, csize);
    free(dst);
    ZL_LOG(V, "prepared %zu -> %zu", srcSize, csize);
    return csize;
}

size_t entropyDecode_outSize(void const* src, size_t srcSize)
{
    (void)src;
    return 10 * srcSize;
}

size_t entropyDecode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;

    ZS_Entropy_DecodeParameters params = ZS_Entropy_DecodeParameters_default();

    ZL_RC rc            = ZL_RC_wrap((uint8_t const*)src, srcSize);
    ZL_Report const ret = ZS_Entropy_decode(dst, dstCapacity, &rc, 1, &params);
    ZL_REQUIRE_SUCCESS(ret);

    return ZL_validResult(ret);
}
void entropyDecode_displayResult(
        const char* srcname,
        const char* fname,
        BMK_runTime_t rt,
        size_t srcSize)
{
    double const sec           = rt.nanoSecPerRun / 1e+9;
    double const nbRunsPerSec  = 1. / sec;
    double const nbBytesPerSec = nbRunsPerSec * (double)rt.sumOfReturn;

    printf("%s: decode %zu bytes into %zu 8-bit tokens in %.2f ms  ==> %.1f MB/s \n",
           fname,
           srcSize,
           rt.sumOfReturn,
           sec * 1000.,
           nbBytesPerSec / (1 << 20));
    (void)srcname;
    fflush(NULL);
}

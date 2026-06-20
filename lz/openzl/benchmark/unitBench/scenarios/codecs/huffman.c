// Copyright (c) Meta Platforms, Inc. and affiliates.

/// MinGW: Use the ANSI stdio functions (e.g. to get correct printf for 64-bits)
#undef __USE_MINGW_ANSI_STDIO
#define __USE_MINGW_ANSI_STDIO 1

#include "benchmark/unitBench/scenarios/codecs/huffman.h"

#include "openzl/codecs/entropy/decode_huffman_kernel.h"
#include "openzl/codecs/entropy/encode_huffman_kernel.h"

size_t largeHuffmanEncode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;

    const uint16_t* const src16 = src;
    size_t const src16Size      = srcSize / sizeof(uint16_t);
    uint8_t* dst8               = dst;
    size_t const dst8Size       = dstCapacity;

    ZL_WC wc = ZL_WC_wrap(dst8, dst8Size);
    ZL_REQUIRE_SUCCESS(
            ZS_largeHuffmanEncode(&wc, src16, src16Size, (uint16_t)-1, 0));
    size_t const csize = (size_t)(ZL_WC_ptr(&wc) - dst8);

    if (0) {
        FILE* f = fopen("out.lh", "wb");
        ZL_REQUIRE_NN(f);
        ZL_REQUIRE_EQ(csize, fwrite(dst8, 1, csize, f));
        fclose(f);
    }

    return csize;
}
void largeHuffmanEncode_displayResult(
        const char* srcname,
        const char* fname,
        BMK_runTime_t rt,
        size_t srcSize)
{
    double const sec              = rt.nanoSecPerRun / 1000000000.;
    double const nbRunsPerSec     = 1. / sec;
    double const nbBytesPerSec    = nbRunsPerSec * (double)srcSize;
    double const compressionRatio = (double)srcSize / (double)rt.sumOfReturn;

    printf("%s: encode %zu 16-bit tokens into %zu bytes (%.2f) in %.2f ms  ==> %.1f MB/s \n",
           fname,
           srcSize / 2,
           rt.sumOfReturn,
           compressionRatio,
           sec * 1000.,
           nbBytesPerSec / (1 << 20));
    (void)srcname;
    fflush(NULL);
}

size_t largeHuffmanDecode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;

    uint16_t* const dst16      = dst;
    size_t const dst16Capacity = dstCapacity / sizeof(uint16_t);

    ZL_RC rc            = ZL_RC_wrap((uint8_t const*)src, srcSize);
    ZL_Report const ret = ZS_largeHuffmanDecode(dst16, dst16Capacity, &rc);
    ZL_REQUIRE_SUCCESS(ret);

    return ZL_validResult(ret);
}
void largeHuffmanDecode_displayResult(
        const char* srcname,
        const char* fname,
        BMK_runTime_t rt,
        size_t srcSize)
{
    double const sec           = rt.nanoSecPerRun / 1000000000.;
    double const nbRunsPerSec  = 1. / sec;
    double const nbBytesPerSec = nbRunsPerSec * (double)(srcSize * 2);

    printf("%s: decode %zu bytes into %zu 16-bit tokens in %.2f ms  ==> %.1f MB/s \n",
           fname,
           srcSize,
           rt.sumOfReturn,
           sec * 1000.,
           nbBytesPerSec / (1 << 20));
    (void)srcname;
    fflush(NULL);
}

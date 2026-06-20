// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "benchmark/unitBench/scenarios/codecs/dispatch_by_tag.h"

// splitBy test : splitting an input into a nb of dst buffers decided at runtime
//                splitting is guided by a context stream
#include "openzl/codecs/dispatch_by_tag/encode_dispatch_by_tag_kernel.h"

static size_t splitBy_internal(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        size_t eltSize)
{
    // In this scenario, input is presumed prepared.
    // It consists in elts of fixed size eltSize (rounded down)
    // IndexBuffer is effectively overlapped with first part of the input
    // In order to limit the nb of dst output to 4,
    // input's values must be modified using splitBy_preparation.
    // dstCapacity must be >= 4 x srcSize to guarantee non overlapping output

#define SB8_NB_DST_BUFFERS 4
    size_t const nbElts = srcSize / eltSize;
    assert(dstCapacity >= SB8_NB_DST_BUFFERS * srcSize);
    (void)dstCapacity;

    void* dstBuffers[SB8_NB_DST_BUFFERS] = { dst,
                                             (char*)dst + srcSize,
                                             (char*)dst + 2 * srcSize,
                                             (char*)dst + 3 * srcSize };

    ZS_DispatchByTag_encode(
            dstBuffers, SB8_NB_DST_BUFFERS, src, nbElts, eltSize, src);

    return srcSize;
}

size_t splitBy8_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    return splitBy_internal(src, srcSize, dst, dstCapacity, 8);
}

// --8<-- [start:splitBy8_preparation]
static size_t splitBy_prepInternal(void* src, size_t srcSize, size_t eltSize)
{
    size_t const nbElts = srcSize / eltSize;
    uint8_t* const src8 = src;
    for (size_t n = 0; n < nbElts; n++) {
        src8[n] = src8[n] % SB8_NB_DST_BUFFERS;
    }
    return srcSize;
}

size_t splitBy8_preparation(void* src, size_t srcSize, const BenchPayload* bp)
{
    (void)bp;
    return splitBy_prepInternal(src, srcSize, 8);
}
// --8<-- [end:splitBy8_preparation]

size_t splitBy4_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    return splitBy_internal(src, srcSize, dst, dstCapacity, 4);
}

size_t splitBy4_preparation(void* src, size_t srcSize, const BenchPayload* bp)
{
    (void)bp;
    return splitBy_prepInternal(src, srcSize, 4);
}

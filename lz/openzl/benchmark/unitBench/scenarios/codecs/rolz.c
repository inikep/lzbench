// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "benchmark/unitBench/scenarios/codecs/rolz.h"

#include "openzl/codecs/rolz/encode_rolz_kernel.h"

size_t rolzc_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    ZL_Report const res = ZS_rolzCompress(dst, dstCapacity, src, srcSize);
    assert(!ZL_isError(res));
    return ZL_validResult(res);
}

size_t fastlz_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    ZL_Report const res = ZS_fastLzCompress(dst, dstCapacity, src, srcSize);
    assert(!ZL_isError(res));
    return ZL_validResult(res);
}

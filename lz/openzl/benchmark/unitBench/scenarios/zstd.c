// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "benchmark/unitBench/scenarios/zstd.h"

#include <zstd.h>

size_t zstd_outcSize(const void* src, size_t srcSize)
{
    (void)src;
    return ZSTD_compressBound(srcSize);
}
size_t zstd_outdSize(const void* src, size_t srcSize)
{
    unsigned long long const dSize = ZSTD_getFrameContentSize(src, srcSize);
    assert(dSize < (2ULL << 30));
    return (size_t)dSize;
}
size_t zstd_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    BenchPayload* payload = customPayload;
    size_t const res =
            ZSTD_compress(dst, dstCapacity, src, srcSize, payload->intParam);
    assert(!ZSTD_isError(res));
    return res;
}
size_t zstdd_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    size_t const res = ZSTD_decompress(dst, dstCapacity, src, srcSize);
    assert(!ZSTD_isError(res));
    return res;
}
size_t zstddctx_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    static ZSTD_DCtx* dctx = NULL;
    if (dctx == NULL) {
        dctx = ZSTD_createDCtx();
        assert(dctx != NULL);
    }
    (void)customPayload;
    size_t const res =
            ZSTD_decompressDCtx(dctx, dst, dstCapacity, src, srcSize);
    assert(!ZSTD_isError(res));
    return res;
}

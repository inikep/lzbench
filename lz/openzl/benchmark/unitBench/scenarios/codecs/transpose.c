// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "benchmark/unitBench/scenarios/codecs/transpose.h"

#include "openzl/codecs/transpose/decode_transpose_kernel.h"
#include "openzl/codecs/transpose/encode_transpose_kernel.h"

size_t transposeEncode16_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload, (void)dstCapacity;
    assert(dstCapacity >= srcSize);
    ZS_transposeEncode(dst, src, srcSize / 2, 2);
    return srcSize;
}
size_t transposeEncode32_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload, (void)dstCapacity;
    assert(dstCapacity >= srcSize);
    ZS_transposeEncode(dst, src, srcSize / 4, 4);
    return srcSize;
}
size_t transposeEncode64_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload, (void)dstCapacity;
    assert(dstCapacity >= srcSize);
    ZS_transposeEncode(dst, src, srcSize / 8, 8);
    return srcSize;
}

size_t transposeDecode16_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload, (void)dstCapacity;
    assert(dstCapacity >= srcSize);
    ZS_transposeDecode(dst, src, srcSize / 2, 2);
    return srcSize;
}
size_t transposeDecode32_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload, (void)dstCapacity;
    assert(dstCapacity >= srcSize);
    ZS_transposeDecode(dst, src, srcSize / 4, 4);
    return srcSize;
}
size_t transposeDecode64_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload, (void)dstCapacity;
    assert(dstCapacity >= srcSize);
    ZS_transposeDecode(dst, src, srcSize / 8, 8);
    return srcSize;
}

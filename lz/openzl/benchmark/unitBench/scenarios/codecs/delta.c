// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "benchmark/unitBench/scenarios/codecs/delta.h"

#include "openzl/codecs/delta/decode_delta_kernel.h"
#include "openzl/codecs/delta/encode_delta_kernel.h"

size_t deltaEncode32_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dstCapacity;
    uint32_t first;
    ZS_deltaEncode32(&first, dst, src, srcSize / 4); // can't fail
    return srcSize;
}
size_t deltaEncode64_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dstCapacity;
    uint64_t first;
    ZS_deltaEncode64(&first, dst, src, srcSize / 8); // can't fail
    return srcSize;
}

size_t deltaDecode8_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dstCapacity;
    uint8_t const* src8 = (uint8_t const*)src;
    ZS_deltaDecode8(dst, src8[0], src8 + 1, srcSize); // can't fail
    return srcSize;
}
size_t deltaDecode16_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dstCapacity;
    uint16_t const* src16 = (uint16_t const*)src;
    ZS_deltaDecode16(dst, src16[0], src16 + 1, srcSize / 2); // can't fail
    return srcSize;
}
size_t deltaDecode32_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dstCapacity;
    uint32_t const* src32 = (uint32_t const*)src;
    ZS_deltaDecode32(dst, src32[0], src32 + 1, srcSize / 4); // can't fail
    return srcSize;
}
size_t deltaDecode64_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    (void)dstCapacity;
    uint64_t const* src64 = (uint64_t const*)src;
    ZS_deltaDecode64(dst, src64[0], src64 + 1, srcSize / 8); // can't fail
    return srcSize;
}

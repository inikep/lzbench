// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "benchmark/unitBench/scenarios/codecs/flatpack.h"

#include "openzl/codecs/flatpack/decode_flatpack_kernel.h"
#include "openzl/codecs/flatpack/encode_flatpack_kernel.h"

static size_t
flatpackDecodeX_prep(void* src, size_t srcSize, uint8_t mask, int mod)
{
    ZL_REQUIRE_NE(srcSize, 0);

    uint8_t* out = (uint8_t*)malloc(256 + srcSize + 2);
    ZL_REQUIRE_NN(out);

    uint8_t* const ip = src;
    if (mod) {
        for (size_t i = 0; i < srcSize; ++i) {
            ip[i] = (uint8_t)(mask + ip[i] % mod);
        }
    } else {
        for (size_t i = 0; i < srcSize; ++i) {
            ip[i] = (uint8_t)(ip[i] & mask);
        }
    }

    ZS_FlatPackSize const size = ZS_flatpackEncode(
            out + 1, 256, out + 1 + 256, srcSize + 1, src, srcSize);
    ZL_REQUIRE(!ZS_FlatPack_isError(size));
    out[0] = (uint8_t)(ZS_FlatPack_alphabetSize(size) - 1);

    size_t const outSize = 1 + 256 + ZS_FlatPack_packedSize(size, srcSize);
    memcpy(src, out, outSize);

    free(out);

    return outSize;
}

size_t flatpackDecode16_prep(void* src, size_t srcSize, const BenchPayload* bp)
{
    (void)bp;
    return flatpackDecodeX_prep(src, srcSize, 0x55, 0);
}

size_t flatpackDecode32_prep(void* src, size_t srcSize, const BenchPayload* bp)
{
    (void)bp;
    return flatpackDecodeX_prep(src, srcSize, 0xd5, 0);
}

size_t flatpackDecode48_prep(void* src, size_t srcSize, const BenchPayload* bp)
{
    (void)bp;
    return flatpackDecodeX_prep(src, srcSize, 42, 48);
}

size_t flatpackDecode64_prep(void* src, size_t srcSize, const BenchPayload* bp)
{
    (void)bp;
    return flatpackDecodeX_prep(src, srcSize, 0xdd, 0);
}

size_t flatpackDecode128_prep(void* src, size_t srcSize, const BenchPayload* bp)
{
    (void)bp;
    return flatpackDecodeX_prep(src, srcSize, 0xfd, 0);
}

size_t flatpackDecode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;

    uint8_t const* const ip       = (uint8_t const*)src;
    size_t const alphabetSize     = (size_t)ip[0] + 1;
    uint8_t const* const alphabet = ip + 1;
    uint8_t const* const packed   = alphabet + 256;
    size_t const packedSize       = srcSize - 256 - 1;
    ZL_ASSERT_GE(srcSize, 256 + 1);
    ZS_FlatPackSize const size = ZS_flatpackDecode(
            dst, dstCapacity, alphabet, alphabetSize, packed, packedSize);
    ZL_ASSERT(!ZS_FlatPack_isError(size));

    return ZS_FlatPack_nbElts(alphabetSize, packed, packedSize);
}

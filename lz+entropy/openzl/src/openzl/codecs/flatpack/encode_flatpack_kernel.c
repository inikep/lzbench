// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "openzl/codecs/flatpack/encode_flatpack_kernel.h"

#include "openzl/shared/mem.h"

static ZS_FlatPackSize const ZS_FlatPack_kError = { 257 };

static void ZS_FlatPack_pack_generic(
        uint8_t const symbolMap[256],
        size_t nbBits,
        uint8_t* packed,
        size_t packedSize,
        uint8_t const* src,
        size_t srcSize)
{
    uint8_t* const packedEnd    = packed + packedSize;
    uint8_t const* const srcEnd = src + srcSize;
    size_t bits                 = 0;
    size_t state                = 0;
    while (src < srcEnd) {
        state |= (size_t)symbolMap[*src++] << bits;
        bits += nbBits;
        if (bits >= 8) {
            *packed++ = (uint8_t)state;
            bits -= 8;
            state >>= 8;
        }
    }
    ZL_ASSERT_LT(bits, 8);
    state |= (size_t)1 << bits;
    *packed++ = (uint8_t)state;
    ZL_ASSERT_EQ(packed, packedEnd);
}

#if ZL_HAS_BMI2

#    include <immintrin.h>

static void ZS_FlatPack_pack_bmi2(
        uint8_t const symbolMap[256],
        size_t nbBits,
        uint8_t* packed,
        size_t packedSize,
        uint8_t const* src,
        size_t srcSize)
{
    uint8_t* const packedEnd    = packed + packedSize;
    uint8_t const* const srcEnd = src + srcSize;
    {
        uint8_t* const packedLimit = packedEnd - 8;
        uint64_t const mask = ((1ull << nbBits) - 1) * 0x0101010101010101ULL;
        size_t const bytesPerLoop = nbBits;
        while (packed < packedLimit) {
            uint8_t symbols[8];
            for (size_t i = 0; i < 8; ++i) {
                symbols[i] = symbolMap[src[i]];
            }
            uint64_t const bytes = ZL_readLE64(symbols);
            uint64_t const bits  = _pext_u64(bytes, mask);
            ZL_writeLE64(packed, bits);
            packed += bytesPerLoop;
            src += 8;
        }
    }
    ZL_ASSERT_LE(src, srcEnd);
    ZS_FlatPack_pack_generic(
            symbolMap,
            nbBits,
            packed,
            (size_t)(packedEnd - packed),
            src,
            (size_t)(srcEnd - src));
}

#endif

static void ZS_FlatPack_pack(
        uint8_t const symbolMap[256],
        size_t nbBits,
        uint8_t* packed,
        size_t packedSize,
        uint8_t const* src,
        size_t srcSize)
{
#if ZL_HAS_BMI2
    ZS_FlatPack_pack_bmi2(symbolMap, nbBits, packed, packedSize, src, srcSize);
#else
    ZS_FlatPack_pack_generic(
            symbolMap, nbBits, packed, packedSize, src, srcSize);
#endif
}

ZS_FlatPackSize ZS_flatpackEncode(
        uint8_t* alphabet,
        size_t alphabetCapacity,
        uint8_t* packed,
        size_t packedCapacity,
        uint8_t const* src,
        size_t srcSize)
{
    if (srcSize == 0) {
        return (ZS_FlatPackSize){ 0 };
    }

    uint8_t symbolMap[256] = { 0 };
    for (size_t i = 0; i < srcSize; ++i) {
        symbolMap[src[i]] = 1;
    }
    size_t nbSymbols = 0;
    for (size_t s = 0; s < 256; ++s) {
        if (symbolMap[s]) {
            symbolMap[s] = (uint8_t)nbSymbols;
            if (nbSymbols >= alphabetCapacity) {
                return ZS_FlatPack_kError;
            }
            alphabet[nbSymbols] = (uint8_t)s;
            ++nbSymbols;
        }
    }
    ZS_FlatPackSize size    = { nbSymbols };
    size_t const nbBits     = ZS_FlatPack_nbBits(size);
    size_t const packedSize = ZS_FlatPack_packedSize(size, srcSize);
    if (packedSize > packedCapacity) {
        return ZS_FlatPack_kError;
    }

    ZS_FlatPack_pack(symbolMap, nbBits, packed, packedSize, src, srcSize);

    return size;
}

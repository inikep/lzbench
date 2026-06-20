// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "openzl/codecs/flatpack/decode_flatpack_kernel.h"

#include "openzl/shared/mem.h"

// Local noinline attribute for decode_flatpack_kernel
#ifdef _MSC_VER
#    define ZL_FLATPACK_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#    define ZL_FLATPACK_NOINLINE __attribute__((noinline))
#else
#    define ZL_FLATPACK_NOINLINE
#endif

#define ZS_FLAT_HAS_NEEDED_EXTS (ZL_HAS_BMI2 && ZL_HAS_SSE42)

static ZS_FlatPackSize const ZS_FlatPack_kError = { 257 };

static ZS_FlatPackSize ZS_flatpackDecodeEnd(
        uint8_t* dst,
        uint8_t* dstEnd,
        uint8_t const* alphabet,
        size_t alphabetSize,
        uint8_t const* packed,
        uint8_t const* packedEnd,
        size_t nbBits)
{
    ZS_FlatPackSize const size = { alphabetSize };
    size_t bits                = 0;
    size_t state               = 0;
    size_t const mask          = (1u << nbBits) - 1;
    while (dst < dstEnd) {
        if (bits < nbBits) {
            if (packed == packedEnd) {
                return ZS_FlatPack_kError;
            }
            state |= (size_t)*packed++ << bits;
            bits += 8;
        }
        size_t const idx = state & mask;
        if (idx >= alphabetSize) {
            return ZS_FlatPack_kError;
        }
        *dst++ = alphabet[idx];
        state >>= nbBits;
        bits -= nbBits;
    }
    if (packed < packedEnd) {
        state |= (size_t)*packed++ << bits;
        bits += 8;
    }
    if (packed != packedEnd) {
        return ZS_FlatPack_kError;
    }
    if (state != 1 || bits > 8) {
        return ZS_FlatPack_kError;
    }

    return size;
}

#if ZS_FLAT_HAS_NEEDED_EXTS

#    include <immintrin.h>

static __m128i
ZS_loadAlphabet16(uint8_t const* alphabet, size_t offset, size_t alphabetSize)
{
    if (offset >= alphabetSize) {
        return _mm_setzero_si128();
    }
    if (alphabetSize - offset >= 16) {
        return _mm_loadu_si128(
                (__m128i_u const*)(void const*)(alphabet + offset));
    }
    uint8_t tmp[16] = { 0 };
    ZL_memcpy(tmp, alphabet + offset, alphabetSize - offset);
    return _mm_loadu_si128((__m128i_u const*)(void const*)tmp);
}

#    define ZS_blend0(b, s0, s1) _mm_blendv_epi8(s0, s1, b)
#    define ZS_blend1(b, s0, s1) \
        _mm_or_si128(_mm_andnot_si128(b, s0), _mm_and_si128(b, s1))

static ZL_FLATPACK_NOINLINE ZS_FlatPackSize ZS_flatpackDecode16(
        uint8_t* dst,
        uint8_t* dstEnd,
        uint8_t const* alphabet,
        size_t alphabetSize,
        uint8_t const* packed,
        uint8_t const* packedEnd,
        size_t nbBits)
{
    {
        __m128i const shuffle = ZS_loadAlphabet16(alphabet, 0, alphabetSize);
        uint8_t const* const packedLimit = packedEnd - 16;
        size_t const bytesPerLoop        = 2 * nbBits;
        uint64_t const mask = ((1ull << nbBits) - 1) * 0x0101010101010101ULL;
        while (packed < packedLimit) {
            uint64_t const bits0    = ZL_readLE64(packed);
            uint64_t const bits8    = ZL_readLE64(packed + nbBits);
            uint64_t const indices0 = _pdep_u64(bits0, mask);
            uint64_t const indices8 = _pdep_u64(bits8, mask);
            __m128i const indices =
                    _mm_set_epi64x((int64_t)indices8, (int64_t)indices0);
            __m128i const symbols = _mm_shuffle_epi8(shuffle, indices);
            _mm_storeu_si128((__m128i_u*)(void*)dst, symbols);
            packed += bytesPerLoop;
            dst += 16;
        }
    }
    ZL_ASSERT_LE(dst, dstEnd);
    return ZS_flatpackDecodeEnd(
            dst, dstEnd, alphabet, alphabetSize, packed, packedEnd, nbBits);
}

static ZL_FLATPACK_NOINLINE ZS_FlatPackSize ZS_flatpackDecode32(
        uint8_t* dst,
        uint8_t* dstEnd,
        uint8_t const* alphabet,
        size_t alphabetSize,
        uint8_t const* packed,
        uint8_t const* packedEnd,
        size_t nbBits)
{
    {
        __m128i const shuffle0 = ZS_loadAlphabet16(alphabet, 0, alphabetSize);
        __m128i const shuffle1 = ZS_loadAlphabet16(alphabet, 16, alphabetSize);
        __m128i const cmp      = _mm_set1_epi8(15);
        uint8_t const* const packedLimit = packedEnd - 16;
        size_t const bytesPerLoop        = 2 * nbBits;
        uint64_t const mask = ((1ull << nbBits) - 1) * 0x0101010101010101ULL;
        while (packed < packedLimit) {
            uint64_t const bits0    = ZL_readLE64(packed);
            uint64_t const bits8    = ZL_readLE64(packed + nbBits);
            uint64_t const indices0 = _pdep_u64(bits0, mask);
            uint64_t const indices8 = _pdep_u64(bits8, mask);
            __m128i const indices =
                    _mm_set_epi64x((int64_t)indices8, (int64_t)indices0);

            __m128i const blend    = _mm_cmpgt_epi8(indices, cmp);
            __m128i const symbols0 = _mm_shuffle_epi8(shuffle0, indices);
            __m128i const symbols1 = _mm_shuffle_epi8(shuffle1, indices);
            __m128i const symbols  = ZS_blend1(blend, symbols0, symbols1);

            _mm_storeu_si128((__m128i_u*)(void*)dst, symbols);
            packed += bytesPerLoop;
            dst += 16;
        }
    }
    ZL_ASSERT_LE(dst, dstEnd);
    return ZS_flatpackDecodeEnd(
            dst, dstEnd, alphabet, alphabetSize, packed, packedEnd, nbBits);
}

static ZL_FLATPACK_NOINLINE ZS_FlatPackSize ZS_flatpackDecode48(
        uint8_t* dst,
        uint8_t* dstEnd,
        uint8_t const* alphabet,
        size_t alphabetSize,
        uint8_t const* packed,
        uint8_t const* packedEnd,
        size_t nbBits)
{
    {
        __m128i const shuffle0 = ZS_loadAlphabet16(alphabet, 0, alphabetSize);
        __m128i const shuffle1 = ZS_loadAlphabet16(alphabet, 16, alphabetSize);
        __m128i const shuffle2 = ZS_loadAlphabet16(alphabet, 32, alphabetSize);
        __m128i const cmp0     = _mm_set1_epi8(15);
        __m128i const cmp1     = _mm_set1_epi8(31);
        uint8_t const* const packedLimit = packedEnd - 16;
        size_t const bytesPerLoop        = 2 * nbBits;
        uint64_t const mask = ((1ull << nbBits) - 1) * 0x0101010101010101ULL;
        while (packed < packedLimit) {
            uint64_t const bits0    = ZL_readLE64(packed);
            uint64_t const bits8    = ZL_readLE64(packed + nbBits);
            uint64_t const indices0 = _pdep_u64(bits0, mask);
            uint64_t const indices8 = _pdep_u64(bits8, mask);
            __m128i const indices =
                    _mm_set_epi64x((int64_t)indices8, (int64_t)indices0);

            __m128i const blend0   = _mm_cmpgt_epi8(indices, cmp0);
            __m128i const blend1   = _mm_cmpgt_epi8(indices, cmp1);
            __m128i const symbols0 = _mm_shuffle_epi8(shuffle0, indices);
            __m128i const symbols1 = _mm_shuffle_epi8(shuffle1, indices);
            __m128i const symbols2 = _mm_shuffle_epi8(shuffle2, indices);

            __m128i const symbols3 = ZS_blend1(blend0, symbols0, symbols1);
            __m128i const symbols  = ZS_blend1(blend1, symbols3, symbols2);

            _mm_storeu_si128((__m128i_u*)(void*)dst, symbols);
            packed += bytesPerLoop;
            dst += 16;
        }
    }
    ZL_ASSERT_LE(dst, dstEnd);
    return ZS_flatpackDecodeEnd(
            dst, dstEnd, alphabet, alphabetSize, packed, packedEnd, nbBits);
}

static ZL_FLATPACK_NOINLINE ZS_FlatPackSize ZS_flatpackDecode64(
        uint8_t* dst,
        uint8_t* dstEnd,
        uint8_t const* alphabet,
        size_t alphabetSize,
        uint8_t const* packed,
        uint8_t const* packedEnd,
        size_t nbBits)
{
    {
        __m128i const shuffle0 = ZS_loadAlphabet16(alphabet, 0, alphabetSize);
        __m128i const shuffle1 = ZS_loadAlphabet16(alphabet, 16, alphabetSize);
        __m128i const shuffle2 = ZS_loadAlphabet16(alphabet, 32, alphabetSize);
        __m128i const shuffle3 = ZS_loadAlphabet16(alphabet, 48, alphabetSize);
        __m128i const cmp0     = _mm_set1_epi8(15);
        __m128i const cmp1     = _mm_set1_epi8(31);
        __m128i const cmp2     = _mm_set1_epi8(47);
        uint8_t const* const packedLimit = packedEnd - 16;
        size_t const bytesPerLoop        = 2 * nbBits;
        uint64_t const mask = ((1ull << nbBits) - 1) * 0x0101010101010101ULL;
        while (packed < packedLimit) {
            uint64_t const bits0    = ZL_readLE64(packed);
            uint64_t const bits8    = ZL_readLE64(packed + nbBits);
            uint64_t const indices0 = _pdep_u64(bits0, mask);
            uint64_t const indices8 = _pdep_u64(bits8, mask);
            __m128i const indices =
                    _mm_set_epi64x((int64_t)indices8, (int64_t)indices0);

            __m128i const blend0   = _mm_cmpgt_epi8(indices, cmp0);
            __m128i const blend1   = _mm_cmpgt_epi8(indices, cmp1);
            __m128i const blend2   = _mm_cmpgt_epi8(indices, cmp2);
            __m128i const symbols0 = _mm_shuffle_epi8(shuffle0, indices);
            __m128i const symbols1 = _mm_shuffle_epi8(shuffle1, indices);
            __m128i const symbols2 = _mm_shuffle_epi8(shuffle2, indices);
            __m128i const symbols3 = _mm_shuffle_epi8(shuffle3, indices);

            __m128i const symbols4 = ZS_blend1(blend0, symbols0, symbols1);
            __m128i const symbols5 = ZS_blend0(blend2, symbols2, symbols3);
            __m128i const symbols  = ZS_blend1(blend1, symbols4, symbols5);

            _mm_storeu_si128((__m128i_u*)(void*)dst, symbols);
            packed += bytesPerLoop;
            dst += 16;
        }
    }
    ZL_ASSERT_LE(dst, dstEnd);
    return ZS_flatpackDecodeEnd(
            dst, dstEnd, alphabet, alphabetSize, packed, packedEnd, nbBits);
}

static ZL_FLATPACK_NOINLINE ZS_FlatPackSize ZS_flatpackDecode128(
        uint8_t* dst,
        uint8_t* dstEnd,
        uint8_t const* alphabet,
        size_t alphabetSize,
        uint8_t const* packed,
        uint8_t const* packedEnd,
        size_t nbBits)
{
    {
        __m128i const shuffle0 = ZS_loadAlphabet16(alphabet, 0, alphabetSize);
        __m128i const shuffle1 = ZS_loadAlphabet16(alphabet, 16, alphabetSize);
        __m128i const shuffle2 = ZS_loadAlphabet16(alphabet, 32, alphabetSize);
        __m128i const shuffle3 = ZS_loadAlphabet16(alphabet, 48, alphabetSize);
        __m128i const shuffle4 = ZS_loadAlphabet16(alphabet, 64, alphabetSize);
        __m128i const shuffle5 = ZS_loadAlphabet16(alphabet, 80, alphabetSize);
        __m128i const shuffle6 = ZS_loadAlphabet16(alphabet, 96, alphabetSize);
        __m128i const shuffle7 = ZS_loadAlphabet16(alphabet, 112, alphabetSize);
        __m128i const cmp0     = _mm_set1_epi8(15);
        __m128i const cmp1     = _mm_set1_epi8(31);
        __m128i const cmp2     = _mm_set1_epi8(47);
        __m128i const cmp3     = _mm_set1_epi8(63);
        __m128i const cmp4     = _mm_set1_epi8(79);
        __m128i const cmp5     = _mm_set1_epi8(95);
        __m128i const cmp6     = _mm_set1_epi8(111);
        uint8_t const* const packedLimit = packedEnd - 16;
        size_t const bytesPerLoop        = 2 * nbBits;
        uint64_t const mask = ((1ull << nbBits) - 1) * 0x0101010101010101ULL;
        while (packed < packedLimit) {
            uint64_t const bits0    = ZL_readLE64(packed);
            uint64_t const bits8    = ZL_readLE64(packed + nbBits);
            uint64_t const indices0 = _pdep_u64(bits0, mask);
            uint64_t const indices8 = _pdep_u64(bits8, mask);
            __m128i const indices =
                    _mm_set_epi64x((int64_t)indices8, (int64_t)indices0);

            __m128i const blend0   = _mm_cmpgt_epi8(indices, cmp0);
            __m128i const blend1   = _mm_cmpgt_epi8(indices, cmp1);
            __m128i const blend2   = _mm_cmpgt_epi8(indices, cmp2);
            __m128i const blend3   = _mm_cmpgt_epi8(indices, cmp3);
            __m128i const blend4   = _mm_cmpgt_epi8(indices, cmp4);
            __m128i const blend5   = _mm_cmpgt_epi8(indices, cmp5);
            __m128i const blend6   = _mm_cmpgt_epi8(indices, cmp6);
            __m128i const symbols0 = _mm_shuffle_epi8(shuffle0, indices);
            __m128i const symbols1 = _mm_shuffle_epi8(shuffle1, indices);
            __m128i const symbols2 = _mm_shuffle_epi8(shuffle2, indices);
            __m128i const symbols3 = _mm_shuffle_epi8(shuffle3, indices);
            __m128i const symbols4 = _mm_shuffle_epi8(shuffle4, indices);
            __m128i const symbols5 = _mm_shuffle_epi8(shuffle5, indices);
            __m128i const symbols6 = _mm_shuffle_epi8(shuffle6, indices);
            __m128i const symbols7 = _mm_shuffle_epi8(shuffle7, indices);

            __m128i const symbols8  = ZS_blend1(blend0, symbols0, symbols1);
            __m128i const symbols9  = ZS_blend0(blend2, symbols2, symbols3);
            __m128i const symbols10 = ZS_blend1(blend4, symbols4, symbols5);
            __m128i const symbols11 = ZS_blend0(blend6, symbols6, symbols7);
            __m128i const symbols12 = ZS_blend1(blend1, symbols8, symbols9);
            __m128i const symbols13 = ZS_blend1(blend5, symbols10, symbols11);
            __m128i const symbols   = ZS_blend1(blend3, symbols12, symbols13);

            _mm_storeu_si128((__m128i_u*)(void*)dst, symbols);
            packed += bytesPerLoop;
            dst += 16;
        }
    }
    ZL_ASSERT_LE(dst, dstEnd);
    return ZS_flatpackDecodeEnd(
            dst, dstEnd, alphabet, alphabetSize, packed, packedEnd, nbBits);
}

static ZL_FLATPACK_NOINLINE ZS_FlatPackSize ZS_flatpackDecodeGeneric(
        uint8_t* dst,
        uint8_t* dstEnd,
        uint8_t const* alphabet,
        size_t alphabetSize,
        uint8_t const* packed,
        uint8_t const* packedEnd,
        size_t nbBits)
{
    ZL_ASSERT_LE(nbBits, 8);
    ZL_ASSERT_LE(alphabetSize, 256);
    {
        uint8_t const* const packedLimit = packedEnd - 8;
        size_t const bytesPerLoop        = nbBits;
        uint64_t const mask = ((1ull << nbBits) - 1) * 0x0101010101010101ULL;
        uint8_t safeAlphabet[256];
        ZL_memcpy(safeAlphabet, alphabet, alphabetSize);
        ZL_memset(safeAlphabet + alphabetSize, 0, 256 - alphabetSize);

        while (packed < packedLimit) {
            uint64_t const bits  = ZL_readLE64(packed);
            uint64_t const bytes = _pdep_u64(bits, mask);

            for (size_t i = 0; i < 8; ++i) {
                dst[i] = safeAlphabet[(bytes >> (8 * i)) & 0xFF];
            }

            packed += bytesPerLoop;
            dst += 8;
        }
    }
    ZL_ASSERT_LE(dst, dstEnd);
    return ZS_flatpackDecodeEnd(
            dst, dstEnd, alphabet, alphabetSize, packed, packedEnd, nbBits);
}

#endif

ZS_FlatPackSize ZS_flatpackDecode(
        uint8_t* dst,
        size_t dstCapacity,
        uint8_t const* alphabet,
        size_t alphabetSize,
        uint8_t const* packed,
        size_t packedSize)
{
    ZS_FlatPackSize size = { alphabetSize };
    if (packedSize == 0 || alphabetSize == 0) {
        return size;
    }

    size_t const nbBits = ZS_FlatPack_nbBits(size);
    size_t const nbElts = ZS_FlatPack_nbElts(alphabetSize, packed, packedSize);
    if (dstCapacity < nbElts) {
        return ZS_FlatPack_kError;
    }

    ZL_ASSERT_LE(nbBits, 8);

#if ZS_FLAT_HAS_NEEDED_EXTS
    if (alphabetSize <= 16) {
        return ZS_flatpackDecode16(
                dst,
                dst + nbElts,
                alphabet,
                alphabetSize,
                packed,
                packed + packedSize,
                nbBits);
    } else if (alphabetSize <= 32) {
        return ZS_flatpackDecode32(
                dst,
                dst + nbElts,
                alphabet,
                alphabetSize,
                packed,
                packed + packedSize,
                nbBits);
    } else if (alphabetSize <= 48) {
        return ZS_flatpackDecode48(
                dst,
                dst + nbElts,
                alphabet,
                alphabetSize,
                packed,
                packed + packedSize,
                nbBits);
    } else if (alphabetSize <= 64) {
        return ZS_flatpackDecode64(
                dst,
                dst + nbElts,
                alphabet,
                alphabetSize,
                packed,
                packed + packedSize,
                nbBits);
    } else if (alphabetSize <= 128) {
        return ZS_flatpackDecode128(
                dst,
                dst + nbElts,
                alphabet,
                alphabetSize,
                packed,
                packed + packedSize,
                nbBits);
    } else {
        return ZS_flatpackDecodeGeneric(
                dst,
                dst + nbElts,
                alphabet,
                alphabetSize,
                packed,
                packed + packedSize,
                nbBits);
    }
#else
    return ZS_flatpackDecodeEnd(
            dst,
            dst + nbElts,
            alphabet,
            alphabetSize,
            packed,
            packed + packedSize,
            nbBits);
#endif
}

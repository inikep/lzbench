// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/parse_int/encode_parse_int_kernel.h"
#include <inttypes.h>
#include <stdint.h>
#include <string.h>
#include "openzl/codecs/parse_int/common_parse_int.h"
#include "openzl/codecs/parse_int/encode_parse_int_gen_lut.h"
#include "openzl/common/assertion.h"
#include "openzl/shared/overflow.h"

#if ZL_HAS_AVX2
#    include <immintrin.h>
#endif

static bool checkNumStringEquality(const char* data, size_t size, int64_t num)
{
    // 21 characters isenough for a 64-bitunsigned integer. One extra character
    // is used for the sign bit.
    char numBuff[ZL_PARSE_INT_MAX_STRING_LENGTH + 1];
    snprintf(numBuff, ZL_PARSE_INT_MAX_STRING_LENGTH + 1, "%" PRId64, num);
    return memcmp(data, numBuff, size) == 0;
}

#if ZL_HAS_AVX2
/// Multiply @p result by 10000 and add @p add.
/// @returns true if either operation overflows.
static bool overflowAccumulate(uint64_t* result, uint64_t add)
{
    // Max value that won't overflow when multiplying by 10000.
    uint64_t const kMaxResult = ((uint64_t)-1) / 10000;
#    ifndef NDEBUG
    uint64_t tmp;
    ZL_ASSERT(!ZL_overflowMulU64(kMaxResult, 10000, &tmp));
    ZL_ASSERT(ZL_overflowMulU64(kMaxResult + 1, 10000, &tmp));
#    endif
    if (*result > kMaxResult) {
        return true;
    }
    *result *= 10000;

    // Add & check for wrapping, we know the add is small so this is
    // correct.
    ZL_ASSERT_LT(add, 10000);
    uint64_t const next = *result + add;
    if (next < *result) {
        return true;
    }
    *result = next;

    return false;
}

static bool
ZL_parseInt64Unsafe_AVX(int64_t* value, char const* ptr, char const* end)
{
    // Determine if it is negative, remove '-' & validate it is non-empty
    if (ptr == end) {
        return false;
    }
    bool const negative = ptr[0] == '-';
    if (negative) {
        ++ptr;
        if (ptr == end) {
            return false;
        }
    }

    // Validate the length isn't >20 = guaranteed overflow
    size_t const len = (size_t)(end - ptr);
    if (len > 20) {
        return false;
    }

    // Validate it doesn't have a leading zero
    if (ptr[0] == '0') {
        if (len != 1 || negative) {
            return false;
        }
        *value = 0;
        return true;
    }

    // 1. Validate all the characters are between '0' & '9'
    // 2. Subtract '0' from each character
    // 3. Zero the bytes before `ptr`
    __m256i v       = _mm256_loadu_si256((__m256i_u const*)(end - 32));
    __m256i m       = _mm256_load_si256((__m256i const*)kNonZeroMask[len]);
    __m256i invalid = _mm256_cmpgt_epi8(v, _mm256_set1_epi8('9'));
    invalid         = _mm256_or_si256(
            invalid, _mm256_cmpgt_epi8(_mm256_set1_epi8('0'), v));
    invalid = _mm256_and_si256(invalid, m);
    if (_mm256_movemask_epi8(invalid) != 0) {
        return false;
    }
    v = _mm256_sub_epi8(v, _mm256_set1_epi8('0'));
    v = _mm256_and_si256(v, m);

    // Store the vector:
    // [0, 32 - len) = 0
    // [32 - len, 32) = [ptr, end) - '0'
    uint8_t data[32];
    _mm256_storeu_si256((__m256i_u*)data, v);

    // Always read the last 20 bytes, values
    // before the beginning of the int are 0.
    uint8_t* src = data + 32 - 20;

#    ifndef NDEBUG
    for (size_t i = 0; i < (32 - len); ++i) {
        ZL_ASSERT_EQ(data[i], 0);
    }
#    endif
    ZL_ASSERT_LE(len, 20);

    // Add to the result in 5 loops of 4
    // We only need to check the last iteration for overflow, because integers
    // of 19 digits or less cannot overflow a uint64_t. We check for signed
    // overflow below.
    uint64_t uresult = 0;
    for (size_t idx = 0; idx < 20;) {
        uint64_t sum = 0;
        for (size_t u = 0; u < 4; ++u, ++idx) {
            ZL_ASSERT_LT(src[idx], 10);
            sum += kLookup[u][src[idx]];
        }
        if (idx == 20) {
            if (overflowAccumulate(&uresult, sum)) {
                return false;
            }
        } else {
            uresult = uresult * 10000 + sum;
        }
    }

    // Convert the uint64_t into an int64_t and check for overflows.
    if (negative) {
        *value = (int64_t)((uint64_t)0 - uresult);
        if (*value > 0) {
            return false;
        }
        return true;
    } else {
        *value = (int64_t)(uresult);
        if (*value < 0) {
            return false;
        }
        return true;
    }
}
#endif // ZL_HAS_AVX2

bool ZL_parseInt(
        int64_t* nums,
        const char* data,
        uint32_t const* sizes,
        size_t const nbElts)
{
    // Use safe parse until we've read 32 bytes from the input stream
    size_t idx;
    {
        size_t offset;
        for (idx = 0, offset = 0; idx < nbElts && offset < 32; ++idx) {
            int64_t value = 0;
            bool success =
                    ZL_parseInt64_fallback(&value, data, data + sizes[idx]);
            if (!success) {
                return false;
            }
            ZL_ASSERT(checkNumStringEquality(data, sizes[idx], value));
            nums[idx] = value;
            data += sizes[idx];
            offset += sizes[idx];
        }
    }

    // Decode the remainder using the fast parsing function
    for (; idx < nbElts; ++idx) {
        int64_t value = 0;
        bool success  = ZL_parseInt64Unsafe(&value, data, data + sizes[idx]);
        if (!success) {
            return false;
        }
        ZL_ASSERT(checkNumStringEquality(data, sizes[idx], value));
        nums[idx] = value;
        data += sizes[idx];
    }
    return true;
}

/**
 * Strictly parses an int64 contained in [ptr, end), and is allowed to read
 * up to 32 bytes before @p end. Fails if:
 * - The string is not an integer
 * - The integer begins with +
 * - The integer has leading zeros
 * - The integer overlows an int64_t
 */
bool ZL_parseInt64Unsafe(int64_t* value, const char* ptr, const char* end)
{
#if ZL_HAS_AVX2
    return ZL_parseInt64Unsafe_AVX(value, ptr, end);
#else
    return ZL_parseInt64_fallback(value, ptr, end);
#endif
}

/**
 * Strictly parses an int64 contained in [ptr, end) and will not read any bytes
 * before @p end. Has all other requirement in @ref parseInt64Unsafe.
 */
bool ZL_parseInt64_fallback(int64_t* value, const char* ptr, const char* end)
{
    // Determine if it is negative, remove '-' & validate it is non-empty
    if (ptr == end) {
        return false;
    }
    bool const negative = ptr[0] == '-';
    if (negative) {
        ++ptr;
        if (ptr == end) {
            return false;
        }
    }

    // Validate the length isn't >20 = guaranteed overflow
    size_t const len = (size_t)(end - ptr);
    if (len > 20) {
        return false;
    }

    // Validate it doesn't have a leading zero
    if (ptr[0] == '0') {
        if (len != 1 || negative) {
            return false;
        }
        *value = 0;
        return true;
    }
    uint64_t uresult = 0;
    for (size_t i = 0; i < len; ++i) {
        // Validate the character is a digit
        if (ptr[i] < '0' || ptr[i] > '9') {
            return false;
        }
        bool overflow = ZL_overflowMulU64(uresult, 10, &uresult);
        overflow |=
                ZL_overflowAddU64(uresult, (uint64_t)(ptr[i] - '0'), &uresult);
        if (overflow) {
            return false;
        }
    }

    if (negative) {
        *value = (int64_t)((uint64_t)0 - uresult);
        if (*value > 0) {
            return false;
        }
        return true;
    } else {
        *value = (int64_t)(uresult);
        if (*value < 0) {
            return false;
        }
        return true;
    }
}

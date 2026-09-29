// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/parse_int/decode_parse_int_kernel.h"
#include <stdint.h>
#include "openzl/codecs/parse_int/common_parse_int.h"
#include "openzl/codecs/parse_int/decode_parse_int_gen_lut.h"
#include "openzl/common/assertion.h"
#include "openzl/shared/bits.h"

// Calculates the number of digits in the base 10 representation of `x`.
// Assumes `x` is non-negative.
static uint8_t u64BaseTenDigits(uint64_t x)
{
    // We calculate the rounded-up base-10 logarithm of `x` based on its base-2
    // logarithm (`64-clz`).
    // However, for some values of the base-2 logarithm we might have two
    // possible values we distinguish between the two by comparing to the
    // 10-power of the approximation we get and checking if the value is greater
    // than our approximation.
    const int clz        = ZL_clz64((uint64_t)x | 1);
    const uint8_t approx = (uint8_t)(((64 - clz) * 1233) >> 12);
    return (uint8_t)(approx + (x >= tenPower[clz]));
}

static size_t numberStringLength(int64_t value)
{
    const uint64_t absValue =
            value < 0 ? ((~(uint64_t)(int64_t)value) + 1) : (uint64_t)value;
    return u64BaseTenDigits(absValue) + (value < 0 ? 1 : 0);
}

// Copies a 4 byte string representation of the number `x` into `dst`.
// `x` must be a positive number and at most 9999.
ZL_INLINE void copy4FromTable(char* dst, uint64_t x)
{
    ZL_ASSERT_LT(x, 10000);
    memcpy(dst, &characterTable[4 * x], 4);
}

// Converts the 64-bit unsigned number `x` to a string of digits in decimal.
// The number is written *backwards* into `out`.
// `out` must have at least 20 digits available backwards (i.e `out-20...out-1`
// should be writable and may be overwritten). `len` is the maximum number of
// digits in the decimal representation for the number. It's used for
// optimization and the function may write more then `len` characters into
// `out`.
// The functionality is based on
// `https://lemire.me/blog/2021/11/18/converting-integers-to-fix-digit-representations-quickly/`
// but we have made multiple modification:
// 1. Branch based on `len` so we skip unneeded calculations and memory
//   accesses. This works as long as most of our numbers are consistent in their
//   lengths, which is to be expected in most use-cases we handle.s
// 2. We write backwards.
// 3. We support 20 digits insteaf o 16.
static void u64numberToStringBackwardsUnsafe(uint64_t x, char* out, uint8_t len)
{
    if (len <= 4) {
        ZL_ASSERT_LT(x, 10000);
        copy4FromTable(out - 4, x);
        return;
    }

    uint64_t bottom       = x % 100000000;
    uint64_t bottomtop    = bottom / 10000;
    uint64_t bottombottom = bottom % 10000;
    copy4FromTable(out - 8, bottomtop);
    copy4FromTable(out - 4, bottombottom);

    if (len >= 8) {
        uint64_t top         = x / 100000000;
        uint64_t toptop      = top / 10000;
        uint64_t toptoptop   = toptop / 10000;
        uint64_t toptopottom = toptop % 10000;
        uint64_t topbottom   = top % 10000;
        copy4FromTable(out - 20, toptoptop);
        copy4FromTable(out - 16, toptopottom);
        copy4FromTable(out - 12, topbottom);
    }
}

// Converts the integer `value` to a string of digits in decimal.
// If the number type is signed, the character '-' is prepended.
// The number is written *backwards* into `out`.
// `out` must have at least `maxStrLen` bytes available backwards (i.e
// `out-maxStrLen...out-1` should be writable and may be overwritten). `len` is
// the maximum number of digits in the decimal representation for the number.
// It's used for optimization and the function may write more then `len`
// characters into `out`.
ZL_INLINE void
numberToStringBackwardsUnsafe(int64_t value, char* out, uint8_t len)
{
    const uint8_t neg = value < 0 ? 1 : 0;
    const uint64_t absValue =
            neg ? ((~(uint64_t)(int64_t)value) + 1) : (uint64_t)value;
    u64numberToStringBackwardsUnsafe(absValue, out, len - neg);
    *(out - len + neg - 1) = '-';
}

size_t ZL_DecodeParseInt_fillFieldSizes(
        uint32_t* fieldSizes,
        size_t nbElts,
        int64_t const* nums)
{
    size_t outSize = 0;
    for (size_t i = 0; i < nbElts; ++i) {
        size_t fieldSize = 0;
        fieldSize        = numberStringLength(nums[i]);
        fieldSizes[i]    = (uint32_t)fieldSize;
        outSize += fieldSize;
    }
    return outSize;
}

// Fills the content of the buffer `dst` with the decoded parsed integers
// from `nums` and the exceptions.
// `dst` must be large enough to contain the decoded integers and exceptions
// (must be at least the length from `parseDecodeIntFillFieldSizes`).
// `dstSize` must be the exact size returned by `parseDecodeIntFillFieldSizes`.
// `fieldSizes` must also be correct and match the actual length of each of the
// fields.
void ZL_DecodeParseInt_fillContent(
        char* dst,
        size_t const dstSize,
        const size_t nbElts,
        int64_t const* nums,
        uint32_t const* const fieldSizes)
{
    /* We start by writing backwards. Since @ref numberToStringBackwards uses
     * up to ZL_PARSE_INT_MAX_STRING_LENGTH bytes backwards, we can directly
     * write to @param dst we have at least maxStrLen bytes of backwards space
     * available. */
    size_t backwardsIndex  = nbElts;
    const int64_t* numsEnd = nums + nbElts;
    const char* dstStart   = dst;
    {
        const int64_t* currNums = numsEnd;
        char* currDst           = dst + dstSize;
        while (currDst >= dst + ZL_PARSE_INT_MAX_STRING_LENGTH) {
            backwardsIndex--;
            const uint32_t fieldSize = fieldSizes[backwardsIndex];
            currDst -= fieldSize;

            ZL_ASSERT_GT((const char*)currDst + fieldSize, dstStart);
            ZL_ASSERT_GE(
                    (const char*)currDst + fieldSize
                            - ZL_PARSE_INT_MAX_STRING_LENGTH,
                    dstStart);
            ZL_ASSERT_LE(fieldSize, ZL_PARSE_INT_MAX_STRING_LENGTH);
            /* The while loop conditional makes this operation safe to use*/
            numberToStringBackwardsUnsafe(
                    *(--currNums),
                    (char*)currDst + fieldSize,
                    (uint8_t)fieldSize);
        }
        ZL_ASSERT_GE((const char*)currDst, dstStart);
    }

    // Write the rest of the fields forwards, requires using scratch
    // space with enough "backwards" space to allow overwirte.
    {
        char* currDst           = dst;
        const int64_t* currNums = nums;
        char scratchSpace[ZL_PARSE_INT_MAX_STRING_LENGTH];
        char* scratchEnd = scratchSpace + sizeof(scratchSpace);
        for (size_t i = 0; i < backwardsIndex; ++i) {
            uint32_t fieldSize = fieldSizes[i];
            ZL_ASSERT_LE(fieldSize, ZL_PARSE_INT_MAX_STRING_LENGTH);
            /* This operation is safe because we use scratch space with enough
             * allocated space*/
            numberToStringBackwardsUnsafe(
                    *(currNums++), scratchEnd, (uint8_t)fieldSize);
            memcpy(currDst, scratchEnd - fieldSize, fieldSize);
            currDst += fieldSize;
        }
    }
}

// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <assert.h>
#include <limits.h> // UCHAR_MAX, USHRT_MAX, UINT_MAX, ULLONG_MAX
#include <stddef.h> // size_t
#include <stdint.h> // uintX_t

#include "openzl/codecs/divide_by/decode_divide_by_kernel.h"

void ZS_divideByDecode(
        void* numericArrayOutput,
        const void* numericArrayInput,
        const size_t inputLength,
        const uint64_t multiplier,
        const size_t nbBytes)
{
    assert(multiplier != 0);
    switch (nbBytes) {
        case 1:
            ZS_divideByDecode8(
                    (uint8_t*)numericArrayOutput,
                    (const uint8_t*)numericArrayInput,
                    inputLength,
                    (uint8_t)multiplier);
            break;
        case 2:
            ZS_divideByDecode16(
                    (uint16_t*)numericArrayOutput,
                    (const uint16_t*)numericArrayInput,
                    inputLength,
                    (uint16_t)multiplier);
            break;
        case 4:
            ZS_divideByDecode32(
                    (uint32_t*)numericArrayOutput,
                    (const uint32_t*)numericArrayInput,
                    inputLength,
                    (uint32_t)multiplier);
            break;
        case 8:
            ZS_divideByDecode64(
                    (uint64_t*)numericArrayOutput,
                    (const uint64_t*)numericArrayInput,
                    inputLength,
                    (uint64_t)multiplier);
            break;
        default:
            assert(0); /* Ensure nbBytes is 1, 2, 4 or 8 */
    }
}

void ZS_divideByDecode8(
        uint8_t* numericArrayOutput,
        const uint8_t* numericArrayInput,
        const size_t inputLength,
        const uint8_t multiplier)
{
    assert(multiplier != 0);
    for (size_t i = 0; i < inputLength; i++) {
        assert(numericArrayInput[i] <= UCHAR_MAX / multiplier);
        numericArrayOutput[i] = numericArrayInput[i] * multiplier;
    }
}

void ZS_divideByDecode16(
        uint16_t* numericArrayOutput,
        const uint16_t* numericArrayInput,
        const size_t inputLength,
        const uint16_t multiplier)
{
    assert(multiplier != 0);
    for (size_t i = 0; i < inputLength; i++) {
        assert(numericArrayInput[i] <= USHRT_MAX / multiplier);
        numericArrayOutput[i] = numericArrayInput[i] * multiplier;
    }
}

void ZS_divideByDecode32(
        uint32_t* numericArrayOutput,
        const uint32_t* numericArrayInput,
        const size_t inputLength,
        const uint32_t multiplier)
{
    assert(multiplier != 0);
    for (size_t i = 0; i < inputLength; i++) {
        assert(numericArrayInput[i] <= UINT_MAX / multiplier);
        numericArrayOutput[i] = numericArrayInput[i] * multiplier;
    }
}

void ZS_divideByDecode64(
        uint64_t* numericArrayOutput,
        const uint64_t* numericArrayInput,
        const size_t inputLength,
        const uint64_t multiplier)
{
    assert(multiplier != 0);
    for (size_t i = 0; i < inputLength; i++) {
        assert(numericArrayInput[i] <= ULLONG_MAX / multiplier);
        numericArrayOutput[i] = numericArrayInput[i] * multiplier;
    }
}

// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <assert.h>
#include <stddef.h> // size_t
#include <stdint.h> // uintX_t

#include "openzl/codecs/divide_by/common_gcd.h"
#include "openzl/codecs/divide_by/encode_divide_by_kernel.h"

void ZS_divideByEncode(
        void* numericArrayOutput,
        const void* numericArrayInput,
        const size_t inputLength,
        const uint64_t divisor,
        const size_t nbBytes)
{
    assert(divisor != 0);
    switch (nbBytes) {
        case 1:
            ZS_divideByEncode8(
                    (uint8_t*)numericArrayOutput,
                    (const uint8_t*)numericArrayInput,
                    inputLength,
                    (uint8_t)divisor);
            break;
        case 2:
            ZS_divideByEncode16(
                    (uint16_t*)numericArrayOutput,
                    (const uint16_t*)numericArrayInput,
                    inputLength,
                    (uint16_t)divisor);
            break;
        case 4:
            ZS_divideByEncode32(
                    (uint32_t*)numericArrayOutput,
                    (const uint32_t*)numericArrayInput,
                    inputLength,
                    (uint32_t)divisor);
            break;
        case 8:
            ZS_divideByEncode64(
                    (uint64_t*)numericArrayOutput,
                    (const uint64_t*)numericArrayInput,
                    inputLength,
                    (uint64_t)divisor);
            break;
        default:
            assert(0); /* Ensure nbBytes is 1, 2, 4 or 8 */
    }
}

void ZS_divideByEncode8(
        uint8_t* numericArrayOutput,
        const uint8_t* numericArrayInput,
        const size_t inputLength,
        const uint8_t divisor)
{
    assert(divisor != 0);
    int shift;
    uint8_t inverse = ZL_getMultiplicativeInverse8(&shift, divisor);
    for (size_t i = 0; i < inputLength; i++) {
        assert(numericArrayInput[i] % divisor == 0);
        numericArrayOutput[i] =
                (uint8_t)(numericArrayInput[i] * inverse >> shift);
    }
}

void ZS_divideByEncode16(
        uint16_t* numericArrayOutput,
        const uint16_t* numericArrayInput,
        const size_t inputLength,
        const uint16_t divisor)
{
    int shift;
    // We use the 32-bit version here because by default, 16-bit multiplication
    // is optimized to use 32-bit data types. Hence there is no reason to write
    // a 16-bit version for inverse.
    uint32_t inverse = ZL_getMultiplicativeInverse32(&shift, divisor);
    for (size_t i = 0; i < inputLength; i++) {
        assert(numericArrayInput[i] % divisor == 0);
        numericArrayOutput[i] =
                (uint16_t)(numericArrayInput[i] * inverse >> shift);
    }
}

void ZS_divideByEncode32(
        uint32_t* numericArrayOutput,
        const uint32_t* numericArrayInput,
        const size_t inputLength,
        const uint32_t divisor)
{
    int shift;
    uint32_t inverse = ZL_getMultiplicativeInverse32(&shift, divisor);
    for (size_t i = 0; i < inputLength; i++) {
        assert(numericArrayInput[i] % divisor == 0);
        numericArrayOutput[i] = numericArrayInput[i] * inverse >> shift;
    }
}

void ZS_divideByEncode64(
        uint64_t* numericArrayOutput,
        const uint64_t* numericArrayInput,
        const size_t inputLength,
        const uint64_t divisor)
{
    int shift;
    uint64_t inverse = ZL_getMultiplicativeInverse64(&shift, divisor);
    for (size_t i = 0; i < inputLength; i++) {
        assert(numericArrayInput[i] % divisor == 0);
        numericArrayOutput[i] = numericArrayInput[i] * inverse >> shift;
    }
}

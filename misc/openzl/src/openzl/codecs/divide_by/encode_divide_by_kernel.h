// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_ENCODE_DIVIDE_BY_KERNEL_H
#define ZSTRONG_TRANSFORMS_ENCODE_DIVIDE_BY_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // uintX_t

#include "openzl/zl_portability.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Divide by encodes elements in @p numericArrayInput, storing the quotient with
 * @p divisor in @p numericArrayOutput.
 *
 * @param numericArrayOutput Output for the values of  @p numericArrayInput
 * divided by @p divisor.
 * @param numericArrayInput The input array of @p inputLength elements.
 * @param inputLength The length of  @p numericArrayInput.
 * @param divisor The divisor to divide the input by. Should not be
 * null and should not point to 0.
 *
 * @param nbBytes The byte size of the data in @p numericArrayInput.
 *
 * NOTE: Assumes the elements of @p numericArrayInput are divisible by @p
 * divisor.
 */

void ZS_divideByEncode(
        void* numericArrayOutput,
        const void* numericArrayInput,
        const size_t inputLength,
        const uint64_t divisor,
        const size_t nbBytes);

void ZS_divideByEncode8(
        uint8_t* numericArrayOutput,
        const uint8_t* numericArrayInput,
        const size_t inputLength,
        const uint8_t divisor);

void ZS_divideByEncode16(
        uint16_t* numericArrayOutput,
        const uint16_t* numericArrayInput,
        const size_t inputLength,
        const uint16_t divisor);

void ZS_divideByEncode32(
        uint32_t* numericArrayOutput,
        const uint32_t* numericArrayInput,
        const size_t inputLength,
        const uint32_t divisor);

void ZS_divideByEncode64(
        uint64_t* numericArrayOutput,
        const uint64_t* numericArrayInput,
        const size_t inputLength,
        const uint64_t divisor);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif

// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_DECODE_DIVIDE_BY_KERNEL_H
#define ZSTRONG_TRANSFORMS_DECODE_DIVIDE_BY_KERNEL_H

#include <stddef.h> // size_t
#include <stdint.h> // uintX_t

#include "openzl/zl_portability.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Divide by decodes elements in @p numericArrayInput, storing its value
 * multiplied by
 * @p multiplier in @p numericArrayOutput to reverse the encoding process.
 *
 * @param numericArrayOutput Ouput for the values of  @p numericArrayInput
 * decoded by multiplying by @p multiplier.
 * @param numericArrayInput The input array of @p inputLength elements.
 * @param inputLength The length of  @p numericArrayInput.
 * @param multiplier The multiplier to multiply the input by.
 * Should not be null.
 * @param nbBytes The byte size of the data in @p numericArrayInput.
 */

void ZS_divideByDecode(
        void* numericArrayOutput,
        const void* numericArrayInput,
        const size_t inputLength,
        const uint64_t multiplier,
        const size_t nbBytes);

void ZS_divideByDecode8(
        uint8_t* numericArrayOutput,
        const uint8_t* numericArrayInput,
        const size_t inputLength,
        const uint8_t multiplier);

void ZS_divideByDecode16(
        uint16_t* numericArrayOutput,
        const uint16_t* numericArrayInput,
        const size_t inputLength,
        const uint16_t multiplier);

void ZS_divideByDecode32(
        uint32_t* numericArrayOutput,
        const uint32_t* numericArrayInput,
        const size_t inputLength,
        const uint32_t multiplier);

void ZS_divideByDecode64(
        uint64_t* numericArrayOutput,
        const uint64_t* numericArrayInput,
        const size_t inputLength,
        const uint64_t multiplier);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif

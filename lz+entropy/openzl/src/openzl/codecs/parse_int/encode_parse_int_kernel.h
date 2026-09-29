// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_ENCODE_PARSE_INT_KERNEL_H
#define ZSTRONG_ENCODE_PARSE_INT_KERNEL_H

#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS
/** @brief Parses a single string containing an integer and stores the result in
 * @p value.
 *
 * @returns true if the string is an integer, false otherwise.
 *
 * @param ptr the pointer the characters where the integer string is stored
 * @param end the pointer to the past the end element of the integer string
 */
bool ZL_parseInt64_fallback(int64_t* value, const char* ptr, const char* end);

/** @brief Has identical functionality to @ref ZL_parseInt64_fallback but uses
 * AVX2 instructions to do faster parsing.
 */
bool ZL_parseInt64Unsafe(int64_t* value, const char* ptr, const char* end);

/** @brief Parses @p data where @p sizes are the number of characters in each
 * integer string and stores their integer representation in @p nums.
 *
 * @returns true if all parsed strings are integers, false otherwise.
 *
 * @param nums A preallocated buffer to write the resulting integer read from @p
 * data
 * @param data The buffer to read characters to parse into integers
 * @param sizes The sizes of each string representing a number in @p data
 * @param nbElts The number of integer strings the parse
 */
bool ZL_parseInt(
        int64_t* nums,
        const char* data,
        uint32_t const* sizes,
        size_t const nbElts);
ZL_END_C_DECLS

#endif

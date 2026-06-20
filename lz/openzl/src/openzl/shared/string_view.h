// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_SHARED_STRING_VIEW_H
#define ZSTRONG_SHARED_STRING_VIEW_H

#include <string.h>

#include "openzl/common/assertion.h" // ZL_ASSERT
#include "openzl/shared/xxhash.h"
#include "openzl/zl_errors.h"

ZL_BEGIN_C_DECLS
typedef struct StringView {
    const char* data;
    size_t size;
} StringView;

/**
 * @brief Hashes function for StringView type
 *
 * @returns The hash of the StringView
 */
ZL_FORCE_INLINE size_t StringView_hash(StringView const* key)
{
    return XXH3_64bits(key->data, key->size);
}

/**
 * @brief Checks equality of two string views,
 *
 * @returns True if buffers contain the same characters and are the same size.
 * False otherwise
 */
ZL_FORCE_INLINE bool StringView_eq(StringView const* lhs, StringView const* rhs)
{
    if (lhs->size != rhs->size) {
        return false;
    }

    return memcmp(lhs->data, rhs->data, lhs->size) == 0;
}

/**
 * @brief Checks equality of a StringView and a C string,
 *
 * @returns true iff the buffers contain the same characters and are the same
 *          size, false otherwise.
 */
ZL_FORCE_INLINE bool StringView_eqCStr(
        StringView const* lhs,
        const char* const rhs)
{
    size_t const rhs_size = strlen(rhs);
    if (lhs->size != rhs_size) {
        return false;
    }
    return memcmp(lhs->data, rhs, lhs->size) == 0;
}

/**
 * @brief Gets the substring in the interval [ @p pos, @p pos + @p rLen )
 *
 * @param pos The position of the start of the substring
 * @param rLen The length of the substring
 *
 * @returns A StringView of the resulting substring requested
 */
ZL_FORCE_INLINE StringView
StringView_substr(StringView const* sv, size_t pos, size_t rLen)
{
    ZL_ASSERT((rLen <= (sv->size - pos)) && (pos <= sv->size));
    return (StringView){ sv->data + pos, rLen };
}

/**
 * @brief Initializes StringView given a sized buffer
 *
 * @returns The StringView based on initialization params
 */
ZL_FORCE_INLINE StringView StringView_init(const char* data, size_t size)
{
    return (StringView){ data, size };
}

/**
 * @brief Initializes StringView given a cstring
 *
 * @returns The StringView based on initialization params
 */
ZL_FORCE_INLINE StringView StringView_initFromCStr(const char* cstr)
{
    size_t size = strlen(cstr);
    return (StringView){ cstr, size };
}

/**
 * @brief Advances the data buffer pointer and adjusts size to match.
 */
ZL_FORCE_INLINE void StringView_advance(StringView* sv, size_t n)
{
    ZL_ASSERT_LE(n, sv->size);
    sv->data += n;
    sv->size -= n;
}

ZL_RESULT_DECLARE_TYPE(StringView);

ZL_END_C_DECLS
#endif

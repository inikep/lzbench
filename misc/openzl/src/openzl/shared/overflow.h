// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZSTRONG_COMMON_OVERFLOW_H
#define ZSTRONG_COMMON_OVERFLOW_H

#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/**
 * Multiply @p x and @p y, and store the result in @p r.
 * @note @p r always contains the multiplication result,
 * even in the case of overflow.
 * @returns true iff the multiplication overflowed @p r.
 */
ZL_NODISCARD ZL_INLINE bool
ZL_overflowMulU32(uint32_t x, uint32_t y, uint32_t* result);

/**
 * Multiply @p x and @p y, and store the result in @p r.
 * @note @p r always contains the multiplication result,
 * even in the case of overflow.
 * @returns true iff the multiplication overflowed @p r.
 */
ZL_NODISCARD ZL_INLINE bool
ZL_overflowMulU64(uint64_t x, uint64_t y, uint64_t* result);

/**
 * Multiply @p x and @p y, and store the result in @p r.
 * @note @p r always contains the multiplication result,
 * even in the case of overflow.
 * @returns true iff the multiplication overflowed @p r.
 */
ZL_NODISCARD ZL_INLINE bool
ZL_overflowMulST(size_t x, size_t y, size_t* result);

/**
 * Add @p x and @p y, and store the result in @p r.
 * @note @p r always contains the addition result,
 * even in the case of overflow.
 * @returns true iff the addition overflowed @p r.
 */
ZL_NODISCARD ZL_INLINE bool
ZL_overflowAddU32(uint32_t x, uint32_t y, uint32_t* result);

/**
 * Add @p x and @p y, and store the result in @p r.
 * @note @p r always contains the addition result,
 * even in the case of overflow.
 * @returns true iff the addition overflowed @p r.
 */
ZL_NODISCARD ZL_INLINE bool
ZL_overflowAddU64(uint64_t x, uint64_t y, uint64_t* result);

/**
 * Add @p x and @p y, and store the result in @p r.
 * @note @p r always contains the addition result,
 * even in the case of overflow.
 * @returns true iff the addition overflowed @p r.
 */
ZL_NODISCARD ZL_INLINE bool
ZL_overflowAddST(size_t x, size_t y, size_t* result);

ZL_NODISCARD ZL_INLINE bool
ZL_overflowMulU32_fallback(uint32_t x, uint32_t y, uint32_t* result)
{
    *result = x * y;
    if (y > 0 && x > UINT32_MAX / y) {
        return true;
    } else {
        return false;
    }
}

ZL_NODISCARD ZL_INLINE bool
ZL_overflowMulU64_fallback(uint64_t x, uint64_t y, uint64_t* result)
{
    *result = x * y;
    if (y > 0 && x > UINT64_MAX / y) {
        return true;
    } else {
        return false;
    }
}

ZL_NODISCARD ZL_INLINE bool
ZL_overflowMulST_fallback(size_t x, size_t y, size_t* result)
{
    if (sizeof(size_t) == 8) {
        uint64_t r;
        bool const overflow =
                ZL_overflowMulU64_fallback((uint64_t)x, (uint64_t)y, &r);
        *result = (size_t)r;
        return overflow;
    } else {
        uint32_t r;
        bool const overflow =
                ZL_overflowMulU32_fallback((uint32_t)x, (uint32_t)y, &r);
        *result = (size_t)r;
        return overflow;
    }
}

ZL_NODISCARD ZL_INLINE bool
ZL_overflowMulU32(uint32_t x, uint32_t y, uint32_t* result)
{
#ifdef __GNUC__
    return __builtin_mul_overflow(x, y, result);
#else
    return ZL_overflowMulU32_fallback(x, y, result);
#endif
}

ZL_NODISCARD ZL_INLINE bool
ZL_overflowMulU64(uint64_t x, uint64_t y, uint64_t* result)
{
#ifdef __GNUC__
    return __builtin_mul_overflow(x, y, result);
#else
    return ZL_overflowMulU64_fallback(x, y, result);
#endif
}

ZL_NODISCARD ZL_INLINE bool ZL_overflowMulST(size_t x, size_t y, size_t* result)
{
#ifdef __GNUC__
    return __builtin_mul_overflow(x, y, result);
#else
    return ZL_overflowMulST_fallback(x, y, result);
#endif
}

ZL_NODISCARD ZL_INLINE bool
ZL_overflowAddU32_fallback(uint32_t x, uint32_t y, uint32_t* result)
{
    *result = x + y;
    if (x > UINT32_MAX - y) {
        return true;
    } else {
        return false;
    }
}

ZL_NODISCARD ZL_INLINE bool
ZL_overflowAddU64_fallback(uint64_t x, uint64_t y, uint64_t* result)
{
    *result = x + y;
    if (x > UINT64_MAX - y) {
        return true;
    } else {
        return false;
    }
}

ZL_NODISCARD ZL_INLINE bool
ZL_overflowAddST_fallback(size_t x, size_t y, size_t* result)
{
    if (sizeof(size_t) == 8) {
        uint64_t r;
        bool const overflow =
                ZL_overflowAddU64_fallback((uint64_t)x, (uint64_t)y, &r);
        *result = (size_t)r;
        return overflow;
    } else {
        uint32_t r;
        bool const overflow =
                ZL_overflowAddU32_fallback((uint32_t)x, (uint32_t)y, &r);
        *result = (size_t)r;
        return overflow;
    }
}

ZL_NODISCARD ZL_INLINE bool
ZL_overflowAddU32(uint32_t x, uint32_t y, uint32_t* result)
{
#ifdef __GNUC__
    return __builtin_add_overflow(x, y, result);
#else
    return ZL_overflowAddU32_fallback(x, y, result);
#endif
}

ZL_NODISCARD ZL_INLINE bool
ZL_overflowAddU64(uint64_t x, uint64_t y, uint64_t* result)
{
#ifdef __GNUC__
    return __builtin_add_overflow(x, y, result);
#else
    return ZL_overflowAddU64_fallback(x, y, result);
#endif
}

ZL_NODISCARD ZL_INLINE bool ZL_overflowAddST(size_t x, size_t y, size_t* result)
{
#ifdef __GNUC__
    return __builtin_add_overflow(x, y, result);
#else
    return ZL_overflowAddST_fallback(x, y, result);
#endif
}

ZL_END_C_DECLS

#endif

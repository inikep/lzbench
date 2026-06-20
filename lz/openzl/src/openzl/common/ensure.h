// Copyright (c) Meta Platforms, Inc. and affiliates.

/*
 *  ensure.h - Compile-time validation for C
 */
#ifndef ZSTRONG_ENSURE_H
#define ZSTRONG_ENSURE_H

#include "openzl/common/assertion.h"

#if defined(ZS_ENABLE_ENSURE) && defined(__GNUC__) && !defined(__clang__)
/* This compile-time test only works on gcc (not clang)
 *
 * It is relying on Value Range Analysis combined with Dead Code Elimination.
 * VRA requires at least `-O1` to be enabled.
 * Consequently, this check doesn't work with `-O0`.
 *
 * Some conditions may only be fulfilled thanks to assert() or assume()
 * in which case, these macros should be enabled too.
 *
 * Due to above conditions, this compile-time analysis is only enabled when
 * build macro ZS_ENABLE_ENSURE is set,
 * otherwise it decays into a standard runtime assertion test.
 */

/* ZL_ENSURE(c) :
 * triggers a warning at compile time if condition is not _always_ true
 */
#    define ZL_ENSURE(condition) \
        (void)((condition) ? (void)0 : _condition_not_ensured())

__attribute__((noinline)) __attribute__((unused))
__attribute__((warning("required condition is not ensured"))) static void
_condition_not_ensured(void)
{
    // must generate a side effect to not be optimized away
    ZL_ABORT();
}

#else /* other compilers, or ZS_ENABLE_ENSURE not set */

/* compile-time verification only works on gcc (not clang),
 * switch to runtime validation (assert) for other compilers */

#    define ZL_ENSURE(c) ZL_ASSERT(c)

#endif // correct compiler

#endif //  ZSTRONG_ENSURE_H  (header guard)

// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_COMPRESS_LOCALPARAMS_H
#define ZSTRONG_COMPRESS_LOCALPARAMS_H

#include "openzl/common/allocation.h" // Arena
#include "openzl/shared/portability.h"
#include "openzl/zl_compressor.h" // ZL_LocalParams

ZL_BEGIN_C_DECLS

/* Transfer local parameters @lp into @arena,
 * update @lparam content (pointers to arrays) to their dst locations */
ZL_Report LP_transferLocalParams(Arena* arena, ZL_LocalParams* lparam);

/* get all IntParams */
ZL_LocalIntParams LP_getLocalIntParams(const ZL_LocalParams* lp);

/* get one specific intParam */
ZL_IntParam LP_getLocalIntParam(const ZL_LocalParams* lp, int intParamId);

ZL_RefParam LP_getLocalRefParam(const ZL_LocalParams* lp, int refParamId);

/**
 * These functions return whether the given param sets are logically equal.
 *
 * This is somewhat more complicated than a strict equality check, since the
 * params can be written in any order and the representation doesn't prevent
 * repetitions of keys. :/
 *
 * E.g., `{ 1: "foo", 2: "bar" }` is equal to `{ 2: "bar", 1: "foo" }` and
 * `{ 1: "foo", 1: "huh", 2: "bar" }`.
 */
bool ZL_LocalParams_eq(const ZL_LocalParams* lhs, const ZL_LocalParams* rhs);

bool ZL_LocalIntParams_eq(
        const ZL_LocalIntParams* lhs,
        const ZL_LocalIntParams* rhs);

/**
 * The param value is compared by inspecting the contents. So two pointers with
 * different values, but which point to identical content, will compare as
 * equal.
 */
bool ZL_LocalCopyParams_eq(
        const ZL_LocalCopyParams* lhs,
        const ZL_LocalCopyParams* rhs);

/**
 * The param value is compared by comparing the pointer values.
 */
bool ZL_LocalRefParams_eq(
        const ZL_LocalRefParams* lhs,
        const ZL_LocalRefParams* rhs);

/**
 * These functions provide a hash of the logical param set.
 *
 * Two param sets that compare equal must have the same hash. Two param sets
 * that are logically unequal *should* have different hashes.
 *
 * These hash functions are not stable.
 */
size_t ZL_LocalParams_hash(const ZL_LocalParams* lp);

size_t ZL_LocalIntParams_hash(const ZL_LocalIntParams* lip);

size_t ZL_LocalCopyParams_hash(const ZL_LocalCopyParams* lcp);

size_t ZL_LocalRefParams_hash(const ZL_LocalRefParams* lrp);

ZL_END_C_DECLS

#endif // ZSTRONG_COMPRESS_LOCALPARAMS_H

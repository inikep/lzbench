// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZS_COMMON_REP_H
#define ZS_COMMON_REP_H

#include <string.h>

#include "openzl/codecs/rolz/common_rolz.h"
#include "openzl/common/debug.h"
#include "openzl/shared/portability.h"

#define kRepML 0

ZL_BEGIN_C_DECLS

#define ZS_REP_NUM 3
#define REP_SUB 4

typedef struct {
    uint32_t reps[ZS_REP_NUM];
    uint32_t mls[ZS_REP_NUM];
} ZS_rep;

static ZS_rep const ZS_initialReps = { { 1, 4, 8 }, { 1, 4, 8 } };

ZL_INLINE uint32_t ZS_rep_matchLength(ZS_rep const* reps, uint32_t matchLength)
{
    if (!kRepML) {
        return matchLength;
    }
    // if (reps->mls[0] >= 16) {
    //   return kRepMinMatch + 16 + matchLength;
    // }
    // int const delta = (int)matchLength - (int)reps->mls[0];
    // if (delta >= 0) {
    //   return 1 + ((uint32_t)delta << 1);
    // }
    // return ((uint32_t)-delta << 1) + 2;
    for (uint32_t i = 0; i < 3; ++i) {
        if (reps->mls[i] == matchLength) {
            return MINMATCH + i;
        }
    }
    return matchLength + MINMATCH + 2;
}

#define ZS_NO_REP 3
ZL_INLINE ZS_rep ZS_rep_update(
        ZS_rep const* reps,
        uint32_t repcode,
        uint32_t offset,
        uint32_t matchLength)
{
    ZS_rep ret;
    uint32_t const rep = repcode & 3;
    uint32_t const off = repcode >> 2;
    if (rep == ZS_NO_REP || rep == 2) {
        if (rep == 2) {
            ZL_ASSERT_EQ(reps->reps[2] + (off - REP_SUB), offset);
        }
        ret.reps[2] = reps->reps[1];
        ret.reps[1] = reps->reps[0];
        ret.reps[0] = offset;
    } else if (rep == 1) {
        ZL_ASSERT_EQ(reps->reps[1] + (off - REP_SUB), offset);
        ret.reps[2] = reps->reps[2];
        ret.reps[1] = reps->reps[0];
        ret.reps[0] = offset;
    } else {
        // TODO: Handle Rep0 better. It doesn't handle offsets...
        ZL_ASSERT_EQ(reps->reps[0], offset);
        memcpy(ret.reps, reps, sizeof(ret.reps));
    }
    if (rep == ZS_NO_REP) {
        memcpy(ret.mls, reps->mls, sizeof(ret.mls));
    } else if (matchLength == reps->mls[0]) {
        memcpy(ret.mls, reps->mls, sizeof(ret.mls));
    } else if (matchLength == reps->mls[1]) {
        matchLength = reps->mls[1];
        ret.mls[2]  = reps->mls[2];
        ret.mls[1]  = reps->mls[0];
        ret.mls[0]  = matchLength;
    } else {
        ret.mls[2] = reps->mls[1];
        ret.mls[1] = reps->mls[0];
        ret.mls[0] = matchLength;
    }
    return ret;
}

ZL_END_C_DECLS

#endif // ZS_COMMON_REP_H

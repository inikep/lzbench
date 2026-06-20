// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZS_COMMON_MARKOV_H
#define ZS_COMMON_MARKOV_H

#include "openzl/codecs/rolz/common_rolz.h"
#include "openzl/codecs/rolz/common_rolz_sequences.h"
#include "openzl/common/debug.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

#define MULTI 1

#define ZS_MARKOV_NUM_STATES (MULTI ? (LITS_ARE_SEQ ? 12 : 6) : 4)
#define ZS_MARKOV_LZ_INITIAL_STATE (LITS_ARE_SEQ ? 0 : 0)
#define ZS_MARKOV_RZ_INITIAL_STATE (LITS_ARE_SEQ ? 2 : 0)

ZL_INLINE uint32_t
ZS_markov_nextState(uint32_t const state, ZS_matchType const matchType)
{
    ZL_ASSERT_LT(matchType, 5);
#if MULTI
#    if LITS_ARE_SEQ
    static uint32_t const kNextState[ZS_MARKOV_NUM_STATES][5] = {
        { 1, 3, 6, 7, 11 }, // lit-lz (0)
        { 1, 3, 6, 7, 11 }, // *-lz (1)
        { 1, 3, 5, 7, 8 },  // lit-rolz (2)
        { 1, 3, 5, 7, 8 },  // *-rolz (3)
        { 1, 3, 6, 7 },     // lit-rep0 (4)
        {},                 // rolz-rep0 (5)
        {},                 // *-rep0 (6)
        {},                 // *-rep (7)
        {},                 // rolz-lit (8)
        {},                 // rep0-lit (9)
        {},                 // rep-lit (10)
        {},                 // *-lit (11)
    };
    return kNextState[state][matchType];
#    else
    static uint32_t const kNextState[ZS_MARKOV_NUM_STATES][4] = {
        { 0, 1, 2, 5 }, // ZS_mt_lz
        { 0, 1, 3, 5 }, // ZS_mt_rolz
        { 0, 1, 4, 5 }, // (lz|rep)-ZS_mt_rep0
        { 0, 1, 4, 5 }, // RZ-ZS_mt_rep0
        { 0, 1, 4, 5 }, // Rep0-ZS_mt_rep0
        { 0, 1, 2, 5 }, // ZS_mt_rep
    };
    return kNextState[state][matchType];
#    endif
#else
    (void)state;
    return matchType;
#endif
}

ZL_END_C_DECLS

#endif // ZS_COMMON_MARKOV_H

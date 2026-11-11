// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZS_TRANSFORMS_ROLZ_COMMON_ROLZ_SEQUENCES_H
#define ZS_TRANSFORMS_ROLZ_COMMON_ROLZ_SEQUENCES_H

#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

typedef enum {
    ZS_mt_lz   = 0, //< LZ match
    ZS_mt_rolz = 1, //< ROLZ match
    ZS_mt_rep0 = 2, //< rep0
    ZS_mt_rep  = 3, //< rep1+
    ZS_mt_lits = 4, //< Literals
    ZS_mt_lzn  = 5, //< LZ match w/o insert. Haven't tried this yet.
} ZS_matchType;

#define LITS_ARE_SEQ 0

typedef struct {
    uint32_t literalLength;
    uint32_t matchCode;
    uint32_t matchLength;
    uint32_t matchType;
} ZS_sequence;

ZL_END_C_DECLS

#endif

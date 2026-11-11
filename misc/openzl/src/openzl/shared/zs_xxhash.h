// Copyright (c) Meta Platforms, Inc. and affiliates.

// Host adaptations of libxxhash to zstrong,
// notably Build macros.
// To be included from xxhash.h

#ifndef ZS_COMMON_ZS_XXHASH_H
#define ZS_COMMON_ZS_XXHASH_H

#include "openzl/shared/portability.h" // ZL_FALLTHROUGH

#ifndef XXH_INLINE_ALL
#    define XXH_INLINE_ALL
#endif
#ifndef XXH_STATIC_LINKING_ONLY
#    define XXH_STATIC_LINKING_ONLY
#endif

#endif // ZS_COMMON_ZS_XXHASH_H

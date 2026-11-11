// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_CODECS_BRUTE_FORCE_SELECTOR_H
#define ZSTRONG_CODECS_BRUTE_FORCE_SELECTOR_H

#include <stddef.h>

#include "openzl/zl_opaque_types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Parameterized brute force selector that selects the best successor from a
 * user-provided list of candidates.
 * @param successors the list of successors to select from. Each successor must
 * be equipped to handle the input stream type.
 */
ZL_GraphID ZL_Compressor_registerBruteForceSelectorGraph(
        ZL_Compressor* cgraph,
        const ZL_GraphID* successors,
        size_t numSuccessors);

#if defined(__cplusplus)
}
#endif

#endif

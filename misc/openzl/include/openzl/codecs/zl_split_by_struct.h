// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_CODECS_SPLIT_BY_STRUCT_H
#define ZSTRONG_CODECS_SPLIT_BY_STRUCT_H

#include <stddef.h>

#include "openzl/zl_opaque_types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** Split-by-struct
 * This operation splits a serialized input
 * defined as an array of structures of fixed size,
 * by grouping same fields into their own stream.
 * All fields are considered concatenated back-to-back (no alignment).
 * For this transform to work, input must be an exact multiple of struct_size,
 * with struct_size = sum(field_sizes).
 * Each output stream is then assigned a successor Graph.
 */
ZL_GraphID ZL_Compressor_registerSplitByStructGraph(
        ZL_Compressor* cgraph,
        const size_t* fieldSizes,
        const ZL_GraphID* successors,
        size_t nbFields);

#if defined(__cplusplus)
}
#endif

#endif

// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_SDDL_PROFILE_H
#define OPENZL_SDDL_PROFILE_H

#include "openzl/zl_compressor.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Builds a Simple Data Description Language graph in @p compressor.
 *
 * Use @ref ZL_Compressor_buildSDDLGraph() if you want to control the successor
 * graph that dispatched data is sent to rather than just use the default this
 * profile sets up. (That default is the generic clustering graph, with its
 * default successor set.)
 *
 * @param description should point to a compiled description, such as is
 *                    produced by @ref openzl::sddl::Compiler::compile().
 */
ZL_RESULT_OF(ZL_GraphID)
ZL_SDDL_setupProfile(
        ZL_Compressor* compressor,
        const void* description,
        size_t descriptionSize);

#if defined(__cplusplus)
}
#endif

#endif // OPENZL_SDDL_PROFILE_H

// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_CODECS_ACE_H
#define ZSTRONG_CODECS_ACE_H

#include "openzl/zl_errors.h"
#include "openzl/zl_opaque_types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Inserts a placeholder for the Automated Compressor Explorer (ACE) to
 * replace with an automatically generated graph. It accepts a single input of
 * any type.
 *
 * Before training, this graph will just forward to `ZL_GRAPH_COMPRESS_GENERIC`.
 * After training the top-level graph, ACE will replace this component with an
 * automatically generated graph that performs well on the training data.
 *
 * If the same ACE GraphID is used multiple times within a graph, all the inputs
 * get passed to the same training, and only a single graph is generated. If
 * there are multiple ACE graphs in a top-level graph, each created with a call
 * to `ZL_Compressor_buildACEGraph()`, then each unique GraphID will be replaced
 * with a trained graph optimized for the inputs passed to that specific
 * GraphID.
 *
 * Input 0: Any type
 *
 * @returns The placeholder ACE graphID.
 */
ZL_GraphID ZL_Compressor_buildACEGraph(ZL_Compressor* compressor);

/// @see ZL_Compressor_buildACEGraph
ZL_RESULT_OF(ZL_GraphID)
ZL_Compressor_buildACEGraph2(ZL_Compressor* compressor);

/**
 * The same as `ZL_Compressor_buildACEGraph`, but uses `defaultGraph` to
 * compress until it is trained.
 *
 * @see ZL_Compressor_buildACEGraph
 */
ZL_GraphID ZL_Compressor_buildACEGraphWithDefault(
        ZL_Compressor* compressor,
        ZL_GraphID defaultGraph);

/// @see ZL_Compressor_buildACEGraphWithDefault
ZL_RESULT_OF(ZL_GraphID)
ZL_Compressor_buildACEGraphWithDefault2(
        ZL_Compressor* compressor,
        ZL_GraphID defaultGraph);

#if defined(__cplusplus)
}
#endif

#endif

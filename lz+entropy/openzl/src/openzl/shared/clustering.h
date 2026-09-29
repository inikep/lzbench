// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_COMMON_TRANSFORMS_CLUSTERING_H
#define ZSTRONG_COMMON_TRANSFORMS_CLUSTERING_H

#include "openzl/common/cursor.h"
#include "openzl/shared/portability.h"
#include "openzl/zl_errors.h"

ZL_BEGIN_C_DECLS

typedef struct {
    // All values must be < numClusters
    uint8_t contextToCluster[256];
    // Must be exact, so no clusters may be empty
    size_t numClusters;
    size_t maxSymbol;
} ZL_ContextClustering;

/// Encode `clustering` into `dst`
ZL_Report ZL_ContextClustering_encode(
        ZL_WC* dst,
        ZL_ContextClustering const* clustering);
/// Decode `clustering` from `src`
ZL_Report ZL_ContextClustering_decode(
        ZL_ContextClustering* clustering,
        ZL_RC* src);

typedef enum {
    /// Returns the identity clustering.
    ZL_ClusteringMode_identity,
    /// This algorithm results in a good clustering, but can be slow.
    ZL_ClusteringMode_greedy,
    /// This algorithm only prunes small contexts.
    /// It can be extremely fast.
    ZL_ClusteringMode_prune,
} ZL_ClusteringMode;

/**
 * Clusters the contexts into at most `maxClusters`.
 *
 * @param clustering The resulting clustering.
 * @param src The source data.
 * @param context The context data. Must be the same size as the source.
 * @param maxContext The maximum context value. Set to 255 if unknown.
 * @param maxClusters The maximum number of output clusters. If the clustering
 * algorithm produces more clusters an error is returned.
 * @param mode The clustering algorithm to use.
 */
ZL_Report ZL_cluster(
        ZL_ContextClustering* clustering,
        ZL_RC src,
        ZL_RC context,
        uint32_t maxContext,
        size_t maxClusters,
        ZL_ClusteringMode mode);

/// @see ZL_cluster
void ZL_ContextClustering_identity(ZL_ContextClustering* clustering, ZL_RC ctx);

/// @see ZL_cluster
void ZL_ContextClustering_greedy(
        ZL_ContextClustering* clustering,
        ZL_RC src,
        ZL_RC context,
        uint32_t maxContext,
        size_t maxClusters);

/// @see ZL_cluster
/// Prunes all contexts with too few values. Then prunes context by size.
void ZL_ContextClustering_prune(
        ZL_ContextClustering* clustering,
        ZL_RC context,
        uint32_t maxContext,
        size_t maxClusters);

ZL_END_C_DECLS

#endif // ZSTRONG_COMMON_TRANSFORMS_CLUSTERING_H

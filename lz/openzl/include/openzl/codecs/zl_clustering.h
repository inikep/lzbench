// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_CLUSTERING_H
#define OPENZL_CODECS_CLUSTERING_H

#include "openzl/zl_compressor.h"
#include "openzl/zl_graphs.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define ZL_CLUSTERING_TAG_METADATA_ID 0

/**
 * ZL_ClusteringConfig_TypeSuccessor contains information about the successor
 * for a specific type, eltWidth pair.
 */
typedef struct {
    ZL_Type type;
    size_t eltWidth;
    size_t successorIdx;
    size_t clusteringCodecIdx;
} ZL_ClusteringConfig_TypeSuccessor;

/**
 * Cluster specific information containing the pointer to the stable identifiers
 * of all inputs inside the cluster, as well as the successor associated with
 * the specific type, eltWidth of the cluster.
 */
typedef struct {
    ZL_ClusteringConfig_TypeSuccessor typeSuccessor;
    int* memberTags;
    size_t nbMemberTags;
} ZL_ClusteringConfig_Cluster;

/**
 * A serializable configuration used to provide information on how to do
 * clustering. ZL_ClusteringConfig provides a way to specify how previous
 * similar data should be clustered, and the configuration of successors to use
 * to achieve the desired goals for the compressor such as compression ratio,
 * compression speed etc.
 *
 * A valid configuration requires that across all clusters, tags are unique, and
 * all types in typeDefaults have a unique (type, eltWidth) pair.
 *
 *
 *
 * (TODO:)If the graph is unconfigured, all inputs will be unclustered and sent
 * to a generic compressor.
 */
typedef struct {
    ZL_ClusteringConfig_Cluster* clusters;
    size_t nbClusters;
    ZL_ClusteringConfig_TypeSuccessor* typeDefaults;
    size_t nbTypeDefaults;
} ZL_ClusteringConfig;

/**
* @brief Registers the clustering graph. This graph takes n inputs of any type,
* clustering them with the concat codecs according to the provided
* configuration and sends them to the successor graphs specified in
* @p successors
*
* @returns The graph ID registered for the clustering graph
* @param config The clustering configuration
* @param successors The set of custom successors to send clusters to
* @param nbSuccessors The number of custom successors

*/
ZL_GraphID ZL_Clustering_registerGraph(
        ZL_Compressor* compressor,
        const ZL_ClusteringConfig* config,
        const ZL_GraphID* successors,
        size_t nbSuccessors);

/** A specialization of @ref ZL_Clustering_registerGraph that clusters using
 * nodes in @p clusteringCodecs. A valid clustering codec is defined as one
 * which has exactly one input that is variable, and has an optional numeric
 * output with one typed output same as the input type. Falls back to the concat
 codec if tag is unconfigured and no default is set for the type.

 * @param clusteringCodecs The set of clustering codecs to use
 * @param nbClusteringCodecs The number of clustering codecs
 */
ZL_GraphID ZL_Clustering_registerGraphWithCustomClusteringCodecs(
        ZL_Compressor* compressor,
        const ZL_ClusteringConfig* config,
        const ZL_GraphID* successors,
        size_t nbSuccessors,
        const ZL_NodeID* clusteringCodecs,
        size_t nbClusteringCodecs);

#if defined(__cplusplus)
}
#endif

#endif

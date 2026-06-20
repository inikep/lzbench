// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <map>
#include <optional>
#include <vector>

#include "openzl/common/allocation.h"
#include "openzl/zl_compressor.h"
#include "tools/training/train_params.h"
#include "tools/training/utils/utils.h"

namespace openzl::training {

/**
 * @brief Performs training on a set of @p samples, in a columnar format by
 * finding a good choice of clustering for the sample inputs, and choosing
 * appropriate successors for these clusters.
 *
 * @param compressor The compressor to train which must have ZL_GRAPH_CLUSTERING
 * as the starting graph, otherwise an exception is thrown.
 * @param arena The arena where C memory allocation is done for this function.
 * @param samples The samples to train on. Each sample is a vector of inputs
 * which will be compressed using the clustering graph.
 * @param successors The successors to use for the clustering graph.
 * @param clusteringCodecs The codecs to use for clustering
 * @param typeToDefaultSuccessorIdxMap A map from (type, default successor
 * index) pairs to the index of the successor to use for that type.
 * @param clusteringTrainer The clustering trainer to use.
 * @param maxThreads The maximum number of threads to use for training.
 * @param clusteringGraphNonUniqueName The clustering graph prefix
 *
 * @returns ZL_GraphID with update local params showing a good clustering
 * configuration chosen by the specified training algorithm
 */

ZL_GraphID train_cluster(
        ZL_Compressor* compressor,
        Arena& arena,
        const std::vector<MultiInput>& samples,
        const std::vector<ZL_GraphID>& successors,
        const std::vector<ZL_NodeID>& clusteringCodecs,
        const std::map<std::pair<ZL_Type, size_t>, size_t>&
                typeToDefaultSuccessorIdxMap,
        const TrainParams& trainParams);

} // namespace openzl::training

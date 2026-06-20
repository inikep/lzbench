// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "tools/training/clustering/clustering_config_builder.h"
#include "tools/training/clustering/compression_utils.h"
#include "tools/training/clustering/trainers/trainer.h"

namespace openzl::training {

/* Runs a sequential clustering algorithm building up clusters from a
 * full split configuration where no inputs are clustered. This starts in a
 * state that is minimally clustered whiich differs from greedy which starts in
 * a state where all inputs of the same type are clustered. The algorithm
 * explores mutations in the config adding the next input to a known list of
 * clustered inputs. The mutation is accepted if it improves compression ratio,
 * otherwise is rejected and this input becomes a new cluster.
 *
 * The bottom up clustering algorithm is intended to be used where many inputs
 * are correlated, however not all inputs should be correlated with each other.
 */
class BottomUpTrainer : public Trainer {
   public:
    explicit BottomUpTrainer(int maxThreads, poly::optional<size_t> maxTimeSecs)
            : Trainer(maxThreads, maxTimeSecs)
    {
    }

    ClusteringConfigBuilder getTrainedClusteringConfig(
            const ZL_Compressor* compressor,
            const std::vector<MultiInput>& samples,
            const std::vector<ZL_GraphID>& successors,
            const std::vector<ZL_NodeID>& clusteringCodecs,
            const std::map<std::pair<ZL_Type, size_t>, size_t>&
                    typeToDefaultSuccessorIdxMap) override;

   private:
    /* Builds a config where all inputs are configured to have their own
     * clusster and trained to have the best successor out of the ones provided
     * for each of these clusters.
     */
    ClusteringConfigBuilder buildTrainedFullSplitConfig(
            const CompressionUtils& cUtils,
            const ColumnMetadata& metadata,
            const std::map<std::pair<ZL_Type, size_t>, size_t>&
                    typeToDefaultSuccessorIdxMap);

    /* Builds a config where the input with @p tag is moved out of its existing
     * cluster (if it exists) and put into the cluster with index @p clusterIdx.
     * The updated cluster is then trained to have the best successor out of the
     * ones provided.
     */
    ClusteringConfigBuilder buildTrainedConfigAddInputToCluster(
            const CompressionUtils& cUtils,
            const ColumnMetadata& metadata,
            const ClusteringConfigBuilder& config,
            int tag,
            ZL_Type type,
            size_t eltWidth,
            size_t clusterIdx);
};

} // namespace openzl::training

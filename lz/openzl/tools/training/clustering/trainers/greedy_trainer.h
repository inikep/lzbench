// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "tools/training/clustering/compression_utils.h"
#include "tools/training/clustering/trainers/trainer.h"
#include "tools/training/utils/thread_pool.h"

namespace openzl::training {

/* Runs a greedy alogrithm that produces a clustering config that optimizes for
 * compression ratio. The algorithm starts with a configuration where all inputs
 * of the same type are clustered and sent to the default successor. It then
 * explores mutations in the configuration, such as splitting an input from a
 * cluster and accepts mutations with a greedy methodology.
 *
 * The greedy clustering algorithm is intended to be used where inputs are
 * expected to be correlated in some manner.
 */
class GreedyTrainer : public Trainer {
   public:
    explicit GreedyTrainer(int maxThreads, poly::optional<size_t> maxTimeSecs)
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
    // Resets the trainer to its initial state.
    void reset()
    {
        topColumns_.clear();
        marginalCosts_.clear();
        similarColumns_.clear();
    }

    void initTopInputs(
            const CompressionUtils& cUtils,
            ColumnMetadata& metadata);
    void initInputMarginalCosts(
            const CompressionUtils& cUtils,
            const std::map<std::pair<ZL_Type, size_t>, size_t>&
                    typeToDefaultSuccessorIdxMap,
            ColumnMetadata& metadata);
    void initSimilarInputs(
            const CompressionUtils& cUtils,
            const std::map<std::pair<ZL_Type, size_t>, size_t>&
                    typeToDefaultSuccessorIdxMap,
            ColumnMetadata& metadata);

    std::vector<ColumnInfo> topColumns_;
    size_t maxInputs_              = 500;
    size_t numGreedyIters_         = 2;
    size_t maxPairSplitCandidates_ = 2;
    std::unordered_map<ColumnInfo, size_t, ColumnInfoHasher> marginalCosts_;
    std::unordered_map<ColumnInfo, std::vector<ColumnInfo>, ColumnInfoHasher>
            similarColumns_;
};

} // namespace openzl::training

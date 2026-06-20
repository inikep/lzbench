// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "tools/training/clustering/compression_utils.h"
#include "tools/training/clustering/trainers/trainer.h"

namespace openzl::training {

/* Performs full split clustering, producing a clustering configuration where
 * there are no clustered inputs. After setting each input to its own cluster,
 * explores all provided successors and chooses the best compressed successor
 * for each input.
 *
 * The full split clustering algorithm is intended to be a lightweight tool for
 * when the user expects inputs to be independent of each other.
 */
class FullSplitTrainer : public Trainer {
   public:
    explicit FullSplitTrainer(
            int maxThreads,
            poly::optional<size_t> maxTimeSecs)
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
};

} // namespace openzl::training

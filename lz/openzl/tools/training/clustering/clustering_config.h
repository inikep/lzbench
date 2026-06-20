// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include <unordered_set>
#include <vector>
#include "openzl/compress/graphs/generic_clustering_graph.h"
#include "openzl/shared/xxhash.h"

namespace openzl::training {

struct ColumnInfo {
    int tag;
    ZL_Type type;
    size_t width;
    bool operator==(const ColumnInfo& other) const
    {
        return tag == other.tag && type == other.type && width == other.width;
    }

    bool operator<(const ColumnInfo& other) const
    {
        return tag < other.tag;
    }
};

struct ColumnInfoHasher {
    std::size_t operator()(const ColumnInfo& c) const
    {
        XXH3_state_t state{};
        XXH3_64bits_reset(&state);
        XXH3_64bits_update(&state, &c.tag, sizeof(c.tag));
        XXH3_64bits_update(&state, &c.type, sizeof(c.type));
        XXH3_64bits_update(&state, &c.width, sizeof(c.width));
        return XXH3_64bits_digest(&state);
    }
};

// The required metadata for exploration
using ColumnMetadata = std::unordered_set<ColumnInfo, ColumnInfoHasher>;

class ClusteringConfig {
   public:
    ClusteringConfig()  = default;
    ~ClusteringConfig() = default;
    explicit ClusteringConfig(ZL_ClusteringConfig config)
            : config_(config),
              typeDefaultsStorage_(
                      config.typeDefaults,
                      config.typeDefaults + config.nbTypeDefaults),
              clustersStorage_(
                      config.clusters,
                      config.clusters + config.nbClusters)
    {
        config_.typeDefaults = typeDefaultsStorage_.data();
        config_.clusters     = clustersStorage_.data();
        clusterMemberTagsStorage_.reserve(config_.nbClusters);
        for (size_t i = 0; i < config_.nbClusters; ++i) {
            clusterMemberTagsStorage_.emplace_back(
                    config.clusters[i].memberTags,
                    config.clusters[i].memberTags
                            + config.clusters[i].nbMemberTags);
            config_.clusters[i].memberTags =
                    clusterMemberTagsStorage_[i].data();
        }
    }

    ClusteringConfig(const ClusteringConfig&)            = delete;
    ClusteringConfig(ClusteringConfig&&)                 = default;
    ClusteringConfig& operator=(const ClusteringConfig&) = delete;
    ClusteringConfig& operator=(ClusteringConfig&&)      = default;

    const ZL_ClusteringConfig* get() const
    {
        return &config_;
    }

    const ZL_ClusteringConfig* operator->() const
    {
        return get();
    }
    const ZL_ClusteringConfig& operator*() const
    {
        return *get();
    }

    void push_back(const ZL_ClusteringConfig_TypeSuccessor& typeSuccessor)
    {
        typeDefaultsStorage_.push_back(typeSuccessor);
        config_.nbTypeDefaults = typeDefaultsStorage_.size();
        config_.typeDefaults   = typeDefaultsStorage_.data();
    }

    void push_back(const ZL_ClusteringConfig_Cluster& cluster)
    {
        clustersStorage_.push_back(cluster);
        clusterMemberTagsStorage_.emplace_back(
                cluster.memberTags, cluster.memberTags + cluster.nbMemberTags);
        clustersStorage_.back().memberTags =
                clusterMemberTagsStorage_.back().data();
        config_.nbClusters = clustersStorage_.size();
        config_.clusters   = clustersStorage_.data();
    }

   private:
    ZL_ClusteringConfig config_{ .nbClusters = 0, .nbTypeDefaults = 0 };

    std::vector<ZL_ClusteringConfig_TypeSuccessor> typeDefaultsStorage_;
    std::vector<ZL_ClusteringConfig_Cluster> clustersStorage_;
    std::vector<std::vector<int>> clusterMemberTagsStorage_;
};
} // namespace openzl::training

// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "openzl/cpp/Exception.hpp"
#include "openzl/zl_compress.h"
#include "openzl/zl_errors.h"
#include "tools/logger/Logger.h"
#include "tools/training/clustering/clustering_config.h"

namespace openzl::training {

class Utils {
   public:
    static void ZL_Input_Deleter(ZL_Input* ptr)
    {
        ZL_TypedRef_free(ptr);
    }

    static void printClusteringConfig(const ZL_ClusteringConfig& config)
    {
        std::stringstream ss;
        ss << "ZL_ClusteringConfig: " << std::endl;
        ss << "nbClusters: " << config.nbClusters << std::endl;
        for (size_t i = 0; i < config.nbClusters; ++i) {
            ss << "cluster " << i << ": Successor: "
               << config.clusters[i].typeSuccessor.successorIdx;
            ss << " memberTags: ";
            for (size_t j = 0; j < config.clusters[i].nbMemberTags; ++j) {
                ss << " " << config.clusters[i].memberTags[j];
            }
            ss << std::endl;
        }
        ss << "nbTypeDefaults: " << config.nbTypeDefaults << std::endl;
        for (size_t i = 0; i < config.nbTypeDefaults; ++i) {
            ss << "typeDefault " << i
               << ": Successor: " << config.typeDefaults[i].successorIdx
               << " type: " << config.typeDefaults[i].type << std::endl;
        }
        tools::logger::Logger::log(tools::logger::VERBOSE1, ss.str());
    }

    static void printClusteringConfig(const ClusteringConfig& config)
    {
        printClusteringConfig(*config);
    }

    static void throwIfError(ZL_Report report, const std::string& errorMsg)
    {
        if (ZL_isError(report)) {
            throw Exception(errorMsg);
        }
    }
};

} // namespace openzl::training

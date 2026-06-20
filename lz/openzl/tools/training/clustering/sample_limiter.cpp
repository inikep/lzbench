// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "tools/training/clustering/sample_limiter.h"
#include <iostream>
#include <list>
#include <random>
#include "openzl/cpp/Exception.hpp"
#include "tools/io/InputSet.h"
#include "tools/io/InputSetStatic.h"
#include "tools/logger/Logger.h"

namespace openzl::training {

using namespace openzl::tools::logger;

std::vector<size_t> SampleLimiter::pickSampleIndicesWithLimits(
        const std::vector<size_t>& sampleSizes,
        const std::function<bool(size_t, size_t)>& stopCondition,
        size_t maxFileSize)
{
    std::vector<size_t> pickedIndices;
    std::list<size_t> unsampledIndices;
    for (size_t i = 0; i < sampleSizes.size(); ++i) {
        if (sampleSizes[i] < maxFileSize) {
            unsampledIndices.push_back(i);
        }
    }

    auto subsetSampleSize = 0;
    while (unsampledIndices.size() > 0
           && !stopCondition(subsetSampleSize, pickedIndices.size())) {
        std::uniform_int_distribution<int> dist(
                0, (int)unsampledIndices.size() - 1);
        auto index = dist(gen);
        auto it    = unsampledIndices.begin();
        std::advance(it, index);
        pickedIndices.push_back(*it);
        subsetSampleSize += sampleSizes[*it];
        unsampledIndices.erase(it);
    }
    return pickedIndices;
}

std::unique_ptr<tools::io::InputSet> SampleLimiter::getFilteredInputsPtr(
        const tools::io::InputSet& inputs)
{
    size_t totalSampleSize = 0;
    std::vector<size_t> sampleSizes;
    std::vector<std::shared_ptr<tools::io::Input>> filteredInputs;
    std::vector<std::shared_ptr<tools::io::Input>> inputPtrs;
    for (const auto& inputPtr : inputs) {
        totalSampleSize += inputPtr->size().value();
        sampleSizes.push_back(inputPtr->size().value());
        inputPtrs.push_back(inputPtr);
    }
    if (!numSamples_.has_value()) {
        if (totalSampleSize > maxTotalSize_) {
            Logger::log(
                    INFO,
                    "Total file size is too large, doing random sampling");
        }
    } else {
        if (numSamples_.value() != sampleSizes.size()) {
            Logger::log_c(INFO, "Using %zu samples", numSamples_.value());

        } else {
            Logger::log_c(
                    INFO,
                    "Using all provided training samples, total size %zu",
                    totalSampleSize);
            return std::make_unique<tools::io::InputSetStatic>(
                    std::move(inputPtrs));
        }
    }

    auto stopCondition = [this](size_t subsetSampleSize, size_t numPicked) {
        if (numSamples_.has_value()) {
            return numPicked == numSamples_.value();
        }
        return subsetSampleSize > maxTotalSize_ - maxFileSize_;
    };
    auto pickedIndices = pickSampleIndicesWithLimits(
            sampleSizes, stopCondition, maxFileSize_);
    size_t subsetSampleSize = 0;
    for (auto index : pickedIndices) {
        filteredInputs.push_back(inputPtrs[index]);
        subsetSampleSize += sampleSizes[index];
    }
    if (sampleSizes.size() == 0) {
        throw Exception("No samples found");
    }
    size_t numPicked = filteredInputs.size();
    if (numPicked == 0) {
        throw Exception("All samples exceed the max training file size limit.");
    }
    Logger::log_c(
            INFO,
            "Picked %zu samples out of %zu samples with total size %zu",
            numPicked,
            inputPtrs.size(),
            subsetSampleSize);
    auto filteredInputsPtr = std::make_unique<tools::io::InputSetStatic>(
            std::move(filteredInputs));
    return std::move(filteredInputsPtr);
}
} // namespace openzl::training

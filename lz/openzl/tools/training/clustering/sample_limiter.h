// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
#include <random>
#include "tools/io/InputSet.h"

namespace openzl::training {
class SampleLimiter {
   public:
    SampleLimiter(
            size_t maxTotalSize,
            size_t maxFileSize,
            std::optional<size_t> numSamples = std::nullopt,
            uint32_t seed                    = 0)
            : maxTotalSize_(maxTotalSize),
              maxFileSize_(maxFileSize),
              numSamples_(numSamples),
              gen(seed)
    {
    }

    // Choose a subset of samples that have size <= maxFileSize. Stops
    // picking when the subset has size >= minTotalSize. This
    // effectively limits the total size to minTotalSize + maxFileSize
    // - 1.
    std::vector<size_t> pickSampleIndicesWithLimits(
            const std::vector<size_t>& sampleSizes,
            const std::function<bool(size_t, size_t)>& stopCondition,
            size_t maxFileSize);

    // Uses pickSampleIndicesWithLimits to filter the inputs and returns a
    // pointer to the filtered inputSet. Uses the limit kMaxSingleSampleSize for
    // both minTotalSize and maxFileSize.
    std::unique_ptr<tools::io::InputSet> getFilteredInputsPtr(
            const tools::io::InputSet& inputs);

   private:
    size_t maxTotalSize_;
    size_t maxFileSize_;
    std::optional<size_t> numSamples_;
    std::mt19937 gen;
};
} // namespace openzl::training

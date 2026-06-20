// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <map>
#include <vector>

#include "tools/training/utils/utils.h"

namespace openzl::training {

/**
 * Collects input streams from a set of multi-input samples for training an
 * unconfigured node.
 *
 * This function compresses the samples providedusing the provided compression
 * context, and captures the input streams that would be processed by the
 * unconfigured node. These input streams can then be used for training.
 *
 * @param inputs Multi input samples to process
 * @param untrainedGraphName Name of the unconfigured graph node to train
 * @param cctx Compression context to use for processing samples
 * @return Vector of samples, where each sample is a vector of shared pointers
 * to input streams
 */

std::vector<MultiInput> collectInputStreamsForGraph(
        const std::vector<MultiInput>& inputs,
        const std::string& untrainedGraphName,
        CCtx& cctx);

std::map<std::string, std::vector<MultiInput>> collectInputStreamsForGraphs(
        const std::vector<MultiInput>& inputs,
        const std::vector<std::string>& untrainedGraphNames,
        CCtx& cctx);

} // namespace openzl::training

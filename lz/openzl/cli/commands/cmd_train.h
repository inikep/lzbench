// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cli/args/TrainArgs.h"

namespace openzl::cli {

/*
 * Train a compression profile and write it to a file.
 *
 * @param outputPath Output file for writing the trained compressor
 * @param global_args The global arguments.
 * @param train_args The training arguments (training algorithm, etc.)
 *
 * @return 0 on success, 1 on failure.
 */
int cmdTrain(const TrainArgs& args);

} // namespace openzl::cli

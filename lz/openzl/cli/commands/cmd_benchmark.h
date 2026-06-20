// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cli/args/BenchmarkArgs.h"

namespace openzl::cli {

struct BenchmarkResult {
    double compressionRatio;
    double decompressionSpeed;
    double compressionSpeed;
};

/// @return a return code for the CLI to return to the shell
int cmdBenchmark(const BenchmarkArgs& args);

/// @return The result of benchmarking the files specified
BenchmarkResult runCompressionBenchmarks(const BenchmarkArgs& args);

} // namespace openzl::cli

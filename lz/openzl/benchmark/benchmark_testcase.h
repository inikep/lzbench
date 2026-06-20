// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

namespace zstrong::bench {

/**
 * BenchmarkTestcase:
 * Defines the interface that all Zstrong benchmark testcases
 * should follow. There can be many kinds of testcases but all of them should
 * support registration in the Google Benchmark harness through
 * `registerBenchmarks`.
 * Other functionalities can be added in the future such as configuration or
 * filtering.
 */
class BenchmarkTestcase {
   public:
    BenchmarkTestcase()          = default;
    virtual ~BenchmarkTestcase() = default;
    /**
     * Registers the testcase with Google Benchmark harness.
     */
    virtual void registerBenchmarks() = 0;
};

} // namespace zstrong::bench

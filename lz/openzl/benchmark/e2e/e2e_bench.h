// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "benchmark/benchmark_data.h"
#include "benchmark/benchmark_testcase.h"
#include "benchmark/e2e/e2e_compressor.h"

namespace zstrong::bench::e2e {

/**
 * E2EBenchmarkTestcase:
 * An E2E benchmark testcases runs a ZstrongCompressor on BenchmarkData
 * and measures the compression ratio, compression speed and decompression
 * speed.
 * When registered it adds two benchmarks, one for compression and another for
 * decompression.
 */
class E2EBenchmarkTestcase : public BenchmarkTestcase {
   private:
    std::shared_ptr<ZstrongCompressor> _compressor;
    std::shared_ptr<BenchmarkData> _data;

   public:
    E2EBenchmarkTestcase(
            std::shared_ptr<ZstrongCompressor> compressor,
            std::shared_ptr<BenchmarkData> data)
            : _compressor(compressor), _data(data)
    {
    }
    void registerBenchmarks() override;
};

/**
 * registerE2EBenchmarks:
 * Generates and registers all E2E benchmark cases.
 */
void registerE2EBenchmarks();

} // namespace zstrong::bench::e2e

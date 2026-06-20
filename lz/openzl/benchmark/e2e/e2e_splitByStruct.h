// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <benchmark/benchmark.h>
#include <cstdint>
#include <initializer_list>
#include <memory>

#include "benchmark/benchmark_data.h"
#include "benchmark/benchmark_testcase.h"
#include "benchmark/e2e/e2e_bench.h"
#include "benchmark/e2e/e2e_compressor.h"
#include "benchmark/e2e/e2e_zstrong_utils.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_public_nodes.h"

namespace zstrong::bench::e2e {
namespace splitByStruct {

using namespace zstrong::bench::utils;

namespace {

// Graph generation ; this is the same split operation as in SAO graph.
// The goal is to focus on benchmarking the split operation specifically.
inline ZL_GraphID saoSplit_graph(ZL_Compressor* cgraph)
{
    if (ZL_isError(ZL_Compressor_setParameter(
                cgraph, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION))) {
        abort();
    }
    /* The SAO format contains mostly an array of structures,
     * containing 6 fields, of total length 28 bytes:
     * Real*8 SRA0      B1950 Right Ascension (radians)
     * Real*8 SDEC0     B1950 Declination (radians)
     * Character*2 IS   Spectral type (2 characters)
     * Integer*2 MAG    V Magnitude * 100
     * Real*4 XRPM      R.A. proper motion (radians per year)
     * Real*4 XDPM      Dec. proper motion (radians per year)
     * We replicate __only__ the split operation in this simplified graph.
     */
    return ZL_Compressor_registerSplitByStructGraph(
            cgraph,
            std::initializer_list<size_t>{ 8, 8, 2, 2, 4, 4 }.begin(),
            std::initializer_list<ZL_GraphID>{ ZL_GRAPH_STORE,
                                               ZL_GRAPH_STORE,
                                               ZL_GRAPH_STORE,
                                               ZL_GRAPH_STORE,
                                               ZL_GRAPH_STORE,
                                               ZL_GRAPH_STORE }
                    .begin(),
            6);
}

class SplitByStructCompressor : public ZstrongCompressor {
   private:
    ZL_GraphID configureGraph(ZL_Compressor* cgraph) override
    {
        return saoSplit_graph(cgraph);
    }

   public:
    using ZstrongCompressor::ZstrongCompressor;

    virtual std::string name() override
    {
        return "SAO-Splitter";
    }
};
} // namespace

// Register the SplitByStruct benchmarks, this allows us to specify pairs of
// compressors and corpora and register a different benchmark testcase of each.
// In practice, each registered benchmark testcase will register two benchmarks
// - one for compression and another for decompression.
inline void registerBenchmarks()
{
    // Create instances of the benchmark corpora.
    // We want to benchmark multiple different sizes.
    // The main requirement is to generate a size which is strict multiple of
    // structure size
    std::vector<std::shared_ptr<BenchmarkData>> corpora = {
        // 400 structures of 28 bytes (7 * 4) => ~ 10 K
        std::make_shared<UniformDistributionData<uint32_t>>(
                7 * 400, std::nullopt),
        // 4000 structures of 28 bytes (7 * 4) => ~ 100 K
        std::make_shared<UniformDistributionData<uint32_t>>(
                7 * 4000, std::nullopt),
        // 40,000 structures of 28 bytes (7 * 4) => ~ 1 MB
        std::make_shared<UniformDistributionData<uint32_t>>(
                7 * 40000, std::nullopt),
        // 258,997 structures of 28 bytes (7 * 4) => SAO size
        std::make_shared<UniformDistributionData<uint32_t>>(
                7 * 258997, std::nullopt),
    };
    for (auto corpus : corpora) {
        // Test SplitByStructCompressor with each corpus.
        // Note : in the future, this compressor could receive parameters
        // in order to test different split configurations.
        auto compressor = std::make_shared<SplitByStructCompressor>();
        E2EBenchmarkTestcase(compressor, corpus).registerBenchmarks();
    }
}
} // namespace splitByStruct
} // namespace zstrong::bench::e2e

// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <benchmark/benchmark.h>
#include <algorithm>
#include "benchmark/benchmark_config.h"
#include "benchmark/e2e/e2e_bench.h"
#include "benchmark/micro/micro_bench.h"

int main(int argc, char** argv)
{
    std::vector<std::string_view> args{ argv, argv + argc };
    // Find if the flag "--short" exists in args and remove it if it does.
    // @todo: in the future, provide help about this flag.
    auto shortFlag = std::find(args.begin(), args.end(), "--short");
    if (shortFlag != args.end()) {
        args.erase(shortFlag);
        zstrong::bench::BenchmarkConfig::instance().setUseShortList(true);
    }
    // Register benchmarks
    zstrong::bench::e2e::registerE2EBenchmarks();
    zstrong::bench::micro::registerMicroBenchmarks();

    // Initialize the harness, pass through command line arguments
    {
        int newArgc    = (int)args.size();
        char** newArgv = new char*[(size_t)newArgc];
        for (size_t i = 0; i < (size_t)newArgc; ++i) {
            newArgv[i] = const_cast<char*>(args[i].data());
        }
        benchmark::Initialize(&newArgc, newArgv);
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}

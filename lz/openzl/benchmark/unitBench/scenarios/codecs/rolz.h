// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_ROLZ_H
#define ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_ROLZ_H

#include <stddef.h>
#include "benchmark/unitBench/bench_entry.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * ROLZ compression wrapper function
 */
size_t rolzc_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * FastLZ compression wrapper function
 */
size_t fastlz_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

#ifdef __cplusplus
}
#endif

#endif // ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_ROLZ_H

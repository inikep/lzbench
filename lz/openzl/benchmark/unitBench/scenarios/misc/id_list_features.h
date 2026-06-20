// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_MISC_ID_LIST_FEATURES_H
#define ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_MISC_ID_LIST_FEATURES_H

#include <stddef.h>
#include "benchmark/unitBench/bench_entry.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * ID score list features compression wrapper function
 */
size_t id_score_list_features_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * ID list features compression wrapper function
 */
size_t id_list_features_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

#ifdef __cplusplus
}
#endif

#endif // ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_MISC_ID_LIST_FEATURES_H

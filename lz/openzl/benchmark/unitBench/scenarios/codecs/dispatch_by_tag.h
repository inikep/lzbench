// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_DISPATCH_BY_TAG_H
#define ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_DISPATCH_BY_TAG_H

#include <stddef.h>
#include "benchmark/unitBench/bench_entry.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Split by 8-byte elements wrapper function
 */
size_t splitBy8_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Preparation function for 8-byte element splitting
 */
size_t splitBy8_preparation(void* src, size_t srcSize, const BenchPayload* bp);

/**
 * Split by 4-byte elements wrapper function
 */
size_t splitBy4_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Preparation function for 4-byte element splitting
 */
size_t splitBy4_preparation(void* src, size_t srcSize, const BenchPayload* bp);

#ifdef __cplusplus
}
#endif

#endif // ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_DISPATCH_BY_TAG_H

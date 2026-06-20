// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_FLATPACK_H
#define ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_FLATPACK_H

#include <stddef.h>
#include "benchmark/unitBench/bench_entry.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Preparation function for flatpack decoding with 16-element alphabet
 */
size_t flatpackDecode16_prep(void* src, size_t srcSize, const BenchPayload* bp);

/**
 * Preparation function for flatpack decoding with 32-element alphabet
 */
size_t flatpackDecode32_prep(void* src, size_t srcSize, const BenchPayload* bp);

/**
 * Preparation function for flatpack decoding with 48-element alphabet
 */
size_t flatpackDecode48_prep(void* src, size_t srcSize, const BenchPayload* bp);

/**
 * Preparation function for flatpack decoding with 64-element alphabet
 */
size_t flatpackDecode64_prep(void* src, size_t srcSize, const BenchPayload* bp);

/**
 * Preparation function for flatpack decoding with 128-element alphabet
 */
size_t
flatpackDecode128_prep(void* src, size_t srcSize, const BenchPayload* bp);

/**
 * Flatpack decoding wrapper function
 */
size_t flatpackDecode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

#ifdef __cplusplus
}
#endif

#endif // ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_FLATPACK_H

// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_DISPATCH_STRING_H
#define ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_DISPATCH_STRING_H

#include <stddef.h>
#include "benchmark/unitBench/bench_entry.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Calculate output size for dispatch string encoding
 */
size_t dispatchStringEncode_outSize(const void* src, size_t srcSize);

/**
 * Dispatch string encoding wrapper function
 */
size_t dispatchStringEncode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Dispatch string decoding wrapper function
 */
size_t dispatchStringDecode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

#ifdef __cplusplus
}
#endif

#endif // ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_DISPATCH_STRING_H

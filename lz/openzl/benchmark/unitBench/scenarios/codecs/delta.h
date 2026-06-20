// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_BENCHMARK_UNITBENCH_DELTA_H
#define ZSTRONG_BENCHMARK_UNITBENCH_DELTA_H

#include "benchmark/unitBench/bench_entry.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t deltaEncode32_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);
size_t deltaEncode64_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

size_t deltaDecode8_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);
size_t deltaDecode16_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);
size_t deltaDecode32_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);
size_t deltaDecode64_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ZSTRONG_BENCHMARK_UNITBENCH_DELTA_H

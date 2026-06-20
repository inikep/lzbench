// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_ZSTD_H
#define ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_ZSTD_H

#include <stddef.h>
#include "benchmark/unitBench/bench_entry.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Calculate the maximum possible compressed size for ZSTD compression
 */
size_t zstd_outcSize(const void* src, size_t srcSize);

/**
 * Calculate the decompressed size from ZSTD compressed data
 */
size_t zstd_outdSize(const void* src, size_t srcSize);

/**
 * ZSTD compression wrapper function
 */
size_t zstd_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * ZSTD decompression wrapper function
 */
size_t zstdd_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * ZSTD decompression wrapper function using ZSTD_DCtx
 */
size_t zstddctx_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

#ifdef __cplusplus
}
#endif

#endif // ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_ZSTD_H

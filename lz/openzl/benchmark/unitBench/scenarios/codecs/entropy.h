// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_ENTROPY_H
#define ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_ENTROPY_H

#include <stddef.h>
#include "benchmark/unitBench/bench_entry.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Entropy encoding wrapper function
 */
size_t entropyEncode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Preparation function for entropy decoding
 */
size_t
entropyDecode_preparation(void* src, size_t srcSize, const BenchPayload* bp);

/**
 * FSE encoding wrapper function
 */
size_t fseEncode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Preparation function for FSE decoding
 */
size_t fseDecode_preparation(void* src, size_t srcSize, const BenchPayload* bp);

/**
 * Calculate output size for entropy decoding
 */
size_t entropyDecode_outSize(void const* src, size_t srcSize);

/**
 * Entropy decoding wrapper function
 */
size_t entropyDecode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Display function for entropy decoding results
 */
void entropyDecode_displayResult(
        const char* srcname,
        const char* fname,
        BMK_runTime_t rt,
        size_t srcSize);

#ifdef __cplusplus
}
#endif

#endif // ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_ENTROPY_H

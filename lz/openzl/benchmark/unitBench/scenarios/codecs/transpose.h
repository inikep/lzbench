// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_TRANSPOSE_H
#define ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_TRANSPOSE_H

#include <stddef.h>
#include "benchmark/unitBench/bench_entry.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Transpose encoding wrapper function for 16-bit elements
 */
size_t transposeEncode16_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Transpose encoding wrapper function for 32-bit elements
 */
size_t transposeEncode32_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Transpose encoding wrapper function for 64-bit elements
 */
size_t transposeEncode64_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Transpose decoding wrapper function for 16-bit elements
 */
size_t transposeDecode16_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Transpose decoding wrapper function for 32-bit elements
 */
size_t transposeDecode32_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Transpose decoding wrapper function for 64-bit elements
 */
size_t transposeDecode64_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

#ifdef __cplusplus
}
#endif

#endif // ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_TRANSPOSE_H

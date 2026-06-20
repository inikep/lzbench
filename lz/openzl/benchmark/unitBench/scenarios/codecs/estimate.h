// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_ESTIMATE_H
#define ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_ESTIMATE_H

#include <stddef.h>
#include "benchmark/unitBench/bench_entry.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Exact cardinality calculation for 2-byte elements
 */
size_t exact2_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Cardinality estimation for 1-byte elements
 */
size_t estimate1_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Cardinality estimation for 2-byte elements
 */
size_t estimate2_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Linear counting cardinality estimation for 4-byte elements
 */
size_t estimateLC4_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * HyperLogLog cardinality estimation for 4-byte elements
 */
size_t estimateHLL4_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Linear counting cardinality estimation for 8-byte elements
 */
size_t estimateLC8_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * HyperLogLog cardinality estimation for 8-byte elements
 */
size_t estimateHLL8_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Dimensionality estimation for 1-byte elements
 */
size_t dimensionality1_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Dimensionality estimation for 2-byte elements
 */
size_t dimensionality2_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Dimensionality estimation for 3-byte elements
 */
size_t dimensionality3_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Dimensionality estimation for 4-byte elements
 */
size_t dimensionality4_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Dimensionality estimation for 8-byte elements
 */
size_t dimensionality8_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

#ifdef __cplusplus
}
#endif

#endif // ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_ESTIMATE_H

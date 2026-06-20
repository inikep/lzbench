// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_TOKENIZE_H
#define ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_TOKENIZE_H

#include <stddef.h>
#include "benchmark/unitBench/bench_entry.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Tokenize 2-to-1 encoding wrapper function
 */
size_t tokenize2to1Encode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Tokenize 2-to-1 decoding wrapper function
 */
size_t tokenize2to1Decode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Display function for tokenize 2-to-1 decoding results
 */
void tokenize2to1Decode_displayResult(
        const char* srcname,
        const char* fname,
        BMK_runTime_t rt,
        size_t srcSize);

/**
 * Tokenize 4-to-2 encoding wrapper function
 */
size_t tokenize4to2Encode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Tokenize variable-to-4 encoding wrapper function
 */
size_t tokenizeVarto4Encode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Preparation function for tokenize variable-to-4 encoding
 */
size_t tokenizeVarto4_preparation(void* s, size_t ss, const BenchPayload* bp);

/**
 * Tokenize variable-to-4 decoding wrapper function
 */
size_t tokenizeVarto4Decode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Preparation function for tokenize variable decoding
 */
size_t tokVarDecode_prep(void* s, size_t ss, const BenchPayload* bp);

/**
 * Calculate output size for tokenize variable decoding
 */
size_t tokVarDecode_outSize(const void* src, size_t srcSize);

#ifdef __cplusplus
}
#endif

#endif // ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_TOKENIZE_H

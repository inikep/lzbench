// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_HUFFMAN_H
#define ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_HUFFMAN_H

#include <stddef.h>
#include "benchmark/unitBench/bench_entry.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Large Huffman encoding wrapper function
 */
size_t largeHuffmanEncode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Display function for large Huffman encoding results
 */
void largeHuffmanEncode_displayResult(
        const char* srcname,
        const char* fname,
        BMK_runTime_t rt,
        size_t srcSize);

/**
 * Large Huffman decoding wrapper function
 */
size_t largeHuffmanDecode_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/**
 * Display function for large Huffman decoding results
 */
void largeHuffmanDecode_displayResult(
        const char* srcname,
        const char* fname,
        BMK_runTime_t rt,
        size_t srcSize);

#ifdef __cplusplus
}
#endif

#endif // ZSTRONG_BENCHMARK_UNITBENCH_SCENARIOS_CODECS_HUFFMAN_H
